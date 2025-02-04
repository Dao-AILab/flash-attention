
import csv
import math
import torch
import os
import random
import functools
import triton
import triton.language as tl

AUTOTUNE = os.environ.get('FLASH_ATTENTION_TRITON_AMD_AUTOTUNE', '0').lower() in ('1', 'true', 'yes')
DEBUG = os.environ.get('FLASH_ATTENTION_TRITON_AMD_DEBUG', '0').lower() in ('1', 'true', 'yes')
PERF = os.environ.get('FLASH_ATTENTION_TRITON_AMD_PERF', '0').lower() in ('1', 'true', 'yes')
USE_SINGLE_BWD_KERNEL = os.environ.get('USE_SINGLE_BWD_KERNEL', '0').lower() in ('1', 'true', 'yes')
USE_TRITON_ROCM = os.getenv("FLASH_ATTENTION_TRITON_AMD_ENABLE", "FALSE") == "TRUE"
DEBUG_TRITON = os.environ.get('DEBUG_TRITON', '0').lower() in ('1', 'true', 'yes') and os.environ.get('TRITON_INTERPRET', '0').lower() in ('1', 'true', 'yes')
DEBUG_TRITON_DETAIL = os.environ.get('DEBUG_TRITON_DETAIL', '0').lower() in ('1', 'true', 'yes')
if USE_TRITON_ROCM: # TODO remove this
    random.seed(42)
DROPOUT_USE_PYTORCH = False
DROPOUT_DUMP = False

class MetaData():
    cu_seqlens_q = None
    cu_seqlens_k = None
    max_seqlens_q = 0
    max_seqlens_k = 0
    bias = None
    alibi_slopes = None
    causal = False
    num_contexts = 0
    varlen = False
    layout = None
    cache_seqlens = None
    cache_batch_idx = None
    new_kv = False
    seqlen_new = None
    k_new = None
    v_new = None
    return_scores= False
    dropout_p= 0.0
    philox_seed, philox_offset = None, None # if dropout_p > 0.0 seed the RNG so we get reproducible results for testing.
    # NOTE: scale sm_scale by log_2(e) and use 2^x in the loop as we do not have native e^x support in HW.
    use_exp2 = False
    rotary_sin = None
    rotary_cos = None
    rotary_interleaved = False
    rotary_conjunction = False
    

    def __repr__(self) -> str:
        return (f"MetaData(\n"
                f"  sm_scale={self.sm_scale},\n"
                f"  cu_seqlens_q={self.cu_seqlens_q},\n"
                f"  cu_seqlens_k={self.cu_seqlens_k},\n"
                f"  max_seqlens_q={self.max_seqlens_q},\n"
                f"  max_seqlens_k={self.max_seqlens_k},\n"
                f"  bias={self.bias},\n"
                f"  alibi_slopes={self.alibi_slopes},\n"
                f"  causal={self.causal},\n"
                f"  num_contexts={self.num_contexts},\n"
                f"  varlen={self.varlen},\n"
                f"  layout={self.layout},\n"
                f"  cache_seqlens={self.cache_seqlens},\n"
                f"  cache_batch_idx={self.cache_batch_idx},\n"
                f"  new_kv={self.new_kv},\n"
                f"  seqlen_new={self.seqlen_new},\n"
                f"  k_new={self.k_new},\n"
                f"  v_new={self.v_new},\n"
                f"  dropout_p={self.dropout_p},\n"
                f"  return_scores={self.return_scores}\n"
                f")")

    def __init__(self, sm_scale=1.0):
        self.sm_scale = sm_scale

    def set_varlen_params(self, cu_seqlens_q, cu_seqlens_k):
        self.varlen = True
        self.layout = 'thd'
        self.cu_seqlens_q = cu_seqlens_q
        self.cu_seqlens_k = cu_seqlens_k
        # Without "varlen", there should still be one sequence.
        assert len(cu_seqlens_q) >= 2
        assert len(cu_seqlens_q) == len(cu_seqlens_k)
        self.num_contexts = len(cu_seqlens_q) - 1
        for i in range(0, self.num_contexts):
            self.max_seqlens_q = max(cu_seqlens_q[i + 1].item() - cu_seqlens_q[i].item(), self.max_seqlens_q)
            self.max_seqlens_k = max(cu_seqlens_k[i + 1].item() - cu_seqlens_k[i].item(), self.max_seqlens_k)

    def need_bias(self, bias, batch, nheads, seqlen_q, seqlen_k):
        assert bias.is_cuda
        assert bias.dim() == 4
        assert bias.shape[0] == 1
        assert bias.shape[2:] == (seqlen_q, seqlen_k)
        self.bias = bias

    def need_alibi(self, alibi_slopes, batch, nheads):
        assert alibi_slopes.is_cuda
        assert alibi_slopes.dim() == 2
        assert alibi_slopes.shape[0] == batch
        assert alibi_slopes.shape[1] == nheads
        self.alibi_slopes = alibi_slopes

    def need_causal(self):
        self.causal = True

    def need_rotary(self, sin, cos, rotary_interleaved, rotary_conjunction=False):
        self.rotary_sin = sin
        self.rotary_cos = cos
        self.rotary_interleaved = rotary_interleaved
        self.rotary_conjunction = rotary_conjunction

    def need_dropout(self, dropout_p):
        self.dropout_p = dropout_p
        self.return_scores = True
        self.philox_seed, self.philox_offset = 0x1BF58, 0x1D4B49

    def check_args(self, q, k, v, o):
        assert q.dim() == k.dim() and q.dim() == v.dim()

        batch, nheads_q, nheads_k, head_size, _, _ = get_shape_from_layout(q, k, self.layout, self.cu_seqlens_q, self.cu_seqlens_k, self.max_seqlens_q, self.max_seqlens_k)
        if self.varlen:
            assert q.dim() == 3
            assert self.cu_seqlens_q is not None
            assert self.cu_seqlens_k is not None
            assert len(self.cu_seqlens_q) == len(self.cu_seqlens_k)
            # TODO: Remove once bias is supported with varlen
            assert self.bias is None
            # assert not self.return_scores
        else:
            assert q.dim() == 4
            assert self.max_seqlens_q > 0 and self.max_seqlens_k > 0
            assert self.cu_seqlens_q is None and self.cu_seqlens_k is None
        assert k.shape == v.shape
        assert q.shape[-1] == k.shape[-1] and q.shape[-1] == v.shape[-1]
        # TODO: Change assert if we support qkl f8 and v f16
        assert q.dtype == k.dtype and q.dtype == v.dtype
        assert head_size <= 256
        assert o.shape == q.shape
        assert (nheads_q % nheads_k) == 0
        assert self.layout is not None
        assert self.layout == 'thd' or not self.varlen

def input_helper(Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype, layout, device="cuda", DEBUG_INPUT=False):
    torch.manual_seed(20)

    # Initialize q, k, v
    if layout == 'bhsd':
        q_tensor_shape = (Z, HQ, N_CTX_Q, D_HEAD)
        k_tensor_shape = (Z, HK, N_CTX_K, D_HEAD)
    elif layout == 'bshd':
        q_tensor_shape = (Z, N_CTX_Q, HQ, D_HEAD)
        k_tensor_shape = (Z, N_CTX_K, HK, D_HEAD)
    else:
        assert False, f'Got unsupported tensor layout: {layout}'

    if DEBUG_INPUT:
        if layout == "bhsd":
            q = torch.arange(N_CTX_Q, dtype=dtype, device=device).view(1, 1, N_CTX_Q, 1).expand(*q_tensor_shape).contiguous().requires_grad_()
            k = torch.arange(N_CTX_K, dtype=dtype, device=device).view(1, 1, N_CTX_K, 1).expand(*k_tensor_shape).contiguous().requires_grad_()
            v = torch.arange(N_CTX_K, dtype=dtype, device=device).view(1, 1, N_CTX_K, 1).expand(*k_tensor_shape).contiguous().requires_grad_()
        elif layout == "bshd":
            q = torch.arange(N_CTX_Q, dtype=dtype, device=device).view(1, N_CTX_Q, 1, 1).expand(*q_tensor_shape).contiguous().requires_grad_()
            k = torch.arange(N_CTX_K, dtype=dtype, device=device).view(1, N_CTX_K, 1, 1).expand(*k_tensor_shape).contiguous().requires_grad_()
            v = torch.arange(N_CTX_K, dtype=dtype, device=device).view(1, N_CTX_K, 1, 1).expand(*k_tensor_shape).contiguous().requires_grad_()
    else:
        q = torch.randn(q_tensor_shape, dtype=dtype, device=device, requires_grad=True)
        k = torch.randn(k_tensor_shape, dtype=dtype, device=device, requires_grad=True)
        v = torch.randn(k_tensor_shape, dtype=dtype, device=device, requires_grad=True)
    
    if DEBUG_INPUT:
        sm_scale = 1
    else:
        sm_scale = D_HEAD**-0.5
    input_metadata = MetaData(sm_scale=sm_scale)
    input_metadata.max_seqlens_q = N_CTX_Q
    input_metadata.max_seqlens_k = N_CTX_K
    input_metadata.layout = layout
    return q, k, v, input_metadata


def random_seqlens_composition(N, Z):
    # generate a random composition of N into Z positive parts.
    idx = torch.randperm(N - 1)[: Z - 1] + 1
    idx, _ = torch.sort(idx)
    breakpoints = torch.cat([
        torch.tensor([0], dtype=torch.long),
        idx,
        torch.tensor([N], dtype=torch.long),
    ])
    seqlens = (breakpoints[1:] - breakpoints[:-1]).to(torch.int32)
    return seqlens

def varlen_input_helper(Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype, device="cuda", equal_seqlens=False, DEBUG_INPUT=False):
    torch.manual_seed(20)

    # Random or equal sequence lengths based on 'equal_seqlens' flag
    if not equal_seqlens:
        seqlens_q = random_seqlens_composition(N_CTX_Q, Z)
        seqlens_k = random_seqlens_composition(N_CTX_K, Z)
    else:
        seqlens_q = torch.full((Z,), N_CTX_Q // Z, dtype=torch.int32)
        seqlens_k = torch.full((Z,), N_CTX_K // Z, dtype=torch.int32)

    # calculate cumulative sequence lengths
    cu_seqlens_q = torch.cat([torch.tensor([0], dtype=torch.int32), seqlens_q.cumsum(dim=0)])
    cu_seqlens_k = torch.cat([torch.tensor([0], dtype=torch.int32), seqlens_k.cumsum(dim=0)])
    cu_seqlens_q = cu_seqlens_q.to(device=device).to(torch.int32)
    cu_seqlens_k = cu_seqlens_k.to(device=device).to(torch.int32)

    # total lengths
    total_q = cu_seqlens_q[-1].item()
    total_k = cu_seqlens_k[-1].item()

    if DEBUG_INPUT:
        sm_scale = 1.0
        
        q = torch.empty(total_q, HQ, D_HEAD, dtype=dtype, device=device)
        k = torch.empty(total_k, HK, D_HEAD, dtype=dtype, device=device)
        v = torch.empty(total_k, HK, D_HEAD, dtype=dtype, device=device)
        for i in range(Z):
            q_start = cu_seqlens_q[i].item()
            q_end   = cu_seqlens_q[i+1].item()
            q_length  = q_end - q_start
            k_start = cu_seqlens_k[i].item()
            k_end   = cu_seqlens_k[i+1].item()
            k_length  = k_end - k_start
            
          
            q[q_start:q_end, :, :] = (
                torch.arange(q_length, dtype=dtype, device=device)
                .view(q_length, 1, 1)
                .expand(q_length, HQ, D_HEAD)
            )
            k[k_start:k_end, :, :] = (
                torch.arange(k_length, dtype=dtype, device=device)
                .view(k_length, 1, 1)
                .expand(k_length, HK, D_HEAD)
            )
            v[k_start:k_end, :, :] = (
                torch.arange(k_length, dtype=dtype, device=device)
                .view(k_length, 1, 1)
                .expand(k_length, HK, D_HEAD)
            )
        q.requires_grad_()
        k.requires_grad_()
        v.requires_grad_()
      
    else:
        # Initialize q, k, v with random values
        q = torch.randn((total_q, HQ, D_HEAD), dtype=dtype, device=device).requires_grad_()
        k = torch.randn((total_k, HK, D_HEAD), dtype=dtype, device=device).requires_grad_()
        v = torch.randn((total_k, HK, D_HEAD), dtype=dtype, device=device).requires_grad_()
        sm_scale = D_HEAD ** -0.5

    input_metadata = MetaData(sm_scale=sm_scale)
    input_metadata.set_varlen_params(cu_seqlens_q, cu_seqlens_k)

    return q, k, v, input_metadata


def get_shape_from_layout(q, k, layout, cu_seqlens_q = None, cu_seqlens_k = None, max_seqlen_q=None, max_seqlen_k=None):
    if layout == 'bhsd':
        batch_q, nheads_q, max_seqlen_q, head_size_q = q.shape
        batch_k, nheads_k, max_seqlen_k, head_size_k = k.shape
    elif layout == 'bshd':
        batch_q, max_seqlen_q, nheads_q, head_size_q = q.shape
        batch_k, max_seqlen_k, nheads_k, head_size_k = k.shape
    elif  layout == 'thd':
        batch_q, max_seqlen_q, nheads_q, head_size_q = len(cu_seqlens_q) - 1, max_seqlen_q, q.shape[1], q.shape[2]
        batch_k, max_seqlen_k, nheads_k, head_size_k = len(cu_seqlens_k) - 1, max_seqlen_k, k.shape[1], k.shape[2]
    else:
        assert False, "Got unsupported layout."
    
    # assert
    assert batch_q == batch_k
    assert head_size_q == head_size_k

    return batch_q, nheads_q, nheads_k, head_size_q, max_seqlen_q, max_seqlen_k

def get_strides_from_layout(q, k, v, o, layout):
    if layout == 'thd':
        q_strides = (0, q.stride(1), q.stride(0), q.stride(2))
        k_strides = (0, k.stride(1), k.stride(0), k.stride(2))
        v_strides = (0, v.stride(1), v.stride(0), v.stride(2))
        o_strides = (0, o.stride(1), o.stride(0), o.stride(2))
    elif layout == 'bhsd':
        q_strides = (q.stride(0), q.stride(1), q.stride(2), q.stride(3))
        k_strides = (k.stride(0), k.stride(1), k.stride(2), k.stride(3))
        v_strides = (v.stride(0), v.stride(1), v.stride(2), v.stride(3))
        o_strides = (o.stride(0), o.stride(1), o.stride(2), o.stride(3))
    elif layout == 'bshd':
        q_strides = (q.stride(0), q.stride(2), q.stride(1), q.stride(3))
        k_strides = (k.stride(0), k.stride(2), k.stride(1), k.stride(3))
        v_strides = (v.stride(0), v.stride(2), v.stride(1), v.stride(3))
        o_strides = (o.stride(0), o.stride(2), o.stride(1), o.stride(3))
    else:
        assert False, 'Got unsupported layout.'
    return q_strides, k_strides, v_strides, o_strides

def get_padded_headsize(size):
    # Get closest power of 2 over or equal to 32.
    padded_d_model = 1 << (size - 1).bit_length()
    # Smallest head_dim supported is 16. If smaller, the tile in the
    # kernel is padded - there is no padding in memory for any dims.
    padded_d_model = max(padded_d_model, 16)
    return padded_d_model

def compute_alibi_tensor_ref(alibi_slopes, seqlen_q, seqlen_k):
    q_idx = torch.arange(seqlen_q, dtype=torch.int32, device="cuda").unsqueeze(-1)  # (N_CTX_Q, 1)
    k_idx = torch.arange(seqlen_k, dtype=torch.int32, device="cuda").unsqueeze(0)  # (1, N_CTX_K)
    relative_pos = torch.abs(q_idx + seqlen_k - seqlen_q - k_idx)  # (N_CTX_Q, N_CTX_K)
    return -1 * alibi_slopes.unsqueeze(-1).unsqueeze(-1) * relative_pos  # (Z, H, N_CTX_Q, N_CTX_K)

def create_dropout_mask(dropout_p, shape, seed):
    device = "cuda"
    rand_vals = torch.rand(shape, generator=torch.Generator(device=device).manual_seed(seed), device=device, dtype=torch.float32)
    return rand_vals > dropout_p

def create_dropout_mask_varlen(dropout_p, batch, nheads_q, cu_seqlens_q, cu_seqlens_k, philox_seed):
    device = "cuda"
    qlens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1])
    klens = (cu_seqlens_k[1:] - cu_seqlens_k[:-1])
    max_qlen = qlens.max()
    max_klen = klens.max()
    dropout_mask = torch.zeros((batch, nheads_q, max_qlen, max_klen), device=device)
    for b in range(batch):
        qlen = qlens[b]
        klen = klens[b]
        rand_vals = torch.rand((nheads_q, qlen, klen), generator=torch.Generator(device=device).manual_seed(philox_seed), device=device, dtype=torch.float32)
        submask = rand_vals > dropout_p
        dropout_mask[b, :, :qlen, :klen] = submask

    return dropout_mask

def write_dropout_mask(x, tensor_name = "tensor"):
    batch, head, seqlen_m, seqlen_n = x.shape
    x = x.tolist()

    with open(f'{tensor_name}.csv', 'w') as f:
        writer = csv.writer(f)
        for b in range(batch):
            for h in range(head):
                dropout_mask = x[b][h]
                if True:
                    BLOCK_M = 64
                    BLOCK_N = 64
                
                    # Calculate number of blocks in each dimension
                    m_blocks = math.ceil(seqlen_m / BLOCK_M)
                    n_blocks = math.ceil(seqlen_n / BLOCK_N)
                    
                    # Process each block
                    for m_block in range(m_blocks):
                        # Calculate row range for current block
                        row_start = m_block * BLOCK_M
                        row_end = min(row_start + BLOCK_M, seqlen_m)
                        
                        for n_block in range(n_blocks):
                            # Calculate column range for current block
                            col_start = n_block * BLOCK_N
                            col_end = min(col_start + BLOCK_N, seqlen_n)
                            
                            # Extract and write the current block
                            for row_idx in range(row_start, row_end):
                                row_data = dropout_mask[row_idx][col_start:col_end]
                                writer.writerow(row_data)
                else:
                    writer.writerows(dropout_mask)

def _strides(x: torch.Tensor, *stride_names: str):
    if x is None:
        return {f"stride_{s}": 0 for i, s in enumerate(stride_names)}

    assert x.ndim == len(stride_names)
    return {f"stride_{s}": x.stride(i) for i, s in enumerate(stride_names)}

def get_input_shapes():
    cases = [(max(1, 2**(16 - i)), 1, 2**i, 16, 1, 128)
             for i in range(8, 18)] + [(max(1, 2**(16 - i)), 1, 2**i, 16, 2, 128) for i in range(8, 18)]
    return cases

@functools.cache
def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"

@functools.cache
def get_arch():
    return triton.runtime.driver.active.get_current_target().arch

@functools.cache
def is_cdna():
    return is_hip() and get_arch() in ('gfx908', 'gfx90a', 'gfx940', 'gfx941', 'gfx942')

@functools.cache
def is_rdna():
    return is_hip() and get_arch() in ("gfx1030", "gfx1100", "gfx1101", "gfx1102", "gfx1200", "gfx1201")

@functools.cache
def arch_supports_fp8():
    return is_hip() and get_arch() in ('gfx942')