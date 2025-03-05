import csv
import math
from typing import Optional
import torch
import os
import random
import functools
import triton
import triton.language as tl

# -------------------------------
# Gloabl Variables
# -------------------------------
AUTOTUNE = os.environ.get('FLASH_ATTENTION_TRITON_AMD_AUTOTUNE', '0').lower() in ('1', 'true', 'yes')
DEBUG = os.environ.get('FLASH_ATTENTION_TRITON_AMD_DEBUG', '0').lower() in ('1', 'true', 'yes')
PERF = os.environ.get('FLASH_ATTENTION_TRITON_AMD_PERF', '0').lower() in ('1', 'true', 'yes')
USE_SINGLE_BWD_KERNEL = os.environ.get('USE_SINGLE_BWD_KERNEL', '0').lower() in ('1', 'true', 'yes')
USE_TRITON_ROCM = os.getenv("FLASH_ATTENTION_TRITON_AMD_ENABLE", "FALSE") == "TRUE"
USE_TRITON_INTERPRET = os.environ.get('TRITON_INTERPRET', '0').lower() in ('1', 'true', 'yes')
DEBUG_TRITON = os.environ.get('DEBUG_TRITON', '0').lower() in ('1', 'true', 'yes') and USE_TRITON_INTERPRET
DEBUG_TRITON_DETAIL = os.environ.get('DEBUG_TRITON_DETAIL', '0').lower() in ('1', 'true', 'yes') and USE_TRITON_INTERPRET
if USE_TRITON_ROCM: # TODO remove this
    random.seed(42)
DROPOUT_USE_PYTORCH = False
DROPOUT_DUMP = False


# -------------------------------
# Metadata
# -------------------------------
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
        assert o.shape == q.shape
        assert (nheads_q % nheads_k) == 0
        assert self.layout is not None
        assert self.layout == 'thd' or not self.varlen

# -------------------------------
# Input Helper
# -------------------------------
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

def generate_varlen_tensor(
    batch_size: int,
    total_seqlen: int,
    num_heads: int,
    head_size: int,
    equal_seqlens: bool = False,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    DEBUG_INPUT: bool = False
) -> tuple[torch.Tensor, torch.Tensor, int]:
    # get seqlens
    if equal_seqlens:
        seqlens = torch.full(
        (batch_size,),
        total_seqlen // batch_size,
        dtype=torch.int32,
        device=device
        )
        seqlens[-1] += total_seqlen % batch_size
    else:
        seqlens = random_seqlens_composition(total_seqlen, batch_size).to(device=device)

    # create cumulative sequence lengths
    cu_seqlens = torch.cat([torch.tensor([0], dtype=torch.int32, device=device), seqlens.cumsum(dim=0)]).to(torch.int32).to(device=device)
    max_seqlen = torch.max(seqlens).to(torch.int32).item()

    # create varlen tensor
    if DEBUG_INPUT:
        x = torch.zeros(total_seqlen, num_heads, head_size, dtype=dtype, device=device)
        for i in range(batch_size):
            start = cu_seqlens[i].item()
            end   = cu_seqlens[i+1].item()
            length  = end - start

            x[start:end, :, :] = (
                torch.arange(length, dtype=dtype, device=device)
                .view(length, 1, 1)
                .expand(length, num_heads, head_size)
            )
    else:
        x = torch.randn((total_seqlen, num_heads, head_size), dtype=dtype, device=device)

    # requires grad
    x.requires_grad_()

    return x, cu_seqlens, max_seqlen

def varlen_input_helper(BATCH, HQ, HK, TOTAL_SEQLENS_Q, TOTAL_SEQLENS_K, D_HEAD, dtype, device="cuda", equal_seqlens=False, DEBUG_INPUT=False):
    torch.manual_seed(20)
    q, cu_seqlens_q, _ = generate_varlen_tensor(BATCH, TOTAL_SEQLENS_Q, HQ, D_HEAD, dtype=dtype, device=device, equal_seqlens=equal_seqlens, DEBUG_INPUT=DEBUG_INPUT)
    k, cu_seqlens_k, _ = generate_varlen_tensor(BATCH, TOTAL_SEQLENS_K, HK, D_HEAD, dtype=dtype, device=device, equal_seqlens=equal_seqlens, DEBUG_INPUT=DEBUG_INPUT)
    v, _, _ = generate_varlen_tensor(BATCH, TOTAL_SEQLENS_K, HK, D_HEAD, dtype=dtype, device=device, equal_seqlens=equal_seqlens, DEBUG_INPUT=DEBUG_INPUT)
    sm_scale = D_HEAD ** -0.5

    input_metadata = MetaData(sm_scale=sm_scale)
    input_metadata.set_varlen_params(cu_seqlens_q, cu_seqlens_k)

    return q, k, v, input_metadata

def nonvarlen_input_helper(Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype, layout, device="cuda", DEBUG_INPUT=False):
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

def input_helper(BATCH, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype, layout, device="cuda", DEBUG_INPUT=False):
    if layout == "thd":
        q, k, v, metadata = varlen_input_helper(BATCH, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype, device=device, DEBUG_INPUT=DEBUG_INPUT)
    else:
        q, k, v, metadata = nonvarlen_input_helper(BATCH, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype, layout, device=device, DEBUG_INPUT=DEBUG_INPUT)

    return q, k, v, metadata

# -------------------------------
# FP8
# -------------------------------
@triton.jit
def compute_fp8_scaling_factors(x, fp8_max: tl.constexpr):
    # compute fp8 scaling and descaling factor for a block
    x_amax = tl.max(tl.abs(x)) # NOTE: abs deals with negative values
    x_amax = tl.where(x_amax <= 1e-9, 1e-9, x_amax)
    scale_x = fp8_max / x_amax
    descale_x = x_amax / fp8_max
    return scale_x, descale_x

def is_fp8(x):
    if x.dtype in {torch.float8_e4m3fnuz, torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e5m2fnuz}:
        if arch_supports_fp8():
            return True
        else:
            raise RuntimeError("This device doesnot support fp8")
    else:
        return False

def cast_nonvarlen_to_fp8(
    x: torch.Tensor,
    fp8_dtype,
    layout,
    clamp_val=1e-9,
):
    if layout == "bshd":
        if len(x.shape) != 4:
            raise ValueError(f"'bshd' tensor should have shape [batch, seqlen, heads, dim], got {x.shape}")
        reduce_dims = (1, 3)  # seq_len and dim dimensions
    elif layout == "bhsd":
        if len(x.shape) != 4:
            raise ValueError(f"'bhsd' tensor should have shape [batch, heads, seqlen, dim], got {x.shape}")
        reduce_dims = (2, 3)  # seq_len and dim dimensions
    else:
        raise ValueError(f"Unknown layout: {layout}")
  

    # Compute the absolute max along reduce_dims, clamped to avoid 0-scale
    x_abs_max = x.abs().amax(dim=reduce_dims)
    x_abs_max = torch.maximum(x_abs_max, x.new_tensor(clamp_val))

    # Unsqueeze back to a shape suitable for broadcast
    unsqueeze_dims = sorted(reduce_dims)
    for d in unsqueeze_dims:
        x_abs_max = x_abs_max.unsqueeze(d)

    # compute scale and descale
    fp8_max = torch.finfo(fp8_dtype).max
    scale = fp8_max / x_abs_max
    descale_factor = x_abs_max / fp8_max

    # cast to FP8, optionally setting requires_grad
    x_fp8 = (x * scale).to(fp8_dtype)

    return x_fp8, descale_factor

def cast_varlen_to_fp8(
    x: torch.Tensor,
    fp8_dtype: torch.dtype,
    cu_seqlens,
    clamp_val: float = 1e-9,
) -> tuple[torch.Tensor, torch.Tensor]:
    # validate tensor shape
    if len(x.shape) != 3:
        raise ValueError(f"tensor should have shape [total_seqlen, heads, dim], got {x.shape}")
    num_heads = x.shape[1]
    
    # Get batch size from cu_seqlens
    batch = cu_seqlens.shape[0] - 1
    fp8_max = torch.finfo(fp8_dtype).max
    
    # Compute scale and descale factors per sequence
    x_fp8 = torch.zeros_like(x, dtype=fp8_dtype)
    descale_factors = torch.zeros((batch, num_heads), device=x.device, dtype=torch.float32)
    
    for i in range(batch):
        start = cu_seqlens[i]
        end = cu_seqlens[i + 1]
        x_slice = x[start:end]  # Slice for current sequence
        
        # Standard tensor (0: seq_len, 2: head_dim)
        x_abs_max = x_slice.abs().amax(dim=(0, 2))  # [heads]
        
        # apply minimum clamping
        x_abs_max = torch.maximum(x_abs_max, x.new_tensor(clamp_val))
        
        # compute scale and descale factors
        scale_i = fp8_max / x_abs_max
        descale_i = x_abs_max / fp8_max
        
        # store descale factors
        descale_factors[i, :] = descale_i
        
        scale_reshape = scale_i.reshape(1, num_heads, 1)
        
        # scale and cast to FP8
        x_fp8[start:end] = (x_slice * scale_reshape).to(fp8_dtype)
        
    return x_fp8, descale_factors

def decast_fp8(
    x_fp8: torch.Tensor,
    descale_factor: torch.Tensor,
    original_dtype: torch.dtype,
    layout: str,
    cu_seqlens: Optional[torch.Tensor] = None
) -> torch.Tensor:
    x_orig = x_fp8.to(original_dtype)
    
    if layout in ("bshd", "bhsd"):
        return x_orig * descale_factor
    elif layout == "thd":
        if cu_seqlens is None:
            raise ValueError("cu_seqlens must be provided for varlen layout ('thd')")
        
        if len(x_orig.shape) != 3:
            raise ValueError(f"tensor should have shape [total_seqlen, heads, dim], got {x_orig.shape}")
        
        # create output tensor
        x_out = x_orig.clone()
        batch = cu_seqlens.shape[0] - 1
        
        # apply descaling per sequence
        for i in range(batch):
            start = int(cu_seqlens[i].item())
            end = int(cu_seqlens[i + 1].item())
            
            # reshape to [1, heads, 1]
            factor = descale_factor[i].reshape(1, -1, 1)
            
            # apply descaling
            x_out[start:end] = x_out[start:end] * factor
        
        return x_out
    else:
        raise ValueError(f"Unknown layout: {layout}")

def cast_to_fp8(
    x: torch.Tensor,
    fp8_dtype: torch.dtype,
    layout: str,
    clamp_val: float = 1e-9,
    cu_seqlens=None
) -> tuple[torch.Tensor, torch.Tensor]:
    if layout in ("bshd", "bhsd"):
        return cast_nonvarlen_to_fp8(x, fp8_dtype, layout, clamp_val=clamp_val)
    elif layout == "thd":
        if cu_seqlens is None:
            raise ValueError("cu_seqlens must be provided for varlen (thd) layout")
        return cast_varlen_to_fp8(x, fp8_dtype, cu_seqlens, clamp_val=clamp_val)
    else:
        raise ValueError(f"Unknown layout: {layout}")
# -------------------------------
# Misc
# -------------------------------
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

def _strides(x: torch.Tensor, *stride_names: str):
    if x is None:
        return {f"stride_{s}": 0 for i, s in enumerate(stride_names)}

    assert x.ndim == len(stride_names)
    return {f"stride_{s}": x.stride(i) for i, s in enumerate(stride_names)}

def get_input_shapes():
    cases = [(max(1, 2**(16 - i)), 1, 2**i, 16, 1, 128)
             for i in range(8, 18)] + [(max(1, 2**(16 - i)), 1, 2**i, 16, 2, 128) for i in range(8, 18)]
    return cases

# -------------------------------
# Dropouts
# -------------------------------
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

# -------------------------------
# Runtime info
# -------------------------------
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
