import csv
import math
import torch
import os
import random
import functools
import triton
import triton.language as tl
from typing import Literal, Optional, Union

# -------------------------------
# Gloabl Variables
# -------------------------------
AUTOTUNE = os.environ.get('FLASH_ATTENTION_TRITON_AMD_AUTOTUNE', '0').lower() in ('1', 'true', 'yes')
DEBUG = os.environ.get('FLASH_ATTENTION_TRITON_AMD_DEBUG', '0').lower() in ('1', 'true', 'yes')
USE_REF = os.environ.get('FLASH_ATTENTION_TRITON_AMD_REF', '0').lower() in ('1', 'true', 'yes')
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
    cu_seqlens_q: Optional[torch.Tensor] = None
    cu_seqlens_k: Optional[torch.Tensor] = None
    max_seqlens_q: int = 0
    max_seqlens_k: int = 0
    bias: Optional[torch.Tensor] = None
    alibi_slopes: Optional[torch.Tensor] = None
    causal: bool = False
    num_contexts = 0
    varlen: bool = False
    layout: Optional[Literal["bshd", "bhsd", "thd"]] = None
    cache_seqlens: Optional[Union[(int, torch.Tensor)]] = None
    cache_batch_idx = None
    packing: Optional[bool] = None
    return_scores: bool = False
    dropout_p: float = 0.0
    philox_seed: Optional[int] = None
    philox_offset : Optional[int]= None # if dropout_p > 0.0 seed the RNG so we get reproducible results for testing.
    # NOTE: scale sm_scale by log_2(e) and use 2^x in the loop as we do not have native e^x support in HW.
    use_exp2: bool = False
    rotary_sin: Optional[torch.Tensor] = None
    rotary_cos: Optional[torch.Tensor] = None
    rotary_interleaved: bool = False
    rotary_conjunction: bool = False
    

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
                f"  dropout_p={self.dropout_p},\n"
                f"  return_scores={self.return_scores}\n"
                f")")

    def __init__(self, sm_scale=1.0):
        self.sm_scale = sm_scale

    def set_varlen_params(self, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k):
        self.varlen = True
        self.layout = 'thd'
        self.cu_seqlens_q = cu_seqlens_q
        self.cu_seqlens_k = cu_seqlens_k
        self.max_seqlens_q = max_seqlen_q
        self.max_seqlens_k = max_seqlen_k

        # Without "varlen", there should still be one sequence.
        assert len(cu_seqlens_q) >= 2
        assert len(cu_seqlens_q) == len(cu_seqlens_k)

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

    def need_causal(self, causal):
        self.causal = causal

    def need_rotary(self, sin, cos, rotary_interleaved, rotary_conjunction=False):
        self.rotary_sin = sin
        self.rotary_cos = cos
        self.rotary_interleaved = rotary_interleaved
        self.rotary_conjunction = rotary_conjunction

    def need_dropout(self, dropout_p, return_softmax = True):
        self.dropout_p = dropout_p
        self.return_softmax = return_softmax
        self.philox_seed, self.philox_offset = 0x1BF58, 0x1D4B49

    def check_args(self, q, k, v, o):
        assert q.dim() == k.dim() and q.dim() == v.dim()

        batch, nheads_q, nheads_k, head_size, _, _ = get_shapes_from_layout(q, k, self.layout, self.cu_seqlens_q, self.cu_seqlens_k, self.max_seqlens_q, self.max_seqlens_k)
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
def random_seqlens_composition(SEQ_LEN, BATCH):
    # generate a random composition of N into Z positive parts.
    idx = torch.randperm(SEQ_LEN - 1)[: BATCH - 1] + 1
    idx, _ = torch.sort(idx)
    breakpoints = torch.cat([
        torch.tensor([0], dtype=torch.long),
        idx,
        torch.tensor([SEQ_LEN], dtype=torch.long),
    ])
    seqlens = (breakpoints[1:] - breakpoints[:-1]).to(torch.int32)
    return seqlens

def generate_varlen_tensor(
    total_seqlen: int,
    num_heads: int,
    head_size: int,
    batch_size: Optional[int] = None,
    equal_seqlens: bool = False,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    DEBUG_INPUT: bool = False
):
    if DEBUG:
        print("total_seqlen", total_seqlen)
        print("num_heads", num_heads)
        print("head_size", head_size)

    # save fp8 type
    is_fp8_dtype = is_dtype_fp8(dtype)
    if is_fp8_dtype:
        og_fp8_dtype = dtype
        dtype = torch.float32

    # get valid batch_size
    if batch_size is None:
        valid_batch_sizes = [bs for bs in [1, 2, 4, 8, 16, 32, 64] if bs <= total_seqlen]
        batch_size = random.choice(valid_batch_sizes)
    
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

    if is_fp8_dtype:
        # cast to fp8
        x, descale_x = cast_to_fp8(x, og_fp8_dtype, "thd", cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        x.requires_grad_()
        return x, cu_seqlens, max_seqlen, descale_x
    else:
        x.requires_grad_()
        return x, cu_seqlens, max_seqlen

def generate_bshd_tensor(BATCH, SEQ_LEN, NUM_HEADS, D_HEAD, dtype, device="cuda", DEBUG_INPUT=False):
    # save fp8 type
    is_fp8_dtype = is_dtype_fp8(dtype)
    if is_fp8_dtype:
        og_fp8_dtype = dtype
        dtype = torch.float32

    # gen tensor
    tensor_shape = (BATCH, SEQ_LEN, NUM_HEADS, D_HEAD)
    if DEBUG_INPUT:
        x = torch.arange(SEQ_LEN, dtype=dtype, device=device).view(1, SEQ_LEN, 1, 1).expand(*tensor_shape).contiguous()
    else:
        x = torch.randn(tensor_shape, dtype=dtype, device=device)
    
    if is_fp8_dtype:
        # cast to fp8
        x, descale_x = cast_to_fp8(x, og_fp8_dtype, "bshd")
        x.requires_grad_()
        return x, descale_x
    else:
        x.requires_grad_()
        return x

def generate_bhsd_tensor(BATCH, NUM_HEADS, SEQ_LEN, D_HEAD, dtype, device="cuda", DEBUG_INPUT=False):
    # save fp8 type
    is_fp8_dtype = is_dtype_fp8(dtype)
    if is_fp8_dtype:
        og_fp8_dtype = dtype
        dtype = torch.float32
    
    # gen tensor
    tensor_shape = (BATCH, NUM_HEADS, SEQ_LEN, D_HEAD)
    if DEBUG_INPUT:
        x = torch.arange(SEQ_LEN, dtype=dtype, device=device).view(1, 1, SEQ_LEN, 1).expand(*tensor_shape).contiguous()
    else:
        x = torch.randn(tensor_shape, dtype=dtype, device=device)
    

    if is_fp8_dtype:
        # cast to fp8
        x, descale_x = cast_to_fp8(x, og_fp8_dtype, "bhsd") # FIXME: I don't the casting fn supports this atm
        x.requires_grad_()
        return x, descale_x
    else:
        x.requires_grad_()
        return x

def input_helper(
    BATCH: int,
    HQ: int,
    HK: int,
    N_CTX_Q: int,
    N_CTX_K: int,
    D_HEAD: int,
    CAUSAL: bool,
    DROPOUT_P: float,
    dtype: torch.dtype,
    layout: Literal["bshd", "bhsd", "thd"],
    packing: Optional[Literal["kv", "qkv"]] = None,
    device: Literal["cpu", "cuda"] = "cuda",
    DEBUG_INPUT: bool = False,
):
    torch.manual_seed(20)
    is_fp8_dtype = is_dtype_fp8(dtype)

    if layout == "thd":
        # set params
        TOTAL_SEQLENS_Q = BATCH * N_CTX_Q
        TOTAL_SEQLENS_K = BATCH * N_CTX_K
        equal_seqlens=False
        
        # gen tensors
        # TODO: the gen functions should maybe have different gen modes like random, ones, increasing seqlen
        if is_fp8_dtype:
            q, cu_seqlens_q, max_seqlen_q, descale_q = generate_varlen_tensor(TOTAL_SEQLENS_Q, HQ, D_HEAD, batch_size=BATCH, dtype=dtype, device=device, equal_seqlens=equal_seqlens, DEBUG_INPUT=DEBUG_INPUT)
            k, cu_seqlens_k, max_seqlen_k, descale_k = generate_varlen_tensor(TOTAL_SEQLENS_K, HK, D_HEAD, batch_size=BATCH, dtype=dtype, device=device, equal_seqlens=equal_seqlens, DEBUG_INPUT=DEBUG_INPUT)
            v, _, _ , descale_v = generate_varlen_tensor(TOTAL_SEQLENS_K, HK, D_HEAD, batch_size=BATCH, dtype=dtype, device=device, equal_seqlens=equal_seqlens, DEBUG_INPUT=DEBUG_INPUT)
            do, _, _ , descale_do = generate_varlen_tensor(TOTAL_SEQLENS_Q, HQ, D_HEAD, batch_size=BATCH, dtype=dtype, device=device, equal_seqlens=equal_seqlens)
        else:
            q, cu_seqlens_q, max_seqlen_q = generate_varlen_tensor(TOTAL_SEQLENS_Q, HQ, D_HEAD, batch_size=BATCH, dtype=dtype, device=device, equal_seqlens=equal_seqlens, DEBUG_INPUT=DEBUG_INPUT)
            k, cu_seqlens_k, max_seqlen_k = generate_varlen_tensor(TOTAL_SEQLENS_K, HK, D_HEAD, batch_size=BATCH, dtype=dtype, device=device, equal_seqlens=equal_seqlens, DEBUG_INPUT=DEBUG_INPUT)
            v, _, _ = generate_varlen_tensor(TOTAL_SEQLENS_K, HK, D_HEAD, batch_size=BATCH, dtype=dtype, device=device, equal_seqlens=equal_seqlens, DEBUG_INPUT=DEBUG_INPUT)
            do = torch.ones_like(q) if DEBUG_INPUT else torch.randn_like(q)
        
        # setup metadata
        if DEBUG_INPUT:
            sm_scale = 1
        else:
            sm_scale = D_HEAD**-0.5
        metadata = MetaData(sm_scale=sm_scale)
        metadata.set_varlen_params(cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)
        metadata.need_causal(CAUSAL)
        metadata.need_dropout(DROPOUT_P)
    elif layout == 'bshd' or layout == "bhsd":
        # gen tensors
        if layout == "bshd":
            if is_fp8_dtype:
                q, descale_q = generate_bshd_tensor(BATCH, N_CTX_Q, HQ, D_HEAD, dtype=dtype, device=device, DEBUG_INPUT=DEBUG_INPUT)
                k, descale_k = generate_bshd_tensor(BATCH, N_CTX_K, HK, D_HEAD, dtype=dtype, device=device, DEBUG_INPUT=DEBUG_INPUT)
                v, descale_v = generate_bshd_tensor(BATCH, N_CTX_K, HK, D_HEAD, dtype=dtype, device=device, DEBUG_INPUT=DEBUG_INPUT)
                do, descale_do = generate_bshd_tensor(BATCH, N_CTX_Q, HQ, D_HEAD, dtype=dtype, device=device)
            else:
                q = generate_bshd_tensor(BATCH, N_CTX_Q, HQ, D_HEAD, dtype=dtype, device=device, DEBUG_INPUT=DEBUG_INPUT)
                k = generate_bshd_tensor(BATCH, N_CTX_K, HK, D_HEAD, dtype=dtype, device=device, DEBUG_INPUT=DEBUG_INPUT)
                v = generate_bshd_tensor(BATCH, N_CTX_K, HK, D_HEAD, dtype=dtype, device=device, DEBUG_INPUT=DEBUG_INPUT)
                do = torch.ones_like(q) if DEBUG_INPUT else torch.randn_like(q)
        elif layout == "bhsd":
            if is_fp8_dtype:
                q, descale_q = generate_bhsd_tensor(BATCH, HQ, N_CTX_Q, D_HEAD, dtype=dtype, device=device, DEBUG_INPUT=DEBUG_INPUT)
                k, descale_k = generate_bhsd_tensor(BATCH, HK, N_CTX_K, D_HEAD, dtype=dtype, device=device, DEBUG_INPUT=DEBUG_INPUT)
                v, descale_v = generate_bhsd_tensor(BATCH, HK, N_CTX_K, D_HEAD, dtype=dtype, device=device, DEBUG_INPUT=DEBUG_INPUT)
                do, descale_do = generate_bhsd_tensor(BATCH, HQ, N_CTX_Q, D_HEAD, dtype=dtype, device=device)
            else:
                q = generate_bhsd_tensor(BATCH, HQ, N_CTX_Q, D_HEAD, dtype=dtype, device=device, DEBUG_INPUT=DEBUG_INPUT)
                k = generate_bhsd_tensor(BATCH, HK, N_CTX_K, D_HEAD, dtype=dtype, device=device, DEBUG_INPUT=DEBUG_INPUT)
                v = generate_bhsd_tensor(BATCH, HK, N_CTX_K, D_HEAD, dtype=dtype, device=device, DEBUG_INPUT=DEBUG_INPUT)
                do = torch.ones_like(q) if DEBUG_INPUT else torch.randn_like(q)

        # setup metadata
        if DEBUG_INPUT:
            sm_scale = 1
        else:
            sm_scale = D_HEAD**-0.5
        metadata = MetaData(sm_scale=sm_scale)
        metadata.max_seqlens_q = N_CTX_Q
        metadata.max_seqlens_k = N_CTX_K
        metadata.layout = layout
        metadata.need_causal(CAUSAL)
        metadata.need_dropout(DROPOUT_P)
    else:
        raise ValueError(f"Unknown layout: {layout}")

    # deal with packing
    if packing is None:
        if is_fp8_dtype:
            return (q, descale_q), (k, descale_k), (v, descale_v), (do, descale_do), metadata
        else:
            return q, k, v, do, metadata
    elif packing == "kv":
        # pack k and v
        if layout in ["bhsd", "thd"]:
            kv = torch.stack([k, v], dim=1)
        elif layout == "bshd":
            kv = torch.stack([k, v], dim=2)
        else:
            raise ValueError(f"Unknown layout: {layout}")

        if is_fp8_dtype:
            raise ValueError("FP8 not supported kv packing yet")
        else:
            return q, kv, do, metadata
    elif packing == "qkv":
        # qkv packing - requires same sequence length for q and k
        assert N_CTX_Q == N_CTX_K, "For QKV packing, Q and K must have same sequence length"
        assert HQ == HK, "For QKV packing, Q and K must have same number of heads"

        # pack q, k, and v
        if layout in ["bhsd", "thd"]:
            qkv = torch.stack([q, k, v], dim=1)
        elif layout == "bshd":
            qkv = torch.stack([q, k, v], dim=2)
        else:
            raise ValueError(f"Unknown layout: {layout}")

        if is_fp8_dtype:
            raise ValueError("FP8 not supported qkv packing yet")
        else:
            return qkv, do, metadata
    else:
        assert False, f"Unsupported packing mode: {packing}"

# -------------------------------
# Alibi
# -------------------------------
@triton.jit
def compute_alibi_block(alibi_slope, seqlen_q, seqlen_k, offs_m, offs_n, transpose=False):
    # when seqlen_k and seqlen_q are different we want the diagonal to stick to the bottom right of the attention matrix
    # for casual mask we want something like this where (1 is kept and 0 is masked)
    # seqlen_q = 2 and seqlen_k = 5
    #   1 1 1 1 0
    #   1 1 1 1 1
    # seqlen_q = 5 and seqlen_k = 2
    #        0 0
    #        0 0
    #        0 0
    #        1 0
    #        1 1
    # for alibi the diagonal is 0 indicating no penalty for attending to that spot and increasing penalty for attending further from the diagonal
    # e.g. alibi_slope = 1, seqlen_q = 2, seqlen_k = 5, offs_m = [0, 1, 2, 3], offs_n = [0, 1, 2, 3, 4], transpose = False
    # 1. offs_m[:,None] = [[0],
    #                       [1],
    # 2. offs_m[:,None] + seqlen_k = [[5],
    #                                  [6],
    # 3. offs_m[:,None] + seqlen_k - seqlen_q = [[3],
    #                                             [4],
    # 4. offs_m[:,None] + seqlen_k - seqlen_q - offs_n[None,:] = [[3], - [[0, 1, 2, 3, 4]] =  [[ 3, 2, 1, 0,-1],
    #                                                            [4],                           [ 4, 3, 2, 1, 0]]
    # 5. -1 * alibi_slope * tl.abs(relative_pos_block) = [[ -3, -2, -1, 0,-1],
    #                                                     [ -4, -3, -2, -1, 0]],
    relative_pos_block = offs_m[:, None] + seqlen_k - seqlen_q - offs_n[None, :]
    alibi_block = -1 * alibi_slope * tl.abs(relative_pos_block)
    if transpose:
        return alibi_block.T
    else:
        return alibi_block

# -------------------------------
# FP8
# -------------------------------
def is_dtype_fp8(dtype):
    if dtype in {torch.float8_e4m3fnuz, torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e5m2fnuz}:
        if arch_supports_fp8():
            return True
        else:
            raise RuntimeError("This device doesnot support fp8")
    else:
        return False

def is_fp8(x):
    return is_dtype_fp8(x.dtype)

@triton.jit
def compute_fp8_scaling_factors(x, fp8_max: tl.constexpr):
    # compute fp8 scaling and descaling factor for a block
    x_amax = tl.max(tl.abs(x)) # NOTE: abs deals with negative values
    x_amax = tl.where(x_amax <= 1e-9, 1e-9, x_amax)
    scale_x = fp8_max / x_amax
    descale_x = x_amax / fp8_max
    return scale_x, descale_x

@triton.jit
def _cast_varlen_to_fp8_kernel_2d(
    X, X_fp8, Descale,
    cu_seqlens, H, MAX_SEQLEN,
    stride_batch, stride_seq, stride_head, stride_dim,
    stride_out_batch, stride_out_seq, stride_out_head, stride_out_dim,
    stride_desc_batch, stride_desc_head,
    FP8_CLAMP_VAL, 
    FP8_MAX,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ACTUAL_HEAD_DIM: tl.constexpr,
    IS_VARLEN: tl.constexpr
    ):
    # Process one (batch, head) pair per kernel
    b_id = tl.program_id(0)
    h_id = tl.program_id(1)
    
    # Get sequence bounds for this batch
    if IS_VARLEN:
        seq_start = tl.load(cu_seqlens + b_id)
        seq_end = tl.load(cu_seqlens + b_id + 1)
        seqlen = seq_end - seq_start
    else:
        seq_start = 0
        seqlen = MAX_SEQLEN  
    
    # initialize max value tracker
    x_max_val = 0.0
    
    # STEP 1: Find max absolute value across the entire sequence
    num_of_blocks = tl.cdiv(seqlen, BLOCK_SIZE)
    for blk_idx in range(0, num_of_blocks):
        # print("blk_idx:", blk_idx)
        # offsets
        offs_seq = blk_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        offs_dim = tl.arange(0, HEAD_DIM)
        
        # Create mask for valid elements
        mask_seq = offs_seq[:, None] < seqlen
        if ACTUAL_HEAD_DIM != HEAD_DIM:
            mask_dim = offs_dim[None, :] < ACTUAL_HEAD_DIM
            mask_seq = mask_seq & mask_dim

        # Load block
        adj_x =  b_id * stride_batch + h_id * stride_head + seq_start * stride_seq + offs_seq[:, None] * stride_seq + offs_dim[None, :] * stride_dim
        x_block = tl.load(X + adj_x, mask=mask_seq, other=0.0)
        # print("x_block:", x_block)
        
        # Find max absolute value in this block
        block_max = tl.max(tl.abs(x_block))
        # print("block_max:", block_max)
        
        # Update overall max
        x_max_val = tl.maximum(x_max_val, block_max)
        # print("x_max_val:", x_max_val)
    
    # clamp to avoid division by zero issues
    x_max_val = tl.maximum(x_max_val, FP8_CLAMP_VAL)
    
    # compute scale and descale factors for the entire sequence
    scale = FP8_MAX / x_max_val
    descale = x_max_val / FP8_MAX
    
    # store descale factor for this (batch, head) pair
    desc_ptr = Descale + b_id * stride_desc_batch + h_id# * stride_desc_head
    tl.store(desc_ptr, descale)

    # STEP 2: Apply scaling to the entire sequence and convert to FP8
    for blk_idx in range(0, num_of_blocks):
        # offsets
        offs_seq = blk_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        offs_dim = tl.arange(0, HEAD_DIM)
        
        # Create mask for valid elements
        mask_seq = offs_seq[:, None] < seqlen
        if ACTUAL_HEAD_DIM != HEAD_DIM:
            mask_dim = offs_dim[None, :] < ACTUAL_HEAD_DIM
            mask_seq = mask_seq & mask_dim
        
        # Load block - Using the fixed addressing
        addr = b_id * stride_batch + h_id * stride_head + seq_start * stride_seq + offs_seq[:, None] * stride_seq + offs_dim[None, :] * stride_dim
        x_block = tl.load(X + addr, mask=mask_seq, other=0.0)
        
        # Apply scale and convert to FP8
        x_fp8_block = (x_block * scale).to(X_fp8.type.element_ty)
        
        # Store results
        addr_out = b_id * stride_out_batch + h_id * stride_out_head + seq_start * stride_out_seq + offs_seq[:, None] * stride_out_seq + offs_dim[None, :] * stride_out_dim
        tl.store(X_fp8 + addr_out, x_fp8_block, mask=mask_seq)

def cast_to_fp8(
    x: torch.Tensor,
    fp8_dtype: torch.dtype,
    layout: Literal["bshd", "thd"],
    clamp_val: float = 1e-9,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    if False:
        print()
        print("cast_to_fp8")
        print("x:", x, x.shape)
        print("fp8_dtype:", fp8_dtype)
        print("cu_seqlens:", cu_seqlens)
        print("max_seqlen:", max_seqlen)
        print("clamp_val:", clamp_val)

    # check types are valid
    assert x.dtype in {torch.float16, torch.float32, torch.float64, torch.bfloat16} and is_dtype_fp8(fp8_dtype), f"Cannot cast {x.dtype} to {fp8_dtype}"

    # extract dimensions
    batch, max_seqlen_final, num_heads, head_dim = get_shape_from_layout(x, layout, cu_seqlens, max_seqlen)
    is_varlen = layout == "thd"
    fp8_max = torch.finfo(fp8_dtype).max
    if False:
        print("batch:", batch)
        print("max_seqlen_final:", max_seqlen_final)
        print("num_heads:", num_heads)
        print("head_dim:", head_dim)

    # get closest power of 2 for head_dim
    padded_head_dim = 1 << (head_dim - 1).bit_length()
    padded_head_dim = max(padded_head_dim, 32)

    # kernel params
    x_fp8 = torch.zeros_like(x, dtype=fp8_dtype)
    descale_factors = torch.zeros((batch, num_heads), device=x.device, dtype=torch.float32)
    BLOCK_SIZE = 128

    # calculate strides
    stride_batch, stride_head, stride_seq, stride_dim = get_stride_from_layout(x, layout)
    stride_out_batch, stride_out_head, stride_out_seq, stride_out_dim = get_stride_from_layout(x_fp8, layout)
    stride_desc_batch, stride_desc_head = descale_factors.stride()

    if False:
        print("stride_batch", stride_batch)
        print("stride_head", stride_head)
        print("stride_seq", stride_seq)
        print("stride_dim", stride_dim)
        print("stride_out_batch", stride_out_batch)
        print("stride_out_head", stride_out_head)
        print("stride_out_seq", stride_out_seq)
        print("stride_out_dim", stride_out_dim)
        print("stride_desc_batch", stride_desc_batch)
        print("stride_desc_head", stride_desc_head)

    grid = (batch, num_heads)
    _cast_varlen_to_fp8_kernel_2d[grid](
        x, x_fp8, descale_factors,
        cu_seqlens, num_heads, max_seqlen_final,
        stride_batch, stride_seq, stride_head, stride_dim,
        stride_out_batch, stride_out_seq, stride_out_head, stride_out_dim,
        stride_desc_batch, stride_desc_head,
        clamp_val, fp8_max,
        BLOCK_SIZE=BLOCK_SIZE,
        HEAD_DIM=padded_head_dim, 
        ACTUAL_HEAD_DIM=head_dim,
        IS_VARLEN=is_varlen
    )
    
    if False:
        print("x_fp8:", x_fp8, x_fp8.shape)
        print("descale_factors:", descale_factors, descale_factors.shape)
    return x_fp8, descale_factors

# -------------------------------
# Misc
# -------------------------------
def get_shape_from_layout(
    x: torch.Tensor,
    layout: Literal["bshd", "bhsd", "thd"],
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
) -> tuple[int, int, int, int]:
    if layout == 'bhsd':
        batch, num_heads, max_seqlen_final, head_dim = x.shape
    elif layout == 'bshd':
        batch, max_seqlen_final, num_heads, head_dim = x.shape
    elif  layout == 'thd':
        total_seqlen, num_heads, head_dim = x.shape
        if cu_seqlens is None:
            raise ValueError("cu_seqlens must be provided for varlen (thd) layout") 
        if max_seqlen is None:
            raise ValueError("max_seqlen must be provided for varlen (thd) layout")
        
        batch, max_seqlen_final, num_heads, head_dim = len(cu_seqlens) - 1, max_seqlen, num_heads, head_dim
    else:
        assert False, "Got unsupported layout."

    return batch, max_seqlen_final, num_heads, head_dim


def get_shapes_from_layout(q, k, layout, cu_seqlens_q = None, cu_seqlens_k = None, max_seqlen_q=None, max_seqlen_k=None):
    batch_q, seqlen_q, nheads_q, head_size_q = get_shape_from_layout(q, layout, cu_seqlens_q, max_seqlen_q)
    batch_k, seqlen_k, nheads_k, head_size_k = get_shape_from_layout(k, layout, cu_seqlens_k, max_seqlen_k)
    
    # assert
    assert batch_q == batch_k
    assert head_size_q == head_size_k

    return batch_q, nheads_q, nheads_k, head_size_q, seqlen_q, seqlen_k

def get_stride_from_layout(x: torch.Tensor, layout:Literal["bshd", "bhsd", "thd"]):
    if layout == 'thd':
        strides = (0, x.stride(1), x.stride(0), x.stride(2))  
    elif layout == 'bhsd':
        strides = (x.stride(0), x.stride(1), x.stride(2), x.stride(3))
    elif layout == 'bshd':
        strides = (x.stride(0), x.stride(2), x.stride(1), x.stride(3))
    else:
        assert False, 'Got unsupported layout.'
    return strides

def get_shape_and_strides_from_layout(x: torch.Tensor, layout: Literal["bshd", "bhsd", "thd"], cu_seqlens: Optional[torch.Tensor] = None, max_seqlen: Optional[int] = None):
    return get_shape_from_layout(x, layout, cu_seqlens, max_seqlen), get_stride_from_layout(x, layout)

def get_strides_from_layout(q, k, v, o, layout):
    q_strides = get_stride_from_layout(q, layout)
    k_strides = get_stride_from_layout(k, layout)
    v_strides = get_stride_from_layout(v, layout)
    o_strides = get_stride_from_layout(o, layout)
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
    return is_hip() and get_arch() in ('gfx908', 'gfx90a', 'gfx940', 'gfx941', 'gfx942', 'gfx950')

@functools.cache
def is_rdna():
    return is_hip() and get_arch() in ("gfx1030", "gfx1100", "gfx1101", "gfx1102", "gfx1200", "gfx1201")

@functools.cache
def arch_supports_fp8():
    return is_hip() and get_arch() in ('gfx942')
