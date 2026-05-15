# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# [2025-07-04] Version in Cute-DSL, for Hopper and Blackwell. You'll need install nvidia-cutlass-dsl==4.2.0.

# Supported features:
# - BF16 & FP16 dtype
# - noncausal & causal attention
# - MHA, GQA, MQA
# - hdim 64, 96, 128.
# - (hdim_qk, hdim_v) = (192, 128) for Blackwell (i.e. DeepSeek shape)
# - varlen
# - sliding window
# - bwd pass for Ampere (will also run on Hopper/Blackwell, but will be slow)

# Features not supported yet:
# - split (i.e. FlashDecoding)
# - tuned block sizes
# - paged KV
# - append KV to existing KV cache
# - FP8
# - bwd pass optimized for Hopper/Blackwell

import math
import os
from functools import lru_cache
from typing import Optional, Tuple, Callable, Union, Type

import torch


@lru_cache(maxsize=None)
def _get_device_capability():
    """Cached device capability check."""
    return torch.cuda.get_device_capability()[0]

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from flash_attn.cute import utils
from flash_attn.cute.flash_fwd import FlashAttentionForwardSm80, FlashAttentionForwardSm90
from flash_attn.cute.flash_fwd_sm100 import FlashAttentionForwardSm100
from flash_attn.cute.flash_fwd_sm100_fp4 import FlashAttentionForwardSm100 as FlashAttentionForwardSm100FP4
from flash_attn.cute.flash_bwd_preprocess import FlashAttentionBackwardPreprocess
from flash_attn.cute.flash_bwd import FlashAttentionBackwardSm80
from flash_attn.cute.flash_bwd_sm90 import FlashAttentionBackwardSm90
from flash_attn.cute.flash_bwd_sm100 import FlashAttentionBackwardSm100
from flash_attn.cute.flash_bwd_postprocess import FlashAttentionBackwardPostprocess
from flash_attn.cute.flash_fwd_combine import FlashAttentionForwardCombine

from flash_attn.cute.block_sparsity import (
    BlockSparseTensorsTorch,
    to_cute_block_sparse_tensors,
    normalize_block_sparse_tensors,
    get_block_sparse_expected_shapes,
)

def dump_kernel_attributes(compiled_kernel):
    from cuda.bindings import driver
    from cutlass.utils import HardwareInfo
    import torch
    device_id = torch.cuda.current_device()
    hardware_info = HardwareInfo(device_id=device_id)
    cubin_data = compiled_kernel.artifacts.CUBIN
    if cubin_data is None:
        print("cubin_data is None, need '--keep-cubin' option when compiling to print kernel attributes")
        return
    cuda_library = hardware_info._checkCudaErrors(
        driver.cuLibraryLoadData(cubin_data, None, None, 0, None, None, 0)
    )
    kernels = hardware_info._checkCudaErrors(driver.cuLibraryEnumerateKernels(1, cuda_library))
    kernel = hardware_info._checkCudaErrors(driver.cuKernelGetFunction(kernels[0]))
    # more metrics: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html#group__CUDA__EXEC_1g5e92a1b0d8d1b82cb00dcfb2de15961b
    local_size_bytes = hardware_info._checkCudaErrors(
        driver.cuFuncGetAttribute(
            driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
            kernel,
        )
    )
    num_regs = hardware_info._checkCudaErrors(
        driver.cuFuncGetAttribute(
            driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS,
            kernel,
        )
    )

    print(f"--- Kernel Info ---")
    print(f"local_size_bytes: {local_size_bytes}")
    print(f"num_regs: {num_regs}")
    print(f"--- End Kernel Info ---")

def maybe_contiguous(x):
    if x is None or x.stride(-1) == 1:
        return x
    # FP4 dtypes have no copy_ kernel — .contiguous() would fail. Callers (FP4 V)
    # are responsible for handing in already-properly-laid-out FP4 tensors.
    if hasattr(torch, "float4_e2m1fn_x2") and x.dtype == torch.float4_e2m1fn_x2:
        return x
    return x.contiguous()


def _validate_tensor(t, name, expected_shape, expected_dtype, expected_device):
    assert t.shape == expected_shape, f"{name} shape {t.shape} != expected {expected_shape}"
    assert t.dtype == expected_dtype, f"{name} dtype {t.dtype} != expected {expected_dtype}"
    assert t.device == expected_device, f"{name} device {t.device} != expected {expected_device}"
    assert t.is_cuda, f"{name} must be on CUDA"

def to_cute_tensor(t, assumed_align=16, leading_dim=-1, fully_dynamic=False):
    """Convert torch tensor to cute tensor for TVM FFI. leading_dim=-1 defaults to t.ndim-1."""
    if isinstance(t, cute.Tensor):
        return t
    tensor = from_dlpack(t.detach(), assumed_align=assumed_align, enable_tvm_ffi=True)
    if fully_dynamic:
        return tensor.mark_layout_dynamic()
    if leading_dim == -1:
        leading_dim = t.ndim - 1
    return tensor.mark_layout_dynamic(leading_dim=leading_dim)


torch2cute_dtype_map = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
    torch.float8_e4m3fn: cutlass.Float8E4M3FN,
    torch.float8_e5m2: cutlass.Float8E5M2,
}

cute2torch_dtype_map = {
    cutlass.Float16: torch.float16,
    cutlass.BFloat16: torch.bfloat16,
    cutlass.Float32: torch.float32,
    cutlass.Float4E2M1FN: torch.bfloat16,  # FP4 outputs as bfloat16
}

# Check if dtype is nvfp4
def is_nvfp4_dtype(dtype):
    """Check if a torch dtype is nvfp4 (FP4)"""
    # Check by dtype name or string representation
    dtype_str = str(dtype).lower()
    # NOTE: cutlass uses int8 for nvfp4 for now, change later.
    return 'nvfp4' in dtype_str or 'fp4' in dtype_str or 'e2m1' in dtype_str or hasattr(dtype, 'nvfp4') or dtype == torch.int8


def is_valid_dtypes_and_scale_factor_vec_size(
    ab_dtype: Type[cutlass.Numeric],
    sf_dtype: Type[cutlass.Numeric],
    sf_vec_size: int,
) -> bool:
    """
    Check if the dtypes and sf_vec_size are valid combinations for block-scaled quantization.

    :param ab_dtype: The data type of the Q/K/V operands (typically Float4E2M1FN for FP4)
    :type ab_dtype: Type[cutlass.Numeric]
    :param sf_dtype: The data type of the scale factor
    :type sf_dtype: Type[cutlass.Numeric]
    :param sf_vec_size: The vector size of the scale factor
    :type sf_vec_size: int

    :return: True if the dtypes and sf_vec_size are valid, False otherwise
    :rtype: bool
    """
    is_valid = True

    # Check valid ab_dtype (for FP4 flash attention, this should be Float4E2M1FN)
    if ab_dtype not in {
        cutlass.Float4E2M1FN,
        cutlass.Float8E5M2,
        cutlass.Float8E4M3FN,
    }:
        is_valid = False

    # Check valid sf_vec_size
    if sf_vec_size not in {16, 32}:
        is_valid = False

    # Check valid sf_dtype
    if sf_dtype not in {cutlass.Float8E8M0FNU, cutlass.Float8E4M3FN}:
        is_valid = False

    # Check valid sf_dtype and sf_vec_size combinations
    if sf_dtype == cutlass.Float8E4M3FN and sf_vec_size == 32:
        is_valid = False
    if ab_dtype in {cutlass.Float8E5M2, cutlass.Float8E4M3FN} and sf_vec_size == 16:
        is_valid = False

    return is_valid


def num_splits_heuristic(total_mblocks, num_SMs, num_n_blocks, max_splits):
    # If num_n_blocks is too small, use 1 split. For example, we never split for hdim = 128 and seqlen_k = 512.
    if num_n_blocks <= 4:
        return 1

    # NOTE: We should revisit this heuristic after persistence is supported for split KV.
    # Sometimes, it's ideal to over-schedule splits for better efficiency.
    return min(num_SMs // total_mblocks, max_splits, num_n_blocks)


def _flash_attn_fwd(
    q: Union[torch.Tensor, cute.Tensor],
    k: Union[torch.Tensor, cute.Tensor],
    v: Union[torch.Tensor, cute.Tensor],
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    softcap: Optional[float] = None,
    window_size_left: Optional[int] = None,
    window_size_right: Optional[int] = None,
    learnable_sink: Optional[torch.Tensor] = None,
    # m_block_size: int = 128,
    # n_block_size: int = 64,
    # num_threads: int = 128,
    m_block_size: int = 128,
    n_block_size: int = 128,
    num_threads: int = 384,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    _compute_capability: Optional[int] = None,
    score_mod: Optional[Callable] = None,
    mask_mod: Optional[Callable] = None,
    block_sparse_tensors: Optional[BlockSparseTensorsTorch] = None,
    return_lse: bool = False,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    aux_tensors: Optional[list[torch.Tensor]] = None,
    mSFQ: Optional[Union[torch.Tensor, cute.Tensor]] = None,  # Scale factor for Q
    mSFK: Optional[Union[torch.Tensor, cute.Tensor]] = None,  # Scale factor for K
    mSFV: Optional[Union[torch.Tensor, cute.Tensor]] = None,  # Scale factor for V
    force_fp4_impl: bool = False, # Test fp4 attn impl under bf16 precision w/o sf
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward pass for FlashAttention.

    Args:
        ...
        score_mod: A callable that takes the attention scores and applies a modification.
        mask_mod: A callable that takes token position information and selectively masks
        block_sparse_tensors: A tuple of tensors used for block sparsity. 
        return_lse: Whether to return the log softmax of the attention scores. If set to True will always calculate
        out: Optional pre-allocated output tensor. If None, will be allocated internally.
        lse: Optional pre-allocated log-sum-exp tensor. If None, will be allocated when needed.
        aux_tensors: Some score_mods will want to read from global aux_tensors. This is how we thread them through to the inner kernel.
    """
    # Handle CUTE tensors - use them directly, no conversion needed
    # Only make contiguous if they are torch tensors
    if not isinstance(q, cute.Tensor):
        q, k, v = [maybe_contiguous(t) for t in (q, k, v)]
    num_head, head_dim = q.shape[-2:]
    # For FP4 packed dtypes (float4_e2m1fn_x2), the last dim is headdim/2.
    # For int8 FP4 buffers (from cute_tensor_like), shape already has full headdim — no correction needed.
    if not isinstance(q, cute.Tensor) and hasattr(torch, 'float4_e2m1fn_x2') and q.dtype == torch.float4_e2m1fn_x2:
        head_dim = head_dim * 2
    if cu_seqlens_q is None:
        batch_size, seqlen_q = q.shape[:2]
        total_q = batch_size * seqlen_q
    else:
        batch_size = cu_seqlens_q.shape[0] - 1
        seqlen_q = None
        total_q = q.shape[0]
    if page_table is not None:
        assert cu_seqlens_k is None, "page_table is not supported with cu_seqlens_k"
        assert page_table.dtype == torch.int32, "page_table must be int32"
        assert page_table.stride(-1) == 1, "page_table must be contiguous in the last dimension"
        max_num_pages_per_seq = page_table.shape[1]
        assert page_table.shape == (batch_size, max_num_pages_per_seq)
        num_pages, page_size = k.shape[:2]
        seqlen_k = num_pages * page_size
    else:
        num_pages, page_size = None, None
        seqlen_k = k.shape[-3]
    num_head_kv = k.shape[-2]
    # For FP4 V (float4_e2m1fn_x2), V is passed as the K-major (= S-major) FP4
    # tensor with shape (b, h, d, s/2): each int8 byte holds two seqlen-adjacent
    # FP4 values (high/low nibble), so the FP4 stride for S is 0.5 byte = 1
    # element. nvfp4_quantize on `v.permute(0,2,3,1).reshape(b*h*d, s)` produces
    # this layout naturally.
    is_fp4_v = (
        not isinstance(v, cute.Tensor)
        and hasattr(torch, "float4_e2m1fn_x2")
        and v.dtype == torch.float4_e2m1fn_x2
    )
    if is_fp4_v:
        # v.shape = (b, h, d, s/2): num_head_kv = h, head_dim_v = d
        assert v.shape[0] == batch_size and v.shape[3] * 2 == seqlen_k, (
            f"FP4 V shape {v.shape} must be (b={batch_size}, h, d, s/2={seqlen_k//2})"
        )
        num_head_kv = v.shape[1]
        head_dim_v = v.shape[2]
    else:
        head_dim_v = v.shape[-1]
    # For shape assertions, use the packed dim (what's actually in the tensor)
    head_dim_packed = q.shape[-1]
    if cu_seqlens_k is None:
        if page_table is None:
            assert k.shape == (batch_size, seqlen_k, num_head_kv, head_dim_packed), f"k shape {k.shape} != expected {(batch_size, seqlen_k, num_head_kv, head_dim_packed)}"
            if not is_fp4_v:
                assert v.shape == (batch_size, seqlen_k, num_head_kv, head_dim_v)
        else:
            assert k.shape == (num_pages, page_size, num_head_kv, head_dim_packed)
            if not is_fp4_v:
                assert v.shape == (num_pages, page_size, num_head_kv, head_dim_v)
    else:
        assert k.shape == (seqlen_k, num_head_kv, head_dim_packed)
        if not is_fp4_v:
            assert v.shape == (seqlen_k, num_head_kv, head_dim_v)
        assert cu_seqlens_k.shape == (batch_size + 1,), (
            "cu_seqlens_k must have shape (batch_size + 1,)"
        )

    if cu_seqlens_q is not None:
        assert cu_seqlens_q.shape == (batch_size + 1,), (
            "cu_seqlens_q must have shape (batch_size + 1,)"
        )
    assert seqused_q is None or seqused_q.shape == (batch_size,), (
        "seqused_q must have shape (batch_size,)"
    )
    assert seqused_k is None or seqused_k.shape == (batch_size,), (
        "seqused_k must have shape (batch_size,)"
    )
    # Check if using FP4 (nvfp4)
    # Handle both torch and CUTE tensors
    is_cute_q = isinstance(q, cute.Tensor)
    if is_cute_q:
        q_dtype = q.element_type
        use_fp4 = q.element_type == cutlass.Float4E2M1FN 
        if not use_fp4:
            k_dtype = k.element_type if isinstance(k, cute.Tensor) else k.dtype
            v_dtype = v.element_type if isinstance(v, cute.Tensor) else v.dtype
            # Allow pure FP8 Q/K/V (no block scale) — routes through
            # FlashAttentionForwardSm100 with kind::f8f6f4. Gated on the env flag
            # so the default behavior (reject pure FP8 → force block-scale) is
            # preserved; set FA4_ALLOW_PURE_FP8_QK=1 to opt into A/B testing.
            allowed_q_dtypes = [cutlass.Float16, cutlass.BFloat16]
            if mSFQ is not None or os.environ.get("FA4_ALLOW_PURE_FP8_QK", "0") == "1":
                allowed_q_dtypes.extend([cutlass.Float8E4M3FN, cutlass.Float8E5M2])
            assert q_dtype in allowed_q_dtypes, (
                "inputs must be float16/bfloat16, or FP8 when block-scaled QK scale factors are provided"
            )
            assert q_dtype == k_dtype, "Q and K must have the same dtype"
            _mixed_ok = os.environ.get("FA4_ALLOW_PURE_FP8_QK", "0") == "1"
            if mSFQ is None and not _mixed_ok:
                assert q_dtype == v_dtype, "inputs must have the same dtype"
            else:
                _allowed_v = [cutlass.Float16, cutlass.BFloat16, cutlass.Float8E4M3FN, cutlass.Float8E5M2,
                              torch.float16, torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2]
                assert v_dtype in _allowed_v, "V must be BF16/FP16 or FP8 when block-scaled QK is enabled"
    else:
        use_fp4 = is_nvfp4_dtype(q.dtype)
        if not use_fp4:
            allowed_q_dtypes = [torch.float16, torch.bfloat16]
            if mSFQ is not None or os.environ.get("FA4_ALLOW_PURE_FP8_QK", "0") == "1":
                allowed_q_dtypes.extend([torch.float8_e4m3fn, torch.float8_e5m2])
            assert q.dtype in allowed_q_dtypes, (
                "inputs must be float16/bfloat16, or FP8 when block-scaled QK scale factors are provided"
            )
            assert q.dtype == k.dtype, "Q and K must have the same dtype"
            _mixed_ok = os.environ.get("FA4_ALLOW_PURE_FP8_QK", "0") == "1"
            if mSFQ is None and not _mixed_ok:
                assert q.dtype == v.dtype, "inputs must have the same dtype"
            else:
                assert v.dtype in [
                    torch.float16,
                    torch.bfloat16,
                    torch.float8_e4m3fn,
                    torch.float8_e5m2,
                ], "V must be BF16/FP16 or FP8 when block-scaled QK is enabled"
    
    # Store is_cute_q for later use
    for t in [cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k]:
        if t is not None:
            assert t.dtype == torch.int32, (
                "cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k must be int32"
            )
            assert t.stride(0) == 1, (
                "cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k must be contiguous"
            )
    if learnable_sink is not None:
        assert learnable_sink.shape == (num_head,)
        assert learnable_sink.dtype == torch.bfloat16, "learnable_sink must be bfloat16"

    # Check CUDA device - CUTE tensors are always on device, torch tensors need .is_cuda check
    assert all(
        t is None or (isinstance(t, cute.Tensor) or t.is_cuda)
        for t in (
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            page_table,
            learnable_sink,
        )
    ), "inputs must be on CUDA device"
    assert num_head % num_head_kv == 0, "num_head must be divisible by num_head_kv"
    assert head_dim <= 256, "head_dim must be less than or equal to 256"
    # Handle both torch and CUTE tensors for element_size
    if is_cute_q:
        # For CUTE tensors, FP4 is 4 bits but packed as 1 byte (2 values per byte)
        # For alignment purposes, use 1 byte for FP4
        if q.element_type == cutlass.Float4E2M1FN:
            q_element_size = 1
        else:
            q_element_size = q.element_type.width // 8
    else:
        q_element_size = q.element_size()
    alignment = 16 // q_element_size
    assert head_dim % alignment == 0, f"head_dim must be divisible by {alignment}"
    assert head_dim_v % alignment == 0, f"head_dim_v must be divisible by {alignment}"
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    if softcap == 0.0:
        softcap = None
    qhead_per_kvhead = num_head // num_head_kv
    if pack_gqa is None:
        pack_gqa = qhead_per_kvhead > 1

    use_blockscaled_impl = use_fp4 or mSFQ is not None or force_fp4_impl

    # Handle both torch and CUTE tensors for dtype and device
    if is_cute_q:
        # For CUTE tensors, we need to get the torch dtype for output allocation
        if q_dtype == cutlass.Float4E2M1FN or use_blockscaled_impl:
            out_torch_dtype = torch.bfloat16
        else:
            out_torch_dtype = cute2torch_dtype_map[q_dtype]
        device = torch.device('cuda')  # CUTE tensors are always on CUDA
    else:
        if use_blockscaled_impl:
            out_torch_dtype = torch.bfloat16
        elif q.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) and v.dtype in (torch.float16, torch.bfloat16):
            # Mixed FP8 QK + BF16/FP16 PV: output follows v_dtype (the widest operand).
            out_torch_dtype = v.dtype
        else:
            out_torch_dtype = q.dtype
        device = q.device
    q_batch_seqlen_shape = (batch_size, seqlen_q) if cu_seqlens_q is None else (total_q,)
    lse_shape = (batch_size, num_head, seqlen_q) if cu_seqlens_q is None else (num_head, total_q)
    if isinstance(q, torch.Tensor):
        requires_grad = q.requires_grad or k.requires_grad or v.requires_grad
    else:
        requires_grad = False # NOTE(Wenxuan): cute tensor has no grad attr, currently hardcode for inference

    if out is None:
        out = torch.empty(
            *q_batch_seqlen_shape, num_head, head_dim_v, dtype=out_torch_dtype, device=device
        )
    else:
        expected_out_shape = (*q_batch_seqlen_shape, num_head, head_dim_v)
        assert out.shape == expected_out_shape, (
            f"out tensor shape {out.shape} does not match expected shape {expected_out_shape}"
        )
        assert out.dtype == out_torch_dtype, (
            f"out tensor dtype {out.dtype} does not match expected dtype {out_torch_dtype}"
        )
        assert out.device == device, (
            f"out tensor device {out.device} does not match input device {device}"
        )
        assert out.is_cuda, "out tensor must be on CUDA device"

    if lse is None:
        lse = (
            torch.empty(lse_shape, dtype=torch.float32, device=device)
            if requires_grad or return_lse
            else None
        )
    elif lse is not None:
        assert lse.shape == lse_shape, (
            f"lse tensor shape {lse.shape} does not match expected shape {lse_shape}"
        )
        assert lse.dtype == torch.float32, (
            f"lse tensor dtype {lse.dtype} does not match expected dtype torch.float32"
        )
        assert lse.device == device, (
            f"lse tensor device {lse.device} does not match input device {device}"
        )
        assert lse.is_cuda, "lse tensor must be on CUDA device"

    # Get CUTE dtype - use directly if CUTE tensor, otherwise convert from torch dtype
    if is_cute_q:
        dtype = q.element_type
    else:
        dtype = torch2cute_dtype_map.get(q.dtype, cutlass.Float16) if not use_fp4 else cutlass.Float16


    compute_capability = (
        _get_device_capability()
        if _compute_capability is None
        else _compute_capability
    )

    assert compute_capability in [9, 10], "Unsupported compute capability. Supported: 9.x, 10.x"

    use_block_sparsity = block_sparse_tensors is not None

    if mask_mod is None:
        if causal:
            window_size_right = 0
        local = window_size_left is not None or window_size_right is not None
        if window_size_left is not None or window_size_right is not None:
            if window_size_left is None and window_size_right == 0:
                causal, local = True, False
            else:
                causal, local = False, True
    else:
        causal, local = False, False

    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    if compute_capability == 9:  # TODO: tune block size according to hdim.
        if head_dim == head_dim_v == 128 and not causal and not local and not use_block_sparsity:
            n_block_size = 192
    if compute_capability == 10:
        # TODO: fix the varlen case
        if (
            pack_gqa
            and (128 % qhead_per_kvhead != 0)
            or (cu_seqlens_q is not None or seqused_q is not None)
        ):
            pack_gqa = False
        # TODO: fix GQA + SplitKV + non-varlen
        if pack_gqa and num_splits != 1 and cu_seqlens_q is None:
            pack_gqa = False

    if num_splits < 1:
        max_seqlen_k = seqlen_k if cu_seqlens_k is None else (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).max().item()
        max_seqlen_q = seqlen_q if cu_seqlens_q is None else (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item()
        seqlen_q_packgqa = max_seqlen_q * qhead_per_kvhead
        seqlen_k_loaded = max_seqlen_k if not local else max(0, min(max_seqlen_k, window_size_right + window_size_left + 1 + m_block_size))
        num_n_blocks = (seqlen_k_loaded + n_block_size - 1) // n_block_size
        num_m_blocks = (seqlen_q_packgqa + m_block_size - 1) // m_block_size
        total_mblocks = batch_size * num_head_kv * num_m_blocks
        num_splits = num_splits_heuristic(
            total_mblocks,
            torch.cuda.get_device_properties(device).multi_processor_count,
            num_n_blocks,
            128,
        )

    is_split_kv = num_splits > 1
    if is_split_kv:
        out_partial = torch.empty(num_splits, *q_batch_seqlen_shape, num_head, head_dim_v, dtype=torch.float32, device=device)
        lse_partial = torch.empty(num_splits, *lse_shape, dtype=torch.float32, device=device)

    # hash score and mask mods for compile cache
    score_mod_hash = utils.hash_callable(score_mod) if score_mod is not None else False
    mask_mod_hash = utils.hash_callable(mask_mod) if mask_mod is not None else False

    if softcap is not None:
        assert score_mod is None, "softcap and score_mod cannot be used together"
        score_mod = utils.create_softcap_scoremod(softcap)

    is_varlen = (
        cu_seqlens_q is not None
        or cu_seqlens_k is not None
        or seqused_q is not None
        or seqused_k is not None
    )
    if score_mod is not None:
        if is_varlen:
            raise NotImplementedError(
                "score_mod with aux_tensors is not yet supported for varlen sequences. This will be fixed in a future PR."
            )

    if mask_mod is not None:
        if not use_block_sparsity:
            raise NotImplementedError(
                "mask_mod requires the use of block sparsity. This will be fixed in a future PR."
            )
        if is_varlen:
            raise NotImplementedError(
                "mask_mod with aux_tensors is not yet supported for varlen sequences. This will be fixed in a future PR."
            )
        if pack_gqa:
            raise NotImplementedError(
                "mask_mod with aux_tensors is not yet supported with pack_gqa=True. This will be fixed in a future PR."
            )

    if use_block_sparsity:
        if is_varlen:
            raise NotImplementedError(
                "Block sparsity is not yet supported for varlen sequences. This will be fixed in a future PR."
            )
        if pack_gqa:
            raise NotImplementedError(
                "Block sparsity is not yet supported with pack_gqa=True. This will be fixed in a future PR."
            )
        if is_split_kv:
            raise NotImplementedError(
                "Block sparsity is not yet supported with SplitKV. TODO: partition sparse block lists per split."
            )

    # Block-scaled QK ab/sf dtype must be in the compile key — otherwise NVFP4 and
    # MXFP8 share the same key and the second mode silently reuses the kernel
    # compiled for the first (was producing NaN/inf for whichever was second).
    # We can derive the QK ab dtype from the Q tensor's element_type here (cute.Tensor
    # path) and from the FP4 hint (torch path); for the SF dtype, derive from mSFQ's
    # element_type if it's already a cute.Tensor, else infer from sf_vec_size which we
    # also derive from the SF tensor shape.
    if isinstance(q, cute.Tensor):
        _key_qk_ab_dtype = q.element_type
    elif use_fp4:
        _key_qk_ab_dtype = cutlass.Float4E2M1FN
    else:
        _key_qk_ab_dtype = torch2cute_dtype_map.get(q.dtype, cutlass.BFloat16)

    if mSFQ is None:
        _key_sf_dtype = None
    elif isinstance(mSFQ, cute.Tensor):
        _key_sf_dtype = mSFQ.element_type
    else:
        # The torch underlying tensor for an FP8/E4M3/E8M0 SF is plain int8 with no
        # dtype hint — we have to infer from the AB dtype which is already known.
        # MXFP8 (Float8E4M3FN/E5M2 ab) uses E8M0 SF; NVFP4 (Float4E2M1FN ab) uses E4M3.
        if _key_qk_ab_dtype in (cutlass.Float8E4M3FN, cutlass.Float8E5M2):
            _key_sf_dtype = cutlass.Float8E8M0FNU
        else:
            _key_sf_dtype = cutlass.Float8E4M3FN

    # v_dtype key: distinguish mixed FP8 QK + BF16 PV vs pure FP8 QK + FP8 PV.
    if isinstance(v, cute.Tensor):
        _key_v_dtype = v.element_type
    else:
        _key_v_dtype = torch2cute_dtype_map.get(v.dtype, cutlass.BFloat16)
    compile_key = (
        dtype,
        _key_v_dtype,
        head_dim,
        head_dim_v,
        qhead_per_kvhead,
        causal,
        score_mod_hash,
        mask_mod_hash,
        use_block_sparsity,
        len(aux_tensors) if aux_tensors is not None else 0,
        lse is None,
        cu_seqlens_q is None,
        cu_seqlens_k is None,
        seqused_q is None,
        seqused_k is None,
        page_table is not None,
        window_size_left is not None,
        window_size_right is not None,
        learnable_sink is not None,
        m_block_size,
        n_block_size,
        num_threads,
        is_split_kv,
        pack_gqa,
        compute_capability,
        page_size not in [None, 128],  # paged KV non-TMA
        use_fp4,  # Include FP4 flag in compile key
        mSFQ is not None,  # Include block-scaled path activation
        mSFQ is not None,  # Include scale factor flags
        mSFK is not None,
        mSFV is not None,
        force_fp4_impl,
        _key_qk_ab_dtype,  # NVFP4 (Float4E2M1FN) vs MXFP8 (Float8E4M3FN/E5M2)
        _key_sf_dtype,     # E4M3 (NVFP4) vs E8M0 (MXFP8) — was previously not in the
                           # key; both modes shared a slot and the second silently
                           # reused the kernel compiled for the first.
        local,
    )
    fp4_qk = use_fp4 and not is_cute_q
    # FP4 V also needs the make_ptr path: dlpack reports half-headdim shape for
    # float4_e2m1fn_x2, but the kernel needs to know the full headdim (and the
    # K-major stride) to build SFV's TMA descriptor correctly.
    fp4_v = (
        use_fp4 and not isinstance(v, cute.Tensor)
        and hasattr(torch, "float4_e2m1fn_x2")
        and v.dtype == torch.float4_e2m1fn_x2
    )
    # Compute q_shape, k_shape and qk_ab_dtype for pointer-based Q/K path (used by FP4 kernel)
    if fp4_qk:
        q_ptr_shape = tuple(int(s) for s in (*q.shape[:-1], q.shape[-1] * 2))
        k_ptr_shape = tuple(int(s) for s in (*k.shape[:-1], k.shape[-1] * 2))
        qk_ab_dtype = cutlass.Float4E2M1FN
    elif is_cute_q:
        q_ptr_shape = tuple(int(s) for s in q.shape)
        k_ptr_shape = tuple(int(s) for s in k.shape)
        qk_ab_dtype = q.element_type
    else:
        q_ptr_shape = tuple(int(s) for s in q.shape)
        k_ptr_shape = tuple(int(s) for s in k.shape)
        qk_ab_dtype = torch2cute_dtype_map.get(q.dtype, cutlass.BFloat16)
    if fp4_v:
        # K-major V FP4 tensor: torch shape (b, h, d, s/2) packs 2 seqlen-adjacent
        # FP4 per byte. Convert to logical (b, s, h, d) for the kernel.
        b_v, h_v, d_v, s_half = (int(x) for x in v.shape)
        v_ptr_shape = (b_v, s_half * 2, h_v, d_v)
    else:
        v_ptr_shape = ()
    if compile_key not in _flash_attn_fwd.compile_cache:
        (
            cu_seqlens_q_tensor,
            cu_seqlens_k_tensor,
            seqused_q_tensor,
            seqused_k_tensor,
            learnable_sink_tensor,
        ) = [
            to_cute_tensor(t, assumed_align=4, leading_dim=0)
            if t is not None
            else None
            for t in (cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, learnable_sink)
        ]
        page_table_tensor = (
            to_cute_tensor(page_table, assumed_align=4, leading_dim=1)
            if page_table is not None
            else None
        )
        # Extract scale factor dtype and vec_size before converting tensors
        sf_dtype = None
        sf_vec_size = 16  # Default for FP4
        if use_blockscaled_impl and mSFQ is not None:
            if isinstance(mSFQ, cute.Tensor):
                sf_dtype = mSFQ.element_type
            else:
                # Convert torch dtype to CUTLASS dtype
                sf_dtype = torch2cute_dtype_map.get(mSFQ.dtype, cutlass.Float8E4M3FN)
                if (
                    sf_dtype == cutlass.Float8E4M3FN
                    and qk_ab_dtype in {cutlass.Float8E4M3FN, cutlass.Float8E5M2}
                    and mSFQ.dtype not in {torch.float8_e4m3fn, torch.float8_e5m2}
                ):
                    # Float8E8M0FNU scale tensors currently round-trip through the benchmark
                    # as byte-backed torch tensors, so infer MXFP8 scale metadata from Q/K dtype.
                    sf_dtype = cutlass.Float8E8M0FNU
            # Set default sf_vec_size based on sf_dtype
            if sf_dtype == cutlass.Float8E4M3FN:
                sf_vec_size = 16
            elif sf_dtype == cutlass.Float8E8M0FNU:
                sf_vec_size = 32
            else:
                raise ValueError(f"Invalid scale factor dtype: {sf_dtype}")
        
        # For torch FP4 tensors: use make_ptr path (kernel builds tensor from pointer + shape)
        # For cute tensors or non-FP4 torch tensors: use to_cute_tensor as before
        if fp4_qk:
            from cutlass.cute.runtime import make_ptr
            q_tensor = make_ptr(qk_ab_dtype, 0, cute.AddressSpace.gmem, assumed_align=16)
            k_tensor = make_ptr(qk_ab_dtype, 0, cute.AddressSpace.gmem, assumed_align=16)
        else:
            q_tensor = to_cute_tensor(q)
            k_tensor = to_cute_tensor(k)
        if fp4_v:
            from cutlass.cute.runtime import make_ptr
            v_tensor = make_ptr(cutlass.Float4E2M1FN, 0, cute.AddressSpace.gmem, assumed_align=16)
        else:
            v_tensor = to_cute_tensor(v)
        o_tensor = to_cute_tensor(out if not is_split_kv else out_partial)
        # Pass through scale factor tensors when using the SM100 block-scaled kernel.
        mSFQ_tensor = mSFK_tensor = mSFV_tensor = None
        if use_blockscaled_impl:
            mSFQ_tensor = to_cute_tensor(mSFQ, leading_dim=3, assumed_align=16) if mSFQ is not None else None
            mSFK_tensor = to_cute_tensor(mSFK, leading_dim=3, assumed_align=16) if mSFK is not None else None
            mSFV_tensor = to_cute_tensor(mSFV, leading_dim=3, assumed_align=16) if mSFV is not None else None
        if is_split_kv:
            lse_tensor = to_cute_tensor(lse_partial, assumed_align=4)
        elif lse is not None:
            lse_tensor = to_cute_tensor(lse, assumed_align=4)
        else:
            lse_tensor = None

        sparse_tensors = None
        if block_sparse_tensors is not None:
            if seqlen_q is None:
                raise ValueError("Block sparsity requires fixed-length sequences (seqlen_q must be known).")
            expected_count_shape, expected_index_shape = get_block_sparse_expected_shapes(
                batch_size, num_head, seqlen_q, seqlen_k,
                m_block_size, n_block_size, compute_capability,
            )
            compile_time_normalized = normalize_block_sparse_tensors(
                block_sparse_tensors,
                expected_count_shape=expected_count_shape,
                expected_index_shape=expected_index_shape,
            )
            sparse_tensors = to_cute_block_sparse_tensors(compile_time_normalized)

        cute_aux_tensors = None
        if aux_tensors is not None:
            cute_aux_tensors = [to_cute_tensor(buf, assumed_align=None, fully_dynamic=True) for buf in aux_tensors]

        if compute_capability == 9:
            assert page_table is None, "paged KV not supported on SM 9.0"
            assert not is_split_kv, "SplitKV not supported on SM 9.0"
            # fa_fwd = FlashAttentionForwardSm80(
            fa_fwd = FlashAttentionForwardSm90(
                dtype,
                head_dim,
                head_dim_v,
                qhead_per_kvhead,
                is_causal=causal,
                is_local=local,
                pack_gqa=pack_gqa,
                tile_m=m_block_size,
                tile_n=n_block_size,
                # num_stages=1,
                num_stages=2,
                num_threads=num_threads,
                Q_in_regs=False,
                intra_wg_overlap=True,
                mma_pv_is_rs=True,
                mask_mod=mask_mod,
                score_mod=score_mod,
                has_aux_tensors=aux_tensors is not None,
            )
        elif compute_capability == 10:
            if use_blockscaled_impl:
                # Use the SM100 block-scaled attention kernel for NVFP4 / MXFP8 QK.
                if sf_dtype is None:
                    sf_dtype = cutlass.Float8E4M3FN  # Default scale factor dtype for FP4
                # Validate dtype and scale factor combinations
                ab_dtype = q_tensor.value_type if fp4_qk else q_tensor.element_type
                if not force_fp4_impl and not is_valid_dtypes_and_scale_factor_vec_size(ab_dtype, sf_dtype, sf_vec_size):
                    raise ValueError(
                        f"Invalid dtype combination: ab_dtype={ab_dtype}, "
                        f"sf_dtype={sf_dtype}, sf_vec_size={sf_vec_size}"
                    )
                fa_fwd = FlashAttentionForwardSm100FP4(
                    head_dim,
                    head_dim_v,
                    qhead_per_kvhead=qhead_per_kvhead,
                    is_causal=causal,
                    is_local=local,
                    is_split_kv=is_split_kv,
                    pack_gqa=pack_gqa,
                    m_block_size=m_block_size,
                    n_block_size=n_block_size,
                    is_persistent=not causal
                        and not local
                        and cu_seqlens_q is None
                        and seqused_q is None
                        and not is_split_kv,
                    score_mod=score_mod,
                    mask_mod=mask_mod,
                    has_aux_tensors=aux_tensors is not None,
                    paged_kv_non_tma=page_size not in [None, 128],
                    is_varlen_q=cu_seqlens_q is not None
                        or seqused_q is not None,
                    sf_dtype=sf_dtype,
                    sf_vec_size=sf_vec_size,
                )
            else:
                import os as _os
                _requested_disable_2cta = _os.environ.get("FA_DISABLE_2CTA", "0") == "1"
                _head_dim_padded = int(math.ceil(head_dim / 16) * 16)
                _head_dim_v_padded = int(math.ceil(head_dim_v / 16) * 16)
                _use_2cta_instrs = (
                    compute_capability // 10 in [10, 11]
                    and not _requested_disable_2cta
                    and not causal
                    and not local
                    and not is_split_kv
                    and cu_seqlens_q is None
                    and seqused_q is None
                    and page_size in [None, 128]
                    and _head_dim_padded in [128, 192]
                    and _head_dim_v_padded == 128
                    and seqlen_q_packgqa > 2 * m_block_size
                    and (m_block_size % qhead_per_kvhead == 0 or not pack_gqa)
                )
                _use_clc = (
                    not (cu_seqlens_q is None and not causal and not local)
                    and not (cu_seqlens_q is not None and qhead_per_kvhead == 1)
                )
                fa_fwd = FlashAttentionForwardSm100(
                    head_dim,
                    head_dim_v,
                    qhead_per_kvhead=qhead_per_kvhead,
                    is_causal=causal,
                    is_local=local,
                    is_split_kv=is_split_kv,
                    pack_gqa=pack_gqa,
                    m_block_size=m_block_size,
                    n_block_size=n_block_size,
                    is_persistent=not causal
                        and not local
                        and cu_seqlens_q is None
                        and seqused_q is None
                        and not is_split_kv,
                    score_mod=score_mod,
                    mask_mod=mask_mod,
                    has_aux_tensors=aux_tensors is not None,
                    paged_kv_non_tma=page_size not in [None, 128],
                    is_varlen_q=cu_seqlens_q is not None
                        or seqused_q is not None,
                    use_2cta_instrs=_use_2cta_instrs,
                    use_clc_scheduler=_use_clc,
                )
        else:
            raise ValueError(
                f"Unsupported compute capability: {compute_capability}. Supported: 9.x, 10.x"
            )
        # TODO: check @can_implement
        fake_stream = cute.runtime.make_fake_stream()
        compile_args = [
            fa_fwd,
            q_tensor,
            k_tensor,
            v_tensor,
            o_tensor,
            lse_tensor,
            softmax_scale,
            # current_stream,
            fake_stream,
            cu_seqlens_q_tensor,
            cu_seqlens_k_tensor,
            seqused_q_tensor,
            seqused_k_tensor,
            page_table_tensor,
            window_size_left,
            window_size_right,
            learnable_sink_tensor,
            sparse_tensors,
            cute_aux_tensors,
        ]
        # Add scale factor tensors if using the block-scaled SM100 kernel
        if use_blockscaled_impl:
            compile_args.extend([mSFQ_tensor, mSFK_tensor, mSFV_tensor])
        # Add q/k shapes for the block-scaled kernel (it always builds tensors from pointer + shape)
        if use_blockscaled_impl:
            if fp4_qk:
                sym_q_shape = tuple(cutlass.Int32(0) for _ in q_ptr_shape)
                sym_k_shape = tuple(cutlass.Int32(0) for _ in k_ptr_shape)
            else:
                sym_q_shape = tuple(cutlass.Int32(0) for _ in range(len(q_tensor.shape)))
                sym_k_shape = tuple(cutlass.Int32(0) for _ in range(len(k_tensor.shape)))
            compile_args.extend([sym_q_shape, sym_k_shape])
            if fp4_v:
                sym_v_shape = tuple(cutlass.Int32(0) for _ in v_ptr_shape)
                compile_args.append(sym_v_shape)
        _flash_attn_fwd.compile_cache[compile_key] = cute.compile(
            *compile_args,
            options="--enable-tvm-ffi",
        )
        # dump_kernel_attributes(_flash_attn_fwd.compile_cache[compile_key])

    # Expand block sparse tensors to match actual head count (may be broadcast from 1)
    normalized_block_sparse_tensors = None
    if block_sparse_tensors is not None:
        expected_count_shape, expected_index_shape = get_block_sparse_expected_shapes(
            batch_size, num_head, seqlen_q, seqlen_k,
            m_block_size, n_block_size, compute_capability,
        )
        normalized_block_sparse_tensors = normalize_block_sparse_tensors(
            block_sparse_tensors,
            expected_count_shape=expected_count_shape,
            expected_index_shape=expected_index_shape,
        )
    if fp4_qk:
        from cutlass.cute.runtime import make_ptr as _make_ptr
        q_data_ptr = q.data_ptr() if hasattr(q, 'data_ptr') else q.iterator.data_ptr
        k_data_ptr = k.data_ptr() if hasattr(k, 'data_ptr') else k.iterator.data_ptr
        q_call = _make_ptr(qk_ab_dtype, q_data_ptr, cute.AddressSpace.gmem, assumed_align=16)
        k_call = _make_ptr(qk_ab_dtype, k_data_ptr, cute.AddressSpace.gmem, assumed_align=16)
    else:
        q_call = q
        k_call = k
    if fp4_v:
        from cutlass.cute.runtime import make_ptr as _make_ptr
        v_data_ptr = v.data_ptr()
        v_call = _make_ptr(cutlass.Float4E2M1FN, v_data_ptr, cute.AddressSpace.gmem, assumed_align=16)
    else:
        v_call = v
    call_args = [
        q_call,
        k_call,
        v_call,
        out if not is_split_kv else out_partial,
        lse_partial if is_split_kv else lse,
        softmax_scale,
        current_stream,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        page_table,
        window_size_left,
        window_size_right,
        learnable_sink,
        normalized_block_sparse_tensors,
        aux_tensors,
    ]

    # Add scale factor tensors if using the block-scaled SM100 kernel
    if use_blockscaled_impl:
        call_args.extend([mSFQ, mSFK, mSFV])
    # Add q/k shapes for the block-scaled kernel
    if use_blockscaled_impl:
        call_args.extend([q_ptr_shape, k_ptr_shape])
        if fp4_v:
            call_args.append(v_ptr_shape)

    _flash_attn_fwd.compile_cache[compile_key](*call_args)

    if is_split_kv:
        _flash_attn_fwd_combine(
            out_partial,
            lse_partial.transpose(-1, -2),
            out,
            lse.transpose(-1, -2) if lse is not None else None,
            cu_seqlens_q,
            seqused_q,
        )
    return out, lse


_flash_attn_fwd.compile_cache = {}


def _flash_attn_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    softcap: float = 0.0,
    m_block_size: int = 64,
    n_block_size: int = 128,
    num_threads: int = 256,
    pack_gqa: bool = False,
    num_stages_Q: int = 2,
    num_stages_dO: int = 2,
    SdP_swapAB: bool = False,
    dKV_swapAB: bool = False,
    dQ_swapAB: bool = False,
    AtomLayoutMSdP: int = 2,
    AtomLayoutNdKV: int = 2,
    AtomLayoutMdQ: int = 2,
    V_in_regs: bool = False,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    compute_capability = _get_device_capability()
    assert compute_capability in [9, 10], "Unsupported compute capability. Supported: 9.x, 10.x"

    if compute_capability == 9:
        m_block_size = 80 if not causal else 64
        n_block_size = 128
        num_stages_Q = 2
        num_stages_dO = 2
        num_stages_PdS = 2
        SdP_swapAB = True
        dKV_swapAB = False
        dQ_swapAB = not causal
        AtomLayoutMSdP = 1
        AtomLayoutNdKV = 2
        AtomLayoutMdQ = 1
        cluster_size = 1
    else:
        m_block_size = 128
        n_block_size = 128
        dQ_swapAB = False
        dKV_swapAB = False
        AtomLayoutMdQ = 1
        AtomLayoutNdKV = 1
        # TODO: support cluster size 2
        cluster_size = 1
    q, k, v, out, dout, lse, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k = [
        maybe_contiguous(t)
        for t in (q, k, v, out, dout, lse, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k)
    ]
    num_head, head_dim = q.shape[-2:]
    if cu_seqlens_q is None:
        batch_size, seqlen_q = q.shape[:2]
        total_q = batch_size * seqlen_q
    else:
        batch_size = cu_seqlens_q.shape[0] - 1
        seqlen_q = None
        total_q = q.shape[0]

    if cu_seqlens_k is None:
        batch_size, seqlen_k = k.shape[:2]
        total_k = batch_size * seqlen_k
    else:
        batch_size = cu_seqlens_k.shape[0] - 1
        seqlen_k = None
        total_k = k.shape[0]

    num_head_kv = k.shape[-2]
    head_dim_v = v.shape[-1]

    if cu_seqlens_k is None:
        assert k.shape == (batch_size, seqlen_k, num_head_kv, head_dim)
        assert v.shape == (batch_size, seqlen_k, num_head_kv, head_dim_v)
    else:
        assert k.shape == (total_k, num_head_kv, head_dim)
        assert v.shape == (total_k, num_head_kv, head_dim_v)
        assert cu_seqlens_k.shape == (batch_size + 1,), (
            "cu_seqlens_k must have shape (batch_size + 1,)"
        )

    if cu_seqlens_q is not None:
        assert cu_seqlens_q.shape == (batch_size + 1,), (
            "cu_seqlens_q must have shape (batch_size + 1,)"
        )

        assert out.shape == (total_q, num_head, head_dim_v)
        assert dout.shape == (total_q, num_head, head_dim_v)
        assert lse.shape == (num_head, total_q), "lse must have shape (num_head, total_q)"
    else:
        assert out.shape == (batch_size, seqlen_q, num_head, head_dim_v)
        assert dout.shape == (batch_size, seqlen_q, num_head, head_dim_v)
        assert lse.shape == (batch_size, num_head, seqlen_q), (
            "lse must have shape (batch_size, num_head, seqlen_q)"
        )

    assert q.dtype in [torch.float16, torch.bfloat16], "inputs must be float16 or bfloat16"
    assert q.dtype == k.dtype == v.dtype == out.dtype == dout.dtype, (
        "inputs must have the same dtype"
    )
    for t in [cu_seqlens_q, cu_seqlens_k]:
        if t is not None:
            assert t.dtype == torch.int32, "cu_seqlens_q, cu_seqlens_k must be int32"
    assert lse.dtype == torch.float32, "lse must be float32"
    assert all(
        t is None or t.is_cuda for t in (q, k, v, out, dout, lse, cu_seqlens_q, cu_seqlens_k)
    ), "inputs must be on CUDA device"
    assert num_head % num_head_kv == 0, "num_head must be divisible by num_head_kv"
    assert head_dim <= 256, "head_dim must be less than or equal to 256"
    alignment = 16 // q.element_size()
    assert head_dim % alignment == 0, f"head_dim must be divisible by {alignment}"
    assert head_dim_v % alignment == 0, f"head_dim_v must be divisible by {alignment}"
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    qhead_per_kvhead = num_head // num_head_kv
    if pack_gqa is None:
        pack_gqa = qhead_per_kvhead > 1
    if compute_capability == 10:
        pack_gqa = False # override for now

    device = q.device
    # TODO: check if this is the right rounding
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    head_dim_rounded = (head_dim + 32 - 1) // 32 * 32

    if cu_seqlens_q is None:
        seqlen_q_rounded = (seqlen_q + m_block_size - 1) // m_block_size * m_block_size
        dq_accum = torch.empty(
            batch_size,
            num_head,
            seqlen_q_rounded * head_dim_rounded,
            dtype=torch.float32,
            device=device,
        )
        dpsum = torch.empty(
            batch_size, num_head, seqlen_q_rounded, dtype=torch.float32, device=device
        )
        lse_log2 = torch.empty(
            batch_size, num_head, seqlen_q_rounded, dtype=torch.float32, device=device
        )
    else:
        total_q_rounded_padded = (
            (total_q + cu_seqlens_q.shape[0] * m_block_size - 1) // m_block_size * m_block_size
        )
        dq_accum = torch.empty(
            num_head, total_q_rounded_padded * head_dim_rounded, dtype=torch.float32, device=device
        )
        dpsum = torch.empty(num_head, total_q_rounded_padded, dtype=torch.float32, device=device)
        lse_log2 = torch.empty(num_head, total_q_rounded_padded, dtype=torch.float32, device=device)

    if qhead_per_kvhead > 1:
        head_dim_v_rounded = (head_dim_v + 32 - 1) // 32 * 32
        if cu_seqlens_k is None:
            seqlen_k_rounded = (seqlen_k + n_block_size - 1) // n_block_size * n_block_size
            num_n_blocks = seqlen_k_rounded // n_block_size
            if cluster_size == 2 and num_n_blocks % cluster_size != 0:
                seqlen_k_rounded = seqlen_k_rounded + n_block_size
            dk_accum = torch.zeros(
                batch_size,
                num_head_kv,
                seqlen_k_rounded * head_dim_rounded,
                dtype=torch.float32,
                device=device,
            )
            dv_accum = torch.zeros(
                batch_size,
                num_head_kv,
                seqlen_k_rounded * head_dim_v_rounded,
                dtype=torch.float32,
                device=device,
            )
        else:
            total_k_rounded_padded = (
                (total_k + cu_seqlens_k.shape[0] * n_block_size - 1) // n_block_size * n_block_size
            )
            num_n_blocks = total_k_rounded_padded // n_block_size
            if cluster_size == 2 and num_n_blocks % cluster_size != 0:
                total_k_rounded_padded = total_k_rounded_padded + n_block_size
            dk_accum = torch.zeros(
                num_head_kv,
                total_k_rounded_padded * head_dim_rounded,
                dtype=torch.float32,
                device=device,
            )
            dv_accum = torch.zeros(
                num_head_kv,
                total_k_rounded_padded * head_dim_v_rounded,
                dtype=torch.float32,
                device=device,
            )

    dtype = torch2cute_dtype_map[q.dtype]
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    if deterministic:
        dQ_semaphore = torch.zeros(batch_size, num_head, seqlen_q_rounded // m_block_size, 1, dtype=torch.int32, device="cuda")
    else:
        dQ_semaphore = None

    if deterministic and qhead_per_kvhead > 1:
        dK_semaphore = torch.zeros(batch_size, num_head_kv, seqlen_k_rounded // n_block_size, 2, dtype=torch.int32, device="cuda")
        dV_semaphore = torch.zeros(batch_size, num_head_kv, seqlen_k_rounded // n_block_size, 2, dtype=torch.int32, device="cuda")
    else:
        dK_semaphore = None
        dV_semaphore = None

    # Preprocess kernel: compute (o * dout).sum(dim=-1), lse * log2_e, and zero out dq_accum.
    compile_key_pre = (compute_capability, dtype, head_dim_v, m_block_size, num_threads)
    if compile_key_pre not in _flash_attn_bwd.compile_cache_pre:
        o_tensor, do_tensor = [to_cute_tensor(t) for t in (out, dout)]
        dq_accum_tensor, dpsum_tensor, lse_log2_tensor = [
            to_cute_tensor(t) for t in (dq_accum, dpsum, lse_log2)
        ]
        lse_tensor = to_cute_tensor(lse, assumed_align=4)
        cu_seqlens_q_tensor, seqused_q_tensor = [
            to_cute_tensor(t, assumed_align=4) if t is not None else None
            for t in (cu_seqlens_q, seqused_q)
        ]
        fa_bwd_pre = FlashAttentionBackwardPreprocess(
            dtype,
            head_dim_v,
            m_block_size,
            num_threads=num_threads,
        )
        # TODO: check @can_implement
        _flash_attn_bwd.compile_cache_pre[compile_key_pre] = cute.compile(
            fa_bwd_pre,
            o_tensor,
            do_tensor,
            dpsum_tensor,
            lse_tensor,
            lse_log2_tensor,
            dq_accum_tensor,
            cu_seqlens_q_tensor,
            seqused_q_tensor,
            current_stream,
            options="--enable-tvm-ffi",
        )
    _flash_attn_bwd.compile_cache_pre[compile_key_pre](
        out,
        dout,
        dpsum,
        lse,
        lse_log2,
        dq_accum,
        cu_seqlens_q,
        seqused_q,
        current_stream,
    )

    # Backward kernel: compute dk, dv, dq_accum.
    if compute_capability == 9:
        compile_key = (
            compute_capability,
            dtype,
            head_dim,
            head_dim_v,
            qhead_per_kvhead,
            causal,
            softcap != 0.0,
            m_block_size,
            n_block_size,
            num_threads,
            pack_gqa,
            num_stages_Q,
            num_stages_dO,
            SdP_swapAB,
            dKV_swapAB,
            dQ_swapAB,
            AtomLayoutMSdP,
            AtomLayoutNdKV,
            AtomLayoutMdQ,
            V_in_regs,
        )
    else:
        compile_key = (
            compute_capability,
            dtype,
            head_dim,
            head_dim_v,
            qhead_per_kvhead,
            causal,
            softcap != 0.0,
            m_block_size,
            n_block_size,
            num_threads,
            pack_gqa,
            cluster_size,
        )
    num_threads = 384
    if compile_key not in _flash_attn_bwd.compile_cache:
        q_tensor, k_tensor, v_tensor, do_tensor, dq_tensor, dk_tensor, dv_tensor = [
            to_cute_tensor(t) for t in (q, k, v, dout, dq, dk, dv)
        ]
        dq_accum_tensor, dpsum_tensor, lse_log2_tensor = [
            to_cute_tensor(t) for t in (dq_accum, dpsum, lse_log2)
        ]
        if qhead_per_kvhead > 1:
            dk_accum_tensor, dv_accum_tensor = [
                to_cute_tensor(t) for t in (dk_accum, dv_accum)
            ]
        cu_seqlens_q_tensor, cu_seqlens_k_tensor, seqused_q_tensor, seqused_k_tensor = [
            to_cute_tensor(t, assumed_align=4) if t is not None else None
            for t in (cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k)
        ]
        dQ_semaphore_tensor, dK_semaphore_tensor, dV_semaphore_tensor = [
            utils.convert_from_dlpack_leading_static(t.detach(), leading_dim=3, alignment=4, stride_order=t.dim_order())
            if t is not None else None
            for t in (dQ_semaphore, dK_semaphore, dV_semaphore)
        ]
        fa_bwd_sm80 = FlashAttentionBackwardSm80(
            dtype,
            head_dim,
            head_dim_v,
            qhead_per_kvhead,
            m_block_size,
            n_block_size,
            num_stages_Q,
            num_stages_dO,
            num_threads,
            pack_gqa,
            causal,
            SdP_swapAB,
            dKV_swapAB,
            dQ_swapAB,
            AtomLayoutMSdP,
            AtomLayoutNdKV,
            AtomLayoutMdQ,
            V_in_regs=V_in_regs,
        )
        if compute_capability == 9:
            fa_bwd_obj = FlashAttentionBackwardSm90(
                dtype,
                head_dim,
                head_dim_v,
                qhead_per_kvhead,
                causal,
                m_block_size,
                n_block_size,
                num_stages_Q,
                num_stages_dO,
                num_stages_PdS,
                SdP_swapAB,
                dKV_swapAB,
                dQ_swapAB,
                AtomLayoutMSdP,
                AtomLayoutNdKV,
                AtomLayoutMdQ,
                num_threads,
                V_in_regs=V_in_regs,
            )
        else:
            fa_bwd_obj = FlashAttentionBackwardSm100(
                head_dim,
                head_dim_v,
                is_causal=causal,
                qhead_per_kvhead=qhead_per_kvhead,
                # tile_m=m_block_size,
                # tile_n=n_block_size,
                cluster_size=cluster_size,
                # cluster_size=1,
            )
        # TODO: check @can_implement
        _flash_attn_bwd.compile_cache[compile_key] = cute.compile(
            fa_bwd_obj,
            q_tensor,
            k_tensor,
            v_tensor,
            do_tensor,
            lse_log2_tensor,
            dpsum_tensor,
            dq_accum_tensor,
            dk_tensor if qhead_per_kvhead == 1 else dk_accum_tensor,
            dv_tensor if qhead_per_kvhead == 1 else dv_accum_tensor,
            softmax_scale,
            current_stream,
            cu_seqlens_q_tensor,
            cu_seqlens_k_tensor,
            seqused_q_tensor,
            seqused_k_tensor,
            None,  # softcap - not yet supported in backward
            window_size_left,
            window_size_right,
            dQ_semaphore_tensor,
            dK_semaphore_tensor,
            dV_semaphore_tensor,
            options="--enable-tvm-ffi",
        )
    _flash_attn_bwd.compile_cache[compile_key](
        q,
        k,
        v,
        dout,
        lse_log2,
        dpsum,
        dq_accum,
        dk if qhead_per_kvhead == 1 else dk_accum,
        dv if qhead_per_kvhead == 1 else dv_accum,
        softmax_scale,
        current_stream,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        None,  # softcap - not yet supported in backward
        window_size_left,
        window_size_right,
        dQ_semaphore,
        dK_semaphore,
        dV_semaphore,
    )

    num_threads = 256 if compute_capability == 9 else 128
    # Postprocess kernel: convert dq_accum from float32 to dq in bf16/fp16
    compile_key_post = (dtype, head_dim, m_block_size, num_threads, AtomLayoutMdQ, dQ_swapAB)
    if compile_key_post not in _flash_attn_bwd.compile_cache_post:
        dq_accum_tensor = to_cute_tensor(dq_accum)
        dq_tensor = to_cute_tensor(dq)
        cu_seqlens_q_tensor, seqused_q_tensor = [
            to_cute_tensor(t, assumed_align=4) if t is not None else None
            for t in (cu_seqlens_q, seqused_q)
        ]
        arch = compute_capability * 10
        fa_bwd_post = FlashAttentionBackwardPostprocess(
            dtype, head_dim, arch, m_block_size, num_threads, AtomLayoutMdQ, dQ_swapAB
        )
        # TODO: check @can_implement
        _flash_attn_bwd.compile_cache_post[compile_key_post] = cute.compile(
            fa_bwd_post,
            dq_accum_tensor,
            dq_tensor,
            softmax_scale,
            cu_seqlens_q_tensor,
            seqused_q_tensor,
            current_stream,
            options="--enable-tvm-ffi",
        )
    _flash_attn_bwd.compile_cache_post[compile_key_post](
        dq_accum,
        dq,
        softmax_scale,
        cu_seqlens_q,
        seqused_q,
        current_stream,
    )

    if qhead_per_kvhead > 1:
        # Postprocess kernel: convert dk_accum & dv_accum from float32 to bf16/fp16
        compile_key_post = (dtype, head_dim, n_block_size, num_threads, AtomLayoutNdKV, dKV_swapAB)
        if compile_key_post not in _flash_attn_bwd.compile_cache_post:
            dk_accum_tensor = to_cute_tensor(dk_accum)
            dk_tensor = to_cute_tensor(dk)
            cu_seqlens_k_tensor, seqused_k_tensor = [
                to_cute_tensor(t, assumed_align=4) if t is not None else None
                for t in (cu_seqlens_k, seqused_k)
            ]
            fa_bwd_post = FlashAttentionBackwardPostprocess(
                dtype, head_dim, n_block_size, num_threads, AtomLayoutNdKV, dKV_swapAB
            )
            # TODO: check @can_implement
            _flash_attn_bwd.compile_cache_post[compile_key_post] = cute.compile(
                fa_bwd_post,
                dk_accum_tensor,
                dk_tensor,
                softmax_scale,
                cu_seqlens_k_tensor,
                seqused_k_tensor,
                current_stream,
                options="--enable-tvm-ffi",
            )
        _flash_attn_bwd.compile_cache_post[compile_key_post](
            dk_accum,
            dk,
            softmax_scale,
            cu_seqlens_k,
            seqused_k,
            current_stream,
        )
        compile_key_post = (
            dtype,
            head_dim_v,
            n_block_size,
            num_threads,
            AtomLayoutNdKV,
            dKV_swapAB,
        )
        if compile_key_post not in _flash_attn_bwd.compile_cache_post:
            dv_accum_tensor = to_cute_tensor(dv_accum)
            dv_tensor = to_cute_tensor(dv)
            cu_seqlens_k_tensor, seqused_k_tensor = [
                to_cute_tensor(t, assumed_align=4) if t is not None else None
                for t in (cu_seqlens_k, seqused_k)
            ]
            fa_bwd_post = FlashAttentionBackwardPostprocess(
                dtype, head_dim_v, n_block_size, num_threads, AtomLayoutNdKV, dKV_swapAB
            )
            # TODO: check @can_implement
            _flash_attn_bwd.compile_cache_post[compile_key_post] = cute.compile(
                fa_bwd_post,
                dv_accum_tensor,
                dv_tensor,
                cutlass.Float32(1.0),
                cu_seqlens_k_tensor,
                seqused_k_tensor,
                current_stream,
                options="--enable-tvm-ffi",
            )
        _flash_attn_bwd.compile_cache_post[compile_key_post](
            dv_accum,
            dv,
            1.0,
            cu_seqlens_k,
            seqused_k,
            current_stream,
        )

    return dq, dk, dv


_flash_attn_bwd.compile_cache_pre = {}
_flash_attn_bwd.compile_cache = {}
_flash_attn_bwd.compile_cache_post = {}


class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        window_size: Tuple[Optional[int], Optional[int]] = (None, None),
        learnable_sink: Optional[torch.Tensor] = None,
        softcap: float = 0.0,
        num_splits: int = 1,
        pack_gqa: Optional[bool] = None,
        mask_mod: Optional[Callable] = None,
        full_block_cnt: Optional[torch.Tensor] = None,
        full_block_idx: Optional[torch.Tensor] = None,
        mask_block_cnt: Optional[torch.Tensor] = None,
        mask_block_idx: Optional[torch.Tensor] = None,
        mSFQ: Optional[torch.Tensor] = None,
        mSFK: Optional[torch.Tensor] = None,
        mSFV: Optional[torch.Tensor] = None,
        force_fp4_impl: bool = False,
    ):
        # Only create block sparse tensors if at least one block sparse parameter is provided
        block_sparse_tensors = None
        if any(t is not None for t in [full_block_cnt, full_block_idx, mask_block_cnt, mask_block_idx]):
            block_sparse_tensors = BlockSparseTensorsTorch(
                full_block_cnt=full_block_cnt,
                full_block_idx=full_block_idx,
                mask_block_cnt=mask_block_cnt,
                mask_block_idx=mask_block_idx,
            )
        out, lse = _flash_attn_fwd(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            learnable_sink=learnable_sink,
            softcap=softcap,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
            mask_mod=mask_mod,
            block_sparse_tensors=block_sparse_tensors,
            mSFQ=mSFQ,
            mSFK=mSFK,
            mSFV=mSFV,
            force_fp4_impl=force_fp4_impl,
        )
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        return out, lse

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = _flash_attn_bwd(
            q,
            k,
            v,
            out,
            dout,
            lse,
            ctx.softmax_scale,
            ctx.causal,
            ctx.softcap,
        )
        return dq, dk, dv, *((None,) * 20)  # Extra Nones is fine


class FlashAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: Optional[torch.Tensor],
        cu_seqlens_k: Optional[torch.Tensor],
        seqused_q: Optional[torch.Tensor] = None,
        seqused_k: Optional[torch.Tensor] = None,
        page_table: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        window_size: Tuple[Optional[int], Optional[int]] = (None, None),
        learnable_sink: Optional[torch.Tensor] = None,
        softcap: float = 0.0,
        num_splits: int = 1,
        pack_gqa: Optional[bool] = None,
    ):
        out, lse = _flash_attn_fwd(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            page_table=page_table,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            learnable_sink=learnable_sink,
            softcap=softcap,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
        )
        ctx.save_for_backward(q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        return out, lse

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k = ctx.saved_tensors
        assert seqused_q == seqused_k == None
        assert ctx.softcap == 0.0
        dq, dk, dv = _flash_attn_bwd(
            q,
            k,
            v,
            out,
            dout,
            lse,
            ctx.softmax_scale,
            ctx.causal,
            ctx.softcap,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
        )

        return dq, dk, dv, *((None,) * 20)


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    learnable_sink: Optional[torch.Tensor] = None,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    mask_mod: Optional[Callable] = None,
    full_block_cnt: Optional[torch.Tensor] = None,
    full_block_idx: Optional[torch.Tensor] = None,
    mask_block_cnt: Optional[torch.Tensor] = None,
    mask_block_idx: Optional[torch.Tensor] = None,
    mSFQ: Optional[torch.Tensor] = None,
    mSFK: Optional[torch.Tensor] = None,
    mSFV: Optional[torch.Tensor] = None,
    force_fp4_impl: bool = False,
):
    return FlashAttnFunc.apply(
        q,
        k,
        v,
        softmax_scale,
        causal,
        window_size,
        learnable_sink,
        softcap,
        num_splits,
        pack_gqa,
        mask_mod,
        full_block_cnt,
        full_block_idx,
        mask_block_cnt,
        mask_block_idx,
        mSFQ,
        mSFK,
        mSFV,
        force_fp4_impl,
    )


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    learnable_sink: Optional[torch.Tensor] = None,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
):
    return FlashAttnVarlenFunc.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        page_table,
        softmax_scale,
        causal,
        window_size,
        learnable_sink,
        softcap,
        num_splits,
        pack_gqa,
    )


def _flash_attn_fwd_combine(
    out_partial: torch.Tensor,
    lse_partial: torch.Tensor,
    out: torch.Tensor,
    lse: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    num_splits_dynamic_ptr: Optional[torch.Tensor] = None,
    semaphore_to_reset: Optional[torch.Tensor] = None,
) -> None:
    """Forward combine kernel for split attention computation.

    Combines partial outputs and log-sum-exp values from multiple splits
    of attention computation into final outputs.

    Args:
        out_partial: Partial outputs tensor (num_splits, batch, seqlen, nheads, headdim) or
                                            (num_splits, total_q, nheads, headdim) if there's cu_seqlens
        lse_partial: Partial LSE tensor (num_splits, batch, seqlen, nheads) or
                                       (num_splits, total_q, nheads) if there's cu_seqlens
        out: Output tensor (batch, seqlen, nheads, headdim) or (total_q, nheads, headdim) if there's cu_seqlens
        lse: Output LSE tensor (batch, seqlen, nheads) or (total_q, nheads) if there's cu_seqlens.
        cu_seqlens: Cumulative sequence lengths for variable length sequences
        seqused: Used sequence lengths for each batch
        num_splits_dynamic_ptr: Dynamic number of splits per batch
        semaphore_to_reset: Semaphore for synchronization
        k_block_size: Block size for head dimension

    Returns:
        None
    """
    # Input validation
    assert out_partial.dim() in [4, 5], "out_partial must have 4 or 5 dimensions"
    assert lse_partial.dim() in [3, 4], "lse_partial must have 3 or 4 dimensions"
    assert out_partial.dtype in [torch.float16, torch.bfloat16, torch.float32], (
        "out_partial must be fp16, bf16, or fp32"
    )
    assert lse_partial.dtype == torch.float32, "lse_partial must be fp32"
    assert out_partial.is_cuda and lse_partial.is_cuda, "tensors must be on CUDA device"
    assert out_partial.stride(-1) == 1, "out_partial must be contiguous in the last dimension"
    assert lse_partial.stride(-2) == 1, "lse_partial must be contiguous in the seqlen dimension"
    assert lse_partial.shape == out_partial.shape[:-1]

    # Determine if this is variable length based on dimensions
    is_varlen = out_partial.dim() == 4

    # Validate output tensor shapes and types
    assert out.shape == out_partial.shape[1:], "out shape mismatch"
    if lse is not None:
        assert lse.shape == lse_partial.shape[1:], "lse shape mismatch"
        assert lse.dtype == torch.float32, "lse must be fp32"

    # Validate optional tensors
    for t, name in [
        (cu_seqlens, "cu_seqlens"),
        (seqused, "seqused"),
        (num_splits_dynamic_ptr, "num_splits_dynamic_ptr"),
    ]:
        if t is not None:
            assert t.dtype == torch.int32, f"{name} must be int32"
            assert t.is_cuda, f"{name} must be on CUDA device"
            assert t.is_contiguous(), f"{name} must be contiguous"

    head_dim = out_partial.shape[-1]
    num_splits = out_partial.shape[0]
    assert num_splits <= 256
    # If hdim is 96 or 192, it's faster to round them to 128 or 256 respectively
    # so that kBlockM is smaller and we have more parallelism.
    k_block_size = 64 if head_dim <= 64 else 128
    # We want kBlockM to be as small as possible to maximize parallelism.
    # E.g., if hdim is 64, we want kBlockM to be 16 so that we can use 256 threads, each reading 4 elements (floats).
    m_block_size = 8 if k_block_size % 128 == 0 else (16 if k_block_size % 64 == 0 else 32)
    log_max_splits = max(math.ceil(math.log2(num_splits)), 4)
    if m_block_size == 8:
        # If kBlockM == 8 then the minimum number of splits is 32.
        # TODO: we can deal w this by using 128 threads instead
        log_max_splits = max(log_max_splits, 5)

    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    # Create combine kernel configuration
    dtype = torch2cute_dtype_map[out.dtype]
    dtype_partial = torch2cute_dtype_map[out_partial.dtype]

    compile_key = (
        dtype,
        dtype_partial,
        head_dim,
        m_block_size,
        k_block_size,
        log_max_splits,
        cu_seqlens is not None,
        seqused is not None,
        lse is not None,
    )

    if compile_key not in _flash_attn_fwd_combine.compile_cache:
        out_partial_tensor = to_cute_tensor(
            out_partial, leading_dim=4 if not is_varlen else 3
        )
        lse_partial_tensor = to_cute_tensor(
            lse_partial, assumed_align=4, leading_dim=lse_partial.ndim - 2
        )
        out_tensor = to_cute_tensor(out, leading_dim=3 if not is_varlen else 2)
        lse_tensor = (
            to_cute_tensor(lse, assumed_align=4, leading_dim=lse.ndim - 2)
            if lse is not None
            else None
        )

        optional_tensors = [
            to_cute_tensor(t, assumed_align=4, leading_dim=0)
            if t is not None
            else None
            for t in (cu_seqlens, seqused, num_splits_dynamic_ptr, semaphore_to_reset)
        ]
        cu_seqlens_tensor, seqused_tensor, num_splits_dynamic_tensor, semaphore_tensor = (
            optional_tensors
        )
        fa_combine = FlashAttentionForwardCombine(
            dtype=dtype,
            dtype_partial=dtype_partial,
            head_dim=head_dim,
            m_block_size=m_block_size,
            k_block_size=k_block_size,
            log_max_splits=log_max_splits,
        )

        # Check if implementation is supported
        if not fa_combine.can_implement(
            dtype,
            dtype_partial,
            head_dim,
            m_block_size,
            k_block_size,
            log_max_splits,
            num_threads=256,
        ):
            raise RuntimeError(
                "FlashAttention combine kernel cannot be implemented with given parameters"
            )

        _flash_attn_fwd_combine.compile_cache[compile_key] = cute.compile(
            fa_combine,
            out_partial_tensor,
            lse_partial_tensor,
            out_tensor,
            lse_tensor,
            cu_seqlens_tensor,
            seqused_tensor,
            num_splits_dynamic_tensor,
            semaphore_tensor,
            current_stream,
            options="--enable-tvm-ffi",
        )
    _flash_attn_fwd_combine.compile_cache[compile_key](
        out_partial,
        lse_partial,
        out,
        lse,
        cu_seqlens,
        seqused,
        num_splits_dynamic_ptr,
        semaphore_to_reset,
        current_stream,
    )


_flash_attn_fwd_combine.compile_cache = {}


def flash_attn_combine(
    out_partial: torch.Tensor,
    lse_partial: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    return_lse: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Flash Attention combine function for split attention computation.

    Combines partial outputs and log-sum-exp values from multiple splits
    of attention computation into final outputs. This is the main user-facing
    interface for the combine kernel.

    Args:
        out_partial: Partial outputs tensor with shape:
            - (num_splits, batch_size, seqlen, num_heads, head_size) for regular batched input
            - (num_splits, total_q, num_heads, head_size) for variable length input
        lse_partial: Partial LSE tensor with shape:
            - (num_splits, batch_size, seqlen, num_heads) for regular batched input
            - (num_splits, total_q, num_heads) for variable length input
        out: Optional output tensor. If None, will be created automatically.
        out_dtype: Optional output dtype. If None, will use fp16/bf16 based on input.
        cu_seqlens: Cumulative sequence lengths for variable length sequences
        seqused: Used sequence lengths for each batch
        return_lse: Whether to return the combined LSE tensor. Default is True.

    Returns:
        Tuple of (out, lse) where:
        - out: Combined output tensor with shape (batch_size, seqlen, num_heads, head_size)
              or (total_q, num_heads, head_size) for varlen
        - lse: Combined log-sum-exp tensor with shape (batch_size, seqlen, num_heads)
              or (total_q, num_heads) for varlen. None if return_lse=False

    Note:
        This function expects the input tensors to be in the format produced by
        split attention computation, where the first dimension is num_splits.
        The permuting from user format to kernel format is now done inside the kernel.
    """
    # Input validation
    assert out_partial.dim() in [4, 5], "out_partial must have 4 or 5 dimensions"
    assert lse_partial.dim() in [3, 4], "lse_partial must have 3 or 4 dimensions"
    assert out_partial.dtype == torch.float32, "out_partial must be fp32 (from accumulation)"
    assert lse_partial.dtype == torch.float32, "lse_partial must be fp32"

    # Determine if this is variable length based on dimensions
    is_varlen = out_partial.dim() == 4

    if is_varlen:
        # Variable length: (num_splits, total_q, num_heads, head_size)
        num_splits, total_q, num_heads, head_size = out_partial.shape
        assert lse_partial.shape == (num_splits, total_q, num_heads), (
            "lse_partial shape mismatch for varlen"
        )
        batch_size = 1  # Treat as single batch for varlen
        seqlen = total_q
    else:
        # Regular batched: (num_splits, batch_size, seqlen, num_heads, head_size)
        num_splits, batch_size, seqlen, num_heads, head_size = out_partial.shape
        assert lse_partial.shape == (num_splits, batch_size, seqlen, num_heads), (
            "lse_partial shape mismatch"
        )

    # Determine output dtype
    if out_dtype is None:
        out_dtype = out_partial.dtype

    # Create output if not provided
    device = out_partial.device
    if out is None:
        if is_varlen:
            out = torch.empty(total_q, num_heads, head_size, dtype=out_dtype, device=device)
        else:
            out = torch.empty(
                batch_size, seqlen, num_heads, head_size, dtype=out_dtype, device=device
            )

    # Create lse output only if requested
    if return_lse:
        if is_varlen:
            lse = torch.empty(num_heads, total_q, dtype=torch.float32, device=device).transpose(
                0, 1
            )
        else:
            lse = torch.empty(
                batch_size, num_heads, seqlen, dtype=torch.float32, device=device
            ).transpose(1, 2)
    else:
        lse = None

    _flash_attn_fwd_combine(
        out_partial,
        lse_partial,
        out,
        lse,
        cu_seqlens,
        seqused,
    )
    return out, lse
