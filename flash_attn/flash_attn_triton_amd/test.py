import os
import glob
import shutil
import time
import torch
import pytest
import logging
import numpy as np
from pathlib import Path
from flash_attn import (
    flash_attn_func,
    flash_attn_fp8_func,
    flash_attn_kvpacked_func, 
    flash_attn_qkvpacked_func,
    flash_attn_qkvpacked_fp8_func,
    flash_attn_varlen_func,
    flash_attn_varlen_fp8_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_varlen_qkvpacked_func,
    flash_attn_varlen_qkvpacked_fp8_func
)

from .utils import DEBUG, input_helper, arch_supports_fp8
from .fwd_ref import attention_forward_pytorch_ref_impl
from .fwd_prefill import attention_prefill_forward_triton_impl
from .bwd_prefill_split import attention_prefill_backward_triton_split_impl
from .bwd_ref import attention_backward_pytorch_ref_impl

# set print options
# torch.set_printoptions(linewidth=5e5, edgeitems=10, sci_mode=False)
# np.set_printoptions(linewidth=5000, threshold=1e4, suppress=True, precision=4)

# defailt fp16 tolerance is ATOL, RTOL = 1e-5, 1e-3. See table https://pytorch.org/docs/stable/testing.html
ATOL, RTOL = 1e-2, 1e-2 # old standard. maybe to lose. 
# ATOL, RTOL = 1e-3, 1e-3  # catchs fa mismatch issues
# ATOL, RTOL = 1e-4, 1e-3 # to strict. there will be small diffs
# ATOL, RTOL = 1e-5, 1e-3 # # default fp16. there will be small diffs
# ATOL_fp8, RTOL_fp8 = 1e-1, 1e-1 # to strict for larger tensors in fp8
ATOL_fp8, RTOL_fp8 = 2.5e-1, 2.5e-1 #  fp8
# ATOL_fp8, RTOL_fp8 = 2e-2, 2e-2 #  fp8
EQUAL_NAN = True

@pytest.mark.parametrize(
    "BATCH, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD",
    [
        (1, 1, 1, 1, 1, 1),
        (1, 1, 1, 2, 4, 16),
        (1, 2, 2, 2, 4, 16),
        (1, 4, 1, 2, 4, 16),
        (1, 4, 2, 2, 4, 16),
        (1, 1, 1, 4, 2, 16),
        (1, 1, 1, 4, 4, 16),
        (1, 2, 2, 4, 4, 16),
        (2, 1, 1, 4, 4, 16),
        (2, 2, 2, 4, 4, 16),
        (1, 1, 1, 128, 64, 16),
        (2, 2, 2, 2, 128, 1),
        (2, 3, 3, 2, 128, 16),
        (3, 2, 2, 256, 512, 16),
        (3, 3, 3, 128, 128, 64),
        (2, 4, 4, 1024, 1024, 64),
        (4, 6, 6, 108, 256, 224),
        (4, 8, 8, 2048, 2048, 128),
        (4, 16, 16, 4096, 4096, 64),
        (2, 4, 4, 8192, 8192, 32),
        # fa configs
        (4, 6, 1, 113, 203, 256),
        (4, 6, 1, 128, 217, 256),
        (4, 6, 2, 113, 211, 128),
        (4, 6, 2, 108, 256, 128),
        (4, 6, 1, 256, 512, 64),
        (4, 6, 1, 512, 256, 64),
        (4, 6, 2, 1024, 1024, 32),
        (4, 6, 2, 1023, 1024, 32),
        (4, 6, 6, 1024, 1023, 32),
        (4, 6, 6, 2048, 2048, 32),
    ],
)
@pytest.mark.parametrize('causal', [True, False])
@pytest.mark.parametrize('dropout_p', [0.0])
@pytest.mark.parametrize('alibi_slopes', [None])
@pytest.mark.parametrize('layout', ["bshd", "thd"])
@pytest.mark.parametrize('dtype', [torch.float16])
@pytest.mark.parametrize('use_exp2', [True, False]) # works when use_exp2 is false
@pytest.mark.parametrize('DEBUG_INPUT', [False]) # NOTE: debug input can overflow when the tensors are large. Just use to figure out issues
def test_op_prefill_fwd_impl(BATCH, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, causal, dropout_p, alibi_slopes, layout, dtype, use_exp2, DEBUG_INPUT):
    torch.manual_seed(42)
    device = "cuda"

    q, k, v, do, metadata = input_helper(BATCH, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, causal, dropout_p, dtype, layout=layout, device=device)

    if DEBUG:
        if HQ // HK != 1:
            print("MQA/GQA")
        else:
            print("MHA")

    # update metadata
    metadata.use_exp2 = use_exp2
    if causal:
        metadata.need_causal(True)

    # NOTE: the returned score is not the same as the reference because we need to adjust as we find new maxes per block. We are not doing that
    metadata.need_dropout(dropout_p)


    # call Triton's forward implementation directly
    q_triton = q.clone()
    k_triton = k.clone()
    v_triton = v.clone()
    o_triton = torch.zeros_like(q).contiguous() if DEBUG_INPUT else torch.empty_like(q)
    softmax_lse_triton, sd_mask_triton = attention_prefill_forward_triton_impl(
                                                q_triton, 
                                                k_triton, 
                                                v_triton, 
                                                o_triton, 
                                                metadata.sm_scale, 
                                                metadata.alibi_slopes, 
                                                metadata.causal, 
                                                metadata.bias, 
                                                metadata.layout, 
                                                metadata.cu_seqlens_q, 
                                                metadata.cu_seqlens_k,
                                                metadata.max_seqlens_q, 
                                                metadata.max_seqlens_k,
                                                metadata.cache_seqlens, 
                                                metadata.cache_batch_idx,
                                                metadata.dropout_p,
                                                metadata.philox_seed, 
                                                metadata.philox_offset, 
                                                metadata.return_scores, 
                                                metadata.use_exp2,
                                                None,
                                                None,
                                                None,
                                                None)

    # ref forward
    q_ref = q.clone()
    k_ref = k.clone()
    v_ref = v.clone()
    o_ref = torch.zeros_like(q).contiguous() if DEBUG_INPUT else torch.empty_like(q)
    softmax_lse_ref, sd_mask_ref  = attention_forward_pytorch_ref_impl(
        q_ref, 
        k_ref, 
        v_ref,
        o_ref,
        metadata.sm_scale,
        metadata.alibi_slopes,
        causal, 
        layout,
        metadata.cu_seqlens_q,
        metadata.cu_seqlens_k,
        metadata.max_seqlens_q,
        metadata.max_seqlens_k,
        metadata.dropout_p,
        metadata.philox_seed, 
        metadata.philox_offset, 
        use_exp2
    )

    if DEBUG:
        print()
        print("Compare Triton Impl with refernce Pytorch Impl")

    # this can be set to true manually or when using dropout
    if metadata.return_scores:
        if DEBUG:
            print("sd_mask_triton:", sd_mask_triton, sd_mask_triton.shape)
            print("sd_mask_ref:", sd_mask_ref, sd_mask_ref.shape)
        torch.testing.assert_close(sd_mask_triton.to(sd_mask_ref.dtype), sd_mask_ref, atol=ATOL, rtol=RTOL)

    if DEBUG:
        print("softmax_lse_triton:", softmax_lse_triton, softmax_lse_triton.shape)
        print("softmax_lse_ref:", softmax_lse_ref, softmax_lse_ref.shape)
    torch.testing.assert_close(softmax_lse_triton, softmax_lse_ref, atol=ATOL, rtol=RTOL)
    
    if DEBUG:
        print("output_triton:", o_triton, o_triton.shape)
        print("output_ref:", o_ref, o_ref.shape)
    torch.testing.assert_close(o_triton, o_ref, atol=ATOL, rtol=RTOL)

@pytest.mark.parametrize(
    "BATCH, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD", [
    (1, 1, 1, 1, 1, 1),
    (1, 1, 1, 4, 4, 4),
    (2, 1, 1, 4, 4, 16),
    (1, 2, 2, 4, 4, 16),
    (1, 4, 1, 2, 4, 16),
    (1, 8, 1, 2, 4, 16),
    (1, 16, 1, 2, 4, 16),
    (1, 32, 1, 2, 4, 16),
    (1, 64, 1, 2, 4, 16),
    (1, 4, 2, 2, 4, 16),
    (2, 2, 2, 4, 4, 16),
    (1, 1, 1, 4, 4, 16),
    (2, 1, 1, 4, 4 , 16),
    (4, 6, 6, 8, 8 , 16),
    (1, 1, 1, 4, 4, 32),
    (1, 1, 1, 16, 16, 16),
    (1, 1, 1, 32, 32, 16),
    (1, 1, 1, 64, 64, 16),
    (1, 1, 1, 64, 64, 16),
    (1, 1, 1, 64, 128, 16),
    (1, 1, 1, 64, 64, 32),
    (1, 1, 1, 64, 128, 32),
    (1, 1, 1, 128, 128, 64),
    (1, 1, 1, 128, 256, 45),
    (1, 1, 1, 113, 203, 192),
    (1, 1, 1, 256, 256, 64),
    (1, 1, 1, 256, 512, 16),
    (1, 1, 1, 512, 512, 64),
    (1, 1, 1, 1024, 1024, 64),
    # fa configs
    (2, 2, 2, 128, 128, 65),
    (2, 2, 2, 128, 128, 224),
    (4, 6, 6, 108, 256, 224),
    (1, 1, 1, 256, 512, 16),
    # old tests that work
    (4, 48, 6, 1024, 1024, 64),
    (4, 48, 12, 2048, 1024, 64),
    (4, 48, 24, 1024, 1024, 64),
    (4, 48, 48, 1024, 1024, 64),
    (4, 48, 48, 1024, 1024, 73),
    (4, 48, 48, 2048, 2048, 64),
    (1, 24, 24, 4096, 4096, 64),
    (1, 16, 16, 1024, 1024, 64),
    (1, 16, 16, 1024, 1024, 128),
    # testcase new
    # seqlen q == k
    (1, 1, 1, 2, 2, 2),  # small enough to debug
    (1, 1, 1, 128, 128, 32),  # only one block
    (1, 1, 1, 127, 127, 32),  # only one block but with masking
    (1, 1, 1, 129, 129, 1),  # two blocks with 2nd block small enough to debug
    (1, 1, 1, 350, 350, 1),  # two blocks with 2nd block small enough to debug
    (1, 1, 1, 350, 350, 68),  # generic masking on q, k and head
    (4, 1, 1, 512, 512, 128),  # batch > 1
    (4, 8, 2, 512, 512, 128),  # GQA
    (4, 8, 2, 512, 512, 68),   # non-power-of-2 head_dim
    (4, 8, 2, 500, 500, 68),  # comprehensive case for seqlen q == k
    # seqlen q > k
    (1, 1, 1, 64, 32, 8),  # seqlen_q > seqlen_k
    (1, 1, 1, 192, 128, 32),  # seqlen_q > seqlen_k
    (4, 8, 2, 1024, 512, 68),  # seqlen_q < seqlen_k
    (1, 1, 1, 729, 516, 68),  # seqlen_q > seqlen_k
    (16, 16, 4, 2753, 1528, 68),  # a comprehensive seqlen_q > seqlen_k
    # seqlen q < k
    (1, 1, 1, 32, 64, 8),  # seqlen_q > seqlen_k
    (1, 1, 1, 128, 192, 32),  # seqlen_q < seqlen_k
    (4, 8, 2, 512, 1024, 68),  # seqlen_q < seqlen_k
    (1, 1, 1, 200, 413, 1),  # seqlen_q < seqlen_k
    (1, 1, 1, 782, 1546, 1),  # seqlen_q < seqlen_k
    (16, 16, 4, 1528, 2753, 68),  # a comprehensive seqlen_q < seqlen_k
])
@pytest.mark.parametrize('causal', [True, False])
@pytest.mark.parametrize('dropout_p', [0.0])
@pytest.mark.parametrize('alibi_slopes', [None])
@pytest.mark.parametrize('layout', ["bshd", "thd"])
@pytest.mark.parametrize('dtype', [torch.float16])
@pytest.mark.parametrize('use_exp2', [False]) # FIXME: using exp2 causes issue when used with causal
@pytest.mark.parametrize('DEBUG_INPUT', [False]) # debug output causes nans on larger tensors
def test_op_prefill_bwd_impl(BATCH, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, causal, dropout_p, alibi_slopes, layout, dtype, use_exp2, DEBUG_INPUT):
    torch.manual_seed(20)
    device="cuda"

    # gen inputs
    q, k, v, do, metadata = input_helper(BATCH, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, causal, dropout_p, dtype, layout=layout, device=device)

    # NOTE: the returned score is not the same as the reference because we need to adjust as we find new maxes per block. We are not doing that
    metadata.need_dropout(dropout_p)

    # =============================================== Reference ==============================================================
    # fwd
    q_ref = q.clone() 
    k_ref = k.clone()
    v_ref = v.clone()
    output_ref = torch.zeros_like(q).contiguous() if DEBUG_INPUT else torch.empty_like(q)
    softmax_lse_ref, sd_mask_ref = attention_forward_pytorch_ref_impl(
        q_ref,
        k_ref, 
        v_ref,
        output_ref,
        metadata.sm_scale,
        metadata.alibi_slopes,
        causal, 
        layout,
        metadata.cu_seqlens_q,
        metadata.cu_seqlens_k,
        metadata.max_seqlens_q,
        metadata.max_seqlens_k,
        metadata.dropout_p,
        metadata.philox_seed, 
        metadata.philox_offset, 
        use_exp2
    )

    # bwd
    do_ref = do.clone()
    dq_ref = torch.zeros_like(q).contiguous() if DEBUG_INPUT else torch.empty_like(q)
    dk_ref = torch.zeros_like(k).contiguous() if DEBUG_INPUT else torch.empty_like(k)
    dv_ref = torch.zeros_like(v).contiguous() if DEBUG_INPUT else torch.empty_like(v)
    delta_ref = attention_backward_pytorch_ref_impl(
        do_ref,
        q_ref,
        k_ref,
        v_ref,
        output_ref,
        softmax_lse_ref,
        dq_ref,
        dk_ref,
        dv_ref,
        metadata.sm_scale,
        metadata.alibi_slopes,
        causal,
        layout,
        metadata.cu_seqlens_q,
        metadata.cu_seqlens_k,
        metadata.max_seqlens_q,
        metadata.max_seqlens_k,
        metadata.dropout_p,
        metadata.philox_seed, 
        metadata.philox_offset, 
        use_exp2
    )

    # =============================================== Triton ==============================================================
    do_triton = do.clone()
    q_triton = q.clone()
    k_triton = k.clone()
    v_triton = v.clone()
    o_triton = output_ref.clone().contiguous()
    softmax_lse_triton = softmax_lse_ref.clone().contiguous()
    dq_triton = torch.zeros_like(q_triton, dtype=q.dtype) # NOTE: the kernel does inplace accumlation on dq so dq has to be zeros
    dk_triton = torch.zeros_like(k_triton, dtype=k.dtype) if DEBUG_INPUT else torch.empty_like(k_triton, dtype=k.dtype)
    dv_triton = torch.zeros_like(v_triton, dtype=v.dtype) if DEBUG_INPUT else torch.empty_like(v_triton, dtype=v.dtype)
    delta_triton = attention_prefill_backward_triton_split_impl(
        do_triton,
        q_triton,
        k_triton,
        v_triton,
        o_triton,
        softmax_lse_triton,
        dq_triton,
        dk_triton,
        dv_triton,
        metadata.sm_scale,
        alibi_slopes,
        causal,
        layout,
        metadata.cu_seqlens_q,
        metadata.cu_seqlens_k,
        metadata.max_seqlens_q,
        metadata.max_seqlens_k,
        metadata.dropout_p,
        metadata.philox_seed, 
        metadata.philox_offset, 
        use_exp2,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )

    # =============================================== Check ==============================================================
    if DEBUG:
        print()
    if DEBUG:
        print("delta_triton:", delta_triton, delta_triton.shape)
        print("delta_ref:", delta_ref, delta_ref.shape)
    torch.testing.assert_close(delta_triton, delta_ref, atol=ATOL, rtol=RTOL, equal_nan=EQUAL_NAN)

    if DEBUG:
        print("dv_triton:", dv_triton, dv_triton.shape)
        print("dv_ref:", dv_ref, dv_ref.shape)
    torch.testing.assert_close(dv_triton, dv_ref, atol=ATOL, rtol=RTOL, equal_nan=EQUAL_NAN)

    if DEBUG:
        print("dk_triton:", dk_triton, dk_triton.shape)
        print("dk_ref:", dk_ref, dk_ref.shape)
    torch.testing.assert_close(dk_triton, dk_ref, atol=ATOL, rtol=RTOL, equal_nan=EQUAL_NAN)

    if DEBUG:
        print("dq_triton:", dq_triton, dq_triton.shape)
        print("dq_ref:", dq_ref, dq_ref.shape)
    torch.testing.assert_close(dq_triton, dq_ref, atol=ATOL, rtol=RTOL, equal_nan=EQUAL_NAN)

def fp8_assert_close(tensor_a, tensor_b, atol=ATOL_fp8, rtol=RTOL_fp8, max_diff_percentage=0.5):
    """Assert tensors are close with tolerance for small percentage of elements"""
    # standard comparison
    abs_diff = torch.abs(tensor_a - tensor_b)
    rel_diff = abs_diff / torch.abs(tensor_b.clamp(min=1e-6))
    
    # calculate elements that exceed tolerance
    abs_check = abs_diff > atol
    rel_check = rel_diff > rtol
    failed_check = torch.logical_and(abs_check, rel_check)
    
    # calculate percentage of failed elements
    failed_percentage = failed_check.sum().item() / failed_check.numel() * 100
    
    # if percentage is small enough, test passes
    if failed_percentage <= max_diff_percentage:
        return True
    
    # Otherwise, provide diagnostic information
    max_abs_idx = torch.argmax(abs_diff).item()
    max_rel_idx = torch.argmax(rel_diff).item()
    
    flat_to_idx = lambda flat_idx, shape: np.unravel_index(flat_idx, shape)
    
    max_abs_pos = flat_to_idx(max_abs_idx, tensor_a.shape)
    max_rel_pos = flat_to_idx(max_rel_idx, tensor_a.shape)
    
    max_abs_diff = abs_diff.flatten()[max_abs_idx].item()
    max_rel_diff = rel_diff.flatten()[max_rel_idx].item()
    
    raise AssertionError(
        f"Tensors not close enough! {failed_percentage:.6f}% elements exceed tolerance.\n"
        f"Greatest absolute difference: {max_abs_diff} at index {max_abs_pos} (up to {atol} allowed)\n"
        f"Greatest relative difference: {max_rel_diff} at index {max_rel_pos} (up to {rtol} allowed)"
    )

@pytest.mark.parametrize(
    "Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD",
    [
        # seqlen q == k
        (1, 1, 1, 1, 1, 1),
        (1, 1, 1, 2, 2, 2),  # small enough to debug
        (1, 1, 1, 4, 4, 16),
        (1, 2, 2, 4, 4, 16),
        (2, 1, 1, 4, 4, 16),
        (2, 2, 2, 4, 4, 16),
        (1, 1, 1, 128, 128, 32),  # only one block
        (3, 3, 3, 128, 128, 64),
        (1, 1, 1, 127, 127, 32),  # only one block but with masking
        # (1, 1, 1, 129, 129, 1),  # two blocks with 2nd block small enough to debug # fails
        (1, 2, 2, 129, 129, 32),  # two blocks with 2nd block small enough to debug
        (1, 1, 1, 350, 350, 32),  # two blocks with 2nd block small enough to debug
        (1, 1, 1, 350, 350, 68),  # generic masking on q, k and head
        (4, 1, 1, 512, 512, 128), # batch > 1
        (4, 2, 2, 512, 512, 128),
        (4, 2, 2, 512, 512, 68),
        (4, 2, 2, 500, 500, 68),
        (2, 4, 4, 1024, 1024, 64),
        (4, 8, 8, 2048, 2048, 128),
        (2, 8, 8, 4096, 4096, 64),
        (2, 4, 4, 8192, 8192, 32),
        # seqlen q > k
        (1, 1, 1, 4, 2, 16),
        (1, 1, 1, 64, 32, 8),
        (1, 1, 1, 128, 64, 16),
        (1, 1, 1, 192, 128, 32),
        (1, 2, 2, 1024, 512, 68),
        (1, 4, 4, 729, 516, 68),
        (2, 4, 4, 2753, 1528, 68),  # a comprehensive seqlen_q > seqlen_k
        # seqlen q < k
        (1, 1, 1, 2, 4, 16),
        (1, 2, 2, 2, 4, 16),
        (1, 4, 1, 2, 4, 16),
        (1, 4, 2, 2, 4, 16),
        (2, 2, 2, 2, 128, 1),
        (2, 3, 3, 2, 128, 16),
        (1, 1, 1, 32, 64, 8),
        (1, 1, 1, 128, 192, 32),
        (4, 6, 6, 108, 256, 32),
        (3, 2, 2, 256, 512, 16),
        (2, 2, 2, 512, 1024, 68),
        (1, 1, 1, 200, 413, 32),
        (1, 1, 1, 782, 1546, 32),
        # gqa/mqa                   # mismatch issue on varlen
        (4, 8, 2, 500, 500, 68), 
        (4, 8, 2, 512, 512, 68),
        (4, 8, 2, 512, 512, 128),
        (4, 8, 2, 512, 1024, 68),
        (4, 8, 2, 1024, 512, 64),
        (4, 16, 4, 1528, 2753, 68),
        # fa configs
        (2, 4, 1, 113, 203, 64),
        (2, 4, 2, 128, 217, 64),
        (2, 6, 2, 113, 211, 128),
        (2, 6, 2, 108, 256, 128),
        (2, 6, 2, 256, 512, 64),
        (2, 6, 2, 512, 256, 64),
        (2, 6, 2, 1024, 1024, 32),
        (2, 6, 2, 1023, 1024, 32),
        (2, 6, 6, 1024, 1023, 32),
        (2, 6, 6, 2048, 2048, 32),
    ],
)
@pytest.mark.parametrize('causal', [False, True])
@pytest.mark.parametrize('dropout_p', [0.0])
@pytest.mark.parametrize('layout', ["bshd", "thd"])
@pytest.mark.parametrize('packing', [None, "qkv"])
@pytest.mark.parametrize('DEBUG_INPUT', [False])
@pytest.mark.flaky(reruns=3, reason="Retry failures")
@pytest.mark.skipif(not arch_supports_fp8(), reason="fp8 not supported on this device")
def test_fp8(Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, causal, dropout_p, layout, packing, DEBUG_INPUT):
    torch.manual_seed(20)
    test_backward = True
    device = "cuda"
    window_size = (-1, -1)
    softcap = 0.0
    alibi_slopes = None
    deterministic = False
    ref_dtype = torch.float32
    is_varlen = True if layout == "thd" else False

    # skip QKV packing tests for uneven sequence lengths and head sizes
    if packing == 'qkv':
        if N_CTX_Q != N_CTX_K:
            pytest.skip("QKV packing requires N_CTX_Q == N_CTX_K")
        if HQ != HK:
            pytest.skip("QKV packing requires HQ == HK")

    # test apis
    if packing == 'qkv':
        # generate inputs
        qkv, do, metadata = input_helper(Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, causal, dropout_p, ref_dtype, layout, packing=packing, device=device, DEBUG_INPUT=DEBUG_INPUT)

        # ----------------------------------------------------------------
        # --- FP8 ---
        # ----------------------------------------------------------------
        qkv_fp8 = qkv.clone()
        do_fp8= do.clone()

        if is_varlen:
            out_fp8, lse_fp8, S_dmask_fp8 = flash_attn_varlen_qkvpacked_fp8_func(
                qkv_fp8,
                metadata.cu_seqlens_q,
                metadata.max_seqlens_q,
                dropout_p,
                causal=causal,
                window_size=window_size,
                softcap=softcap,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                return_attn_probs=True,
            )
        else:
            out_fp8, lse_fp8, S_dmask_fp8 = flash_attn_qkvpacked_fp8_func(
                qkv_fp8,
                dropout_p,
                causal=causal,
                window_size=window_size,
                softcap=softcap,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                return_attn_probs=True,
            )

        # ----------------------------------------------------------------
        # --- Reference ---
        # ----------------------------------------------------------------
        # reference forward pass
        qkv_ref = qkv.clone()
        do_ref= do.clone()

        if is_varlen:
            out_ref, lse_ref, S_dmask_ref = flash_attn_varlen_qkvpacked_func(
                qkv_ref,
                metadata.cu_seqlens_q,
                metadata.max_seqlens_q,
                dropout_p,
                causal=causal,
                window_size=window_size,
                softcap=softcap,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                return_attn_probs=True,
            )
        else:
            out_ref, lse_ref, S_dmask_ref = flash_attn_qkvpacked_func(
                qkv_ref,
                dropout_p,
                causal=causal,
                window_size=window_size,
                softcap=softcap,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                return_attn_probs=True,
            )

        # ----------------------------------------------------------------
        # --- Compare ---
        # ----------------------------------------------------------------
        # compare forward
        if DEBUG:
            print()
            print(f"Compare fp8 against ref with dtype {ref_dtype}")

        if DEBUG:
            print("out_ref:", out_ref, out_ref.shape)
            print("out_fp8:", out_fp8, out_fp8.shape)
        fp8_assert_close(out_ref, out_fp8, atol=ATOL_fp8, rtol=RTOL_fp8 )
        

        if DEBUG:
            print("lse_ref:", lse_ref, lse_ref.shape)
            print("lse_fp8:", lse_fp8, lse_fp8.shape)
        fp8_assert_close(lse_ref, lse_fp8, atol=ATOL_fp8, rtol=RTOL_fp8)


        if dropout_p > 0.0:
            if DEBUG:
                print("S_dmask_ref:", S_dmask_ref, S_dmask_ref.shape)
                print("S_dmask_fp8:", S_dmask_fp8, S_dmask_fp8.shape)
            fp8_assert_close(S_dmask_ref, S_dmask_fp8, atol=ATOL_fp8, rtol=RTOL_fp8)

        if not test_backward:
            return
        
        # fp8 backward pass
        dqkv_fp8, = torch.autograd.grad(out_fp8, (qkv_fp8), do_fp8)

        # ref backward pass
        dqkv_ref, = torch.autograd.grad(out_ref, (qkv_ref), do_ref)

        # compare backward gradients
        if DEBUG:
            print("dqkv_ref:", dqkv_ref, dqkv_ref.shape)
            print("dqkv_fp8:", dqkv_fp8, dqkv_fp8.shape)
        fp8_assert_close(dqkv_ref, dqkv_fp8, atol=ATOL_fp8, rtol=RTOL_fp8)

    elif packing is None:
        # generate inputs
        q, k, v, do, metadata = input_helper(Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, causal, dropout_p, ref_dtype, layout, device=device, DEBUG_INPUT=DEBUG_INPUT)

        # ----------------------------------------------------------------
        # --- FP8 ---
        # ----------------------------------------------------------------
        if DEBUG:
            print()
            print(f"Compute Fp8 Forward")
        q_fp8 = q.clone()
        k_fp8 = k.clone()
        v_fp8 = v.clone()
        do_fp8= do.clone()

        if is_varlen:
            out_fp8, lse_fp8, S_dmask_fp8 = flash_attn_varlen_fp8_func(
                q_fp8,
                k_fp8,
                v_fp8,
                metadata.cu_seqlens_q,
                metadata.cu_seqlens_k,
                metadata.max_seqlens_q,
                metadata.max_seqlens_k,
                dropout_p,
                causal=causal,
                window_size=window_size,
                softcap=softcap,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                return_attn_probs=True,
            )
        else:
            out_fp8, lse_fp8, S_dmask_fp8 = flash_attn_fp8_func(
                q_fp8,
                k_fp8,
                v_fp8,
                dropout_p,
                causal=causal,
                window_size=window_size,
                softcap=softcap,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                return_attn_probs=True,
            )

        # ----------------------------------------------------------------
        # --- Reference ---
        # ----------------------------------------------------------------
        if DEBUG:
            print()
            print(f"Compute Reference Forward")
        # reference forward pass
        q_ref = q.clone()
        k_ref = k.clone()
        v_ref = v.clone()
        do_ref = do.clone()

        if is_varlen:
            out_ref, lse_ref, S_dmask_ref = flash_attn_varlen_func(
                q_ref,
                k_ref,
                v_ref,
                metadata.cu_seqlens_q,
                metadata.cu_seqlens_k,
                metadata.max_seqlens_q,
                metadata.max_seqlens_k,
                dropout_p,
                causal=causal,
                window_size=window_size,
                softcap=softcap,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                return_attn_probs=True,
            )
        else:
            out_ref, lse_ref, S_dmask_ref = flash_attn_func(
                q_ref,
                k_ref,
                v_ref,
                dropout_p,
                causal=causal,
                window_size=window_size,
                softcap=softcap,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                return_attn_probs=True,
            )

        # ----------------------------------------------------------------
        # --- Compare ---
        # ----------------------------------------------------------------
        # compare forward
        if DEBUG:
            print()
            print(f"Compare fp8 against ref with dtype {ref_dtype}")

        if DEBUG:
            print("out_ref:", out_ref, out_ref.shape)
            print("out_fp8:", out_fp8, out_fp8.shape)
        # torch.testing.assert_close(out_ref, out_fp8, atol=ATOL_fp8, rtol=RTOL_fp8)
        fp8_assert_close(out_ref, out_fp8, atol=ATOL_fp8, rtol=RTOL_fp8 )
        

        if DEBUG:
            print("lse_ref:", lse_ref, lse_ref.shape)
            print("lse_fp8:", lse_fp8, lse_fp8.shape)
        # torch.testing.assert_close(lse_ref, lse_fp8, atol=ATOL_fp8, rtol=RTOL_fp8)
        fp8_assert_close(lse_ref, lse_fp8, atol=ATOL_fp8, rtol=RTOL_fp8)


        if dropout_p > 0.0:
            if DEBUG:
                print("S_dmask_ref:", S_dmask_ref, S_dmask_ref.shape)
                print("S_dmask_fp8:", S_dmask_fp8, S_dmask_fp8.shape)
            # torch.testing.assert_close(S_dmask_ref, S_dmask_fp8, atol=ATOL_fp8, rtol=RTOL_fp8)
            fp8_assert_close(S_dmask_ref, S_dmask_fp8, atol=ATOL_fp8, rtol=RTOL_fp8)

        if not test_backward:
            return
        
        if DEBUG:
            print()
            print(f"Compute Fp8 Backward")
        # fp8 backward pass
        dq_fp8, dk_fp8, dv_fp8 = torch.autograd.grad(out_fp8, (q_fp8, k_fp8, v_fp8), do_fp8)
        
        if DEBUG:
            print()
            print(f"Compute Reference Backward")
        # ref backward pass
        dq_ref, dk_ref, dv_ref = torch.autograd.grad(out_ref, (q_ref, k_ref, v_ref), do_ref)

        # compare backward gradients
        if DEBUG:
            print("dv_ref:", dv_ref, dv_ref.shape)
            print("dv_fp8:", dv_fp8, dv_fp8.shape)
        # torch.testing.assert_close(dv_ref, dv_fp8, atol=ATOL_fp8, rtol=RTOL_fp8, equal_nan=EQUAL_NAN)
        fp8_assert_close(dv_ref, dv_fp8, atol=ATOL_fp8, rtol=RTOL_fp8 )

        if DEBUG:
            print("dk_ref:", dk_ref, dk_ref.shape)
            print("dk_fp8:", dk_fp8, dk_fp8.shape)
        # torch.testing.assert_close(dk_ref, dk_fp8, atol=ATOL_fp8, rtol=RTOL_fp8, equal_nan=EQUAL_NAN)
        fp8_assert_close(dk_ref, dk_fp8, atol=ATOL_fp8, rtol=RTOL_fp8 )

        if DEBUG:
            print("dq_ref:", dq_ref, dq_ref.shape)
            print("dq_fp8:", dq_fp8, dq_fp8.shape)
        # torch.testing.assert_close(dq_ref, dq_fp8, atol=ATOL_fp8, rtol=RTOL_fp8, equal_nan=EQUAL_NAN)
        fp8_assert_close(dq_ref, dq_fp8, atol=ATOL_fp8, rtol=RTOL_fp8 )

@pytest.mark.parametrize(
    "BATCH, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD",
    [
        (2, 4, 4, 512, 512, 128),
    ],
)
@pytest.mark.parametrize('causal', [False, True])
@pytest.mark.parametrize('dropout_p', [0.0, 0.1])
@pytest.mark.parametrize('layout', ['bshd'])
@pytest.mark.parametrize('packing', [None])
@pytest.mark.parametrize('test_backward', [False, True])
@pytest.mark.skipif(not arch_supports_fp8(), reason="fp8 not supported on this device")
@pytest.mark.skip("Breaks on CI but works locally")
def test_ir(BATCH, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, causal, dropout_p, layout, packing, test_backward): # Don't run this test in parallel. It clears the cache so it doesnot work properly if run in parallel.
    torch.manual_seed(20)
    device = "cuda"
    window_size = (-1, -1)
    softcap = 0.0
    alibi_slopes = None
    deterministic = False
    ref_dtype = torch.float32
    is_varlen = True if layout == "thd" else False

    # remove cache
    cache_path = Path(os.path.expanduser("~/.triton/cache"))
    if cache_path.exists():
        shutil.rmtree(cache_path)
        os.makedirs(cache_path)

    # inputs
    q, k, v, do, metadata = input_helper(BATCH, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, causal, dropout_p, ref_dtype, layout=layout, packing=packing, device=device)

    if packing == None:
        # fp8 forward pass
        if is_varlen:
            out, lse, S_dmask = flash_attn_varlen_fp8_func(
                q,
                k,
                v,
                metadata.cu_seqlens_q,
                metadata.cu_seqlens_k,
                metadata.max_seqlens_q,
                metadata.max_seqlens_k,
                dropout_p,
                causal=causal,
                window_size=window_size,
                softcap=softcap,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                return_attn_probs=True,
            )
        else:
            out, lse, S_dmask = flash_attn_fp8_func(
                    q,
                    k,
                    v,
                    dropout_p,
                    causal=causal,
                    window_size=window_size,
                    softcap=softcap,
                    alibi_slopes=alibi_slopes,
                    deterministic=deterministic,
                    return_attn_probs=True,
                )

        # fp8 backward pass
        if test_backward:
            dq, dk, dv = torch.autograd.grad(out, (q, k, v), do)
    elif packing == "qkv":
        # qkv packing path
        # pack input tensors (use dim=1 for varlen, else dim=2)
        if is_varlen:
            qkv = torch.stack([q, k, v], dim=1)
        else:
            qkv = torch.stack([q, k, v], dim=2)

        # fp8 forward pass for qkv-packed input
        if is_varlen:
            out, lse, S_dmask = flash_attn_varlen_qkvpacked_fp8_func(
                qkv,
                metadata.cu_seqlens_q,
                metadata.max_seqlens_q,
                dropout_p,
                causal=causal,
                window_size=window_size,
                softcap=softcap,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                return_attn_probs=True,
            )
        else:
            out, lse, S_dmask = flash_attn_qkvpacked_fp8_func(
                qkv,
                dropout_p,
                causal=causal,
                window_size=window_size,
                softcap=softcap,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                return_attn_probs=True,
            )

        # fp8 backward pass for qkv-packed input
        if test_backward:
            dqkv, = torch.autograd.grad(out, (qkv,), do)
    else:
        raise ValueError(f"unknown packing type {packing}")
    
    # search for .ttir files
    max_retries = 5
    retry_delay = 0.5
    ttir_files = []
    logging.info(f"Checking for .ttir files in {cache_path}...")
    for attempt in range(max_retries):
        # search for .ttir files recursively within the cache path
        ttir_files = glob.glob(str(cache_path) + "/**/*.ttir", recursive=True)

        if ttir_files:
            # Files found, log success and exit the loop
            logging.info(f"Found {len(ttir_files)} .ttir files on attempt {attempt + 1}.")
            break
        else:
            # Files not found yet
            if attempt < max_retries - 1:
                # If not the last attempt, wait and log before retrying
                logging.warning(
                    f"No .ttir files found on attempt {attempt + 1}. "
                    f"Retrying in {retry_delay}s..."
                )
                time.sleep(retry_delay)
            else:
                pytest.fail(
                    f"FATAL: No .ttir files found in cache {cache_path} "
                    f"after {max_retries} attempts."
                )

    # check if there is fp8
    ttir_files_fp8_found_status = {}
    fp8_types = ['f8E4M3', 'f8E5M2']
    for ttir_file in ttir_files:
        base_name = os.path.basename(ttir_file)
        with open(ttir_file, 'r') as f:
            content = f.read()

            # check content for fp8
            fp8_found = False
            for f8_type in fp8_types:
                if f8_type in content:
                    fp8_found = True
            ttir_files_fp8_found_status[base_name] = fp8_found

    for file, fp8_found in ttir_files_fp8_found_status.items():
        assert fp8_found, f"{fp8_types} not found in {file}"
