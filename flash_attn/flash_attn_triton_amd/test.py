import torch
import pytest

from .utils import MetaData, get_input_shapes, input_helper, varlen_input_helper, DEBUG
from .interface_torch import attention_prefill, attention_decode
from .fwd_ref import attention_forward_pytorch_ref_impl, compute_alibi_tensor_ref
from .fwd_prefill import attention_prefill_forward_triton_impl
from .bwd_prefill import attention_prefill_backward_triton_impl
from .bwd_ref import attention_backward_pytorch_ref_impl
from .fwd_decode import dequantize_kv_fp16, quantize_kv_int4

# defailt fp16 tolerance is ATOL, RTOL = 1e-5, 1e-3. See table https://pytorch.org/docs/stable/testing.html
ATOL, RTOL = 1e-2, 1e-2 # old standard. maybe to lose. 
# ATOL, RTOL = 1e-3, 1e-3  # catchs fa mismatch issues
# ATOL, RTOL = 1e-4, 1e-3 # to strict. there will be small diffs
# ATOL, RTOL = 1e-5, 1e-3 # # default fp16. there will be small diffs
EQUAL_NAN = True

@pytest.mark.parametrize('Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD', [
    (4, 48, 24, 1024, 1024, 64),
    (1, 24, 6, 8192, 8192, 64),
    (1, 4, 2, 16384, 16384, 128),
    (2, 16, 4, 1020, 987, 128),
    (2, 16, 4, 15498, 2, 128),
    (2, 16, 2, 7, 16219, 64),
    (4, 48, 12, 1, 1, 64),
    (4, 48, 48, 1, 1, 128),
    (4, 48, 24, 3, 3, 128),
    (4, 48, 48, 1001, 990, 64),
    (1, 8, 8, 8081, 7099, 64),
    (1, 4, 4, 16330, 15989, 128),
    (4, 4, 1, 1024, 1024, 33),
    (4, 4, 2, 65, 1018, 65),
    (4, 4, 4, 128, 128, 65),
    (4, 4, 4, 113, 123, 1),
])
@pytest.mark.parametrize('causal', [True, False])
@pytest.mark.parametrize('use_alibi', [True, False])
@pytest.mark.parametrize('layout', ['bshd', 'bhsd'])
def test_op_fwd_prefill(Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, causal, use_alibi, layout, dtype=torch.float16):
    torch.manual_seed(20)
    q, k, v, input_metadata = input_helper(Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype, layout)
    if causal:
        input_metadata.need_causal()

    if use_alibi:
        # for n heads the set of slopes is the geometric sequence that starts 2^(-8/n)
        alibi_slopes = torch.tensor([2**(-8 / HQ * i) for i in range(1, HQ + 1)], dtype=torch.float32,
                                    device="cuda").repeat(Z, 1)
        input_metadata.need_alibi(alibi_slopes, Z, HQ)
    else:
        alibi_slopes = None

    o = torch.empty_like(q)

    # triton implementation
    tri_out, _, _ = attention_prefill(q, k, v, o, input_metadata)

    # Transpose here if layout is bshd so we have same reference code for all layouts
    if layout == 'bshd':
        q = q.transpose(1, 2).clone()
        k = k.transpose(1, 2).clone()
        v = v.transpose(1, 2).clone()
    # Replicate K and V if using MQA/GQA
    if HQ != HK:
        k = k.view(k.shape[0], k.shape[1], -1, k.shape[2],
                   k.shape[3]).expand(-1, -1, HQ // HK, -1, -1).reshape(k.shape[0], -1, k.shape[2], k.shape[3])
        v = v.view(v.shape[0], v.shape[1], -1, v.shape[2],
                   v.shape[3]).expand(-1, -1, HQ // HK, -1, -1).reshape(v.shape[0], -1, v.shape[2], v.shape[3])

    scores = torch.einsum('bhqd,bhkd->bhqk', q, k).float() * input_metadata.sm_scale
    if causal:
        mask = torch.tril(torch.ones(N_CTX_Q, N_CTX_K, device="cuda"), diagonal=N_CTX_K - N_CTX_Q)
        scores[:, :, mask == 0] = float("-inf")
    if use_alibi:
        scores += compute_alibi_tensor_ref(alibi_slopes, N_CTX_Q, N_CTX_K)

    p = torch.softmax(scores, dim=-1)
    if causal:
        # If N_CTX_Q > N_CTX_K, there is at least one row of all -infs going into
        # the softmax. This produces a row of NaNs as -inf - -inf == NaN. So we fix
        # this by converting the NaNs to 0s, which is what they should be out of the softmax.
        nan_mask = torch.isnan(p)
        p[nan_mask == 1] = 0
    ref_out = torch.einsum('bhqk,bhkd->bhqd', p.half(), v)
    # compare
    if layout == 'bshd':
        ref_out = ref_out.transpose(1, 2).clone()
    torch.testing.assert_close(ref_out, tri_out, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize('Z, H, N_CTX_Q, N_CTX_K, D_HEAD', [
    (4, 48, 1024, 1024, 64),
    (4, 12, 8192, 8192, 64),
    (2, 4, 16384, 16384, 128),
    (2, 16, 15498, 2, 128),
    (2, 4, 7, 16219, 64),
    (4, 48, 1, 1, 64),
    (4, 48, 1, 1, 128),
    (4, 48, 3, 3, 128),
    (4, 48, 1001, 990, 64),
    (1, 8, 8081, 7099, 64),
    (1, 8, 16330, 15989, 128),
    (4, 4, 1024, 1024, 33),
    (4, 4, 65, 1019, 65),
    (4, 4, 128, 128, 65),
    # TODO: This config fails. Disabled until triaged and fixed.
    #  (2, 16, 1020, 987, 128),
    #   (4, 4, 113, 123, 1),
])
@pytest.mark.parametrize('causal', [True, False])
@pytest.mark.parametrize('use_bias', [True])
def test_op_fwd_prefill_bias(Z, H, N_CTX_Q, N_CTX_K, D_HEAD, causal, use_bias, dtype=torch.float16):
    torch.manual_seed(20)
    sm_scale = D_HEAD**-0.5
    input_metadata = MetaData(sm_scale=sm_scale)
    q, k, v, input_metadata = input_helper(Z, H, H, N_CTX_Q, N_CTX_K, D_HEAD, dtype, layout='bhsd')
    if causal:
        input_metadata.need_causal()
    if use_bias:
        bias = torch.randn((1, H, N_CTX_Q, N_CTX_K), dtype=torch.float32, device="cuda")
        input_metadata.need_bias(bias, Z, H, N_CTX_Q, N_CTX_K)
    else:
        bias = None
    o = torch.empty_like(q)

    # triton implementation
    tri_out, _, _ = attention_prefill(q, k, v, o, input_metadata)
    # reference implementation:171

    scores = torch.einsum('bhqd,bhkd->bhqk', q, k).float() * sm_scale
    if causal:
        mask = torch.tril(torch.ones(N_CTX_Q, N_CTX_K, device="cuda"), diagonal=N_CTX_K - N_CTX_Q)
        scores[:, :, mask == 0] = float("-inf")
    if use_bias:
        scores += input_metadata.bias
    p = torch.softmax(scores, dim=-1)
    if causal:
        # If N_CTX_Q > N_CTX_K, there is at least one row of all -infs going into
        # the softmax. This produces a row of NaNs as -inf - -inf == NaN. So we fix
        # this by converting the NaNs to 0s, which is what they should be out of the softmax.
        nan_mask = torch.isnan(p)
        p[nan_mask == 1] = 0
    ref_out = torch.einsum('bhqk,bhkd->bhqd', p.half(), v)
    # compare
    torch.testing.assert_close(ref_out, tri_out, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize('Z, H, N_CTX, D_HEAD', [
                                                (4, 48, 8192, 64), 
                                                 (4, 48, 256, 64), 
                                                 (4, 48, 512, 64),
                                                 (4, 48, 1024, 64), 
                                                 (8, 48, 4096, 64), 
                                                 (4, 48, 8192, 64),
                                                 (4, 48, 128, 128), 
                                                 (4, 48, 4096, 128), 
                                                 (4, 48, 16384, 128),
                                                 (4, 16, 1024, 128), 
                                                 (4, 16, 8192, 128), 
                                                 (32, 48, 8192, 128)
                                                 ]
                                                 )
@pytest.mark.parametrize('causal', [True, False])
def test_op_varlen_fwd(Z, H, N_CTX, D_HEAD, causal, dtype=torch.float16):

    q, k, v, input_metadata = varlen_input_helper(Z, H, H, N_CTX, N_CTX, D_HEAD, dtype)

    tri_out = torch.empty_like(q)
    ref_out = torch.empty_like(q)

    for i in range(0, input_metadata.num_contexts):
        start_q, start_k = input_metadata.cu_seqlens_q[i], input_metadata.cu_seqlens_k[i]
        end_q, end_k = input_metadata.cu_seqlens_q[i + 1], input_metadata.cu_seqlens_k[i + 1]
        scores = torch.einsum('qhd,khd->qhk', q[start_q:end_q], k[start_k:end_k]).float()
        p = torch.softmax(scores * input_metadata.sm_scale, dim=-1).half()
        ref_out[start_q:end_q] = torch.einsum('qhk,khd->qhd', p, v[start_k:end_k])
    attention_prefill(q, k, v, tri_out, input_metadata)
    torch.testing.assert_close(ref_out, tri_out, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize('Z, HQ, HK, N_CTX, D_HEAD', [(2, 48, 24, 128, 64), (4, 48, 12, 256, 64), (4, 48, 4, 512, 64),
                                                      (4, 48, 2, 1024, 64), (8, 48, 6, 4096, 64), (4, 48, 8, 16384, 64),
                                                      (4, 64, 16, 128, 128), (4, 64, 4, 4096, 128),
                                                      (4, 64, 8, 16384, 128), (4, 16, 4, 1024, 128),
                                                      (4, 16, 2, 8192, 128), (32, 128, 32, 8192, 128)])
@pytest.mark.parametrize('causal', [False])
def test_op_varlen_mqa_fwd(Z, HQ, HK, N_CTX, D_HEAD, causal, dtype=torch.float16):
    q, k, v, input_metadata = varlen_input_helper(Z, HQ, HK, N_CTX, N_CTX, D_HEAD, dtype)
    ref_out = torch.empty_like(q)
    tri_out = torch.empty_like(q)
    # Make KV look like HQ/HK "groups" of HK. Later, we will reshape so the
    # size aligns with Q.
    k_ref = k.view(k.shape[0], k.shape[1], 1, k.shape[2]).expand(-1, -1, HQ // HK, -1)
    v_ref = v.view(v.shape[0], v.shape[1], 1, v.shape[2]).expand(-1, -1, HQ // HK, -1)
    for i in range(0, input_metadata.num_contexts):
        start_q, start_k = input_metadata.cu_seqlens_q[i], input_metadata.cu_seqlens_k[i]
        end_q, end_k = input_metadata.cu_seqlens_q[i + 1], input_metadata.cu_seqlens_k[i + 1]
        k_curr = k_ref[start_k:end_k]
        k_curr = k_curr.reshape(k_curr.shape[0], -1, k_curr.shape[3])
        v_curr = v_ref[start_k:end_k]
        v_curr = v_curr.reshape(v_curr.shape[0], -1, v_curr.shape[3])
        scores = torch.einsum('qhd,khd->qhk', q[start_q:end_q], k_curr).float()
        p = torch.softmax(scores * input_metadata.sm_scale, dim=-1).half()
        ref_out[start_q:end_q] = torch.einsum('qhk,khd->qhd', p, v_curr)
    attention_prefill(q, k, v, tri_out, input_metadata)
    torch.testing.assert_close(ref_out, tri_out, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize('Z, H, N_CTX_Q, N_CTX_K, D_HEAD', [
    # smallest config test
    (1, 1, 16, 16, 64), # pass on new # fail on old
    (1, 1, 32, 32, 64), # pass on new # fail on old
    (1, 1, 64, 64, 16), # pass # smallest head_size = 16
    (1, 1, 64, 64, 64), # pass # smallest seq len seems to be 64
    (1, 1, 128, 128, 64), # pass
    (1, 1, 256, 256, 64), # pass
    (1, 1, 512, 512, 64), # pass
    # failing FA
    (1, 1, 256, 512, 16),
    # old tests that work
    (4, 48, 1024, 1024, 64), # pass
    (4, 48, 2048, 2048, 64), # pass
    (2, 48, 4096, 4096, 64), # pass
    (1, 16, 1024, 1024, 64), # pass
    (1, 16, 1024, 1024, 128), # pass
    # old tests that were commented out
    # (1, 16, 8192, 8192, 63),
    # (1, 16, 1022, 1022, 64),
])
# @pytest.mark.parametrize('torch_sdpa_test', [False, True])
@pytest.mark.parametrize('torch_sdpa_test', [False])
# @pytest.mark.parametrize('causal', [True, False])
@pytest.mark.parametrize('causal', [False])
# @pytest.mark.parametrize('use_alibi', [False, True])
@pytest.mark.parametrize('use_alibi', [False])
def test_op_bwd(Z, H, N_CTX_Q, N_CTX_K, D_HEAD, causal, torch_sdpa_test, use_alibi, dtype=torch.float16):
    torch.manual_seed(20)

    DEBUG_INPUT = False

    # seqlens
    seqlen_q = N_CTX_Q
    seqlen_k = N_CTX_K

    # setup up metadata
    if DEBUG_INPUT:
        sm_scale = 1
    else:
        sm_scale = D_HEAD**-0.5
    input_metadata = MetaData(sm_scale=sm_scale)
    input_metadata.max_seqlens_q = seqlen_q
    input_metadata.max_seqlens_k = seqlen_k
    input_metadata.layout = "bhsd"

    dropout_p = 0
    if DEBUG_INPUT:
        q = torch.arange(seqlen_q, dtype=dtype, device="cuda").view(1, 1, seqlen_q, 1).expand(Z, H, seqlen_q, D_HEAD).requires_grad_()
        k = torch.arange(seqlen_k, dtype=dtype, device="cuda").view(1, 1, seqlen_k, 1).expand(Z, H, seqlen_k, D_HEAD).requires_grad_()
        v = torch.arange(seqlen_k, dtype=dtype, device="cuda").view(1, 1, seqlen_k, 1).expand(Z, H, seqlen_k, D_HEAD).requires_grad_()
        o = torch.zeros_like(q)
    else:
        # Generate random inputs
        q = torch.randn(Z, H, N_CTX_Q, D_HEAD, device='cuda', dtype=dtype, requires_grad=True)
        k = torch.randn(Z, H, N_CTX_K, D_HEAD, device='cuda', dtype=dtype, requires_grad=True)
        v = torch.randn(Z, H, N_CTX_K, D_HEAD, device='cuda', dtype=dtype, requires_grad=True)
        o = torch.empty_like(q)

    if causal:
        input_metadata.need_causal()

    if use_alibi and not torch_sdpa_test:
        # for n heads the set of slopes is the geometric sequence that starts 2^(-8/n)
        alibi_slopes = torch.tensor([2**(-8 / H * i) for i in range(1, H + 1)], dtype=torch.float32,
                                    device="cuda").repeat(Z, 1)
        input_metadata.need_alibi(alibi_slopes, Z, H)

    if DEBUG_INPUT:
        dout = torch.ones_like(q)
    else:
        dout = torch.randn_like(q)

    # reference implementation
    if torch_sdpa_test:
        ref_out, ref_softmax = torch.ops.aten._scaled_dot_product_attention_math(q, k, v, dropout_p=dropout_p,
                                                                                 is_causal=causal, scale=sm_scale,
                                                                                 dropout_mask=None)
        ref_out.backward(dout.to(device=ref_out.device, dtype=ref_out.dtype))
        ref_dv, v.grad = v.grad.clone(), None
        ref_dk, k.grad = k.grad.clone(), None
        ref_dq, q.grad = q.grad.clone(), None
    else:
        M = torch.tril(torch.ones((seqlen_q, seqlen_k), device="cuda"))
        p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
        if use_alibi:
            p += compute_alibi_tensor_ref(alibi_slopes, N_CTX_Q, N_CTX_K)
        if causal:
            p[:, :, M == 0] = float("-inf")

        p = torch.softmax(p.float(), dim=-1).type(dtype=p.dtype)
        ref_out = torch.matmul(p, v)
        ref_out.backward(dout)
        ref_dv, v.grad = v.grad.clone(), None
        ref_dk, k.grad = k.grad.clone(), None
        ref_dq, q.grad = q.grad.clone(), None

    # # triton implementation
    tri_out, _, _ = attention_prefill(q, k, v, o, input_metadata)
    tri_out.backward(dout)
    tri_dv, v.grad = v.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dq, q.grad = q.grad.clone(), None
    # compare
    if DEBUG:
        print("tri_out:", tri_out)
        print("ref_out:",ref_out )
    torch.testing.assert_close(ref_out, tri_out, atol=1e-2, rtol=0)
    
    # The current block size for MI200 series is 64x64. This results in
    # larger differences in float results due to rounding.
    if dtype == torch.bfloat16:
        ATOL = 1e-1 * max(1.0, (seqlen_q + D_HEAD) / 64.0)
    if dtype == torch.float32:
        ATOL = 1e-3 * max(1.0, (seqlen_q + D_HEAD) / 64.0)
    else:
        ATOL = 1e-1 * max(1.0, (seqlen_q + D_HEAD) / 64.0)

    RTOL = 0

    if DEBUG:
        print("ref_dv:", ref_dv)
        print("tri_dv:", tri_dv)
        print("ref_dk:", ref_dk)
        print("tri_dk:", tri_dk)
        print("ref_dq:", ref_dq)
        print("tri_dq:", tri_dq)

    torch.testing.assert_close(ref_dv, tri_dv, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(ref_dk, tri_dk, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(ref_dq, tri_dq, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize('Z, H, N_CTX_Q, N_CTX_K, D_HEAD', [
    (1, 1, 1, 1, 1),
    (1, 1, 2, 4, 16),
    (1, 1, 4, 2, 16),
    (1, 1, 4, 4, 16),
    (1, 2, 4, 4, 16),
    (2, 1, 4, 4, 16),
    (2, 2, 4, 4, 16),
    (1, 1, 128, 64, 16),
    (2, 2, 2, 128, 1),
    (2, 3, 2, 128, 16),
    (3, 2, 256, 512, 16),
    (3, 3, 128, 128, 64),
    (2, 4, 1024, 1024, 64),
    (4, 6, 108, 256, 224),
    (4, 8, 2048, 2048, 128),
    (4, 16, 4096, 4096, 64),
    (2, 4, 8192, 8192, 32),
    # # fa configs
    (4, 6, 113, 203, 256),
    (4, 6, 128, 217, 256),
    (4, 6, 113, 211, 128),
    (4, 6, 108, 256, 128),
    (4, 6, 256, 512, 64),
    (4, 6, 512, 256, 64),
    (4, 6, 1024, 1024, 32),
    (4, 6, 1023, 1024, 32),
    (4, 6, 1024, 1023, 32),
    (4, 6, 2048, 2048, 32),
])
@pytest.mark.parametrize('causal', [True, False])
@pytest.mark.parametrize('return_scores', [False])
@pytest.mark.parametrize('layout', ["bhsd", "bshd", "thd"])
@pytest.mark.parametrize('use_exp2', [True, False]) # works when use_exp2 is false
@pytest.mark.parametrize('DEBUG_INPUT', [False]) # NOTE: debug input can overflow when the tensors are large. Just use to figure out issues
def test_op_prefill_fwd_impl(Z, H, N_CTX_Q, N_CTX_K, D_HEAD, causal, return_scores, layout, use_exp2, DEBUG_INPUT):
    dtype = torch.float16
    torch.manual_seed(0)
    alibi_slopes = None
    dropout_p = 0.0
    device = "cuda"

    if layout == "thd":
        q, k, v, metadata = varlen_input_helper(Z, H, H, N_CTX_Q, N_CTX_K, D_HEAD, dtype, device=device, DEBUG_INPUT=DEBUG_INPUT)
    else:
        q, k, v, metadata = input_helper(Z, H, H, N_CTX_Q, N_CTX_K, D_HEAD, dtype, layout, device=device, DEBUG_INPUT=DEBUG_INPUT)
    if DEBUG_INPUT:
        output_triton = torch.zeros_like(q).contiguous()
    else:
        output_triton = torch.empty_like(q)

    # update metadata
    metadata.use_exp2 = use_exp2
    if causal:
        metadata.need_causal()

    # NOTE: the returned score is not the same as the reference because we need to adjust as we find new maxes per block. We are not doing that
    if return_scores:
        metadata.return_scores = True

    # call Triton's forward implementation directly
    ( output_triton, 
        softmax_lse_triton, 
        exp_scores_triton, 
        _, 
        _, 
        _, 
        _, 
        _, 
        _) = attention_prefill_forward_triton_impl(
                                                q, 
                                                k, 
                                                v, 
                                                output_triton, 
                                                metadata.sm_scale, 
                                                metadata.alibi_slopes, 
                                                metadata.causal, 
                                                metadata.bias, 
                                                metadata.dropout_p, 
                                                metadata.layout, 
                                                metadata.cu_seqlens_q, 
                                                metadata.cu_seqlens_k,
                                                metadata.max_seqlens_q, 
                                                metadata.max_seqlens_k, 
                                                metadata.return_scores, 
                                                metadata.use_exp2)

    (
        output_ref,
        softmax_lse_ref,
        exp_scores_ref,
        softmax_ref,
        attention_shifted_scaled_scores_ref,
        attention_scaled_scores_ref,
        attention_scores_ref,
    ) = attention_forward_pytorch_ref_impl(
        q.clone(), 
        k.clone(), 
        v.clone(), 
        metadata.sm_scale, 
        causal, 
        layout,
        metadata.cu_seqlens_q,
        metadata.cu_seqlens_k,
        metadata.max_seqlens_q,
        metadata.max_seqlens_k,
        use_exp2
    )

    if DEBUG:
        print("softmax_lse_triton:", softmax_lse_triton, softmax_lse_triton.shape)
        print("softmax_lse_ref:", softmax_lse_ref, softmax_lse_ref.shape)
    torch.testing.assert_close(softmax_lse_triton, softmax_lse_ref, atol=ATOL, rtol=RTOL)

    if layout != "thd":
        # use trick with lse to get the softmax. you need the scores but is it
        softmax_triton = torch.exp(attention_scaled_scores_ref - softmax_lse_triton.unsqueeze(-1))
        if DEBUG:
            print("attention_scaled_scores_ref:", attention_scaled_scores_ref, attention_scaled_scores_ref.shape)
            print("softmax_lse_triton:", softmax_lse_triton, softmax_lse_triton.shape)
            print("softmax_triton:", softmax_triton, softmax_triton.shape)
            print("softmax_ref:", softmax_ref, softmax_ref.shape)
        torch.testing.assert_close(softmax_triton, softmax_ref, atol=ATOL, rtol=RTOL)
    
    if DEBUG:
        print("output_triton:", output_triton, output_triton.shape)
        print("output_ref:", output_ref, output_ref.shape)
    torch.testing.assert_close(output_triton, output_ref, atol=ATOL, rtol=RTOL)


    # compare with pytorch expect thd and causal impl is different
    if False and layout in ["bhsd", "bshd"] and not causal:
        out_pytorch, softmax_pytorch = torch.ops.aten._scaled_dot_product_attention_math(
                                                                            q.transpose(1, 2) if layout == "bshd" else q , 
                                                                            k.transpose(1, 2) if layout == "bshd" else k, 
                                                                            v.transpose(1, 2) if layout == "bshd" else v, 
                                                                            dropout_p=dropout_p,
                                                                            is_causal=causal, scale=metadata.sm_scale,
                                                                            dropout_mask=None)
        out_pytorch = out_pytorch.transpose(1, 2) if layout == "bshd" else out_pytorch

        if DEBUG:
            print("o:", output_triton, output_triton.shape)
            print("out_pytorch:", out_pytorch, out_pytorch.shape)
        torch.testing.assert_close(output_triton, out_pytorch, atol=ATOL, rtol=RTOL)
        
        # compare with pytorch output
        if DEBUG:
            print("softmax_triton:", softmax_triton, softmax_triton.shape)
            print("softmax_pytorch:", softmax_pytorch, softmax_pytorch.shape)
        torch.testing.assert_close(softmax_triton, softmax_pytorch.to(torch.float32), atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize('Z, H, N_CTX_Q, N_CTX_K, D_HEAD', [
    (1, 1, 1, 1, 1),
    (1, 1, 4, 4, 4),
    (2, 1, 4, 4, 16),
    (1, 2, 4, 4, 16),
    (2, 2, 4, 4, 16),
    (1, 1, 4, 4, 16),
    (2, 1, 4, 4 , 16),
    (4, 6, 8, 8 , 16),
    (1, 1, 4, 4, 32),
    (1, 1, 16, 16, 16),
    (1, 1, 32, 32, 16),
    (1, 1, 64, 64, 16),
    (1, 1, 64, 64, 64),
    (1, 1, 64, 128, 32),
    (1, 1, 128, 128, 64),
    (1, 1, 128, 256, 45),
    (1, 1, 113, 203, 192),
    (1, 1, 256, 256, 64),
    (1, 1, 256, 512, 16),
    (1, 1, 512, 512, 64), 
    (1, 1, 1024, 1024, 64),
    # fa configs
    (2, 2, 128, 128, 65),
    (2, 2, 128, 128, 224),
    (4, 6, 108, 256, 224),
    (1, 1, 256, 512, 16),
    # old tests that work
    (4, 48, 1024, 1024, 73),
    (4, 48, 1024, 1024, 64),
    (4, 48, 2048, 2048, 64),
    (1, 24, 4096, 4096, 64),
    (1, 16, 1024, 1024, 64),
    (1, 16, 1024, 1024, 128),
])
@pytest.mark.parametrize('causal', [True, False])
@pytest.mark.parametrize('use_exp2', [False]) # FIXME: using exp2 causes issue when used with causal
@pytest.mark.parametrize('layout', ["bhsd", "bshd", "thd"])
@pytest.mark.parametrize('sequence_parallel', [True, False])
@pytest.mark.parametrize('DEBUG_INPUT', [False]) # debug output causes nans in both new and old backend
def test_op_prefill_bwd_impl(Z, H, N_CTX_Q, N_CTX_K, D_HEAD, causal, use_exp2, layout, sequence_parallel, DEBUG_INPUT):
    dtype = torch.float16
    torch.manual_seed(20) # seed from test_op_bwd

    alibi_slopes = None
    if layout == "thd":
        q, k, v, metadata = varlen_input_helper(Z, H, H, N_CTX_Q, N_CTX_K, D_HEAD, dtype, DEBUG_INPUT=DEBUG_INPUT)
    else:
        q, k, v, metadata = input_helper(Z, H, H, N_CTX_Q, N_CTX_K, D_HEAD, dtype, layout, DEBUG_INPUT=DEBUG_INPUT)
    if DEBUG_INPUT:
        do = torch.ones_like(q).contiguous()
    else:
        do = torch.randn_like(q)

    # =============================================== Reference ==============================================================
    q_ref = q.clone() 
    k_ref = k.clone()
    v_ref = v.clone()    
    (
        o_ref,
        softmax_lse_ref,
        _,
        _,
        _,
        _,
        _,
    ) = attention_forward_pytorch_ref_impl(
        q_ref,
        k_ref, 
        v_ref,
        metadata.sm_scale, 
        causal, 
        layout,
        metadata.cu_seqlens_q,
        metadata.cu_seqlens_k,
        metadata.max_seqlens_q,
        metadata.max_seqlens_k,
        use_exp2
    )

    dq = torch.zeros_like(q, dtype=q.dtype) # NOTE: the kernel does inplace accumlation on dq so dq has to be zeros
    if DEBUG_INPUT:
        dk = torch.zeros_like(k, dtype=k.dtype)
        dv = torch.zeros_like(v, dtype=v.dtype)
    else:
        dk = torch.empty_like(k, dtype=k.dtype)
        dv = torch.empty_like(v, dtype=v.dtype)

    do_ref = do.clone()
    dq_ref, dk_ref, dv_ref, delta_ref = attention_backward_pytorch_ref_impl(
        do_ref,
        q_ref,
        k_ref,
        v_ref,
        o_ref,
        softmax_lse_ref,
        metadata.sm_scale,
        causal,
        layout,
        metadata.cu_seqlens_q,
        metadata.cu_seqlens_k,
        metadata.max_seqlens_q,
        metadata.max_seqlens_k,
        use_exp2
    )

    # =============================================== Triton ==============================================================
    o = o_ref.clone().contiguous()
    softmax_lse = softmax_lse_ref.clone().contiguous()
    dq_triton, dk_triton, dv_triton, delta_triton, _, _ = attention_prefill_backward_triton_impl(
        do,
        q,
        k,
        v,
        o,
        softmax_lse,
        dq,
        dk,
        dv,
        metadata.sm_scale,
        alibi_slopes,
        causal,
        layout,
        metadata.cu_seqlens_q,
        metadata.cu_seqlens_k,
        metadata.max_seqlens_q,
        metadata.max_seqlens_k,
        use_exp2,
        sequence_parallel=sequence_parallel
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


@pytest.mark.parametrize('batch_size, seqlen_q, seqlen_k, group_q, group_k, dim', get_input_shapes())
def test_op_fwd_decode(batch_size, seqlen_q, seqlen_k, group_q, group_k, dim, dtype=torch.bfloat16):
    if DEBUG:
        print()
        print(f"batch_size = {batch_size}, seqlen_q = {seqlen_q}, seqlen_k = {seqlen_k}, group_q = {group_q}, group_k = {group_k}, dim = {dim}")
    torch.manual_seed(20)
    query_group_head_size = (group_q + group_k - 1) // group_k
    q = (torch.empty((batch_size, seqlen_q, group_k, query_group_head_size, dim), dtype=dtype,
                     device="cuda").normal_(mean=0., std=0.5).requires_grad_())
    k = (torch.empty((batch_size, seqlen_k, group_k, 1, dim), dtype=dtype,
                     device="cuda").normal_(mean=0.,
                                            std=0.5).requires_grad_()).expand(-1, -1, -1, query_group_head_size, -1)
    v = (torch.empty((batch_size, seqlen_k, group_k, 1, dim), dtype=dtype,
                     device="cuda").normal_(mean=0.,
                                            std=0.5).requires_grad_()).expand(-1, -1, -1, query_group_head_size, -1)
    scale = 1 / dim**0.5
    input_metadata = MetaData(sm_scale=scale)
    input_metadata.layout = "bsghd"
    tri_out, _ = attention_decode(q, k, v, input_metadata)

    q = q.reshape([batch_size, seqlen_q, -1, dim]).permute(0, 2, 1, 3)
    k = k.reshape([batch_size, seqlen_k, -1, dim]).permute(0, 2, 1, 3)
    v = v.reshape([batch_size, seqlen_k, -1, dim]).permute(0, 2, 1, 3)
    attn = (q @ k.transpose(-1, -2) * scale).softmax(-1)
    ref_out = attn @ v

    # compare
    torch.testing.assert_close(ref_out, tri_out, atol=1e-3, rtol=0)

def test_quantization():
    a = torch.randn((2, 4, 32), dtype=torch.float16, device='cuda')
    qa = quantize_kv_int4(a, num_groups=4)
    dqa = dequantize_kv_fp16(qa, num_groups=4)
    torch.testing.assert_close(a, dqa, atol=1.5e-1, rtol=1e-1)

@pytest.mark.parametrize('B, Mq, Mkv, Hq, Hkv, K', get_input_shapes())
def test_op_fwd_decode_int4_kv(B, Mq, Mkv, Hq, Hkv, K, dtype=torch.float16):
    pytest.skip("Decode kernel doesnot support quantization yet")
    torch.manual_seed(2)
    q = (torch.empty((B, Mq, Hkv, (Hq + Hkv - 1) // Hkv, K), dtype=dtype,
                     device="cuda").normal_(mean=1.0, std=0.5).requires_grad_())
    k = (torch.empty((B, Mkv, Hkv, 1, K), dtype=dtype,
                     device="cuda").normal_(mean=1.0,
                                            std=0.5).requires_grad_()).expand(-1, -1, -1, (Hq + Hkv - 1) // Hkv, -1)
    v = (torch.empty((B, Mkv, Hkv, 1, K), dtype=dtype,
                     device="cuda").normal_(mean=1.0,
                                            std=0.5).requires_grad_()).expand(-1, -1, -1, (Hq + Hkv - 1) // Hkv, -1)

    num_groups = 1
    quant_k = (quantize_kv_int4(k, num_groups=num_groups).contiguous().view(torch.int32))
    quant_v = (quantize_kv_int4(v, num_groups=num_groups).contiguous().view(torch.int32))
    scale = 1 / K**0.5
    input_metadata = MetaData(sm_scale=scale)
    input_metadata.layout = "bsghd"
    tri_out, _ = attention_decode(q, quant_k, quant_v, input_metadata)

    q = q.reshape([B, Mq, -1, K]).permute(0, 2, 1, 3)
    k = k.reshape([B, Mkv, -1, K]).permute(0, 2, 1, 3)
    v = v.reshape([B, Mkv, -1, K]).permute(0, 2, 1, 3)
    attn = (q @ k.transpose(-1, -2) * scale).softmax(-1)
    ref_out = attn @ v
    # compare
    torch.testing.assert_close(ref_out, tri_out, atol=2.1e-2, rtol=0)

    # since quantization introduces rounding error, use the
    # dequantized kv as inputs to the ref implementation to reduce
    # the tolerance to 1e-3
    dqk = dequantize_kv_fp16(quant_k, num_groups=num_groups)
    dqv = dequantize_kv_fp16(quant_v, num_groups=num_groups)
    dqk = dqk.reshape([B, Mkv, -1, K]).permute(0, 2, 1, 3)
    dqv = dqv.reshape([B, Mkv, -1, K]).permute(0, 2, 1, 3)
    dq_attn = (q @ dqk.transpose(-1, -2) * scale).softmax(-1)
    dq_ref_out = dq_attn @ dqv
    torch.testing.assert_close(dq_ref_out, tri_out, atol=1e-3, rtol=0)
