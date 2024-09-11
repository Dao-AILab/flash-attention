import math

import pytest
import torch
import torch.nn.functional as F

from einops import rearrange, repeat
from flash_attn_interface import flash_attn_func, flash_attn_varlen_func, _flash_attn_forward
from tests.test_util import generate_random_padding_mask, generate_qkv, construct_local_mask, attention_ref

ABS_TOL = 5e-3
REL_TOL = 1e-1

def print_diffs(out, out_ref):
    out_1d = out.flatten()
    out_ref_1d = out_ref.flatten()
    for idx, (e_o, e_o_ref) in enumerate(zip(out_1d, out_ref_1d)):
        diff = e_o - e_o_ref
        abs_diff = abs(diff)
        abs_ref = abs(e_o_ref + 1e-5)
        relative_diff = abs_diff / abs_ref
        if abs_diff > ABS_TOL or relative_diff > REL_TOL:
            print(f"==== diff ==== {idx}, test: {e_o}, ref: {e_o_ref}")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
# @pytest.mark.parametrize("dtype", [torch.float8_e4m3fn])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
# @pytest.mark.parametrize("mha_type", ["mha"])
@pytest.mark.parametrize("causal", [False, True])
# @pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("deterministic", [False, True])
# @pytest.mark.parametrize("deterministic", [True])
# @pytest.mark.parametrize("d", [32, 64, 96, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize('d', [32, 40, 64, 80, 96, 128, 160, 192])
# @pytest.mark.parametrize('d', [32, 64, 96, 128, 160, 192])
# @pytest.mark.parametrize('d', [56, 80])
# @pytest.mark.parametrize("d", [64, 128, 256])
# @pytest.mark.parametrize("d", [64, 96, 128])
# @pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("d", [64, 128, 256])
@pytest.mark.parametrize("descale", [1.0])
# @pytest.mark.parametrize("descale", [1.0, 2.0, 3.0, 4.0])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (1, 1),
        (257, 1),
        (64, 128),
        (128, 128),
        (256, 256),
        (113, 203),
        (128, 217),
        (113, 211),
        (108, 256),
        (256, 512),
        (384, 256),
        (640, 128),
        (512, 256),
        (1024, 1024),
        (1023, 1024),
        (1024, 1023),
        (4096, 4096),
    ],
)
# @pytest.mark.parametrize('seqlen_q,seqlen_k', [(128, 128)])
def test_flash_attn_output(
    seqlen_q, seqlen_k, d, causal, deterministic, mha_type, dtype, descale
):
    device = "cuda"
    if(dtype == torch.float8_e4m3fn):
        dtype_init = torch.float16
    else:
        dtype_init = dtype    
    print(dtype)
    # set seed
    torch.random.manual_seed(0)
    # batch_size = 40
    # nheads = 16
    batch_size = 4
    nheads = 6
    nheads_kv = 6 if mha_type == "mha" else (2 if mha_type == "gqa" else 1)
    # nheads_kv = 2
    # batch_size = 9
    # nheads = 6
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype_init, requires_grad=True)
    k = torch.randn(batch_size, seqlen_k, nheads_kv, d, device=device, dtype=dtype_init, requires_grad=True)
    v = torch.randn(batch_size, seqlen_k, nheads_kv, d, device=device, dtype=dtype_init, requires_grad=True)

    q = q.to(dtype)
    k = k.to(dtype)
    v = v.to(dtype)

    softmax_scale = q.shape[-1] ** (-0.5)
    descale_q = torch.tensor([descale], dtype=torch.float32, device='cuda')
    descale_k = torch.tensor([descale], dtype=torch.float32, device='cuda')
    descale_v = torch.tensor([descale], dtype=torch.float32, device='cuda')
    if(dtype != torch.float8_e4m3fn):
        out, lse = flash_attn_func(q, k, v, causal=causal, deterministic=deterministic)
    else:
        out, q, k, v, out_padded, lse, S_dmask = _flash_attn_forward(
            q, k, v, softmax_scale, causal, descale_q=descale_q, descale_k=descale_k, descale_v=descale_v
        )

    q = q.to(dtype_init)
    k = k.to(dtype_init)
    v = v.to(dtype_init)

    if(dtype == torch.float8_e4m3fn):
        descale_q = descale_q.to(dtype_init)
        descale_k = descale_k.to(dtype_init)
        descale_v = descale_v.to(dtype_init)
        q = q * descale_q
        k = k * descale_k
        v = v * descale_v
        
    out_ref, attn_ref = attention_ref(
        q,
        k,
        v,
        None,
        None,
        causal=causal,
    )
    out_pt, attn_pt = attention_ref(
        q,
        k,
        v,
        None,
        None,
        causal=causal,
        upcast=False,
        reorder_ops=True,
    )

    # qk = torch.einsum('bshd,bthd->bhst', q, k).float()
    # m = qk.amax(-1, keepdim=True)
    # s_tmp = torch.exp((qk - m) / math.sqrt(d))
    # exp_sum = s_tmp.sum(-1)
    # qk = torch.einsum('bthd,bshd->bhts', q.float() / math.sqrt(d), k.float())
    # lse_ref = torch.logsumexp(qk, dim=-1)

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")
    
    # if not causal:
    #     print(f"LSE max diff: {(lse - lse_ref).abs().max().item()}")                
    # breakpoint()

    if d <= 128 and dtype != torch.float8_e4m3fn:
        g = torch.randn_like(out)
        do_o = (g.float() * out.float()).sum(-1)
        dq, dk, dv = torch.autograd.grad(out, (q, k, v), g)
        dq_ref, dk_ref, dv_ref = torch.autograd.grad(out_ref, (q, k, v), g)
        dq_pt, dk_pt, dv_pt = torch.autograd.grad(out_pt, (q, k, v), g)
        print(f"dQ max diff: {(dq - dq_ref).abs().max().item()}")
        print(f"dK max diff: {(dk - dk_ref).abs().max().item()}")
        print(f"dV max diff: {(dv - dv_ref).abs().max().item()}")
        print(f"dQ mean diff: {(dq - dq_ref).abs().mean().item()}")
        print(f"dK mean diff: {(dk - dk_ref).abs().mean().item()}")
        print(f"dV mean diff: {(dv - dv_ref).abs().mean().item()}")
        print(f"dQ Pytorch max diff: {(dq_pt - dq_ref).abs().max().item()}")
        print(f"dK Pytorch max diff: {(dk_pt - dk_ref).abs().max().item()}")
        print(f"dV Pytorch max diff: {(dv_pt - dv_ref).abs().max().item()}")
        print(f"dQ Pytorch mean diff: {(dq_pt - dq_ref).abs().mean().item()}")
        print(f"dK Pytorch mean diff: {(dk_pt - dk_ref).abs().mean().item()}")
        print(f"dV Pytorch mean diff: {(dv_pt - dv_ref).abs().mean().item()}")

    # dS = torch.einsum('bthd,bshd->bhts', g.float(), v.float())
    # P = torch.softmax(qk, -1)
    # dP = P * (dS - do_o.unsqueeze(1))
    # dQ = torch.einsum('bhts,bshd->bthd', dP, k.float())
    # dV = torch.einsum('bhts,bthd->bshd', P, g.float())
    # dK = torch.einsum('bhts,bthd->bshd', dP, q.float())
    # breakpoint()

    # Check that FlashAttention's numerical error is at most twice the numerical error
    # of a Pytorch implementation.
    # breakpoint()
    if(dtype != torch.float8_e4m3fn):
        assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item() + 3e-5
    else:       
        # just test correctness of fp8 kernel w/o further quantization techniques
        assert (out - out_ref).abs().max().item() <= 40 * (out_pt - out_ref).abs().max().item()

    if d <= 128 and dtype != torch.float8_e4m3fn:
        assert (dq - dq_ref).abs().max().item() <= 2 * (dq_pt - dq_ref).abs().max().item() + 3e-5
        assert (dk - dk_ref).abs().max().item() <= 2 * (dk_pt - dk_ref).abs().max().item() + 3e-5
        assert (dv - dv_ref).abs().max().item() <= 2 * (dv_pt - dv_ref).abs().max().item() + 3e-5


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
# @pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
# @pytest.mark.parametrize("mha_type", ["mha"])
@pytest.mark.parametrize("causal", [False, True])
# @pytest.mark.parametrize("causal", [False])
@pytest.mark.parametrize("deterministic", [False, True])
# @pytest.mark.parametrize("deterministic", [False])
# @pytest.mark.parametrize("d", [32, 59, 64, 80, 96, 111, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize("d", [32, 64, 96, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize('d', [128])
# @pytest.mark.parametrize("d", [64, 128, 256])
@pytest.mark.parametrize("d", [64, 128])
# @pytest.mark.parametrize("d", [128])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (1, 1),
        (1, 3),
        (2, 1),
        (511, 1),
        (3, 513),
        (64, 128),
        (113, 203),
        (128, 128),
        (128, 217),
        (113, 211),
        (108, 256),
        (256, 512),
        (384, 256),
        (512, 256),
        (640, 128),
        (1024, 1024),
        (1023, 1024),
        (1024, 1023),
        (2048, 2048),
    ],
)
# @pytest.mark.parametrize('seqlen_q,seqlen_k', [(128, 128)])
def test_flash_attn_varlen_output(
    seqlen_q, seqlen_k, d, causal, deterministic, mha_type, dtype
):
    if (
        max(seqlen_q, seqlen_k) >= 2048
        and torch.cuda.get_device_properties("cuda").total_memory <= 16 * 2**30
    ):
        pytest.skip()  # Reference implementation OOM
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    # batch_size = 1
    # nheads = 1
    batch_size = 9
    nheads = 6
    nheads_kv = 6 if mha_type == "mha" else (2 if mha_type == "gqa" else 1)
 
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(
        batch_size, seqlen_k, nheads_kv, d, device=device, dtype=dtype, requires_grad=True
    )
    v = torch.randn(
        batch_size, seqlen_k, nheads_kv, d, device=device, dtype=dtype, requires_grad=True
    )

    query_padding_mask = generate_random_padding_mask(seqlen_q, batch_size, device, mode="random", zero_lengths=False)
    key_padding_mask = generate_random_padding_mask(seqlen_k, batch_size, device, mode="random", zero_lengths=True)
    # key_padding_mask = generate_random_padding_mask(seqlen_k, batch_size, device, mode='full')

    (
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        q,
        k,
        v,
        output_pad_fn,
        dq_pad_fn,
        dk_pad_fn,
    ) = generate_qkv(q, k, v, query_padding_mask, key_padding_mask, kvpacked=False)
    # print("cu_seqlens_q: ", cu_seqlens_q)
    # print("cu_seqlens_k: ", cu_seqlens_k)
    # print("q_unpad, shape: ", q_unpad.shape)
    # print("k_unpad, shape: ", k_unpad.shape)
    # print("v_unpad, shape: ", v_unpad.shape)
    out_unpad, sm_lse = flash_attn_varlen_func(
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        causal=causal,
        deterministic=deterministic,
    )
    out = output_pad_fn(out_unpad)
    dropout_mask = None

    out_ref, attn_ref = attention_ref(
        q,
        k,
        v,
        query_padding_mask,
        key_padding_mask,
        causal=causal,
    )
    out_pt, attn_pt = attention_ref(
        q,
        k,
        v,
        query_padding_mask,
        key_padding_mask,
        causal=causal,
        upcast=False,
        reorder_ops=True,
    )

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")

    g = torch.randn_like(out)
    if d <= 128:
        (
            dq_unpad,
            dk_unpad,
            dv_unpad,
        ) = torch.autograd.grad(out, (q_unpad, k_unpad, v_unpad), g)
        dk = dk_pad_fn(dk_unpad)
        dv = dk_pad_fn(dv_unpad)
        (
            dq_ref,
            dk_ref,
            dv_ref,
        ) = torch.autograd.grad(out_ref, (q, k, v), g)
        zero_masking = rearrange(torch.logical_not(torch.any(key_padding_mask, 1)), "b -> b 1 1 1")
        dk_ref.masked_fill_(zero_masking, 0.0)
        dv_ref.masked_fill_(zero_masking, 0.0)
        (
            dq_pt,
            dk_pt,
            dv_pt,
        ) = torch.autograd.grad(out_pt, (q, k, v), g)
        dk_pt.masked_fill_(zero_masking, 0.0)
        dv_pt.masked_fill_(zero_masking, 0.0)
        dq = dq_pad_fn(dq_unpad)
        print(f"dQ max diff: {(dq - dq_ref).abs().max().item()}")
        print(f"dK max diff: {(dk - dk_ref).abs().max().item()}")
        print(f"dV max diff: {(dv - dv_ref).abs().max().item()}")
        print(f"dQ mean diff: {(dq - dq_ref).abs().mean().item()}")
        print(f"dK mean diff: {(dk - dk_ref).abs().mean().item()}")
        print(f"dV mean diff: {(dv - dv_ref).abs().mean().item()}")
        print(f"dQ Pytorch max diff: {(dq_pt - dq_ref).abs().max().item()}")
        print(f"dK Pytorch max diff: {(dk_pt - dk_ref).abs().max().item()}")
        print(f"dV Pytorch max diff: {(dv_pt - dv_ref).abs().max().item()}")
        print(f"dQ Pytorch mean diff: {(dq_pt - dq_ref).abs().mean().item()}")
        print(f"dK Pytorch mean diff: {(dk_pt - dk_ref).abs().mean().item()}")
        print(f"dV Pytorch mean diff: {(dv_pt - dv_ref).abs().mean().item()}")

    # Check that FlashAttention's numerical error is at most twice the numerical error
    # of a Pytorch implementation.
    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item()

    if d <= 128:
        assert (dq - dq_ref).abs().max().item() < 1e-4 or (dq - dq_ref).abs().max().item() <= 3 * (dq_pt - dq_ref).abs().max().item()
        assert (dk - dk_ref).abs().max().item() < 1e-4 or (dk - dk_ref).abs().max().item() <= 3 * (dk_pt - dk_ref).abs().max().item()
        assert (dk - dk_ref).abs().max().item() < 1e-4 or (dv - dv_ref).abs().max().item() <= 3 * (dv_pt - dv_ref).abs().max().item()
