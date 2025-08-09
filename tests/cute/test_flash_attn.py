# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.

import math
import itertools

import pytest
import torch

from einops import rearrange, repeat
try:
    from flash_attn.layers.rotary import apply_rotary_emb
except ImportError:
    apply_rotary_emb = None

# from padding import pad_input, unpad_input
from flash_attn.utils.testing import attention_ref, generate_qkv, generate_random_padding_mask
from flash_attn.cute.interface import flash_attn_func, flash_attn_varlen_func


# @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
# @pytest.mark.parametrize("mha_type", ["mha"])
@pytest.mark.parametrize("has_learnable_sink", [False, True])
# @pytest.mark.parametrize("has_learnable_sink", [False])
# @pytest.mark.parametrize("has_qv", [False, True])
@pytest.mark.parametrize("has_qv", [False])
# @pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("deterministic", [False])
# @pytest.mark.parametrize("softcap", [0.0, 15.0])
@pytest.mark.parametrize("softcap", [0.0])
@pytest.mark.parametrize("local", [False, True])
# @pytest.mark.parametrize("local", [False])
@pytest.mark.parametrize("causal", [False, True])
# @pytest.mark.parametrize("causal", [False])
# @pytest.mark.parametrize("d", [32, 64, 96, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize('d', [32, 40, 64, 80, 96, 128, 160, 192, 256])
# @pytest.mark.parametrize('d', [32, 64, 96, 128, 160, 192])
# @pytest.mark.parametrize('d', [56, 80])
# @pytest.mark.parametrize("d", [64, 128, 256])
# @pytest.mark.parametrize('d', [32, 40, 64, 80, 96, 128])
# @pytest.mark.parametrize("d", [64, 96, 128, 192])
# @pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("d", [128, 192])
# @pytest.mark.parametrize("d", [128])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (1, 1),
        (64, 128),
        (128, 192),
        (256, 256),
        (239, 1),
        (799, 3),
        (113, 203),
        (113, 128),
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
        (4224, 4224),
    ],
)
# @pytest.mark.parametrize('seqlen_q,seqlen_k', [(128, 128)])
def test_flash_attn_output(
    seqlen_q, seqlen_k, d, causal, local, softcap, deterministic, has_qv, has_learnable_sink, mha_type, dtype
):
    if (causal or local) and seqlen_k < seqlen_q:
        pytest.skip("Causal attention requires seqlen_k >= seqlen_q")
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    batch_size = 9 if seqlen_k <= 2048 else 2
    # batch_size = 1
    nheads = 6
    # nheads = 1
    nheads_kv = nheads if mha_type == "mha" else (2 if mha_type == "gqa" else 1)
    dtype_ref = torch.bfloat16 if dtype == torch.float8_e4m3fn else dtype
    # dv_vals = [128, d] if d > 128 and d <= 192 else ([256, 512, d] if d <= 64 else [d])
    dv_vals = [128] if d == 192 else ([d] if d != 128 else [64, d])
    if dtype == torch.float8_e4m3fn:
        dv_vals = [d]
    # attention_chunk_vals = [torch.randint(1, seqlen_k * 2, (1,)).item(), 0] if not DISABLE_LOCAL else [0]
    attention_chunk_vals = [0]
    for dv, attention_chunk in itertools.product(dv_vals, attention_chunk_vals):
        q_ref = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype_ref)
        if softcap > 0.0:
            # Ensure the values of qk are at least within softcap range.
            q_ref = (q_ref * softcap / 4)
        q_ref = q_ref.to(dtype).to(dtype_ref).requires_grad_()
        k_ref = torch.randn(batch_size, seqlen_k, nheads_kv, d, device=device, dtype=dtype_ref).to(dtype).to(dtype_ref).requires_grad_()
        v_ref = torch.randn(batch_size, seqlen_k, nheads_kv, dv, device=device, dtype=dtype_ref).to(dtype).to(dtype_ref).requires_grad_()
        if has_qv:
            qv_ref = torch.randn(batch_size, seqlen_q, nheads, dv, device=device, dtype=dtype_ref).to(dtype).to(dtype_ref)
        else:
            qv_ref = None
        # Put window_size after QKV randn so that window_size changes from test to test
        window_size = (None, None) if not local else torch.randint(0, seqlen_k, (2,)).tolist()
        # window_size = (-1, -1) if not local else (16, 0)
        if has_learnable_sink:
            learnable_sink = torch.randn(nheads, dtype=torch.bfloat16, device=device)
        else:
            learnable_sink = None
        if dtype == torch.float8_e4m3fn:
            q_descale, k_descale, v_descale = [torch.rand(batch_size, nheads_kv, device=device, dtype=torch.float32) * 2 for _ in range(3)]
        else:
            q_descale, k_descale, v_descale = None, None, None
        q, k, v = [x.detach().to(dtype).requires_grad_() for x in (q_ref, k_ref, v_ref)]
        qv = qv_ref.detach().to(dtype).requires_grad_() if has_qv else None
        out_ref, attn_ref = attention_ref(
            q_ref,
            k_ref,
            v_ref,
            None,
            None,
            causal=causal,
            qv=qv_ref,
            q_descale=q_descale, k_descale=k_descale, v_descale=v_descale,
            window_size=window_size,
            attention_chunk=attention_chunk,
            learnable_sink=learnable_sink,
            softcap=softcap
        )
        out_pt, attn_pt = attention_ref(
            q_ref,
            k_ref,
            v_ref,
            None,
            None,
            causal=causal,
            qv=qv_ref,
            q_descale=q_descale, k_descale=k_descale, v_descale=v_descale,
            window_size=window_size,
            attention_chunk=attention_chunk,
            learnable_sink=learnable_sink,
            softcap=softcap,
            upcast=False,
            reorder_ops=True,
            intermediate_dtype=dtype if dtype == torch.float8_e4m3fn else None,
        )

        # k_extended = repeat(k_ref, "b s h d -> b s (h k) d", k=nheads // nheads_kv)
        # qk = torch.einsum('bshd,bthd->bhst', q_ref, k_extended).float()
        # # if qv is not None:
        # #     qk += torch.einsum('bshd,bthd->bhst', qv_ref, v_ref).float()
        # m = qk.amax(-1, keepdim=True)
        # s_tmp = torch.exp((qk - m) / math.sqrt(d))
        # exp_sum = s_tmp.sum(-1)
        # # qk = torch.einsum('bthd,bshd->bhts', q_ref.float() / math.sqrt(d), k_ref.float())
        # # lse_ref = torch.logsumexp(qk, dim=-1)

        # Numerical error if we just do any arithmetic on out_ref
        fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
        rtol = 2 if softcap == 0.0 else 3

        print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
        print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")
        # pack_gqa_vals = [False, True] if not DISABLE_PACKGQA else [False]
        # num_splits_vals = [1, 3] if not DISABLE_SPLIT else [1]
        pack_gqa_vals = [False]
        num_splits_vals = [1]
        for pack_gqa, num_splits in itertools.product(pack_gqa_vals, num_splits_vals):
            out, lse = flash_attn_func(
                q,
                k,
                v,
                causal=causal,
                # qv=qv,
                # q_descale=q_descale, k_descale=k_descale, v_descale=v_descale,
                window_size=window_size,
                # attention_chunk=attention_chunk,
                softcap=softcap,
                learnable_sink=learnable_sink,
                # pack_gqa=pack_gqa,
                # num_splits=num_splits
            )
            print(f"Output max diff: {(out - out_ref).abs().max().item()}")
            print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
            # if not causal:
            #     print(f"LSE max diff: {(lse - lse_ref).abs().max().item()}")
            # breakpoint()

            # Check that FlashAttention's numerical error is at most twice the numerical error
            # of a Pytorch implementation.
            assert (out - out_ref).abs().max().item() <= rtol * (out_pt - out_ref).abs().max().item() + fwd_atol

        if (
            dtype != torch.float8_e4m3fn
            and not has_qv
            and not dv > 256
            and not attention_chunk != 0
            and softcap == 0.0
            and not local
            and dv == d
            and learnable_sink is None
            and False
        ):
            g = torch.randn_like(out)
            # do_o = ((g.float() * out.float()).sum(-1)).transpose(1, 2)
            dq, dk, dv = torch.autograd.grad(out, (q, k, v), g)
            # print(f"dO_O max diff: {(softmax_d - do_o).abs().max().item()}")
            # assert (softmax_d - do_o).abs().max().item() <= 1e-5
            # assert dq_accum.abs().max().item() == 0.0

            # dS = torch.einsum('bthd,bshd->bhts', g.float(), v.float())
            # P = torch.softmax(qk, -1)
            # dP = P * (dS - do_o.transpose(1, 2).unsqueeze(1))
            # dQ = torch.einsum('bhts,bshd->bthd', dP, k.float())
            # dV = torch.einsum('bhts,bthd->bshd', P, g.float())
            # dK = torch.einsum('bhts,bthd->bshd', dP, q.float())

            # dq, dk, dv = torch.autograd.grad(out, (q, k, v), g)
            dq_ref, dk_ref, dv_ref = torch.autograd.grad(out_ref, (q_ref, k_ref, v_ref), g)
            dq_pt, dk_pt, dv_pt = torch.autograd.grad(out_pt, (q_ref, k_ref, v_ref), g)
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
            # breakpoint()
            dq_atol = 2 * (dq_ref + 0.3 - 0.3 - dq_ref).abs().max().item() + (0 if softcap == 0 else 3e-4)
            assert (dq - dq_ref).abs().max().item() <= rtol * (dq_pt - dq_ref).abs().max().item() + dq_atol
            dk_atol = 2 * (dk_ref + 0.3 - 0.3 - dk_ref).abs().max().item() + (0 if softcap == 0 else 3e-4)
            assert (dk - dk_ref).abs().max().item() <= rtol * (dk_pt - dk_ref).abs().max().item() + dk_atol
            dv_atol = 2 * (dv_ref + 0.3 - 0.3 - dv_ref).abs().max().item() + (0 if softcap == 0 else 3e-4)
            assert (dv - dv_ref).abs().max().item() <= rtol * (dv_pt - dv_ref).abs().max().item() + dv_atol


# @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
# @pytest.mark.parametrize("mha_type", ["mha"])
@pytest.mark.parametrize("has_learnable_sink", [False, True])
# @pytest.mark.parametrize("has_learnable_sink", [False])
# @pytest.mark.parametrize("has_qv", [False, True])
@pytest.mark.parametrize("has_qv", [False])
# @pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("deterministic", [False])
# @pytest.mark.parametrize("softcap", [0.0, 15.0])
@pytest.mark.parametrize("softcap", [0.0])
@pytest.mark.parametrize("local", [False, True])
# @pytest.mark.parametrize("local", [False])
@pytest.mark.parametrize("causal", [False, True])
# @pytest.mark.parametrize("causal", [False])
# @pytest.mark.parametrize("add_unused_qkv", [False, True])
@pytest.mark.parametrize("add_unused_qkv", [False])
# @pytest.mark.parametrize("d", [32, 64, 96, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize('d', [32, 40, 64, 80, 96, 128, 160, 192, 256])
# @pytest.mark.parametrize('d', [32, 64, 96, 128, 160, 192])
# @pytest.mark.parametrize('d', [56, 80])
# @pytest.mark.parametrize('d', [32, 40, 64, 80, 96, 128])
# @pytest.mark.parametrize("d", [64, 96, 128])
@pytest.mark.parametrize("d", [128, 192])
# @pytest.mark.parametrize("d", [128])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        # (1, 1),
        # (1, 3),
        # (2, 1),
        (511, 1),
        (3, 513),
        (64, 128),
        (128, 128),
        (256, 256),
        (113, 203),
        (128, 217),
        (113, 211),
        (108, 256),
        (256, 512),
        (307, 256),
        (640, 128),
        (512, 256),
        (1024, 1024),
        (1023, 1024),
        (1024, 1023),
        (2048, 2048),
    ],
)
def test_flash_attn_varlen_output(
    seqlen_q, seqlen_k, d, add_unused_qkv, causal, local, softcap, deterministic, has_qv, has_learnable_sink, mha_type, dtype
):
    if (causal or local):  # Right now we only support causal attention with seqlen_k == seqlen_q
        seqlen_k = seqlen_q
    device = "cuda"
    # set seed
    torch.random.manual_seed(seqlen_q + seqlen_k + d + int(causal) * 2 + int(local))
    batch_size = 49 if seqlen_q <= 2048 else 2
    nheads = 6
    # batch_size = 1
    # nheads = 1
    nheads_kv = nheads if mha_type == "mha" else (2 if mha_type == "gqa" else 1)
    dtype_ref = torch.bfloat16 if dtype == torch.float8_e4m3fn else dtype
    # dv_vals = [128, d] if d > 128 and d <= 192 else ([256, 512, d] if d <= 64 else [d])
    dv_vals = [128] if d == 192 else ([d] if d != 128 else [64, d])
    if dtype == torch.float8_e4m3fn:
        dv_vals = [d]
    # attention_chunk_vals = [torch.randint(1, seqlen_k * 2, (1,)).item(), 0] if seqlen_q <= seqlen_k and not DISABLE_LOCAL else [0]
    attention_chunk_vals = [0]
    for dv, attention_chunk in itertools.product(dv_vals, attention_chunk_vals):
        q_ref = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype_ref)
        if softcap > 0.0:
            # Ensure the values of qk are at least within softcap range.
            q_ref = (q_ref * softcap / 4).detach().requires_grad_()
        q_ref = q_ref.to(dtype).to(dtype_ref).requires_grad_()
        k_ref = torch.randn(batch_size, seqlen_k, nheads_kv, d, device=device, dtype=dtype_ref).to(dtype).to(dtype_ref).requires_grad_()
        v_ref = torch.randn(batch_size, seqlen_k, nheads_kv, dv, device=device, dtype=dtype_ref).to(dtype).to(dtype_ref).requires_grad_()
        if has_qv:
            qv_ref = torch.randn(batch_size, seqlen_q, nheads, dv, device=device, dtype=dtype_ref).to(dtype).to(dtype_ref)
        else:
            qv_ref = None
        # Put window_size after QKV randn so that window_size changes from test to test
        window_size = (None, None) if not local else torch.randint(0, seqlen_k, (2,)).tolist()
        if has_learnable_sink:
            learnable_sink = torch.randn(nheads, dtype=torch.bfloat16, device=device)
        else:
            learnable_sink = None
        if dtype == torch.float8_e4m3fn:
            q_descale, k_descale, v_descale = [torch.rand(batch_size, nheads_kv, device=device, dtype=torch.float32) * 2 for _ in range(3)]
        else:
            q_descale, k_descale, v_descale = None, None, None
        q, k, v = [x.detach().requires_grad_() for x in (q_ref, k_ref, v_ref)]
        qv = qv_ref.detach() if has_qv else None
        query_padding_mask = generate_random_padding_mask(
            seqlen_q, batch_size, device, mode="random", zero_lengths=False
        )
        # TODO: test zero_lengths
        key_padding_mask = generate_random_padding_mask(
            # seqlen_k, batch_size, device, mode="random", zero_lengths=True
            seqlen_k, batch_size, device, mode="random", zero_lengths=False
        )

        def _gen_unused_masks(padding_mask, add_unused, max_seq_len, bs, device):
            if add_unused:
                another_mask = generate_random_padding_mask(max_seq_len, bs, device)
                attn_mask = torch.logical_and(padding_mask, another_mask)
                unused_mask = torch.logical_xor(
                    torch.logical_or(padding_mask, another_mask), attn_mask
                )
            else:
                attn_mask = padding_mask
                unused_mask = None
            return attn_mask, unused_mask

        query_padding_mask, query_unused_mask = _gen_unused_masks(
            query_padding_mask, add_unused_qkv, seqlen_q, batch_size, q.device
        )
        # query_padding_mask[:] = True
        # query_unused_mask = None
        key_padding_mask, key_unused_mask = _gen_unused_masks(
            key_padding_mask, add_unused_qkv, seqlen_k, batch_size, k.device
        )

        if causal or local:
            key_padding_mask = query_padding_mask

        (
            q_unpad,
            k_unpad,
            v_unpad,
            qv_unpad,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            max_seqlen_q,
            max_seqlen_k,
            q,
            k,
            v,
            qv,
            output_pad_fn,
            dq_pad_fn,
            dk_pad_fn,
        ) = generate_qkv(q, k, v, query_padding_mask, key_padding_mask, qv=qv, kvpacked=False,
                        query_unused_mask=query_unused_mask, key_unused_mask=key_unused_mask)
        q_unpad, k_unpad, v_unpad = [x.detach().to(dtype).requires_grad_() for x in (q_unpad, k_unpad, v_unpad)]
        out_ref, attn_ref = attention_ref(
            q_ref,
            k_ref,
            v_ref,
            query_padding_mask,
            key_padding_mask,
            causal=causal,
            qv=qv_ref,
            q_descale=q_descale, k_descale=k_descale, v_descale=v_descale,
            window_size=window_size,
            attention_chunk=attention_chunk,
            learnable_sink=learnable_sink,
            softcap=softcap
        )
        out_pt, attn_pt = attention_ref(
            q_ref,
            k_ref,
            v_ref,
            query_padding_mask,
            key_padding_mask,
            causal=causal,
            qv=qv_ref,
            q_descale=q_descale, k_descale=k_descale, v_descale=v_descale,
            window_size=window_size,
            attention_chunk=attention_chunk,
            learnable_sink=learnable_sink,
            softcap=softcap,
            upcast=False,
            reorder_ops=True,
            intermediate_dtype=dtype if dtype == torch.float8_e4m3fn else None,
        )

        print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
        print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")

        if query_unused_mask is not None:
            q_zero_masking = rearrange(query_unused_mask, "b s -> b s 1 1")

        # Numerical error if we just do any arithmetic on out_ref
        fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
        rtol = 2 if softcap == 0.0 else 3

        # pack_gqa_vals = [False, True] if not DISABLE_PACKGQA else [False]
        # num_splits_vals = [1, 3] if not DISABLE_SPLIT else [1]
        pack_gqa_vals = [False]
        num_splits_vals = [1]
        for pack_gqa, num_splits in itertools.product(pack_gqa_vals, num_splits_vals):
            out_unpad, lse = flash_attn_varlen_func(
                q_unpad,
                k_unpad,
                v_unpad,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                # max_seqlen_k,
                # seqused_q=seqused_q,
                # seqused_k=seqused_k,
                causal=causal,
                # qv=qv_unpad,
                # q_descale=q_descale,
                # k_descale=k_descale, v_descale=v_descale,
                window_size=window_size,
                # attention_chunk=attention_chunk,
                learnable_sink=learnable_sink,
                softcap=softcap,
            )
            out = output_pad_fn(out_unpad)
            if query_unused_mask is not None:
                out.masked_fill_(q_zero_masking, 0.0)
            print(f"Output max diff: {(out - out_ref).abs().max().item()}")
            print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
            # if not causal:
            #     print(f"LSE max diff: {(lse - lse_ref).abs().max().item()}")
            # breakpoint()

            # Check that FlashAttention's numerical error is at most 3x the numerical error
            # of a Pytorch implementation.
            assert (out - out_ref).abs().max().item() <= rtol * (out_pt - out_ref).abs().max().item() + fwd_atol


        if (
            dtype != torch.float8_e4m3fn
            and not has_qv
            and not dv > 256
            and not attention_chunk != 0
            and dv == d
            and not has_learnable_sink
            and False
        ):
            g_unpad = torch.randn_like(out_unpad)
            do_o = ((g_unpad.float() * out_unpad.float()).sum(-1)).transpose(-1, -2)
            # import flash_attn_3_cuda
            # dq_unpad, dk_unpad, dv_unpad, softmax_d, dq_accum, lse_log2 = flash_attn_3_cuda.bwd_varlen(
            #     g_unpad,
            #     q_unpad,
            #     k_unpad,
            #     v_unpad,
            #     out_unpad,
            #     lse,
            #     None,
            #     None,
            #     None,
            #     cu_seqlens_q,
            #     cu_seqlens_k,
            #     None, None,
            #     max_seqlen_q,
            #     max_seqlen_k,
            #     d ** (-0.5),
            #     causal,
            #     window_size[0], window_size[1],
            #     softcap,
            #     deterministic,
            #     0,  # sm_margin
            # )
            dq_unpad, dk_unpad, dv_unpad = torch.autograd.grad(out_unpad, (q_unpad, k_unpad, v_unpad), g_unpad)
            dq = dq_pad_fn(dq_unpad)
            dk = dk_pad_fn(dk_unpad)
            dv = dk_pad_fn(dv_unpad)
            if key_unused_mask is not None:
                k_zero_masking = rearrange(key_unused_mask, "b s -> b s 1 1")
                dk.masked_fill_(k_zero_masking, 0.0)
                dv.masked_fill_(k_zero_masking, 0.0)
            if query_unused_mask is not None:
                dq.masked_fill_(q_zero_masking, 0.0)
            # print(f"dO_O max diff: {(softmax_d - do_o).abs().max().item()}")
            # assert (softmax_d - do_o).abs().max().item() <= 1e-5
            # assert dq_accum.abs().max().item() == 0.0
            g = output_pad_fn(g_unpad)

            # qk = torch.einsum('bthd,bshd->bhts', q / (d ** 0.5), k).float()
            # qk = torch.masked_fill(qk, rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"))
            # dS = torch.einsum('bthd,bshd->bhts', g.float(), v.float())
            # P = torch.softmax(qk, -1)
            # dP = P * (dS - (g.float() * out.float()).sum(-1).transpose(1, 2).unsqueeze(-1))
            # dQ = torch.einsum('bhts,bshd->bthd', dP, k.float())
            # dV = torch.einsum('bhts,bthd->bshd', P, g.float())
            # dK = torch.einsum('bhts,bthd->bshd', dP, q.float())


            # dq, dk, dv = torch.autograd.grad(out, (q, k, v), g)
            dq_ref, dk_ref, dv_ref = torch.autograd.grad(out_ref, (q_ref, k_ref, v_ref), g)
            dq_pt, dk_pt, dv_pt = torch.autograd.grad(out_pt, (q_ref, k_ref, v_ref), g)
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
            # breakpoint()
            dq_atol = 2 * (dq_ref + 0.3 - 0.3 - dq_ref).abs().max().item() + (0 if softcap == 0 else 3e-4)
            assert (dq - dq_ref).abs().max().item() <= rtol * (dq_pt - dq_ref).abs().max().item() + dq_atol
            dk_atol = 2 * (dk_ref + 0.3 - 0.3 - dk_ref).abs().max().item() + (0 if softcap == 0 else 3e-4)
            assert (dk - dk_ref).abs().max().item() <= rtol * (dk_pt - dk_ref).abs().max().item() + dk_atol
            dv_atol = 2 * (dv_ref + 0.3 - 0.3 - dv_ref).abs().max().item() + (0 if softcap == 0 else 3e-4)
            assert (dv - dv_ref).abs().max().item() <= rtol * (dv_pt - dv_ref).abs().max().item() + dv_atol
