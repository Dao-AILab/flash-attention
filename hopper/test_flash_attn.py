import os
import math
import itertools

import pytest
import torch
import torch.nn.functional as F
from torch._C import parse_schema

from einops import rearrange, repeat
try:
    from flash_attn.layers.rotary import apply_rotary_emb
except ImportError:
    apply_rotary_emb = None

from padding import pad_input, unpad_input
from test_util import (
    attention_ref,
    generate_qkv,
    generate_random_padding_mask,
)

from flash_attn_interface import flash_attn_func, flash_attn_varlen_func, flash_attn_combine
from flash_attn_interface import flash_attn_with_kvcache, get_scheduler_metadata


DISABLE_BACKWARD = os.getenv("FLASH_ATTENTION_DISABLE_BACKWARD", "FALSE") == "TRUE"
DISABLE_SPLIT = os.getenv("FLASH_ATTENTION_DISABLE_SPLIT", "FALSE") == "TRUE"
DISABLE_PAGEDKV = os.getenv("FLASH_ATTENTION_DISABLE_PAGEDKV", "FALSE") == "TRUE"
DISABLE_APPENDKV = os.getenv("FLASH_ATTENTION_DISABLE_APPENDKV", "FALSE") == "TRUE"
DISABLE_LOCAL = os.getenv("FLASH_ATTENTION_DISABLE_LOCAL", "FALSE") == "TRUE"
DISABLE_SOFTCAP = os.getenv("FLASH_ATTENTION_DISABLE_SOFTCAP", "FALSE") == "TRUE"
DISABLE_PACKGQA = os.getenv("FLASH_ATTENTION_DISABLE_PACKGQA", "FALSE") == "TRUE"
DISABLE_FP16 = os.getenv("FLASH_ATTENTION_DISABLE_FP16", "FALSE") == "TRUE"
DISABLE_FP8 = os.getenv("FLASH_ATTENTION_DISABLE_FP8", "FALSE") == "TRUE" or torch.cuda.get_device_capability("cuda")[0] < 9
DISABLE_HDIM64 = os.getenv("FLASH_ATTENTION_DISABLE_HDIM64", "FALSE") == "TRUE"
DISABLE_HDIM96 = os.getenv("FLASH_ATTENTION_DISABLE_HDIM96", "FALSE") == "TRUE"
DISABLE_HDIM128 = os.getenv("FLASH_ATTENTION_DISABLE_HDIM128", "FALSE") == "TRUE"
DISABLE_HDIM192 = os.getenv("FLASH_ATTENTION_DISABLE_HDIM192", "FALSE") == "TRUE"
DISABLE_HDIM256 = os.getenv("FLASH_ATTENTION_DISABLE_HDIM256", "FALSE") == "TRUE"

COMPILED_HDIMS = (
    []
    + ([64] if not DISABLE_HDIM64 else [])
    + ([96] if not DISABLE_HDIM96 else [])
    + ([128] if not DISABLE_HDIM128 else [])
    + ([192] if not DISABLE_HDIM192 else [])
    + ([256] if not DISABLE_HDIM256 else [])
)


# @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("dtype", [torch.bfloat16] + ([torch.float16] if not DISABLE_FP16 else []) + ([torch.float8_e4m3fn] if not DISABLE_FP8 else []))
# @pytest.mark.parametrize("dtype", [torch.bfloat16])
# @pytest.mark.parametrize("dtype", [torch.float8_e4m3fn])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
# @pytest.mark.parametrize("mha_type", ["mha"])
@pytest.mark.parametrize("has_qv", [False, True])
# @pytest.mark.parametrize("has_qv", [True])
# @pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("deterministic", [False])
@pytest.mark.parametrize("softcap", [0.0] + ([15.0] if not DISABLE_SOFTCAP else []))
# @pytest.mark.parametrize("softcap", [0.0])
@pytest.mark.parametrize("local", [False] + ([True] if not DISABLE_LOCAL else []))
# @pytest.mark.parametrize("local", [False])
@pytest.mark.parametrize("causal", [False, True])
# @pytest.mark.parametrize("causal", [True])
# @pytest.mark.parametrize("V_colmajor", [False, True])
@pytest.mark.parametrize("V_colmajor", [False])
# @pytest.mark.parametrize("d", [32, 64, 96, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize('d', [32, 40, 64, 80, 96, 128, 160, 192, 256])
# @pytest.mark.parametrize('d', [32, 64, 96, 128, 160, 192])
# @pytest.mark.parametrize('d', [56, 80])
# @pytest.mark.parametrize("d", [64, 128, 256])
# @pytest.mark.parametrize('d', [32, 40, 64, 80, 96, 128])
# @pytest.mark.parametrize("d", [64, 96, 128, 192])
@pytest.mark.parametrize("d", COMPILED_HDIMS)
# @pytest.mark.parametrize("d", [64])
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
        seqlen_q, seqlen_k, d, causal, local, softcap, V_colmajor, deterministic, has_qv, mha_type, dtype
):
    if V_colmajor and (seqlen_k % 16 != 0 or dtype != torch.float8_e4m3fn):
        pytest.skip("V_colmajor requires seqlen_k to be a multiple of 16 and dtype to be float8_e4m3fn")
    if has_qv and (d != 64 or dtype == torch.float8_e4m3fn):
        pytest.skip("Has Qv requires hdim 64 and dtype to be float16 or bfloat16 (not float8_e4m3fn)")
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    # batch_size = 40
    # nheads = 16
    batch_size = 9 if seqlen_k <= 2048 else 2
    # batch_size = 1
    nheads = 6
    # nheads = 1
    nheads_kv = nheads if mha_type == "mha" else (2 if mha_type == "gqa" else 1)
    dtype_ref = torch.bfloat16 if dtype == torch.float8_e4m3fn else dtype
    dv_vals = [128, d] if d > 128 and d <= 192 else ([256, 512, d] if d <= 64 else [d])
    if dtype == torch.float8_e4m3fn:
        dv_vals = [d]
    if has_qv:
        dv_vals = [256, 512]
    attention_chunk_vals = [torch.randint(1, seqlen_k * 2, (1,)).item(), 0] if not DISABLE_LOCAL else [0]
    for dv, attention_chunk in itertools.product(dv_vals, attention_chunk_vals):
        print(f"{dv = }, {attention_chunk = }")
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
        window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,)).tolist()
        # window_size = (-1, -1) if not local else (16, 0)
        if dtype == torch.float8_e4m3fn:
            q_descale, k_descale, v_descale = [torch.rand(batch_size, nheads_kv, device=device, dtype=torch.float32) * 2 for _ in range(3)]
        else:
            q_descale, k_descale, v_descale = None, None, None
        q, k, v = [x.detach().to(dtype).requires_grad_() for x in (q_ref, k_ref, v_ref)]
        qv = qv_ref.detach().to(dtype).requires_grad_() if has_qv else None
        if V_colmajor:
            v = rearrange(rearrange(v.detach(), "b s h d -> b h d s").contiguous(), "b h d s -> b s h d").requires_grad_()
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
            softcap=softcap,
            upcast=False,
            reorder_ops=True,
            intermediate_dtype=dtype if dtype == torch.float8_e4m3fn else None,
        )

        # qk = torch.einsum('bshd,bthd->bhst', q_ref, k_ref).float()
        # if qv is not None:
        #     qk += torch.einsum('bshd,bthd->bhst', qv_ref, v_ref).float()
        # m = qk.amax(-1, keepdim=True)
        # s_tmp = torch.exp((qk - m) / math.sqrt(d))
        # exp_sum = s_tmp.sum(-1)
        # qk = torch.einsum('bthd,bshd->bhts', q_ref.float() / math.sqrt(d), k_ref.float())
        # lse_ref = torch.logsumexp(qk, dim=-1)

        # Numerical error if we just do any arithmetic on out_ref
        fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
        rtol = 2 if softcap == 0.0 else 3

        print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
        print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")
        pack_gqa_vals = [False, True] if not DISABLE_PACKGQA else [False]
        num_splits_vals = [1, 3] if not DISABLE_SPLIT else [1]
        for pack_gqa, num_splits in itertools.product(pack_gqa_vals, num_splits_vals):
            print(f"{pack_gqa = }, {num_splits = }")
            out = flash_attn_func(
                q,
                k,
                v,
                causal=causal,
                qv=qv,
                q_descale=q_descale, k_descale=k_descale, v_descale=v_descale,
                window_size=window_size,
                attention_chunk=attention_chunk,
                softcap=softcap,
                pack_gqa=pack_gqa,
                num_splits=num_splits
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
            not DISABLE_BACKWARD 
            and dtype != torch.float8_e4m3fn 
            and not V_colmajor 
            and not has_qv
            and not dv > 256
            and not attention_chunk != 0
        ):
            g = torch.randn_like(out)
            do_o = ((g.float() * out.float()).sum(-1)).transpose(1, 2)
            # import flash_attn_3_cuda
            # dq, dk, dv, softmax_d, dq_accum, dk_accum, dv_accum = flash_attn_3_cuda.bwd(
            #     g,
            #     q,
            #     k,
            #     v,
            #     out,
            #     lse,
            #     None,
            #     None,
            #     None,
            #     d ** (-0.5),
            #     causal,
            #     window_size[0], window_size[1],
            #     softcap,
            #     deterministic,
            #     0,  # sm_margin
            # )
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
@pytest.mark.parametrize("dtype", [torch.bfloat16] + ([torch.float16] if not DISABLE_FP16 else []) + ([torch.float8_e4m3fn] if not DISABLE_FP8 else []))
# @pytest.mark.parametrize("dtype", [torch.bfloat16])
# @pytest.mark.parametrize("dtype", [torch.float8_e4m3fn])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
# @pytest.mark.parametrize("mha_type", ["mha"])
@pytest.mark.parametrize("has_qv", [False, True])
# @pytest.mark.parametrize("has_qv", [False])
# @pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("deterministic", [False])
@pytest.mark.parametrize("softcap", [0.0] + ([15.0] if not DISABLE_SOFTCAP else []))
# @pytest.mark.parametrize("softcap", [0.0])
@pytest.mark.parametrize("local", [False] + ([True] if not DISABLE_LOCAL else []))
# @pytest.mark.parametrize("local", [False])
@pytest.mark.parametrize("causal", [False, True])
# @pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("add_unused_qkv", [False, True])
# @pytest.mark.parametrize("add_unused_qkv", [True])
# @pytest.mark.parametrize("d", [32, 64, 96, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize('d', [32, 40, 64, 80, 96, 128, 160, 192, 256])
# @pytest.mark.parametrize('d', [32, 64, 96, 128, 160, 192])
# @pytest.mark.parametrize('d', [56, 80])
# @pytest.mark.parametrize('d', [32, 40, 64, 80, 96, 128])
# @pytest.mark.parametrize("d", [64, 96, 128])
@pytest.mark.parametrize("d", COMPILED_HDIMS)
# @pytest.mark.parametrize("d", [64])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (1, 1),
        (1, 3),
        (2, 1),
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
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
    ],
)
def test_flash_attn_varlen_output(
    seqlen_q, seqlen_k, d, add_unused_qkv, causal, local, softcap, deterministic, has_qv, mha_type, dtype,
):
    if has_qv and (d != 64 or dtype == torch.float8_e4m3fn):
        pytest.skip("Has Qv requires hdim 64 and dtype to be float16 or bfloat16 (not float8_e4m3fn)")
    device = "cuda"
    # set seed
    torch.random.manual_seed(seqlen_q + seqlen_k + d + int(causal) * 2 + int(local))
    # batch_size = 40
    # nheads = 16
    batch_size = 9 if seqlen_q <= 2048 else 2
    # batch_size = 32
    nheads = 6
    nheads_kv = nheads if mha_type == "mha" else (2 if mha_type == "gqa" else 1)
    # batch_size = 2
    # nheads = 1
    # nheads_kv = nheads
    
    dtype_ref = torch.bfloat16 if dtype == torch.float8_e4m3fn else dtype
    dv_vals = [128, d] if d > 128 and d <= 192 else ([256, 512, d] if d <= 64 else [d])
    if dtype == torch.float8_e4m3fn:
        dv_vals = [d]
    if has_qv:
        dv_vals = [256, 512]
    attention_chunk_vals = [torch.randint(1, seqlen_k * 2, (1,)).item(), 0] if seqlen_q <= seqlen_k and not DISABLE_LOCAL else [0]
    for dv, attention_chunk in itertools.product(dv_vals, attention_chunk_vals):
        print(f"{dv = }, {attention_chunk = }")
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
        window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))
        if dtype == torch.float8_e4m3fn:
            q_descale, k_descale, v_descale = [torch.rand(batch_size, nheads_kv, device=device, dtype=torch.float32) * 2 for _ in range(3)]
        else:
            q_descale, k_descale, v_descale = None, None, None
        q, k, v = [x.detach().requires_grad_() for x in (q_ref, k_ref, v_ref)]
        qv = qv_ref.detach() if has_qv else None
        query_padding_mask = generate_random_padding_mask(
            seqlen_q, batch_size, device, mode="random", zero_lengths=False
        )
        key_padding_mask = generate_random_padding_mask(
            seqlen_k, batch_size, device, mode="random", zero_lengths=True
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
        key_padding_mask, key_unused_mask = _gen_unused_masks(
            key_padding_mask, add_unused_qkv, seqlen_k, batch_size, k.device
        )

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

        pack_gqa_vals = [False, True] if not DISABLE_PACKGQA else [False]
        # pack_gqa_vals = [False]
        num_splits_vals = [1, 3, 0] if not DISABLE_SPLIT else [1]
        # num_splits_vals = [1]
        # print("cu_seqlens_q: ", cu_seqlens_q)
        # print("cu_seqlens_k: ", cu_seqlens_k)
        # print("seqused_q: ", seqused_q)
        # print("seqused_k: ", seqused_k)
        for pack_gqa, num_splits in itertools.product(pack_gqa_vals, num_splits_vals):
            print(f"{pack_gqa = }, {num_splits = }")
            out_unpad = flash_attn_varlen_func(
                q_unpad,
                k_unpad,
                v_unpad,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                seqused_q=seqused_q,
                seqused_k=seqused_k,
                causal=causal,
                qv=qv_unpad,
                q_descale=q_descale,
                k_descale=k_descale, v_descale=v_descale,
                window_size=window_size,
                attention_chunk=attention_chunk,
                softcap=softcap,
                pack_gqa=pack_gqa,
                num_splits=num_splits,
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
            not DISABLE_BACKWARD 
            and dtype != torch.float8_e4m3fn 
            and not has_qv
            and not dv > 256
            and not attention_chunk != 0
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


# @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("dtype", [torch.bfloat16] + ([torch.float8_e4m3fn] if not DISABLE_FP8 else []))
# @pytest.mark.parametrize("dtype", [torch.bfloat16])
# @pytest.mark.parametrize("dtype", [torch.float8_e4m3fn])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
# @pytest.mark.parametrize("mha_type", ["mha"])
@pytest.mark.parametrize("new_kv", [False] + ([True] if not DISABLE_APPENDKV else []))
# @pytest.mark.parametrize("new_kv", [False])
@pytest.mark.parametrize("causal,local", [(False, False), (True, False)] + ([(False, True)] if not DISABLE_LOCAL else []))
# @pytest.mark.parametrize("causal,local", [(False, False), (True, False)])
# @pytest.mark.parametrize("causal,local", [(True, False)])
@pytest.mark.parametrize("seqlen_new_eq_seqlen_q", [True, False] if not DISABLE_APPENDKV else [True])
# @pytest.mark.parametrize("seqlen_new_eq_seqlen_q", [False])
# @pytest.mark.parametrize("has_rotary_seqlens", [False, True])
@pytest.mark.parametrize("has_rotary_seqlens", [False])
@pytest.mark.parametrize("rotary_interleaved", [False, True] if not DISABLE_APPENDKV else [False])
# @pytest.mark.parametrize("rotary_interleaved", [False])
@pytest.mark.parametrize("rotary_fraction", [0.0, 0.5, 1.0] if (not DISABLE_APPENDKV) and (apply_rotary_emb is not None) else [0.0])
# @pytest.mark.parametrize("rotary_fraction", [0.0])
@pytest.mark.parametrize("page_size", [None] + ([1, 4, 128] if not DISABLE_PAGEDKV else []))
# @pytest.mark.parametrize("page_size", [None])
@pytest.mark.parametrize("has_leftpad", [False, True])
# @pytest.mark.parametrize("has_leftpad", [False])
@pytest.mark.parametrize("has_batch_idx", [False, True])
# @pytest.mark.parametrize("has_batch_idx", [True])
@pytest.mark.parametrize("varlen_q", [False, True])
# @pytest.mark.parametrize("varlen_q", [True])
# @pytest.mark.parametrize("d", [32, 59, 64, 80, 128, 256])
# @pytest.mark.parametrize("d", [32, 64, 96, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize('d', [32, 40, 64, 80, 96, 128, 160, 192])
# @pytest.mark.parametrize('d', [56, 80])
@pytest.mark.parametrize("d", [128])
# @pytest.mark.parametrize("d", [192])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (1, 128),
        (1, 339),
        (3, 1024),
        (64, 800),
        (64, 256),
        (3, 799),
        (64, 2048),
        (16, 20000),
        # (1, 128 * 1024),
        # (16, 128 * 1024),
        (128, 128),
        (256, 512),  # To test appending KV with more than 1 block
        (2048, 3577),  # Enough tile to test persistent scheduler
    ],
)
# @pytest.mark.parametrize('seqlen_q,seqlen_k', [(256, 128)])
def test_flash_attn_kvcache(
    seqlen_q,
    seqlen_k,
    d,
    varlen_q,
    has_batch_idx,
    has_leftpad,
    page_size,
    rotary_fraction,
    rotary_interleaved,
    has_rotary_seqlens,
    seqlen_new_eq_seqlen_q,
    causal,
    local,
    new_kv,
    mha_type,
    dtype,
):
    if page_size is not None and seqlen_k % page_size != 0:
        pytest.skip()
    if seqlen_q > seqlen_k and new_kv:
        pytest.skip()
    if not new_kv and rotary_fraction > 0.0:
        pytest.skip()
    if rotary_fraction == 0.0 and has_rotary_seqlens:
        pytest.skip()
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    batch_size = 5
    # batch_size = 1
    batch_size_cache = batch_size if not has_batch_idx else batch_size * 2
    nheads = 6
    # nheads = 1
    # rotary_dim must be a multiple of 16, and must be <= d
    rotary_dim = math.floor(int(rotary_fraction * d) / 16) * 16
    nheads_k = nheads if mha_type == "mha" else (1 if mha_type == "mqa" else 3)
    assert nheads % nheads_k == 0
    dtype_ref = torch.bfloat16 if dtype == torch.float8_e4m3fn else dtype
    dv_vals = [128, d] if d > 128 and d <= 192 else ([256, 512, d] if d <= 64 else [d])
    if dtype == torch.float8_e4m3fn:
        dv_vals = [d]
    attention_chunk_vals = [torch.randint(1, seqlen_k * 2, (1,)).item(), 0] if (causal or local) and not DISABLE_LOCAL else [0]
    for dv, attention_chunk in itertools.product(dv_vals, attention_chunk_vals):
        print(f"{dv = }, {attention_chunk = }")
        has_qv = d == 64 and dv >= 256
        q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype_ref).to(dtype).to(dtype_ref)
        if has_qv:
            qv = torch.randn(batch_size, seqlen_q, nheads, dv, device=device, dtype=dtype_ref).to(dtype).to(dtype_ref)
        else:
            qv = None
        if varlen_q:
            query_padding_mask = generate_random_padding_mask(seqlen_q, batch_size, device, mode="random")
            q_unpad, indices_q, cu_seqlens_q, max_seqlen_q, *rest = unpad_input(q, query_padding_mask)
            output_pad_fn = lambda output_unpad: pad_input(
                output_unpad, indices_q, batch_size, seqlen_q
            )
            qv_unpad = rearrange(qv, "b s ... -> (b s) ...")[indices_q] if has_qv else None
        else:
            query_padding_mask = None
            q_unpad = q
            qv_unpad = qv
            cu_seqlens_q, max_seqlen_q = None, None
        # Put window_size after QKV randn so that window_size changes from test to test
        window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))

        seqlen_new = seqlen_q if seqlen_new_eq_seqlen_q else torch.randint(1, seqlen_q + 1, (1,)).item()
        cu_seqlens_k_new = None
        key_new_padding_mask = None
        if new_kv:
            k = torch.randn(batch_size, seqlen_new, nheads_k, d, device=device, dtype=dtype_ref).to(dtype).to(dtype_ref)
            v = torch.randn(batch_size, seqlen_new, nheads_k, dv, device=device, dtype=dtype_ref).to(dtype).to(dtype_ref)
            if varlen_q:  # k & v are also varlen
                key_new_padding_mask = generate_random_padding_mask(seqlen_new, batch_size, device, mode="random")
                k_unpad, indices_k, cu_seqlens_k_new, *rest = unpad_input(k, key_new_padding_mask)
                v_unpad, *rest = unpad_input(v, key_new_padding_mask)
            else:
                k_unpad, v_unpad = k, v
        else:
            k, v, k_unpad, v_unpad = None, None, None, None
        if page_size is None:
            k_cache = torch.randn(batch_size_cache, seqlen_k, nheads_k, d, device=device, dtype=dtype_ref).to(dtype).to(dtype_ref)
            v_cache = torch.randn(batch_size_cache, seqlen_k, nheads_k, dv, device=device, dtype=dtype_ref).to(dtype).to(dtype_ref)
            page_table = None
        else:
            (
                k_cache,
                v_cache,
                page_table,
                k_cache_paged,
                v_cache_paged,
                num_blocks,
            ) = _generate_block_kvcache(
                seqlen_k, page_size, batch_size_cache, nheads_k, d, dv, device, dtype, dtype_ref
            )
        cache_seqlens = torch.randint(
            0 if new_kv else 1,
            # If we don't use seqlen_q in the case of causal and rotary, cos/sin won't be long enough
            (
                (seqlen_k - (seqlen_q if (causal or local) and rotary_dim > 1 else seqlen_new) + 1)
                if new_kv
                else (seqlen_k + 1)
            ),
            (batch_size,),
            dtype=torch.int32,
            device=device,
        )
        if has_leftpad:
            cache_leftpad = torch.cat([torch.randint(0, cache_seqlens[i].item(), (1,), dtype=torch.int32, device=device)
                                    if cache_seqlens[i].item() > 0 else torch.zeros(1, dtype=torch.int32, device=device)
                                    for i in range(batch_size)])
        else:
            cache_leftpad = None
        if has_batch_idx:
            cache_batch_idx = torch.randperm(batch_size_cache, dtype=torch.int32, device=device)[
                :batch_size
            ]
        else:
            cache_batch_idx = None
        arange = rearrange(torch.arange(seqlen_k, device=device), "s -> 1 s")
        cache_seqlens_expanded = rearrange(cache_seqlens, "b -> b 1")
        if not new_kv:
            key_padding_mask = arange < cache_seqlens_expanded
        else:
            k_new_seqlens = key_new_padding_mask.sum(-1, keepdims=True) if varlen_q else seqlen_new
            key_padding_mask = arange < cache_seqlens_expanded + k_new_seqlens
        if has_leftpad:
            key_padding_mask = torch.logical_and(
                key_padding_mask, arange >= cache_leftpad.unsqueeze(-1).expand(-1, seqlen_k)
            )
        # cache_seqlens = torch.tensor([64], dtype=torch.int32, device=device)
        rotary_seqlens = cache_seqlens if not has_rotary_seqlens else cache_seqlens // 2
        if rotary_dim > 0:
            angle = (
                torch.rand(
                    seqlen_k if page_size is None else num_blocks * page_size,
                    rotary_dim // 2,
                    device=device,
                )
                * 2
                * math.pi
            )
            cos = torch.cos(angle).to(dtype=dtype_ref).to(dtype).to(dtype_ref)
            sin = torch.sin(angle).to(dtype=dtype_ref).to(dtype).to(dtype_ref)
            if causal or local:
                q_ro = apply_rotary_emb(
                    q, cos, sin, seqlen_offsets=rotary_seqlens, interleaved=rotary_interleaved
                )
            else:
                q_ro = rearrange(
                    apply_rotary_emb(
                        rearrange(q, "b s h d -> b 1 (s h) d"),
                        cos,
                        sin,
                        seqlen_offsets=rotary_seqlens,
                        interleaved=rotary_interleaved,
                    ),
                    "b 1 (s h) d -> b s h d",
                    s=seqlen_q,
                )
            # q_ro = q
            k_ro = apply_rotary_emb(
                k, cos, sin, seqlen_offsets=rotary_seqlens, interleaved=rotary_interleaved
            )
        else:
            cos, sin = None, None
            q_ro, k_ro = q, k
        # k_cache[:, 64:] = -1
        k_cache_ref = (k_cache if not has_batch_idx else k_cache[cache_batch_idx]).clone()
        v_cache_ref = (v_cache if not has_batch_idx else v_cache[cache_batch_idx]).clone()
        if new_kv:
            update_mask = torch.logical_and(
                cache_seqlens_expanded <= arange, arange < cache_seqlens_expanded + k_new_seqlens
            )
            k_to_update = rearrange(k_ro, "b s ... -> (b s) ...")
            v_to_update = rearrange(v, "b s ... -> (b s) ...")
            if varlen_q:
                k_to_update = k_to_update[indices_k]
                v_to_update = v_to_update[indices_k]
            k_cache_ref[update_mask] = k_to_update
            v_cache_ref[update_mask] = v_to_update
        k_cache_rep = repeat(k_cache_ref, "b s h d -> b s (h g) d", g=nheads // nheads_k)
        v_cache_rep = repeat(v_cache_ref, "b s h d -> b s (h g) d", g=nheads // nheads_k)
        out_ref, _ = attention_ref(
            q_ro,
            k_cache_rep,
            v_cache_rep,
            query_padding_mask,
            key_padding_mask,
            causal=causal,
            qv=qv,
            window_size=window_size,
            attention_chunk=attention_chunk,
            key_leftpad=cache_leftpad,
        )
        out_pt, _ = attention_ref(
            q_ro,
            k_cache_rep,
            v_cache_rep,
            query_padding_mask,
            key_padding_mask,
            causal=causal,
            qv=qv,
            window_size=window_size,
            attention_chunk=attention_chunk,
            upcast=False,
            reorder_ops=True,
            key_leftpad=cache_leftpad,
            intermediate_dtype=dtype if dtype == torch.float8_e4m3fn else None
        )
        q = q.to(dtype)
        q_unpad = q_unpad.to(dtype) if varlen_q else None
        k_cache = k_cache.to(dtype)
        v_cache = v_cache.to(dtype)
        k_cache_paged = k_cache_paged.to(dtype) if page_size is not None else None
        v_cache_paged = v_cache_paged.to(dtype) if page_size is not None else None
        k = k.to(dtype) if k is not None else None
        v = v.to(dtype) if v is not None else None
        k_unpad = k_unpad.to(dtype) if k_unpad is not None else None
        v_unpad = v_unpad.to(dtype) if v_unpad is not None else None
        qv = qv.to(dtype) if qv is not None else None
        qv_unpad = qv_unpad.to(dtype) if (varlen_q and qv is not None) else None
        cos = cos.to(dtype) if cos is not None else None
        sin = sin.to(dtype) if sin is not None else None
        k_cache_saved = k_cache.clone() if page_size is None else k_cache_paged.clone()
        v_cache_saved = v_cache.clone() if page_size is None else v_cache_paged.clone()
        num_splits_vals = [1, 3, 0] if not DISABLE_SPLIT else [1]
        precompute_metadata_vals = [False, True]
        for num_splits, precompute_metadata in itertools.product(num_splits_vals, precompute_metadata_vals):
            print(f"{num_splits = }, {precompute_metadata = }")
            if precompute_metadata:
                scheduler_metadata = get_scheduler_metadata(
                    batch_size,
                    max_seqlen_q if varlen_q else seqlen_q,
                    seqlen_k if page_size is None else page_table.shape[1] * page_size,
                    nheads, nheads_k, d,
                    cache_seqlens, q.dtype, headdim_v=dv, cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k_new=cu_seqlens_k_new, cache_leftpad=cache_leftpad,
                    max_seqlen_k_new=seqlen_new, page_size=page_size,
                    causal=causal, window_size=window_size, attention_chunk=attention_chunk,
                    num_splits=num_splits,
                )
            else:
                scheduler_metadata = None
            # Repeat to test metadata reuse
            for _ in range(1 if not precompute_metadata else 2):
                if page_size is None:
                    k_cache.copy_(k_cache_saved)
                    v_cache.copy_(v_cache_saved)
                else:
                    k_cache_paged.copy_(k_cache_saved)
                    v_cache_paged.copy_(v_cache_saved)
                out, lse, *rest = flash_attn_with_kvcache(
                    q if not varlen_q else q_unpad,
                    k_cache if page_size is None else k_cache_paged,
                    v_cache if page_size is None else v_cache_paged,
                    k if not new_kv or not varlen_q else k_unpad,
                    v if not new_kv or not varlen_q else v_unpad,
                    qv=qv if not varlen_q else qv_unpad,
                    rotary_cos=cos,
                    rotary_sin=sin,
                    cache_seqlens=cache_seqlens,
                    cache_batch_idx=cache_batch_idx,
                    cache_leftpad=cache_leftpad,
                    page_table=page_table,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k_new=cu_seqlens_k_new,
                    max_seqlen_q=max_seqlen_q,
                    rotary_seqlens=rotary_seqlens,
                    causal=causal,
                    window_size=window_size,
                    attention_chunk=attention_chunk,
                    rotary_interleaved=rotary_interleaved,
                    scheduler_metadata=scheduler_metadata,
                    num_splits=num_splits,
                    return_softmax_lse=True,
                )
                if varlen_q:
                    out = output_pad_fn(out)
                # out = flash_attn_with_kvcache(
                #     q, k_cache, v_cache, cache_seqlens=cache_seqlens, causal=causal, window_size=window_size
                # )
                # out = flash_attn_with_kvcache(q, k_cache, v_cache, causal=causal, window_size=window_size)
                # qk = torch.einsum("bqhd,bkhd->bhqk", q, k_cache_ref)
                # m = qk.amax(-1, keepdim=True)
                # s_tmp = torch.exp((qk - m) / math.sqrt(d))
                # o1 = torch.einsum('bhst,bthd->bshd', s_tmp, v_cache_ref)
                # lse_ref = torch.logsumexp(qk / math.sqrt(d), -1)
                # probs = torch.softmax(qk, dim=-1)
                print(f"Output max diff: {(out - out_ref).abs().max().item()}")
                print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
                print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
                print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")
                # breakpoint()

                # Check that FlashAttention's numerical error is at most twice the numerical error
                # of a Pytorch implementation.
                if new_kv:
                    if page_size is None:
                        k_cache_select = (
                            k_cache.to(dtype_ref) if not has_batch_idx else k_cache.to(dtype_ref)[cache_batch_idx]
                        )
                        v_cache_select = (
                            v_cache.to(dtype_ref) if not has_batch_idx else v_cache.to(dtype_ref)[cache_batch_idx]
                        )
                    else:
                        k_cache_select = rearrange(
                            k_cache_paged.to(dtype_ref)[(page_table if not has_batch_idx else page_table[cache_batch_idx]).flatten()],
                            "(b nblocks) block_size ... -> b (nblocks block_size) ...",
                            b=batch_size,
                        )[:, :seqlen_k].to(dtype_ref)
                        v_cache_select = rearrange(
                            v_cache_paged.to(dtype_ref)[(page_table if not has_batch_idx else page_table[cache_batch_idx]).flatten()],
                            "(b nblocks) block_size ... -> b (nblocks block_size) ...",
                            b=batch_size,
                        )[:, :seqlen_k].to(dtype_ref)
                    k_cache_ref = k_cache_ref.to(dtype).to(dtype_ref)
                    v_cache_ref = v_cache_ref.to(dtype).to(dtype_ref)
                    if dtype is not torch.float8_e4m3fn:
                        assert torch.equal(v_cache_select, v_cache_ref)
                    else:
                        assert torch.allclose(v_cache_select, v_cache_ref, rtol=1e-3, atol=1e-3)
                    # breakpoint()
                    # if rotary_dim == 0 and dtype is not torch.float8_e4m3fn:
                    if rotary_dim == 0:
                        assert torch.equal(k_cache_select, k_cache_ref)
                    else:
                        # if not torch.allclose(k_cache_select, k_cache_ref, rtol=1e-3, atol=1e-3):
                        #     breakpoint()
                        if dtype is not torch.float8_e4m3fn:
                            assert torch.allclose(k_cache_select, k_cache_ref, rtol=1e-3, atol=1e-3)
                        else:
                            assert torch.allclose(k_cache_select, k_cache_ref, rtol=1e-1, atol=1e-1)
                mult = 4 if dtype == torch.float8_e4m3fn else 2
                assert (out - out_ref).abs().max().item() <= mult * (out_pt - out_ref).abs().max().item() + 1e-5
                mult_mean = 3 if dtype == torch.float8_e4m3fn else 1.5
                assert (out - out_ref).abs().mean().item() <= mult_mean * (out_pt - out_ref).abs().mean().item()


def _generate_block_kvcache(seqlen_k, page_size, batch_size, nheads_k, d, dv, device, dtype, dtype_ref):
    num_blocks = math.ceil(seqlen_k / page_size) * batch_size * 3
    k_cache_paged = torch.randn(
        num_blocks, page_size, nheads_k, d, device=device, dtype=dtype_ref
    ).to(dtype).to(dtype_ref)
    v_cache_paged = torch.randn(
        num_blocks, page_size, nheads_k, dv, device=device, dtype=dtype_ref
    ).to(dtype).to(dtype_ref)
    page_table = rearrange(
        torch.randperm(num_blocks, dtype=torch.int32, device=device),
        "(b nblocks) -> b nblocks",
        b=batch_size,
    )
    k_cache = rearrange(
        k_cache_paged[page_table.flatten()],
        "(b nblocks) block_size ... -> b (nblocks block_size) ...",
        b=batch_size,
    )[:, :seqlen_k]
    v_cache = rearrange(
        v_cache_paged[page_table.flatten()],
        "(b nblocks) block_size ... -> b (nblocks block_size) ...",
        b=batch_size,
    )[:, :seqlen_k]
    return k_cache, v_cache, page_table, k_cache_paged, v_cache_paged, num_blocks


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
# @pytest.mark.parametrize('causal', [False])
@pytest.mark.parametrize('d', [128])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (64, 8192),
    ],
)
def test_flash_attn_cluster(seqlen_q, seqlen_k, d, causal, dtype):
    device = "cuda"
    torch.random.manual_seed(0)
    batch_size = 2
    nheads = 16
    nheads_kv = 4
    # There was a bug where this would cause "unspecified launch failure" due to Cluster
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)
    k = torch.randn(batch_size, seqlen_k, nheads_kv, d, device=device, dtype=dtype)
    v = torch.randn(batch_size, seqlen_k, nheads_kv, d, device=device, dtype=dtype)
    for _ in range(100):
        flash_attn_func(q, k, v, causal=causal)


# @pytest.mark.parametrize("dtype", ([torch.float16] if is_sm75 else [torch.float16, torch.bfloat16]))
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
# @pytest.mark.parametrize('causal', [False])
@pytest.mark.parametrize("d", [32, 40, 59, 64, 80, 96, 111, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize("d", [32, 40, 59, 64, 80, 96, 111, 128])
# @pytest.mark.parametrize('d', [32, 56, 64, 80, 96, 128])
# @pytest.mark.parametrize("d", [32, 64, 96, 128, 160, 192])
# @pytest.mark.parametrize('d', [80])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (1, 239),
        (239, 1),
        (3, 799),
        (799, 3),
        (1024, 128),
        (97, 97),
        (128, 128),
        (200, 200),
        (256, 256),
        (257, 257),
        (384, 384),
        (512, 512),
        (768, 768),
        (1024, 1024),
        (2048, 2048),
    ],
)
def test_flash_attn_race_condition(seqlen_q, seqlen_k, d, causal, dtype):
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    # Simulate under memory load
    dummy = torch.empty(70 * 1024 ** 3, dtype=torch.uint8, device=device)
    batch_size = 60  # Sometimes we need large batch size for the race conditions to trigger
    nheads = 4
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=True)
    torch.random.manual_seed(42)
    out0 = flash_attn_func(q, k, v, causal=causal)
    g = torch.randn_like(out0)
    dq0, dk0, dv0 = torch.autograd.grad(out0, (q, k, v), g)
    # Numerical error if we just do any arithmetic on dq
    dq_atol = 2 * ((dq0 + 0.3 - 0.3) - dq0).abs().max().item()

    for i in range(1000):
        torch.random.manual_seed(42)
        out = flash_attn_func(q, k, v, causal=causal)
        assert torch.equal(out, out0)
        # assert torch.equal(lse, lse0)

        dq, dk, dv = torch.autograd.grad(out, (q, k, v), g)
        dq_equal = torch.allclose(dq, dq0, atol=dq_atol)
        if not dq_equal:
            print(f"Iter {i}, {dq_atol = }, dQ max diff: {(dq - dq0).abs().max().item()}")
            # breakpoint()
        assert torch.equal(dv, dv0)
        assert torch.equal(dk, dk0)
        assert dq_equal


def attention_combine_ref(out_partial, lse_partial):
    """
    out_partial: (num_splits, batch_size, seqlen, nheads, d)
    lse_partial: (num_splits, batch_size, nheads, seqlen)
    """
    lse = torch.logsumexp(lse_partial, dim=0)
    scale = torch.exp(lse_partial - lse)
    scale = torch.where(torch.isinf(scale) | torch.isnan(scale), torch.zeros_like(scale), scale)
    out = (scale.unsqueeze(-1) * out_partial).sum(0)
    return out, lse


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
# @pytest.mark.parametrize("dtype", [torch.float32])
# @pytest.mark.parametrize("d", [32, 40, 59, 64, 80, 96, 111, 128, 160, 192, 224, 256])
@pytest.mark.parametrize("d", [64, 96, 128, 192, 256, 512])
# @pytest.mark.parametrize("d", [128])
@pytest.mark.parametrize("seqlen", [1, 2, 3, 32, 64, 256, 113, 108, 640, 1024])
# @pytest.mark.parametrize("seqlen", [12, 32, 64, 256, 112, 108, 640, 1024, 2048, 8192])
# @pytest.mark.parametrize("seqlen", [15])
@pytest.mark.parametrize("num_splits", [1, 2, 3, 5, 17, 32, 55, 97, 133])
# @pytest.mark.parametrize("num_splits", [1, 2, 3, 5, 11])
# @pytest.mark.parametrize("num_splits", [128])
def test_flash_attn_combine(num_splits, seqlen, d, dtype):
    if DISABLE_SPLIT:
        pytest.skip()
    device = "cuda"
    # set seed
    torch.random.manual_seed(1)
    batch_size = 5
    nheads = 16
    # batch_size = 1
    # nheads = 1
    out_partial = torch.randn(num_splits * 2, batch_size, nheads, seqlen, d, device=device, dtype=torch.float32).transpose(2, 3)[:num_splits]  # To test non-contiguous tensor
    lse_partial = torch.randn(num_splits, batch_size, nheads * 2, seqlen, device=device, dtype=torch.float32).transpose(-1, -2)[:, :, :, :nheads]  # To test non-contiguous tensor
    # To test short-circuiting based on num_splits
    lse_partial[num_splits // 2:, :batch_size // 3] = -float("inf")
    out, lse = flash_attn_combine(out_partial, lse_partial, out_dtype=dtype)
    out_ref, lse_ref = attention_combine_ref(out_partial, lse_partial)
    out_pt = out_ref.to(dtype)

    print(f"LSE max diff: {(lse - lse_ref).abs().max().item()}")
    print(f"LSE mean diff: {(lse - lse_ref).abs().mean().item()}")
    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")
    # breakpoint()

    assert torch.allclose(lse, lse_ref, atol=1e-5, rtol=1e-5)
    multiple = 2
    assert ((out - out_ref).abs().max().item() <= multiple * (out_pt - out_ref).abs().max().item()) or torch.allclose(out, out_pt, atol=1e-5, rtol=1e-5)

    # from flash_attn.utils.benchmark import pytorch_profiler
    # # pytorch_profiler(torch.sum, lse_partial)
    # pytorch_profiler(flash_attn_combine, out_partial, lse_partial)
    # pytorch_profiler(torch.sum, out_partial)

def test_flash3_bw_compatibility() -> None:
    # Let's try to always stay backward compatible! This will make life easier
    # for downstream libaries, users, and exported models.
    # 1/ Instead of removing arguments, error out if their value is no longer supported
    # 2/ When adding arguments, add them at the end with a default value
    assert torch.ops.flash_attn_3.fwd.default._schema.is_backward_compatible_with(parse_schema(
        "flash_attn_3::fwd(Tensor q, Tensor k, Tensor v, Tensor(k_new!)? k_new=None, "
        "Tensor(v_new!)? v_new=None, Tensor? q_v=None, Tensor(out!)? out=None, "
        "Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_k=None, "
        "Tensor? cu_seqlens_k_new=None, Tensor? seqused_q=None, Tensor? seqused_k=None, "
        "int? max_seqlen_q=None, int? max_seqlen_k=None, Tensor? page_table=None, "
        "Tensor? kv_batch_idx=None, Tensor? leftpad_k=None, Tensor? rotary_cos=None, Tensor? rotary_sin=None, "
        "Tensor? seqlens_rotary=None, Tensor? q_descale=None, Tensor? k_descale=None, Tensor? v_descale=None, "
        "float? softmax_scale=None, bool is_causal=False, int window_size_left=-1, int window_size_right=-1, "
        "int attention_chunk=0, float softcap=0., bool is_rotary_interleaved=False, "
        "Tensor? scheduler_metadata=None, int num_splits=0, bool? pack_gqa=None, int sm_margin=0) "
        "-> (Tensor(out!), Tensor, Tensor, Tensor)"
    ))
    assert torch.ops.flash_attn_3.bwd.default._schema.is_backward_compatible_with(parse_schema(
        "flash_attn_3::bwd(Tensor dout, Tensor q, Tensor k, Tensor v, Tensor out, Tensor softmax_lse, "
        "Tensor(dq!)? dq=None, Tensor(dk!)? dk=None, Tensor(dv!)? dv=None, Tensor? cu_seqlens_q=None, "
        "Tensor? cu_seqlens_k=None, Tensor? seqused_q=None, Tensor? seqused_k=None, int? max_seqlen_q=None, "
        "int? max_seqlen_k=None, float? softmax_scale=None, bool is_causal=False, int window_size_left=-1, "
        "int window_size_right=-1, float softcap=0., bool deterministic=False, int sm_margin=0) "
        "-> (Tensor(dq!), Tensor(dk!), Tensor(dv!), Tensor, Tensor, Tensor, Tensor, Tensor)"
    ))
    assert torch.ops.flash_attn_3.fwd_combine.default._schema.is_backward_compatible_with(parse_schema(
        "flash_attn_3::fwd_combine(Tensor out_partial, Tensor lse_partial, Tensor(out!)? out=None, "
        "ScalarType? out_dtype=None) -> (Tensor(out!), Tensor)"
    ))
    assert torch.ops.flash_attn_3.get_scheduler_metadata.default._schema.is_backward_compatible_with(parse_schema(
        "flash_attn_3::get_scheduler_metadata(int batch_size, int max_seqlen_q, int max_seqlen_k, "
        "int num_heads, int num_heads_k, int headdim, int headdim_v, ScalarType qkv_dtype, Tensor seqused_k, "
        "Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_k=None, Tensor? cu_seqlens_k_new=None, "
        "Tensor? seqused_q=None, Tensor? leftpad_k=None, int? page_size=None, int max_seqlen_k_new=0, "
        "bool is_causal=False, int window_size_left=-1, int window_size_right=-1, "
        "int attention_chunk=0, bool has_softcap=False, int num_splits=0, bool? pack_gqa=None, "
        "int sm_margin=0) -> Tensor"
    ))
