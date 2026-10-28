# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao, Siyu Wang, Shengbin Di, Yuxi Chi, Johnsonms, Linfeng Zheng, Haoyan Huang, Lanbo Li, Yong Li, Wei Lin.

import itertools
import os
import random

import pytest
import torch
from einops import rearrange

from flash_attn.cute.testing import (
    attention_ref,
    generate_qkv,
    generate_random_padding_mask,
    is_fake_mode,
    maybe_fake_tensor_mode,
)
from flash_attn.cute.interface import flash_attn_varlen_func, get_scheduler_metadata
from flash_attn_test_utils import retry_on_oom

USE_FAKE_TENSOR = int(os.getenv("FLASH_ATTENTION_FAKE_TENSOR", 0)) == 1
DISABLE_SPLIT = os.getenv("FLASH_ATTENTION_DISABLE_SPLIT", "FALSE") == "TRUE"
IS_SM90 = torch.cuda.get_device_capability()[0] == 9
HD256_GENERIC_BWD = os.getenv("FLASH_ATTENTION_HD256_GENERIC_BWD", "0") == "1"
IS_SM100 = torch.cuda.get_device_capability()[0] == 10
IS_SM120 = torch.cuda.get_device_capability()[0] == 12
TEST_BWD_ONLY = False
VERBOSE = True


# @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
# @pytest.mark.parametrize("mha_type", ["mha"])
# @pytest.mark.parametrize("has_learnable_sink", [False, True])
@pytest.mark.parametrize("has_learnable_sink", [False])
# @pytest.mark.parametrize("has_qv", [False, True])
@pytest.mark.parametrize("has_qv", [False])
@pytest.mark.parametrize("deterministic", [False, True])
# @pytest.mark.parametrize("deterministic", [False])
@pytest.mark.parametrize("softcap", [0.0, 15.0])
# @pytest.mark.parametrize("softcap", [0.0])
@pytest.mark.parametrize(
    "causal,local_enum",
    [(False, 0), (False, 1), (False, 2), (False, 3), (True, 0)],
)
# @pytest.mark.parametrize("add_unused_qkv", [False, True])
@pytest.mark.parametrize("add_unused_qkv", [False])
# @pytest.mark.parametrize("d", [32, 64, 96, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize('d', [32, 40, 64, 80, 96, 128, 160, 192, 256])
# @pytest.mark.parametrize('d', [32, 64, 96, 128, 160, 192])
# @pytest.mark.parametrize('d', [56, 80])
# @pytest.mark.parametrize('d', [32, 40, 64, 80, 96, 128])
# @pytest.mark.parametrize("d", [64, 96, 128])
# @pytest.mark.parametrize("d", [128, 192])
@pytest.mark.parametrize("d", [64, 128, 192, 256])
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
        # SM100 hd256 2CTA test cases
        (64, 1),
        (255, 256),
        (4096, 4096),
        (4224, 4224),
    ],
)
@pytest.mark.parametrize("varlen_mode", ["random", "third", "full"])
# @pytest.mark.parametrize("varlen_mode", ["full"])
@pytest.mark.parametrize(
    "zero_lengths_q, zero_lengths_k",
    [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ],
)
@pytest.mark.parametrize(
    "unpad_q, unpad_kv",
    [
        (True, True),
        (False, False),
        (True, False),
        (False, True),
    ],
)
@retry_on_oom
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_varlen_output(
    seqlen_q,
    seqlen_k,
    d,
    add_unused_qkv,
    causal,
    local_enum,
    softcap,
    deterministic,
    has_qv,
    has_learnable_sink,
    mha_type,
    dtype,
    varlen_mode,
    zero_lengths_q,
    zero_lengths_k,
    unpad_q,
    unpad_kv,
):
    local = local_enum > 0
    if local and causal:
        pytest.skip()
    # The SM100 head_dim=256 backward does not support these features yet (the forward does).
    # Remove these skips when support is added.
    if d == 256 and IS_SM100 and not HD256_GENERIC_BWD:
        if has_learnable_sink:
            pytest.skip("SM100 head_dim=256 backward does not support learnable_sink yet")
        if local:
            pytest.skip("SM100 head_dim=256 backward does not support local attention yet")
        if softcap > 0.0:
            pytest.skip("SM100 head_dim=256 backward does not support softcap yet")
        if not unpad_q and unpad_kv:
            pytest.skip("SM100 head_dim=256 backward: varlen-packed K without varlen Q is untested")
    if (
        causal or local
    ):  # Right now reference only supports causal attention with seqlen_k == seqlen_q
        seqlen_k = seqlen_q
    device = "cuda"
    # set seed
    seed = seqlen_q + seqlen_k + d + int(causal) * 2 + int(local)
    random.seed(seed)
    torch.random.manual_seed(seed)
    batch_size = 49 if seqlen_q <= 512 else 7
    nheads = 6
    # nheads = 1
    nheads_kv = nheads if mha_type == "mha" else (3 if mha_type == "gqa" else 1)
    dtype_ref = torch.bfloat16 if dtype == torch.float8_e4m3fn else dtype
    # dv_vals = [128, d] if d > 128 and d <= 192 else ([256, 512, d] if d <= 64 else [d])
    dv_vals = [128] if d == 192 else ([d] if d != 128 else [64, d])
    if d == 256:
        dv_vals = [256]  # (256, 256) is the only hd256 shape on SM100
    if dtype == torch.float8_e4m3fn:
        dv_vals = [d]
    # attention_chunk_vals = [torch.randint(1, seqlen_k * 2, (1,)).item(), 0] if seqlen_q <= seqlen_k else [0]
    attention_chunk_vals = [0]
    for dv, attention_chunk in itertools.product(dv_vals, attention_chunk_vals):
        q_ref = torch.randn(
            batch_size, seqlen_q, nheads, d, device=device, dtype=dtype_ref
        )
        if softcap > 0.0:
            # Ensure the values of qk are at least within softcap range.
            q_ref = (q_ref * softcap / 4).detach().requires_grad_()
        q_ref = q_ref.to(dtype).to(dtype_ref).requires_grad_()
        k_ref = (
            torch.randn(
                batch_size, seqlen_k, nheads_kv, d, device=device, dtype=dtype_ref
            )
            .to(dtype)
            .to(dtype_ref)
            .requires_grad_()
        )
        v_ref = (
            torch.randn(
                batch_size, seqlen_k, nheads_kv, dv, device=device, dtype=dtype_ref
            )
            .to(dtype)
            .to(dtype_ref)
            .requires_grad_()
        )
        if has_qv:
            qv_ref = (
                torch.randn(
                    batch_size, seqlen_q, nheads, dv, device=device, dtype=dtype_ref
                )
                .to(dtype)
                .to(dtype_ref)
            )
        else:
            qv_ref = None
        # Put window_size after QKV randn so that window_size changes from test to test
        window_size = (
            (None, None) if not local else tuple(random.randrange(0, seqlen_k) for _ in range(2))
        )
        if local_enum == 2:
            window_size = (None, window_size[1])
        elif local_enum == 3:
            window_size = (window_size[0], None)
        if local:
            print("window size = ", window_size)
        if has_learnable_sink:
            learnable_sink = torch.randn(nheads, dtype=torch.bfloat16, device=device)
        else:
            learnable_sink = None
        if dtype == torch.float8_e4m3fn:
            q_descale, k_descale, v_descale = [
                torch.rand(batch_size, nheads_kv, device=device, dtype=torch.float32)
                * 2
                for _ in range(3)
            ]
        else:
            q_descale, k_descale, v_descale = None, None, None
        q, k, v = [x.detach().requires_grad_() for x in (q_ref, k_ref, v_ref)]
        qv = qv_ref.detach() if has_qv else None
        query_padding_mask = generate_random_padding_mask(
            seqlen_q,
            batch_size,
            device,
            mode=varlen_mode,
            zero_lengths=zero_lengths_q,
        )
        key_padding_mask = generate_random_padding_mask(
            seqlen_k,
            batch_size,
            device,
            mode=varlen_mode,
            zero_lengths=zero_lengths_k,
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
        ) = generate_qkv(
            q,
            k,
            v,
            query_padding_mask,
            key_padding_mask,
            qv=qv,
            kvpacked=False,
            query_unused_mask=query_unused_mask,
            key_unused_mask=key_unused_mask,
        )
        if unpad_q:
            print("cu_seqlens_q = ", cu_seqlens_q)
        else:
            print("seqused_q = ", seqused_q)
        if unpad_kv:
            print("cu_seqlens_k = ", cu_seqlens_k)
        else:
            print("seqused_k = ", seqused_k)
        q_unpad, k_unpad, v_unpad = [
            x.detach().to(dtype).requires_grad_() for x in (q_unpad, k_unpad, v_unpad)
        ]

        out_ref, attn_ref = attention_ref(
            q_ref,
            k_ref,
            v_ref,
            query_padding_mask,
            key_padding_mask,
            causal=causal,
            qv=qv_ref,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            window_size=window_size,
            attention_chunk=attention_chunk,
            learnable_sink=learnable_sink,
            softcap=softcap,
        )
        out_pt, attn_pt = attention_ref(
            q_ref,
            k_ref,
            v_ref,
            query_padding_mask,
            key_padding_mask,
            causal=causal,
            qv=qv_ref,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            window_size=window_size,
            attention_chunk=attention_chunk,
            learnable_sink=learnable_sink,
            softcap=softcap,
            upcast=False,
            reorder_ops=True,
            intermediate_dtype=dtype if dtype == torch.float8_e4m3fn else None,
        )

        if not is_fake_mode():
            print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
            print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")

            if query_unused_mask is not None:
                q_zero_masking = rearrange(query_unused_mask, "b s -> b s 1 1")

            # Numerical error if we just do any arithmetic on out_ref
            fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
            rtol = 2 if softcap == 0.0 else 3

        pack_gqa_vals = [False, True, None] if not TEST_BWD_ONLY else [False]
        # pack_gqa_vals = [False]
        # num_splits_vals = [1, 3]
        # SplitKV is not supported for hdim >= 192, except hd256 on SM100
        split_hdim_ok = d < 192 or (IS_SM100 and d == 256 and dv == 256)
        num_splits_vals = [1, 3] if split_hdim_ok and not DISABLE_SPLIT and not TEST_BWD_ONLY else [1]
        precompute_metadata_vals = [False, True]
        for pack_gqa, num_splits, precompute_metadata in itertools.product(
            pack_gqa_vals, num_splits_vals, precompute_metadata_vals
        ):
            # SplitKV not supported on SM90/SM120 - skip this iteration
            if (IS_SM90 or IS_SM120) and num_splits > 1:
                continue
            if precompute_metadata and is_fake_mode():
                continue
            if precompute_metadata:
                scheduler_metadata = get_scheduler_metadata(
                    max_seqlen_q=seqlen_q,
                    max_seqlen_k=seqlen_k,
                    nheads=nheads,
                    nheads_kv=nheads_kv,
                    headdim=d,
                    headdim_v=dv,
                    num_splits=num_splits,
                    causal=causal,
                    cu_seqlens_q=cu_seqlens_q if unpad_q else None,
                    cu_seqlens_k=cu_seqlens_k if unpad_kv else None,
                    seqused_q=seqused_q if not unpad_q else None,
                    seqused_k=seqused_k if not unpad_kv else None,
                )
            else:
                scheduler_metadata = None
            # Repeat to exercise metadata reuse across calls.
            for _ in range(1 if not precompute_metadata else 2):
                out_unpad, lse = flash_attn_varlen_func(
                    q_unpad if unpad_q else q,
                    k_unpad if unpad_kv else k,
                    v_unpad if unpad_kv else v,
                    cu_seqlens_q=cu_seqlens_q if unpad_q else None,
                    cu_seqlens_k=cu_seqlens_k if unpad_kv else None,
                    max_seqlen_q=seqlen_q,
                    max_seqlen_k=seqlen_k,
                    seqused_q=seqused_q if not unpad_q else None,
                    seqused_k=seqused_k if not unpad_kv else None,
                    causal=causal,
                    # qv=qv_unpad,
                    # q_descale=q_descale,
                    # k_descale=k_descale, v_descale=v_descale,
                    window_size=window_size,
                    # attention_chunk=attention_chunk,
                    learnable_sink=learnable_sink,
                    softcap=softcap,
                    scheduler_metadata=scheduler_metadata,
                    num_splits=num_splits,
                    pack_gqa=pack_gqa,
                    deterministic=deterministic,
                )
                out = output_pad_fn(out_unpad) if unpad_q else out_unpad
                if is_fake_mode():
                    # no more flash_attn cutedsl calls for the rest of the loop
                    # skip data-dependent postprocessing
                    continue
                if query_unused_mask is not None:
                    out.masked_fill_(q_zero_masking, 0.0)
                # When unpad_q=False with seqused_q, the kernel doesn't write positions
                # beyond seqused_q, so those contain uninitialized values. Mask them out
                # before comparing.
                out_cmp, out_ref_cmp, out_pt_cmp = out, out_ref, out_pt
                if not unpad_q and seqused_q is not None:
                    seqused_mask = torch.arange(seqlen_q, device=device)[None, :] < seqused_q[:, None]
                    seqused_mask = rearrange(seqused_mask, "b s -> b s 1 1")
                    out_cmp = out.clone().masked_fill_(~seqused_mask, 0.0)
                    out_ref_cmp = out_ref.clone().masked_fill_(~seqused_mask, 0.0)
                    out_pt_cmp = out_pt.clone().masked_fill_(~seqused_mask, 0.0)
                print(f"Output max diff: {(out_cmp - out_ref_cmp).abs().max().item()}")
                print(f"Output mean diff: {(out_cmp - out_ref_cmp).abs().mean().item()}")
                # if not causal:
                #     print(f"LSE max diff: {(lse - lse_ref).abs().max().item()}")
                # breakpoint()

                # Check that FlashAttention's numerical error is at most 3x the numerical error
                # of a Pytorch implementation.
                assert (out_cmp - out_ref_cmp).abs().max().item() <= rtol * (
                    out_pt_cmp - out_ref_cmp
                ).abs().max().item() + fwd_atol

        if (
            dtype != torch.float8_e4m3fn
            and not has_qv
            and not dv > 256
            and not attention_chunk != 0
            and (
                (dv == d and d <= 128)
                or (d == 192 and dv == 128)
                or (IS_SM100 and d == 256 and dv == 256 and (softcap == 0.0 or HD256_GENERIC_BWD))
            )
            and not has_learnable_sink
            # and False
        ):
            if d > 192 and IS_SM90:
                pytest.xfail("hdim > 192 backward: SM90 not supported yet")
            if d != dv and mha_type != "mha" and IS_SM90:
                pytest.xfail("SM90 GQA bwd currently requires headdim == headdim_v")
            g_unpad = torch.randn_like(out_unpad)
            # do_o = ((g_unpad.float() * out_unpad.float()).sum(-1)).transpose(-1, -2)
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
            dq_unpad, dk_unpad, dv_unpad = torch.autograd.grad(
                out_unpad,
                (
                    q_unpad if unpad_q else q,
                    k_unpad if unpad_kv else k,
                    v_unpad if unpad_kv else v,
                ),
                g_unpad
            )
            if is_fake_mode():
                # no more flash_attn cutedsl calls for the rest of the loop
                # skip data-dependent postprocessing
                continue
            dq = dq_pad_fn(dq_unpad) if unpad_q else dq_unpad
            dk = dk_pad_fn(dk_unpad) if unpad_kv else dk_unpad
            dv = dk_pad_fn(dv_unpad) if unpad_kv else dv_unpad
            if key_unused_mask is not None:
                k_zero_masking = rearrange(key_unused_mask, "b s -> b s 1 1")
                dk.masked_fill_(k_zero_masking, 0.0)
                dv.masked_fill_(k_zero_masking, 0.0)
            if query_unused_mask is not None:
                dq.masked_fill_(q_zero_masking, 0.0)
            if not unpad_kv:
                dk.masked_fill_(rearrange(~key_padding_mask, "b s -> b s 1 1"), 0.0)
                dv.masked_fill_(rearrange(~key_padding_mask, "b s -> b s 1 1"), 0.0)
            if not unpad_q:
                dq.masked_fill_(rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0)
            # print(f"dO_O max diff: {(softmax_d - do_o).abs().max().item()}")
            # assert (softmax_d - do_o).abs().max().item() <= 1e-5
            # assert dq_accum.abs().max().item() == 0.0
            g = output_pad_fn(g_unpad) if unpad_q else g_unpad

            # qk = torch.einsum('bthd,bshd->bhts', q / (d ** 0.5), k).float()
            # qk = torch.masked_fill(qk, rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"))
            # dS = torch.einsum('bthd,bshd->bhts', g.float(), v.float())
            # P = torch.softmax(qk, -1)
            # dP = P * (dS - (g.float() * out.float()).sum(-1).transpose(1, 2).unsqueeze(-1))
            # dQ = torch.einsum('bhts,bshd->bthd', dP, k.float())
            # dV = torch.einsum('bhts,bthd->bshd', P, g.float())
            # dK = torch.einsum('bhts,bthd->bshd', dP, q.float())

            # dq, dk, dv = torch.autograd.grad(out, (q, k, v), g)
            dq_ref, dk_ref, dv_ref = torch.autograd.grad(
                out_ref, (q_ref, k_ref, v_ref), g
            )
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
            if VERBOSE:
                diff_dq = (dq - dq_ref).abs()
                max_idx = diff_dq.argmax()
                coords = torch.unravel_index(max_idx, diff_dq.shape)
                print(f"dQ max diff: {diff_dq.max().item()}")
                print(f"  at coordinates {tuple(c.item() for c in coords)}: dQ={dq[coords].item()}, dQ_ref={dq_ref[coords].item()}")

                diff_dk = (dk - dk_ref).abs()
                max_idx = diff_dk.argmax()
                coords = torch.unravel_index(max_idx, diff_dk.shape)
                print(f"dK max diff: {diff_dk.max().item()}")
                print(f"  at coordinates {tuple(c.item() for c in coords)}: dK={dk[coords].item()}, dK_ref={dk_ref[coords].item()}")

                diff_dv = (dv - dv_ref).abs()
                max_idx = diff_dv.argmax()
                coords = torch.unravel_index(max_idx, diff_dv.shape)
                print(f"dV max diff: {diff_dv.max().item()}")
                print(f"  at coordinates {tuple(c.item() for c in coords)}: dV={dv[coords].item()}, dV_ref={dv_ref[coords].item()}")
            # breakpoint()
            dq_atol = 2 * (dq_ref + 0.3 - 0.3 - dq_ref).abs().max().item() + (
                0 if softcap == 0 else 3e-4
            )
            assert (dq - dq_ref).abs().max().item() <= rtol * (
                dq_pt - dq_ref
            ).abs().max().item() + dq_atol
            dk_atol = 2 * (dk_ref + 0.3 - 0.3 - dk_ref).abs().max().item() + (
                0 if softcap == 0 else 3e-4
            )
            assert (dk - dk_ref).abs().max().item() <= rtol * (
                dk_pt - dk_ref
            ).abs().max().item() + dk_atol
            dv_atol = 2 * (dv_ref + 0.3 - 0.3 - dv_ref).abs().max().item() + (
                0 if softcap == 0 else 3e-4
            )
            assert (dv - dv_ref).abs().max().item() <= rtol * (
                dv_pt - dv_ref
            ).abs().max().item() + dv_atol
