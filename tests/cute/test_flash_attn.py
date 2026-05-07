# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao, Siyu Wang, Shengbin Di, Yuxi Chi, Johnsonms, Linfeng Zheng, Haoyan Huang, Lanbo Li, Yun Zhong, Man Yuan, Minmin Sun, Yong Li, Wei Lin.

import math
import itertools
import os
import random
import re
import gc
from functools import wraps

import pytest
import torch

from einops import rearrange, repeat

try:
    from flash_attn.layers.rotary import apply_rotary_emb
except ImportError:
    apply_rotary_emb = None

from flash_attn.cute.testing import (
    attention_ref,
    generate_qkv,
    generate_random_padding_mask,
    pad_input,
    unpad_input,
    maybe_fake_tensor_mode,
    is_fake_mode,
)
from flash_attn.cute.interface import (
    flash_attn_func,
    flash_attn_varlen_func,
    _flash_attn_fwd,
    _flash_attn_bwd,
)

def retry_on_oom(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except torch.OutOfMemoryError as e:
            if "out of memory" in str(e).lower():
                if hasattr(_flash_attn_fwd, "compile_cache"):
                    _flash_attn_fwd.compile_cache.clear()
                if hasattr(_flash_attn_bwd, "compile_cache"):
                    _flash_attn_bwd.compile_cache.clear()
                gc.collect()
                torch.cuda.empty_cache()
                return func(*args, **kwargs)
            else:
                raise
    return wrapper

# torch FakeTensorMode would enable fast cutedsl kernel compilation without allocating the actual GPU memory or running the kernel
# When operating fake tensors, we cannot perform data-dependent operations (e.g., `tensor.max()`).
USE_FAKE_TENSOR = int(os.getenv("FLASH_ATTENTION_FAKE_TENSOR", 0)) == 1
DISABLE_SPLIT = os.getenv("FLASH_ATTENTION_DISABLE_SPLIT", "FALSE") == "TRUE"
# SplitKV is not supported on SM90
IS_SM90 = torch.cuda.get_device_capability()[0] == 9
IS_SM100 = torch.cuda.get_device_capability()[0] == 10
TEST_BWD_ONLY = False
VERBOSE = True

# @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
# @pytest.mark.parametrize("mha_type", ["mha"])
@pytest.mark.parametrize("has_learnable_sink", [False, True])
# @pytest.mark.parametrize("has_learnable_sink", [False])
# @pytest.mark.parametrize("has_qv", [False, True])
@pytest.mark.parametrize("has_qv", [False])
@pytest.mark.parametrize("deterministic", [False, True])
# @pytest.mark.parametrize("deterministic", [False])
@pytest.mark.parametrize("softcap", [0.0, 15.0])
# @pytest.mark.parametrize("softcap", [0.0])
@pytest.mark.parametrize("local_enum", [0, 1, 2, 3])
# @pytest.mark.parametrize("local_enum", [0])
@pytest.mark.parametrize("causal", [False, True])
# @pytest.mark.parametrize("causal", [False])
# @pytest.mark.parametrize("d", [32, 64, 96, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize('d', [32, 40, 64, 80, 96, 128, 160, 192, 256])
# @pytest.mark.parametrize('d', [32, 64, 96, 128, 160, 192])
# @pytest.mark.parametrize('d', [56, 80])
# @pytest.mark.parametrize("d", [64, 128, 256])
# @pytest.mark.parametrize('d', [32, 40, 64, 80, 96, 128])
# @pytest.mark.parametrize("d", [64, 96, 128, 192])
# @pytest.mark.parametrize("d", [128, 192])
@pytest.mark.parametrize("d", [64, 96, 128, 192, 256])
# @pytest.mark.parametrize("d", [128])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (1, 1),
        (3, 3),
        (64, 32),
        (64, 128),
        (64, 1),  # SM100 hd256 2CTA test case
        (128, 128),
        (128, 192),
        (256, 256),
        (255, 256),  # SM100 hd256 2CTA test case
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
        (2048, 2048),
        (4096, 4096),
        (4224, 4224),
    ],
)
# @pytest.mark.parametrize('seqlen_q,seqlen_k', [(128, 128)])
@retry_on_oom
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_output(
    seqlen_q,
    seqlen_k,
    d,
    causal,
    local_enum,
    softcap,
    deterministic,
    has_qv,
    has_learnable_sink,
    mha_type,
    dtype,
):
    local = local_enum > 0
    if local and causal:
        pytest.skip()
    if has_qv and d != 64:
        pytest.skip()
    if has_qv and local:
        pytest.xfail("has_qv: local not supported yet")
    if has_qv and has_learnable_sink:
        pytest.xfail("has_qv: learnable sink not supported yet")
    # TODO(wangsiyu): SM100 head_dim=256 2CTA kernel currently does not support the following features.
    # Remove these skips when support is added.
    if d == 256 and IS_SM100:
        if has_learnable_sink:
            pytest.skip("SM100 head_dim=256 2CTA kernel does not support learnable_sink yet")
        if local:
            pytest.skip("SM100 head_dim=256 2CTA kernel does not support local attention yet")
        if softcap > 0.0:
            pytest.skip("SM100 head_dim=256 2CTA kernel does not support softcap yet")
        if deterministic:
            pytest.skip("SM100 head_dim=256 2CTA kernel does not support deterministic mode yet")
        if causal and seqlen_q > seqlen_k:
            pytest.skip("SM100 head_dim=256 2CTA kernel does not support causal attention with seqlen_q > seqlen_k yet")
    device = "cuda"
    # set seed
    seed = 0
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    batch_size = 9 if seqlen_k <= 2048 else 2
    # batch_size = 2
    nheads = 6 if not has_qv else 128
    # nheads = 1
    if not has_qv:
        nheads_kv = nheads if mha_type == "mha" else (3 if mha_type == "gqa" else 1)
    else:
        nheads_kv = nheads if mha_type == "mha" else (8 if mha_type == "gqa" else 1)
    dtype_ref = torch.bfloat16 if dtype == torch.float8_e4m3fn else dtype
    # dv_vals = [128, d] if d > 128 and d <= 192 else ([256, 512, d] if d <= 64 else [d])
    dv_vals = [128] if d == 192 else ([d] if d != 128 else [64, d])
    if dtype == torch.float8_e4m3fn:
        dv_vals = [d]
    if has_qv:
        assert d == 64
        dv_vals = [512]
    # attention_chunk_vals = [torch.randint(1, seqlen_k * 2, (1,)).item(), 0]
    attention_chunk_vals = [0]
    for dv, attention_chunk in itertools.product(dv_vals, attention_chunk_vals):
        q_ref = torch.randn(
            batch_size, seqlen_q, nheads, d, device=device, dtype=dtype_ref
        )
        if softcap > 0.0:
            # Ensure the values of qk are at least within softcap range.
            q_ref = q_ref * softcap / 4
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
            window_size = (None, -window_size[1])
        elif local_enum == 3:
            window_size = (-window_size[0], None)
        if local:
            print("window size = ", window_size)
        # window_size = (-1, -1) if not local else (16, 0)
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
            None,
            None,
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
        if not is_fake_mode():
            fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
            rtol = 2 if softcap == 0.0 else 3

            print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
            print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")
        # num_splits_vals = [1, 3]
        pack_gqa_vals = [True] if has_qv else [False, True, None] if not TEST_BWD_ONLY else [False]
        # SplitKV is not supported for hdim >= 192
        # pack_gqa_vals = [False]
        num_splits_vals = [1, 3] if d < 192 and not DISABLE_SPLIT and not TEST_BWD_ONLY and not has_qv else [1]
        for pack_gqa, num_splits in itertools.product(pack_gqa_vals, num_splits_vals):
            # SplitKV not supported on SM90 - skip this iteration
            if IS_SM90 and num_splits > 1:
                continue
            if IS_SM100 and (d >= 192 and dv >= 192) and not (d == 256 and dv == 256):
                continue
            # TODO(wangsiyu): SM100 head_dim=256 2CTA kernel does not support pack_gqa yet.
            # pack_gqa=None means auto-enable for GQA/MQA (qhead_per_kvhead > 1)
            # Remove this when support is added.
            if d == 256 and IS_SM100:
                if pack_gqa is True:
                    continue
                if pack_gqa is None and mha_type != "mha":
                    continue
            out, lse = flash_attn_func(
                q,
                k,
                v,
                qv=qv,
                causal=causal,
                # q_descale=q_descale, k_descale=k_descale, v_descale=v_descale,
                window_size=window_size,
                # attention_chunk=attention_chunk,
                softcap=softcap,
                learnable_sink=learnable_sink,
                pack_gqa=pack_gqa,
                num_splits=num_splits,
                deterministic=deterministic,
            )
            if is_fake_mode():
                # no more flash_attn cutedsl calls for the rest of the loop
                # skip data-dependent postprocessing
                continue
            print(f"Output max diff: {(out - out_ref).abs().max().item()}")
            print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
            # if not causal:
            #     print(f"LSE max diff: {(lse - lse_ref).abs().max().item()}")
            # breakpoint()

            # Check that FlashAttention's numerical error is at most twice the numerical error
            # of a Pytorch implementation.
            assert (out - out_ref).abs().max().item() <= rtol * (
                out_pt - out_ref
            ).abs().max().item() + fwd_atol

        if (
            dtype != torch.float8_e4m3fn
            and not has_qv
            and not dv > 256
            and not attention_chunk != 0
            and softcap == 0.0
            and (
                (dv == d and d <= 128)
                or (d == 192 and dv == 128)
                or (IS_SM100 and d == 256 and dv == 256)
            )
            and learnable_sink is None
            # and False
            and not ((causal or local) and seqlen_k < seqlen_q)
        ):
            if d > 192 and IS_SM90:
                pytest.xfail("hdim > 192 backward: SM90 not supported yet")
            if d != dv and mha_type != "mha" and IS_SM90:
                pytest.xfail("SM90 GQA bwd currently requires headdim == headdim_v")
            g = torch.randn_like(out)
            # do_o = ((g.float() * out.float()).sum(-1)).transpose(1, 2)
            dq, dk, dv = torch.autograd.grad(out, (q, k, v), g)
            if is_fake_mode():
                # no more flash_attn cutedsl calls for the rest of the loop
                # skip data-dependent postprocessing
                continue
            # print(f"dO_O max diff: {(softmax_d - do_o).abs().max().item()}")
            # assert (softmax_d - do_o).abs().max().item() <= 1e-5
            # assert dq_accum.abs().max().item() == 0.0

            # dS = torch.einsum('bthd,bshd->bhts', g.float(), v.float())
            # P = torch.softmax(qk, -1)
            # dP = P * (dS - do_o.transpose(1, 2).unsqueeze(1))
            # dQ = torch.einsum('bhts,bshd->bthd', dP, k.float())
            # dV = torch.einsum('bhts,bthd->bshd', P, g.float())
            # dK = torch.einsum('bhts,bthd->bshd', dP, q.float())
            # breakpoint()

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
@pytest.mark.parametrize("local_enum", [0, 1, 2, 3])
# @pytest.mark.parametrize("local_enum", [0])
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
    # TODO(wangsiyu): SM100 head_dim=256 2CTA kernel currently does not support the following features.
    # Remove these skips when support is added.
    if d == 256 and IS_SM100:
        if has_learnable_sink:
            pytest.skip("SM100 head_dim=256 2CTA kernel does not support learnable_sink yet")
        if local:
            pytest.skip("SM100 head_dim=256 2CTA kernel does not support local attention yet")
        if softcap > 0.0:
            pytest.skip("SM100 head_dim=256 2CTA kernel does not support softcap yet")
        if deterministic:
            pytest.skip("SM100 head_dim=256 2CTA kernel does not support deterministic mode yet")
        if causal and seqlen_q > seqlen_k:
            pytest.skip("SM100 head_dim=256 2CTA kernel does not support causal attention with seqlen_q > seqlen_k yet")
        if zero_lengths_q or zero_lengths_k:
            pytest.skip("SM100 head_dim=256 2CTA kernel does not support zero-length sequences yet")
        if not unpad_q or not unpad_kv:
            pytest.skip("SM100 head_dim=256 2CTA kernel does not support seqused_q/seqused_k mode yet (requires unpad_q=True and unpad_kv=True)")
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
        dv_vals = [256]  # SM100 hd=256 2CTA kernel only supports dv=256
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
        # SplitKV is not supported for hdim >= 192
        num_splits_vals = [1, 3] if d < 192 and not DISABLE_SPLIT and not TEST_BWD_ONLY else [1]
        for pack_gqa, num_splits in itertools.product(pack_gqa_vals, num_splits_vals):
            # SplitKV not supported on SM90 - skip this iteration
            if IS_SM90 and num_splits > 1:
                continue
            # TODO(wangsiyu): SM100 head_dim=256 2CTA kernel does not support pack_gqa yet.
            # pack_gqa=None means auto-enable for GQA/MQA (qhead_per_kvhead > 1)
            # Remove this when support is added.
            if d == 256 and IS_SM100:
                if pack_gqa is True:
                    continue
                if pack_gqa is None and mha_type != "mha":
                    continue
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
                or (IS_SM100 and d == 256 and dv == 256)
            )
            and not has_learnable_sink
            and softcap == 0.0 # TODO: support softcap != 0.0 in varlen bwd
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


# @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
# @pytest.mark.parametrize("dtype", [torch.float8_e4m3fn])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
# @pytest.mark.parametrize("mha_type", ["mha"])
@pytest.mark.parametrize("has_learnable_sink", [False, True])
# @pytest.mark.parametrize("has_learnable_sink", [False])
# @pytest.mark.parametrize("new_kv", [False, True])
@pytest.mark.parametrize("new_kv", [False])
@pytest.mark.parametrize("local", [False, True])
# @pytest.mark.parametrize("local", [False])
@pytest.mark.parametrize("causal", [False, True])
# @pytest.mark.parametrize("causal", [True])
# @pytest.mark.parametrize("seqlen_new_eq_seqlen_q", [True, False])
@pytest.mark.parametrize("seqlen_new_eq_seqlen_q", [False])
# @pytest.mark.parametrize("has_rotary_seqlens", [False, True])
@pytest.mark.parametrize("has_rotary_seqlens", [False])
# @pytest.mark.parametrize("rotary_interleaved", [False, True])
@pytest.mark.parametrize("rotary_interleaved", [True])
# @pytest.mark.parametrize("rotary_fraction", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("rotary_fraction", [0.0])
@pytest.mark.parametrize("page_size", [None] + ([1, 4, 128]))
# @pytest.mark.parametrize("page_size", [None, 128])
# @pytest.mark.parametrize("page_size", [128])
# @pytest.mark.parametrize("has_leftpad", [False, True])
@pytest.mark.parametrize("has_leftpad", [False])
# @pytest.mark.parametrize("has_batch_idx", [False, True])
@pytest.mark.parametrize("has_batch_idx", [False])
@pytest.mark.parametrize("varlen_q", [False, True])
# @pytest.mark.parametrize("varlen_q", [False])
# @pytest.mark.parametrize("d", [32, 59, 64, 80, 128, 256])
# @pytest.mark.parametrize("d", [32, 64, 96, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize('d', [32, 40, 64, 80, 96, 128, 160, 192])
# @pytest.mark.parametrize('d', [56, 80])
# @pytest.mark.parametrize("d", [128])
@pytest.mark.parametrize("d", [64, 128])
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
        # # (1, 128 * 1024),
        # # (16, 128 * 1024),
        # (128, 128),
        # (256, 512),  # To test appending KV with more than 1 block
        # (2048, 3577),  # Enough tile to test persistent scheduler
    ],
)
# @pytest.mark.parametrize('seqlen_q,seqlen_k', [(256, 128)])
@retry_on_oom
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
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
    has_learnable_sink,
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
    seed = 0
    random.seed(seed)
    torch.random.manual_seed(seed)
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
    # dv_vals = [128, d] if d > 128 and d <= 192 else ([256, 512, d] if d <= 64 else [d])
    dv_vals = [d]
    if dtype == torch.float8_e4m3fn:
        dv_vals = [d]
    # attention_chunk_vals = [torch.randint(1, seqlen_k * 2, (1,)).item(), 0] if (causal or local) else [0]
    attention_chunk_vals = [0]
    for dv, attention_chunk in itertools.product(dv_vals, attention_chunk_vals):
        # has_qv = d == 64 and dv >= 256
        has_qv = False
        q = (
            torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype_ref)
            .to(dtype)
            .to(dtype_ref)
        )
        if has_qv:
            qv = (
                torch.randn(
                    batch_size, seqlen_q, nheads, dv, device=device, dtype=dtype_ref
                )
                .to(dtype)
                .to(dtype_ref)
            )
        else:
            qv = None
        if varlen_q:
            query_padding_mask = generate_random_padding_mask(
                seqlen_q, batch_size, device, mode="random"
            )
            q_unpad, indices_q, cu_seqlens_q, max_seqlen_q, *rest = unpad_input(
                q, query_padding_mask
            )
            output_pad_fn = lambda output_unpad: pad_input(
                output_unpad, indices_q, batch_size, seqlen_q
            )
            qv_unpad = (
                rearrange(qv, "b s ... -> (b s) ...")[indices_q] if has_qv else None
            )
        else:
            query_padding_mask = None
            q_unpad = q
            qv_unpad = qv
            cu_seqlens_q, max_seqlen_q = None, None
        # Put window_size after QKV randn so that window_size changes from test to test
        window_size = (
            (None, None) if not local else tuple(random.randrange(0, seqlen_k) for _ in range(2))
        )
        if has_learnable_sink:
            learnable_sink = torch.randn(nheads, dtype=torch.bfloat16, device=device)
        else:
            learnable_sink = None

        seqlen_new = (
            seqlen_q
            if seqlen_new_eq_seqlen_q
            else random.randrange(1, seqlen_q + 1)
        )
        cu_seqlens_k_new = None
        key_new_padding_mask = None
        if new_kv:
            k = (
                torch.randn(
                    batch_size, seqlen_new, nheads_k, d, device=device, dtype=dtype_ref
                )
                .to(dtype)
                .to(dtype_ref)
            )
            v = (
                torch.randn(
                    batch_size, seqlen_new, nheads_k, dv, device=device, dtype=dtype_ref
                )
                .to(dtype)
                .to(dtype_ref)
            )
            if varlen_q:  # k & v are also varlen
                key_new_padding_mask = generate_random_padding_mask(
                    seqlen_new, batch_size, device, mode="random"
                )
                k_unpad, indices_k, cu_seqlens_k_new, *rest = unpad_input(
                    k, key_new_padding_mask
                )
                v_unpad, *rest = unpad_input(v, key_new_padding_mask)
            else:
                k_unpad, v_unpad = k, v
        else:
            k, v, k_unpad, v_unpad = None, None, None, None
        if page_size is None:
            k_cache = (
                torch.randn(
                    batch_size_cache,
                    seqlen_k,
                    nheads_k,
                    d,
                    device=device,
                    dtype=dtype_ref,
                )
                .to(dtype)
                .to(dtype_ref)
            )
            v_cache = (
                torch.randn(
                    batch_size_cache,
                    seqlen_k,
                    nheads_k,
                    dv,
                    device=device,
                    dtype=dtype_ref,
                )
                .to(dtype)
                .to(dtype_ref)
            )
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
                seqlen_k,
                page_size,
                batch_size_cache,
                nheads_k,
                d,
                dv,
                device,
                dtype,
                dtype_ref,
            )
        if not is_fake_mode():
            cache_seqlens = torch.randint(
                0 if new_kv else 1,
                # If we don't use seqlen_q in the case of causal and rotary, cos/sin won't be long enough
                (
                    (
                        seqlen_k
                        - (seqlen_q if (causal or local) and rotary_dim > 1 else seqlen_new)
                        + 1
                    )
                    if new_kv
                    else (seqlen_k + 1)
                ),
                (batch_size,),
                dtype=torch.int32,
                device=device,
            )
        else:
            cache_seqlens = torch.ones(
                batch_size,
                dtype=torch.int32,
                device=device,
            )
        if has_leftpad:
            if not is_fake_mode():
                cache_leftpad = torch.cat(
                    [
                        torch.randint(
                            0,
                            cache_seqlens[i].item(),
                            (1,),
                            dtype=torch.int32,
                            device=device,
                        )
                        if cache_seqlens[i].item() > 0
                        else torch.zeros(1, dtype=torch.int32, device=device)
                        for i in range(batch_size)
                    ]
                )
            else:
                cache_leftpad = torch.zeros(batch_size, dtype=torch.int32, device=device)
        else:
            cache_leftpad = None
        if has_batch_idx:
            if not is_fake_mode():
                cache_batch_idx = torch.randperm(
                    batch_size_cache, dtype=torch.int32, device=device
                )[:batch_size]
            else:
                cache_batch_idx = torch.arange(
                    batch_size, dtype=torch.int32, device=device
                )
        else:
            cache_batch_idx = None
        arange = rearrange(torch.arange(seqlen_k, device=device), "s -> 1 s")
        cache_seqlens_expanded = rearrange(cache_seqlens, "b -> b 1")
        if not new_kv:
            key_padding_mask = arange < cache_seqlens_expanded
        else:
            k_new_seqlens = (
                key_new_padding_mask.sum(-1, keepdims=True) if varlen_q else seqlen_new
            )
            key_padding_mask = arange < cache_seqlens_expanded + k_new_seqlens
        if has_leftpad:
            key_padding_mask = torch.logical_and(
                key_padding_mask,
                arange >= cache_leftpad.unsqueeze(-1).expand(-1, seqlen_k),
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
                    q,
                    cos,
                    sin,
                    seqlen_offsets=rotary_seqlens,
                    interleaved=rotary_interleaved,
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
                k,
                cos,
                sin,
                seqlen_offsets=rotary_seqlens,
                interleaved=rotary_interleaved,
            )
        else:
            cos, sin = None, None
            q_ro, k_ro = q, k
        # k_cache[:, 64:] = -1
        k_cache_ref = (
            k_cache if not has_batch_idx else k_cache[cache_batch_idx]
        ).clone()
        v_cache_ref = (
            v_cache if not has_batch_idx else v_cache[cache_batch_idx]
        ).clone()
        if new_kv:
            update_mask = torch.logical_and(
                cache_seqlens_expanded <= arange,
                arange < cache_seqlens_expanded + k_new_seqlens,
            )
            k_to_update = rearrange(k_ro, "b s ... -> (b s) ...")
            v_to_update = rearrange(v, "b s ... -> (b s) ...")
            if varlen_q:
                k_to_update = k_to_update[indices_k]
                v_to_update = v_to_update[indices_k]
            k_cache_ref[update_mask] = k_to_update
            v_cache_ref[update_mask] = v_to_update
        k_cache_rep = repeat(
            k_cache_ref, "b s h d -> b s (h g) d", g=nheads // nheads_k
        )
        v_cache_rep = repeat(
            v_cache_ref, "b s h d -> b s (h g) d", g=nheads // nheads_k
        )
        out_ref, _ = attention_ref(
            q_ro,
            k_cache_rep,
            v_cache_rep,
            query_padding_mask,
            key_padding_mask,
            causal=causal,
            qv=qv,
            window_size=window_size,
            learnable_sink=learnable_sink,
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
            learnable_sink=learnable_sink,
            attention_chunk=attention_chunk,
            upcast=False,
            reorder_ops=True,
            key_leftpad=cache_leftpad,
            intermediate_dtype=dtype if dtype == torch.float8_e4m3fn else None,
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
        # num_splits_vals = [1, 0]
        # SplitKV is not supported for hdim >= 192
        num_splits_vals = [1, 3] if d < 192 and not DISABLE_SPLIT else [1]
        # precompute_metadata_vals = [False, True]
        precompute_metadata_vals = [False]
        for num_splits, precompute_metadata in itertools.product(
            num_splits_vals, precompute_metadata_vals
        ):
            # SplitKV not supported on SM90 - skip this iteration
            if IS_SM90 and num_splits > 1:
                continue
            # if precompute_metadata:
            #     scheduler_metadata = get_scheduler_metadata(
            #         batch_size, max_seqlen_q if varlen_q else seqlen_q, seqlen_k, nheads, nheads_k, d,
            #         cache_seqlens, q.dtype, headdim_v=dv, cu_seqlens_q=cu_seqlens_q,
            #         cu_seqlens_k_new=cu_seqlens_k_new, cache_leftpad=cache_leftpad,
            #         max_seqlen_k_new=seqlen_new, page_size=page_size,
            #         causal=causal, window_size=window_size, attention_chunk=attention_chunk,
            #         num_splits=num_splits
            #     )
            # else:
            #     scheduler_metadata = None
            scheduler_metadata = None
            # Repeat to test metadata reuse
            for _ in range(1 if not precompute_metadata else 2):
                if page_size is None:
                    k_cache.copy_(k_cache_saved)
                    v_cache.copy_(v_cache_saved)
                else:
                    k_cache_paged.copy_(k_cache_saved)
                    v_cache_paged.copy_(v_cache_saved)
                # out, lse, *rest = flash_attn_with_kvcache(
                out, lse, *rest = flash_attn_varlen_func(
                    q if not varlen_q else q_unpad,
                    k_cache if page_size is None else k_cache_paged,
                    v_cache if page_size is None else v_cache_paged,
                    # k if not new_kv or not varlen_q else k_unpad,
                    # v if not new_kv or not varlen_q else v_unpad,
                    # qv=qv if not varlen_q else qv_unpad,
                    # rotary_cos=cos,
                    # rotary_sin=sin,
                    seqused_k=cache_seqlens,
                    # cache_batch_idx=cache_batch_idx,
                    # cache_leftpad=cache_leftpad,
                    page_table=page_table,
                    cu_seqlens_q=cu_seqlens_q,
                    # cu_seqlens_k_new=cu_seqlens_k_new,
                    # rotary_seqlens=rotary_seqlens,
                    causal=causal,
                    window_size=window_size,
                    learnable_sink=learnable_sink,
                    # attention_chunk=attention_chunk,
                    # rotary_interleaved=rotary_interleaved,
                    # scheduler_metadata=scheduler_metadata,
                    num_splits=num_splits,
                    # return_softmax_lse=True
                )
                if varlen_q:
                    out = output_pad_fn(out)
                if is_fake_mode():
                    # no more flash_attn cutedsl calls for the rest of the loop
                    # skip data-dependent postprocessing
                    continue
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
                            k_cache.to(dtype_ref)
                            if not has_batch_idx
                            else k_cache.to(dtype_ref)[cache_batch_idx]
                        )
                        v_cache_select = (
                            v_cache.to(dtype_ref)
                            if not has_batch_idx
                            else v_cache.to(dtype_ref)[cache_batch_idx]
                        )
                    else:
                        k_cache_select = rearrange(
                            k_cache_paged.to(dtype_ref)[
                                (
                                    page_table
                                    if not has_batch_idx
                                    else page_table[cache_batch_idx]
                                ).flatten()
                            ],
                            "(b nblocks) block_size ... -> b (nblocks block_size) ...",
                            b=batch_size,
                        )[:, :seqlen_k].to(dtype_ref)
                        v_cache_select = rearrange(
                            v_cache_paged.to(dtype_ref)[
                                (
                                    page_table
                                    if not has_batch_idx
                                    else page_table[cache_batch_idx]
                                ).flatten()
                            ],
                            "(b nblocks) block_size ... -> b (nblocks block_size) ...",
                            b=batch_size,
                        )[:, :seqlen_k].to(dtype_ref)
                    k_cache_ref = k_cache_ref.to(dtype).to(dtype_ref)
                    v_cache_ref = v_cache_ref.to(dtype).to(dtype_ref)
                    if dtype is not torch.float8_e4m3fn:
                        assert torch.equal(v_cache_select, v_cache_ref)
                    else:
                        assert torch.allclose(
                            v_cache_select, v_cache_ref, rtol=1e-3, atol=1e-3
                        )
                    # breakpoint()
                    # if rotary_dim == 0 and dtype is not torch.float8_e4m3fn:
                    if rotary_dim == 0:
                        assert torch.equal(k_cache_select, k_cache_ref)
                    else:
                        # if not torch.allclose(k_cache_select, k_cache_ref, rtol=1e-3, atol=1e-3):
                        #     breakpoint()
                        if dtype is not torch.float8_e4m3fn:
                            assert torch.allclose(
                                k_cache_select, k_cache_ref, rtol=1e-3, atol=1e-3
                            )
                        else:
                            assert torch.allclose(
                                k_cache_select, k_cache_ref, rtol=1e-1, atol=1e-1
                            )
                mult = 4 if dtype == torch.float8_e4m3fn else 2
                assert (out - out_ref).abs().max().item() <= mult * (
                    out_pt - out_ref
                ).abs().max().item() + 1e-5
                mult_mean = 3 if dtype == torch.float8_e4m3fn else 1.5
                assert (out - out_ref).abs().mean().item() <= mult_mean * (
                    out_pt - out_ref
                ).abs().mean().item()


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("seqlen_q,seqlen_k", [(128, 128), (256, 256)])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_bwd_preallocated_outputs(seqlen_q, seqlen_k, d, causal, dtype):
    from flash_attn.cute.interface import _flash_attn_fwd, _flash_attn_bwd

    device = "cuda"
    torch.random.manual_seed(42)
    batch_size = 2
    nheads = 4

    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=True)

    out, lse = _flash_attn_fwd(q, k, v, causal=causal, return_lse=True)
    dout = torch.randn_like(out)

    dq_ref, dk_ref, dv_ref = _flash_attn_bwd(q, k, v, out, dout, lse, causal=causal)

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    dq_out, dk_out, dv_out = _flash_attn_bwd(
        q, k, v, out, dout, lse, causal=causal, dq=dq, dk=dk, dv=dv
    )

    if is_fake_mode():
        return
    assert dq_out is dq
    assert dk_out is dk
    assert dv_out is dv
    assert torch.allclose(dq, dq_ref, atol=1e-5, rtol=1e-5)
    assert torch.allclose(dk, dk_ref, atol=1e-5, rtol=1e-5)
    assert torch.allclose(dv, dv_ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("seqlen_q,seqlen_k", [(128, 128), (256, 256)])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_lse_grad(seqlen_q, seqlen_k, d, causal, dtype):
    """Test that gradient flows through the returned LSE tensor."""
    device = "cuda"
    torch.random.manual_seed(42)
    batch_size = 2
    nheads = 4

    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=True)

    out, lse = flash_attn_func(q, k, v, causal=causal, return_lse=True)

    if is_fake_mode():
        return

    assert lse is not None
    assert lse.requires_grad

    # Compute loss = sum(out * g) + sum(lse * dlse_weight) to test gradient flows through both
    g = torch.randn_like(out)
    dlse_weight = torch.randn_like(lse)
    loss = (out * g).sum() + (lse * dlse_weight).sum()
    dq, dk, dv = torch.autograd.grad(loss, (q, k, v))

    # Compare against reference: manually compute what the gradients should be
    # Reference: standard attention in float
    q_ref = q.detach().float().requires_grad_()
    k_ref = k.detach().float().requires_grad_()
    v_ref = v.detach().float().requires_grad_()
    # (batch, seqlen_q, nheads, d) -> (batch, nheads, seqlen_q, d)
    qk = torch.einsum("bshd,bthd->bhst", q_ref, k_ref) / (d ** 0.5)
    if causal:
        mask = torch.triu(torch.ones(seqlen_q, seqlen_k, device=device, dtype=torch.bool), diagonal=seqlen_k - seqlen_q + 1)
        qk = qk.masked_fill(mask, float("-inf"))
    lse_ref = torch.logsumexp(qk, dim=-1)  # (batch, nheads, seqlen_q)
    p = torch.softmax(qk, dim=-1)
    # v_ref: (batch, seqlen_k, nheads, d)
    out_ref = torch.einsum("bhst,bthd->bshd", p, v_ref)
    loss_ref = (out_ref * g.float()).sum() + (lse_ref * dlse_weight.float()).sum()
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(loss_ref, (q_ref, k_ref, v_ref))

    # Use relaxed tolerances since flash_attn operates in bf16 while reference is float32.
    # The reference is also not a perfect bf16 simulation (it doesn't reorder ops), so
    # we use a generous tolerance.
    print(f"dQ max diff: {(dq.float() - dq_ref).abs().max().item()}")
    print(f"dK max diff: {(dk.float() - dk_ref).abs().max().item()}")
    print(f"dV max diff: {(dv.float() - dv_ref).abs().max().item()}")
    # Absolute tolerance: bf16 has ~0.004-0.02 error for these sizes
    atol = 0.02
    assert (dq.float() - dq_ref).abs().max().item() <= atol, f"dQ error too large"
    assert (dk.float() - dk_ref).abs().max().item() <= atol, f"dK error too large"
    assert (dv.float() - dv_ref).abs().max().item() <= atol, f"dV error too large"

    # Also test: gradient with only dLSE (no dO)
    out2, lse2 = flash_attn_func(q, k, v, causal=causal, return_lse=True)
    loss_lse_only = (lse2 * dlse_weight).sum()
    dq2, dk2, dv2 = torch.autograd.grad(loss_lse_only, (q, k, v))

    q_ref2 = q.detach().float().requires_grad_()
    k_ref2 = k.detach().float().requires_grad_()
    qk2 = torch.einsum("bshd,bthd->bhst", q_ref2, k_ref2) / (d ** 0.5)
    if causal:
        qk2 = qk2.masked_fill(mask, float("-inf"))
    lse_ref2 = torch.logsumexp(qk2, dim=-1)
    loss_ref2 = (lse_ref2 * dlse_weight.float()).sum()
    dq_ref2, dk_ref2 = torch.autograd.grad(loss_ref2, (q_ref2, k_ref2))

    print(f"LSE-only dQ max diff: {(dq2.float() - dq_ref2).abs().max().item()}")
    print(f"LSE-only dK max diff: {(dk2.float() - dk_ref2).abs().max().item()}")
    # dV should be zero when only LSE gradient flows (LSE doesn't depend on V)
    print(f"LSE-only dV max: {dv2.abs().max().item()}")
    assert dv2.abs().max().item() == 0.0, "dV should be zero when loss depends only on LSE"


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("d", [128])
@pytest.mark.parametrize("seqlen_q,seqlen_k", [(128, 128)])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_lse_grad_unused(seqlen_q, seqlen_k, d, causal, dtype):
    """Test return_lse=True when LSE is returned but not used in the loss.

    With set_materialize_grads(False), dlse should be None (not a zero tensor),
    so no extra zeroing kernel is launched. Gradients should match the standard
    backward (without return_lse).
    """
    device = "cuda"
    torch.random.manual_seed(42)
    batch_size = 2
    nheads = 4

    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=True)
    g = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)

    # Case 1: return_lse=False (standard path, lse marked non-differentiable)
    out1, lse1 = flash_attn_func(q, k, v, causal=causal, return_lse=False)
    if is_fake_mode():
        return
    dq1, dk1, dv1 = torch.autograd.grad(out1, (q, k, v), g)

    # Case 2: return_lse=True but lse NOT used in loss (dlse should be None)
    out2, lse2 = flash_attn_func(q, k, v, causal=causal, return_lse=True)
    dq2, dk2, dv2 = torch.autograd.grad(out2, (q, k, v), g)

    # Case 3: return_lse=True and lse IS used in loss
    out3, lse3 = flash_attn_func(q, k, v, causal=causal, return_lse=True)
    dlse_weight = torch.randn_like(lse3)
    loss3 = (out3 * g).sum() + (lse3 * dlse_weight).sum()
    dq3, dk3, dv3 = torch.autograd.grad(loss3, (q, k, v))

    # Cases 1 and 2 should produce identical gradients
    assert torch.equal(dq1, dq2), "dQ should be identical when LSE is unused"
    assert torch.equal(dk1, dk2), "dK should be identical when LSE is unused"
    assert torch.equal(dv1, dv2), "dV should be identical when LSE is unused"

    # Case 3 should differ from case 1 (LSE gradient adds extra contribution to dQ, dK)
    assert not torch.equal(dq1, dq3), "dQ should differ when LSE gradient is included"
    assert not torch.equal(dk1, dk3), "dK should differ when LSE gradient is included"
    # dV should be the same since LSE doesn't depend on V
    assert torch.equal(dv1, dv3), "dV should be identical since LSE doesn't depend on V"

    print("Case 1 vs 2 (unused LSE): dQ diff =", (dq1 - dq2).abs().max().item())
    print("Case 1 vs 3 (used LSE):   dQ diff =", (dq1 - dq3).abs().max().item())
    print("Case 1 vs 3 (used LSE):   dK diff =", (dk1 - dk3).abs().max().item())
    print("Case 1 vs 3 (used LSE):   dV diff =", (dv1 - dv3).abs().max().item())


def _generate_block_kvcache(
    seqlen_k, page_size, batch_size, nheads_k, d, dv, device, dtype, dtype_ref
):
    num_blocks = math.ceil(seqlen_k / page_size) * batch_size * 3
    k_cache_paged = (
        torch.randn(num_blocks, page_size, nheads_k, d, device=device, dtype=dtype_ref)
        .to(dtype)
        .to(dtype_ref)
    )
    v_cache_paged = (
        torch.randn(num_blocks, page_size, nheads_k, dv, device=device, dtype=dtype_ref)
        .to(dtype)
        .to(dtype_ref)
    )
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


@pytest.mark.parametrize("page_size", [16, 64, 256])
@pytest.mark.parametrize("seqlen_q", [64, 128, 256])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_paged_deepseek(seqlen_q, page_size):
    """Regression test: paged non-TMA with DeepSeek MLA shape (d=192, dv=128).
    seqlen_q<=128 triggers q_stage=1, seqlen_q>128 triggers q_stage=2.
    """
    if IS_SM90:
        pytest.skip("paged KV not supported on SM90")
    device = "cuda"
    dtype = torch.bfloat16
    d, dv = 192, 128
    nheads = 16
    nheads_kv = 16

    torch.random.manual_seed(0)
    q = torch.randn(seqlen_q, nheads, d, device=device, dtype=dtype)
    k = torch.randn(seqlen_q, nheads_kv, d, device=device, dtype=dtype)
    v = torch.randn(seqlen_q, nheads_kv, dv, device=device, dtype=dtype)
    cu_seqlens = torch.tensor([0, seqlen_q], dtype=torch.int32, device=device)

    # Non-paged reference
    out_ref, _ = flash_attn_varlen_func(
        q, k, v, cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
        max_seqlen_q=seqlen_q, max_seqlen_k=seqlen_q, causal=True,
    )

    # Paged
    num_pages = (seqlen_q + page_size - 1) // page_size
    k_cache_paged = torch.zeros(num_pages, page_size, nheads_kv, d, device=device, dtype=dtype)
    v_cache_paged = torch.zeros(num_pages, page_size, nheads_kv, dv, device=device, dtype=dtype)
    for i in range(seqlen_q):
        k_cache_paged[i // page_size, i % page_size] = k[i]
        v_cache_paged[i // page_size, i % page_size] = v[i]
    page_table = torch.arange(num_pages, dtype=torch.int32, device=device).unsqueeze(0)
    cache_seqlens = torch.tensor([seqlen_q], dtype=torch.int32, device=device)

    out, _ = flash_attn_varlen_func(
        q, k_cache_paged, v_cache_paged,
        cu_seqlens_q=cu_seqlens, cu_seqlens_k=None,
        max_seqlen_q=seqlen_q, max_seqlen_k=None,
        seqused_k=cache_seqlens, page_table=page_table, causal=True,
    )

    if is_fake_mode():
        return

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    assert torch.equal(out, out_ref)


@pytest.mark.parametrize("seqlen_q", [128, 512, 2048])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_paged_hd256_sm100_tma(seqlen_q):
    """TMA paged KV in the SM100 hd256 2CTA forward kernel.

    Verifies paged KV (page_table + TMA) matches the non-paged varlen reference
    and is deterministic across runs. page_size must equal tile_n=128.
    """
    if not IS_SM100:
        pytest.skip("SM100-specific paged hd256 test")
    device = "cuda"
    dtype = torch.bfloat16
    d = 256
    batch_size = 2
    nheads = 16
    nheads_kv = 16
    page_size = 128
    assert seqlen_q % page_size == 0

    torch.random.manual_seed(0)
    q = torch.randn(batch_size * seqlen_q, nheads, d, device=device, dtype=dtype)
    k = torch.randn(batch_size * seqlen_q, nheads_kv, d, device=device, dtype=dtype)
    v = torch.randn(batch_size * seqlen_q, nheads_kv, d, device=device, dtype=dtype)
    cu_seqlens_q = torch.arange(0, batch_size + 1, dtype=torch.int32, device=device) * seqlen_q
    cu_seqlens_k = cu_seqlens_q.clone()

    # Non-paged reference (varlen).
    out_ref, _ = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=seqlen_q, max_seqlen_k=seqlen_q,
    )

    # Repack into paged layout: (total_pages, page_size, nheads_kv, d).
    num_pages_per_seq = seqlen_q // page_size
    total_pages = batch_size * num_pages_per_seq
    k_paged = torch.zeros(total_pages, page_size, nheads_kv, d, device=device, dtype=dtype)
    v_paged = torch.zeros(total_pages, page_size, nheads_kv, d, device=device, dtype=dtype)
    for b in range(batch_size):
        for s in range(seqlen_q):
            pi = b * num_pages_per_seq + s // page_size
            po = s % page_size
            k_paged[pi, po] = k[b * seqlen_q + s]
            v_paged[pi, po] = v[b * seqlen_q + s]
    page_table = torch.arange(total_pages, dtype=torch.int32, device=device).reshape(
        batch_size, num_pages_per_seq
    )

    # Paged via hd256 2CTA TMA paged path — run twice for determinism.
    out_paged_0, _ = flash_attn_varlen_func(
        q, k_paged, v_paged,
        cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=None,
        max_seqlen_q=seqlen_q, max_seqlen_k=seqlen_q,
        page_table=page_table,
    )
    out_paged_1, _ = flash_attn_varlen_func(
        q, k_paged, v_paged,
        cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=None,
        max_seqlen_q=seqlen_q, max_seqlen_k=seqlen_q,
        page_table=page_table,
    )

    if is_fake_mode():
        return

    print(f"Paged vs non-paged max diff: {(out_paged_0 - out_ref).abs().max().item()}")
    print(f"Paged determinism diff: {(out_paged_1 - out_paged_0).abs().max().item()}")
    assert torch.allclose(out_paged_0, out_ref, atol=1e-3, rtol=1e-3), "Paged output does not match non-paged reference"
    assert torch.equal(out_paged_1, out_paged_0), "Paged output is not deterministic"


@pytest.mark.parametrize("nheads_kv", [2, 4, 8])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_paged_hd256_sm100_tma_gqa(nheads_kv):
    """TMA paged KV for SM100 hd256 2CTA with GQA (nheads_q > nheads_kv).

    Exercises the head_kv_coord derivation for qhead_per_kvhead > 1 — the MHA
    test passes by coincidence since modulo and integer division agree when
    qhead_per_kvhead == 1.
    """
    if not IS_SM100:
        pytest.skip("SM100-specific paged hd256 test")
    device = "cuda"
    dtype = torch.bfloat16
    d = 256
    batch_size = 2
    nheads = 16
    page_size = 128
    seqlen_q = 512
    assert nheads % nheads_kv == 0 and seqlen_q % page_size == 0

    torch.random.manual_seed(0)
    q = torch.randn(batch_size * seqlen_q, nheads, d, device=device, dtype=dtype)
    k = torch.randn(batch_size * seqlen_q, nheads_kv, d, device=device, dtype=dtype)
    v = torch.randn(batch_size * seqlen_q, nheads_kv, d, device=device, dtype=dtype)
    cu_seqlens_q = torch.arange(0, batch_size + 1, dtype=torch.int32, device=device) * seqlen_q
    cu_seqlens_k = cu_seqlens_q.clone()

    out_ref, _ = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=seqlen_q, max_seqlen_k=seqlen_q,
    )

    num_pages_per_seq = seqlen_q // page_size
    total_pages = batch_size * num_pages_per_seq
    k_paged = torch.zeros(total_pages, page_size, nheads_kv, d, device=device, dtype=dtype)
    v_paged = torch.zeros(total_pages, page_size, nheads_kv, d, device=device, dtype=dtype)
    for b in range(batch_size):
        for s in range(seqlen_q):
            pi = b * num_pages_per_seq + s // page_size
            po = s % page_size
            k_paged[pi, po] = k[b * seqlen_q + s]
            v_paged[pi, po] = v[b * seqlen_q + s]
    page_table = torch.arange(total_pages, dtype=torch.int32, device=device).reshape(
        batch_size, num_pages_per_seq
    )

    out_paged, _ = flash_attn_varlen_func(
        q, k_paged, v_paged,
        cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=None,
        max_seqlen_q=seqlen_q, max_seqlen_k=seqlen_q,
        page_table=page_table,
    )

    if is_fake_mode():
        return

    print(f"GQA nheads_kv={nheads_kv} paged vs non-paged max diff: {(out_paged - out_ref).abs().max().item()}")
    assert torch.allclose(out_paged, out_ref, atol=1e-3, rtol=1e-3), (
        f"Paged GQA output does not match non-paged reference (nheads_kv={nheads_kv})"
    )


@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_paged_hd256_sm100_tma_shuffled():
    """TMA paged KV for SM100 hd256 2CTA with a non-identity (shuffled) page_table.

    An identity page_table passes even if the kernel ignores it. This test
    shuffles physical pages so a kernel that bypasses page_table would silently
    read wrong data, proving the remapping path is exercised.
    """
    if not IS_SM100:
        pytest.skip("SM100-specific paged hd256 test")
    device = "cuda"
    dtype = torch.bfloat16
    d = 256
    batch_size = 2
    nheads = 16
    nheads_kv = 16
    page_size = 128
    seqlen_q = 512
    num_pages_per_seq = seqlen_q // page_size
    total_pages = batch_size * num_pages_per_seq

    torch.random.manual_seed(42)
    q = torch.randn(batch_size * seqlen_q, nheads, d, device=device, dtype=dtype)
    k = torch.randn(batch_size * seqlen_q, nheads_kv, d, device=device, dtype=dtype)
    v = torch.randn(batch_size * seqlen_q, nheads_kv, d, device=device, dtype=dtype)
    cu_seqlens_q = torch.arange(0, batch_size + 1, dtype=torch.int32, device=device) * seqlen_q
    cu_seqlens_k = cu_seqlens_q.clone()

    out_ref, _ = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=seqlen_q, max_seqlen_k=seqlen_q,
    )

    # Shuffle physical pages: reverse order within each batch item.
    # Build as Python list of ints to avoid .item() calls on FakeTensors during compilation.
    perm = [
        list(range((b + 1) * num_pages_per_seq - 1, b * num_pages_per_seq - 1, -1))
        for b in range(batch_size)
    ]
    page_table = torch.tensor(perm, dtype=torch.int32, device=device)

    k_paged = torch.zeros(total_pages, page_size, nheads_kv, d, device=device, dtype=dtype)
    v_paged = torch.zeros(total_pages, page_size, nheads_kv, d, device=device, dtype=dtype)
    for b in range(batch_size):
        for s in range(seqlen_q):
            phys = perm[b][s // page_size]  # Python int, safe in FakeTensorMode
            po = s % page_size
            k_paged[phys, po] = k[b * seqlen_q + s]
            v_paged[phys, po] = v[b * seqlen_q + s]

    out_paged, _ = flash_attn_varlen_func(
        q, k_paged, v_paged,
        cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=None,
        max_seqlen_q=seqlen_q, max_seqlen_k=seqlen_q,
        page_table=page_table,
    )

    if is_fake_mode():
        return

    print(f"Shuffled paged vs non-paged max diff: {(out_paged - out_ref).abs().max().item()}")
    assert torch.allclose(out_paged, out_ref, atol=1e-3, rtol=1e-3), (
        "Shuffled paged output does not match non-paged reference"
    )


@pytest.mark.parametrize("head_dim", [4, 148, 288])
def test_flash_attn_invalid_head_dim(head_dim):
    device = "cuda"
    dtype = torch.bfloat16
    batch_size, seqlen, nheads = 1, 64, 4

    q = torch.randn(batch_size, seqlen, nheads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, seqlen, nheads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, seqlen, nheads, head_dim, device=device, dtype=dtype)

    with pytest.raises(AssertionError, match=re.escape(f"(head_dim, head_dim_v)=({head_dim}, {head_dim}) is not supported on SM")):
        flash_attn_func(q, k, v)


# @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
# @pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
@pytest.mark.parametrize("mha_type", ["mqa"])
@pytest.mark.parametrize("has_learnable_sink", [False])
@pytest.mark.parametrize("softcap", [0.0])
@pytest.mark.parametrize("deterministic", [False])
# @pytest.mark.parametrize("local_enum", [0, 1])
@pytest.mark.parametrize("local_enum", [0])
@pytest.mark.parametrize("causal", [False, True])
# @pytest.mark.parametrize("causal", [False])
@pytest.mark.parametrize("d", [64])
@pytest.mark.parametrize("nheads", [16, 128])
@pytest.mark.parametrize("kv_sparsity", [False, True])
# @pytest.mark.parametrize("kv_sparsity", [True])
@pytest.mark.parametrize("gather_kv_length", [1024, 2048])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (1, 1),
        (3, 3),
        (64, 32),
        (64, 128),
        (128, 128),
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
        (2048, 2048),
        (1, 8192),
        (4096, 4096),
    ],
)
# @pytest.mark.parametrize('seqlen_q,seqlen_k', [(128, 128)])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_mla_absorbed(
    seqlen_q,
    seqlen_k,
    d,
    nheads,
    causal,
    local_enum,
    softcap,
    deterministic,
    has_learnable_sink,
    mha_type,
    dtype,
    kv_sparsity,
    gather_kv_length,
):
    has_qv = True
    if not IS_SM100:
        pytest.skip()
    if kv_sparsity and seqlen_k < gather_kv_length:
        seqlen_k += gather_kv_length
    local = local_enum > 0
    if local and causal:
        pytest.skip()
    if local:
        pytest.xfail("mla absorbed: local not supported yet")
    if kv_sparsity and nheads != 128:
        pytest.skip()
    device = "cuda"
    # set seed
    seed = 0
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    batch_size = 9 if seqlen_k <= 2048 else 2
    # batch_size = 2
    # nheads = 128
    nheads_kv = nheads if mha_type == "mha" else (8 if mha_type == "gqa" else 1)
    dtype_ref = torch.bfloat16 if dtype == torch.float8_e4m3fn else dtype
    dv_vals = [512]
    # attention_chunk_vals = [torch.randint(1, seqlen_k * 2, (1,)).item(), 0]
    attention_chunk_vals = [0]
    for dv, attention_chunk in itertools.product(dv_vals, attention_chunk_vals):
        q_ref = torch.randn(
            batch_size, seqlen_q, nheads, d, device=device, dtype=dtype_ref
        )
        if softcap > 0.0:
            # Ensure the values of qk are at least within softcap range.
            q_ref = q_ref * softcap / 4
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
        if kv_sparsity:
            gather_kv_indices = torch.rand(batch_size, seqlen_q, gather_kv_length, device=device).argsort(dim=-1).to(torch.int32)
        else:
            gather_kv_indices = None
        # Put window_size after QKV randn so that window_size changes from test to test
        window_size = (
            (None, None) if not local else tuple(random.randrange(0, seqlen_k) for _ in range(2))
        )
        if local_enum == 2:
            window_size = (None, -window_size[1])
        elif local_enum == 3:
            window_size = (-window_size[0], None)
        if local:
            print("window size = ", window_size)
        # window_size = (-1, -1) if not local else (16, 0)
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
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            window_size=window_size,
            attention_chunk=attention_chunk,
            learnable_sink=learnable_sink,
            softcap=softcap,
            gather_kv_indices=gather_kv_indices,
        )
        out_pt, attn_pt = attention_ref(
            q_ref,
            k_ref,
            v_ref,
            None,
            None,
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
            gather_kv_indices=gather_kv_indices,
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
        if not is_fake_mode():
            fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
            rtol = 2 if softcap == 0.0 else 3
            print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
            print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")
        num_splits_vals = [1]
        pack_gqa_vals = [True]
        for pack_gqa, num_splits in itertools.product(pack_gqa_vals, num_splits_vals):
            out, lse = flash_attn_func(
                q,
                k,
                v,
                qv=qv,
                gather_kv_indices=gather_kv_indices,
                causal=causal,
                # q_descale=q_descale, k_descale=k_descale, v_descale=v_descale,
                window_size=window_size,
                # attention_chunk=attention_chunk,
                softcap=softcap,
                learnable_sink=learnable_sink,
                pack_gqa=pack_gqa,
                num_splits=num_splits,
                deterministic=deterministic,
            )
            if is_fake_mode():
                # no more flash_attn cutedsl calls for the rest of the loop
                # skip data-dependent postprocessing
                continue
            print(f"Output max diff: {(out - out_ref).abs().max().item()}")
            print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
            # if not causal:
            #     print(f"LSE max diff: {(lse - lse_ref).abs().max().item()}")
            # breakpoint()

            # Check that FlashAttention's numerical error is at most twice the numerical error
            # of a Pytorch implementation.
            assert (out - out_ref).abs().max().item() <= rtol * (
                out_pt - out_ref
            ).abs().max().item() + fwd_atol
            assert not torch.isnan(lse).any(), "LSE contains NaN"

            repeats = 1000
            for iter in range(repeats):
                out2, lse2 = flash_attn_func(
                    q,
                    k,
                    v,
                    qv=qv,
                    gather_kv_indices=gather_kv_indices,
                    causal=causal,
                    # q_descale=q_descale, k_descale=k_descale, v_descale=v_descale,
                    window_size=window_size,
                    # attention_chunk=attention_chunk,
                    softcap=softcap,
                    learnable_sink=learnable_sink,
                    pack_gqa=pack_gqa,
                    num_splits=num_splits,
                    deterministic=deterministic,
                )
                # print(f"out max: {out.abs().max().item()}, {iter=}")
                # print(f"out vs out2 max diff: {(out - out2).abs().max().item()}, {iter=}")
                # print(f"out vs out2 mean diff: {(out - out2).abs().mean().item()}, {iter=}")
                assert torch.equal(out, out2), f"non-deterministic with max diff = {(out - out2).abs().max().item()} on {iter=}"


# @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
# @pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
@pytest.mark.parametrize("mha_type", ["mqa"])
@pytest.mark.parametrize("has_learnable_sink", [False])
@pytest.mark.parametrize("deterministic", [False])
@pytest.mark.parametrize("softcap", [0.0])
@pytest.mark.parametrize("local_enum", [0])
@pytest.mark.parametrize("causal", [False, True])
# @pytest.mark.parametrize("causal", [False])
# @pytest.mark.parametrize("add_unused_qkv", [False, True])
@pytest.mark.parametrize("add_unused_qkv", [False])
@pytest.mark.parametrize("kv_sparsity", [False, True])
# @pytest.mark.parametrize("kv_sparsity", [False])
@pytest.mark.parametrize("gather_kv_length", [1024, 2048])
@pytest.mark.parametrize("d", [64])
@pytest.mark.parametrize("nheads", [16, 128])
# @pytest.mark.parametrize("nheads", [128])
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
@pytest.mark.parametrize("varlen_mode", ["random", "full"])
# @pytest.mark.parametrize("varlen_mode", ["random"])
@pytest.mark.parametrize(
    "zero_lengths_q, zero_lengths_k",
    [
        (False, False),
        # (True, False),
    ],
)
@pytest.mark.parametrize(
    "unpad_q, unpad_kv",
    [
        (True, True),
        (True, False),
        (False, False),
        (False, True),
    ],
)
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_mla_absorbed_varlen(
    seqlen_q,
    seqlen_k,
    d,
    nheads,
    add_unused_qkv,
    causal,
    local_enum,
    softcap,
    deterministic,
    has_learnable_sink,
    mha_type,
    dtype,
    varlen_mode,
    zero_lengths_q,
    zero_lengths_k,
    unpad_q,
    unpad_kv,
    kv_sparsity,
    gather_kv_length,
):
    has_qv = True
    if not IS_SM100:
        pytest.skip()
    if kv_sparsity and seqlen_k < gather_kv_length:
        seqlen_k += gather_kv_length
    local = local_enum > 0
    if local and causal:
        pytest.skip()
    if has_qv and local:
        pytest.xfail("has_qv: local not supported yet")
    if kv_sparsity and nheads != 128:
        pytest.skip()
    seqlen_q_og = seqlen_q
    seqlen_k_og = seqlen_k
    if (
        causal or local
    ):  # Right now reference only supports causal attention with seqlen_k == seqlen_q
        seqlen_q = max(seqlen_q_og, seqlen_k_og)
        seqlen_k = max(seqlen_q_og, seqlen_k_og)
    device = "cuda"
    # set seed
    seed = seqlen_q + seqlen_k + d + int(causal) * 2 + int(local)
    random.seed(seed)
    torch.random.manual_seed(seed)
    batch_size = 7 if seqlen_q <= 512 else 3
    nheads_kv = nheads if mha_type == "mha" else (8 if mha_type == "gqa" else 1)
    dtype_ref = torch.bfloat16 if dtype == torch.float8_e4m3fn else dtype
    dv_vals = [512]
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
        if kv_sparsity:
            gather_kv_indices = torch.rand(batch_size, seqlen_q, gather_kv_length, device=device).argsort(dim=-1).to(torch.int32)
        else:
            gather_kv_indices = None
            
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
            min_seqlen=gather_kv_length if kv_sparsity else None,
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
        # unpad gather_kv_indices
        if kv_sparsity:
            _, indices_q, _, _, _ = unpad_input(
                q, query_padding_mask, query_unused_mask
            )
            gather_kv_indices_unpad = rearrange(gather_kv_indices, "b s ... -> (b s) ...")[indices_q]
        else:
            gather_kv_indices_unpad = None
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
            gather_kv_indices=gather_kv_indices,
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
            gather_kv_indices=gather_kv_indices,
        )

        if not is_fake_mode():
            print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
            print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")

            if query_unused_mask is not None:
                q_zero_masking = rearrange(query_unused_mask, "b s -> b s 1 1")

            # Numerical error if we just do any arithmetic on out_ref
            fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
            rtol = 2 if softcap == 0.0 else 3

        pack_gqa_vals = [True]
        num_splits_vals = [1]
        for pack_gqa, num_splits in itertools.product(pack_gqa_vals, num_splits_vals):
            # SplitKV not supported on SM90 - skip this iteration
            if IS_SM90 and num_splits > 1:
                continue
            out_unpad, lse = flash_attn_varlen_func(
                q_unpad if unpad_q else q,
                k_unpad if unpad_kv else k,
                v_unpad if unpad_kv else v,
                qv_unpad if unpad_q else qv,
                cu_seqlens_q=cu_seqlens_q if unpad_q else None,
                cu_seqlens_k=cu_seqlens_k if unpad_kv else None,
                max_seqlen_q=seqlen_q,
                max_seqlen_k=seqlen_k,
                min_seqlen_k=gather_kv_length if kv_sparsity else None,
                seqused_q=seqused_q if not unpad_q else None,
                seqused_k=seqused_k if not unpad_kv else None,
                causal=causal,
                window_size=window_size,
                learnable_sink=learnable_sink,
                softcap=softcap,
                num_splits=num_splits,
                pack_gqa=pack_gqa,
                deterministic=deterministic,
                gather_kv_indices=gather_kv_indices_unpad if unpad_q else gather_kv_indices,
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
            # LSE sanity: only valid positions (packed unpad path; padded path
            # can legitimately contain uninit tail beyond seqused_q).
            if unpad_q:
                assert not torch.isnan(lse).any(), "LSE contains NaN"

            repeats = 1000
            for iter in range(repeats):
                out_unpad2, lse = flash_attn_varlen_func(
                    q_unpad if unpad_q else q,
                    k_unpad if unpad_kv else k,
                    v_unpad if unpad_kv else v,
                    qv_unpad if unpad_q else qv,
                    cu_seqlens_q=cu_seqlens_q if unpad_q else None,
                    cu_seqlens_k=cu_seqlens_k if unpad_kv else None,
                    max_seqlen_q=seqlen_q,
                    max_seqlen_k=seqlen_k,
                    min_seqlen_k=gather_kv_length if kv_sparsity else None,
                    seqused_q=seqused_q if not unpad_q else None,
                    seqused_k=seqused_k if not unpad_kv else None,
                    causal=causal,
                    window_size=window_size,
                    learnable_sink=learnable_sink,
                    softcap=softcap,
                    num_splits=num_splits,
                    pack_gqa=pack_gqa,
                    deterministic=deterministic,
                    gather_kv_indices=gather_kv_indices_unpad if unpad_q else gather_kv_indices,
                )
                out2 = output_pad_fn(out_unpad2) if unpad_q else out_unpad2
                if query_unused_mask is not None:
                    out2.masked_fill_(q_zero_masking, 0.0)
                # When unpad_q=False with seqused_q, the kernel doesn't write positions
                # beyond seqused_q, so those contain uninitialized values. Mask them out
                # before comparing.
                if not unpad_q and seqused_q is not None:
                    seqused_mask = torch.arange(seqlen_q, device=device)[None, :] < seqused_q[:, None]
                    seqused_mask = rearrange(seqused_mask, "b s -> b s 1 1")
                    out2.masked_fill_(~seqused_mask, 0.0)
                # print(f"out2 max: {out2.abs().max().item()}, {iter=}")
                # print(f"out vs out2 max diff: {(out_cmp - out2).abs().max().item()}, {iter=}")
                # print(f"out vs out2 mean diff: {(out_cmp - out2).abs().mean().item()}, {iter=}")
                assert torch.equal(out_cmp, out2), f"non-deterministic with max diff = {(out_cmp - out2).abs().max().item()} on {iter=}"


# ---------------------------------------------------------------------------
# Regression test: seqlen_k=0 must not crash (CUDA graph padding scenario)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("d", [128, 192])
@pytest.mark.parametrize("seqlen_q", [1, 64, 128, 256])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_seqlen_k_zero(seqlen_q, d, causal):
    """K/V with physical seqlen dim == 0 must not crash.

    seqlen_k == 0 violates two downstream invariants, producing two
    different crashes depending on the mask:

      causal=False -> TMA descriptor over a 0-length K tensor goes OOB
                      on first tile load -> PTX IllegalInstruction.

      causal=True  -> SingleTileLPTScheduler's L2-swizzle heuristic in
                      tile_scheduler.py evaluates
                          size_l2 // (seqlen_k * (d + d_v) * elem_size)
                      -> host SIGFPE before the kernel launches.

    Varlen paths (cu_seqlens_k / seqused_k with K physical seqlen > 0)
    are not exercised here: per-batch empty slots are already handled
    by the kernel's fake-iteration path and do not hit either invariant.
    """
    if IS_SM90:
        pytest.skip("SM90 uses a different kernel path")

    device = "cuda"
    dtype = torch.bfloat16
    dv = 128 if d == 192 else d
    batch_size = 4
    nheads = 16
    nheads_kv = 16

    torch.manual_seed(0)

    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)
    # K/V have physical seqlen dim == 0 — this is what crashes on unpatched FA4.
    # causal=False hits GPU IllegalInstruction (TMA OOB on 0-length K tensor).
    # causal=True  hits host SIGFPE in tile_scheduler.py LPT L2-swizzle heuristic
    #              (size_l2 // size_one_head with size_one_head = seqlen_k*... = 0).
    k = torch.empty(batch_size, 0, nheads_kv, d, device=device, dtype=dtype)
    v = torch.empty(batch_size, 0, nheads_kv, dv, device=device, dtype=dtype)

    out, lse = flash_attn_func(q, k, v, causal=causal)

    if is_fake_mode():
        return

    # No crash above already validates the fix. Below validates the contract
    # the early-return promises: zero output, -inf LSE.
    assert out.shape == (batch_size, seqlen_q, nheads, dv), \
        f"Unexpected output shape: {out.shape}"
    assert torch.all(out == 0).item(), \
        f"Expected all-zero output when seqlen_k=0, got max={out.abs().max().item():.6f}"
    if lse is not None:
        assert torch.all(torch.isinf(lse) & (lse < 0)).item(), \
            f"Expected all -inf LSE when seqlen_k=0, got: {lse}"
