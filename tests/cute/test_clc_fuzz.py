"""Adversarial regression tests for CLC tile scheduling.

These cases intentionally target scheduler-sensitive shapes: mismatched
sequence lengths, non-aligned tiles, GQA ratios, minimal problems, and
larger persistent workloads. This is deterministic adversarial coverage,
not randomized fuzzing.
"""

from contextlib import contextmanager
import os
from unittest import mock

import pytest
import torch

from flash_attn.cute import utils as cute_utils
from flash_attn.cute.flash_fwd_sm100 import FlashAttentionForwardSm100
from flash_attn.cute.interface import flash_attn_func, flash_attn_varlen_func
from flash_attn.cute.testing import attention_ref
from flash_attn.cute.tile_scheduler import SchedulingMode, SingleTileLPTScheduler, SingleTileVarlenScheduler


if torch.cuda.is_available():
    COMPUTE_CAPABILITY = torch.cuda.get_device_capability()[0]
    SM_COUNT = torch.cuda.get_device_properties("cuda").multi_processor_count
else:
    COMPUTE_CAPABILITY = 0
    SM_COUNT = 0
pytestmark = pytest.mark.skipif(
    COMPUTE_CAPABILITY not in (10, 11),
    reason="CLC adversarial tests require SM100/SM110 persistent forward",
)

_captured_schedulers: list[tuple[type, SchedulingMode]] = []
_orig_init = FlashAttentionForwardSm100.__init__


def _spy_init(self_inner, *a, **kw):
    _orig_init(self_inner, *a, **kw)
    _captured_schedulers.append((self_inner.TileScheduler, self_inner.scheduling_mode))


@contextmanager
def clc_scheduler_enabled():
    with (
        mock.patch.dict(os.environ, {"FA_CLC": "1"}, clear=False),
        mock.patch.object(cute_utils, "_fa_clc_enabled", True),
        mock.patch.object(FlashAttentionForwardSm100, "__init__", _spy_init),
    ):
        yield


def check_output(q, k, v, *, causal=False, window_size=(None, None), num_splits=1, assert_clc=True):
    _captured_schedulers.clear()
    out, _ = flash_attn_func(q, k, v, causal=causal, window_size=window_size, num_splits=num_splits)
    torch.cuda.synchronize()
    if assert_clc and _captured_schedulers:
        sched_cls, sched_mode = _captured_schedulers[-1]
        assert sched_cls is SingleTileLPTScheduler, f"Expected SingleTileLPTScheduler, got {sched_cls.__name__}"
        assert sched_mode == SchedulingMode.CLC, f"Expected CLC scheduling mode, got {sched_mode!r}"
    out_ref, _ = attention_ref(q, k, v, causal=causal, window_size=window_size)
    out_pt, _ = attention_ref(q, k, v, causal=causal, window_size=window_size, upcast=False, reorder_ops=True)
    fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
    assert (out - out_ref).abs().max().item() <= 2 * (
        out_pt - out_ref
    ).abs().max().item() + fwd_atol, (
        f"max_diff={(out - out_ref).abs().max().item()}, "
        f"pt_max_diff={(out_pt - out_ref).abs().max().item()}, "
        f"fwd_atol={fwd_atol}, "
        f"q={list(q.shape)} k={list(k.shape)} v={list(v.shape)} "
        f"causal={causal} window_size={window_size} num_splits={num_splits}"
    )


def randn(b, s, h, d):
    return torch.randn(b, s, h, d, device="cuda", dtype=torch.bfloat16)


def expected_total_tiles_mha(batch, seqlen_q, heads):
    q_stage = 2 if COMPUTE_CAPABILITY == 10 and seqlen_q > 128 else 1
    num_block = (seqlen_q + q_stage * 128 - 1) // (q_stage * 128)
    return num_block * heads * batch


@pytest.fixture(autouse=True)
def seed():
    torch.random.manual_seed(42)


@pytest.fixture(autouse=True)
def enable_clc_scheduler():
    with clc_scheduler_enabled():
        yield


class TestCLCMismatchedSeqlens:

    @pytest.mark.parametrize("sq,sk", [
        (128, 512),
        (128, 1024),
        (128, 2048),
        (256, 64),
        (256, 128),
        (512, 127),
        (512, 129),
        (64, 4096),
        (1, 128),
        (1, 512),
        (1, 1024),
    ])
    def test_qk_mismatch(self, sq, sk):
        check_output(randn(4, sq, 4, 128), randn(4, sk, 4, 128), randn(4, sk, 4, 128))

    @pytest.mark.parametrize("sq,sk", [
        (128, 513),
        (256, 1023),
        (64, 257),
        (192, 383),
        (1, 255),
    ])
    def test_qk_mismatch_nonaligned_k(self, sq, sk):
        check_output(randn(4, sq, 4, 128), randn(4, sk, 4, 128), randn(4, sk, 4, 128))

    @pytest.mark.parametrize("sq,sk", [
        (1, 128),
        (1, 256),
        (1, 1024),
        (2, 128),
        (3, 512),
    ])
    def test_tiny_q_long_k(self, sq, sk):
        check_output(randn(2, sq, 4, 128), randn(2, sk, 4, 128), randn(2, sk, 4, 128))


class TestCLCNonAlignedShapes:
    @pytest.mark.parametrize("sq", [1, 3, 7, 15, 31, 33, 63, 65, 127, 129, 191, 193, 255, 257])
    def test_nonaligned_q(self, sq):
        check_output(randn(2, sq, 4, 128), randn(2, 256, 4, 128), randn(2, 256, 4, 128))

    @pytest.mark.parametrize("sk", [1, 7, 31, 33, 63, 65, 127, 129, 255, 257, 511, 513])
    def test_nonaligned_k(self, sk):
        check_output(randn(2, 256, 4, 128), randn(2, sk, 4, 128), randn(2, sk, 4, 128))


class TestCLCPrimes:
    @pytest.mark.parametrize("batch,heads,sq,sk", [
        (1, 1, 127, 131),
        (3, 5, 131, 127),
        (7, 3, 257, 251),
        (11, 7, 67, 509),
        (13, 1, 191, 193),
        (5, 11, 61, 67),
        (2, 3, 509, 127),
    ])
    def test_all_prime(self, batch, heads, sq, sk):
        check_output(
            randn(batch, sq, heads, 128),
            randn(batch, sk, heads, 128),
            randn(batch, sk, heads, 128),
        )


class TestCLC2CTA:
    @pytest.mark.parametrize("sq,sk", [
        (128, 512),
        (256, 127),
        (256, 129),
        (128, 2048),
        (1, 512),
        (64, 1024),
        (512, 64),
    ])
    def test_2cta_qk_mismatch(self, sq, sk):
        check_output(randn(4, sq, 4, 128), randn(4, sk, 4, 128), randn(4, sk, 4, 128))

    @pytest.mark.parametrize("batch,heads,sq,sk", [
        (1, 1, 128, 128),
        (1, 1, 256, 512),
        (3, 5, 128, 1024),
        (7, 3, 512, 127),
        (9, 7, 256, 257),
        (13, 1, 128, 64),
    ])
    def test_2cta_adversarial_combos(self, batch, heads, sq, sk):
        check_output(
            randn(batch, sq, heads, 128),
            randn(batch, sk, heads, 128),
            randn(batch, sk, heads, 128),
        )


class TestCLCGQA:
    @pytest.mark.parametrize("q_heads,kv_heads,sq,sk", [
        (4, 1, 128, 512),
        (4, 1, 256, 127),
        (8, 1, 64, 1024),
        (8, 2, 512, 129),
        (8, 4, 1, 256),
        (6, 2, 192, 383),
        (6, 3, 128, 1),
        (12, 4, 257, 511),
    ])
    def test_gqa_mismatch(self, q_heads, kv_heads, sq, sk):
        check_output(
            randn(4, sq, q_heads, 128),
            randn(4, sk, kv_heads, 128),
            randn(4, sk, kv_heads, 128),
        )

    @pytest.mark.parametrize("q_heads,kv_heads", [
        (4, 1), (4, 2), (8, 1), (8, 2), (8, 4), (6, 2), (6, 3), (12, 4),
    ])
    def test_gqa_ratios(self, q_heads, kv_heads):
        check_output(
            randn(4, 512, q_heads, 128),
            randn(4, 512, kv_heads, 128),
            randn(4, 512, kv_heads, 128),
        )


class TestCLCHeadDim:
    @pytest.mark.parametrize("d,dv,sq,sk", [
        (64, 64, 128, 512),
        (64, 64, 1, 256),
        (96, 96, 255, 127),
        (128, 64, 192, 384),
        (128, 64, 1, 1024),
    ])
    def test_head_dims_adversarial(self, d, dv, sq, sk):
        check_output(randn(4, sq, 4, d), randn(4, sk, 4, d), randn(4, sk, 4, dv))

    def test_overlap_sO_sQ_fallback(self):
        from flash_attn.cute.tile_scheduler import SingleTileScheduler

        _captured_schedulers.clear()
        check_output(randn(4, 128, 4, 192), randn(4, 257, 4, 192), randn(4, 257, 4, 128), assert_clc=False)
        assert _captured_schedulers, "No scheduler was captured"
        sched_cls, sched_mode = _captured_schedulers[-1]
        assert sched_cls is SingleTileScheduler, f"Expected SingleTileScheduler fallback, got {sched_cls.__name__}"
        assert sched_mode == SchedulingMode.STATIC, f"Expected STATIC fallback, got {sched_mode!r}"


class TestCLCFallback:

    def test_varlen_fallback(self):
        _captured_schedulers.clear()
        batch, seqlen, heads, d = 4, 256, 4, 128
        lens = torch.tensor([64, 128, 32, 32], dtype=torch.int32)
        cu_seqlens = torch.cat([torch.zeros(1, dtype=torch.int32), lens.cumsum(0)]).to(device="cuda", dtype=torch.int32)
        total = int(cu_seqlens[-1])
        q = torch.randn(total, heads, d, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(total, heads, d, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(total, heads, d, device="cuda", dtype=torch.bfloat16)
        out, _ = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=int(lens.max()),
            max_seqlen_k=int(lens.max()),
        )
        torch.cuda.synchronize()
        assert _captured_schedulers, "No scheduler was captured"
        sched_cls, sched_mode = _captured_schedulers[-1]
        assert sched_cls is SingleTileVarlenScheduler, (
            f"Expected SingleTileVarlenScheduler fallback for varlen, got {sched_cls.__name__}"
        )
        assert sched_mode == SchedulingMode.STATIC, f"Expected STATIC fallback, got {sched_mode!r}"

    @pytest.mark.parametrize("sq,sk,wl,wr", [
        (512, 512, 128, 128),
        (256, 1024, 64, 64),
        (512, 512, 255, 0),
        (128, 2048, 32, 512),
    ])
    def test_local_window_with_clc(self, sq, sk, wl, wr):
        check_output(
            randn(4, sq, 4, 128),
            randn(4, sk, 4, 128),
            randn(4, sk, 4, 128),
            window_size=(wl, wr),
        )


class TestCLCMinimal:
    @pytest.mark.parametrize("sq,sk", [(1, 1), (1, 2), (2, 1), (1, 128), (128, 1)])
    def test_minimal(self, sq, sk):
        check_output(randn(1, sq, 1, 128), randn(1, sk, 1, 128), randn(1, sk, 1, 128))

    def test_single_element(self):
        check_output(randn(1, 1, 1, 64), randn(1, 1, 1, 64), randn(1, 1, 1, 64))


class TestCLCCausal:

    @pytest.mark.parametrize("batch,heads,sq,sk", [
        (3, 5, 259, 259),
        (7, 3, 513, 513),
        (1, 7, 1023, 1023),
        (5, 11, 2049, 2049),
        (2, 3, 4097, 4097),
    ])
    def test_causal_square(self, batch, heads, sq, sk):
        check_output(randn(batch, sq, heads, 128), randn(batch, sk, heads, 128), randn(batch, sk, heads, 128), causal=True)

    @pytest.mark.parametrize("batch,heads,sq,sk", [
        (3, 7, 127, 513),
        (5, 3, 259, 1023),
        (7, 5, 63, 2049),
        (11, 1, 1, 511),
        (2, 9, 1, 1025),
        (9, 3, 33, 4097),
    ])
    def test_causal_qk_mismatch(self, batch, heads, sq, sk):
        check_output(randn(batch, sq, heads, 128), randn(batch, sk, heads, 128), randn(batch, sk, heads, 128), causal=True)

    @pytest.mark.parametrize("batch,heads,sq,sk", [
        (3, 7, 191, 191),
        (7, 5, 193, 193),
        (5, 3, 383, 383),
        (11, 1, 129, 509),
        (2, 13, 1, 131),
        (9, 3, 67, 251),
    ])
    def test_causal_nonaligned(self, batch, heads, sq, sk):
        check_output(randn(batch, sq, heads, 128), randn(batch, sk, heads, 128), randn(batch, sk, heads, 128), causal=True)

    @pytest.mark.parametrize("batch,q_heads,kv_heads,sq", [
        (3, 6, 2, 513),
        (7, 8, 1, 259),
        (5, 12, 4, 1023),
        (2, 8, 2, 2049),
        (11, 4, 1, 191),
    ])
    def test_causal_gqa(self, batch, q_heads, kv_heads, sq):
        check_output(
            randn(batch, sq, q_heads, 128),
            randn(batch, sq, kv_heads, 128),
            randn(batch, sq, kv_heads, 128),
            causal=True,
        )

    def test_causal_large(self):
        check_output(randn(3, 4097, 13, 128), randn(3, 4097, 13, 128), randn(3, 4097, 13, 128), causal=True)


class TestCLCLargeScale:
    def test_large_batch(self):
        check_output(randn(32, 512, 8, 128), randn(32, 512, 8, 128), randn(32, 512, 8, 128))

    def test_long_seq(self):
        check_output(randn(2, 4096, 4, 128), randn(2, 4096, 4, 128), randn(2, 4096, 4, 128))

    def test_many_heads(self):
        check_output(randn(4, 512, 32, 128), randn(4, 512, 32, 128), randn(4, 512, 32, 128))

    @pytest.mark.parametrize("batch,heads,sq,sk", [
        (24, 8, 768, 2048),
        (16, 8, 1536, 4096),
        (12, 8, 2305, 4096),
    ])
    def test_work_stealing_pressure(self, batch, heads, sq, sk):
        total_tiles = expected_total_tiles_mha(batch, sq, heads)
        assert total_tiles > SM_COUNT, f"expected total_tiles={total_tiles} > sm_count={SM_COUNT}"
        check_output(
            randn(batch, sq, heads, 128),
            randn(batch, sk, heads, 128),
            randn(batch, sk, heads, 128),
        )

    def test_long_k_short_q(self):
        check_output(randn(8, 64, 8, 128), randn(8, 8192, 8, 128), randn(8, 8192, 8, 128))

    def test_long_q_short_k(self):
        check_output(randn(4, 4096, 4, 128), randn(4, 64, 4, 128), randn(4, 64, 4, 128))


class TestCLCRepeatability:
    @pytest.mark.parametrize("trial", range(5))
    def test_repeat_mismatch(self, trial):
        torch.random.manual_seed(trial)
        check_output(randn(7, 192, 5, 128), randn(7, 513, 5, 128), randn(7, 513, 5, 128))

    @pytest.mark.parametrize("trial", range(5))
    def test_repeat_2cta(self, trial):
        torch.random.manual_seed(trial)
        check_output(randn(9, 257, 3, 128), randn(9, 511, 3, 128), randn(9, 511, 3, 128))

    @pytest.mark.parametrize("trial", range(5))
    def test_repeat_gqa_mismatch(self, trial):
        torch.random.manual_seed(trial)
        check_output(randn(5, 128, 8, 128), randn(5, 1024, 2, 128), randn(5, 1024, 2, 128))

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
