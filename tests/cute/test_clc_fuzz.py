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
from flash_attn.cute.interface import flash_attn_func
from flash_attn.cute.testing import attention_ref


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


@contextmanager
def clc_scheduler_enabled():
    with (
        mock.patch.dict(os.environ, {"FA4_CLC": "1"}, clear=False),
        mock.patch.object(cute_utils, "_fa4_clc_enabled", True),
    ):
        yield


def check_output(q, k, v, *, causal=False, num_splits=1, atol=0.02):
    out, _ = flash_attn_func(q, k, v, causal=causal, num_splits=num_splits)
    torch.cuda.synchronize()
    out_ref, _ = attention_ref(q, k, v, causal=causal)
    max_diff = (out - out_ref).abs().max().item()
    mean_diff = (out - out_ref).abs().mean().item()
    assert max_diff < atol, (
        f"max_diff={max_diff} (mean={mean_diff}), "
        f"q={list(q.shape)} k={list(k.shape)} v={list(v.shape)} "
        f"causal={causal} num_splits={num_splits}"
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
        (192, 128, 128, 257),
        (192, 128, 64, 512),
    ])
    def test_head_dims_adversarial(self, d, dv, sq, sk):
        check_output(randn(4, sq, 4, d), randn(4, sk, 4, d), randn(4, sk, 4, dv))


class TestCLCMinimal:
    @pytest.mark.parametrize("sq,sk", [(1, 1), (1, 2), (2, 1), (1, 128), (128, 1)])
    def test_minimal(self, sq, sk):
        check_output(randn(1, sq, 1, 128), randn(1, sk, 1, 128), randn(1, sk, 1, 128))

    def test_single_element(self):
        check_output(randn(1, 1, 1, 64), randn(1, 1, 1, 64), randn(1, 1, 1, 64))


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


class TestCLCSplitK:

    @pytest.mark.parametrize("num_splits", [2, 3, 4, 7, 8])
    def test_basic_split_k(self, num_splits):
        check_output(
            randn(2, 128, 4, 128), randn(2, 2048, 4, 128), randn(2, 2048, 4, 128),
            num_splits=num_splits,
        )

    @pytest.mark.parametrize("num_splits", [2, 4, 8])
    def test_split_k_causal(self, num_splits):
        check_output(
            randn(2, 512, 4, 128), randn(2, 2048, 4, 128), randn(2, 2048, 4, 128),
            causal=True, num_splits=num_splits,
        )

    @pytest.mark.parametrize("sq,sk,num_splits", [
        (1, 1024, 2),
        (1, 2048, 4),
        (1, 4096, 8),
        (2, 1024, 3),
        (3, 2048, 5),
    ])
    def test_split_k_decode(self, sq, sk, num_splits):
        check_output(
            randn(4, sq, 8, 128), randn(4, sk, 8, 128), randn(4, sk, 8, 128),
            num_splits=num_splits,
        )

    @pytest.mark.parametrize("q_heads,kv_heads,num_splits", [
        (8, 1, 2),
        (8, 2, 4),
        (12, 4, 3),
        (6, 2, 8),
    ])
    def test_split_k_gqa(self, q_heads, kv_heads, num_splits):
        check_output(
            randn(4, 128, q_heads, 128),
            randn(4, 2048, kv_heads, 128),
            randn(4, 2048, kv_heads, 128),
            num_splits=num_splits,
        )

    @pytest.mark.parametrize("sq,sk,num_splits", [
        (127, 1023, 2),
        (65, 2049, 4),
        (193, 1537, 3),
    ])
    def test_split_k_nonaligned(self, sq, sk, num_splits):
        check_output(
            randn(4, sq, 4, 128), randn(4, sk, 4, 128), randn(4, sk, 4, 128),
            num_splits=num_splits,
        )

    @pytest.mark.parametrize("d,dv,num_splits", [
        (64, 64, 4),
        (96, 96, 2),
        (128, 64, 3),
        (192, 128, 2),
    ])
    def test_split_k_head_dims(self, d, dv, num_splits):
        check_output(
            randn(4, 128, 4, d), randn(4, 2048, 4, d), randn(4, 2048, 4, dv),
            num_splits=num_splits,
        )

    def test_split_k_auto_heuristic(self):
        check_output(
            randn(1, 64, 1, 128), randn(1, 8192, 1, 128), randn(1, 8192, 1, 128),
            num_splits=0,
        )


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