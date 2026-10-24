"""SM100 backward with the query heads of a KV head packed into the Q tile
(FLASH_ATTENTION_BWD_PACK_GQA=1) against the per-head path (FLASH_ATTENTION_BWD_PACK_GQA=0) and
against FP64 eager.

See NOTE [bwd pack_gqa] in pack_gqa.py: one tile per KV head iterates over the packed
rows of all its query heads, dK / dV stay in TMEM for the group (no fp32 accumulate, no
postprocess), dQaccum / LSE / dPsum are per KV head in packed row order.
"""

import itertools
import os

import pytest
import torch

from eager_reference import check_against_reference, make_cu_seqlens

from flash_attn.cute import utils
from flash_attn.cute.interface import (
    _flash_attn_bwd,
    flash_attn_func,
    flash_attn_varlen_func,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] not in (10, 11),
    reason="SM100/SM110-only backward test",
)

SEED = 20261004
HEAD_DIMS = [64, 128, 256]
HEADS = {"gqa8_2": (8, 2), "gqa16_4": (16, 4), "gqa32_4": (32, 4), "mqa8_1": (8, 1)}
FEATURE_KWARGS = {
    "none": {},
    "local": {"window_size": (64, 32)},
    "softcap": {"softcap": 15.0},
    "deterministic": {"deterministic": True},
}
# (q_lens, k_lens) per layout; dense uses equal lengths per batch and the dense API.
SHAPES = [
    pytest.param((256, 256), (192, 192), id="2x256x192"),
    pytest.param((1000,), (1000,), id="1x1000"),
    pytest.param((16, 16), (640, 640), id="2x16x640"),
]


@pytest.fixture
def pack_gqa_env():
    saved = os.environ.get("FLASH_ATTENTION_BWD_PACK_GQA")
    yield
    if saved is None:
        os.environ.pop("FLASH_ATTENTION_BWD_PACK_GQA", None)
    else:
        os.environ["FLASH_ATTENTION_BWD_PACK_GQA"] = saved


def hd256_on_general_kernel():
    getter = getattr(utils, "_get_hd256_generic_bwd", None)
    return getter is None or getter()


def run_fwd_bwd_layout(layout, q, k, v, dout, q_lens, k_lens, causal, pack, **kwargs):
    os.environ["FLASH_ATTENTION_BWD_PACK_GQA"] = "1" if pack else "0"
    head_dim = q.shape[-1]
    if layout == "dense":
        b = len(q_lens)
        heads_q, heads_kv = q.shape[1], k.shape[1]
        out, _ = flash_attn_func(
            q.view(b, q_lens[0], heads_q, head_dim),
            k.view(b, k_lens[0], heads_kv, head_dim),
            v.view(b, k_lens[0], heads_kv, head_dim),
            causal=causal,
            **kwargs,
        )
        out = out.flatten(0, 1)
    else:
        out, _ = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=make_cu_seqlens(q_lens),
            cu_seqlens_k=make_cu_seqlens(k_lens),
            max_seqlen_q=max(q_lens),
            max_seqlen_k=max(k_lens),
            causal=causal,
            **kwargs,
        )
    dq, dk, dv = torch.autograd.grad(out, (q, k, v), dout)
    torch.cuda.synchronize()
    return out.detach(), dq, dk, dv


@pytest.mark.parametrize("causal", [False, True], ids=["noncausal", "causal"])
@pytest.mark.parametrize("feature", list(FEATURE_KWARGS))
@pytest.mark.parametrize("layout", ["dense", "varlen"])
@pytest.mark.parametrize("mha_type", list(HEADS))
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("q_lens,k_lens", SHAPES)
def test_bwd_pack_gqa_matches_unpacked(
    q_lens, k_lens, head_dim, mha_type, layout, feature, causal, pack_gqa_env
):
    if head_dim == 256 and not hd256_on_general_kernel():
        pytest.skip("hdim 256 backward runs on the dedicated kernels (no packing)")
    if layout == "varlen":
        # Ragged version of the same budget: tile edges, a short slot, an empty slot.
        q_lens, k_lens = (
            (q_lens[0] - 3, 0, 65) if len(q_lens) == 2 else (q_lens[0] - 7,),
            ((k_lens[0] + 5, 0, 129) if len(k_lens) == 2 else (k_lens[0] + 1,)),
        )
    if causal and any(nq > nk for nq, nk in zip(q_lens, k_lens)):
        pytest.skip("bottom-right causal leaves query rows without keys")
    torch.manual_seed(SEED)
    heads_q, heads_kv = HEADS[mha_type]
    dtype = torch.bfloat16
    q = torch.randn(
        sum(q_lens), heads_q, head_dim, device="cuda", dtype=dtype, requires_grad=True
    )
    k = torch.randn(
        sum(k_lens), heads_kv, head_dim, device="cuda", dtype=dtype, requires_grad=True
    )
    v = torch.randn_like(k, requires_grad=True)
    dout = torch.randn_like(q)
    kwargs = FEATURE_KWARGS[feature]
    args = (layout, q, k, v, dout, q_lens, k_lens, causal)
    ref = run_fwd_bwd_layout(*args, False, **kwargs)
    got = run_fwd_bwd_layout(*args, True, **kwargs)
    assert torch.equal(got[0], ref[0]), (
        "forward does not depend on the backward packing"
    )
    for name, a, b_ in zip(("dQ", "dK", "dV"), ref[1:], got[1:]):
        # dQ: same fp32 reduction per KV head; dK / dV: one TMEM accumulation instead of fp32
        # atomics over the group -> at most bf16 rounding of the result.
        assert torch.isfinite(b_.float()).all(), f"{name} has non-finite values"
        tol = 2 * torch.finfo(dtype).eps * a.float().abs().max()
        assert (a.float() - b_.float()).abs().max() <= tol, (
            f"{name} differs from the unpacked path"
        )
    if feature == "deterministic":
        again = run_fwd_bwd_layout(*args, True, **kwargs)
        assert all(torch.equal(x, y) for x, y in zip(got, again)), (
            "deterministic run differs"
        )
    if feature in ("none", "deterministic"):
        check_against_reference(got, q, k, v, dout, q_lens, k_lens, causal, dtype)


@pytest.mark.parametrize("causal", [False, True], ids=["noncausal", "causal"])
@pytest.mark.parametrize("layout", ["dense", "varlen"])
@pytest.mark.parametrize("mha_type", ["gqa8_2", "mqa8_1"])
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_bwd_pack_gqa_unused_kv_tail(head_dim, mha_type, layout, causal, pack_gqa_env):
    """K/V rows past seqused_k must not reach dQ / dK / dV on the packed kernel either.

    The packed kernel bounds its K/V TMA loads like the per-head one (per-CTA descriptor copies
    rewritten to the sequence's row extent), but it runs one tile per KV head over qh times the
    rows: the descriptor slots, the packed LSE / dPsum / dQaccum offsets and the seqused_k masking
    are exercised with a different grid. Same layouts as test_flash_attn_sm100_bwd_unused_kv_tail:
    packed K/V with cu_seqlens_k + seqused_k, and dense (batch, seqlen, heads, d) with seqused_k.
    """
    if head_dim == 256 and not hd256_on_general_kernel():
        pytest.skip("hdim 256 backward runs on the dedicated kernels (no packing)")
    torch.manual_seed(SEED)
    heads_q, heads_kv = HEADS[mha_type]
    dtype = torch.bfloat16
    lengths = [100, 256, 37, 200, 129]
    batch, capacity = len(lengths), 256
    seqused_k = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    if layout == "varlen":
        q = torch.randn(sum(lengths), heads_q, head_dim, device="cuda", dtype=dtype)
        k = torch.randn(
            batch * capacity, heads_kv, head_dim, device="cuda", dtype=dtype
        )
        v = torch.randn_like(k)
        kwargs = dict(
            causal=causal,
            cu_seqlens_q=torch.tensor(
                [0] + list(itertools.accumulate(lengths)),
                device="cuda",
                dtype=torch.int32,
            ),
            cu_seqlens_k=torch.arange(
                0, (batch + 1) * capacity, capacity, device="cuda", dtype=torch.int32
            ),
            seqused_k=seqused_k,
            max_seqlen_q=max(lengths),
            max_seqlen_k=capacity,
        )
    else:
        seqlen_q = 128
        q = torch.randn(batch, seqlen_q, heads_q, head_dim, device="cuda", dtype=dtype)
        k = torch.randn(batch, capacity, heads_kv, head_dim, device="cuda", dtype=dtype)
        v = torch.randn_like(k)
        kwargs = dict(
            causal=causal,
            seqused_k=seqused_k,
            max_seqlen_q=seqlen_q,
            max_seqlen_k=capacity,
        )
    out, lse = flash_attn_varlen_func(q, k, v, return_lse=True, **kwargs)
    dout = torch.randn_like(out)

    def backward(k_in, v_in, pack):
        os.environ["FLASH_ATTENTION_BWD_PACK_GQA"] = "1" if pack else "0"
        dq, dk, dv = _flash_attn_bwd(q, k_in, v_in, out, dout, lse, **kwargs)[:3]
        torch.cuda.synchronize()
        return dq, dk, dv

    valid = torch.zeros(batch, capacity, dtype=torch.bool, device="cuda")
    for b, length in enumerate(lengths):
        valid[b, :length] = True
    if layout == "varlen":
        valid = valid.flatten()
    eps = torch.finfo(dtype).eps
    unpacked = backward(k, v, False)
    ref = backward(k, v, True)
    for name, a, b_ in zip(("dQ", "dK", "dV"), unpacked, ref):
        if name != "dQ":
            # dK / dV rows past seqused_k are never written; compare the valid rows only.
            a, b_ = a[valid], b_[valid]
        assert torch.isfinite(b_.float()).all(), f"{name} has non-finite values"
        tol = 2 * eps * a.float().abs().max()
        assert (a.float() - b_.float()).abs().max() <= tol, (
            f"{name} differs from the unpacked path"
        )
    # Rows past seqused_k hold NaN: a K/V load that is not bounded by the sequence end leaks it
    # through 0 * NaN in dP = dO @ V^T / dQ = dS @ K. The clean and poisoned runs may still
    # differ by an ulp (fp32 reduce-add order).
    for poison in ("k", "v", "kv"):
        k_nan, v_nan = k.clone(), v.clone()
        if "k" in poison:
            k_nan[~valid] = float("nan")
        if "v" in poison:
            v_nan[~valid] = float("nan")
        got = backward(k_nan, v_nan, True)
        for name, a, b_ in zip(("dQ", "dK", "dV"), ref, got):
            if name != "dQ":
                a, b_ = a[valid], b_[valid]
            assert torch.isfinite(b_.float()).all(), f"{poison}: {name} has NaN / Inf"
            torch.testing.assert_close(
                b_, a, atol=4 * eps * a.abs().max().item(), rtol=4 * eps
            )


def test_bwd_pack_gqa_heuristic():
    """Pack when the grid still hides the qh-times longer packed tiles; keep ragged / long causal
    MQA unpacked (measured on B200: MQA 16/1 hd256, one 8k causal sequence plus 16 x 256 in a
    varlen pack: packed 3946 us, unpacked 2297 us)."""
    from flash_attn.cute.interface import _bwd_pack_gqa_heuristic

    def decide(
        batch,
        heads,
        heads_kv,
        seqlen_q,
        seqlen_k,
        causal,
        varlen=False,
        total_q=None,
        total_k=None,
    ):
        return _bwd_pack_gqa_heuristic(
            num_sms=148,
            batch_size=batch,
            num_head=heads,
            num_head_kv=heads_kv,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            total_q=total_q if total_q is not None else batch * seqlen_q,
            total_k=total_k if total_k is not None else batch * seqlen_k,
            m_block_size=128,
            n_block_size=128,
            cluster_size=2,
            causal=causal,
            local=False,
            varlen=varlen,
        )

    # dense 2k / 4k GQA and MQA batches: enough packed tiles, pack
    assert decide(8, 16, 4, 2048, 2048, False)
    assert decide(4, 16, 1, 4096, 4096, False)
    # one long causal MQA sequence: the first packed KV tile runs 16 x 64 iterations, serial
    assert not decide(1, 16, 1, 8192, 8192, True)
    # ragged varlen pack, one 8k sequence + 16 x 256 (PR 2993's workload), causal and not
    lens = (8192,) + (256,) * 16
    assert not decide(
        17, 16, 1, 8192, 8192, True, varlen=True, total_q=sum(lens), total_k=sum(lens)
    )
    assert not decide(
        17,
        16,
        1,
        8192,
        4096,
        False,
        varlen=True,
        total_q=sum(lens),
        total_k=sum(lens) // 2,
    )
