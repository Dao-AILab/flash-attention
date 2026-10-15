"""SM100 backward with a 64-row KV tile per CTA (FLASH_ATTENTION_BWD_TILE_N=64) against the
default 128-row tile and against FP64 eager.

The 64-row tile puts the accumulators in the M=64 TMEM layout and stages P / dS in SMEM (see
NOTE [M=64 accumulator layout] in flash_bwd_sm100.py); it is the layout the hd256 backward
needs. GQA goes through the fp32 dK/dV accumulate (written row-major, see the postprocess),
varlen through the non-TMA epilogue.
"""

import os

import pytest
import torch

from eager_reference import check_against_reference, make_cu_seqlens

from flash_attn.cute.interface import flash_attn_func, flash_attn_varlen_func
from score_mod_definitions import score_mod_bwd_rel_bias, score_mod_rel_bias

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] not in (10, 11),
    reason="SM100/SM110-only backward tile test",
)

SEED = 20261004
HEAD_DIM = 128
# (batch_size, seqlen_q, seqlen_k): tile edges, a partial last KV tile, long, and Q/K mismatch.
SHAPES = [
    (2, 128, 128),
    (2, 256, 192),
    (2, 512, 512),
    (1, 1000, 1000),
    (1, 4096, 4096),
    (2, 128, 640),
    (2, 640, 128),
]


@pytest.fixture
def bwd_tile_n_env():
    saved = os.environ.get("FLASH_ATTENTION_BWD_TILE_N")
    yield
    if saved is None:
        os.environ.pop("FLASH_ATTENTION_BWD_TILE_N", None)
    else:
        os.environ["FLASH_ATTENTION_BWD_TILE_N"] = saved


def run_fwd_bwd(leaves, q, k, v, dout, causal, tile_n):
    """Forward + backward on the dense views; gradients land in the packed leaves."""
    os.environ["FLASH_ATTENTION_BWD_TILE_N"] = str(tile_n)
    out, _ = flash_attn_func(q, k, v, causal=causal)
    dq, dk, dv = torch.autograd.grad(out, leaves, dout)
    torch.cuda.synchronize()
    return out.detach(), dq, dk, dv


@pytest.mark.parametrize("causal", [False, True], ids=["noncausal", "causal"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("batch_size,seqlen_q,seqlen_k", SHAPES)
def test_bwd_tile_n_64_matches_default(
    batch_size, seqlen_q, seqlen_k, dtype, causal, bwd_tile_n_env
):
    if causal and seqlen_q > seqlen_k:
        pytest.skip("bottom-right causal leaves query rows without keys")
    torch.manual_seed(SEED)
    nheads = 4
    q = torch.randn(
        batch_size * seqlen_q,
        nheads,
        HEAD_DIM,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    k = torch.randn(
        batch_size * seqlen_k,
        nheads,
        HEAD_DIM,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    v = torch.randn_like(k, requires_grad=True)
    dout = torch.randn_like(q)
    q_d = q.view(batch_size, seqlen_q, nheads, HEAD_DIM)
    k_d = k.view(batch_size, seqlen_k, nheads, HEAD_DIM)
    v_d = v.view(batch_size, seqlen_k, nheads, HEAD_DIM)
    dout_d = dout.view_as(q_d)
    ref = run_fwd_bwd((q, k, v), q_d, k_d, v_d, dout_d, causal, 128)
    got = run_fwd_bwd((q, k, v), q_d, k_d, v_d, dout_d, causal, 64)
    # The 64-row tile reorders no reduction that the reference check does not already allow;
    # dK / dV are produced by the same MMAs and must match the default tile exactly.
    assert torch.equal(got[0], ref[0]), "forward does not depend on the backward tile"
    assert torch.equal(got[2], ref[2]), "dK differs from the 128-row tile"
    assert torch.equal(got[3], ref[3]), "dV differs from the 128-row tile"
    q_lens, k_lens = (seqlen_q,) * batch_size, (seqlen_k,) * batch_size
    check_against_reference(
        (got[0].flatten(0, 1), got[1], got[2], got[3]),
        q,
        k,
        v,
        dout,
        q_lens,
        k_lens,
        causal,
        dtype,
    )


HEADS = {"mha": (4, 4), "gqa8_2": (8, 2)}
FEATURE_KWARGS = {
    "none": {},
    "local": {"window_size": (64, 32)},
    "softcap": {"softcap": 15.0},
    "deterministic": {"deterministic": True},
    # position-dependent score_mod: the M=64 layout must hand every warpgroup its own
    # (q_idx, kv_idx) coordinates (score_mod and score_mod_bwd)
    "rel_bias": {
        "score_mod": score_mod_rel_bias,
        "score_mod_bwd": score_mod_bwd_rel_bias,
    },
}
# (q_lens, k_lens) per layout; dense uses equal lengths and the dense API.
FEATURE_SHAPES = [
    pytest.param((256, 256), (192, 192), id="2x256x192"),
    pytest.param((1000,), (1000,), id="1x1000"),
]


def ragged_lens(q_lens, k_lens):
    """Ragged version of a dense budget: keeps the tile edges and adds a short slot."""
    return (
        (q_lens[0] - 3, 65) if len(q_lens) == 2 else (q_lens[0] - 7,),
        ((k_lens[0] + 5, 129) if len(k_lens) == 2 else (k_lens[0] + 1,)),
    )


def run_fwd_bwd_layout(layout, q, k, v, dout, q_lens, k_lens, causal, tile_n, **kwargs):
    os.environ["FLASH_ATTENTION_BWD_TILE_N"] = str(tile_n)
    if layout == "dense":
        b = len(q_lens)
        out, _ = flash_attn_func(
            q.view(b, q_lens[0], *q.shape[1:]),
            k.view(b, k_lens[0], *k.shape[1:]),
            v.view(b, k_lens[0], *v.shape[1:]),
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


def check_tile_n_64_against_128(ref, got, mha_type, dtype):
    assert torch.equal(got[0], ref[0]), "forward does not depend on the backward tile"
    if mha_type == "mha":
        # Same MMAs, same stores: dK / dV must match the 128-row tile exactly.
        assert torch.equal(got[2], ref[2]), "dK differs from the 128-row tile"
        assert torch.equal(got[3], ref[3]), "dV differs from the 128-row tile"
    else:
        # GQA accumulates the Q heads into fp32 in a different order: allow bf16 rounding.
        for name, a, b_ in zip(("dK", "dV"), ref[2:], got[2:]):
            tol = 2 * torch.finfo(dtype).eps * a.float().abs().max()
            assert (a.float() - b_.float()).abs().max() <= tol, name


@pytest.mark.parametrize("causal", [False, True], ids=["noncausal", "causal"])
@pytest.mark.parametrize("feature", list(FEATURE_KWARGS))
@pytest.mark.parametrize("layout", ["dense", "varlen"])
@pytest.mark.parametrize("mha_type", list(HEADS))
@pytest.mark.parametrize("q_lens,k_lens", FEATURE_SHAPES)
def test_bwd_tile_n_64_features(
    q_lens, k_lens, mha_type, layout, feature, causal, bwd_tile_n_env
):
    """GQA accumulate, varlen epilogue, deterministic, local and softcap on the 64-row tile."""
    if layout == "varlen":
        q_lens, k_lens = ragged_lens(q_lens, k_lens)
    if causal and any(nq > nk for nq, nk in zip(q_lens, k_lens)):
        pytest.skip("bottom-right causal leaves query rows without keys")
    torch.manual_seed(SEED)
    heads_q, heads_kv = HEADS[mha_type]
    dtype = torch.bfloat16
    q = torch.randn(
        sum(q_lens), heads_q, HEAD_DIM, device="cuda", dtype=dtype, requires_grad=True
    )
    k = torch.randn(
        sum(k_lens), heads_kv, HEAD_DIM, device="cuda", dtype=dtype, requires_grad=True
    )
    v = torch.randn_like(k, requires_grad=True)
    dout = torch.randn_like(q)
    kwargs = FEATURE_KWARGS[feature]
    args = (layout, q, k, v, dout, q_lens, k_lens, causal)
    ref = run_fwd_bwd_layout(*args, 128, **kwargs)
    got = run_fwd_bwd_layout(*args, 64, **kwargs)
    check_tile_n_64_against_128(ref, got, mha_type, dtype)
    if feature == "deterministic":
        again = run_fwd_bwd_layout(*args, 64, **kwargs)
        assert all(torch.equal(x, y) for x, y in zip(got, again)), (
            "deterministic run differs"
        )
    if feature in ("none", "deterministic"):
        check_against_reference(got, q, k, v, dout, q_lens, k_lens, causal, dtype)


@pytest.mark.parametrize("causal", [False, True], ids=["noncausal", "causal"])
@pytest.mark.parametrize("layout", ["dense", "varlen"])
@pytest.mark.parametrize("mha_type", list(HEADS))
@pytest.mark.parametrize("head_dim,head_dim_v", [(128, 64)])
def test_bwd_tile_n_64_head_dim_v(
    head_dim, head_dim_v, mha_type, layout, causal, bwd_tile_n_env
):
    """head_dim_v != head_dim: the dV epilogue derives its TMEM load width, staging and TMA
    stores from head_dim_v (MHA: TMA / direct store; GQA: fp32 accumulate)."""
    # keys cover every query row, so causal has no empty rows (NaN in the eager reference)
    q_lens, k_lens = (192, 192), (256, 256)
    if layout == "varlen":
        q_lens, k_lens = ragged_lens(q_lens, k_lens)
    torch.manual_seed(SEED)
    heads_q, heads_kv = HEADS[mha_type]
    dtype = torch.bfloat16
    q = torch.randn(
        sum(q_lens), heads_q, head_dim, device="cuda", dtype=dtype, requires_grad=True
    )
    k = torch.randn(
        sum(k_lens), heads_kv, head_dim, device="cuda", dtype=dtype, requires_grad=True
    )
    v = torch.randn(
        sum(k_lens),
        heads_kv,
        head_dim_v,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    dout = torch.randn(sum(q_lens), heads_q, head_dim_v, device="cuda", dtype=dtype)
    args = (layout, q, k, v, dout, q_lens, k_lens, causal)
    got = run_fwd_bwd_layout(*args, 64)
    if mha_type == "mha":
        ref = run_fwd_bwd_layout(*args, 128)
        check_tile_n_64_against_128(ref, got, mha_type, dtype)
    # GQA on the 128-row 2CTA tile does not compile for head_dim_v != head_dim (its fp32 dV
    # staging is sized by head_dim), so every case is also checked against the eager reference.
    check_against_reference(got, q, k, v, dout, q_lens, k_lens, causal, dtype)
