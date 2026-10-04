"""SM100 backward with a 64-row KV tile per CTA (FLASH_ATTENTION_BWD_TILE_N=64) against the
default 128-row tile and against FP64 eager.

The 64-row tile puts the accumulators in the M=64 TMEM layout and stages P / dS in SMEM (see
NOTE [M=64 accumulator layout] in flash_bwd_sm100.py); it is the layout the hd256 backward
needs. Dense MHA only for now: GQA and varlen go through epilogue paths that are not wired yet.
"""

import os

import pytest
import torch

from eager_reference import check_against_reference

from flash_attn.cute.interface import flash_attn_func

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
