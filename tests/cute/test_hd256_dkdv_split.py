"""hd256 GQA/MQA dK/dV split on SM100/SM110: split against unsplit and against FP64 eager.

The split only changes the fp32 summation order of dK/dV. O and dQ stay bitwise equal to the
unsplit kernel, dK/dV stay within the FP64 reference tolerance, and repeated runs are bitwise
identical. The packs hold one long sequence among short ones, the case the split targets, with a
host max_seqlen and with max_seqlen=None (varlen scheduler).
"""

import pytest
import torch

from eager_reference import check_against_reference, make_cu_seqlens

from flash_attn.cute import interface
from flash_attn.cute.interface import flash_attn_varlen_func

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (10, 11),
    reason="SM100/SM110-only hd256 backward",
)

HEAD_DIM = 256
SEED = 20261012
HEADS = {"mqa16_1": (16, 1), "gqa24_4": (24, 4)}
# Self-attention packs (q_lens == k_lens).
PACKS = [
    pytest.param((4096, 128, 300, 77), id="long_first"),
    pytest.param((128, 300, 77, 4096), id="long_last"),
    pytest.param((2048, 2048), id="two_long"),
]


@pytest.fixture
def dkdv_splits(monkeypatch):
    """Set FLASH_ATTENTION_HD256_DKDV_SPLITS (read once per process, so clear the cache)."""

    def set_splits(n):
        monkeypatch.setenv("FLASH_ATTENTION_HD256_DKDV_SPLITS", str(n))
        interface._hd256_dkdv_split_env.cache_clear()

    yield set_splits
    monkeypatch.undo()
    interface._hd256_dkdv_split_env.cache_clear()


def run(lens, heads, causal, dtype, host_max_seqlen):
    torch.manual_seed(SEED)
    heads_q, heads_kv = heads
    rows = sum(lens)
    q = torch.randn(rows, heads_q, HEAD_DIM, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn(rows, heads_kv, HEAD_DIM, device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    dout = torch.randn_like(q)
    cu = make_cu_seqlens(lens)
    max_seqlen = max(lens) if host_max_seqlen else None
    out, _ = flash_attn_varlen_func(
        q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
        causal=causal,
    )
    dq, dk, dv = torch.autograd.grad(out, (q, k, v), dout)
    torch.cuda.synchronize()
    return (out.detach(), dq, dk, dv), (q, k, v, dout)


@pytest.mark.parametrize("lens", PACKS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("mha_type", list(HEADS))
@pytest.mark.parametrize("causal", [False, True], ids=["noncausal", "causal"])
@pytest.mark.parametrize("host_max_seqlen", [True, False], ids=["max_seqlen", "max_seqlen_none"])
def test_hd256_dkdv_split(lens, dtype, mha_type, causal, host_max_seqlen, dkdv_splits):
    heads = HEADS[mha_type]
    qhead_per_kvhead = heads[0] // heads[1]

    dkdv_splits(1)
    unsplit, _ = run(lens, heads, causal, dtype, host_max_seqlen)
    dkdv_splits(4)
    assert interface._hd256_dkdv_split_config(qhead_per_kvhead, max(lens), None, None)[0] > 1
    split, inputs = run(lens, heads, causal, dtype, host_max_seqlen)
    again, _ = run(lens, heads, causal, dtype, host_max_seqlen)

    for name, a, b in zip(("O", "dQ"), unsplit[:2], split[:2]):
        assert torch.equal(a, b), f"{name} changed with the dK/dV split"
    for name, a, b in zip(("O", "dQ", "dK", "dV"), split, again):
        assert torch.equal(a, b), f"{name} not deterministic with the dK/dV split"
    check_against_reference(split, *inputs, lens, lens, causal, dtype)
