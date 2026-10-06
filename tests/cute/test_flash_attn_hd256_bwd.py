"""hd256 forward + backward on SM100/SM110 against FP64 eager, dense and varlen.

Shapes sit on the 128-row tile edges (127/128/129), a partial last tile (1000), a long
sequence (4096) and Q/K length mismatches in both directions, for MHA and 24/4 GQA, causal
and non-causal. The features the hd256 backward does not support yet (softcap,
learnable_sink, local windows, deterministic) are parametrized, and skipped, in
test_flash_attn.py.
"""

import pytest
import torch

from eager_reference import check_against_reference, make_cu_seqlens

from flash_attn.cute.interface import flash_attn_func, flash_attn_varlen_func

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (10, 11),
    reason="SM100/SM110-only hd256 backward",
)

HEAD_DIM = 256
SEED = 20261004

# (batch_size, seqlen_q, seqlen_k)
DENSE_SHAPES = [
    (2, 127, 127),
    (2, 128, 128),
    (2, 129, 129),
    (2, 512, 512),
    (1, 1000, 1000),
    (1, 4096, 4096),
    (2, 128, 640),
    (2, 640, 128),
]
# (q_lens, k_lens): ragged, an empty batch slot, Q shorter than K, Q longer than K.
VARLEN_CASES = [
    pytest.param((128, 256), (128, 256), id="ragged"),
    pytest.param((129, 0, 257), (129, 0, 257), id="empty_slot"),
    pytest.param((65, 257), (129, 513), id="q_shorter"),
    pytest.param((257, 65), (129, 513), id="q_longer"),
]
HEADS = {"mha": (6, 6), "gqa24_4": (24, 4)}
DTYPES = pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
MHA_TYPES = pytest.mark.parametrize("mha_type", list(HEADS))
CAUSAL = pytest.mark.parametrize("causal", [False, True], ids=["noncausal", "causal"])


def make_packed_inputs(q_rows, k_rows, mha_type, dtype):
    """Leaf tensors in the packed (rows, heads, HEAD_DIM) layout the reference consumes."""
    torch.manual_seed(SEED)
    heads_q, heads_kv = HEADS[mha_type]
    q = torch.randn(q_rows, heads_q, HEAD_DIM, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn(k_rows, heads_kv, HEAD_DIM, device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    dout = torch.randn_like(q)
    return q, k, v, dout


def skip_causal_without_keys(causal, q_lens, k_lens):
    if causal and any(nq > nk for nq, nk in zip(q_lens, k_lens)):
        pytest.skip("bottom-right causal leaves query rows without keys")


@pytest.mark.parametrize("batch_size,seqlen_q,seqlen_k", DENSE_SHAPES)
@DTYPES
@MHA_TYPES
@CAUSAL
def test_hd256_dense_fwd_bwd(batch_size, seqlen_q, seqlen_k, dtype, mha_type, causal):
    q_lens, k_lens = (seqlen_q,) * batch_size, (seqlen_k,) * batch_size
    skip_causal_without_keys(causal, q_lens, k_lens)
    q, k, v, dout = make_packed_inputs(sum(q_lens), sum(k_lens), mha_type, dtype)
    # Views of the packed leaves: gradients land in the packed layout.
    q_dense = q.view(batch_size, seqlen_q, *q.shape[1:])
    k_dense = k.view(batch_size, seqlen_k, *k.shape[1:])
    v_dense = v.view(batch_size, seqlen_k, *v.shape[1:])
    out, _ = flash_attn_func(q_dense, k_dense, v_dense, causal=causal)
    dq, dk, dv = torch.autograd.grad(out, (q, k, v), dout.view_as(out))
    torch.cuda.synchronize()
    check_against_reference(
        (out.detach().flatten(0, 1), dq, dk, dv), q, k, v, dout, q_lens, k_lens, causal, dtype
    )


@pytest.mark.parametrize("q_lens,k_lens", VARLEN_CASES)
@DTYPES
@MHA_TYPES
@CAUSAL
def test_hd256_varlen_fwd_bwd(q_lens, k_lens, dtype, mha_type, causal):
    skip_causal_without_keys(causal, q_lens, k_lens)
    q, k, v, dout = make_packed_inputs(sum(q_lens), sum(k_lens), mha_type, dtype)
    out, _ = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=make_cu_seqlens(q_lens),
        cu_seqlens_k=make_cu_seqlens(k_lens),
        max_seqlen_q=max(q_lens),
        max_seqlen_k=max(k_lens),
        causal=causal,
    )
    dq, dk, dv = torch.autograd.grad(out, (q, k, v), dout)
    torch.cuda.synchronize()
    check_against_reference((out.detach(), dq, dk, dv), q, k, v, dout, q_lens, k_lens, causal, dtype)
