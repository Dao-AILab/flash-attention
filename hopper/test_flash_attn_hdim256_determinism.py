import itertools
import os

import pytest
import torch

from flash_attn_interface import flash_attn_func, flash_attn_varlen_func


pytestmark = pytest.mark.skipif(
    os.getenv("FLASH_ATTENTION_DISABLE_BACKWARD", "FALSE") == "TRUE"
    or os.getenv("FLASH_ATTENTION_DISABLE_HDIM256", "FALSE") == "TRUE",
    reason="requires hdim256 backward",
)


def _attention(samples, order, varlen, options):
    ordered = [samples[i] for i in order]
    combine = torch.cat if varlen else torch.stack
    q, k, v = [
        combine([sample[j] for sample in ordered]).detach().requires_grad_()
        for j in range(3)
    ]
    dout = combine([sample[3] for sample in ordered])
    if varlen:
        lengths_q = [sample[0].shape[0] for sample in ordered]
        lengths_k = [sample[1].shape[0] for sample in ordered]
        cu_q = torch.tensor(
            [0, *itertools.accumulate(lengths_q)], device="cuda", dtype=torch.int32
        )
        cu_k = torch.tensor(
            [0, *itertools.accumulate(lengths_k)], device="cuda", dtype=torch.int32
        )
        out, lse = flash_attn_varlen_func(
            q, k, v, cu_q, cu_k, max(lengths_q), max(lengths_k), **options
        )
    else:
        out, lse = flash_attn_func(q, k, v, **options)
    dq, dk, dv = torch.autograd.grad(out, (q, k, v), dout)
    if varlen:
        parts = zip(
            out.detach().split(lengths_q),
            lse.detach().split(lengths_q, dim=-1),
            dq.split(lengths_q),
            dk.split(lengths_k),
            dv.split(lengths_k),
        )
    else:
        parts = zip(
            out.detach().unbind(), lse.detach().unbind(), dq.unbind(), dk.unbind(), dv.unbind()
        )
    return dict(zip(order, parts))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("d", [224, 256])
@pytest.mark.parametrize("kv_heads", [8, 2, 1])
@pytest.mark.parametrize("seqlen_q,seqlen_k", [(63, 97), (65, 33)])
@pytest.mark.parametrize("mode", ["full", "causal", "local", "softcap"])
@pytest.mark.parametrize("varlen", [False, True])
def test_hdim256_fresh_autograd_replay(dtype, d, kv_heads, seqlen_q, seqlen_k, mode, varlen):
    if dtype == torch.float16 and os.getenv("FLASH_ATTENTION_DISABLE_FP16", "FALSE") == "TRUE":
        pytest.skip("FP16 is disabled")
    if varlen and os.getenv("FLASH_ATTENTION_DISABLE_VARLEN", "FALSE") == "TRUE":
        pytest.skip("variable-length attention is disabled")
    if mode == "local" and os.getenv("FLASH_ATTENTION_DISABLE_LOCAL", "FALSE") == "TRUE":
        pytest.skip("local attention is disabled")
    if mode == "softcap" and os.getenv("FLASH_ATTENTION_DISABLE_SOFTCAP", "FALSE") == "TRUE":
        pytest.skip("softcap is disabled")
    torch.manual_seed(0)
    options = dict(
        causal=mode == "causal",
        window_size=(37, 11) if mode == "local" else (-1, -1),
        softcap=15.0 if mode == "softcap" else 0.0,
        deterministic=True,
        num_splits=1,
        pack_gqa=False,
        return_attn_probs=True,
    )
    lengths = (
        [(seqlen_q, seqlen_k), (seqlen_q - 7, seqlen_k - 3), (17, 1)]
        if varlen else [(seqlen_q, seqlen_k)] * 3
    )
    samples = []
    for sq, sk in lengths:
        q = torch.randn(sq, 8, d, device="cuda", dtype=dtype)
        k, v = [torch.randn(sk, kv_heads, d, device="cuda", dtype=dtype) for _ in range(2)]
        samples.append((q, k, v, torch.randn_like(q)))
    if torch.cuda.get_device_capability()[0] < 9:
        with pytest.raises(RuntimeError, match="hdim 256 requires SM90"):
            _attention(samples, [0], varlen, options)
        return
    expected = {i: _attention(samples, [i], varlen, options)[i] for i in range(3)}
    # Every invocation performs a fresh public forward and autograd backward;
    # no saved forward output or LSE is injected into a later backward call.
    for order in ([0, 1, 2], [2, 0, 1]):
        for _ in range(3):
            actual = _attention(samples, order, varlen, options)
            for i in order:
                for name, left, right in zip(("out", "lse", "dq", "dk", "dv"), actual[i], expected[i]):
                    assert not torch.isnan(left).any(), name
                    if name != "lse":
                        assert torch.isfinite(left).all(), name
                    assert torch.equal(
                        left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8)
                    ), f"{name} changed with batch order or replay"
