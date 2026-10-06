"""Eager FP64/low-precision attention references and tolerance checks for varlen tests.

Shared by the hd256 scheduler tests and the hd256 backward shape grid.
"""

from itertools import accumulate

import torch


def make_cu_seqlens(lengths):
    return torch.tensor((0, *accumulate(lengths)), dtype=torch.int32, device="cuda")


def varlen_reference(q, k, v, dout, q_lens, k_lens, causal, ref_dtype):
    """Return eager O/dQ/dK/dV in ref_dtype for active rows only.

    Ignore extra input capacity; return only O when dout is None.
    """
    scale = q.shape[-1] ** -0.5
    heads_q, heads_kv = q.shape[1], k.shape[1]
    qr, kr, vr = [t.detach().to(ref_dtype).requires_grad_(dout is not None) for t in (q, k, v)]
    outputs = []
    q_off = k_off = 0
    for nq, nk in zip(q_lens, k_lens):
        if nq > 0:
            assert nk > 0, "a query row with no keys has undefined softmax output"
            qi = qr[q_off : q_off + nq].transpose(0, 1)
            ki = (
                kr[k_off : k_off + nk].transpose(0, 1).repeat_interleave(heads_q // heads_kv, dim=0)
            )
            vi = (
                vr[k_off : k_off + nk].transpose(0, 1).repeat_interleave(heads_q // heads_kv, dim=0)
            )
            scores = (qi @ ki.transpose(-1, -2)) * scale
            if causal:
                # Bottom-right aligned causal mask, also correct when nq != nk.
                rows = torch.arange(nq, device=q.device)[:, None]
                cols = torch.arange(nk, device=q.device)[None, :]
                scores = scores.masked_fill(cols > rows + nk - nq, float("-inf"))
            outputs.append((scores.softmax(-1) @ vi).transpose(0, 1))
        q_off += nq
        k_off += nk
    out = torch.cat(outputs)
    if dout is None:
        return (out,)
    dq, dk, dv = torch.autograd.grad(out, (qr, kr, vr), dout[:q_off].to(ref_dtype))
    return out, dq[:q_off], dk[:k_off], dv[:k_off]


def rms(t):
    return t.square().mean().sqrt()


def check_against_reference(actual, q, k, v, dout, q_lens, k_lens, causal, dtype):
    """Compare O/dQ/dK/dV against FP64 eager, allowing the measured low-precision eager error.

    The fused kernel orders its softmax/reductions differently from eager and
    rounds P and the accumulator to the output dtype, so on top of the measured
    unfused low-precision eager error allow two output ULPs (pointwise/max and RMS).
    """
    truth = varlen_reference(q, k, v, dout, q_lens, k_lens, causal, torch.float64)
    eager = varlen_reference(q, k, v, dout, q_lens, k_lens, causal, dtype)
    assert len(actual) == len(truth) == len(eager) == (1 if dout is None else 4)
    ulp = torch.finfo(dtype).eps
    for name, got, want, low in zip(("O", "dQ", "dK", "dV"), actual, truth, eager):
        assert got.shape == want.shape, (name, got.shape, want.shape)
        assert torch.isfinite(got).all(), name
        error = (got.double() - want).abs()
        eager_error = (low.double() - want).abs()
        allowance = 2 * ulp * want.abs()
        assert error.max() <= eager_error.max() + allowance.max(), (
            name,
            error.max().item(),
            eager_error.max().item(),
            allowance.max().item(),
        )
        assert rms(error) <= rms(eager_error) + rms(allowance), (
            name,
            rms(error).item(),
            rms(eager_error).item(),
            rms(allowance).item(),
        )
