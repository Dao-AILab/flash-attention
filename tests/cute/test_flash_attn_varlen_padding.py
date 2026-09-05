"""SM80/SM120 varlen forward: over-provisioned grid tiles must not touch memory.

The varlen tile scheduler sizes the grid from ``total_q + num_batch *
(tile_m - 1)`` rows, so a batch whose sequence lengths do not fill whole
query tiles (a zero-length sequence, or two sequences with tile remainders)
gets extra CTAs whose work tile maps to batch index ``num_batch``. The SM80
and SM120 kernels used to derive those CTAs' sequence lengths from
``cu_seqlens[num_batch + 1]``, one element past the tensor, and then loaded
K/V and stored O for the garbage length. The test plants a large value right
after ``cu_seqlens`` and a NaN canary right after ``out``: a kernel that runs
the extra tiles overwrites the canary (and faults when the garbage offset
leaves mapped memory); a kernel that skips them leaves it intact and matches
a per-sequence reference.
"""

import pytest
import torch

from flash_attn.cute.interface import _flash_attn_fwd

TILE_M = 128
CANARY_ROWS = 4096


def _wasted_tiles(lens):
    total = sum(lens)
    provisioned = (total + len(lens) * (TILE_M - 1)) // TILE_M
    used = sum((length + TILE_M - 1) // TILE_M for length in lens)
    return provisioned - used


def _reference(q, k, v, lens, scale):
    outs = []
    start = 0
    for length in lens:
        if length == 0:
            continue
        qs = q[start : start + length].transpose(0, 1).float()
        ks = k[start : start + length].transpose(0, 1).float()
        vs = v[start : start + length].transpose(0, 1).float()
        out = torch.nn.functional.scaled_dot_product_attention(qs, ks, vs, is_causal=True, scale=scale)
        outs.append(out.transpose(0, 1))
        start += length
    return torch.cat(outs, dim=0)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (8, 12),
    reason="SM80/SM120 forward kernel",
)
@pytest.mark.parametrize("head_dim,head_dim_v", [(128, 128), (192, 128)])
@pytest.mark.parametrize("lens", [[3615, 0], [1055, 744], [1057, 111], [744, 1055]])
def test_varlen_extra_tiles_do_not_touch_memory(lens, head_dim, head_dim_v):
    assert _wasted_tiles(lens) >= 1, "batch must over-provision the grid"
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    nheads = 11
    total = sum(lens)
    q = torch.randn(total, nheads, head_dim, dtype=dtype, device=device)
    k = torch.randn(total, nheads, head_dim, dtype=dtype, device=device)
    v = torch.randn(total, nheads, head_dim_v, dtype=dtype, device=device)

    # cu_seqlens is a prefix view of a buffer whose next element is huge, the
    # worst case for a kernel that reads one past the end.
    cu_buffer = torch.full((len(lens) + 2,), 1 << 30, dtype=torch.int32, device=device)
    cu_buffer[0] = 0
    cu_buffer[1 : len(lens) + 1] = torch.cumsum(torch.tensor(lens, dtype=torch.int32), dim=0).to(device)
    cu_seqlens = cu_buffer[: len(lens) + 1]

    canary = torch.full((total + CANARY_ROWS, nheads, head_dim_v), float("nan"), dtype=dtype, device=device)
    canary[:total].zero_()
    out = canary[:total]

    scale = head_dim**-0.5
    _flash_attn_fwd(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max(lens),
        max_seqlen_k=max(lens),
        softmax_scale=scale,
        causal=True,
        out=out,
    )
    torch.cuda.synchronize()

    assert torch.isnan(canary[total:].float()).all(), "extra grid tiles wrote past the end of the output"
    assert cu_buffer[len(lens) + 1].item() == 1 << 30
    torch.testing.assert_close(out.float(), _reference(q, k, v, lens, scale), atol=2e-2, rtol=2e-2)
