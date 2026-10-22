import pytest
import torch

# flash_attn/ops/triton/linear.py imports triton.ops.matmul_perf_model, which
# only exists in triton < 3.0
pytest.importorskip("triton.ops")

from flash_attn.ops.triton.linear import triton_linear_act  # noqa: E402


@pytest.mark.parametrize(
    "M, N, K",
    [
        (513, 1000, 256),  # tails on both output axes
        (5, 17, 9),
        (512, 1024, 256),  # tile-aligned control
    ],
)
@pytest.mark.parametrize("with_bias", [False, True])
def test_linear_act_non_tile_aligned(M, N, K, with_bias):
    """kernel_fwd's output store must respect the (rm < M) & (rn < N) bounds
    mask: the host grid rounds up with cdiv, so unmasked tail tiles write past
    the end of the output and onto the next row of it."""
    torch.manual_seed(0)
    device = "cuda"
    x = torch.randn(M, K, device=device, dtype=torch.float16)
    w = torch.randn(N, K, device=device, dtype=torch.float16)
    bias = torch.randn(N, device=device, dtype=torch.float16) if with_bias else None

    ref = x.float() @ w.float().t()
    if bias is not None:
        ref += bias.float()

    for _ in range(3):  # the in-bounds corruption is a tile race, so repeat
        out = triton_linear_act(x, w, bias)
        torch.testing.assert_close(out.float(), ref, atol=1e-1, rtol=1e-2)
