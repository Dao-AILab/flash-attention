import gc
import random
import time
from functools import wraps

import pytest
import torch

from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd, flash_attn_varlen_func
from flash_attn.cute.testing import generate_random_padding_mask


COMPUTE_CAPABILITY = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
IS_SM90 = COMPUTE_CAPABILITY == 9
IS_SM100_OR_SM110 = COMPUTE_CAPABILITY in (10, 11)
SUPPORTS_DETERMINISTIC_BWD = COMPUTE_CAPABILITY in (9, 10, 11)
USE_FAKE_TENSOR = False


def retry_on_oom(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except torch.OutOfMemoryError as e:
            if "out of memory" not in str(e).lower():
                raise
            if hasattr(_flash_attn_fwd, "compile_cache"):
                _flash_attn_fwd.compile_cache.clear()
            if hasattr(_flash_attn_bwd, "compile_cache"):
                _flash_attn_bwd.compile_cache.clear()
            gc.collect()
            torch.cuda.empty_cache()
            return func(*args, **kwargs)

    return wrapper


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    seqlen, num_kv_heads, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :].expand(seqlen, num_kv_heads, n_rep, head_dim)
    return hidden_states.reshape(seqlen, num_kv_heads * n_rep, head_dim)


def _make_cu_seqlens(seqlens, device="cuda") -> torch.Tensor:
    return torch.tensor([0] + list(torch.tensor(seqlens).cumsum(0).tolist()), device=device, dtype=torch.int32)


def _make_chunk_sizes(mode: str, batch_size: int, base_chunk_size: int, device="cuda") -> torch.Tensor:
    if mode == "constant":
        chunk_sizes = [base_chunk_size] * batch_size
    elif mode == "per_batch":
        chunk_sizes = [max(1, base_chunk_size + (i % 3 - 1) * 7) for i in range(batch_size)]
    elif mode == "tail_oversized":
        chunk_sizes = [base_chunk_size if i % 2 == 0 else base_chunk_size * 3 + 17 for i in range(batch_size)]
    else:
        raise ValueError(f"unknown chunk size mode: {mode}")
    return torch.tensor(chunk_sizes, device=device, dtype=torch.int32)


def _heads_for_mha_type(mha_type: str) -> tuple[int, int]:
    num_heads = 8
    if mha_type == "mha":
        return num_heads, num_heads
    if mha_type == "gqa":
        return num_heads, 2
    if mha_type == "mqa":
        return num_heads, 1
    raise ValueError(f"unknown mha_type: {mha_type}")


def _tolerances(dtype: torch.dtype) -> tuple[float, float]:
    # bf16 dK/dV can differ by about one ULP from the PyTorch reference because
    # the kernel and reference accumulate reductions in different orders.
    return (2e-2, 4e-2) if dtype == torch.bfloat16 else (3e-3, 3e-3)


def chunk_attn_ref_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    chunk_size: torch.Tensor,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Reference for chunk attention on packed varlen tensors.

    For batch b:
        keep(q, k) = ((q + seqlen_k - seqlen_q) // chunk_size[b] + 1) * chunk_size[b] > k
    """
    dtype = q.dtype
    head_dim = q.shape[-1]
    num_q_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    n_rep = num_q_heads // num_kv_heads
    softmax_scale = head_dim ** (-0.5) if softmax_scale is None else softmax_scale

    outputs = []
    for batch_idx in range(len(cu_seqlens_q) - 1):
        q_start, q_end = cu_seqlens_q[batch_idx].item(), cu_seqlens_q[batch_idx + 1].item()
        k_start, k_end = cu_seqlens_k[batch_idx].item(), cu_seqlens_k[batch_idx + 1].item()
        seqlen_q = q_end - q_start
        seqlen_k = k_end - k_start
        kq_diff = seqlen_k - seqlen_q
        chunk = chunk_size[batch_idx].item()

        q_b = q[q_start:q_end]
        k_b = _repeat_kv(k[k_start:k_end], n_rep)
        v_b = _repeat_kv(v[k_start:k_end], n_rep)

        scores = torch.bmm(q_b.permute(1, 0, 2), k_b.permute(1, 2, 0)).float() * softmax_scale
        q_range = torch.arange(seqlen_q, device=q.device).unsqueeze(1)
        k_range = torch.arange(seqlen_k, device=q.device).unsqueeze(0)
        chunk_mask = ((q_range + kq_diff) // chunk + 1) * chunk > k_range
        scores.masked_fill_(~chunk_mask.unsqueeze(0), float("-inf"))
        probs = torch.softmax(scores, dim=-1).to(dtype)
        outputs.append(torch.bmm(probs, v_b.permute(1, 0, 2)).permute(1, 0, 2))

    return torch.cat(outputs, dim=0)


def _run_chunk_varlen_case(
    *,
    seqlens_q: list[int],
    seqlens_k: list[int],
    head_dim: int,
    dtype: torch.dtype,
    mha_type: str,
    chunk_size: int,
    chunk_size_mode: str = "constant",
    deterministic: bool = False,
    kv_layout: str = "packed",
):
    assert len(seqlens_q) == len(seqlens_k)
    assert kv_layout in ("packed", "padded")
    device = "cuda"
    batch_size = len(seqlens_q)
    num_heads, num_kv_heads = _heads_for_mha_type(mha_type)
    chunk_size_tensor = _make_chunk_sizes(chunk_size_mode, batch_size, chunk_size, device=device)
    cu_seqlens_q = _make_cu_seqlens(seqlens_q, device=device)
    cu_seqlens_k = _make_cu_seqlens(seqlens_k, device=device)
    total_q = sum(seqlens_q)
    total_k = sum(seqlens_k)
    max_seqlen_q = max(seqlens_q)
    max_seqlen_k = max(seqlens_k)

    q_ref = torch.randn(
        total_q, num_heads, head_dim, device=device, dtype=dtype, requires_grad=True
    )
    k_ref = torch.randn(
        total_k, num_kv_heads, head_dim, device=device, dtype=dtype, requires_grad=True
    )
    v_ref = torch.randn(
        total_k, num_kv_heads, head_dim, device=device, dtype=dtype, requires_grad=True
    )

    out_ref = chunk_attn_ref_varlen(q_ref, k_ref, v_ref, cu_seqlens_q, cu_seqlens_k, chunk_size_tensor)

    q = q_ref.detach().clone().requires_grad_(True)
    if kv_layout == "packed":
        k = k_ref.detach().clone().requires_grad_(True)
        v = v_ref.detach().clone().requires_grad_(True)
        cu_seqlens_k_arg = cu_seqlens_k
        seqused_k_arg = None
    else:
        k_padded = torch.zeros(
            batch_size, max_seqlen_k, num_kv_heads, head_dim, device=device, dtype=dtype
        )
        v_padded = torch.zeros_like(k_padded)
        for batch_idx, seqlen in enumerate(seqlens_k):
            start, end = cu_seqlens_k[batch_idx].item(), cu_seqlens_k[batch_idx + 1].item()
            k_padded[batch_idx, :seqlen] = k_ref.detach()[start:end]
            v_padded[batch_idx, :seqlen] = v_ref.detach()[start:end]
        k = k_padded.requires_grad_(True)
        v = v_padded.requires_grad_(True)
        cu_seqlens_k_arg = None
        seqused_k_arg = torch.tensor(seqlens_k, device=device, dtype=torch.int32)

    out, _ = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k_arg,
        seqused_k=seqused_k_arg,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        causal=True,
        deterministic=deterministic,
        chunk_size=chunk_size_tensor,
    )

    fwd_tol, bwd_tol = _tolerances(dtype)
    torch.testing.assert_close(out_ref, out, atol=fwd_tol, rtol=fwd_tol)

    grad = torch.randn_like(out)
    out_ref.backward(grad)
    out.backward(grad)
    torch.testing.assert_close(q_ref.grad, q.grad, atol=bwd_tol, rtol=bwd_tol)
    if kv_layout == "packed":
        torch.testing.assert_close(k_ref.grad, k.grad, atol=bwd_tol, rtol=bwd_tol)
        torch.testing.assert_close(v_ref.grad, v.grad, atol=bwd_tol, rtol=bwd_tol)
    else:
        k_grad = torch.cat([k.grad[b, :seqlen] for b, seqlen in enumerate(seqlens_k)], dim=0)
        v_grad = torch.cat([v.grad[b, :seqlen] for b, seqlen in enumerate(seqlens_k)], dim=0)
        torch.testing.assert_close(k_ref.grad, k_grad, atol=bwd_tol, rtol=bwd_tol)
        torch.testing.assert_close(v_ref.grad, v_grad, atol=bwd_tol, rtol=bwd_tol)

    return out.detach(), q.detach(), k.detach(), v.detach(), chunk_size_tensor


def _run_flash_chunk_once(q, k, v, grad, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, chunk_size):
    q_run = q.detach().clone().requires_grad_(True)
    k_run = k.detach().clone().requires_grad_(True)
    v_run = v.detach().clone().requires_grad_(True)
    out, _ = flash_attn_varlen_func(
        q_run,
        k_run,
        v_run,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        causal=True,
        deterministic=True,
        chunk_size=chunk_size,
    )
    out.backward(grad)
    return out.detach(), q_run.grad.detach(), k_run.grad.detach(), v_run.grad.detach()


CHUNK_CASES = [
    pytest.param([41, 128, 257], [41, 128, 257], 8, "constant", id="equal-varlen-small-chunk"),
    pytest.param([68, 99], [112, 154], 4, "constant", id="k-longer-kqdiff-not-multiple"),
    pytest.param([128, 31, 256], [193, 203, 320], 60, "per_batch", id="mixed-qk-per-batch"),
    pytest.param([511, 37], [511, 37], 300, "tail_oversized", id="oversized-tail-chunk"),
    pytest.param([1024], [1024], 500, "constant", id="1k-long-chunk-500"),
]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mha_type", ["mha", "gqa", "mqa"])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("seqlens_q,seqlens_k,chunk_size,chunk_size_mode", CHUNK_CASES)
@retry_on_oom
def test_chunk_attn_varlen_output(
    seqlens_q,
    seqlens_k,
    chunk_size,
    chunk_size_mode,
    deterministic,
    head_dim,
    mha_type,
    dtype,
):
    if deterministic and not SUPPORTS_DETERMINISTIC_BWD:
        pytest.skip("deterministic backward is supported on SM90/SM100/SM110")
    seed = 1000 + head_dim + len(seqlens_q) * 17 + int(deterministic)
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.empty_cache()
    _run_chunk_varlen_case(
        seqlens_q=seqlens_q,
        seqlens_k=seqlens_k,
        head_dim=head_dim,
        dtype=dtype,
        mha_type=mha_type,
        chunk_size=chunk_size,
        chunk_size_mode=chunk_size_mode,
        deterministic=deterministic,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mha_type", ["mha", "gqa", "mqa"])
@pytest.mark.parametrize("kv_layout", ["packed", "padded"])
@pytest.mark.parametrize("head_dim", [64, 128])
@retry_on_oom
def test_chunk_attn_varlen_from_padding_masks(head_dim, kv_layout, mha_type, dtype):
    """Covers cu_seqlens_q with either packed KV or padded KV + seqused_k."""
    device = "cuda"
    seed = 1234 + head_dim + (0 if kv_layout == "packed" else 100)
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.empty_cache()

    batch_size = 5
    max_seqlen_q, max_seqlen_k = 257, 384
    query_padding_mask = generate_random_padding_mask(
        max_seqlen_q, batch_size, device, mode="random"
    )
    key_padding_mask = generate_random_padding_mask(
        max_seqlen_k, batch_size, device, mode="third"
    )
    seqlens_q = query_padding_mask.sum(-1).tolist()
    seqlens_k = key_padding_mask.sum(-1).tolist()
    # Keep every sequence non-empty so chunk causal rows always have at least one valid key.
    seqlens_q = [max(1, int(s)) for s in seqlens_q]
    seqlens_k = [max(sq, int(sk)) for sq, sk in zip(seqlens_q, seqlens_k)]

    _run_chunk_varlen_case(
        seqlens_q=seqlens_q,
        seqlens_k=seqlens_k,
        head_dim=head_dim,
        dtype=dtype,
        mha_type=mha_type,
        chunk_size=49,
        chunk_size_mode="per_batch",
        kv_layout=kv_layout,
    )


@pytest.mark.skipif(
    not SUPPORTS_DETERMINISTIC_BWD,
    reason="deterministic backward is supported on SM90/SM100/SM110",
)
@pytest.mark.parametrize(
    "seqlens_q,seqlens_k,chunk_size",
    [
        ([128, 192], [128, 192], 8),
        ([68, 99], [112, 154], 4),
        ([256, 37, 113], [320, 113, 203], 60),
    ],
)
@retry_on_oom
def test_chunk_attn_deterministic_reproducibility(seqlens_q, seqlens_k, chunk_size):
    """Repeated deterministic=True backward runs should be bit-identical."""
    seed = 2024 + chunk_size
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.empty_cache()

    device = "cuda"
    num_heads, num_kv_heads = _heads_for_mha_type("gqa")
    head_dim = 64
    dtype = torch.float16
    cu_seqlens_q = _make_cu_seqlens(seqlens_q, device=device)
    cu_seqlens_k = _make_cu_seqlens(seqlens_k, device=device)
    chunk_size_tensor = _make_chunk_sizes("per_batch", len(seqlens_q), chunk_size, device=device)
    q = torch.randn(sum(seqlens_q), num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(sum(seqlens_k), num_kv_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(sum(seqlens_k), num_kv_heads, head_dim, device=device, dtype=dtype)
    max_seqlen_q = max(seqlens_q)
    max_seqlen_k = max(seqlens_k)

    out_warmup, _, _, _ = _run_flash_chunk_once(
        q, k, v, torch.ones(sum(seqlens_q), num_heads, head_dim, device=device, dtype=dtype),
        cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, chunk_size_tensor,
    )
    grad = torch.randn_like(out_warmup)
    out_ref, dq_ref, dk_ref, dv_ref = _run_flash_chunk_once(
        q, k, v, grad, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, chunk_size_tensor
    )
    for _ in range(3):
        out_i, dq_i, dk_i, dv_i = _run_flash_chunk_once(
            q, k, v, grad, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, chunk_size_tensor
        )
        assert torch.equal(out_ref, out_i), "fwd output not bit-identical across deterministic runs"
        assert torch.equal(dq_ref, dq_i), "dq not bit-identical under deterministic=True"
        assert torch.equal(dk_ref, dk_i), "dk not bit-identical under deterministic=True"
        assert torch.equal(dv_ref, dv_i), "dv not bit-identical under deterministic=True"


@pytest.mark.skipif(
    not IS_SM100_OR_SM110,
    reason="regression exercises SM100/SM110 deterministic semaphore ordering",
)
def test_chunk_attn_sm100_deterministic_boundary_regression():
    """Regression for chunks whose boundary bumps n_block_max_for_m_block in SM100 bwd."""
    random.seed(60)
    torch.random.manual_seed(60)
    torch.cuda.empty_cache()
    _run_chunk_varlen_case(
        seqlens_q=[71, 119],
        seqlens_k=[127, 185],
        head_dim=64,
        dtype=torch.float16,
        mha_type="gqa",
        chunk_size=60,
        chunk_size_mode="constant",
        deterministic=True,
    )


def test_chunk_attn_api_validation():
    device = "cuda"
    q = torch.randn(4, 2, 64, device=device, dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    cu_seqlens = torch.tensor([0, 4], device=device, dtype=torch.int32)
    chunk_size = torch.tensor([2], device=device, dtype=torch.int32)

    with pytest.raises(AssertionError, match="causal"):
        flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=4,
            max_seqlen_k=4,
            causal=False,
            chunk_size=chunk_size,
        )
    with pytest.raises(AssertionError, match="global attention"):
        flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=4,
            max_seqlen_k=4,
            causal=True,
            window_size=(32, -1),
            chunk_size=chunk_size,
        )
    with pytest.raises(AssertionError, match="int32"):
        flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=4,
            max_seqlen_k=4,
            causal=True,
            chunk_size=chunk_size.to(torch.int64),
        )


def demo_chunk_attn_varlen():
    """Small executable demo: `python tests/cute/test_chunk_attn.py`."""
    device = "cuda"
    dtype = torch.bfloat16
    batch_size = 2
    seqlen = 1024
    num_heads = 16
    head_dim = 128
    chunk_size = 64
    torch.random.manual_seed(0)

    cu_seqlens = torch.tensor([0, seqlen, batch_size * seqlen], device=device, dtype=torch.int32)
    chunk_size_tensor = torch.tensor([chunk_size] * batch_size, device=device, dtype=torch.int32)
    q = torch.randn(batch_size * seqlen, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    for _ in range(10):
        flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=seqlen,
            max_seqlen_k=seqlen,
            causal=True,
            chunk_size=chunk_size_tensor,
        )

    torch.cuda.synchronize()
    start = time.time()
    repeat = 50
    for _ in range(repeat):
        out, _ = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=seqlen,
            max_seqlen_k=seqlen,
            causal=True,
            chunk_size=chunk_size_tensor,
        )
    torch.cuda.synchronize()
    elapsed_ms = (time.time() - start) * 1000 / repeat
    print(
        "flash_attn_varlen_func chunk demo: "
        f"batch={batch_size}, seqlen={seqlen}, heads={num_heads}, "
        f"head_dim={head_dim}, chunk_size={chunk_size}, time={elapsed_ms:.3f} ms, "
        f"out_norm={out.float().norm().item():.3f}"
    )


if __name__ == "__main__":
    demo_chunk_attn_varlen()