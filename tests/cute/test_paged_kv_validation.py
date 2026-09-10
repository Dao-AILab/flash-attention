"""Host-side validation tests for paged-KV configurations.

These tests do NOT require a GPU. They exercise the early input validation in
``_flash_attn_fwd`` that rejects paged-KV configs whose ``n_block_size`` (tile_n)
is not divisible by the cp.async KV-load thread count. Such configs would
otherwise fail with an opaque error deep inside JIT compilation (or, worse,
silently drop KV rows).

The pattern mirrors the fast two-pass test workflow: we run under
``FakeTensorMode`` (no GPU memory) and replace ``cute.compile`` with a sentinel
so that reaching compilation is observable without a CUDA device. The kernel
arch is forced via the private ``_arch`` argument so the selected forward path
is deterministic on any host.
"""

import pytest

# CUDA / CuTeDSL are Linux+NVIDIA only; skip cleanly where they are unavailable
# (e.g. macOS dev hosts) instead of erroring at collection time.
torch = pytest.importorskip("torch")
pytest.importorskip("cutlass")

from torch._subclasses.fake_tensor import FakeTensorMode  # noqa: E402

from flash_attn.cute import interface  # noqa: E402
from flash_attn.cute.interface import _flash_attn_fwd  # noqa: E402


class _CompileReached(Exception):
    """Raised by the stubbed ``cute.compile`` to mark that host validation passed."""


def _install_compile_sentinel(monkeypatch):
    """Replace cute.compile with a sentinel and reset the compile cache.

    If host validation passes, control reaches ``cute.compile`` and the sentinel
    fires; we treat that as "the config was accepted".
    """

    def _sentinel(*args, **kwargs):
        raise _CompileReached

    monkeypatch.setattr(interface.cute, "compile", _sentinel)
    # Ensure we never short-circuit on a previously cached kernel.
    monkeypatch.setattr(_flash_attn_fwd, "compile_cache", {})


def _make_paged_inputs(
    *,
    batch_size=1,
    seqlen_q=128,
    num_head=4,
    num_head_kv=4,
    head_dim=128,
    head_dim_v=128,
    num_pages=4,
    page_size=64,
    max_num_pages_per_seq=4,
    dtype=torch.bfloat16,
):
    """Build fake (meta) tensors for a paged-KV forward call."""
    q = torch.empty(batch_size, seqlen_q, num_head, head_dim, dtype=dtype)
    k = torch.empty(num_pages, page_size, num_head_kv, head_dim, dtype=dtype)
    v = torch.empty(num_pages, page_size, num_head_kv, head_dim_v, dtype=dtype)
    page_table = torch.zeros(batch_size, max_num_pages_per_seq, dtype=torch.int32)
    return q, k, v, page_table


@pytest.mark.parametrize("arch", [90, 100])
def test_paged_kv_indivisible_tile_n_raises(monkeypatch, arch):
    """tile_n not divisible by the KV-load thread count must raise ValueError.

    tile_n is forced to 96 via ``tile_mn``. On SM90 the cp.async loader uses 128
    threads; on SM100 (q_stage == 1 here, since seqlen_q == tile_m) it also uses
    128. 96 % 128 != 0, so the config is rejected before compilation.
    """
    _install_compile_sentinel(monkeypatch)
    with FakeTensorMode():
        q, k, v, page_table = _make_paged_inputs(page_size=64)
        with pytest.raises(ValueError, match=r"cp\.async KV-load path"):
            _flash_attn_fwd(
                q,
                k,
                v,
                page_table=page_table,
                tile_mn=(128, 96),  # force n_block_size = 96
                _arch=arch,
            )


@pytest.mark.parametrize("arch", [90, 100])
def test_paged_kv_divisible_tile_n_passes_guard(monkeypatch, arch):
    """A valid paged config (tile_n divisible by load threads) clears the guard.

    tile_n is forced to 128 (divisible by both 128 and 64), with page_size=64 so
    the cp.async path is selected. The new validation must NOT raise; execution
    should reach the stubbed ``cute.compile`` (``_CompileReached``).
    """
    _install_compile_sentinel(monkeypatch)
    with FakeTensorMode():
        q, k, v, page_table = _make_paged_inputs(page_size=64)
        with pytest.raises(_CompileReached):
            _flash_attn_fwd(
                q,
                k,
                v,
                page_table=page_table,
                tile_mn=(128, 128),  # divisible by the KV-load thread count
                _arch=arch,
            )


def test_paged_kv_tma_path_skips_guard(monkeypatch):
    """When page_size == tile_n the TMA path is used and the guard is bypassed.

    Even with tile_n=96 (not divisible by 128), page_size == tile_n means
    PagedKVManager's cp.async loader is never used, so the config is accepted and
    reaches compilation.
    """
    _install_compile_sentinel(monkeypatch)
    with FakeTensorMode():
        q, k, v, page_table = _make_paged_inputs(page_size=96, num_pages=4)
        with pytest.raises(_CompileReached):
            _flash_attn_fwd(
                q,
                k,
                v,
                page_table=page_table,
                tile_mn=(128, 96),  # page_size == tile_n -> TMA path
                _arch=90,
            )
