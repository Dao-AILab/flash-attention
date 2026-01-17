# mask mod test script
# REFACTORED to use _flash_attn_fwd as the kernel entrypoint
#
# Test Organization:
# - test_static_masks: Fast tests for masks that don't need per-seqlen compilation
#   (identity, document, block_diagonal, etc.) with comprehensive seqlen coverage
# - test_parameterized_masks: Slower tests for masks that require recompilation per
#   seqlen pair (causal, block_causal, sliding_window) with reduced seqlen coverage
#
# Usage:
#   pytest test_mask_mod.py::test_static_masks         # Run only fast tests
#   pytest test_mask_mod.py::test_parameterized_masks  # Run only slow tests
#   pytest test_mask_mod.py                            # Run all tests

import math
from unittest import mock

import pytest
import torch
import cutlass
import cutlass.cute as cute
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
import torch.nn.functional as F

from flash_attn.cute.interface import _flash_attn_fwd, _flash_attn_bwd
from flash_attn.cute.block_sparsity import (
    BlockSparseTensorsTorch,
    fast_sampling,
    normalize_block_sparse_config,
)
from flash_attn.cute import utils
from mask_mod_definitions import get_mask_pair, random_doc_id_tensor
COMPUTE_CAPABILITY = torch.cuda.get_device_capability()[0]


@pytest.fixture(autouse=True)
def reset_torch_state():
    """Reset torch dynamo/compile state between tests to avoid state pollution."""
    torch._dynamo.reset()
    torch.cuda.empty_cache()

    yield

    torch._dynamo.reset()
    torch.cuda.empty_cache()

def create_tensors(
    batch_size, seqlen_q, seqlen_k, nheads, nheads_kv, headdim, headdim_v, dtype
):
    device = "cuda"
    q = torch.randn(batch_size, seqlen_q, nheads, headdim, device=device, dtype=dtype)
    k = torch.randn(
        batch_size, seqlen_k, nheads_kv, headdim, device=device, dtype=dtype
    )
    v = torch.randn(
        batch_size, seqlen_k, nheads_kv, headdim_v, device=device, dtype=dtype
    )
    out = torch.empty(
        batch_size, seqlen_q, nheads, headdim_v, device=device, dtype=dtype
    )
    lse = torch.empty(batch_size, nheads, seqlen_q, device=device, dtype=torch.float32)

    return {
        "q": q,
        "k": k,
        "v": v,
        "out": out,
        "lse": lse,
    }


def compute_reference_flex_attn(tensors, mask_mod_flex, block_size: tuple[int, int] | None = None):
    """Compute reference using flex_attention for custom mask_mods"""
    batch_size, seqlen_q, nheads, headdim = tensors["q"].shape
    _, seqlen_k, nheads_kv, _ = tensors["k"].shape

    q = tensors["q"].transpose(1, 2)
    k = tensors["k"].transpose(1, 2)
    v = tensors["v"].transpose(1, 2)

    if nheads != nheads_kv:
        repeat_factor = nheads // nheads_kv
        k = k.repeat_interleave(repeat_factor, dim=1)
        v = v.repeat_interleave(repeat_factor, dim=1)

    scale = 1.0 / math.sqrt(headdim)

    # Handle identity (no masking) case
    if mask_mod_flex is None:
        out_ref = F.scaled_dot_product_attention(q, k, v, scale=scale)
        return out_ref.transpose(1, 2).contiguous()

    block_mask_kwargs = {}
    if block_size is not None:
        block_mask_kwargs["BLOCK_SIZE"] = block_size

    block_mask = create_block_mask(
        mask_mod_flex,
        B=batch_size,
        H=nheads,
        Q_LEN=seqlen_q,
        KV_LEN=seqlen_k,
        device=q.device,
        **block_mask_kwargs,
    )
    out_ref = flex_attention(q, k, v, block_mask=block_mask, scale=scale, enable_gqa=True)
    return out_ref.transpose(1, 2).contiguous()


def get_coarse_block_mask_pair(sparse_tile_m: int, tile_n: int, last_block: int):
    @fast_sampling
    @cute.jit
    def _cute_coarse_block_mask(
        batch: cute.TensorSSA,
        head: cute.TensorSSA,
        m_idx: cute.TensorSSA,
        n_idx: cute.TensorSSA,
        seqlen_info,
        aux_tensors,
    ) -> cute.TensorSSA:
        sparse_tile_m_ssa = utils.scalar_to_ssa(sparse_tile_m, cutlass.Int32)
        tile_n_ssa = utils.scalar_to_ssa(tile_n, cutlass.Int32)
        q_block = m_idx // sparse_tile_m_ssa
        n_block = n_idx // tile_n_ssa
        zero = utils.scalar_to_ssa(0, cutlass.Int32)
        one = utils.scalar_to_ssa(1, cutlass.Int32)
        last = utils.scalar_to_ssa(last_block, cutlass.Int32)
        return ((q_block == zero) & (n_block == zero)) | ((q_block == one) & (n_block == last))

    def _flex_coarse_block_mask(b, h, q_idx, kv_idx):
        q_block = q_idx // sparse_tile_m
        n_block = kv_idx // tile_n
        return ((q_block == 0) & (n_block == 0)) | ((q_block == 1) & (n_block == last_block))

    return _cute_coarse_block_mask, _flex_coarse_block_mask


SEQLEN_PAIRS_COMPREHENSIVE = [
    (1, 1),
    (64, 128),
    (128, 192),
    (256, 256),
    (239, 1),
    (799, 3),
    (113, 203),
    (113, 128),
    (128, 217),
    (113, 211),
    (108, 256),
    (256, 512),
    (384, 256),
    (640, 128),
    (512, 256),
    (1024, 1024),
    (1023, 1024),
    (1024, 1023),
    (4096, 4096),
    (4224, 4224),
]

SEQLEN_PAIRS_SMOKE = [
    (128, 128),
    (256, 256),
    (113, 203),
    (1024, 1024),
    (128, 8192)
]


def _run_mask_test(
    seqlen_q,
    seqlen_k,
    nheads,
    kv_mode,
    headdim,
    dtype,
    mask_name,
    window_size,
    window_left,
    window_right,
    tile_m,
    tile_n,
    use_block_sparsity,
    needs_backward=False,
):
    torch.manual_seed(42)

    if mask_name == "sliding_window":
        assert window_size is not None, (
            "window_size must be specified for sliding_window"
        )
        if seqlen_q > seqlen_k:
            pytest.skip(
                f"seqlen_q={seqlen_q} > seqlen_k={seqlen_k} not supported for sliding_window"
            )

    # Determine nheads_kv based on mode
    if kv_mode == "mha":
        nheads_kv = nheads
        pack_gqa = False
    elif kv_mode == "gqa":
        if COMPUTE_CAPABILITY < 9:
            pytest.xfail("pack_gqa requires SM90+")
        nheads_kv = nheads // 4
        pack_gqa = True
    elif kv_mode == "mqa":
        nheads_kv = 1
        pack_gqa = False
    else:
        raise ValueError(f"Unknown kv_mode: {kv_mode}")

    batch_size = 1
    headdim_v = headdim

    aux_tensors_arg = None
    mask_mod_cute, mask_mod_flex = get_mask_pair(
        mask_name, seqlen_q=seqlen_q, seqlen_k=seqlen_k, window_size=window_size
    )
    if mask_name == "document":
        doc_len = max(seqlen_q, seqlen_k)
        doc_ids = random_doc_id_tensor(nheads, batch_size, doc_len, device="cuda").to(
            dtype=torch.int32, device="cuda"
        )
        original_flex_mask = mask_mod_flex

        def mask_mod_flex(b, h, q_idx, kv_idx, doc_ids=doc_ids):
            return original_flex_mask(b, h, q_idx, kv_idx, doc_ids)

        aux_tensors_arg = [doc_ids]
    elif mask_name == "ima":
        bias_threshold = (seqlen_k // 4) * 3
        bias = torch.full((seqlen_k,), bias_threshold, dtype=torch.int32, device="cuda")
        original_flex_mask = mask_mod_flex

        def mask_mod_flex(b, h, q_idx, kv_idx, bias=bias):
            return original_flex_mask(b, h, q_idx, kv_idx, bias)

        aux_tensors_arg = [bias]
    causal = False

    if causal and seqlen_k < seqlen_q:
        pytest.skip("causal masking requires seqlen_k >= seqlen_q")

    tensors = create_tensors(
        batch_size, seqlen_q, seqlen_k, nheads, nheads_kv, headdim, headdim_v, dtype
    )

    # SM100 uses sparse_tile_m = 2*tile_m to match forward q_stage=2 pipelining
    if COMPUTE_CAPABILITY == 10:
        sparse_tile_m = 2 * tile_m
    else:
        sparse_tile_m = tile_m

    block_mask_nheads = 1 if pack_gqa else nheads
    bm = create_block_mask(
        mask_mod_flex,
        batch_size,
        block_mask_nheads,
        seqlen_q,
        seqlen_k,
        device="cuda",
        BLOCK_SIZE=(sparse_tile_m, tile_n),
    )
    (
        _seq_q,
        _seq_k,
        kv_mask_cnt,
        kv_mask_idx,
        full_kv_cnt,
        full_kv_idx,
        q_mask_cnt,
        q_mask_idx,
        full_q_cnt,
        full_q_idx,
        *_,
    ) = bm.as_tuple()

    # SM90 block-sparse backward expects BlockMask granularity (128, 128) regardless of fwd tiling.
    sparse_tile_m_bwd = sparse_tile_m
    if COMPUTE_CAPABILITY == 9 and use_block_sparsity and (sparse_tile_m, tile_n) != (128, 128):
        bm_bwd = create_block_mask(
            mask_mod_flex,
            batch_size,
            nheads,
            seqlen_q,
            seqlen_k,
            device="cuda",
            BLOCK_SIZE=(128, 128),
        )
        (
            _seq_q,
            _seq_k,
            _kv_mask_cnt,
            _kv_mask_idx,
            _full_kv_cnt,
            _full_kv_idx,
            q_mask_cnt,
            q_mask_idx,
            full_q_cnt,
            full_q_idx,
            *_,
        ) = bm_bwd.as_tuple()
        sparse_tile_m_bwd = 128

    softmax_scale = 1.0 / math.sqrt(headdim)

    block_sparse_mask_fwd = (
        BlockSparseTensorsTorch(
            mask_block_cnt=kv_mask_cnt,
            mask_block_idx=kv_mask_idx,
            full_block_cnt=full_kv_cnt,
            full_block_idx=full_kv_idx,
            block_size=(sparse_tile_m, tile_n),
        )
        if use_block_sparsity
        else None
    )

    # Backward uses Q-direction (transposed) sparse tensors
    block_sparse_mask_bwd = (
        BlockSparseTensorsTorch(
            mask_block_cnt=q_mask_cnt,
            mask_block_idx=q_mask_idx,
            full_block_cnt=full_q_cnt,
            full_block_idx=full_q_idx,
            block_size=(sparse_tile_m_bwd, tile_n),
        )
        if use_block_sparsity
        else None
    )

    out_tuple = _flash_attn_fwd(
        q=tensors["q"],
        k=tensors["k"],
        v=tensors["v"],
        out=tensors["out"],
        lse=tensors["lse"],
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        seqused_q=None,
        seqused_k=None,
        page_table=None,
        softmax_scale=softmax_scale,
        causal=causal,
        softcap=None,
        window_size_left=window_left,
        window_size_right=window_right,
        learnable_sink=None,
        m_block_size=tile_m,
        n_block_size=tile_n,
        pack_gqa=pack_gqa,
        _compute_capability=None,
        score_mod=None,
        mask_mod=mask_mod_cute,
        block_sparse_tensors=block_sparse_mask_fwd,
        return_lse=True,
        aux_tensors=aux_tensors_arg,
    )

    out_cute = out_tuple[0]
    lse_cute = out_tuple[1]
    tensors_fp32 = {
        k: v.float() if v.dtype in [torch.float16, torch.bfloat16] else v
        for k, v in tensors.items()
    }

    block_size = (tile_m, tile_n)
    out_ref_fp32 = compute_reference_flex_attn(tensors_fp32, mask_mod_flex, block_size)
    out_ref = compute_reference_flex_attn(tensors, mask_mod_flex, block_size)
    out_pt = out_ref.clone()

    # Check for invalid values
    assert out_cute.shape == out_ref_fp32.shape == out_ref.shape
    assert not torch.isnan(out_cute).any()
    assert not torch.isnan(out_ref_fp32).any()
    assert torch.isfinite(out_cute).all()
    assert torch.isfinite(out_ref_fp32).all()

    # Compute numerical tolerance (matching flash attention tests)
    fwd_atol = 2 * (out_ref_fp32 + 0.3 - 0.3 - out_ref_fp32).abs().max().item()
    rtol = 2

    ref_error = (out_ref - out_ref_fp32).abs().max().item()
    pt_error = (out_pt - out_ref_fp32).abs().max().item()
    cute_error = (out_cute - out_ref_fp32).abs().max().item()

    mask_desc = f"mask_mod={mask_name}"
    if mask_name == "sliding_window" and window_size is not None:
        mask_desc += f"(w={window_size})"

    print(
        f"\n{mask_desc} @ Q={seqlen_q}, K={seqlen_k}, H={nheads}/{nheads_kv} ({kv_mode}), "
        f"D={headdim}, M={tile_m}, N={tile_n}"
    )
    print("  Reference implementation: FlexAttention")
    print(f"  Reference vs FP32: {ref_error:.2e}")
    print(f"  PyTorch vs FP32: {pt_error:.2e}")
    print(f"  Kernel vs FP32: {cute_error:.2e}")
    print(f"  Tolerance: rtol={rtol} * {pt_error:.2e} + {fwd_atol:.2e}")
    print(f"  Error ratio: {cute_error / max(pt_error, 1e-10):.2f}")

    # Debug: show some sample values if error is large
    if cute_error > 1e-2:
        print(f"  DEBUG: Sample kernel output: {out_cute[0, 0, 0, :5]}")
        print(f"  DEBUG: Sample reference output: {out_ref_fp32[0, 0, 0, :5]}")
        print(f"  DEBUG: Max diff location: {(out_cute - out_ref_fp32).abs().argmax()}")
        max_diff_idx = (out_cute - out_ref_fp32).abs().argmax()
        max_diff_coords = torch.unravel_index(max_diff_idx, out_cute.shape)
        print(f"  DEBUG: Max diff at coords: {max_diff_coords}")
        print(f"  DEBUG: Kernel value: {out_cute[max_diff_coords]:.6f}")
        print(f"  DEBUG: Reference value: {out_ref_fp32[max_diff_coords]:.6f}")

    # Use the same assertion logic as FlashAttention tests
    assert cute_error <= rtol * pt_error + fwd_atol, (
        f"Kernel error {cute_error:.2e} exceeds {rtol}x PyTorch error {pt_error:.2e} + {fwd_atol:.2e}"
    )

    if needs_backward:
        q = tensors["q"]
        k = tensors["k"]
        v = tensors["v"]

        # Create grad_out once and reuse
        grad_out = torch.randn_like(out_cute)

        # Create block_mask for flex reference
        flex_block_mask = create_block_mask(
            mask_mod_flex, batch_size, nheads, seqlen_q, seqlen_k,
            device="cuda", BLOCK_SIZE=(tile_m, tile_n),
        )

        dq_cute, dk_cute, dv_cute = run_cute_mask_bwd(
            q, k, v, out_cute, lse_cute, grad_out, mask_mod_cute,
            block_sparse_mask_bwd=block_sparse_mask_bwd, tile_m=tile_m, tile_n=tile_n,
            aux_tensors=aux_tensors_arg,
        )
        _, dq_ref_fp32, dk_ref_fp32, dv_ref_fp32 = run_flex_reference_bwd(
            q, k, v, flex_block_mask, grad_out, dtype=torch.float32
        )
        _, dq_pt, dk_pt, dv_pt = run_flex_reference_bwd(
            q, k, v, flex_block_mask, grad_out
        )

        # Check for invalid values
        assert not torch.isnan(dq_cute).any(), "dQ contains NaN"
        assert not torch.isnan(dk_cute).any(), "dK contains NaN"
        assert not torch.isnan(dv_cute).any(), "dV contains NaN"

        bwd_rtol = 2
        min_seqlen = min(seqlen_q, seqlen_k)
        bwd_atol_floor = 1e-5 if min_seqlen >= 64 else 3e-5
        dq_atol = max(bwd_atol_floor, 2 * (dq_ref_fp32 + 0.3 - 0.3 - dq_ref_fp32).abs().max().item())
        dk_atol = max(bwd_atol_floor, 2 * (dk_ref_fp32 + 0.3 - 0.3 - dk_ref_fp32).abs().max().item())
        dv_atol = max(bwd_atol_floor, 2 * (dv_ref_fp32 + 0.3 - 0.3 - dv_ref_fp32).abs().max().item())

        dq_ref = dq_ref_fp32.to(dtype)
        dk_ref = dk_ref_fp32.to(dtype)
        dv_ref = dv_ref_fp32.to(dtype)

        pt_dq_err = (dq_pt - dq_ref).abs().max().item()
        pt_dk_err = (dk_pt - dk_ref).abs().max().item()
        pt_dv_err = (dv_pt - dv_ref).abs().max().item()

        cute_dq_err = (dq_cute - dq_ref).abs().max().item()
        cute_dk_err = (dk_cute - dk_ref).abs().max().item()
        cute_dv_err = (dv_cute - dv_ref).abs().max().item()

        print("  Backward comparison:")
        print(f"    dQ: PT err={pt_dq_err:.2e}, CuTE err={cute_dq_err:.2e}, atol={dq_atol:.2e}")
        print(f"    dK: PT err={pt_dk_err:.2e}, CuTE err={cute_dk_err:.2e}, atol={dk_atol:.2e}")
        print(f"    dV: PT err={pt_dv_err:.2e}, CuTE err={cute_dv_err:.2e}, atol={dv_atol:.2e}")

        assert cute_dq_err <= bwd_rtol * pt_dq_err + dq_atol, f"dQ error too large: {cute_dq_err:.2e}"
        assert cute_dk_err <= bwd_rtol * pt_dk_err + dk_atol, f"dK error too large: {cute_dk_err:.2e}"
        assert cute_dv_err <= bwd_rtol * pt_dv_err + dv_atol, f"dV error too large: {cute_dv_err:.2e}"


def test_mask_mod_ima_partial_block():
    _run_mask_test(
        seqlen_q=257,
        seqlen_k=257,
        nheads=1,
        kv_mode="mha",
        headdim=128,
        dtype=torch.bfloat16,
        mask_name="ima",
        window_size=None,
        window_left=None,
        window_right=None,
        tile_m=128,
        tile_n=128,
        use_block_sparsity=True,
        needs_backward=True,
    )


# Q boundary seqlens: NOT multiples of tile_m (128)
# These exercise the fix for is_full_block tiles not masking OOB Q rows in backward
Q_BOUNDARY_SEQLEN_PAIRS = [
    (200, 200),    # Last m_block: rows 128-199 valid, 200-255 should be masked
    (300, 300),    # Last m_block: rows 256-299 valid, 300-383 should be masked
    (129, 129),    # Just 1 element into second tile
    (255, 255),    # Just 1 element short of 2 full tiles
    (500, 512),    # Q boundary only (K aligned)
    (512, 500),    # K boundary only (Q aligned)
    (333, 444),    # Both non-aligned
]


@pytest.mark.parametrize("seqlen_q,seqlen_k", Q_BOUNDARY_SEQLEN_PAIRS)
@pytest.mark.parametrize("mask_name", ["block_diagonal", "document"])
def test_q_boundary_masking_block_sparse_bwd(seqlen_q, seqlen_k, mask_name):
    """Test Q boundary masking for block-sparse backward pass.

    This test specifically exercises the fix for the bug where Q rows beyond seqlen_q
    were not masked in backward pass for is_full_block=True tiles.

    The bug occurred because:
    - In forward, apply_mask_sm100 always checks both Q and K bounds
    - In backward, apply_mask_sm100_transposed with is_full_block=True only checked K bounds
    - Result: partial last m_blocks had unmasked garbage Q rows contributing to gradients

    Key conditions:
    - seqlen_q NOT a multiple of tile_m (128): creates partial last m_block
    - Block-sparse with mask_mod: exercises is_full_block=True path
    - Backward pass: where the bug manifested
    """
    _run_mask_test(
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        nheads=4,
        kv_mode="mha",
        headdim=128,
        dtype=torch.bfloat16,
        mask_name=mask_name,
        window_size=None,
        window_left=None,
        window_right=None,
        tile_m=128,
        tile_n=128,
        use_block_sparsity=True,
        needs_backward=True,
    )


@pytest.mark.skipif(COMPUTE_CAPABILITY != 10, reason="Test uses SM100 block mask conventions (2*tile_m)")
def test_single_doc_bwd_minimal():
    """Minimal test to isolate single-document backward pass bug.

    This test uses batch=1, nheads=1, and a single document (all same doc_id)
    to make debugging easier. The bug manifests as large numerical errors
    in dQ, dK, dV when blocks are classified as "full blocks" due to
    the mask returning True for all positions.

    Run with: pytest tests/cute/test_mask_mod.py::test_single_doc_bwd_minimal -v -s
    """
    import random
    random.seed(42)
    torch.manual_seed(42)

    seqlen_q = 384
    seqlen_k = 300
    batch_size = 1
    nheads = 1
    headdim = 128
    tile_m = 128
    tile_n = 128
    dtype = torch.bfloat16

    # Create single-document doc_ids (all same doc_id = 0)
    doc_ids = torch.zeros(batch_size, nheads, max(seqlen_q, seqlen_k), dtype=torch.int32, device="cuda")

    from mask_mod_definitions import get_mask_pair
    mask_mod_cute, mask_mod_flex = get_mask_pair("document", seqlen_q=seqlen_q, seqlen_k=seqlen_k)

    original_flex_mask = mask_mod_flex
    def mask_mod_flex(b, h, q_idx, kv_idx, doc_ids=doc_ids):
        return original_flex_mask(b, h, q_idx, kv_idx, doc_ids)

    aux_tensors_arg = [doc_ids]

    # Create tensors
    q = torch.randn(batch_size, seqlen_q, nheads, headdim, device="cuda", dtype=dtype)
    k = torch.randn(batch_size, seqlen_k, nheads, headdim, device="cuda", dtype=dtype)
    v = torch.randn(batch_size, seqlen_k, nheads, headdim, device="cuda", dtype=dtype)
    out = torch.empty(batch_size, seqlen_q, nheads, headdim, device="cuda", dtype=dtype)
    lse = torch.empty(batch_size, nheads, seqlen_q, device="cuda", dtype=torch.float32)

    sparse_tile_m = 2 * tile_m
    bm = create_block_mask(
        mask_mod_flex, batch_size, nheads, seqlen_q, seqlen_k,
        device="cuda", BLOCK_SIZE=(sparse_tile_m, tile_n),
    )
    (
        _seq_q, _seq_k,
        kv_mask_cnt, kv_mask_idx, full_kv_cnt, full_kv_idx,
        q_mask_cnt, q_mask_idx, full_q_cnt, full_q_idx, *_,
    ) = bm.as_tuple()

    block_sparse_mask_fwd = BlockSparseTensorsTorch(
        mask_block_cnt=kv_mask_cnt,
        mask_block_idx=kv_mask_idx,
        full_block_cnt=full_kv_cnt,
        full_block_idx=full_kv_idx,
        block_size=(sparse_tile_m, tile_n),
    )
    block_sparse_mask_bwd = BlockSparseTensorsTorch(
        mask_block_cnt=q_mask_cnt,
        mask_block_idx=q_mask_idx,
        full_block_cnt=full_q_cnt,
        full_block_idx=full_q_idx,
        block_size=(sparse_tile_m, tile_n),
    )


    out_tuple = _flash_attn_fwd(
        q=q, k=k, v=v, out=out, lse=lse,
        cu_seqlens_q=None, cu_seqlens_k=None,
        seqused_q=None, seqused_k=None, page_table=None,
        causal=False, softcap=None,
        window_size_left=-1, window_size_right=-1,
        m_block_size=tile_m, n_block_size=tile_n, pack_gqa=False,
        _compute_capability=None, score_mod=None,
        mask_mod=mask_mod_cute,
        block_sparse_tensors=block_sparse_mask_fwd,
        return_lse=True, aux_tensors=aux_tensors_arg,
    )
    out_cute = out_tuple[0]
    lse_cute = out_tuple[1]

    # Backward pass
    grad_out = torch.randn_like(out_cute)

    dq_cute, dk_cute, dv_cute = run_cute_mask_bwd(
        q, k, v, out_cute, lse_cute, grad_out, mask_mod_cute,
        block_sparse_mask_bwd=block_sparse_mask_bwd,
        tile_m=tile_m, tile_n=tile_n,
        aux_tensors=aux_tensors_arg,
    )

    flex_block_mask = create_block_mask(
        mask_mod_flex, batch_size, nheads, seqlen_q, seqlen_k,
        device="cuda", BLOCK_SIZE=(tile_m, tile_n),
    )
    out_ref, dq_ref, dk_ref, dv_ref = run_flex_reference_bwd(
        q, k, v, flex_block_mask, grad_out, dtype=torch.float32
    )

    # Compare
    dq_err = (dq_cute - dq_ref.to(dtype)).abs().max().item()
    dk_err = (dk_cute - dk_ref.to(dtype)).abs().max().item()
    dv_err = (dv_cute - dv_ref.to(dtype)).abs().max().item()

    print(f"dQ error: {dq_err:.2e}")
    print(f"dK error: {dk_err:.2e}")
    print(f"dV error: {dv_err:.2e}")

    # Assert gradients are correct (this will fail, demonstrating the bug)
    assert dq_err < 0.05, f"dQ error too large: {dq_err:.2e}"
    assert dk_err < 0.05, f"dK error too large: {dk_err:.2e}"
    assert dv_err < 0.05, f"dV error too large: {dv_err:.2e}"


@pytest.mark.parametrize("seqlen_q,seqlen_k", SEQLEN_PAIRS_COMPREHENSIVE)
@pytest.mark.parametrize("nheads", [16])
@pytest.mark.parametrize("kv_mode", ["mha", "gqa", "mqa"])
@pytest.mark.parametrize("headdim", [128])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("use_block_sparsity", [True, False])
@pytest.mark.parametrize(
    "mask_name",
    ["block_diagonal", "mini_causal"],
)
@pytest.mark.parametrize("tile_m,tile_n", [(128, 128), (128, 112)])
def test_static_masks(
    seqlen_q, seqlen_k, nheads, kv_mode, headdim, dtype, use_block_sparsity, mask_name, tile_m, tile_n
):
    """Test static masks that don't require recompilation per seqlen pair.

    Known good masks:
    - block_diagonal: Masks by 64-element diagonal blocks
    - mini_causal: Local causal within 128-element tiles
    """
    if COMPUTE_CAPABILITY == 10 and (tile_m, tile_n) != (128, 128):
        pytest.skip("TODO: Non-128x128 tiles currently not supported on SM 10.0. due to TMEM")

    _run_mask_test(
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        nheads=nheads,
        kv_mode=kv_mode,
        headdim=headdim,
        dtype=dtype,
        mask_name=mask_name,
        window_size=None,
        window_left=None,
        window_right=None,
        tile_m=tile_m,
        tile_n=tile_n,
        use_block_sparsity=use_block_sparsity,
        needs_backward=True,
    )


@pytest.mark.parametrize("seqlen_q,seqlen_k", SEQLEN_PAIRS_SMOKE)
@pytest.mark.parametrize("nheads", [16])
@pytest.mark.parametrize("kv_mode", ["mha", "gqa"])
@pytest.mark.parametrize("headdim", [128])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("use_block_sparsity", [True, False])
@pytest.mark.parametrize(
    "mask_name,window_size",
    [
        ("causal", None),
        ("block_causal", None),
        ("sliding_window", 128),
        ("sliding_window", 256),
        ("sliding_window", 512),
        ("document", None),
    ],
)
@pytest.mark.parametrize("tile_m,tile_n", [(128, 128), (128, 112), (64, 128)])
def test_parameterized_masks(
    seqlen_q, seqlen_k, nheads, kv_mode, headdim, dtype, use_block_sparsity, mask_name, window_size, tile_m, tile_n
):
    """Test parameterized masks that require recompilation per seqlen pair.

    Uses fewer seqlen combinations to reduce test time.

    Masks tested:
    - causal, block_causal: Require offset = seqlen_k - seqlen_q
    - sliding_window: Requires window size and offset parameters
    - document: Slower to check
    """
    if COMPUTE_CAPABILITY == 10 and (tile_m, tile_n) != (128, 128):
        pytest.skip("TODO: Non-128x128 tiles currently not supported on SM 10.0. due to TMEM")

    _run_mask_test(
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        nheads=nheads,
        kv_mode=kv_mode,
        headdim=headdim,
        dtype=dtype,
        mask_name=mask_name,
        window_size=window_size,
        window_left=None,
        window_right=None,
        tile_m=tile_m,
        tile_n=tile_n,
        use_block_sparsity=use_block_sparsity,
        needs_backward=True,
    )


def test_sm100_block_sparse_sink_all_masked():
    """Block-sparse regression for the sink path"""
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100-only test")
    device = "cuda"
    dtype = torch.bfloat16
    batch_size = 1
    seqlen_q = 256
    seqlen_k = 128
    nheads = 8
    headdim = 128
    q = torch.randn(batch_size, seqlen_q, nheads, headdim, dtype=dtype, device=device)
    k = torch.randn(batch_size, seqlen_k, nheads, headdim, dtype=dtype, device=device)
    v = torch.randn(batch_size, seqlen_k, nheads, headdim, dtype=dtype, device=device)
    learnable_sink = torch.full((nheads,), 0.5, dtype=torch.bfloat16, device=device)
    zero_cnt = torch.zeros((batch_size, nheads, 1), dtype=torch.int32, device=device)
    zero_idx = torch.zeros((batch_size, nheads, 1, 1), dtype=torch.int32, device=device)
    sparse = BlockSparseTensorsTorch(
        mask_block_cnt=zero_cnt,
        mask_block_idx=zero_idx,
        full_block_cnt=zero_cnt,
        full_block_idx=zero_idx,
        block_size=(256, 128),
    )
    softmax_scale = 1.0 / math.sqrt(headdim)
    _, lse = _flash_attn_fwd(
        q=q,
        k=k,
        v=v,
        softmax_scale=softmax_scale,
        causal=False,
        window_size_left=None,
        window_size_right=None,
        learnable_sink=learnable_sink,
        m_block_size=128,
        n_block_size=128,
        num_threads=384,
        pack_gqa=False,
        block_sparse_tensors=sparse,
        return_lse=True,
    )
    # Fully masked tile ⇒ probability mass sits entirely on the sink, so LSE equals sink logit.
    expected = learnable_sink.float()[None, :, None].expand_as(lse)
    assert torch.allclose(lse, expected, atol=0.0, rtol=0.0)


@pytest.mark.skipif(COMPUTE_CAPABILITY != 10, reason="SM100-only test")
def test_sm100_block_sparse_q_stage1():
    from flash_attn.cute import flash_fwd_sm100
    from flash_attn.cute.interface import _flash_attn_fwd

    observed = {}
    original_init = flash_fwd_sm100.FlashAttentionForwardSm100.__init__

    def wrapped_init(self, *args, **kwargs):
        observed["q_stage"] = kwargs.get("q_stage")
        return original_init(self, *args, **kwargs)

    with mock.patch.object(
        flash_fwd_sm100.FlashAttentionForwardSm100,
        "__init__",
        wrapped_init,
    ):
        compile_cache = dict(_flash_attn_fwd.compile_cache)
        _flash_attn_fwd.compile_cache.clear()
        try:
            _run_mask_test(
                seqlen_q=128,
                seqlen_k=128,
                nheads=4,
                kv_mode="mha",
                headdim=128,
                dtype=torch.bfloat16,
                mask_name="block_diagonal",
                window_size=None,
                window_left=None,
                window_right=None,
                tile_m=128,
                tile_n=128,
                use_block_sparsity=True,
                needs_backward=False,
            )
        finally:
            _flash_attn_fwd.compile_cache.clear()
            _flash_attn_fwd.compile_cache.update(compile_cache)
    assert observed.get("q_stage") == 1


@pytest.mark.skipif(COMPUTE_CAPABILITY != 10, reason="SM100-only test")
def test_sm100_block_sparse_coarse_blocks():
    torch.manual_seed(42)
    seqlen_q = 512
    seqlen_k = 512
    nheads = 4
    headdim = 128
    dtype = torch.bfloat16
    tile_m = 128
    tile_n = 128
    sparse_tile_m = 512
    batch_size = 1

    mask_mod_cute, mask_mod_flex = get_mask_pair(
        "block_diagonal", seqlen_q=seqlen_q, seqlen_k=seqlen_k, window_size=None
    )
    tensors = create_tensors(
        batch_size, seqlen_q, seqlen_k, nheads, nheads, headdim, headdim, dtype
    )

    bm = create_block_mask(
        mask_mod_flex,
        batch_size,
        nheads,
        seqlen_q,
        seqlen_k,
        device="cuda",
        BLOCK_SIZE=(sparse_tile_m, tile_n),
    )
    (
        _seq_q,
        _seq_k,
        kv_mask_cnt,
        kv_mask_idx,
        full_kv_cnt,
        full_kv_idx,
        *_,
    ) = bm.as_tuple()

    block_sparse_mask_fwd = BlockSparseTensorsTorch(
        mask_block_cnt=kv_mask_cnt,
        mask_block_idx=kv_mask_idx,
        full_block_cnt=full_kv_cnt,
        full_block_idx=full_kv_idx,
        block_size=(sparse_tile_m, tile_n),
    )

    out_cute, _ = _flash_attn_fwd(
        q=tensors["q"],
        k=tensors["k"],
        v=tensors["v"],
        out=tensors["out"],
        lse=tensors["lse"],
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        seqused_q=None,
        seqused_k=None,
        page_table=None,
        softmax_scale=1.0 / math.sqrt(headdim),
        causal=False,
        softcap=None,
        window_size_left=None,
        window_size_right=None,
        learnable_sink=None,
        m_block_size=tile_m,
        n_block_size=tile_n,
        pack_gqa=False,
        _compute_capability=None,
        score_mod=None,
        mask_mod=mask_mod_cute,
        block_sparse_tensors=block_sparse_mask_fwd,
        return_lse=True,
    )

    tensors_fp32 = {
        k: v.float() if v.dtype in [torch.float16, torch.bfloat16] else v
        for k, v in tensors.items()
    }
    out_ref_fp32 = compute_reference_flex_attn(
        tensors_fp32, mask_mod_flex, (sparse_tile_m, tile_n)
    )
    out_ref = compute_reference_flex_attn(tensors, mask_mod_flex, (sparse_tile_m, tile_n))

    assert out_cute.shape == out_ref_fp32.shape == out_ref.shape
    assert not torch.isnan(out_cute).any()
    assert not torch.isnan(out_ref_fp32).any()
    assert torch.isfinite(out_cute).all()
    assert torch.isfinite(out_ref_fp32).all()

    fwd_atol = 2 * (out_ref_fp32 + 0.3 - 0.3 - out_ref_fp32).abs().max().item()
    rtol = 2
    pt_error = (out_ref - out_ref_fp32).abs().max().item()
    cute_error = (out_cute - out_ref_fp32).abs().max().item()
    assert cute_error <= rtol * pt_error + fwd_atol, (
        f"Kernel error {cute_error:.2e} exceeds {rtol}x PyTorch error {pt_error:.2e} + {fwd_atol:.2e}"
    )


@pytest.mark.skipif(COMPUTE_CAPABILITY != 10, reason="SM100-only test")
def test_sm100_block_sparse_coarse_blocks_mismatch():
    torch.manual_seed(0)
    seqlen_q = 1024
    seqlen_k = 512
    nheads = 2
    headdim = 128
    dtype = torch.bfloat16
    tile_m = 128
    tile_n = 128
    sparse_tile_m = 512
    batch_size = 1

    mask_mod_cute, mask_mod_flex = get_coarse_block_mask_pair(
        sparse_tile_m, tile_n, last_block=3
    )
    tensors = create_tensors(
        batch_size, seqlen_q, seqlen_k, nheads, nheads, headdim, headdim, dtype
    )

    bm = create_block_mask(
        mask_mod_flex,
        batch_size,
        nheads,
        seqlen_q,
        seqlen_k,
        device="cuda",
        BLOCK_SIZE=(sparse_tile_m, tile_n),
    )
    (
        _seq_q,
        _seq_k,
        kv_mask_cnt,
        kv_mask_idx,
        full_kv_cnt,
        full_kv_idx,
        *_,
    ) = bm.as_tuple()

    block_sparse_mask_fwd = BlockSparseTensorsTorch(
        mask_block_cnt=kv_mask_cnt,
        mask_block_idx=kv_mask_idx,
        full_block_cnt=full_kv_cnt,
        full_block_idx=full_kv_idx,
        block_size=(sparse_tile_m, tile_n),
    )

    observed = {}
    original_normalize = normalize_block_sparse_config

    def wrapped_normalize(*args, **kwargs):
        normalized, pattern, q_subtile_factor = original_normalize(*args, **kwargs)
        observed["q_subtile_factor"] = q_subtile_factor
        return normalized, pattern, q_subtile_factor

    with mock.patch("flash_attn.cute.interface.normalize_block_sparse_config", wrapped_normalize):
        out_cute, _ = _flash_attn_fwd(
            q=tensors["q"],
            k=tensors["k"],
            v=tensors["v"],
            out=tensors["out"],
            lse=tensors["lse"],
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            seqused_q=None,
            seqused_k=None,
            page_table=None,
            softmax_scale=1.0 / math.sqrt(headdim),
            causal=False,
            softcap=None,
            window_size_left=None,
            window_size_right=None,
            learnable_sink=None,
            m_block_size=tile_m,
            n_block_size=tile_n,
            pack_gqa=False,
            _compute_capability=None,
            score_mod=None,
            mask_mod=mask_mod_cute,
            block_sparse_tensors=block_sparse_mask_fwd,
            return_lse=True,
        )
    assert observed.get("q_subtile_factor") == 2

    tensors_fp32 = {
        k: v.float() if v.dtype in [torch.float16, torch.bfloat16] else v
        for k, v in tensors.items()
    }
    out_ref_fp32 = compute_reference_flex_attn(
        tensors_fp32, mask_mod_flex, (sparse_tile_m, tile_n)
    )
    out_ref = compute_reference_flex_attn(tensors, mask_mod_flex, (sparse_tile_m, tile_n))

    assert out_cute.shape == out_ref_fp32.shape == out_ref.shape
    assert not torch.isnan(out_cute).any()
    assert not torch.isnan(out_ref_fp32).any()
    assert torch.isfinite(out_cute).all()
    assert torch.isfinite(out_ref_fp32).all()

    fwd_atol = 2 * (out_ref_fp32 + 0.3 - 0.3 - out_ref_fp32).abs().max().item()
    rtol = 2
    pt_error = (out_ref - out_ref_fp32).abs().max().item()
    cute_error = (out_cute - out_ref_fp32).abs().max().item()
    assert cute_error <= rtol * pt_error + fwd_atol, (
        f"Kernel error {cute_error:.2e} exceeds {rtol}x PyTorch error {pt_error:.2e} + {fwd_atol:.2e}"
    )


# =============================================================================
# Backward Helper Functions
# =============================================================================

def run_cute_mask_bwd(
    q, k, v, out, lse, grad_out, mask_mod_cute,
    block_sparse_mask_bwd=None, tile_m=128, tile_n=128,
    aux_tensors=None,
):
    """Run flash attention backward with mask_mod.

    Args:
        q, k, v: Input tensors in BSHD format
        out: Forward output tensor
        lse: Log-sum-exp from forward pass
        grad_out: Gradient of output
        mask_mod_cute: CuTE mask modification function
        block_sparse_mask_bwd: Block sparse tensors for backward pass
        tile_m, tile_n: Tile sizes
        aux_tensors: Auxiliary tensors for mask_mod (e.g., doc_ids for document masking)

    Returns (dq, dk, dv) all in BSHD format.
    """
    dq, dk, dv = _flash_attn_bwd(
        q=q,
        k=k,
        v=v,
        out=out,
        dout=grad_out,
        lse=lse,
        causal=False,
        m_block_size=tile_m,
        n_block_size=tile_n,
        mask_mod=mask_mod_cute,
        block_sparse_tensors=block_sparse_mask_bwd,
        aux_tensors=aux_tensors,
    )

    return dq, dk, dv


def run_flex_reference_bwd(q, k, v, block_mask, grad_out, dtype=None):
    """Run flex_attention forward + backward for reference.

    Args:
        q, k, v: Input tensors in BSHD format
        block_mask: Pre-created block mask for flex_attention
        grad_out: Gradient of output in BSHD format
        dtype: Optional dtype to cast inputs to (e.g., torch.float32 for reference)

    Returns (out, dq, dk, dv) all in BSHD format.
    """
    # Transpose to BHSD for flex_attention
    if dtype is not None:
        q_ref = q.transpose(1, 2).to(dtype).requires_grad_(True)
        k_ref = k.transpose(1, 2).to(dtype).requires_grad_(True)
        v_ref = v.transpose(1, 2).to(dtype).requires_grad_(True)
        grad_out_ref = grad_out.transpose(1, 2).to(dtype)
    else:
        q_ref = q.transpose(1, 2).requires_grad_(True)
        k_ref = k.transpose(1, 2).requires_grad_(True)
        v_ref = v.transpose(1, 2).requires_grad_(True)
        grad_out_ref = grad_out.transpose(1, 2)

    # Use flex_attention directly without torch.compile for backward tests
    # torch.compile can hang on certain mask patterns (e.g., mini_causal with float32)
    out_ref = flex_attention(q_ref, k_ref, v_ref, block_mask=block_mask, enable_gqa=True)
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(out_ref, (q_ref, k_ref, v_ref), grad_out_ref)

    # Transpose back to BSHD
    return (
        out_ref.transpose(1, 2),
        dq_ref.transpose(1, 2),
        dk_ref.transpose(1, 2),
        dv_ref.transpose(1, 2),
    )


def test_sm90_block_sparse_bwd_mismatched_q_block_granularity_error_message():
    if COMPUTE_CAPABILITY != 9:
        pytest.skip("SM90-only test")

    batch_size = 1
    seqlen_q = 256
    seqlen_k = 256
    nheads = 4
    nheads_kv = nheads
    headdim = 128
    dtype = torch.bfloat16
    tile_m = 80
    tile_n = 128

    tensors = create_tensors(batch_size, seqlen_q, seqlen_k, nheads, nheads_kv, headdim, headdim, dtype)
    mask_mod_cute, mask_mod_flex = get_mask_pair("block_diagonal", seqlen_q=seqlen_q, seqlen_k=seqlen_k)
    bm = create_block_mask(
        mask_mod_flex,
        batch_size,
        nheads,
        seqlen_q,
        seqlen_k,
        device="cuda",
        BLOCK_SIZE=(tile_m, tile_n),
    )
    (
        _seq_q,
        _seq_k,
        _kv_mask_cnt,
        _kv_mask_idx,
        _full_kv_cnt,
        _full_kv_idx,
        q_mask_cnt,
        q_mask_idx,
        full_q_cnt,
        full_q_idx,
        *_,
    ) = bm.as_tuple()

    block_sparse_mask_bwd = BlockSparseTensorsTorch(
        mask_block_cnt=q_mask_cnt,
        mask_block_idx=q_mask_idx,
        full_block_cnt=full_q_cnt,
        full_block_idx=full_q_idx,
        block_size=(tile_m, tile_n),
    )

    softmax_scale = 1.0 / math.sqrt(headdim)
    out = torch.empty(batch_size, seqlen_q, nheads, headdim, device="cuda", dtype=dtype)
    lse = torch.empty(batch_size, nheads, seqlen_q, device="cuda", dtype=torch.float32)
    grad_out = torch.randn_like(out)

    with pytest.raises(
        ValueError,
        match=r"Block sparsity expects sparse_block_size_q=128 for subtile_factor=2\.",
    ):
        _flash_attn_bwd(
            q=tensors["q"],
            k=tensors["k"],
            v=tensors["v"],
            out=out,
            dout=grad_out,
            lse=lse,
            softmax_scale=softmax_scale,
            causal=False,
            m_block_size=tile_m,
            n_block_size=tile_n,
            mask_mod=mask_mod_cute,
            block_sparse_tensors=block_sparse_mask_bwd,
        )


def test_gqa_block_sparse_broadcast_pattern_recompilation():
    """Test that different block sparse broadcast patterns trigger recompilation.

    This is a regression test for a bug where:
    1. First call with block_mask H=1 (broadcasts across all query heads)
    2. Second call with block_mask H=nheads (no broadcast)
    3. Second call incorrectly reused cached kernel from first call

    The fix adds block_sparse_broadcast_pattern to the compile key so that
    kernels are recompiled when broadcast patterns change. CuTe's
    mark_layout_dynamic() keeps stride=0 as static, so different broadcast
    patterns require different compiled kernels.
    """
    torch.manual_seed(42)

    batch_size = 2
    nheads = 8
    nheads_kv = 2
    seqlen = 257
    headdim = 64
    dtype = torch.bfloat16
    tile_m = 128
    tile_n = 128

    sparse_tile_m = 2 * tile_m if COMPUTE_CAPABILITY == 10 else tile_m

    def causal_mask(b, h, q, kv):
        return q >= kv

    mask_mod_cute, _ = get_mask_pair("causal", seqlen_q=seqlen, seqlen_k=seqlen)

    tensors = create_tensors(batch_size, seqlen, seqlen, nheads, nheads_kv, headdim, headdim, dtype)
    q, k, v = tensors["q"], tensors["k"], tensors["v"]
    grad_out = torch.randn_like(tensors["out"])
    softmax_scale = 1.0 / math.sqrt(headdim)

    def run_with_block_mask_nheads(block_mask_nheads: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bm = create_block_mask(
            causal_mask, batch_size, block_mask_nheads, seqlen, seqlen,
            device="cuda", BLOCK_SIZE=(sparse_tile_m, tile_n),
        )
        (
            _seq_q, _seq_k,
            kv_mask_cnt, kv_mask_idx, full_kv_cnt, full_kv_idx,
            q_mask_cnt, q_mask_idx, full_q_cnt, full_q_idx, *_,
        ) = bm.as_tuple()

        block_sparse_fwd = BlockSparseTensorsTorch(
            mask_block_cnt=kv_mask_cnt,
            mask_block_idx=kv_mask_idx,
            full_block_cnt=full_kv_cnt,
            full_block_idx=full_kv_idx,
            block_size=(sparse_tile_m, tile_n),
        )
        block_sparse_bwd = BlockSparseTensorsTorch(
            mask_block_cnt=q_mask_cnt,
            mask_block_idx=q_mask_idx,
            full_block_cnt=full_q_cnt,
            full_block_idx=full_q_idx,
            block_size=(sparse_tile_m, tile_n),
        )

        out = torch.empty_like(tensors["out"])
        lse = torch.empty_like(tensors["lse"])

        out_tuple = _flash_attn_fwd(
            q=q, k=k, v=v, out=out, lse=lse,
            softmax_scale=softmax_scale, causal=False,
            window_size_left=-1, window_size_right=-1,
            m_block_size=tile_m, n_block_size=tile_n, pack_gqa=False,
            mask_mod=mask_mod_cute, block_sparse_tensors=block_sparse_fwd,
            return_lse=True,
        )
        out_cute, lse_cute = out_tuple[0], out_tuple[1]

        dq, dk, dv = run_cute_mask_bwd(
            q, k, v, out_cute, lse_cute, grad_out, mask_mod_cute,
            block_sparse_mask_bwd=block_sparse_bwd, tile_m=tile_m, tile_n=tile_n,
        )
        return dq, dk, dv

    flex_block_mask = create_block_mask(
        causal_mask, batch_size, nheads, seqlen, seqlen,
        device="cuda", BLOCK_SIZE=(tile_m, tile_n),
    )
    _, dq_ref, dk_ref, dv_ref = run_flex_reference_bwd(q, k, v, flex_block_mask, grad_out, dtype=torch.float32)
    dq_ref, dk_ref, dv_ref = dq_ref.to(dtype), dk_ref.to(dtype), dv_ref.to(dtype)

    dq_broadcast, dk_broadcast, dv_broadcast = run_with_block_mask_nheads(1)
    dq_no_broadcast, dk_no_broadcast, dv_no_broadcast = run_with_block_mask_nheads(nheads)

    err_broadcast_dq = (dq_broadcast - dq_ref).abs().max().item()
    err_no_broadcast_dq = (dq_no_broadcast - dq_ref).abs().max().item()

    print("\nGQA block sparse broadcast pattern test:")
    print(f"  dQ error (H=1 broadcast): {err_broadcast_dq:.2e}")
    print(f"  dQ error (H={nheads} no broadcast): {err_no_broadcast_dq:.2e}")

    assert err_broadcast_dq < 0.1, f"Broadcast dQ error too large: {err_broadcast_dq:.2e}"
    assert err_no_broadcast_dq < 0.1, f"No-broadcast dQ error too large: {err_no_broadcast_dq:.2e}"


def test_gqa_expand_stride_zero_bug():
    """Test that GQA with expand()-created K/V tensors works correctly.

    This is a regression test for bugs with expand()-created tensors:

    Forward bug: cute.assume() fails when tensor strides are Python int 0
    (from expand()) instead of MLIR values.
    Error: AttributeError: 'int' object has no attribute 'type'

    Backward bug: mark_layout_dynamic fails with expanded tensors.
    Error: RuntimeError: Expected strides[leading_dim] == 1, but got N.

    Trigger: expand() + transpose() creates stride=0 dimensions (GQA pattern).
    """
    torch.manual_seed(42)

    batch_size = 1
    seqlen = 2048
    headdim = 128
    n_heads = 4
    n_kv_heads = 1
    dtype = torch.bfloat16
    device = "cuda"

    q = torch.randn(batch_size, seqlen, n_heads, headdim, device=device, dtype=dtype)
    k_orig = torch.randn(batch_size, seqlen, n_kv_heads, headdim, device=device, dtype=dtype)
    v_orig = torch.randn(batch_size, seqlen, n_kv_heads, headdim, device=device, dtype=dtype)

    k = k_orig.expand(batch_size, seqlen, n_heads, headdim)
    v = v_orig.expand(batch_size, seqlen, n_heads, headdim)

    assert k.stride()[2] == 0, "K should have stride=0 in head dim from expand()"
    assert v.stride()[2] == 0, "V should have stride=0 in head dim from expand()"

    out = torch.empty_like(q)
    lse = torch.empty(batch_size, n_heads, seqlen, device=device, dtype=torch.float32)
    softmax_scale = 1.0 / math.sqrt(headdim)

    out_tuple = _flash_attn_fwd(
        q=q, k=k, v=v, out=out, lse=lse,
        softmax_scale=softmax_scale,
        causal=True,
        m_block_size=128, n_block_size=128,
        return_lse=True,
    )
    out_fwd, lse_fwd = out_tuple[0], out_tuple[1]

    assert not torch.isnan(out_fwd).any(), "Forward output contains NaN"
    assert torch.isfinite(out_fwd).all(), "Forward output contains non-finite values"

    tensors_for_ref = {"q": q, "k": k, "v": v}
    tensors_fp32 = {"q": q.float(), "k": k.float(), "v": v.float()}

    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    out_ref = compute_reference_flex_attn(tensors_for_ref, causal_mask)
    out_ref_fp32 = compute_reference_flex_attn(tensors_fp32, causal_mask)

    fwd_atol = 2 * (out_ref_fp32 + 0.3 - 0.3 - out_ref_fp32).abs().max().item()
    rtol = 2
    pt_error = (out_ref - out_ref_fp32).abs().max().item()
    cute_error = (out_fwd - out_ref_fp32).abs().max().item()

    print(f"\nGQA expand stride=0 test:")
    print(f"  Forward: kernel err={cute_error:.2e}, ref err={pt_error:.2e}, atol={fwd_atol:.2e}")
    assert cute_error <= rtol * pt_error + fwd_atol, (
        f"Forward error {cute_error:.2e} exceeds {rtol}x ref error {pt_error:.2e} + {fwd_atol:.2e}"
    )

    grad_out = torch.randn_like(out_fwd)
    dq, dk, dv = _flash_attn_bwd(
        q=q, k=k, v=v, out=out_fwd, dout=grad_out, lse=lse_fwd,
        softmax_scale=softmax_scale,
        causal=True,
        m_block_size=128, n_block_size=128,
    )

    assert not torch.isnan(dq).any(), "dQ contains NaN"
    assert not torch.isnan(dk).any(), "dK contains NaN"
    assert not torch.isnan(dv).any(), "dV contains NaN"

    flex_block_mask = create_block_mask(
        causal_mask, batch_size, n_heads, seqlen, seqlen,
        device=device, BLOCK_SIZE=(128, 128),
    )
    _, dq_ref, dk_ref, dv_ref = run_flex_reference_bwd(q, k, v, flex_block_mask, grad_out, dtype=torch.float32)

    bwd_rtol = 2
    bwd_atol_floor = 1e-5

    dq_atol = max(bwd_atol_floor, 2 * (dq_ref + 0.3 - 0.3 - dq_ref).abs().max().item())
    dk_atol = max(bwd_atol_floor, 2 * (dk_ref + 0.3 - 0.3 - dk_ref).abs().max().item())
    dv_atol = max(bwd_atol_floor, 2 * (dv_ref + 0.3 - 0.3 - dv_ref).abs().max().item())

    _, dq_pt, dk_pt, dv_pt = run_flex_reference_bwd(q, k, v, flex_block_mask, grad_out)

    pt_dq_err = (dq_pt - dq_ref.to(dtype)).abs().max().item()
    pt_dk_err = (dk_pt - dk_ref.to(dtype)).abs().max().item()
    pt_dv_err = (dv_pt - dv_ref.to(dtype)).abs().max().item()

    cute_dq_err = (dq - dq_ref.to(dtype)).abs().max().item()
    cute_dk_err = (dk - dk_ref.to(dtype)).abs().max().item()
    cute_dv_err = (dv - dv_ref.to(dtype)).abs().max().item()

    print(f"  Backward dQ: kernel err={cute_dq_err:.2e}, ref err={pt_dq_err:.2e}, atol={dq_atol:.2e}")
    print(f"  Backward dK: kernel err={cute_dk_err:.2e}, ref err={pt_dk_err:.2e}, atol={dk_atol:.2e}")
    print(f"  Backward dV: kernel err={cute_dv_err:.2e}, ref err={pt_dv_err:.2e}, atol={dv_atol:.2e}")

    assert cute_dq_err <= bwd_rtol * pt_dq_err + dq_atol, f"dQ error too large: {cute_dq_err:.2e}"
    assert cute_dk_err <= bwd_rtol * pt_dk_err + dk_atol, f"dK error too large: {cute_dk_err:.2e}"
    assert cute_dv_err <= bwd_rtol * pt_dv_err + dv_atol, f"dV error too large: {cute_dv_err:.2e}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
