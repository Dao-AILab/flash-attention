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
from typing import Optional

import pytest
import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
import torch.nn.functional as F

from flash_attn.cute.interface import _flash_attn_fwd
from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch
from flash_attn.cute.mask_definitions import (
    get_mask_pair,
    STATIC_MASKS,
    random_doc_id_tensor,
)
from flash_attn.cute.testing import attention_ref


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
        "q": q.contiguous(),
        "k": k.contiguous(),
        "v": v.contiguous(),
        "out": out.contiguous(),
        "lse": lse.contiguous(),
    }


def compute_reference_flash_attn(tensors, causal, window_size, dtype_ref, upcast=True):
    """Compute reference using FlashAttention's attention_ref function"""
    q = tensors["q"].to(dtype_ref)
    k = tensors["k"].to(dtype_ref)
    v = tensors["v"].to(dtype_ref)

    out_ref, attn_ref = attention_ref(
        q,
        k,
        v,
        query_padding_mask=None,
        key_padding_mask=None,
        causal=causal,
        window_size=window_size,
        upcast=upcast,
        reorder_ops=False,
    )

    return out_ref


def compute_reference_flex_attn(tensors, mask_mod_flex, block_size: Optional[tuple[int, int]] = None):
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
    out_ref = flex_attention(q, k, v, block_mask=block_mask, scale=scale)
    return out_ref.transpose(1, 2).contiguous()


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
    elif kv_mode == "gqa":
        nheads_kv = nheads // 2
    elif kv_mode == "mqa":
        nheads_kv = 1
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
    causal = False

    if causal and seqlen_k < seqlen_q:
        pytest.skip("causal masking requires seqlen_k >= seqlen_q")

    tensors = create_tensors(
        batch_size, seqlen_q, seqlen_k, nheads, nheads_kv, headdim, headdim_v, dtype
    )

    # Compute block sparsity for mask_mod
    bm = create_block_mask(
        mask_mod_flex,
        batch_size,
        nheads,
        seqlen_q,
        seqlen_k,
        device="cuda",
        BLOCK_SIZE=(tile_m, tile_n),
    )
    _, _, mask_cnt, mask_idx, full_cnt, full_idx, *_ = bm.as_tuple()

    softmax_scale = 1.0 / math.sqrt(headdim)

    # if full_cnt is not None:
    #     print(f"Block sparsity info for {mask_name}:")
    #     print(f"  full_cnt shape: {full_cnt.shape}")
    #     print(f"  full_idx shape: {full_idx.shape}")
    #     print(f"  mask_cnt shape: {mask_cnt.shape}")
    #     print(f"  mask_idx shape: {mask_idx.shape}")
    #     print(f"  full_cnt: {full_cnt}")
    #     print(f"  full_idx: {full_idx}")
    #     print(f"  mask_cnt: {mask_cnt}")
    #     print(f"  mask_idx: {mask_idx}")
    #     if full_cnt[0,0,0] > 0:
    #         print(f"  First Q block - full indices: {full_idx[0,0,0,:full_cnt[0,0,0].item()]}")
    #     if mask_cnt[0,0,0] > 0:
    #         print(f"  First Q block - mask indices: {mask_idx[0,0,0,:mask_cnt[0,0,0].item()]}")
    block_sparse_mask = BlockSparseTensorsTorch(
        mask_block_cnt=mask_cnt,
        mask_block_idx=mask_idx,
        full_block_cnt=full_cnt,
        full_block_idx=full_idx,
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
        num_threads=384,
        pack_gqa=False,
        _compute_capability=None,
        score_mod=None,
        mask_mod=mask_mod_cute,
        block_sparse_tensors=block_sparse_mask,
        return_lse=True,
        aux_tensors=aux_tensors_arg,
    )

    out_cute = out_tuple[0]
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


@pytest.mark.parametrize("seqlen_q,seqlen_k", SEQLEN_PAIRS_COMPREHENSIVE)
@pytest.mark.parametrize("nheads", [16])
@pytest.mark.parametrize("kv_mode", ["mha", "gqa", "mqa"])
@pytest.mark.parametrize("headdim", [128])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize(
    "mask_name",
    ["block_diagonal", "mini_causal"],
)
@pytest.mark.parametrize("tile_m,tile_n", [(128, 128), (128, 112)])
def test_static_masks(
    seqlen_q, seqlen_k, nheads, kv_mode, headdim, dtype, mask_name, tile_m, tile_n
):
    """Test static masks that don't require recompilation per seqlen pair.

    Known good masks:
    - block_diagonal: Masks by 64-element diagonal blocks
    - mini_causal: Local causal within 128-element tiles
    """
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
    )


@pytest.mark.parametrize("seqlen_q,seqlen_k", SEQLEN_PAIRS_SMOKE)
@pytest.mark.parametrize("nheads", [16])
@pytest.mark.parametrize("kv_mode", ["mha"])
@pytest.mark.parametrize("headdim", [128])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
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
    seqlen_q, seqlen_k, nheads, kv_mode, headdim, dtype, mask_name, window_size, tile_m, tile_n
):
    """Test parameterized masks that require recompilation per seqlen pair.

    Uses fewer seqlen combinations to reduce test time.

    Masks tested:
    - causal, block_causal: Require offset = seqlen_k - seqlen_q
    - sliding_window: Requires window size and offset parameters
    - document: Slower to check
    """
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
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
