# mask mod test script

import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import pytest
import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
import torch.nn.functional as F

from flash_attn.cute.block_sparsity import compute_block_sparsity
from flash_attn.cute.flash_fwd import (
    FlashAttentionForwardSm80,
    FlashAttentionForwardSm90,
)
from flash_attn.cute.flash_fwd_sm100 import FlashAttentionForwardSm100
from flash_attn.cute.mask_definitions import MASK_FUNCTIONS, flex_causal_mask, create_flex_sliding_window_mask, create_cute_sliding_window_mask
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


def compile_and_run_kernel(
    tensors,
    mask_mod_cute,
    causal,
    is_local,
    window_left,
    window_right,
    tile_m,
    tile_n,
    full_block_cnt=None,
    full_block_idx=None,
    mask_block_cnt=None,
    mask_block_idx=None,
):
    dtype_map = {
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
        torch.float32: cutlass.Float32,
    }
    cute_dtype = dtype_map[tensors["q"].dtype]

    batch_size, seqlen_q, nheads, headdim = tensors["q"].shape
    _, seqlen_k, nheads_kv, _ = tensors["k"].shape
    headdim_v = tensors["v"].shape[-1]

    compute_capability = torch.cuda.get_device_capability()
    if compute_capability >= (10, 0):
        kernel_class = FlashAttentionForwardSm100
    elif compute_capability >= (9, 0):
        kernel_class = FlashAttentionForwardSm90
    else:
        kernel_class = FlashAttentionForwardSm80

    qhead_per_kvhead = nheads // nheads_kv
    kernel = kernel_class(
        cute_dtype,
        headdim,
        headdim_v,
        qhead_per_kvhead,
        is_causal=causal,
        is_local=is_local,
        pack_gqa=False,
        tile_m=tile_m,
        tile_n=tile_n,
        num_stages=2,
        num_threads=384,
        intra_wg_overlap=True,
        mma_pv_is_rs=True,
        mask_mod=mask_mod_cute,
        has_buffers=False,
        Q_in_regs=False,
    )

    softmax_scale = 1.0 / math.sqrt(headdim)
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    q_cute = from_dlpack(tensors["q"].detach(), assumed_align=16).mark_layout_dynamic(
        leading_dim=tensors["q"].ndim - 1
    )
    k_cute = from_dlpack(tensors["k"].detach(), assumed_align=16).mark_layout_dynamic(
        leading_dim=tensors["k"].ndim - 1
    )
    v_cute = from_dlpack(tensors["v"].detach(), assumed_align=16).mark_layout_dynamic(
        leading_dim=tensors["v"].ndim - 1
    )
    out_cute = from_dlpack(
        tensors["out"].detach(), assumed_align=16
    ).mark_layout_dynamic(leading_dim=tensors["out"].ndim - 1)
    lse_cute = from_dlpack(
        tensors["lse"].detach(), assumed_align=4
    ).mark_layout_dynamic(leading_dim=tensors["lse"].ndim - 1)

    full_block_cnt_cute = (
        from_dlpack(full_block_cnt.detach(), assumed_align=4)
        if full_block_cnt is not None
        else None
    )
    full_block_idx_cute = (
        from_dlpack(full_block_idx.detach(), assumed_align=4)
        if full_block_idx is not None
        else None
    )
    mask_block_cnt_cute = (
        from_dlpack(mask_block_cnt.detach(), assumed_align=4)
        if mask_block_cnt is not None
        else None
    )
    mask_block_idx_cute = (
        from_dlpack(mask_block_idx.detach(), assumed_align=4)
        if mask_block_idx is not None
        else None
    )

    # Window parameters for is_local
    window_left_cute = (
        cutlass.Int32(window_left) if window_left is not None else None
    )
    window_right_cute = (
        cutlass.Int32(window_right) if window_right is not None else None
    )

    compiled = cute.compile(
        kernel,
        q_cute,
        k_cute,
        v_cute,
        out_cute,
        lse_cute,
        softmax_scale,
        current_stream,
        None,  # cu_seqlens_q
        None,  # cu_seqlens_k
        None,  # seqused_q
        None,  # seqused_k
        None,  # page_table
        window_left_cute,
        window_right_cute,
        None,  # learnable_sink
        full_block_cnt_cute,
        full_block_idx_cute,
        mask_block_cnt_cute,
        mask_block_idx_cute,
        None,  # buffers
    )

    compiled(
        q_cute,
        k_cute,
        v_cute,
        out_cute,
        lse_cute,
        softmax_scale,
        current_stream,
        None,  # cu_seqlens_q
        None,  # cu_seqlens_k
        None,  # seqused_q
        None,  # seqused_k
        None,  # page_table
        window_left_cute,
        window_right_cute,
        None,  # learnable_sink
        full_block_cnt_cute,
        full_block_idx_cute,
        mask_block_cnt_cute,
        mask_block_idx_cute,
        None,  # buffers
    )

    torch.cuda.synchronize()
    return tensors["out"]


def compute_reference_flash_attn(
    tensors, causal, window_size, dtype_ref, upcast=True
):
    """Compute reference using FlashAttention's attention_ref function"""
    batch_size, seqlen_q, nheads, headdim = tensors["q"].shape
    _, seqlen_k, nheads_kv, _ = tensors["k"].shape
    
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


def compute_reference_flex_attn(
    tensors, mask_mod_flex, mask_mod_name, tile_m, tile_n
):
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

    # Wrap mask_mod_flex to pass seqlen_q and seqlen_k
    def mask_fn(b, h, q_idx, kv_idx):
        return mask_mod_flex(b, h, q_idx, kv_idx, seqlen_q, seqlen_k)

    if mask_mod_name == "block_causal":
        n_blocks_q = (seqlen_q + tile_m - 1) // tile_m
        n_blocks_k = (seqlen_k + tile_n - 1) // tile_n

        mask = torch.zeros(seqlen_q, seqlen_k, dtype=torch.bool, device=q.device)

        for q_block in range(n_blocks_q):
            q_start = q_block * tile_m
            q_end = min((q_block + 1) * tile_m, seqlen_q)
            for k_block in range(n_blocks_k):
                if k_block <= q_block:
                    k_start = k_block * tile_n
                    k_end = min((k_block + 1) * tile_n, seqlen_k)
                    mask[q_start:q_end, k_start:k_end] = True

        attn_mask = (
            mask.unsqueeze(0).unsqueeze(0).expand(batch_size, nheads, -1, -1)
        )
        out_ref = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, scale=scale
        )
    else:
        block_mask = create_block_mask(
            mask_fn,
            B=batch_size,
            H=nheads,
            Q_LEN=seqlen_q,
            KV_LEN=seqlen_k,
        ).to(q.device)
        out_ref = flex_attention(q, k, v, block_mask=block_mask, scale=scale)

    return out_ref.transpose(1, 2).contiguous()


@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
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
    ],
)
# @pytest.mark.parametrize("nheads", [4, 16, 32])
@pytest.mark.parametrize("nheads", [16])
@pytest.mark.parametrize("kv_mode", ["mha", "gqa", "mqa"])
# @pytest.mark.parametrize("headdim", [64, 128])
@pytest.mark.parametrize("headdim", [128])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize(
    "use_mask_mod,is_local,mask_name,window_size,window_left,window_right",
    [
        (False, False, "identity", None, None, None),
        (False, False, "causal", None, None, None),
        (True, False, "identity", None, None, None),
        (True, False, "causal", None, None, None),
        # (True, False, "block_causal", None, None, None),
        # Mask mod sliding window
        (True, False, "sliding_window", 128, None, None),
        (True, False, "sliding_window", 256, None, None),
        (True, False, "sliding_window", 512, None, None),
        # Base local attention
        # (False, True, None, None, 128, 0),
        # (False, True, None, None, 256, 0),
        # (False, True, None, None, 512, 0),
    ],
)
@pytest.mark.parametrize("tile_m,tile_n", [(128, 128),])
def test_mask_mod_output(
    seqlen_q, seqlen_k, nheads, kv_mode, headdim, dtype, 
    use_mask_mod, is_local, mask_name, window_size, window_left, window_right,
    tile_m, tile_n
):
    torch.manual_seed(42)

    # Validate configuration
    if is_local:
        assert not use_mask_mod, "Cannot use both is_local and use_mask_mod"
        assert window_left is not None or window_right is not None, \
            "Must specify window_left or window_right for is_local"
    
    if use_mask_mod and mask_name == "sliding_window":
        assert window_size is not None, "window_size must be specified for sliding_window"
        # Skip if seqlen_k is too small for the window
        # if seqlen_k < window_size // 2:
        #     pytest.skip(f"seqlen_k={seqlen_k} too small for window_size={window_size}")
        # Skip if seqlen_q > seqlen_k (problematic for sliding window)
        if seqlen_q > seqlen_k:
            pytest.skip(f"seqlen_q={seqlen_q} > seqlen_k={seqlen_k} not supported for sliding_window")
    
    if is_local:
        window_left_val = window_left if window_left is not None else 0
        window_right_val = window_right if window_right is not None else 0
        total_window = window_left_val + window_right_val + 1
        # Skip if seqlen_k is too small for the window
        if seqlen_k < total_window // 2:
            pytest.skip(f"seqlen_k={seqlen_k} too small for window={total_window}")
        # Skip if seqlen_q > seqlen_k (problematic for local window)
        if seqlen_q > seqlen_k:
            pytest.skip(f"seqlen_q={seqlen_q} > seqlen_k={seqlen_k} not supported for is_local")

    # Determine nheads_kv based on mode
    if kv_mode == "mha":
        nheads_kv = nheads
    elif kv_mode == "gqa":
        nheads_kv = nheads // 2
    elif kv_mode == "mqa":
        nheads_kv = 1
    else:
        raise ValueError(f"Unknown kv_mode: {kv_mode}")

    batch_size = 2
    headdim_v = headdim

    # Determine mask_mod functions and causal flag
    if use_mask_mod:
        if mask_name == "sliding_window":
            # Use factory function for custom window size
            mask_mod_cute = create_cute_sliding_window_mask(window_size)
            mask_mod_flex = create_flex_sliding_window_mask(window_size)
        else:
            mask_mod_cute, mask_mod_flex = MASK_FUNCTIONS[mask_name]
        causal = (mask_name == "causal")
    elif is_local:
        # Base local attention - no mask_mod
        mask_mod_cute = None
        mask_mod_flex = None
        causal = False
    else:
        mask_mod_cute = None
        mask_mod_flex = None
        causal = (mask_name == "causal") if mask_name else False
    
    if causal and seqlen_k < seqlen_q:
        pytest.skip("causal masking requires seqlen_k >= seqlen_q")

    tensors = create_tensors(
        batch_size, seqlen_q, seqlen_k, nheads, nheads_kv, headdim, headdim_v, dtype
    )

    # Compute block sparsity for mask_mod
    full_cnt, full_idx, mask_cnt, mask_idx = None, None, None, None
    if use_mask_mod:
        from dataclasses import dataclass

        @dataclass
        class Config:
            seqlen_q: int
            seqlen_k: int
            nheads: int
            nheads_kv: int
            batch_size: int
            tile_m: int
            tile_n: int
            use_mask_mod: bool
            mask_mod_name: str
            window_size: int = 1024
            verbose: bool = False

        config = Config(
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            nheads=nheads,
            nheads_kv=nheads_kv,
            batch_size=batch_size,
            tile_m=tile_m,
            tile_n=tile_n,
            use_mask_mod=True,
            mask_mod_name=mask_name,
            window_size=window_size if window_size is not None else 1024,
        )

        full_cnt, full_idx, mask_cnt, mask_idx = compute_block_sparsity(
            config=config, mask_mod_flex=mask_mod_flex, device="cuda"
        )

    # Run kernel
    out_cute = compile_and_run_kernel(
        tensors,
        mask_mod_cute,
        causal=causal,
        is_local=is_local,
        window_left=window_left,
        window_right=window_right,
        tile_m=tile_m,
        tile_n=tile_n,
        full_block_cnt=full_cnt,
        full_block_idx=full_idx,
        mask_block_cnt=mask_cnt,
        mask_block_idx=mask_idx,
    )

    # Determine which reference implementation to use
    dtype_ref = torch.bfloat16
    use_flash_attn_ref = False
    
    # Use FlashAttention reference for causal and local window cases
    if mask_name == "causal" and not use_mask_mod:
        use_flash_attn_ref = True
        window_size_ref = (None, None)  # attention_ref handles causal internally
    elif mask_name == "identity" and not use_mask_mod and not is_local:
        use_flash_attn_ref = True
        window_size_ref = (None, None)  # No window for identity
    elif is_local:
        use_flash_attn_ref = True
        # For is_local, we need to pass the window parameters
        # When window_right=0, this is inherently causal
        window_size_ref = (window_left, window_right)
        if window_right == 0:
            causal = True  # Override causal flag for reference computation
    elif use_mask_mod and mask_name == "sliding_window":
        use_flash_attn_ref = True
        # For sliding window mask_mod, window_size corresponds directly to window_left
        # in attention_ref (number of previous tokens that can be attended to)
        # Sliding window with window_right=0 is inherently causal
        window_size_ref = (window_size, 0)
        causal = True  # Override causal flag for reference computation
    
    if use_flash_attn_ref:
        # Compute reference using FlashAttention's attention_ref
        out_ref_fp32 = compute_reference_flash_attn(
            tensors, causal=causal, window_size=window_size_ref, dtype_ref=torch.float32, upcast=True
        )
        out_ref = compute_reference_flash_attn(
            tensors, causal=causal, window_size=window_size_ref, dtype_ref=dtype_ref, upcast=False
        )
        
        # Also compute PyTorch reference for comparison (with reorder_ops for better accuracy)
        out_pt = compute_reference_flash_attn(
            tensors, causal=causal, window_size=window_size_ref, dtype_ref=dtype, upcast=False
        )
    else:
        # Use flex_attention for custom mask_mods
        tensors_fp32 = {
            k: v.float() if v.dtype in [torch.float16, torch.bfloat16] else v
            for k, v in tensors.items()
        }
        
        out_ref_fp32 = compute_reference_flex_attn(
            tensors_fp32, mask_mod_flex, mask_name, tile_m, tile_n
        )
        out_ref = compute_reference_flex_attn(
            tensors, mask_mod_flex, mask_name, tile_m, tile_n
        )
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

    # Build description string
    if is_local:
        mask_desc = f"is_local(L={window_left},R={window_right})"
    elif use_mask_mod:
        mask_desc = f"mask_mod={mask_name}"
        if mask_name == "sliding_window" and window_size is not None:
            mask_desc += f"(w={window_size})"
    else:
        mask_desc = mask_name if mask_name else "identity"
    
    print(
        f"\n{mask_desc} @ Q={seqlen_q}, K={seqlen_k}, H={nheads}/{nheads_kv} ({kv_mode}), "
        f"D={headdim}, M={tile_m}, N={tile_n}"
    )
    print(f"  Reference implementation: {'FlashAttention' if use_flash_attn_ref else 'FlexAttention'}")
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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])