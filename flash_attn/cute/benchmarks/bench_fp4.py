"""Standalone benchmark for FP4 Flash Attention.

This benchmark tests FP4 attention kernels and compares them against
the standard Python interface implementation.
"""

import time
from typing import NamedTuple

import torch
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack

from flash_attn.cute.interface import flash_attn_func as flash_attn_func_python

from triton.testing import do_bench


def bench_gpu_time(fn, rep=25, warmup=10):
    ms = do_bench(fn, rep=rep, warmup=warmup)
    return NamedTuple('Timing', [('mean', float)])(mean=ms)


Timing = NamedTuple('timing', [('mean', float)])

def check_tensor_for_nans(tensor, name="tensor"):
    """Check a tensor (CuTe or PyTorch) for NaN values.
    
    Note: If you have a CuTe tensor created via cute_tensor_like, use the returned
    torch tensor directly instead of the CuTe tensor for checking.
    """
    # If it's already a torch tensor, use it directly
    if isinstance(tensor, torch.Tensor):
        torch_tensor = tensor
    else:
        # Try to convert CuTe tensor to PyTorch using DLPack
        # CuTe tensors created from torch tensors store the DLPack capsule in _dlpack_data
        try:
            if hasattr(tensor, '_dlpack_data'):
                # Use torch.from_dlpack to convert the DLPack capsule back to torch tensor
                torch_tensor = torch.from_dlpack(tensor._dlpack_data)
            else:
                print(f"Warning: {name} is a CuTe tensor but cannot access DLPack data. "
                      "If created with cute_tensor_like, use the returned torch tensor instead.")
                raise ValueError(f"Tensor {name} is a CuTe tensor but cannot access DLPack data!")
        except Exception as e:
            print(f"Warning: Could not convert {name} to PyTorch tensor: {e}")
            raise ValueError(f"Could not convert {name} to PyTorch tensor: {e}")
    
    has_nan = torch.isnan(torch_tensor).any().item()
    has_inf = torch.isinf(torch_tensor).any().item()
    
    if has_nan:
        nan_count = torch.isnan(torch_tensor).sum().item()
        print(f"ERROR: {name} contains {nan_count} NaN values!")
        print(f"  Shape: {torch_tensor.shape}")
        print(f"  Dtype: {torch_tensor.dtype}")
        if torch_tensor.numel() > 0:
            finite_mask = torch.isfinite(torch_tensor)
            if finite_mask.any():
                print(f"  Finite values - Min: {torch_tensor[finite_mask].min().item():.6f}, "
                      f"Max: {torch_tensor[finite_mask].max().item():.6f}")
        raise ValueError(f"Tensor {name} contains NaN values!")
    
    if has_inf:
        inf_count = torch.isinf(torch_tensor).sum().item()
        print(f"WARNING: {name} contains {inf_count} Inf values!")
        print(f"  Shape: {torch_tensor.shape}")
        print(f"  Dtype: {torch_tensor.dtype}")
        raise ValueError(f"Tensor {name} contains Inf values!")
    
    return torch_tensor


def flops(batch, nheads, seqlen_q, seqlen_k, headdim, headdim_v, causal=False, window_size=(None, None)):
    """Calculate FLOPS for attention computation."""
    if causal:
        avg_seqlen = (max(0, seqlen_k - seqlen_q) + seqlen_k) / 2
    else:
        if window_size == (None, None):
            avg_seqlen = seqlen_k
        else:
            row_idx = torch.arange(seqlen_q, device='cuda')
            col_left = torch.maximum(row_idx + seqlen_k - seqlen_q - window_size[0], torch.tensor(0)) if window_size[0] is not None else torch.zeros_like(row_idx)
            col_right = torch.minimum(row_idx + seqlen_k - seqlen_q + window_size[1], torch.tensor(seqlen_k - 1)) if window_size[1] is not None else torch.full_like(row_idx, seqlen_k - 1)
            avg_seqlen = (col_right - col_left + 1).float().mean().item()
    return batch * nheads * 2 * seqlen_q * avg_seqlen * (headdim + headdim_v)


@cute.jit
def cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
    sf_ref_tensor: cute.Tensor,
    sf_mma_tensor: cute.Tensor,
    atom_k: cute.Int32,
):
    """Convert scale factor tensor from MKL layout to mma specification M(32x4xrest_m)xK(4xrest_k)x(nheads,batch) layout"""
    # sf_ref_tensor has shape (mn, sf_k, batch, nheads) after permute
    # sf_mma_tensor has shape (32, 4, rest_m, 4, rest_k, nheads, batch) 
    # Convert coordinates: (mn_idx, sf_k_idx, batch_idx, nhead_idx) -> (atom_m_0, atom_m_1, rest_m_idx, atom_k_idx, rest_k_idx, nhead_idx, batch_idx)
    atom_m = (32, 4)
    for i in cutlass.range(cute.size(sf_ref_tensor)):
        mkl_coord = sf_ref_tensor.layout.get_hier_coord(i)
        mn_idx, sf_k_idx, batch_idx, nhead_idx = mkl_coord
        # Convert mn_idx to (rest_m_idx, atom_m_0, atom_m_1)
        rest_m_idx = mn_idx // (atom_m[0] * atom_m[1])
        mn_in_tile = mn_idx % (atom_m[0] * atom_m[1])
        atom_m_0 = mn_in_tile // atom_m[1]
        atom_m_1 = mn_in_tile % atom_m[1]
        # Convert sf_k_idx to (rest_k_idx, atom_k)
        rest_k_idx = sf_k_idx // atom_k
        atom_k_idx = sf_k_idx % atom_k
        # Create mma coordinate matching permuted shape (32, 4, rest_m, 4, rest_k, nheads, batch)
        mma_coord = (atom_m_0, atom_m_1, rest_m_idx, atom_k_idx, rest_k_idx, nhead_idx, batch_idx)
        sf_mma_tensor[mma_coord] = sf_ref_tensor[mkl_coord]


def create_scale_factor_tensor(batch, seqlen, nheads, headdim, sf_vec_size, sf_dtype, q_dtype, device='cuda', debug=False, sf_value=1.0):
    """Create scale factor tensor for Q/K/V.

    Args:
        batch: Batch size
        seqlen: Sequence length
        nheads: Number of heads
        headdim: Head dimension
        sf_vec_size: Scale factor vector size (typically 16 for NVFP4)
        sf_dtype: Scale factor dtype (typically Float8E4M3FN for NVFP4)
        device: Device to create tensor on
        sf_value: Scale factor value (default 1.0)

    Returns:
        Tuple of (ref_tensor_cpu, cute_tensor, cute_torch_tensor)
    """
    def ceil_div(a, b):
        return (a + b - 1) // b

    # Scale factor shape: (batch, nheads, seqlen, ceil_div(headdim, sf_vec_size))
    # For attention, we need scale factors per head dimension
    # Split batch and nheads so we can index them separately in the kernel
    mn = seqlen
    k = headdim
    sf_k = ceil_div(k, sf_vec_size)
    ref_shape = (batch, nheads, mn, sf_k)

    atom_m = (32, 4)
    atom_k = 4
    # mma_shape keeps batch and nheads separate: (batch, nheads, rest_m, rest_k, 32, 4, 4)
    # This allows indexing batch and head separately in the kernel like mQ
    mma_shape = (
        batch,
        nheads,
        ceil_div(mn, atom_m[0] * atom_m[1]),
        ceil_div(sf_k, atom_k),
        atom_m[0],
        atom_m[1],
        atom_k,
    )

    # Permute (batch, nheads, mn, sf_k) to (mn, sf_k, batch, nheads) for ref tensor
    # This allows indexing by batch and nheads in the kernel
    ref_permute_order = (2, 3, 0, 1)
    # Permute mma_shape (batch, nheads, rest_m, rest_k, 32, 4, 4) to (32, 4, rest_m, 4, rest_k, nheads, batch)
    mma_permute_order = (4, 5, 2, 6, 3, 1, 0)

    # Create f32 ref torch tensor (cpu)
    init_type = cutlass_torch.TensorInitType.SCALAR
    init_config = cutlass_torch.ScalarInitConfig(
        value=sf_value
    )
    ref_f32_torch_tensor_cpu = cutlass_torch.create_and_permute_torch_tensor(
        ref_shape, torch.float32, permute_order=ref_permute_order, init_type=init_type, init_config=init_config
    )
    
    # Create f32 cute torch tensor (cpu)
    cute_f32_torch_tensor_cpu = cutlass_torch.create_and_permute_torch_tensor(
        mma_shape,
        torch.float32,
        permute_order=mma_permute_order,
        init_type=init_type,
        init_config=init_config,
    )
    
    # convert ref f32 tensor to cute f32 tensor
    cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
        from_dlpack(ref_f32_torch_tensor_cpu),
        from_dlpack(cute_f32_torch_tensor_cpu),
        atom_k,
    )
    cute_f32_torch_tensor = cute_f32_torch_tensor_cpu.cuda()
    
    # reshape makes memory contiguous
    # After permute with ref_permute_order, shape is (mn, sf_k, batch, nheads)
    # Permute to (batch, nheads, mn, sf_k), then expand and reshape
    l = batch * nheads
    ref_f32_torch_tensor_cpu = (
        ref_f32_torch_tensor_cpu.permute(2, 3, 0, 1)  # (mn, sf_k, batch, nheads) -> (batch, nheads, mn, sf_k)
        .unsqueeze(-1)
        .expand(batch, nheads, mn, sf_k, sf_vec_size)
        .reshape(batch, nheads, mn, sf_k * sf_vec_size)
        .permute(2, 3, 0, 1)  # (batch, nheads, mn, sf_k * sf_vec_size) -> (mn, sf_k * sf_vec_size, batch, nheads)
        .reshape(l, mn, sf_k * sf_vec_size)  # Flatten batch and nheads for compatibility
        .permute(1, 2, 0)  # (l, mn, sf_k * sf_vec_size) -> (mn, sf_k * sf_vec_size, l)
    )
    # prune to actual k dimension
    ref_f32_torch_tensor_cpu = ref_f32_torch_tensor_cpu[:, :k, :]
    
    # Create dtype cute torch tensor (cpu)
    cute_tensor, cute_torch_tensor = cutlass_torch.cute_tensor_like(
        cute_f32_torch_tensor_cpu,
        sf_dtype,
        is_dynamic_layout=True,
        assumed_align=16,
    )
    
    # Convert f32 cute tensor to dtype cute tensor
    cute_tensor = cutlass_torch.convert_cute_tensor(
        cute_f32_torch_tensor,
        cute_tensor,
        sf_dtype,
        is_dynamic_layout=True,
    )

    return ref_f32_torch_tensor_cpu, cute_tensor, cute_torch_tensor


def _compact_fp4_data(torch_underlying, ab_dtype):
    """Compact FP4 data in the int8 buffer to match CuTe's FP4 stride interpretation.

    cute_tensor_like creates an int8 tensor with the FULL shape of the reference,
    then overrides element_type to FP4. The CuTe tensor has int8-unit strides
    (e.g., head stride = D bytes), but FP4 element_type halves byte offsets
    (head byte offset = D/2). So the kernel reads at half the byte positions.

    convert_cute_tensor correctly writes packed FP4 data at int8 byte positions
    (rows at D-byte intervals). This function compacts the data so it's at the
    FP4 byte positions (rows at D/2-byte intervals) where the kernel expects it.
    """
    if ab_dtype.width >= 8:
        return

    D = torch_underlying.shape[-1]  # Last dim = headdim in int8 units
    packed_size = D // (8 // ab_dtype.width)  # D//2 for FP4

    # Extract packed FP4 data from each row (first packed_size bytes, rest is gap)
    rows = torch_underlying.reshape(-1, D)
    packed = rows[:, :packed_size].contiguous()  # Copy before overwriting

    # Clear buffer and write data at compacted positions
    torch_underlying.fill_(0)
    torch_underlying.flatten()[:packed.numel()].copy_(packed.flatten())


def create_blockscaled_attention_tensors(batch, seqlen_q, seqlen_k, nheads, nheads_kv, headdim, headdim_v,
                                         device='cuda', dtype_gen=torch.bfloat16, pv_mode='bf16', return_torch=True,
                                         ab_dtype=None, sf_dtype=None, sf_vec_size=None, pv_fp8_dtype=None, debug=False):
    """Create block-scaled Q/K attention tensors with optional FP4 / FP8 V.
    
    Args:
        batch: Batch size
        seqlen_q: Query sequence length
        seqlen_k: Key sequence length
        nheads: Number of query heads
        nheads_kv: Number of key/value heads
        headdim: Head dimension for Q/K
        headdim_v: Head dimension for V
        device: Device to create tensors on
        dtype_gen: Dtype to generate random data in (before conversion to FP4)
        pv_mode: One of {'bf16', 'fp4', 'fp8'}
        return_torch: Whether to return torch tensors (default: True)
        ab_dtype: Data type for block-scaled Q/K matrices
        sf_dtype: Q/K scale factor dtype
        sf_vec_size: Q/K scale factor vector size
    Returns:
        Tuple of (q_tensor, k_tensor, v_tensor, q_sf, k_sf, v_sf, q_ref, k_ref, v_ref)
        where q_tensor, k_tensor are block-scaled operand tensors, v_tensor depends on pv_mode,
        q_sf, k_sf are Q/K scale factor tensors, v_sf is scale factor tensor only for pv_mode='fp4',
        and *_ref are reference FP32 tensors
    """
    if pv_mode not in {"bf16", "fp4", "fp8"}:
        raise ValueError(f"Invalid pv_mode={pv_mode}")

    # Default block-scaled parameters
    if ab_dtype is None:
        ab_dtype = cutlass.Float4E2M1FN
    if sf_dtype is None:
        sf_dtype = cutlass.Float8E4M3FN
    if sf_vec_size is None:
        sf_vec_size = 16
    if pv_fp8_dtype is None:
        pv_fp8_dtype = cutlass.Float8E4M3FN
    
    # Create reference FP32 tensors
    if debug:
        q_ref = torch.full((batch, seqlen_q, nheads, headdim), fill_value=1.0, device=device, dtype=torch.float32)
        k_ref = torch.full((batch, seqlen_k, nheads_kv, headdim), fill_value=1.0, device=device, dtype=torch.float32)
        # V = block_index mod 4, keeping values in FP4 range (max 6.0)
        # With uniform attention, expected output = mean of V values
        n_block_size = 128
        n_blocks = seqlen_k // n_block_size
        v_ref = torch.zeros((batch, seqlen_k, nheads_kv, headdim_v), device=device, dtype=torch.float32)
        for s in range(seqlen_k):
            v_ref[:, s, :, :] = (s // n_block_size) % 4
        # breakpoint()
    else:
        q_ref = torch.randn(batch, seqlen_q, nheads, headdim, device=device, dtype=torch.float32)
        k_ref = torch.randn(batch, seqlen_k, nheads_kv, headdim, device=device, dtype=torch.float32)
        v_ref = torch.randn(batch, seqlen_k, nheads_kv, headdim_v, device=device, dtype=torch.float32)

    # Create Q/K tensors for the selected block-scaled operand type.
    if ab_dtype == cutlass.Float4E2M1FN:
        qk_divisibility = 32
    else:
        qk_divisibility = 16
    q_tensor, q_torch_underlying = cutlass_torch.cute_tensor_like(
        q_ref, ab_dtype, is_dynamic_layout=True, assumed_align=16
    )
    k_tensor, k_torch_underlying = cutlass_torch.cute_tensor_like(
        k_ref, ab_dtype, is_dynamic_layout=True, assumed_align=16
    )
    # Get the correct stride_order from the reference tensors
    # stride_order should match the layout of the original tensor
    q_stride_order = tuple(q_ref.dim_order())
    k_stride_order = tuple(k_ref.dim_order())
    # Mark tensors to be byte aligned (FP4 needs divisibility of 2)
    # For flash attention, headdim is the last dimension (index 3), which should be mode 1
    # Mode 0 is for batch/seqlen/nheads, mode 1 is for headdim
    q_tensor.mark_compact_shape_dynamic(
        mode=1,  # headdim dimension needs divisibility for FP4
        stride_order=q_stride_order,
        divisibility=qk_divisibility,
    )
    k_tensor.mark_compact_shape_dynamic(
        mode=1,  # headdim dimension needs divisibility for FP4
        stride_order=k_stride_order,
        divisibility=qk_divisibility,
    )

    # Convert FP32 tensors to FP4 format for Q and K
    q_tensor = cutlass_torch.convert_cute_tensor(
        q_ref, q_tensor, ab_dtype, is_dynamic_layout=True
    )
    k_tensor = cutlass_torch.convert_cute_tensor(
        k_ref, k_tensor, ab_dtype, is_dynamic_layout=True
    )

    # Note: convert_cute_tensor writes FP4 data contiguously, matching the
    # FP4 stride interpretation. No compaction or stride fix needed.
    
    # Handle V according to the requested PV path.
    if pv_mode == "fp4":
        # FP4 block-scaled MMA requires V to be K-major (seqlen contiguous in SMEM).
        # Create V with seqlen as the contiguous dimension by physically transposing.
        # v_ref: (batch, seqlen, nheads, headdim) with headdim contiguous
        # After permute+contiguous+permute: same shape but seqlen has stride 1
        v_ref_kmajor = v_ref.permute(0, 3, 2, 1).contiguous().permute(0, 3, 2, 1)
        print(f"  V layout for FP4: shape={v_ref_kmajor.shape}, strides={v_ref_kmajor.stride()}")
        v_tensor, v_torch_underlying = cutlass_torch.cute_tensor_like(
            v_ref_kmajor, ab_dtype, is_dynamic_layout=True, assumed_align=16
        )
        # Get the correct stride_order from the transposed reference tensor
        v_stride_order = tuple(v_ref_kmajor.dim_order())
        v_tensor.mark_compact_shape_dynamic(
            mode=1,  # headdim dimension needs divisibility for FP4
            stride_order=v_stride_order,
            divisibility=32,
        )
        v_tensor = cutlass_torch.convert_cute_tensor(
            v_ref_kmajor, v_tensor, ab_dtype, is_dynamic_layout=True
        )
        # V is (batch, seqlen, nheads, headdim) with seqlen contiguous
    elif pv_mode == "fp8":
        v_tensor, v_torch_underlying = cutlass_torch.cute_tensor_like(
            v_ref, pv_fp8_dtype, is_dynamic_layout=True, assumed_align=16
        )
        v_stride_order = tuple(v_ref.dim_order())
        v_tensor.mark_compact_shape_dynamic(
            mode=1,
            stride_order=v_stride_order,
            divisibility=16,
        )
        v_tensor = cutlass_torch.convert_cute_tensor(
            v_ref, v_tensor, pv_fp8_dtype, is_dynamic_layout=True
        )
    else:
        # V stays in regular dtype (BF16/FP16).
        # Convert torch dtype to CUTE dtype
        assert dtype_gen in [torch.bfloat16, torch.float16]
        if dtype_gen == torch.bfloat16:
            v_cute_dtype = cutlass.BFloat16
        elif dtype_gen == torch.float16:
            v_cute_dtype = cutlass.Float16

        v_tensor, v_torch_underlying = cutlass_torch.cute_tensor_like(
            v_ref, v_cute_dtype, is_dynamic_layout=True, assumed_align=16
        )
        # Get the correct stride_order from the reference tensor
        v_stride_order = tuple(v_ref.dim_order())
        v_tensor.mark_compact_shape_dynamic(
            mode=1,  # headdim_v dimension
            stride_order=v_stride_order,
            divisibility=16, 
        )
        v_tensor = cutlass_torch.convert_cute_tensor(
            v_ref, v_tensor, v_cute_dtype, is_dynamic_layout=True
        )
    
    # Create scale factor tensors for Q and K (V scale factors are optional)
    # For Q: (batch, nheads, seqlen_q, headdim) -> scale factors for headdim dimension
    # Scale factors are per (batch * nheads, seqlen_q, ceil_div(headdim, sf_vec_size))
    q_sf_ref, q_sf_tensor, q_sf_torch_underlying = create_scale_factor_tensor(
        batch, seqlen_q, nheads, headdim, sf_vec_size, sf_dtype, ab_dtype, 
        device, debug=debug
    )
    # For K: (batch, nheads_kv, seqlen_k, headdim) -> scale factors for headdim dimension
    k_sf_value = 2.0 if debug else 1.0  # Use 2.0 in debug to test SF reading (expect S=256)
    k_sf_ref, k_sf_tensor, k_sf_torch_underlying = create_scale_factor_tensor(
        batch, seqlen_k, nheads_kv, headdim, sf_vec_size, sf_dtype, ab_dtype,
        device, debug=debug, sf_value=k_sf_value
    )
    
    # Create V scale factors only if V is block-scaled FP4.
    if pv_mode == "fp4":
        v_sf_ref, v_sf_tensor, v_sf_torch_underlying = create_scale_factor_tensor(
            batch, seqlen_k, nheads_kv, headdim_v, sf_vec_size, sf_dtype, ab_dtype, device, debug=debug
        )
    else:
        v_sf_tensor = None
        v_sf_torch_underlying = None

    # TODO: multiply qkv ref with scale factor tensors

    if debug:
        k_sf_torch_underlying.fill_(0x40)  # 2.0 in E4M3
        torch.cuda.synchronize()

    if return_torch:
        # For FP8 V: convert int8 backing to torch.float8_e4m3fn so interface recognizes it
        if pv_mode == "fp8" and v_torch_underlying is not None and v_torch_underlying.dtype == torch.int8:
            v_torch_underlying = v_torch_underlying.view(torch.float8_e4m3fn)
        return (q_torch_underlying, k_torch_underlying, v_torch_underlying, q_sf_torch_underlying, k_sf_torch_underlying, v_sf_torch_underlying,
                q_ref, k_ref, v_ref)
    else:
        # Return torch underlying for SF tensors so interface can create TVM-FFI cute tensors
        # The cute tensors from cute_tensor_like lack enable_tvm_ffi=True
        return (q_tensor, k_tensor, v_tensor, q_sf_torch_underlying, k_sf_torch_underlying, v_sf_torch_underlying,
                q_ref, k_ref, v_ref)


def create_nvfp4_attention_tensors(batch, seqlen_q, seqlen_k, nheads, nheads_kv, headdim, headdim_v,
                                   device='cuda', dtype_gen=torch.bfloat16, pv_mode='bf16',
                                   pv_fp8_dtype=None):
    """Create block-scaled attention tensors using flashinfer's nvfp4_quantize.

    Uses proper per-block adaptive scale factors (SF = amax/6 per sf_vec_size=16 block),
    producing packed float4_e2m1fn_x2 Q/K with SF in BlockScaledBasicChunk MMA layout.
    This matches production quantization and achieves cos >= 0.99 vs BF16.
    """
    from flashinfer.quantization import nvfp4_quantize, SfLayout

    sf_vec_size = 16
    tile_m = 128

    q_ref = torch.randn(batch, seqlen_q, nheads, headdim, device=device, dtype=torch.float32)
    k_ref = torch.randn(batch, seqlen_k, nheads_kv, headdim, device=device, dtype=torch.float32)
    v_ref = torch.randn(batch, seqlen_k, nheads_kv, headdim_v, device=device, dtype=torch.float32)

    def _quantize_and_reshape_sf(ref, batch_, seqlen_, nheads_, headdim_):
        t2d = ref.to(dtype_gen).reshape(batch_ * seqlen_, nheads_ * headdim_)
        one = torch.ones(1, device=device, dtype=torch.float32)
        fp4_data, sf_data = nvfp4_quantize(t2d, one, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
        fp4 = fp4_data.reshape(batch_, seqlen_, nheads_, headdim_ // 2).view(torch.uint8).view(torch.float4_e2m1fn_x2)
        rest_m = seqlen_ // tile_m
        sf_k = headdim_ // sf_vec_size
        rest_k = sf_k // 4
        total_m = batch_ * rest_m
        total_k = (nheads_ * sf_k) // 4
        sf = sf_data.reshape(total_m, total_k, 32, 4, 4)
        sf = sf.reshape(batch_, rest_m, nheads_, rest_k, 32, 4, 4)
        sf = sf.permute(0, 2, 1, 3, 4, 5, 6).contiguous().permute(4, 5, 2, 6, 3, 1, 0)
        return fp4, sf

    q_fp4, q_sf = _quantize_and_reshape_sf(q_ref, batch, seqlen_q, nheads, headdim)
    k_fp4, k_sf = _quantize_and_reshape_sf(k_ref, batch, seqlen_k, nheads_kv, headdim)

    # V stays in BF16/FP8 (no block-scaled PV)
    if pv_mode == "fp8":
        _fp8 = pv_fp8_dtype or cutlass.Float8E4M3FN
        _torch_fp8 = torch.float8_e4m3fn if _fp8 == cutlass.Float8E4M3FN else torch.float8_e5m2
        v_tensor = v_ref.to(dtype_gen).to(_torch_fp8)
    else:
        v_tensor = v_ref.to(dtype_gen)

    return q_fp4, k_fp4, v_tensor, q_sf, k_sf, None, q_ref, k_ref, v_ref


def time_fwd(func, *args, repeats=10, verbose=True, desc="", **kwargs):
    """Time forward pass via triton.testing.do_bench.

    CUPTI (flashinfer's default path) requires CUDA 13+ which is unavailable on
    this build host, and bench_gpu_time's fallback path has ~6% per-iter overhead.

    `rep` and `warmup` are in milliseconds; triton auto-picks iteration counts.
    We use rep=25ms intentionally — the sweet spot on B200 for this kernel:
      rep=10ms  → too few iterations, median dominated by warmup tail
      rep=25ms  → ~3 iterations, matches torch.cuda.Event peak (1804 TF)
      rep=50ms+ → SM clock throttles (sustained >25 ms heavy work trips
                  the power-limit governor), numbers drop ~40 TF to ~1765 TF

    triton.do_bench always zero-fills an L2-sized buffer between iterations;
    with only ~3 iterations the cumulative L2-flush overhead stays negligible.
    """
    import triton.testing
    fn = lambda: func(*args, **kwargs)
    ms = triton.testing.do_bench(fn, rep=25, warmup=10, return_mode="median")
    return Timing(ms * 1e-3)


def main(ab_dtype, sf_dtype, sf_vec_size, pv_mode="bf16", pv_fp8_dtype=cutlass.Float8E4M3FN, debug=False, causal=False):
    """Main benchmark function."""
    torch.manual_seed(0)
    repeats = 10
    device = 'cuda'
    verbose = True
    dtype_gen = torch.bfloat16
    
    # Benchmark configurations: (batch, seqlen, nheads, headdim)
    # Covers bench_fp4 defaults, video-gen (Wan2.1-1.3B: nheads=12, hdim=128), and larger models
    configs = [
        # bench_fp4 defaults (nheads=16, headdim=128)
        (1, 256, 16, 128),
        (1, 1024, 16, 128),
        (4, 4096, 16, 128),
        (1, 32768, 16, 128),
        (4, 4096, 32, 128),
        # Video gen shapes (Wan2.1-T2V-1.3B: nheads=12, headdim=128)
        (1, 4096, 12, 128),
        (1, 32768, 12, 128),
        # Larger models (nheads=24)
        (1, 4096, 24, 128),
        (1, 32768, 24, 128),
    ]
    if pv_mode != "fp4" and sf_vec_size * 4 <= 64:
        # headdim=64 requires head_dim >= sf_vec_size*4 (block-scaled MMA atom K constraint).
        # MXFP8 (sf_vec_size=32) needs headdim >= 128; NVFP4 (sf_vec_size=16) supports d=64.
        configs.append((1, 32768, 24, 64))
    print("=" * 80)
    print("FP4 Flash Attention Benchmark")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Causal: {causal}")
    print(f"PV mode: {pv_mode}")
    print(f"QK ab_dtype: {ab_dtype}")
    print(f"QK sf_dtype: {sf_dtype}")
    print(f"QK sf_vec_size: {sf_vec_size}")
    print("=" * 80)

    # Use nvfp4_quantize for NVFP4 (proper per-block SF, cos>=0.99).
    # Fall back to cute_tensor_like for MXFP8 or FP4 PV (nvfp4_quantize only handles NVFP4).
    use_nvfp4 = (ab_dtype == cutlass.Float4E2M1FN and pv_mode != "fp4")

    for i, (batch_size, seqlen, nheads, headdim) in enumerate(configs):
        nheads_kv = nheads
        headdim_v = headdim
        seqlen_q = seqlen
        window_size = (None, None)

        print(f"\n### Batch={batch_size}, SeqLen={seqlen}, Nheads={nheads}, Headdim={headdim} ###")

        if use_nvfp4:
            (q_fp4, k_fp4, v_tensor, q_sf, k_sf, v_sf,
                q_ref, k_ref, v_ref) = create_nvfp4_attention_tensors(
                batch_size, seqlen_q, seqlen, nheads, nheads_kv,
                headdim, headdim_v, device, dtype_gen, pv_mode=pv_mode,
                pv_fp8_dtype=pv_fp8_dtype)
        else:
            (q_fp4, k_fp4, v_tensor, q_sf, k_sf, v_sf,
                q_ref, k_ref, v_ref) = create_blockscaled_attention_tensors(
                batch_size, seqlen_q, seqlen, nheads, nheads_kv,
                headdim, headdim_v, device, dtype_gen, pv_mode=pv_mode, return_torch=False,
                ab_dtype=ab_dtype, sf_dtype=sf_dtype, sf_vec_size=sf_vec_size,
                pv_fp8_dtype=pv_fp8_dtype, debug=debug)
        q_sf_torch = check_tensor_for_nans(q_sf, name="q_sf")
        k_sf_torch = check_tensor_for_nans(k_sf, name="k_sf")

        if pv_mode == "fp4":
            v_sf_torch = check_tensor_for_nans(v_sf, name="v_sf")

        # Calculate FLOPS
        nFLOPS = flops(batch_size, nheads, seqlen_q, seqlen, headdim, headdim_v,
                      causal=causal, window_size=window_size)
        
        # Benchmark FP4 attention
        # Pass CUTE tensors directly (like dense GEMM example)
        m_fp4 = None
        fp4_out = None
        try:
            # The interface should detect nvfp4 dtype and dispatch to FP4 kernel
            # Pass scale factor tensors (V scale factors only if quant_v=True)
            desc_str = f'Block-scaled Attention (pv_mode={pv_mode})'
            m_fp4 = time_fwd(
                flash_attn_func_python,
                q_fp4, k_fp4, v_tensor,
                causal=causal,
                window_size=window_size,
                mSFQ=q_sf,
                mSFK=k_sf,
                mSFV=v_sf,
                repeats=repeats,
                verbose=verbose,
                desc=desc_str
            )
            print(f'FP4 Attention fwd: {m_fp4.mean * 1e3:.3f}ms, {(nFLOPS / m_fp4.mean * 1e-12):.1f} TFLOPS')

            torch.cuda.synchronize()  # flush GPU printf buffer
            fp4_out = flash_attn_func_python(
                q_fp4, k_fp4, v_tensor,
                causal=causal,
                window_size=window_size,
                mSFQ=q_sf,
                mSFK=k_sf,
                mSFV=v_sf,
            )
            # fp4_out = flash_attn_func_python(
            #     q_ref.to(torch.bfloat16), k_ref.to(torch.bfloat16), v_ref.to(torch.bfloat16),
            #     causal=causal,
            #     window_size=window_size,
            #     force_fp4_impl=True,
            # )
        except Exception as e:
            print(f"FP4 attention failed: {e}")
            import traceback
            traceback.print_exc()

        
        # Benchmark reference (FP16/BF16) attention for comparison
        ref_out = None
        try:
            # Create reference tensors in standard dtype
            q_ref = q_ref.to(dtype_gen)
            k_ref = k_ref.to(dtype_gen)
            v_ref = v_ref.to(dtype_gen)
            m_ref = time_fwd(
                flash_attn_func_python,
                q_ref, k_ref, v_ref,
                causal=causal,
                window_size=window_size,
                repeats=repeats,
                verbose=verbose,
                desc='Reference (FP16/BF16) Attention'
            )
            print(f'Reference fwd: {m_ref.mean * 1e3:.3f}ms, {(nFLOPS / m_ref.mean * 1e-12):.1f} TFLOPS')
            ref_out = flash_attn_func_python(
                q_ref, k_ref, v_ref,
                causal=causal,
                window_size=window_size,
            )
            if m_fp4 is not None:
                speedup = m_ref.mean / m_fp4.mean
                print(f'Speedup: {speedup:.2f}x')
        except Exception as e:
            print(f"Reference attention failed: {e}")
            import traceback
            traceback.print_exc()

        # Test: force block-scaled SM100 path with bf16 data (no scale factors).
        # This runs the FP4 kernel code path but with bf16 Q/K/V
        force_fp4_out = None
        try:
            force_fp4_out = flash_attn_func_python(
                q_ref, k_ref, v_ref,
                causal=causal,
                window_size=window_size,
                force_fp4_impl=True,
            )
            force_fp4_t = force_fp4_out[0] if isinstance(force_fp4_out, tuple) else force_fp4_out
            ref_t = ref_out[0] if isinstance(ref_out, tuple) else ref_out
            print("  force_fp4_impl bf16 test:")
            print(f"    force_fp4[0,0,0,:4]: {force_fp4_t[0,0,0,:4]}")
            print(f"    ref[0,0,0,:4]:       {ref_t[0,0,0,:4]}")
            m_block_size_test = 128
            for mb in range(min(seqlen_q // m_block_size_test, 4)):
                s = mb * m_block_size_test
                stage = mb % 2
                print(f"    m_block={mb} stage={stage}: force_fp4={force_fp4_t[0,s,0,0].item():.4f} ref={ref_t[0,s,0,0].item():.4f}")
        except Exception as e:
            print(f"  force_fp4_impl test failed: {e}")
            import traceback
            traceback.print_exc()

        # Compare FP4 and reference outputs
        if fp4_out is not None and ref_out is not None:
            try:
                fp4_cmp = fp4_out[0] if isinstance(fp4_out, tuple) else fp4_out
                ref_cmp = ref_out[0] if isinstance(ref_out, tuple) else ref_out
                fp4_f = fp4_cmp.float()
                ref_f = ref_cmp.float()
                abs_diff = (fp4_f - ref_f).abs()
                has_nan = fp4_cmp.isnan().any().item()
                max_diff = abs_diff.max().item()
                mean_diff = abs_diff.mean().item()
                cos_sim = torch.nn.functional.cosine_similarity(
                    fp4_f.flatten().unsqueeze(0), ref_f.flatten().unsqueeze(0)
                ).item()
                print(f"  FP4 vs ref: cos_sim={cos_sim:.6f}, max_diff={max_diff:.4f}, mean_diff={mean_diff:.6f}, has_nan={has_nan}")
                # FP4 quantization error: max_diff~2-3, mean_diff~0.03-0.09 with SF=1.0
                if debug:
                    torch.testing.assert_close(fp4_cmp, ref_cmp, atol=1e-2, rtol=1e-2)
                else:
                    torch.testing.assert_close(fp4_cmp, ref_cmp, atol=3.0, rtol=0.5)
            except Exception as e:
                print(f"FP4 and reference outputs differ: {e}")
                fp4_t = fp4_out[0] if isinstance(fp4_out, tuple) else fp4_out
                ref_t = ref_out[0] if isinstance(ref_out, tuple) else ref_out
                print(f"  fp4[0,0,0,:8]: {fp4_t[0,0,0,:8]}")
                print(f"  ref[0,0,0,:8]: {ref_t[0,0,0,:8]}")
                # Print fp4 output for each m_block (stage 0 and stage 1 alternate)
                m_block_size = 128
                for mb in range(min(seqlen_q // m_block_size, 8)):
                    s = mb * m_block_size
                    stage = mb % 2
                    print(f"  m_block={mb} stage={stage}: fp4[0,{s},0,0]={fp4_t[0,s,0,0].item():.4f} ref={ref_t[0,s,0,0].item():.4f}")
                import traceback
                traceback.print_exc()

        if (i + 1) % 3 == 0:
            torch.cuda.synchronize()
            time.sleep(2)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark block-scaled Flash Attention")
    parser.add_argument(
        "--quant_v",
        action="store_true",
        help="Deprecated alias for --pv_mode=fp4"
    )
    parser.add_argument(
        "--qk_mode",
        choices=["nvfp4", "mxfp8"],
        default="nvfp4",
        help="Block-scaled QK mode",
    )
    parser.add_argument(
        "--pv_mode",
        choices=["bf16", "fp4", "fp8"],
        default=None,
        help="PV path: bf16 baseline V, fp4 block-scaled V, or pure fp8 V",
    )
    parser.add_argument(
        "--fp8_dtype",
        choices=["e4m3", "e5m2"],
        default="e4m3",
        help="FP8 operand dtype used for MXFP8 QK / FP8 V modes",
    )
    parser.add_argument("--debug", action="store_true", help="Debug precision, set all tensors to 1.0")
    parser.add_argument("--causal", action="store_true", help="Causal attention")
    # parser.add_argument("--ab_dtype", type=cutlass.dtype, default=cutlass.Float4E2M1FN)
    # parser.add_argument("--sf_dtype", type=cutlass.dtype, default=cutlass.Float8E4M3FN)

    args = parser.parse_args()
    pv_mode = "fp4" if args.quant_v else (args.pv_mode or "bf16")
    fp8_dtype = cutlass.Float8E4M3FN if args.fp8_dtype == "e4m3" else cutlass.Float8E5M2
    if args.qk_mode == "nvfp4":
        ab_dtype = cutlass.Float4E2M1FN
        sf_dtype = cutlass.Float8E4M3FN
        sf_vec_size = 16
    else:
        ab_dtype = fp8_dtype
        sf_dtype = cutlass.Float8E8M0FNU
        sf_vec_size = 32
    main(ab_dtype, sf_dtype, sf_vec_size, pv_mode, fp8_dtype, args.debug, args.causal)
