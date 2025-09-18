import torch
import os
from .fwd_prefill import attention_prefill_forward_triton_impl
from .bwd_prefill_split import attention_prefill_backward_triton_split_impl
from .bwd_prefill_fused_atomics import attention_prefill_backward_triton_fused_atomics_impl
from .bwd_prefill_fused_no_atomics import attention_prefill_backward_triton_split_fused_no_atomics_impl
from .fwd_decode import attention_decode_forward_triton_impl
from .fwd_ref import attention_prefill_forward_ref_impl, attention_decode_forward_ref_impl
from .bwd_ref import attention_backward_pytorch_ref_impl
from .utils import DEBUG, USE_REF, MetaData, is_fp8
from einops import rearrange, repeat
from flash_attn.layers.rotary import apply_rotary_emb
from typing import Optional, Union, Tuple

USE_EXP2 = True
BWD_MODE = os.environ.get('BWD_MODE', 'fused_no_atomics').lower()
USE_DECODE_PATH = os.environ.get('FLASH_ATTENTION_V3_USE_DECODE', '0') == '1'

def fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_new: Optional[torch.Tensor],
    v_new: Optional[torch.Tensor],
    qv: Optional[torch.Tensor],
    out: Optional[torch.Tensor],
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    cu_seqlens_k_new: Optional[torch.Tensor],
    seqused_q: Optional[torch.Tensor],
    seqused_k: Optional[torch.Tensor],
    max_seqlen_q: Optional[int],
    max_seqlen_k: Optional[int],
    page_table: Optional[torch.Tensor],
    kv_batch_idx: Optional[torch.Tensor],
    leftpad_k: Optional[torch.Tensor],
    rotary_cos: Optional[torch.Tensor],
    rotary_sin: Optional[torch.Tensor],
    seqlens_rotary: Optional[torch.Tensor],
    q_descale: Optional[torch.Tensor],
    k_descale: Optional[torch.Tensor],
    v_descale: Optional[torch.Tensor],
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    attention_chunk: int,
    softcap: float,
    rotary_interleaved: bool,
    scheduler_metadata=None,
    num_splits: int = 1,
    pack_gqa=None,
    sm_margin: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Flash Attention v3 forward pass compatible interface for AMD Triton implementation.
    
    This function maps v3 parameters to the existing AMD Triton implementation.
    """
    
    if DEBUG:
        print()
        print("interface_fa_v3.py::fwd inputs")
        print("q:", q, q.shape)
        print("k:", k, k.shape)
        print("v:", v, v.shape)
        print("k_new:", k_new, k_new.shape if k_new is not None else None)
        print("v_new:", v_new, v_new.shape if v_new is not None else None)
        print("qv:", qv, qv.shape if qv is not None else None)
        print("out:", out, out.shape if out is not None else None)
        print("cu_seqlens_q:", cu_seqlens_q, cu_seqlens_q.shape if cu_seqlens_q is not None else None)
        print("cu_seqlens_k:", cu_seqlens_k, cu_seqlens_k.shape if cu_seqlens_k is not None else None)
        print("cu_seqlens_k_new:", cu_seqlens_k_new, cu_seqlens_k_new.shape if cu_seqlens_k_new is not None else None)
        print("seqused_q:", seqused_q, seqused_q.shape if seqused_q is not None else None)
        print("seqused_k:", seqused_k, seqused_k.shape if seqused_k is not None else None)
        print("max_seqlen_q:", max_seqlen_q)
        print("max_seqlen_k:", max_seqlen_k)
        print("page_table:", page_table, page_table.shape if page_table is not None else None)
        print("kv_batch_idx:", kv_batch_idx, kv_batch_idx.shape if kv_batch_idx is not None else None)
        print("leftpad_k:", leftpad_k, leftpad_k.shape if leftpad_k is not None else None)
        print("rotary_cos:", rotary_cos, rotary_cos.shape if rotary_cos is not None else None)
        print("rotary_sin:", rotary_sin, rotary_sin.shape if rotary_sin is not None else None)
        print("seqlens_rotary:", seqlens_rotary, seqlens_rotary.shape if seqlens_rotary is not None else None)
        print("q_descale:", q_descale, q_descale.shape if q_descale is not None else None)
        print("k_descale:", k_descale, k_descale.shape if k_descale is not None else None)
        print("v_descale:", v_descale, v_descale.shape if v_descale is not None else None)
        print("softmax_scale:", softmax_scale)
        print("causal:", causal)
        print("window_size_left:", window_size_left)
        print("window_size_right:", window_size_right)
        print("attention_chunk:", attention_chunk)
        print("softcap:", softcap)
        print("rotary_interleaved:", rotary_interleaved)
        print("scheduler_metadata:", scheduler_metadata)
        print("num_splits:", num_splits)
        print("pack_gqa:", pack_gqa)
        print("sm_margin:", sm_margin)
    
    # Handle qv packed input
    if qv is not None:
        raise NotImplementedError("QV packed input is not yet supported in the AMD Triton backend")
    
    
    # Handle softcap
    if softcap != 0.0:
        raise NotImplementedError(f"Softcap is not yet supported in the AMD Triton backend (got softcap={softcap}, expected 0.0)")
    
    # Handle attention_chunk
    if attention_chunk != 0 and attention_chunk != 1:
        raise NotImplementedError(f"attention_chunk is not yet supported in the AMD Triton backend (got attention_chunk={attention_chunk})")
    
    
    # Handle scheduler metadata
    if scheduler_metadata is not None:
        raise NotImplementedError("Scheduler metadata is not yet supported in the AMD Triton backend")
    
    # Handle pack_gqa
    if pack_gqa is not None and pack_gqa is not False:
        raise NotImplementedError(f"pack_gqa is not yet supported in the AMD Triton backend (got pack_gqa={pack_gqa})")
    
    # Handle num_splits
    if num_splits != 1:
        raise NotImplementedError(f"Split attention (num_splits > 1) is not yet supported in the AMD Triton backend (got num_splits={num_splits})")
    
    # Handle sm_margin
    if sm_margin != 0:
        raise NotImplementedError(f"sm_margin is not yet supported in the AMD Triton backend (got sm_margin={sm_margin}, expected 0)")
    
    # Handle leftpad_k
    if leftpad_k is not None:
        raise NotImplementedError("Left padding (leftpad_k) is not yet supported in the AMD Triton backend")
    
    # Handle cu_seqlens_k_new
    if cu_seqlens_k_new is not None:
        raise NotImplementedError("cu_seqlens_k_new is not yet supported in the AMD Triton backend")

    # if seqlens_rotary is not None:
    #     raise NotImplementedError("seqlens_rotary is not yet supported in the AMD Triton backend")
    
    # Setup metadata
    metadata = MetaData(sm_scale=softmax_scale)


    # Handle variable length sequences first to determine layout
    # Determine layout based on tensor dimensions and cu_seqlens presence
    if cu_seqlens_q is not None:
        # Q has variable length - check tensor dimensions to confirm
        if len(q.shape) == 3:  # [total_seqlen, nheads, head_dim]
            metadata.layout = "thd"
            metadata.varlen = True
            metadata.cu_seqlens_q = cu_seqlens_q
            metadata.max_seqlens_q = max_seqlen_q
            
            # K might be varlen or batch mode
            if cu_seqlens_k is not None:
                metadata.cu_seqlens_k = cu_seqlens_k
                metadata.max_seqlens_k = max_seqlen_k
            else:
                # K is in batch mode while Q is varlen (KV cache scenario)
                metadata.cu_seqlens_k = None
                metadata.max_seqlens_k = k.shape[1] if len(k.shape) == 4 else max_seqlen_k
        else:
            raise ValueError(f"cu_seqlens_q provided but q has shape {q.shape}, expected 3D tensor for varlen")
    else:
        # Regular batch mode
        metadata.layout = "bshd"
        metadata.varlen = False
        metadata.cu_seqlens_q = None
        metadata.cu_seqlens_k = None
        metadata.max_seqlens_q = q.shape[1] if max_seqlen_q is None else max_seqlen_q
        metadata.max_seqlens_k = k.shape[1] if max_seqlen_k is None else max_seqlen_k

    # Now determine if we should use decode or prefill kernel
    # Decode kernel should be used for KV cache scenarios where:
    # 1. k_new/v_new are provided - incremental KV cache update (primary KV cache indicator)
    # 2. kv_batch_idx is provided - KV cache batch indexing (primary KV cache indicator)
    # 3. seqused_k without seqused_q - indicates KV cache fill levels (not varlen masking)
    # Note: In varlen, both seqused_q and seqused_k are used for sequence masking
    #       In KV cache, only seqused_k is used to track cache fill levels
    if USE_DECODE_PATH:
        # Force decode path
        use_decode = True
    else:
        # Detect KV cache scenarios:
        # - Clear KV cache indicators (k_new, v_new, kv_batch_idx)
        # - OR seqused_k without seqused_q (KV cache fill tracking, not varlen masking)
        use_decode = (
            k_new is not None or               # Have new KV to append (KV cache indicator)
            v_new is not None or               # Have new KV to append (KV cache indicator)
            kv_batch_idx is not None or        # Have KV cache batch indexing (KV cache indicator)
            (seqused_k is not None and seqused_q is None)  # KV cache fill levels (not varlen)
        )
    
    # Check for unsupported features with decode kernel
    if use_decode:
        if metadata.layout == "thd":
            raise NotImplementedError("Varlen is not yet supported with the decode kernel in the AMD Triton backend")
        if kv_batch_idx is not None:
            raise NotImplementedError("kv_batch_idx is not yet supported with the decode kernel in the AMD Triton backend")
        
    
    if out is None:
        out_dtype = torch.float32 if is_fp8(q) else q.dtype
        if metadata.layout == "bshd":
            out = torch.zeros(q.shape[0], q.shape[1], q.shape[2], v.shape[-1], dtype=out_dtype, device=q.device)
        elif metadata.layout == "thd":
            out = torch.zeros(q.shape[0], q.shape[1], v.shape[-1], dtype=out_dtype, device=q.device)
        else:
            raise ValueError(f"Unsupported layout: {metadata.layout}. Only 'bshd' and 'thd' layouts are supported.")
    else:
        out = out.zero_()
    
    if is_fp8(q):
        if (q_descale is None) or (k_descale is None) or (v_descale is None):
            import warnings
            warnings.warn("FP8 tensors detected but descale factors not provided. Using default scale of 1.0", UserWarning)
    
    # Get shape
    if metadata.layout == "bshd":
        batch, _, nheads_q, _ = q.shape
    else:  # "thd" layout for varlen
        _, nheads_q, _ = q.shape
        batch = len(cu_seqlens_q) - 1 if cu_seqlens_q is not None else 1
    
    # Handle causal mask
    if causal:
        metadata.need_causal(True)
    
    # Handle alibi slopes (not directly supported in v3 interface, but we'll keep the logic)
    alibi_slopes = None  # V3 doesn't have alibi_slopes in the signature
    if alibi_slopes is not None:
        if alibi_slopes.dim() == 2:
            pass
        elif alibi_slopes.dim() == 1:
            alibi_slopes = alibi_slopes.unsqueeze(0).expand(batch, -1)
        else:
            raise ValueError(f"Alibi can be (nheads,) or (batch_size, nheads). Given tensor with shape {alibi_slopes.shape}")
        metadata.need_alibi(alibi_slopes, batch, nheads_q)
    
    # Handle dropout (v3 doesn't have dropout in forward)
    dropout_p = 0.0
    return_softmax = False
    metadata.need_dropout(dropout_p, return_softmax)
    
    # Handle rotary embeddings
    if rotary_cos is not None and rotary_sin is not None:
        metadata.need_rotary(rotary_sin, rotary_cos, rotary_interleaved)
        
        # Apply rotary embeddings if provided
        if metadata.causal or window_size_left != -1 or window_size_right != -1:
            q_rot = apply_rotary_emb(
                q,
                rotary_cos,
                rotary_sin,
                seqlen_offsets=seqlens_rotary,
                interleaved=rotary_interleaved,
            )
            q = q_rot.to(q.dtype)
            
            if k_new is not None:
                k_rot = apply_rotary_emb(
                    k_new,
                    rotary_cos,
                    rotary_sin,
                    seqlen_offsets=seqlens_rotary,
                    interleaved=rotary_interleaved,
                )
                k_new = k_rot.to(k.dtype)
    
    # Store RNG state
    rng_state = torch.as_tensor([metadata.philox_seed, metadata.philox_offset])

    # Call implementation
    if USE_REF:
        if DEBUG:
            print("Using reference implementation")
        
        if use_decode:
            if DEBUG:
                print(f"Using decode reference implementation ( layout={metadata.layout}, cache_seqlens={seqused_k is not None}, k_new={k_new is not None}, v_new={v_new is not None}, kv_batch_idx={kv_batch_idx is not None})")
            # Use decode reference implementation
            softmax_lse = attention_decode_forward_ref_impl(
                q,
                k,  # k_cache
                v,  # v_cache
                k_new,
                v_new,
                out,
                metadata.sm_scale,
                metadata.causal,
                window_size_left,
                window_size_right,
                metadata.alibi_slopes,
                metadata.layout,
                seqused_k,  # cache_seqlens
                kv_batch_idx,  # cache_batch_idx
                page_table,  # block_table
                q_descale,
                k_descale,
                v_descale,
            )
        else:
            if DEBUG:
                print("Using prefill reference implementation")
            # Use prefill reference implementation
            softmax_lse_ref, sd_mask_ref = attention_prefill_forward_ref_impl(
                q, k, v, out,
                metadata.sm_scale,
                metadata.alibi_slopes,
                metadata.causal,
                window_size_left,
                window_size_right,
                metadata.layout,
                metadata.cu_seqlens_q,
                metadata.cu_seqlens_k,
                metadata.max_seqlens_q,
                metadata.max_seqlens_k,
                metadata.dropout_p,
                metadata.philox_seed,
                metadata.philox_offset,
                USE_EXP2
            )
            softmax_lse = softmax_lse_ref
    else:
        if DEBUG:
            print("Using Triton implementation")
        
        if use_decode:
            if DEBUG:
                print(f"Using Decode Triton implementation (cache_seqlens={seqused_k is not None}, k_new={k_new is not None}, v_new={v_new is not None}, kv_batch_idx={kv_batch_idx is not None})")
            
            # Use decode kernel for KV cache scenarios
            # Note: seqused_k can serve as cache_seqlens in v3
            softmax_lse = attention_decode_forward_triton_impl(
                q,
                k,  # k_cache in v2 terminology
                v,  # v_cache in v2 terminology
                k_new,  # New KV values to append to cache
                v_new,  # New KV values to append to cache
                out,
                metadata.sm_scale,
                metadata.causal,
                window_size_left,
                window_size_right,
                metadata.alibi_slopes,
                metadata.layout,
                seqused_k,  # cache_seqlens
                kv_batch_idx,  # cache_batch_idx
                page_table,  # block_table for paged attention
                q_descale,
                k_descale,
                v_descale,
            )
            # Decode kernel returns only softmax_lse, not sd_mask
            sd_mask_triton = None
        else:
            if DEBUG:
                print("Using prefill Triton implementation")
            # Use prefill kernel
            softmax_lse_triton, sd_mask_triton = attention_prefill_forward_triton_impl(
                q, k, v, out,
                metadata.sm_scale,
                metadata.alibi_slopes,
                metadata.causal,
                window_size_left,
                window_size_right,
                None,  # block_table
                metadata.layout,
                metadata.cu_seqlens_q,
                metadata.cu_seqlens_k,
                metadata.max_seqlens_q,
                metadata.max_seqlens_k,
                metadata.dropout_p,
                metadata.philox_seed,
                metadata.philox_offset,
                metadata.return_softmax,
                USE_EXP2,
                q_descale,
                k_descale,
                v_descale,
                seqused_q,
                seqused_k,
            )
            softmax_lse = softmax_lse_triton
    
    if DEBUG:
        print("interface_fa_v3.py::fwd outputs")
        print("out:", out, out.shape)
        print("softmax_lse:", softmax_lse, softmax_lse.shape)
    
    # Return format compatible with v3
    # V3 returns (out, softmax_lse, *rest) where rest can be empty or contain additional outputs
    return out, softmax_lse


def bwd(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: Optional[torch.Tensor],
    dk: Optional[torch.Tensor],
    dv: Optional[torch.Tensor],
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    seqused_q: Optional[torch.Tensor],
    seqused_k: Optional[torch.Tensor],
    max_seqlen_q: Optional[int],
    max_seqlen_k: Optional[int],
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    deterministic: bool,
    sm_margin: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Flash Attention v3 backward pass compatible interface for AMD Triton implementation.
    
    This function maps v3 parameters to the existing AMD Triton implementation.
    """
    
    if DEBUG:
        print()
        print("interface_fa_v3.py::bwd inputs")
        print("dout:", dout, dout.shape)
        print("q:", q, q.shape)
        print("k:", k, k.shape)
        print("v:", v, v.shape)
        print("out:", out, out.shape)
        print("softmax_lse:", softmax_lse, softmax_lse.shape)
        print("dq:", dq, dq.shape if dq is not None else None)
        print("dk:", dk, dk.shape if dk is not None else None)
        print("dv:", dv, dv.shape if dv is not None else None)
        print("cu_seqlens_q:", cu_seqlens_q, cu_seqlens_q.shape if cu_seqlens_q is not None else None)
        print("cu_seqlens_k:", cu_seqlens_k, cu_seqlens_k.shape if cu_seqlens_k is not None else None)
        print("seqused_q:", seqused_q, seqused_q.shape if seqused_q is not None else None)
        print("seqused_k:", seqused_k, seqused_k.shape if seqused_k is not None else None)
        print("max_seqlen_q:", max_seqlen_q)
        print("max_seqlen_k:", max_seqlen_k)
        print("softmax_scale:", softmax_scale)
        print("causal:", causal)
        print("window_size_left:", window_size_left)
        print("window_size_right:", window_size_right)
        print("softcap:", softcap)
        print("deterministic:", deterministic)
        print("sm_margin:", sm_margin)
    
    # Check for unsupported features in backward pass
    
    # Handle softcap
    if softcap != 0.0:
        raise NotImplementedError(f"Softcap is not yet supported in the AMD Triton backend backward pass (got softcap={softcap}, expected 0.0)")

    # Handle sm_margin  
    if sm_margin != 0:
        raise NotImplementedError(f"sm_margin is not yet supported in the AMD Triton backend backward pass (got sm_margin={sm_margin}, expected 0)")
    
    # Initialize gradient tensors if not provided
    dq = torch.zeros_like(q) if dq is None else dq.zero_()
    dk = torch.zeros_like(k) if dk is None else dk.zero_()
    dv = torch.zeros_like(v) if dv is None else dv.zero_()
    
    # Determine layout based on cu_seqlens
    if cu_seqlens_q is not None and cu_seqlens_k is not None:
        # Variable length sequence mode
        layout = "thd"
        batch = len(cu_seqlens_q) - 1
        _, nheads_q, _ = q.shape
    else:
        # Regular batch mode
        layout = "bshd"
        batch, _, nheads_q, _ = q.shape
        max_seqlen_q = q.shape[1] if max_seqlen_q is None else max_seqlen_q
        max_seqlen_k = k.shape[1] if max_seqlen_k is None else max_seqlen_k
    
    # V3 backward doesn't have dropout or alibi slopes
    dropout_p = 0.0
    philox_seed, philox_offset = None, None
    alibi_slopes = None
    
    # For fp8, we would need descale factors, but v3 interface doesn't expose them
    # So we'll pass None for now
    descale_q = None
    descale_k = None
    descale_v = None
    descale_o = None
    descale_do = None
    descale_dq = None
    descale_dk = None
    descale_dv = None
    
    # Call implementation
    if USE_REF:
        if DEBUG:
            print("Using reference implementation")
        delta_ref = attention_backward_pytorch_ref_impl(
            dout, q, k, v, out, softmax_lse,
            dq, dk, dv,
            softmax_scale,
            alibi_slopes,
            causal,
            window_size_left,
            window_size_right,
            layout,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            philox_seed,
            philox_offset,
            USE_EXP2,
        )
        delta = delta_ref
    else:
        if DEBUG:
            print("Using Triton implementation")
        
        if BWD_MODE == "split":
            delta_triton = attention_prefill_backward_triton_split_impl(
                dout, q, k, v, out, softmax_lse,
                dq, dk, dv,
                softmax_scale,
                alibi_slopes,
                causal,
                layout,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                dropout_p,
                philox_seed,
                philox_offset,
                USE_EXP2,
                descale_q, descale_k, descale_v, descale_o,
                descale_do, descale_dq, descale_dk, descale_dv,
                seqused_q, seqused_k,
            )
            delta = delta_triton
        elif BWD_MODE == "fused_atomics":
            delta_triton = attention_prefill_backward_triton_fused_atomics_impl(
                dout, q, k, v, out, softmax_lse,
                dq, dk, dv,
                softmax_scale,
                alibi_slopes,
                causal,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                dropout_p,
                philox_seed,
                philox_offset,
                descale_q, descale_k, descale_v, descale_o,
                True,
            )
            delta = delta_triton
        elif BWD_MODE == "fused_no_atomics":
            delta_triton = attention_prefill_backward_triton_split_fused_no_atomics_impl(
                dout, q, k, v, out, softmax_lse,
                dq, dk, dv,
                softmax_scale,
                alibi_slopes,
                causal,
                layout,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                dropout_p,
                philox_seed,
                philox_offset,
                USE_EXP2,
                descale_q, descale_k, descale_v, descale_o,
                descale_do, descale_dq, descale_dk, descale_dv,
                seqused_q, seqused_k,
            )
            delta = delta_triton
        else:
            raise ValueError(f"Unknown bwd mode {BWD_MODE}")
    
    if DEBUG:
        print("interface_fa_v3.py::bwd outputs")
        print("dq:", dq, dq.shape)
        print("dk:", dk, dk.shape)
        print("dv:", dv, dv.shape)
        print("delta:", delta, delta.shape if delta is not None else None)
    
    # V3 expects (dq, dk, dv, softmax_d, *rest)
    # delta is the softmax_d in this case
    return dq, dk, dv, delta


def fwd_combine(
    out_partial: torch.Tensor,
    lse_partial: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Combine partial outputs from split attention computation.
    
    This is used when num_splits > 1 to combine the partial results.
    
    Args:
        out_partial: Partial output tensor from split computation
        lse_partial: Partial log-sum-exp tensor
        out: Optional output tensor to write to
        out_dtype: Optional dtype for output
    
    Returns:
        Combined output tensor
    """
    raise NotImplementedError("fwd_combine is not yet implemented in the AMD Triton backend")


def get_scheduler_metadata(
    batch_size: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    num_heads_q: int,
    num_heads_kv: int,
    headdim: int,
    headdim_v: int,
    qkv_dtype: torch.dtype,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    cu_seqlens_k_new: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    page_size: Optional[int] = None,
    max_seqlen_k_new: int = 0,
    causal: bool = False,
    window_size_left: int = -1,
    window_size_right: int = -1,
    attention_chunk: int = 0,
    has_softcap: bool = False,
    num_splits: int = 0,
    pack_gqa: Optional[bool] = None,
    sm_margin: int = 0,
):
    """
    Get scheduler metadata for optimized kernel selection.
    
    This function is used to precompute metadata for kernel scheduling in FA3.
    The AMD Triton backend currently doesn't use scheduler metadata, so this
    raises an error.
    
    Args:
        Various attention parameters used for scheduling decisions
    
    Returns:
        None - scheduler metadata is not used in AMD Triton backend
    """
    raise NotImplementedError("get_scheduler_metadata is not supported in the AMD Triton backend yet.")