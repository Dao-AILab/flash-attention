import os
import warnings
import torch
from typing import Optional, Union, Tuple
from .fwd_prefill import attention_forward_prefill_triton_impl
from .fwd_decode import attention_forward_decode_triton_impl
from .bwd import attention_backward_triton_impl
from .utils import (
    DEBUG,
    USE_EXP2,
    BWD_MODE,
    PHILOX_SEED,
    PHILOX_OFFSET,
    is_fp8,
    get_recommended_fp8_dtype,
)


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
        print("q:", q.dtype if q is not None else None, q.shape)
        print("k:", k.dtype if k is not None else None, k.shape)
        print("v:", v.dtype if v is not None else None, v.shape)
        print(
            "k_new:",
            k_new.dtype if k_new is not None else None,
            k_new.shape if k_new is not None else None,
        )
        print(
            "v_new:",
            v_new.dtype if v_new is not None else None,
            v_new.shape if v_new is not None else None,
        )
        print(
            "qv:",
            qv.dtype if qv is not None else None,
            qv.shape if qv is not None else None,
        )
        print(
            "out:",
            out.dtype if out is not None else None,
            out.shape if out is not None else None,
        )
        print(
            "cu_seqlens_q:",
            cu_seqlens_q,
            cu_seqlens_q.shape if cu_seqlens_q is not None else None,
        )
        print(
            "cu_seqlens_k:",
            cu_seqlens_k,
            cu_seqlens_k.shape if cu_seqlens_k is not None else None,
        )
        print(
            "cu_seqlens_k_new:",
            cu_seqlens_k_new,
            cu_seqlens_k_new.shape if cu_seqlens_k_new is not None else None,
        )
        print(
            "seqused_q:", seqused_q, seqused_q.shape if seqused_q is not None else None
        )
        print(
            "seqused_k:", seqused_k, seqused_k.shape if seqused_k is not None else None
        )
        print("max_seqlen_q:", max_seqlen_q)
        print("max_seqlen_k:", max_seqlen_k)
        print(
            "page_table:",
            page_table,
            page_table.shape if page_table is not None else None,
        )
        print(
            "kv_batch_idx:",
            kv_batch_idx,
            kv_batch_idx.shape if kv_batch_idx is not None else None,
        )
        print(
            "leftpad_k:", leftpad_k, leftpad_k.shape if leftpad_k is not None else None
        )
        print(
            "rotary_cos:",
            rotary_cos,
            rotary_cos.shape if rotary_cos is not None else None,
        )
        print(
            "rotary_sin:",
            rotary_sin,
            rotary_sin.shape if rotary_sin is not None else None,
        )
        print(
            "seqlens_rotary:",
            seqlens_rotary,
            seqlens_rotary.shape if seqlens_rotary is not None else None,
        )
        print(
            "q_descale:",
            q_descale.dtype if q_descale is not None else None,
            q_descale.shape if q_descale is not None else None,
        )
        print(
            "k_descale:",
            k_descale.dtype if k_descale is not None else None,
            k_descale.shape if k_descale is not None else None,
        )
        print(
            "v_descale:",
            v_descale.dtype if v_descale is not None else None,
            v_descale.shape if v_descale is not None else None,
        )
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
        raise NotImplementedError(
            "QV packed input is not yet supported in the AMD Triton backend"
        )

    # Handle softcap
    if softcap != 0.0:
        raise NotImplementedError(
            f"Softcap is not yet supported in the AMD Triton backend (got softcap={softcap}, expected 0.0)"
        )

    # Handle attention_chunk
    if attention_chunk != 0 and attention_chunk != 1:
        raise NotImplementedError(
            f"attention_chunk is not yet supported in the AMD Triton backend (got attention_chunk={attention_chunk})"
        )

    # Handle scheduler metadata
    if scheduler_metadata is not None:
        raise NotImplementedError(
            "Scheduler metadata is not yet supported in the AMD Triton backend"
        )

    # Handle pack_gqa
    if pack_gqa is not None and pack_gqa is not False:
        raise NotImplementedError(
            f"pack_gqa is not yet supported in the AMD Triton backend (got pack_gqa={pack_gqa})"
        )

    # Handle num_splits
    if num_splits != 1:
        raise NotImplementedError(
            f"Split attention (num_splits > 1) is not yet supported in the AMD Triton backend (got num_splits={num_splits})"
        )

    # Handle sm_margin
    if sm_margin != 0:
        raise NotImplementedError(
            f"sm_margin is not yet supported in the AMD Triton backend (got sm_margin={sm_margin}, expected 0)"
        )

    # Handle leftpad_k
    if leftpad_k is not None:
        raise NotImplementedError(
            "Left padding (leftpad_k) is not yet supported in the AMD Triton backend"
        )

    # Handle cu_seqlens_k_new
    if cu_seqlens_k_new is not None:
        raise NotImplementedError(
            "cu_seqlens_k_new is not yet supported in the AMD Triton backend"
        )

    # establish layout / varlen & max seq lens
    if cu_seqlens_q is not None:
        if len(q.shape) != 3:
            raise ValueError(
                f"cu_seqlens_q provided but q has shape {q.shape}, expected 3D tensor for varlen"
            )
        layout = "thd"
        cu_seqlens_q_local = cu_seqlens_q
        max_seqlens_q_local = max_seqlen_q
        if cu_seqlens_k is not None:
            cu_seqlens_k_local = cu_seqlens_k
            max_seqlens_k_local = max_seqlen_k
        else:
            cu_seqlens_k_local = None
            max_seqlens_k_local = k.shape[1] if len(k.shape) == 4 else max_seqlen_k
    else:
        layout = "bshd"
        cu_seqlens_q_local = None
        cu_seqlens_k_local = None
        max_seqlens_q_local = q.shape[1] if max_seqlen_q is None else max_seqlen_q
        max_seqlens_k_local = k.shape[1] if max_seqlen_k is None else max_seqlen_k

    # Now determine if we should use decode or prefill kernel
    # Decode kernel should be used for KV cache scenarios where:
    # 1. k_new/v_new are provided - incremental KV cache update (primary KV cache indicator)
    # 2. kv_batch_idx is provided - KV cache batch indexing (primary KV cache indicator)
    # 3. seqused_k without seqused_q - indicates KV cache fill levels (not varlen masking)
    # Note: In varlen, both seqused_q and seqused_k are used for sequence masking
    #       In KV cache, only seqused_k is used to track cache fill levels
    # Detect KV cache scenarios:
    # - Clear KV cache indicators (k_new, v_new, kv_batch_idx)
    # - OR seqused_k without seqused_q (KV cache fill tracking, not varlen masking)
    use_decode = (
        k_new is not None  # Have new KV to append (KV cache indicator)
        or v_new is not None  # Have new KV to append (KV cache indicator)
        or kv_batch_idx is not None  # Have KV cache batch indexing (KV cache indicator)
        or (
            seqused_k is not None and seqused_q is None
        )  # KV cache fill levels (not varlen)
    )

    # Check for unsupported features with decode kernel
    if use_decode:
        if layout == "thd":
            raise NotImplementedError(
                "Varlen is not yet supported with the decode kernel in the AMD Triton backend"
            )
        if kv_batch_idx is not None:
            raise NotImplementedError(
                "kv_batch_idx is not yet supported with the decode kernel in the AMD Triton backend"
            )

    if out is None:
        # NOTE: Using types that are lower precision than float32 such as bfloat16 for fp8 causes mismatches on a small set of tests.
        out_dtype = torch.float32 if is_fp8([q, k, v]) else q.dtype
        if layout == "bshd":
            out = torch.zeros(
                q.shape[0],
                q.shape[1],
                q.shape[2],
                v.shape[-1],
                dtype=out_dtype,
                device=q.device,
            )
        elif layout == "thd":
            out = torch.zeros(
                q.shape[0], q.shape[1], v.shape[-1], dtype=out_dtype, device=q.device
            )
        else:
            raise ValueError(
                f"Unsupported layout: {layout}. Only 'bshd' and 'thd' layouts are supported."
            )
    else:
        out = out.zero_()

    # Handle causal mask
    causal_flag = bool(causal)

    # Handle alibi slopes
    alibi_slopes = None

    # Handle dropout
    dropout_p = 0.0
    return_softmax = False
    philox_seed = PHILOX_SEED
    philox_offset = PHILOX_OFFSET

    # Call implementation
    if DEBUG:
        print("Using Triton implementation")

    if use_decode:
        if DEBUG:
            print(
                f"Using Decode Triton implementation (cache_seqlens={seqused_k is not None}, k_new={k_new is not None}, v_new={v_new is not None}, kv_batch_idx={kv_batch_idx is not None})"
            )

        # Create softmax_lse tensor for decode - always exact shape (B, Hq, Sq)
        batch, seqlen_q, nheads_q, _ = q.shape
        softmax_lse = torch.zeros(
            (batch, nheads_q, seqlen_q), device=q.device, dtype=torch.float32
        )

        attention_forward_decode_triton_impl(
            q,
            k,
            v,
            k_new,
            v_new,
            out,
            softmax_lse,
            softmax_scale,
            causal_flag,
            window_size_left,
            window_size_right,
            alibi_slopes,
            layout,
            seqused_k,
            kv_batch_idx,
            page_table,
            q_descale,
            k_descale,
            v_descale,
            rotary_cos=rotary_cos,
            rotary_sin=rotary_sin,
            rotary_interleaved=rotary_interleaved,
            seqlens_rotary=seqlens_rotary,
        )
    else:
        if DEBUG:
            print("Using Prefill Triton implementation")

        # Create softmax_lse tensor - FA3 always uses exact shapes
        if layout == "thd":
            # varlen: (Hq, Total_Q)
            total_q, nheads_q, _ = q.shape
            softmax_lse = torch.zeros(
                (nheads_q, total_q), device=q.device, dtype=torch.float32
            )
        else:
            # bshd: (B, Hq, Sq)
            batch, seqlen_q, nheads_q, _ = q.shape
            softmax_lse = torch.zeros(
                (batch, nheads_q, seqlen_q), device=q.device, dtype=torch.float32
            )

        # sd_mask is not returned in v3 interface
        sd_mask = None

        attention_forward_prefill_triton_impl(
            q,
            k,
            v,
            out,
            softmax_lse,
            sd_mask,
            softmax_scale,
            alibi_slopes,
            causal_flag,
            window_size_left,
            window_size_right,
            None,
            layout,
            cu_seqlens_q_local,
            cu_seqlens_k_local,
            max_seqlens_q_local,
            max_seqlens_k_local,
            dropout_p,
            philox_seed,
            philox_offset,
            return_softmax,
            USE_EXP2,
            q_descale,
            k_descale,
            v_descale,
            seqused_q,
            seqused_k,
            rotary_cos=rotary_cos,
            rotary_sin=rotary_sin,
            rotary_interleaved=rotary_interleaved,
            seqlens_rotary=seqlens_rotary,
        )

    if DEBUG:
        print("interface_fa_v3.py::fwd outputs")
        print(
            "out:",
            out.dtype if out is not None else None,
            out.shape if out is not None else None,
        )
        print(
            "softmax_lse:",
            softmax_lse.dtype if softmax_lse is not None else None,
            softmax_lse.shape if softmax_lse is not None else None,
        )

    # --- Assertions (FA3 always expects exact shapes) ---
    # out: same shape as q except last dim is v's head_dim
    if layout == "thd":
        # varlen: (Total_Q, Hq, Dv)
        assert (
            out.shape[0] == q.shape[0]
        ), f"[fwd_v3] out.shape[0] {out.shape[0]} != q.shape[0] {q.shape[0]}"
        assert (
            out.shape[1] == q.shape[1]
        ), f"[fwd_v3] out.shape[1] {out.shape[1]} != q.shape[1] {q.shape[1]}"
        assert (
            out.shape[2] == v.shape[-1]
        ), f"[fwd_v3] out.shape[2] {out.shape[2]} != v.shape[-1] {v.shape[-1]}"
    else:
        # bshd: (B, Sq, Hq, Dv)
        assert (
            out.shape[0] == q.shape[0]
        ), f"[fwd_v3] out.shape[0] {out.shape[0]} != q.shape[0] {q.shape[0]}"
        assert (
            out.shape[1] == q.shape[1]
        ), f"[fwd_v3] out.shape[1] {out.shape[1]} != q.shape[1] {q.shape[1]}"
        assert (
            out.shape[2] == q.shape[2]
        ), f"[fwd_v3] out.shape[2] {out.shape[2]} != q.shape[2] {q.shape[2]}"
        assert (
            out.shape[3] == v.shape[-1]
        ), f"[fwd_v3] out.shape[3] {out.shape[3]} != v.shape[-1] {v.shape[-1]}"

    # softmax_lse dtype
    assert (
        softmax_lse.dtype == torch.float32
    ), f"[fwd_v3] softmax_lse dtype {softmax_lse.dtype} != torch.float32"
    # softmax_lse shape depends on layout
    if layout == "thd":
        # varlen: (Hq, Total_Q)
        expected_lse_shape = (q.shape[1], q.shape[0])
    else:
        # bshd: (B, Hq, Sq)
        expected_lse_shape = (q.shape[0], q.shape[2], q.shape[1])
    assert (
        softmax_lse.shape == expected_lse_shape
    ), f"[fwd_v3] softmax_lse shape {softmax_lse.shape} != {expected_lse_shape}"

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
        print(
            "dout:",
            dout.dtype if dout is not None else None,
            dout.shape if dout is not None else None,
        )
        print(
            "q:", q.dtype if q is not None else None, q.shape if q is not None else None
        )
        print(
            "k:", k.dtype if k is not None else None, k.shape if k is not None else None
        )
        print(
            "v:", v.dtype if v is not None else None, v.shape if v is not None else None
        )
        print(
            "out:",
            out.dtype if out is not None else None,
            out.shape if out is not None else None,
        )
        print(
            "softmax_lse:",
            softmax_lse.dtype if softmax_lse is not None else None,
            softmax_lse.shape if softmax_lse is not None else None,
        )
        print(
            "dq:",
            dq.dtype if dq is not None else None,
            dq.shape if dq is not None else None,
        )
        print(
            "dk:",
            dk.dtype if dk is not None else None,
            dk.shape if dk is not None else None,
        )
        print(
            "dv:",
            dv.dtype if dv is not None else None,
            dv.shape if dv is not None else None,
        )
        print(
            "cu_seqlens_q:",
            cu_seqlens_q,
            cu_seqlens_q.shape if cu_seqlens_q is not None else None,
        )
        print(
            "cu_seqlens_k:",
            cu_seqlens_k,
            cu_seqlens_k.shape if cu_seqlens_k is not None else None,
        )
        print(
            "seqused_q:", seqused_q, seqused_q.shape if seqused_q is not None else None
        )
        print(
            "seqused_k:", seqused_k, seqused_k.shape if seqused_k is not None else None
        )
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
        raise NotImplementedError(
            f"Softcap is not yet supported in the AMD Triton backend backward pass (got softcap={softcap}, expected 0.0)"
        )

    # Handle sm_margin
    if sm_margin != 0:
        raise NotImplementedError(
            f"sm_margin is not yet supported in the AMD Triton backend backward pass (got sm_margin={sm_margin}, expected 0)"
        )

    # Initialize gradient tensors if not provided
    # NOTE: Using types that are lower precision than float32 such as bfloat16 for fp8 causes mismatches on a small set of tests.
    grad_dtype = torch.float32 if is_fp8([q, k, v]) else q.dtype
    dq = torch.zeros_like(q, dtype=grad_dtype) if dq is None else dq.zero_()
    dk = torch.zeros_like(k, dtype=grad_dtype) if dk is None else dk.zero_()
    dv = torch.zeros_like(v, dtype=grad_dtype) if dv is None else dv.zero_()

    # Determine layout based on cu_seqlens
    if cu_seqlens_q is not None and cu_seqlens_k is not None:
        # Variable length sequence mode
        layout = "thd"
        batch = len(cu_seqlens_q) - 1
        total_q, nheads_q, _ = q.shape
        # Create delta tensor - varlen: (Hq, Total_Q)
        delta = torch.zeros((nheads_q, total_q), device=q.device, dtype=torch.float32)
    else:
        # Regular batch mode
        layout = "bshd"
        batch, seqlen_q, nheads_q, _ = q.shape
        max_seqlen_q = q.shape[1] if max_seqlen_q is None else max_seqlen_q
        max_seqlen_k = k.shape[1] if max_seqlen_k is None else max_seqlen_k
        # Create delta tensor - bshd: (B, Hq, Sq)
        delta = torch.zeros(
            (batch, nheads_q, seqlen_q), device=q.device, dtype=torch.float32
        )

    # V3 backward doesn't have dropout or alibi slopes
    dropout_p = 0.0
    philox_seed, philox_offset = None, None
    alibi_slopes = None

    # Call implementation
    if DEBUG:
        print(f"Using Triton implementation in {BWD_MODE} mode")
    attention_backward_triton_impl(
        do=dout,
        q=q,
        k=k,
        v=v,
        o=out,
        softmax_lse=softmax_lse,
        dq=dq,
        dk=dk,
        dv=dv,
        delta=delta,
        sm_scale=softmax_scale,
        alibi_slopes=alibi_slopes,
        causal=causal,
        layout=layout,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        dropout_p=dropout_p,
        philox_seed=philox_seed,
        philox_offset=philox_offset,
        use_exp2=USE_EXP2,
        mode=BWD_MODE,
    )

    if DEBUG:
        print("interface_fa_v3.py::bwd outputs")
        print(
            "dq:",
            dq.dtype if dq is not None else None,
            dq.shape if dq is not None else None,
        )
        print(
            "dk:",
            dk.dtype if dk is not None else None,
            dk.shape if dk is not None else None,
        )
        print(
            "dv:",
            dv.dtype if dv is not None else None,
            dv.shape if dv is not None else None,
        )
        print(
            "delta:",
            delta.dtype if delta is not None else None,
            delta.shape if delta is not None else None,
        )

    # --- Assertions (FA3 always expects exact shapes) ---
    # Gradients should match input shapes
    assert dq.shape == q.shape, f"[bwd_v3] dq shape {dq.shape} != q shape {q.shape}"
    assert dk.shape == k.shape, f"[bwd_v3] dk shape {dk.shape} != k shape {k.shape}"
    assert dv.shape == v.shape, f"[bwd_v3] dv shape {dv.shape} != v shape {v.shape}"
    # delta (softmax_d) should match softmax_lse shape
    assert (
        delta.dtype == torch.float32
    ), f"[bwd_v3] delta dtype {delta.dtype} != torch.float32"
    if layout == "thd":
        # varlen: (Hq, Total_Q)
        expected_delta_shape = (q.shape[1], q.shape[0])
    else:
        # bshd: (B, Hq, Sq)
        expected_delta_shape = (q.shape[0], q.shape[2], q.shape[1])
    assert (
        delta.shape == expected_delta_shape
    ), f"[bwd_v3] delta shape {delta.shape} != {expected_delta_shape}"

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
    raise NotImplementedError(
        "fwd_combine is not yet implemented in the AMD Triton backend"
    )


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
    raise NotImplementedError(
        "get_scheduler_metadata is not supported in the AMD Triton backend yet."
    )
