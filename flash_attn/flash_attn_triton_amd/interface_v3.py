import torch
import os
from typing import Optional, Union, Tuple
from .fwd_prefill import attention_forward_prefill_triton_impl
from .fwd_decode import attention_forward_decode_triton_impl
from .bwd import attention_backward_triton_impl
from .utils import DEBUG, USE_EXP2, BWD_MODE, PHILOX_SEED, PHILOX_OFFSET, is_fp8


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
            "q_descale:", q_descale, q_descale.shape if q_descale is not None else None
        )
        print(
            "k_descale:", k_descale, k_descale.shape if k_descale is not None else None
        )
        print(
            "v_descale:", v_descale, v_descale.shape if v_descale is not None else None
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

    # if seqlens_rotary is not None:
    #     raise NotImplementedError("seqlens_rotary is not yet supported in the AMD Triton backend")

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
        out_dtype = torch.float32 if is_fp8(q) else q.dtype
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

    if is_fp8(q):
        if (q_descale is None) or (k_descale is None) or (v_descale is None):
            import warnings

            warnings.warn(
                "FP8 tensors detected but descale factors not provided. Using default scale of 1.0",
                UserWarning,
            )
        else:
            # Enforce exact expected shapes; no reshaping or normalization.
            if layout == "bshd":
                expected_batch = q.shape[0]
                expected_q_heads = q.shape[2]
                expected_kv_heads = k.shape[2]
            else:  # thd layout
                expected_batch = (
                    (len(cu_seqlens_q_local) - 1)
                    if cu_seqlens_q_local is not None
                    else 1
                )
                expected_q_heads = q.shape[1]
                expected_kv_heads = k.shape[1]

            assert (
                q_descale.dim() == 2
                and q_descale.shape[0] == expected_batch
                and q_descale.shape[1] == expected_kv_heads
            ), f"q_descale expected shape ({expected_batch}, {expected_kv_heads}) got {tuple(q_descale.shape)}"
            assert (
                k_descale.dim() == 2
                and k_descale.shape[0] == expected_batch
                and k_descale.shape[1] == expected_kv_heads
            ), f"k_descale expected shape ({expected_batch}, {expected_kv_heads}) got {tuple(k_descale.shape)}"
            assert (
                v_descale.dim() == 2
                and v_descale.shape[0] == expected_batch
                and v_descale.shape[1] == expected_kv_heads
            ), f"v_descale expected shape ({expected_batch}, {expected_kv_heads}) got {tuple(v_descale.shape)}"

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

        softmax_lse = attention_forward_decode_triton_impl(
            q,
            k,
            v,
            k_new,
            v_new,
            out,
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
        softmax_lse, _ = attention_forward_prefill_triton_impl(
            q,
            k,
            v,
            out,
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

    # Call implementation
    if DEBUG:
        print("Using Triton implementation (unified backward dispatcher)")
    delta = attention_backward_triton_impl(
        do=dout,
        q=q,
        k=k,
        v=v,
        o=out,
        softmax_lse=softmax_lse,
        dq=dq,
        dk=dk,
        dv=dv,
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
