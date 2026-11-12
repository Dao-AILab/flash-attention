"""FA3 low-level ABI backed by aiter's Triton MHA v3."""

from types import SimpleNamespace

from flash_attn_interface import AMD_BACKEND


def fwd(
    q,
    k,
    v,
    k_new,
    v_new,
    qv,
    out,
    cu_seqlens_q,
    cu_seqlens_k,
    cu_seqlens_k_new,
    seqused_q,
    seqused_k,
    max_seqlen_q,
    max_seqlen_k,
    page_table,
    kv_batch_idx,
    leftpad_k,
    rotary_cos,
    rotary_sin,
    seqlens_rotary,
    q_descale,
    k_descale,
    v_descale,
    softmax_scale,
    causal,
    window_size_left,
    window_size_right,
    attention_chunk,
    softcap,
    rotary_interleaved,
    scheduler_metadata,
    num_splits,
    pack_gqa,
    sm_margin,
):
    if AMD_BACKEND.impl != "triton":
        raise NotImplementedError(
            "FA3 is not supported by the aiter general implementation. "
            "Use FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE."
        )
    if scheduler_metadata is not None:
        raise NotImplementedError("scheduler_metadata is not supported by the FA3 aiter adapter")
    from aiter.ops.triton.attention import mha_v3 as triton_mha_v3

    out, softmax_lse = triton_mha_v3._flash_attn_forward(
        q,
        k,
        v,
        softmax_scale,
        causal,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        attention_chunk=attention_chunk,
        softcap=softcap,
        k_new=k_new,
        v_new=v_new,
        qv=qv,
        out=out,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        cu_seqlens_k_new=cu_seqlens_k_new,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        page_table=page_table,
        kv_batch_idx=kv_batch_idx,
        leftpad_k=leftpad_k,
        rotary_cos=rotary_cos,
        rotary_sin=rotary_sin,
        seqlens_rotary=seqlens_rotary,
        rotary_interleaved=rotary_interleaved,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        sm_margin=sm_margin,
    )
    # Like the Dao-AI forward, no split-KV accumulators are returned.
    return out, softmax_lse, None, None


def bwd(
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    dq,
    dk,
    dv,
    cu_seqlens_q,
    cu_seqlens_k,
    seqused_q,
    seqused_k,
    max_seqlen_q,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size_left,
    window_size_right,
    softcap,
    deterministic,
    sm_margin,
):
    if AMD_BACKEND.impl != "triton":
        raise NotImplementedError(
            "FA3 is not supported by the aiter general implementation. "
            "Use FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE."
        )
    from aiter.ops.triton.attention import mha_v3 as triton_mha_v3

    softmax_d = triton_mha_v3._flash_attn_backward(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        dq,
        dk,
        dv,
        softmax_scale,
        causal,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        softcap=softcap,
        deterministic=deterministic,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        sm_margin=sm_margin,
    )
    return (softmax_d,)


def fwd_combine(out_partial, lse_partial, out=None, out_dtype=None):
    if AMD_BACKEND.impl != "triton":
        raise NotImplementedError(
            "FA3 is not supported by the aiter general implementation. "
            "Use FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE."
        )
    from aiter.ops.triton.attention import mha_v3 as triton_mha_v3

    return triton_mha_v3._flash_attn_combine(out_partial, lse_partial, out, out_dtype)


def get_scheduler_metadata(*args, **kwargs):
    raise NotImplementedError("get_scheduler_metadata is not supported by the FA3 aiter adapter")


flash_attn_3 = SimpleNamespace(
    fwd=fwd,
    bwd=bwd,
    fwd_combine=fwd_combine,
    get_scheduler_metadata=get_scheduler_metadata,
)
