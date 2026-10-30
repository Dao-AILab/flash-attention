"""FA2 low-level ABI backed by the ROCm Triton or general aiter dispatcher."""

from types import SimpleNamespace

from aiter.ops.triton._triton_kernels.attention.dao_ai import flash_attn_2 as _triton_fa2
from flash_attn.flash_attn_interface import AMD_BACKEND


def _check_general_args(softcap, gen):
    if softcap != 0.0:
        raise NotImplementedError("softcap is not supported by the FA2 aiter adapter")
    if gen is not None:
        raise NotImplementedError(
            "an explicit dropout generator is not supported by the FA2 aiter adapter"
        )


def fwd(
    q,
    k,
    v,
    out,
    alibi_slopes,
    dropout_p,
    softmax_scale,
    causal,
    window_size_left,
    window_size_right,
    softcap,
    return_softmax,
    gen=None,
):
    if AMD_BACKEND == "triton":
        return _triton_fa2.fwd(
            q,
            k,
            v,
            out,
            alibi_slopes,
            dropout_p,
            softmax_scale,
            causal,
            window_size_left,
            window_size_right,
            softcap,
            return_softmax,
            gen,
        )
    else:
        _check_general_args(softcap, gen)
        from aiter.ops import mha

        return mha._flash_attn_forward(
            q,
            k,
            v,
            dropout_p,
            softmax_scale,
            causal,
            window_size_left,
            window_size_right,
            0,
            None,
            alibi_slopes,
            None,
            None,
            None,
            True,
            return_softmax and dropout_p > 0,
            out=out,
        )


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
    alibi_slopes,
    dropout_p,
    softmax_scale,
    causal,
    window_size_left,
    window_size_right,
    softcap,
    deterministic,
    gen=None,
    rng_state=None,
):
    if AMD_BACKEND == "triton":
        return _triton_fa2.bwd(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            alibi_slopes,
            dropout_p,
            softmax_scale,
            causal,
            window_size_left,
            window_size_right,
            softcap,
            deterministic,
            gen,
            rng_state,
        )
    else:
        _check_general_args(softcap, gen)
        from aiter.ops import mha
        import torch

        dq = torch.empty_like(q) if dq is None else dq
        dk = torch.empty_like(k) if dk is None else dk
        dv = torch.empty_like(v) if dv is None else dv
        softmax_d = mha._flash_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            None,
            dropout_p,
            softmax_scale,
            causal,
            window_size_left,
            window_size_right,
            None,
            alibi_slopes,
            deterministic,
            rng_state,
        )
        return dq, dk, dv, softmax_d


def varlen_fwd(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    cu_seqlens_k,
    seqused_k,
    leftpad_k,
    block_table,
    alibi_slopes,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p,
    softmax_scale,
    zero_tensors,
    causal,
    window_size_left,
    window_size_right,
    softcap,
    return_softmax,
    gen=None,
    num_splits=0,
):
    if AMD_BACKEND == "triton":
        return _triton_fa2.varlen_fwd(
            q,
            k,
            v,
            out,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_k,
            leftpad_k,
            block_table,
            alibi_slopes,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            softmax_scale,
            zero_tensors,
            causal,
            window_size_left,
            window_size_right,
            softcap,
            return_softmax,
            gen,
            num_splits,
        )
    else:
        _check_general_args(softcap, gen)
        if seqused_k is not None or leftpad_k is not None:
            raise NotImplementedError(
                "seqused_k and leftpad_k are not supported by the FA2 aiter adapter"
            )
        if num_splits != 0:
            raise NotImplementedError("num_splits is not supported by the FA2 aiter adapter")
        from aiter.ops import mha

        return mha._flash_attn_varlen_forward(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            None,
            None,
            max_seqlen_q,
            max_seqlen_k,
            0,
            dropout_p,
            softmax_scale,
            causal,
            0.0,
            window_size_left,
            window_size_right,
            0,
            None,
            alibi_slopes,
            None,
            None,
            None,
            True,
            return_softmax and dropout_p > 0,
            block_table=block_table,
            out=out,
            zero_tensors=zero_tensors,
        )


def varlen_bwd(
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
    alibi_slopes,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p,
    softmax_scale,
    zero_tensors,
    causal,
    window_size_left,
    window_size_right,
    softcap,
    deterministic,
    gen=None,
    rng_state=None,
):
    if AMD_BACKEND == "triton":
        return _triton_fa2.varlen_bwd(
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
            alibi_slopes,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            softmax_scale,
            zero_tensors,
            causal,
            window_size_left,
            window_size_right,
            softcap,
            deterministic,
            gen,
            rng_state,
        )
    else:
        _check_general_args(softcap, gen)
        from aiter.ops import mha
        import torch

        dq = torch.empty_like(q) if dq is None else dq
        dk = torch.empty_like(k) if dk is None else dk
        dv = torch.empty_like(v) if dv is None else dv
        softmax_d = mha._flash_attn_varlen_backward(
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
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            softmax_scale,
            causal,
            window_size_left,
            window_size_right,
            alibi_slopes,
            deterministic,
            rng_state,
            zero_tensors=zero_tensors,
        )
        return dq, dk, dv, softmax_d


def fwd_kvcache(
    q,
    k_cache,
    v_cache,
    k,
    v,
    cache_seqlens,
    rotary_cos,
    rotary_sin,
    cache_batch_idx,
    cache_leftpad,
    block_table,
    alibi_slopes,
    out,
    softmax_scale,
    causal,
    window_size_left,
    window_size_right,
    softcap,
    rotary_interleaved,
    num_splits,
):
    if AMD_BACKEND == "triton":
        return _triton_fa2.fwd_kvcache(
            q,
            k_cache,
            v_cache,
            k,
            v,
            cache_seqlens,
            rotary_cos,
            rotary_sin,
            cache_batch_idx,
            cache_leftpad,
            block_table,
            alibi_slopes,
            out,
            softmax_scale,
            causal,
            window_size_left,
            window_size_right,
            softcap,
            rotary_interleaved,
            num_splits,
        )
    else:
        if out is not None:
            raise NotImplementedError("out is not supported by the FA2 aiter KV-cache adapter")
        from aiter.ops.triton.attention.mha import flash_attn_with_kvcache

        return flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            k=k,
            v=v,
            cache_seqlens=cache_seqlens,
            rotary_cos=rotary_cos,
            rotary_sin=rotary_sin,
            cache_batch_idx=cache_batch_idx,
            cache_leftpad=cache_leftpad,
            block_table=block_table,
            alibi_slopes=alibi_slopes,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=(window_size_left, window_size_right),
            softcap=softcap,
            rotary_interleaved=rotary_interleaved,
            num_splits=num_splits,
            return_softmax_lse=True,
        )


flash_attn_2 = SimpleNamespace(
    fwd=fwd,
    bwd=bwd,
    varlen_fwd=varlen_fwd,
    varlen_bwd=varlen_bwd,
    fwd_kvcache=fwd_kvcache,
)
