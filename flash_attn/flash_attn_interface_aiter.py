"""FA2 low-level ABI backed by aiter's Triton MHA (implementation="dao_ai") or its general MHA."""

import math
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from flash_attn.flash_attn_interface import AMD_BACKEND


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
    if softcap != 0.0:
        raise NotImplementedError("softcap is not supported by the FA2 aiter adapter")
    if gen is not None:
        raise NotImplementedError(
            "an explicit dropout generator is not supported by the FA2 aiter adapter"
        )
    if AMD_BACKEND.impl == "triton":
        from aiter.ops.triton.attention import mha as triton_mha

        out, softmax_lse, s_dmask, _, _, rng_state = triton_mha._flash_attn_forward(
            q,
            k,
            v,
            dropout_p,
            softmax_scale,
            causal,
            window_size_left,
            window_size_right,
            None,
            alibi_slopes,
            True,
            return_softmax,
            q.shape[1],
            k.shape[1],
            implementation="dao_ai",
        )
    else:
        if return_softmax and dropout_p == 0:
            raise NotImplementedError(
                "return_softmax without dropout is not supported by the aiter general implementation"
            )
        from aiter.ops import mha as general_mha

        out, softmax_lse, s_dmask, rng_state = general_mha._flash_attn_forward(
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
        if return_softmax and dropout_p > 0:
            # aiter returns CK's random bytes. In FA's record, a value >= 0 is kept and < 0 is dropped.
            s_dmask = (math.floor(255.0 * (1 - dropout_p)) - s_dmask.to(torch.float32)).to(q.dtype)
        else:
            s_dmask = torch.empty((0,), dtype=q.dtype, device=q.device)
    # FA's register_fake for _flash_attn_forward describes these returns.
    batch, seqlen_q, nheads, _ = q.shape
    s_dmask_shape = (batch, nheads, seqlen_q, k.shape[1]) if return_softmax else (0,)
    assert s_dmask.shape == s_dmask_shape and s_dmask.dtype == q.dtype, s_dmask.shape
    assert out.shape == q.shape[:-1] + v.shape[-1:] and out.dtype == q.dtype, out.shape
    assert softmax_lse.shape == (batch, nheads, seqlen_q), softmax_lse.shape
    assert softmax_lse.dtype == torch.float32, softmax_lse.dtype
    assert rng_state.shape == (2,) and rng_state.dtype == torch.int64, rng_state
    return out, softmax_lse, s_dmask, rng_state


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
    if softcap != 0.0:
        raise NotImplementedError("softcap is not supported by the FA2 aiter adapter")
    if gen is not None:
        raise NotImplementedError(
            "an explicit dropout generator is not supported by the FA2 aiter adapter"
        )
    if AMD_BACKEND.impl == "triton":
        from aiter.ops.triton.attention import mha as triton_mha

        softmax_d = triton_mha._flash_attn_backward(
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
            alibi_slopes,
            deterministic,
            q.shape[1],
            k.shape[1],
            rng_state=rng_state,
            implementation="dao_ai",
        )
    else:
        from aiter.ops import mha as general_mha

        dq = torch.empty_like(q) if dq is None else dq
        dk = torch.empty_like(k) if dk is None else dk
        dv = torch.empty_like(v) if dv is None else dv
        softmax_d = general_mha._flash_attn_backward(
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
    # FA's register_fake for _flash_attn_backward describes softmax_d.
    assert softmax_d.shape == q.shape[:1] + (q.shape[2], q.shape[1]), softmax_d.shape
    assert softmax_d.dtype == torch.float32, softmax_d.dtype
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
    if softcap != 0.0:
        raise NotImplementedError("softcap is not supported by the FA2 aiter adapter")
    if gen is not None:
        raise NotImplementedError(
            "an explicit dropout generator is not supported by the FA2 aiter adapter"
        )
    if seqused_k is not None or leftpad_k is not None:
        raise NotImplementedError(
            "seqused_k and leftpad_k are not supported by the FA2 aiter adapter"
        )
    if num_splits not in (0, 1):
        raise NotImplementedError("num_splits is not supported by the FA2 aiter adapter")
    if AMD_BACKEND.impl == "triton":
        if block_table is not None:
            raise NotImplementedError(
                "block_table is not supported by the aiter triton implementation"
            )
        from aiter.ops.triton.attention import mha as triton_mha

        out, softmax_lse, s_dmask, _, _, rng_state = triton_mha._flash_attn_forward(
            q,
            k,
            v,
            dropout_p,
            softmax_scale,
            causal,
            window_size_left,
            window_size_right,
            None,
            alibi_slopes,
            True,
            return_softmax,
            max_seqlen_q,
            max_seqlen_k,
            cu_seqlens_q,
            cu_seqlens_k,
            implementation="dao_ai",
        )
    else:
        if return_softmax and dropout_p == 0:
            raise NotImplementedError(
                "return_softmax without dropout is not supported by the aiter general implementation"
            )
        from aiter.ops import mha as general_mha

        out, softmax_lse, s_dmask, rng_state = general_mha._flash_attn_varlen_forward(
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
        if return_softmax and dropout_p > 0:
            # aiter returns CK's random bytes, shaped (heads, total_q, max_seqlen_k). In FA's record,
            # a value >= 0 is kept and < 0 is dropped, shaped (batch, heads, seqlen_q, seqlen_k).
            pattern = math.floor(255.0 * (1 - dropout_p)) - s_dmask.to(torch.float32)
            lengths = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).tolist()
            s_dmask = torch.stack([
                F.pad(rows, (0, max_seqlen_k - rows.shape[2], 0, max_seqlen_q - rows.shape[1]))
                for rows in pattern.split(lengths, dim=1)
            ]).to(q.dtype)
        else:
            s_dmask = torch.empty((0,), dtype=q.dtype, device=q.device)
    # FA's register_fake for _flash_attn_varlen_forward describes these returns.
    batch = cu_seqlens_q.numel() - 1
    total_q, nheads, _ = q.shape
    s_dmask_shape = (batch, nheads, max_seqlen_q, max_seqlen_k) if return_softmax else (0,)
    assert s_dmask.shape == s_dmask_shape and s_dmask.dtype == q.dtype, s_dmask.shape
    assert out.shape == q.shape[:-1] + v.shape[-1:] and out.dtype == q.dtype, out.shape
    assert softmax_lse.shape == (nheads, total_q), softmax_lse.shape
    assert softmax_lse.dtype == torch.float32, softmax_lse.dtype
    assert rng_state.shape == (2,) and rng_state.dtype == torch.int64, rng_state
    return out, softmax_lse, s_dmask, rng_state


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
    if softcap != 0.0:
        raise NotImplementedError("softcap is not supported by the FA2 aiter adapter")
    if gen is not None:
        raise NotImplementedError(
            "an explicit dropout generator is not supported by the FA2 aiter adapter"
        )
    if AMD_BACKEND.impl == "triton":
        from aiter.ops.triton.attention import mha as triton_mha

        softmax_d = triton_mha._flash_attn_backward(
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
            alibi_slopes,
            deterministic,
            max_seqlen_q,
            max_seqlen_k,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            rng_state=rng_state,
            implementation="dao_ai",
        )
    else:
        from aiter.ops import mha as general_mha

        dq = torch.empty_like(q) if dq is None else dq
        dk = torch.empty_like(k) if dk is None else dk
        dv = torch.empty_like(v) if dv is None else dv
        softmax_d = general_mha._flash_attn_varlen_backward(
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
        # CK allocates (batch, heads, total_q) and writes every token to batch row 0.
        if softmax_d.dim() == 3:
            softmax_d = softmax_d[0]
    # FA's register_fake for _flash_attn_varlen_backward describes softmax_d.
    assert softmax_d.shape == (q.shape[1], q.shape[0]), softmax_d.shape
    assert softmax_d.dtype == torch.float32, softmax_d.dtype
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
    if AMD_BACKEND.impl == "triton":
        if out is not None:
            raise NotImplementedError("out is not supported by the FA2 aiter adapter")
        from aiter.ops.triton.attention import mha as triton_mha

        out, softmax_lse = triton_mha.flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            k,
            v,
            cache_seqlens,
            softmax_scale,
            causal,
            (window_size_left, window_size_right),
            softcap,
            num_splits,
            rotary_cos,
            rotary_sin,
            cache_batch_idx,
            cache_leftpad,
            block_table,
            alibi_slopes,
            rotary_interleaved,
            return_softmax_lse=True,
        )
        # What flash_attn_with_kvcache returns on the other backends.
        assert out.shape == q.shape[:-1] + v_cache.shape[-1:] and out.dtype == q.dtype, out.shape
        assert softmax_lse.shape == (q.shape[0], q.shape[2], q.shape[1]), softmax_lse.shape
        assert softmax_lse.dtype == torch.float32, softmax_lse.dtype
        return out, softmax_lse
    raise NotImplementedError(
        "KV-cache attention is not supported by the aiter general implementation. "
        "Use FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE."
    )

flash_attn_2 = SimpleNamespace(
    fwd=fwd,
    bwd=bwd,
    varlen_fwd=varlen_fwd,
    varlen_bwd=varlen_bwd,
    fwd_kvcache=fwd_kvcache,
)
