import torch
import os
from .fwd_prefill import attention_prefill_forward_triton_impl
from .bwd_prefill import attention_prefill_backward_triton_impl
from .bwd_prefill_split import attention_prefill_backward_triton_split_impl
from .fwd_decode import attention_decode_forward_triton_impl
from .fwd_ref import attention_forward_pytorch_ref_impl
from .bwd_ref import attention_backward_pytorch_ref_impl
from .utils import DEBUG_TRITON, DEBUG_TRITON_DETAIL, MetaData, get_shapes_from_layout, DEBUG, is_fp8
from einops import rearrange, repeat
from flash_attn.layers.rotary import apply_rotary_emb
from typing import Optional, Union

USE_REF = os.environ.get('FLASH_ATTENTION_TRITON_AMD_REF', '0').lower() in ('1', 'true', 'yes')

def fwd(q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: Optional[torch.Tensor],
        alibi_slopes: Optional[torch.Tensor],
        dropout_p: float,
        softmax_scale: float,
        causal: bool,
        window_size_left: int,
        window_size_right: int,
        softcap: float,
        return_softmax: bool,
        gen_: Optional[torch.Tensor] = None,
        descale_q: Optional[torch.Tensor] = None,
        descale_k: Optional[torch.Tensor] = None,
        descale_v: Optional[torch.Tensor] = None
    ):

    if DEBUG:
        print()
        print("flash_attn_triton_amd.py::fwd inputs")
        print("q:", q, q.shape)
        print("k:", k, k.shape)
        print("v:", v, v.shape)
        print("out:", out, out.shape if out else None)
        print("alibi_slopes:", alibi_slopes)
        print("dropout_p:", dropout_p)
        print("softmax_scale:", softmax_scale)
        print("causal:", causal)
        print("window_size_left:", window_size_left)
        print("window_size_right:", window_size_right)
        print("softcap:", softcap)
        print("return_softmax:", return_softmax)
        print("descale_q:", descale_q, descale_q.shape if descale_q is not None else None)
        print("descale_k:", descale_k, descale_k.shape if descale_k is not None else None)
        print("descale_v:", descale_v, descale_v.shape if descale_v is not None else None)

    if is_fp8(q):
        if out is None:
            out = torch.zeros_like(q, dtype=torch.float32) 
        else:
            assert out.dtype == torch.float32, "fp8 output tensor should be fp32 type" # the high level api doesnot provide
    else:
        out = torch.zeros_like(q) if out is None else out.zero_()

    # Setup metadata
    metadata = MetaData(sm_scale=softmax_scale)
    metadata.max_seqlens_q = q.shape[1]
    metadata.max_seqlens_k = k.shape[1]
    metadata.layout = "bshd"
    if return_softmax:
        metadata.return_scores = True

    batch, nheads_q, nheads_k, head_size, _, _ = get_shapes_from_layout(q, k, metadata.layout)

    if causal:
        metadata.need_causal()

    if alibi_slopes is not None:
        metadata.need_alibi(alibi_slopes, batch, nheads_q)

    if dropout_p > 0.0:
        metadata.need_dropout(dropout_p)
        rng_state = torch.as_tensor([metadata.philox_seed, metadata.philox_offset]) # as_tensors uses the underlying data and doesnot cast
    else:
        rng_state = None

    # check arguments
    metadata.check_args(q, k, v, out)

    # call implementation
    if USE_REF:
        if DEBUG:
            print("Using reference implementation")
        softmax_lse_ref, sd_mask_ref = attention_forward_pytorch_ref_impl(
                                                q,
                                                k,
                                                v,
                                                metadata.sm_scale,
                                                metadata.causal,
                                                metadata.layout,
                                                metadata.cu_seqlens_q,
                                                metadata.cu_seqlens_k,
                                                metadata.max_seqlens_q,
                                                metadata.max_seqlens_k,
                                                metadata.dropout_p,
                                                metadata.philox_seed,
                                                metadata.philox_offset,
                                                metadata.use_exp2)
        softmax_lse=softmax_lse_ref
        sd_mask=sd_mask_ref
    else:
        if DEBUG:
            print("Using Triton implementation")
        softmax_lse_triton, sd_mask_triton = attention_prefill_forward_triton_impl(
                                                q,
                                                k,
                                                v,
                                                out,
                                                metadata.sm_scale,
                                                metadata.alibi_slopes,
                                                metadata.causal,
                                                metadata.bias,
                                                metadata.layout,
                                                metadata.cu_seqlens_q,
                                                metadata.cu_seqlens_k,
                                                metadata.max_seqlens_q,
                                                metadata.max_seqlens_k,
                                                metadata.dropout_p,
                                                metadata.philox_seed,
                                                metadata.philox_offset,
                                                metadata.return_scores,
                                                metadata.use_exp2,
                                                descale_q,
                                                descale_k,
                                                descale_v)
        softmax_lse=softmax_lse_triton
        sd_mask=sd_mask_triton

    if DEBUG:
        print("flash_attn_triton_amd.py::fwd outputs")
        print("o:", out, out.shape)
        print("softmax_lse:", softmax_lse, softmax_lse.shape)
        print("exp_scores:", sd_mask, sd_mask.shape if sd_mask is not None else None )

    return out, softmax_lse, sd_mask, rng_state

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
    alibi_slopes: Optional[torch.Tensor],
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    deterministic: bool,
    gen_: Optional[torch.Tensor] = None,
    rng_state:Optional[torch.Tensor] = None,
    descale_q: Optional[torch.Tensor] = None,
    descale_k: Optional[torch.Tensor] = None,
    descale_v: Optional[torch.Tensor] = None,
    descale_do: Optional[torch.Tensor] = None,
):
    if DEBUG:
        print()
        print("flash_attn_triton_amd.py::bwd inputs")
        print("dout:", dout, dout.shape)
        print("q:", q, q.shape)
        print("k:", k, k.shape)
        print("v:", v, v.shape)
        print("out:", out, out.shape)
        print("softmax_lse:", softmax_lse, softmax_lse.shape)
        print("dq:", dq, dq.shape if dq is not None else None)
        print("dk:", dk, dk.shape if dk is not None else None)
        print("dv:", dv, dv.shape if dv is not None else None)
        print("alibi_slopes:", alibi_slopes)
        print("dropout_p:", dropout_p)
        print("out:", out)
        print("softmax_scale:", softmax_scale)
        print("causal:", causal)
        print("window_size_left:", window_size_left)
        print("window_size_right:", window_size_right)
        print("deterministic:", deterministic)
        print("gen_:", gen_)
        print("rng_state:", rng_state)
        print("descale_q:", descale_q, descale_q.shape if descale_q is not None else None)
        print("descale_k:", descale_k, descale_k.shape if descale_k is not None else None)
        print("descale_v:", descale_v, descale_v.shape if descale_v is not None else None)
        print("descale_do:", descale_do, descale_do.shape if descale_do is not None else None)

    dq = torch.zeros_like(q) if dq is None else dq.zero_()
    dk = torch.zeros_like(k) if dk is None else dk.zero_()
    dv = torch.zeros_like(v) if dv is None else dv.zero_()

    if dropout_p > 0.0:
        assert rng_state is not None
        philox_seed, philox_offset = rng_state[0].item(), rng_state[1].item()
    else:
        philox_seed, philox_offset = None, None

    # call implementation
    if USE_REF:
        if DEBUG:
            print("Using reference implementation")

        delta_ref = attention_backward_pytorch_ref_impl(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            softmax_scale,
            causal,
            "bshd",
            None,
            None,
            None,
            None,
            dropout_p,
            philox_seed,
            philox_offset,
            False,
        )
        delta = delta_ref
    else:
        if DEBUG:
            print("Using Triton implementation")
        delta_triton = attention_prefill_backward_triton_split_impl(
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
            alibi_slopes,
            causal,
            "bshd",
            None,
            None,
            None,
            None,
            dropout_p,
            philox_seed,
            philox_offset,
            False,
            descale_q = descale_q,
            descale_k = descale_k,
            descale_v = descale_v,
            descale_do = descale_do,
            DEBUG_TRITON=DEBUG_TRITON,
            DEBUG_TRITON_DETAIL=DEBUG_TRITON_DETAIL,
        )
        delta = delta_triton

    if DEBUG:
        print("flash_attn_triton_amd.py::bwd outputs")
        print("dv:", dv, dv.shape)
        print("dk:", dk, dk.shape)
        print("dq:", dq, dq.shape)
    return dq, dk, dv, delta

def varlen_fwd(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: Optional[torch.Tensor],
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        seqused_k: Optional[torch.Tensor],
        leftpad_k: Optional[torch.Tensor],
        block_table_: Optional[torch.Tensor],
        alibi_slopes: Optional[torch.Tensor],
        max_seqlen_q: int,
        max_seqlen_k: int,
        dropout_p: float,
        softmax_scale: float,
        zero_tensors: bool ,
        causal: bool ,
        window_size_left: int,
        window_size_right: int,
        softcap: float,
        return_softmax: bool,
        gen_: Optional[torch.Tensor] = None,
        descale_q: Optional[torch.Tensor] = None,
        descale_k: Optional[torch.Tensor] = None,
        descale_v: Optional[torch.Tensor] = None
    ):

    if DEBUG:
        print()
        print("flash_attn_triton_amd.py::varlen_fwd")
        print("q:", q, q.shape)
        print("k:", k, k.shape)
        print("v:", v, v.shape)
        print("cu_seqlens_q:", cu_seqlens_q, cu_seqlens_q.shape)
        print("cu_seqlens_k:", cu_seqlens_k, cu_seqlens_k.shape)
        print("alibi_slopes:", alibi_slopes)
        print("max_seqlen_q:", max_seqlen_q)
        print("max_seqlen_k:", max_seqlen_k)
        print("dropout_p:", dropout_p)
        print("softmax_scale:", softmax_scale)
        print("causal:", causal)
        print("window_size_left:", window_size_left)
        print("window_size_right:", window_size_right)
        print("gen_:", gen_)
        print("descale_q:", descale_q, descale_q.shape if descale_q is not None else None)
        print("descale_k:", descale_k, descale_k.shape if descale_k is not None else None)
        print("descale_v:", descale_v, descale_v.shape if descale_v is not None else None)

    if is_fp8(q):
        if out is None:
            out = torch.zeros_like(q, dtype=torch.float32) 
        else:
            assert out.dtype == torch.float32, "fp8 output tensor should be fp32 type" # the high level api doesnot provide
    else:
        out = torch.zeros_like(q) if out is None else out.zero_()

    # Setup metadata
    metadata = MetaData(sm_scale=softmax_scale)
    if return_softmax:
        metadata.return_scores = True
    metadata.set_varlen_params(cu_seqlens_q, cu_seqlens_k)  # set layout to "thd" and other metdata
    assert metadata.layout is not None

    # get shapes
    batch, nheads_q, nheads_k, head_size , seqlen_q, seqlen_k = get_shapes_from_layout(q, k, metadata.layout, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)

    if causal:
        metadata.need_causal()

    if alibi_slopes is not None:
        metadata.need_alibi(alibi_slopes, batch, nheads_q)

    if dropout_p > 0.0:
        metadata.need_dropout(dropout_p)
        rng_state = torch.as_tensor([metadata.philox_seed, metadata.philox_offset]) # as_tensors uses the underlying data and doesnot cast
    else:
        rng_state = None

    # Check arguments
    metadata.check_args(q, k, v, out)

    # call implementation
    if USE_REF:
        if DEBUG:
            print("Using reference implementation")
        softmax_lse_ref, sd_mask_ref = attention_forward_pytorch_ref_impl(
                                                q,
                                                k,
                                                v,
                                                metadata.sm_scale,
                                                metadata.causal,
                                                metadata.layout,
                                                metadata.cu_seqlens_q,
                                                metadata.cu_seqlens_k,
                                                metadata.max_seqlens_q,
                                                metadata.max_seqlens_k,
                                                metadata.dropout_p,
                                                metadata.philox_seed,
                                                metadata.philox_offset,
                                                metadata.use_exp2)
        softmax_lse=softmax_lse_ref
        sd_mask=sd_mask_ref
    else:
        if DEBUG:
            print("Using Triton implementation")
        softmax_lse_triton, sd_mask_triton = attention_prefill_forward_triton_impl(
                                                            q,
                                                            k,
                                                            v,
                                                            out,
                                                            metadata.sm_scale,
                                                            metadata.alibi_slopes,
                                                            metadata.causal,
                                                            metadata.bias,
                                                            metadata.layout,
                                                            metadata.cu_seqlens_q,
                                                            metadata.cu_seqlens_k,
                                                            metadata.max_seqlens_q,
                                                            metadata.max_seqlens_k,
                                                            metadata.dropout_p,
                                                            metadata.philox_seed,
                                                            metadata.philox_offset,
                                                            metadata.return_scores,
                                                            metadata.use_exp2,
                                                            descale_q,
                                                            descale_k,
                                                            descale_v)
        softmax_lse=softmax_lse_triton
        sd_mask=sd_mask_triton

    if DEBUG:
        print("varlen_fwd outputs")
        print("out:", out, out.shape)
        print("softmax_lse:", softmax_lse, softmax_lse.shape)
        print("sd_mask:", sd_mask, sd_mask.shape if sd_mask is not None else None )


    return out, softmax_lse, sd_mask, rng_state

def varlen_bwd(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: Optional[torch.Tensor],
    dk: Optional[torch.Tensor],
    dv: Optional[torch.Tensor],
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    alibi_slopes: Optional[torch.Tensor],
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    softmax_scale: float,
    zero_tensors: bool,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    deterministic: bool,
    gen_ : Optional[torch.Tensor] = None,
    rng_state: Optional[torch.Tensor] = None,
    descale_q: Optional[torch.Tensor] = None,
    descale_k: Optional[torch.Tensor] = None,
    descale_v: Optional[torch.Tensor] = None,
    descale_do: Optional[torch.Tensor] = None,
):
    if DEBUG:
        print()
        print("varlen_bwd")
        print("dout:", dout, dout.shape)
        print("q:", q, q.shape)
        print("k:", k, k.shape)
        print("v:", v, v.shape)
        print("out:", out)
        print("softmax_lse:", softmax_lse, softmax_lse.shape)
        print("dq:", dq, dq.shape if dq is not None else None)
        print("dk:", dk, dk.shape if dk is not None else None)
        print("dv:", dv, dv.shape if dv is not None else None)
        print("cu_seqlens_q:", cu_seqlens_q, cu_seqlens_q.shape)
        print("cu_seqlens_k:", cu_seqlens_k, cu_seqlens_k.shape)
        print("alibi_slopes:", alibi_slopes)
        print("max_seqlen_q:", max_seqlen_q)
        print("max_seqlen_k:", max_seqlen_k)
        print("dropout_p:", dropout_p)
        print("softmax_scale:", softmax_scale)
        print("causal:", causal)
        print("window_size_left:", window_size_left)
        print("window_size_right:", window_size_right)
        print("deterministic:", deterministic)
        print("gen_:", gen_)
        print("rng_state:", rng_state)
        print("descale_q:", descale_q, descale_q.shape if descale_q is not None  else None)
        print("descale_k:", descale_k, descale_k.shape if descale_k is not None  else None)
        print("descale_v:", descale_v, descale_v.shape if descale_v is not None  else None)
        print("descale_do:", descale_do, descale_do.shape if descale_do else None)

    dq = torch.zeros_like(q) if dq is None else dq.zero_()
    dk = torch.zeros_like(k) if dk is None else dk.zero_()
    dv = torch.zeros_like(v) if dv is None else dv.zero_()

    if dropout_p > 0.0:
        assert rng_state is not None
        philox_seed, philox_offset = rng_state[0].item(), rng_state[1].item()
    else:
        philox_seed, philox_offset = None, None

    # call implementation
    if USE_REF:
        if DEBUG:
            print("Using reference implementation")
        delta_ref = attention_backward_pytorch_ref_impl(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            softmax_scale,
            causal,
            "thd",
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            philox_seed,
            philox_offset,
            False,
        )
        delta = delta_ref
    else:
        if DEBUG:
            print("Using Triton implementation") 
        delta_triton = attention_prefill_backward_triton_split_impl(
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
            alibi_slopes,
            causal,
            "thd",
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            philox_seed,
            philox_offset,
            False,
            descale_q = descale_q,
            descale_k = descale_k,
            descale_v = descale_v,
            descale_do = descale_do,
            DEBUG_TRITON=DEBUG_TRITON,
            DEBUG_TRITON_DETAIL=DEBUG_TRITON_DETAIL,
        )
        delta = delta_triton

    if DEBUG:
        print("varlen_bwd outputs")
        print("delta:", delta, delta.shape)
        print("dv:", dv, dv.shape)
        print("dk:", dk, dk.shape)
        print("dq:", dq, dq.shape)

    return dq, dk, dv, delta

def fwd_kvcache(
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        cache_seqlens: Optional[Union[(int, torch.Tensor)]],
        rotary_cos: Optional[torch.Tensor],
        rotary_sin: Optional[torch.Tensor],
        cache_batch_idx: Optional[torch.Tensor],
        cache_leftpad: Optional[torch.Tensor],
        block_table: Optional[torch.Tensor],
        alibi_slopes: Optional[torch.Tensor],
        out: Optional[torch.Tensor],
        softmax_scale: float,
        causal: bool,
        window_size_left: int,
        window_size_right: int,
        softcap: float,
        rotary_interleaved: bool,
        num_splits: int
    ):

    # fill metadata
    metadata = MetaData(sm_scale=softmax_scale)
    metadata.layout = "bshd"
    metadata.max_seqlens_q = q.shape[1]
    metadata.max_seqlens_k = k_cache.shape[1]
    metadata.cache_seqlens = cache_seqlens
    metadata.cache_batch_idx = cache_batch_idx

    k_new = k
    v_new = v

    if causal:
        metadata.need_causal()

    if alibi_slopes is not None:
        batch, _ , nheads_q, _= q.shape
        metadata.need_alibi(alibi_slopes, batch, nheads_q)

    # rotary boolean
    apply_rotary = torch.is_tensor(rotary_cos) and torch.is_tensor(rotary_sin)
    if apply_rotary:
        metadata.need_rotary(rotary_sin, rotary_cos, rotary_interleaved)

    # Rotary Embedding Implementation
    if apply_rotary:
        if metadata.causal:     # NOTE: when support is added. Add `or metadata.local`
            q_ro = apply_rotary_emb(
                q,
                metadata.rotary_cos,
                metadata.rotary_sin,
                seqlen_offsets=metadata.cache_seqlens,
                interleaved=metadata.rotary_interleaved,
            )
        else:
            q_ro = rearrange(
                apply_rotary_emb(
                    rearrange(q, "b s h d -> b 1 (s h) d"),
                    metadata.rotary_cos,
                    metadata.rotary_sin,
                    seqlen_offsets=metadata.cache_seqlens,
                    interleaved=metadata.rotary_interleaved,
                ),
                "b 1 (s h) d -> b s h d",
                s=metadata.max_seqlens_q,
            )
        k_ro = apply_rotary_emb(
            k_new,
            metadata.rotary_cos,
            metadata.rotary_sin,
            seqlen_offsets=metadata.cache_seqlens,
            interleaved=metadata.rotary_interleaved,
        )

        q, k_new = q_ro.to(q.dtype), k_ro.to(q.dtype)

    # launch kernel
    # TODO: pass output as an arg. Maybe we are copying output which is causing slow down
    output_triton, softmax_lse_triton = attention_decode_forward_triton_impl(
        q,
        k_cache,
        v_cache,
        k_new,
        v_new,
        metadata.sm_scale,
        metadata.causal,
        metadata.alibi_slopes,
        metadata.layout,
        metadata.cache_seqlens,
        metadata.cache_batch_idx,
    )
    out = output_triton
    softmax_lse = softmax_lse_triton


    return out, softmax_lse
