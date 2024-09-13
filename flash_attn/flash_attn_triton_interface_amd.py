import torch
import triton
from .flash_attn_triton_kernel_prefill_amd import MetaData, get_shape_from_layout, attention_prefill
from .flash_attn_triton_kernel_decode_amd import attention_decode

def fwd(q,
        k,
        v,
        o,
        alibi_slopes,
        dropout_p,
        softmax_scale,
        causal,
        window_size_left,
        window_size_right,
        softcap,
        return_softmax,
        gen_):

    if dropout_p != 0.0:
        raise ValueError("dropout is not supported on AMD's Triton Backend yet")

    if o is None:
        o = torch.empty_like(q)

    # Setup metadata
    input_metadata = MetaData(sm_scale=softmax_scale)
    input_metadata.max_seqlens_q = q.shape[1]
    input_metadata.max_seqlens_k = k.shape[1]
    input_metadata.layout = "bshd"
    if return_softmax:
        input_metadata.return_encoded_softmax = True

    batch, nheads_q, nheads_k, head_size = get_shape_from_layout(q, k, input_metadata)
    
    if causal:
        input_metadata.need_causal()
    
    if alibi_slopes is not None:
        input_metadata.need_alibi(alibi_slopes, batch, nheads_q)
    
    if dropout_p > 0.0:
        input_metadata.need_dropout(dropout_p, return_softmax)
    
    # Check arguments
    input_metadata.check_args(q, k, v, o)
    tri_out, softmax_lse, softmax_dmask= attention_prefill(q, k, v, o, input_metadata)

    return tri_out, q , k , v, o, softmax_lse, softmax_dmask, None

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
    deterministic,
    gen_,
    rng_state,
):
    raise ValueError("bwd is not supported on AMD's Triton Backend yet")

def varlen_fwd(
        q, 
        k, 
        v, 
        o,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_k,
        leftpad_k,
        block_table_,
        alibi_slopes,\
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
        gen_):

    if dropout_p != 0.0:
        raise ValueError("dropout is not supported on AMD's Triton Backend yet")
    
    if o is None:
        o = torch.empty_like(q)

    # Setup metadata
    input_metadata = MetaData(sm_scale=softmax_scale)
    if return_softmax:
        input_metadata.return_encoded_softmax = True
    input_metadata.set_varlen_params(cu_seqlens_q, cu_seqlens_k)  # set layout to "thd" and other metdata

    # get shapes
    batch, nheads_q, nheads_k, head_size = get_shape_from_layout(q, k, input_metadata)

    if causal:
        input_metadata.need_causal()

    if alibi_slopes is not None:
        input_metadata.need_alibi(alibi_slopes, batch, nheads_q)
    
    if dropout_p > 0.0:
        input_metadata.need_dropout(dropout_p, return_softmax)
    
    # Check arguments
    input_metadata.check_args(q, k, v, o)

    tri_out, softmax_lse, softmax_dmask= attention_prefill(q, k, v, o, input_metadata)

    return tri_out, q , k , v, o, softmax_lse, softmax_dmask, None

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
    gen_,
    rng_state,
):
    raise ValueError("varlen_bwd is not supported on AMD's Triton Backend yet")

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
        num_splits):

    if out is None:
        out = torch.empty_like(q)

    # fill metadata
    input_metadata = MetaData(sm_scale=softmax_scale)
    input_metadata.layout = "bshd"
    input_metadata.max_seqlens_q = q.shape[1]
    input_metadata.max_seqlens_k = k_cache.shape[1]
    input_metadata.cache_seqlens = cache_seqlens
    input_metadata.cache_batch_idx = cache_batch_idx

    if k is not None and v is not None:
        input_metadata.new_kv = True
        input_metadata.seqlen_new = k.shape[1]
        input_metadata.k_new = k
        input_metadata.v_new = v

    if causal:
        input_metadata.need_causal()

    if alibi_slopes is not None:
        batch, _ , nheads_q, _= q.shape
        input_metadata.need_alibi(alibi_slopes, batch, nheads_q)

    # launch kernel
    tri_out, softmax_lse = attention_decode(q, k_cache, v_cache, input_metadata)
    return tri_out, softmax_lse
