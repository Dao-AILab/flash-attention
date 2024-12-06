import torch
from .fwd_prefill import attention_prefill_forward_triton_impl
from .bwd_prefill import attention_prefill_backward_triton_impl
from .fwd_decode import attention_decode_forward_triton_impl


class _attention_prefill(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, o, metadata):
        output, softmax_lse, sd_mask = attention_prefill_forward_triton_impl(
                                                q, 
                                                k, 
                                                v, 
                                                o, 
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
                                                metadata.use_exp2)

        ctx.save_for_backward(q, k, v, o, softmax_lse)
        ctx.sm_scale = metadata.sm_scale
        ctx.causal = metadata.causal
        ctx.alibi_slopes = metadata.alibi_slopes
        ctx.dropout_p = metadata.dropout_p
        ctx.philox_seed = metadata.philox_seed
        ctx.philox_offset = metadata.philox_offset
        ctx.return_scores = metadata.return_scores
        ctx.layout = metadata.layout
        ctx.use_exp2 = metadata.use_exp2
        return output, softmax_lse, sd_mask

    @staticmethod
    def backward(ctx, do, *args):
        q, k, v, o, softmax_lse = ctx.saved_tensors
        return attention_prefill_backward_triton_impl(
            do,
            q,
            k,
            v,
            o,
            softmax_lse,
            None,
            None,
            None,
            ctx.sm_scale,
            ctx.alibi_slopes,
            ctx.causal,
            ctx.layout,
            None,
            None,
            None,
            None,
            ctx.dropout_p, 
            ctx.philox_seed, 
            ctx.philox_offset,
            ctx.use_exp2
        )

attention_prefill = _attention_prefill.apply


class _attention_decode(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, metadata):
        output, softmax_lse = attention_decode_forward_triton_impl(
            q,
            k,
            v,
            metadata.sm_scale,
            metadata.causal,
            metadata.alibi_slopes,
            metadata.layout,
            metadata.cache_seqlens,
            metadata.cache_batch_idx,
            metadata.new_kv,
            metadata.k_new,
            metadata.v_new,
        )
        return output, softmax_lse

attention_decode = _attention_decode.apply
