import torch
import os
from .fwd_prefill import attention_prefill_forward_triton_impl
from .bwd_prefill import attention_prefill_backward_triton_impl
from .fwd_decode import attention_decode_forward_triton_impl
from einops import rearrange, repeat, parse_shape
from flash_attn.layers.rotary import apply_rotary_emb

ENABLE_FUSED_ROTARY = os.environ.get('FLASH_ATTENTION_TRITON_AMD_ENABLE_FUSED_ROTARY', '0').lower() in ('1', 'true', 'yes')

class _attention_prefill(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, o, metadata):
        (output, 
        softmax_lse, 
        exp_scores, 
        grid, 
        head_size, 
        philox_seed, 
        philox_offset, 
        _, 
        _) = attention_prefill_forward_triton_impl(
                                                q, 
                                                k, 
                                                v, 
                                                o, 
                                                metadata.sm_scale, 
                                                metadata.alibi_slopes, 
                                                metadata.causal, 
                                                metadata.bias, 
                                                metadata.dropout_p, 
                                                metadata.layout, 
                                                metadata.cu_seqlens_q, 
                                                metadata.cu_seqlens_k,
                                                metadata.max_seqlens_q, 
                                                metadata.max_seqlens_k, 
                                                metadata.return_scores, 
                                                metadata.use_exp2)

        ctx.save_for_backward(q, k, v, o, softmax_lse)
        ctx.grid = grid
        ctx.sm_scale = metadata.sm_scale
        ctx.head_size = head_size
        ctx.causal = metadata.causal
        ctx.alibi_slopes = metadata.alibi_slopes
        ctx.dropout_p = metadata.dropout_p
        ctx.philox_seed = philox_seed
        ctx.philox_offset = philox_offset
        ctx.exp_scores = exp_scores
        ctx.return_scores = metadata.return_scores
        ctx.layout = metadata.layout
        ctx.use_exp2 = metadata.use_exp2
        return output, softmax_lse, exp_scores

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
            ctx.use_exp2
        )

attention_prefill = _attention_prefill.apply


class _attention_decode(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, metadata):
        if not ENABLE_FUSED_ROTARY:
            q_original_shape = parse_shape(q, 'b s g h d')
            # Non-fused rotary kernel
            if metadata.rotary_dim > 0.0:
                if metadata.causal:     # NOTE: when local support is added. Add `or metadata.local`
                    q_ro = apply_rotary_emb(
                        q,
                        metadata.rotary_cos,
                        metadata.rotary_sin,
                        seqlen_offsets=metadata.cache_seqlens if metadata.cache_seqlens else 0,
                        interleaved=metadata.rotary_interleaved,
                    )
                else:
                    q_ro = rearrange(
                        apply_rotary_emb(
                            rearrange(q, "b s g h d -> b 1 (s g h) d"),
                            metadata.rotary_cos,
                            metadata.rotary_sin,
                            seqlen_offsets=metadata.cache_seqlens if metadata.cache_seqlens else 0,
                            interleaved=metadata.rotary_interleaved,
                        ),
                        "b 1 (s g h) d -> b s g h d",
                        s=q_original_shape['s'],
                        g=q_original_shape['g'],
                        h=q_original_shape['h']
                    )

                # NOTE: since we don't have new kv we don't need to rotate k

                # k_ro = apply_rotary_emb(
                #     metadata.k_new,
                #     metadata.rotary_cos,
                #     metadata.rotary_sin,
                #     seqlen_offsets=metadata.cache_seqlens,
                #     interleaved=metadata.rotary_interleaved,
                # )

                q, metadata.k_new = q_ro.to(q.dtype), None

                # nullify rotary parameters so that the fused rotary implementation is not executed within the triton decode fwd kernel
                metadata.need_rotary(0, None, None, False)

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
            metadata.rotary_cos,
            metadata.rotary_sin,
            metadata.rotary_dim,
            metadata.rotary_interleaved,
            metadata.rotary_conjugate
        )
        return output, softmax_lse

attention_decode = _attention_decode.apply
