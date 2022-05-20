# Adapted from https://github.com/mlcommons/training_results_v1.1/blob/main/NVIDIA/benchmarks/bert/implementations/pytorch/fmha.py
import torch
import torch.nn as nn

import stream_attn_cuda


def _stream_attn_forward(qkv, cu_seqlens, dropout_p, max_s, softmax_scale, causal, return_softmax):
    context, softmax_lse, *rest = stream_attn_cuda.fwd(qkv, cu_seqlens, dropout_p, max_s, softmax_scale,
                                                       False, causal, return_softmax, None)
    # if context.isnan().any() or softmax_lse.isnan().any():
    #     breakpoint()
    S_dmask = rest[0] if return_softmax else None
    return context, softmax_lse, S_dmask


def _stream_attn_backward(dout, qkv, out, S_dmask, softmax_lse, cu_seqlens, dropout_p, max_s,
                   softmax_scale, causal):
    dqkv, dp, softmax_d = stream_attn_cuda.bwd(dout, qkv, out, S_dmask, softmax_lse, cu_seqlens, dropout_p,
                                               softmax_scale, max_s, False, causal, None)
    # if dqkv.isnan().any() or softmax_d.isnan().any():
    #     breakpoint()
    return dqkv


class StreamAttnFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, qkv, cu_seqlens, dropout_p, max_s, softmax_scale, causal):
        # Save rng_state because the backward pass will regenerate the dropout mask
        rng_state = torch.cuda.get_rng_state() if dropout_p > 0 else None
        if softmax_scale is None:
            softmax_scale = qkv.shape[-1] ** (-0.5)
        context, softmax_lse, S_dmask = _stream_attn_forward(
            qkv, cu_seqlens, dropout_p, max_s, softmax_scale, causal=causal, return_softmax=False
        )
        ctx.save_for_backward(qkv, context, S_dmask, softmax_lse, cu_seqlens, rng_state)
        ctx.dropout_p = dropout_p
        ctx.max_s = max_s
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        return context

    @staticmethod
    def backward(ctx, dout):
        qkv, context, S_dmask, softmax_lse, cu_seqlens, rng_state = ctx.saved_tensors
        if rng_state is not None:
            cur_rng_state = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(rng_state)
        # S_dmask is None, temporarily use another tensor just to get it running
        dqkv = _stream_attn_backward(
            dout, qkv, context, context, softmax_lse, cu_seqlens, ctx.dropout_p,
            ctx.max_s, ctx.softmax_scale, ctx.causal
        )
        if rng_state is not None:
            torch.cuda.set_rng_state(cur_rng_state)
        return dqkv, None, None, None, None, None, None


# We duplicate code to return both the output and the softmax for testing
# Returning both makes backward a bit slower, so we want to keep using the other version for speed.
class StreamAttnFunWithS(torch.autograd.Function):

    @staticmethod
    def forward(ctx, qkv, cu_seqlens, dropout_p, max_s, softmax_scale, causal):
        # Save rng_state because the backward pass is gonna regenerate the dropout mask
        rng_state = torch.cuda.get_rng_state() if dropout_p > 0 else None
        if softmax_scale is None:
            softmax_scale = qkv.shape[-1] ** (-0.5)
        context, softmax_lse, S_dmask = _stream_attn_forward(
            qkv, cu_seqlens, dropout_p, max_s, softmax_scale, causal=causal, return_softmax=True
        )
        ctx.save_for_backward(qkv, context, S_dmask, softmax_lse, cu_seqlens, rng_state)
        ctx.dropout_p = dropout_p
        ctx.max_s = max_s
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        return context, S_dmask, softmax_lse

    @staticmethod
    def backward(ctx, dout, _dS_dmask_ignored, _dsoftmax_sum_ignored):
        qkv, context, S_dmask, softmax_lse, cu_seqlens, rng_state = ctx.saved_tensors
        if rng_state is not None:
            cur_rng_state = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(rng_state)
        dqkv = _stream_attn_backward(
            dout, qkv, context, S_dmask, softmax_lse, cu_seqlens, ctx.dropout_p,
            ctx.max_s, ctx.softmax_scale, ctx.causal
        )
        if rng_state is not None:
            torch.cuda.set_rng_state(cur_rng_state)
        return dqkv, None, None, None, None, None


def stream_attn_func(qkv, cu_seqlens, dropout_p, max_s, softmax_scale=None, causal=False,
                     return_attn_probs=False):
    """dropout_p should be set to 0.0 during evaluation
    """
    func = StreamAttnFun if not return_attn_probs else StreamAttnFunWithS
    return func.apply(qkv, cu_seqlens, dropout_p, max_s, softmax_scale, causal)
