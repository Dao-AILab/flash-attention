import torch
from flash_attn import flash_attn_sink_func
from flash_attn.flash_attn_interface import _flash_attn_sink_backward


class FlashAttentionWithSinkFused(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        sink: torch.Tensor,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        softcap=0.0,
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
    ):
        # Check device
        if q.device.type != 'cuda':
            raise RuntimeError(
                f"Flash Attention only supports CUDA devices, "
                f"current device: {q.device}"
            )
        
        ctx.save_for_backward(q, k, v, sink, alibi_slopes)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.deterministic = deterministic
        ctx.return_attn_probs = return_attn_probs
        ctx.sink_shape = sink.shape  # Save original sink shape

        out, lse, _ = flash_attn_sink_func(
            q,
            k,
            v,
            sink,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_attn_probs=True,
        )
        ctx.lse = lse
        ctx.output = out

        return out


    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, sink, alibi_slopes = ctx.saved_tensors
        lse = ctx.lse

        grad_q = torch.empty_like(q)
        grad_k = torch.empty_like(k)
        grad_v = torch.empty_like(v)
        grad_sink = torch.empty_like(sink)

        _flash_attn_sink_backward(
            grad_output,  # dout: main path gradient
            q,                # q
            k,                # k
            v,                # v
            sink,
            ctx.output,       # out: original output
            lse,              # softmax_lse
            grad_q,           # dq: main path
            grad_k,           # dk: main path
            grad_v,           # dv: main path
            grad_sink,        # ds: sink gradient
            ctx.dropout_p,    # dropout_p
            ctx.softmax_scale,  # softmax_scale
            ctx.causal,       # causal
            ctx.window_size[0],  # window_size_left
            ctx.window_size[1],  # window_size_right
            ctx.softcap,      # softcap
            alibi_slopes,     # alibi_slopes
            ctx.deterministic,  # deterministic
        )


        return (grad_q, grad_k, grad_v, grad_sink, None, None, None, None, 
                None, None, None, None)


def flash_attn_with_sink_fused_func(
    q,
    k,
    v,
    sink: torch.Tensor,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0,  # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Flash Attention requires CUDA devices. "
            "Current device does not support CUDA."
        )
    
    return FlashAttentionWithSinkFused.apply(
        q, k, v, sink, dropout_p, softmax_scale, causal,
        window_size, softcap, alibi_slopes, deterministic, return_attn_probs
    )
