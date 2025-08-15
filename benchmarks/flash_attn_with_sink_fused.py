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

        # import pdb; pdb.set_trace()

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
        print("==== lse shape: ", lse.shape)

        origin_dtype = out.dtype

        ctx.raw_output = out.clone()
        ctx.lse = lse.clone()

        lse = lse.transpose(-2, -1).unsqueeze(dim=-1)
        # sink = sink.reshape(1, 1, -1, 1)

        # multiplier = 1 / (torch.exp(sink - lse) + 1)
        # out = (out * multiplier).to(origin_dtype)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, sink, alibi_slopes = ctx.saved_tensors
        raw_output = ctx.raw_output
        lse = ctx.lse

        lse = lse.transpose(-2, -1).unsqueeze(dim=-1)
        sink_reshaped = sink.reshape(1, 1, -1, 1)
        multiplier = 1 / (torch.exp(sink_reshaped - lse) + 1)

        # 1) Main path via multiplier
        grad_raw_output = (grad_output * multiplier).to(q.dtype)

        # Use flash attention backward function for main path
        grad_q_main = torch.empty_like(q)
        grad_k_main = torch.empty_like(k)
        grad_v = torch.empty_like(v)

        _flash_attn_sink_backward(
            grad_raw_output,  # dout: main path gradient
            q,                # q
            k,                # k
            v,                # v
            sink,
            ctx.raw_output,   # out: original output
            lse,              # softmax_lse
            grad_q_main,      # dq: main path
            grad_k_main,      # dk: main path
            grad_v,           # dv: main path
            torch.empty_like(sink),        # ds: sink gradient
            ctx.dropout_p,    # dropout_p
            ctx.softmax_scale,  # softmax_scale
            ctx.causal,       # causal
            ctx.window_size[0],  # window_size_left
            ctx.window_size[1],  # window_size_right
            ctx.softcap,      # softcap
            alibi_slopes,     # alibi_slopes
            ctx.deterministic,  # deterministic
        )

        # 2) Sink gradient path
        # g_r = (grad_output * raw_output).sum(dim=-1)  # [B,H,Nq]
        g_r = torch.sum(grad_output * raw_output, dim=-1)
        
        # g_ell = g_r * multiplier * (1 - multiplier)  # [B,H,Nq]
        # Based on debug output:
        # g_r shape: [1, 512, 64] (batch, seq_len, heads)
        # multiplier shape: [1, 512, 64, 1] (batch, seq_len, heads, 1)
        # We need multiplier_for_grad to have shape [1, 512, 64]
        # [1, 512, 64, 1] -> [1, 512, 64]
        multiplier_for_grad = multiplier.squeeze(-1)
        
        g_ell = g_r * multiplier_for_grad * (1 - multiplier_for_grad)
        # Based on shapes: g_ell [1, 512, 64], we need to sum over seq_len (dim=1)
        # to get [1, 64], then sum over batch (dim=0) to get [64]
        grad_sink = -torch.sum(g_ell, dim=1)  # Sum over seq_len -> [1, 64]
        # Sum over batch dimension and reshape to match original sink shape
        grad_sink = grad_sink.sum(dim=0)  # Sum over batch -> [64]
        grad_sink = grad_sink.reshape(ctx.sink_shape)

        # 3) Additional Q gradient via sink
        # dQ_extra = scale * g_ell * attention(Q,K,K)
        scale = ctx.softmax_scale or (1.0 / q.shape[-1] ** 0.5)

        # Compute attention(Q,K,K) for additional Q gradient
        mu_k = flash_attn_sink_func(
            q, k, k, sink,
            dropout_p=ctx.dropout_p,
            softmax_scale=ctx.softmax_scale,
            causal=ctx.causal,
            window_size=ctx.window_size,
            softcap=ctx.softcap,
            alibi_slopes=alibi_slopes,
            deterministic=ctx.deterministic,
            return_attn_probs=False,
        )
        grad_q_extra = scale * g_ell.unsqueeze(-1) * mu_k

        # 4) Additional K gradient via sink
        # dK_extra = scale * P^T (g_ell * Q)
        x = (g_ell.unsqueeze(-1) * q).to(q.dtype)

        # Use flash attention backward to compute P^T X
        grad_k_extra = torch.empty_like(k)
        _flash_attn_sink_backward(
            x,                  # dout: g_ell * Q
            q,                  # q
            k,                  # k
            k,                  # v (dummy, using K as V)
            sink,
            ctx.raw_output,     # out: original output
            lse,                # softmax_lse
            None,               # dq: not needed
            None,               # dk: not needed
            grad_k_extra,       # dv: this will be dK_extra
            torch.empty_like(sink),
            ctx.dropout_p,      # dropout_p
            ctx.softmax_scale,  # softmax_scale
            ctx.causal,         # causal
            ctx.window_size[0],  # window_size_left
            ctx.window_size[1],  # window_size_right
            ctx.softcap,        # softcap
            alibi_slopes,       # alibi_slopes
            ctx.deterministic,  # deterministic
        )
        grad_k_extra = scale * grad_k_extra

        # 5) Sum all gradients
        grad_q = grad_q_main + grad_q_extra
        grad_k = grad_k_main + grad_k_extra
        # grad_v already from main path

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
