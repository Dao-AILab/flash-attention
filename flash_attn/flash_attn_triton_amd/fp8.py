from typing import Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
from .utils import cast_to_fp8, is_fp8
from . import interface_fa as flash_attn_gpu


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x

class FlashAttnFP8Func(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_softmax,
        is_grad_enabled,
        descale_q: Optional[torch.Tensor] = None,
        descale_k: Optional[torch.Tensor] = None,
        descale_v: Optional[torch.Tensor] = None,
        descale_do: Optional[torch.Tensor] = None
    ):
        is_grad = is_grad_enabled and any(
            x.requires_grad for x in [q, k, v]
        )
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        head_size_og = q.size(3)
        if head_size_og % 8 != 0:
            q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
            k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
            v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])
        
        # figure out fwd parameters
        if is_fp8(q) or is_fp8(k) or is_fp8(v): # fp8 input and output
            raise ValueError("fp8 input and out not supported yet for this function.")
            assert (descale_q is not None) and (descale_k is not None) and (descale_v is not None), f"You need to pass descale factors for q, k and v"
            q_fp8 = q
            k_fp8 = k
            v_fp8 = v
            out_fp8, descale_o = torch.zeros_like(q_fp8), torch.zeros_like(descale_q)
        else: # cast to fp8 and return output in the fp32. (accumulator type) 
            assert (descale_q is None) and (descale_k is None) and (descale_v is None), f"Found {q.dtype} input tensor with descale factors. In this case, we cast to fp8 and compute the descale factors. You can pass an fp8 tensor with its descale factors if desired."
            q_fp8, descale_q = cast_to_fp8(q, torch.float8_e4m3fnuz, "bshd")
            k_fp8, descale_k = cast_to_fp8(k, torch.float8_e4m3fnuz, "bshd")
            v_fp8, descale_v = cast_to_fp8(v, torch.float8_e4m3fnuz, "bshd")
            out_fp8, descale_o = torch.zeros_like(q_fp8, dtype=torch.float32), None
                
        q_fp8, k_fp8, v_fp8 = [maybe_contiguous(x) for x in (q_fp8, k_fp8, v_fp8)]
        _, softmax_lse, S_dmask, rng_state = flash_attn_gpu.fwd(
            q_fp8,
            k_fp8,
            v_fp8,
            out_fp8,
            alibi_slopes,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            softcap=softcap,
            return_softmax=return_softmax and dropout_p > 0,
            descale_q=descale_q,
            descale_k=descale_k,
            descale_v=descale_v,
            descale_o=descale_o
        )
        if is_grad:
            ctx.save_for_backward(q_fp8, k_fp8, v_fp8, out_fp8, softmax_lse, rng_state, descale_q, descale_k, descale_v, descale_o, descale_do)
            ctx.dropout_p = dropout_p
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.window_size = window_size
            ctx.softcap = softcap
            ctx.alibi_slopes = alibi_slopes
            ctx.deterministic = deterministic
        out = out_fp8[..., :head_size_og] # NOTE: this used to be out_padded. It might cause issue doing an empty

        # check output type
        assert out.dtype == q.dtype, "Input and output type must match otherwise there will be implicit casting by autograd"
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        q_fp8, k_fp8, v_fp8, out_fp8, softmax_lse, rng_state, descale_q, descale_k, descale_v, descale_o, descale_do = ctx.saved_tensors
        head_size_og = dout.size(3)
        dout_padded = dout
        if head_size_og % 8 != 0:
            dout_padded = torch.nn.functional.pad(dout, [0, 8 - head_size_og % 8])

        # figure out bwd parameters
        if is_fp8(dout): # fp8 input and output
            raise ValueError("fp8 input and out not supported yet for this function.")
            assert (descale_do is not None), f"You need to pass descale factors for do"
            dout_padded_fp8 = dout_padded
            dq, descale_dq = torch.zeros_like(q_fp8), torch.zeros_like(descale_q)
            dk, descale_dk = torch.zeros_like(k_fp8), torch.zeros_like(descale_k)
            dv, descale_dv = torch.zeros_like(v_fp8), torch.zeros_like(descale_v)
        else: # cast to fp8 and return output in the fp32. (accumulator type) 
            assert (descale_do is None), f"Found {dout.dtype} input tensor with descale factors. In this case, we cast to fp8 and compute the descale factors. You can pass an fp8 tensor with its descale factors if desired."
            dout_padded_fp8, descale_do = cast_to_fp8(dout_padded, torch.float8_e4m3fnuz, "bshd")
            dq, descale_dq = torch.zeros_like(q_fp8, dtype=torch.float32), None
            dk, descale_dk = torch.zeros_like(k_fp8, dtype=torch.float32), None
            dv, descale_dv = torch.zeros_like(v_fp8, dtype=torch.float32), None
        
        # dq, dk, dv are allocated by us so they should already be contiguous
        dout_padded_fp8, q_fp8, k_fp8, v_fp8, out_fp8 = [maybe_contiguous(x) for x in (dout_padded_fp8, q_fp8, k_fp8, v_fp8, out_fp8)]
        flash_attn_gpu.bwd(
            dout_padded_fp8,
            q_fp8,
            k_fp8,
            v_fp8,
            out_fp8,
            softmax_lse,
            dq,
            dk,
            dv,
            ctx.alibi_slopes,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size[0],
            ctx.window_size[1],
            ctx.softcap,
            ctx.deterministic,
            None, # gen_
            rng_state,
            descale_q,
            descale_k,
            descale_v,
            descale_o,
            descale_do,
            descale_dq,
            descale_dk,
            descale_dv,
        )
        dq = dq[..., : dout.shape[-1]]  # We could have padded the head dimension
        dk = dk[..., : dout.shape[-1]]
        dv = dv[..., : dout.shape[-1]]
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None

def flash_attn_fp8_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0, # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    descale_q: Optional[torch.Tensor] = None,
    descale_k: Optional[torch.Tensor] = None,
    descale_v: Optional[torch.Tensor] = None,
    descale_do: Optional[torch.Tensor] = None
):
    return FlashAttnFP8Func.apply(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        torch.is_grad_enabled(),
        descale_q,
        descale_k,
        descale_v,
        descale_do
    )

class FlashAttnVarlenFP8Func(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_softmax,
        block_table,
        is_grad_enabled,
        descale_q: Optional[torch.Tensor] = None,
        descale_k: Optional[torch.Tensor] = None,
        descale_v: Optional[torch.Tensor] = None,
        descale_do: Optional[torch.Tensor] = None
    ):
        is_grad = is_grad_enabled and any(
            x.requires_grad for x in [q, k, v]
        )
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        head_size_og = q.size(2)
        if head_size_og % 8 != 0:
            q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
            k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
            v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])
        
        # figure out fwd parameters
        if is_fp8(q) or is_fp8(k) or is_fp8(v): # fp8 input and output
            raise ValueError("fp8 input and out not supported yet for this function.")
            assert (descale_q is not None) and (descale_k is not None) and (descale_v is not None), f"You need to pass descale factors for q, k and v"
            q_fp8 = q
            k_fp8 = k
            v_fp8 = v
            out_fp8, descale_o = torch.zeros_like(q_fp8), torch.zeros_like(descale_q)
        else: # cast to fp8 and return output in the fp32. (accumulator type) 
            assert (descale_q is None) and (descale_k is None) and (descale_v is None), f"Found {q.dtype} input tensor with descale factors. In this case, we cast to fp8 and compute the descale factors. You can pass an fp8 tensor with its descale factors if desired."
            q_fp8, descale_q = cast_to_fp8(q, torch.float8_e4m3fnuz, "thd", cu_seqlens=cu_seqlens_q, max_seqlen=max_seqlen_q)
            k_fp8, descale_k = cast_to_fp8(k, torch.float8_e4m3fnuz, "thd", cu_seqlens=cu_seqlens_k, max_seqlen=max_seqlen_k)
            v_fp8, descale_v = cast_to_fp8(v, torch.float8_e4m3fnuz, "thd", cu_seqlens=cu_seqlens_k, max_seqlen=max_seqlen_k)
            out_fp8, descale_o = torch.zeros_like(q_fp8, dtype=torch.float32), None
                
        q_fp8, k_fp8, v_fp8 = [maybe_contiguous(x) for x in (q_fp8, k_fp8, v_fp8)]
        _, softmax_lse, S_dmask, rng_state = flash_attn_gpu.varlen_fwd(
            q_fp8,
            k_fp8,
            v_fp8,
            out_fp8,
            cu_seqlens_q,
            cu_seqlens_k,
            None,
            None,
            block_table,
            alibi_slopes,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            softmax_scale,
            False,
            causal,
            window_size[0],
            window_size[1],
            softcap,
            return_softmax,
            None,
            descale_q=descale_q,
            descale_k=descale_k,
            descale_v=descale_v,
            descale_o=descale_o
        )
        if is_grad:
            ctx.save_for_backward(q_fp8, k_fp8, v_fp8, out_fp8, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state, descale_q, descale_k, descale_v, descale_o, descale_do)
            ctx.dropout_p = dropout_p
            ctx.max_seqlen_q = max_seqlen_q
            ctx.max_seqlen_k = max_seqlen_k
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.window_size = window_size
            ctx.softcap = softcap
            ctx.alibi_slopes = alibi_slopes
            ctx.deterministic = deterministic
        out = out_fp8[..., :head_size_og] # NOTE: this used to be out_padded. It might cause issue doing an empty

        # check output type
        assert out.dtype == q.dtype, "Input and output type must match otherwise there will be implicit casting by autograd"
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        q_fp8, k_fp8, v_fp8, out_fp8, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state, descale_q, descale_k, descale_v, descale_o, descale_do = ctx.saved_tensors
        head_size_og = dout.size(2)
        dout_padded = dout
        if head_size_og % 8 != 0:
            dout_padded = torch.nn.functional.pad(dout, [0, 8 - head_size_og % 8])

        # figure out bwd parameters
        if is_fp8(dout_padded): # fp8 input and output
            raise ValueError("fp8 input and out not supported yet for this function.")
            assert (descale_do is not None), f"You need to pass descale factors for do"
            dout_padded_fp8 = dout_padded
            dq, descale_dq = torch.zeros_like(q_fp8), torch.zeros_like(descale_q)
            dk, descale_dk = torch.zeros_like(k_fp8), torch.zeros_like(descale_k)
            dv, descale_dv = torch.zeros_like(v_fp8), torch.zeros_like(descale_v)
        else: # cast to fp8 and return output in the fp32. (accumulator type) 
            assert (descale_do is None), f"Found {dout.dtype} input tensor with descale factors. In this case, we cast to fp8 and compute the descale factors. You can pass an fp8 tensor with its descale factors if desired."
            dout_padded_fp8, descale_do = cast_to_fp8(dout_padded, torch.float8_e4m3fnuz, "thd", cu_seqlens=cu_seqlens_q, max_seqlen=ctx.max_seqlen_q)
            dq, descale_dq = torch.zeros_like(q_fp8, dtype=torch.float32), None
            dk, descale_dk = torch.zeros_like(k_fp8, dtype=torch.float32), None
            dv, descale_dv = torch.zeros_like(v_fp8, dtype=torch.float32), None
        
        # dq, dk, dv are allocated by us so they should already be contiguous
        dout_padded_fp8, q_fp8, k_fp8, v_fp8, out_fp8 = [maybe_contiguous(x) for x in (dout_padded_fp8, q_fp8, k_fp8, v_fp8, out_fp8)]
        flash_attn_gpu.varlen_bwd(
            dout_padded_fp8,
            q_fp8,
            k_fp8,
            v_fp8,
            out_fp8,
            softmax_lse,
            dq,
            dk,
            dv,
            cu_seqlens_q,
            cu_seqlens_k,
            ctx.alibi_slopes,
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            ctx.dropout_p,
            ctx.softmax_scale,
            False,
            ctx.causal,
            ctx.window_size[0],
            ctx.window_size[1],
            ctx.softcap,
            ctx.deterministic,
            None,
            rng_state,
            descale_q,
            descale_k,
            descale_v,
            descale_o,
            descale_do,
            descale_dq,
            descale_dk,
            descale_dv,
        )
        dq = dq[..., : dout.shape[-1]]  # We could have padded the head dimension
        dk = dk[..., : dout.shape[-1]]
        dv = dv[..., : dout.shape[-1]]
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None


def flash_attn_varlen_fp8_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0, # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None
):
    return FlashAttnVarlenFP8Func.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        block_table,
        torch.is_grad_enabled()
    )

class FlashAttnQKVPackedFP8Func(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_softmax,
        is_grad_enabled,
        descale_q: Optional[torch.Tensor] = None,
        descale_k: Optional[torch.Tensor] = None,
        descale_v: Optional[torch.Tensor] = None,
        descale_do: Optional[torch.Tensor] = None
    ):
        is_grad = is_grad_enabled and qkv.requires_grad
        if softmax_scale is None:
            softmax_scale = qkv.shape[-1] ** (-0.5)
        q, k, v = qkv[:, :, 0].detach(), qkv[:, :, 1].detach(), qkv[:, :, 2].detach()
        head_size_og = q.size(3)
        if head_size_og % 8 != 0:
            q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
            k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
            v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])

        # figure out fwd parameters
        if is_fp8(q) or is_fp8(k) or is_fp8(v): # fp8 input and output
            raise ValueError("fp8 input and out not supported yet for this function.")
            assert (descale_q is not None) and (descale_k is not None) and (descale_v is not None), f"You need to pass descale factors for q, k and v"
            q_fp8 = q
            k_fp8 = k
            v_fp8 = v
            out_fp8, descale_o = torch.zeros_like(q_fp8), torch.zeros_like(descale_q)
        else: # cast to fp8 and return output in the fp32. (accumulator type) 
            assert (descale_q is None) and (descale_k is None) and (descale_v is None), f"Found {q.dtype} input tensor with descale factors. In this case, we cast to fp8 and compute the descale factors. You can pass an fp8 tensor with its descale factors if desired."
            q_fp8, descale_q = cast_to_fp8(q, torch.float8_e4m3fnuz, "bshd")
            k_fp8, descale_k = cast_to_fp8(k, torch.float8_e4m3fnuz, "bshd")
            v_fp8, descale_v = cast_to_fp8(v, torch.float8_e4m3fnuz, "bshd")
            out_fp8, descale_o = torch.zeros_like(q_fp8, dtype=torch.float32), None

        q_fp8, k_fp8, v_fp8 = [maybe_contiguous(x) for x in (q_fp8, k_fp8, v_fp8)]
        _, softmax_lse, S_dmask, rng_state = flash_attn_gpu.fwd(
            q_fp8,
            k_fp8,
            v_fp8,
            out_fp8,
            alibi_slopes,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            softcap=softcap,
            return_softmax=return_softmax and dropout_p > 0,
            descale_q=descale_q,
            descale_k=descale_k,
            descale_v=descale_v,
            descale_o=descale_o,
        )
        if is_grad:
            ctx.save_for_backward(q_fp8, k_fp8, v_fp8, out_fp8, softmax_lse, rng_state, descale_q, descale_k, descale_v, descale_o, descale_do)
            ctx.dropout_p = dropout_p
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.window_size = window_size
            ctx.softcap = softcap
            ctx.alibi_slopes = alibi_slopes
            ctx.deterministic = deterministic
        out = out_fp8[..., :head_size_og]
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        q_fp8, k_fp8, v_fp8, out_fp8, softmax_lse, rng_state, descale_q, descale_k, descale_v, descale_o, descale_do = ctx.saved_tensors
        qkv_shape = q_fp8.shape[:-2] + (3, *q_fp8.shape[-2:])
        head_size_og = dout.size(3)
        dout_padded = dout
        if head_size_og % 8 != 0:
            dout_padded = torch.nn.functional.pad(dout, [0, 8 - head_size_og % 8])
        
        # figure out bwd parameters
        if is_fp8(dout): # fp8 input and output
            raise ValueError("fp8 input and out not supported yet for this function.")
            assert (descale_do is not None), f"You need to pass descale factors for do"
            dout_padded_fp8 = dout_padded
            dqkv, descale_dqkv = torch.zeros(qkv_shape, device=q_fp8.device), torch.zeros_like(descale_q)
        else: # cast to fp8 and return output in the fp32. (accumulator type) 
            assert (descale_do is None), f"Found {dout.dtype} input tensor with descale factors. In this case, we cast to fp8 and compute the descale factors. You can pass an fp8 tensor with its descale factors if desired."
            dout_padded_fp8, descale_do = cast_to_fp8(dout_padded, torch.float8_e4m3fnuz, "bshd")
            dqkv, descale_dqkv = torch.zeros(qkv_shape, dtype=torch.float32, device=q_fp8.device), None
        
        
        # dq, dk, dv are allocated by us so they should already be contiguous
        dout_padded_fp8, q_fp8, k_fp8, v_fp8, out_fp8 = [maybe_contiguous(x) for x in (dout_padded_fp8, q_fp8, k_fp8, v_fp8, out_fp8)]
        flash_attn_gpu.bwd(
            dout_padded_fp8,
            q_fp8,
            k_fp8,
            v_fp8,
            out_fp8,
            softmax_lse,
            dqkv[:, :, 0],
            dqkv[:, :, 1],
            dqkv[:, :, 2],
            ctx.alibi_slopes,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size[0],
            ctx.window_size[1],
            ctx.softcap,
            ctx.deterministic,
            None, # gen_
            rng_state,
            descale_q,
            descale_k,
            descale_v,
            descale_o,
            descale_do,
            None,
            None,
            None,
        )
        dqkv = dqkv[..., : dout.shape[-1]]  # We could have padded the head dimension
        return dqkv, None, None, None, None, None, None, None, None, None


def flash_attn_qkvpacked_fp8_func(
    qkv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0,  # <=0.0 means deactivate
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    return FlashAttnQKVPackedFP8Func.apply(
        qkv,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        torch.is_grad_enabled(),
    )


class FlashAttnVarlenQKVPackedFP8Func(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv,
        cu_seqlens,
        max_seqlen,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_softmax,
        is_grad_enabled,
        descale_q: Optional[torch.Tensor] = None,
        descale_k: Optional[torch.Tensor] = None,
        descale_v: Optional[torch.Tensor] = None,
        descale_do: Optional[torch.Tensor] = None
    ):
        is_grad = is_grad_enabled and qkv.requires_grad
        if softmax_scale is None:
            softmax_scale = qkv.shape[-1] ** (-0.5)
        q, k, v = qkv[:, 0].detach(), qkv[:, 1].detach(), qkv[:, 2].detach()
        head_size_og = q.size(2)
        if head_size_og % 8 != 0:
            q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
            k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
            v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])

        # figure out fwd parameters
        if is_fp8(q) or is_fp8(k) or is_fp8(v): # fp8 input and output
            raise ValueError("fp8 input and out not supported yet for this function.")
            assert (descale_q is not None) and (descale_k is not None) and (descale_v is not None), f"You need to pass descale factors for q, k and v"
            q_fp8 = q
            k_fp8 = k
            v_fp8 = v
            out_fp8, descale_o = torch.zeros_like(q_fp8), torch.zeros_like(descale_q)
        else: # cast to fp8 and return output in the fp32. (accumulator type) 
            assert (descale_q is None) and (descale_k is None) and (descale_v is None), f"Found {q.dtype} input tensor with descale factors. In this case, we cast to fp8 and compute the descale factors. You can pass an fp8 tensor with its descale factors if desired."
            q_fp8, descale_q = cast_to_fp8(q, torch.float8_e4m3fnuz, "thd", cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
            k_fp8, descale_k = cast_to_fp8(k, torch.float8_e4m3fnuz, "thd", cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
            v_fp8, descale_v = cast_to_fp8(v, torch.float8_e4m3fnuz, "thd", cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
            out_fp8, descale_o = torch.zeros_like(q_fp8, dtype=torch.float32), None

        q_fp8, k_fp8, v_fp8 = [maybe_contiguous(x) for x in (q_fp8, k_fp8, v_fp8)]
        _, softmax_lse, S_dmask, rng_state = flash_attn_gpu.varlen_fwd(
            q_fp8,
            k_fp8,
            v_fp8,
            out_fp8,
            cu_seqlens,
            cu_seqlens,
            None,
            None,
            None,
            alibi_slopes,
            max_seqlen,
            max_seqlen,
            dropout_p,
            softmax_scale,
            False,
            causal,
            window_size[0],
            window_size[1],
            softcap,
            return_softmax,
            None,
            descale_q=descale_q,
            descale_k=descale_k,
            descale_v=descale_v,
            descale_o=descale_o
        )
        if is_grad:
            ctx.save_for_backward(q_fp8, k_fp8, v_fp8, out_fp8, softmax_lse, cu_seqlens, rng_state, descale_q, descale_k, descale_v, descale_o, descale_do)
            ctx.dropout_p = dropout_p
            ctx.max_seqlen = max_seqlen
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.window_size = window_size
            ctx.softcap = softcap
            ctx.alibi_slopes = alibi_slopes
            ctx.deterministic = deterministic
        out = out_fp8[..., :head_size_og]
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        q_fp8, k_fp8, v_fp8, out_fp8, softmax_lse, cu_seqlens, rng_state, descale_q, descale_k, descale_v, descale_o, descale_do = ctx.saved_tensors
        qkv_shape = q_fp8.shape[:-2] + (3, *q_fp8.shape[-2:])
        head_size_og = dout.size(2)
        dout_padded = dout
        if head_size_og % 8 != 0:
            dout_padded = torch.nn.functional.pad(dout, [0, 8 - head_size_og % 8])

        # figure out bwd parameters
        if is_fp8(dout_padded): # fp8 input and output
            raise ValueError("fp8 input and out not supported yet for this function.")
            assert (descale_do is not None), f"You need to pass descale factors for do"
            dout_padded_fp8 = dout_padded
            dqkv, descale_dqkv = torch.zeros(qkv_shape, device=q_fp8.device), torch.zeros_like(descale_q)
        else: # cast to fp8 and return output in the fp32. (accumulator type) 
            assert (descale_do is None), f"Found {dout.dtype} input tensor with descale factors. In this case, we cast to fp8 and compute the descale factors. You can pass an fp8 tensor with its descale factors if desired."
            dout_padded_fp8, descale_do = cast_to_fp8(dout_padded, torch.float8_e4m3fnuz, "thd", cu_seqlens=cu_seqlens, max_seqlen=ctx.max_seqlen)
            dqkv, descale_dqkv = torch.zeros(qkv_shape, dtype=torch.float32, device=q_fp8.device), None
        
        # dq, dk, dv are allocated by us so they should already be contiguous
        dout_padded_fp8, q_fp8, k_fp8, v_fp8, out_fp8 = [maybe_contiguous(x) for x in (dout_padded_fp8, q_fp8, k_fp8, v_fp8, out_fp8)]
        flash_attn_gpu.varlen_bwd(
            dout_padded_fp8,
            q_fp8,
            k_fp8,
            v_fp8,
            out_fp8,
            softmax_lse,
            dqkv[:, 0],
            dqkv[:, 1],
            dqkv[:, 2],
            cu_seqlens,
            cu_seqlens,
            ctx.alibi_slopes,
            ctx.max_seqlen,
            ctx.max_seqlen,
            ctx.dropout_p,
            ctx.softmax_scale,
            False,
            ctx.causal,
            ctx.window_size[0],
            ctx.window_size[1],
            ctx.softcap,
            ctx.deterministic,
            None,
            rng_state,
            descale_q,
            descale_k,
            descale_v,
            descale_o,
            descale_do,
            None,
            None,
            None,
        )
        dqkv = dqkv[..., : dout.shape[-1]]  # We could have padded the head dimension
        return dqkv, None, None, None, None, None, None, None, None, None, None, None


def flash_attn_varlen_qkvpacked_fp8_func(
    qkv,
    cu_seqlens,
    max_seqlen,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0, # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    return FlashAttnVarlenQKVPackedFP8Func.apply(
        qkv,
        cu_seqlens,
        max_seqlen,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        torch.is_grad_enabled(),
    )
