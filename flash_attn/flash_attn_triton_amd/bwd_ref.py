import torch
import math
from typing import Literal, Optional
from .utils import DEBUG, compute_alibi_tensor_ref

DEBUG_CORE = False

def attention_backward_core_ref_impl(
    do, q, k, v, o, softmax_lse, sm_scale, causal, dropout_p, philox_seed, philox_offset, alibi_slopes, use_exp2
):
    if DEBUG_CORE:
        print()
        print("attention_backward_core_ref_impl")
        print("do:", do, do.shape)
        print("q:", q, q.shape)
        print("k:", k, k.shape)
        print("v:", v, v.shape)
        print("o:", o, o.shape) # is a bad number
        print("softmax_lse:", softmax_lse, softmax_lse.shape)
        print("sm_scale:", sm_scale)
        print("causal:", causal)
        print("dropout_p:", dropout_p)
        print("philox_seed:", philox_seed)
        print("philox_offset:", philox_offset)
        print("use_exp2:", use_exp2)
    
    # cast to float32
    do = do.to(torch.float32)
    q = q.to(torch.float32)
    k = k.to(torch.float32)
    v = v.to(torch.float32)
    o = o.to(torch.float32)
    softmax_lse = softmax_lse.to(torch.float32)


    # recompute attention_scores. Make sure it matches the forward impl. i.e. It use float32
    attention_scores = torch.matmul(q, k.transpose(-2, -1))
    if DEBUG_CORE:
        print("attention_scores:", attention_scores, attention_scores.shape)

    # scale scores
    attention_scaled_scores = sm_scale * attention_scores
    if DEBUG_CORE:
        print("attention_scaled_scores:", attention_scaled_scores, attention_scaled_scores.shape)

    if alibi_slopes is not None:
        L_q, L_k = q.shape[1], k.shape[1]
        if DEBUG_CORE:
            print("alibi_slopes:", alibi_slopes, alibi_slopes.shape)
        alibi_bias = compute_alibi_tensor_ref(alibi_slopes, L_q, L_k)
        alibi_bias = alibi_bias.reshape(-1, L_q, L_k)
        if True:
            print("alibi_bias:", alibi_bias, alibi_bias.shape)
        attention_scaled_scores = attention_scaled_scores + alibi_bias
        if DEBUG_CORE:
            print("attention_scaled_scores after alibi:", attention_scaled_scores, attention_scaled_scores.shape)

    # Apply causal mask if necessary
    if causal:
        L_q, L_k = q.shape[1], k.shape[1]
        row_idx = torch.arange(L_q, device=q.device).unsqueeze(1)
        col_idx = torch.arange(L_k, device=q.device).unsqueeze(0)
        col_offset = L_q-L_k
        causal_mask = row_idx >= (col_offset + col_idx)
        if DEBUG_CORE:
            print("causal_mask:", causal_mask)
        # set -inf to places the causal mask is false
        attention_scaled_scores = attention_scaled_scores.masked_fill(
             torch.logical_not(causal_mask.unsqueeze(0)), float('-inf')
        )
        if DEBUG_CORE:
            print("attention_scaled_scores after causal:", attention_scaled_scores, attention_scaled_scores.shape)

    # compute probabilities using softmax_lse
    if use_exp2:
        RCP_LN = 1 / math.log(2)
        attention_scaled_scores_base2 = attention_scaled_scores * RCP_LN
        softmax_lse_base2 = softmax_lse * RCP_LN
        softmax_lse_3d =  softmax_lse_base2.unsqueeze(-1)
        p = torch.exp2(attention_scaled_scores_base2 - softmax_lse_3d)
    else:
        softmax_lse_3d =  softmax_lse.unsqueeze(-1)
        p = torch.exp(attention_scaled_scores - softmax_lse_3d)
    if DEBUG_CORE:
        print("softmax_lse_3d:", softmax_lse_3d, softmax_lse_3d.shape)
        print("p:", p, p.shape)

    if dropout_p > 0.0:
        rand_vals = torch.rand(p.shape, generator=torch.Generator(device=p.device).manual_seed(philox_seed), device=p.device, dtype=p.dtype)
        dropout_mask, dropout_scale = rand_vals > dropout_p,  (1.0 / (1 - dropout_p))
        if DEBUG:
            print("dropout_scale:", dropout_scale)
            print("dropout_mask:", dropout_mask)
            
        p_drop = torch.where(dropout_mask, p, torch.zeros_like(p))
        p_drop_scaled =  p_drop * dropout_scale
        if DEBUG_CORE:
            print("dropout_scale:", dropout_scale)
            print("p_drop:", p_drop, p_drop.shape)
            print("p_drop_scaled:", p_drop_scaled, p_drop_scaled.shape)
        
        # compute dv
        dv = torch.matmul(p_drop_scaled.transpose(-2, -1), do)
        if DEBUG_CORE:
            print("dv:", dv, dv.shape)

        # compute dp
        dp_dropout = torch.matmul(do, v.transpose(-2, -1))
        dp = torch.where(dropout_mask, dp_dropout , torch.zeros_like(dp_dropout)) * dropout_scale
        if DEBUG_CORE:
            print("dp_dropout:", dp_dropout, dp_dropout.shape)
            print("dp:", dp, dp.shape)
    else:
        # compute dv
        dv = torch.matmul(p.transpose(-2, -1), do)
        if DEBUG_CORE:
            print("dv:", dv, dv.shape)

        # compute dp
        dp = torch.matmul(do, v.transpose(-2, -1))
        if DEBUG_CORE:
            print("dp:", dp, dp.shape)

    # calculate ds
    if False:
        delta = torch.sum(o * do, axis=-1).unsqueeze(-1)
    else:
        delta = torch.sum(p * dp, axis=-1).unsqueeze(-1)
    if DEBUG:
        print("delta:", delta, delta.shape)
    dscores_scaled = p * (dp - delta)
    ds = dscores_scaled * sm_scale
    if DEBUG_CORE:
        print("dscores_scaled:", dscores_scaled, dscores_scaled.shape)
        print("ds:", ds, ds.shape)

    # compute gradient wrt k & q
    dk = torch.matmul(ds.transpose(-2, -1), q)
    dq = torch.matmul(ds, k)
    if DEBUG_CORE:
        print("dk:", dk, dk.shape)
        print("dq:", dq, dq.shape)

    # cast back to original dtype
    dq = dq.to(torch.float16)
    dk = dk.to(torch.float16)
    dv = dv.to(torch.float16)
    # remove d dim with size 1
    delta = delta.squeeze(-1)

    if DEBUG_CORE:
        print("attention_backward_core_ref_impl output")
        print("delta:", delta, delta.shape)
        print("dv:", dv, dv.shape)
        print("dk:", dk, dk.shape)
        print("dq:", dq, dq.shape)

    return dq, dk, dv, delta

def attention_varlen_backward_pytorch_ref_impl(
    do,
    q,
    k,
    v,
    o,
    softmax_lse,
    sm_scale,
    causal,
    layout,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p, 
    philox_seed, 
    philox_offset,
    alibi_slopes,
    use_exp2,
):
    # Ensure the layout is 'thd'
    if layout != 'thd':
        raise ValueError(f"Unsupported layout {layout}. Expected 'thd'.")

    batch_size = cu_seqlens_q.shape[0] - 1
    nheads_q, head_dim = q.shape[1], q.shape[2]
    nheads_k = k.shape[1]

    group_size = nheads_q // nheads_k
    if nheads_q % nheads_k != 0:
        raise ValueError("nheads_q must be divisible by nheads_k")

    # Pre-allocate outputs
    total_L_q = q.shape[0]
    total_L_k = k.shape[0]

    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)
    # delta has the same shape as softmax_lse: [total_L_q, nheads_q]
    delta = torch.zeros((total_L_q, nheads_q), dtype=torch.float32, device=o.device)

    for i in range(batch_size):
        # Get the start and end indices for the current sequence
        start_q = cu_seqlens_q[i].item()
        end_q = cu_seqlens_q[i + 1].item()
        start_k = cu_seqlens_k[i].item()
        end_k = cu_seqlens_k[i + 1].item()

        # Extract q_i, k_i, v_i, do_i, o_i, softmax_lse_i
        q_i = q[start_q:end_q, :, :]      # [L_q_i, nheads_q, head_dim]
        k_i = k[start_k:end_k, :, :]      # [L_k_i, nheads_k, head_dim]
        v_i = v[start_k:end_k, :, :]      # [L_k_i, nheads_k, head_dim]
        do_i = do[start_q:end_q, :, :]    # [L_q_i, nheads_q, head_dim]
        o_i = o[start_q:end_q, :, :]      # [L_q_i, nheads_q, head_dim]
        softmax_lse_i = softmax_lse[start_q:end_q, :] # [L_q_i, nheads_q]

        if group_size != 1:
            # MQA or GQA case
            # Reshape tensors to include group dimension
            q_i = q_i.view(q_i.shape[0], nheads_k, group_size, head_dim)
            do_i = do_i.view(do_i.shape[0], nheads_k, group_size, head_dim)
            o_i = o_i.view(o_i.shape[0], nheads_k, group_size, head_dim)
            softmax_lse_i = softmax_lse_i.view(softmax_lse_i.shape[0], nheads_k, group_size)
            # Expand k_i and v_i to match group_size
            k_i = k_i.unsqueeze(2).expand(-1, -1, group_size, -1)
            v_i = v_i.unsqueeze(2).expand(-1, -1, group_size, -1)
            # Flatten the nheads_k and group_size dimensions
            q_i = q_i.reshape(q_i.shape[0], nheads_k * group_size, head_dim)
            do_i = do_i.reshape(do_i.shape[0], nheads_k * group_size, head_dim)
            o_i = o_i.reshape(o_i.shape[0], nheads_k * group_size, head_dim)
            softmax_lse_i = softmax_lse_i.reshape(softmax_lse_i.shape[0], nheads_k * group_size)
            k_i = k_i.reshape(k_i.shape[0], nheads_k * group_size, head_dim)
            v_i = v_i.reshape(v_i.shape[0], nheads_k * group_size, head_dim)
        # Permute to [nheads_total, L, head_dim]
        q_i = q_i.permute(1, 0, 2)
        k_i = k_i.permute(1, 0, 2)
        v_i = v_i.permute(1, 0, 2)
        do_i = do_i.permute(1, 0, 2)
        o_i = o_i.permute(1, 0, 2)
        softmax_lse_i = softmax_lse_i.transpose(0, 1)
        if alibi_slopes is not None:
            alibi_slopes_i = alibi_slopes[i]
        else:
            alibi_slopes_i = None

        # Call the core backward function for this sequence
        dq_i, dk_i, dv_i, delta_i = attention_backward_core_ref_impl(
            do_i,
            q_i,
            k_i,
            v_i,
            o_i,
            softmax_lse_i,
            sm_scale,
            causal,
            dropout_p, 
            philox_seed, 
            philox_offset,
            alibi_slopes_i,
            use_exp2
        )

        # Convert back to 'thd' layout
        dq_i = dq_i.permute(1, 0, 2)  # [L_q_i, nheads_total, head_dim]
        dk_i = dk_i.permute(1, 0, 2)  # [L_k_i, nheads_total, head_dim]
        dv_i = dv_i.permute(1, 0, 2)  # [L_k_i, nheads_total, head_dim]
        delta_i = delta_i.transpose(1, 0)  # [L_q_i, nheads_total]

        if group_size != 1:
            # Reshape dq_i and delta_i back to original shape
            dq_i = dq_i.view(dq_i.shape[0], nheads_k, group_size, head_dim)
            delta_i = delta_i.view(delta_i.shape[0], nheads_k, group_size)
            # Sum dk_i and dv_i over group dimension
            dk_i = dk_i.view(dk_i.shape[0], nheads_k, group_size, head_dim)
            dv_i = dv_i.view(dv_i.shape[0], nheads_k, group_size, head_dim)
            dk_i = dk_i.sum(dim=2)
            dv_i = dv_i.sum(dim=2)
            # Reshape dq_i back to [L_q_i, nheads_q, head_dim]
            dq_i = dq_i.reshape(dq_i.shape[0], nheads_q, head_dim)
            delta_i = delta_i.reshape(delta_i.shape[0], nheads_q)
        else:
            # No need to reshape
            pass

        # Place outputs in pre-allocated tensors
        dq[start_q:end_q, :, :] = dq_i
        dk[start_k:end_k, :, :] += dk_i  # Accumulate gradients for shared keys
        dv[start_k:end_k, :, :] += dv_i  # Accumulate gradients for shared values
        delta[start_q:end_q, :] = delta_i

    return dq, dk, dv, delta

def attention_vanilla_backward_pytorch_ref_impl(
    do,
    q,
    k,
    v,
    o,
    softmax_lse,
    sm_scale,
    causal,
    layout,
    dropout_p,
    philox_seed,
    philox_offset,
    alibi_slopes,
    use_exp2,
):
    if layout == "bshd":
        if DEBUG:
            print()
            print("Changing layout to bhsd!")
        do = do.transpose(1, 2).contiguous()
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        o = o.transpose(1, 2).contiguous()
    elif layout == "bhsd":
        pass
    else:
        raise ValueError(f"Unknown layout {layout}")
    
    # Prepare tensors
    batch_size, nheads_q, seq_len_q, head_dim = q.shape
    batch_size, nheads_k, seq_len_k, head_dim = k.shape

    group_size = nheads_q // nheads_k
    if nheads_q % nheads_k != 0:
        raise ValueError("nheads_q must be divisible by nheads_k")

    if group_size != 1:
        # MQA or GQA case
        # Reshape do, q, o to [batch_size, nheads_k, group_size, seq_len_q, head_dim]
        do = do.reshape(batch_size, nheads_k, group_size, seq_len_q, head_dim)
        q = q.reshape(batch_size, nheads_k, group_size, seq_len_q, head_dim)
        o = o.reshape(batch_size, nheads_k, group_size, seq_len_q, head_dim)
        # Reshape softmax_lse to [batch_size, nheads_k, group_size, seq_len_q]
        softmax_lse = softmax_lse.reshape(batch_size, nheads_k, group_size, seq_len_q)
        # Expand k and v to match group_size
        k = k.unsqueeze(2).expand(-1, -1, group_size, -1, -1)  # [batch_size, nheads_k, group_size, seq_len_k, head_dim]
        v = v.unsqueeze(2).expand(-1, -1, group_size, -1, -1)
        # Flatten the first three dimensions for computation
        do = do.reshape(batch_size * nheads_k * group_size, seq_len_q, head_dim)
        q = q.reshape(batch_size * nheads_k * group_size, seq_len_q, head_dim)
        k = k.reshape(batch_size * nheads_k * group_size, seq_len_k, head_dim)
        v = v.reshape(batch_size * nheads_k * group_size, seq_len_k, head_dim)
        o = o.reshape(batch_size * nheads_k * group_size, seq_len_q, head_dim)
        softmax_lse = softmax_lse.reshape(batch_size * nheads_k * group_size, seq_len_q)
    else:
        # Standard case
        do = do.reshape(batch_size * nheads_q, seq_len_q, head_dim)
        q = q.reshape(batch_size * nheads_q, seq_len_q, head_dim)
        k = k.reshape(batch_size * nheads_k, seq_len_k, head_dim)
        v = v.reshape(batch_size * nheads_k, seq_len_k, head_dim)
        o = o.reshape(batch_size * nheads_q, seq_len_q, head_dim)
        softmax_lse = softmax_lse.reshape(batch_size * nheads_q, seq_len_q)

    # Call the core backward function
    dq, dk, dv, delta = attention_backward_core_ref_impl(
        do,
        q,
        k,
        v,
        o,
        softmax_lse,
        sm_scale,
        causal,
        dropout_p, 
        philox_seed, 
        philox_offset,
        alibi_slopes,
        use_exp2
    )

    if group_size != 1:
        # Reshape dq back to [batch_size, nheads_k, group_size, seq_len_q, head_dim]
        dq = dq.reshape(batch_size, nheads_k, group_size, seq_len_q, head_dim)
        # Reshape delta back to [batch_size, nheads_k, group_size, seq_len_q]
        delta = delta.reshape(batch_size, nheads_k, group_size, seq_len_q)
        # Sum dk and dv over group_size dimension, since k and v are shared across groups
        dk = dk.reshape(batch_size, nheads_k, group_size, seq_len_k, head_dim)
        dk = dk.sum(dim=2)  # Sum over group_size dimension
        dv = dv.reshape(batch_size, nheads_k, group_size, seq_len_k, head_dim)
        dv = dv.sum(dim=2)
        # Reshape dq to [batch_size, nheads_q, seq_len_q, head_dim]
        dq = dq.reshape(batch_size, nheads_k * group_size, seq_len_q, head_dim)
        delta = delta.reshape(batch_size, nheads_k * group_size, seq_len_q)
    else:
        # Standard case
        dq = dq.reshape(batch_size, nheads_q, seq_len_q, head_dim)
        dk = dk.reshape(batch_size, nheads_k, seq_len_k, head_dim)
        dv = dv.reshape(batch_size, nheads_k, seq_len_k, head_dim)
        delta = delta.reshape(batch_size, nheads_q, seq_len_q)

    # Go back to original layout
    if layout == "bshd":
        if DEBUG:
            print()
            print("Changing back to bshd!")
        dq = dq.transpose(1, 2)
        dk = dk.transpose(1, 2)
        dv = dv.transpose(1, 2)
    elif layout == "bhsd":
        pass
    else:
        raise ValueError(f"Unknown layout {layout}")

    return dq, dk, dv, delta

def attention_backward_pytorch_ref_impl(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    sm_scale: float,
    alibi_slopes: Optional[torch.Tensor],
    causal: bool,
    layout: Literal["bshd", "bhsd", "thd"],
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    max_seqlen_q: Optional[int],
    max_seqlen_k: Optional[int],
    dropout_p: float, 
    philox_seed: Optional[int], 
    philox_offset: Optional[int],
    use_exp2: bool
):
    if layout == "thd":
        dq_ref, dk_ref, dv_ref, delta = attention_varlen_backward_pytorch_ref_impl(
            do,
            q,
            k,
            v,
            o,
            softmax_lse,
            sm_scale,
            causal,
            layout,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p, 
            philox_seed,
            philox_offset,
            alibi_slopes,
            use_exp2,
        )
    else:
        dq_ref, dk_ref, dv_ref, delta = attention_vanilla_backward_pytorch_ref_impl(
            do,
            q,
            k,
            v,
            o,
            softmax_lse,
            sm_scale,
            causal,
            layout,
            dropout_p, 
            philox_seed, 
            philox_offset,
            alibi_slopes,
            use_exp2,
        )
        

    # copy into output tensor
    dv.copy_(dv_ref.to(dv.dtype))
    dk.copy_(dk_ref.to(dk.dtype))
    dq.copy_(dq_ref.to(dq.dtype))

    return delta
