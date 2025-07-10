import torch
import math
from typing import Literal, Optional, Union
from .utils import compute_alibi_tensor_ref

DEBUG = False
DEBUG_CORE = False

def attention_forward_core_ref_impl(
    q, k, v, sm_scale, causal, window_size_left, window_size_right, 
    dropout_p, philox_seed, philox_offset, alibi_slopes, use_exp2,
    cache_seqlens=None
):
    if DEBUG_CORE:
        print()
        print("attention_forward_core_ref_impl")
        print("q:", q, q.shape)
        print("k:", k, k.shape)
        print("v:", v, v.shape)
        print("sm_scale:", sm_scale)
        print("causal:", causal)
        print("window_size_left:", window_size_left)
        print("window_size_right:", window_size_right)
        print("dropout_p:", dropout_p)
        print("philox_seed:", philox_seed)
        print("philox_offset:", philox_offset)
        print("use_exp2:", use_exp2)
        print("cache_seqlens:", cache_seqlens)

    # cast to float32
    q = q.to(torch.float32)
    k = k.to(torch.float32)
    v = v.to(torch.float32)

    # get seqlens
    L_q, L_k = q.shape[1], k.shape[1]
    
    # Compute attention scores
    attention_scores = torch.matmul(q, k.transpose(-2, -1))
    if DEBUG_CORE:
        print("attention_scores:", attention_scores, attention_scores.shape)

    # Scale scores
    attention_scaled_scores = sm_scale * attention_scores
    if DEBUG_CORE:
        print("attention_scaled_scores:", attention_scaled_scores, attention_scaled_scores.shape)

    # Apply ALiBi if slopes are provided
    if alibi_slopes is not None:
        if cache_seqlens is not None:
            # DECODE MODE: Special ALiBi handling
            # In decode mode, k has shape [nheads, max_cache_len, head_dim]
            # but only cache_seqlens positions are valid
            
            # The test's attn_bias_from_alibi_slopes uses this formula:
            # relative_pos = torch.abs(row_idx + sk - sq - col_idx)
            # where sk = actual valid key length, sq = query length
            
            row_idx = torch.arange(L_q, device=q.device, dtype=torch.float32).unsqueeze(1)
            col_idx = torch.arange(L_k, device=q.device, dtype=torch.float32).unsqueeze(0)
            
            # Compute relative positions
            # cache_seqlens is the actual number of valid keys (sk in the test)
            # L_q is the query sequence length (sq in the test)
            relative_pos = torch.abs(row_idx + cache_seqlens - L_q - col_idx)
            
            # Apply slopes
            if alibi_slopes.dim() == 1:
                # Shape: [nheads] -> [nheads, 1, 1]
                alibi_slopes_expanded = alibi_slopes.view(-1, 1, 1)
            else:
                # Already has batch dimension
                alibi_slopes_expanded = alibi_slopes
            
            alibi_bias = -alibi_slopes_expanded * relative_pos
            
            if DEBUG_CORE:
                print(f"Decode ALiBi: cache_seqlens={cache_seqlens}, L_q={L_q}, L_k={L_k}")
                print(f"relative_pos shape: {relative_pos.shape}")
                print(f"alibi_bias shape: {alibi_bias.shape}")
        else:
            if DEBUG_CORE:
                print("alibi_slopes:", alibi_slopes, alibi_slopes.shape)
            alibi_bias = compute_alibi_tensor_ref(alibi_slopes, L_q, L_k)
            if DEBUG_CORE:
                print("alibi_bias:", alibi_bias, alibi_bias.shape)
            alibi_bias = alibi_bias.reshape(-1, L_q, L_k)
            if DEBUG_CORE:
                print("alibi_bias_flat:", alibi_bias, alibi_bias.shape)

        attention_scaled_scores = attention_scaled_scores + alibi_bias
        if DEBUG_CORE:
            print("attention_scaled_scores after alibi:", attention_scaled_scores, attention_scaled_scores.shape)

    # Apply masks
    row_idx = torch.arange(L_q, device=q.device).unsqueeze(1)
    col_idx = torch.arange(L_k, device=q.device).unsqueeze(0)
    
    if cache_seqlens is not None:
        # We're in decode mode with a KV cache
        # k and v are full allocated size, but only cache_seqlens positions are valid
        
        # Create a mask for valid cache positions
        cache_mask = col_idx < cache_seqlens
        
        # Use cache_seqlens for offset calculation to match test's construct_local_mask
        # which uses key_padding_mask.sum() as the sequence length
        col_offset = cache_seqlens - L_q
        
        if DEBUG_CORE:
            print(f"Cache mode: valid_len={cache_seqlens}, L_k={L_k}")
            print(f"Using col_offset={col_offset} based on valid cache length")
    else:
        # Calculate offset for when seqlen_q != seqlen_k
        # This offset aligns query positions to key positions
        # When L_q < L_k, offset is positive, meaning query i maps to key position (i + offset)
        # This is consistent with construct_local_mask in the tests which uses (sk - sq)
        col_offset = L_k - L_q
        cache_mask = None

    mask_applied = False
    if causal and (window_size_left, window_size_right) == (-1, -1):
        # Pure causal: ensure query doesn't attend to future keys
        # With offset, query i can attend to keys up to position (i + col_offset)
        mask = row_idx >= (col_idx - col_offset)
        mask_applied = True
        if DEBUG_CORE:
            print("causal_mask:", mask)
    elif (window_size_left, window_size_right) != (-1, -1):
        # Handle the case where window sizes exceed sequence length
        if window_size_left >= L_k:
            window_size_left = -1  # No left limit
        if window_size_right >= L_k:
            window_size_right = -1  # No right limit
        
        if causal:
            # Causal + sliding window: ensure we don't attend to future
            window_size_right = min(window_size_right, 0) if window_size_right != -1 else 0
        
        # Create sliding window mask
        # Each query at position i attends to keys in [i + offset - left, i + offset + right]
        if window_size_left == -1 and window_size_right == -1:
            # No window restriction
            mask = torch.ones((L_q, L_k), dtype=torch.bool, device=q.device)
        else:
            mask = torch.ones((L_q, L_k), dtype=torch.bool, device=q.device)
            if window_size_left != -1:
                # Each query at position i attends to keys from position (i - left) accounting for offset
                mask = mask & (col_idx >= (row_idx + col_offset - window_size_left))
            if window_size_right != -1:
                # Each query at position i attends to keys up to position (i + right) accounting for offset
                mask = mask & (col_idx <= (row_idx + col_offset + window_size_right))
        
        # Apply causal constraint
        if causal:
            causal_mask = row_idx >= (col_idx - col_offset)
            mask = mask & causal_mask
        
        mask_applied = True
        if DEBUG_CORE:
            print(f"sliding_window_mask (left={window_size_left}, right={window_size_right}):", mask)
        
    # Apply cache mask if needed
    if cache_mask is not None:
        if mask_applied:
            mask = mask & cache_mask
        else:
            mask = cache_mask
            mask_applied = True
    
    # Apply the mask if created
    if mask_applied:
        attention_scaled_scores = attention_scaled_scores.masked_fill(
            torch.logical_not(mask.unsqueeze(0)), float('-inf')
        )
        if DEBUG_CORE:
            print("attention_scaled_scores after masking:", attention_scaled_scores, attention_scaled_scores.shape)

    # Compute max for numerical stability
    max_scores = torch.max(attention_scaled_scores, dim=-1, keepdim=True)[0]
    if DEBUG_CORE:
        print("max_scores:", max_scores, max_scores.shape)
    if mask_applied:
        # Replace -inf in max_scores with zeros to avoid NaN in subtraction
        max_scores = torch.where(
            torch.isinf(max_scores), torch.zeros_like(max_scores), max_scores
        )
        if DEBUG_CORE:
            print("max_scores after mask handling:", max_scores, max_scores.shape)

    # Shift scores
    attention_shifted_scaled_scores = attention_scaled_scores - max_scores
    if DEBUG_CORE:
            print("attention_shifted_scaled_scores:", attention_shifted_scaled_scores, attention_shifted_scaled_scores.shape)

    # Exponentiate
    if use_exp2:
        RCP_LN = 1 / math.log(2)
        exp_scores = torch.exp2(RCP_LN * attention_shifted_scaled_scores)
    else:
        exp_scores = torch.exp(attention_shifted_scaled_scores)

    if DEBUG_CORE:
        print("exp_scores:", exp_scores, exp_scores.shape)

    # Sum of exponentials
    sum_exp_scores = torch.sum(exp_scores, dim=-1, keepdim=True)
    if DEBUG_CORE:
        print("sum_exp_scores:", sum_exp_scores, sum_exp_scores.shape)
    if mask_applied:
        # if sum of exp scores is 0.0 it means scores where -inf, we cannot compute softmax and softmax_lse. Setting to 1 deals with -inf case cleanly 
        sum_exp_scores = torch.where(
        sum_exp_scores == 0,
        torch.ones_like(sum_exp_scores),
        sum_exp_scores
        )
    if DEBUG_CORE:
        print("sum_exp_scores:", sum_exp_scores, sum_exp_scores.shape)

    # Compute softmax probabilities
    p = exp_scores / sum_exp_scores

    if DEBUG_CORE:
        print("softmax:", p, p.shape)
        
    # apply dropout if specified
    if dropout_p > 0.0:
        rand_vals = torch.rand(p.shape, generator=torch.Generator(device=p.device).manual_seed(philox_seed), device=p.device, dtype=p.dtype)
        dropout_mask, dropout_scale = rand_vals > dropout_p,  (1.0 / (1 - dropout_p))
        if DEBUG_CORE:
            print("dropout_scale:", dropout_scale)
            print("dropout_mask:", dropout_mask)
        # Apply dropout mask and scale
        # Set -1 for dropped positions and 1 for kept positions in exp_scores 
        sd_mask = torch.where(dropout_mask, exp_scores, -exp_scores)
        p = torch.where(dropout_mask, p , torch.zeros_like(p)) * dropout_scale
        if DEBUG_CORE:
            print("softmax after dropout:", p)
            print("sd_mask:", sd_mask)
    else:
        sd_mask = exp_scores
    
    # Compute log-sum-exp
    if use_exp2:
        LN2 = math.log(2)
        RCP_LN = 1 / math.log(2)
        max_scores_base2 = max_scores * RCP_LN
        softmax_lse_base2 = max_scores_base2 + torch.log2(sum_exp_scores)
        softmax_lse = softmax_lse_base2 * LN2
        softmax_lse.squeeze_(-1)
    else:
        softmax_lse = max_scores + torch.log(sum_exp_scores)
        softmax_lse = softmax_lse.squeeze(-1)

    if DEBUG_CORE:
        print("softmax_lse:", softmax_lse, softmax_lse.shape)

    # Compute output
    o = torch.matmul(p, v)
    if DEBUG_CORE:
        print("o:", o, o.shape)

    # cast back to original dtype
    o = o.to(torch.float16)
    # softmax_lse = softmax_lse.to(torch.float16) # NOTE: if you cast lse to fp16 it cause accuracy issues. keep fp32
    sd_mask = sd_mask.to(torch.float16)

    return o, softmax_lse, sd_mask

def attention_vanilla_forward_pytorch_ref_impl(q, k, v, sm_scale, causal, window_size_left, window_size_right, layout, dropout_p, philox_seed, philox_offset, alibi_slopes, use_exp2):
    """Compute reference output and softmax_lse using PyTorch's built-in function"""

    # Ensure the layout is 'bhsd'
    if layout == "bshd":
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
    elif layout != "bhsd":
        raise ValueError(f"Unknown layout {layout}")

    # Prepare tensors
    batch_size, nheads_q, seq_len_q, head_dim = q.shape
    batch_size, nheads_k, seq_len_k, head_dim = k.shape
    group_size = nheads_q // nheads_k
    if nheads_q % nheads_k != 0:
        raise ValueError("nheads_q must be divisible by nheads_k")

    if group_size != 1:
        # MQA or GQA case
        # Reshape q to [batch_size, nheads_k, group_size, seq_len_q, head_dim]
        q = q.reshape(batch_size, nheads_k, group_size, seq_len_q, head_dim)
        # Expand k and v to match group_size
        k = k.unsqueeze(2).expand(-1, -1, group_size, -1, -1)
        v = v.unsqueeze(2).expand(-1, -1, group_size, -1, -1)
        # Flatten the first three dimensions for computation
        q = q.reshape(batch_size * nheads_k * group_size, seq_len_q, head_dim)
        k = k.reshape(batch_size * nheads_k * group_size, seq_len_k, head_dim)
        v = v.reshape(batch_size * nheads_k * group_size, seq_len_k, head_dim)
    else:
        q = q.reshape(batch_size * nheads_q, seq_len_q, head_dim)
        k = k.reshape(batch_size * nheads_k, seq_len_k, head_dim)
        v = v.reshape(batch_size * nheads_k, seq_len_k, head_dim)

    # Call the core attention function
    o, softmax_lse, sd_mask = attention_forward_core_ref_impl(
        q, k, v, sm_scale, causal, window_size_left, window_size_right, dropout_p, philox_seed, philox_offset, alibi_slopes, use_exp2
    )

    if group_size != 1:
        # Reshape outputs back to original dimensions
        o = o.reshape(batch_size, nheads_k, group_size, seq_len_q, head_dim)
        o = o.reshape(batch_size, nheads_q, seq_len_q, head_dim)
        softmax_lse = softmax_lse.reshape(batch_size, nheads_k, group_size, seq_len_q)
        softmax_lse = softmax_lse.reshape(batch_size, nheads_q, seq_len_q)
        sd_mask = sd_mask.reshape(batch_size, nheads_k, group_size, seq_len_q, seq_len_k)
        sd_mask = sd_mask.reshape(batch_size, nheads_q, seq_len_q, seq_len_k)
    else:
        # Standard case
        o = o.reshape(batch_size, nheads_q, seq_len_q, head_dim)
        softmax_lse = softmax_lse.reshape(batch_size, nheads_q, seq_len_q)
        sd_mask = sd_mask.reshape(batch_size, nheads_q, seq_len_q, seq_len_k)

    # Restore original layout if necessary
    if layout == "bshd":
        o = o.transpose(1, 2)

    return o, softmax_lse, sd_mask


def attention_varlen_forward_pytorch_ref_impl(
    q,
    k,
    v,
    sm_scale,
    causal,
    window_size_left,
    window_size_right,
    layout,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p, 
    philox_seed, 
    philox_offset,
    alibi_slopes,
    use_exp2
):
    # Ensure the layout is 'thd'
    if layout != 'thd':
        raise ValueError(f"Unsupported layout {layout}. Expected 'thd'.")

    batch_size = cu_seqlens_q.shape[0] - 1
    nheads_q, nheads_k = q.shape[1], k.shape[1]
    head_dim = q.shape[2]

    # Pre-allocate outputs
    total_L_q = q.shape[0]
    total_L_k = k.shape[0]

    o = torch.zeros((total_L_q, nheads_q, head_dim), dtype=q.dtype, device=q.device)
    softmax_lse = torch.zeros((nheads_q, total_L_q), dtype=torch.float32, device=q.device)
    sd_mask = torch.zeros((batch_size, nheads_q, max_seqlen_q, max_seqlen_k), dtype=torch.float32, device=q.device)

    # Compute group_size for MQA/GQA handling
    group_size = nheads_q // nheads_k
    if nheads_q % nheads_k != 0:
        raise ValueError("nheads_q must be divisible by nheads_k")

    for i in range(batch_size):
        # Get the start and end indices for the current sequence
        start_q = cu_seqlens_q[i].item()
        end_q = cu_seqlens_q[i + 1].item()
        start_k = cu_seqlens_k[i].item()
        end_k = cu_seqlens_k[i + 1].item()

        seqlen_q = end_q - start_q
        seqlen_k = end_k - start_k

        if DEBUG:
            print(f"Batch {i} with seqlen_q = {seqlen_q}, seqlen_k = {seqlen_k}, Hq= {nheads_q}, Hk = {nheads_k}")

        # Extract q_i, k_i, v_i
        q_i = q[start_q:end_q, :, :]  # [L_q_i, nheads_q, head_dim]
        k_i = k[start_k:end_k, :, :]  # [L_k_i, nheads_k, head_dim]
        v_i = v[start_k:end_k, :, :]  # [L_k_i, nheads_k, head_dim]

        # Permute to [nheads, L_q_i, head_dim]
        q_i = q_i.permute(1, 0, 2)
        k_i = k_i.permute(1, 0, 2)
        v_i = v_i.permute(1, 0, 2)

        # Handle MQA/GQA by adjusting shapes based on group_size
        if group_size != 1:
            # Reshape q_i to [nheads_k, group_size, L_q_i, head_dim]
            q_i = q_i.reshape(nheads_k, group_size, seqlen_q, head_dim)
            # Expand k_i and v_i to match group_size
            k_i = k_i.unsqueeze(1).expand(-1, group_size, -1, -1)
            v_i = v_i.unsqueeze(1).expand(-1, group_size, -1, -1)
            # Flatten the first two dimensions for computation
            q_i = q_i.reshape(nheads_k * group_size, seqlen_q, head_dim)
            k_i = k_i.reshape(nheads_k * group_size, seqlen_k, head_dim)
            v_i = v_i.reshape(nheads_k * group_size, seqlen_k, head_dim)
        else:
            # Standard case
            q_i = q_i.reshape(nheads_q, seqlen_q, head_dim)
            k_i = k_i.reshape(nheads_k, seqlen_k, head_dim)
            v_i = v_i.reshape(nheads_k, seqlen_k, head_dim)

        if alibi_slopes is not None:
            alibi_slopes_i = alibi_slopes[i]
        else:
            alibi_slopes_i = None

        # Call the core attention function for this sequence
        o_i, softmax_lse_i, sd_mask_i = attention_forward_core_ref_impl(q_i, k_i, v_i, sm_scale, causal, window_size_left, window_size_right, dropout_p, philox_seed, philox_offset, alibi_slopes_i, use_exp2)

        # Reshape outputs back to original dimensions
        if group_size != 1:
            # Reshape outputs to [nheads_k, group_size, seqlen_q, head_dim]
            o_i = o_i.reshape(nheads_k, group_size, seqlen_q, head_dim)
            # Combine the first two dimensions back to nheads_q
            o_i = o_i.reshape(nheads_q, seqlen_q, head_dim)
            # Reshape softmax_lse_i similarly
            softmax_lse_i = softmax_lse_i.reshape(nheads_k, group_size, seqlen_q)
            softmax_lse_i = softmax_lse_i.reshape(nheads_q, seqlen_q)
        else:
            # Outputs are already in the correct shape
            pass

        # Convert back to 'thd' layout
        o_i = o_i.permute(1, 0, 2)  # [L_q_i, nheads_q, head_dim]
        sd_mask_i = sd_mask_i # [nheads_q, L_q_i, L_k_i]

        # Place outputs in pre-allocated tensors
        o[start_q:end_q, :, :] = o_i
        softmax_lse[:, start_q:end_q] = softmax_lse_i
        sd_mask[i, :, :seqlen_q, :seqlen_k] = sd_mask_i

    return o, softmax_lse, sd_mask

def attention_forward_pytorch_ref_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    sm_scale: float,
    alibi_slopes: Optional[torch.Tensor],
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    layout: Literal["bshd", "bhsd", "thd"],
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    philox_seed: Optional[int],
    philox_offset: Optional[int],
    use_exp2: bool
):
    # compute reference
    if layout == "thd":
        o_ref, softmax_lse_ref, sd_mask_ref = attention_varlen_forward_pytorch_ref_impl(
            q.clone(), 
            k.clone(), 
            v.clone(), 
            sm_scale, 
            causal,
            window_size_left,
            window_size_right,
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
        o_ref, softmax_lse_ref, sd_mask_ref = attention_vanilla_forward_pytorch_ref_impl(
                                                       q.clone(),
                                                       k.clone(),
                                                       v.clone(),
                                                       sm_scale,
                                                       causal,
                                                       window_size_left,
                                                        window_size_right,
                                                       layout,
                                                       dropout_p,
                                                       philox_seed,
                                                       philox_offset,
                                                       alibi_slopes,
                                                       use_exp2)

    # copy back to ouput tensor
    out.copy_(o_ref.to(out.dtype))
    
    return softmax_lse_ref, sd_mask_ref

def attention_decode_forward_ref_impl(
        q: torch.Tensor, 
        k_cache: torch.Tensor, 
        v_cache: torch.Tensor,
        k_new: Optional[torch.Tensor],
        v_new: Optional[torch.Tensor],
        out: torch.Tensor,
        sm_scale: float, 
        causal: bool,
        window_size_left: int, 
        window_size_right: int,
        alibi_slopes: Optional[torch.Tensor], 
        layout: Literal["bshd"], 
        cache_seqlens: Optional[torch.Tensor], 
        cache_batch_idx: Optional[torch.Tensor],
):
    """Compute reference output for decode attention using PyTorch's built-in functions"""
    
    # get batch size before any layout conversion
    batch_size = q.shape[0]
    
    # handle cache_batch_idx
    if cache_batch_idx is not None:
        # remap batch indices for cache access
        batch_indices = cache_batch_idx
    else:
        batch_indices = torch.arange(batch_size, device=q.device)
    
    # copy new keys and values into cache if provided (before any layout conversion)
    if k_new is not None and v_new is not None:
        _, seq_len_new, _, _ = k_new.shape  # shape is [batch, seq_len, nheads, head_dim] for bshd layout
        
        for b in range(batch_size):
            cache_idx = batch_indices[b].item() if torch.is_tensor(batch_indices) else batch_indices
            
            # determine where to place new k/v in cache
            if cache_seqlens is not None:
                if torch.is_tensor(cache_seqlens):
                    start_pos = cache_seqlens[b].item()
                else:
                    start_pos = cache_seqlens
            else:
                # if no cache_seqlens, assume we're filling from the beginning
                start_pos = 0
            
            end_pos = start_pos + seq_len_new
            
            # copy new keys and values into cache (both are in bshd layout)
            k_cache[cache_idx, start_pos:end_pos, :, :] = k_new[b, :, :, :]
            v_cache[cache_idx, start_pos:end_pos, :, :] = v_new[b, :, :, :]
    
    # ensure the layout is 'bhsd'
    if layout == "bshd":
        q = q.transpose(1, 2).contiguous()
        k_cache = k_cache.transpose(1, 2).contiguous()
        v_cache = v_cache.transpose(1, 2).contiguous()
    elif layout != "bhsd":
        raise ValueError(f"Unknown layout {layout}")
    
    # prepare tensors
    batch_size_q, nheads_q, seq_len_q, head_dim = q.shape
    batch_size_cache, nheads_k, max_cache_len, head_dim_k = k_cache.shape
    _, nheads_v, _, head_dim_v = v_cache.shape
    
    # validate dimensions
    assert head_dim == head_dim_k == head_dim_v, f"Head dimensions must match: {head_dim}, {head_dim_k}, {head_dim_v}"
    
    # handle MQA/GQA
    group_size = nheads_q // nheads_k
    if nheads_q % nheads_k != 0:
        raise ValueError("nheads_q must be divisible by nheads_k")
    
    # handle cache_batch_idx
    if cache_batch_idx is not None:
        # remap batch indices for cache access
        batch_indices = cache_batch_idx
    else:
        batch_indices = torch.arange(batch_size, device=q.device)
    
    # prepare outputs
    o = torch.zeros_like(q)
    softmax_lse = torch.zeros((batch_size, nheads_q, seq_len_q), dtype=torch.float32, device=q.device)
    
    # process each batch element
    for b in range(batch_size):
        cache_idx = batch_indices[b].item() if torch.is_tensor(batch_indices) else batch_indices
        
        # determine valid cache length for this batch element
        if cache_seqlens is not None:
            if torch.is_tensor(cache_seqlens):
                cache_len = cache_seqlens[b].item()
                if k_new is not None:
                    _, seq_len_new, _, _ = k_new.shape
                    cache_len += seq_len_new
            else:
                cache_len = cache_seqlens
                if k_new is not None:
                    _, seq_len_new, _, _ = k_new.shape
                    cache_len += seq_len_new
        else:
            cache_len = max_cache_len
        
        # CHANGE: Extract the full cache, not just valid portion
        # This matches what the test does - it uses full k_cache_rep/v_cache_rep
        k_b = k_cache[cache_idx, :, :, :]  # [nheads_k, max_cache_len, head_dim]
        v_b = v_cache[cache_idx, :, :, :]  # [nheads_v, max_cache_len, head_dim]
        q_b = q[b:b+1, :, :, :]  # [1, nheads_q, seq_len_q, head_dim]
        
        # handle MQA/GQA by expanding k and v
        if group_size != 1:
            # expand k and v to match q's number of heads
            k_b = k_b.unsqueeze(1).expand(-1, group_size, -1, -1)
            k_b = k_b.reshape(nheads_q, max_cache_len, head_dim)
            
            v_b = v_b.unsqueeze(1).expand(-1, group_size, -1, -1)
            v_b = v_b.reshape(nheads_q, max_cache_len, head_dim)
        
        # reshape for attention_forward_core_ref_impl
        q_b = q_b.reshape(nheads_q, seq_len_q, head_dim)
        
        # handle alibi slopes for this batch
        alibi_slopes_b = None
        if alibi_slopes is not None:
            if alibi_slopes.dim() == 2:
                alibi_slopes_b = alibi_slopes[b]
            else:
                alibi_slopes_b = alibi_slopes
        
        # call core attention function with cache information
        o_b, softmax_lse_b, _ = attention_forward_core_ref_impl(
            q_b, k_b, v_b, sm_scale, causal, window_size_left, window_size_right,
            dropout_p=0.0, philox_seed=None, philox_offset=None, 
            alibi_slopes=alibi_slopes_b, use_exp2=True,
            cache_seqlens=cache_len,      # Pass valid cache length
        )
        
        # store outputs
        o[b, :, :, :] = o_b.reshape(nheads_q, seq_len_q, head_dim)
        softmax_lse[b, :, :] = softmax_lse_b.reshape(nheads_q, seq_len_q)
    
    # restore original layout if necessary
    if layout == "bshd":
        o = o.transpose(1, 2)
    
    # copy output to the provided tensor
    out.copy_(o.to(out.dtype))
    
    return softmax_lse