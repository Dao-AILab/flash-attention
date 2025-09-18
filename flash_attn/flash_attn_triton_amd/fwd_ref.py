import torch
import math
from typing import Literal, Optional, Union
from .utils import compute_alibi_tensor_ref

DEBUG = False
DEBUG_CORE = False

def attention_forward_core_ref_impl(
    q, k, v, sm_scale, causal, window_size_left, window_size_right, 
    dropout_p, philox_seed, philox_offset, alibi_slopes, use_exp2,
    cache_seqlens=None, block_table=None, paged_kv_block_size=None
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
        print("block_table:", block_table)
        print("paged_kv_block_size:", paged_kv_block_size)

    # cast to float32
    q = q.to(torch.float32)
    
    # Check if we're in paged KV mode
    is_paged = block_table is not None and paged_kv_block_size is not None
    
    if False:  # Debug paged attention (disabled for production)
        print(f"\n=== attention_forward_core_ref_impl DEBUG ===")
        print(f"is_paged: {is_paged}")
        print(f"block_table: {block_table.shape if block_table is not None else None}")
        print(f"paged_kv_block_size: {paged_kv_block_size}")
        if is_paged:
            print(f"k shape (paged): {k.shape}")
            print(f"v shape (paged): {v.shape}")
        print(f"cache_seqlens: {cache_seqlens}")
    
    if is_paged:
        # In paged mode, k and v are [num_blocks, block_size, nheads_k, head_dim]
        # We'll compute attention on-the-fly without reconstructing
        nheads_q = q.shape[0]
        L_q = q.shape[1]
        head_dim = q.shape[2]
        
        # Get number of KV heads from the cache
        nheads_k = k.shape[2]  # k shape: [num_blocks, block_size, nheads_k, head_dim]
        
        # Handle MQA/GQA
        assert nheads_q % nheads_k == 0, f"nheads_q ({nheads_q}) must be divisible by nheads_k ({nheads_k})"
        group_size = nheads_q // nheads_k
        
        # Determine the actual KV sequence length from cache_seqlens
        L_k = cache_seqlens if isinstance(cache_seqlens, int) else cache_seqlens.item()
        
        if False:  # Debug disabled
            print(f"L_q: {L_q}, L_k: {L_k}, nheads_q: {nheads_q}, nheads_k: {nheads_k}, group_size: {group_size}, head_dim: {head_dim}")
            print(f"block_table contents: {block_table if block_table is not None else 'None'}")
        
        # Initialize attention scores
        attention_scores = torch.zeros((nheads_q, L_q, L_k), dtype=torch.float32, device=q.device)
        
        # Compute attention scores on-the-fly by accessing blocks directly
        for kv_pos in range(L_k):
            # Calculate which block and position within block
            block_idx = kv_pos // paged_kv_block_size
            within_block_idx = kv_pos % paged_kv_block_size
            
            # Get the physical block number from block_table
            # block_table is [1, num_blocks] for single batch in core function
            if block_table.dim() == 2:
                physical_block = block_table[0, block_idx].item()
            else:
                physical_block = block_table[block_idx].item()
            
            # Debug output disabled
            # if kv_pos == 0:
            #     print(f"First KV access: block_idx={block_idx}, within_block={within_block_idx}, physical_block={physical_block}")
            #     print(f"k_vec shape will be: {k[physical_block, within_block_idx, :, :].shape}")
            
            # Access k values directly from paged cache
            # k shape: [num_blocks, block_size, nheads_k, head_dim]
            k_vec = k[physical_block, within_block_idx, :, :].to(torch.float32)  # [nheads_k, head_dim]
            
            # For GQA/MQA, we need to repeat k_vec for each group
            if group_size > 1:
                # Expand k_vec to match query heads
                # k_vec: [nheads_k, head_dim] -> [nheads_q, head_dim]
                k_vec = k_vec.repeat_interleave(group_size, dim=0)
            
            # Compute dot product with all query positions
            # q is [nheads_q, L_q, head_dim], k_vec is [nheads_q, head_dim]
            # Result should be [nheads_q, L_q] for this kv_pos
            attention_scores[:, :, kv_pos] = torch.sum(q * k_vec.unsqueeze(1), dim=-1)
        
        # Keep k and v in original format for later v computation
        k_paged = k
        v_paged = v
        
        # Debug output disabled
        # print(f"attention_scores computed shape: {attention_scores.shape}")
        # print(f"attention_scores sample values: {attention_scores[0, 0, :5]}")
    else:
        # Standard non-paged mode
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
    if is_paged:
        # Compute output on-the-fly using paged v cache
        nheads_q = p.shape[0]
        L_q = p.shape[1]
        nheads_v = v_paged.shape[2]  # [num_blocks, block_size, nheads_v, head_dim]
        head_dim = v_paged.shape[3]
        
        # Handle MQA/GQA for v
        assert nheads_q % nheads_v == 0, f"nheads_q ({nheads_q}) must be divisible by nheads_v ({nheads_v})"
        v_group_size = nheads_q // nheads_v
        
        o = torch.zeros((nheads_q, L_q, head_dim), dtype=torch.float32, device=p.device)
        
        # Accumulate weighted v values
        for kv_pos in range(L_k):
            # Calculate which block and position within block
            block_idx = kv_pos // paged_kv_block_size
            within_block_idx = kv_pos % paged_kv_block_size
            
            # Get the physical block number from block_table
            if block_table.dim() == 2:
                physical_block = block_table[0, block_idx].item()
            else:
                physical_block = block_table[block_idx].item()
            
            # Access v values directly from paged cache
            # v_paged shape: [num_blocks, block_size, nheads_v, head_dim]
            v_vec = v_paged[physical_block, within_block_idx, :, :].to(torch.float32)  # [nheads_v, head_dim]
            
            # For GQA/MQA, we need to repeat v_vec for each group
            if v_group_size > 1:
                # Expand v_vec to match query heads
                # v_vec: [nheads_v, head_dim] -> [nheads_q, head_dim]
                v_vec = v_vec.repeat_interleave(v_group_size, dim=0)
            
            # Weight by attention probabilities
            # p is [nheads_q, L_q, L_k], we need p[:, :, kv_pos] which is [nheads_q, L_q]
            # v_vec is [nheads_q, head_dim]
            # We want to add p[:, :, kv_pos] * v_vec to each query position
            weights = p[:, :, kv_pos].unsqueeze(-1)  # [nheads_q, L_q, 1]
            o += weights * v_vec.unsqueeze(1)  # [nheads_q, L_q, head_dim]
    else:
        o = torch.matmul(p, v)
    
    # Debug output disabled
    # if False:
    #     print(f"Output o shape: {o.shape}")
    #     print(f"Output o sample values: {o[0, 0, :5]}")
    
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

def attention_prefill_forward_ref_impl(
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
        block_table: Optional[torch.Tensor] = None,
        q_descale: Optional[torch.Tensor] = None,
        k_descale: Optional[torch.Tensor] = None,
        v_descale: Optional[torch.Tensor] = None,
):
    """Compute reference output for decode attention using PyTorch's built-in functions"""
    
    if False:  # Permanently disabled old debug output
        pass
        # print("\n========== attention_decode_forward_ref_impl inputs ==========")
        # print(f"q shape: {q.shape}, dtype: {q.dtype}, device: {q.device}")
        print(f"q values:\n{q}")
        print(f"\nk_cache shape: {k_cache.shape}, dtype: {k_cache.dtype}, device: {k_cache.device}")
        print(f"k_cache values:\n{k_cache}")
        print(f"\nv_cache shape: {v_cache.shape}, dtype: {v_cache.dtype}, device: {v_cache.device}")
        print(f"v_cache values:\n{v_cache}")
        print(f"\nk_new: {k_new.shape if k_new is not None else None}, dtype: {k_new.dtype if k_new is not None else None}")
        if k_new is not None:
            print(f"k_new values:\n{k_new}")
        print(f"\nv_new: {v_new.shape if v_new is not None else None}, dtype: {v_new.dtype if v_new is not None else None}")
        if v_new is not None:
            print(f"v_new values:\n{v_new}")
        print(f"\nout shape: {out.shape}, dtype: {out.dtype}, device: {out.device}")
        print(f"out values:\n{out}")
        print(f"\nsm_scale: {sm_scale}")
        print(f"causal: {causal}")
        print(f"window_size_left: {window_size_left}")
        print(f"window_size_right: {window_size_right}")
        print(f"\nalibi_slopes: {alibi_slopes.shape if alibi_slopes is not None else None}, dtype: {alibi_slopes.dtype if alibi_slopes is not None else None}")
        if alibi_slopes is not None:
            print(f"alibi_slopes values:\n{alibi_slopes}")
        print(f"\nlayout: {layout}")
        print(f"cache_seqlens: {cache_seqlens}")
        if cache_seqlens is not None and torch.is_tensor(cache_seqlens):
            print(f"cache_seqlens values: {cache_seqlens}")
        print(f"cache_batch_idx: {cache_batch_idx}")
        if cache_batch_idx is not None:
            print(f"cache_batch_idx values: {cache_batch_idx}")
        print(f"\nblock_table: {block_table.shape if block_table is not None else None}, dtype: {block_table.dtype if block_table is not None else None}")
        if block_table is not None:
            print(f"block_table values:\n{block_table}")
        print("=" * 60)
    
    # get batch size before any layout conversion
    batch_size = q.shape[0]
    
    # Determine if we're in paged KV mode
    is_paged = block_table is not None
    if is_paged:
        # Infer block size from cache shape
        # k_cache shape for paged: [num_blocks, block_size, nheads, head_dim]
        paged_kv_block_size = k_cache.shape[1]
    else:
        paged_kv_block_size = None
    
    # handle cache_batch_idx
    if cache_batch_idx is not None:
        # remap batch indices for cache access
        batch_indices = cache_batch_idx
    else:
        batch_indices = torch.arange(batch_size, device=q.device)
    
    # copy new keys and values into cache if provided (before any layout conversion)
    if k_new is not None and v_new is not None:
        if is_paged:
            # For paged KV cache, we need to update the blocks with new k/v values
            _, seq_len_new, _, _ = k_new.shape  # shape is [batch, seq_len, nheads, head_dim] for bshd layout
            
            for b in range(batch_size):
                # Determine where to place new k/v in cache
                if cache_seqlens is not None:
                    if torch.is_tensor(cache_seqlens):
                        start_pos = cache_seqlens[b].item()
                    else:
                        start_pos = cache_seqlens
                else:
                    start_pos = 0
                
                # For each new position, find the corresponding block and update it
                for pos_offset in range(seq_len_new):
                    kv_pos = start_pos + pos_offset
                    
                    # Calculate which block and position within block
                    block_idx = kv_pos // paged_kv_block_size
                    within_block_idx = kv_pos % paged_kv_block_size
                    
                    # Get the physical block number from block_table
                    physical_block = block_table[b, block_idx].item()
                    
                    # Update the k and v values in the paged cache
                    # k_cache shape: [num_blocks, block_size, nheads, head_dim]
                    # k_new shape: [batch, seq_len, nheads, head_dim]
                    k_cache[physical_block, within_block_idx, :, :] = k_new[b, pos_offset, :, :]
                    v_cache[physical_block, within_block_idx, :, :] = v_new[b, pos_offset, :, :]
        else:
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
        if not is_paged:
            k_cache = k_cache.transpose(1, 2).contiguous()
            v_cache = v_cache.transpose(1, 2).contiguous()
    elif layout != "bhsd":
        raise ValueError(f"Unknown layout {layout}")
    
    # prepare tensors
    batch_size_q, nheads_q, seq_len_q, head_dim = q.shape
    
    if is_paged:
        # For paged cache: [num_blocks, block_size, nheads, head_dim]
        num_blocks, block_size, nheads_k, head_dim_k = k_cache.shape
        _, _, nheads_v, head_dim_v = v_cache.shape
        max_cache_len = None  # Not directly available in paged mode
        batch_size_cache = None  # Not applicable in paged mode
    else:
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
        if not is_paged:
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
            if is_paged:
                # For paged mode, we need cache_seqlens to know the valid length
                raise ValueError("cache_seqlens must be provided for paged KV cache")
            else:
                cache_len = max_cache_len
        
        if is_paged:
            # For paged KV cache, pass the cache and block table directly
            # Extract block table for this batch element
            block_table_b = block_table[b:b+1, :]  # [1, num_blocks]
            k_b = k_cache  # Pass entire paged cache
            v_b = v_cache  # Pass entire paged cache
            q_b = q[b:b+1, :, :, :]  # [1, nheads_q, seq_len_q, head_dim]
            
            # For paged mode with MQA/GQA, we handle expansion in the core function
            # Just reshape q for now
            q_b = q_b.reshape(nheads_q, seq_len_q, head_dim)
        else:
            # Standard non-paged mode
            k_b = k_cache[cache_idx, :, :, :]  # [nheads_k, max_cache_len, head_dim]
            v_b = v_cache[cache_idx, :, :, :]  # [nheads_v, max_cache_len, head_dim]
            q_b = q[b:b+1, :, :, :]  # [1, nheads_q, seq_len_q, head_dim]
            block_table_b = None
            
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
            block_table=block_table_b,    # Pass block table for paged mode
            paged_kv_block_size=paged_kv_block_size,  # Pass block size for paged mode
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