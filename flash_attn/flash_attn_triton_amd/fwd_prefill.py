import os
import torch
import triton
import triton.language as tl
from typing import Literal, Optional, Union
from .utils import DROPOUT_USE_PYTORCH, DROPOUT_DUMP, AUTOTUNE, compute_alibi_block, compute_fp8_scaling_factors, get_arch, is_cdna, is_fp8, is_rdna, create_dropout_mask, get_fwd_prefill_autotune_configs

DEBUG = False

# NOTE: triton fails to import tl.constexprs so create them here for the file
tl_DROPOUT_USE_PYTORCH: tl.constexpr = triton.language.constexpr(DROPOUT_USE_PYTORCH)
tl_DROPOUT_DUMP: tl.constexpr = triton.language.constexpr(DROPOUT_DUMP)

fwd_prefill_autotune_configs, fwd_prefill_autotune_keys = get_fwd_prefill_autotune_configs()

@triton.jit
def _attn_fwd_no_mask(acc, l_i, m_i, 
                    q, k_base_ptrs, v_base_ptrs, bias_base_ptrs, 
                    stride_kn, stride_vk, stride_bn, stride_sn, 
                    start_m, seqlen_k, seqlen_q, 
                    dropout_p, philox_seed, philox_base_ptrs, 
                    sd_mask_base_ptrs, dropout_mask_base_ptrs,
                    offs_m, offs_n, offs_d,
                    block_min, block_max, alibi_slope,
                    descale_q, descale_k, descale_v, IS_FP8: tl.constexpr, FP8_MAX: tl.constexpr,
                    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
                    PRE_LOAD_V: tl.constexpr,
                    ENABLE_DROPOUT: tl.constexpr, PADDED_HEAD: tl.constexpr,
                    ACTUAL_BLOCK_DMODEL: tl.constexpr, SM_SCALE: tl.constexpr, USE_ALIBI: tl.constexpr, USE_EXP2: tl.constexpr,
                    RETURN_SCORES: tl.constexpr, ACCUMULATOR_TYPE):
    if USE_EXP2:
        RCP_LN2: tl.constexpr = 1.4426950408889634
    
    # loop over k, v, and update accumulator
    for start_n in range(block_min, block_max, BLOCK_N):
        # get ptrs
        k_ptrs = k_base_ptrs + start_n * stride_kn
        v_ptrs = v_base_ptrs + start_n * stride_vk

        kv_offs_n = start_n + tl.arange(0, BLOCK_N)
        if PADDED_HEAD:
            k_mask, k_mask_other = (offs_d[:, None] < ACTUAL_BLOCK_DMODEL), 0.0
            v_mask, v_mask_other = (offs_d[None, :] < ACTUAL_BLOCK_DMODEL), 0.0
            
        # load k and if preload_v then v
        k = tl.load(k_ptrs, mask=k_mask, other=k_mask_other) if PADDED_HEAD else tl.load(k_ptrs)
        if PRE_LOAD_V:
            v = tl.load(v_ptrs, mask=v_mask, other=v_mask_other) if PADDED_HEAD else tl.load(v_ptrs)
        
        # setup qk accumlator
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=ACCUMULATOR_TYPE)

        # -- compute qk ----
        if IS_FP8 :
            qk += (tl.dot(q, k) * descale_q * descale_k)
        else:
            qk += tl.dot(q, k)
        qk_scaled =  qk * SM_SCALE

        if USE_ALIBI:
            # compute the global position of each token within the sequence
            q_offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
            alibi_block = compute_alibi_block(alibi_slope, seqlen_q, seqlen_k, q_offs_m,
                                              kv_offs_n)
            qk_scaled += alibi_block

        # compute qk mask
        qk_mask = (offs_m[:, None] < seqlen_q) & (kv_offs_n[None, :] < seqlen_k)
        
        # compute bias
        if bias_base_ptrs is not None:
            bias_ptrs = bias_base_ptrs + start_n * stride_bn
            bias = tl.load(bias_ptrs, mask=qk_mask, other=0.0)
            qk_scaled += bias

        # get max scores so far
        m_ij = tl.maximum(m_i, tl.max(qk_scaled, 1))

        # scale and subtract max
        q_shifted = tl.where(m_ij[:, None] == float("-inf"), 
                                float("-inf"), 
                                qk_scaled - m_ij[:, None])
        
        # Compute scaled QK and softmax probabilities
        if USE_EXP2:
            p = tl.math.exp2(q_shifted * RCP_LN2)
        else:
            p = tl.math.exp(q_shifted)

        # CAVEAT: Must update l_ij before applying dropout
        l_ij = tl.sum(p, 1)
        if ENABLE_DROPOUT:
            dropout_mask_ptrs = dropout_mask_base_ptrs + start_n * stride_sn
            sd_mask_ptrs = sd_mask_base_ptrs + start_n * stride_sn
            philox_ptrs = philox_base_ptrs + start_n * stride_sn
            if tl_DROPOUT_USE_PYTORCH:
                dropout_mask = tl.load(dropout_mask_ptrs, mask=qk_mask)
            else:
                rng_output = tl.rand(philox_seed, philox_ptrs)  # TODO: use tl.randint for better performance
                dropout_mask = rng_output > dropout_p
                if tl_DROPOUT_DUMP:
                    tl.store(dropout_mask_ptrs, dropout_mask, mask=qk_mask)

            # return scores with negative values for dropped vals
            sd_mask = tl.where(dropout_mask, p, -p)
            tl.store(sd_mask_ptrs, sd_mask, mask=qk_mask)

            # apply dropout mask in place
            p = tl.where(dropout_mask, p, 0.0)
        elif RETURN_SCORES:
            # NOTE: the returned score is not the same as the reference because we need to adjust as we find new maxes per block. We are not doing that
            sd_mask_ptrs = sd_mask_base_ptrs + start_n * stride_sn
            tl.store(sd_mask_ptrs, p, mask=qk_mask)
        
        # -- update output accumulator --
        # alpha is an adjustment factor for acc and li as we loop and find new maxes
        # store the diff in maxes to adjust acc and li as we discover new maxes
        m_diff = tl.where(m_ij == float("-inf"), 
                                float("-inf"), 
                                m_i - m_ij)
        if USE_EXP2:
            alpha = tl.math.exp2(m_diff * RCP_LN2)
        else:
            alpha = tl.math.exp(m_diff)
        acc = acc * alpha[:, None]
        if not PRE_LOAD_V:
            v = tl.load(v_ptrs, mask=v_mask, other=v_mask_other) if PADDED_HEAD else tl.load(v_ptrs)
        
        # -- update m_i and l_i
        l_i = l_i * alpha + l_ij
        m_i = m_ij

        if IS_FP8:
            scale_p, descale_p = compute_fp8_scaling_factors(p, FP8_MAX)
            acc += (tl.dot((p * scale_p).to(v.type.element_ty), v) * descale_p * descale_v)
        else:
            acc += tl.dot(p.to(v.type.element_ty), v)
    
    return acc, l_i, m_i

@triton.jit
def _attn_fwd_mask(acc, l_i, m_i, 
                    q, k_base_ptrs, v_base_ptrs, bias_base_ptrs, 
                    stride_kn, stride_vk, stride_bn, stride_sn, start_m,
                    seqlen_k, seqlen_q, 
                    dropout_p, philox_seed, philox_base_ptrs, 
                    sd_mask_base_ptrs, dropout_mask_base_ptrs,
                    offs_m, offs_n, offs_d,
                    block_min, block_max, n_extra_tokens, alibi_slope,
                    descale_q, descale_k, descale_v, IS_FP8: tl.constexpr, FP8_MAX: tl.constexpr,
                    IS_CAUSAL: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
                    PRE_LOAD_V: tl.constexpr,
                    ENABLE_DROPOUT: tl.constexpr, PADDED_HEAD: tl.constexpr,
                    ACTUAL_BLOCK_DMODEL: tl.constexpr, SM_SCALE: tl.constexpr, USE_ALIBI: tl.constexpr, USE_EXP2: tl.constexpr,
                    RETURN_SCORES: tl.constexpr, 
                    USE_SLIDING_WINDOW: tl.constexpr, WINDOW_SIZE_LEFT: tl.constexpr, WINDOW_SIZE_RIGHT: tl.constexpr,
                    ACCUMULATOR_TYPE):
    if USE_EXP2:
        RCP_LN2: tl.constexpr = 1.4426950408889634

    # seqlen diff
    seqlen_delta_qk = seqlen_k - seqlen_q
    
    # loop over k, v, and update accumulator
    for start_n in range(block_min, block_max, BLOCK_N):
        # get ptrs
        k_ptrs = k_base_ptrs + start_n * stride_kn
        v_ptrs = v_base_ptrs + start_n * stride_vk

        # For padded blocks, we will overrun the tensor size if
        # we load all BLOCK_N. For others, the blocks are all within range.
        kv_offs_n = start_n + tl.arange(0, BLOCK_N)
        k_mask = (kv_offs_n[None, :] < seqlen_k)
        v_mask = (kv_offs_n[:, None] < seqlen_k)
        if PADDED_HEAD:
            k_mask = k_mask & (offs_d[:, None] < ACTUAL_BLOCK_DMODEL)
            v_mask = v_mask & (offs_d[None, :] < ACTUAL_BLOCK_DMODEL)
        
        # load k and if preload_v then v
        k = tl.load(k_ptrs, mask=k_mask, other = 0.0)
        if PRE_LOAD_V:
            v = tl.load(v_ptrs, mask=v_mask, other=0.0)
        
        # setup qk accumlator
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=ACCUMULATOR_TYPE)
        
        # We start from end of seqlen_k so only the first iteration would need
        # to be checked for padding if it is not a multiple of block_n
        # TODO: This can be optimized to only be true for the padded block.
        # If this is the last block / iteration, we want to
        # mask if the sequence length is not a multiple of block size
        # a solution is to always do BLOCK_M // BLOCK_N + 1 steps if not is_modulo_mn.
        # last step might get wasted but that is okay. check if this masking works For
        # that case.
        if (n_extra_tokens != 0) and (start_n + BLOCK_N == block_max):
            boundary_m = tl.full([BLOCK_M], seqlen_k, dtype=tl.int32)
            size_n = start_n + offs_n[None, :]
            mask = size_n < boundary_m[:, None]
            qk = tl.where(mask, qk, float("-inf"))

        # -- compute qk ----
        if IS_FP8 :
            qk += (tl.dot(q, k) * descale_q * descale_k)
        else:
            qk += tl.dot(q, k)
        qk_scaled =  qk * SM_SCALE

        if USE_ALIBI:
            # compute the global position of each token within the sequence
            q_offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
            alibi_block = compute_alibi_block(alibi_slope, seqlen_q, seqlen_k, q_offs_m,
                                              kv_offs_n)
            qk_scaled += alibi_block

        if USE_SLIDING_WINDOW:
            if IS_CAUSAL:
                # ========== CAUSAL SLIDING WINDOW MASKING ==========
                # For causal sliding window, we need to apply both constraints:
                # 1. Causal: col_idx <= row_idx + (seqlen_k - seqlen_q)
                # 2. Sliding window: row_idx - window_left <= col_idx <= row_idx + window_right
                
                # Get positions
                row_idx = offs_m  # Query positions
                col_idx = kv_offs_n  # Key positions
                
                # Expand for broadcasting
                row_idx_expanded = row_idx[:, None]  # [BLOCK_M, 1]
                col_idx_expanded = col_idx[None, :]  # [1, BLOCK_N]
                
                # Apply causal constraint: can only attend to positions before or at the diagonal
                causal_offset = seqlen_k - seqlen_q
                causal_mask = col_idx_expanded > (row_idx_expanded + causal_offset)
                
                # Apply sliding window constraint
                if WINDOW_SIZE_LEFT < 0:
                    # Only right window constraint
                    window_mask = col_idx_expanded > (row_idx_expanded + causal_offset + WINDOW_SIZE_RIGHT)
                else:
                    # Both left and right window constraints
                    # Adjust window bounds by causal offset
                    left_bound = row_idx_expanded + causal_offset - WINDOW_SIZE_LEFT
                    right_bound = row_idx_expanded + causal_offset + WINDOW_SIZE_RIGHT
                    
                    # Can't attend to positions outside the window
                    window_mask = (col_idx_expanded < left_bound) | (col_idx_expanded > right_bound)
                
                # Final mask is the union of both constraints (True = cannot attend)
                mask = causal_mask | window_mask
                
                # Apply mask
                qk_scaled = tl.where(mask, float("-inf"), qk_scaled)
            else:
                # ========== NON-CAUSAL SLIDING WINDOW MASKING ==========
                # Exactly matching reference construct_local_mask:
                # row_idx = query positions, col_idx = key positions
                # sk = seqlen_k, sq = seqlen_q
                
                # Get positions  
                row_idx = offs_m  # Query positions
                col_idx = kv_offs_n  # Key positions
                
                # sk and sq from reference (no padding masks in this test)
                sk = seqlen_k
                sq = seqlen_q
                
                # Expand for broadcasting
                row_idx_expanded = row_idx[:, None]  # [BLOCK_M, 1]
                col_idx_expanded = col_idx[None, :]  # [1, BLOCK_N]
                
                # Reference logic for mask computation
                if WINDOW_SIZE_LEFT < 0:
                    # Reference: return col_idx > row_idx + sk - sq + window_size[1]
                    mask = col_idx_expanded > (row_idx_expanded + sk - sq + WINDOW_SIZE_RIGHT)
                else:
                    # Reference: 
                    # sk = torch.full_like(col_idx, seqlen_k) if key_padding_mask is None else sk
                    # return torch.logical_or(
                    #     col_idx > torch.minimum(row_idx + sk - sq + window_size[1], sk),
                    #     col_idx < row_idx + sk - sq - window_size[0],
                    # )
                    # Create sk tensor with proper shape for broadcasting
                    # sk represents the key sequence length, which should be compared per column
                    sk_full = tl.full((1, BLOCK_N), sk, dtype=tl.int32)
                    
                    # Compute boundaries
                    right_bound_val = row_idx_expanded + sk - sq + WINDOW_SIZE_RIGHT
                    right_bound = tl.minimum(right_bound_val, sk_full)
                    left_bound = row_idx_expanded + sk - sq - WINDOW_SIZE_LEFT
                    
                    # Mask where True = cannot attend (matching reference)
                    mask = (col_idx_expanded > right_bound) | (col_idx_expanded < left_bound)
                
                # Apply mask (set to -inf where mask is True)
                qk_scaled = tl.where(mask, float("-inf"), qk_scaled)
        else:
            if IS_CAUSAL:
                causal_boundary = start_n + offs_n - seqlen_delta_qk
                causal_mask = offs_m[:, None] >= causal_boundary[None, :]
                qk_scaled = tl.where(causal_mask, qk_scaled, float("-inf"))
        
        # compute qk mask
        qk_mask = (offs_m[:, None] < seqlen_q) & (kv_offs_n[None, :] < seqlen_k)

        # compute bias
        if bias_base_ptrs is not None:
            bias_ptrs = bias_base_ptrs + start_n * stride_bn
            bias = tl.load(bias_ptrs, mask=qk_mask, other=0.0)
            qk_scaled += bias

        # get max scores so far
        m_ij = tl.maximum(m_i, tl.max(qk_scaled, 1))

        # scale and subtract max
        # IMPORTANT: Handle the case where all values are -inf
        # When m_ij = -inf and qk_scaled = -inf, subtraction gives NaN
        # We need to handle this explicitly
        if USE_SLIDING_WINDOW:
            # Check if this block has any valid values (m_ij != -inf)
            # For rows where everything is -inf, set q_shifted to -inf (not NaN)
            q_shifted = tl.where(m_ij[:, None] == float("-inf"), 
                                float("-inf"), 
                                qk_scaled - m_ij[:, None])
        else:
            q_shifted = qk_scaled - m_ij[:, None]
        
        # Compute scaled QK and softmax probabilities
        if USE_EXP2:
            p = tl.math.exp2(q_shifted * RCP_LN2)
        else:
            p = tl.math.exp(q_shifted)

        # CAVEAT: Must update l_ij before applying dropout
        l_ij = tl.sum(p, 1)
        if ENABLE_DROPOUT:
            dropout_mask_ptrs = dropout_mask_base_ptrs + start_n * stride_sn
            sd_mask_ptrs = sd_mask_base_ptrs + start_n * stride_sn
            philox_ptrs = philox_base_ptrs + start_n * stride_sn
            if tl_DROPOUT_USE_PYTORCH:
                dropout_mask = tl.load(dropout_mask_ptrs, mask=qk_mask)
            else:
                rng_output = tl.rand(philox_seed, philox_ptrs)  # TODO: use tl.randint for better performance
                dropout_mask = rng_output > dropout_p
                if tl_DROPOUT_DUMP:
                    tl.store(dropout_mask_ptrs, dropout_mask, mask=qk_mask)

            # return scores with negative values for dropped vals
            sd_mask = tl.where(dropout_mask, p, -p)
            tl.store(sd_mask_ptrs, sd_mask, mask=qk_mask)

            # apply dropout mask in place
            p = tl.where(dropout_mask, p, 0.0)
        elif RETURN_SCORES:
            # NOTE: the returned score is not the same as the reference because we need to adjust as we find new maxes per block. We are not doing that
            sd_mask_ptrs = sd_mask_base_ptrs + start_n * stride_sn
            tl.store(sd_mask_ptrs, p, mask=qk_mask)
        
        # -- update output accumulator --
        # alpha is an adjustment factor for acc and li as we loop and find new maxes
        # store the diff in maxes to adjust acc and li as we discover new maxes
        m_diff = tl.where(m_ij == float("-inf"), 
                                float("-inf"), 
                                m_i - m_ij)
        if USE_EXP2:
            alpha = tl.math.exp2(m_diff * RCP_LN2)
        else:
            alpha = tl.math.exp(m_diff)
        acc = acc * alpha[:, None]
        if not PRE_LOAD_V:
            v = tl.load(v_ptrs, mask=v_mask, other=0.0)
        
        # -- update m_i and l_i
        l_i = l_i * alpha + l_ij
        m_i = m_ij

        if IS_FP8:
            scale_p, descale_p = compute_fp8_scaling_factors(p, FP8_MAX)
            acc += (tl.dot((p * scale_p).to(v.type.element_ty), v) * descale_p * descale_v)
        else:
            acc += tl.dot(p.to(v.type.element_ty), v)
    
    return acc, l_i, m_i


@triton.jit
def compute_masking(seqlen_k, seqlen_q, start_m,
                      IS_CAUSAL: tl.constexpr, USE_SLIDING_WINDOW: tl.constexpr,
                      WINDOW_SIZE_LEFT: tl.constexpr, WINDOW_SIZE_RIGHT: tl.constexpr,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """
    Classify K blocks for attention computation with sliding window support.
    
    Returns:
        - n_front_skip_blocks: Blocks completely before the window
        - n_front_masked_blocks: Blocks partially overlapping window front
        - n_full_blocks: Blocks completely inside the window
        - n_back_masked_blocks: Blocks partially overlapping window back
        - n_extra_tokens: Padding tokens in last K block
    """
    # Example case
    # BLOCK_M = 4, BLOCK_N = 4, seqlen_q = 8, seqlen_k = 10

    # Total K blocks in the key sequence
    total_k_blocks = tl.cdiv(seqlen_k, BLOCK_N)

    # check if we will need to do masking due either BLOCK_N being bigger than seqlen_k or seqlen_k not being a factor of BLOCK_N
    # n_extra_tokens = 10 % 4 = 2
    # This means the last K block has 2 valid tokens and 2 padding positions
    # K blocks visualization:
    #         Block 0         Block 1         Block 2 (last)
    #         K0 K1 K2 K3    K4 K5 K6 K7     K8 K9 ?? ??
    #         ↑---------↑    ↑---------↑     ↑---↑ ↑---↑
    #         full block     full block      valid  pad
    if seqlen_k < BLOCK_N:
        n_extra_tokens = BLOCK_N - seqlen_k
    elif seqlen_k % BLOCK_N:
        n_extra_tokens = seqlen_k % BLOCK_N
    else:
        n_extra_tokens = 0 
    
    if USE_SLIDING_WINDOW:
        # TODO: Optimize by computing which blocks can be fully skipped
        # For now, process all blocks with the mask function
        if IS_CAUSAL:
            return 0, 0, 0, total_k_blocks, n_extra_tokens
        else:
            return 0, 0, 0, total_k_blocks, n_extra_tokens
    else:
        if IS_CAUSAL:
            # ========== CAUSAL MODE: Classify K Blocks ==========
            # Calculate causal boundary for this Q block
            #          [K0 K1 K2 K3] [K4 K5 K6 K7] [K8 K9 ?? ??]
            # Q0-Q3:   [ 1  0  0  0] [ 0  0  0  0] [ 0  0 -- --]  ← Q0
            #          [ 1  1  0  0] [ 0  0  0  0] [ 0  0 -- --]  ← Q1
            #          [ 1  1  1  0] [ 0  0  0  0] [ 0  0 -- --]  ← Q2
            #          [ 1  1  1  1] [ 1  1  0  0] [ 0  0 -- --]  ← Q3
            #                            ↑ can see up to K5
            #
            # Q4-Q7:   [ 1  1  1  1] [ 1  1  1  0] [ 0  0 -- --]  ← Q4
            #          [ 1  1  1  1] [ 1  1  1  1] [ 0  0 -- --]  ← Q5
            #          [ 1  1  1  1] [ 1  1  1  1] [ 1  0 -- --]  ← Q6
            #          [ 1  1  1  1] [ 1  1  1  1] [ 1  1 -- --]  ← Q7

            # ------------------------------------------------------------
            # 1. figure out, in tokens, the right-most K position
            #    this Q-block may attend to
            # ------------------------------------------------------------
            q_start      = start_m * BLOCK_M
            q_end        = tl.minimum((start_m + 1) * BLOCK_M - 1, seqlen_q - 1)

            # causal diagonal offset between the two streams
            diag         = seqlen_k - seqlen_q          # 0 when |Q| == |K|
            k_max_token  = q_end + diag                 # last visible K index

            # this Q-block is entirely above the diagonal ⇒ nothing to do
            if k_max_token < 0:
                return 0, 0, 0, 0, n_extra_tokens

            k_max_token  = tl.minimum(k_max_token, seqlen_k - 1)

            # ------------------------------------------------------------
            # 2. translate token indices into K-block indices
            # ------------------------------------------------------------
            last_visible_k_block = k_max_token // BLOCK_N
            n_visible_k_blocks   = tl.minimum(last_visible_k_block + 1, total_k_blocks)

            # ------------------------------------------------------------
            # 3. classify those visible blocks
            #    – we *never* skip or mask blocks in front, because causal
            #      attention always starts at K0
            #    – the back side can require several masked blocks:
            #         • intersection of the causal diagonal with K-grid
            #           (at most  ⌈BLOCK_M / BLOCK_N⌉ blocks)
            #         • plus one extra block if this Q-block stops in the
            #           middle of a K-block or the last K-block is padded
            # ------------------------------------------------------------
            padded_last_k = n_extra_tokens != 0
            is_modulo_mn  = (not padded_last_k) & (seqlen_q % BLOCK_M == 0)

            n_back_masked_blocks = BLOCK_M // BLOCK_N + tl.where(is_modulo_mn, 0, 1)
            n_back_masked_blocks = tl.minimum(n_back_masked_blocks, n_visible_k_blocks)

            n_front_skip_blocks   = 0        # causal never skips the left side
            n_front_masked_blocks = 0        # ditto
            n_full_blocks         = n_visible_k_blocks - n_back_masked_blocks
        else:
            # ========== NON-CAUSAL MODE ==========
            # Without causal mask, all positions can attend to all positions
            # Only need to handle the padding in the last block
            #          [K0 K1 K2 K3] [K4 K5 K6 K7] [K8 K9 ?? ??]
            # Q0-Q3:   [ 1  1  1  1] [ 1  1  1  1] [ 1  1 -∞ -∞]
            #          [ 1  1  1  1] [ 1  1  1  1] [ 1  1 -∞ -∞]
            #          [ 1  1  1  1] [ 1  1  1  1] [ 1  1 -∞ -∞]
            #          [ 1  1  1  1] [ 1  1  1  1] [ 1  1 -∞ -∞]
            #
            # Q4-Q7:   [ 1  1  1  1] [ 1  1  1  1] [ 1  1 -∞ -∞]
            #          [ 1  1  1  1] [ 1  1  1  1] [ 1  1 -∞ -∞]
            #          [ 1  1  1  1] [ 1  1  1  1] [ 1  1 -∞ -∞]
            #          [ 1  1  1  1] [ 1  1  1  1] [ 1  1 -∞ -∞]
            
            n_front_skip_blocks   = 0        # never skips the left side
            n_front_masked_blocks = 0        # ditto
            if n_extra_tokens != 0:
                n_back_masked_blocks = 1  # Last block needs padding mask
                n_full_blocks = total_k_blocks - 1
            else:
                n_back_masked_blocks = 0  # All blocks are aligned
                n_full_blocks = total_k_blocks
    
        return n_front_skip_blocks, n_front_masked_blocks, n_full_blocks, n_back_masked_blocks, n_extra_tokens

@triton.autotune(
    configs=fwd_prefill_autotune_configs,
    key=fwd_prefill_autotune_keys,
    use_cuda_graph=True,
)
@triton.jit
def attn_fwd(Q, K, V, bias,
             Descale_Q, Descale_K, Descale_V, Descale_O, stride_descale_q_z, stride_descale_k_z, stride_descale_v_z, stride_descale_o_z,
             SM_SCALE: tl.constexpr, LSE, Out, stride_qz, stride_qh, stride_qm, stride_qk,
             stride_kz, stride_kh, stride_kn, stride_kk, stride_vz, stride_vh, stride_vk, stride_vn,
             stride_oz, stride_oh, stride_om, stride_on, stride_bz, stride_bh, stride_bm, stride_bn, stride_az, stride_ah,
             stride_sz, stride_sh, stride_sm, stride_sn, stride_lse_z, stride_lse_h, stride_lse_m, cu_seqlens_q, cu_seqlens_k,
             dropout_p, philox_seed, philox_offset_base, sd_mask, dropout_mask, alibi_slopes, HQ: tl.constexpr,
             HK: tl.constexpr, ACTUAL_BLOCK_DMODEL: tl.constexpr, MAX_SEQLENS_Q: tl.constexpr,
             MAX_SEQLENS_K: tl.constexpr, IS_VARLEN: tl.constexpr, IS_CAUSAL: tl.constexpr,
             USE_SLIDING_WINDOW: tl.constexpr, WINDOW_SIZE_LEFT: tl.constexpr, WINDOW_SIZE_RIGHT: tl.constexpr, BLOCK_M: tl.constexpr,
             BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr, PRE_LOAD_V: tl.constexpr, USE_BIAS: tl.constexpr,
             ENABLE_DROPOUT: tl.constexpr, RETURN_SCORES: tl.constexpr, NEEDS_SDMASK : tl.constexpr, USE_ALIBI: tl.constexpr, USE_EXP2: tl.constexpr, 
             IS_FP8: tl.constexpr, FP8_MAX: tl.constexpr, FP8_OUTPUT: tl.constexpr):
    # set params
    ACCUMULATOR_TYPE = tl.float32

    # compute offsets
    off_z = tl.program_id(0)
    off_h_q = tl.program_id(1)
    start_m = tl.program_id(2)
    # If MQA / GQA, set the K and V head offsets appropriately.
    GROUP_SIZE: tl.constexpr = HQ // HK
    if GROUP_SIZE != 1:
        off_h_k = off_h_q // GROUP_SIZE
    else:
        off_h_k = off_h_q
    # Determine if we need to mask the heads
    PADDED_HEAD: tl.constexpr = (ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # handle seqlen
    if IS_VARLEN:
        cu_seqlens_q_start = tl.load(cu_seqlens_q + off_z)
        cu_seqlens_q_end = tl.load(cu_seqlens_q + off_z + 1)
        seqlen_q = cu_seqlens_q_end - cu_seqlens_q_start
        
        # we have a one-size-fits-all grid in id(0). Some seqlens might be too small for all start_m so for those we return early.
        if start_m * BLOCK_M > seqlen_q:
            return
        cu_seqlens_k_start = tl.load(cu_seqlens_k + off_z)
        cu_seqlens_k_end = tl.load(cu_seqlens_k + off_z + 1)
        seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start
    else:
        cu_seqlens_q_start = 0
        cu_seqlens_k_start = 0
        seqlen_q = MAX_SEQLENS_Q
        seqlen_k = MAX_SEQLENS_K

    # Load scale factors if IS_FP8.
    if IS_FP8:
        descale_q = tl.load(Descale_Q + off_z * stride_descale_q_z + off_h_q)
        descale_k = tl.load(Descale_K + off_z * stride_descale_k_z + off_h_k)
        descale_v = tl.load(Descale_V + off_z * stride_descale_v_z + off_h_k)
    else:
        descale_q, descale_k, descale_v = 1.0, 1.0, 1.0
    

    # figure out masking pattern
    n_front_skip_blocks, n_front_masked_blocks, n_full_blocks, n_back_masked_blocks, n_extra_tokens = compute_masking(
        seqlen_k, seqlen_q, start_m, IS_CAUSAL, USE_SLIDING_WINDOW, 
        WINDOW_SIZE_LEFT, WINDOW_SIZE_RIGHT, BLOCK_M, BLOCK_N
    )

    # ============================================================
    #          PROGRAM EARLY EXIT (All K Blocks Skipped)
    # ============================================================
    total_visible_blocks = n_front_masked_blocks + n_full_blocks + n_back_masked_blocks
    if total_visible_blocks == 0:
        """
        No K blocks visible - write zeros and exit.
        """
        # Write zeros to output
        o_offset = Out + off_z * stride_oz + off_h_q * stride_oh + cu_seqlens_q_start * stride_om
        o_ptrs = o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on
        o_mask = (offs_m[:, None] < seqlen_q)
        if PADDED_HEAD:
            o_mask = o_mask & (offs_d[None, :] < ACTUAL_BLOCK_DMODEL)
        tl.store(o_ptrs, tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=Out.type.element_ty), mask=o_mask)
        
        # Write zeros to LSE
        l_ptrs = LSE + off_z * stride_lse_z + off_h_q * stride_lse_h + cu_seqlens_q_start * stride_lse_m + offs_m * stride_lse_m
        tl.store(l_ptrs, tl.zeros([BLOCK_M], dtype=tl.float32), mask=offs_m < seqlen_q)
        return
    
    # ============================================================
    #         NORMAL PROCESSING (Some K Blocks Visible)
    # ============================================================
    """
    This program has visible K blocks to process.
    We'll use two calls to handle different block types efficiently.
    """
    
    # Initialize for processing
    # Compute pointers for all the tensors used in this kernel.
    q_offset = Q + off_z * stride_qz + off_h_q * stride_qh + cu_seqlens_q_start * stride_qm
    q_ptrs = q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_offset = K + off_z * stride_kz + off_h_k * stride_kh + cu_seqlens_k_start * stride_kn
    k_ptrs = k_offset + offs_d[:, None] * stride_kk + offs_n[None, :] * stride_kn
    v_offset = V + off_z * stride_vz + off_h_k * stride_vh + cu_seqlens_k_start * stride_vk
    v_ptrs = v_offset + offs_n[:, None] * stride_vk + offs_d[None, :] * stride_vn
    if USE_BIAS:
        # Note: this might get large enough to overflow on some configs
        bias_offset = off_h_q * stride_bh
        bias_ptrs = bias + bias_offset + offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn
    else:
        bias_ptrs = None

    if USE_ALIBI:
        a_offset = off_z * stride_az + off_h_q * stride_ah
        alibi_slope = tl.load(alibi_slopes + a_offset)
    else:
        alibi_slope = None

    if NEEDS_SDMASK:
        sd_mask_offset = sd_mask + off_z * stride_sz + off_h_q * stride_sh #+ cu_seqlens_q_start * stride_sm
        sd_mask_ptrs = sd_mask_offset + offs_m[:, None] * stride_sm + offs_n[None, :] * stride_sn
    else:
        sd_mask_ptrs = None

    if ENABLE_DROPOUT:
        dropout_mask_offset = dropout_mask + off_z * stride_sz + off_h_q * stride_sh #+ cu_seqlens_q_start * stride_sm
        dropout_mask_ptrs = dropout_mask_offset + offs_m[:, None] * stride_sm + offs_n[None, :] * stride_sn
        batch_philox_offset = philox_offset_base + off_z * stride_sz + off_h_q * stride_sh #+ cu_seqlens_q_start * stride_sm
        philox_ptrs = batch_philox_offset + offs_m[:, None] * stride_sm + offs_n[None, :] * stride_sn
    else:
        dropout_mask_ptrs = None
        philox_ptrs = 0

    # initialize pointer to m and l
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=ACCUMULATOR_TYPE)
    l_i = tl.full([BLOCK_M], 1.0, dtype=ACCUMULATOR_TYPE)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=ACCUMULATOR_TYPE)

    # Q is loaded once at the beginning and shared by all N blocks.
    q_ptrs_mask = offs_m[:, None] < seqlen_q
    if PADDED_HEAD:
        q_ptrs_mask = q_ptrs_mask & (offs_d[None, :] < ACTUAL_BLOCK_DMODEL)
    q = tl.load(q_ptrs, mask=q_ptrs_mask, other=0.0)


    # ========== Process MASKED K Blocks in the front ==========
    # NOTE: we use USE_SLIDING_WINDOW as guard because the compiler will crash other wise. front masking is only for sliding window so that is fine.
    if n_front_masked_blocks > 0 and USE_SLIDING_WINDOW:
        block_min = n_front_skip_blocks * BLOCK_N
        block_max = (n_front_skip_blocks + n_front_masked_blocks) * BLOCK_N
        
        acc, l_i, m_i = _attn_fwd_mask(
            acc, l_i, m_i, 
            q, k_ptrs, v_ptrs, bias_ptrs, 
            stride_kn, stride_vk, stride_bn, stride_sn,
            start_m, seqlen_k, seqlen_q, 
            dropout_p, philox_seed, philox_ptrs,
            sd_mask_ptrs, dropout_mask_ptrs,
            offs_m, offs_n, offs_d,
            block_min,        # Start of front masked blocks
            block_max,        # End of front masked blocks
            0,                # n_extra_tokens (0 for front blocks, only relevant for last block)
            alibi_slope, 
            descale_q, descale_k, descale_v, IS_FP8, FP8_MAX,
            IS_CAUSAL,
            BLOCK_M, BLOCK_DMODEL, BLOCK_N,
            PRE_LOAD_V,
            ENABLE_DROPOUT, PADDED_HEAD,
            ACTUAL_BLOCK_DMODEL, SM_SCALE, 
            USE_ALIBI=USE_ALIBI, USE_EXP2=USE_EXP2, 
            RETURN_SCORES=RETURN_SCORES, 
            USE_SLIDING_WINDOW=USE_SLIDING_WINDOW, 
            WINDOW_SIZE_LEFT=WINDOW_SIZE_LEFT, 
            WINDOW_SIZE_RIGHT=WINDOW_SIZE_RIGHT,
            ACCUMULATOR_TYPE=ACCUMULATOR_TYPE
        )
    
    # ========== Process FULL K Blocks (Fast Path) ==========
    if n_full_blocks > 0:
        block_min = (n_front_skip_blocks + n_front_masked_blocks) * BLOCK_N
        block_max = (n_front_skip_blocks + n_front_masked_blocks + n_full_blocks) * BLOCK_N
        
        acc, l_i, m_i = _attn_fwd_no_mask(
            acc, l_i, m_i, 
            q, k_ptrs, v_ptrs, bias_ptrs, 
            stride_kn, stride_vk, stride_bn, stride_sn,
            start_m, seqlen_k, seqlen_q, 
            dropout_p, philox_seed, philox_ptrs,
            sd_mask_ptrs, dropout_mask_ptrs,
            offs_m, offs_n, offs_d,
            block_min,        # Start of range: 0
            block_max,        # End of range: n_full_blocks * BLOCK_N
            alibi_slope,
            descale_q, descale_k, descale_v, IS_FP8, FP8_MAX,
            BLOCK_M, BLOCK_DMODEL, BLOCK_N,
            PRE_LOAD_V,
            ENABLE_DROPOUT, PADDED_HEAD,
            ACTUAL_BLOCK_DMODEL, SM_SCALE, 
            USE_ALIBI=USE_ALIBI, USE_EXP2=USE_EXP2, 
            RETURN_SCORES=RETURN_SCORES, ACCUMULATOR_TYPE=ACCUMULATOR_TYPE
        )
    
    # ========== Process MASKED K Blocks in the back ==========
    if n_back_masked_blocks > 0:
        block_min = (n_front_skip_blocks + n_front_masked_blocks + n_full_blocks) * BLOCK_N
        block_max = (n_front_skip_blocks + n_front_masked_blocks + n_full_blocks + n_back_masked_blocks) * BLOCK_N
        
        acc, l_i, m_i = _attn_fwd_mask(
            acc, l_i, m_i, 
            q, k_ptrs, v_ptrs, bias_ptrs, 
            stride_kn, stride_vk, stride_bn, stride_sn,
            start_m, seqlen_k, seqlen_q, 
            dropout_p, philox_seed, philox_ptrs,
            sd_mask_ptrs, dropout_mask_ptrs,
            offs_m, offs_n, offs_d,
            block_min,        # Start of range: n_full_blocks * BLOCK_N
            block_max,        # End of range: n_visible_k_blocks * BLOCK_N
            n_extra_tokens,   # Padding tokens in last block
            alibi_slope, 
            descale_q, descale_k, descale_v, IS_FP8, FP8_MAX,
            IS_CAUSAL,        # Use actual causal flag
            BLOCK_M, BLOCK_DMODEL, BLOCK_N,
            PRE_LOAD_V,
            ENABLE_DROPOUT, PADDED_HEAD,
            ACTUAL_BLOCK_DMODEL, SM_SCALE, 
            USE_ALIBI=USE_ALIBI, USE_EXP2=USE_EXP2, 
            RETURN_SCORES=RETURN_SCORES, 
            USE_SLIDING_WINDOW=USE_SLIDING_WINDOW, 
            WINDOW_SIZE_LEFT=WINDOW_SIZE_LEFT, 
            WINDOW_SIZE_RIGHT=WINDOW_SIZE_RIGHT,
            ACCUMULATOR_TYPE=ACCUMULATOR_TYPE
        )

    # ============================================================
    #                        EPILOGUE
    # ============================================================
    # This helps the compiler do Newton Raphson on l_i vs on acc which is much larger.
    # Instead of directly computing 1/l_i which can be inf, 
    # we check for the invalid case first
    if USE_SLIDING_WINDOW:
        # For rows where m_i is still -inf, no keys were valid
        # Set l_i to 1.0 to avoid division by zero (acc is already 0)
        invalid_mask = m_i == float("-inf")
        l_i_safe = tl.where(invalid_mask, 1.0, l_i)
        l_recip = 1 / l_i_safe[:, None]
    else:
        invalid_mask = None
        l_recip = 1 / l_i[:, None]
    acc = acc * l_recip
    if ENABLE_DROPOUT:
        dropout_scale = 1 / (1 - dropout_p)
        acc = acc * dropout_scale

    # compute log-sum-exp
    if USE_EXP2:
        RCP_LN2: tl.constexpr = 1.4426950408889634
        LN2: tl.constexpr = 0.6931471824645996
        # compute log-sum-exp in base 2 units
        mi_base2 = m_i * RCP_LN2
        # For invalid rows, log(l_i) would be -inf, but we want LSE to be -inf
        # So we handle this case explicitly
        if USE_SLIDING_WINDOW:
            log_l_i = tl.where(invalid_mask, 0.0, tl.math.log2(l_i))
            softmax_lse = mi_base2 + log_l_i
            # Ensure invalid rows have LSE = -inf
            softmax_lse = tl.where(invalid_mask, float("-inf"), softmax_lse)
        else:
            softmax_lse = mi_base2 + tl.math.log2(l_i)
        # convert back to natural units
        softmax_lse *= LN2
    else:
        if USE_SLIDING_WINDOW:
            log_l_i = tl.where(invalid_mask, 0.0, tl.math.log(l_i))
            softmax_lse = m_i + log_l_i
            softmax_lse = tl.where(invalid_mask, float("-inf"), softmax_lse)
        else:
            softmax_lse = m_i + tl.math.log(l_i)

    # handle masking edge cases
    if USE_SLIDING_WINDOW:
        if IS_CAUSAL:
            pass
        else:
            pass
    else:
        if IS_CAUSAL:
            # When seqlen_q > seqlen_k, some rows are completely above the causal diagonal
            # These rows have all -inf attention scores, resulting in NaN after softmax
            # e.g.
            # Q length: 6, K length: 4
            # Causal mask (X = can attend, . = cannot):
            #    K0 K1 K2 K3
            # Q0   .  .  .  .  <- All masked, would give NaN
            # Q1   .  .  .  .  <- All masked, would give NaN  
            # Q2   X  .  .  .  <- First valid row
            # Q3   X  X  .  .
            # Q4   X  X  X  .
            # Q5   X  X  X  X
            causal_start_idx = seqlen_q - seqlen_k
            start_m_idx = start_m * BLOCK_M
            
            # Create mask for rows that need zeroing
            row_indices = start_m_idx + tl.arange(0, BLOCK_M)
            causal_mask = row_indices < causal_start_idx
            
            # Zero out both acc and LSE for these rows
            if causal_start_idx > start_m_idx:
                end_m_idx = (start_m + 1) * BLOCK_M
                if causal_start_idx < end_m_idx:
                    # This block contains the boundary - need to mask acc
                    out_mask_boundary = tl.full((BLOCK_DMODEL, ), causal_start_idx, dtype=tl.int32)
                    out_ptrs_mask = row_indices[:, None] >= out_mask_boundary[None, :]
                    z = 0.0
                    acc = tl.where(out_ptrs_mask, acc, z.to(acc.type.element_ty))
            
            # Zero out LSE for rows above diagonal
            softmax_lse = tl.where(causal_mask, 0.0, softmax_lse)

    # write back LSE(Log Sum Exponents), the log of the normalization constant
    l_offset = LSE + off_z * stride_lse_z + off_h_q * stride_lse_h + cu_seqlens_q_start * stride_lse_m
    l_ptrs = l_offset + offs_m * stride_lse_m

    # If seqlen_q not multiple of BLOCK_M, we need to mask out the last few rows.
    # This is only true for the last Q block. For others, overflow_size will be -ve
    end_m_idx = (start_m + 1) * BLOCK_M
    overflow_size = end_m_idx - seqlen_q
    if overflow_size > 0:
        boundary = tl.full((BLOCK_M, ), BLOCK_M - overflow_size, dtype=tl.int32)
        l_ptrs_mask = tl.arange(0, BLOCK_M) < boundary
        tl.store(l_ptrs, softmax_lse, mask=l_ptrs_mask)
    else:
        tl.store(l_ptrs, softmax_lse)

    # write back O
    o_offset = Out + off_z * stride_oz + off_h_q * stride_oh + cu_seqlens_q_start * stride_om
    o_ptrs = o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on
    o_ptrs_mask = tl.full([BLOCK_M, BLOCK_DMODEL], 1, dtype=tl.int1)
    if overflow_size > 0:
        o_ptrs_mask = o_ptrs_mask & (offs_m[:, None] < seqlen_q)
    if PADDED_HEAD:
        o_ptrs_mask = o_ptrs_mask & (offs_d[None, :] < ACTUAL_BLOCK_DMODEL)

    if FP8_OUTPUT:
        scale_acc, descale_acc = compute_fp8_scaling_factors(acc, FP8_MAX)
        tl.store(Descale_O + off_z * stride_descale_o_z + off_h_q, descale_acc)
        tl.store(o_ptrs, (acc * scale_acc).to(Out.type.element_ty), mask=o_ptrs_mask)
    else:
        tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=o_ptrs_mask)

def attention_prefill_forward_triton_impl(
                                        q: torch.Tensor,
                                        k: torch.Tensor,
                                        v: torch.Tensor,
                                        o: torch.Tensor,
                                        sm_scale: float,
                                        alibi_slopes: Optional[torch.Tensor],
                                        causal: bool,
                                        window_size_left: int,
                                        window_size_right: int,
                                        bias: Optional[torch.Tensor],
                                        layout: Literal["bshd", "bhsd", "thd"],
                                        # varlen
                                        cu_seqlens_q: Optional[torch.Tensor], 
                                        cu_seqlens_k: Optional[torch.Tensor],
                                        max_seqlens_q: int, 
                                        max_seqlens_k: int,
                                        # dropout
                                        dropout_p: float,
                                        philox_seed: Optional[int],
                                        philox_offset: Optional[int],
                                        # misc
                                        return_softmax: bool,
                                        use_exp2: bool,
                                        # fp8
                                        descale_q: Optional[torch.Tensor],
                                        descale_k: Optional[torch.Tensor],
                                        descale_v: Optional[torch.Tensor],
                                        descale_o: Optional[torch.Tensor],
):
    # get params, strides and shape
    IS_VARLEN = layout == "thd"

    # common assertions
    assert 0.0 <= dropout_p <= 1.0, f"dropout_p must be between 0 and 1, got {dropout_p}"
    assert q.device == k.device == v.device == o.device, \
        f"All tensors must be on the same device. Got: q={q.device}, k={k.device}, v={v.device}, o={o.device}"
    assert q.dtype == k.dtype == v.dtype, "q, k, v must have the same dtype"
    current_device = torch.cuda.current_device()
    assert q.is_cuda and q.device.index == current_device, f"Device mismatch: Kernel will launch on cuda:{current_device}, but tensors are on {q.device}"
    
    # get shapes and strides
    if IS_VARLEN:
        # shape
        total_seqlen_q, nheads_q, head_size_q = q.shape
        total_seqlen_k, nheads_k, head_size_k = k.shape
        total_seqlen_v, nheads_v, head_size_v = v.shape
        
        # assert shapes
        assert cu_seqlens_q is not None, "cu_seqlens_q must be provided for varlen layout"
        assert cu_seqlens_k is not None, "cu_seqlens_k must be provided for varlen layout"
        assert max_seqlens_q is not None and max_seqlens_q > 0, "max_seqlens_q must be provided and positive for varlen layout"
        assert max_seqlens_k is not None and max_seqlens_k > 0, "max_seqlens_k must be provided and positive for varlen layout"
        
        # assert head dimensions
        assert head_size_q == head_size_k == head_size_v, f"head sizes must match: q={head_size_q}, k={head_size_k}, v={head_size_v}"
        assert nheads_k == nheads_v, f"k and v must have same number of heads: k={nheads_k}, v={nheads_v}"
        assert nheads_q % nheads_k == 0, f"nheads_q {nheads_q} must be divisible by nheads_k {nheads_k} for GQA/MQA"
        
        # assert output shapes
        assert o.shape == (total_seqlen_q, nheads_q, head_size_q), f"o shape {o.shape} != expected {(total_seqlen_q, nheads_q, head_size_q)}"
        
        # assert cu_seqlens
        assert cu_seqlens_q.dtype == torch.int32, f"cu_seqlens_q must be int32, got {cu_seqlens_q.dtype}"
        assert cu_seqlens_k.dtype == torch.int32, f"cu_seqlens_k must be int32, got {cu_seqlens_k.dtype}"
        assert cu_seqlens_q[0] == 0, "cu_seqlens_q must start with 0"
        assert cu_seqlens_k[0] == 0, "cu_seqlens_k must start with 0"
        assert cu_seqlens_q[-1] == total_seqlen_q, f"cu_seqlens_q[-1] {cu_seqlens_q[-1]} != total_seqlen_q {total_seqlen_q}"
        assert cu_seqlens_k[-1] == total_seqlen_k, f"cu_seqlens_k[-1] {cu_seqlens_k[-1]} != total_seqlen_k {total_seqlen_k}"
        
        # set vars
        batch = len(cu_seqlens_q) - 1
        head_size = head_size_q
        
        # softmax_lse shape
        softmax_lse = torch.zeros((nheads_q, total_seqlen_q), device=q.device, dtype=torch.float32)
        
        # strides
        stride_qb, stride_qh, stride_qm, stride_qd = 0, q.stride(1), q.stride(0), q.stride(2)
        stride_kb, stride_kh, stride_kn, stride_kd = 0, k.stride(1), k.stride(0), k.stride(2)
        stride_vb, stride_vh, stride_vn, stride_vd = 0, v.stride(1), v.stride(0), v.stride(2)
        stride_ob, stride_oh, stride_om, stride_od = 0, o.stride(1), o.stride(0), o.stride(2)
        stride_lse_z, stride_lse_h, stride_lse_m = 0, softmax_lse.stride(0), softmax_lse.stride(1)
    else:
        # shapes
        batch_q, seqlen_q, nheads_q, head_size_q = q.shape
        batch_k, seqlen_k, nheads_k, head_size_k = k.shape
        batch_v, seqlen_v, nheads_v, head_size_v = v.shape
        
        # assert batch dimensions
        assert batch_q == batch_k == batch_v, f"batch sizes must match: q={batch_q}, k={batch_k}, v={batch_v}"
        
        # assert head dimensions
        assert head_size_q == head_size_k == head_size_v, f"head sizes must match: q={head_size_q}, k={head_size_k}, v={head_size_v}"
        assert nheads_k == nheads_v, f"k and v must have same number of heads: k={nheads_k}, v={nheads_v}"
        assert nheads_q % nheads_k == 0, f"nheads_q {nheads_q} must be divisible by nheads_k {nheads_k} for GQA/MQA"
        
        # assert sequence lengths
        assert seqlen_k == seqlen_v, f"k and v sequence lengths must match: k={seqlen_k}, v={seqlen_v}"
        
        # assert output shapes
        assert o.shape == (batch_q, seqlen_q, nheads_q, head_size_q), f"o shape {o.shape} != expected {(batch_q, seqlen_q, nheads_q, head_size_q)}"
        
        # set vars
        batch = batch_q
        head_size = head_size_q
        max_seqlens_q = seqlen_q
        max_seqlens_k = seqlen_k
        
        # softmax_lse shape
        softmax_lse = torch.zeros((batch, nheads_q, seqlen_q), device=q.device, dtype=torch.float32)
        
        # strides
        stride_qb, stride_qh, stride_qm, stride_qd = q.stride(0), q.stride(2), q.stride(1), q.stride(3)
        stride_kb, stride_kh, stride_kn, stride_kd = k.stride(0), k.stride(2), k.stride(1), k.stride(3)
        stride_vb, stride_vh, stride_vn, stride_vd = v.stride(0), v.stride(2), v.stride(1), v.stride(3)
        stride_ob, stride_oh, stride_om, stride_od = o.stride(0), o.stride(2), o.stride(1), o.stride(3)
        stride_lse_z, stride_lse_h, stride_lse_m = softmax_lse.stride()

    # fp8 setup and assertions
    IS_FP8 = is_fp8(q)
    if IS_FP8:
        # we already asserted that q, k, v all have the same dtype, so no need to check each one

        FP8_MAX = torch.finfo(q.dtype).max

        # Check descale tensors
        assert descale_q is not None, "descale_q must be provided when using fp8"
        assert descale_k is not None, "descale_k must be provided when using fp8"
        assert descale_v is not None, "descale_v must be provided when using fp8"
        
        if is_fp8(o):
            FP8_OUTPUT = True
            assert descale_o is not None, f"descale_o is None. In fp8, you need to pass a tensor for descale_o along with a tensor o."
        else:
            FP8_OUTPUT = False
            # o should be fp32 or fp16/bf16
            assert o.dtype in [torch.float16, torch.bfloat16, torch.float32], \
                f"Output tensor o must be fp16, bf16, or fp32 when using fp8, got {o.dtype}"

        stride_descale_q_z = descale_q.stride(0) if descale_q is not None else None
        stride_descale_k_z = descale_k.stride(0) if descale_k is not None else None
        stride_descale_v_z = descale_v.stride(0) if descale_v is not None else None
        stride_descale_o_z = descale_o.stride(0) if descale_o is not None else None
    else:
        FP8_MAX = None
        FP8_OUTPUT = False
        descale_q = descale_k = descale_v = descale_o = None
        stride_descale_q_z = stride_descale_k_z = stride_descale_v_z = stride_descale_o_z = None
        
        # check output dtype matches input dtype when not using fp8
        assert o.dtype == q.dtype, f"Output dtype {o.dtype} must match input dtype {q.dtype} when not using fp8"
    
    # check features
    use_sliding_window = window_size_left != -1 or window_size_right!= -1
    use_alibi, (stride_az, stride_ah) = (True, alibi_slopes.stride()) if alibi_slopes is not None else (False, (0, 0))
    # NOTE: a large bias tensor leads to overflow during pointer arithmetic
    if (bias is not None):
        assert (bias.numel() < 2**31)

    # Get closest power of 2 over or equal to 32.
    padded_d_model = 1 << (head_size - 1).bit_length()
    # Smallest head_dim supported is 16. If smaller, the tile in the
    # kernel is padded - there is no padding in memory for any dims.
    padded_d_model = max(padded_d_model, 16)

    # sd_mask is used to validate dropout behavior vs the PyTorch SDPA math backend reference.  We zero this out
    # to give a consistent starting point and then populate it with the output of softmax with the sign bit set according
    # to the dropout mask. The resulting return allows this mask to be fed into the reference implementation for testing
    # only. This return holds no useful output aside from debugging.
    NEEDS_SDMASK = (dropout_p > 0.0) or return_softmax
    if NEEDS_SDMASK:
        sd_mask = torch.zeros((batch, nheads_q, max_seqlens_q, max_seqlens_k), device=q.device,
                                        dtype=torch.float32)
        if DROPOUT_USE_PYTORCH:
            dropout_mask = create_dropout_mask(dropout_p, (batch, nheads_q, max_seqlens_q, max_seqlens_k), seed = philox_seed)
        else:
            dropout_mask = torch.zeros((batch, nheads_q, max_seqlens_q, max_seqlens_k), device=q.device,
                                        dtype=torch.float32)
        stride_sz, stride_sh, stride_sm, stride_sn = (sd_mask.stride(0), sd_mask.stride(1), sd_mask.stride(2), sd_mask.stride(3))
    else:
        sd_mask = None
        dropout_mask = None
        stride_sz, stride_sh, stride_sm, stride_sn = (0, 0, 0, 0)

    if bias is not None:
        stride_bz, stride_bh, stride_bm, stride_bn = (bias.stride(0), bias.stride(1),bias.stride(2),
                        bias.stride(3))
    else:
        stride_bz, stride_bh, stride_bm, stride_bn = (0, 0, 0, 0)

    # launch kernel
    grid = lambda META: (batch, nheads_q, triton.cdiv(max_seqlens_q, META['BLOCK_M']))
    attn_fwd[grid](q, k, v, bias,
                    descale_q, descale_k, descale_v, descale_o, stride_descale_q_z, stride_descale_k_z, stride_descale_v_z, stride_descale_o_z,
                    sm_scale, softmax_lse, o,
                    stride_qb, stride_qh, stride_qm, stride_qd, 
                    stride_kb, stride_kh, stride_kn, stride_kd,
                    stride_vb, stride_vh, stride_vn, stride_vd,
                    stride_ob, stride_oh, stride_om, stride_od,
                    stride_bz, stride_bh, stride_bm, stride_bn, 
                    stride_az, stride_ah, 
                    stride_sz, stride_sh, stride_sm, stride_sn, 
                    stride_lse_z, stride_lse_h, stride_lse_m,
                    cu_seqlens_q, cu_seqlens_k,
                    dropout_p=dropout_p, philox_seed=philox_seed, philox_offset_base=philox_offset, sd_mask=sd_mask, dropout_mask=dropout_mask, alibi_slopes=alibi_slopes, 
                    HQ=nheads_q, HK=nheads_k, ACTUAL_BLOCK_DMODEL=head_size, MAX_SEQLENS_Q=max_seqlens_q,
                    MAX_SEQLENS_K=max_seqlens_k, IS_CAUSAL=causal, 
                    USE_SLIDING_WINDOW=use_sliding_window, WINDOW_SIZE_LEFT=window_size_left, WINDOW_SIZE_RIGHT=window_size_right, 
                    IS_VARLEN=IS_VARLEN,
                    BLOCK_DMODEL=padded_d_model, USE_BIAS=False if bias is None else True,
                    USE_ALIBI=use_alibi, ENABLE_DROPOUT=dropout_p > 0.0, USE_EXP2=use_exp2, RETURN_SCORES=return_softmax, NEEDS_SDMASK=NEEDS_SDMASK,
                    IS_FP8=IS_FP8, FP8_MAX=FP8_MAX, FP8_OUTPUT=FP8_OUTPUT)

    return softmax_lse, sd_mask if return_softmax else None