import torch
import triton
import triton.language as tl

from typing import Optional, Tuple

@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y

@triton.jit
def compute_fp8_scaling_factors(x, fp8_max: tl.constexpr):
    # compute fp8 scaling and descaling factor for a block
    x_amax = tl.max(tl.abs(x)) # NOTE: abs deals with negative values
    x_amax = tl.where(x_amax <= 1e-9, 1e-9, x_amax)
    scale_x = fp8_max / x_amax
    descale_x = x_amax / fp8_max
    return scale_x, descale_x

def is_fp8(x):
    if x.dtype in {torch.float8_e4m3fnuz, torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e5m2fnuz}:
        if arch_supports_fp8():
            return True
        else:
            raise RuntimeError("This device does not support fp8")
    else:
        return False


def cast_to_fp8(
    x: torch.Tensor,
    fp8_dtype,
    layout,
    clamp_val=1e-9,
):
    if len(x.shape) != 4:
        raise ValueError(f"'bshd' tensor should have shape [batch, seqlen, heads, dim], got {x.shape}")
    reduce_dims = (1, 3)  # seq_len and dim dimensions
   
    # Compute the absolute max along reduce_dims, clamped to avoid 0-scale
    x_abs_max = x.abs().amax(dim=reduce_dims)
    x_abs_max = torch.maximum(x_abs_max, x.new_tensor(clamp_val))

    # Unsqueeze back to a shape suitable for broadcast
    unsqueeze_dims = sorted(reduce_dims)
    for d in unsqueeze_dims:
        x_abs_max = x_abs_max.unsqueeze(d)

    # compute scale and descale
    fp8_max = torch.finfo(fp8_dtype).max
    scale = fp8_max / x_abs_max
    descale_factor = x_abs_max / fp8_max

    # cast to FP8, optionally setting requires_grad
    x_fp8 = (x * scale).to(fp8_dtype)

    return x_fp8, descale_factor


def cast_varlen_to_fp8(
    x: torch.Tensor,
    fp8_dtype: torch.dtype,
    cu_seqlens,
    clamp_val: float = 1e-9,
) -> tuple[torch.Tensor, torch.Tensor]:
    # validate tensor shape
    if len(x.shape) != 3:
        raise ValueError(f"tensor should have shape [total_seqlen, heads, dim], got {x.shape}")
    num_heads = x.shape[1]
    
    # Get batch size from cu_seqlens
    batch = cu_seqlens.shape[0] - 1
    fp8_max = torch.finfo(fp8_dtype).max
    
    # Compute scale and descale factors per sequence
    x_fp8 = torch.zeros_like(x, dtype=fp8_dtype)
    descale_factors = torch.zeros((batch, num_heads), device=x.device, dtype=torch.float32)
    
    for i in range(batch):
        start = cu_seqlens[i]
        end = cu_seqlens[i + 1]
        x_slice = x[start:end]  # Slice for current sequence
        
        # Standard tensor (0: seq_len, 2: head_dim)
        x_abs_max = x_slice.abs().amax(dim=(0, 2))  # [heads]
        
        # apply minimum clamping
        x_abs_max = torch.maximum(x_abs_max, x.new_tensor(clamp_val))
        
        # compute scale and descale factors
        scale_i = fp8_max / x_abs_max
        descale_i = x_abs_max / fp8_max
        
        # store descale factors
        descale_factors[i, :] = descale_i
        
        scale_reshape = scale_i.reshape(1, num_heads, 1)
        
        # scale and cast to FP8
        x_fp8[start:end] = (x_slice * scale_reshape).to(fp8_dtype)
        
    return x_fp8, descale_factors


#TODO Move this to a common folder. Will need to add future arch list
def get_arch():
    return triton.runtime.driver.active.get_current_target().arch

def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"

def arch_supports_fp8():
    return is_hip() and get_arch() in ('gfx942')

@triton.jit
def load_fn(ptrs, offset_first, offset_second, boundary_first, boundary_second):
    if offset_first is not None and offset_second is not None:
        mask = (offset_first[:, None] < boundary_first) & \
               (offset_second[None, :] < boundary_second)
        tensor = tl.load(ptrs, mask=mask, other=0.0)
    elif offset_first is not None:
        mask = offset_first[:, None] < boundary_first
        tensor = tl.load(ptrs, mask=mask, other=0.0)
    elif offset_second is not None:
        mask = offset_second[None, :] < boundary_second
        tensor = tl.load(ptrs, mask=mask, other=0.0)
    else:
        tensor = tl.load(ptrs)
    return tensor

@triton.jit
def compute_alibi_block(alibi_slope, seqlen_q, seqlen_k, offs_m, offs_n, transpose=False):
    # when seqlen_k and seqlen_q are different we want the diagonal to stick to the bottom right of the attention matrix
    # for casual mask we want something like this where (1 is kept and 0 is masked)
    # seqlen_q = 2 and seqlen_k = 5
    #   1 1 1 1 0
    #   1 1 1 1 1
    # seqlen_q = 5 and seqlen_k = 2
    #        0 0
    #        0 0
    #        0 0
    #        1 0
    #        1 1
    # for alibi the diagonal is 0 indicating no penalty for attending to that spot and increasing penalty for attending further from the diagonal
    # e.g. alibi_slope = 1, seqlen_q = 2, seqlen_k = 5, offs_m = [0, 1, 2, 3], offs_n = [0, 1, 2, 3, 4], transpose = False
    # 1. offs_m[:,None] = [[0],
    #                       [1],
    # 2. offs_m[:,None] + seqlen_k = [[5],
    #                                  [6],
    # 3. offs_m[:,None] + seqlen_k - seqlen_q = [[3],
    #                                             [4],
    # 4. offs_m[:,None] + seqlen_k - seqlen_q - offs_n[None,:] = [[3], - [[0, 1, 2, 3, 4]] =  [[ 3, 2, 1, 0,-1],
    #                                                            [4],                           [ 4, 3, 2, 1, 0]]
    # 5. -1 * alibi_slope * tl.abs(relative_pos_block) = [[ -3, -2, -1, 0,-1],
    #                                                     [ -4, -3, -2, -1, 0]],
    relative_pos_block = offs_m[:, None] + seqlen_k - seqlen_q - offs_n[None, :]
    alibi_block = -1 * alibi_slope * tl.abs(relative_pos_block)
    if transpose:
        return alibi_block.T
    else:
        return alibi_block

@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,
    k_ptrs,
    v_ptrs,
    stride_kn,
    stride_vk,
    stride_sn,
    start_m,
    seqlen_k,
    seqlen_q, 
    dropout_p,
    sd_mask_ptrs,
    dropout_mask_ptrs,
    philox_seed,
    philox_ptrs,
    block_min,
    block_max,
    offs_n_causal,
    masked_blocks,
    n_extra_tokens,
    alibi_slope,
    descale_q, 
    descale_k, 
    descale_v,
    OFFS_M: tl.constexpr,
    OFFS_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DMODEL_POW2: tl.constexpr,
    SM_SCALE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    MASK_STEPS: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    RETURN_SCORES: tl.constexpr,
    PADDED_HEAD: tl.constexpr,
    IS_FP8: tl.constexpr, 
    FP8_MAX: tl.constexpr,
):
    RCP_LN2: tl.constexpr = 1.4426950408889634

    # loop over k, v, and update accumulator

    for start_n in range(block_min, block_max, BLOCK_N):
        # For padded blocks, we will overrun the tensor size if
        # we load all BLOCK_N. For others, the blocks are all within range.
        if MASK_STEPS:
            k_offs_n = start_n + tl.arange(0, BLOCK_N)
        else:
            k_offs_n = None
        k_offs_k = None if not PADDED_HEAD else tl.arange(0, BLOCK_DMODEL_POW2)
        k = load_fn(k_ptrs, k_offs_k, k_offs_n, BLOCK_DMODEL, seqlen_k)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        # We start from end of seqlen_k so only the first iteration would need
        # to be checked for padding if it is not a multiple of block_n
        # TODO: This can be optimized to only be true for the padded block.
        if MASK_STEPS:
            # If this is the last block / iteration, we want to
            # mask if the sequence length is not a multiple of block size
            # a solution is to always do BLOCK_M // BLOCK_N + 1 steps if not is_modulo_mn.
            # last step might get wasted but that is okay. check if this masking works For
            # that case.
            if (start_n + BLOCK_N == block_max) and (n_extra_tokens != 0):
                boundary_m = tl.full([BLOCK_M], seqlen_k, dtype=tl.int32)
                size_n = start_n + OFFS_N[None, :]
                mask = size_n < boundary_m[:, None]
                qk = tl.where(mask, qk, float("-inf"))

        # compute masks
        q_mask = (OFFS_M[:, None] < seqlen_q)
        k_mask = ((start_n + tl.arange(0, BLOCK_N))[None, :] < seqlen_k)
        p_mask = q_mask & k_mask

        # -- compute qk ----
        if IS_FP8:
            qk += (tl.dot(q, k) * descale_q * descale_k)
        else:
            qk += tl.dot(q, k)
        qk_scaled =  qk * SM_SCALE
        if IS_CAUSAL:
            causal_boundary = start_n + offs_n_causal
            causal_mask = OFFS_M[:, None] >= causal_boundary[None, :]
            qk_scaled = tl.where(causal_mask, qk_scaled, float("-inf"))

        if alibi_slope is not None:
            # Compute the global position of each token within the sequence
            global_m_positions = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
            global_n_positions = start_n + tl.arange(0, BLOCK_N)
            alibi_block = compute_alibi_block(alibi_slope, seqlen_q, seqlen_k, global_m_positions,
                                              global_n_positions)
            qk_scaled += alibi_block
        # get max scores so far
        m_ij = tl.maximum(m_i, tl.max(qk_scaled, 1))

        # scale and subtract max
        q_shifted = qk_scaled - m_ij[:, None]
        
        # Compute scaled QK and softmax probabilities
        p = tl.math.exp2(q_shifted * RCP_LN2)

        # CAVEAT: Must update l_ij before applying dropout
        l_ij = tl.sum(p, 1)
        if ENABLE_DROPOUT:
            rng_output = tl.rand(philox_seed, philox_ptrs)  # TODO: use tl.randint for better performance
            dropout_mask = rng_output > dropout_p
            tl.store(dropout_mask_ptrs, dropout_mask, mask=p_mask)

            # return scores with negative values for dropped vals
            sd_mask = tl.where(dropout_mask, p, -p)
            tl.store(sd_mask_ptrs, sd_mask, mask=p_mask)

            # apply dropout mask in place
            p = tl.where(dropout_mask, p, 0.0)
        elif RETURN_SCORES:
            # NOTE: the returned score is not the same as the reference because we need to adjust as we find new maxes per block. We are not doing that
            tl.store(sd_mask_ptrs, p, mask=p_mask)
        
        # -- update output accumulator --
        # alpha is an adjustment factor for acc and li as we loop and find new maxes
        # store the diff in maxes to adjust acc and li as we discover new maxes
        m_diff = m_i - m_ij
        alpha = tl.math.exp2(m_diff * RCP_LN2)
        acc = acc * alpha[:, None]
        v = load_fn(v_ptrs, k_offs_n, k_offs_k, seqlen_k, BLOCK_DMODEL)
        # -- update m_i and l_i
        l_i = l_i * alpha + l_ij
        # update m_i and l_i
        m_i = m_ij

        if IS_FP8:
            scale_p, descale_p = compute_fp8_scaling_factors(p, FP8_MAX)
            acc += (tl.dot((p * scale_p).to(v.type.element_ty), v) * descale_p * descale_v)
        else:
            acc += tl.dot(p.to(v.type.element_ty), v)

        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vk
        if RETURN_SCORES:
            sd_mask_ptrs += BLOCK_N * stride_sn
        
        if ENABLE_DROPOUT:
            dropout_mask_ptrs += BLOCK_N * stride_sn
            philox_ptrs += BLOCK_N * stride_sn
    
    return acc, l_i, m_i


@triton.jit
def _attn_fwd(q_ptr: torch.Tensor, 
            k_ptr: torch.Tensor, 
            v_ptr: torch.Tensor,
            descale_q_ptr: torch.Tensor,
            descale_k_ptr: torch.Tensor,
            descale_v_ptr: torch.Tensor,
            out_ptr: torch.Tensor,
            alibi_slopes_ptr: torch.Tensor,
            s_dmask_ptr: torch.Tensor,
            dropout_mask_ptr: torch.Tensor,
            softmax_lse_ptr: torch.Tensor,
            stride_qz, stride_qh, stride_qm, stride_qk,
            stride_kz, stride_kh, stride_kn, stride_kk,
            stride_vz, stride_vh, stride_vn, stride_vk,
            stride_descale_q_z, stride_descale_k_z, stride_descale_v_z,
            stride_oz, stride_oh, stride_om, stride_on,
            stride_alibi_z, stride_alibi_h,
            stride_sd_z, stride_sd_h, stride_sd_m, stride_sd_n,
            stride_lse_z, stride_lse_h, stride_lse_m,
            sm_scale,
            cu_seqlens_q,
            cu_seqlens_k,
            dropout_p,
            philox_seed,
            philox_offset,
            SEQLEN_Q: tl.constexpr,
            SEQLEN_K: tl.constexpr,
            IS_CAUSAL: tl.constexpr,
            NUM_Q_HEADS: tl.constexpr,
            NUM_K_HEADS: tl.constexpr,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,
            BLOCK_DMODEL: tl.constexpr,
            BLOCK_DMODEL_POW2: tl.constexpr,
            RETURN_SCORES: tl.constexpr,
            ENABLE_DROPOUT: tl.constexpr,
            IS_FP8: tl.constexpr,
            FP8_MAX: tl.constexpr,
            VARLEN: tl.constexpr,
):
    #calculate offsets
    start_m = tl.program_id(0) #seqlen_q
    off_q_head = tl.program_id(1)  #num_q_heads
    off_z = tl.program_id(2) #batch

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL_POW2)

    if VARLEN:
        cu_seqlens_q_start = tl.load(cu_seqlens_q + off_z)
        cu_seqlens_q_end = tl.load(cu_seqlens_q + off_z + 1)

        seqlen_q = cu_seqlens_q_end - cu_seqlens_q_start
        # We have a one-size-fits-all grid in id(0). Some seqlens might be too
        # small for all start_m so for those we return early.
        if start_m * BLOCK_M > seqlen_q:
            return
        cu_seqlens_k_start = tl.load(cu_seqlens_k + off_z)
        cu_seqlens_k_end = tl.load(cu_seqlens_k + off_z + 1)
        seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start
    else:
        cu_seqlens_q_start = 0
        cu_seqlens_k_start = 0
        seqlen_q = SEQLEN_Q
        seqlen_k = SEQLEN_K

    n_blocks = cdiv_fn(seqlen_k, BLOCK_N)

    # Now we compute whether we need to exit early due to causal masking.
    # This is because for seqlen_q > seqlen_k, M rows of the attn scores
    # are completely masked, resulting in 0s written to the output, and
    # inf written to LSE. We don't need to do any GEMMs in this case.
    # This block of code determines what N is, and if this WG is operating
    # on those M rows.
    if (IS_CAUSAL):
        # If seqlen_q == seqlen_k, the attn scores are a square matrix.
        # If seqlen_q != seqlen_k, attn scores are rectangular which means
        # the causal mask boundary is bottom right aligned, and ends at either
        # the top edge (seqlen_q < seqlen_k) or left edge.

        # This captures the decrease in n_blocks if we have a rectangular attn matrix
        n_blocks_seqlen = cdiv_fn((start_m + 1) * BLOCK_M + seqlen_k - seqlen_q, BLOCK_N)

        # This is what adjusts the block_max for the current WG, only
        # if IS_CAUSAL. Otherwise we want to always iterate through all n_blocks
        n_blocks = min(n_blocks, n_blocks_seqlen)

        # If we have no blocks after adjusting for seqlen deltas, this WG is part of
        # the blocks that are all 0. We exit early.
        if n_blocks <= 0:
            offs_out = (off_z * stride_oz + 
                        off_q_head * stride_oh + 
                        cu_seqlens_q_start * stride_om +
                        offs_m[:, None] * stride_om + 
                        offs_d[None, :] * stride_on)
            acc = tl.zeros([BLOCK_M, BLOCK_DMODEL_POW2], dtype=out_ptr.type.element_ty)
            out_mask = (offs_m[:, None] < seqlen_q) & (offs_d < BLOCK_DMODEL)
            tl.store(out_ptr + offs_out, acc, mask=out_mask)

            if softmax_lse_ptr is not None:
                offs_lse = (off_z * stride_lse_z + 
                            off_q_head * stride_lse_h +
                            cu_seqlens_q_start * stride_lse_m + 
                            offs_m*stride_lse_m
                            )
                lse_mask = offs_m < SEQLEN_Q
                lse = tl.full([BLOCK_M], value=0.0, dtype=tl.float32)
                tl.store(softmax_lse_ptr + offs_lse, lse, mask=lse_mask)
                # TODO: Should dropout and return encoded softmax be handled here too?

            return

    grp_sz:tl.constexpr = NUM_Q_HEADS // NUM_K_HEADS 
    if grp_sz != 1: #Grouped Query Attention
        off_k_head = off_q_head // grp_sz 
    else: 
        off_k_head = off_q_head

    #q,k,v offsets
    q_offs = (off_z * stride_qz + 
                off_q_head * stride_qh +
                cu_seqlens_q_start * stride_qm +
                offs_m[:, None] * stride_qm + offs_d[None, :]*stride_qk
    )
    q_ptrs = q_ptr + q_offs

    k_offs = (off_z * stride_kz + 
                off_k_head * stride_kh +
                cu_seqlens_k_start * stride_kn +
                offs_d[:, None] * stride_kk + offs_n[None, :]*stride_kn
    )
    k_ptrs = k_ptr + k_offs

    v_offs = (off_z * stride_vz + 
                off_k_head * stride_vh +
                cu_seqlens_k_start * stride_vn +
                offs_n[:, None] * stride_vn + offs_d[None, :]*stride_vk
    )
    v_ptrs = v_ptr + v_offs

    #alibi slopes
    if alibi_slopes_ptr is not None:
        alibi_offs = off_z * stride_alibi_z + off_q_head * stride_alibi_h
        alibi_slope = tl.load(alibi_slopes + alibi_offs)
    else:
        alibi_slope = None

    #s_dmask (return_scores)
    if s_dmask_ptr is not None:
        s_dmask_offs =  (off_z * stride_sd_z + 
                        off_q_head * stride_sd_h + 
                        offs_m[:, None] * stride_sd_m +
                        offs_n[None, :] * stride_sd_n
        )
        s_dmask_ptrs = s_dmask_ptr + s_dmask_offs
    else:
        s_dmask_ptrs = None

    #dropout 
    if dropout_mask_ptr is not None:
        dropout_mask_offs =  (off_z * stride_sd_z + 
                        off_q_head * stride_sd_h + 
                        offs_m[:, None] * stride_sd_m +
                        offs_n[None, :] * stride_sd_n
        )
        dropout_mask_ptrs = dropout_mask_ptr + dropout_mask_offs
        philox_ptrs = (philox_offset + 
                        off_z * stride_sd_z + 
                        off_q_head * stride_sd_h  + 
                        offs_m[:, None] * stride_sd_m + 
                        offs_n[None, :] * stride_sd_n
        )
    else:
        dropout_mask_ptrs = None
        philox_ptrs = None

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL_POW2], dtype=tl.float32)
    if (BLOCK_DMODEL == BLOCK_DMODEL_POW2):
        q_mask = (offs_m[:, None] < seqlen_q) 
    else:
        q_mask = (offs_m[:, None] < seqlen_q) & (offs_d[None, :] < BLOCK_DMODEL)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)
    if IS_FP8:
        descale_q = tl.load(descale_q_ptr + off_z * stride_descale_q_z + off_q_head)
        descale_k = tl.load(descale_k_ptr + off_z * stride_descale_k_z + off_k_head)
        descale_v = tl.load(descale_v_ptr + off_z * stride_descale_v_z + off_k_head)
    else:
        descale_q, descale_k ,descale_v = 1.0, 1.0, 1.0

    n_extra_tokens = 0
    if seqlen_k < BLOCK_N:
        n_extra_tokens = BLOCK_N -seqlen_k 
    elif seqlen_k % BLOCK_N:
        n_extra_tokens = seqlen_k % BLOCK_N
    
    #if CAUSAL, then determine masked_blocks and full blocks
    # Here we compute how many full and masked blocks we have.
    padded_block_k = n_extra_tokens != 0
    is_modulo_mn = not padded_block_k and (seqlen_q % BLOCK_M == 0)
    if IS_CAUSAL:
        # There are always at least BLOCK_M // BLOCK_N masked blocks.
        # Additionally there might be one more due to dissimilar seqlens.
        masked_blocks = BLOCK_M // BLOCK_N + (not is_modulo_mn)
    else:
        # Padding on Q does not need to be masked in the FA loop.
        masked_blocks = padded_block_k
    # if IS_CAUSAL, not is_modulo_mn does not always result in an additional block.
    # In this case we might exceed n_blocks so pick the min.
    masked_blocks = min(masked_blocks, n_blocks)
    n_full_blocks = n_blocks - masked_blocks
    block_min = 0
    block_max = n_blocks * BLOCK_N
    # Compute for full blocks. Here we set causal to false regardless of its actual
    # value because there is no masking. Similarly we do not need padding.
    if n_full_blocks > 0:
        block_max = (n_blocks - masked_blocks) * BLOCK_N
        acc, l_i, m_i = _attn_fwd_inner(acc, 
                                        l_i, 
                                        m_i, 
                                        q, 
                                        k_ptrs, 
                                        v_ptrs, 
                                        stride_kn, 
                                        stride_vn, 
                                        stride_sd_n,
                                        start_m, 
                                        seqlen_k, 
                                        seqlen_q, 
                                        dropout_p, 
                                        s_dmask_ptrs, dropout_mask_ptrs, philox_seed, philox_ptrs,
                                        block_min, block_max, 0, 0, 0, alibi_slope, 
                                        descale_q, descale_k, descale_v,
                                        offs_m, offs_n, BLOCK_M, BLOCK_N, BLOCK_DMODEL,BLOCK_DMODEL_POW2,
                                        sm_scale, False, MASK_STEPS=False, ENABLE_DROPOUT=ENABLE_DROPOUT, 
                                        RETURN_SCORES=RETURN_SCORES, PADDED_HEAD=BLOCK_DMODEL!=BLOCK_DMODEL_POW2,
                                        IS_FP8=IS_FP8, FP8_MAX=FP8_MAX
                                        )
        block_min = block_max
        block_max = n_blocks * BLOCK_N

      # Remaining blocks, if any, are full / not masked.
    if (masked_blocks > 0):
        if IS_CAUSAL:
            offs_n_causal = offs_n + (seqlen_q - seqlen_k)
        else:
            offs_n_causal = 0
        k_ptrs += n_full_blocks * BLOCK_N * stride_kn
        v_ptrs += n_full_blocks * BLOCK_N * stride_vn
        if RETURN_SCORES:
            s_dmask_ptrs += n_full_blocks * BLOCK_N * stride_sd_n
        if ENABLE_DROPOUT:
            dropout_mask_ptrs += n_full_blocks * BLOCK_N * stride_sd_n
        acc, l_i, m_i = _attn_fwd_inner(acc, 
                                        l_i, 
                                        m_i, 
                                        q, 
                                        k_ptrs, 
                                        v_ptrs, 
                                        stride_kn, stride_vn, stride_sd_n,
                                        start_m, seqlen_k, seqlen_q, 
                                        dropout_p, 
                                        s_dmask_ptrs, dropout_mask_ptrs, philox_seed, philox_ptrs,
                                        block_min, block_max, offs_n_causal, masked_blocks, n_extra_tokens, alibi_slope, 
                                        descale_q, descale_k, descale_v,
                                        offs_m, offs_n, BLOCK_M, BLOCK_N, BLOCK_DMODEL,BLOCK_DMODEL_POW2,
                                        sm_scale, IS_CAUSAL, MASK_STEPS=True, ENABLE_DROPOUT=ENABLE_DROPOUT, 
                                        RETURN_SCORES=RETURN_SCORES, PADDED_HEAD=BLOCK_DMODEL!=BLOCK_DMODEL_POW2,
                                        IS_FP8=IS_FP8, FP8_MAX=FP8_MAX
                                        )
    # epilogue
    # This helps the compiler do Newton Raphson on l_i vs on acc which is much larger.
    l_recip = 1 / l_i[:, None]
    acc = acc * l_recip
    if ENABLE_DROPOUT:
        dropout_scale = 1 / (1 - dropout_p)
        acc = acc * dropout_scale
    # If seqlen_q > seqlen_k but the delta is not a multiple of BLOCK_M,
    # then we have one block with a row of all NaNs which come from computing
    # softmax over a row of all -infs (-inf - inf = NaN). We check for that here
    # and store 0s where there are NaNs as these rows should've been zeroed out.
    end_m_idx = (start_m + 1) * BLOCK_M
    start_m_idx = start_m * BLOCK_M
    causal_start_idx = seqlen_q - seqlen_k
    if IS_CAUSAL:
        if causal_start_idx > start_m_idx and causal_start_idx < end_m_idx:
            out_mask_boundary = tl.full((BLOCK_DMODEL_POW2, ), causal_start_idx, dtype=tl.int32)
            mask_m_offsets = start_m_idx + tl.arange(0, BLOCK_M)
            out_ptrs_mask = mask_m_offsets[:, None] >= out_mask_boundary[None, :]
            z = 0.0
            acc = tl.where(out_ptrs_mask, acc, z.to(acc.type.element_ty))

    # write back LSE(Log Sum Exponents), the log of the normalization constant
    overflow_size = end_m_idx - seqlen_q 
    if softmax_lse_ptr is not None: 
        RCP_LN2: tl.constexpr = 1.4426950408889634
        LN2: tl.constexpr = 0.6931471824645996
        # compute log-sum-exp in base 2 units
        mi_base2 = m_i * RCP_LN2
        softmax_lse = mi_base2 + tl.math.log2(l_i)
        # convert back to natural units
        softmax_lse *= LN2
    
        if IS_CAUSAL:
            # zero out nans caused by -infs when doing causal
            lse_causal_mask = (start_m_idx + tl.arange(0, BLOCK_M)) < causal_start_idx
            softmax_lse = tl.where(lse_causal_mask, 0.0, softmax_lse)

        # If seqlen_q not multiple of BLOCK_M, we need to mask out the last few rows.
        # This is only true for the last M block. For others, overflow_size will be -ve
        offs_lse = off_z * stride_lse_z + off_q_head * stride_lse_h +  cu_seqlens_q_start * stride_lse_m + offs_m*stride_lse_m
        if overflow_size > 0:
            boundary = tl.full((BLOCK_M, ), BLOCK_M - overflow_size, dtype=tl.int32)
            lse_mask = tl.arange(0, BLOCK_M) < boundary
            tl.store(softmax_lse_ptr + offs_lse, softmax_lse, mask=lse_mask) # the log of the normalization constant
        else:
            tl.store(softmax_lse_ptr + offs_lse, softmax_lse) # the log of the normalization constant

    # write back O
    offs_out = (off_z * stride_oz + 
                off_q_head * stride_oh + 
                cu_seqlens_q_start * stride_om +
                offs_m[:, None] * stride_om + 
                offs_d[None, :] * stride_on) 
    out_mask = tl.full([BLOCK_M, BLOCK_DMODEL_POW2], 1, dtype=tl.int1)
    if overflow_size > 0:
        out_mask = out_mask & (offs_m[:, None] < seqlen_q)
    if BLOCK_DMODEL != BLOCK_DMODEL_POW2:
        out_mask = out_mask & (offs_d[None, :] < BLOCK_DMODEL)
    op =  acc.to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + offs_out, op, mask=out_mask)

def _flash_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    alibi_slopes: Optional[torch.Tensor],
    return_lse: bool,
    return_softmax: bool,
    max_seqlen_q: int,
    max_seqlen_k: int,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    descale_q: Optional[torch.Tensor] = None,
    descale_k: Optional[torch.Tensor] = None,
    descale_v: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    #FP8
    IS_FP8 = is_fp8(q)
    FP8_MAX: tl.constexpr=torch.finfo(q.dtype).max
    is_varlen = True if cu_seqlens_q is not None else False

    if IS_FP8:
        o = torch.zeros_like(q, dtype=torch.float32) 
    else:
        o = torch.zeros_like(q)
    if is_varlen:
        #Layout for q,k,v is thd ie [total_tokens, num_head, head_dim] 
        batch, seqlen_q, num_q_heads, head_sz  = len(cu_seqlens_q) - 1, max_seqlen_q, q.shape[1], q.shape[2]
        seqlen_k, num_k_heads =  max_seqlen_k, k.shape[1] 
        q_strides = (0, q.stride(1), q.stride(0), q.stride(2))
        k_strides = (0, k.stride(1), k.stride(0), k.stride(2))
        v_strides = (0, v.stride(1), v.stride(0), v.stride(2))
        o_strides = (0, o.stride(1), o.stride(0), o.stride(2))
    else:
        #Layout for q,k,v is bshd ie [batch, seq_len, num_head, head_dim] 
        batch, seqlen_q, num_q_heads, head_sz = q.shape
        seqlen_k = k.shape[1]
        num_k_heads = k.shape[2]
        q_strides = (q.stride(0), q.stride(2), q.stride(1), q.stride(3))
        k_strides = (k.stride(0), k.stride(2), k.stride(1), k.stride(3))
        v_strides = (v.stride(0), v.stride(2), v.stride(1), v.stride(3))
        o_strides = (o.stride(0), o.stride(2), o.stride(1), o.stride(3))

    #padding for head_dim. Power of 2 or 16
    BLOCK_DMODEL_POW2 = triton.next_power_of_2(head_sz)
    BLOCK_DMODEL_POW2 = max(BLOCK_DMODEL_POW2, 16)

    #softmax_lse [batch, num_q_heads, seqlen_q]
    if return_lse:
        if is_varlen:
            softmax_lse = torch.zeros((q.shape[0], num_q_heads), device=q.device, dtype=torch.float32)
            stride_lse_z, stride_lse_h, stride_lse_m = 0, softmax_lse.stride(1), softmax_lse.stride(0)
        else:
            softmax_lse = torch.zeros((batch, num_q_heads, max_seqlen_q), device=q.device, dtype=torch.float32)
            stride_lse_z, stride_lse_h, stride_lse_m = softmax_lse.stride()
    else:
        softmax_lse = None

    #exp_scores [batch, num_q_heads, seqlen_q, seqlen_k]
    enable_dropout = dropout_p > 0.0
    if enable_dropout:
        philox_seed = torch.randint(0, 0xffffff, (1,))[0].item() #No specific reason to restrict range to 0xffffff
        philox_offset = torch.randint(0, 0xffffff, (1,))[0].item() #Pass in an int, not Tensor
    else:
        philox_seed = 0
        philox_offset = 0
    if return_softmax or enable_dropout:
        s_dmask = torch.zeros((batch, num_q_heads, max_seqlen_q, max_seqlen_k), device=q.device, dtype=torch.float32)
        dropout_mask = torch.zeros((batch, num_q_heads, max_seqlen_q, max_seqlen_k), device=q.device, dtype=torch.float32)
    else:
        s_dmask = None
        dropout_mask = None


    # Best config from ROCm/triton/python/perf-kernels/flash_attention.py::attn_fwd autotuning is BLOCK_M: 128, BLOCK_N: 64, waves_per_eu: 2, num_warps: 4, num_ctas: 1, num_stages: 1
    # Tuned for MI300x
    config = {
        'BLOCK_M': 128,
        'BLOCK_N': 32, # BLOCK_N: 64 spills for _attn_fwd
        'waves_per_eu': 2,
        'num_warps': 4,
        'num_ctas': 1,
        'num_stages': 1,
    }

    grid = lambda META:(triton.cdiv(seqlen_q, META['BLOCK_M']), num_q_heads, batch)
    _attn_fwd[grid](q,
                    k,
                    v,
                    descale_q,
                    descale_k,
                    descale_v,
                    o,
                    alibi_slopes,
                    s_dmask,
                    dropout_mask,
                    softmax_lse,
                    *q_strides,
                    *k_strides, 
                    *v_strides, 
                    descale_q.stride(0) if descale_q is not None else 0,
                    descale_k.stride(0) if descale_k is not None else 0,
                    descale_v.stride(0) if descale_v is not None else 0,
                    *o_strides,
                    alibi_slopes.stride(0) if alibi_slopes is not None else 0,
                    alibi_slopes.stride(1) if alibi_slopes is not None else 0,
                    s_dmask.stride(0) if s_dmask is not None else 0,
                    s_dmask.stride(1) if s_dmask is not None else 0,
                    s_dmask.stride(2) if s_dmask is not None else 0,
                    s_dmask.stride(3) if s_dmask is not None else 0,
                    stride_lse_z if softmax_lse is not None else 0,
                    stride_lse_h if softmax_lse is not None else 0,
                    stride_lse_m if softmax_lse is not None else 0,
                    softmax_scale, 
                    cu_seqlens_q,
                    cu_seqlens_k,
                    dropout_p,
                    philox_seed,
                    philox_offset,
                    SEQLEN_Q=max_seqlen_q,
                    SEQLEN_K=max_seqlen_k,
                    IS_CAUSAL=causal,
                    NUM_Q_HEADS=num_q_heads,
                    NUM_K_HEADS=num_k_heads,
                    BLOCK_DMODEL=head_sz,
                    BLOCK_DMODEL_POW2=BLOCK_DMODEL_POW2,
                    RETURN_SCORES=return_softmax,
                    ENABLE_DROPOUT=enable_dropout,
                    IS_FP8=IS_FP8,
                    FP8_MAX=FP8_MAX,
                    VARLEN=is_varlen,
                    **config
    )

    return o, softmax_lse, s_dmask, philox_seed, philox_offset 

# This function computes delta given output Out and gradient DO
# Here is the I/O shape:
# Out: (batch, nhead_q, max_seqlens_q, headDim)
# DO: (batch, nhead_q, max_seqlens_q, headDim)
# Delta: (batch, nheads_q, max_seqlens_q), same as softmax_lse defined at
@triton.jit
def _bwd_preprocess(
    o_ptr, do_ptr,  # noqa: E741
    delta_ptr,
    stride_o_b, stride_o_h, stride_o_m, stride_o_k,
    stride_delta_b, stride_delta_h, stride_delta_m,
    stride_descale_do_z,
    cu_seqlens_q, max_seqlen_q,
    descale_do_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_D_MODEL: tl.constexpr,
    BLOCK_D_MODEL_POW2: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_FP8: tl.constexpr
):
    pid_m = tl.program_id(0) #seqlen
    bid = tl.program_id(1) #batch
    hid = tl.program_id(2) #head

    # Handle varlen
    q_start = 0
    seqlen_q = max_seqlen_q
    if IS_VARLEN:
        q_start = tl.load(cu_seqlens_q + bid)
        q_end = tl.load(cu_seqlens_q + bid + 1)
        seqlen_q = q_end - q_start
    else:
        q_start = 0
        seqlen_q = max_seqlen_q

    # Compute offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_D_MODEL_POW2)

    # Offset O/DO by batch, head and q_start
    offs = (bid * stride_o_b +
                hid * stride_o_h +
                q_start * stride_o_m + offs_m[:, None] * stride_o_m +
                offs_k[None, :] * stride_o_k)

    # create masks
    mask_m = offs_m < seqlen_q
    mask = mask_m[:, None]
    PADDED_HEAD: tl.constexpr = (BLOCK_D_MODEL != BLOCK_D_MODEL_POW2)
    if PADDED_HEAD:
        mask &= offs_k[None, :] < BLOCK_D_MODEL

    # load [BLOCK_M, BLOCK_D_MODEL_POW2]
    o = tl.load(o_ptr + offs, mask=mask, other=0.0)
    do = tl.load(do_ptr + offs, mask=mask, other=0.0)

    # compute and write-back to delta
    if IS_FP8:
        descale_do = tl.load(descale_do_ptr + bid * stride_descale_do_z + hid)

        # NOTE: do is in the fp8 range and o is not in fp8
        delta = tl.sum(o.to(tl.float32) * (do.to(tl.float32) * descale_do), axis=1)
    else:
        delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)

    offs_delta = (bid * stride_delta_b + 
                hid * stride_delta_h + 
                q_start * stride_delta_m + offs_m * stride_delta_m)
    tl.store(delta_ptr + offs_delta, delta, mask=mask_m)

@triton.jit
def _bwd_dq_inner(
    dq,
    q, K, V, do, m, Delta, sm_scale,
    stride_qm, stride_qk, stride_kn, stride_kk, stride_vn, stride_vk,
    stride_dropout_m, stride_dropout_n,
    stride_deltam,
    seqlen_q, seqlen_k,
    dropout_p, philox_seed, batch_philox_offset, dropout_offset,
    start_m, start_n, end_n, num_steps, 
    descale_q, descale_k, descale_v, descale_do,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D_MODEL: tl.constexpr,
    BLOCK_D_MODEL_POW2: tl.constexpr,
    MASK: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    IS_FP8: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    RCP_LN2: tl.constexpr = 1.4426950408889634

    PADDED_HEAD: tl.constexpr = (BLOCK_D_MODEL != BLOCK_D_MODEL_POW2)
    delta_qk = seqlen_q - seqlen_k
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_D_MODEL_POW2)

    # mask to make sure not OOB of seqlen_q
    mask_m = offs_m < seqlen_q

    kT_ptrs = K + offs_n[None, :] * stride_kn + offs_k[:, None] * stride_kk
    vT_ptrs = V + offs_n[None, :] * stride_vn + offs_k[:, None] * stride_vk

    # D (= delta) is pre-divided by ds_scale.
    Di = tl.load(Delta + offs_m * stride_deltam, mask=mask_m, other=0.0)

    curr_n = start_n
    step_n = BLOCK_N
    curr_philox_offset = batch_philox_offset
    curr_dropout_offset = dropout_offset
    for blk_idx in range(num_steps):
        offs_n = curr_n + tl.arange(0, BLOCK_N)
        # end_n is needed because the end of causal True might not be perfectly
        # aligned with the end of the block
        mask_n = offs_n < end_n
        mask_kT = mask_n[None, :]
        mask_mn = mask_m[:, None] & (offs_n[None, :] < end_n)
        if PADDED_HEAD:
            mask_kT &= offs_k[:, None] < BLOCK_D_MODEL

        kT = tl.load(kT_ptrs, mask=mask_kT, other=0.0)
        vT = tl.load(vT_ptrs, mask=mask_kT, other=0.0)        

        #dropout
        if ENABLE_DROPOUT:
            philox_offs = (curr_philox_offset + 
                            offs_m[:, None] * stride_dropout_m +
                            offs_n[None, :] * stride_dropout_n)
            rand_vals = tl.rand(philox_seed, philox_offs)
            dropout_mask = rand_vals > dropout_p
            dropout_scale = 1 / (1 - dropout_p)

        #qk
        if IS_FP8:
            qk = tl.dot(q, kT) * descale_q * descale_k
        else:
            qk = tl.dot(q, kT)
        p = tl.math.exp2(qk * sm_scale * RCP_LN2 - m * RCP_LN2)

        if MASK:
            causal_mask = (offs_m[:, None] - delta_qk) >= offs_n[None, :]
            mask = causal_mask * mask_mn
            p = tl.where(mask, p, 0.0)

        #dp
        if IS_FP8:
            dp = (tl.dot(do, vT) * descale_do * descale_v)
        else:
            dp = tl.dot(do, vT)
        
        if ENABLE_DROPOUT:
            dp = tl.where(dropout_mask, dp, 0.0) * dropout_scale
        
        #ds
        delta_i = Di[:, None]
        ds = p * (dp - delta_i)

        #dq
        # NOTE: We need to de-scale dq in the end, because kT was pre-scaled.
        if IS_FP8:
            scale_ds, descale_ds = compute_fp8_scaling_factors(ds, FP8_MAX)
            dq += (tl.dot((ds*scale_ds).to(kT.type.element_ty), tl.trans(kT)) * descale_ds * descale_k)
        else:
            dq += tl.dot(ds.to(kT.type.element_ty), tl.trans(kT))

        curr_n += step_n
        kT_ptrs += step_n * stride_kn
        vT_ptrs += step_n * stride_vn
    return dq


@triton.jit
def _bwd_dkdv_inner(
    dk, dv,
    Q, k, v, DO, M, D, sm_scale,
    stride_q_m, stride_q_k,
    stride_do_m, stride_do_k,
    stride_dropout_m, stride_dropout_n,
    stride_deltam,
    dropout_p, philox_seed, batch_philox_offset, dropout_offset,
    seqlen_q, seqlen_k,
    start_n, start_m, num_steps,
    descale_q, descale_k, descale_v, descale_do,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D_MODEL: tl.constexpr,
    BLOCK_D_MODEL_POW2: tl.constexpr,
    MASK: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    IS_FP8: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    PADDED_HEAD: tl.constexpr = (BLOCK_D_MODEL != BLOCK_D_MODEL_POW2)
    delta_qk = seqlen_q - seqlen_k
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_D_MODEL_POW2)

    # mask to make sure not OOB of seqlen_q
    mask_n = offs_n < seqlen_k
    qT_ptrs = Q + offs_m[None, :] * stride_q_m + offs_k[:, None] * stride_q_k #[BLOCK_D_MODEL_POW2, BLOCK_M]
    do_ptrs = DO + offs_m[:, None] * stride_do_m + offs_k[None,: ] * stride_do_k
    curr_m = start_m
    step_m = BLOCK_M
    curr_philox_offset = batch_philox_offset
    curr_dropout_offset = dropout_offset
    RCP_LN2: tl.constexpr = 1.4426950408889634

    #Iterate over blocks(BLOCK_M size) of Q while calculating 
    #a fixed block(BLOCK_N) of dk and dv. Note, during backward
    #pass P has to be recomputed. However, this kernel computes 
    #dV and dK, so we compute we need P^T and S^T. See backward pass
    #equations
    # 
    #From Flash Attention Paper:
    #ForwardPass: S = QkT, P=softmax(S), O=PV
    #
    #BackwardPass equations
    #dV = P^TdO 
    #dP = dOV^T
    #dS = dsoftmax(dP)
    #dQ = dSK
    #dK = QdS^T
    for blk_idx in range(num_steps):
        offs_m = curr_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < seqlen_q
        mask_qT = mask_m[None, :]
        mask_do = mask_m[:, None]
        mask_nm = mask_n[:, None] & (offs_m[None, :] < seqlen_q)
        if PADDED_HEAD:
            mask_qT &= offs_k[:, None] < BLOCK_D_MODEL
            mask_do &= offs_k[None, :] < BLOCK_D_MODEL

        #load qT
        qT = tl.load(qT_ptrs, mask=mask_qT, other=0.0)
        
        #dropout
        if ENABLE_DROPOUT:
             # NOTE: dropout is transposed because it is used to mask pT
            philox_offs = (curr_philox_offset + 
                            offs_m[None, :] * stride_dropout_m +
                            offs_n[:, None] * stride_dropout_n)
            rand_vals = tl.rand(philox_seed, philox_offs)
            dropout_mask = rand_vals > dropout_p
            dropout_scale = 1.0 / (1 - dropout_p)

        #Load M
        m = tl.load(M + offs_m * stride_deltam, mask=mask_m, other=0.0)

        #Compute qkT
        if IS_FP8:
            qkT = (tl.dot(k, qT) * descale_q * descale_k)
        else:
            qkT = tl.dot(k, qT)
        
        #Compute pT(use m and also apply sm_scale)
        pT = tl.math.exp(qkT * sm_scale - m[None, :])

        if MASK:
            causal_mask = (offs_m[None, :] - delta_qk) >= offs_n[:, None]
            mask = causal_mask & mask_nm
            pT = tl.where(mask, pT, 0.0)

        #load DO
        do = tl.load(do_ptrs, mask=mask_do, other=0.0)
        
        #dV
        if ENABLE_DROPOUT:
            pT_dropout = tl.where(dropout_mask, pT, 0.0) * dropout_scale
            if IS_FP8:
                scale_p_dropout, descale_p_dropout = compute_fp8_scaling_factors(pT_dropout, FP8_MAX)
                dv += (tl.dot((pT_dropout * scale_p_dropout).to(do.type.element_ty), do) * descale_p_dropout * descale_do)
            else:
                dv += tl.dot(pT_dropout.to(do.type.element_ty), do)
        else:
            if IS_FP8:
                scale_pT, descale_pT = compute_fp8_scaling_factors(pT, FP8_MAX)
                dv += (tl.dot((pT * scale_pT).to(do.type.element_ty), do) * descale_pT * descale_do)
            else:
                dv += tl.dot(pT.to(do.type.element_ty), do)

        #Load delta
        Di = tl.load(D + offs_m * stride_deltam, mask=mask_m)

        #Compute dP and dS
        if IS_FP8:
            dpT = tl.dot(v, tl.trans(do)) * descale_v * descale_do
        else:
            dpT = tl.dot(v, tl.trans(do))

        if ENABLE_DROPOUT:
            dpT = tl.where(dropout_mask, dpT, 0.0) * dropout_scale

        delta_i = Di[None, :]
        dsT = pT * (dpT - delta_i)
        
        #compute dk
        if IS_FP8:
            scale_dsT, descale_dsT = compute_fp8_scaling_factors(dsT, FP8_MAX)
            dk += (tl.dot((dsT * scale_dsT).to(qT.type.element_ty), tl.trans(qT)) * descale_dsT * descale_q)
        else:
            dk += tl.dot(dsT.to(qT.type.element_ty), tl.trans(qT)) 

        #increment pointers
        curr_m += step_m
        qT_ptrs += step_m * stride_q_m
        do_ptrs += step_m * stride_do_m

    return dk, dv


@triton.jit
def _bwd_dkdvdq_inner(
    dk, dv,
    Q, k, v, DO, DQ, M, D, sm_scale,
    stride_q_m, stride_q_k,
    stride_do_m, stride_do_k,
    stride_dropout_m, stride_dropout_n,
    stride_deltam,
    dropout_p, philox_seed, batch_philox_offset, dropout_offset,
    seqlen_q, seqlen_k,
    start_n, start_m, num_steps,
    descale_q, descale_k, descale_v, descale_do,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D_MODEL: tl.constexpr,
    BLOCK_D_MODEL_POW2: tl.constexpr,
    MASK: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    IS_FP8: tl.constexpr,
    FP8_MAX: tl.constexpr,
    workgroup_id: tl.int32,
):
    PADDED_HEAD: tl.constexpr = (BLOCK_D_MODEL != BLOCK_D_MODEL_POW2)
    delta_qk = seqlen_q - seqlen_k
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_D_MODEL_POW2)

    # mask to make sure not OOB of seqlen_q
    mask_n = offs_n < seqlen_k
    
    qT_ptrs_start = Q + offs_m[None, :] * stride_q_m + offs_k[:, None] * stride_q_k #[BLOCK_D_MODEL_POW2, BLOCK_M]
    dq_ptrs_start = DQ + offs_m[:, None] * stride_q_m + offs_k[None,:] * stride_q_k #[BLOCK_M, BLOCK_D_MODEL_POW2]
    
    do_ptrs_start = DO + offs_m[:, None] * stride_do_m + offs_k[None,: ] * stride_do_k
    curr_m = start_m
    step_m = BLOCK_M
    curr_philox_offset = batch_philox_offset
    curr_dropout_offset = dropout_offset
    RCP_LN2: tl.constexpr = 1.4426950408889634

    #Iterate over blocks(BLOCK_M size) of Q while calculating 
    #a fixed block(BLOCK_N) of dk and dv. Note, during backward
    #pass P has to be recomputed. However, this kernel computes 
    #dV and dK, so we compute we need P^T and S^T. See backward pass
    #equations
    # 
    #From Flash Attention Paper:
    #ForwardPass: S = QkT, P=softmax(S), O=PV
    #
    #BackwardPass equations
    #dV = P^TdO 
    #dP = dOV^T
    #dS = dsoftmax(dP)
    #dQ = dSK
    #dK = QdS^T

    # Compute a starting index and step based on workgroup_id
    # Use a simple hash-like function to spread out the starting points
    start_idx = (workgroup_id * 17) % num_steps  # 17 is an arbitrary prime to spread indices
    # Ensure step is coprime with num_steps to visit all indices exactly once
    step = 1 # 3 if num_steps > 1 or num_steps==3 else 1 # coprime with num_steps


    for iter in range(num_steps):
        # Compute the permuted block index
        blk_idx = (start_idx + iter * step) % num_steps

        curr_m = start_m + blk_idx * step_m
        qT_ptrs = qT_ptrs_start + blk_idx * step_m * stride_q_m
        dq_ptrs = dq_ptrs_start + blk_idx * step_m * stride_q_m
        do_ptrs = do_ptrs_start + blk_idx * step_m * stride_do_m

        offs_m = curr_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < seqlen_q
        mask_qT = mask_m[None, :]
        mask_do = mask_m[:, None]
        mask_nm = mask_n[:, None] & (offs_m[None, :] < seqlen_q)
        
        if PADDED_HEAD:
            mask_qT &= offs_k[:, None] < BLOCK_D_MODEL
            mask_do &= offs_k[None, :] < BLOCK_D_MODEL

        #load qT
        qT = tl.load(qT_ptrs, mask=mask_qT, other=0.0)
        
        #dropout
        if ENABLE_DROPOUT:
             # NOTE: dropout is transposed because it is used to mask pT
            philox_offs = (curr_philox_offset + 
                            offs_m[None, :] * stride_dropout_m +
                            offs_n[:, None] * stride_dropout_n)
            rand_vals = tl.rand(philox_seed, philox_offs)
            dropout_mask = rand_vals > dropout_p
            dropout_scale = 1.0 / (1 - dropout_p)

        #Load M
        m = tl.load(M + offs_m * stride_deltam, mask=mask_m, other=0.0)

        #Compute qkT
        if IS_FP8:
            qkT = (tl.dot(k, qT) * descale_q * descale_k)
        else:
            qkT = tl.dot(k, qT)
        
        #Compute pT(use m and also apply sm_scale)
        pT = tl.math.exp(qkT * sm_scale - m[None, :])

        if MASK:
            causal_mask = (offs_m[None, :] - delta_qk) >= (offs_n[:, None])
            mask = causal_mask & mask_nm
            pT = tl.where(mask, pT, 0.0)

        #load DO
        do = tl.load(do_ptrs, mask=mask_do, other=0.0)
        
        #dV
        if ENABLE_DROPOUT:
            pT_dropout = tl.where(dropout_mask, pT, 0.0) * dropout_scale
            if IS_FP8:
                scale_p_dropout, descale_p_dropout = compute_fp8_scaling_factors(pT_dropout, FP8_MAX)
                dv += (tl.dot((pT_dropout * scale_p_dropout).to(do.type.element_ty), do) * descale_p_dropout * descale_do)
            else:
                dv += tl.dot(pT_dropout.to(do.type.element_ty), do)
        else:
            if IS_FP8:
                scale_pT, descale_pT = compute_fp8_scaling_factors(pT, FP8_MAX)
                dv += (tl.dot((pT * scale_pT).to(do.type.element_ty), do) * descale_pT * descale_do)
            else:
                dv += tl.dot(pT.to(do.type.element_ty), do)

        #Load delta
        Di = tl.load(D + offs_m * stride_deltam, mask=mask_m)

        #Compute dP and dS
        if IS_FP8:
            dpT = tl.dot(v, tl.trans(do)) * descale_v * descale_do
        else:
            dpT = tl.dot(v, tl.trans(do))

        if ENABLE_DROPOUT:
            dpT = tl.where(dropout_mask, dpT, 0.0) * dropout_scale

        delta_i = Di[None, :]
        dsT = pT * (dpT - delta_i)
        
        #compute dk
        if IS_FP8:
            scale_dsT, descale_dsT = compute_fp8_scaling_factors(dsT, FP8_MAX)
            dk += (tl.dot((dsT * scale_dsT).to(qT.type.element_ty), tl.trans(qT)) * descale_dsT * descale_q)
        else:
            dk += tl.dot(dsT.to(qT.type.element_ty), tl.trans(qT)) 


        # We can compute the dq_partial here and do a atomic add to the correct memory location
        # NOTE: Possible problems with the atomic add: contention, is inside a loop which has achieved bad perf before
        # (BLOCK_M, BLOCK_N) x (BLOCK_N, D)
        if IS_FP8:
            dq_partial = tl.dot((dsT * scale_dsT).to(k.dtype).T, k) * descale_dsT * descale_k
        else:
            dq_partial = tl.dot(dsT.to(k.dtype).T, k) 
        tl.atomic_add(
            dq_ptrs,
            dq_partial * sm_scale,
            mask=mask_m[:, None],
            sem="relaxed",
        )

    return dk, dv


@triton.jit
def _bwd_kernel_dkdvdq_causal(
    q_ptr, k_ptr, v_ptr, sm_scale, do_ptr, dk_ptr, dv_ptr, dq_ptr,
    m_ptr, delta_ptr,
    stride_q_b, stride_q_h, stride_q_m, stride_q_k,
    stride_k_b, stride_k_h, stride_k_n, stride_k_k,
    stride_v_b, stride_v_h, stride_v_n, stride_v_k,
    stride_dk_b, stride_dk_h, stride_dk_n, stride_dk_k,
    stride_delta_b, stride_delta_h, stride_delta_m,
    stride_do_b, stride_do_h, stride_do_m, stride_do_k,
    stride_dropout_b, stride_dropout_h, stride_dropout_m, stride_dropout_n,
    stride_descale_q_z, stride_descale_k_z, stride_descale_v_z, stride_descale_do_z,
    cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k,
    dropout_mask, dropout_p, philox_seed, philox_offset_base,
    descale_q_ptr, descale_k_ptr, descale_v_ptr, descale_do_ptr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    BATCH,
    NUM_K_PIDS, 
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLK_SLICE_FACTOR: tl.constexpr,
    BLOCK_D_MODEL: tl.constexpr,
    BLOCK_D_MODEL_POW2: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_FP8: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    wid = tl.program_id(0) # workgoup id: 0, ..., NUM_K_PIDS * BATCH * NUM_K_HEADS - 1

    # workgroups get launched first along batch dim, then in head_k dim, and then in seq k block dim
    batch_idx = wid % BATCH 
    head_k_idx = wid // BATCH % NUM_K_HEADS 
    seq_k_blk_idx = wid // (BATCH * NUM_K_HEADS) % NUM_K_PIDS

    #Determine q and k start along with seqlen_q and seqlen_k
    q_start = 0
    k_start = 0
    seqlen_q = max_seqlen_q
    seqlen_k = max_seqlen_k
    if IS_VARLEN:
        q_start = tl.load(cu_seqlens_q + batch_idx)
        q_end = tl.load(cu_seqlens_q + batch_idx + 1)
        k_start = tl.load(cu_seqlens_k + batch_idx)
        k_end = tl.load(cu_seqlens_k + batch_idx + 1)
        seqlen_q = q_end - q_start
        seqlen_k = k_end - k_start

    dk = tl.zeros([BLOCK_N, BLOCK_D_MODEL_POW2], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_D_MODEL_POW2], dtype=tl.float32)

    # Figure out causal starting block since we have seqlen_q >=< seqlen_k.
    # Unlike forward pass where we tile on M dim and iterate on N dim, so that
    # we can skip some M blocks, in backward pass, we tile on the N dim for kv
    # and iterate over the M. In this way, we cannot skip N blocks, but only to
    # determine the starting M blocks to skip some initial blocks masked by
    # causal.
    delta_qk = seqlen_q - seqlen_k

    # q > k: diretcly skip all the way until the start of causal block
    start_delta_q_gt_k = delta_qk

    # q < k: some blocks will have no Masked block, other needs to re-calc
    # starting position
    # delta_qk is negative so flip it, only multiple of BLOCK_N can skip the
    # masked op
    num_blocks_skip = -delta_qk // BLOCK_N
    delta_aligned = (num_blocks_skip + 1) * BLOCK_N + delta_qk
    start_delta_q_lt_k = delta_aligned // BLOCK_M * BLOCK_M
    if delta_qk >= 0:
        start_delta = delta_qk
    else:
        start_delta = start_delta_q_lt_k
    
    start_n =  seq_k_blk_idx * BLOCK_N

    offs_k = tl.arange(0, BLOCK_D_MODEL_POW2)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    # Mask for loading K and V
    mask_kv = offs_n[:, None] < seqlen_k
    PADDED_HEAD: tl.constexpr = (BLOCK_D_MODEL != BLOCK_D_MODEL_POW2)
    if PADDED_HEAD:
        mask_k = offs_k < BLOCK_D_MODEL
        mask_kv &= mask_k[None, :]
    
    GROUP_SIZE = NUM_Q_HEADS // NUM_K_HEADS
    adj_k = (batch_idx * stride_k_b + 
            head_k_idx * stride_k_h + 
            k_start * stride_k_n + offs_n[:, None] * stride_k_n + 
            offs_k[None, :] * stride_k_k)
    adj_v = (batch_idx * stride_v_b + 
            head_k_idx * stride_v_h + 
            k_start * stride_v_n + offs_n[:, None] * stride_v_n + 
            offs_k[None, :] * stride_v_k)
    # load K and V: they stay in SRAM throughout the inner loop.
    k = tl.load(k_ptr + adj_k , mask=mask_kv, other=0.0)
    v = tl.load(v_ptr + adj_v, mask=mask_kv, other=0.0) 

    # If MQA / GQA, set the K and V head offsets appropriately.
    for head_q_idx in range(head_k_idx * GROUP_SIZE, head_k_idx * GROUP_SIZE + GROUP_SIZE):
        if delta_qk >= 0:
            start_m = start_n + start_delta
            len_m = BLOCK_N
        else:
            start_m = max(start_n + delta_qk, 0)
            start_m = (start_m // BLOCK_M) * BLOCK_M
            # because we might shift the masked blocks up, we are deeper into
            # the masked out region, so we would potentially increase the total
            # steps with masked operation to get out of it
            residue_m = max(start_n + delta_qk - start_m, 0)
            len_m = BLOCK_N + residue_m

        # offset input and output tensor by batch and Q/K heads
        adj_q = batch_idx * stride_q_b + head_q_idx * stride_q_h + q_start * stride_q_m
        
        q_ptr_adj = q_ptr + adj_q
        dq_ptr_adj = dq_ptr + adj_q
        
        adj_do = batch_idx * stride_do_b + head_q_idx * stride_do_h + q_start * stride_do_m
        do_ptr_adj = do_ptr + adj_do
        adj_delta = batch_idx * stride_delta_b + head_q_idx * stride_delta_h + q_start * stride_delta_m
        m_ptr_adj = m_ptr + adj_delta
        delta_ptr_adj = delta_ptr + adj_delta

        # batch_philox_offset is the ACTUALLY dropout offset
        # dropout_offset is for debug purpose and will be removed later
        batch_philox_offset = 0
        dropout_offset = 0
        if ENABLE_DROPOUT:
            batch_philox_offset = (philox_offset_base + batch_idx * stride_dropout_b + 
                                  head_q_idx * stride_dropout_h)
            dropout_offset = (dropout_mask + batch_idx * stride_dropout_b + 
                             head_q_idx * stride_dropout_h)

        MASK_BLOCK_M: tl.constexpr = BLOCK_M // BLK_SLICE_FACTOR
        # bound the masked operation to q len so it does not have to wast cycles
        len_m = min(len_m, seqlen_q)
        num_steps = tl.cdiv(len_m, MASK_BLOCK_M)
        
        
        # when q < k, we may skip the initial masked op
        # if seq_k_blk_idx < num_blocks_skip:
        #     num_steps = 0

        if IS_FP8:
            descale_q = tl.load(descale_q_ptr + batch_idx * stride_descale_q_z + head_q_idx)
            descale_k = tl.load(descale_k_ptr + batch_idx * stride_descale_k_z + head_k_idx)
            descale_v = tl.load(descale_v_ptr + batch_idx * stride_descale_v_z + head_k_idx)
            descale_do = tl.load(descale_do_ptr + batch_idx * stride_descale_do_z + head_q_idx)
        else:
            descale_q, descale_k, descale_v, descale_do = 1.0, 1.0, 1.0, 1.0

        # if start_m is negative, the current N-tile has no block on the
        #   diagonal of causal mask, so everything have no causal mask
        dk, dv = _bwd_dkdvdq_inner(
            dk, dv,  # output tensors
            q_ptr_adj, k, v, do_ptr_adj, dq_ptr_adj, m_ptr_adj, delta_ptr_adj, sm_scale, # input tensors
            stride_q_m, stride_q_k,  # strides for q
            stride_do_m, stride_do_k,  # strides for o
            stride_dropout_m, stride_dropout_n,  # strides for dropout
            stride_delta_m,
            dropout_p, philox_seed, batch_philox_offset, dropout_offset,  #
            seqlen_q, seqlen_k,  # max sequence length for q and k
            start_n, start_m, num_steps,  # iteration numbers
            descale_q, descale_k, descale_v, descale_do, # fp8 descale factors from user 
            MASK_BLOCK_M, BLOCK_N,  # block dim
            BLOCK_D_MODEL, BLOCK_D_MODEL_POW2,  # head dim
            MASK=True,  # causal masking
            ENABLE_DROPOUT=ENABLE_DROPOUT,  # activate dropout
            IS_FP8=IS_FP8,
            FP8_MAX=FP8_MAX,
            workgroup_id=seq_k_blk_idx,
        )
        start_m += num_steps * MASK_BLOCK_M
        num_steps = tl.cdiv(seqlen_q - start_m, BLOCK_M)
        end_m = start_m + num_steps * BLOCK_M

        dk, dv = _bwd_dkdvdq_inner(
            dk, dv,  # output tensors
            q_ptr_adj, k, v, do_ptr_adj, dq_ptr_adj, m_ptr_adj, delta_ptr_adj, sm_scale, # input tensors
            stride_q_m, stride_q_k,  # strides for q
            stride_do_m, stride_do_k,  # strides for o
            stride_dropout_m, stride_dropout_n,  # strides for dropout
            stride_delta_m,
            dropout_p, philox_seed, batch_philox_offset, dropout_offset,  #
            seqlen_q, seqlen_k,  # max sequence length for q and k
            start_n, start_m, num_steps,  # iteration numbers
            descale_q, descale_k, descale_v, descale_do, # fp8 descale factors from user
            BLOCK_M, BLOCK_N,  # block dim
            BLOCK_D_MODEL, BLOCK_D_MODEL_POW2,  # head dim
            MASK=False,  # causal masking
            ENABLE_DROPOUT=ENABLE_DROPOUT,  # activate dropout
            IS_FP8=IS_FP8,
            FP8_MAX=FP8_MAX,
            workgroup_id=seq_k_blk_idx,
        )

    # Write back dV and dK.
    offs_dkdv = (batch_idx * stride_dk_b + 
                head_k_idx * stride_dk_h + 
                k_start * stride_dk_n + offs_n[:, None] * stride_dk_n + 
                offs_k[None, :] * stride_dk_k)
    tl.store(dv_ptr + offs_dkdv, dv, mask=mask_kv)
    dk *= sm_scale
    tl.store(dk_ptr + offs_dkdv, dk, mask=mask_kv)


@triton.jit
def _bwd_kernel_dkdv_causal(
    q_ptr, k_ptr, v_ptr, sm_scale, do_ptr, dk_ptr, dv_ptr,
    m_ptr, delta_ptr,
    stride_q_b, stride_q_h, stride_q_m, stride_q_k,
    stride_k_b, stride_k_h, stride_k_n, stride_k_k,
    stride_v_b, stride_v_h, stride_v_n, stride_v_k,
    stride_dk_b, stride_dk_h, stride_dk_n, stride_dk_k,
    stride_delta_b, stride_delta_h, stride_delta_m,
    stride_do_b, stride_do_h, stride_do_m, stride_do_k,
    stride_dropout_b, stride_dropout_h, stride_dropout_m, stride_dropout_n,
    stride_descale_q_z, stride_descale_k_z, stride_descale_v_z, stride_descale_do_z,
    cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k,
    dropout_mask, dropout_p, philox_seed, philox_offset_base,
    descale_q_ptr, descale_k_ptr, descale_v_ptr, descale_do_ptr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLK_SLICE_FACTOR: tl.constexpr,
    BLOCK_D_MODEL: tl.constexpr,
    BLOCK_D_MODEL_POW2: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_FP8: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    #seq block, batch, head_k
    seq_k_blk_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)
    head_k_idx = tl.program_id(2)

    #Determine q and k start along with seqlen_q and seqlen_k
    q_start = 0
    k_start = 0
    seqlen_q = max_seqlen_q
    seqlen_k = max_seqlen_k
    if IS_VARLEN:
        q_start = tl.load(cu_seqlens_q + batch_idx)
        q_end = tl.load(cu_seqlens_q + batch_idx + 1)
        k_start = tl.load(cu_seqlens_k + batch_idx)
        k_end = tl.load(cu_seqlens_k + batch_idx + 1)
        seqlen_q = q_end - q_start
        seqlen_k = k_end - k_start

    dk = tl.zeros([BLOCK_N, BLOCK_D_MODEL_POW2], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_D_MODEL_POW2], dtype=tl.float32)

    # Figure out causal starting block since we have seqlen_q >=< seqlen_k.
    # Unlike forward pass where we tile on M dim and iterate on N dim, so that
    # we can skip some M blocks, in backward pass, we tile on the N dim for kv
    # and iterate over the M. In this way, we cannot skip N blocks, but only to
    # determine the starting M blocks to skip some initial blocks masked by
    # causal.
    delta_qk = seqlen_q - seqlen_k

    # q > k: diretcly skip all the way until the start of causal block
    start_delta_q_gt_k = delta_qk

    # q < k: some blocks will have no Masked block, other needs to re-calc
    # starting position
    # delta_qk is negative so flip it, only multiple of BLOCK_N can skip the
    # masked op
    num_blocks_skip = -delta_qk // BLOCK_N
    delta_aligned = (num_blocks_skip + 1) * BLOCK_N + delta_qk
    start_delta_q_lt_k = delta_aligned // BLOCK_M * BLOCK_M
    if delta_qk >= 0:
        start_delta = delta_qk
    else:
        start_delta = start_delta_q_lt_k
    
    start_n =  seq_k_blk_idx *BLOCK_N

    offs_k = tl.arange(0, BLOCK_D_MODEL_POW2)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    # Mask for loading K and V
    mask_kv = offs_n[:, None] < seqlen_k
    PADDED_HEAD: tl.constexpr = (BLOCK_D_MODEL != BLOCK_D_MODEL_POW2)
    if PADDED_HEAD:
        mask_k = offs_k < BLOCK_D_MODEL
        mask_kv &= mask_k[None, :]
    
    GROUP_SIZE = NUM_Q_HEADS // NUM_K_HEADS
    adj_k = (batch_idx * stride_k_b + 
            head_k_idx * stride_k_h + 
            k_start * stride_k_n + offs_n[:, None] * stride_k_n + 
            offs_k[None, :] * stride_k_k)
    adj_v = (batch_idx * stride_v_b + 
            head_k_idx * stride_v_h + 
            k_start * stride_v_n + offs_n[:, None] * stride_v_n + 
            offs_k[None, :] * stride_v_k)
    # load K and V: they stay in SRAM throughout the inner loop.
    k = tl.load(k_ptr + adj_k , mask=mask_kv, other=0.0)
    v = tl.load(v_ptr + adj_v, mask=mask_kv, other=0.0) 

    # If MQA / GQA, set the K and V head offsets appropriately.
    for head_q_idx in range(head_k_idx * GROUP_SIZE, head_k_idx * GROUP_SIZE + GROUP_SIZE):
        if delta_qk >= 0:
            start_m = start_n + start_delta
            len_m = BLOCK_N
        else:
            start_m = max(start_n + delta_qk, 0)
            start_m = start_m // BLOCK_M * BLOCK_M
            # because we might shift the masked blocks up, we are deeper into
            # the masked out region, so we would potentially increase the total
            # steps with masked operation to get out of it
            residue_m = max(start_n + delta_qk - start_m, 0)
            len_m = BLOCK_N + residue_m

        # offset input and output tensor by batch and Q/K heads
        adj_q = batch_idx * stride_q_b + head_q_idx * stride_q_h + q_start * stride_q_m
        q_ptr_adj = q_ptr + adj_q
        adj_do = batch_idx * stride_do_b + head_q_idx * stride_do_h + q_start * stride_do_m
        do_ptr_adj = do_ptr + adj_do
        adj_delta = batch_idx * stride_delta_b + head_q_idx * stride_delta_h + q_start * stride_delta_m
        m_ptr_adj = m_ptr + adj_delta
        delta_ptr_adj = delta_ptr + adj_delta

        # batch_philox_offset is the ACTUALLY dropout offset
        # dropout_offset is for debug purpose and will be removed later
        batch_philox_offset = 0
        dropout_offset = 0
        if ENABLE_DROPOUT:
            batch_philox_offset = (philox_offset_base + batch_idx * stride_dropout_b + 
                                  head_q_idx * stride_dropout_h)
            dropout_offset = (dropout_mask + batch_idx * stride_dropout_b + 
                             head_q_idx * stride_dropout_h)

        MASK_BLOCK_M: tl.constexpr = BLOCK_M // BLK_SLICE_FACTOR
        # bound the masked operation to q len so it does not have to wast cycles
        len_m = min(len_m, seqlen_q)
        num_steps = tl.cdiv(len_m, MASK_BLOCK_M)
        # when q < k, we may skip the initial masked op
        if seq_k_blk_idx < num_blocks_skip:
            num_steps = 0

        if IS_FP8:
            descale_q = tl.load(descale_q_ptr + batch_idx * stride_descale_q_z + head_q_idx)
            descale_k = tl.load(descale_k_ptr + batch_idx * stride_descale_k_z + head_k_idx)
            descale_v = tl.load(descale_v_ptr + batch_idx * stride_descale_v_z + head_k_idx)
            descale_do = tl.load(descale_do_ptr + batch_idx * stride_descale_do_z + head_q_idx)
        else:
            descale_q, descale_k, descale_v, descale_do = 1.0, 1.0, 1.0, 1.0

        # if start_m is negative, the current N-tile has no block on the
        #   diagonal of causal mask, so everything have no causal mask
        dk, dv = _bwd_dkdv_inner(
            dk, dv,  # output tensors
            q_ptr_adj, k, v, do_ptr_adj, m_ptr_adj, delta_ptr_adj, sm_scale, # input tensors
            stride_q_m, stride_q_k,  # strides for q
            stride_do_m, stride_do_k,  # strides for o
            stride_dropout_m, stride_dropout_n,  # strides for dropout
            stride_delta_m,
            dropout_p, philox_seed, batch_philox_offset, dropout_offset,  #
            seqlen_q, seqlen_k,  # max sequence length for q and k
            start_n, start_m, num_steps,  # iteration numbers
            descale_q, descale_k, descale_v, descale_do, # fp8 descale factors from user 
            MASK_BLOCK_M, BLOCK_N,  # block dim
            BLOCK_D_MODEL, BLOCK_D_MODEL_POW2,  # head dim
            MASK=True,  # causal masking
            ENABLE_DROPOUT=ENABLE_DROPOUT,  # activate dropout
            IS_FP8=IS_FP8,
            FP8_MAX=FP8_MAX,
        )
        start_m += num_steps * MASK_BLOCK_M
        num_steps = tl.cdiv(seqlen_q - start_m, BLOCK_M)
        end_m = start_m + num_steps * BLOCK_M

        dk, dv = _bwd_dkdv_inner(
            dk, dv,  # output tensors
            q_ptr_adj, k, v, do_ptr_adj, m_ptr_adj, delta_ptr_adj, sm_scale, # input tensors
            stride_q_m, stride_q_k,  # strides for q
            stride_do_m, stride_do_k,  # strides for o
            stride_dropout_m, stride_dropout_n,  # strides for dropout
            stride_delta_m,
            dropout_p, philox_seed, batch_philox_offset, dropout_offset,  #
            seqlen_q, seqlen_k,  # max sequence length for q and k
            start_n, start_m, num_steps,  # iteration numbers
            descale_q, descale_k, descale_v, descale_do, # fp8 descale factors from user
            BLOCK_M, BLOCK_N,  # block dim
            BLOCK_D_MODEL, BLOCK_D_MODEL_POW2,  # head dim
            MASK=False,  # causal masking
            ENABLE_DROPOUT=ENABLE_DROPOUT,  # activate dropout
            IS_FP8=IS_FP8,
            FP8_MAX=FP8_MAX,
        )

    # Write back dV and dK.
    offs_dkdv = (batch_idx * stride_dk_b + 
                head_k_idx * stride_dk_h + 
                k_start * stride_dk_n + offs_n[:, None] * stride_dk_n + 
                offs_k[None, :] * stride_dk_k)
    tl.store(dv_ptr + offs_dkdv, dv, mask=mask_kv)
    dk *= sm_scale
    tl.store(dk_ptr + offs_dkdv, dk, mask=mask_kv)

@triton.jit
def _bwd_kernel_dq_causal(
    q_ptr, k_ptr, v_ptr, sm_scale, do_ptr, dq_ptr,
    m_ptr, delta_ptr,
    stride_q_b, stride_q_h, stride_q_m, stride_q_k,
    stride_k_b, stride_k_h, stride_k_n, stride_k_k,
    stride_v_b, stride_v_h, stride_v_n, stride_v_k,
    stride_dq_b, stride_dq_h, stride_dq_m, stride_dq_k,
    stride_delta_b, stride_delta_h, stride_delta_m,
    stride_do_b, stride_do_h, stride_do_m, stride_do_k,
    stride_dropout_b, stride_dropout_h, stride_dropout_m, stride_dropout_n,
    stride_descale_q_z, stride_descale_k_z, stride_descale_v_z, stride_descale_do_z,
    cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k,
    dropout_mask, dropout_p, philox_seed, philox_offset_base,
    descale_q_ptr, descale_k_ptr, descale_v_ptr, descale_do_ptr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLK_SLICE_FACTOR: tl.constexpr,
    BLOCK_D_MODEL: tl.constexpr,
    BLOCK_D_MODEL_POW2: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_FP8: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    seq_q_blk_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)
    head_k_idx = tl.program_id(2)

    q_start = 0
    k_start = 0
    seqlen_q = max_seqlen_q
    seqlen_k = max_seqlen_k
    if IS_VARLEN:
        q_start = tl.load(cu_seqlens_q + batch_idx)
        q_end = tl.load(cu_seqlens_q + batch_idx + 1)
        k_start = tl.load(cu_seqlens_k + batch_idx)
        k_end = tl.load(cu_seqlens_k + batch_idx + 1)
        seqlen_q = q_end - q_start
        seqlen_k = k_end - k_start
    
    # Figure out causal starting block since we have seqlen_q <=> seqlen_k.
    # Unlike forward pass where we tile on M dim and iterate on N dim, so that
    # we can skip some M blocks, in backward pass, we tile on the N dim for kv
    # and iterate over the M. In this way, we cannot skip N blocks, but only to
    # determine the starting M blocks to skip some initial blocks masked by
    # causal.
    # DQ tiles on M dim and iterate on N dim, so we there could be some tiles we
    # can simply skip and we need to adjust starting position.
    start_m = seq_q_blk_idx * BLOCK_M
    # seqlen_q > seqlen_k, no need to process these tile for dq
    delta_qk = seqlen_q - seqlen_k
    if start_m + BLOCK_M < delta_qk:
        return
    
    offs_k = tl.arange(0, BLOCK_D_MODEL_POW2)
    offs_m = start_m + tl.arange(0, BLOCK_M)
    # Mask for loading K and V
    mask_q = offs_m[:, None] < seqlen_q
    PADDED_HEAD: tl.constexpr = (BLOCK_D_MODEL != BLOCK_D_MODEL_POW2)
    if PADDED_HEAD:
        mask_k = offs_k < BLOCK_D_MODEL
        mask_q &= mask_k[None, :]
    offs_q = offs_m[:, None] * stride_q_m + offs_k[None, :] * stride_q_k
    offs_do = offs_m[:, None] * stride_do_m + offs_k[None, :] * stride_do_k
    adj_k = batch_idx * stride_k_b + head_k_idx * stride_k_h + k_start * stride_k_n
    adj_v = batch_idx * stride_v_b + head_k_idx * stride_v_h + k_start * stride_v_n
    k_ptr_adj = k_ptr
    v_ptr_adj = v_ptr
    k_ptr_adj +=  adj_k
    v_ptr_adj +=  adj_v

    # If MQA / GQA, set the K and V head offsets appropriately.
    GROUP_SIZE = NUM_Q_HEADS // NUM_K_HEADS
    for head_q_idx in range(head_k_idx * GROUP_SIZE, head_k_idx * GROUP_SIZE + GROUP_SIZE):
        # seqlen_q < seqlen_k: delta_qk more kv tokens are added at the front
        #   for every M-tile
        end_n = start_m + BLOCK_M - delta_qk
        # clamp end_n at [0, seqlen_k]
        end_n = max(min(end_n, seqlen_k), 0)

        # offset input and output tensor by batch and Q/K heads
        adj_q = (batch_idx * stride_q_b + 
                head_q_idx * stride_q_h + 
                q_start * stride_q_m)
        adj_do = (batch_idx * stride_do_b + 
                head_q_idx * stride_do_h + 
                q_start * stride_do_m)
        adj_delta = (batch_idx * stride_delta_b + 
                    head_q_idx * stride_delta_h + 
                    q_start * stride_delta_m)
        delta_ptr_adj = delta_ptr + adj_delta

        # batch_philox_offset is the ACTUALLY dropout offset
        # dropout_offset is for debug purpose and will be removed later
        batch_philox_offset = 0
        dropout_offset = 0
        if ENABLE_DROPOUT:
            batch_philox_offset = (philox_offset_base + 
                                  batch_idx * stride_dropout_b + 
                                  head_q_idx * stride_dropout_h)
            dropout_offset = (dropout_mask + 
                            batch_idx * stride_dropout_b + 
                            head_q_idx * stride_dropout_h)

        q = tl.load(q_ptr + adj_q + offs_q, mask=mask_q, other=0.0)
        do = tl.load(do_ptr + adj_do + offs_do, mask=mask_q, other=0.0)
        m = tl.load(m_ptr + adj_delta + offs_m * stride_delta_m,
                    mask=offs_m < seqlen_q)
        m = m[:, None]

        MASK_BLOCK_N: tl.constexpr = BLOCK_N // BLK_SLICE_FACTOR
        # start can only be 0 at minimum
        start_n = max(end_n - BLOCK_M, 0)
        num_steps = tl.cdiv(end_n - start_n, MASK_BLOCK_N)

        if IS_FP8:
            descale_q = tl.load(descale_q_ptr + batch_idx * stride_descale_q_z + head_q_idx)
            descale_k = tl.load(descale_k_ptr + batch_idx * stride_descale_k_z + head_k_idx)
            descale_v = tl.load(descale_v_ptr + batch_idx * stride_descale_v_z + head_k_idx)
            descale_do = tl.load(descale_do_ptr + batch_idx * stride_descale_do_z + head_q_idx)
        else:
            descale_q, descale_k, descale_v, descale_do = 1.0, 1.0, 1.0, 1.0

        dq = tl.zeros([BLOCK_M, BLOCK_D_MODEL_POW2], dtype=tl.float32)
        # Compute dQ for masked (diagonal) blocks.
        # NOTE: This code scans each row of QK^T backward (from right to left,
        # but inside each call to _bwd_dq_inner, from left to right), but that's
        # not due to anything important.  I just wanted to reuse the loop
        # structure for dK & dV above as much as possible.
        dq = _bwd_dq_inner(
            dq,
            q, k_ptr_adj, v_ptr_adj, do, m, delta_ptr_adj, sm_scale,
            stride_q_m, stride_q_k, stride_k_n, stride_k_k, stride_v_n, stride_v_k,
            stride_dropout_m, stride_dropout_n,
            stride_delta_m,
            seqlen_q, seqlen_k,
            dropout_p, philox_seed, batch_philox_offset, dropout_offset,
            start_m, start_n, end_n, num_steps,
            descale_q, descale_k, descale_v, descale_do,
            BLOCK_M, MASK_BLOCK_N,
            BLOCK_D_MODEL, BLOCK_D_MODEL_POW2,
            MASK=True,
            ENABLE_DROPOUT=ENABLE_DROPOUT,
            IS_FP8=IS_FP8,
            FP8_MAX=FP8_MAX,
        )
        end_n -= num_steps * MASK_BLOCK_N
        num_steps = tl.cdiv(end_n, BLOCK_N)
        start_n = max(end_n - num_steps * BLOCK_N, 0)
        dq = _bwd_dq_inner(
            dq,
            q, k_ptr_adj, v_ptr_adj, do, m, delta_ptr_adj, sm_scale,
            stride_q_m, stride_q_k, stride_k_n, stride_k_k, stride_v_n, stride_v_k,
            stride_dropout_m, stride_dropout_n,
            stride_delta_m,
            seqlen_q, seqlen_k,
            dropout_p, philox_seed, batch_philox_offset, dropout_offset,
            start_m, start_n, end_n, num_steps,
            descale_q, descale_k, descale_v, descale_do,
            BLOCK_M, BLOCK_N,
            BLOCK_D_MODEL, BLOCK_D_MODEL_POW2,
            MASK=False,
            ENABLE_DROPOUT=ENABLE_DROPOUT,
            IS_FP8=IS_FP8,
            FP8_MAX=FP8_MAX,
        )
        # Write back dQ.
        offs_dq = (batch_idx * stride_dq_b + 
                    head_q_idx * stride_dq_h +
                    q_start * stride_dq_m +
                    offs_m[:, None] * stride_dq_m + 
                    offs_k[None, :] * stride_dq_k)
        dq *= sm_scale
        tl.store(dq_ptr + offs_dq, dq, mask=mask_q)


@triton.jit
def _bwd_kernel_dkdvdq_noncausal(
    Q, K, V, sm_scale, DO, DK, DV, DQ,
    M, Delta,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_dkb, stride_dkh, stride_dkn, stride_dkk,
    stride_deltab, stride_deltah, stride_deltam,
    stride_dob, stride_doh, stride_dom, stride_dok,
    stride_dropoutb, stride_dropouth, stride_dropoutm, stride_dropoutn,
    stride_descale_q_z, stride_descale_k_z, stride_descale_v_z, stride_descale_do_z,
    cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k,
    dropout_mask, dropout_p, philox_seed, philox_offset,
    descale_q_ptr, descale_k_ptr, descale_v_ptr, descale_do_ptr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    BATCH,
    NUM_K_PIDS,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLK_SLICE_FACTOR: tl.constexpr,
    BLOCK_D_MODEL: tl.constexpr,
    BLOCK_D_MODEL_POW2: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_FP8: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    # workgroup id
    wid = tl.program_id(0) # 0, ..., NUM_K_PIDS * BATCH * NUM_K_HEADS - 1

    # Workgroups get launched first along batch dim, then in head_k dim, and then in seq k block dim
    # This is in order to avoid contention for the tl.atomic_add (inside _bwd_dkdvdq_inner) that happens between workgroups that share the same batch and head_k. 
    bid = wid % BATCH 
    hkid = wid // BATCH % NUM_K_HEADS 
    pid = wid // (BATCH * NUM_K_HEADS) % NUM_K_PIDS 

    q_start = 0
    k_start = 0
    seqlen_q = max_seqlen_q
    seqlen_k = max_seqlen_k

    if IS_VARLEN:
        q_start = tl.load(cu_seqlens_q + bid)
        q_end = tl.load(cu_seqlens_q + bid + 1)
        k_start = tl.load(cu_seqlens_k + bid)
        k_end = tl.load(cu_seqlens_k + bid + 1)
        seqlen_q = q_end - q_start
        seqlen_k = k_end - k_start


    dk = tl.zeros([BLOCK_N, BLOCK_D_MODEL_POW2], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_D_MODEL_POW2], dtype=tl.float32)

    start_n = pid * BLOCK_N

    offs_k = tl.arange(0, BLOCK_D_MODEL_POW2)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    mask_kv = offs_n[:, None] < seqlen_k
    PADDED_HEAD: tl.constexpr = (BLOCK_D_MODEL != BLOCK_D_MODEL_POW2)
    if PADDED_HEAD:
        mask_kv &= offs_k < BLOCK_D_MODEL

    GROUP_SIZE = NUM_Q_HEADS // NUM_K_HEADS
    adj_k = (bid * stride_kb + 
            hkid * stride_kh + 
            k_start * stride_kn + 
            offs_n[:, None] * stride_kn + 
            offs_k[None, :] * stride_kk)
    adj_v = (bid * stride_vb + 
            hkid * stride_vh + 
            k_start * stride_vn + 
            offs_n[:, None] * stride_vn + 
            offs_k[None, :] * stride_vk)

    k = tl.load(K + adj_k, mask=mask_kv, other=0.0)
    v = tl.load(V + adj_v, mask=mask_kv, other=0.0)

    for hqid in range(hkid * GROUP_SIZE, hkid * GROUP_SIZE + GROUP_SIZE):
        adj_q = (bid * stride_qb + hqid * stride_qh + q_start * stride_qm)
        
        Q_ptr = Q + adj_q
        DQ_ptr = DQ  + adj_q
        
        adj_do = (bid * stride_dob + hqid * stride_doh + q_start * stride_dom)
        DO_ptr = DO + adj_do
        adj_delta = (bid * stride_deltab + hqid * stride_deltah + q_start * stride_deltam)
        M_ptr = M + adj_delta
        Delta_ptr = Delta + adj_delta

        #dropout 
        batch_philox_offset = 0
        dropout_offset = 0
        if ENABLE_DROPOUT:
            batch_philox_offset = philox_offset + bid * stride_dropoutb + \
                                  hqid * stride_dropouth
            dropout_offset = dropout_mask + bid * stride_dropoutb + \
                             hqid * stride_dropouth

        if IS_FP8:
            descale_q = tl.load(descale_q_ptr + bid * stride_descale_q_z + hqid)
            descale_k = tl.load(descale_k_ptr + bid * stride_descale_k_z + hkid)
            descale_v = tl.load(descale_v_ptr + bid * stride_descale_v_z + hkid)
            descale_do = tl.load(descale_do_ptr + bid * stride_descale_do_z + hqid)
        else:
            descale_q, descale_k, descale_v, descale_do = 1.0, 1.0, 1.0, 1.0

        start_m = 0
        num_steps = tl.cdiv(seqlen_q, BLOCK_M)

        dk, dv = _bwd_dkdvdq_inner(
            dk, dv,
            Q_ptr, k, v, DO_ptr, DQ_ptr, M_ptr, Delta_ptr, sm_scale,
            stride_qm, stride_qk,
            stride_dom, stride_dok,
            stride_dropoutm, stride_dropoutn,
            stride_deltam,
            dropout_p, philox_seed, batch_philox_offset, dropout_offset,
            seqlen_q, seqlen_k,
            start_n, start_m, num_steps,
            descale_q, descale_k, descale_v, descale_do,
            BLOCK_M, BLOCK_N,
            BLOCK_D_MODEL, BLOCK_D_MODEL_POW2,
            MASK=False,
            ENABLE_DROPOUT=ENABLE_DROPOUT,
            IS_FP8=IS_FP8,
            FP8_MAX=FP8_MAX,
            workgroup_id=pid,
        )

    adj_dkdv = (bid * stride_dkb +
                hkid * stride_dkh +
                k_start * stride_dkn + offs_n[:, None] * stride_dkn + 
                offs_k[None, :] * stride_dkk)
    tl.store(DV + adj_dkdv, dv, mask=mask_kv)
    dk *= sm_scale
    tl.store(DK + adj_dkdv, dk, mask=mask_kv)



@triton.jit
def _bwd_kernel_dkdv_noncausal(
    Q, K, V, sm_scale, DO, DK, DV,
    M, Delta,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_dkb, stride_dkh, stride_dkn, stride_dkk,
    stride_deltab, stride_deltah, stride_deltam,
    stride_dob, stride_doh, stride_dom, stride_dok,
    stride_dropoutb, stride_dropouth, stride_dropoutm, stride_dropoutn,
    stride_descale_q_z, stride_descale_k_z, stride_descale_v_z, stride_descale_do_z,
    cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k,
    dropout_mask, dropout_p, philox_seed, philox_offset,
    descale_q_ptr, descale_k_ptr, descale_v_ptr, descale_do_ptr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLK_SLICE_FACTOR: tl.constexpr,
    BLOCK_D_MODEL: tl.constexpr,
    BLOCK_D_MODEL_POW2: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_FP8: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    pid = tl.program_id(0)
    bid = tl.program_id(1)
    hkid = tl.program_id(2)

    q_start = 0
    k_start = 0
    seqlen_q = max_seqlen_q
    seqlen_k = max_seqlen_k

    if IS_VARLEN:
        q_start = tl.load(cu_seqlens_q + bid)
        q_end = tl.load(cu_seqlens_q + bid + 1)
        k_start = tl.load(cu_seqlens_k + bid)
        k_end = tl.load(cu_seqlens_k + bid + 1)
        seqlen_q = q_end - q_start
        seqlen_k = k_end - k_start


    dk = tl.zeros([BLOCK_N, BLOCK_D_MODEL_POW2], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_D_MODEL_POW2], dtype=tl.float32)

    start_n = pid * BLOCK_N

    offs_k = tl.arange(0, BLOCK_D_MODEL_POW2)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    mask_kv = offs_n[:, None] < seqlen_k
    PADDED_HEAD: tl.constexpr = (BLOCK_D_MODEL != BLOCK_D_MODEL_POW2)
    if PADDED_HEAD:
        mask_kv &= offs_k < BLOCK_D_MODEL

    GROUP_SIZE = NUM_Q_HEADS // NUM_K_HEADS
    adj_k = (bid * stride_kb + 
            hkid * stride_kh + 
            k_start * stride_kn + 
            offs_n[:, None] * stride_kn + 
            offs_k[None, :] * stride_kk)
    adj_v = (bid * stride_vb + 
            hkid * stride_vh + 
            k_start * stride_vn + 
            offs_n[:, None] * stride_vn + 
            offs_k[None, :] * stride_vk)

    k = tl.load(K + adj_k, mask=mask_kv, other=0.0)
    v = tl.load(V + adj_v, mask=mask_kv, other=0.0)

    for hqid in range(hkid * GROUP_SIZE, hkid * GROUP_SIZE + GROUP_SIZE):
        adj_q = (bid * stride_qb + hqid * stride_qh + q_start * stride_qm)
        Q_ptr = Q + adj_q
        adj_do = (bid * stride_dob + hqid * stride_doh + q_start * stride_dom)
        DO_ptr = DO + adj_do
        adj_delta = (bid * stride_deltab + hqid * stride_deltah + q_start * stride_deltam)
        M_ptr = M + adj_delta
        Delta_ptr = Delta + adj_delta

        #dropout 
        batch_philox_offset = 0
        dropout_offset = 0
        if ENABLE_DROPOUT:
            batch_philox_offset = philox_offset + bid * stride_dropoutb + \
                                  hqid * stride_dropouth
            dropout_offset = dropout_mask + bid * stride_dropoutb + \
                             hqid * stride_dropouth

        if IS_FP8:
            descale_q = tl.load(descale_q_ptr + bid * stride_descale_q_z + hqid)
            descale_k = tl.load(descale_k_ptr + bid * stride_descale_k_z + hkid)
            descale_v = tl.load(descale_v_ptr + bid * stride_descale_v_z + hkid)
            descale_do = tl.load(descale_do_ptr + bid * stride_descale_do_z + hqid)
        else:
            descale_q, descale_k, descale_v, descale_do = 1.0, 1.0, 1.0, 1.0

        start_m = 0
        num_steps = tl.cdiv(seqlen_q, BLOCK_M)
        dk, dv = _bwd_dkdv_inner(
            dk, dv,
            Q_ptr, k, v, DO_ptr, M_ptr, Delta_ptr, sm_scale,
            stride_qm, stride_qk,
            stride_dom, stride_dok,
            stride_dropoutm, stride_dropoutn,
            stride_deltam,
            dropout_p, philox_seed, batch_philox_offset, dropout_offset,
            seqlen_q, seqlen_k,
            start_n, start_m, num_steps,
            descale_q, descale_k, descale_v, descale_do,
            BLOCK_M, BLOCK_N,
            BLOCK_D_MODEL, BLOCK_D_MODEL_POW2,
            MASK=False,
            ENABLE_DROPOUT=ENABLE_DROPOUT,
            IS_FP8=IS_FP8,
            FP8_MAX=FP8_MAX,
        )

    adj_dkdv = (bid * stride_dkb +
                hkid * stride_dkh +
                k_start * stride_dkn + offs_n[:, None] * stride_dkn + 
                offs_k[None, :] * stride_dkk)
    tl.store(DV + adj_dkdv, dv, mask=mask_kv)
    dk *= sm_scale
    tl.store(DK + adj_dkdv, dk, mask=mask_kv)


@triton.jit
def _bwd_kernel_dq_noncausal(
    Q, K, V, sm_scale, DO, DQ,
    M, delta,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_dqb, stride_dqh, stride_dqm, stride_dqk,
    stride_deltab, stride_deltah, stride_deltam,
    stride_dob, stride_doh, stride_dom, stride_dok,
    stride_dropoutb, stride_dropouth, stride_dropoutm, stride_dropoutn,
    stride_descale_q_z, stride_descale_k_z, stride_descale_v_z, stride_descale_do_z,
    cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k,
    dropout_mask, dropout_p, philox_seed, philox_offset_base,
    descale_q_ptr, descale_k_ptr, descale_v_ptr, descale_do_ptr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLK_SLICE_FACTOR: tl.constexpr,
    BLOCK_D_MODEL: tl.constexpr,
    BLOCK_D_MODEL_POW2: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_FP8: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    pid = tl.program_id(0) #seqlen
    bid = tl.program_id(1) #batch
    hkid = tl.program_id(2) #head_k

    q_start = 0
    k_start = 0
    seqlen_q = max_seqlen_q
    seqlen_k = max_seqlen_k

    if IS_VARLEN:
        # Compute actual sequence lengths
        q_start = tl.load(cu_seqlens_q + bid)
        q_end = tl.load(cu_seqlens_q + bid + 1)
        k_start = tl.load(cu_seqlens_k + bid)
        k_end = tl.load(cu_seqlens_k + bid + 1)
        seqlen_q = q_end - q_start
        seqlen_k = k_end - k_start
    
    start_m = pid * BLOCK_M

    offs_k = tl.arange(0, BLOCK_D_MODEL_POW2)
    offs_m = start_m + tl.arange(0, BLOCK_M)

    #mask for loading K and V
    mask_q = offs_m[:, None] < seqlen_q
    PADDED_HEAD: tl.constexpr = (BLOCK_D_MODEL != BLOCK_D_MODEL_POW2)
    if PADDED_HEAD:
        mask_k = offs_k < BLOCK_D_MODEL
        mask_q &= mask_k[None, :]
    offs_q = offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    offs_do = offs_m[:, None] * stride_dom + offs_k[None, :] * stride_dok
    adj_k = bid * stride_kb + hkid * stride_kh + k_start * stride_kn
    adj_v = bid * stride_vb + hkid * stride_vh + k_start * stride_vn
    K += adj_k
    V += adj_v

    GROUP_SIZE = NUM_Q_HEADS // NUM_K_HEADS
    for hqid in range(hkid * GROUP_SIZE, hkid * GROUP_SIZE + GROUP_SIZE):
        adj_q = bid * stride_qb + hqid * stride_qh + q_start * stride_qm
        adj_do = bid * stride_dob + hqid * stride_doh + q_start * stride_dom
        adj_delta = bid * stride_deltab + hqid * stride_deltah + q_start * stride_deltam
        delta_ptr = delta + adj_delta

        batch_philox_offset = 0
        dropout_offset = 0
        if ENABLE_DROPOUT:
            batch_philox_offset = (philox_offset_base + 
                                  bid * stride_dropoutb + 
                                  hqid * stride_dropouth)
            dropout_offset = (
                dropout_mask + bid * stride_dropoutb + hqid * stride_dropouth)

        q = tl.load(Q + adj_q + offs_q, mask=mask_q, other=0.0)
        do = tl.load(DO + adj_do + offs_do, mask=mask_q, other=0.0)
        m = tl.load(M + adj_delta + offs_m * stride_deltam, mask=offs_m < seqlen_q)
        m = m[:, None]

        #FP8
        if IS_FP8:
            descale_q = tl.load(descale_q_ptr + bid * stride_descale_q_z + hqid)
            descale_k = tl.load(descale_k_ptr + bid * stride_descale_k_z + hkid)
            descale_v = tl.load(descale_v_ptr + bid * stride_descale_v_z + hkid)
            descale_do = tl.load(descale_do_ptr + bid * stride_descale_do_z + hqid)
        else:
            descale_q, descale_k, descale_v, descale_do = 1.0, 1.0, 1.0, 1.0

        start_n = 0
        end_n = seqlen_k
        num_steps = tl.cdiv(seqlen_k, BLOCK_N)
        dq = tl.zeros([BLOCK_M, BLOCK_D_MODEL_POW2], dtype=tl.float32)
        dq = _bwd_dq_inner(
            dq,
            q, K, V, do, m, delta_ptr, sm_scale,
            stride_qm, stride_qk, stride_kn, stride_kk, stride_vn, stride_vk,
            stride_dropoutm, stride_dropoutn,
            stride_deltam,
            seqlen_q, seqlen_k,
            dropout_p, philox_seed, batch_philox_offset, dropout_offset,
            start_m, start_n, end_n, num_steps,
            descale_q, descale_k, descale_v, descale_do,
            BLOCK_M, BLOCK_N,
            BLOCK_D_MODEL, BLOCK_D_MODEL_POW2,
            MASK=False,
            ENABLE_DROPOUT=ENABLE_DROPOUT,
            IS_FP8=IS_FP8,
            FP8_MAX=FP8_MAX,
        )

        adj_dq = bid * stride_dqb + hqid * stride_dqh + q_start * stride_dqm
        offs_dq = offs_m[:, None] * stride_dqm + offs_k[None, :] * stride_dqk
        dq *= sm_scale
        tl.store(DQ + adj_dq + offs_dq, dq, mask=mask_q)

def _flash_attn_backward(
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
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    philox_seed: Optional[int] = 0,
    philox_offset: Optional[int] = 0,
    descale_q: Optional[torch.Tensor] = None,
    descale_k: Optional[torch.Tensor] = None,
    descale_v: Optional[torch.Tensor] = None,
    descale_do: Optional[torch.Tensor] = None,
    fused: bool = False,
):
    IS_FP8 = is_fp8(q)
    if IS_FP8:
        FP8_MAX = torch.finfo(q.dtype).max
        descale_strides = (descale_q.stride(0),descale_k.stride(0),descale_v.stride(0),descale_do.stride(0) )
    else:
        FP8_MAX = None
        stride_descale_q_z = stride_descale_k_z = stride_descale_v_z = stride_descale_do_z = None
        descale_strides = (stride_descale_q_z, stride_descale_k_z, stride_descale_v_z, stride_descale_do_z)

    IS_VARLEN = True if cu_seqlens_q is not None else False

    #get strides and shape
    if IS_VARLEN:  
        #Layout for q,k,v is thd ie [total tokens, num_head, head_dim] 
        batch, seqlen_q, num_q_heads, head_sz  = len(cu_seqlens_q) - 1, max_seqlen_q, q.shape[1], q.shape[2]
        seqlen_k, num_k_heads =  max_seqlen_k, k.shape[1] 
        q_strides = (0, q.stride(1), q.stride(0), q.stride(2))
        q_strides = (0, q.stride(1), q.stride(0), q.stride(2))
        k_strides = (0, k.stride(1), k.stride(0), k.stride(2))
        v_strides = (0, v.stride(1), v.stride(0), v.stride(2))
        o_strides = (0, o.stride(1), o.stride(0), o.stride(2))
        dq_strides = (0, dq.stride(1), dq.stride(0), dq.stride(2))
        dk_strides = (0, dk.stride(1), dk.stride(0), dk.stride(2))
        dv_strides = (0, dv.stride(1), dv.stride(0), dv.stride(2))
        do_strides = (0, do.stride(1), do.stride(0), do.stride(2))
    else:
        #Layout for q,k,v is bshd ie [batch, seq_len, num_head, head_dim] 
        batch, seqlen_q, num_q_heads, head_sz = q.shape
        seqlen_k, num_k_heads  = k.shape[1], k.shape[2]
        q_strides = (q.stride(0), q.stride(2), q.stride(1), q.stride(3))
        k_strides = (k.stride(0), k.stride(2), k.stride(1), k.stride(3))
        v_strides = (v.stride(0), v.stride(2), v.stride(1), v.stride(3))
        o_strides = (o.stride(0), o.stride(2), o.stride(1), o.stride(3))
        dq_strides = (dq.stride(0), dq.stride(2), dq.stride(1), dq.stride(3))
        dk_strides = (dk.stride(0), dk.stride(2), dk.stride(1), dk.stride(3))
        dv_strides = (dv.stride(0), dv.stride(2), dv.stride(1), dv.stride(3))
        do_strides = (do.stride(0), do.stride(2), do.stride(1), do.stride(3))

    #BLOCK_D_MODEL, BLOCK_D_MODEL_POW2
    #padding for head_dim. Power of 2 or 16
    BLOCK_D_MODEL_POW2 = triton.next_power_of_2(head_sz)
    BLOCK_D_MODEL_POW2 = max(BLOCK_D_MODEL_POW2, 16)

    #Configs
    #PRE_BLOCK, BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2
    #BLK_SLICE_FACTOR
    NUM_WARPS, NUM_STAGES = 4, 1
    WAVES_PER_EU = 1
    PRE_BLOCK = 128
    #BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 128, 128, 32
    BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 64, 64, 64, 16 
    BLK_SLICE_FACTOR = 2

    #init delta
    delta = torch.zeros_like(softmax_lse) 
    if IS_VARLEN:
        #[total_tokens, num_q_heads, seqlen_q]
        delta_strides = (0, delta.stride(1), delta.stride(0))
    else:
        #[batch, num_q_heads, seqlen_q]
        delta_strides = delta.stride()

    #preprocess
    #compute D(delta) = rowsum(dO*O). Note, multiplication is element-wise.
    pre_grid = (triton.cdiv(max_seqlen_q, PRE_BLOCK), batch, num_q_heads)
    _bwd_preprocess[pre_grid](
        o, do,
        delta,
        *o_strides,
        *delta_strides,
        descale_strides[3],
        cu_seqlens_q, max_seqlen_q,
        descale_do,
        BLOCK_M=PRE_BLOCK,
        BLOCK_D_MODEL=head_sz,
        BLOCK_D_MODEL_POW2=BLOCK_D_MODEL_POW2,
        IS_VARLEN=IS_VARLEN,
        IS_FP8=IS_FP8
    )

    #dropout_mask
    use_dropout = (dropout_p > 0.0)
    if use_dropout:
        dropout_mask = torch.zeros(
            (batch, num_q_heads, max_seqlen_q, max_seqlen_k),
            device=q.device,
            dtype=torch.float32)
        dropout_strides = dropout_mask.stride()
    else:
        dropout_mask = None
        dropout_strides = (0, 0, 0, 0)

    grid_dkdv = ((max_seqlen_k + BLOCK_N1 - 1) // BLOCK_N1, batch, num_k_heads)
    grid_dq = ((max_seqlen_q + BLOCK_M2 - 1) // BLOCK_M2, batch, num_k_heads)
    
    if fused: # fuses dk, dv, dq computations into one kernel by computing the dq using atomic adds between workgroups
        
        BLOCK_N = 128
        config = {
            "BLOCK_M": 32,
            "BLOCK_N": BLOCK_N,
            "num_warps": 4,
            "num_stages": 1,
            "waves_per_eu": 1,
            "BLK_SLICE_FACTOR": 2,
        }
        
        num_k_pids = (max_seqlen_k + BLOCK_N - 1) // BLOCK_N
        grid_dkdvdq = (batch * num_k_heads * num_k_pids,) 

        if causal:
            _bwd_kernel_dkdvdq_causal[grid_dkdvdq](
                q, k, v, sm_scale, do, dk, dv, dq,
                softmax_lse, delta,
                *q_strides,
                *k_strides,
                *v_strides,
                *dk_strides,
                *delta_strides,
                *do_strides,
                *dropout_strides,
                *descale_strides,
                cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                dropout_mask,dropout_p, philox_seed, philox_offset,
                descale_q, descale_k, descale_v, descale_do,
                NUM_Q_HEADS=num_q_heads,
                NUM_K_HEADS=num_k_heads,
                BATCH=batch,
                NUM_K_PIDS=num_k_pids, 
                BLOCK_D_MODEL=head_sz,
                BLOCK_D_MODEL_POW2=BLOCK_D_MODEL_POW2,
                ENABLE_DROPOUT=use_dropout,
                IS_VARLEN=IS_VARLEN,
                IS_FP8=IS_FP8,
                FP8_MAX=FP8_MAX,
                **config,
            )
        else:
            _bwd_kernel_dkdvdq_noncausal[grid_dkdvdq](
                q, k, v, sm_scale, do, dk, dv, dq,
                softmax_lse, delta,
                *q_strides,
                *k_strides,
                *v_strides,
                *dk_strides,
                *delta_strides,
                *do_strides,
                *dropout_strides,
                *descale_strides,
                cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                dropout_mask,dropout_p, philox_seed, philox_offset,
                descale_q, descale_k, descale_v, descale_do,
                NUM_Q_HEADS=num_q_heads,
                NUM_K_HEADS=num_k_heads,
                BATCH=batch,
                NUM_K_PIDS=num_k_pids,
                BLOCK_D_MODEL=head_sz,
                BLOCK_D_MODEL_POW2=BLOCK_D_MODEL_POW2,
                ENABLE_DROPOUT=use_dropout,
                IS_VARLEN=IS_VARLEN,
                IS_FP8=IS_FP8,
                FP8_MAX=FP8_MAX,
                **config,
            )
        
        return delta
    
    # split kernels solution: one kernel computes dk, dv and the other computes dq

    if causal:
        _bwd_kernel_dkdv_causal[grid_dkdv](
            q, k, v, sm_scale, do, dk, dv,
            softmax_lse, delta,
            *q_strides,
            *k_strides,
            *v_strides,
            *dk_strides,
            *delta_strides,
            *do_strides,
            *dropout_strides,
            *descale_strides,
            cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_mask,dropout_p, philox_seed, philox_offset,
            descale_q, descale_k, descale_v, descale_do,
            NUM_Q_HEADS=num_q_heads,
            NUM_K_HEADS=num_k_heads,
            BLOCK_M=BLOCK_M1,
            BLOCK_N=BLOCK_N1,
            BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,
            BLOCK_D_MODEL=head_sz,
            BLOCK_D_MODEL_POW2=BLOCK_D_MODEL_POW2,
            ENABLE_DROPOUT=use_dropout,
            IS_VARLEN=IS_VARLEN,
            IS_FP8=IS_FP8,
            FP8_MAX=FP8_MAX,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
            waves_per_eu=WAVES_PER_EU,
        )
        _bwd_kernel_dq_causal[grid_dq](
            q, k, v, sm_scale, do, dq,
            softmax_lse, delta,
            *q_strides,
            *k_strides,
            *v_strides,
            *dq_strides,
            *delta_strides,
            *do_strides,
            *dropout_strides,
            *descale_strides,
            cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k,
            dropout_mask,dropout_p, philox_seed, philox_offset,
            descale_q, descale_k, descale_v, descale_do,
            NUM_Q_HEADS=num_q_heads,
            NUM_K_HEADS=num_k_heads,
            BLOCK_M=BLOCK_M2,
            BLOCK_N=BLOCK_N2,
            BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,
            BLOCK_D_MODEL=head_sz,
            BLOCK_D_MODEL_POW2=BLOCK_D_MODEL_POW2,
            ENABLE_DROPOUT=use_dropout,
            IS_VARLEN=IS_VARLEN,
            IS_FP8=IS_FP8,
            FP8_MAX=FP8_MAX,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
            waves_per_eu=WAVES_PER_EU,
        )
    else:
        _bwd_kernel_dkdv_noncausal[grid_dkdv](
            q, k, v, sm_scale, do, dk, dv,
            softmax_lse, delta,
            *q_strides,
            *k_strides,
            *v_strides,
            *dk_strides,
            *delta_strides,
            *do_strides,
            *dropout_strides,
            *descale_strides,
            cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_mask,dropout_p, philox_seed, philox_offset,
            descale_q, descale_k, descale_v, descale_do,
            NUM_Q_HEADS=num_q_heads,
            NUM_K_HEADS=num_k_heads,
            BLOCK_M=BLOCK_M1,
            BLOCK_N=BLOCK_N1,
            BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,
            BLOCK_D_MODEL=head_sz,
            BLOCK_D_MODEL_POW2=BLOCK_D_MODEL_POW2,
            ENABLE_DROPOUT=use_dropout,
            IS_VARLEN=IS_VARLEN,
            IS_FP8=IS_FP8,
            FP8_MAX=FP8_MAX,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
            waves_per_eu=WAVES_PER_EU,
        )

        _bwd_kernel_dq_noncausal[grid_dq](
            q, k, v, sm_scale, do, dq,
            softmax_lse, delta,
            *q_strides,
            *k_strides,
            *v_strides,
            *dq_strides,
            *delta_strides,
            *do_strides,
            *dropout_strides,
            *descale_strides,
            cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_mask,dropout_p, philox_seed, philox_offset,
            descale_q, descale_k, descale_v, descale_do,
            NUM_Q_HEADS=num_q_heads,
            NUM_K_HEADS=num_k_heads,
            BLOCK_M=BLOCK_M2,
            BLOCK_N=BLOCK_N2,
            BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,
            BLOCK_D_MODEL=head_sz,
            BLOCK_D_MODEL_POW2=BLOCK_D_MODEL_POW2,
            ENABLE_DROPOUT=use_dropout,
            IS_VARLEN=IS_VARLEN,
            IS_FP8=IS_FP8,
            FP8_MAX=FP8_MAX,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
            waves_per_eu=WAVES_PER_EU,
        )

    return delta


class FlashAttnFunc(torch.autograd.Function):
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
        alibi_slopes,
        deterministic,
        return_lse,
        return_softmax,
        is_grad_enabled,
        fused_backward,
    ):
        is_grad = is_grad_enabled and any(
            x.requires_grad for x in [q,k,v]
        )
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
    
        
        head_size_og = q.size(3)
        if head_size_og % 8 != 0:
            q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
            k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
            v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])
        out_padded, softmax_lse, S_dmask, philox_seed, philox_offset = _flash_attn_forward(
            q,
            k,
            v,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            alibi_slopes=alibi_slopes,
            return_lse=return_lse,
            return_softmax=return_softmax and dropout_p > 0,
            max_seqlen_q=q.shape[1],
            max_seqlen_k=k.shape[1],
        )

        if is_grad:
            ctx.save_for_backward(q, k, v, out_padded, softmax_lse)
            ctx.philox_seed = philox_seed
            ctx.philox_offset = philox_offset
            ctx.dropout_p = dropout_p
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.window_size = window_size
            ctx.alibi_slopes = alibi_slopes
            ctx.deterministic = deterministic
            ctx.fused_backward = fused_backward


        out = out_padded[..., :head_size_og]
        result = [out]
        if return_lse:
            result.append(softmax_lse)
        if return_softmax:
            result.append(S_dmask)

        return tuple(result)

    @staticmethod
    def backward(ctx, do, *args):
        q, k, v, out, softmax_lse = ctx.saved_tensors
        dq, dk, dv = torch.zeros_like(q), torch.empty_like(k), torch.empty_like(v)
        head_size_v_og = do.size(3)
        do_padded = do
        if head_size_v_og % 8 != 0:
            do_padded = torch.nn.functional.pad(do, [0, 8 - head_size_v_og % 8])
        _flash_attn_backward(
            do_padded,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            ctx.softmax_scale,
            ctx.alibi_slopes,
            ctx.causal,
            None,
            None,
            max_seqlen_q=q.shape[1],
            max_seqlen_k=k.shape[1],
            dropout_p=ctx.dropout_p,
            philox_seed=ctx.philox_seed,
            philox_offset=ctx.philox_offset,
            fused=ctx.fused_backward,
        )
        dq = dq[..., : q.shape[-1]]  # We could have padded the head dimension
        dk = dk[..., : k.shape[-1]]
        dv = dv[..., : v.shape[-1]]
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None

def flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1,-1),
    alibi_slopes=None,
    deterministic=True,
    return_lse=False,
    return_attn_probs=False,
    fused_backward=False,
):
    """dropout_p should be set to 0.0 during evaluation
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        k: (batch_size, seqlen, nheads_k, headdim)
        v: (batch_size, seqlen, nheads_k, headdim)
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_lse=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    return FlashAttnFunc.apply(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_lse,
        return_attn_probs,
        torch.is_grad_enabled(),
        fused_backward,
    )


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
        alibi_slopes,
        deterministic,
        return_lse,
        return_softmax,
        is_grad_enabled, 
    ):
        is_grad = is_grad_enabled and any(
            x.requires_grad for x in [q,k,v]
        )
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        head_size_og = q.size(3)
        if head_size_og % 8 != 0:
            q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
            k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
            v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])

        # cast input to fp8
        fp8_dtype = torch.float8_e4m3fnuz 
        q_fp8, descale_q = cast_to_fp8(q, fp8_dtype, "bshd")
        k_fp8, descale_k = cast_to_fp8(k, fp8_dtype, "bshd")
        v_fp8, descale_v = cast_to_fp8(v, fp8_dtype, "bshd")

        out_padded, softmax_lse, S_dmask, philox_seed, philox_offset = _flash_attn_forward(
            q,
            k,
            v,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            alibi_slopes=alibi_slopes,
            return_lse=return_lse,
            return_softmax=return_softmax and dropout_p > 0,
            max_seqlen_q=q.shape[1],
            max_seqlen_k=k.shape[1],
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            descale_q=descale_q,
            descale_k=descale_k,
            descale_v=descale_v
        )

        if is_grad:
            ctx.save_for_backward(q_fp8, k_fp8, v_fp8, out_padded, softmax_lse, descale_q, descale_k, descale_v)
            ctx.philox_seed = philox_seed
            ctx.philox_offset = philox_offset
            ctx.dropout_p = dropout_p
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.window_size = window_size
            ctx.alibi_slopes = alibi_slopes
        
        out = out_padded[..., :head_size_og]
        result = [out]
        if return_lse:
            result.append(softmax_lse)
        if return_softmax:
            result.append(S_dmask)

        return tuple(result)

    @staticmethod
    def backward(ctx, do, *args):
        q_fp8, k_fp8, v_fp8, out, softmax_lse, descale_q, descale_k, descale_v = ctx.saved_tensors
        dq, dk, dv = torch.zeros_like(q_fp8, dtype=torch.float32), torch.zeros_like(k_fp8, dtype=torch.float32), torch.zeros_like(v_fp8, dtype=torch.float32)
        head_size_v_og = do.size(3)
        do_padded = do
        if head_size_v_og % 8 != 0:
            do_padded = torch.nn.functional.pad(do, [0, 8 - head_size_v_og % 8])
        
        fp8_dtype = torch.float8_e4m3fnuz
        do_padded_fp8, descale_do = cast_to_fp8(do_padded, fp8_dtype, "bshd")
        _flash_attn_backward(
            do_padded_fp8,
            q_fp8,
            k_fp8,
            v_fp8,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            ctx.softmax_scale,
            ctx.alibi_slopes,
            ctx.causal,
            None,
            None,
            max_seqlen_q=q_fp8.shape[1],
            max_seqlen_k=k_fp8.shape[1],
            dropout_p=ctx.dropout_p,
            philox_seed=ctx.philox_seed,
            philox_offset=ctx.philox_offset,
            descale_q=descale_q,
            descale_k=descale_k,
            descale_v=descale_v,
            descale_do=descale_do,
        )
        #dq = dq[..., : q_fp8.shape[-1]]  # We could have padded the head dimension
        #dk = dk[..., : k_fp8.shape[-1]]
        #dv = dv[..., : v_fp8.shape[-1]]
        return dq, dk, dv, None, None, None, None, None, None, None, None, None

def flash_attn_fp8_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    alibi_slopes=None,
    deterministic=False,
    return_lse=False,
    return_attn_probs=False
):
    return FlashAttnFP8Func.apply(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_lse,
        return_attn_probs,
        torch.is_grad_enabled()
    ) 

class FlashAttnVarlenFunc(torch.autograd.Function):
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
        alibi_slopes,
        deterministic,
        return_lse,
        return_softmax,
        block_table,
        is_grad_enabled,
        fused_backward,
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
        out_padded, softmax_lse, S_dmask, philox_seed, philox_offset =  _flash_attn_forward(
            q,
            k,
            v,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            alibi_slopes=alibi_slopes,
            return_lse=return_lse,
            return_softmax=return_softmax and dropout_p > 0.0,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
        )
        if is_grad:
            ctx.save_for_backward(q, k, v, out_padded, softmax_lse, cu_seqlens_q, cu_seqlens_k)
            ctx.max_seqlen_q = max_seqlen_q
            ctx.max_seqlen_k = max_seqlen_k
            ctx.philox_seed = philox_seed
            ctx.philox_offset = philox_offset
            ctx.dropout_p = dropout_p
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.window_size = window_size
            ctx.alibi_slopes = alibi_slopes
            ctx.fused_backward = fused_backward
        out = out_padded[..., :head_size_og]

        result = [out]
        if return_lse:
            result.append(softmax_lse)
        if return_softmax:
            result.append(S_dmask)
        
        return tuple(result)

    @staticmethod
    def backward(ctx, do, *args):
        q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors
        dq, dk, dv = torch.zeros_like(q), torch.empty_like(k), torch.empty_like(v)
        head_size_og = do.size(2)
        do_padded = do
        if head_size_og % 8 != 0:
            do_padded = torch.nn.functional.pad(do, [0, 8 - head_size_og % 8])
        _flash_attn_backward(
            do_padded,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            ctx.softmax_scale,
            ctx.alibi_slopes,
            ctx.causal,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q=ctx.max_seqlen_q,
            max_seqlen_k=ctx.max_seqlen_k,
            dropout_p=ctx.dropout_p,
            philox_seed=ctx.philox_seed,
            philox_offset=ctx.philox_offset,
            fused=ctx.fused_backward,
        )
        dq = dq[..., : q.shape[-1]]  # We could have padded the head dimension
        dk = dk[..., : k.shape[-1]]
        dv = dv[..., : v.shape[-1]]
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None


def flash_attn_varlen_func(
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
    window_size=(-1,-1),
    alibi_slopes=None,
    deterministic=False,
    return_lse=False,
    return_attn_probs=False,
    block_table=None,
    fused_backward=False,
):
    """dropout_p should be set to 0.0 during evaluation
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in K, V with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Arguments:
        q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        k: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        v: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        cu_seqlens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into q.
        cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into kv.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (total, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (nheads, total_q_seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    return FlashAttnVarlenFunc.apply(
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
        alibi_slopes,
        deterministic,
        return_lse,
        return_attn_probs,
        block_table,
        torch.is_grad_enabled(),
        fused_backward,
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
        alibi_slopes,
        deterministic,
        return_lse,
        return_softmax,
        block_table,
        is_grad_enabled,
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
        
        # cast input to fp8
        fp8_dtype = torch.float8_e4m3fnuz 
        q_fp8, descale_q = cast_varlen_to_fp8(q, fp8_dtype, cu_seqlens=cu_seqlens_q) 
        k_fp8, descale_k = cast_varlen_to_fp8(k, fp8_dtype,  cu_seqlens=cu_seqlens_k)
        v_fp8, descale_v = cast_varlen_to_fp8(v, fp8_dtype,  cu_seqlens=cu_seqlens_k)

        out_padded, softmax_lse, S_dmask, philox_seed, philox_offset = _flash_attn_forward(
            q_fp8,
            k_fp8,
            v_fp8,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            alibi_slopes=alibi_slopes,
            return_lse=return_lse,
            return_softmax=return_softmax and dropout_p > 0,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            descale_q=descale_q,
            descale_k=descale_k,
            descale_v=descale_v
        )
        if is_grad:
            ctx.save_for_backward(q_fp8, k_fp8, v_fp8, out_padded, softmax_lse, cu_seqlens_q, cu_seqlens_k, descale_q, descale_k, descale_v)
            ctx.max_seqlen_q = max_seqlen_q
            ctx.max_seqlen_k = max_seqlen_k
            ctx.philox_seed = philox_seed
            ctx.philox_offset = philox_offset
            ctx.dropout_p = dropout_p
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.window_size = window_size
            ctx.alibi_slopes = alibi_slopes
        out = out_padded[..., :head_size_og]
        result = [out]
        if return_lse:
            result.append(softmax_lse)
        if return_softmax:
            result.append(S_dmask)
        
        return tuple(result)
    
    @staticmethod
    def backward(ctx, do, *args):
        q_fp8, k_fp8, v_fp8, out, softmax_lse, cu_seqlens_q, cu_seqlens_q, descale_q, descale_k, descale_v = ctx.saved_tensors
        dq, dk, dv = torch.zeros_like(q, dtype=torch.float32), torch.zeros_like(k, dtype=torch.float32), torch.zeros_like(v, dtype=torch.float32)
        head_size_v_og = do.size(3)
        do_padded = do
        if head_size_v_og % 8 != 0:
            do_padded = torch.nn.functional.pad(do, [0, 8 - head_size_v_og % 8])
        
        fp8_dtype = torch.float8_e4m3fnuz 
        do_padded_fp8, descale_do = cast_varlen_to_fp8(dout_padded, fp8_dtype, "thd", cu_seqlens_q)
        
        _flash_attn_backward(
            do_padded_fp8,
            q_fp8,
            k_fp8,
            v_fp8,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            ctx.softmax_scale,
            ctx.alibi_slopes,
            ctx.causal,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dropout_p=ctx.dropout_p,
            philox_seed=ctx.philox_seed,
            philox_offset=ctx.philox_offset,
            descale_q=descale_q,
            descale_k=descale_k,
            descale_v=descale_v,
            descale_do=descale_do
        )
        dq = dq[..., : q.shape[-1]]  # We could have padded the head dimension
        dk = dk[..., : k.shape[-1]]
        dv = dv[..., : v.shape[-1]]
        return dq, dk, dv, None, None, None, None, None, None, None, None, None

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
    alibi_slopes=None,
    deterministic=False,
    return_lse=False,
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
        alibi_slopes,
        deterministic,
        return_lse,
        return_attn_probs,
        block_table,
        torch.is_grad_enabled()
    )