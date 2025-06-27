import torch
import triton
import triton.language as tl
from flash_attn.flash_attn_triton_amd.utils import compute_fp8_scaling_factors

from typing import Optional, Tuple

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
    stride_dq_m, stride_dq_k,
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
    dq_ptrs_start = DQ + offs_m[:, None] * stride_dq_m + offs_k[None,:] * stride_dq_k #[BLOCK_M, BLOCK_D_MODEL_POW2]
    
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
        dq_ptrs = dq_ptrs_start + blk_idx * step_m * stride_dq_m
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
        adj_dq = batch_idx * stride_dq_b + head_q_idx * stride_dq_h + q_start * stride_dq_m
        
        q_ptr_adj = q_ptr + adj_q
        dq_ptr_adj = dq_ptr + adj_dq
        
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

        # if unaligned start_m is negative, the current N-tile has no block on the
        #   diagonal of causal mask, so everything have no causal mask
        dk, dv = _bwd_dkdvdq_inner(
            dk, dv,  # output tensors
            q_ptr_adj, k, v, do_ptr_adj, dq_ptr_adj, m_ptr_adj, delta_ptr_adj, sm_scale, # input tensors
            stride_q_m, stride_q_k,  # strides for q
            stride_dq_m, stride_dq_k,  # strides for q
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
            stride_dq_m, stride_dq_k,  # strides for dq
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
    stride_dqb, stride_dqh, stride_dqm, stride_dqk,
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
        adj_dq = (bid * stride_dqb + hqid * stride_dqh + q_start * stride_dqm)

        Q_ptr = Q + adj_q
        DQ_ptr = DQ  + adj_dq
        
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
            stride_dqm, stride_dqk,
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

def attention_prefill_backward_triton_fused_atomics_impl(
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
        
        BLOCK_N = 128 if BLOCK_D_MODEL_POW2 < 160 else 64 # larger head sizes lead to oom
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