import torch
import triton # type: ignore
import triton.language as tl # type: ignore
from typing import Literal, Optional
from .utils import DEBUG, DROPOUT_USE_PYTORCH, DROPOUT_DUMP, compute_fp8_scaling_factors, get_shapes_from_layout, \
    get_strides_from_layout, create_dropout_mask, create_dropout_mask_varlen, is_fp8

# NOTE: triton fails to import tl.constexprs so create them here for the file
tl_DROPOUT_USE_PYTORCH: tl.constexpr = triton.language.constexpr(DROPOUT_USE_PYTORCH)
tl_DROPOUT_DUMP: tl.constexpr = triton.language.constexpr(DROPOUT_DUMP)

# This function computes delta given output Out and gradient DO
# Here is the I/O shape:
# Out: (batch, nhead_q, max_seqlens_q, headDim)
# DO: (batch, nhead_q, max_seqlens_q, headDim)
# Delta: (batch, nheads_q, max_seqlens_q), same as softmax_lse defined at
#   fwd_prefill.py line 607
@triton.jit
def _bwd_preprocess(
    O, DO,  # noqa: E741
    Delta,
    stride_ob, stride_oh, stride_om, stride_ok,
    stride_deltab, stride_deltah, stride_deltam,
    stride_descale_do_z,
    cu_seqlens_q, max_seqlen_q,
    Descale_do,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ACTUAL_HEAD_DIM: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_FP8: tl.constexpr
):
    pid_m = tl.program_id(0)
    bid = tl.program_id(1)
    hid = tl.program_id(2)
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
    offs_k = tl.arange(0, HEAD_DIM)
    # Offset O/DO by batch, head and q_start
    O += bid * stride_ob + hid * stride_oh + q_start * stride_om  # noqa: E741
    DO += bid * stride_ob + hid * stride_oh + q_start * stride_om
    # create masks
    mask_m = offs_m < seqlen_q
    mask_md = mask_m[:, None]
    PADDED_HEAD: tl.constexpr = (ACTUAL_HEAD_DIM != HEAD_DIM)
    if PADDED_HEAD:
        mask_md &= offs_k[None, :] < ACTUAL_HEAD_DIM
    # compute pointers
    offs_do = offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
    out_ptrs = O + offs_do
    do_ptrs = DO + offs_do
    # load
    o = tl.load(out_ptrs, mask=mask_md, other=0.0)
    do = tl.load(do_ptrs, mask=mask_md, other=0.0)
    # compute and write-back to delta
    if IS_FP8:
        descale_do = tl.load(Descale_do + bid * stride_descale_do_z + hid)

        # NOTE: do is in the fp8 range and o is not in fp8
        delta = tl.sum(o.to(tl.float32) * (do.to(tl.float32) * descale_do), axis=1)
    else:
        delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
    delta_offset = Delta + bid * stride_deltab + hid * stride_deltah + q_start * stride_deltam
    tl.store(delta_offset + offs_m * stride_deltam, delta, mask=mask_m)


# The main inner-loop logic for computing dK and dV.
@triton.jit
def _bwd_dkdv_inner(
    dk, dv,  # output
    Q, k, v, DO, M, D, sm_scale,  # input tensor
    stride_qm, stride_qk,
    stride_dom, stride_dok,
    stride_dropoutm, stride_dropoutn,
    stride_deltam,
    BLOCK_M: tl.constexpr,  # 16
    BLOCK_N: tl.constexpr,  # 128
    HEAD_DIM: tl.constexpr,  #
    ACTUAL_HEAD_DIM: tl.constexpr,  #
    dropout_p, philox_seed, batch_philox_offset, dropout_offset,
    alibi_slope,
    seqlen_q, seqlen_k,  # max sequence length for q and k
    # Filled in by the wrapper.
    start_n, start_m, num_steps,  # iteration numbers
    descale_q, descale_k, descale_v, descale_do, # fp8 descale factors from user
    MASK: tl.constexpr,  # causal masking, only apply to tiles on mask diagonal
    ENABLE_DROPOUT: tl.constexpr,  # activate dropout
    USE_ALIBI: tl.constexpr,
    USE_EXP2: tl.constexpr,  # activate exp2
    IS_FP8: tl.constexpr,
    FP8_MAX: tl.constexpr,
    DEBUG_TRITON: tl.constexpr,
    DEBUG_TRITON_DETAIL: tl.constexpr,
):
    # if HEAD_DIM is padded
    PADDED_HEAD: tl.constexpr = (ACTUAL_HEAD_DIM != HEAD_DIM)
    delta_qk = seqlen_q - seqlen_k
    offs_m = start_m + tl.arange(0, BLOCK_M)  # start_m + (0, 15)
    offs_n = start_n + tl.arange(0, BLOCK_N)  # start_m + (0, 127)
    offs_k = tl.arange(0, HEAD_DIM)
    # mask to make sure not OOB of seqlen_q
    mask_n = offs_n < seqlen_k
    # Q and DO are (seqlen_q, head_dim)
    # qT_ptrs = (1, BLOCK_M) + (HEAD_DIM, 1), transpose of q
    qT_ptrs = Q + offs_m[None, :] * stride_qm + offs_k[:, None] * stride_qk
    # do_ptrs = (BLOCK_M, 1) + (1, HEAD_DIM), NOT transposed
    do_ptrs = DO + offs_m[:, None] * stride_dom + offs_k[None, :] * stride_dok
    # BLOCK_N must be a multiple of BLOCK_M, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N % BLOCK_M == 0)
    curr_m = start_m
    step_m = BLOCK_M
    curr_philox_offset = batch_philox_offset
    curr_dropout_offset = dropout_offset
    RCP_LN2: tl.constexpr = 1.4426950408889634  # = 1.0 / ln(2)

    for blk_idx in range(num_steps):
        if DEBUG_TRITON: print(f"iter {blk_idx}: curr_m = {curr_m}")  # noqa: E701
        offs_m = curr_m + tl.arange(0, BLOCK_M)
        # update the mask because offs_m advanced
        mask_m = offs_m < seqlen_q
        mask_qT = mask_m[None, :]
        mask_do = mask_m[:, None]
        mask_nm = mask_n[:, None] & (offs_m[None, :] < seqlen_q)
        if PADDED_HEAD:
            mask_qT &= offs_k[:, None] < ACTUAL_HEAD_DIM
            mask_do &= offs_k[None, :] < ACTUAL_HEAD_DIM
        qT = tl.load(qT_ptrs, mask=mask_qT, other=0.0)
        # generate dropout mask
        if ENABLE_DROPOUT:
            # NOTE: dropout is transposed because it is used to mask pT
            philox_offs = curr_philox_offset + \
                          offs_m[None, :] * stride_dropoutm + \
                          offs_n[:, None] * stride_dropoutn
            if tl_DROPOUT_USE_PYTORCH:
                dropout_offs = offs_m[None, :] * stride_dropoutm + \
                               offs_n[:, None] * stride_dropoutn
                dropout_mask = tl.load(
                    curr_dropout_offset + dropout_offs,
                    mask=mask_nm
                )
            else:
                rand_vals = tl.rand(philox_seed, philox_offs)
                dropout_mask = rand_vals > dropout_p
            dropout_scale = 1.0 / (1 - dropout_p)
        # Load m before computing qk to reduce pipeline stall.
        m = tl.load(M + offs_m * stride_deltam, mask=mask_m, other=0.0)
        if IS_FP8:
            qkT = (tl.dot(k, qT) * descale_q * descale_k)
        else:
            qkT = tl.dot(k, qT)
        qkT_scaled =  qkT * sm_scale

        if USE_ALIBI:
            relative_pos_block = offs_n[:, None] + seqlen_q - seqlen_k - offs_m[None, :]
            alibi_block = -1 * alibi_slope * tl.abs(relative_pos_block)
            qkT_scaled += alibi_block

        if DEBUG_TRITON_DETAIL:
            if start_n == 256:
                print(f"qT: {qT.shape}\n", qT)
                print(f"k: {k.shape}\n", k)
                print(f"qkT scaled: {qkT.shape}\n", qkT_scaled)
        # TODO: remove the scaling of m later when we removed re-scaling in fwd
        if USE_EXP2:
            pT = tl.math.exp2(qkT_scaled * RCP_LN2 - m[None, :] * RCP_LN2)
        else:
            pT = tl.math.exp(qkT_scaled - m[None, :])

        # Autoregressive masking.
        if MASK:
            # offset offs_m with delta_qk since the causal mask starts at
            # bottom right of the (seqlen_q, seqlen_k) matrix
            causal_mask = (offs_m[None, :] - delta_qk) >= offs_n[:, None]
            mask = causal_mask & mask_nm
            if DEBUG_TRITON_DETAIL:
                if start_n == 256:
                    print(f"causal_mask: {causal_mask.shape}\n", causal_mask)
                    print(f"qkT after causal: {qkT.shape}\n", tl.where(causal_mask, qkT * sm_scale, 0.0))
            pT = tl.where(mask, pT, 0.0)
        do = tl.load(do_ptrs, mask=mask_do, other=0.0)
        # Compute dV.
        if ENABLE_DROPOUT:
            pT_dropout = tl.where(dropout_mask, pT, 0.0) * dropout_scale
            if IS_FP8:
                scale_p_dropout, descale_p_dropout = compute_fp8_scaling_factors(pT_dropout, FP8_MAX)
                dv += (tl.dot((pT_dropout * scale_p_dropout).to(do.type.element_ty), do)* descale_p_dropout * descale_do)
            else:
                dv += tl.dot(pT_dropout.to(do.type.element_ty), do)
        else:
            if IS_FP8:
                scale_pT, descale_pT = compute_fp8_scaling_factors(pT, FP8_MAX)
                dv += (tl.dot((pT * scale_pT).to(do.type.element_ty), do) * descale_pT * descale_do)
            else:
                dv += tl.dot(pT.to(do.type.element_ty), do)

        if DEBUG_TRITON_DETAIL:
            if start_n == 256:
                print(f"pT: {pT.shape}\n", pT)
        # D (= delta) is pre-divided by ds_scale.
        Di = tl.load(D + offs_m * stride_deltam, mask=mask_m)
        # Compute dP and dS.
        if IS_FP8:
            dpT = (tl.dot(v, tl.trans(do)) * descale_v * descale_do)
        else:
            dpT = tl.dot(v, tl.trans(do))
        if ENABLE_DROPOUT:
            dpT = tl.where(dropout_mask, dpT, 0.0) * dropout_scale
        delta_i = Di[None, :]
        dsT = pT * (dpT - delta_i)
        if IS_FP8:
            scale_dsT, descale_dsT = compute_fp8_scaling_factors(dsT, FP8_MAX)
            dk += (tl.dot((dsT * scale_dsT).to(qT.type.element_ty), tl.trans(qT)) * descale_dsT * descale_q)
        else:
            dk += tl.dot(dsT.to(qT.type.element_ty), tl.trans(qT))
        # Increment pointers.
        curr_m += step_m
        qT_ptrs += step_m * stride_qm
        do_ptrs += step_m * stride_dom
    return dk, dv


# grid = (max_seqlen_k // BLOCK_N, batch, nheads_q)
@triton.jit
def _bwd_kernel_dkdv_causal(
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
    stride_az, stride_ah,
    HQ, HK,
    cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k,
    Dropout_mask, dropout_p, philox_seed, philox_offset_base,
    Alibi_slopes,
    Descale_q, Descale_k, Descale_v, Descale_do,
    BLOCK_M: tl.constexpr,  # 32
    BLOCK_N: tl.constexpr,  # 128
    BLK_SLICE_FACTOR: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ACTUAL_HEAD_DIM: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_ALIBI: tl.constexpr,
    USE_EXP2: tl.constexpr,
    IS_FP8: tl.constexpr,
    FP8_MAX: tl.constexpr,
    FP8_OUTPUT: tl.constexpr,
    DEBUG_TRITON: tl.constexpr,
    DEBUG_TRITON_DETAIL: tl.constexpr,
):
    # program ids
    pid = tl.program_id(0)
    bid = tl.program_id(1)
    hkid = tl.program_id(2)
    # figure out varlen start and end
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

    dk = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    # Figure out causal starting block since we have seqlen_q >=< seqlen_k.
    # Unlike forward pass where we tile on M dim and iterate on N dim, so that
    # we can skip some M blocks, in backward pass, we tile on the N dim for kv
    # and iterate over the M. In this way, we cannot skip N blocks, but only to
    # determine the starting M blocks to skip some initial blocks masked by
    # causal.
    delta_qk = seqlen_q - seqlen_k
    if DEBUG_TRITON: print(f"\npid: {pid}, bid: {bid}, hkid: {hkid}")
    if DEBUG_TRITON: print(f"delta_qk = {delta_qk}")
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
        if DEBUG_TRITON: print(f"q >= k: start_delta = delta_qk aligned to BLOCK_M = {start_delta_q_gt_k}")
    else:
        start_delta = start_delta_q_lt_k
        if DEBUG_TRITON: print(f"q < k: start_delta = residue btw multiple BLOCK_N and delta_qk = {delta_aligned} = aligned to BLOCK_M = {start_delta_q_lt_k}")
    # align the delta_qk
    start_n = pid * BLOCK_N

    offs_k = tl.arange(0, HEAD_DIM)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    # Mask for loading K and V
    mask_kv = offs_n[:, None] < seqlen_k
    PADDED_HEAD: tl.constexpr = (ACTUAL_HEAD_DIM != HEAD_DIM)
    if PADDED_HEAD:
        mask_k = offs_k < ACTUAL_HEAD_DIM
        mask_kv &= mask_k[None, :]

    GROUP_SIZE = HQ // HK
    # K/V tensors not changed for the group
    adj_k = bid * stride_kb + hkid * stride_kh + k_start * stride_kn + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk
    adj_v = bid * stride_vb + hkid * stride_vh + k_start * stride_vn + offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk
    # load K and V: they stay in SRAM throughout the inner loop.
    k = tl.load(K + adj_k , mask=mask_kv, other=0.0)
    v = tl.load(V + adj_v, mask=mask_kv, other=0.0)
    # If MQA / GQA, set the K and V head offsets appropriately.
    for hqid in range(hkid * GROUP_SIZE, hkid * GROUP_SIZE + GROUP_SIZE):
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
            if DEBUG_TRITON: print(f"residue_m = {residue_m}")

        # offset input and output tensor by batch and Q/K heads
        adj_q = bid * stride_qb + hqid * stride_qh + q_start * stride_qm
        Q_ptr = Q + adj_q
        adj_do = bid * stride_dob + hqid * stride_doh + q_start * stride_dom
        DO_ptr = DO + adj_do
        adj_delta = bid * stride_deltab + hqid * stride_deltah + q_start * stride_deltam
        M_ptr = M + adj_delta
        Delta_ptr = Delta + adj_delta

        if USE_ALIBI:
            alibi_offset = bid * stride_az + hqid * stride_ah
            alibi_slope = tl.load(Alibi_slopes + alibi_offset)
        else:
            alibi_slope = None

        # batch_philox_offset is the ACTUALLY dropout offset
        # dropout_offset is for debug purpose and will be removed later
        batch_philox_offset = 0
        dropout_offset = 0
        if ENABLE_DROPOUT:
            batch_philox_offset = philox_offset_base + bid * stride_dropoutb + \
                                  hqid * stride_dropouth
            dropout_offset = Dropout_mask + bid * stride_dropoutb + \
                             hqid * stride_dropouth

        MASK_BLOCK_M: tl.constexpr = BLOCK_M // BLK_SLICE_FACTOR
        # bound the masked operation to q len so it does not have to wast cycles
        len_m = min(len_m, seqlen_q)
        num_steps = tl.cdiv(len_m, MASK_BLOCK_M)
        # when q < k, we may skip the initial masked op
        if pid < num_blocks_skip:
            num_steps = 0

        if IS_FP8:
            descale_q = tl.load(Descale_q + bid * stride_descale_q_z + hqid)
            descale_k = tl.load(Descale_k + bid * stride_descale_k_z + hkid)
            descale_v = tl.load(Descale_v + bid * stride_descale_v_z + hkid)
            descale_do = tl.load(Descale_do + bid * stride_descale_do_z + hqid)
        else:
            descale_q, descale_k, descale_v, descale_do = 1.0, 1.0, 1.0, 1.0

        # if start_m is negative, the current N-tile has no block on the
        #   diagonal of causal mask, so everything have no causal mask
        if DEBUG_TRITON: print(f"Masked: start_n: {start_n}; start_m: {start_m}, num_steps: {num_steps}")
        dk, dv = _bwd_dkdv_inner(
            dk, dv,  # output tensors
            Q_ptr, k, v, DO_ptr, M_ptr, Delta_ptr, sm_scale, # input tensors
            stride_qm, stride_qk,  # strides for q
            stride_dom, stride_dok,  # strides for o
            stride_dropoutm, stride_dropoutn,  # strides for dropout
            stride_deltam,
            MASK_BLOCK_M, BLOCK_N,  # block dim
            HEAD_DIM, ACTUAL_HEAD_DIM,  # head dim
            dropout_p, philox_seed, batch_philox_offset, dropout_offset,  #
            alibi_slope,
            seqlen_q, seqlen_k,  # max sequence length for q and k
            start_n, start_m, num_steps,  # iteration numbers
            descale_q, descale_k, descale_v, descale_do, # fp8 descale factors from user 
            MASK=True,  # causal masking
            ENABLE_DROPOUT=ENABLE_DROPOUT,  # activate dropout
            USE_ALIBI=USE_ALIBI, 
            USE_EXP2=USE_EXP2,
            IS_FP8=IS_FP8,
            FP8_MAX=FP8_MAX,
            DEBUG_TRITON=DEBUG_TRITON,
            DEBUG_TRITON_DETAIL=DEBUG_TRITON_DETAIL,
        )
        start_m += num_steps * MASK_BLOCK_M
        num_steps = tl.cdiv(seqlen_q - start_m, BLOCK_M)
        end_m = start_m + num_steps * BLOCK_M

        if DEBUG_TRITON: print(f"start_m after Masked step: {start_m}; num_steps: {num_steps}")  # noqa: E701
        if DEBUG_TRITON: print(f"unMasked: start_n: {start_n}, start_m: {start_m}, end_m: {end_m}, num_steps: {num_steps}")  # noqa: E701
        if DEBUG_TRITON: print("unMasked")  # noqa: E701
        dk, dv = _bwd_dkdv_inner(
            dk, dv,  # output tensors
            Q_ptr, k, v, DO_ptr, M_ptr, Delta_ptr, sm_scale, # input tensors
            stride_qm, stride_qk,  # strides for q
            stride_dom, stride_dok,  # strides for o
            stride_dropoutm, stride_dropoutn,  # strides for dropout
            stride_deltam,
            BLOCK_M, BLOCK_N,  # block dim
            HEAD_DIM, ACTUAL_HEAD_DIM,  # head dim
            dropout_p, philox_seed, batch_philox_offset, dropout_offset,  #
            alibi_slope,
            seqlen_q, seqlen_k,  # max sequence length for q and k
            start_n, start_m, num_steps,  # iteration numbers
            descale_q, descale_k, descale_v, descale_do, # fp8 descale factors from user
            MASK=False,  # causal masking
            ENABLE_DROPOUT=ENABLE_DROPOUT,  # activate dropout
            USE_ALIBI=USE_ALIBI,
            USE_EXP2=USE_EXP2,
            IS_FP8=IS_FP8,
            FP8_MAX=FP8_MAX,
            DEBUG_TRITON=DEBUG_TRITON,
            DEBUG_TRITON_DETAIL=DEBUG_TRITON_DETAIL,
        )

    # Write back dV and dK.
    adj_dkdv = bid * stride_dkb + hkid * stride_kh + k_start * stride_dkn
    offs_dkdv = offs_n[:, None] * stride_dkn + offs_k[None, :] * stride_dkk
    tl.store(DV + adj_dkdv + offs_dkdv, dv, mask=mask_kv)
    dk *= sm_scale
    tl.store(DK + adj_dkdv + offs_dkdv, dk, mask=mask_kv)


# the main inner-loop logic for computing dQ
@triton.jit
def _bwd_dq_inner(
    dq,  # output
    q, K, V, do, m, Delta, sm_scale, # input
    # shared by Q/K/V.
    stride_qm, stride_qk, stride_kn,  stride_kk, stride_vn, stride_vk,
    stride_dropoutm, stride_dropoutn,  # stride for dropout
    stride_deltam,
    seqlen_q, seqlen_k,  #
    BLOCK_M2: tl.constexpr,  #
    BLOCK_N2: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,
    ACTUAL_HEAD_DIM: tl.constexpr,  #
    dropout_p, philox_seed, batch_philox_offset, dropout_offset,
    alibi_slope,
    # Filled in by the wrapper.
    start_m, start_n, end_n, num_steps,  #
    descale_q, descale_k, descale_v, descale_do, # fp8 descale factors from user
    MASK: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    USE_ALIBI: tl.constexpr,
    USE_EXP2: tl.constexpr,
    IS_FP8: tl.constexpr,
    FP8_MAX: tl.constexpr,
    DEBUG_TRITON: tl.constexpr,
    DEBUG_TRITON_DETAIL: tl.constexpr,
):
    # if HEAD_DIM is padded
    PADDED_HEAD: tl.constexpr = (ACTUAL_HEAD_DIM != HEAD_DIM)
    delta_qk = seqlen_q - seqlen_k
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    offs_k = tl.arange(0, HEAD_DIM)

    # mask to make sure not OOB of seqlen_q
    mask_m = offs_m < seqlen_q

    kT_ptrs = K + offs_n[None, :] * stride_kn + offs_k[:, None] * stride_kk
    vT_ptrs = V + offs_n[None, :] * stride_vn + offs_k[:, None] * stride_vk
    # D (= delta) is pre-divided by ds_scale.
    Di = tl.load(Delta + offs_m * stride_deltam, mask=mask_m, other=0.0)
    # BLOCK_M2 must be a multiple of BLOCK_N2, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    curr_n = start_n
    step_n = BLOCK_N2
    curr_philox_offset = batch_philox_offset
    curr_dropout_offset = dropout_offset
    RCP_LN2: tl.constexpr = 1.4426950408889634  # = 1.0 / ln(2)
    for blk_idx in range(num_steps):
        if DEBUG_TRITON: print(f"iter {blk_idx}: curr_n = {curr_n}")  # noqa: E701
        offs_n = curr_n + tl.arange(0, BLOCK_N2)
        # end_n is needed because the end of causal True might not be perfectly
        # aligned with the end of the block
        mask_n = offs_n < end_n
        if DEBUG_TRITON_DETAIL: print(f"start_n = {start_n}, end_n = {end_n}, offs_n: {offs_n.shape}\n{offs_n}")  # noqa: E701
        if DEBUG_TRITON_DETAIL: print(f"mask_n: {mask_n.shape}\n{mask_n}")  # noqa: E701
        mask_kT = mask_n[None, :]
        mask_mn = mask_m[:, None] & (offs_n[None, :] < end_n)
        if PADDED_HEAD:
            mask_kT &= offs_k[:, None] < ACTUAL_HEAD_DIM

        kT = tl.load(kT_ptrs, mask=mask_kT, other=0.0)
        vT = tl.load(vT_ptrs, mask=mask_kT, other=0.0)

        if ENABLE_DROPOUT:
            # NOTE: dropout is transposed because it is used to mask pT
            philox_offs = curr_philox_offset + \
                          offs_m[:, None] * stride_dropoutm + \
                          offs_n[None, :] * stride_dropoutn
            if tl_DROPOUT_USE_PYTORCH:
                dropout_offs = offs_m[:, None] * stride_dropoutm + \
                               offs_n[None, :] * stride_dropoutn
                dropout_mask = tl.load(
                    curr_dropout_offset + dropout_offs,
                    mask=mask_mn)
            else:
                rand_vals = tl.rand(philox_seed, philox_offs)
                dropout_mask = rand_vals > dropout_p
            dropout_scale = 1 / (1 - dropout_p)

        if IS_FP8:
            qk = (tl.dot(q, kT) * descale_q * descale_k)
        else:
            qk = tl.dot(q, kT)
        qk_scaled = qk * sm_scale

        if USE_ALIBI:
            relative_pos_block = offs_m[:, None] + seqlen_k - seqlen_q - offs_n[None, :]
            alibi_block = -1 * alibi_slope * tl.abs(relative_pos_block)
            qk_scaled += alibi_block

        if DEBUG_TRITON_DETAIL: print(f"qk scaled: {qk.shape}\n", qk_scaled)  # noqa: E701
        if USE_EXP2:
            p = tl.math.exp2(qk_scaled * RCP_LN2 - m * RCP_LN2)
        else:
            p = tl.math.exp(qk_scaled - m)

        # Autoregressive masking.
        if MASK:
            causal_mask = (offs_m[:, None] - delta_qk) >= offs_n[None, :]
            mask = causal_mask & mask_mn
            p = tl.where(mask, p, 0.0)
        # Compute dP and dS.
        if IS_FP8:
            dp = (tl.dot(do, vT) * descale_do * descale_v)
        else:
            dp = tl.dot(do, vT)
        if ENABLE_DROPOUT:
            dp = tl.where(dropout_mask, dp, 0.0) * dropout_scale
        delta_i = Di[:, None]
        ds = p * (dp -delta_i)
        # Compute dQ.
        # NOTE: We need to de-scale dq in the end, because kT was pre-scaled.
        if IS_FP8:
            scale_ds, descale_ds = compute_fp8_scaling_factors(ds, FP8_MAX)
            dq += (tl.dot((ds * scale_ds).to(kT.type.element_ty), tl.trans(kT)) * descale_ds * descale_k)
        else:
            dq += tl.dot(ds.to(kT.type.element_ty), tl.trans(kT))
        # Increment pointers.
        curr_n += step_n
        kT_ptrs += step_n * stride_kn
        vT_ptrs += step_n * stride_vn
    return dq


# grid = (tl.cdiv(max_seqlen_q // BLOCK_M2), batch, nheads_q)
@triton.jit
def _bwd_kernel_dq_causal(
    Q, K, V, sm_scale, DO, DQ,
    M, Delta,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_dqb, stride_dqh, stride_dqm, stride_dqk,
    stride_deltab, stride_deltah, stride_deltam,
    stride_dob, stride_doh, stride_dom, stride_dok,
    stride_dropoutb, stride_dropouth, stride_dropoutm, stride_dropoutn,
    stride_descale_q_z, stride_descale_k_z, stride_descale_v_z, stride_descale_do_z,
    stride_az, stride_ah,
    HQ, HK,
    cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k,
    Dropout_mask, dropout_p, philox_seed, philox_offset_base,
    Alibi_slopes,
    Descale_q, Descale_k, Descale_v, Descale_do,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLK_SLICE_FACTOR: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ACTUAL_HEAD_DIM: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_ALIBI: tl.constexpr,
    USE_EXP2: tl.constexpr,
    IS_FP8: tl.constexpr,
    FP8_MAX: tl.constexpr,
    FP8_OUTPUT: tl.constexpr,
    DEBUG_TRITON: tl.constexpr,
    DEBUG_TRITON_DETAIL: tl.constexpr,
):
    # program ids
    pid = tl.program_id(0)
    bid = tl.program_id(1)
    hkid = tl.program_id(2)
    # figure out varlen start and end
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

    # Figure out causal starting block since we have seqlen_q <=> seqlen_k.
    # Unlike forward pass where we tile on M dim and iterate on N dim, so that
    # we can skip some M blocks, in backward pass, we tile on the N dim for kv
    # and iterate over the M. In this way, we cannot skip N blocks, but only to
    # determine the starting M blocks to skip some initial blocks masked by
    # causal.
    # DQ tiles on M dim and iterate on N dim, so we there could be some tiles we
    # can simply skip and we need to adjust starting position.
    start_m = pid * BLOCK_M
    # seqlen_q > seqlen_k, no need to process these tile for dq
    delta_qk = seqlen_q - seqlen_k
    if DEBUG_TRITON: print(f"end_n = start_m + BLOCK_M = {start_m} + {BLOCK_M} = {start_m + BLOCK_M}")  # noqa: E701
    if start_m + BLOCK_M < delta_qk:
        if DEBUG_TRITON: print(f"start_m + BLOCK_M = {start_m} + {BLOCK_M} = {start_m + BLOCK_M} < delta_qk of {delta_qk}")  # noqa: E701
        return

    offs_k = tl.arange(0, HEAD_DIM)
    offs_m = start_m + tl.arange(0, BLOCK_M)
    # Mask for loading K and V
    mask_q = offs_m[:, None] < seqlen_q
    PADDED_HEAD: tl.constexpr = (ACTUAL_HEAD_DIM != HEAD_DIM)
    if PADDED_HEAD:
        mask_k = offs_k < ACTUAL_HEAD_DIM
        mask_q &= mask_k[None, :]
    offs_q = offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    offs_do = offs_m[:, None] * stride_dom + offs_k[None, :] * stride_dok
    adj_k = bid * stride_kb + hkid * stride_kh + k_start * stride_kn
    adj_v = bid * stride_vb + hkid * stride_vh + k_start * stride_vn
    K +=  adj_k
    V +=  adj_v
    # If MQA / GQA, set the K and V head offsets appropriately.
    GROUP_SIZE = HQ // HK
    for hqid in range(hkid * GROUP_SIZE, hkid * GROUP_SIZE + GROUP_SIZE):
        # seqlen_q < seqlen_k: delta_qk more kv tokens are added at the front
        #   for every M-tile
        end_n = start_m + BLOCK_M - delta_qk
        # clamp end_n at [0, seqlen_k]
        end_n = max(min(end_n, seqlen_k), 0)
        if DEBUG_TRITON: print(f"delta_qk: {delta_qk}; end_n: {end_n}")  # noqa: E701
        # offset input and output tensor by batch and Q/K heads
        adj_q = bid * stride_qb + hqid * stride_qh + q_start * stride_qm
        adj_do = bid * stride_dob + hqid * stride_doh + q_start * stride_dom
        adj_delta = \
            bid * stride_deltab + hqid * stride_deltah + q_start * stride_deltam
        Delta_ptr = Delta + adj_delta

        if USE_ALIBI:
            alibi_offset = bid * stride_az + hqid * stride_ah
            alibi_slope = tl.load(Alibi_slopes + alibi_offset)
        else:
            alibi_slope = None

        # batch_philox_offset is the ACTUALLY dropout offset
        # dropout_offset is for debug purpose and will be removed later
        batch_philox_offset = 0
        dropout_offset = 0
        if ENABLE_DROPOUT:
            batch_philox_offset = philox_offset_base + \
                                  bid * stride_dropoutb + \
                                  hqid * stride_dropouth
            dropout_offset = \
                Dropout_mask + bid * stride_dropoutb + hqid * stride_dropouth

        q = tl.load(Q + adj_q + offs_q, mask=mask_q, other=0.0)
        do = tl.load(DO + adj_do + offs_do, mask=mask_q, other=0.0)
        m = tl.load(M + adj_delta + offs_m * stride_deltam,
                    mask=offs_m < seqlen_q)
        m = m[:, None]

        MASK_BLOCK_N: tl.constexpr = BLOCK_N // BLK_SLICE_FACTOR
        # start can only be 0 at minimum
        start_n = max(end_n - BLOCK_M, 0)
        num_steps = tl.cdiv(end_n - start_n, MASK_BLOCK_N)

        if IS_FP8:
            descale_q = tl.load(Descale_q + bid * stride_descale_q_z + hqid)
            descale_k = tl.load(Descale_k + bid * stride_descale_k_z + hkid)
            descale_v = tl.load(Descale_v + bid * stride_descale_v_z + hkid)
            descale_do = tl.load(Descale_do + bid * stride_descale_do_z + hqid)
        else:
            descale_q, descale_k, descale_v, descale_do = 1.0, 1.0, 1.0, 1.0

        dq = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
        if DEBUG_TRITON: print(f"pid: {pid}; end_n: {end_n}, start_m: {start_m}")  # noqa: E701
        # Compute dQ for masked (diagonal) blocks.
        # NOTE: This code scans each row of QK^T backward (from right to left,
        # but inside each call to _bwd_dq_inner, from left to right), but that's
        # not due to anything important.  I just wanted to reuse the loop
        # structure for dK & dV above as much as possible.
        if DEBUG_TRITON: print(f"Masked: start_m: {start_m}, start_n: {start_n}, end_n: {end_n}, num_steps: {num_steps}")  # noqa: E701
        dq = _bwd_dq_inner(
            dq,
            q, K, V, do, m, Delta_ptr, sm_scale,
            stride_qm, stride_qk, stride_kn, stride_kk, stride_vn, stride_vk,
            stride_dropoutm, stride_dropoutn,
            stride_deltam,
            seqlen_q, seqlen_k,
            BLOCK_M, MASK_BLOCK_N,
            HEAD_DIM, ACTUAL_HEAD_DIM,
            dropout_p, philox_seed, batch_philox_offset, dropout_offset,
            alibi_slope,
            start_m, start_n, end_n, num_steps,
            descale_q, descale_k, descale_v, descale_do,
            MASK=True,
            ENABLE_DROPOUT=ENABLE_DROPOUT,
            USE_ALIBI=USE_ALIBI,
            USE_EXP2=USE_EXP2,
            IS_FP8=IS_FP8,
            FP8_MAX=FP8_MAX,
            DEBUG_TRITON=DEBUG_TRITON,
            DEBUG_TRITON_DETAIL=DEBUG_TRITON_DETAIL,
        )
        end_n -= num_steps * MASK_BLOCK_N
        num_steps = tl.cdiv(end_n, BLOCK_N)
        start_n = max(end_n - num_steps * BLOCK_N, 0)
        if DEBUG_TRITON: print(f"unMasked: start_m: {start_m}, start_n: {start_n}, end_n: {end_n}, num_steps: {num_steps}")  # noqa: E701
        dq = _bwd_dq_inner(
            dq,
            q, K, V, do, m, Delta_ptr, sm_scale,
            stride_qm, stride_qk, stride_kn, stride_kk, stride_vn, stride_vk,
            stride_dropoutm, stride_dropoutn,
            stride_deltam,
            seqlen_q, seqlen_k,
            BLOCK_M, BLOCK_N,
            HEAD_DIM, ACTUAL_HEAD_DIM,
            dropout_p, philox_seed, batch_philox_offset, dropout_offset,
            alibi_slope,
            start_m, start_n, end_n, num_steps,
            descale_q, descale_k, descale_v, descale_do,
            MASK=False,
            ENABLE_DROPOUT=ENABLE_DROPOUT,
            USE_ALIBI=USE_ALIBI,
            USE_EXP2=USE_EXP2,
            IS_FP8=IS_FP8,
            FP8_MAX=FP8_MAX,
            DEBUG_TRITON=DEBUG_TRITON,
            DEBUG_TRITON_DETAIL=DEBUG_TRITON_DETAIL,
        )
        # Write back dQ.
        adj_dq = bid * stride_dqb + hqid * stride_dqh + q_start * stride_dqm
        offs_dq = offs_m[:, None] * stride_dqm + offs_k[None, :] * stride_dqk
        dq *= sm_scale
        tl.store(DQ + adj_dq + offs_dq, dq, mask=mask_q)


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
    stride_az, stride_ah,
    HQ, HK,
    cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k,
    Dropout_mask, dropout_p, philox_seed, philox_offset_base,
    Alibi_slopes,
    Descale_q, Descale_k, Descale_v, Descale_do,
    BLOCK_M: tl.constexpr,  # 32
    BLOCK_N: tl.constexpr,  # 128
    BLK_SLICE_FACTOR: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ACTUAL_HEAD_DIM: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_ALIBI: tl.constexpr,
    USE_EXP2: tl.constexpr,
    IS_FP8: tl.constexpr,
    FP8_MAX: tl.constexpr,
    FP8_OUTPUT: tl.constexpr,
    DEBUG_TRITON: tl.constexpr,
    DEBUG_TRITON_DETAIL: tl.constexpr,
):
    # program ids
    pid = tl.program_id(0)
    bid = tl.program_id(1)
    hkid = tl.program_id(2)
    # figure out varlen start and end
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

    dk = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)

    start_n = pid * BLOCK_N

    offs_k = tl.arange(0, HEAD_DIM)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    # Mask for loading K and V
    mask_kv = offs_n[:, None] < seqlen_k
    PADDED_HEAD: tl.constexpr = (ACTUAL_HEAD_DIM != HEAD_DIM)
    if PADDED_HEAD:
        mask_k = offs_k < ACTUAL_HEAD_DIM
        mask_kv &= mask_k[None, :]

    GROUP_SIZE = HQ // HK
    # K/V tensors not changed for the group
    adj_k = bid * stride_kb + hkid * stride_kh + k_start * stride_kn + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk
    adj_v = bid * stride_vb + hkid * stride_vh + k_start * stride_vn + offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk
    # load K and V: they stay in SRAM throughout the inner loop.
    k = tl.load(K + adj_k, mask=mask_kv, other=0.0)
    v = tl.load(V + adj_v, mask=mask_kv, other=0.0)
    # If MQA / GQA, set the K and V head offsets appropriately.
    for hqid in range(hkid * GROUP_SIZE, hkid * GROUP_SIZE + GROUP_SIZE):
        # offset input and output tensor by batch and Q/K heads
        adj_q = bid * stride_qb + hqid * stride_qh + q_start * stride_qm
        Q_ptr = Q + adj_q
        adj_do = bid * stride_dob + hqid * stride_doh + q_start * stride_dom
        DO_ptr = DO + adj_do
        adj_delta = bid * stride_deltab + hqid * stride_deltah + q_start * stride_deltam
        M_ptr = M + adj_delta
        Delta_ptr = Delta + adj_delta

        if USE_ALIBI:
            alibi_offset = bid * stride_az + hqid * stride_ah
            alibi_slope = tl.load(Alibi_slopes + alibi_offset)
        else:
            alibi_slope = None

        # batch_philox_offset is the ACTUALLY dropout offset
        # dropout_offset is for debug purpose and will be removed later
        batch_philox_offset = 0
        dropout_offset = 0
        if ENABLE_DROPOUT:
            batch_philox_offset = philox_offset_base + bid * stride_dropoutb + \
                                  hqid * stride_dropouth
            dropout_offset = Dropout_mask + bid * stride_dropoutb + \
                             hqid * stride_dropouth
            
        if IS_FP8:
            descale_q = tl.load(Descale_q + bid * stride_descale_q_z + hqid)
            descale_k = tl.load(Descale_k + bid * stride_descale_k_z + hkid)
            descale_v = tl.load(Descale_v + bid * stride_descale_v_z + hkid)
            descale_do = tl.load(Descale_do + bid * stride_descale_do_z + hqid)
        else:
            descale_q, descale_k, descale_v, descale_do = 1.0, 1.0, 1.0, 1.0

        # because there is no causal, we always start from the beginning
        start_m = 0
        num_steps = tl.cdiv(seqlen_q, BLOCK_M)
        dk, dv = _bwd_dkdv_inner(
            dk, dv,  # output tensors
            Q_ptr, k, v, DO_ptr, M_ptr, Delta_ptr, sm_scale, # input tensors
            stride_qm, stride_qk,  # strides for q
            stride_dom, stride_dok,  # strides for o
            stride_dropoutm, stride_dropoutn,  # strides for dropout
            stride_deltam,
            BLOCK_M, BLOCK_N,  # block dim
            HEAD_DIM, ACTUAL_HEAD_DIM,  # head dim
            dropout_p, philox_seed, batch_philox_offset, dropout_offset,  #
            alibi_slope,
            seqlen_q, seqlen_k,  # max sequence length for q and k
            start_n, start_m, num_steps,  # iteration numbers
            descale_q, descale_k, descale_v, descale_do, # fp8 descale factors from user
            MASK=False,  # causal masking
            ENABLE_DROPOUT=ENABLE_DROPOUT,  # activate dropout
            USE_ALIBI=USE_ALIBI,
            USE_EXP2=USE_EXP2,
            IS_FP8=IS_FP8,
            FP8_MAX=FP8_MAX,
            DEBUG_TRITON=DEBUG_TRITON,
            DEBUG_TRITON_DETAIL=DEBUG_TRITON_DETAIL,
        )

    # Write back dV and dK.
    adj_dkdv = bid * stride_dkb + hkid * stride_kh + k_start * stride_dkn
    offs_dkdv = offs_n[:, None] * stride_dkn + offs_k[None, :] * stride_dkk
    tl.store(DV + adj_dkdv + offs_dkdv, dv, mask=mask_kv)
    dk *= sm_scale
    tl.store(DK + adj_dkdv + offs_dkdv, dk, mask=mask_kv)


@triton.jit
def _bwd_kernel_dq_noncausal(
    Q, K, V, sm_scale, DO, DQ,
    M, Delta,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_dqb, stride_dqh, stride_dqm, stride_dqk,
    stride_deltab, stride_deltah, stride_deltam,
    stride_dob, stride_doh, stride_dom, stride_dok,
    stride_dropoutb, stride_dropouth, stride_dropoutm, stride_dropoutn,
    stride_descale_q_z, stride_descale_k_z, stride_descale_v_z, stride_descale_do_z,
    stride_az, stride_ah,
    HQ, HK,
    cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k,
    Dropout_mask, dropout_p, philox_seed, philox_offset_base,
    Alibi_slopes,
    Descale_q, Descale_k, Descale_v, Descale_do,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLK_SLICE_FACTOR: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ACTUAL_HEAD_DIM: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_ALIBI: tl.constexpr,
    USE_EXP2: tl.constexpr,
    IS_FP8: tl.constexpr,
    FP8_MAX: tl.constexpr,
    FP8_OUTPUT: tl.constexpr,
    DEBUG_TRITON: tl.constexpr,
    DEBUG_TRITON_DETAIL: tl.constexpr,
):
    # program ids
    pid = tl.program_id(0)
    bid = tl.program_id(1)
    hkid = tl.program_id(2)
    # figure out varlen start and end
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

    offs_k = tl.arange(0, HEAD_DIM)
    offs_m = start_m + tl.arange(0, BLOCK_M)
    # Mask for loading K and V
    mask_q = offs_m[:, None] < seqlen_q
    PADDED_HEAD: tl.constexpr = (ACTUAL_HEAD_DIM != HEAD_DIM)
    if PADDED_HEAD:
        mask_k = offs_k < ACTUAL_HEAD_DIM
        mask_q &= mask_k[None, :]
    offs_q = offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    offs_do = offs_m[:, None] * stride_dom + offs_k[None, :] * stride_dok
    adj_k = bid * stride_kb + hkid * stride_kh + k_start * stride_kn
    adj_v = bid * stride_vb + hkid * stride_vh + k_start * stride_vn
    K +=  adj_k
    V +=  adj_v
    # If MQA / GQA, set the K and V head offsets appropriately.
    GROUP_SIZE = HQ // HK
    for hqid in range(hkid * GROUP_SIZE, hkid * GROUP_SIZE + GROUP_SIZE):
        # offset input and output tensor by batch and Q/K heads
        adj_q = bid * stride_qb + hqid * stride_qh + q_start * stride_qm
        adj_do = bid * stride_dob + hqid * stride_doh + q_start * stride_dom
        adj_delta = \
            bid * stride_deltab + hqid * stride_deltah + q_start * stride_deltam
        Delta_ptr = Delta + adj_delta

        if USE_ALIBI:
            alibi_offset = bid * stride_az + hqid * stride_ah
            alibi_slope = tl.load(Alibi_slopes + alibi_offset)
        else:
            alibi_slope = None

        # batch_philox_offset is the ACTUALLY dropout offset
        # dropout_offset is for debug purpose and will be removed later
        batch_philox_offset = 0
        dropout_offset = 0
        if ENABLE_DROPOUT:
            batch_philox_offset = philox_offset_base + \
                                  bid * stride_dropoutb + \
                                  hqid * stride_dropouth
            dropout_offset = \
                Dropout_mask + bid * stride_dropoutb + hqid * stride_dropouth

        q = tl.load(Q + adj_q + offs_q, mask=mask_q, other=0.0)
        do = tl.load(DO + adj_do + offs_do, mask=mask_q, other=0.0)
        m = tl.load(M + adj_delta + offs_m * stride_deltam,
                    mask=offs_m < seqlen_q)
        m = m[:, None]

        if IS_FP8:
            descale_q = tl.load(Descale_q + bid * stride_descale_q_z + hqid)
            descale_k = tl.load(Descale_k + bid * stride_descale_k_z + hkid)
            descale_v = tl.load(Descale_v + bid * stride_descale_v_z + hkid)
            descale_do = tl.load(Descale_do + bid * stride_descale_do_z + hqid)
        else:
            descale_q, descale_k, descale_v, descale_do = 1.0, 1.0, 1.0, 1.0

        # start can only be 0 at minimum
        start_n = 0
        end_n = seqlen_k
        num_steps = tl.cdiv(seqlen_k, BLOCK_N)
        dq = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
        dq = _bwd_dq_inner(
            dq,
            q, K, V, do, m, Delta_ptr, sm_scale,
            stride_qm, stride_qk, stride_kn, stride_kk, stride_vn, stride_vk,
            stride_dropoutm, stride_dropoutn,
            stride_deltam,
            seqlen_q, seqlen_k,
            BLOCK_M, BLOCK_N,
            HEAD_DIM, ACTUAL_HEAD_DIM,
            dropout_p, philox_seed, batch_philox_offset, dropout_offset,
            alibi_slope,
            start_m, start_n, end_n, num_steps,
            descale_q, descale_k, descale_v, descale_do,
            MASK=False,
            ENABLE_DROPOUT=ENABLE_DROPOUT,
            USE_ALIBI=USE_ALIBI,
            USE_EXP2=USE_EXP2,
            IS_FP8=IS_FP8,
            FP8_MAX=FP8_MAX,
            DEBUG_TRITON=DEBUG_TRITON,
            DEBUG_TRITON_DETAIL=DEBUG_TRITON_DETAIL,
        )
        # Write back dQ.
        adj_dq = bid * stride_dqb + hqid * stride_dqh + q_start * stride_dqm
        offs_dq = offs_m[:, None] * stride_dqm + offs_k[None, :] * stride_dqk
        dq *= sm_scale
        tl.store(DQ + adj_dq + offs_dq, dq, mask=mask_q)


def attention_prefill_backward_triton_split_impl(
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
    use_exp2: bool,
    # fp8
    descale_q: Optional[torch.Tensor],
    descale_k: Optional[torch.Tensor],
    descale_v: Optional[torch.Tensor],
    descale_o: Optional[torch.Tensor],
    descale_do: Optional[torch.Tensor],
    descale_dq: Optional[torch.Tensor],
    descale_dk: Optional[torch.Tensor],
    descale_dv: Optional[torch.Tensor],
):
    # debug
    DEBUG_TRITON: bool = False
    DEBUG_TRITON_DETAIL: bool = False

    # fp8
    IS_FP8 = is_fp8(q)
    if IS_FP8:
        FP8_MAX = torch.finfo(q.dtype).max
        # assert that the main inputs are fp8
        assert is_fp8(do) and is_fp8(q) and is_fp8(k) and is_fp8(v), f"Non fp8 type found: do.dtype={do.dtype}, q.dtype={q.dtype}, k.dtype={k.dtype}, v.dtype={v.dtype}. All tensors must be fp8."
        if is_fp8(o):
            FP8_OUTPUT = True
            assert descale_o is not None, f"descale_o is None. In fp8, you need to pass a tensor for descale_o along with a tensor o."
            assert descale_dq is not None, f"descale_dq is None. In fp8, you need to pass a tensor for descale_dq along with a tensor dq."
            assert descale_dk is not None, f"descale_dk is None. In fp8, you need to pass a tensor for descale_dk along with a tensor dk."
            assert descale_dv is not None, f"descale_dv is None. In fp8, you need to pass a tensor for descale_dv along with a tensor dv."
        else:
            FP8_OUTPUT = False

        stride_descale_q_z = descale_q.stride(0) if descale_q is not None else None
        stride_descale_k_z = descale_k.stride(0) if descale_k is not None else None
        stride_descale_v_z = descale_v.stride(0) if descale_v is not None else None
        stride_descale_o_z = descale_o.stride(0) if descale_o is not None else None
        stride_descale_do_z = descale_do.stride(0) if descale_do is not None else None
    else:
        FP8_MAX = None
        FP8_OUTPUT = False
        stride_descale_q_z = stride_descale_k_z = stride_descale_v_z = stride_descale_o_z = stride_descale_do_z = None


    # get strides and shape
    batch, nheads_q, nheads_k, head_size, max_seqlen_q_final, max_seqlen_k_final = \
        get_shapes_from_layout(
            q, k, layout,
            cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k
        )
    q_strides, k_strides, v_strides, o_strides = \
        get_strides_from_layout(q, k, v, o, layout)
    stride_qb, stride_qh, stride_qm, stride_qk =  q_strides
    stride_kb, stride_kh, stride_kn, stride_kk = k_strides
    stride_vb, stride_vh, stride_vn, stride_vk = v_strides
    stride_ob, stride_oh, stride_om, stride_ok = o_strides
    dq_strides, dk_strides, dv_strides, do_strides = \
        get_strides_from_layout(dq, dk, dv, do, layout)
    stride_dqb, stride_dqh, stride_dqm, stride_dqk =  dq_strides
    stride_dkb, stride_dkh, stride_dkn, stride_dkk = dk_strides
    stride_dvb, stride_dvh, stride_dvn, stride_dvk = dv_strides
    stride_dob, stride_doh, stride_dom, stride_dok = do_strides
    IS_VARLEN = layout == "thd"
    use_dropout = (dropout_p > 0.0)
    use_alibi, (stride_az, stride_ah) = (True, alibi_slopes.stride()) if alibi_slopes is not None else (False, (0, 0))

    # get closest power of 2 over or equal to 32.
    padded_d_model = 1 << (head_size - 1).bit_length()
    padded_d_model = max(padded_d_model, 32) # NOTE: the causal path expects a min of 32. It will cause a compiler assert.
    HEAD_DIM = padded_d_model
    ACTUAL_HEAD_DIM = head_size
    # meta-parameters
    # TODO: fix num_stages later
    NUM_WARPS, NUM_STAGES = 4, 1
    WAVES_PER_EU = 1
    PRE_BLOCK = 128
    BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 128, 128, 32
    BLK_SLICE_FACTOR = 2

    # init delta
    delta = torch.zeros_like(softmax_lse)
    if IS_VARLEN:
        stride_deltab = 0
        stride_deltah, stride_deltam = delta.stride()
    else:
        stride_deltab, stride_deltah, stride_deltam = delta.stride()
    pre_grid = (triton.cdiv(max_seqlen_q_final, PRE_BLOCK), batch, nheads_q)
    _bwd_preprocess[pre_grid](
        o, do,
        delta,
        stride_ob, stride_oh, stride_om, stride_ok,
        stride_deltab, stride_deltah, stride_deltam,
        stride_descale_do_z,
        cu_seqlens_q, max_seqlen_q_final,
        descale_do,
        BLOCK_M=PRE_BLOCK,
        HEAD_DIM=HEAD_DIM,
        ACTUAL_HEAD_DIM=ACTUAL_HEAD_DIM,
        IS_VARLEN=IS_VARLEN,
        IS_FP8=IS_FP8
    )
    
    if DEBUG:
        print("delta:", delta, delta.shape)

    # dropout mask tensor for debugging. We dump the dropout mask created in
    #   the kernel for testing
    dropout_mask = None
    stride_dropoutb, stride_dropouth, stride_dropoutm, stride_dropoutn = \
        (0, 0 , 0 , 0)
    if use_dropout:
        dropout_mask = torch.zeros(
            (batch, nheads_q, max_seqlen_q_final, max_seqlen_k_final),
            device=q.device,
            dtype=torch.float32
        )

        if DROPOUT_USE_PYTORCH:
            if not IS_VARLEN:
                dropout_mask = create_dropout_mask(
                    dropout_p,
                    (batch, nheads_q, max_seqlen_q_final, max_seqlen_k_final),
                    seed = philox_seed
                )
            else:
                dropout_mask = create_dropout_mask_varlen(
                    dropout_p, batch, nheads_q,
                    cu_seqlens_q, cu_seqlens_k, philox_seed
                )
        stride_dropoutb, stride_dropouth, stride_dropoutm, stride_dropoutn = \
            dropout_mask.stride()

    grid_dkdv = ((max_seqlen_k_final + BLOCK_N1 - 1) // BLOCK_N1, batch, nheads_k)
    grid_dq = ((max_seqlen_q_final + BLOCK_M2 - 1) // BLOCK_M2, batch, nheads_k)
    if causal:
        if DEBUG_TRITON: print(f"_bwd_kernel_dkdv: grid = {grid_dkdv}, block_size = ({BLOCK_M1, BLOCK_N1})", )  # noqa: E701
        _bwd_kernel_dkdv_causal[grid_dkdv](
            q, k, v, sm_scale, do, dk, dv,
            softmax_lse, delta,
            stride_qb, stride_qh, stride_qm, stride_qk,
            stride_kb, stride_kh, stride_kn, stride_kk,
            stride_vb, stride_vh, stride_vn, stride_vk,
            stride_dkb, stride_dkh, stride_dkn, stride_dkk,
            stride_deltab, stride_deltah, stride_deltam,
            stride_dob, stride_doh, stride_dom, stride_dok,
            stride_dropoutb, stride_dropouth, stride_dropoutm, stride_dropoutn,
            stride_descale_q_z, stride_descale_k_z, stride_descale_v_z, stride_descale_do_z,
            stride_az, stride_ah,
            nheads_q, nheads_k,
            cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q_final, max_seqlen_k_final,
            dropout_mask, dropout_p, philox_seed, philox_offset,
            alibi_slopes,
            descale_q, descale_k, descale_v, descale_do,
            BLOCK_M1, BLOCK_N1, BLK_SLICE_FACTOR,
            HEAD_DIM, ACTUAL_HEAD_DIM,
            ENABLE_DROPOUT=use_dropout,
            IS_VARLEN=IS_VARLEN,
            USE_ALIBI=use_alibi,
            USE_EXP2=use_exp2,
            IS_FP8=IS_FP8,
            FP8_MAX=FP8_MAX,
            FP8_OUTPUT=FP8_OUTPUT,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
            waves_per_eu = WAVES_PER_EU,
            DEBUG_TRITON=DEBUG_TRITON,
            DEBUG_TRITON_DETAIL=DEBUG_TRITON_DETAIL,
        )

        if DEBUG_TRITON: print(f"\n_bwd_kernel_dq: grid = {grid_dq}, block_size = ({BLOCK_M2, BLOCK_N2})", )  # noqa: E701
        _bwd_kernel_dq_causal[grid_dq](
            q, k, v, sm_scale, do, dq,
            softmax_lse, delta,
            stride_qb, stride_qh, stride_qm, stride_qk,
            stride_kb, stride_kh, stride_kn, stride_kk,
            stride_vb, stride_vh, stride_vn, stride_vk,
            stride_dqb, stride_dqh, stride_dqm, stride_dqk,
            stride_deltab, stride_deltah, stride_deltam,
            stride_dob, stride_doh, stride_dom, stride_dok,
            stride_dropoutb, stride_dropouth, stride_dropoutm, stride_dropoutn,
            stride_descale_q_z, stride_descale_k_z, stride_descale_v_z, stride_descale_do_z,
            stride_az, stride_ah,
            nheads_q, nheads_k,
            cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q_final, max_seqlen_k_final,
            dropout_mask, dropout_p, philox_seed, philox_offset,
            alibi_slopes,
            descale_q, descale_k, descale_v, descale_do,
            BLOCK_M2, BLOCK_N2, BLK_SLICE_FACTOR,
            HEAD_DIM, ACTUAL_HEAD_DIM,
            ENABLE_DROPOUT=use_dropout,
            IS_VARLEN=IS_VARLEN,
            USE_ALIBI=use_alibi,
            USE_EXP2=use_exp2,
            IS_FP8=IS_FP8,
            FP8_MAX=FP8_MAX,
            FP8_OUTPUT=FP8_OUTPUT,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
            waves_per_eu = WAVES_PER_EU,
            DEBUG_TRITON=DEBUG_TRITON,
            DEBUG_TRITON_DETAIL=DEBUG_TRITON_DETAIL,
        )
    else:
        _bwd_kernel_dkdv_noncausal[grid_dkdv](
            q, k, v, sm_scale, do, dk, dv,
            softmax_lse, delta,
            stride_qb, stride_qh, stride_qm, stride_qk,
            stride_kb, stride_kh, stride_kn, stride_kk,
            stride_vb, stride_vh, stride_vn, stride_vk,
            stride_dkb, stride_dkh, stride_dkn, stride_dkk,
            stride_deltab, stride_deltah, stride_deltam,
            stride_dob, stride_doh, stride_dom, stride_dok,
            stride_dropoutb, stride_dropouth, stride_dropoutm, stride_dropoutn,
            stride_descale_q_z, stride_descale_k_z, stride_descale_v_z, stride_descale_do_z,
            stride_az, stride_ah,
            nheads_q, nheads_k,
            cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q_final, max_seqlen_k_final,
            dropout_mask, dropout_p, philox_seed, philox_offset,
            alibi_slopes,
            descale_q, descale_k, descale_v, descale_do,
            BLOCK_M1, BLOCK_N1, BLK_SLICE_FACTOR,
            HEAD_DIM, ACTUAL_HEAD_DIM,
            ENABLE_DROPOUT=use_dropout,
            IS_VARLEN=IS_VARLEN,
            USE_ALIBI=use_alibi,
            USE_EXP2=use_exp2,
            IS_FP8=IS_FP8,
            FP8_MAX=FP8_MAX,
            FP8_OUTPUT=FP8_OUTPUT,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
            waves_per_eu = WAVES_PER_EU,
            DEBUG_TRITON=DEBUG_TRITON,
            DEBUG_TRITON_DETAIL=DEBUG_TRITON_DETAIL,
        )

        _bwd_kernel_dq_noncausal[grid_dq](
            q, k, v, sm_scale, do, dq,
            softmax_lse, delta,
            stride_qb, stride_qh, stride_qm, stride_qk,
            stride_kb, stride_kh, stride_kn, stride_kk,
            stride_vb, stride_vh, stride_vn, stride_vk,
            stride_dqb, stride_dqh, stride_dqm, stride_dqk,
            stride_deltab, stride_deltah, stride_deltam,
            stride_dob, stride_doh, stride_dom, stride_dok,
            stride_dropoutb, stride_dropouth, stride_dropoutm, stride_dropoutn,
            stride_descale_q_z, stride_descale_k_z, stride_descale_v_z, stride_descale_do_z,
            stride_az, stride_ah,
            nheads_q, nheads_k,
            cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q_final, max_seqlen_k_final,
            dropout_mask, dropout_p, philox_seed, philox_offset,
            alibi_slopes,
            descale_q, descale_k, descale_v, descale_do,
            BLOCK_M2, BLOCK_N2, BLK_SLICE_FACTOR,
            HEAD_DIM, ACTUAL_HEAD_DIM,
            ENABLE_DROPOUT=use_dropout,
            IS_VARLEN=IS_VARLEN,
            USE_ALIBI=use_alibi,
            USE_EXP2=use_exp2,
            IS_FP8=IS_FP8,
            FP8_MAX=FP8_MAX,
            FP8_OUTPUT=FP8_OUTPUT,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
            waves_per_eu = WAVES_PER_EU,
            DEBUG_TRITON=DEBUG_TRITON,
            DEBUG_TRITON_DETAIL=DEBUG_TRITON_DETAIL,
        )

    return delta
