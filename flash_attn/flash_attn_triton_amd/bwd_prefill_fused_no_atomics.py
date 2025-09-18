import os
import torch
import triton # type: ignore
import triton.language as tl # type: ignore
from typing import Literal, Optional
from .utils import AUTOTUNE, DROPOUT_USE_PYTORCH, DROPOUT_DUMP, DEBUG, compute_fp8_scaling_factors, \
    create_dropout_mask, create_dropout_mask_varlen, is_cdna, is_fp8, is_rdna, round_multiple

# NOTE: triton fails to import tl.constexprs so create them here for the file
tl_DROPOUT_USE_PYTORCH: tl.constexpr = triton.language.constexpr(DROPOUT_USE_PYTORCH)
tl_DROPOUT_DUMP: tl.constexpr = triton.language.constexpr(DROPOUT_DUMP)


def get_autotune_configs():
    if False:
        if is_cdna():
            # shared meta-parameters
            NUM_STAGES = 1
            NUM_WARPS = 4
            WAVES_PER_EU = 2
            MATRIX_INSTR_NONKDIM = 16

            preprocess_autotune_configs = [
                triton.Config({"PRE_BLOCK": 128, "waves_per_eu": WAVES_PER_EU, "matrix_instr_nonkdim": MATRIX_INSTR_NONKDIM}, num_stages=NUM_STAGES, num_warps=NUM_WARPS), # og config
                triton.Config({"PRE_BLOCK": 64, "waves_per_eu": WAVES_PER_EU, "matrix_instr_nonkdim": MATRIX_INSTR_NONKDIM}, num_stages=NUM_STAGES, num_warps=NUM_WARPS),
                triton.Config({"PRE_BLOCK": 32, "waves_per_eu": WAVES_PER_EU, "matrix_instr_nonkdim": MATRIX_INSTR_NONKDIM}, num_stages=NUM_STAGES, num_warps=NUM_WARPS),
                triton.Config({"PRE_BLOCK": 16, "waves_per_eu": WAVES_PER_EU, "matrix_instr_nonkdim": MATRIX_INSTR_NONKDIM}, num_stages=NUM_STAGES, num_warps=NUM_WARPS),
            ]
            preprocess_autotune_keys = [
                "IS_CAUSAL", "dropout_p", "MAX_SEQLENS_Q", "MAX_SEQLENS_K", 
                "ACTUAL_HEAD_DIM_QK", "ACTUAL_HEAD_DIM_V", "IS_VARLEN", "HQ", "HK",
            ]
            causal_autotune_configs = [
                triton.Config({"BLOCK_M1": 32, "BLOCK_N1": 128, "BLOCK_M2": 128, "BLOCK_N2": 32, "BLK_SLICE_FACTOR": 2, "waves_per_eu": WAVES_PER_EU, "matrix_instr_nonkdim": MATRIX_INSTR_NONKDIM}, num_stages=NUM_STAGES, num_warps=NUM_WARPS),  # og config
                triton.Config({'BLOCK_M1': 16, 'BLOCK_N1': 128, 'BLOCK_M2': 128, 'BLOCK_N2': 16, 'BLK_SLICE_FACTOR': 2, "waves_per_eu": WAVES_PER_EU, "matrix_instr_nonkdim": MATRIX_INSTR_NONKDIM}, num_stages=NUM_STAGES, num_warps=NUM_WARPS),
                triton.Config({'BLOCK_M1': 16, 'BLOCK_N1': 64, 'BLOCK_M2': 64, 'BLOCK_N2': 16, 'BLK_SLICE_FACTOR': 2, "waves_per_eu": WAVES_PER_EU, "matrix_instr_nonkdim": MATRIX_INSTR_NONKDIM}, num_stages=NUM_STAGES, num_warps=NUM_WARPS),
                triton.Config({'BLOCK_M1': 32, 'BLOCK_N1': 64, 'BLOCK_M2': 64, 'BLOCK_N2': 32, 'BLK_SLICE_FACTOR': 2, "waves_per_eu": WAVES_PER_EU, "matrix_instr_nonkdim": MATRIX_INSTR_NONKDIM}, num_stages=NUM_STAGES, num_warps=NUM_WARPS),
            ]
            causal_autotune_keys = [
                "IS_CAUSAL", "dropout_p", "MAX_SEQLENS_Q", "MAX_SEQLENS_K", 
                "ACTUAL_HEAD_DIM_QK", "ACTUAL_HEAD_DIM_V", "IS_VARLEN", "HQ", "HK",
            ]
            noncausal_autotune_configs = [
                triton.Config({"BLOCK_M1": 32, "BLOCK_N1": 128, "BLOCK_M2": 128, "BLOCK_N2": 32, "BLK_SLICE_FACTOR": 2, "waves_per_eu": WAVES_PER_EU, "matrix_instr_nonkdim": MATRIX_INSTR_NONKDIM}, num_stages=NUM_STAGES, num_warps=NUM_WARPS),  # og config
                triton.Config({'BLOCK_M1': 16, 'BLOCK_N1': 128, 'BLOCK_M2': 128, 'BLOCK_N2': 16, 'BLK_SLICE_FACTOR': 2, "waves_per_eu": WAVES_PER_EU, "matrix_instr_nonkdim": MATRIX_INSTR_NONKDIM}, num_stages=NUM_STAGES, num_warps=NUM_WARPS),
                triton.Config({'BLOCK_M1': 16, 'BLOCK_N1': 64, 'BLOCK_M2': 64, 'BLOCK_N2': 16, 'BLK_SLICE_FACTOR': 2, "waves_per_eu": WAVES_PER_EU, "matrix_instr_nonkdim": MATRIX_INSTR_NONKDIM}, num_stages=NUM_STAGES, num_warps=NUM_WARPS),
                triton.Config({'BLOCK_M1': 32, 'BLOCK_N1': 64, 'BLOCK_M2': 64, 'BLOCK_N2': 32, 'BLK_SLICE_FACTOR': 2, "waves_per_eu": WAVES_PER_EU, "matrix_instr_nonkdim": MATRIX_INSTR_NONKDIM}, num_stages=NUM_STAGES, num_warps=NUM_WARPS),
            ]
            noncausal_autotune_keys = [
                "IS_CAUSAL", "dropout_p", "MAX_SEQLENS_Q", "MAX_SEQLENS_K", 
                "ACTUAL_HEAD_DIM_QK", "ACTUAL_HEAD_DIM_V", "IS_VARLEN", "HQ", "HK",
            ]
            
            return (preprocess_autotune_configs, preprocess_autotune_keys), (causal_autotune_configs, causal_autotune_keys), (noncausal_autotune_configs, noncausal_autotune_keys)
        else:
            raise ValueError("Unknown Device Type")
    else:
        # meta-parameters
        # TODO: fix num_stages later
        NUM_WARPS, NUM_STAGES = 4, 1
        WAVES_PER_EU = 1
        PRE_BLOCK = 128
        BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 128, 128, 32
        BLK_SLICE_FACTOR = 2

        assert BLOCK_N1 == BLOCK_M2
        
        # configs for the kernels
        preprocess_autotune_configs = [
            triton.Config({"PRE_BLOCK": PRE_BLOCK, "waves_per_eu": WAVES_PER_EU}, num_stages=NUM_STAGES, num_warps=NUM_WARPS),
        ]
        preprocess_autotune_keys = [
            "max_seqlen_q",
           "ACTUAL_HEAD_DIM_V", "IS_VARLEN",
        ]
        causal_autotune_configs = [
            triton.Config({"BLOCK_M1": BLOCK_M1, "BLOCK_N1": BLOCK_N1, "BLOCK_M2": BLOCK_M2, "BLOCK_N2": BLOCK_N2, "BLK_SLICE_FACTOR": BLK_SLICE_FACTOR, "waves_per_eu": WAVES_PER_EU}, num_stages=NUM_STAGES, num_warps=NUM_WARPS),
        ]
        causal_autotune_keys = [
            "dropout_p", "max_seqlen_q", "max_seqlen_k", 
            "ACTUAL_HEAD_DIM_QK", "ACTUAL_HEAD_DIM_V", "IS_VARLEN", "HQ", "HK",
        ]
        noncausal_autotune_configs = [
            triton.Config({"BLOCK_M1": BLOCK_M1, "BLOCK_N1": BLOCK_N1, "BLOCK_M2": BLOCK_M2, "BLOCK_N2": BLOCK_N2, "BLK_SLICE_FACTOR": BLK_SLICE_FACTOR, "waves_per_eu": WAVES_PER_EU}, num_stages=NUM_STAGES, num_warps=NUM_WARPS),
        ]
        noncausal_autotune_keys = [
            "dropout_p", "max_seqlen_q", "max_seqlen_k", 
            "ACTUAL_HEAD_DIM_QK", "ACTUAL_HEAD_DIM_V", "IS_VARLEN", "HQ", "HK",
        ]
        return (preprocess_autotune_configs, preprocess_autotune_keys), (causal_autotune_configs, causal_autotune_keys), (noncausal_autotune_configs, noncausal_autotune_keys)
    


(preprocess_autotune_configs, preprocess_autotune_keys), (causal_autotune_configs, causal_autotune_keys), (noncausal_autotune_configs, noncausal_autotune_keys) = get_autotune_configs()


# This function computes delta given output Out and gradient DO
# Here is the I/O shape:
# Out: (batch, nhead_q, max_seqlens_q, headDim)
# DO: (batch, nhead_q, max_seqlens_q, headDim)
# Delta: (batch, nheads_q, max_seqlens_q)
@triton.autotune(
    configs=preprocess_autotune_configs,
    key=preprocess_autotune_keys,
    use_cuda_graph=True,
)
@triton.jit
def _bwd_preprocess(
    O,
    DO,  # noqa: E741
    Delta,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_dob, stride_doh, stride_dom, stride_dod,
    stride_delta_b, stride_delta_h, stride_delta_m,
    stride_descale_do_z,
    cu_seqlens_q, max_seqlen_q,
    Descale_do,
    PRE_BLOCK: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
    ACTUAL_HEAD_DIM_V: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_FP8: tl.constexpr
):
    pid_m = tl.program_id(0)
    bid = tl.program_id(1)
    hid = tl.program_id(2)
    # Handle varlen
    if IS_VARLEN:
        q_start = tl.load(cu_seqlens_q + bid)
        q_end = tl.load(cu_seqlens_q + bid + 1)
        seqlen_q = q_end - q_start
    else:
        q_start = 0
        seqlen_q = max_seqlen_q

    # Compute offsets
    offs_m = pid_m * PRE_BLOCK + tl.arange(0, PRE_BLOCK)
    offs_d = tl.arange(0, HEAD_DIM_V)
    # pointer offsets for O & DO
    off_o = ( bid * stride_ob 
            + hid * stride_oh 
            + q_start * stride_om 
            + offs_m[:, None] * stride_om 
            + offs_d[None, :] * stride_od) # noqa: E741
    off_do = (bid * stride_dob 
              + hid * stride_doh 
              + q_start * stride_dom 
              + offs_m[:, None] * stride_dom 
              + offs_d[None, :] * stride_dod)

    # create masks
    mask_m = offs_m < seqlen_q
    mask_md = mask_m[:, None]
    PADDED_HEAD_V: tl.constexpr = (ACTUAL_HEAD_DIM_V != HEAD_DIM_V)
    if PADDED_HEAD_V:
        mask_md &= offs_d[None, :] < ACTUAL_HEAD_DIM_V
    # load
    o = tl.load(O + off_o, mask=mask_md, other=0.0)
    do = tl.load(DO + off_do, mask=mask_md, other=0.0)
    # compute and write-back to delta
    if IS_FP8:
        off_descale_do = bid * stride_descale_do_z + hid
        descale_do = tl.load(Descale_do + off_descale_do)

        # NOTE: do is in the fp8 range and o is not in fp8
        delta = tl.sum(o.to(tl.float32) * (do.to(tl.float32) * descale_do), axis=1)
    else:
        delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
    off_delta = (bid * stride_delta_b 
                 + hid * stride_delta_h 
                 + q_start * stride_delta_m
                 + offs_m * stride_delta_m)
    tl.store(Delta + off_delta , delta, mask=mask_m)


# The main inner-loop logic for computing dK and dV.
@triton.jit
def _bwd_dkdv_inner(
    dk, dv,  # output
    Q, k, v, DO, M, D, sm_scale,  # input tensor
    stride_qm, stride_qk,
    stride_dom, stride_dok,
    stride_dropoutm, stride_dropoutn,
    stride_lse_m, stride_delta_m,
    BLOCK_M: tl.constexpr,  # 16
    BLOCK_N: tl.constexpr,  # 128
    HEAD_DIM_QK: tl.constexpr,  #
    HEAD_DIM_V: tl.constexpr,  #
    ACTUAL_HEAD_DIM_QK: tl.constexpr,  #
    ACTUAL_HEAD_DIM_V: tl.constexpr,  #
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
    PADDED_HEAD_QK: tl.constexpr = (ACTUAL_HEAD_DIM_QK != HEAD_DIM_QK)
    PADDED_HEAD_V: tl.constexpr = (ACTUAL_HEAD_DIM_V != HEAD_DIM_V)
    delta_qk = seqlen_q - seqlen_k
    offs_m = start_m + tl.arange(0, BLOCK_M)  # start_m + (0, 15)
    offs_n = start_n + tl.arange(0, BLOCK_N)  # start_m + (0, 127)
    offs_k_qk = tl.arange(0, HEAD_DIM_QK)
    offs_k_v = tl.arange(0, HEAD_DIM_V)
    # mask to make sure not OOB of seqlen_q
    mask_n = offs_n < seqlen_k
    # Q and DO are (seqlen_q, head_dim)
    # qT_ptrs = (1, BLOCK_M) + (HEAD_DIM_QK, 1), transpose of q
    qT_ptrs = Q + offs_m[None, :] * stride_qm + offs_k_qk[:, None] * stride_qk
    # do_ptrs = (BLOCK_M, 1) + (1, HEAD_DIM_V), NOT transposed
    do_ptrs = DO + offs_m[:, None] * stride_dom + offs_k_v[None, :] * stride_dok
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
        if PADDED_HEAD_QK:
            mask_qT &= offs_k_qk[:, None] < ACTUAL_HEAD_DIM_QK
        if PADDED_HEAD_V:
            mask_do &= offs_k_v[None, :] < ACTUAL_HEAD_DIM_V
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
        m = tl.load(M + offs_m * stride_lse_m, mask=mask_m, other=0.0)
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
        Di = tl.load(D + offs_m * stride_delta_m, mask=mask_m)
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

# the main inner-loop logic for computing dQ
@triton.jit
def _bwd_dq_inner(
    dq,  # output
    q, K, V, do, m, Delta, sm_scale, # input
    # shared by Q/K/V.
    stride_qm, stride_qk, stride_kn,  stride_kk, stride_vn, stride_vk,
    stride_dropoutm, stride_dropoutn,  # stride for dropout
    stride_lse_m,
    stride_delta_m,
    seqlen_q, seqlen_k,  #
    BLOCK_M2: tl.constexpr,  #
    BLOCK_N2: tl.constexpr,  #
    HEAD_DIM_QK: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
    ACTUAL_HEAD_DIM_QK: tl.constexpr,
    ACTUAL_HEAD_DIM_V: tl.constexpr,  #
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
    PADDED_HEAD_QK: tl.constexpr = (ACTUAL_HEAD_DIM_QK != HEAD_DIM_QK)
    PADDED_HEAD_V: tl.constexpr = (ACTUAL_HEAD_DIM_V != HEAD_DIM_V)
    delta_qk = seqlen_q - seqlen_k
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    offs_k_qk = tl.arange(0, HEAD_DIM_QK)
    offs_k_v = tl.arange(0, HEAD_DIM_V)

    # mask to make sure not OOB of seqlen_q
    mask_m = offs_m < seqlen_q

    kT_ptrs = K + offs_n[None, :] * stride_kn + offs_k_qk[:, None] * stride_kk
    vT_ptrs = V + offs_n[None, :] * stride_vn + offs_k_v[:, None] * stride_vk
    # D (= delta) is pre-divided by ds_scale.
    Di = tl.load(Delta + offs_m * stride_delta_m, mask=mask_m, other=0.0)
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
        mask_vT = mask_n[None, :]
        mask_mn = mask_m[:, None] & (offs_n[None, :] < end_n)
        if PADDED_HEAD_QK:
            mask_kT &= offs_k_qk[:, None] < ACTUAL_HEAD_DIM_QK
        if PADDED_HEAD_V:
            mask_vT &= offs_k_v[:, None] < ACTUAL_HEAD_DIM_V

        kT = tl.load(kT_ptrs, mask=mask_kT, other=0.0)
        vT = tl.load(vT_ptrs, mask=mask_vT, other=0.0)

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

@triton.autotune(
    configs=causal_autotune_configs,
    key=causal_autotune_keys,
    use_cuda_graph=True,
)
@triton.jit
def bwd_kernel_causal( # grid = (nheads_k, tl.cdiv(max_seqlen_q // BLOCK_M2), batch)
    Q, K, V, sm_scale, DO, DQ, DK, DV,
    M, Delta,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_dqb, stride_dqh, stride_dqm, stride_dqd,
    stride_dkb, stride_dkh, stride_dkn, stride_dkd,
    stride_dvb, stride_dvh, stride_dvn, stride_dvd,
    stride_lse_b, stride_lse_h, stride_lse_m,
    stride_delta_b, stride_delta_h, stride_delta_m,
    stride_dob, stride_doh, stride_dom, stride_dod,
    stride_dropoutb, stride_dropouth, stride_dropoutm, stride_dropoutn,
    stride_descale_q_z, stride_descale_k_z, stride_descale_v_z, stride_descale_do_z,
    stride_az, stride_ah,
    HQ, HK,
    cu_seqlens_q, cu_seqlens_k,
    seqused_q, seqused_k,  # Add seqused parameters
    max_seqlen_q, max_seqlen_k,
    Dropout_mask, dropout_p, philox_seed, philox_offset_base,
    Alibi_slopes,
    Descale_q, Descale_k, Descale_v, Descale_do,
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    BLOCK_M2: tl.constexpr,
    BLOCK_N2: tl.constexpr,
    BLK_SLICE_FACTOR: tl.constexpr,
    HEAD_DIM_QK: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
    ACTUAL_HEAD_DIM_QK: tl.constexpr,
    ACTUAL_HEAD_DIM_V: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_ALIBI: tl.constexpr,
    USE_EXP2: tl.constexpr,
    IS_FP8: tl.constexpr,
    FP8_MAX: tl.constexpr,
    FP8_OUTPUT: tl.constexpr,
    USE_SEQUSED: tl.constexpr,  # Add flag for seqused
    DEBUG_TRITON: tl.constexpr,
    DEBUG_TRITON_DETAIL: tl.constexpr,
):
    # program ids
    hkid = tl.program_id(0)
    pid = tl.program_id(1)
    bid = tl.program_id(2)
    if DEBUG_TRITON: print(f"\npid: {pid}, bid: {bid}, hkid: {hkid}")  # noqa: E701
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
        
        # If seqused is provided, use it to limit the actual sequence length
        if USE_SEQUSED:
            actual_seqlen_q = tl.load(seqused_q + bid) if seqused_q is not None else q_end - q_start
            seqlen_q = tl.minimum(actual_seqlen_q, q_end - q_start)
            actual_seqlen_k = tl.load(seqused_k + bid) if seqused_k is not None else k_end - k_start
            seqlen_k = tl.minimum(actual_seqlen_k, k_end - k_start)
        else:
            seqlen_q = q_end - q_start
            seqlen_k = k_end - k_start

    delta_qk = seqlen_q - seqlen_k
    if DEBUG_TRITON: print(f"delta_qk = {delta_qk}")  # noqa: E701
    PADDED_HEAD_QK: tl.constexpr = (ACTUAL_HEAD_DIM_QK != HEAD_DIM_QK)
    PADDED_HEAD_V: tl.constexpr = (ACTUAL_HEAD_DIM_V != HEAD_DIM_V)
    offs_d_qk = tl.arange(0, HEAD_DIM_QK)
    offs_d_v = tl.arange(0, HEAD_DIM_V)
    GROUP_SIZE: tl.constexpr = HQ // HK

    # align the delta_qk
    start_n = pid * BLOCK_N1
    if start_n < seqlen_k:
        # This section does dk and dv
        dk = tl.zeros([BLOCK_N1, HEAD_DIM_QK], dtype=tl.float32)
        dv = tl.zeros([BLOCK_N1, HEAD_DIM_V], dtype=tl.float32)

        # q > k: diretcly skip all the way until the start of causal block
        start_delta_q_gt_k = delta_qk
        # q < k: some blocks will have no Masked block, other needs to re-calc
        # starting position
        # delta_qk is negative so flip it, only multiple of BLOCK_N can skip the
        # masked op
        num_blocks_skip = -delta_qk // BLOCK_N1
        delta_aligned = (num_blocks_skip + 1) * BLOCK_N1 + delta_qk
        start_delta_q_lt_k = delta_aligned // BLOCK_M1 * BLOCK_M1
        if delta_qk >= 0:
            start_delta = delta_qk
            if DEBUG_TRITON: print(f"q >= k: start_delta = delta_qk aligned to BLOCK_M = {start_delta_q_gt_k}")  # noqa: E701
        else:
            start_delta = start_delta_q_lt_k
            if DEBUG_TRITON: print(f"q < k: start_delta = residue btw multiple BLOCK_N and delta_qk = {delta_aligned} = aligned to BLOCK_M = {start_delta_q_lt_k}")  # noqa: E701

        offs_n = start_n + tl.arange(0, BLOCK_N1)
        # Mask for loading K and V
        mask_k = offs_n[:, None] < seqlen_k
        mask_v = offs_n[:, None] < seqlen_k
        if PADDED_HEAD_QK:
            mask_d_qk = offs_d_qk < ACTUAL_HEAD_DIM_QK
            mask_k &= mask_d_qk[None, :]
        if PADDED_HEAD_V:
            mask_d_v = offs_d_v < ACTUAL_HEAD_DIM_V
            mask_v &= mask_d_v[None, :]

        # K/V tensors not changed for the group
        adj_k = bid * stride_kb + hkid * stride_kh + k_start * stride_kn + offs_n[:, None] * stride_kn + offs_d_qk[None, :] * stride_kd
        adj_v = bid * stride_vb + hkid * stride_vh + k_start * stride_vn + offs_n[:, None] * stride_vn + offs_d_v[None, :] * stride_vd
        # load K and V: they stay in SRAM throughout the inner loop.
        k = tl.load(K + adj_k, mask=mask_k, other=0.0)
        v = tl.load(V + adj_v, mask=mask_v, other=0.0)
        # If MQA / GQA, set the K and V head offsets appropriately.
        # hqid = hkid
        for hqid in range(hkid * GROUP_SIZE, hkid * GROUP_SIZE + GROUP_SIZE):
            if delta_qk >= 0:
                start_m = start_n + start_delta
                len_m = BLOCK_N1
            else:
                start_m = max(start_n + delta_qk, 0)
                start_m = start_m // BLOCK_M1 * BLOCK_M1
                # because we might shift the masked blocks up, we are deeper into
                # the masked out region, so we would potentially increase the total
                # steps with masked operation to get out of it
                residue_m = max(start_n + delta_qk - start_m, 0)
                len_m = BLOCK_N1 + residue_m
                if DEBUG_TRITON: print(f"residue_m = {residue_m}")  # noqa: E701

            # offset input and output tensor by batch and Q/K heads
            adj_q = bid * stride_qb + hqid * stride_qh + q_start * stride_qm
            Q_ptr = Q + adj_q
            adj_do = bid * stride_dob + hqid * stride_doh + q_start * stride_dom
            DO_ptr = DO + adj_do
            adj_delta = bid * stride_delta_b + hqid * stride_delta_h + q_start * stride_delta_m
            Delta_ptr = Delta + adj_delta
            adj_m = bid * stride_lse_b + hqid * stride_lse_h + q_start * stride_lse_m 
            M_ptr = M + adj_m

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

            MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR
            # bound the masked operation to q len so it does not have to wast cycles
            len_m = min(len_m, seqlen_q)
            num_steps = tl.cdiv(len_m, MASK_BLOCK_M1)
            # when q < k, we may skip the initial masked op
            if pid < num_blocks_skip:
                num_steps = 0

            # if start_m is negative, the current N-tile has no block on the
            #   diagonal of causal mask, so everything have no causal mask
            if DEBUG_TRITON: print(f"Masked: start_n: {start_n}; start_m: {start_m}, num_steps: {num_steps}")  # noqa: E701
            dk, dv = _bwd_dkdv_inner(
                dk, dv,  # output tensors
                Q_ptr, k, v, DO_ptr, M_ptr, Delta_ptr, sm_scale, # input tensors
                stride_qm, stride_qd,  # strides for q
                stride_dom, stride_dod,  # strides for o
                stride_dropoutm, stride_dropoutn,  # strides for dropout
                stride_lse_m, stride_delta_m,
                MASK_BLOCK_M1, BLOCK_N1,  # block dim
                HEAD_DIM_QK, HEAD_DIM_V, ACTUAL_HEAD_DIM_QK, ACTUAL_HEAD_DIM_V,  # head dim
                dropout_p, philox_seed, batch_philox_offset, dropout_offset,
                alibi_slope,
                seqlen_q, seqlen_k,  # max sequence length for q and k
                start_n, start_m, num_steps,  # iteration numbers
                descale_q, descale_k, descale_v, descale_do,
                MASK=True,  # causal masking
                ENABLE_DROPOUT=ENABLE_DROPOUT,  # activate dropout
                USE_ALIBI=USE_ALIBI,
                USE_EXP2=USE_EXP2,
                IS_FP8=IS_FP8,
                FP8_MAX=FP8_MAX,
                DEBUG_TRITON=DEBUG_TRITON,
                DEBUG_TRITON_DETAIL=DEBUG_TRITON_DETAIL,
            )
            start_m += num_steps * MASK_BLOCK_M1
            num_steps = tl.cdiv(seqlen_q - start_m, BLOCK_M1)
            end_m = start_m + num_steps * BLOCK_M1

            if DEBUG_TRITON: print(f"start_m after Masked step: {start_m}; num_steps: {num_steps}")  # noqa: E701
            if DEBUG_TRITON: print(f"unMasked: start_n: {start_n}, start_m: {start_m}, end_m: {end_m}, num_steps: {num_steps}")  # noqa: E701
            if DEBUG_TRITON: print("unMasked")  # noqa: E701
            dk, dv = _bwd_dkdv_inner(
                dk, dv,  # output tensors
                Q_ptr, k, v, DO_ptr, M_ptr, Delta_ptr, sm_scale, # input tensors
                stride_qm, stride_qd,  # strides for q
                stride_dom, stride_dod,  # strides for o
                stride_dropoutm, stride_dropoutn,  # strides for dropout
                stride_lse_m, stride_delta_m,
                BLOCK_M1, BLOCK_N1,  # block dim
                HEAD_DIM_QK, HEAD_DIM_V, ACTUAL_HEAD_DIM_QK, ACTUAL_HEAD_DIM_V,  # head dim
                dropout_p, philox_seed, batch_philox_offset, dropout_offset,
                alibi_slope,
                seqlen_q, seqlen_k,  # max sequence length for q and k
                start_n, start_m, num_steps,  # iteration numbers
                descale_q, descale_k, descale_v, descale_do,
                MASK=False,  # causal masking
                ENABLE_DROPOUT=ENABLE_DROPOUT,  # activate dropout
                USE_ALIBI=USE_ALIBI,
                USE_EXP2=USE_EXP2,
                IS_FP8=IS_FP8,
                FP8_MAX=FP8_MAX,
                DEBUG_TRITON=DEBUG_TRITON,
                DEBUG_TRITON_DETAIL=DEBUG_TRITON_DETAIL,
            )
        # end of GQA/MQA of dkdv
        # Write back dV
        adj_dv = bid * stride_dvb + hkid * stride_dvh + k_start * stride_dvn
        offs_dv = offs_n[:, None] * stride_dvn + offs_d_v[None, :] * stride_dvd
        tl.store(DV + adj_dv + offs_dv, dv, mask=mask_v)
        # write back dk
        adj_dk = bid * stride_dkb + hkid * stride_dkh + k_start * stride_dkn
        offs_dk = offs_n[:, None] * stride_dkn + offs_d_qk[None, :] * stride_dkd
        dk *= sm_scale
        tl.store(DK + adj_dk + offs_dk, dk, mask=mask_k)

    # This part does dq
    start_m = pid * BLOCK_M2
    if start_m < seqlen_q:
        # seqlen_q > seqlen_k, no need to process these tile for dq
        if DEBUG_TRITON: print(f"end_n = start_m + BLOCK_M = {start_m} + {BLOCK_M2} = {start_m + BLOCK_M2}")  # noqa: E701
        if start_m + BLOCK_M2 < delta_qk:
            if DEBUG_TRITON: print(f"start_m + BLOCK_M2 = {start_m} + {BLOCK_M2} = {start_m + BLOCK_M2} < delta_qk of {delta_qk}")  # noqa: E701
            return

        offs_m = start_m + tl.arange(0, BLOCK_M2)
        # Mask for loading K and V
        mask_q = offs_m[:, None] < seqlen_q
        mask_do = offs_m[:, None] < seqlen_q
        if PADDED_HEAD_QK:
            mask_d_qk = offs_d_qk < ACTUAL_HEAD_DIM_QK
            mask_q &= mask_d_qk[None, :]
        if PADDED_HEAD_V:
            mask_d_v = offs_d_v < ACTUAL_HEAD_DIM_V
            mask_do &= mask_d_v[None, :]
        offs_q = offs_m[:, None] * stride_qm + offs_d_qk[None, :] * stride_qd
        offs_do = offs_m[:, None] * stride_dom + offs_d_v[None, :] * stride_dod
        # NOTE: don't assume that the strides for k and v are the same!
        K +=  bid * stride_kb + hkid * stride_kh + k_start * stride_kn
        V +=  bid * stride_vb + hkid * stride_vh + k_start * stride_vn

        # If MQA / GQA, set the K and V head offsets appropriately.
        for hqid in range(hkid * GROUP_SIZE, hkid * GROUP_SIZE + GROUP_SIZE):
            # seqlen_q < seqlen_k: delta_qk more kv tokens are added at the front
            #   for every M-tile
            end_n = start_m + BLOCK_M2 - delta_qk
            # clamp end_n at [0, seqlen_k]
            end_n = max(min(end_n, seqlen_k), 0)
            if DEBUG_TRITON: print(f"delta_qk: {delta_qk}; end_n: {end_n}")  # noqa: E701
            # offset input and output tensor by batch and Q/K heads
            adj_q = bid * stride_qb + hqid * stride_qh + q_start * stride_qm
            adj_do = bid * stride_dob + hqid * stride_doh + q_start * stride_dom
            adj_delta = \
                bid * stride_delta_b + hqid * stride_delta_h + q_start * stride_delta_m
            Delta_ptr = Delta + adj_delta
            adj_m = bid * stride_lse_b + hqid * stride_lse_h + q_start * stride_lse_m 
            M_ptr = M + adj_m

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
            do = tl.load(DO + adj_do + offs_do, mask=mask_do, other=0.0)
            m = tl.load(M + adj_m + offs_m * stride_lse_m,
                        mask=offs_m < seqlen_q)
            m = m[:, None]

            MASK_BLOCK_N2: tl.constexpr = BLOCK_N2 // BLK_SLICE_FACTOR
            # start can only be 0 at minimum
            start_n = max(end_n - BLOCK_M2, 0)
            num_steps = tl.cdiv(end_n - start_n, MASK_BLOCK_N2)

            if IS_FP8:
                descale_q = tl.load(Descale_q + bid * stride_descale_q_z + hqid)
                descale_k = tl.load(Descale_k + bid * stride_descale_k_z + hkid)
                descale_v = tl.load(Descale_v + bid * stride_descale_v_z + hkid)
                descale_do = tl.load(Descale_do + bid * stride_descale_do_z + hqid)
            else:
                descale_q, descale_k, descale_v, descale_do = 1.0, 1.0, 1.0, 1.0

            dq = tl.zeros([BLOCK_M2, HEAD_DIM_QK], dtype=tl.float32)
            dq = _bwd_dq_inner(
                dq,
                q, K, V, do, m, Delta_ptr, sm_scale,
                stride_qm, stride_qd, stride_kn, stride_kd, stride_vn, stride_vd,
                stride_dropoutm, stride_dropoutn,
                stride_lse_m,
                stride_delta_m,
                seqlen_q, seqlen_k,
                BLOCK_M2, MASK_BLOCK_N2,
                HEAD_DIM_QK, HEAD_DIM_V, ACTUAL_HEAD_DIM_QK, ACTUAL_HEAD_DIM_V,
                dropout_p, philox_seed, batch_philox_offset, dropout_offset,
                alibi_slope,
                start_m, start_n, end_n, num_steps,
                descale_q, descale_k, descale_v, descale_do,
                MASK=True,  #
                ENABLE_DROPOUT=ENABLE_DROPOUT,
                USE_ALIBI=USE_ALIBI,
                USE_EXP2=USE_EXP2,
                IS_FP8=IS_FP8,
                FP8_MAX=FP8_MAX,
                DEBUG_TRITON=DEBUG_TRITON,
                DEBUG_TRITON_DETAIL=DEBUG_TRITON_DETAIL,
            )
            end_n -= num_steps * MASK_BLOCK_N2
            num_steps = tl.cdiv(end_n, BLOCK_N2)
            start_n = max(end_n - num_steps * BLOCK_N2, 0)
            if DEBUG_TRITON: print(f"unMasked: start_m: {start_m}, start_n: {start_n}, end_n: {end_n}, num_steps: {num_steps}")  # noqa: E701
            dq = _bwd_dq_inner(
                dq,
                q, K, V, do, m, Delta_ptr, sm_scale,
                stride_qm, stride_qd, stride_kn, stride_kd, stride_vn, stride_vd,
                stride_dropoutm, stride_dropoutn,
                stride_lse_m,
                stride_delta_m,
                seqlen_q, seqlen_k,
                BLOCK_M2, BLOCK_N2,
                HEAD_DIM_QK, HEAD_DIM_V, ACTUAL_HEAD_DIM_QK, ACTUAL_HEAD_DIM_V,
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
            offs_dq = offs_m[:, None] * stride_dqm + offs_d_qk[None, :] * stride_dqd
            dq *= sm_scale
            tl.store(DQ + adj_dq + offs_dq, dq, mask=mask_q)
            # end of GQA/MQA of dq

@triton.autotune(
    configs=noncausal_autotune_configs,
    key=noncausal_autotune_keys,
    use_cuda_graph=True,
)
@triton.jit
def bwd_kernel_noncausal(
    Q, K, V, sm_scale, DO, DQ, DK, DV,
    M, Delta,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_dqb, stride_dqh, stride_dqm, stride_dqd,
    stride_dkb, stride_dkh, stride_dkn, stride_dkd,
    stride_dvb, stride_dvh, stride_dvn, stride_dvd,
    stride_lse_b, stride_lse_h, stride_lse_m,
    stride_delta_b, stride_delta_h, stride_delta_m,
    stride_dob, stride_doh, stride_dom, stride_dod,
    stride_dropoutb, stride_dropouth, stride_dropoutm, stride_dropoutn,
    stride_descale_q_z, stride_descale_k_z, stride_descale_v_z, stride_descale_do_z,
    stride_az, stride_ah,
    HQ, HK,
    cu_seqlens_q, cu_seqlens_k,
    seqused_q, seqused_k,  # Add seqused parameters
    max_seqlen_q, max_seqlen_k,
    Dropout_mask, dropout_p, philox_seed, philox_offset_base,
    Alibi_slopes,
    Descale_q, Descale_k, Descale_v, Descale_do,
    BLOCK_M1: tl.constexpr,  # 32
    BLOCK_N1: tl.constexpr,  # 128
    BLOCK_M2: tl.constexpr,  # 128
    BLOCK_N2: tl.constexpr,  # 32
    BLK_SLICE_FACTOR: tl.constexpr,
    HEAD_DIM_QK: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
    ACTUAL_HEAD_DIM_QK: tl.constexpr,
    ACTUAL_HEAD_DIM_V: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_ALIBI: tl.constexpr,
    USE_EXP2: tl.constexpr,
    IS_FP8: tl.constexpr,
    FP8_MAX: tl.constexpr,
    FP8_OUTPUT: tl.constexpr,
    USE_SEQUSED: tl.constexpr,  # Add flag for seqused
    DEBUG_TRITON: tl.constexpr,
    DEBUG_TRITON_DETAIL: tl.constexpr,
):
    # program ids
    hkid = tl.program_id(0)
    pid = tl.program_id(1)
    bid = tl.program_id(2)
    if DEBUG_TRITON: print(f"\npid: {pid}, bid: {bid}, hkid: {hkid}")  # noqa: E701
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
        
        # If seqused is provided, use it to limit the actual sequence length
        if USE_SEQUSED:
            actual_seqlen_q = tl.load(seqused_q + bid) if seqused_q is not None else q_end - q_start
            seqlen_q = tl.minimum(actual_seqlen_q, q_end - q_start)
            actual_seqlen_k = tl.load(seqused_k + bid) if seqused_k is not None else k_end - k_start
            seqlen_k = tl.minimum(actual_seqlen_k, k_end - k_start)
        else:
            seqlen_q = q_end - q_start
            seqlen_k = k_end - k_start

    PADDED_HEAD_QK: tl.constexpr = (ACTUAL_HEAD_DIM_QK != HEAD_DIM_QK)
    PADDED_HEAD_V: tl.constexpr = (ACTUAL_HEAD_DIM_V != HEAD_DIM_V)
    offs_d_qk = tl.arange(0, HEAD_DIM_QK)
    offs_d_v = tl.arange(0, HEAD_DIM_V)
    GROUP_SIZE: tl.constexpr = HQ // HK

    start_n = pid * BLOCK_N1
    if start_n < seqlen_k:
        dk = tl.zeros([BLOCK_N1, HEAD_DIM_QK], dtype=tl.float32)
        dv = tl.zeros([BLOCK_N1, HEAD_DIM_V], dtype=tl.float32)

        offs_n = start_n + tl.arange(0, BLOCK_N1)
        # Mask for loading K and V
        mask_k = offs_n[:, None] < seqlen_k
        mask_v = offs_n[:, None] < seqlen_k
        if PADDED_HEAD_QK:
            mask_d_qk = offs_d_qk < ACTUAL_HEAD_DIM_QK
            mask_k &= mask_d_qk[None, :]
        if PADDED_HEAD_V:
            mask_d_v = offs_d_v < ACTUAL_HEAD_DIM_V
            mask_v &= mask_d_v[None, :]
        # NOTE: don't assume that the strides for k and v are the same!
        # K/V tensors not changed for the group
        adj_k = bid * stride_kb + hkid * stride_kh + k_start * stride_kn + offs_n[:, None] * stride_kn + offs_d_qk[None, :] * stride_kd
        adj_v = bid * stride_vb + hkid * stride_vh + k_start * stride_vn + offs_n[:, None] * stride_vn + offs_d_v[None, :] * stride_vd
        # load K and V: they stay in SRAM throughout the inner loop.
        k = tl.load(K + adj_k, mask=mask_k, other=0.0)
        v = tl.load(V + adj_v, mask=mask_v, other=0.0)
        # If MQA / GQA, set the K and V head offsets appropriately.
        for hqid in range(hkid * GROUP_SIZE, hkid * GROUP_SIZE + GROUP_SIZE):
            # offset input and output tensor by batch and Q/K heads
            adj_q = bid * stride_qb + hqid * stride_qh + q_start * stride_qm
            Q_ptr = Q + adj_q
            adj_do = bid * stride_dob + hqid * stride_doh + q_start * stride_dom
            DO_ptr = DO + adj_do
            adj_delta = bid * stride_delta_b + hqid * stride_delta_h + q_start * stride_delta_m
            Delta_ptr = Delta + adj_delta
            adj_m = bid * stride_lse_b + hqid * stride_lse_h + q_start * stride_lse_m 
            M_ptr = M + adj_m

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
            num_steps = tl.cdiv(seqlen_q, BLOCK_M1)
            dk, dv = _bwd_dkdv_inner(
                dk, dv,  # output tensors
                Q_ptr, k, v, DO_ptr, M_ptr, Delta_ptr, sm_scale, # input tensors
                stride_qm, stride_qd,  # strides for q
                stride_dom, stride_dod,  # strides for o
                stride_dropoutm, stride_dropoutn,  # strides for dropout
                stride_lse_m,
                stride_delta_m,
                BLOCK_M1, BLOCK_N1,  # block dim
                HEAD_DIM_QK, HEAD_DIM_V, ACTUAL_HEAD_DIM_QK, ACTUAL_HEAD_DIM_V,  # head dim
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

        # Write back dV
        adj_dv = bid * stride_dvb + hkid * stride_dvh + k_start * stride_dvn
        offs_dv = offs_n[:, None] * stride_dvn + offs_d_v[None, :] * stride_dvd
        tl.store(DV + adj_dv + offs_dv, dv, mask=mask_v)
        # write back dk
        adj_dk = bid * stride_dkb + hkid * stride_dkh + k_start * stride_dkn
        offs_dk = offs_n[:, None] * stride_dkn + offs_d_qk[None, :] * stride_dkd
        dk *= sm_scale
        tl.store(DK + adj_dk + offs_dk, dk, mask=mask_k)

    # THIS PART DOES DQ
    start_m = pid * BLOCK_M2
    if start_m < seqlen_q:
        offs_m = start_m + tl.arange(0, BLOCK_M2)
        # Mask for loading K and V
        mask_q = offs_m[:, None] < seqlen_q
        mask_do = offs_m[:, None] < seqlen_q
        if PADDED_HEAD_QK:
            mask_d_qk = offs_d_qk < ACTUAL_HEAD_DIM_QK
            mask_q &= mask_d_qk[None, :]
        if PADDED_HEAD_V:
            mask_d_v = offs_d_v < ACTUAL_HEAD_DIM_V
            mask_do &= mask_d_v[None, :]
        offs_q = offs_m[:, None] * stride_qm + offs_d_qk[None, :] * stride_qd
        offs_do = offs_m[:, None] * stride_dom + offs_d_v[None, :] * stride_dod
        K +=  bid * stride_kb + hkid * stride_kh + k_start * stride_kn
        V +=  bid * stride_vb + hkid * stride_vh + k_start * stride_vn
        # If MQA / GQA, set the K and V head offsets appropriately.
        for hqid in range(hkid * GROUP_SIZE, hkid * GROUP_SIZE + GROUP_SIZE):
            # offset input and output tensor by batch and Q/K heads
            adj_q = bid * stride_qb + hqid * stride_qh + q_start * stride_qm
            adj_do = bid * stride_dob + hqid * stride_doh + q_start * stride_dom
            adj_delta = \
                bid * stride_delta_b + hqid * stride_delta_h + q_start * stride_delta_m
            Delta_ptr = Delta + adj_delta
            adj_m = bid * stride_lse_b + hqid * stride_lse_h + q_start * stride_lse_m 
            M_ptr = M + adj_m

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
            do = tl.load(DO + adj_do + offs_do, mask=mask_do, other=0.0)
            m = tl.load(M + adj_m + offs_m * stride_lse_m,
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
            num_steps = tl.cdiv(seqlen_k, BLOCK_N2)

            dq = tl.zeros([BLOCK_M2, HEAD_DIM_QK], dtype=tl.float32)
            dq = _bwd_dq_inner(
                dq,
                q, K, V, do, m, Delta_ptr, sm_scale,
                stride_qm, stride_qd, stride_kn, stride_kd, stride_vn, stride_vd,
                stride_dropoutm, stride_dropoutn,
                stride_lse_m,
                stride_delta_m,
                seqlen_q, seqlen_k,
                BLOCK_M2, BLOCK_N2,
                HEAD_DIM_QK, HEAD_DIM_V, ACTUAL_HEAD_DIM_QK, ACTUAL_HEAD_DIM_V,
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
            offs_dq = offs_m[:, None] * stride_dqm + offs_d_qk[None, :] * stride_dqd
            dq *= sm_scale
            tl.store(DQ + adj_dq + offs_dq, dq, mask=mask_q)

def is_contiguous(x, name):
    if x.is_contiguous():
        return x
    else:
        print(f"{name} is not contiguous")
        return x.contiguous() 
    
OLD_LSE: bool = False
DEBUG_TRITON: bool = False
DEBUG_TRITON_DETAIL: bool = False

def attention_prefill_backward_triton_split_fused_no_atomics_impl(
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
    # seqused for FA v3
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
):
    # get params, strides and shape
    IS_VARLEN = layout == "thd"
    use_dropout = (dropout_p > 0.0)

    # common assertions
    assert 0.0 <= dropout_p <= 1.0, f"dropout_p must be between 0 and 1, got {dropout_p}"
    assert q.device == k.device == v.device == o.device == do.device == softmax_lse.device, \
        f"All tensors must be on the same device. Got: q={q.device}, k={k.device}, v={v.device}, o={o.device}, do={do.device}, softmax_lse={softmax_lse.device}"
    assert q.dtype == k.dtype == v.dtype == do.dtype, "q, k, v, do must have the same dtype"
    current_device = torch.cuda.current_device()
    assert q.is_cuda and q.device.index == current_device, f"Device mismatch: Kernel will launch on cuda:{current_device}, but tensors are on {q.device}"

    # get shapes and strides        
    if IS_VARLEN:
        # shape
        total_seqlen_q, nheads_q, head_size_q = q.shape
        total_seqlen_k, nheads_k, head_size_k = k.shape
        total_seqlen_v, nheads_v, head_size_v = v.shape
        nheads_lse, total_seqlen_lse = softmax_lse.shape

        # assert shapes
        assert total_seqlen_lse == total_seqlen_q, f"softmax_lse seqlen {total_seqlen_lse} != q seqlen {total_seqlen_q}"
        assert cu_seqlens_q is not None, "cu_seqlens_q must be provided for varlen layout"
        assert cu_seqlens_k is not None, "cu_seqlens_k must be provided for varlen layout"
        assert max_seqlen_q is not None, "max_seqlen_q must be provided for varlen layout"
        assert max_seqlen_k is not None, "max_seqlen_k must be provided for varlen layout"
        
        # assert head dimensions
        assert head_size_q == head_size_k, f"head sizes must match: q={head_size_q}, k={head_size_k}"
        assert nheads_k == nheads_v, f"k and v must have same number of heads: k={nheads_k}, v={nheads_v}"
        assert nheads_q % nheads_k == 0, f"nheads_q {nheads_q} must be divisible by nheads_k {nheads_k} for GQA/MQA"
        assert nheads_lse == nheads_q, f"softmax_lse heads {nheads_lse} != q heads {nheads_q}"
        
        # assert output shapes
        assert o.shape == (total_seqlen_q, nheads_q, head_size_v), f"o shape {o.shape} != expected {(total_seqlen_q, nheads_q, head_size_v)}"
        assert do.shape == o.shape, f"do shape {do.shape} != o shape {o.shape}"
        assert dq.shape == q.shape, f"dq shape {dq.shape} != q shape {q.shape}"
        assert dk.shape == k.shape, f"dk shape {dk.shape} != k shape {k.shape}"
        assert dv.shape == v.shape, f"dv shape {dv.shape} != v shape {v.shape}"
        
        # assert cu_seqlens
        assert cu_seqlens_q.dtype == torch.int32, f"cu_seqlens_q must be int32, got {cu_seqlens_q.dtype}"
        assert cu_seqlens_k.dtype == torch.int32, f"cu_seqlens_k must be int32, got {cu_seqlens_k.dtype}"
        assert cu_seqlens_q[0] == 0, "cu_seqlens_q must start with 0"
        assert cu_seqlens_k[0] == 0, "cu_seqlens_k must start with 0"
        assert cu_seqlens_q[-1] == total_seqlen_q, f"cu_seqlens_q[-1] {cu_seqlens_q[-1]} != total_seqlen_q {total_seqlen_q}"
        assert cu_seqlens_k[-1] == total_seqlen_k, f"cu_seqlens_k[-1] {cu_seqlens_k[-1]} != total_seqlen_k {total_seqlen_k}"
        
        # set vars
        batch = len(cu_seqlens_q) - 1
        head_size_qk = head_size_q

        # strides
        stride_qb, stride_qm, stride_qh, stride_qd = 0, q.stride(0), q.stride(1), q.stride(2)
        stride_kb, stride_kn, stride_kh, stride_kd = 0, k.stride(0), k.stride(1), k.stride(2)
        stride_vb, stride_vn, stride_vh, stride_vd = 0, v.stride(0), v.stride(1), v.stride(2)
        stride_ob, stride_om, stride_oh, stride_od = 0, o.stride(0), o.stride(1), o.stride(2)
        stride_dqb, stride_dqm, stride_dqh, stride_dqd = 0, dq.stride(0), dq.stride(1), dq.stride(2)
        stride_dkb, stride_dkn, stride_dkh, stride_dkd = 0, dk.stride(0), dk.stride(1), dk.stride(2)
        stride_dvb, stride_dvn, stride_dvh, stride_dvd = 0, dv.stride(0), dv.stride(1), dv.stride(2)
        stride_dob, stride_dom, stride_doh, stride_dod = 0, do.stride(0), do.stride(1), do.stride(2)
        stride_lse_b, stride_lse_h, stride_lse_m = (0, softmax_lse.stride(0), softmax_lse.stride(1))
    else:
        # shapes
        batch_q, seqlen_q, nheads_q, head_size_q = q.shape
        batch_k, seqlen_k, nheads_k, head_size_k = k.shape
        batch_v, seqlen_v, nheads_v, head_size_v = v.shape
        batch_lse, nheads_lse, seqlen_lse = softmax_lse.shape
        
        # assert batch dimensions
        assert batch_q == batch_k == batch_v, f"batch sizes must match: q={batch_q}, k={batch_k}, v={batch_v}"
        
        # assert head dimensions
        assert head_size_q == head_size_k, f"head sizes must match: q={head_size_q}, k={head_size_k}"
        assert nheads_k == nheads_v, f"k and v must have same number of heads: k={nheads_k}, v={nheads_v}"
        assert nheads_q % nheads_k == 0, f"nheads_q {nheads_q} must be divisible by nheads_k {nheads_k} for GQA/MQA"
        
        # assert sequence lengths
        assert seqlen_k == seqlen_v, f"k and v sequence lengths must match: k={seqlen_k}, v={seqlen_v}"
        
        # assert output shapes
        assert o.shape == (batch_q, seqlen_q, nheads_q, head_size_v), f"o shape {o.shape} != expected"
        assert do.shape == o.shape, f"do shape {do.shape} != o shape {o.shape}"
        assert dq.shape == q.shape, f"dq shape {dq.shape} != q shape {q.shape}"
        assert dk.shape == k.shape, f"dk shape {dk.shape} != k shape {k.shape}"
        assert dv.shape == v.shape, f"dv shape {dv.shape} != v shape {v.shape}"
        
        # assert softmax_lse shape
        assert softmax_lse.shape == (batch_q, nheads_q, seqlen_q), f"softmax_lse shape {softmax_lse.shape} != expected"

        # set vars
        batch = batch_q
        head_size_qk = head_size_q
        max_seqlen_q = seqlen_q
        max_seqlen_k = seqlen_k

        # strides
        stride_qb, stride_qm, stride_qh, stride_qd = q.stride()
        stride_kb, stride_kn, stride_kh, stride_kd = k.stride()
        stride_vb, stride_vn, stride_vh, stride_vd = v.stride()
        stride_ob, stride_om, stride_oh, stride_od = o.stride()
        stride_dqb, stride_dqm, stride_dqh, stride_dqd = dq.stride()
        stride_dkb, stride_dkn, stride_dkh, stride_dkd = dk.stride()
        stride_dvb, stride_dvn, stride_dvh, stride_dvd = dv.stride()
        stride_dob, stride_dom, stride_doh, stride_dod = do.stride()
        stride_lse_b, stride_lse_h, stride_lse_m = softmax_lse.stride()

    # fp8 setup - moved after all assertions
    IS_FP8 = is_fp8(q)
    if IS_FP8:
        FP8_MAX = torch.finfo(q.dtype).max
        # we already asserted that do, q, k, v all have the same dtype, so no need to check each one
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

        if DEBUG:
            print(f"FP8 path triggered in bwd_prefill_fused_no_atomics.py (FP8_OUTPUT={FP8_OUTPUT})")
    else:
        FP8_MAX = None
        FP8_OUTPUT = False
        stride_descale_q_z = stride_descale_k_z = stride_descale_v_z = stride_descale_o_z = stride_descale_do_z = None

    # alibi setup
    use_alibi, (stride_az, stride_ah) = (True, alibi_slopes.stride()) if alibi_slopes is not None else (False, (0, 0))

    # get closest power of 2 over or equal to 32.
    padded_d_model_qk = 1 << (head_size_qk - 1).bit_length()
    padded_d_model_qk = max(padded_d_model_qk, 32)
    padded_d_model_v = 1 << (head_size_v - 1).bit_length()
    padded_d_model_v = max(padded_d_model_v, 32)
    HEAD_DIM_QK = padded_d_model_qk
    HEAD_DIM_V = padded_d_model_v
    ACTUAL_HEAD_DIM_QK = head_size_qk
    ACTUAL_HEAD_DIM_V = head_size_v

    # init delta
    if OLD_LSE:
        delta = torch.empty_like(softmax_lse)
        if IS_VARLEN:
            stride_delta_b, stride_delta_h, stride_delta_m = 0, delta.stride(0), delta.stride(1)
        else:
            stride_delta_b, stride_delta_h, stride_delta_m = delta.stride()
    else:
        if IS_VARLEN:
            # interface expects the varlen sequence dims to rounded like this. Not sure why.
            total_q, num_heads, _ = q.shape
            total_q_rounded = total_q + 128 * batch
            delta_padded = torch.zeros((nheads_q, total_q_rounded), device=q.device, dtype=torch.float32)
            delta = delta_padded[:, :total_q]
            stride_delta_b, stride_delta_h, stride_delta_m = 0, delta.stride(0), delta.stride(1)
        else:
            # the interface expects the sequence dimension to be rounded to 128
            max_seqlen_q_rounded = round_multiple(max_seqlen_q, 128)
            delta_padded = torch.zeros((batch, nheads_q, max_seqlen_q_rounded), 
                            device=q.device, dtype=torch.float32)
            delta = delta_padded[:, :, :max_seqlen_q]
            stride_delta_b, stride_delta_h, stride_delta_m = delta.stride()

    pre_grid = lambda META:  (triton.cdiv(max_seqlen_q, META['PRE_BLOCK']), batch, nheads_q)
    _bwd_preprocess[pre_grid](
        o, do,
        delta,
        stride_ob, stride_oh, stride_om, stride_od,
        stride_dob, stride_doh, stride_dom, stride_dod,
        stride_delta_b, stride_delta_h, stride_delta_m,
        stride_descale_do_z,
        cu_seqlens_q, max_seqlen_q,
        descale_do,
        HEAD_DIM_V=HEAD_DIM_V,
        ACTUAL_HEAD_DIM_V=ACTUAL_HEAD_DIM_V,
        IS_VARLEN=IS_VARLEN,
        IS_FP8=IS_FP8
    )

    if False:
        print("delta:", delta, delta.shape)

    # dropout mask tensor for debugging. We dump the dropout mask created in
    #   the kernel for testing
    dropout_mask = None
    stride_dropoutb, stride_dropouth, stride_dropoutm, stride_dropoutn = \
        (0, 0 , 0 , 0)
    if use_dropout:
        dropout_mask = torch.zeros(
            (batch, nheads_q, max_seqlen_q, max_seqlen_k),
            device=q.device,
            dtype=torch.float32
        )

        if DROPOUT_USE_PYTORCH:
            if not IS_VARLEN:
                dropout_mask = create_dropout_mask(
                    dropout_p,
                    (batch, nheads_q, max_seqlen_q, max_seqlen_k),
                    seed = philox_seed
                )
            else:
                dropout_mask = create_dropout_mask_varlen(
                    dropout_p, batch, nheads_q,
                    cu_seqlens_q, cu_seqlens_k, philox_seed
                )
        stride_dropoutb, stride_dropouth, stride_dropoutm, stride_dropoutn = \
            dropout_mask.stride()

    seqlen = max(max_seqlen_q, max_seqlen_k)
    grid = lambda META: (nheads_k, (seqlen + META['BLOCK_N1'] - 1) // META['BLOCK_N1'], batch, )
    if causal:
        if DEBUG_TRITON: print(f"bwd_kernel: grid = {grid}" )  # noqa: E701
        bwd_kernel_causal[grid](
            q, k, v, sm_scale, do, dq, dk, dv,
            softmax_lse, delta,
            stride_qb, stride_qh, stride_qm, stride_qd,
            stride_kb, stride_kh, stride_kn, stride_kd,
            stride_vb, stride_vh, stride_vn, stride_vd,
            stride_dqb, stride_dqh, stride_dqm, stride_dqd,
            stride_dkb, stride_dkh, stride_dkn, stride_dkd,
            stride_dvb, stride_dvh, stride_dvn, stride_dvd,
            stride_lse_b, stride_lse_h, stride_lse_m,
            stride_delta_b, stride_delta_h, stride_delta_m,
            stride_dob, stride_doh, stride_dom, stride_dod,
            stride_dropoutb, stride_dropouth, stride_dropoutm, stride_dropoutn,
            stride_descale_q_z, stride_descale_k_z, stride_descale_v_z, stride_descale_do_z,
            stride_az, stride_ah,
            nheads_q, nheads_k,
            cu_seqlens_q, cu_seqlens_k,
            seqused_q, seqused_k,  # Pass seqused tensors
            max_seqlen_q, max_seqlen_k,
            dropout_mask, dropout_p, philox_seed, philox_offset,
            alibi_slopes,
            descale_q, descale_k, descale_v, descale_do,
            HEAD_DIM_QK=HEAD_DIM_QK,
            HEAD_DIM_V=HEAD_DIM_V,
            ACTUAL_HEAD_DIM_QK=ACTUAL_HEAD_DIM_QK,
            ACTUAL_HEAD_DIM_V=ACTUAL_HEAD_DIM_V,
            ENABLE_DROPOUT=use_dropout,
            IS_VARLEN=IS_VARLEN,
            USE_ALIBI=use_alibi,
            USE_EXP2=use_exp2,
            IS_FP8=IS_FP8,
            FP8_MAX=FP8_MAX,
            FP8_OUTPUT=FP8_OUTPUT,
            USE_SEQUSED=(seqused_q is not None or seqused_k is not None),  # Add flag for seqused
            DEBUG_TRITON=DEBUG_TRITON,
            DEBUG_TRITON_DETAIL=DEBUG_TRITON_DETAIL,
        )
    else:
        bwd_kernel_noncausal[grid](
            q, k, v, sm_scale, do, dq, dk, dv,
            softmax_lse, delta,
            stride_qb, stride_qh, stride_qm, stride_qd,
            stride_kb, stride_kh, stride_kn, stride_kd,
            stride_vb, stride_vh, stride_vn, stride_vd,
            stride_dqb, stride_dqh, stride_dqm, stride_dqd,
            stride_dkb, stride_dkh, stride_dkn, stride_dkd,
            stride_dvb, stride_dvh, stride_dvn, stride_dvd,
            stride_lse_b, stride_lse_h, stride_lse_m,
            stride_delta_b, stride_delta_h, stride_delta_m,
            stride_dob, stride_doh, stride_dom, stride_dod,
            stride_dropoutb, stride_dropouth, stride_dropoutm, stride_dropoutn,
            stride_descale_q_z, stride_descale_k_z, stride_descale_v_z, stride_descale_do_z,
            stride_az, stride_ah,
            nheads_q, nheads_k,
            cu_seqlens_q, cu_seqlens_k,
            seqused_q, seqused_k,  # Pass seqused tensors
            max_seqlen_q, max_seqlen_k,
            dropout_mask, dropout_p, philox_seed, philox_offset,
            alibi_slopes,
            descale_q, descale_k, descale_v, descale_do,
            HEAD_DIM_QK=HEAD_DIM_QK,
            HEAD_DIM_V=HEAD_DIM_V,
            ACTUAL_HEAD_DIM_QK=ACTUAL_HEAD_DIM_QK,
            ACTUAL_HEAD_DIM_V=ACTUAL_HEAD_DIM_V,
            ENABLE_DROPOUT=use_dropout,
            IS_VARLEN=IS_VARLEN,
            USE_ALIBI=use_alibi,
            USE_EXP2=use_exp2,
            IS_FP8=IS_FP8,
            FP8_MAX=FP8_MAX,
            FP8_OUTPUT=FP8_OUTPUT,
            USE_SEQUSED=(seqused_q is not None or seqused_k is not None),  # Add flag for seqused
            DEBUG_TRITON=DEBUG_TRITON,
            DEBUG_TRITON_DETAIL=DEBUG_TRITON_DETAIL,
        )

    if OLD_LSE:
        return delta
    else:
        return delta_padded
