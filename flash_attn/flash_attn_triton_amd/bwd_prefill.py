from typing import Literal, Optional
import torch
import triton
import triton.language as tl
from .utils import DEBUG, DROPOUT_USE_PYTORCH, DROPOUT_DUMP, compute_fp8_scaling_factors, get_shapes_from_layout, get_strides_from_layout, is_fp8, write_dropout_mask, create_dropout_mask

# TODO: move this into utils.py so it's shared among kernels
# NOTE: triton fails to import tl.constexprs so create them here for the file
tl_DROPOUT_USE_PYTORCH: tl.constexpr = triton.language.constexpr(DROPOUT_USE_PYTORCH)
tl_DROPOUT_DUMP: tl.constexpr = triton.language.constexpr(DROPOUT_DUMP)

@triton.jit
def _bwd_preprocess(
    Out,
    DO,
    Delta,
    stride_oz, stride_oh, stride_om, stride_ok,
    stride_doz, stride_doh, stride_dom, stride_dok,
    stride_deltaz, stride_deltah, stride_deltam,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    DESCALE_do,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    ACTUAL_BLOCK_DMODEL: tl.constexpr,
    N_CTX_Q: tl.constexpr,
    Z: tl.constexpr,
    H: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_FP8: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    # Compute batch and head indices
    off_z = pid_bh // H
    off_h = pid_bh % H

    if IS_VARLEN:
        # Compute sequence lengths for the current batch
        q_start = tl.load(cu_seqlens_q + off_z)
        q_end = tl.load(cu_seqlens_q + off_z + 1)
        k_start = tl.load(cu_seqlens_k + off_z)
        k_end = tl.load(cu_seqlens_k + off_z + 1)

        # Compute actual sequence lengths
        N_CTX_Q = q_end - q_start
        N_CTX_K = k_end - k_start
    else:
        q_start = 0
        k_start = 0
        N_CTX_Q = max_seqlen_q
        N_CTX_K = max_seqlen_k

    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_d = tl.arange(0, BLOCK_DMODEL)

    # create masks
    mask_m = off_m < N_CTX_Q
    mask_d = off_d < ACTUAL_BLOCK_DMODEL

    # compute offsets
    o_offset = Out + off_z * stride_oz + off_h * stride_oh + q_start * stride_om
    do_offset = DO + off_z * stride_oz + off_h * stride_oh + q_start * stride_om

    # compute pointers
    out_ptrs = o_offset + off_m[:, None] * stride_om + off_d[None, :] * stride_ok
    do_ptrs = do_offset + off_m[:, None] * stride_dom + off_d[None, :] * stride_dok

    # load
    o = tl.load(out_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
    do = tl.load(do_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)

    # compute delta
    if IS_FP8:
        stride_descale_q_z = H
        descale_do = tl.load(DESCALE_do + off_z * stride_descale_q_z + off_h)

        # NOTE: do is scaled into the fp8 range and o is in fp8 but should be in the same scale as fp32
        delta = tl.sum(o.to(tl.float32) * (do.to(tl.float32) * descale_do), axis=1)
    else:
        delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)

    # write-back delta
    delta_offset = Delta + off_z * stride_deltaz + off_h * stride_deltah + q_start * stride_deltam
    delta_ptrs = delta_offset + off_m * stride_deltam
    tl.store(delta_ptrs, delta, mask=mask_m)


@triton.jit
def _bwd_kernel_one_col_block(
    Q,
    K,
    V,
    sm_scale,
    Out,
    DO,
    DQ,
    DK,
    DV,
    L,
    D,
    q_offset,
    k_offset,
    v_offset,
    do_offset,
    dq_offset,
    dk_offset,
    dv_offset,
    l_offset,
    delta_offset,
    dropout_offset,
    stride_dq_all,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_deltaz, 
    stride_deltah, 
    stride_deltam,
    stride_dropoutz, stride_dropouth, stride_dropoutm, stride_dropoutn,
    N_CTX_Q,
    N_CTX_K,
    start_n,
    num_block_m,
    num_block_n,
    dropout_p,
    philox_seed,
    batch_philox_offset,
    descale_q, 
    descale_k,
    descale_v,
    descale_do,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    ACTUAL_BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SEQUENCE_PARALLEL: tl.constexpr,
    CAUSAL: tl.constexpr,
    DROPOUT: tl.constexpr,
    USE_EXP2: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    IS_FP8: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    if CAUSAL:
        # TODO: Causal can skip more blocks with something like lo = start_m * BLOCK_M
        lo = 0
    else:
        lo = 0

    # initialize col and head offsets
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # masks
    mask_n = offs_n < N_CTX_K
    mask_d = offs_d < ACTUAL_BLOCK_DMODEL
    kv_mask = mask_n[:, None] & mask_d[None, :]
    

    # initialize grad accumulators
    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)

    # load k and v once per column block
    k_ptrs = k_offset + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    v_ptrs = v_offset + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
    k = tl.load(k_ptrs, mask=kv_mask, other=0.0)
    kT = tl.trans(k)
    vT = tl.trans(tl.load(v_ptrs, mask=kv_mask, other=0.0))

    # loop over rows
    for start_m in range(lo, num_block_m):
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        q_ptrs = q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
        dq_ptrs = dq_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
        do_ptrs = do_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
        
        # update mask as row block changes
        mask_m = offs_m < N_CTX_Q
        q_mask = mask_m[:, None] & mask_d[None, :]

        # load q, k, v, do on-chip
        q = tl.load(q_ptrs, mask=q_mask, other=0.0)
        do = tl.load(do_ptrs, mask=q_mask, other=0.0)

        # recompute p = softmax(qk, dim=-1).T
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if IS_FP8:
            qk += (tl.dot(q, kT) * descale_q * descale_k)
        else:
            qk += tl.dot(q, kT)

        if CAUSAL:
            col_offset = N_CTX_Q - N_CTX_K
            causal_mask = offs_m[:, None] >= (col_offset + offs_n[None, :])
            qk = tl.where(causal_mask, qk, float("-inf"))

        l_ptrs = l_offset + offs_m * stride_deltam
        l_i = tl.load(l_ptrs, mask=mask_m)

        # compute p
        if USE_EXP2:
            RCP_LN2: tl.constexpr = 1.4426950408889634
            qk *= sm_scale * RCP_LN2
            l_i *= RCP_LN2
            p = tl.math.exp2(qk - l_i[:, None])
        else:
            qk *= sm_scale
            p = tl.math.exp(qk - l_i[:, None])

        # mask block in the cases where the data is smaller the block size
        p_mask = mask_m[:, None] & mask_n[None, :]
        p = tl.where(p_mask, p, 0.0)
        
        if DROPOUT:
            # NOTE: must create a new var p_drop to prevent p (which is used later to compute ds) from changing
            philox_offset = batch_philox_offset + offs_m[:, None] * stride_dropoutm + offs_n[None, :] * stride_dropoutn
            # print("philox_seed:", philox_seed)
            # print("philox_offset:", philox_offset)
            if tl_DROPOUT_USE_PYTORCH:
                dropout_ptrs = dropout_offset + offs_m[:, None] * stride_dropoutm + offs_n[None, :] * stride_dropoutn
                dropout_mask = tl.load(dropout_ptrs, mask=p_mask)
            else:
                rand_vals = tl.rand(philox_seed, philox_offset)
                dropout_mask = rand_vals > dropout_p
            dropout_scale = 1/ (1 - dropout_p)

            if tl_DROPOUT_DUMP:
                dropout_ptrs = dropout_offset + offs_m[:, None] * stride_dropoutm + offs_n[None, :] * stride_dropoutn
                tl.store(dropout_ptrs, dropout_mask, mask=p_mask)
            
            # apply dropout mask
            p_drop = tl.where(dropout_mask, p, 0.0)
            p_drop_scaled = p_drop * dropout_scale

            # compute dv
            if IS_FP8:
                scale_p_dropout, descale_p_dropout = compute_fp8_scaling_factors(p_drop_scaled, FP8_MAX)
                dv +=  (tl.dot(tl.trans(p_drop_scaled * scale_p_dropout).to(do.type.element_ty), do) * descale_p_dropout * descale_do)
            else:
                dv += tl.dot(tl.trans(p_drop_scaled).to(do.type.element_ty), do)

            # compute dp
            if IS_FP8:
                dp_drop_scaled = (tl.dot(do, vT) * descale_do * descale_v)
            else:
                dp_drop_scaled = tl.dot(do, vT)
            dp = tl.where(dropout_mask, dp_drop_scaled, 0.0) * dropout_scale
        else:

            # compute dv
            if IS_FP8:
                scale_p, descale_p = compute_fp8_scaling_factors(p, FP8_MAX)
                dv +=  (tl.dot(tl.trans(p * scale_p).to(do.type.element_ty), do) * descale_p * descale_do)
            else:
                dv += tl.dot(tl.trans(p).to(do.type.element_ty), do)

            # compute dp
            if IS_FP8:
                dp = (tl.dot(do, vT) * descale_do * descale_v)
            else:
                dp = tl.dot(do, vT)

        
        # load delta
        delta_ptrs = delta_offset + offs_m * stride_deltam
        delta_i = tl.load(delta_ptrs, mask=mask_m)

        # compute ds
        dscores_scaled = (p * (dp - delta_i[:, None]))
        ds = dscores_scaled * sm_scale
        ds = tl.where(p_mask, ds, 0.0)
        
        # compute descale_ds
        if IS_FP8:
            scale_ds, descale_ds = compute_fp8_scaling_factors(ds, FP8_MAX)
        else:
            scale_ds, descale_ds = 1.0, 1.0
        
        # compute dk
        if IS_FP8:
            dk += (tl.dot(tl.trans(ds * scale_ds).to(q.type.element_ty), q) * descale_ds * descale_q)
        else:
            dk += tl.dot(tl.trans(ds).to(q.type.element_ty), q)

        # compute dq
        if SEQUENCE_PARALLEL:
            if IS_FP8:
                dq = (tl.dot((ds * scale_ds).to(k.type.element_ty), k) * descale_ds * descale_k)
            else:
                dq = tl.dot(ds.to(k.type.element_ty), k)
        else:
            dq = tl.load(dq_ptrs, mask=q_mask, other=0.0)
            if IS_FP8:
                dq += (tl.dot((ds * scale_ds).to(k.type.element_ty), k) * descale_ds * descale_k)
            else:
                dq += tl.dot(ds.to(k.type.element_ty), k)
        tl.store(dq_ptrs, dq.to(Q.dtype.element_ty), mask=q_mask)

    # write-back dv and dk
    dk_ptrs = dk_offset + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    dv_ptrs = dv_offset + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
    
    # write-back
    if GROUP_SIZE != 1:
        # use atomic_add to properly accumulate gradients from multiple query heads
        tl.atomic_add(dk_ptrs, dk.to(K.dtype.element_ty), mask=kv_mask)
        tl.atomic_add(dv_ptrs, dv.to(V.dtype.element_ty), mask=kv_mask)
    else:
        tl.store(dk_ptrs, dk.to(K.dtype.element_ty), mask=kv_mask)
        tl.store(dv_ptrs, dv.to(V.dtype.element_ty), mask=kv_mask)

@triton.jit
def _bwd_kernel(
    Q,
    K,
    V,
    sm_scale,
    Out,
    DO,
    DQ,
    DK,
    DV,
    L,
    Delta,
    Dropout_mask,
    DESCALE_q,
    DESCALE_k,
    DESCALE_v,
    DESCALE_do,
    stride_dq_all,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_deltaz, 
    stride_deltah, 
    stride_deltam,
    stride_dropoutz, stride_dropouth, stride_dropoutm, stride_dropoutn,
    Z,
    HQ,
    HK,
    num_block_m,
    num_block_n,
    cu_seqlens_q,  
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p, 
    philox_seed, 
    philox_offset_base,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    ACTUAL_BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SEQUENCE_PARALLEL: tl.constexpr,
    CAUSAL: tl.constexpr,
    DROPOUT: tl.constexpr,
    USE_EXP2: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    IS_FP8: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    # program ids
    off_zh = tl.program_id(0)
    if SEQUENCE_PARALLEL:
        start_n = tl.program_id(1)
    off_z = off_zh // HQ
    off_hq = off_zh % HQ

    # check if GQA/MQA
    if GROUP_SIZE != 1:
        off_hk = off_hq // GROUP_SIZE
    else:
        off_hk = off_hq

    if IS_VARLEN:
        # Compute sequence lengths for the current batch
        q_start = tl.load(cu_seqlens_q + off_z)
        q_end = tl.load(cu_seqlens_q + off_z + 1)
        k_start = tl.load(cu_seqlens_k + off_z)
        k_end = tl.load(cu_seqlens_k + off_z + 1)

        # Compute actual sequence lengths
        N_CTX_Q = q_end - q_start
        N_CTX_K = k_end - k_start
    else:
        q_start = 0
        k_start = 0
        N_CTX_Q = max_seqlen_q
        N_CTX_K = max_seqlen_k

    # input tensor offsets
    q_offset = Q + off_z * stride_qz + off_hq * stride_qh + q_start * stride_qm
    k_offset = K + off_z * stride_kz + off_hk * stride_kh + k_start * stride_kn
    v_offset = V + off_z * stride_vz + off_hk * stride_vh + k_start * stride_vn
    do_offset = DO + off_z * stride_qz + off_hq * stride_qh + q_start * stride_qm
    l_offset = L + off_z * stride_deltaz + off_hq * stride_deltah + q_start * stride_deltam
    delta_offset = Delta + off_z * stride_deltaz + off_hq * stride_deltah + q_start * stride_deltam

    if DROPOUT:
        batch_philox_offset = philox_offset_base + off_z * stride_dropoutz + off_hq * stride_dropouth #+ q_start * stride_dropoutm
        dropout_offset = Dropout_mask + off_z * stride_dropoutz + off_hq * stride_dropouth #+ q_start * stride_dropoutm
    else:
        batch_philox_offset = 0
        dropout_offset = 0

    if IS_FP8:
        stride_descale_q_z = HQ
        stride_descale_kv_z = HK

        descale_q = tl.load(DESCALE_q + off_z * stride_descale_q_z + off_hq)
        descale_k = tl.load(DESCALE_k + off_z * stride_descale_kv_z + off_hk)
        descale_v = tl.load(DESCALE_v + off_z * stride_descale_kv_z + off_hk)
        descale_do = tl.load(DESCALE_do + off_z * stride_descale_q_z + off_hq)
    else:
        descale_q, descale_k, descale_v, descale_do = 1.0, 1.0, 1.0, 1.0
    
    # output tensor offsets
    dk_offset = DK + off_z * stride_kz + off_hk * stride_kh + k_start * stride_kn
    dv_offset = DV + off_z * stride_vz + off_hk * stride_vh + k_start * stride_vn
    if SEQUENCE_PARALLEL:
        dq_offset = DQ + start_n * stride_dq_all + off_z * stride_qz + off_hq * stride_qh + q_start * stride_qm
    else:
        dq_offset = DQ + off_z * stride_qz + off_hq * stride_qh + q_start * stride_qm

    # inner loop
    if SEQUENCE_PARALLEL:
        _bwd_kernel_one_col_block(
            Q,
            K,
            V,
            sm_scale,
            Out,
            DO,
            DQ,
            DK,
            DV,
            L,
            Delta,
            q_offset,
            k_offset,
            v_offset,
            do_offset,
            dq_offset,
            dk_offset,
            dv_offset,
            l_offset,
            delta_offset,
            dropout_offset,
            stride_dq_all,
            stride_qz,
            stride_qh,
            stride_qm,
            stride_qk,
            stride_kz,
            stride_kh,
            stride_kn,
            stride_kk,
            stride_vz,
            stride_vh,
            stride_vn,
            stride_vk,
            stride_deltaz,
            stride_deltah,
            stride_deltam,
            stride_dropoutz, stride_dropouth, stride_dropoutm, stride_dropoutn,
            N_CTX_Q,
            N_CTX_K,
            start_n,
            num_block_m,
            num_block_n,
            dropout_p,
            philox_seed,
            batch_philox_offset,
            descale_q, 
            descale_k,
            descale_v,
            descale_do,
            BLOCK_M=BLOCK_M,
            BLOCK_DMODEL=BLOCK_DMODEL,
            ACTUAL_BLOCK_DMODEL=ACTUAL_BLOCK_DMODEL,
            BLOCK_N=BLOCK_N,
            SEQUENCE_PARALLEL=SEQUENCE_PARALLEL,
            CAUSAL=CAUSAL,
            DROPOUT=DROPOUT,
            USE_EXP2=USE_EXP2,
            GROUP_SIZE=GROUP_SIZE,
            IS_FP8=IS_FP8,
            FP8_MAX=FP8_MAX
        )
    else:
        for start_n in range(0, num_block_n):
            _bwd_kernel_one_col_block(
                Q,
                K,
                V,
                sm_scale,
                Out,
                DO,
                DQ,
                DK,
                DV,
                L,
                Delta,
                q_offset,
                k_offset,
                v_offset,
                do_offset,
                dq_offset,
                dk_offset,
                dv_offset,
                l_offset,
                delta_offset,
                dropout_offset,
                stride_dq_all,
                stride_qz,
                stride_qh,
                stride_qm,
                stride_qk,
                stride_kz,
                stride_kh,
                stride_kn,
                stride_kk,
                stride_vz,
                stride_vh,
                stride_vn,
                stride_vk,
                stride_deltaz,
                stride_deltah,
                stride_deltam,
                stride_dropoutz, stride_dropouth, stride_dropoutm, stride_dropoutn,
                N_CTX_Q,
                N_CTX_K,
                start_n,
                num_block_m,
                num_block_n,
                dropout_p, 
                philox_seed, 
                batch_philox_offset,
                descale_q, 
                descale_k,
                descale_v,
                descale_do,
                BLOCK_M=BLOCK_M,
                BLOCK_DMODEL=BLOCK_DMODEL,
                ACTUAL_BLOCK_DMODEL=ACTUAL_BLOCK_DMODEL,
                BLOCK_N=BLOCK_N,
                SEQUENCE_PARALLEL=SEQUENCE_PARALLEL,
                CAUSAL=CAUSAL,
                DROPOUT=DROPOUT,
                USE_EXP2=USE_EXP2,
                GROUP_SIZE=GROUP_SIZE,
                IS_FP8=IS_FP8,
                FP8_MAX=FP8_MAX
            )


# NOTE: smaller blocks have lower accuracy. more accumulation error probably 128 * 128 seems good but leads to oom. 64 * 64 has accumulation errors but no oom.
def attention_prefill_backward_triton_impl(
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
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float, 
    philox_seed: Optional[int], 
    philox_offset: Optional[int],
    use_exp2: bool,
    sequence_parallel: bool = True,
    # fp8
    descale_q: Optional[torch.Tensor] = None,
    descale_k: Optional[torch.Tensor] = None,
    descale_v: Optional[torch.Tensor] = None,
    descale_do: Optional[torch.Tensor] = None,
):
    if DEBUG:
        print()
        print("attention_prefill_backward_triton_impl")
        print("do:", do, do.shape)
        print("q:", q, q.shape)
        print("k:", k, k.shape)
        print("v:", v, v.shape)
        print("o:", o, o.shape)
        print("softmax_lse:", softmax_lse, softmax_lse.shape)
        print("dq:", dq, dq.shape if dq is not None else None)
        print("dk:", dk, dk.shape if dk is not None else None)
        print("dv:", dv, dv.shape if dv is not None else None)
        print("sm_scale:", sm_scale)
        print("alibi_slopes:", alibi_slopes)
        print("causal:", causal)
        print("layout:", layout)
        print("cu_seqlens_q:", cu_seqlens_q)
        print("cu_seqlens_k:", cu_seqlens_k)
        print("max_seqlen_q:", max_seqlen_q)
        print("max_seqlen_k:", max_seqlen_k)
        print("dropout_p:", dropout_p)
        print("philox_seed:", philox_seed)
        print("philox_offset:", philox_offset)
        print("use_exp2:", use_exp2)
        print("sequence_parallel:", sequence_parallel)
        print("descale_q:", descale_q)
        print("descale_k:", descale_k)
        print("descale_v:", descale_v)
        print("descale_do:", descale_do)

    IS_FP8 = is_fp8(q)
    if IS_FP8:
        FP8_MAX=torch.finfo(q.dtype).max
    else:
        FP8_MAX=None

    # make contiguous
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    softmax_lse = softmax_lse.contiguous()

    # get strides and shape
    batch, nheads_q, nheads_k, head_size, max_seqlen_q, max_seqlen_k = get_shapes_from_layout(q, k, layout, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)
    q_strides, k_strides, v_strides, o_strides = get_strides_from_layout(q, k, v, o, layout)
    stride_qz, stride_qh, stride_qm, stride_qk =  q_strides
    stride_kz, stride_kh, stride_kn, stride_kk = k_strides
    stride_vz, stride_vh, stride_vn, stride_vk = v_strides
    stride_oz, stride_oh, stride_om, stride_ok = o_strides
    is_varlen = layout == "thd"
    group_size = nheads_q // nheads_k
    use_dropout = (dropout_p > 0.0)

    # FIXME: some configs lead to oom for some reason when using 64 x 64 blocks
    if max_seqlen_q <= 32 or max_seqlen_k <= 32:
        BLOCK_M = 32 
        BLOCK_N = 32
    else:
        BLOCK_M = 64 
        BLOCK_N = 64

    if DEBUG:
        print("BLOCK_M:", BLOCK_M)
        print("BLOCK_N:", BLOCK_N)

    num_warps = 4 # NOTE: original is 8. changing it to 1 caused issues be careful
    num_stages = 1
    waves_per_eu = 1

    # divide up the problem
    num_blocks_m = triton.cdiv(max_seqlen_q, BLOCK_M)
    num_blocks_n = triton.cdiv(max_seqlen_k, BLOCK_N)

    # get closest power of 2 over or equal to 32.
    padded_d_model = 1 << (head_size - 1).bit_length()
    padded_d_model = max(padded_d_model, 16)
    BLOCK_DMODEL = padded_d_model
    ACTUAL_BLOCK_DMODEL = head_size

    do = do.contiguous()

    # deal with dq
    if sequence_parallel:
        dq = dq.unsqueeze(0).repeat(num_blocks_n, *([1] * len(q.shape))) # we do repeat instead of expand because we need to write data so views are not enough
    stride_dq_all = dq.stride()[0]

    # assert contiguous
    assert do.is_contiguous()
    assert q.is_contiguous()
    assert k.is_contiguous()
    assert v.is_contiguous()
    assert o.is_contiguous()
    assert softmax_lse.is_contiguous()

    # init delta
    delta = torch.zeros_like(softmax_lse)
    if is_varlen:
        stride_deltam, stride_deltah = delta.stride()
        stride_deltaz = 0
    else:
        stride_deltaz, stride_deltah, stride_deltam = delta.stride()

    # dropout mask tensor for debugging. We dump the dropout mask created in the kernel for testing
    if use_dropout:
        if DROPOUT_USE_PYTORCH:
            dropout_mask = create_dropout_mask(dropout_p, (batch, nheads_q, max_seqlen_q, max_seqlen_k), seed = philox_seed)
        else:
            dropout_mask = torch.zeros((batch, nheads_q, max_seqlen_q, max_seqlen_k), device=q.device,
                                        dtype=torch.float32)
        stride_dropoutz, stride_dropouth, stride_dropoutm, stride_dropoutn = (dropout_mask.stride(0), dropout_mask.stride(1), dropout_mask.stride(2), dropout_mask.stride(3))
    else:
        dropout_mask = None
        stride_dropoutz, stride_dropouth, stride_dropoutm, stride_dropoutn = (0, 0 , 0 , 0)


    _bwd_preprocess[(batch * nheads_q, num_blocks_m)](
        o,
        do,
        delta,
        stride_oz, stride_oh, stride_om, stride_ok,
        stride_oz, stride_oh, stride_om, stride_ok, # FIXME: don't share strides with derivatives this was causing a lot of issues
        stride_deltaz, stride_deltah, stride_deltam,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        descale_do,
        BLOCK_M=BLOCK_M,
        BLOCK_DMODEL=BLOCK_DMODEL,
        ACTUAL_BLOCK_DMODEL=ACTUAL_BLOCK_DMODEL,
        N_CTX_Q=max_seqlen_q,
        Z=batch,
        H=nheads_q,
        IS_VARLEN=is_varlen,
        IS_FP8=IS_FP8
    )

    if DEBUG:
        print("delta:", delta, delta.shape)
        print("group_size:", group_size)

    _bwd_kernel[(batch * nheads_q, num_blocks_n if sequence_parallel else 1)](
        q,
        k,
        v,
        sm_scale,
        o,
        do,
        dq,
        dk,
        dv,
        softmax_lse,
        delta,
        dropout_mask,
        descale_q,
        descale_k,
        descale_v,
        descale_do,
        stride_dq_all,
        stride_qz, stride_qh, stride_qm, stride_qk, # FIXME: don't share strides with derivatives this was causing a lot of issues
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vn, stride_vk,
        stride_deltaz, stride_deltah, stride_deltam,
        stride_dropoutz, stride_dropouth, stride_dropoutm, stride_dropoutn,
        batch,
        nheads_q,
        nheads_k,
        num_blocks_m,
        num_blocks_n,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p, philox_seed, philox_offset,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
        ACTUAL_BLOCK_DMODEL=ACTUAL_BLOCK_DMODEL,
        SEQUENCE_PARALLEL=sequence_parallel,
        CAUSAL=causal,
        DROPOUT=use_dropout,
        USE_EXP2=use_exp2,
        num_warps=num_warps,
        num_stages=num_stages,
        waves_per_eu = waves_per_eu,
        IS_VARLEN=is_varlen,
        GROUP_SIZE=group_size,
        IS_FP8=IS_FP8,
        FP8_MAX=FP8_MAX
    )

    if sequence_parallel:
        dq = dq.sum(dim=0)

    if DEBUG:
        print("attention_prefill_backward_triton_impl outputs")
        print("dv:", dv, dv.shape)
        print("dk:", dk, dk.shape)
        print("dq:", dq, dq.shape)
        if use_dropout:
            print("dropout_mask:", dropout_mask, dropout_mask.shape if dropout_mask is not None else None)
            print("dropout_fraction bwd:", 1.0 - (dropout_mask.sum()/ dropout_mask.numel()).item())
            write_dropout_mask(dropout_mask, "dropout_mask_bwd")

    return delta
