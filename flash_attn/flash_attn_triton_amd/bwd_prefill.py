import torch
import triton
import triton.language as tl
from .utils import DEBUG, DROPOUT_USE_PYTORCH, DROPOUT_DUMP, get_shape_from_layout, get_strides_from_layout, write_dropout_mask, create_dropout_mask

# TODO: move this into utils.py so it's shared among kernels
# NOTE: triton fails to import tl.constexprs so create them here for the file
tl_DROPOUT_USE_PYTORCH: tl.constexpr = DROPOUT_USE_PYTORCH
tl_DROPOUT_DUMP: tl.constexpr = DROPOUT_DUMP

@triton.jit
def _bwd_preprocess_use_o(
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
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    ACTUAL_BLOCK_DMODEL: tl.constexpr,
    N_CTX_Q: tl.constexpr,
    Z: tl.constexpr,
    H: tl.constexpr,
    IS_VARLEN: tl.constexpr
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
    o = tl.load(out_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
    do = tl.load(do_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    # compute delta
    delta = tl.sum(o * do, axis=1)

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
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    ACTUAL_BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SEQUENCE_PARALLEL: tl.constexpr,
    CAUSAL: tl.constexpr,
    DROPOUT: tl.constexpr,
    USE_EXP2: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
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
    v = tl.load(v_ptrs, mask=kv_mask, other=0.0)

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
        qk += tl.dot(q, tl.trans(k))

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
            p_drop_scaled = p_drop_scaled.to(tl.float16)

            # compute dv
            dv += tl.dot(tl.trans(p_drop_scaled), do)

            # compute dp
            dp_drop_scaled = tl.dot(do, tl.trans(v))
            dp = tl.where(dropout_mask, dp_drop_scaled, 0.0) * dropout_scale

            # compute ds
            delta_ptrs = delta_offset + offs_m * stride_deltam
            delta_i = tl.load(delta_ptrs, mask=mask_m)
            dscores_scaled = (p * (dp - delta_i[:, None]))
            ds = dscores_scaled * sm_scale
            ds = tl.where(p_mask, ds, 0.0)
            ds = ds.to(tl.float16)
        else:
            p = p.to(tl.float16)

            # compute dv
            dv += tl.dot(tl.trans(p), do)

            # compute dp
            dp = tl.dot(do, tl.trans(v))

            # compute ds
            delta_ptrs = delta_offset + offs_m * stride_deltam
            delta_i = tl.load(delta_ptrs, mask=mask_m)
            dscores_scaled = (p * (dp - delta_i[:, None]))
            ds = dscores_scaled * sm_scale
            ds = tl.where(p_mask, ds, 0.0)
            ds = ds.to(tl.float16)
            
        # compute dk
        dk += tl.dot(tl.trans(ds), q)

        # compute dq
        if SEQUENCE_PARALLEL:
            dq = tl.dot(ds, k)
        else:
            dq = tl.load(dq_ptrs, mask=q_mask, other=0.0)
            dq += tl.dot(ds, k)
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
):
    # program ids
    off_zh = tl.program_id(0)
    if SEQUENCE_PARALLEL:
        start_n = tl.program_id(1)
    off_z = off_zh // HQ
    off_hq = off_zh % HQ

    GROUP_SIZE = HQ // HK
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
            dropout_p, philox_seed, batch_philox_offset,
            BLOCK_M=BLOCK_M,
            BLOCK_DMODEL=BLOCK_DMODEL,
            ACTUAL_BLOCK_DMODEL=ACTUAL_BLOCK_DMODEL,
            BLOCK_N=BLOCK_N,
            SEQUENCE_PARALLEL=SEQUENCE_PARALLEL,
            CAUSAL=CAUSAL,
            DROPOUT=DROPOUT,
            USE_EXP2=USE_EXP2,
            GROUP_SIZE=GROUP_SIZE
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
                dropout_p, philox_seed, batch_philox_offset,
                BLOCK_M=BLOCK_M,
                BLOCK_DMODEL=BLOCK_DMODEL,
                ACTUAL_BLOCK_DMODEL=ACTUAL_BLOCK_DMODEL,
                BLOCK_N=BLOCK_N,
                SEQUENCE_PARALLEL=SEQUENCE_PARALLEL,
                CAUSAL=CAUSAL,
                DROPOUT=DROPOUT,
                USE_EXP2=USE_EXP2,
                GROUP_SIZE=GROUP_SIZE
            )


# NOTE: smaller blocks have lower accuracy. more accumlation error probably 128 * 128 seems good but leads to oom. 64 * 64 has accumlation errors but no oom.
def attention_prefill_backward_triton_impl(
    do,
    q,
    k,
    v,
    o,
    softmax_lse,
    dq,
    dk,
    dv,
    sm_scale: float,
    alibi_slopes,
    causal,
    layout: str,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p, 
    philox_seed, 
    philox_offset,
    use_exp2: bool,
    sequence_parallel = True,
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

    # make contigious
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    softmax_lse = softmax_lse.contiguous()

    # get strides and shape
    batch, nheads_q, nheads_k, head_size, max_seqlen_q, max_seqlen_k = get_shape_from_layout(q, k, layout, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)
    q_strides, k_strides, v_strides, o_strides = get_strides_from_layout(q, k, v, o, layout)
    stride_qz, stride_qh, stride_qm, stride_qk =  q_strides
    stride_kz, stride_kh, stride_kn, stride_kk = k_strides
    stride_vz, stride_vh, stride_vn, stride_vk = v_strides
    stride_oz, stride_oh, stride_om, stride_ok = o_strides
    is_varlen = layout == "thd"
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

    num_warps = 4 # NOTE: originial is 8. changing it to 1 caused issues be careful
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
    if dq is None:
        if sequence_parallel:
            dq = torch.zeros((num_blocks_n,) + q.shape, device=q.device, dtype=q.dtype)
        else:
            dq = torch.zeros(q.shape, device=q.device, dtype=q.dtype)
    stride_dq_all = dq.stride()[0]

    # deal with dk, dv
    if (dk is None) or (dv is None):
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)


    # zero out
    dq.zero_()
    dk.zero_()
    dv.zero_()

    # assert contigious
    assert do.is_contiguous()
    assert q.is_contiguous()
    assert k.is_contiguous()
    assert v.is_contiguous()
    assert o.is_contiguous()
    assert softmax_lse.is_contiguous()

    # init delta
    delta = torch.empty_like(softmax_lse)
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


    _bwd_preprocess_use_o[(batch * nheads_q, num_blocks_m)](
        o,
        do,
        delta,
        stride_oz, stride_oh, stride_om, stride_ok,
        stride_oz, stride_oh, stride_om, stride_ok,
        stride_deltaz, stride_deltah, stride_deltam,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        BLOCK_M=BLOCK_M,
        BLOCK_DMODEL=BLOCK_DMODEL,
        ACTUAL_BLOCK_DMODEL=ACTUAL_BLOCK_DMODEL,
        N_CTX_Q=max_seqlen_q,
        Z=batch,
        H=nheads_q,
        IS_VARLEN=is_varlen
    )

    if False:
        print("_bwd_kernel inputs")
        print("do:", do, do.shape)
        print("q:", q, q.shape)
        print("k:", k, k.shape)
        print("v:", v, v.shape)
        print("sm_scale", sm_scale)
        print("o:", o, o.shape)
        print("dq:", dq, dq.shape)
        print("dk:", dk, dk.shape)
        print("dv:", dv, dv.shape)
        print("L:", softmax_lse, softmax_lse.shape)
        print("delta:", delta, delta.shape)
        print("stride_qz, stride_qh, stride_qm, stride_qk:",  stride_qz, stride_qh, stride_qm, stride_qk)
        print("stride_kz, stride_kh, stride_kn, stride_kk:",  stride_kz, stride_kh, stride_kn, stride_kk)
        print("stride_vz, stride_vh, stride_vn, stride_vk:",  stride_vz, stride_vh, stride_vn, stride_vk)
        print("batch_q:", batch)
        print("heads_q:",nheads_q)
        print("max_seqlen_q:",max_seqlen_q)
        print("max_seqlen_k:",max_seqlen_k)
        print("dropout_p:",dropout_p)
        print("philox_seed:", philox_seed)
        print("philox_offset:",philox_offset)
        print("BLOCK_M:",BLOCK_M)
        print("BLOCK_N:",BLOCK_M)
        print("BLOCK_DMODEL:",BLOCK_DMODEL)
        print("ACTUAL_BLOCK_DMODEL:",ACTUAL_BLOCK_DMODEL)
        print("SEQUENCE_PARALLEL:",sequence_parallel)
        print("CAUSAL:",causal)
        print("DROPOUT:", use_dropout)
        print("num_warps:",num_warps)
        print("num_stages:", num_stages)
        print("USE_EXP2:", use_exp2)
        print("num_blocks_m:", num_blocks_m)
        print("num_blocks_n:", num_blocks_n)

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
        stride_dq_all,
        stride_qz, stride_qh, stride_qm, stride_qk,
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
        IS_VARLEN=is_varlen
    )

    if sequence_parallel:
        dq = dq.sum(dim=0)

    if DEBUG:
        print("attention_prefill_backward_triton_impl outputs")
        print("delta:", delta, delta.shape)
        print("dv:", dv, dv.shape)
        print("dk:", dk, dk.shape)
        print("dq:", dq, dq.shape)
        print("copy_back:", copy_back)
        if use_dropout:
            print("dropout_mask:", dropout_mask, dropout_mask.shape if dropout_mask is not None else None)
            print("dropout_fraction bwd:", 1.0 - (dropout_mask.sum()/ dropout_mask.numel()).item())
            write_dropout_mask(dropout_mask, "dropout_mask_bwd")

    return dq, dk, dv, delta, None, None
