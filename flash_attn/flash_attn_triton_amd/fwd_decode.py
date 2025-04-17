import torch
import triton
import triton.language as tl
from typing import Literal, Optional, Union
from .utils import AUTOTUNE, DEBUG, get_padded_headsize, get_shape_and_strides_from_layout, is_cdna

def get_cdna_autotune_configs():
    return [
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'waves_per_eu': 2, 'PRE_LOAD_V': False}, num_stages=1,
                      num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 2, 'PRE_LOAD_V': False}, num_stages=1,
                      num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 3, 'PRE_LOAD_V': False}, num_stages=1,
                      num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 1, 'PRE_LOAD_V': False}, num_stages=1,
                      num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'waves_per_eu': 2, 'PRE_LOAD_V': False}, num_stages=1,
                      num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'waves_per_eu': 1, 'PRE_LOAD_V': False}, num_stages=1,
                      num_warps=4),
        # Fall-back config.
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'waves_per_eu': 1, 'PRE_LOAD_V': False}, num_stages=1,
                      num_warps=4),
    ], ['IS_CAUSAL', 'dropout_p', 'MAX_SEQLENS_Q', 'MAX_SEQLENS_K', 'ACTUAL_BLOCK_DMODEL', 'VARLEN', 'HQ', 'HK']

def get_autotune_configs():
    if AUTOTUNE:
        if is_cdna():
            autotune_configs, autotune_keys = get_cdna_autotune_configs()
            fwd_auto_tune_configs, fwd_autotune_keys= autotune_configs, autotune_keys
            reduce_auto_tune_configs, reduce_autotune_keys = autotune_configs, autotune_keys
            return (fwd_auto_tune_configs, fwd_autotune_keys), (reduce_auto_tune_configs, reduce_autotune_keys)
        else:
            raise ValueError("Unknown Device Type")
    else:
        autotune_configs, autotune_keys = [
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 64, "waves_per_eu": 1, "PRE_LOAD_V": False},
                num_stages=1,
                num_warps=4,
            ),
        ], [
            "IS_CAUSAL",
            "dropout_p",
            "MAX_SEQLENS_Q",
            "MAX_SEQLENS_K",
            "ACTUAL_BLOCK_DMODEL",
            "VARLEN",
            "HQ",
            "HK",
        ]

        fwd_auto_tune_configs, fwd_autotune_keys= autotune_configs, autotune_keys
        reduce_auto_tune_configs, reduce_autotune_keys = autotune_configs, autotune_keys
        return (fwd_auto_tune_configs, fwd_autotune_keys), (reduce_auto_tune_configs, reduce_autotune_keys)


(fwd_auto_tune_configs, fwd_autotune_keys), (reduce_auto_tune_configs, reduce_autotune_keys) = get_autotune_configs()

# @triton.autotune(
#     configs=fwd_auto_tune_configs,
#     key=fwd_autotune_keys,
#     use_cuda_graph=True,
# )
@triton.jit
def _fwd_kernel_splitK(
    Q,
    K,
    V,
    sm_scale,
    Out_splitK,  # [B*H*G, split_k, Mq, K]
    Metadata,  # [B*H*G, 2, split_k, M_ceil] contains [mi, li]
    K_new,
    V_new,
    Cache_seqlens,
    Cache_batch_idx,
    Alibi_slopes,
    stride_qz,
    stride_qm,
    stride_qg,
    stride_qh,
    stride_qd,
    stride_kz,
    stride_kn,
    stride_kg,
    stride_kh,
    stride_kd,
    stride_vz,
    stride_vn,
    stride_vg,
    stride_vh,
    stride_vd,
    stride_osk_zhg,
    stride_osk_s,
    stride_osk_m,
    stride_osk_d,
    stride_mzhg,
    stride_m2,
    stride_ms,
    stride_mm,
    stride_kn_z,
    stride_kn_n,
    stride_kn_g,
    stride_kn_h,
    stride_kn_d,
    stride_vn_z,
    stride_vn_n,
    stride_vn_g,
    stride_vn_h,
    stride_vn_d,
    stride_az, 
    stride_ah,
    Z,
    N_CTX_Q,
    N_CTX_K,
    N_CTX_NEW,
    BLOCK_N_PER_SPLIT,
    H_q: tl.constexpr,
    H_kv: tl.constexpr,
    G_q: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    ACTUAL_BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BOUNDS_CHECKS_N: tl.constexpr,
    USE_CACHE_SEQLENs: tl.constexpr,
    USE_CACHE_BATCH_IDX: tl.constexpr,
    NEW_KV: tl.constexpr,
    IS_GQA: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    USE_ALIBI: tl.constexpr,
    PADDED_HEAD: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    # get program ids
    pid_m = tl.program_id(0)
    pid_zhg = tl.program_id(1)
    pid_splitk = tl.program_id(2)

    # compute z, h and g ids
    z_id = pid_zhg // (H_q * G_q)
    hq_id = (pid_zhg // G_q) % H_q
    g_id = pid_zhg % G_q

    # is gqa
    if IS_GQA:
        hk_id = hq_id // GROUP_SIZE
        hv_id = hk_id
    else:
        hk_id = hq_id
        hv_id = hq_id

    # figure out seqlens
    lo = pid_splitk * BLOCK_N_PER_SPLIT
    if USE_CACHE_SEQLENs:
        cache_seqlen_last_idx = tl.load(Cache_seqlens + z_id)
        if NEW_KV:
            N_CTX_K_FINAL = cache_seqlen_last_idx + N_CTX_NEW
        else:
            N_CTX_K_FINAL = cache_seqlen_last_idx
    else:
        N_CTX_K_FINAL = N_CTX_K
    hi = tl.minimum((pid_splitk + 1) * BLOCK_N_PER_SPLIT, N_CTX_K_FINAL)

    # pick batch index
    if USE_CACHE_BATCH_IDX:
        cache_batch_idx = tl.load(Cache_batch_idx + z_id)
    else:
        cache_batch_idx = z_id

    # compute offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # compute ptrs
    q_offset = Q + hq_id * stride_qh + z_id * stride_qz + g_id * stride_qg
    q_ptrs = q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    k_offset = K + hk_id * stride_kh + cache_batch_idx * stride_kz + g_id * stride_kg
    v_offset = V + hv_id * stride_vh + cache_batch_idx * stride_vz + g_id * stride_vg

    # compute masks
    if PADDED_HEAD:
        q_mask = (offs_m < N_CTX_Q)[:, None] & (offs_d < ACTUAL_BLOCK_DMODEL)[None, :]
        kT_mask = (offs_d < ACTUAL_BLOCK_DMODEL)[:, None] & (offs_n < N_CTX_K_FINAL)[None, :]
        v_mask = (offs_n < N_CTX_K_FINAL)[:, None] & (offs_d < ACTUAL_BLOCK_DMODEL)[None, :]
        osk_mask = (offs_m < N_CTX_Q)[:, None] & (offs_d < ACTUAL_BLOCK_DMODEL)[None, :]
    else:
        q_mask = (offs_m < N_CTX_Q)[:, None]
        kT_mask = (offs_n < N_CTX_K_FINAL)[None, :]
        v_mask = (offs_n < N_CTX_K_FINAL)[:, None]
        osk_mask = (offs_m < N_CTX_Q)[:, None]

    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504
    
    # load q: it will stay in SRAM throughout
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)
    q = (q * qk_scale).to(q.dtype)

    # load ALiBi slope if enabled
    if USE_ALIBI:
        a_offset = z_id * stride_az + hq_id * stride_ah
        alibi_slope = tl.load(Alibi_slopes + a_offset)
    else:
        alibi_slope = None

    # Copy new Keys and Values into Cache
    if NEW_KV:
        knew_base = K_new + hk_id * stride_kn_h + z_id * stride_kn_z + g_id * stride_kn_g
        
        # Determine the starting position for new data in the cache
        if USE_CACHE_SEQLENs:
            start_idx = tl.load(Cache_seqlens + z_id)
        else:
            start_idx = N_CTX_K - N_CTX_NEW

        # Copy new Keys
        for i in range(0, N_CTX_NEW, BLOCK_N):
            # Load from K_new
            k_new_block = tl.load(
                knew_base +
                tl.arange(0, BLOCK_DMODEL)[:, None] * stride_kn_d +
                (tl.arange(0, BLOCK_N) + i)[None, :] * stride_kn_n,
                 mask=(tl.arange(0, BLOCK_N)[None, :] + i < N_CTX_NEW) &
                     (tl.arange(0, BLOCK_DMODEL)[:, None] < ACTUAL_BLOCK_DMODEL),
                other=0
            )
            
            # Store to K
            tl.store(
                k_offset +
                tl.arange(0, BLOCK_DMODEL)[:, None] * stride_kd +
                (tl.arange(0, BLOCK_N) + i + start_idx)[None, :] * stride_kn,
                k_new_block,
                 mask=(tl.arange(0, BLOCK_N)[None, :] + i < N_CTX_NEW) &
                     (tl.arange(0, BLOCK_DMODEL)[:, None] < ACTUAL_BLOCK_DMODEL),
            )

        # Copy new Values
        vnew_base = V_new + hv_id * stride_vn_h + z_id * stride_vn_z + g_id * stride_vn_g
        for i in range(0, N_CTX_NEW, BLOCK_N):
            # Load from V_new
            v_new_block = tl.load(
                vnew_base +
                (tl.arange(0, BLOCK_N) + i)[:, None] * stride_vn_n +
                tl.arange(0, BLOCK_DMODEL)[None, :] * stride_vn_d,
                mask=(tl.arange(0, BLOCK_N)[:, None] + i < N_CTX_NEW) &
                     (tl.arange(0, BLOCK_DMODEL)[None, :] < ACTUAL_BLOCK_DMODEL),
                other=0
            )
            
            # Store to V
            tl.store(
                v_offset + 
                (tl.arange(0, BLOCK_N) + i + start_idx)[:, None] * stride_vn +
                tl.arange(0, BLOCK_DMODEL)[None, :] * stride_vd,
                v_new_block,
                 mask=(tl.arange(0, BLOCK_N)[:, None] + i < N_CTX_NEW) &
                     (tl.arange(0, BLOCK_DMODEL)[None, :] < ACTUAL_BLOCK_DMODEL),
            )


    # initialize pointer to m and l
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)  # noqa: F821


    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        kT_ptrs = k_offset + offs_d[:, None] * stride_kd + (start_n + offs_n)[None, :] * stride_kn
        V_ptrs = v_offset + (start_n + offs_n)[:, None] * stride_vn + offs_d[None, :] * stride_vd

        # load k
        kT = tl.load(kT_ptrs, mask=kT_mask, other=0.0)
        v = tl.load(V_ptrs, mask=v_mask, other=0.0)

        # -- compute qk ---
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, kT)  # noqa: F821

        if USE_ALIBI:
            row_idx = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            col_idx = start_n + tl.arange(0, BLOCK_N)
            
            # Compute relative positions
            relative_pos = row_idx[:, None] + N_CTX_K_FINAL - (N_CTX_Q + col_idx[None, :])
            relative_pos = tl.abs(relative_pos)
            
            # Compute ALiBi bias
            alibi_bias = -1 * alibi_slope * relative_pos
            qk += (alibi_bias * 1.44269504)

        # Apply causal mask if IS_CAUSAL is True
        if IS_CAUSAL:
            row_idx = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            col_idx = start_n + tl.arange(0, BLOCK_N)
            
            # create a N_CTX_Q x kv_len causal mask
            col_offset = N_CTX_Q - N_CTX_K_FINAL
            causal_mask = row_idx[:, None] >= (col_offset + col_idx[None, :])

            # Apply the mask
            qk = tl.where(causal_mask, qk, float("-inf"))

        # TODO: This is slow, and only needed at the last iteration.
        # Maybe we can unroll the last iteration instead?
        if BOUNDS_CHECKS_N:
            qk = tl.where(tl.arange(0, BLOCK_N) < hi - start_n, qk, float("-inf"))

        # -- compute scaling constant ---
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        if IS_CAUSAL:
            alpha = tl.math.exp2(tl.where(m_i > float("-inf"), m_i - m_i_new, float("-inf")))
        else:
            alpha = tl.math.exp2(m_i - m_i_new)
        # cause of nan because subtracting infs
        if IS_CAUSAL:
            qk = tl.where(qk > float("-inf"), qk - m_i_new[:, None], float("-inf"))
        else:
            qk = qk - m_i_new[:, None] 
        
        p = tl.math.exp2(qk)

        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        p = p.to(Q.dtype.element_ty)

        # -- scale and update acc --
        acc *= alpha[:, None]
        acc += tl.dot(p.to(v.dtype), v)

    # write back O
    osk_offset = Out_splitK + pid_zhg * stride_osk_zhg + pid_splitk * stride_osk_s
    osk_ptrs = osk_offset + offs_m[:, None] * stride_osk_m + offs_d[None, :] * stride_osk_d 
    tl.store(
        osk_ptrs,
        acc,
        mask=osk_mask,
    )

    # write metadata for split-K reduction
    metadata_offset = Metadata + pid_zhg * stride_mzhg + pid_splitk * stride_ms
    metadata_ptr = metadata_offset + offs_m
    tl.store(metadata_ptr, m_i)
    tl.store(metadata_ptr + stride_m2, l_i)


# @triton.autotune(
#     configs=reduce_auto_tune_configs,
#     key=reduce_autotune_keys,
#     use_cuda_graph=True,
# )
@triton.jit
def _splitK_reduce(
    Out_splitK,  # [B*H*G, split_k, Mq, K]
    Metadata,  # [B*H*G, 2, split_k, M_ceil] contains [mi, li]
    Out,  # [B, H, G, M, K]
    LSE,  # [B*H*G, M]
    stride_osk_zhg,
    stride_osk_s,
    stride_osk_m,
    stride_osk_k,
    stride_mzhg,
    stride_m2,
    stride_ms,
    stride_mm,
    stride_oz,
    stride_oh,
    stride_og,
    stride_om,
    stride_ok,
    stride_lse_zhg,
    stride_lse_m,
    K_BLOCK_SIZE: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    ACTUAL_BLOCK_DMODEL: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    split_k: tl.constexpr,
    splitK_pow2: tl.constexpr,
    MASK_SPLITK: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    PADDED_HEAD: tl.constexpr,
):
    # get pids
    pid_zhg = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_k = tl.program_id(2)

    # compute offsets
    offs_splitK = tl.arange(0, splitK_pow2)
    offs_k = pid_k * K_BLOCK_SIZE + tl.arange(0, K_BLOCK_SIZE)


    # compute masks
    if PADDED_HEAD:
        o_mask = offs_k < ACTUAL_BLOCK_DMODEL
    else:
        o_mask = None

    # compute ptrs
    metadata_offset = Metadata + pid_zhg * stride_mzhg
    metadata_ptr = metadata_offset + offs_splitK * stride_ms + pid_m * stride_mm

    osk_offset = Out_splitK + pid_zhg * stride_osk_zhg + pid_m * stride_osk_m
    osk_ptr = osk_offset + offs_splitK[:, None] * stride_osk_s + offs_k[None, :] * stride_osk_k

    # read max values of each splitK
    if MASK_SPLITK:
        splitK_mask = offs_splitK < split_k
        l_m = tl.load(metadata_ptr, mask=splitK_mask, other=float("-inf"))
        l_sum = tl.load(metadata_ptr + stride_m2, mask=splitK_mask, other=0.0)
        acc = tl.load(osk_ptr, mask=splitK_mask[:, None], other=0.0)
    else:
        l_m = tl.load(metadata_ptr)
        l_sum = tl.load(metadata_ptr + stride_m2)
        acc = tl.load(osk_ptr)

    g_m = tl.max(l_m, axis=0)
    
    if IS_CAUSAL:
        l_m_offset = l_m - g_m
        alpha = tl.where(l_m_offset > float("-inf"), tl.math.exp2(l_m_offset), 0.0)
    else:
        alpha = tl.math.exp2(l_m - g_m)

    # read sum
    l_sum *= alpha
    g_sum = tl.sum(l_sum, axis=0)
    acc = acc * alpha[:, None]

    if IS_CAUSAL:
        # Avoid division by zero
        g_sum_safe = tl.where(g_sum > 0, g_sum, 1.0)
        acc_out = tl.sum(acc, axis=0) / g_sum_safe
    else:
        acc_out = tl.sum(acc, axis=0) / g_sum

    # Store output
    z_id = pid_zhg // (H * G)
    h_id = (pid_zhg // G) % H
    g_id = pid_zhg % G
    out_offset = Out + z_id * stride_oz + h_id * stride_oh  + g_id * stride_og
    out_ptr = out_offset + pid_m * stride_om + offs_k
    tl.store(out_ptr, acc_out, mask=o_mask)

    # Store lse
    l_ptrs = LSE + pid_zhg * stride_lse_zhg + pid_m
    if IS_CAUSAL:
        lse = tl.where(g_sum > 0, (g_m + tl.math.log2(g_sum)) / 1.44269504, g_m)
        tl.store(l_ptrs, lse)
    else:
        tl.store(l_ptrs, (g_m + tl.math.log2(g_sum)) / 1.44269504)


@triton.jit
def cast_uint32_to_half2(scale_shift):
    # Extract two float16 packed into one int32
    scale = scale_shift & 0xFFFF
    shift = scale_shift >> 16
    scale = scale.to(tl.uint16).to(tl.float16, bitcast=True)
    shift = shift.to(tl.uint16).to(tl.float16, bitcast=True)
    return scale, shift

@triton.jit
def dequantize(
    x_,
    scale,
    shift,
    PACKED_PER_VAL: tl.constexpr = 8,
):
    # PACKED_PER_VAL is the number of values packed into
    # each element x_. For example, for int4 quantization
    #and x_ of type int32, PACKED_PER_VAL is 8.

    BLOCK_N: tl.constexpr = x_.shape[0]
    BLOCK_DMODEL_PACKED: tl.constexpr = x_.shape[1]
    offsets = tl.arange(0, PACKED_PER_VAL) * 4
    quant_offset = (x_[:, None, :] >> offsets[None, :, None])  # (BLOCK_N, PACKED_PER_VAL, D // PACKED_PER_VAL)

    quant_offset = tl.view(quant_offset, (BLOCK_N, BLOCK_DMODEL_PACKED * PACKED_PER_VAL))
    # Trick - instead of converting int4 to float16 we view it as float16
    # and then multiply by 32768 * 512 == 2**24
    quant_offset = (quant_offset & 0xF).to(tl.uint16).to(tl.float16, bitcast=True)
    quant_offset = (quant_offset * 32768.0).to(tl.float16)
    scale_512 = scale * 512

    dequant = quant_offset * scale_512 + shift
    return dequant

def quantize_kv_int4(k: torch.Tensor, num_groups: int = 1) -> torch.Tensor:
    # Scale and shift are such that quantization linearly maps
    # int4 values range [0..15] to input values range min(k)..max(k)
    # individually for every row
    k = k.reshape(*k.shape[:-1], num_groups, k.shape[-1] // num_groups)
    max_vals = torch.max(k, dim=-1, keepdim=True).values
    min_vals = torch.min(k, dim=-1, keepdim=True).values
    scale_k: torch.Tensor = (max_vals - min_vals) / 15

    shift_k = torch.min(k, dim=-1, keepdim=True).values
    scale_k = scale_k.to(torch.float16)
    shift_k = shift_k.to(torch.float16)

    in_bytes = ((k - shift_k.expand(k.shape)) / scale_k.expand(k.shape)) + 0.5
    in_bytes = in_bytes.to(torch.uint8)
    in_int4 = in_bytes & 0xF
    in_int4_packed = in_int4[..., ::2] + (in_int4[..., 1::2] << 4)
    scale_shift = torch.concat([scale_k.view(torch.uint8), shift_k.view(torch.uint8)], dim=-1)
    k_quant = torch.concat(
        [
            scale_shift.flatten(start_dim=-2),
            in_int4_packed.flatten(start_dim=-2),
        ],
        dim=-1,
    ).view(torch.int16)
    return k_quant


def dequantize_kv_fp16(quant_k: torch.Tensor, num_groups: int = 1) -> torch.Tensor:
    k_i16 = quant_k.view(torch.int16)
    k_ui8 = k_i16.view(torch.uint8)

    ss_size = num_groups * 4
    scale_shift_ui8 = k_ui8[..., 0:ss_size]
    scale_shift_ui8 = scale_shift_ui8.reshape(*scale_shift_ui8.shape[:-1], num_groups, 4)
    scale = scale_shift_ui8[..., 0:2].view(torch.float16)
    shift = scale_shift_ui8[..., 2:4].view(torch.float16)

    kv_ui8 = k_ui8[..., ss_size:]
    k_ui8 = kv_ui8.reshape(*kv_ui8.shape[:-1], num_groups, -1)
    k1_i4 = k_ui8 & 0xF
    k2_i4 = (k_ui8 & 0xF0) >> 4
    k_shape = k1_i4.shape
    k1_f16 = k1_i4.to(torch.float16) * scale.expand(k_shape) + shift.expand(k_shape)
    k2_f16 = k2_i4.to(torch.float16) * scale.expand(k_shape) + shift.expand(k_shape)

    out = torch.empty((*k1_f16.shape[:-1], k1_f16.shape[-1] * 2), dtype=torch.float16, device=quant_k.device)
    out[..., ::2] = k1_f16
    out[..., 1::2] = k2_f16
    out = out.reshape(*k_shape[:-2], -1)

    return out


def get_split_k(B: int, G: int, H: int, Mk: int) -> int:
    """Heuristic for the number of splits"""
    bh = max(B * H, 1)  # NOTE: Handle B*h=0 case
    split_k = max(Mk, 1024) // bh
    max_chunk_size = 64
    while split_k > 0 and Mk / split_k < max_chunk_size:
        split_k = split_k // 2
    while B * H * G * split_k >= 1024:
        split_k = split_k // 2
    split_k = min(split_k, 512)
    split_k = max(split_k, 1)
    return split_k

def attention_decode_forward_triton_impl(
        q: torch.Tensor, 
        k_cache: torch.Tensor, 
        v_cache: torch.Tensor,
        k_new: Optional[torch.Tensor],
        v_new: Optional[torch.Tensor],
        out: torch.Tensor,
        sm_scale: float, 
        causal: bool, 
        alibi_slopes: Optional[torch.Tensor], 
        layout: Literal["bshd"], 
        cache_seqlens: Optional[Union[(int, torch.Tensor)]], 
        cache_batch_idx: Optional[torch.Tensor],
):
    # triton configs
    BLOCK_M = 16
    BLOCK_N = 64
    num_stages = 1
    num_warps_fwd = 1
    num_warps_reduce = 4
        
    # kernel_configs
    is_new_kv = True if k_new is not None and v_new is not None else False
    use_alibi = False if alibi_slopes is None else True
    use_cache_seqlens = cache_seqlens is not None
    SPLIT_K = None
    NUM_QUANT_GROUPS = 1

    # get shapes and strides
    (batch_size, seqlen_q, nheads_q, dim_q), (stride_qz, stride_qh, stride_qm, stride_qd) = get_shape_and_strides_from_layout(q, layout)
    (_, seqlen_kc, nheads_kc, dim_kc), (stride_kc_z, stride_kc_h, stride_kc_n, stride_kc_d) = get_shape_and_strides_from_layout(k_cache, layout)
    (_, seqlen_vc, nheads_vc, dim_vc), (stride_vc_z, stride_vc_h, stride_vc_n, stride_vc_d) = get_shape_and_strides_from_layout(v_cache, layout)
    if is_new_kv:
        ( _, seqlen_kn, nheads_kn, dim_kn), (stride_kn_z, stride_kn_h, stride_kn_n, stride_kn_d) = get_shape_and_strides_from_layout(k_new, layout)
        (_, seqlen_vn, nheads_vn, dim_vn), (stride_vn_z, stride_vn_h, stride_vn_n, stride_vn_d) = get_shape_and_strides_from_layout(v_new, layout)
    else:
        ( _, seqlen_kn, nheads_kn, dim_kn), (stride_kn_z, stride_kn_h, stride_kn_n, stride_kn_d) = (None, None, None, None), (None, None, None, None)
        (_, seqlen_vn, nheads_vn, dim_vn), (stride_vn_z, stride_vn_h, stride_vn_n, stride_vn_d) = (None, None, None, None), (None, None, None, None)
    (_, seqlen_o, nheads_o, dim_o), (stride_oz, stride_oh, stride_om, stride_od) = get_shape_and_strides_from_layout(out, layout)
    if use_alibi:
        stride_az, stride_ah = alibi_slopes.stride()
    else:
        stride_az, stride_ah = (None, None)

    assert dim_q == dim_kc == dim_vc, f"Dimensions must match: {dim_q}, {dim_kc}, {dim_vc}"

    # add extra information needed by the kernels
    if layout == "bshd":
        (n_group_q, heads_per_group_q), stride_qg = (1, nheads_q), stride_qm
        (n_group_k, heads_per_group_k), stride_kc_g = (1, nheads_kc), stride_kc_n
        (n_group_v, heads_per_group_v), stride_vc_g = (1, nheads_vc), stride_vc_n
        if is_new_kv:
            (n_group_kn, heads_per_group_kn), stride_kn_g = (1, nheads_kn), stride_kn_n
            (n_group_vn, heads_per_group_vn), stride_vn_g = (1, nheads_vn), stride_vn_n
        else:
            (n_group_kn, heads_per_group_kn), stride_kn_g = (None, None), None
            (n_group_vn, heads_per_group_vn), stride_vn_g = (None, None), None
        (n_group_o, heads_per_group_o), stride_og = (1, nheads_o), stride_om
    else:
        raise ValueError(f"{layout} layout is not supported")

    # get padded size
    dim_padded  = get_padded_headsize(dim_kc)
    is_padded_head = dim_padded != dim_kc

    # Handle MQA/GQA case
    group_size = nheads_q // nheads_kc
    if group_size > 1:
        is_gqa = True
    else:
        is_gqa = False

    if SPLIT_K is not None:
        split_k = SPLIT_K
    else:
        # Use heuristics
        split_k = get_split_k(batch_size, n_group_q, heads_per_group_q, seqlen_kc) # NOTE: should the split think about seqlens?
    split_size = (seqlen_kc + split_k - 1) // split_k

    # setup grid
    seqlen_q_ceil = (seqlen_q + BLOCK_M - 1) // BLOCK_M * BLOCK_M
    grid = lambda META: (triton.cdiv(seqlen_q, META['BLOCK_M']),  batch_size * n_group_q * heads_per_group_q, split_k)
    
    # create intermediate tensors
    out_splitk = torch.empty([batch_size * n_group_q * heads_per_group_q, split_k, seqlen_q_ceil, dim_kc], dtype=torch.float32, device=q.device)
    metadata = torch.empty([batch_size * n_group_q * heads_per_group_q, 2, split_k, seqlen_q_ceil], dtype=torch.float32, device=q.device)
    lse = torch.empty((batch_size * n_group_q * heads_per_group_q, seqlen_q), dtype=torch.float32, device=q.device)
    
    # get intermediate tensor strides
    stride_osk_zhg, stride_osk_s, stride_osk_m, stride_osk_d = out_splitk.stride()
    stride_mzhg, stride_m2, stride_ms, stride_mm = metadata.stride()
    stride_lse_zhg, stride_lse_m = lse.stride()

    if False:
        print("batch_size, seqlen_q, nheads_q, dim_q", (batch_size, seqlen_q, nheads_q, dim_q))
        print("_, seqlen_kc, nheads_kc, dim_kc", (_, seqlen_kc, nheads_kc, dim_kc))
        print("dim_padded:", dim_padded)
        print("stride_qz, stride_qm, stride_qg, stride_qh, stride_qd", (stride_qz, stride_qm, stride_qg, stride_qh, stride_qd))
        print("stride_kc_z, stride_kc_n, stride_kc_g, stride_kc_h, stride_kc_d", (stride_kc_z, stride_kc_n, stride_kc_g, stride_kc_h, stride_kc_d))
        print("stride_vc_z, stride_vc_n, stride_vc_g, stride_vc_h, stride_vc_d", (stride_vc_z, stride_vc_n, stride_vc_g, stride_vc_h, stride_vc_d))
        if is_new_kv:
            print("stride_kn_z, stride_kn_n, stride_kn_g, stride_kn_h, stride_kn_d", (stride_kn_z, stride_kn_n, stride_kn_g, stride_kn_h, stride_kn_d))
            print("stride_vn_z, stride_vn_n, stride_vn_g, stride_vn_h, stride_vn_d", (stride_vn_z, stride_vn_n, stride_vn_g, stride_vn_h, stride_vn_d))
        print("stride_oz, stride_om, stride_og, stride_oh, stride_od", (stride_oz, stride_om, stride_og, stride_oh, stride_od))
        print("stride_osk_zhg, stride_osk_s, stride_osk_m, stride_osk_d", (stride_osk_zhg, stride_osk_s, stride_osk_m, stride_osk_d))
        print("stride_mzhg, stride_m2, stride_ms, stride_mm", (stride_mzhg, stride_m2, stride_ms, stride_mm))
        print("stride_lse_zhg, stride_lse_m", (stride_lse_zhg, stride_lse_m))

    # TODO: enable quantization
    _fwd_kernel_splitK[grid](
        Q=q,
        K=k_cache,
        V=v_cache,
        sm_scale=sm_scale,
        Out_splitK=out_splitk,
        Metadata=metadata,
        K_new=k_new,
        V_new=v_new,
        Cache_seqlens=cache_seqlens,
        Cache_batch_idx=cache_batch_idx,
        Alibi_slopes=alibi_slopes,
        # q strides
        stride_qz=stride_qz,
        stride_qm=stride_qm,
        stride_qg=stride_qg,
        stride_qh=stride_qh,
        stride_qd=stride_qd,
        # k strides
        stride_kz=stride_kc_z,
        stride_kn=stride_kc_n,
        stride_kg=stride_kc_g,
        stride_kh=stride_kc_h,
        stride_kd=stride_kc_d,
        # v strides
        stride_vz=stride_vc_z,
        stride_vn=stride_vc_n,
        stride_vg=stride_vc_g,
        stride_vh=stride_vc_h,
        stride_vd=stride_vc_d,
        # out_splitk strides
        stride_osk_zhg=stride_osk_zhg,
        stride_osk_s=stride_osk_s,
        stride_osk_m=stride_osk_m,
        stride_osk_d=stride_osk_d,
        # metadata strides
        stride_mzhg=stride_mzhg,
        stride_m2=stride_m2,
        stride_ms=stride_ms,
        stride_mm=stride_mm,
        # k_new strides
        stride_kn_z=stride_kn_z,
        stride_kn_n=stride_kn_n,
        stride_kn_g=stride_kn_g,
        stride_kn_h=stride_kn_h,
        stride_kn_d=stride_kn_d,
        # v_new strides
        stride_vn_z=stride_vn_z,
        stride_vn_n=stride_vn_n,
        stride_vn_g=stride_vn_g,
        stride_vn_h=stride_vn_h,
        stride_vn_d=stride_vn_d,
        # alibi strides
        stride_az=stride_az,
        stride_ah=stride_ah,
        Z=batch_size,
        H_q=heads_per_group_q,
        H_kv=heads_per_group_k,
        G_q=n_group_q,
        N_CTX_Q=seqlen_q,
        N_CTX_K=seqlen_kc,
        N_CTX_NEW=seqlen_kn,
        BLOCK_N_PER_SPLIT=split_size,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=dim_padded,
        ACTUAL_BLOCK_DMODEL=dim_kc,
        BOUNDS_CHECKS_N=(split_size % BLOCK_N) > 0 or use_cache_seqlens,
        USE_CACHE_SEQLENs=use_cache_seqlens,
        USE_CACHE_BATCH_IDX=cache_batch_idx is not None,
        NEW_KV=is_new_kv,
        IS_GQA=is_gqa,
        IS_CAUSAL=causal,
        USE_ALIBI=use_alibi,
        PADDED_HEAD=is_padded_head,
        GROUP_SIZE=group_size,
        num_warps=num_warps_fwd,
        num_stages=num_stages,
    )

    if DEBUG:
        print("Out_splitK:", out_splitk, out_splitk.shape)
        print("metadata:", metadata, metadata.shape)
        print("lse:", lse, lse.shape)
        print("Out:", out, out.shape)

    # Merge together
    splitK_pow2 = triton.next_power_of_2(split_k)
    mask_split_k = splitK_pow2 > split_k
    if batch_size * n_group_q * heads_per_group_q * seqlen_q >= 512:
        k_block_num = 1
    else:
        k_block_num = 2
    assert dim_padded % k_block_num == 0
    k_block_size = dim_padded // k_block_num
    grid = (batch_size * n_group_q * heads_per_group_q, seqlen_q, k_block_num)


    if DEBUG:
        print("splitK_pow2:", splitK_pow2)
        print("k_block_num:", k_block_num)
        print("k_block_size:", k_block_size)
        print("grid:", grid)

    _splitK_reduce[grid](
        out_splitk, 
        metadata, 
        out, 
        lse, 
        # Split-K output strides
        stride_osk_zhg=stride_osk_zhg,
        stride_osk_s=stride_osk_s,
        stride_osk_m=stride_osk_m,
        stride_osk_k=stride_osk_d,
        # Metadata strides
        stride_mzhg=stride_mzhg,
        stride_m2=stride_m2,
        stride_ms=stride_ms,
        stride_mm=stride_mm,
        # Output tensor strides
        stride_oz=stride_oz,
        stride_oh=stride_oh,
        stride_og=stride_og,
        stride_om=stride_om,
        stride_ok=stride_od,
        # LSE strides
        stride_lse_zhg=stride_lse_zhg,
        stride_lse_m=stride_lse_m,
        K_BLOCK_SIZE=k_block_size,
        BLOCK_DMODEL=dim_padded,
        ACTUAL_BLOCK_DMODEL=dim_kc,
        G=n_group_q, 
        H=heads_per_group_q,
        # TODO: Tune num_warps
        split_k=split_k, 
        splitK_pow2=splitK_pow2, 
        MASK_SPLITK=mask_split_k,
        IS_CAUSAL=causal,
        PADDED_HEAD=is_padded_head,
        num_warps=num_warps_reduce)

    return lse
