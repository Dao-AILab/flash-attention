/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

// Compiles aiter's fmha_v3_bwd dispatcher (the asm path only) into this extension. The code
// objects it launches are embedded at build time (see generate_aiter_bwd_v3 in setup.py), so no
// AITER_ASM_DIR is needed at runtime and an aiter Python package in the same process is not
// affected.
#define ONLY_FAV3 1
#define ENABLE_CK 1
#define AITER_EMBEDDED_HSA_HEADER "fa_aiter_hsa_embed.hpp"
#define AITER_EMBEDDED_HSA_MAP flash_aiter_embedded_hsa_map()

#include "fa_aiter_mha_bwd.hpp" // third_party/aiter/csrc/cpp_itfs/mha_bwd.cu, copied by setup.py
#include "fa_aiter_hsa_embed.inc"

#include "aiter_bwd_v3.hpp"

namespace flash {

bool run_aiter_bwd_v3(const fmha_bwd_args &args,
                      const std::string &dtype,
                      const ck_tile::stream_config &stream_config,
                      const std::function<void *(size_t bytes, bool zero_init)> &workspace_alloc)
{
    // Only gfx950 code objects are embedded (see AITER_BWD_ARCHS in setup.py).
    if (get_gpu_arch() != "gfx950")
        return false;
    // Where the kernels match fmha_bwd, from probing gfx950 against an fp32 reference. Outside it
    // they give wrong gradients (seqlen_q < 16, or a mask with seqlen_q/seqlen_k off the tile
    // grid, for hdim 65-128; the latter can also fault) or lose precision (fp16 on the hdim-64 and
    // hdim-192 kernels, 2-3x the error of fmha_bwd). Unmasked hdim 64 is faster on fmha_bwd.
    const int d = args.hdim_q;
    const bool masked = args.mask_type != static_cast<ck_tile::index_t>(mask_enum::no_mask);
    if (args.hdim_v != d || d < 64 || d > 192 || (d == 64 && !masked))
        return false;
    if (dtype == "fp16" && (d == 64 || d > 128))
        return false;
    if (args.seqlen_q < 16)
        return false;
    if (masked && (args.seqlen_q % 16 != 0 || args.seqlen_k % 64 != 0))
        return false;
    // The masked hdim 65-128 kernels miss a barrier and can write zeros into dk when the GPU is
    // shared with other processes. setup.py patches the bottom-right-causal ones it recognises;
    // everything else in that range stays on fmha_bwd.
    // They are also wrong (or abort) when seqlen_k % 128 == 64 across more than one KV tile.
    if (masked && d > 64 && d <= 128 &&
        (args.seqlen_k % 128 != 0 ||
         args.mask_type != static_cast<ck_tile::index_t>(mask_enum::mask_bottom_right) ||
         !flash_aiter_bwd_dk_fixed(get_gpu_arch(), dtype)))
        return false;

    aiter::mha_bwd_args a{};
    a.use_asm_v3     = true;
    // fp32 atomics for dq, as PyTorch uses: the a16 kernels need seqlen_q == seqlen_k.
    a.v3_atomic_fp32 = true;
    // gfx950 converts to bf16 with RTNE in hardware; aiter picks the same.
    a.v3_bf16_cvt    = get_gpu_arch() == "gfx950" ? 0 : 1;
    a.v3_api_check   = false;

    a.hdim_q           = args.hdim_q;
    a.hdim_v           = args.hdim_v;
    a.data_type        = dtype;
    a.is_group_mode    = false;
    a.mask_type        = args.mask_type;
    a.bias_type        = 0;
    a.has_dbias        = false;
    a.has_dropout      = false;
    a.is_store_randval = false;
    a.is_deterministic = false;

    a.q_ptr        = args.q_ptr;
    a.k_ptr        = args.k_ptr;
    a.v_ptr        = args.v_ptr;
    a.bias_ptr     = nullptr;
    a.o_ptr        = args.o_ptr;
    a.lse_ptr      = args.lse_ptr;
    a.do_ptr       = args.do_ptr;
    a.d_ptr        = args.d_ptr;
    a.rand_val_ptr = nullptr;
    a.dq_ptr       = args.dq_ptr;
    a.dk_ptr       = args.dk_ptr;
    a.dv_ptr       = args.dv_ptr;
    a.dbias_ptr    = nullptr;

    a.seqlen_q     = args.seqlen_q;
    a.seqlen_k     = args.seqlen_k;
    a.batch        = args.batch;
    a.max_seqlen_q = args.max_seqlen_q;
    a.max_seqlen_k = args.max_seqlen_k;
    a.nhead_q      = args.nhead_q;
    a.nhead_k      = args.nhead_k;
    a.scale        = args.scale;

    a.stride_q  = args.stride_q;
    a.stride_k  = args.stride_k;
    a.stride_v  = args.stride_v;
    a.stride_o  = args.stride_o;
    a.stride_do = args.stride_do;
    a.stride_dq = args.stride_dq;
    a.stride_dk = args.stride_dk;
    a.stride_dv = args.stride_dv;

    a.nhead_stride_q    = args.nhead_stride_q;
    a.nhead_stride_k    = args.nhead_stride_k;
    a.nhead_stride_v    = args.nhead_stride_v;
    a.nhead_stride_o    = args.nhead_stride_o;
    a.nhead_stride_do   = args.nhead_stride_do;
    a.nhead_stride_lsed = args.nhead_stride_lsed;
    a.nhead_stride_dq   = args.nhead_stride_dq;
    a.nhead_stride_dk   = args.nhead_stride_dk;
    a.nhead_stride_dv   = args.nhead_stride_dv;

    a.batch_stride_q    = args.batch_stride_q;
    a.batch_stride_k    = args.batch_stride_k;
    a.batch_stride_v    = args.batch_stride_v;
    a.batch_stride_o    = args.batch_stride_o;
    a.batch_stride_do   = args.batch_stride_do;
    a.batch_stride_lsed = args.batch_stride_lsed;
    a.batch_stride_dq   = args.batch_stride_dq;
    a.batch_stride_dk   = args.batch_stride_dk;
    a.batch_stride_dv   = args.batch_stride_dv;

    a.window_size_left  = args.window_size_left;
    a.window_size_right = args.window_size_right;
    a.p_drop            = 0.f;
    a.p_undrop          = 1.f;
    a.drop_seed_offset  = std::make_pair(uint64_t{0}, uint64_t{0});
    a.workspace_alloc   = workspace_alloc;

    return aiter::fmha_v3_bwd(a, stream_config) >= 0;
}

} // namespace flash
