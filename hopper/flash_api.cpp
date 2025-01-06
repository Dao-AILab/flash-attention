/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

// Include these 2 headers instead of torch/extension.h since we don't need all of the torch headers.
#include <torch/nn/functional.h>
#include <torch/version.h>  // For TORCH_VERSION* macros
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cutlass/numeric_types.h>

#include "flash.h"
#include "static_switch.h"
#include "tile_size.h"
#include "heuristics.h"

// Copied from https://github.com/pytorch/pytorch/commit/7931eee5c5ebcdf468bff4d308510b03355cd909
// This is so that we can pass in torch.dtype as a parameter to the function.
#if TORCH_VERSION_MAJOR < 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR < 4)

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace pybind11::detail {

    template <>
    struct type_caster<at::ScalarType> {
    public:
        // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
        PYBIND11_TYPE_CASTER(at::ScalarType, _("torch.dtype"));
        // PYBIND11_TYPE_CASTER defines a member field called value. at::ScalarType
        // cannot be default-initialized, we provide this constructor to explicitly
        // initialize that field. The value doesn't matter as it will be overwritten
        // after a successful call to load.
        type_caster() : value(at::kFloat) {}
        bool load(handle src, bool) {
            PyObject* obj = src.ptr();
            if (THPDtype_Check(obj)) {
                value = reinterpret_cast<THPDtype*>(obj)->scalar_type;
                return true;
            }
            return false;
        }
        static handle cast(
                           const at::ScalarType& src,
                           return_value_policy /* policy */,
                           handle /* parent */) {
            return Py_NewRef(torch::getTHPDtype(src));
        }
    };

} // namespace pybind11::detail

#endif

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

void set_params_fprop(Flash_fwd_params &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      const size_t d_rounded,
                      // device pointers
                      const at::Tensor q,
                      const at::Tensor k,
                      const at::Tensor v,
                      at::Tensor out,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      void *seqused_q,
                      void *seqused_k,
                      void *softmax_lse_d,
                      float p_dropout,
                      float softmax_scale,
                      int window_size_left,
                      int window_size_right,
                      const float softcap=0.f,
                      const int sm_margin=0) {

    // Reset the parameters
    params = {};

    params.is_bf16 = q.dtype() == torch::kBFloat16;
    params.is_e4m3 = q.dtype() == torch::kFloat8_e4m3fn;

    // Set the pointers and strides.
    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.v_ptr = v.data_ptr();
    // All stride are in elements, not bytes.
    params.q_row_stride = q.stride(-3);
    params.k_row_stride = k.stride(-3);
    params.v_row_stride = v.stride(-3);
    params.q_head_stride = q.stride(-2);
    params.k_head_stride = k.stride(-2);
    params.v_head_stride = v.stride(-2);
    params.v_dim_stride = v.stride(-1);
    params.o_ptr = out.data_ptr();
    params.o_row_stride = out.stride(-3);
    params.o_head_stride = out.stride(-2);

    if (cu_seqlens_q_d == nullptr) {
        params.q_batch_stride = q.stride(0);
        params.o_batch_stride = out.stride(0);
    }
    if (cu_seqlens_k_d == nullptr) {
        params.k_batch_stride = k.stride(0);
        params.v_batch_stride = v.stride(0);
    }

    params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q_d);
    params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k_d);
    params.seqused_q = static_cast<int *>(seqused_q);
    params.seqused_k = static_cast<int *>(seqused_k);

    // Softmax sum
    params.softmax_lse_ptr = softmax_lse_d;

    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.h_k = h_k;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.d = d;
    params.d_rounded = d_rounded;

    // Set the different scale values.
    params.scale_softmax = softmax_scale;
    params.softcap = softcap;

    // Set this to probability of keeping an element to simplify things.
    params.p_dropout = 1.f - p_dropout;
    // Convert p from float to int so we don't have to convert the random uint to float to compare.
    // [Minor] We want to round down since when we do the comparison we use <= instead of <
    // params.p_dropout_in_uint = uint32_t(std::floor(params.p_dropout * 4294967295.0));
    // params.p_dropout_in_uint16_t = uint16_t(std::floor(params.p_dropout * 65535.0));
    params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
    params.rp_dropout = 1.f / params.p_dropout;
    TORCH_CHECK(p_dropout < 1.f);
    #ifdef FLASHATTENTION_DISABLE_DROPOUT
        TORCH_CHECK(p_dropout == 0.0f, "This flash attention build does not support dropout.");
    #endif

    // Causal is the special case where window_size_right == 0 and window_size_left < 0.
    // Local is the more general case where window_size_right >= 0 or window_size_left >= 0.
    params.is_causal = window_size_left < 0 && window_size_right == 0;
    params.is_local = (window_size_left >= 0 || window_size_right >= 0) && !params.is_causal;

    // TODO: check this
    if (window_size_left < 0 && window_size_right >= 0) { window_size_left = seqlen_k - 1; }
    if (window_size_left >= 0 && window_size_right < 0) { window_size_right = seqlen_q - 1; }
    params.window_size_left = window_size_left;
    params.window_size_right = window_size_right;

    params.arch = at::cuda::getCurrentDeviceProperties()->major * 10 + at::cuda::getCurrentDeviceProperties()->minor;
    params.num_sm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount - sm_margin;

    #ifdef FLASHATTENTION_DISABLE_LOCAL
        TORCH_CHECK(!params.is_local, "This flash attention build does not support local attention.");
    #endif
}

void set_params_dgrad(Flash_bwd_params &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      const size_t d_rounded,
                      // device pointers
                      const at::Tensor q,
                      const at::Tensor k,
                      const at::Tensor v,
                      const at::Tensor out,
                      const at::Tensor dout,
                      at::Tensor dq,
                      at::Tensor dk,
                      at::Tensor dv,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      void *seqused_q,
                      void *seqused_k,
                      void *dq_accum_d,
                      void *dk_accum_d,
                      void *dv_accum_d,
                      void *softmax_lse_d,
                      void *dsoftmax_sum_d,
                      float p_dropout,
                      float softmax_scale,
                      int window_size_left,
                      int window_size_right,
                      const float softcap=0.f,
                      bool deterministic=false,
                      int const sm_margin=0) {

    set_params_fprop(params,
                     b, seqlen_q, seqlen_k, seqlen_q_rounded, seqlen_k_rounded, h, h_k, d, d_rounded,
                     q, k, v, out,
                     cu_seqlens_q_d,
                     cu_seqlens_k_d,
                     seqused_q,
                     seqused_k,
                     softmax_lse_d,
                     p_dropout,
                     softmax_scale,
                     window_size_left,
                     window_size_right,
                     softcap,
                     sm_margin);

    // Set the pointers and strides.
    params.do_ptr = dout.data_ptr();
    params.do_row_stride = dout.stride(-3);
    params.do_head_stride = dout.stride(-2);
    params.dq_ptr = dq.data_ptr();
    params.dk_ptr = dk.data_ptr();
    params.dv_ptr = dv.data_ptr();
    params.dq_row_stride = dq.stride(-3);
    params.dk_row_stride = dk.stride(-3);
    params.dv_row_stride = dv.stride(-3);
    params.dq_head_stride = dq.stride(-2);
    params.dk_head_stride = dk.stride(-2);
    params.dv_head_stride = dv.stride(-2);

    if (cu_seqlens_q_d == nullptr) {
        params.do_batch_stride = dout.stride(0);
        params.dq_batch_stride = dq.stride(0);
        params.dk_batch_stride = dk.stride(0);
        params.dv_batch_stride = dv.stride(0);
    }

    params.dq_accum_ptr = dq_accum_d;
    params.dk_accum_ptr = dk_accum_d;
    params.dv_accum_ptr = dv_accum_d;

    // Softmax sum
    params.dsoftmax_sum = dsoftmax_sum_d;

    params.deterministic = deterministic;
}

void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    // HEADDIM_SWITCH(params.d, [&] {
    //     run_mha_fwd_<cutlass::half_t, kHeadSize>(params, stream);
    // });
    TORCH_CHECK(params.num_splits >= 1);
    ARCH_SWITCH(params.arch, Arch, [&] {
        SPLIT_SWITCH(params.num_splits > 1, Split, [&] {
            PAGEDKV_SWITCH(params.page_table, PagedKV, [&] {
                PACKGQA_SWITCH(params.pack_gqa, PackGQA_, [&] {
                    // Always enable PackGQA for Sm8x or PagedKV or Split to reduce compilation
                    static constexpr bool PackGQA = PackGQA_ || Arch < 90 || PagedKV || Split;
                    SOFTCAP_SWITCH(params.softcap > 0.0, Has_softcap, [&] {
                        if (!params.is_e4m3) {
                            if (params.is_bf16) {
                                #ifndef FLASHATTENTION_DISABLE_HDIM64
                                if (params.d <= 64) { return run_mha_fwd_<Arch, cutlass::bfloat16_t, 64, Split, PagedKV, Has_softcap, PackGQA>(params, stream); }
                                #endif
                                #ifndef FLASHATTENTION_DISABLE_HDIM96
                                if (params.d <= 96) { return run_mha_fwd_<Arch, cutlass::bfloat16_t, 96, Split, PagedKV, Has_softcap, PackGQA>(params, stream); }
                                #endif
                                #ifndef FLASHATTENTION_DISABLE_HDIM128
                                if (params.d <= 128) { return run_mha_fwd_<Arch, cutlass::bfloat16_t, 128, Split, PagedKV, Has_softcap, PackGQA>(params, stream); }
                                #endif
                                #ifndef FLASHATTENTION_DISABLE_HDIM192
                                if (params.d <= 192) { return run_mha_fwd_<Arch, cutlass::bfloat16_t, 192, Split, PagedKV, Has_softcap, PackGQA>(params, stream); }
                                #endif
                                #ifndef FLASHATTENTION_DISABLE_HDIM256
                                if (params.d <= 256) { return run_mha_fwd_<Arch, cutlass::bfloat16_t, 256, Split, PagedKV, Has_softcap, PackGQA>(params, stream); }
                                #endif
                            } else {
                                #ifndef FLASHATTENTION_DISABLE_FP16
                                #ifndef FLASHATTENTION_DISABLE_HDIM64
                                if (params.d <= 64) { return run_mha_fwd_<Arch, cutlass::half_t, 64, Split, PagedKV, Has_softcap, PackGQA>(params, stream); }
                                #endif
                                #ifndef FLASHATTENTION_DISABLE_HDIM96
                                if (params.d <= 96) { return run_mha_fwd_<Arch, cutlass::half_t, 96, Split, PagedKV, Has_softcap, PackGQA>(params, stream); }
                                #endif
                                #ifndef FLASHATTENTION_DISABLE_HDIM128
                                if (params.d <= 128) { return run_mha_fwd_<Arch, cutlass::half_t, 128, Split, PagedKV, Has_softcap, PackGQA>(params, stream); }
                                #endif
                                #ifndef FLASHATTENTION_DISABLE_HDIM192
                                if (params.d <= 192) { return run_mha_fwd_<Arch, cutlass::half_t, 192, Split, PagedKV, Has_softcap, PackGQA>(params, stream); }
                                #endif
                                #ifndef FLASHATTENTION_DISABLE_HDIM256
                                if (params.d <= 256) { return run_mha_fwd_<Arch, cutlass::half_t, 256, Split, PagedKV, Has_softcap, PackGQA>(params, stream); }
                                #endif
                                #else
                                TORCH_CHECK(false, "This flash attention build does not support FP16.");
                                #endif
                            }
                        } else {
                            #ifndef FLASHATTENTION_DISABLE_FP8
                            #ifndef FLASHATTENTION_DISABLE_HDIM64
                            if (params.d <= 64) { return run_mha_fwd_<90, cutlass::float_e4m3_t, 64, Split, PagedKV, Has_softcap, PackGQA>(params, stream); }
                            #endif
                            #ifndef FLASHATTENTION_DISABLE_HDIM96
                            if (params.d <= 96) { return run_mha_fwd_<90, cutlass::float_e4m3_t, 96, Split, PagedKV, Has_softcap, PackGQA>(params, stream); }
                            #endif
                            #ifndef FLASHATTENTION_DISABLE_HDIM128
                            if (params.d <= 128) { return run_mha_fwd_<90, cutlass::float_e4m3_t, 128, Split, PagedKV, Has_softcap, PackGQA>(params, stream); }
                            #endif
                            #ifndef FLASHATTENTION_DISABLE_HDIM192
                            if (params.d <= 192) { return run_mha_fwd_<90, cutlass::float_e4m3_t, 192, Split, PagedKV, Has_softcap, PackGQA>(params, stream); }
                            #endif
                            #ifndef FLASHATTENTION_DISABLE_HDIM256
                            if (params.d <= 256) { return run_mha_fwd_<90, cutlass::float_e4m3_t, 256, Split, PagedKV, Has_softcap, PackGQA>(params, stream); }
                            #endif
                            #else
                            TORCH_CHECK(false, "This flash attention build does not support FP8.");
                            #endif
                        }
                    });
                });
            });
        });
    });
}

void run_mha_fwd_combine(Flash_fwd_params &params, cudaStream_t stream) {
    #ifndef FLASHATTENTION_DISABLE_SPLIT
    // If hdim is 96 or 192, it's faster to round them to 128 or 256 respectively
    // so that kBlockM is smaller and we have more parallelism.
    if (params.is_fp32) {
        if (params.d <= 64) {
            run_mha_fwd_combine_<float, float, 64>(params, stream);
        } else if (params.d <= 128) {
            run_mha_fwd_combine_<float, float, 128>(params, stream);
        } else {
            run_mha_fwd_combine_<float, float, 256>(params, stream);
        }
    } else if (params.is_bf16) {
        if (params.d <= 64) {
            run_mha_fwd_combine_<cutlass::bfloat16_t, float, 64>(params, stream);
        } else if (params.d <= 128) {
            run_mha_fwd_combine_<cutlass::bfloat16_t, float, 128>(params, stream);
        } else {
            run_mha_fwd_combine_<cutlass::bfloat16_t, float, 256>(params, stream);
        }
    } else {
        if (params.d <= 64) {
            run_mha_fwd_combine_<cutlass::half_t, float, 64>(params, stream);
        } else if (params.d <= 128) {
            run_mha_fwd_combine_<cutlass::half_t, float, 128>(params, stream);
        } else {
            run_mha_fwd_combine_<cutlass::half_t, float, 256>(params, stream);
        }
    }
    #else
    TORCH_CHECK(false, "This flash attention build does not support combine kernels.");
    #endif
}

inline bool get_pack_gqa(Flash_fwd_params const& params) {
    // Always enable PackGQA for Sm8x or PagedKV or Split to reduce compilation and binary size.
    // Has little effect on speed.
    if (params.arch < 90 || params.page_table || params.num_splits > 1) { return true; }
    #ifdef FLASHATTENTION_DISABLE_PACKGQA
    return false;
    #else
    // params.page_table must already be set
    if (params.h == params.h_k) { return false; }
    // This needs to match the kernel configs
    auto kBlockMN_kernel_args_sm90 = tile_size_fwd_sm90(params.d_rounded, params.is_causal, params.is_local, params.is_e4m3 ? 1 : 2 /*element_size*/, false /*v_colmajor*/, params.page_table, params.softcap > 0.f);
    int const kBlockM = std::get<0>(kBlockMN_kernel_args_sm90);
    return should_pack_gqa(params.cu_seqlens_q || params.seqused_q, params.seqlen_q, params.h / params.h_k, kBlockM);
    #endif
}

inline int get_num_splits(Flash_fwd_params const& params) {
    #ifdef FLASHATTENTION_DISABLE_SPLIT
    return 1;
    #else
    // Always enable PackGQA for Split
    // params.page_table must already be set
    // This needs to match the kernel configs
    bool varlen = params.cu_seqlens_q || params.cu_seqlens_k || params.seqused_q || params.seqused_k || params.leftpad_k;
    auto kBlockMN_kernel_args_sm90 = tile_size_fwd_sm90(params.d_rounded, params.is_causal, params.is_local, params.is_e4m3 ? 1 : 2 /*element_size*/, false /*v_colmajor*/, params.page_table, params.softcap > 0.f);
    // Strictly speaking we need to pass in (varlen && params.num_splits > 1) but num_splits
    // has not been set here. It's OK though because we might just underestimate kBlockN a bit
    auto kBlockMN_kernel_args_sm8x = tile_size_fwd_sm8x(params.arch == 86 || params.arch == 89, params.d_rounded, params.is_causal, params.is_local, params.is_e4m3 ? 1 : 2 /*element_size*/, params.page_table, varlen, params.softcap > 0.f, params.knew_ptr);
    int const kBlockM = params.arch >= 90 ? std::get<0>(kBlockMN_kernel_args_sm90) : std::get<0>(kBlockMN_kernel_args_sm8x);
    int const kBlockN = params.arch >= 90 ? std::get<1>(kBlockMN_kernel_args_sm90) : std::get<1>(kBlockMN_kernel_args_sm8x);
    int seqlen_q_packgqa = params.seqlen_q * (params.h / params.h_k);
    // If is_local, we're not going to load all of seqlen_k
    int const seqlen_k_loaded = !params.is_local
        ? params.seqlen_k
        : std::max(0, std::min(params.seqlen_k, params.window_size_right + params.window_size_left + 1 + kBlockM));
    int const num_n_blocks = (seqlen_k_loaded + kBlockN - 1) / kBlockN;
    int const num_m_blocks = (seqlen_q_packgqa + kBlockM - 1) / kBlockM;
    return num_splits_heuristic(params.b * (!params.pack_gqa ? params.h : params.h_k) * num_m_blocks, params.num_sm, num_n_blocks, 128);
    // return num_splits_heuristic(params.b * params.h_k * num_m_blocks, params.b * params.h_k,
    //                             params.num_sm, num_n_blocks, 128, params.d_rounded);
    #endif
}

inline int get_max_headdim() {
    #ifndef FLASHATTENTION_DISABLE_HDIM256
    return 256;
    #endif
    #ifndef FLASHATTENTION_DISABLE_HDIM192
    return 192;
    #endif
    #ifndef FLASHATTENTION_DISABLE_HDIM128
    return 128;
    #endif
    #ifndef FLASHATTENTION_DISABLE_HDIM96
    return 96;
    #endif
    #ifndef FLASHATTENTION_DISABLE_HDIM64
    return 64;
    #endif
    return 0;
}

inline int round_up_headdim(int head_size) {
    #ifndef FLASHATTENTION_DISABLE_HDIM64
    if (head_size <= 64) { return 64; }
    #endif
    #ifndef FLASHATTENTION_DISABLE_HDIM96
    if (head_size <= 96) { return 96; }
    #endif
    #ifndef FLASHATTENTION_DISABLE_HDIM128
    if (head_size <= 128) { return 128; }
    #endif
    #ifndef FLASHATTENTION_DISABLE_HDIM192
    if (head_size <= 192) { return 192; }
    #endif
    #ifndef FLASHATTENTION_DISABLE_HDIM256
    if (head_size <= 256) { return 256; }
    #endif
    return 256;
}

// b: batch_size
// b_k: batch_size_k
// s_q: seqlen_q
// s_k: seqlen_k
// s_k_new: seqlen_k_new
// h: num_heads
// h_k: num_heads_k
// d: head_size
std::vector<at::Tensor>
mha_fwd(at::Tensor &q,   // (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
        const at::Tensor &k,  // (b_k, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k or (num_pages, page_size, h_k, d) if there is page_table.
        const at::Tensor &v,  // (b_k, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k or (num_pages, page_size, h_k, d) if there is page_table.
        std::optional<const at::Tensor> &k_new_,  // (b, s_k_new, h_k, d) or (total_k_new, h_k, d) if there is cu_seqlens_k_new
        std::optional<const at::Tensor> &v_new_,  // (b, s_k_new, h_k, d) or (total_k_new, h_k, d) if there is cu_seqlens_k_new
        std::optional<at::Tensor> &out_,  // (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
        std::optional<const at::Tensor> &cu_seqlens_q_,  // b+1
        std::optional<const at::Tensor> &cu_seqlens_k_,  // b+1
        std::optional<const at::Tensor> &cu_seqlens_k_new_,  // b+1
        std::optional<const at::Tensor> &seqused_q_, // b. If given, only this many elements of each batch element's queries and outputs are used.
        std::optional<const at::Tensor> &seqused_k_, // b. If given, only this many elements of each batch element's keys are used.
        std::optional<int> max_seqlen_q_,
        // TODO: check if we need max_seqlen_k
        std::optional<int> max_seqlen_k_,
        std::optional<const at::Tensor> &page_table_, // (b_k, max_num_pages_per_seq)
        std::optional<const at::Tensor> &kv_batch_idx_, // b. indices to index into the KV cache
        std::optional<const at::Tensor> &leftpad_k_, // b
        std::optional<const at::Tensor> &rotary_cos_, // seqlen_ro x (rotary_dim / 2)
        std::optional<const at::Tensor> &rotary_sin_, // seqlen_ro x (rotary_dim / 2)
        std::optional<at::Tensor> &q_descale_,  // (b, h_k), not (b, h)
        std::optional<at::Tensor> &k_descale_,  // (b, h_k)
        std::optional<at::Tensor> &v_descale_,  // (b, h_k)
        float const softmax_scale,
        bool is_causal,
        int window_size_left,
        int window_size_right,
        int sink_token_length,
        float const softcap,
        bool const is_rotary_interleaved,   // if true, rotary combines indices 0 & 1, else indices 0 & rotary_dim / 2
        int num_splits,
        std::optional<bool> pack_gqa_,
        int const sm_margin
        ) {

    auto dprops = at::cuda::getCurrentDeviceProperties();
    bool is_sm8x = dprops->major >= 8;
    TORCH_CHECK(is_sm8x, "FlashAttention only supports Ampere GPUs or newer.");

    auto q_type = q.scalar_type();
    TORCH_CHECK(q_type == at::ScalarType::Half || q_type == at::ScalarType::BFloat16 || q_type == at::ScalarType::Float8_e4m3fn,
                "FlashAttention only supports fp16, bf16, and fp8_e4m3 data type");
    if (dprops->major < 9) {
        TORCH_CHECK(q_type == at::ScalarType::Half || q_type == at::ScalarType::BFloat16,
                    "FlashAttention on Ampere/Ada cards only supports fp16 and bf16 data type");
    }
    TORCH_CHECK(k.scalar_type() == q_type, "query and key must have the same dtype");
    TORCH_CHECK(v.scalar_type() == q_type, "query and value must have the same dtype");

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    at::Tensor page_table;
    const bool paged_KV = page_table_.has_value();
    if (paged_KV) {
        page_table = page_table_.value();
        CHECK_DEVICE(page_table);
        TORCH_CHECK(page_table.dtype() == torch::kInt32, "page_table must have dtype torch.int32");
        TORCH_CHECK(page_table.stride(-1) == 1, "page_table must have contiguous last dimension");
    }

    at::Tensor cu_seqlens_q;
    bool const is_varlen_q = cu_seqlens_q_.has_value();
    if (is_varlen_q) {
        cu_seqlens_q = cu_seqlens_q_.value();
        CHECK_DEVICE(cu_seqlens_q); CHECK_CONTIGUOUS(cu_seqlens_q);
        TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must have dtype torch.int32");
        TORCH_CHECK(max_seqlen_q_.has_value(), "max_seqlen_q must be provided if cu_seqlens_q is provided");
    }
    at::Tensor cu_seqlens_k;
    bool const is_varlen_k = cu_seqlens_k_.has_value();
    if (is_varlen_k) {
        cu_seqlens_k = cu_seqlens_k_.value();
        CHECK_DEVICE(cu_seqlens_k); CHECK_CONTIGUOUS(cu_seqlens_k);
        TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must have dtype torch.int32");
        TORCH_CHECK(max_seqlen_k_.has_value(), "max_seqlen_k must be provided if cu_seqlens_k is provided");
        TORCH_CHECK(!paged_KV, "If cu_seqlens_k is passed in, then page table is not supported");
        TORCH_CHECK(!kv_batch_idx_.has_value(), "If cu_seqlens_k is passed in, then page table is not supported");
    }
    // This is what we will template on
    bool const is_varlen = is_varlen_q || is_varlen_k || seqused_q_.has_value() || seqused_k_.has_value() || leftpad_k_.has_value();
    #ifdef FLASHATTENTION_DISABLE_VARLEN
        TORCH_CHECK(!is_varlen, "This flash attention build does not support varlen.");
    #endif

    auto const sizes = q.sizes();
    const int batch_size = !is_varlen_q ? sizes[0] : cu_seqlens_q.size(0) - 1;
    int seqlen_q = !is_varlen_q ? sizes[1] : max_seqlen_q_.value();
    int total_q = !is_varlen_q ? batch_size * sizes[1] : sizes[0];
    int num_heads = q.size(-2);
    int const head_size = q.size(-1);
    int const max_num_pages_per_seq = !paged_KV ? 0 : page_table.size(1);
    int const num_pages = !paged_KV ? 0 : k.size(0);
    int const page_size = !paged_KV ? 1 : k.size(1);
    int const seqlen_k = !is_varlen_k ? (!paged_KV ? k.size(1) : max_num_pages_per_seq * page_size) : max_seqlen_k_.value();
    int const total_k = !is_varlen_k ? batch_size * k.size(1) : k.size(0);
    int const num_heads_k = k.size(-2);
    int const batch_size_k = !paged_KV ? (!is_varlen_k ? k.size(0) : cu_seqlens_k.size(0) - 1) : page_table.size(0);
    if (!kv_batch_idx_.has_value()) {
        TORCH_CHECK(batch_size == batch_size_k, "batch_size must be equal to batch_size_k");
    }
    int const max_headdim = get_max_headdim();
    TORCH_CHECK(head_size <= max_headdim, "FlashAttention forward only supports head dimension at most " + std::to_string(max_headdim));
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    // This needs to go before kBlockM & kBlockN since we rely on the correct window_size and is_causal to set kBlockM
    // TODO: check this
    if (window_size_left >= seqlen_k - 1) { window_size_left = -1; }
    if (window_size_right >= seqlen_q - 1) { window_size_right = -1; }
    if (is_causal) { window_size_right = 0; }
    // There's a case where is_causal=false, window_size=(-1, 0). Then set_params_fprop will set params.is_causal=true.
    // If we don't have is_causal here matching params.is_causal, we might get the wrong kBlockM.
    is_causal = window_size_left < 0 && window_size_right == 0;

    if (!is_varlen_q) {
        CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);
    } else {
        CHECK_SHAPE(q, total_q, num_heads, head_size);
        CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
    }
    if (!paged_KV) {
        if (!is_varlen_k) {
            CHECK_SHAPE(k, batch_size_k, seqlen_k, num_heads_k, head_size);
            CHECK_SHAPE(v, batch_size_k, seqlen_k, num_heads_k, head_size);
        } else {
            CHECK_SHAPE(k, total_k, num_heads_k, head_size);
            CHECK_SHAPE(v, total_k, num_heads_k, head_size);
            CHECK_SHAPE(cu_seqlens_k, batch_size + 1);
        }
    } else {
        CHECK_SHAPE(k, num_pages, page_size, num_heads_k, head_size);
        CHECK_SHAPE(v, num_pages, page_size, num_heads_k, head_size);
        CHECK_SHAPE(page_table, batch_size_k, max_num_pages_per_seq);
    }

    if (seqused_q_.has_value()){
        auto seqused_q = seqused_q_.value();
        TORCH_CHECK(seqused_q.dtype() == torch::kInt32, "seqused_q must have dtype int32");
        CHECK_DEVICE(seqused_q); CHECK_CONTIGUOUS(seqused_q);
        CHECK_SHAPE(seqused_q, batch_size);
    }
    if (seqused_k_.has_value()) {
        auto seqused_k = seqused_k_.value();
        TORCH_CHECK(seqused_k.dtype() == torch::kInt32, "seqused_k must have dtype int32");
        CHECK_DEVICE(seqused_k); CHECK_CONTIGUOUS(seqused_k);
        CHECK_SHAPE(seqused_k, batch_size);
    }

    int const alignment = q_type == torch::kFloat8_e4m3fn ? 16 : 8;
    TORCH_CHECK(head_size % alignment == 0, "head_size should be a multiple of " + std::to_string(alignment));

    auto opts = q.options();
    auto out_type = q_type == at::ScalarType::Float8_e4m3fn ? at::ScalarType::BFloat16 : q_type;
    at::Tensor out;
    if (out_.has_value()) {
        out = out_.value();
        TORCH_CHECK(out.scalar_type() == out_type, "For FP16/BF16 input, output must have the same dtype as inputs. For FP8 input, output must have dtype BF16");
        CHECK_DEVICE(out);
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        if (!is_varlen_q) {
            CHECK_SHAPE(out, batch_size, seqlen_q, num_heads, head_size);
        } else {
            CHECK_SHAPE(out, total_q, num_heads, head_size);
        }
    } else {
        out = torch::empty_like(q, opts.dtype(out_type));
    }

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    int const head_size_rounded = round_up_headdim(head_size);
    int const seqlen_q_rounded = round_multiple(seqlen_q, 128);
    int const seqlen_k_rounded = round_multiple(seqlen_k, 128);

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    at::Tensor softmax_lse;
    if (!is_varlen_q) {
        softmax_lse = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
    } else {
        softmax_lse = torch::empty({num_heads, total_q}, opts.dtype(at::kFloat));
    }

    Flash_fwd_params params;
    set_params_fprop(params,
                     batch_size,
                     seqlen_q, seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q, k, v, out,
                     !is_varlen_q ? nullptr : cu_seqlens_q.data_ptr(),
                     !is_varlen_k ? nullptr : cu_seqlens_k.data_ptr(),
                     seqused_q_.has_value() ? seqused_q_.value().data_ptr() : nullptr,
                     seqused_k_.has_value() ? seqused_k_.value().data_ptr() : nullptr,
                     softmax_lse.data_ptr(),
                     /*p_dropout=*/0.f,
                     softmax_scale,
                     window_size_left,
                     window_size_right,
                     softcap,
                     sm_margin);
    params.total_q = total_q;
    params.total_k = total_k;
    params.sink_token_length = sink_token_length;
    params.b_k = batch_size_k;

    if (paged_KV) {
        params.page_table = page_table.data_ptr<int>();
        params.page_table_batch_stride = page_table.stride(0);
    }
    params.page_size = page_size;
    params.num_pages = num_pages;

    params.num_splits = num_splits <= 0 ? get_num_splits(params) : num_splits;
    params.pack_gqa = pack_gqa_.has_value() ? pack_gqa_.value() : get_pack_gqa(params);

    if (k_new_.has_value()) {
        at::Tensor k_new, v_new;
        TORCH_CHECK(v_new_.has_value(), "If k_new is supplied, v_new must also be passed in");
        TORCH_CHECK(seqused_k_.has_value(), "If k_new is supplied, seqlens_k must also be passed in");
        TORCH_CHECK(seqlen_q <= seqlen_k, "If k_new is supplied, it must have seqlen <= the seqlen of the KV cache");
        at::Tensor cu_seqlens_k_new;
        bool const is_varlen_k_new = cu_seqlens_k_new_.has_value();
        if (is_varlen_k_new) {
            cu_seqlens_k_new = cu_seqlens_k_new_.value();
            CHECK_DEVICE(cu_seqlens_k_new); CHECK_CONTIGUOUS(cu_seqlens_k_new);
            TORCH_CHECK(cu_seqlens_k_new.dtype() == torch::kInt32, "cu_seqlens_k_new must have dtype torch.int32");
        }
        k_new = k_new_.value();
        v_new = v_new_.value();
        TORCH_CHECK(k_new.dtype() == q_type, "k_new must have the same dtype as query");
        TORCH_CHECK(v_new.dtype() == q_type, "v_new must have the same dtype as query");
        CHECK_DEVICE(k_new); CHECK_DEVICE(v_new);
        TORCH_CHECK(k_new.stride(-1) == 1, "k_new tensor must have contiguous last dimension");
        TORCH_CHECK(v_new.stride(-1) == 1, "v_new tensor must have contiguous last dimension");
        // We don't need max_seqlen_k_new, so seqlen_k_new can be whatever when is_varlen_k_new
        int seqlen_k_new = !is_varlen_k_new ? k_new.size(1) : 0;
        int total_k_new = !is_varlen_k_new ? batch_size * k_new.size(1): k_new.size(0);
        if (!is_varlen_k_new) {
            CHECK_SHAPE(k_new, batch_size, seqlen_k_new, num_heads_k, head_size);
            CHECK_SHAPE(v_new, batch_size, seqlen_k_new, num_heads_k, head_size);
        } else {
            CHECK_SHAPE(k_new, total_k_new, num_heads_k, head_size);
            CHECK_SHAPE(v_new, total_k_new, num_heads_k, head_size);
            CHECK_SHAPE(cu_seqlens_k_new, batch_size + 1);
        }
        params.seqlen_knew = seqlen_k_new;
        params.total_knew = total_k_new;
        params.knew_ptr = k_new.data_ptr();
        params.vnew_ptr = v_new.data_ptr();
        // All stride are in elements, not bytes.
        params.knew_row_stride = k_new.stride(-3);
        params.vnew_row_stride = v_new.stride(-3);
        params.knew_head_stride = k_new.stride(-2);
        params.vnew_head_stride = v_new.stride(-2);
        if (!is_varlen_k_new) {
            params.knew_batch_stride = k_new.stride(0);
            params.vnew_batch_stride = v_new.stride(0);
        }
        if (is_varlen_k_new) {
            params.cu_seqlens_knew = static_cast<int*>(cu_seqlens_k_new.data_ptr());
        }
    }

    if (leftpad_k_.has_value()) {
        auto leftpad_k = leftpad_k_.value();
        TORCH_CHECK(leftpad_k.dtype() == torch::kInt32, "leftpad_k must have dtype int32");
        CHECK_DEVICE(leftpad_k); CHECK_CONTIGUOUS(leftpad_k);
        CHECK_SHAPE(leftpad_k, batch_size);
        params.leftpad_k = static_cast<int *>(leftpad_k.data_ptr());
    }

    if (rotary_cos_.has_value()) {
        TORCH_CHECK(k_new_.has_value(), "If rotary cos/sin are provided, new key / value to be appended to KV cache must also be provided");
        auto rotary_cos = rotary_cos_.value();
        CHECK_DEVICE(rotary_cos); CHECK_CONTIGUOUS(rotary_cos);
        params.rotary_dim = rotary_cos.size(1) * 2;
        TORCH_CHECK(params.rotary_dim <= head_size, "rotary_dim must be <= headdim");
        TORCH_CHECK(params.rotary_dim % 16 == 0, "Only rotary dimensions divisible by 16 are currently supported");
        const int seqlen_ro = rotary_cos.size(0);
        if (paged_KV) {
            TORCH_CHECK(seqlen_ro >= seqlen_k, "cos/sin seqlen must be at least the seqlen of KV cache");
        }
        CHECK_SHAPE(rotary_cos, seqlen_ro, params.rotary_dim / 2);
        TORCH_CHECK(rotary_cos.scalar_type() == q_type, "rotary_cos must have the same dtype as query");

        TORCH_CHECK(rotary_sin_.has_value(), "If rotary cos is provided, rotary sin must also be provided");
        auto rotary_sin = rotary_sin_.value();
        CHECK_DEVICE(rotary_sin); CHECK_CONTIGUOUS(rotary_sin);
        CHECK_SHAPE(rotary_sin, seqlen_ro, params.rotary_dim / 2);
        TORCH_CHECK(rotary_sin.scalar_type() == q_type, "rotary_cos must have the same dtype as query");
        params.rotary_cos_ptr = rotary_cos.data_ptr();
        params.rotary_sin_ptr = rotary_sin.data_ptr();
        params.is_rotary_interleaved = is_rotary_interleaved;
    } else {
        params.rotary_dim = 0;
    }

    if (kv_batch_idx_.has_value()) {
        auto kv_batch_idx = kv_batch_idx_.value();
        CHECK_DEVICE(kv_batch_idx); CHECK_CONTIGUOUS(kv_batch_idx);
        TORCH_CHECK(kv_batch_idx.scalar_type() == torch::kInt32, "kv_batch_idx must have dtype int32");
        params.kv_batch_idx = reinterpret_cast<int *>(kv_batch_idx.data_ptr());
    }

    at::Tensor out_accum, softmax_lse_accum;
    auto outaccum_type = at::ScalarType::Float;
    if (params.num_splits > 1) {
        TORCH_CHECK(params.num_splits <= 256, "num_splits > 256 not supported");
        if (!is_varlen_q) {
            out_accum = torch::empty({params.num_splits, batch_size, num_heads, seqlen_q, head_size}, opts.dtype(outaccum_type));
            softmax_lse_accum = torch::empty({params.num_splits, batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
            params.oaccum_batch_stride = out_accum.stride(1);
            params.lseaccum_batch_stride = softmax_lse_accum.stride(1);
        } else {
            out_accum = torch::empty({params.num_splits, num_heads, total_q, head_size}, opts.dtype(outaccum_type));
            softmax_lse_accum = torch::empty({params.num_splits, num_heads, total_q}, opts.dtype(at::kFloat));
        }
        params.is_fp32 = false;
        params.oaccum_ptr = out_accum.data_ptr();
        params.softmax_lseaccum_ptr = softmax_lse_accum.data_ptr();
        params.oaccum_split_stride = out_accum.stride(0);
        params.oaccum_row_stride = out_accum.stride(-2);
        params.oaccum_head_stride = out_accum.stride(-3);
        params.lseaccum_split_stride = softmax_lse_accum.stride(0);
        params.lseaccum_head_stride = softmax_lse_accum.stride(-2);
    }

    at::Tensor tile_count_semaphore;
    // We don't use the persistent scheduler if Split and not Varlen
    bool const persistent_scheduler = params.arch >= 90
        ? (((params.is_causal || params.is_local) && (params.num_splits == 1)) || is_varlen)
        : ((params.is_causal && !is_varlen) || (is_varlen && params.num_splits > 1));
    if (persistent_scheduler) {
        tile_count_semaphore = torch::zeros({1}, opts.dtype(torch::kInt32));
        params.tile_count_semaphore = tile_count_semaphore.data_ptr<int>();
    } else {
        params.tile_count_semaphore = nullptr;
    }

    if (q_type == at::ScalarType::Float8_e4m3fn) {
        if (q_descale_.has_value()) {
            auto q_descale = q_descale_.value();
            CHECK_DEVICE(q_descale);
            CHECK_SHAPE(q_descale, batch_size, num_heads_k);
            params.q_descale_ptr = q_descale.data_ptr<float>();
            params.q_descale_batch_stride = q_descale.stride(0);
            params.q_descale_head_stride = q_descale.stride(1);
        } else {
            params.q_descale_ptr = nullptr;
        }
        if (k_descale_.has_value()) {
            auto k_descale = k_descale_.value();
            CHECK_DEVICE(k_descale);
            CHECK_SHAPE(k_descale, batch_size, num_heads_k);
            params.k_descale_ptr = k_descale.data_ptr<float>();
            params.k_descale_batch_stride = k_descale.stride(0);
            params.k_descale_head_stride = k_descale.stride(1);
        } else {
            params.k_descale_ptr = nullptr;
        }
        if (v_descale_.has_value()) {
            auto v_descale = v_descale_.value();
            CHECK_DEVICE(v_descale);
            CHECK_SHAPE(v_descale, batch_size, num_heads_k);
            params.v_descale_ptr = v_descale.data_ptr<float>();
            params.v_descale_batch_stride = v_descale.stride(0);
            params.v_descale_head_stride = v_descale.stride(1);
        } else {
            params.v_descale_ptr = nullptr;
        }
    }

    #ifdef FLASHATTENTION_DISABLE_LOCAL
    TORCH_CHECK(!params.is_local, "This flash attention build does not support local attention.");
    #endif
    #ifdef FLASHATTENTION_DISABLE_SOFTCAP
    TORCH_CHECK(params.softcap == 0.0, "This flash attention build does not support tanh softcapping.");
    #endif
    #ifdef FLASHATTENTION_DISABLE_SPLIT
    TORCH_CHECK(params.num_splits == 1, "This flash attention build does not support splits.");
    #endif
    #ifdef FLASHATTENTION_DISABLE_PACKGQA
    TORCH_CHECK(!params.pack_gqa || params.arch < 90 || params.page_table || params.num_splits > 1, "This flash attention build does not support pack_gqa.");
    #endif
    #ifdef FLASHATTENTION_DISABLE_PAGEDKV
    TORCH_CHECK(!paged_KV, "This flash attention build does not support paged KV.");
    #endif
    #ifdef FLASHATTENTION_DISABLE_APPENDKV
    TORCH_CHECK(!k_new_.has_value(), "This flash attention build does not support appending KV.");
    #endif

    if (total_q > 0 && (total_k + params.total_knew) > 0 && num_heads_k > 0) {
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        run_mha_fwd(params, stream);
        if (params.num_splits > 1) {
            if (out_type == at::ScalarType::BFloat16) {
                // Since we want output in BF16. Otherwise fwd_combine will output to FP16
                params.is_bf16 = true;
            }
            // Unless there's seqused_q, for the purpose of attn_combine, we can just treat it as batch=1
            // and seqlen = total_q, and don't need to dispatch to Varlen there.
            // if (is_varlen_q && !seqused_q_.has_value()) {
            if (is_varlen_q) {
                params.b = 1;
                params.seqlen_q = total_q;
            }
            run_mha_fwd_combine(params, stream);
        }
    } else if (total_q > 0 && num_heads_k > 0) {
        // If seqlen_k == 0, then we have an empty tensor. We need to set the output to 0.
        out.zero_();
        softmax_lse.fill_(std::numeric_limits<float>::infinity());
    }

    // return {out, softmax_lse};
    return {out, softmax_lse, out_accum, softmax_lse_accum};
}

void run_mha_bwd(Flash_bwd_params &params, cudaStream_t stream) {
    #ifndef FLASHATTENTION_DISABLE_BACKWARD
        // FP16_SWITCH(!params.is_bf16, [&] {
        //     HEADDIM_SWITCH(params.d, [&] {
        //         run_mha_bwd_<elem_type, kHeadDim>(params, stream);
        //     });
        // });
    ARCH_SWITCH(params.arch, Arch, [&] {
        SOFTCAP_SWITCH(params.softcap > 0.f, Has_softcap, [&] {
            if (!params.is_bf16) {
                #ifndef FLASHATTENTION_DISABLE_FP16
                #ifndef FLASHATTENTION_DISABLE_HDIM64
                if (params.d <= 64) { return run_mha_bwd_<Arch, cutlass::half_t, 64, Has_softcap>(params, stream); }
                #endif
                #ifndef FLASHATTENTION_DISABLE_HDIM96
                if (params.d <= 96) { return run_mha_bwd_<Arch, cutlass::half_t, 96, Has_softcap>(params, stream); }
                #endif
                #ifndef FLASHATTENTION_DISABLE_HDIM128
                if (params.d <= 128) { return run_mha_bwd_<Arch, cutlass::half_t, 128, Has_softcap>(params, stream); }
                #endif
                #ifndef FLASHATTENTION_DISABLE_HDIM192
                if (params.d <= 192) { return run_mha_bwd_<Arch, cutlass::half_t, 192, Has_softcap>(params, stream); }
                #endif
                #ifndef FLASHATTENTION_DISABLE_HDIM256
                if (params.d <= 256) { return run_mha_bwd_<Arch, cutlass::half_t, 256, Has_softcap>(params, stream); }
                #endif
                #else
                TORCH_CHECK(false, "This flash attention build does not support FP16.");
                #endif
            } else {
                #ifndef FLASHATTENTION_DISABLE_HDIM64
                if (params.d <= 64) { return run_mha_bwd_<Arch, cutlass::bfloat16_t, 64, Has_softcap>(params, stream); }
                #endif
                #ifndef FLASHATTENTION_DISABLE_HDIM96
                if (params.d <= 96) { return run_mha_bwd_<Arch, cutlass::bfloat16_t, 96, Has_softcap>(params, stream); }
                #endif
                #ifndef FLASHATTENTION_DISABLE_HDIM128
                if (params.d <= 128) { return run_mha_bwd_<Arch, cutlass::bfloat16_t, 128, Has_softcap>(params, stream); }
                #endif
                #ifndef FLASHATTENTION_DISABLE_HDIM192
                if (params.d <= 192) { return run_mha_bwd_<Arch, cutlass::bfloat16_t, 192, Has_softcap>(params, stream); }
                #endif
                #ifndef FLASHATTENTION_DISABLE_HDIM256
                if (params.d <= 256) { return run_mha_bwd_<Arch, cutlass::bfloat16_t, 256, Has_softcap>(params, stream); }
                #endif
            }
        });
    });
    #endif
}


// b: batch_size
// s_q: seqlen_q
// s_k: seqlen_k
// h: num_heads
// h_k: num_heads_k
// d: head_size
std::vector<at::Tensor> mha_bwd(
    const at::Tensor &dout,  // (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
    const at::Tensor &q,     // (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
    const at::Tensor &k,     // (b, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k
    const at::Tensor &v,     // (b, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k
    const at::Tensor &out,   // (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
    const at::Tensor &softmax_lse,    // (b, h, s_q) or (h, total_q) if there is cu_seqlens_q
    std::optional<at::Tensor> &dq_,   // (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
    std::optional<at::Tensor> &dk_,   // (b, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k
    std::optional<at::Tensor> &dv_,   // (b, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k
    std::optional<const at::Tensor> &cu_seqlens_q_,   // b+1
    std::optional<const at::Tensor> &cu_seqlens_k_,   // b+1
    std::optional<const at::Tensor> &seqused_q_, // b. If given, only this many elements of each batch element's queries and outputs are used.
    std::optional<const at::Tensor> &seqused_k_, // b. If given, only this many elements of each batch element's keys are used.
    std::optional<int> max_seqlen_q_,
    std::optional<int> max_seqlen_k_,
    float const softmax_scale,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    int const sink_token_length,
    float const softcap,
    bool const deterministic,
    int const sm_margin) {

    #ifdef FLASHATTENTION_DISABLE_BACKWARD
        TORCH_CHECK(false, "This flash attention build does not support backward.");
    #endif

    auto dprops = at::cuda::getCurrentDeviceProperties();
    bool is_sm8x = dprops->major >= 8;
    TORCH_CHECK(is_sm8x, "FlashAttention only supports Ampere GPUs or newer.");

    auto q_type = q.dtype();
    TORCH_CHECK(q_type == torch::kFloat16 || q_type == torch::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");
    TORCH_CHECK(k.dtype() == q_type, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_type, "query and value must have the same dtype");
    TORCH_CHECK(out.dtype() == q_type, "query and out must have the same dtype");
    TORCH_CHECK(dout.dtype() == q_type, "query and dout must have the same dtype");

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
    CHECK_DEVICE(out); CHECK_DEVICE(dout); CHECK_DEVICE(softmax_lse);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(out.stride(-1) == 1, "out tensor must have contiguous last dimension");
    TORCH_CHECK(dout.stride(-1) == 1, "dout tensor must have contiguous last dimension");

    at::Tensor cu_seqlens_q;
    bool const is_varlen_q = cu_seqlens_q_.has_value();
    if (is_varlen_q) {
        cu_seqlens_q = cu_seqlens_q_.value();
        CHECK_DEVICE(cu_seqlens_q); CHECK_CONTIGUOUS(cu_seqlens_q);
        TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must have dtype torch.int32");
        TORCH_CHECK(max_seqlen_q_.has_value(), "max_seqlen_q must be provided if cu_seqlens_q is provided");
    }
    at::Tensor cu_seqlens_k;
    bool const is_varlen_k = cu_seqlens_k_.has_value();
    if (is_varlen_k) {
        cu_seqlens_k = cu_seqlens_k_.value();
        CHECK_DEVICE(cu_seqlens_k); CHECK_CONTIGUOUS(cu_seqlens_k);
        TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must have dtype torch.int32");
        TORCH_CHECK(max_seqlen_k_.has_value(), "max_seqlen_k must be provided if cu_seqlens_k is provided");
    }
    // This is what we will template on
    bool const is_varlen = is_varlen_q || is_varlen_k || seqused_q_.has_value() || seqused_k_.has_value();
    #ifdef FLASHATTENTION_DISABLE_VARLEN
        TORCH_CHECK(!is_varlen, "This flash attention build does not support varlen.");
    #endif

    auto const sizes = q.sizes();
    int const batch_size = !is_varlen_q ? sizes[0] : cu_seqlens_q.size(0) - 1;
    int const seqlen_q = !is_varlen_q ? sizes[1] : max_seqlen_q_.value();
    int const total_q = !is_varlen_q ? batch_size * sizes[1] : sizes[0];
    int const num_heads = q.size(-2);
    int const head_size = q.size(-1);
    int const seqlen_k = !is_varlen_k ? k.size(1) : max_seqlen_k_.value();
    int const total_k = !is_varlen_k ? batch_size * k.size(1) : k.size(0);
    int const num_heads_k = k.size(-2);
    TORCH_CHECK(head_size % 8 == 0, "head_size should be a multiple of 8");
    int const max_headdim = get_max_headdim();
    TORCH_CHECK(head_size <= max_headdim, "FlashAttention forward only supports head dimension at most " + std::to_string(max_headdim));
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    // This needs to go before kBlockM & kBlockN since we rely on the correct window_size and is_causal to set kBlockM
    if (window_size_left >= seqlen_k - 1) { window_size_left = -1; }
    if (window_size_right >= seqlen_q - 1) { window_size_right = -1; }
    if (is_causal) { window_size_right = 0; }
    // There's a case where is_causal=false, window_size=(-1, 0). Then set_params_bprop will set params.is_causal=true.
    // If we don't have is_causal here matching params.is_causal, we might get the wrong kBlockM (and cause IMA).
    is_causal = window_size_left < 0 && window_size_right == 0;

    int const arch = at::cuda::getCurrentDeviceProperties()->major * 10 + at::cuda::getCurrentDeviceProperties()->minor;
    int const head_size_rounded = round_up_headdim(head_size);
    // Very important that these match the kernel configs
    bool const is_local = (window_size_left >= 0 || window_size_right >= 0) && !is_causal;
    int const kBlockM_sm90 = head_size_rounded <= 64 ? (is_causal && softcap > 0.0 ? 96 : 128)
        : (head_size_rounded <= 96 ? 64
           : (head_size_rounded <= 128 ? (is_causal || is_local || softcap > 0.0 ? 64 : 80)
              : 64));
    int const kBlockM_sm80 = head_size_rounded <= 64 ? 128 : 64;
    int const kBlockM_sm86 = head_size_rounded <= 192 ? 64 : 32;
    int const kBlockM = arch >= 90 ? kBlockM_sm90 : (arch == 86 || arch == 89 ? kBlockM_sm86 : kBlockM_sm80);
    int const kBlockN_sm90 = head_size_rounded <= 128
        ? 128
        : (head_size_rounded <= 192 ? 96 : 80);
    int const kBlockN_sm80 = head_size_rounded <= 128
        ? 128
        : (head_size_rounded <= 192 ? 80 : 64);
    int const kBlockN_sm86 = head_size_rounded <= 64 ? 128
        : (head_size_rounded <= 96 ? 128
           : (head_size_rounded <= 128 ? 96
              : (head_size_rounded <= 192 ? 64 : 64)));
    int const kBlockN = arch >= 90 ? kBlockN_sm90 : (arch == 86 || arch == 89 ? kBlockN_sm86 : kBlockN_sm80);
    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    int const seqlen_q_rounded = round_multiple(seqlen_q, kBlockM);
    int const seqlen_k_rounded = round_multiple(seqlen_k, kBlockN);
    int const total_q_padded_rounded = round_multiple(total_q + batch_size * kBlockM, kBlockM);
    int const total_k_padded_rounded = round_multiple(total_k + batch_size * kBlockN, kBlockN);

    if (!is_varlen_q) {
        CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);
        CHECK_SHAPE(out, batch_size, seqlen_q, num_heads, head_size);
        CHECK_SHAPE(dout, batch_size, seqlen_q, num_heads, head_size);
    } else {
        CHECK_SHAPE(q, total_q, num_heads, head_size);
        CHECK_SHAPE(out, total_q, num_heads, head_size);
        CHECK_SHAPE(dout, total_q, num_heads, head_size);
        CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
    }
    if (!is_varlen_k) {
        CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size);
        CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size);
    } else {
        CHECK_SHAPE(k, total_k, num_heads_k, head_size);
        CHECK_SHAPE(v, total_k, num_heads_k, head_size);
        CHECK_SHAPE(cu_seqlens_k, batch_size + 1);
    }

    if (seqused_q_.has_value()){
        auto seqused_q = seqused_q_.value();
        TORCH_CHECK(seqused_q.dtype() == torch::kInt32, "seqused_q must have dtype int32");
        CHECK_DEVICE(seqused_q); CHECK_CONTIGUOUS(seqused_q);
        CHECK_SHAPE(seqused_q, batch_size);
    }
    if (seqused_k_.has_value()){
        auto seqused_k = seqused_k_.value();
        TORCH_CHECK(seqused_k.dtype() == torch::kInt32, "seqused_k must have dtype int32");
        CHECK_DEVICE(seqused_k); CHECK_CONTIGUOUS(seqused_k);
        CHECK_SHAPE(seqused_k, batch_size);
    }

    at::Tensor dq, dk, dv;
    if (dq_.has_value()) {
        dq = dq_.value();
        TORCH_CHECK(dq.dtype() == q_type, "dq must have the same dtype as q");
        CHECK_DEVICE(dq);
        TORCH_CHECK(dq.stride(-1) == 1, "dq must have contiguous last dimension");
        if (!is_varlen_q) {
            CHECK_SHAPE(dq, batch_size, seqlen_q, num_heads, head_size);
        } else {
            CHECK_SHAPE(dq, total_q, num_heads, head_size);
        }
    } else {
        dq = torch::empty_like(q);
    }
    if (dk_.has_value()) {
        dk = dk_.value();
        TORCH_CHECK(dk.dtype() == q_type, "dk must have the same dtype as q");
        CHECK_DEVICE(dk);
        TORCH_CHECK(dk.stride(-1) == 1, "dk must have contiguous last dimension");
        if (!is_varlen_k) {
            CHECK_SHAPE(dk, batch_size, seqlen_k, num_heads_k, head_size);
        } else {
            CHECK_SHAPE(dk, total_k, num_heads_k, head_size);
        }
    } else {
        dk = torch::empty_like(k);
    }
    if (dv_.has_value()) {
        dv = dv_.value();
        TORCH_CHECK(dv.dtype() == q_type, "dv must have the same dtype as q");
        CHECK_DEVICE(dv);
        TORCH_CHECK(dv.stride(-1) == 1, "dv must have contiguous last dimension");
        if (!is_varlen_k) {
            CHECK_SHAPE(dv, batch_size, seqlen_k, num_heads_k, head_size);
        } else {
            CHECK_SHAPE(dv, total_k, num_heads_k, head_size);
        }
    } else {
        dv = torch::empty_like(v);
    }

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    auto opts = q.options();
    // Need softmax_d to have total_q_padded_rounded since we want its address to be aligned by 16/8 bytes for TMA / LDG.64
    at::Tensor softmax_d, softmax_lse_log2;
    if (!is_varlen) {
        // Need softmax_d to have seqlen_q_rounded since we want its address to be aligned by 16/8 bytes for TMA / LDG.64
        softmax_d = torch::empty({batch_size, num_heads, seqlen_q_rounded}, opts.dtype(at::kFloat));
        softmax_lse_log2 = torch::empty({batch_size, num_heads, seqlen_q_rounded}, opts.dtype(at::kFloat));
    } else {
        softmax_d = torch::empty({num_heads, total_q_padded_rounded}, opts.dtype(at::kFloat));
        softmax_lse_log2 = torch::empty({num_heads, total_q_padded_rounded}, opts.dtype(at::kFloat));
    }
    at::Tensor dq_accum, dk_accum, dv_accum;
    if (!is_varlen) {
        dq_accum = torch::empty({batch_size, num_heads, seqlen_q_rounded * head_size_rounded}, opts.dtype(at::kFloat));
    } else {
        dq_accum = torch::empty({num_heads, total_q_padded_rounded * head_size_rounded}, opts.dtype(at::kFloat));
    }
    if (num_heads_k != num_heads) {  // MQA / GQA
        if (!is_varlen) {
            dk_accum = torch::zeros({batch_size, num_heads_k, seqlen_k_rounded * head_size_rounded}, opts.dtype(at::kFloat));
            dv_accum = torch::zeros({batch_size, num_heads_k, seqlen_k_rounded * head_size_rounded}, opts.dtype(at::kFloat));
        } else {
            dk_accum = torch::zeros({num_heads_k, total_k_padded_rounded, head_size_rounded}, opts.dtype(at::kFloat));
            dv_accum = torch::zeros({num_heads_k, total_k_padded_rounded, head_size_rounded}, opts.dtype(at::kFloat));
        }
    }

    Flash_bwd_params params;
    set_params_dgrad(params,
                     batch_size,
                     seqlen_q, seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q, k, v, out,
                     dout, dq, dk, dv,
                     !is_varlen_q ? nullptr : cu_seqlens_q.data_ptr(),
                     !is_varlen_k ? nullptr : cu_seqlens_k.data_ptr(),
                     seqused_q_.has_value() ? seqused_q_.value().data_ptr() : nullptr,
                     seqused_k_.has_value() ? seqused_k_.value().data_ptr() : nullptr,
                     dq_accum.data_ptr(),
                     num_heads_k != num_heads ? dk_accum.data_ptr() : nullptr,
                     num_heads_k != num_heads ? dv_accum.data_ptr() : nullptr,
                     softmax_lse.data_ptr(),
                     softmax_d.data_ptr(),
                     /*p_dropout=*/0.f,
                     softmax_scale,
                     window_size_left,
                     window_size_right,
                     softcap,
                     deterministic,
                     sm_margin);
    params.total_q = total_q;
    params.total_k = total_k;
    params.softmax_lse_log2_ptr = softmax_lse_log2.data_ptr();
    params.sink_token_length = sink_token_length;

    // auto tile_count_semaphore = (params.is_causal || params.is_local) ? torch::zeros({1}, opts.dtype(torch::kInt32)) : torch::empty({1}, opts.dtype(torch::kInt32));
    // params.tile_count_semaphore = tile_count_semaphore.data_ptr<int>();
    // Will be zero'ed out in the backward preprocess kernel
    at::Tensor dq_semaphore = torch::empty({(seqlen_q + kBlockM - 1) / kBlockM, batch_size, num_heads}, opts.dtype(torch::kInt32));
    params.dq_semaphore = dq_semaphore.data_ptr<int>();
    if (num_heads_k != num_heads && params.deterministic) {
        // TODO: do we need to zero them out?
        at::Tensor dk_semaphore = torch::empty({(seqlen_k + kBlockN - 1) / kBlockN, batch_size, num_heads_k}, opts.dtype(torch::kInt32));
        at::Tensor dv_semaphore = torch::empty({(seqlen_k + kBlockN - 1) / kBlockN, batch_size, num_heads_k}, opts.dtype(torch::kInt32));
        params.dk_semaphore = dk_semaphore.data_ptr<int>();
        params.dv_semaphore = dv_semaphore.data_ptr<int>();
    }

    #ifdef FLASHATTENTION_DISABLE_LOCAL
    TORCH_CHECK(!params.is_local, "This flash attention build does not support local attention.");
    #endif
    #ifdef FLASHATTENTION_DISABLE_SOFTCAP
    TORCH_CHECK(params.softcap == 0.0, "This flash attention build does not support tanh softcapping.");
    #endif

    if (total_q > 0 && total_k > 0 && num_heads_k > 0) {
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        run_mha_bwd(params, stream);
    } else if (total_k > 0 && num_heads_k > 0) {
        // If seqlen_q == 0, then we have an empty tensor. We need to set the output to 0.
        dk.zero_();
        dv.zero_();
        softmax_d.zero_();
    } else if (total_q > 0 && num_heads_k > 0) {
        dq.zero_();
        softmax_d.zero_();
    }

    return { dq, dk, dv, softmax_d, softmax_lse_log2, dq_accum, dk_accum, dv_accum };
}

std::vector<at::Tensor>
mha_combine(const at::Tensor &out_partial,         // num_splits x batch_size x seqlen x num_heads x head_size
            const at::Tensor &lse_partial,         // num_splits x batch_size x seqlen x num_heads
            std::optional<at::Tensor> out_,        // batch_size x seqlen x num_heads x head_size
            std::optional<at::ScalarType> out_dtype_
            ) {

    auto dprops = at::cuda::getCurrentDeviceProperties();
    bool is_sm8x = dprops->major >= 8;
    TORCH_CHECK(is_sm8x, "Attention combine function only supports Ampere GPUs or newer.");

    auto out_partial_type = out_partial.scalar_type();
    TORCH_CHECK(out_partial_type == at::ScalarType::Float, "Attention combine function only support fp32 data type");
    TORCH_CHECK(lse_partial.scalar_type() == at::ScalarType::Float, "Attention combine function only support fp32 data type");

    CHECK_DEVICE(out_partial); CHECK_DEVICE(lse_partial);

    TORCH_CHECK(out_partial.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(lse_partial.stride(-2) == 1, "LSE tensor must be contiguous in the seqlen dimension");

    const auto sizes = out_partial.sizes();

    const int num_splits = sizes[0];
    const int batch_size = sizes[1];
    const int seqlen = sizes[2];
    const int num_heads = sizes[3];
    const int head_size_og = sizes[4];
    TORCH_CHECK(head_size_og <= 256, "FlashAttention combine only supports head dimension at most 256");
    TORCH_CHECK(num_splits <= 256, "FlashAttention combine only supports num_splits at most 256");

    CHECK_SHAPE(out_partial, num_splits, batch_size, seqlen, num_heads, head_size_og);
    CHECK_SHAPE(lse_partial, num_splits, batch_size, seqlen, num_heads);

    int const alignment = 4;
    at::Tensor out_partial_padded;
    auto pad = [](at::Tensor x, int alignment) {
        return x.size(-1) % alignment == 0 ? x : torch::nn::functional::pad(x, torch::nn::functional::PadFuncOptions({0, alignment - x.size(-1) % alignment}));
    };
    out_partial_padded = pad(out_partial, alignment);

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size = round_multiple(head_size_og, alignment);

    auto opts = out_partial.options();
    at::ScalarType out_type = out_dtype_.value_or(out_partial.scalar_type());
    TORCH_CHECK(out_type == at::ScalarType::Float || out_type == at::ScalarType::BFloat16 || out_type == at::ScalarType::Half, "Output type must be FP32, FP16 or BF16");
    at::Tensor out;
    if (out_.has_value()) {
        out = out_.value();
        TORCH_CHECK(out.scalar_type() == out_type);
        CHECK_DEVICE(out);
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        CHECK_SHAPE(out, batch_size, seqlen, num_heads, head_size_og);
        if (head_size_og % alignment != 0) {
            out = torch::empty({batch_size, seqlen, num_heads, head_size}, opts.dtype(out_type));
        }
    } else {
        out = torch::empty({batch_size, seqlen, num_heads, head_size}, opts.dtype(out_type));
    }

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)out_partial.get_device()};

    auto softmax_lse = torch::empty({batch_size, num_heads, seqlen}, opts.dtype(at::kFloat)).transpose(1, 2);

    Flash_fwd_params params {};  // Need to reset the params to set everything to zero
    params.is_fp32 = out_type == at::ScalarType::Float;
    params.is_bf16 = out_type == at::ScalarType::BFloat16;
    params.oaccum_ptr = out_partial_padded.data_ptr();
    params.softmax_lseaccum_ptr = lse_partial.data_ptr();
    params.o_ptr = out.data_ptr();
    params.softmax_lse_ptr = softmax_lse.data_ptr();
    params.b = batch_size;
    params.h = num_heads;
    params.seqlen_q = seqlen;
    params.d = head_size;
    params.num_splits = num_splits;
    params.oaccum_split_stride = out_partial_padded.stride(0);
    params.oaccum_row_stride = out_partial_padded.stride(2);
    params.oaccum_head_stride = out_partial_padded.stride(3);
    params.oaccum_batch_stride = out_partial_padded.stride(1);
    params.lseaccum_split_stride = lse_partial.stride(0);
    params.lseaccum_head_stride = lse_partial.stride(3);
    params.lseaccum_batch_stride = lse_partial.stride(1);
    params.o_row_stride = out.stride(1);
    params.o_head_stride = out.stride(2);
    params.o_batch_stride = out.stride(0);

    if (seqlen > 0 && batch_size > 0) {
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        run_mha_fwd_combine(params, stream);
    }

    at::Tensor out_padded = out;
    if (head_size_og % alignment != 0) {
        out = out.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
        // if (out_.has_value()) { out_.value().copy_(out); }
    }

    return {out, softmax_lse};
}

#ifndef FLASHATTENTION_DISABLE_PYBIND

#include <torch/python.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashAttention";
    m.def("fwd", &mha_fwd, "Forward pass");
    m.def("bwd", &mha_bwd, "Backward pass");
    m.def("fwd_combine", &mha_combine, "Combine partial attention outputs");
}

#endif