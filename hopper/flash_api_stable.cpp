/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#include <Python.h>

#include <cutlass/numeric_types.h>

#include "flash.h"
#include "static_switch.h"
#include "tile_size.h"
#include "heuristics.h"
#include "cuda_check.h"

#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/accelerator.h>

#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#include <cuda_runtime.h>
#include <string>
#include <deque>
#include <mutex>

using torch::stable::Tensor;
namespace tsa = torch::stable::accelerator;

namespace {
inline tsa::DeviceGuard make_device_guard(const Tensor& t) {
  return tsa::DeviceGuard(static_cast<tsa::DeviceIndex>(t.get_device()));
}
std::deque<std::once_flag> device_flags;
std::vector<cudaDeviceProp> device_properties;

void initVectors() {
  static bool init_flag [[maybe_unused]] = []() {
    int device_count;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
      STD_TORCH_CHECK(false, "cudaGetDeviceProperties failed: " +
                                 std::string(cudaGetErrorString(err)));
    }
    device_flags.resize(device_count);
    device_properties.resize(device_count);
    return true;
  }();
}

void initDeviceProperty(int device_index) {
  cudaDeviceProp device_prop{};
  cudaError_t err = cudaGetDeviceProperties(&device_prop, device_index);
  if (err != cudaSuccess) {
    STD_TORCH_CHECK(false, "cudaGetDeviceProperties failed: " +
                               std::string(cudaGetErrorString(err)));
  }
  device_properties[device_index] = device_prop;
}

// Helper function to get device properties using raw CUDA APIs
cudaDeviceProp* get_device_prop() {
  initVectors();
  int device_index;
  cudaError_t err = cudaGetDevice(&device_index);
  if (err != cudaSuccess) {
    STD_TORCH_CHECK(false, "cudaGetDevice failed: " +
                               std::string(cudaGetErrorString(err)));
  }

  std::call_once(device_flags[device_index], initDeviceProperty, device_index);
  return &device_properties[device_index];
}
} // anonymous namespace


extern "C" {
/* Creates a dummy empty _C module that can be imported from Python.
    The import from Python will load the .so consisting of this file
    in this extension, so that the STABLE_TORCH_LIBRARY static initializers
    below are run. */
PyObject* PyInit__C(void)
{
    static struct PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
        "_C",   /* name of module */
        NULL,   /* module documentation, may be NULL */
        -1,     /* size of per-interpreter state of the module,
                    or -1 if the module keeps state in global variables. */
        NULL,   /* methods */
    };
    return PyModule_Create(&module_def);
}
}

#define CHECK_DEVICE(x) STD_TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) \
    do { \
        auto expected_dims = std::vector<int64_t>{__VA_ARGS__}; \
        STD_TORCH_CHECK(x.dim() == static_cast<int64_t>(expected_dims.size()), #x " must have " + std::to_string(expected_dims.size()) + " dimensions, got " + std::to_string(x.dim())); \
        for (size_t i = 0; i < expected_dims.size(); ++i) { \
            STD_TORCH_CHECK(x.size(i) == expected_dims[i], #x " dimension " + std::to_string(i) + " must have size " + std::to_string(expected_dims[i]) + ", got " + std::to_string(x.size(i))); \
        } \
    } while (0)
#define CHECK_CONTIGUOUS(x) STD_TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

#define PREPARE_VARLEN_MAX_BATCHES_1CTA 992

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
                      const Tensor q,
                      const Tensor k,
                      const Tensor v,
                      Tensor out,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      void *seqused_q,
                      void *seqused_k,
                      void *softmax_lse_d,
                      float p_dropout,
                      float softmax_scale,
                      int window_size_left,
                      int window_size_right,
                      int attention_chunk,
                      const float softcap=0.f,
                      const int sm_margin=0) {

    // Reset the parameters
    params = {};

    params.is_bf16 = q.scalar_type() == torch::headeronly::ScalarType::BFloat16;
    params.is_e4m3 = q.scalar_type() == torch::headeronly::ScalarType::Float8_e4m3fn;

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
    STD_TORCH_CHECK(p_dropout < 1.f);
    #ifdef FLASHATTENTION_DISABLE_DROPOUT
        STD_TORCH_CHECK(p_dropout == 0.0f, "This flash attention build does not support dropout.");
    #endif

    // Causal is the special case where window_size_right == 0 and window_size_left < 0.
    // Local is the more general case where window_size_right >= 0 or window_size_left >= 0.
    params.is_causal = window_size_left < 0 && window_size_right == 0 && attention_chunk == 0;
    params.is_local = (window_size_left >= 0 || window_size_right >= 0 || attention_chunk >= 1) && !params.is_causal;

    // TODO: check this
    if (window_size_left < 0) { window_size_left = seqlen_k - 1; }
    if (window_size_right < 0) { window_size_right = seqlen_q - 1; }
    if (attention_chunk > 0) {
        window_size_left = std::min(window_size_left, attention_chunk - 1);
        window_size_right = std::min(window_size_right, attention_chunk - 1);
    }
    params.window_size_left = window_size_left;
    params.window_size_right = window_size_right;
    params.attention_chunk = attention_chunk;

    auto dprops = get_device_prop();
    params.arch = dprops->major * 10 + dprops->minor;
    params.num_sm = dprops->multiProcessorCount - sm_margin;

    #ifdef FLASHATTENTION_DISABLE_LOCAL
        STD_TORCH_CHECK(!params.is_local, "This flash attention build does not support local attention.");
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
                      const Tensor q,
                      const Tensor k,
                      const Tensor v,
                      const Tensor out,
                      const Tensor dout,
                      Tensor dq,
                      Tensor dk,
                      Tensor dv,
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
                      int attention_chunk,
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
                     attention_chunk,
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

template <int Arch, int Split, bool PagedKVNonTMA, bool PackGQA, bool Has_softcap>
void run_mha_fwd_constexpr(Flash_fwd_params &params, cudaStream_t stream) {
    if (!params.is_e4m3) {
        if (params.is_bf16) {
            #ifndef FLASHATTENTION_DISABLE_HDIM64
            if (params.d <= 64) {
                #ifndef FLASHATTENTION_DISABLE_HDIMDIFF64
                if constexpr (Arch == 90) {
                    if (params.dv > 256) {
                        return run_mha_fwd_<Arch, cutlass::bfloat16_t, 64, 512, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                    } else if (params.dv > 64) {
                        return run_mha_fwd_<Arch, cutlass::bfloat16_t, 64, 256, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                    }
                }
                #endif
                return run_mha_fwd_<Arch, cutlass::bfloat16_t, 64, 64, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
            }
            #endif
            #ifndef FLASHATTENTION_DISABLE_HDIM96
            if (params.d <= 96) { return run_mha_fwd_<Arch, cutlass::bfloat16_t, 96, 96, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
            #endif
            #ifndef FLASHATTENTION_DISABLE_HDIM128
            if (params.d <= 128) { return run_mha_fwd_<Arch, cutlass::bfloat16_t, 128, 128, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
            #endif
            #ifndef FLASHATTENTION_DISABLE_HDIM192
            if (params.d <= 192) {
                #ifndef FLASHATTENTION_DISABLE_HDIMDIFF192
                if constexpr (Arch == 90) {
                    if (params.dv <= 128) {
                        return run_mha_fwd_<Arch, cutlass::bfloat16_t, 192, 128, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                    }
                }
                #endif
                return run_mha_fwd_<Arch, cutlass::bfloat16_t, 192, 192, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
            }
            #endif
            #ifndef FLASHATTENTION_DISABLE_HDIM256
            if (params.d <= 256) { return run_mha_fwd_<Arch, cutlass::bfloat16_t, 256, 256, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
            #endif
        } else {
            #ifndef FLASHATTENTION_DISABLE_FP16
            #ifndef FLASHATTENTION_DISABLE_HDIM64
            if (params.d <= 64) {
                #ifndef FLASHATTENTION_DISABLE_HDIMDIFF64
                if constexpr (Arch == 90) {
                    if (params.dv > 256) {
                        return run_mha_fwd_<Arch, cutlass::half_t, 64, 512, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                    } else if (params.dv > 64) {
                        return run_mha_fwd_<Arch, cutlass::half_t, 64, 256, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                    }
                }
                #endif
                return run_mha_fwd_<Arch, cutlass::half_t, 64, 64, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
            }
            #endif
            #ifndef FLASHATTENTION_DISABLE_HDIM96
            if (params.d <= 96) { return run_mha_fwd_<Arch, cutlass::half_t, 96, 96, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
            #endif
            #ifndef FLASHATTENTION_DISABLE_HDIM128
            if (params.d <= 128) { return run_mha_fwd_<Arch, cutlass::half_t, 128, 128, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
            #endif
            #ifndef FLASHATTENTION_DISABLE_HDIM192
            if (params.d <= 192) {
                #ifndef FLASHATTENTION_DISABLE_HDIMDIFF192
                if constexpr (Arch == 90) {
                    if (params.dv <= 128) {
                        return run_mha_fwd_<Arch, cutlass::half_t, 192, 128, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                    }
                }
                #endif
                return run_mha_fwd_<Arch, cutlass::half_t, 192, 192, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
            }
            #endif
            #ifndef FLASHATTENTION_DISABLE_HDIM256
            if (params.d <= 256) { return run_mha_fwd_<Arch, cutlass::half_t, 256, 256, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
            #endif
            #else
            STD_TORCH_CHECK(false, "This flash attention build does not support FP16.");
            #endif
        }
    } else {
        #ifndef FLASHATTENTION_DISABLE_FP8
        #ifndef FLASHATTENTION_DISABLE_HDIM64
        if (params.d <= 64) { return run_mha_fwd_<90, cutlass::float_e4m3_t, 64, 64, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
        #endif
        #ifndef FLASHATTENTION_DISABLE_HDIM96
        if (params.d <= 96) { return run_mha_fwd_<90, cutlass::float_e4m3_t, 96, 96, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
        #endif
        #ifndef FLASHATTENTION_DISABLE_HDIM128
        if (params.d <= 128) { return run_mha_fwd_<90, cutlass::float_e4m3_t, 128, 128, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
        #endif
        #ifndef FLASHATTENTION_DISABLE_HDIM192
        if (params.d <= 192) {
            #ifndef FLASHATTENTION_DISABLE_HDIMDIFF192
            if constexpr (Arch == 90) {
                if (params.dv <= 128) {
                    return run_mha_fwd_<90, cutlass::float_e4m3_t, 192, 128, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                }
            }
            #endif
            return run_mha_fwd_<90, cutlass::float_e4m3_t, 192, 192, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
        }
        #endif
        #ifndef FLASHATTENTION_DISABLE_HDIM256
        if (params.d <= 256) { return run_mha_fwd_<90, cutlass::float_e4m3_t, 256, 256, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
        #endif
        #else
        STD_TORCH_CHECK(false, "This flash attention build does not support FP8.");
        #endif
    }
}

void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    // HEADDIM_SWITCH(params.d, [&] {
    //     run_mha_fwd_<cutlass::half_t, kHeadSize>(params, stream);
    // });
    STD_TORCH_CHECK(params.num_splits >= 1);
    ARCH_SWITCH(params.arch, Arch, [&] {
        SPLIT_SWITCH(params.num_splits > 1, Split, [&] {
            PAGEDKV_SWITCH(params.page_table && !params.pagedkv_tma, PagedKVNonTMA, [&] {
                PACKGQA_SWITCH(params.pack_gqa, PackGQA_, [&] {
                    // Always enable PackGQA for Sm8x or PagedKVNonTMA or Split to reduce compilation
                    static constexpr bool PackGQA = PackGQA_ || Arch < 90 || PagedKVNonTMA || Split;
                    SOFTCAP_SWITCH(params.softcap > 0.0, Has_softcap, [&] {
                        run_mha_fwd_constexpr<Arch, Split, PagedKVNonTMA, PackGQA, Has_softcap>(params, stream);
                    });
                });
            });
        });
    });
}

void run_mha_fwd_combine(Flash_fwd_params &params, cudaStream_t stream, bool enable_pdl=false) {
    #ifndef FLASHATTENTION_DISABLE_SPLIT
    // If hdim is 96 or 192, it's faster to round them to 128 or 256 respectively
    // so that kBlockM is smaller and we have more parallelism.
    if (params.is_fp32) {
        if (params.dv <= 64) {
            run_mha_fwd_combine_<float, float, 64>(params, stream, enable_pdl);
        } else {
            run_mha_fwd_combine_<float, float, 128>(params, stream, enable_pdl);
        }
    } else if (params.is_bf16) {
        if (params.dv <= 64) {
            run_mha_fwd_combine_<cutlass::bfloat16_t, float, 64>(params, stream, enable_pdl);
        } else {
            run_mha_fwd_combine_<cutlass::bfloat16_t, float, 128>(params, stream, enable_pdl);
        }
    } else {
        if (params.dv <= 64) {
            run_mha_fwd_combine_<cutlass::half_t, float, 64>(params, stream, enable_pdl);
        } else {
            run_mha_fwd_combine_<cutlass::half_t, float, 128>(params, stream, enable_pdl);
        }
    }
    #else
    STD_TORCH_CHECK(false, "This flash attention build does not support combine kernels.");
    #endif
}

inline bool get_pagedkv_tma(Flash_fwd_params const& params) {
    if (params.arch < 90 || !params.page_table || params.leftpad_k || params.knew_ptr) { return false; }
    // This needs to match the kernel configs
    auto kBlockMN_kernel_args_sm90 = tile_size_fwd_sm90(params.d_rounded, params.dv_rounded, params.is_causal, params.is_local, params.is_e4m3 ? 1 : 2 /*element_size*/, false /*v_colmajor*/, false /*paged_kv_non_TMA*/, params.softcap > 0.f);
    int const kBlockM = std::get<0>(kBlockMN_kernel_args_sm90);
    int const kBlockN = std::get<1>(kBlockMN_kernel_args_sm90);
    // Heuristic: when seqlen_q <= kBlockM, we're not compute bound, and somehow using TMA is slower,
    // at least for MLA.
    return params.page_size % kBlockN == 0 && params.seqlen_q * (params.h / params.h_k) > kBlockM;
}

inline bool get_pack_gqa(Flash_fwd_params const& params) {
    // Always enable PackGQA for Sm8x or PagedKVNonTMA or Split to reduce compilation and binary size.
    // Has little effect on speed.
    if (params.arch < 90 || (params.page_table && !params.pagedkv_tma) || params.num_splits > 1) { return true; }
    #ifdef FLASHATTENTION_DISABLE_PACKGQA
    return false;
    #else
    // params.page_table must already be set
    if (params.h == params.h_k) { return false; }
    // This needs to match the kernel configs
    auto kBlockMN_kernel_args_sm90 = tile_size_fwd_sm90(params.d_rounded, params.dv_rounded, params.is_causal, params.is_local, params.is_e4m3 ? 1 : 2 /*element_size*/, false /*v_colmajor*/, params.page_table && !params.pagedkv_tma, params.softcap > 0.f);
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
    auto kBlockMN_kernel_args_sm90 = tile_size_fwd_sm90(params.d_rounded, params.dv_rounded, params.is_causal, params.is_local, params.is_e4m3 ? 1 : 2 /*element_size*/, false /*v_colmajor*/, params.page_table && !params.pagedkv_tma, params.softcap > 0.f);
    // Strictly speaking we need to pass in (varlen && params.num_splits > 1) but num_splits
    // has not been set here. It's OK though because we might just underestimate kBlockN a bit
    auto kBlockMN_kernel_args_sm8x = tile_size_fwd_sm8x(params.arch == 86 || params.arch == 89, params.d_rounded, params.dv_rounded, params.is_causal, params.is_local, params.is_e4m3 ? 1 : 2 /*element_size*/, params.page_table, varlen, params.softcap > 0.f, params.knew_ptr);
    int const kBlockM = params.arch >= 90 ? std::get<0>(kBlockMN_kernel_args_sm90) : std::get<0>(kBlockMN_kernel_args_sm8x);
    int const kBlockN = params.arch >= 90 ? std::get<1>(kBlockMN_kernel_args_sm90) : std::get<1>(kBlockMN_kernel_args_sm8x);
    int seqlen_q_packgqa = params.seqlen_q * (params.h / params.h_k);
    // If is_local, we're not going to load all of seqlen_k
    int const seqlen_k_loaded = !params.is_local
        ? params.seqlen_k
        : std::max(0, std::min(params.seqlen_k, params.window_size_right + params.window_size_left + 1 + kBlockM));
    int const num_n_blocks = (seqlen_k_loaded + kBlockN - 1) / kBlockN;
    int const num_m_blocks = (seqlen_q_packgqa + kBlockM - 1) / kBlockM;
    int const size_one_kv_head = params.seqlen_k * (params.d + params.dv) * (params.is_e4m3 ? 1 : 2);
    // Always enable PackGQA for Split
    // If varlen, we use dynamic split, so this heuristic just needs to get an upper bound on num_splits.
    // We assume the case where there's 1 long sequence and the rest are short, i.e. pretending
    // that batch = 1.
    int total_mblocks = (params.num_splits_dynamic_ptr ? 1 : params.b) * params.h_k * num_m_blocks;
    return num_splits_heuristic(total_mblocks, params.num_sm, num_n_blocks, num_m_blocks, size_one_kv_head, params.is_causal || params.is_local, 128);
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

inline int round_up_headdimv(int head_size) {
    if (head_size <= 64) { return 64; }
    if (head_size <= 96) { return 96; }
    if (head_size <= 128) { return 128; }
    if (head_size <= 192) { return 192; }
    if (head_size <= 256) { return 256; }
    return 512;
}

// Only applicable to the case where seqused_k (i.e. cache_seqlens) is available
Tensor
mha_fwd_get_scheduler_metadata(
        int64_t batch_size,
        int64_t max_seqlen_q,
        int64_t max_seqlen_k,
        int64_t num_heads,
        int64_t num_heads_k,
        int64_t headdim,
        int64_t headdim_v,
        torch::headeronly::ScalarType qkv_dtype,
        Tensor seqused_k, // b
        std::optional<Tensor> cu_seqlens_q_,  // b+1
        std::optional<Tensor> cu_seqlens_k_,  // b+1
        std::optional<Tensor> cu_seqlens_k_new_,  // b+1
        std::optional<Tensor> seqused_q_, // b. If given, only this many elements of each batch element's queries and outputs are used.
        std::optional<Tensor> leftpad_k_, // b
        std::optional<int64_t> page_size,
        int64_t max_seqlen_k_new,  // 0 means we're not appending new KV
        bool is_causal,
        int64_t window_size_left,
        int64_t window_size_right,
        int64_t attention_chunk,
        bool has_softcap,
        int64_t num_splits,
        std::optional<bool> pack_gqa_,
        int64_t sm_margin) {

    STD_TORCH_CHECK(qkv_dtype == torch::headeronly::ScalarType::Half || qkv_dtype == torch::headeronly::ScalarType::BFloat16 || qkv_dtype == torch::headeronly::ScalarType::Float8_e4m3fn,
                "FlashAttention only supports fp16, bf16, and fp8_e4m3 data type");
    STD_TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    // Reset the parameters
    Flash_fwd_params params{};
    params.is_bf16 = qkv_dtype == torch::headeronly::ScalarType::BFloat16;
    params.is_e4m3 = qkv_dtype == torch::headeronly::ScalarType::Float8_e4m3fn;
    params.b = batch_size;
    params.seqlen_q = max_seqlen_q;
    params.seqlen_k = max_seqlen_k;
    params.h = num_heads;
    params.h_k = num_heads_k;
    params.d = headdim;
    params.dv = headdim_v;
    params.d_rounded = round_up_headdim(headdim);
    params.dv_rounded = headdim_v == headdim ? params.d_rounded : round_up_headdimv(headdim_v);
    params.seqlen_knew = max_seqlen_k_new;

    bool const is_varlen_q = cu_seqlens_q_.has_value();
    params.cu_seqlens_q = is_varlen_q ? static_cast<int*>(cu_seqlens_q_.value().data_ptr()) : nullptr;
    bool const is_varlen_k = cu_seqlens_k_.has_value();
    params.cu_seqlens_k = is_varlen_k ?  static_cast<int*>(cu_seqlens_k_.value().data_ptr()) : nullptr;
    params.cu_seqlens_knew = cu_seqlens_k_new_.has_value() ? static_cast<int*>(cu_seqlens_k_new_.value().data_ptr()): nullptr;
    params.seqused_q = seqused_q_.has_value() ?  static_cast<int*>(seqused_q_.value().data_ptr()) : nullptr;
    params.seqused_k = static_cast<int*>(seqused_k.data_ptr());
    params.leftpad_k = leftpad_k_.has_value() ? static_cast<int*>(leftpad_k_.value().data_ptr()) : nullptr;
    params.knew_ptr = params.seqlen_knew > 0 ? reinterpret_cast<int*>(1) : nullptr;
    if (window_size_left >= max_seqlen_k - 1) { window_size_left = -1; }
    if (window_size_right >= max_seqlen_q - 1) { window_size_right = -1; }
    // causal=true is the same as causal=false in this case
    if (max_seqlen_q == 1 && window_size_left == -1 && window_size_right == -1 && attention_chunk == 0) {
        // Special case of hdim 128 where we want causal to have kBlockN=128, better for pagedKV and TMA
        if ((headdim <= 64 || headdim > 128) || !page_size.has_value()) {
            is_causal = false;
        }
    }
    if (is_causal) { window_size_right = 0; }

    params.is_causal = window_size_left < 0 && window_size_right == 0 && attention_chunk == 0;
    params.is_local = (window_size_left >= 0 || window_size_right >= 0 || attention_chunk >= 1) && !params.is_causal;
    if (window_size_left < 0) { window_size_left = max_seqlen_k - 1; }
    if (window_size_right < 0) { window_size_right = max_seqlen_q - 1; }
    if (attention_chunk > 0) {
        window_size_left = std::min(window_size_left, attention_chunk - 1);
        window_size_right = std::min(window_size_right, attention_chunk - 1);
    }
    params.window_size_left = window_size_left;
    params.window_size_right = window_size_right;
    params.attention_chunk = attention_chunk;
    auto dprops = get_device_prop();
    params.arch = dprops->major * 10 + dprops->minor;
    params.num_sm = dprops->multiProcessorCount - sm_margin;
    params.softcap = has_softcap ? 1.0f : 0.0f;

    params.page_size = page_size.has_value() ? page_size.value() : 1;
    params.page_table = !page_size.has_value() ? nullptr : reinterpret_cast<int*>(1);

    bool const use_prepare_varlen = true;
    params.prepare_varlen_pdl = use_prepare_varlen && params.b <= PREPARE_VARLEN_MAX_BATCHES_1CTA;
    params.num_splits_dynamic_ptr = !use_prepare_varlen ? nullptr : reinterpret_cast<int*>(1);

    params.pagedkv_tma = get_pagedkv_tma(params);
    params.num_splits = num_splits <= 0 ? get_num_splits(params) : num_splits;
    // Always enable PackGQA for Split, and get_pack_gqa requires params.num_splits to decide
    params.pack_gqa = pack_gqa_.has_value() ? pack_gqa_.value() : get_pack_gqa(params);

    bool is_varlen = true;

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    auto device_guard = make_device_guard(seqused_k);

    // This needs to be set after get_num_splits
    Tensor tile_count_semaphore;  // Contains the semaphore and optionally num_splits_dynamic
    bool const scheduler_needs_semaphore = params.arch >= 90 || params.num_splits > 1;
    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    params.varlen_sort_batches = !params.is_local; // Use this value for Sort in scheduler template
    params.head_swizzle = params.is_causal || params.is_local; // Use this value for LPT in scheduler template
    if (scheduler_needs_semaphore || use_prepare_varlen) {   
        int b_rounded = round_multiple(params.b, 4); // for 16 byte alignment of pointers 
        int num_prepare_batch_vectors = use_prepare_varlen ? 2 : 0;
        if(params.varlen_sort_batches) { num_prepare_batch_vectors += 1; }
        if(params.head_swizzle) { num_prepare_batch_vectors += 1; }
        int head_swizzle_offset = b_rounded * (params.varlen_sort_batches ? 3 : 2);
        int tile_count_semaphore_offset = b_rounded * num_prepare_batch_vectors;
        // printf("(Metadata) num prepare batch vectors = %d.\n", num_prepare_batch_vectors);
        tile_count_semaphore = torch::stable::new_empty(
            seqused_k,
            {int(scheduler_needs_semaphore) + tile_count_semaphore_offset},
            std::make_optional(torch::headeronly::ScalarType::Int));
        // {num_splits_dynamic, num_m_blocks, varlen_batch_idx, num_nheads_in_l2}
        params.num_splits_dynamic_ptr = use_prepare_varlen ? static_cast<int*>(tile_count_semaphore.data_ptr()) : nullptr;
        params.num_m_blocks_ptr =  use_prepare_varlen ? static_cast<int*>(tile_count_semaphore.data_ptr()) + b_rounded : nullptr;
        params.varlen_batch_idx_ptr =  use_prepare_varlen && params.varlen_sort_batches ? static_cast<int*>(tile_count_semaphore.data_ptr()) + b_rounded * 2 : nullptr;
        // params.num_n_blocks_ptr  = use_prepare_varlen && params.head_swizzle ? static_cast<int*>(tile_count_semaphore.data_ptr()) + head_swizzle_offset : nullptr;
        params.num_nheads_in_l2_ptr = use_prepare_varlen && params.head_swizzle ? static_cast<int*>(tile_count_semaphore.data_ptr()) + head_swizzle_offset : nullptr;
        if (scheduler_needs_semaphore) {
            if (!use_prepare_varlen) { torch::stable::zero_(tile_count_semaphore); }  // If varlen we'll manually do the zero-ing
            params.tile_count_semaphore = static_cast<int*>(tile_count_semaphore.data_ptr()) + tile_count_semaphore_offset;
        } else {
            params.tile_count_semaphore = nullptr;
        }
    }

    if (use_prepare_varlen) {
        auto kBlockMN_kernel_args_sm90 = tile_size_fwd_sm90(params.d_rounded, params.dv_rounded, params.is_causal, params.is_local, params.is_e4m3 ? 1 : 2 /*element_size*/, false /*v_colmajor*/, params.page_table && !params.pagedkv_tma, params.softcap > 0.f);
        auto kBlockMN_kernel_args_sm8x = tile_size_fwd_sm8x(params.arch == 86 || params.arch == 89, params.d_rounded, params.dv_rounded, params.is_causal, params.is_local, params.is_e4m3 ? 1 : 2 /*element_size*/, params.page_table, is_varlen && params.num_splits > 1, params.softcap > 0.f, params.knew_ptr);
        int const kBlockM = params.arch >= 90 ? std::get<0>(kBlockMN_kernel_args_sm90) : std::get<0>(kBlockMN_kernel_args_sm8x);
        int const kBlockN = params.arch >= 90 ? std::get<1>(kBlockMN_kernel_args_sm90) : std::get<1>(kBlockMN_kernel_args_sm8x);
        auto device_idx = torch::stable::accelerator::getCurrentDeviceIndex();
        auto stream = (cudaStream_t)torch::stable::accelerator::getCurrentStream(device_idx).id();
        prepare_varlen_num_blocks(params, stream, params.pack_gqa, kBlockM, kBlockN, false /*enable_pdl*/);
        CHECK_CUDA_KERNEL_LAUNCH();
    }
    return tile_count_semaphore;
}

// b: batch_size
// b_k: batch_size_k
// s_q: seqlen_q
// s_k: seqlen_k
// s_k_new: seqlen_k_new
// h: num_heads
// h_k: num_heads_k
// d: head_size
std::tuple<Tensor, Tensor, Tensor, Tensor>
mha_fwd(Tensor q,   // (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
        Tensor k,  // (b_k, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k or (num_pages, page_size, h_k, d) if there is page_table.
        Tensor v,  // (b_k, s_k, h_k, dv) or (total_k, h_k, dv) if there is cu_seqlens_k or (num_pages, page_size, h_k, dv) if there is page_table.
        std::optional<Tensor> k_new_,  // (b, s_k_new, h_k, d) or (total_k_new, h_k, d) if there is cu_seqlens_k_new
        std::optional<Tensor> v_new_,  // (b, s_k_new, h_k, dv) or (total_k_new, h_k, dv) if there is cu_seqlens_k_new
        std::optional<Tensor> q_v_,  // (b, s_q, h, dv) or (total_q_new, h, dv) if there is cu_seqlens_q
        std::optional<Tensor> out_,  // (b, s_q, h, dv) or (total_q, h, dv) if there is cu_seqlens_q
        std::optional<Tensor> cu_seqlens_q_,  // b+1
        std::optional<Tensor> cu_seqlens_k_,  // b+1
        std::optional<Tensor> cu_seqlens_k_new_,  // b+1
        std::optional<Tensor> seqused_q_, // b. If given, only this many elements of each batch element's queries and outputs are used.
        std::optional<Tensor> seqused_k_, // b. If given, only this many elements of each batch element's keys are used.
        std::optional<int64_t> max_seqlen_q_,
        // TODO: check if we need max_seqlen_k
        std::optional<int64_t> max_seqlen_k_,
        std::optional<Tensor> page_table_, // (b_k, max_num_pages_per_seq)
        std::optional<Tensor> kv_batch_idx_, // b. indices to index into the KV cache
        std::optional<Tensor> leftpad_k_, // b
        std::optional<Tensor> rotary_cos_, // seqlen_ro x (rotary_dim / 2)
        std::optional<Tensor> rotary_sin_, // seqlen_ro x (rotary_dim / 2)
        std::optional<Tensor> seqlens_rotary_, // b
        std::optional<Tensor> q_descale_,  // (b, h_k), not (b, h)
        std::optional<Tensor> k_descale_,  // (b, h_k)
        std::optional<Tensor> v_descale_,  // (b, h_k)
        std::optional<double> softmax_scale_,
        bool is_causal,
        int64_t window_size_left,
        int64_t window_size_right,
        int64_t attention_chunk,
        double softcap,
        bool is_rotary_interleaved,   // if true, rotary combines indices 0 & 1, else indices 0 & rotary_dim / 2
        std::optional<Tensor> scheduler_metadata_,  // (b + 1)
        int64_t num_splits,
        std::optional<bool> pack_gqa_,
        int64_t sm_margin
        ) {

    auto dprops = get_device_prop();
    bool is_sm8x = dprops->major >= 8;
    STD_TORCH_CHECK(is_sm8x, "FlashAttention only supports Ampere GPUs or newer.");

    auto q_type = q.scalar_type();
    STD_TORCH_CHECK(q_type == torch::headeronly::ScalarType::Half || q_type == torch::headeronly::ScalarType::BFloat16 || q_type == torch::headeronly::ScalarType::Float8_e4m3fn,
                "FlashAttention only supports fp16, bf16, and fp8_e4m3 data type");
    if (dprops->major < 9) {
        STD_TORCH_CHECK(q_type == torch::headeronly::ScalarType::Half || q_type == torch::headeronly::ScalarType::BFloat16,
                    "FlashAttention on Ampere/Ada cards only supports fp16 and bf16 data type");
    }
    STD_TORCH_CHECK(k.scalar_type() == q_type, "query and key must have the same dtype");
    STD_TORCH_CHECK(v.scalar_type() == q_type, "query and value must have the same dtype");

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);

    STD_TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    STD_TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    STD_TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    Tensor page_table;
    const bool paged_KV = page_table_.has_value();
    if (paged_KV) {
        page_table = page_table_.value();
        CHECK_DEVICE(page_table);
        STD_TORCH_CHECK(page_table.scalar_type() == torch::headeronly::ScalarType::Int, "page_table must have dtype torch.int32");
        STD_TORCH_CHECK(page_table.stride(-1) == 1, "page_table must have contiguous last dimension");
    }

    Tensor cu_seqlens_q;
    bool const is_varlen_q = cu_seqlens_q_.has_value();
    if (is_varlen_q) {
        cu_seqlens_q = cu_seqlens_q_.value();
        CHECK_DEVICE(cu_seqlens_q); CHECK_CONTIGUOUS(cu_seqlens_q);
        STD_TORCH_CHECK(cu_seqlens_q.scalar_type() == torch::headeronly::ScalarType::Int, "cu_seqlens_q must have dtype torch.int32");
        STD_TORCH_CHECK(max_seqlen_q_.has_value(), "max_seqlen_q must be provided if cu_seqlens_q is provided");
    }
    Tensor cu_seqlens_k;
    bool const is_varlen_k = cu_seqlens_k_.has_value();
    if (is_varlen_k) {
        cu_seqlens_k = cu_seqlens_k_.value();
        CHECK_DEVICE(cu_seqlens_k); CHECK_CONTIGUOUS(cu_seqlens_k);
        STD_TORCH_CHECK(cu_seqlens_k.scalar_type() == torch::headeronly::ScalarType::Int, "cu_seqlens_k must have dtype torch.int32");
        STD_TORCH_CHECK(max_seqlen_k_.has_value(), "max_seqlen_k must be provided if cu_seqlens_k is provided");
        STD_TORCH_CHECK(!paged_KV, "If cu_seqlens_k is passed in, then page table is not supported");
        STD_TORCH_CHECK(!kv_batch_idx_.has_value(), "If cu_seqlens_k is passed in, then page table is not supported");
    }

    const int batch_size = !is_varlen_q ? q.size(0) : cu_seqlens_q.size(0) - 1;
    int seqlen_q = !is_varlen_q ? q.size(1) : max_seqlen_q_.value();
    int total_q = !is_varlen_q ? batch_size * q.size(1) : q.size(0);
    int num_heads = q.size(-2);
    int const head_size = q.size(-1);
    int const head_size_v = v.size(-1);
    int const max_num_pages_per_seq = !paged_KV ? 0 : page_table.size(1);
    int const num_pages = !paged_KV ? 0 : k.size(0);
    int const page_size = !paged_KV ? 1 : k.size(1);
    int const seqlen_k = !is_varlen_k ? (!paged_KV ? k.size(1) : max_num_pages_per_seq * page_size) : max_seqlen_k_.value();
    int const total_k = !is_varlen_k ? batch_size * k.size(1) : k.size(0);
    int const num_heads_k = k.size(-2);
    int const batch_size_k = !paged_KV ? (!is_varlen_k ? k.size(0) : cu_seqlens_k.size(0) - 1) : page_table.size(0);
    double softmax_scale = 1.0 / sqrt(double(head_size));
    if (softmax_scale_.has_value()) {
        softmax_scale = softmax_scale_.value();
    }
    if (!kv_batch_idx_.has_value()) {
        STD_TORCH_CHECK(batch_size == batch_size_k, "batch_size must be equal to batch_size_k");
    }
    int const max_headdim = get_max_headdim();
    STD_TORCH_CHECK(head_size <= max_headdim, "FlashAttention forward only supports head dimension at most " + std::to_string(max_headdim));
    STD_TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");
    if (head_size_v != head_size) {
        STD_TORCH_CHECK((head_size > 128 && head_size <= 192 && head_size_v > 96 && head_size_v <= 128) ||
                   (head_size <= 64 && head_size_v <= 512),
                   "If V headdim is different from Q/K dim, we only support Q/K headdim in (128, 192] and V headdim in (96, 128], "
                   "or (Q/K <= 64 and V <= 512).");
        STD_TORCH_CHECK(dprops->major == 9, "Only Hopper supports different V headdim");
        if (head_size_v > 256) {
            STD_TORCH_CHECK(q_type == torch::headeronly::ScalarType::Half || q_type == torch::headeronly::ScalarType::BFloat16,
                        "HeaddimV > 256 requires fp16 and bf16 data type");
        }
    }

    // This needs to go before kBlockM & kBlockN since we rely on the correct window_size and is_causal to set kBlockM
    // TODO: check this
    if (window_size_left >= seqlen_k - 1) { window_size_left = -1; }
    if (window_size_right >= seqlen_q - 1) { window_size_right = -1; }
    // causal=true is the same as causal=false in this case
    if (seqlen_q == 1 && window_size_left == -1 && window_size_right == -1 && attention_chunk == 0) {
        // Special case of hdim 128 where we want causal to have kBlockN=128, better for pagedKV and TMA
        if ((head_size <= 64 || head_size > 128) || !paged_KV) {
            is_causal = false;
        }
    }
    if (is_causal) { window_size_right = 0; }

    if (!is_varlen_q) {
        CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);
    } else {
        CHECK_SHAPE(q, total_q, num_heads, head_size);
        CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
    }
    if (!paged_KV) {
        if (!is_varlen_k) {
            CHECK_SHAPE(k, batch_size_k, seqlen_k, num_heads_k, head_size);
            CHECK_SHAPE(v, batch_size_k, seqlen_k, num_heads_k, head_size_v);
        } else {
            CHECK_SHAPE(k, total_k, num_heads_k, head_size);
            CHECK_SHAPE(v, total_k, num_heads_k, head_size_v);
            CHECK_SHAPE(cu_seqlens_k, batch_size + 1);
        }
    } else {
        CHECK_SHAPE(k, num_pages, page_size, num_heads_k, head_size);
        CHECK_SHAPE(v, num_pages, page_size, num_heads_k, head_size_v);
        CHECK_SHAPE(page_table, batch_size_k, max_num_pages_per_seq);
    }

    if (seqused_q_.has_value()){
        auto seqused_q = seqused_q_.value();
        STD_TORCH_CHECK(seqused_q.scalar_type() == torch::headeronly::ScalarType::Int, "seqused_q must have dtype int32");
        CHECK_DEVICE(seqused_q); CHECK_CONTIGUOUS(seqused_q);
        CHECK_SHAPE(seqused_q, batch_size);
    }
    if (seqused_k_.has_value()) {
        auto seqused_k = seqused_k_.value();
        STD_TORCH_CHECK(seqused_k.scalar_type() == torch::headeronly::ScalarType::Int, "seqused_k must have dtype int32");
        CHECK_DEVICE(seqused_k); CHECK_CONTIGUOUS(seqused_k);
        CHECK_SHAPE(seqused_k, batch_size);
    }

    if (leftpad_k_.has_value()) {
        auto leftpad_k = leftpad_k_.value();
        STD_TORCH_CHECK(leftpad_k.scalar_type() == torch::headeronly::ScalarType::Int, "leftpad_k must have dtype int32");
        CHECK_DEVICE(leftpad_k); CHECK_CONTIGUOUS(leftpad_k);
        CHECK_SHAPE(leftpad_k, batch_size);
    }

    // This is what we will template on
    bool const is_varlen = is_varlen_q || is_varlen_k || seqused_q_.has_value() || seqused_k_.has_value() || leftpad_k_.has_value();
    #ifdef FLASHATTENTION_DISABLE_VARLEN
        STD_TORCH_CHECK(!is_varlen, "This flash attention build does not support varlen.");
    #endif

    int const alignment = q_type == torch::headeronly::ScalarType::Float8_e4m3fn ? 16 : 8;
    STD_TORCH_CHECK(head_size % alignment == 0, "head_size should be a multiple of " + std::to_string(alignment));
    STD_TORCH_CHECK(head_size_v % alignment == 0, "head_size_v should be a multiple of " + std::to_string(alignment));

    auto out_type = q_type == torch::headeronly::ScalarType::Float8_e4m3fn ? torch::headeronly::ScalarType::BFloat16 : q_type;
    Tensor out;
    if (out_.has_value()) {
        out = out_.value();
        STD_TORCH_CHECK(out.scalar_type() == out_type, "For FP16/BF16 input, output must have the same dtype as inputs. For FP8 input, output must have dtype BF16");
        CHECK_DEVICE(out);
        STD_TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        if (!is_varlen_q) {
            CHECK_SHAPE(out, batch_size, seqlen_q, num_heads, head_size_v);
        } else {
            CHECK_SHAPE(out, total_q, num_heads, head_size_v);
        }
    } else {
        out = !is_varlen_q
            ? torch::stable::new_empty(q, {batch_size, seqlen_q, num_heads, head_size_v}, std::make_optional(out_type))
            : torch::stable::new_empty(q, {total_q, num_heads, head_size_v}, std::make_optional(out_type));
    }

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    int const head_size_rounded = round_up_headdim(head_size);
    int const head_size_v_rounded = head_size_v == head_size ? head_size_rounded : round_up_headdimv(head_size_v);
    int const seqlen_q_rounded = round_multiple(seqlen_q, 128);
    int const seqlen_k_rounded = round_multiple(seqlen_k, 128);

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    auto device_guard = make_device_guard(q);

    Tensor softmax_lse;
    if (!is_varlen_q) {
        softmax_lse = torch::stable::new_empty(q, {batch_size, num_heads, seqlen_q}, std::make_optional(torch::headeronly::ScalarType::Float));
    } else {
        softmax_lse = torch::stable::new_empty(q, {num_heads, total_q}, std::make_optional(torch::headeronly::ScalarType::Float));
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
                     attention_chunk,
                     softcap,
                     sm_margin);
    params.total_q = total_q;
    params.total_k = total_k;
    params.b_k = batch_size_k;
    params.dv = head_size_v;
    params.dv_rounded = head_size_v_rounded;
    if (leftpad_k_.has_value()) {  // This needs to be set before get_pagedkv_tma
        params.leftpad_k = static_cast<int *>(leftpad_k_.value().data_ptr());
    }
    if (paged_KV) {
        params.page_table = static_cast<int*>(page_table.data_ptr());
        params.page_table_batch_stride = page_table.stride(0);
    }
    params.page_size = page_size;
    params.num_pages = num_pages;

    if (k_new_.has_value()) {  // This needs to be set before get_pagedkv_tma
        Tensor k_new, v_new;
        STD_TORCH_CHECK(v_new_.has_value(), "If k_new is supplied, v_new must also be passed in");
        STD_TORCH_CHECK(seqused_k_.has_value(), "If k_new is supplied, seqlens_k must also be passed in");
        STD_TORCH_CHECK(seqlen_q <= seqlen_k, "If k_new is supplied, it must have seqlen <= the seqlen of the KV cache");
        Tensor cu_seqlens_k_new;
        bool const is_varlen_k_new = cu_seqlens_k_new_.has_value();
        if (is_varlen_k_new) {
            cu_seqlens_k_new = cu_seqlens_k_new_.value();
            CHECK_DEVICE(cu_seqlens_k_new); CHECK_CONTIGUOUS(cu_seqlens_k_new);
            STD_TORCH_CHECK(cu_seqlens_k_new.scalar_type() == torch::headeronly::ScalarType::Int, "cu_seqlens_k_new must have dtype torch.int32");
        }
        k_new = k_new_.value();
        v_new = v_new_.value();
        STD_TORCH_CHECK(k_new.scalar_type() == q_type, "k_new must have the same dtype as query");
        STD_TORCH_CHECK(v_new.scalar_type() == q_type, "v_new must have the same dtype as query");
        CHECK_DEVICE(k_new); CHECK_DEVICE(v_new);
        STD_TORCH_CHECK(k_new.stride(-1) == 1, "k_new tensor must have contiguous last dimension");
        STD_TORCH_CHECK(v_new.stride(-1) == 1, "v_new tensor must have contiguous last dimension");
        // We don't need max_seqlen_k_new, so seqlen_k_new can be whatever when is_varlen_k_new
        int seqlen_k_new = !is_varlen_k_new ? k_new.size(1) : 0;
        int total_k_new = !is_varlen_k_new ? batch_size * k_new.size(1): k_new.size(0);
        if (!is_varlen_k_new) {
            CHECK_SHAPE(k_new, batch_size, seqlen_k_new, num_heads_k, head_size);
            CHECK_SHAPE(v_new, batch_size, seqlen_k_new, num_heads_k, head_size_v);
        } else {
            CHECK_SHAPE(k_new, total_k_new, num_heads_k, head_size);
            CHECK_SHAPE(v_new, total_k_new, num_heads_k, head_size_v);
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
    
    bool const use_prepare_varlen = is_varlen;
    params.prepare_varlen_pdl = use_prepare_varlen && params.b <= PREPARE_VARLEN_MAX_BATCHES_1CTA;
    // Temporarily set num_splits_dynamic_ptr to 1 since get_num_splits checks it
    params.num_splits_dynamic_ptr = !use_prepare_varlen ? nullptr : reinterpret_cast<int*>(1);

    params.pagedkv_tma = get_pagedkv_tma(params);
    params.num_splits = num_splits <= 0 ? get_num_splits(params) : num_splits;
    // Always enable PackGQA for Split, and get_pack_gqa requires params.num_splits to decide
    params.pack_gqa = pack_gqa_.has_value() ? pack_gqa_.value() : get_pack_gqa(params);

    // This needs to be set after get_num_splits
    Tensor tile_count_semaphore;  // Contains the semaphore and optionally num_splits_dynamic
    // We don't use the persistent scheduler if Split and not Varlen
    bool const scheduler_needs_semaphore = params.arch >= 90
        ? (((params.is_causal || params.is_local) && (params.num_splits == 1)) || is_varlen)
        : ((params.is_causal && !is_varlen) || (is_varlen && params.num_splits > 1));
    params.varlen_sort_batches = !params.is_local; // Use this value for Sort in scheduler template
    params.head_swizzle = params.is_causal || params.is_local; // Use this value for LPT in scheduler template
    if (scheduler_needs_semaphore || use_prepare_varlen) {
        int b_rounded = round_multiple(params.b, 4); // for 16 byte alignment of pointers
        int num_prepare_batch_vectors = use_prepare_varlen ? 2 : 0;
        if(params.varlen_sort_batches) { num_prepare_batch_vectors += 1; }
        if(params.head_swizzle) { num_prepare_batch_vectors += 1; }
        int head_swizzle_offset = b_rounded * (params.varlen_sort_batches ? 3 : 2);
        int tile_count_semaphore_offset = b_rounded * num_prepare_batch_vectors;
        int metadata_size = int(scheduler_needs_semaphore) + tile_count_semaphore_offset;
        // printf("Num prepare batch vectors = %d, metadata_size = %d.\n", num_prepare_batch_vectors, metadata_size);
        params.skip_scheduler_metadata_computation = scheduler_metadata_.has_value();
        if (scheduler_metadata_.has_value()) {
            Tensor scheduler_metadata = scheduler_metadata_.value();
            CHECK_DEVICE(scheduler_metadata);
            CHECK_SHAPE(scheduler_metadata, metadata_size);
            CHECK_CONTIGUOUS(scheduler_metadata);
            STD_TORCH_CHECK(scheduler_metadata.scalar_type() == torch::headeronly::ScalarType::Int, "scheduler_metadata must have dtype int32");
            tile_count_semaphore = scheduler_metadata;
        } else {
            tile_count_semaphore = torch::stable::new_empty(q, {metadata_size}, torch::headeronly::ScalarType::Int);
        }
        if (scheduler_needs_semaphore && !use_prepare_varlen) {
            torch::stable::zero_(tile_count_semaphore);  // If varlen we'll manually do the zero-ing
        }
        // {num_splits_dynamic, num_m_blocks, varlen_batch_idx, num_nheads_in_l2}
        params.num_splits_dynamic_ptr = use_prepare_varlen ? static_cast<int*>(tile_count_semaphore.data_ptr()) : nullptr;
        params.num_m_blocks_ptr =  use_prepare_varlen ? static_cast<int*>(tile_count_semaphore.data_ptr()) + b_rounded : nullptr;
        params.varlen_batch_idx_ptr =  use_prepare_varlen && params.varlen_sort_batches ? static_cast<int*>(tile_count_semaphore.data_ptr()) + b_rounded * 2 : nullptr;
        // params.num_n_blocks_ptr  = use_prepare_varlen && params.head_swizzle ? static_cast<int*>(tile_count_semaphore.data_ptr()) + head_swizzle_offset : nullptr;
        params.num_nheads_in_l2_ptr = use_prepare_varlen && params.head_swizzle ? static_cast<int*>(tile_count_semaphore.data_ptr()) + head_swizzle_offset : nullptr;
        params.tile_count_semaphore = scheduler_needs_semaphore ? static_cast<int*>(tile_count_semaphore.data_ptr()) + tile_count_semaphore_offset : nullptr;
        params.tile_count_semaphore_offset = tile_count_semaphore_offset; // might need to zero out semaphore later
    }

    if (q_v_.has_value()) {
        STD_TORCH_CHECK(head_size <= 64, "q_v is only supported for head_size <= 64");
        STD_TORCH_CHECK(head_size_v >= 256, "q_v is only supported for hdim_v >= 256.");
        STD_TORCH_CHECK(q_type == torch::headeronly::ScalarType::Half || q_type == torch::headeronly::ScalarType::BFloat16,
                    "q_v is only supported for fp16 and bf16 data type");
        STD_TORCH_CHECK(params.arch == 90, "q_v is only supported for Hopper GPUs");
        Tensor q_v = q_v_.value();
        STD_TORCH_CHECK(q_v.scalar_type() == q_type, "q_v must have the same dtype as query");
        CHECK_DEVICE(q_v);
        STD_TORCH_CHECK(q_v.stride(-1) == 1, "q_v tensor must have contiguous last dimension");
        if (!is_varlen_q) {
            CHECK_SHAPE(q_v, batch_size, seqlen_q, num_heads, head_size_v);
        } else {
            CHECK_SHAPE(q_v, total_q, num_heads, head_size_v);
        }
        params.qv_ptr = q_v.data_ptr();
        // All stride are in elements, not bytes.
        params.qv_row_stride = q_v.stride(-3);
        params.qv_head_stride = q_v.stride(-2);
        if (!is_varlen_q) {
            params.qv_batch_stride = q_v.stride(0);
        }
    }

    if (rotary_cos_.has_value()) {
        STD_TORCH_CHECK(k_new_.has_value(), "If rotary cos/sin are provided, new key / value to be appended to KV cache must also be provided");
        auto rotary_cos = rotary_cos_.value();
        CHECK_DEVICE(rotary_cos); CHECK_CONTIGUOUS(rotary_cos);
        params.rotary_dim = rotary_cos.size(1) * 2;
        STD_TORCH_CHECK(params.rotary_dim <= head_size, "rotary_dim must be <= headdim");
        STD_TORCH_CHECK(params.rotary_dim % 16 == 0, "Only rotary dimensions divisible by 16 are currently supported");
        const int seqlen_ro = rotary_cos.size(0);
        if (paged_KV) {
            STD_TORCH_CHECK(seqlen_ro >= seqlen_k, "cos/sin seqlen must be at least the seqlen of KV cache");
        }
        CHECK_SHAPE(rotary_cos, seqlen_ro, params.rotary_dim / 2);
        STD_TORCH_CHECK(rotary_cos.scalar_type() == q_type, "rotary_cos must have the same dtype as query");

        STD_TORCH_CHECK(rotary_sin_.has_value(), "If rotary cos is provided, rotary sin must also be provided");
        auto rotary_sin = rotary_sin_.value();
        CHECK_DEVICE(rotary_sin); CHECK_CONTIGUOUS(rotary_sin);
        CHECK_SHAPE(rotary_sin, seqlen_ro, params.rotary_dim / 2);
        STD_TORCH_CHECK(rotary_sin.scalar_type() == q_type, "rotary_cos must have the same dtype as query");
        params.rotary_cos_ptr = rotary_cos.data_ptr();
        params.rotary_sin_ptr = rotary_sin.data_ptr();
        params.is_rotary_interleaved = is_rotary_interleaved;
        if (seqlens_rotary_.has_value()) {
            Tensor seqlens_rotary = seqlens_rotary_.value();
            CHECK_DEVICE(seqlens_rotary); CHECK_CONTIGUOUS(seqlens_rotary);
            STD_TORCH_CHECK(seqlens_rotary.scalar_type() == torch::headeronly::ScalarType::Int, "seqlens_rotary must have dtype torch.int32");
            CHECK_SHAPE(seqlens_rotary, batch_size);
            params.seqlens_rotary = static_cast<int*>(seqlens_rotary.data_ptr());
        }
    } else {
        params.rotary_dim = 0;
    }

    if (kv_batch_idx_.has_value()) {
        auto kv_batch_idx = kv_batch_idx_.value();
        CHECK_DEVICE(kv_batch_idx); CHECK_CONTIGUOUS(kv_batch_idx);
        STD_TORCH_CHECK(kv_batch_idx.scalar_type() == torch::headeronly::ScalarType::Int, "kv_batch_idx must have dtype int32");
        params.kv_batch_idx = reinterpret_cast<int *>(kv_batch_idx.data_ptr());
    }

    Tensor out_accum, softmax_lse_accum;
    auto outaccum_type = torch::headeronly::ScalarType::Float;
    if (params.num_splits > 1) {
        STD_TORCH_CHECK(params.num_splits <= 256, "num_splits > 256 not supported");
        if (!is_varlen_q) {
            out_accum = torch::stable::new_empty(q, {params.num_splits, batch_size, num_heads, seqlen_q, head_size_v}, std::make_optional(outaccum_type));
            softmax_lse_accum = torch::stable::new_empty(q, {params.num_splits, batch_size, num_heads, seqlen_q}, std::make_optional(torch::headeronly::ScalarType::Float));
            params.oaccum_batch_stride = out_accum.stride(1);
            params.lseaccum_batch_stride = softmax_lse_accum.stride(1);
        } else {
            out_accum = torch::stable::new_empty(q, {params.num_splits, num_heads, total_q, head_size_v}, std::make_optional(outaccum_type));
            softmax_lse_accum = torch::stable::new_empty(q, {params.num_splits, num_heads, total_q}, std::make_optional(torch::headeronly::ScalarType::Float));
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

    if (q_type == torch::headeronly::ScalarType::Float8_e4m3fn) {
        if (q_descale_.has_value()) {
            auto q_descale = q_descale_.value();
            CHECK_DEVICE(q_descale);
            CHECK_SHAPE(q_descale, batch_size, num_heads_k);
            params.q_descale_ptr = static_cast<float*>(q_descale.data_ptr());
            params.q_descale_batch_stride = q_descale.stride(0);
            params.q_descale_head_stride = q_descale.stride(1);
        } else {
            params.q_descale_ptr = nullptr;
        }
        if (k_descale_.has_value()) {
            auto k_descale = k_descale_.value();
            CHECK_DEVICE(k_descale);
            CHECK_SHAPE(k_descale, batch_size, num_heads_k);
            params.k_descale_ptr = static_cast<float*>(k_descale.data_ptr());
            params.k_descale_batch_stride = k_descale.stride(0);
            params.k_descale_head_stride = k_descale.stride(1);
        } else {
            params.k_descale_ptr = nullptr;
        }
        if (v_descale_.has_value()) {
            auto v_descale = v_descale_.value();
            CHECK_DEVICE(v_descale);
            CHECK_SHAPE(v_descale, batch_size, num_heads_k);
            params.v_descale_ptr = static_cast<float*>(v_descale.data_ptr());
            params.v_descale_batch_stride = v_descale.stride(0);
            params.v_descale_head_stride = v_descale.stride(1);
        } else {
            params.v_descale_ptr = nullptr;
        }
    }

    #ifdef FLASHATTENTION_DISABLE_LOCAL
    STD_TORCH_CHECK(!params.is_local, "This flash attention build does not support local attention.");
    #endif
    #ifdef FLASHATTENTION_DISABLE_SOFTCAP
    STD_TORCH_CHECK(params.softcap == 0.0, "This flash attention build does not support tanh softcapping.");
    #endif
    #ifdef FLASHATTENTION_DISABLE_SPLIT
    STD_TORCH_CHECK(params.num_splits == 1, "This flash attention build does not support splits.");
    #endif
    #ifdef FLASHATTENTION_DISABLE_PACKGQA
    STD_TORCH_CHECK(!params.pack_gqa || params.arch < 90 || (params.page_table && !params.pagedkv_tma) || params.num_splits > 1, "This flash attention build does not support pack_gqa.");
    #endif
    #ifdef FLASHATTENTION_DISABLE_PAGEDKV
    STD_TORCH_CHECK(!(params.page_table && !params.pagedkv_tma), "This flash attention build does not support paged KV.");
    #endif
    #ifdef FLASHATTENTION_DISABLE_APPENDKV
    STD_TORCH_CHECK(!k_new_.has_value(), "This flash attention build does not support appending KV.");
    #endif

    if (total_q > 0 && (total_k + params.total_knew) > 0 && num_heads_k > 0) {
        auto device_idx = torch::stable::accelerator::getCurrentDeviceIndex();
        auto stream = (cudaStream_t)torch::stable::accelerator::getCurrentStream(device_idx).id();
        run_mha_fwd(params, stream);
        if (params.num_splits > 1) {
            if (out_type == torch::headeronly::ScalarType::BFloat16) {
                // Since we want output in BF16. Otherwise fwd_combine will output to FP16
                params.is_bf16 = true;
            }
            // Unless there's seqused_q, for the purpose of attn_combine, we can just treat it as batch=1
            // and seqlen = total_q, and don't need to dispatch to Varlen there.
            // However, with dynamic split, each row needs to know which batch it belongs to
            // to read the number of splits, so we just use the varlen version of combine kernel.
            // if (is_varlen_q && !seqused_q_.has_value()) {
            // if (is_varlen_q) {
            //     params.b = 1;
            //     params.seqlen_q = total_q;
            // }
            // This will zero out the semaphore if needed
            run_mha_fwd_combine(params, stream, true /*enable_pdl*/);
        } else if (scheduler_needs_semaphore && params.skip_scheduler_metadata_computation) {
            // need to zero out the semaphore in this case
            auto slice = torch::stable::narrow(tile_count_semaphore, 0, params.tile_count_semaphore_offset, 1);
            torch::stable::zero_(slice);
        }
    } else if (total_q > 0 && num_heads_k > 0) {
        // If seqlen_k == 0, then we have an empty tensor. We need to set the output to 0.
        torch::stable::zero_(out);
        torch::stable::fill_(softmax_lse, std::numeric_limits<float>::infinity());
    }

    // return {out, softmax_lse};
    return {out, softmax_lse, out_accum, softmax_lse_accum};
}

#ifdef FLASHATTENTION_DISABLE_BACKWARD
void run_mha_bwd(Flash_bwd_params &params, cudaStream_t stream) {
    STD_TORCH_CHECK(false, "Flash-Attention was built with backward disabled");
}
#else
template <int Arch, bool Has_softcap>
void run_mha_bwd_constexpr(Flash_bwd_params &params, cudaStream_t stream) {
    if (!params.is_bf16) {
        #ifndef FLASHATTENTION_DISABLE_FP16
        #ifndef FLASHATTENTION_DISABLE_HDIM64
        if (params.d_rounded == 64) { return run_mha_bwd_<Arch, cutlass::half_t, 64, Has_softcap>(params, stream); }
        #endif
        #ifndef FLASHATTENTION_DISABLE_HDIM96
        if (params.d_rounded == 96) { return run_mha_bwd_<Arch, cutlass::half_t, 96, Has_softcap>(params, stream); }
        #endif
        #ifndef FLASHATTENTION_DISABLE_HDIM128
        if (params.d_rounded == 128) { return run_mha_bwd_<Arch, cutlass::half_t, 128, Has_softcap>(params, stream); }
        #endif
        #ifndef FLASHATTENTION_DISABLE_HDIM192
        if (params.d_rounded == 192) { return run_mha_bwd_<Arch, cutlass::half_t, 192, Has_softcap>(params, stream); }
        #endif
        #ifndef FLASHATTENTION_DISABLE_HDIM256
        if (params.d_rounded == 256) { return run_mha_bwd_<Arch, cutlass::half_t, 256, Has_softcap>(params, stream); }
        #endif
        #else
        STD_TORCH_CHECK(false, "This flash attention build does not support FP16.");
        #endif
    } else {
        #ifndef FLASHATTENTION_DISABLE_HDIM64
        if (params.d_rounded == 64) { return run_mha_bwd_<Arch, cutlass::bfloat16_t, 64, Has_softcap>(params, stream); }
        #endif
        #ifndef FLASHATTENTION_DISABLE_HDIM96
        if (params.d_rounded == 96) { return run_mha_bwd_<Arch, cutlass::bfloat16_t, 96, Has_softcap>(params, stream); }
        #endif
        #ifndef FLASHATTENTION_DISABLE_HDIM128
        if (params.d_rounded == 128) { return run_mha_bwd_<Arch, cutlass::bfloat16_t, 128, Has_softcap>(params, stream); }
        #endif
        #ifndef FLASHATTENTION_DISABLE_HDIM192
        if (params.d_rounded == 192) { return run_mha_bwd_<Arch, cutlass::bfloat16_t, 192, Has_softcap>(params, stream); }
        #endif
        #ifndef FLASHATTENTION_DISABLE_HDIM256
        if (params.d_rounded == 256) { return run_mha_bwd_<Arch, cutlass::bfloat16_t, 256, Has_softcap>(params, stream); }
        #endif
    }
}

void run_mha_bwd(Flash_bwd_params &params, cudaStream_t stream) {
        // FP16_SWITCH(!params.is_bf16, [&] {
        //     HEADDIM_SWITCH(params.d, [&] {
        //         run_mha_bwd_<elem_type, kHeadDim>(params, stream);
        //     });
        // });
    ARCH_SWITCH(params.arch, Arch, [&] {
        SOFTCAP_SWITCH(params.softcap > 0.f, Has_softcap, [&] {
            run_mha_bwd_constexpr<Arch, Has_softcap>(params, stream);
        });
    });
}
#endif


// b: batch_size
// s_q: seqlen_q
// s_k: seqlen_k
// h: num_heads
// h_k: num_heads_k
// d: head_size
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> mha_bwd(
    Tensor dout,  // (b, s_q, h, dv) or (total_q, h, dv) if there is cu_seqlens_q
    Tensor q,     // (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
    Tensor k,     // (b, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k
    Tensor v,     // (b, s_k, h_k, dv) or (total_k, h_k, dv) if there is cu_seqlens_k
    Tensor out,   // (b, s_q, h, dv) or (total_q, h, dv) if there is cu_seqlens_q
    Tensor softmax_lse,    // (b, h, s_q) or (h, total_q) if there is cu_seqlens_q
    std::optional<Tensor> dq_,   // (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
    std::optional<Tensor> dk_,   // (b, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k
    std::optional<Tensor> dv_,   // (b, s_k, h_k, dv) or (total_k, h_k, dv) if there is cu_seqlens_k
    std::optional<Tensor> cu_seqlens_q_,   // b+1
    std::optional<Tensor> cu_seqlens_k_,   // b+1
    std::optional<Tensor> seqused_q_, // b. If given, only this many elements of each batch element's queries and outputs are used.
    std::optional<Tensor> seqused_k_, // b. If given, only this many elements of each batch element's keys are used.
    std::optional<int64_t> max_seqlen_q_,
    std::optional<int64_t> max_seqlen_k_,
    std::optional<double> softmax_scale_,
    bool is_causal,
    int64_t window_size_left,
    int64_t window_size_right,
    double softcap,
    bool deterministic,
    int64_t sm_margin
) {

    #ifdef FLASHATTENTION_DISABLE_BACKWARD
        STD_TORCH_CHECK(false, "This flash attention build does not support backward.");
    #endif

    auto dprops = get_device_prop();
    bool is_sm8x = dprops->major >= 8;
    STD_TORCH_CHECK(is_sm8x, "FlashAttention only supports Ampere GPUs or newer.");

    auto q_type = q.scalar_type();
    STD_TORCH_CHECK(q_type == torch::headeronly::ScalarType::Half || q_type == torch::headeronly::ScalarType::BFloat16,
                "FlashAttention only support fp16 and bf16 data type");
    STD_TORCH_CHECK(k.scalar_type() == q_type, "query and key must have the same dtype");
    STD_TORCH_CHECK(v.scalar_type() == q_type, "query and value must have the same dtype");
    STD_TORCH_CHECK(out.scalar_type() == q_type, "query and out must have the same dtype");
    STD_TORCH_CHECK(dout.scalar_type() == q_type, "query and dout must have the same dtype");

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
    CHECK_DEVICE(out); CHECK_DEVICE(dout); CHECK_DEVICE(softmax_lse);

    STD_TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    STD_TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    STD_TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    STD_TORCH_CHECK(out.stride(-1) == 1, "out tensor must have contiguous last dimension");
    STD_TORCH_CHECK(dout.stride(-1) == 1, "dout tensor must have contiguous last dimension");

    Tensor cu_seqlens_q;
    bool const is_varlen_q = cu_seqlens_q_.has_value();
    if (is_varlen_q) {
        cu_seqlens_q = cu_seqlens_q_.value();
        CHECK_DEVICE(cu_seqlens_q); CHECK_CONTIGUOUS(cu_seqlens_q);
        STD_TORCH_CHECK(cu_seqlens_q.scalar_type() == torch::headeronly::ScalarType::Int, "cu_seqlens_q must have dtype torch.int32");
        STD_TORCH_CHECK(max_seqlen_q_.has_value(), "max_seqlen_q must be provided if cu_seqlens_q is provided");
    }
    Tensor cu_seqlens_k;
    bool const is_varlen_k = cu_seqlens_k_.has_value();
    if (is_varlen_k) {
        cu_seqlens_k = cu_seqlens_k_.value();
        CHECK_DEVICE(cu_seqlens_k); CHECK_CONTIGUOUS(cu_seqlens_k);
        STD_TORCH_CHECK(cu_seqlens_k.scalar_type() == torch::headeronly::ScalarType::Int, "cu_seqlens_k must have dtype torch.int32");
        STD_TORCH_CHECK(max_seqlen_k_.has_value(), "max_seqlen_k must be provided if cu_seqlens_k is provided");
    }
    // This is what we will template on
    bool const is_varlen = is_varlen_q || is_varlen_k || seqused_q_.has_value() || seqused_k_.has_value();
    #ifdef FLASHATTENTION_DISABLE_VARLEN
        STD_TORCH_CHECK(!is_varlen, "This flash attention build does not support varlen.");
    #endif

    // auto const sizes = q.sizes();
    int const batch_size = !is_varlen_q ? q.size(0) : cu_seqlens_q.size(0) - 1;
    int const seqlen_q = !is_varlen_q ? q.size(1) : max_seqlen_q_.value();
    int const total_q = !is_varlen_q ? batch_size * q.size(1) : q.size(0);
    int const num_heads = q.size(-2);
    int const head_size = q.size(-1);
    int const head_size_v = v.size(-1);
    int const seqlen_k = !is_varlen_k ? k.size(1) : max_seqlen_k_.value();
    int const total_k = !is_varlen_k ? batch_size * k.size(1) : k.size(0);
    int const num_heads_k = k.size(-2);
    STD_TORCH_CHECK(head_size % 8 == 0, "head_size should be a multiple of 8");
    STD_TORCH_CHECK(head_size_v % 8 == 0, "head_size_v should be a multiple of 8");
    int const max_headdim = get_max_headdim();
    STD_TORCH_CHECK(std::max(head_size, head_size_v) <= max_headdim, "FlashAttention forward only supports head dimension at most " + std::to_string(max_headdim));
    STD_TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");
    double softmax_scale = 1.0 / sqrt(double(head_size));
    if (softmax_scale_.has_value()) {
        softmax_scale = softmax_scale_.value();
    }

    // This needs to go before kBlockM & kBlockN since we rely on the correct window_size and is_causal to set kBlockM
    if (window_size_left >= seqlen_k - 1) { window_size_left = -1; }
    if (window_size_right >= seqlen_q - 1) { window_size_right = -1; }
    if (is_causal) { window_size_right = 0; }
    // There's a case where is_causal=false, window_size=(-1, 0). Then set_params_bprop will set params.is_causal=true.
    // If we don't have is_causal here matching params.is_causal, we might get the wrong kBlockM (and cause IMA).
    is_causal = window_size_left < 0 && window_size_right == 0;

    int const arch = dprops->major * 10 + dprops->minor;
    int const head_size_rounded = round_up_headdim(std::max(head_size, head_size_v));
    int const head_size_v_rounded = head_size_rounded;
    STD_TORCH_CHECK(!deterministic || head_size_rounded < 256, "Deterministic backward not supported for hdim 256.");
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
        CHECK_SHAPE(out, batch_size, seqlen_q, num_heads, head_size_v);
        CHECK_SHAPE(dout, batch_size, seqlen_q, num_heads, head_size_v);
    } else {
        CHECK_SHAPE(q, total_q, num_heads, head_size);
        CHECK_SHAPE(out, total_q, num_heads, head_size_v);
        CHECK_SHAPE(dout, total_q, num_heads, head_size_v);
        CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
    }
    if (!is_varlen_k) {
        CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size);
        CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size_v);
    } else {
        CHECK_SHAPE(k, total_k, num_heads_k, head_size);
        CHECK_SHAPE(v, total_k, num_heads_k, head_size_v);
        CHECK_SHAPE(cu_seqlens_k, batch_size + 1);
    }

    if (seqused_q_.has_value()){
        auto seqused_q = seqused_q_.value();
        STD_TORCH_CHECK(seqused_q.scalar_type() == torch::headeronly::ScalarType::Int, "seqused_q must have dtype int32");
        CHECK_DEVICE(seqused_q); CHECK_CONTIGUOUS(seqused_q);
        CHECK_SHAPE(seqused_q, batch_size);
    }
    if (seqused_k_.has_value()){
        auto seqused_k = seqused_k_.value();
        STD_TORCH_CHECK(seqused_k.scalar_type() == torch::headeronly::ScalarType::Int, "seqused_k must have dtype int32");
        CHECK_DEVICE(seqused_k); CHECK_CONTIGUOUS(seqused_k);
        CHECK_SHAPE(seqused_k, batch_size);
    }

    Tensor dq, dk, dv;
    if (dq_.has_value()) {
        dq = dq_.value();
        STD_TORCH_CHECK(dq.scalar_type() == q_type, "dq must have the same dtype as q");
        CHECK_DEVICE(dq);
        STD_TORCH_CHECK(dq.stride(-1) == 1, "dq must have contiguous last dimension");
        if (!is_varlen_q) {
            CHECK_SHAPE(dq, batch_size, seqlen_q, num_heads, head_size);
        } else {
            CHECK_SHAPE(dq, total_q, num_heads, head_size);
        }
    } else {
        dq = torch::stable::empty_like(q);
    }
    if (dk_.has_value()) {
        dk = dk_.value();
        STD_TORCH_CHECK(dk.scalar_type() == q_type, "dk must have the same dtype as q");
        CHECK_DEVICE(dk);
        STD_TORCH_CHECK(dk.stride(-1) == 1, "dk must have contiguous last dimension");
        if (!is_varlen_k) {
            CHECK_SHAPE(dk, batch_size, seqlen_k, num_heads_k, head_size);
        } else {
            CHECK_SHAPE(dk, total_k, num_heads_k, head_size);
        }
    } else {
        dk = torch::stable::empty_like(k);
    }
    if (dv_.has_value()) {
        dv = dv_.value();
        STD_TORCH_CHECK(dv.scalar_type() == q_type, "dv must have the same dtype as q");
        CHECK_DEVICE(dv);
        STD_TORCH_CHECK(dv.stride(-1) == 1, "dv must have contiguous last dimension");
        if (!is_varlen_k) {
            CHECK_SHAPE(dv, batch_size, seqlen_k, num_heads_k, head_size_v);
        } else {
            CHECK_SHAPE(dv, total_k, num_heads_k, head_size_v);
        }
    } else {
        dv = torch::stable::empty_like(v);
    }

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    auto device_guard = make_device_guard(q);

    // auto opts = q.options();
    // Need softmax_d to have total_q_padded_rounded since we want its address to be aligned by 16/8 bytes for TMA / LDG.64
    Tensor softmax_d, softmax_lse_log2;
    if (!is_varlen) {
        // Need softmax_d to have seqlen_q_rounded since we want its address to be aligned by 16/8 bytes for TMA / LDG.64
        softmax_d = torch::stable::new_empty(q, {batch_size, num_heads, seqlen_q_rounded}, std::make_optional(torch::headeronly::ScalarType::Float));
        softmax_lse_log2 = torch::stable::new_empty(q, {batch_size, num_heads, seqlen_q_rounded}, std::make_optional(torch::headeronly::ScalarType::Float));
    } else {
        softmax_d = torch::stable::new_empty(q, {num_heads, total_q_padded_rounded}, std::make_optional(torch::headeronly::ScalarType::Float));
        softmax_lse_log2 = torch::stable::new_empty(q, {num_heads, total_q_padded_rounded}, std::make_optional(torch::headeronly::ScalarType::Float));
    }
    Tensor dq_accum, dk_accum, dv_accum;
    if (!is_varlen) {
        dq_accum = torch::stable::new_empty(q, {batch_size, num_heads, seqlen_q_rounded * head_size_rounded}, std::make_optional(torch::headeronly::ScalarType::Float));
    } else {
        dq_accum = torch::stable::new_empty(q, {num_heads, total_q_padded_rounded * head_size_rounded}, std::make_optional(torch::headeronly::ScalarType::Float));
    }
    if (num_heads_k != num_heads) {  // MQA / GQA
        if (!is_varlen) {
            dk_accum = torch::stable::new_empty(q, {batch_size, num_heads_k, seqlen_k_rounded * head_size_rounded}, std::make_optional(torch::headeronly::ScalarType::Float));
            dk_accum = torch::stable::fill_(dk_accum, 0.0);
            dv_accum = torch::stable::new_empty(q, {batch_size, num_heads_k, seqlen_k_rounded * head_size_v_rounded}, std::make_optional(torch::headeronly::ScalarType::Float));
            dv_accum = torch::stable::fill_(dv_accum, 0.0);
        } else {
            dk_accum = torch::stable::new_empty(q, {num_heads_k, total_k_padded_rounded, head_size_rounded}, std::make_optional(torch::headeronly::ScalarType::Float));
            dk_accum = torch::stable::fill_(dk_accum, 0.0);
            dv_accum = torch::stable::new_empty(q, {num_heads_k, total_k_padded_rounded, head_size_v_rounded}, std::make_optional(torch::headeronly::ScalarType::Float));
            dv_accum = torch::stable::fill_(dv_accum, 0.0);
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
                     0,  // attention_chunk
                     softcap,
                     deterministic,
                     sm_margin);
    params.total_q = total_q;
    params.total_k = total_k;
    params.softmax_lse_log2_ptr = softmax_lse_log2.data_ptr();
    params.dv = head_size_v;
    params.dv_rounded = head_size_v_rounded;

    // auto tile_count_semaphore = (params.is_causal || params.is_local) ? torch::zeros({1}, opts.dtype(torch::headeronly::ScalarType::Int)) : torch::empty({1}, opts.dtype(torch::headeronly::ScalarType::Int));
    // params.tile_count_semaphore = static_cast<int*>(tile_count_semaphore.data_ptr());
    // Will be zero'ed out in the backward preprocess kernel
    Tensor dq_semaphore = torch::stable::new_empty(q, {(seqlen_q + kBlockM - 1) / kBlockM, batch_size, num_heads}, std::make_optional(torch::headeronly::ScalarType::Int));
    params.dq_semaphore = static_cast<int*>(dq_semaphore.data_ptr());
    if (num_heads_k != num_heads && params.deterministic) {
        // TODO: maybe also zero'ed out dk_semaphore and dv_semaphore in the backward preprocess kernel
        Tensor dk_semaphore = torch::stable::new_zeros(q, {(seqlen_k + kBlockN - 1) / kBlockN, batch_size, num_heads_k}, std::make_optional(torch::headeronly::ScalarType::Int));
        Tensor dv_semaphore = torch::stable::new_zeros(q, {(seqlen_k + kBlockN - 1) / kBlockN, batch_size, num_heads_k}, std::make_optional(torch::headeronly::ScalarType::Int));
        params.dk_semaphore = static_cast<int*>(dk_semaphore.data_ptr());
        params.dv_semaphore = static_cast<int*>(dv_semaphore.data_ptr());
    }

    #ifdef FLASHATTENTION_DISABLE_LOCAL
    STD_TORCH_CHECK(!params.is_local, "This flash attention build does not support local attention.");
    #endif
    #ifdef FLASHATTENTION_DISABLE_SOFTCAP
    STD_TORCH_CHECK(params.softcap == 0.0, "This flash attention build does not support tanh softcapping.");
    #endif

    if (total_q > 0 && total_k > 0 && num_heads_k > 0) {
        auto device_idx = torch::stable::accelerator::getCurrentDeviceIndex();
        auto stream = (cudaStream_t)torch::stable::accelerator::getCurrentStream(device_idx).id();
        run_mha_bwd(params, stream);
    } else if (total_k > 0 && num_heads_k > 0) {
        // If seqlen_q == 0, then we have an empty tensor. We need to set the output to 0.
        torch::stable::zero_(dk);
        torch::stable::zero_(dv);
        torch::stable::zero_(softmax_d);
    } else if (total_q > 0 && num_heads_k > 0) {
        torch::stable::zero_(dq);
        torch::stable::zero_(softmax_d);
    }

    return { dq, dk, dv, softmax_d, softmax_lse_log2, dq_accum, dk_accum, dv_accum };
}

std::tuple<Tensor, Tensor>
mha_combine(Tensor out_partial,         // num_splits x batch_size x seqlen x num_heads x head_size
            Tensor lse_partial,         // num_splits x batch_size x seqlen x num_heads
            std::optional<Tensor> out_,        // batch_size x seqlen x num_heads x head_size
            std::optional<torch::headeronly::ScalarType> out_dtype_
            ) {

    auto dprops = get_device_prop();
    bool is_sm8x = dprops->major >= 8;
    STD_TORCH_CHECK(is_sm8x, "Attention combine function only supports Ampere GPUs or newer.");

    auto out_partial_type = out_partial.scalar_type();
    STD_TORCH_CHECK(out_partial_type == torch::headeronly::ScalarType::Float, "Attention combine function only support fp32 data type");
    STD_TORCH_CHECK(lse_partial.scalar_type() == torch::headeronly::ScalarType::Float, "Attention combine function only support fp32 data type");

    CHECK_DEVICE(out_partial); CHECK_DEVICE(lse_partial);

    STD_TORCH_CHECK(out_partial.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    STD_TORCH_CHECK(lse_partial.stride(-2) == 1, "LSE tensor must be contiguous in the seqlen dimension");

    // const auto sizes = out_partial.sizes();

    const int num_splits = out_partial.size(0);
    const int batch_size = out_partial.size(1);
    const int seqlen = out_partial.size(2);
    const int num_heads = out_partial.size(3);
    const int head_size_og = out_partial.size(4);
    STD_TORCH_CHECK(num_splits <= 256, "FlashAttention combine only supports num_splits at most 256");

    CHECK_SHAPE(out_partial, num_splits, batch_size, seqlen, num_heads, head_size_og);
    CHECK_SHAPE(lse_partial, num_splits, batch_size, seqlen, num_heads);

    int const alignment = 4;
    Tensor out_partial_padded;
    auto pad = [](Tensor x, int alignment) {
        return x.size(-1) % alignment == 0 ? x : torch::stable::pad(x, {0, alignment - x.size(-1) % alignment});
    };
    out_partial_padded = pad(out_partial, alignment);

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size = round_multiple(head_size_og, alignment);

    // auto opts = out_partial.options();
    torch::headeronly::ScalarType out_type = out_dtype_.value_or(out_partial.scalar_type());
    STD_TORCH_CHECK(out_type == torch::headeronly::ScalarType::Float || out_type == torch::headeronly::ScalarType::BFloat16 || out_type == torch::headeronly::ScalarType::Half, "Output type must be FP32, FP16 or BF16");
    Tensor out;
    if (out_.has_value()) {
        out = out_.value();
        STD_TORCH_CHECK(out.scalar_type() == out_type);
        CHECK_DEVICE(out);
        STD_TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        CHECK_SHAPE(out, batch_size, seqlen, num_heads, head_size_og);
        if (head_size_og % alignment != 0) {
            out = torch::stable::new_empty(out_partial, {batch_size, seqlen, num_heads, head_size}, std::make_optional(out_type));
        }
    } else {
        out = torch::stable::new_empty(out_partial, {batch_size, seqlen, num_heads, head_size}, std::make_optional(out_type));
    }

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    auto device_guard = make_device_guard(out_partial);

    auto softmax_lse = torch::stable::new_empty(out_partial, {batch_size, num_heads, seqlen}, std::make_optional(torch::headeronly::ScalarType::Float));
    softmax_lse = torch::stable::transpose(softmax_lse, 1, 2);

    Flash_fwd_params params {};  // Need to reset the params to set everything to zero
    params.is_fp32 = out_type == torch::headeronly::ScalarType::Float;
    params.is_bf16 = out_type == torch::headeronly::ScalarType::BFloat16;
    params.oaccum_ptr = out_partial_padded.data_ptr();
    params.softmax_lseaccum_ptr = lse_partial.data_ptr();
    params.o_ptr = out.data_ptr();
    params.softmax_lse_ptr = softmax_lse.data_ptr();
    params.b = batch_size;
    params.h = num_heads;
    params.seqlen_q = seqlen;
    params.dv = head_size;
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
    params.arch = dprops->major * 10 + dprops->minor;

    if (seqlen > 0 && batch_size > 0) {
        auto device_idx = torch::stable::accelerator::getCurrentDeviceIndex();
        auto stream = (cudaStream_t)torch::stable::accelerator::getCurrentStream(device_idx).id();
        run_mha_fwd_combine(params, stream, false /*enable_pdl*/);
    }

    Tensor out_padded = out;
    if (head_size_og % alignment != 0) {
        out = torch::stable::narrow(out, -1, 0, head_size_og);
        // if (out_.has_value()) { out_.value().copy_(out); }
    }

    return {out, softmax_lse};
}

void boxed_mha_fwd(
    StableIValue* stack,
    uint64_t num_args,
    uint64_t num_outputs
) {
    auto q = to<Tensor>(stack[0]);
    auto k = to<Tensor>(stack[1]);
    auto v = to<Tensor>(stack[2]);
    auto k_new = to<std::optional<Tensor>>(stack[3]);
    auto v_new = to<std::optional<Tensor>>(stack[4]);
    auto q_v = to<std::optional<Tensor>>(stack[5]);
    auto out = to<std::optional<Tensor>>(stack[6]);
    auto cu_seqlens_q = to<std::optional<Tensor>>(stack[7]);
    auto cu_seqlens_k = to<std::optional<Tensor>>(stack[8]);
    auto cu_seqlens_k_new = to<std::optional<Tensor>>(stack[9]);
    auto seqused_q = to<std::optional<Tensor>>(stack[10]);
    auto seqused_k = to<std::optional<Tensor>>(stack[11]);
    auto max_seqlen_q = to<std::optional<int64_t>>(stack[12]);
    auto max_seqlen_k = to<std::optional<int64_t>>(stack[13]);
    auto page_table = to<std::optional<Tensor>>(stack[14]);
    auto kv_batch_idx = to<std::optional<Tensor>>(stack[15]);
    auto leftpad_k = to<std::optional<Tensor>>(stack[16]);
    auto rotary_cos = to<std::optional<Tensor>>(stack[17]);
    auto rotary_sin = to<std::optional<Tensor>>(stack[18]);
    auto seqlens_rotary = to<std::optional<Tensor>>(stack[19]);
    auto q_descale = to<std::optional<Tensor>>(stack[20]);
    auto k_descale = to<std::optional<Tensor>>(stack[21]);
    auto v_descale = to<std::optional<Tensor>>(stack[22]);
    auto softmax_scale = to<std::optional<double>>(stack[23]);
    auto is_causal = to<bool>(stack[24]);
    auto window_size_left = to<int64_t>(stack[25]);
    auto window_size_right = to<int64_t>(stack[26]);
    auto attention_chunk = to<int64_t>(stack[27]);
    auto softcap = to<double>(stack[28]);
    auto is_rotary_interleaved = to<bool>(stack[29]);
    auto scheduler_metadata = to<std::optional<Tensor>>(stack[30]);
    auto num_splits = to<int64_t>(stack[31]);
    auto pack_gqa = to<std::optional<bool>>(stack[32]);
    auto sm_margin = to<int64_t>(stack[33]);

    auto [out_, softmax_lse, out_accum, softmax_lse_accum] = mha_fwd(q, k, v, k_new, v_new, q_v, out, cu_seqlens_q, cu_seqlens_k, cu_seqlens_k_new, seqused_q, seqused_k, max_seqlen_q, max_seqlen_k, page_table, kv_batch_idx, leftpad_k, rotary_cos, rotary_sin, seqlens_rotary, q_descale, k_descale, v_descale, softmax_scale, is_causal, window_size_left, window_size_right, attention_chunk, softcap, is_rotary_interleaved, scheduler_metadata, num_splits, pack_gqa, sm_margin);


    stack[0] = from(out_);
    stack[1] = from(softmax_lse);
    stack[2] = from(out_accum);
    stack[3] = from(softmax_lse_accum);
}

void boxed_mha_bwd(
    StableIValue* stack,
    uint64_t num_args,
    uint64_t num_outputs
) {
    auto dout = to<Tensor>(stack[0]);
    auto q = to<Tensor>(stack[1]);
    auto k = to<Tensor>(stack[2]);
    auto v = to<Tensor>(stack[3]);
    auto out = to<Tensor>(stack[4]);
    auto softmax_lse = to<Tensor>(stack[5]);
    auto dq = to<std::optional<Tensor>>(stack[6]);
    auto dk = to<std::optional<Tensor>>(stack[7]);
    auto dv = to<std::optional<Tensor>>(stack[8]);
    auto cu_seqlens_q = to<std::optional<Tensor>>(stack[9]);
    auto cu_seqlens_k = to<std::optional<Tensor>>(stack[10]);
    auto seqused_q = to<std::optional<Tensor>>(stack[11]);
    auto seqused_k = to<std::optional<Tensor>>(stack[12]);
    auto max_seqlen_q = to<std::optional<int64_t>>(stack[13]);
    auto max_seqlen_k = to<std::optional<int64_t>>(stack[14]);
    auto softmax_scale = to<std::optional<double>>(stack[15]);
    auto is_causal = to<bool>(stack[16]);
    auto window_size_left = to<int64_t>(stack[17]);
    auto window_size_right = to<int64_t>(stack[18]);
    auto softcap = to<double>(stack[19]);
    auto deterministic = to<bool>(stack[20]);
    auto sm_margin = to<int64_t>(stack[21]);

    auto [dq_, dk_, dv_, softmax_d, softmax_lse_log2, dq_accum, dk_accum, dv_accum] = mha_bwd(dout, q, k, v, out, softmax_lse, dq, dk, dv, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, max_seqlen_q, max_seqlen_k, softmax_scale, is_causal, window_size_left, window_size_right, softcap, deterministic, sm_margin);

    stack[0] = from(dq_);
    stack[1] = from(dk_);
    stack[2] = from(dv_);
    stack[3] = from(softmax_d);
    stack[4] = from(softmax_lse_log2);
    stack[5] = from(dq_accum);
    stack[6] = from(dk_accum);
    stack[7] = from(dv_accum);
}

void boxed_mha_combine(
    StableIValue* stack,
    uint64_t num_args,
    uint64_t num_outputs
) {
    auto out_partial = to<Tensor>(stack[0]);
    auto lse_partial = to<Tensor>(stack[1]);
    auto out = to<std::optional<Tensor>>(stack[2]);
    auto out_dtype = to<std::optional<torch::headeronly::ScalarType>>(stack[3]);

    auto [out_, softmax_lse] = mha_combine(out_partial, lse_partial, out, out_dtype);

    stack[0] = from(out_);
    stack[1] = from(softmax_lse);
}

void boxed_mha_fwd_get_scheduler_metadata(
    StableIValue* stack,
    uint64_t num_args,
    uint64_t num_outputs
) {
    auto batch_size = to<int64_t>(stack[0]);
    auto max_seqlen_q = to<int64_t>(stack[1]);
    auto max_seqlen_k = to<int64_t>(stack[2]);
    auto num_heads = to<int64_t>(stack[3]);
    auto num_heads_k = to<int64_t>(stack[4]);
    auto headdim = to<int64_t>(stack[5]);
    auto headdim_v = to<int64_t>(stack[6]);
    auto qkv_dtype = to<torch::headeronly::ScalarType>(stack[7]);
    auto seqused_k = to<Tensor>(stack[8]);
    auto cu_seqlens_q = to<std::optional<Tensor>>(stack[9]);
    auto cu_seqlens_k = to<std::optional<Tensor>>(stack[10]);
    auto cu_seqlens_k_new = to<std::optional<Tensor>>(stack[11]);
    auto seqused_q = to<std::optional<Tensor>>(stack[12]);
    auto leftpad_k = to<std::optional<Tensor>>(stack[13]);
    auto page_size = to<std::optional<int64_t>>(stack[14]);
    auto max_seqlen_k_new = to<int64_t>(stack[15]);
    auto is_causal = to<bool>(stack[16]);
    auto window_size_left = to<int64_t>(stack[17]);
    auto window_size_right = to<int64_t>(stack[18]);
    auto attention_chunk = to<int64_t>(stack[19]);
    auto has_softcap = to<bool>(stack[20]);
    auto num_splits = to<int64_t>(stack[21]);
    auto pack_gqa = to<std::optional<bool>>(stack[22]);
    auto sm_margin = to<int64_t>(stack[23]);

    auto scheduler_metadata = mha_fwd_get_scheduler_metadata(batch_size, max_seqlen_q, max_seqlen_k, num_heads, num_heads_k, headdim, headdim_v, qkv_dtype, seqused_k, cu_seqlens_q, cu_seqlens_k, cu_seqlens_k_new, seqused_q, leftpad_k, page_size, max_seqlen_k_new, is_causal, window_size_left, window_size_right, attention_chunk, has_softcap, num_splits, pack_gqa, sm_margin);

    stack[0] = from(scheduler_metadata);
}

STABLE_TORCH_LIBRARY(flash_attn_3, m) {
    m.def("fwd("
        "Tensor q,"
        "Tensor k,"
        "Tensor v,"
        "Tensor(k_new!)? k_new = None,"
        "Tensor(v_new!)? v_new = None,"
        "Tensor? q_v = None,"
        "Tensor(out!)? out = None,"
        "Tensor? cu_seqlens_q = None,"
        "Tensor? cu_seqlens_k = None,"
        "Tensor? cu_seqlens_k_new = None,"
        "Tensor? seqused_q = None,"
        "Tensor? seqused_k = None,"
        "int? max_seqlen_q = None,"
        "int? max_seqlen_k = None,"
        "Tensor? page_table = None,"
        "Tensor? kv_batch_idx = None,"
        "Tensor? leftpad_k = None,"
        "Tensor? rotary_cos = None,"
        "Tensor? rotary_sin = None,"
        "Tensor? seqlens_rotary = None,"
        "Tensor? q_descale = None,"
        "Tensor? k_descale = None,"
        "Tensor? v_descale = None,"
        "float? softmax_scale = None,"
        "bool is_causal = False,"
        "int window_size_left = -1,"
        "int window_size_right = -1,"
        "int attention_chunk = 0,"
        "float softcap = 0.0,"
        "bool is_rotary_interleaved = False,"
        "Tensor? scheduler_metadata = None,"
        "int num_splits = 0,"
        "bool? pack_gqa = None,"
        "int sm_margin = 0) -> (Tensor(out!), Tensor, Tensor, Tensor)");
    m.def("bwd("
        "Tensor dout,"
        "Tensor q,"
        "Tensor k,"
        "Tensor v,"
        "Tensor out,"
        "Tensor softmax_lse,"
        "Tensor(dq!)? dq = None,"
        "Tensor(dk!)? dk = None,"
        "Tensor(dv!)? dv = None,"
        "Tensor? cu_seqlens_q = None,"
        "Tensor? cu_seqlens_k = None,"
        "Tensor? seqused_q = None,"
        "Tensor? seqused_k = None,"
        "int? max_seqlen_q = None,"
        "int? max_seqlen_k = None,"
        "float? softmax_scale = None,"
        "bool is_causal = False,"
        "int window_size_left = -1,"
        "int window_size_right = -1,"
        "float softcap = 0.0,"
        "bool deterministic = False,"
        "int sm_margin = 0) -> (Tensor(dq!), Tensor(dk!), Tensor(dv!), Tensor, Tensor, Tensor, Tensor, Tensor)");
    m.def("fwd_combine("
        "Tensor out_partial,"
        "Tensor lse_partial,"
        "Tensor(out!)? out = None,"
        "ScalarType? out_dtype = None) -> (Tensor(out!), Tensor)");
    m.def("get_scheduler_metadata("
        "int batch_size,"
        "int max_seqlen_q,"
        "int max_seqlen_k,"
        "int num_heads,"
        "int num_heads_k,"
        "int headdim,"
        "int headdim_v,"
        "ScalarType qkv_dtype,"
        "Tensor seqused_k,"
        "Tensor? cu_seqlens_q = None,"
        "Tensor? cu_seqlens_k = None,"
        "Tensor? cu_seqlens_k_new = None,"
        "Tensor? seqused_q = None,"
        "Tensor? leftpad_k = None,"
        "int? page_size = None,"
        "int max_seqlen_k_new = 0,"
        "bool is_causal = False,"
        "int window_size_left = -1,"
        "int window_size_right = -1,"
        "int attention_chunk = 0,"
        "bool has_softcap = False,"
        "int num_splits = 0,"
        "bool? pack_gqa = None,"
        "int sm_margin = 0) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(flash_attn_3, CUDA, m) {
    m.impl("fwd", &boxed_mha_fwd);
    m.impl("bwd", &boxed_mha_bwd);
    m.impl("fwd_combine", &boxed_mha_combine);
    m.impl("get_scheduler_metadata", &boxed_mha_fwd_get_scheduler_metadata);
}
