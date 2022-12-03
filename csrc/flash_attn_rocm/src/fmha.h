
#pragma once

//#include <cuda.h>
#include <vector>
#include <iostream>

//#ifdef OLD_GENERATOR_PATH
//#include <ATen/CUDAGeneratorImpl.h>
//#else
//#include <ATen/cuda/CUDAGeneratorImpl.h>
//#endif
//
//#include <ATen/cuda/CUDAGraphsUtils.cuh>

#include <fmha_utils.h>

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/tensor_specialization.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_gemm_softmax_gemm_permute_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_batched_gemm.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_softmax.hpp"


constexpr int TOTAL_DIM = 0;
constexpr int H_DIM = 1;
constexpr int D_DIM = 2;

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Qkv_params {
    // The QKV matrices.
    // void *__restrict__ q_ptr;
    // void *__restrict__ k_ptr;
    // void *__restrict__ v_ptr;
    std::vector<const void*> q_ptr; //changed to ck input type
    std::vector<const void*> k_ptr;
    std::vector<const void*> v_ptr;

    // The stride between rows of the Q, K and V matrices.
    // size_t qkv_stride_in_elts;
    // size_t qkv_stride_in_bytes;
    // TD [2022-04-16]: We're using 32-bit indexing to save registers.
    // The code probably won't work for arrays larger than 2GB.
    uint32_t q_row_stride_in_elts;
    uint32_t k_row_stride_in_elts;
    uint32_t v_row_stride_in_elts;
    uint32_t q_head_stride_in_elts;
    uint32_t k_head_stride_in_elts;
    uint32_t v_head_stride_in_elts;

    // The number of heads.
    int h;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct FMHA_fprop_params : public Qkv_params {

    // The O matrix (output).
    // void * __restrict__ o_ptr;
    std::vector<void*> o_ptr;

    // The stride between rows of O.
    // size_t o_stride_in_elts;
    // size_t o_stride_in_bytes;
    uint32_t o_row_stride_in_elts;
    uint32_t o_head_stride_in_elts;
    uint32_t o_tmp_row_stride_in_elts;
    uint32_t o_tmp_head_stride_in_elts;

    // The pointer to the O_tmp matrix, which holds O intermediate value during
    // the loop;
    void *__restrict__ o_tmp_ptr;

    // The pointer to the S matrix.
    void * __restrict__ s_ptr;
    // The stride between rows of the S matrix.
    // int64_t s_stride_in_bytes;
    uint32_t s_stride_in_bytes;

    // The pointer to the softmax sum.
    void * __restrict__ softmax_lse_ptr;

    // The dimensions.
    int b, seqlen_q, seqlen_k, d;

    // The scaling factors for the kernel.
    float scale_bmm1f;
    uint32_t scale_bmm1;

    // array of length b+1 holding starting offset of each sequence.
    int * __restrict__ cu_seqlens_q;
    int * __restrict__ cu_seqlens_k;

    int *__restrict__ blockmask;

    // The dropout probability (probability of keeping an activation).
    float p_dropout;
    uint32_t p_dropout_in_uint;
    uint16_t p_dropout_in_uint16_t;

    // Scale factor of 1 / (1 - p_dropout).
    float rp_dropout;
    float scale_bmm1_rp_dropout;

    // Scale factor of 1 / (1 - p_dropout), in half2.
    uint32_t scale_dropout;

    // Random state.
    // at::PhiloxCudaState philox_args;

    bool is_bf16;
    bool is_causal;

    int num_splits; // How many SMs per attention matrix.
};


template<typename Kernel_params>
struct Launch_params{
    Launch_params(hipDeviceProp_t * props_,
                  hipStream_t stream_,
                  bool is_dropout_,
                  bool return_softmax_)
        : elts_per_thread(0)
        , props(props_)
        , stream(stream_)
        , is_dropout(is_dropout_)
        , return_softmax(return_softmax_) {
    }

    size_t elts_per_thread;

    hipDeviceProp_t * props;

    hipStream_t stream;

    bool is_dropout;
    bool return_softmax;

    Kernel_params params;
    int num_full_heads;
    int num_main_groups;
    int heads_last_wave;
    int main_steps;
    int rest_steps;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

void run_fmha_fp16_bf16_gfx90a(Launch_params<FMHA_fprop_params> &launch_params);

//void run_fmha_dgrad_fp16_gfx90a(FMHA_dgrad_params &params, hipStream_t stream, const bool configure);

//void run_fmha_block_fp16_gfx90a(Launch_params<FMHA_fprop_params> &launch_params, const bool configure);

//void run_fmha_block_dgrad_fp16_gfx90a(const FMHA_dgrad_params &params, hipStream_t stream);
