/******************************************************************************
 * Copyright (c) 2011-2021, NVIDIA CORPORATION.  All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "static_switch.h"
#include "fp16_switch.h"
#include "fmha.h"
#include "fmha_fprop_kernel_1xN.h"

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Return_softmax, bool Need_attn_mask, bool Need_attn_bias>
__global__ void fmha_fprop_fp16_sm80_loop_kernel(FMHA_fprop_params params) {
    fmha::device_1xN_loop<Kernel_traits, Is_dropout, Is_causal, Return_softmax, Need_attn_mask, Need_attn_bias>(params);
}

template<typename Kernel_traits>
void run_fmha_fp16_sm80_loop_(Launch_params<FMHA_fprop_params> &launch_params,
                              const bool configure) {
    constexpr int blocksize_c = Kernel_traits::Cta_tile_p::N;
    const int loop_steps = (launch_params.params.seqlen_k + blocksize_c - 1) / blocksize_c;

    if (configure) {
        using Mma_tile_p = fmha::Hmma_tile<typename Kernel_traits::Cta_tile_p>;
        constexpr int M = Kernel_traits::Cta_tile_p::M;
        size_t STEPS = (launch_params.params.seqlen_q + M - 1) / M;
        constexpr size_t MMAS_M = Mma_tile_p::MMAS_M;
        constexpr size_t MMAS_N = Mma_tile_p::MMAS_N;
        size_t elts_per_head = STEPS * MMAS_M * MMAS_N * 8 * loop_steps;
        launch_params.elts_per_thread = elts_per_head;
        return;
    }

    constexpr int smem_size_softmax_lse = Kernel_traits::Smem_dp_sum::BYTES_PER_TILE;
    // Don't need smem_size_softmax_lse if we're not looping
    const int smem_size = fmha::get_dynamic_smem_size<Kernel_traits>()
        + (loop_steps > 1 ? smem_size_softmax_lse : 0);

    bool has_attn_mask = !(launch_params.params.attn_mask_ptr == nullptr);
    bool has_attn_bias = !(launch_params.params.attn_bias_ptr == nullptr);

    if (has_attn_mask) 
    {
        if (has_attn_bias) {
            // Work-around for gcc 7. It doesn't like nested BOOL_SWITCH.
            // https://github.com/kokkos/kokkos-kernels/issues/349
            // https://github.com/HazyResearch/flash-attention/issues/21
            BOOL_SWITCH(launch_params.is_dropout, IsDropoutConst, [&] {
                auto kernel = launch_params.params.is_causal
                    ? (launch_params.return_softmax
                    ? &fmha_fprop_fp16_sm80_loop_kernel<Kernel_traits, IsDropoutConst, true, true, true, true>
                    : &fmha_fprop_fp16_sm80_loop_kernel<Kernel_traits, IsDropoutConst, true, false, true, true>)
                    : (launch_params.return_softmax
                    ? &fmha_fprop_fp16_sm80_loop_kernel<Kernel_traits, IsDropoutConst, false, true, true, true>
                    : &fmha_fprop_fp16_sm80_loop_kernel<Kernel_traits, IsDropoutConst, false, false, true, true>);
                if( smem_size >= 48 * 1024 ) {
                    FMHA_CHECK_CUDA(cudaFuncSetAttribute(
                        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
                }
                dim3 grid(launch_params.params.b, launch_params.params.h);

                // printf("grid size: %d %d\n", launch_params.params.b, launch_params.params.h);
                // printf("block size: %d\n", Kernel_traits::THREADS);
                kernel<<<grid, Kernel_traits::THREADS, smem_size, launch_params.stream>>>(
                    launch_params.params);
                FMHA_CHECK_CUDA(cudaPeekAtLastError());
            });
        }else{
            // Work-around for gcc 7. It doesn't like nested BOOL_SWITCH.
            // https://github.com/kokkos/kokkos-kernels/issues/349
            // https://github.com/HazyResearch/flash-attention/issues/21
            BOOL_SWITCH(launch_params.is_dropout, IsDropoutConst, [&] {
                auto kernel = launch_params.params.is_causal
                    ? (launch_params.return_softmax
                    ? &fmha_fprop_fp16_sm80_loop_kernel<Kernel_traits, IsDropoutConst, true, true, true, false>
                    : &fmha_fprop_fp16_sm80_loop_kernel<Kernel_traits, IsDropoutConst, true, false, true, false>)
                    : (launch_params.return_softmax
                    ? &fmha_fprop_fp16_sm80_loop_kernel<Kernel_traits, IsDropoutConst, false, true, true, false>
                    : &fmha_fprop_fp16_sm80_loop_kernel<Kernel_traits, IsDropoutConst, false, false, true, false>);
                if( smem_size >= 48 * 1024 ) {
                    FMHA_CHECK_CUDA(cudaFuncSetAttribute(
                        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
                }
                dim3 grid(launch_params.params.b, launch_params.params.h);

                // printf("grid size: %d %d\n", launch_params.params.b, launch_params.params.h);
                // printf("block size: %d\n", Kernel_traits::THREADS);
                kernel<<<grid, Kernel_traits::THREADS, smem_size, launch_params.stream>>>(
                    launch_params.params);
                FMHA_CHECK_CUDA(cudaPeekAtLastError());
            });
        }
    }else{
        if (has_attn_bias) {
            // Work-around for gcc 7. It doesn't like nested BOOL_SWITCH.
            // https://github.com/kokkos/kokkos-kernels/issues/349
            // https://github.com/HazyResearch/flash-attention/issues/21
            BOOL_SWITCH(launch_params.is_dropout, IsDropoutConst, [&] {
                auto kernel = launch_params.params.is_causal
                    ? (launch_params.return_softmax
                    ? &fmha_fprop_fp16_sm80_loop_kernel<Kernel_traits, IsDropoutConst, true, true, false, true>
                    : &fmha_fprop_fp16_sm80_loop_kernel<Kernel_traits, IsDropoutConst, true, false, false, true>)
                    : (launch_params.return_softmax
                    ? &fmha_fprop_fp16_sm80_loop_kernel<Kernel_traits, IsDropoutConst, false, true, false, true>
                    : &fmha_fprop_fp16_sm80_loop_kernel<Kernel_traits, IsDropoutConst, false, false, false, true>);
                if( smem_size >= 48 * 1024 ) {
                    FMHA_CHECK_CUDA(cudaFuncSetAttribute(
                        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
                }
                dim3 grid(launch_params.params.b, launch_params.params.h);

                // printf("grid size: %d %d\n", launch_params.params.b, launch_params.params.h);
                // printf("block size: %d\n", Kernel_traits::THREADS);
                kernel<<<grid, Kernel_traits::THREADS, smem_size, launch_params.stream>>>(
                    launch_params.params);
                FMHA_CHECK_CUDA(cudaPeekAtLastError());
            });
        }else{
            // Work-around for gcc 7. It doesn't like nested BOOL_SWITCH.
            // https://github.com/kokkos/kokkos-kernels/issues/349
            // https://github.com/HazyResearch/flash-attention/issues/21
            BOOL_SWITCH(launch_params.is_dropout, IsDropoutConst, [&] {
                auto kernel = launch_params.params.is_causal
                    ? (launch_params.return_softmax
                    ? &fmha_fprop_fp16_sm80_loop_kernel<Kernel_traits, IsDropoutConst, true, true, false, false>
                    : &fmha_fprop_fp16_sm80_loop_kernel<Kernel_traits, IsDropoutConst, true, false, false, false>)
                    : (launch_params.return_softmax
                    ? &fmha_fprop_fp16_sm80_loop_kernel<Kernel_traits, IsDropoutConst, false, true, false, false>
                    : &fmha_fprop_fp16_sm80_loop_kernel<Kernel_traits, IsDropoutConst, false, false, false, false>);
                if( smem_size >= 48 * 1024 ) {
                    FMHA_CHECK_CUDA(cudaFuncSetAttribute(
                        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
                }
                dim3 grid(launch_params.params.b, launch_params.params.h);

                // printf("grid size: %d %d\n", launch_params.params.b, launch_params.params.h);
                // printf("block size: %d\n", Kernel_traits::THREADS);
                kernel<<<grid, Kernel_traits::THREADS, smem_size, launch_params.stream>>>(
                    launch_params.params);
                FMHA_CHECK_CUDA(cudaPeekAtLastError());
            });
        }
    }
}

void run_fmha_fp16_sm80(Launch_params<FMHA_fprop_params> &launch_params,
                        const bool configure) {
    FP16_SWITCH(launch_params.params.is_bf16, [&] {
        auto dprops = at::cuda::getCurrentDeviceProperties();
        if (launch_params.params.d == 16) {
            if( launch_params.params.seqlen_k == 128 ) {
                using Kernel_traits = FMHA_kernel_traits<128, 16, 16, 1, 4, 0x08u, elem_type>;
                run_fmha_fp16_sm80_loop_<Kernel_traits>(launch_params, configure);
            } 
            else if( launch_params.params.seqlen_k == 256 ) {
                using Kernel_traits = FMHA_kernel_traits<256, 16, 16, 1, 4, 0x08u, elem_type>;
                run_fmha_fp16_sm80_loop_<Kernel_traits>(launch_params, configure);
            } else {
                // TD [2022-05-15] 512 gives wrong results rn
                // using Kernel_traits = FMHA_kernel_traits<512, 16, 16, 1, 4, 0x08u, elem_type>;
                using Kernel_traits = FMHA_kernel_traits<256, 16, 16, 1, 4, 0x08u, elem_type>;
                run_fmha_fp16_sm80_loop_<Kernel_traits>(launch_params, configure);
            }
        }
        else if (launch_params.params.d == 32) {
            if( launch_params.params.seqlen_k == 128 ) {
                using Kernel_traits = FMHA_kernel_traits<128, 32, 16, 1, 4, 0x08u, elem_type>;
                run_fmha_fp16_sm80_loop_<Kernel_traits>(launch_params, configure);
            } else if( launch_params.params.seqlen_k == 256 ) {
                using Kernel_traits = FMHA_kernel_traits<256, 32, 16, 1, 4, 0x08u, elem_type>;
                run_fmha_fp16_sm80_loop_<Kernel_traits>(launch_params, configure);
            } else {
                using Kernel_traits = FMHA_kernel_traits<256, 32, 16, 1, 4, 0x08u, elem_type>;
                run_fmha_fp16_sm80_loop_<Kernel_traits>(launch_params, configure);
            }
        } 
        else if (launch_params.params.d == 64) {
            if( launch_params.params.seqlen_k == 128 ) {
                using Kernel_traits = FMHA_kernel_traits<128, 64, 16, 1, 4, 0x08u, elem_type>;
                run_fmha_fp16_sm80_loop_<Kernel_traits>(launch_params, configure);
            } else if( launch_params.params.seqlen_k >= 256 ) {
                if (dprops->major == 8 && dprops->minor >= 0) {
                    using Kernel_traits = FMHA_kernel_traits<256, 64, 16, 1, 4, 0x08u, elem_type>;
                    run_fmha_fp16_sm80_loop_<Kernel_traits>(launch_params, configure);
                } else if (dprops->major == 7 && dprops->minor == 5) {
                    if (launch_params.is_dropout) { // Need to use the same block size as backward
                        using Kernel_traits = FMHA_kernel_traits<128, 64, 16, 1, 4, 0x08u, elem_type>;
                        run_fmha_fp16_sm80_loop_<Kernel_traits>(launch_params, configure);
                    } else {
                        using Kernel_traits = FMHA_kernel_traits<256, 64, 16, 1, 4, 0x08u, elem_type>;
                        run_fmha_fp16_sm80_loop_<Kernel_traits>(launch_params, configure);
                    }
                }
            }
        } else if (launch_params.params.d == 128) {
            if( launch_params.params.seqlen_k == 128 ) {
                using Kernel_traits = FMHA_kernel_traits<128, 128, 16, 1, 4, 0x08u, elem_type>;
                run_fmha_fp16_sm80_loop_<Kernel_traits>(launch_params, configure);
            } else {
                if (dprops->major == 8 && dprops->minor == 0 && !launch_params.is_dropout) {
                    // TD [2022-06-05] Keep K in registers to reduce register spilling
                    // Gives about 6% speedup compared to using block size 128.
                    using Kernel_traits = FMHA_kernel_traits<256, 128, 16, 1, 4, 0x18u, elem_type>;
                    run_fmha_fp16_sm80_loop_<Kernel_traits>(launch_params, configure);
                } else {  // Need to use the same block size as backward
                    using Kernel_traits = FMHA_kernel_traits<128, 128, 16, 1, 4, 0x08u, elem_type>;
                    run_fmha_fp16_sm80_loop_<Kernel_traits>(launch_params, configure);
                }
            }
        }
        // if (launch_params.params.d == 64) {
        //     // using Kernel_traits = FMHA_kernel_traits<128, 64, 16, 1, 4, 0x08u, elem_type>;
        //     // using Kernel_traits = FMHA_kernel_traits<64, 64, 16, 1, 4, 0x08u, elem_type>;
        //     // using Kernel_traits = FMHA_kernel_traits<512, 64, 16, 1, 8, 0x08u, elem_type>;
        //     using Kernel_traits = FMHA_kernel_traits<128, 64, 16, 1, 4, 0x08u, elem_type>;
        //     run_fmha_fp16_sm80_loop_<Kernel_traits>(launch_params, configure);
        // }
        // if (launch_params.params.d == 64) {
        //     if( launch_params.params.seqlen_k == 128 ) {
        //         using Kernel_traits = FMHA_kernel_traits<128, 64, 16, 1, 4, 0x08u, elem_type>;
        //         run_fmha_fp16_sm80_loop_<Kernel_traits>(launch_params, configure);
        //     } else if( launch_params.params.seqlen_k >= 256 ) {
        //         if (dprops->major == 8 && dprops->minor >= 0) {
        //             using Kernel_traits = FMHA_kernel_traits<256, 64, 16, 1, 4, 0x08u, elem_type>;
        //             run_fmha_fp16_sm80_loop_<Kernel_traits>(launch_params, configure);
        //         } else if (dprops->major == 7 && dprops->minor == 5) {
        //             if (launch_params.is_dropout) { // Need to use the same block size as backward
        //                 using Kernel_traits = FMHA_kernel_traits<128, 64, 16, 1, 4, 0x08u, elem_type>;
        //                 run_fmha_fp16_sm80_loop_<Kernel_traits>(launch_params, configure);
        //             } else {
        //                 using Kernel_traits = FMHA_kernel_traits<256, 64, 16, 1, 4, 0x08u, elem_type>;
        //                 run_fmha_fp16_sm80_loop_<Kernel_traits>(launch_params, configure);
        //             }
        //         }
        //     }
        // }
        // if (launch_params.params.d == 128) {
        //     if( launch_params.params.seqlen_k == 128 ) {
        //         using Kernel_traits = FMHA_kernel_traits<128, 128, 16, 1, 4, 0x08u, elem_type>;
        //         run_fmha_fp16_sm80_loop_<Kernel_traits>(launch_params, configure);
        //     } else {
        //         if (dprops->major == 8 && dprops->minor >= 0 && !launch_params.is_dropout) {
        //             // TD [2022-06-05] Keep K in registers to reduce register spilling
        //             // Gives about 6% speedup compared to using block size 128.
        //             using Kernel_traits = FMHA_kernel_traits<256, 128, 16, 1, 4, 0x18u, elem_type>;
        //             run_fmha_fp16_sm80_loop_<Kernel_traits>(launch_params, configure);
        //         } else {  // Need to use the same block size as backward
        //             using Kernel_traits = FMHA_kernel_traits<128, 128, 16, 1, 4, 0x08u, elem_type>;
        //             run_fmha_fp16_sm80_loop_<Kernel_traits>(launch_params, configure);
        //         }
        //     }
        // }
    });
}