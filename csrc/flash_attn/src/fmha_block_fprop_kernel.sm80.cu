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

#include "fmha.h"
#include "fmha_block_fprop_kernel_1xN.h"
#include "static_switch.h"

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Return_softmax>
__global__ void fmha_block_fprop_sm80_loop_kernel(FMHA_fprop_params params) {
    fmha::device_block_1xN_loop<Kernel_traits, Is_dropout, Is_causal, Return_softmax>(params);
}

template<typename Kernel_traits>
void run_fmha_block_sm80_loop_(Launch_params<FMHA_fprop_params> &launch_params,
                            const bool configure) {
    bool is_causal = launch_params.params.is_causal;
    // TD [2022-04-27]: This case work is pretty ugly, maybe there's a better way?
    auto kernel = launch_params.is_dropout
        ? (is_causal
           ? (launch_params.return_softmax ? &fmha_block_fprop_sm80_loop_kernel<Kernel_traits, true, true, true> : &fmha_block_fprop_sm80_loop_kernel<Kernel_traits, true, true, false>)
           : (launch_params.return_softmax ? &fmha_block_fprop_sm80_loop_kernel<Kernel_traits, true, false, true> : &fmha_block_fprop_sm80_loop_kernel<Kernel_traits, true, false, false>))
        : (is_causal
           ? (launch_params.return_softmax ? &fmha_block_fprop_sm80_loop_kernel<Kernel_traits, false, true, true> : &fmha_block_fprop_sm80_loop_kernel<Kernel_traits, false, true, false>)
           : (launch_params.return_softmax ? &fmha_block_fprop_sm80_loop_kernel<Kernel_traits, false, false, true> : &fmha_block_fprop_sm80_loop_kernel<Kernel_traits, false, false, false>));

    constexpr int blocksize_c = Kernel_traits::Cta_tile_p::N;
    const int loop_steps = (launch_params.params.seqlen_k + blocksize_c - 1) / blocksize_c;
    constexpr int smem_size_softmax_lse = Kernel_traits::Smem_dp_sum::BYTES_PER_TILE;
    // Don't need smem_size_softmax_lse if we're not looping
    const int smem_size = fmha::get_dynamic_smem_size<Kernel_traits>()
        + (loop_steps > 1 ? smem_size_softmax_lse : 0);

    if( smem_size >= 48 * 1024 ) {
        FMHA_CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }

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

    dim3 grid(launch_params.params.b, launch_params.params.h);
    kernel<<<grid, Kernel_traits::THREADS, smem_size, launch_params.stream>>>(
        launch_params.params);

    FMHA_CHECK_CUDA(cudaPeekAtLastError());
}

void run_fmha_block_sm80(Launch_params<FMHA_fprop_params> &launch_params,
                             const bool configure) {
    FP16_SWITCH(launch_params.params.is_bf16, ([&] {
        if (launch_params.params.d == 16) {
            using Kernel_traits = FMHA_kernel_traits<256, 16, 16, 1, 4, 0x08u, elem_type>;
            run_fmha_block_sm80_loop_<Kernel_traits>(launch_params, configure);
        } else if (launch_params.params.d == 32) {
            using Kernel_traits = FMHA_kernel_traits<256, 32, 16, 1, 4, 0x08u, elem_type>;
            run_fmha_block_sm80_loop_<Kernel_traits>(launch_params, configure);
        } else if (launch_params.params.d == 64) {
            using Kernel_traits = FMHA_kernel_traits<256, 64, 16, 1, 4, 0x08u, elem_type>;
            run_fmha_block_sm80_loop_<Kernel_traits>(launch_params, configure);
        } else if (launch_params.params.d == 128) {
            using Kernel_traits = FMHA_kernel_traits<256, 128, 16, 1, 4, 0x08u, elem_type>;
            run_fmha_block_sm80_loop_<Kernel_traits>(launch_params, configure);
        }
    }));
}
