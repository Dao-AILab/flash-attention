/* Copyright (c) 2022, Tri Dao.
 */

#include "static_switch.h"
#include "fp16_switch.h"
#include "fmha.h"
#include "fmha_dgrad_kernel_1xN_loop.h"

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, int loop_steps=-1>
__global__ void fmha_dgrad_fp16_sm80_dq_dk_dv_loop_kernel(FMHA_dgrad_params params) {
    fmha::compute_dq_dk_dv_1xN<Kernel_traits, Is_dropout, Is_causal, loop_steps>(params);
}

template<typename Kernel_traits>
void run_fmha_dgrad_fp16_sm80_loop_(const FMHA_dgrad_params &params, cudaStream_t stream) {
    constexpr int smem_size_softmax = Kernel_traits::Cta_tile_p::M * Kernel_traits::Cta_tile_p::WARPS_N * sizeof(float);
    constexpr int smem_size_q = Kernel_traits::Smem_tile_q::BYTES_PER_TILE;
    constexpr int smem_size_v = Kernel_traits::Smem_tile_v::BYTES_PER_TILE;
    constexpr int smem_size_dq = Kernel_traits::Smem_tile_o::BYTES_PER_TILE;

    using Smem_tile_s = fmha::Smem_tile_mma_transposed<typename Kernel_traits::Cta_tile_p>;
    constexpr int smem_size_s = Smem_tile_s::BYTES_PER_TILE;
    static_assert(smem_size_s == 16 * Kernel_traits::Cta_tile_p::N * 2);
    static_assert(smem_size_dq == 16 * Kernel_traits::Cta_tile_p::K * 4 * Kernel_traits::Cta_tile_p::WARPS_N);

    constexpr int smem_size_dq_dk_dv = smem_size_q * 2 + smem_size_v * (Kernel_traits::V_IN_REGS ? 1 : 2) + smem_size_dq + smem_size_s * 2;
    constexpr int blocksize_c = Kernel_traits::Cta_tile_p::N;
    // printf("blocksize_c = %d, WARPS_N = %d, Smem size = %d\n", blocksize_c, Kernel_traits::Cta_tile_p::WARPS_N, smem_size_dq_dk_dv);

    bool is_dropout = params.p_dropout < 1.f;  // params.p_dropout is the probability of "keeping"
    // Work-around for gcc 7. It doesn't like nested BOOL_SWITCH.
    BOOL_SWITCH(is_dropout, IsDropoutConst, [&] {
        auto kernel = params.is_causal
            ? &fmha_dgrad_fp16_sm80_dq_dk_dv_loop_kernel<Kernel_traits, IsDropoutConst, true>
            : &fmha_dgrad_fp16_sm80_dq_dk_dv_loop_kernel<Kernel_traits, IsDropoutConst, false>;
        if (params.seqlen_k == blocksize_c) {
            kernel = params.is_causal
                ? &fmha_dgrad_fp16_sm80_dq_dk_dv_loop_kernel<Kernel_traits, IsDropoutConst, true, /*loop_steps=*/1>
                : &fmha_dgrad_fp16_sm80_dq_dk_dv_loop_kernel<Kernel_traits, IsDropoutConst, false, /*loop_steps=*/1>;
        } else if (params.seqlen_k == blocksize_c * 2) {
            kernel = params.is_causal
                ? &fmha_dgrad_fp16_sm80_dq_dk_dv_loop_kernel<Kernel_traits, IsDropoutConst, true, /*loop_steps=*/2>
                : &fmha_dgrad_fp16_sm80_dq_dk_dv_loop_kernel<Kernel_traits, IsDropoutConst, false, /*loop_steps=*/2>;
        }
        if( smem_size_dq_dk_dv >= 48 * 1024 ) {
            FMHA_CHECK_CUDA(cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_dq_dk_dv));
        }
        dim3 grid(params.b, params.h);
        kernel<<<grid, Kernel_traits::THREADS, smem_size_dq_dk_dv, stream>>>(params);
        FMHA_CHECK_CUDA(cudaPeekAtLastError());
    });
}

void run_fmha_dgrad_fp16_sm80(const FMHA_dgrad_params &params, cudaStream_t stream) {
    // work around for MSVC issue
    FP16_SWITCH(params.is_bf16, [&] {
        auto dprops = at::cuda::getCurrentDeviceProperties();
        if (params.d == 16) {
            if( params.seqlen_k == 128 ) {
                using Kernel_traits = FMHA_kernel_traits<128, 16, 16, 1, 8, 0x08u, elem_type>;
                run_fmha_dgrad_fp16_sm80_loop_<Kernel_traits>(params, stream);
            } else if( params.seqlen_k == 256 ) {
                using Kernel_traits = FMHA_kernel_traits<256, 16, 16, 1, 8, 0x08u, elem_type>;
                run_fmha_dgrad_fp16_sm80_loop_<Kernel_traits>(params, stream);
            } else {
                // TD [2022-05-15] 512 gives wrong results rn
                // using Kernel_traits = FMHA_kernel_traits<512, 16, 16, 1, 8, 0x08u, elem_type>;
                using Kernel_traits = FMHA_kernel_traits<256, 16, 16, 1, 8, 0x08u, elem_type>;
                run_fmha_dgrad_fp16_sm80_loop_<Kernel_traits>(params, stream);
            }
        } else if (params.d == 32) {
            if( params.seqlen_k == 128 ) {
                using Kernel_traits = FMHA_kernel_traits<128, 32, 16, 1, 8, 0x08u, elem_type>;
                run_fmha_dgrad_fp16_sm80_loop_<Kernel_traits>(params, stream);
            } else if( params.seqlen_k >= 256 ) {
                using Kernel_traits = FMHA_kernel_traits<256, 32, 16, 1, 8, 0x08u, elem_type>;
                run_fmha_dgrad_fp16_sm80_loop_<Kernel_traits>(params, stream);
            }
        } else if (params.d == 64) {
            if( params.seqlen_k == 128 ) {
                using Kernel_traits = FMHA_kernel_traits<128, 64, 16, 1, 8, 0x08u, elem_type>;
                run_fmha_dgrad_fp16_sm80_loop_<Kernel_traits>(params, stream);
            } else if( params.seqlen_k >= 256 ) {
                if (dprops->major == 8 && dprops->minor == 0) {
                    // Don't share smem for K & V, and don't keep V in registers
                    // This speeds things up by 2-3% by avoiding register spills, but it
                    // uses more shared memory, which is fine on A100 but not other GPUs.
                    // For other GPUs, we keep V in registers.
                    using Kernel_traits = FMHA_kernel_traits<256, 64, 16, 1, 8, 0x100u, elem_type>;
                    run_fmha_dgrad_fp16_sm80_loop_<Kernel_traits>(params, stream);
                } else if (dprops->major == 8 && dprops->minor > 0) {
                    using Kernel_traits = FMHA_kernel_traits<256, 64, 16, 1, 8, 0x08u, elem_type>;
                    run_fmha_dgrad_fp16_sm80_loop_<Kernel_traits>(params, stream);
                } else if (dprops->major == 7 && dprops->minor == 5) {
                    using Kernel_traits = FMHA_kernel_traits<128, 64, 16, 1, 8, 0x08u, elem_type>;
                    run_fmha_dgrad_fp16_sm80_loop_<Kernel_traits>(params, stream);
                }
            }
        } else if (params.d == 128) {
            using Kernel_traits = FMHA_kernel_traits<128, 128, 16, 1, 8, 0x100u, elem_type>;
            run_fmha_dgrad_fp16_sm80_loop_<Kernel_traits>(params, stream);
        }
        // if (params.d == 64) {
        //     if (dprops->major == 7 && dprops->minor == 5) {
        //         using Kernel_traits = FMHA_kernel_traits<128, 64, 16, 1, 8, 0x08u, elem_type>;
        //         run_fmha_dgrad_fp16_sm80_loop_<Kernel_traits>(params, stream);
        //     } else {
        //         if( params.seqlen_k == 128 ) {
        //             using Kernel_traits = FMHA_kernel_traits<128, 64, 16, 1, 8, 0x08u, elem_type>;
        //             run_fmha_dgrad_fp16_sm80_loop_<Kernel_traits>(params, stream);
        //         } else if( params.seqlen_k >= 256 ) {
        //             if (dprops->major == 8 && dprops->minor == 0) {
        //                 // Don't share smem for K & V, and don't keep V in registers
        //                 // This speeds things up by 2-3% by avoiding register spills, but it
        //                 // uses more shared memory, which is fine on A100 but not other GPUs.
        //                 // For other GPUs, we keep V in registers.
        //                 using Kernel_traits = FMHA_kernel_traits<256, 64, 16, 1, 8, 0x100u, elem_type>;
        //                 run_fmha_dgrad_fp16_sm80_loop_<Kernel_traits>(params, stream);
        //             } else if (dprops->major == 8 && dprops->minor > 0) {
        //                 using Kernel_traits = FMHA_kernel_traits<256, 64, 16, 1, 8, 0x08u, elem_type>;
        //                 run_fmha_dgrad_fp16_sm80_loop_<Kernel_traits>(params, stream);
        //             }
        //         }
        //     }
        // }
        // if (params.d == 128) {
        //     using Kernel_traits = FMHA_kernel_traits<128, 128, 16, 1, 8, 0x100u_elem_type>;
        //     run_fmha_dgrad_fp16_sm80_loop_<Kernel_traits>(params, stream);
        // }
    });
}