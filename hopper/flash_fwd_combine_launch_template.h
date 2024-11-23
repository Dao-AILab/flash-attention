/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/arch/arch.h"  // For cutlass::arch::Sm80
#include "cutlass/device_kernel.h"  // For device_kernel

#include "static_switch.h"
#include "flash.h"
#include "flash_fwd_combine_kernel.h"

using namespace cute;

template <int kHeadDim, int kBlockM, int kLogMaxSplits, bool IsEvenK, bool Varlen, typename Element, typename ElementPartial>
void run_flash_fwd_combine(Flash_fwd_params &params, cudaStream_t stream) {
    using TileShape_MK = cute::Shape<Int<kBlockM>, Int<kHeadDim>>;
    using CombineKernel = flash::FlashAttnFwdCombine<TileShape_MK, kLogMaxSplits, 256 /*kNThreads*/, 1 /*AlignmentLSE*/,
                                                     IsEvenK, Varlen, Element, ElementPartial, cutlass::arch::Sm80>;

    typename CombineKernel::Arguments args {
        static_cast<ElementPartial const*>(params.oaccum_ptr),
        {!Varlen ? params.seqlen_q : params.total_q, params.d, params.num_splits, params.h, !Varlen ? params.b : 1},  // shape_O_partial
        {params.oaccum_row_stride, _1{}, params.oaccum_split_stride, params.oaccum_head_stride, !Varlen ? params.oaccum_batch_stride : 0},  // stride_O_partial
        static_cast<float*>(params.softmax_lseaccum_ptr),
        {!Varlen ? params.seqlen_q : params.total_q, params.num_splits, params.h, !Varlen ? params.b : 1},  // shape_LSE_partial
        {_1{}, params.lseaccum_split_stride, params.lseaccum_head_stride, !Varlen ? params.lseaccum_batch_stride : 0},  // stride_LSE_partial
        static_cast<Element*>(params.o_ptr),
        {params.o_row_stride, _1{}, params.o_head_stride, !Varlen ? params.o_batch_stride : 0},  // stride_O
        static_cast<float*>(params.softmax_lse_ptr),
        {_1{}, !Varlen ? params.seqlen_q : params.total_q, !Varlen ? params.h * params.seqlen_q : 0},  // stride_LSE
        params.cu_seqlens_q, params.seqused_q
    };

    typename CombineKernel::Params kernel_params = CombineKernel::to_underlying_arguments(args);
    int num_blocks_m = cute::ceil_div(params.seqlen_q * params.h * (!Varlen ? params.b : 1), kBlockM);
    dim3 grid_m(num_blocks_m, !Varlen ? 1 : params.b);
    auto kernel = cutlass::device_kernel<CombineKernel>;
    int smem_size = CombineKernel::SharedStorageSize;
    if (smem_size >= 48 * 1024) {
        CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }
    kernel<<<grid_m, CombineKernel::MaxThreadsPerBlock, smem_size, stream>>>(kernel_params);
    CHECK_CUDA_KERNEL_LAUNCH();
}

template<typename T, typename Tpartial, int kHeadDim>
void run_mha_fwd_combine_(Flash_fwd_params &params, cudaStream_t stream) {
    // We want kBlockM to be as small as possible to maximize parallelism.
    // E.g., if hdim is 64, we want kBlockM to be 16 so that we can use 256 threads, each reading 4 elements (floats).
    static_assert(kHeadDim % 32 == 0, "kHeadDim must be a multiple of 32");
    static constexpr int kBlockM = kHeadDim % 128 == 0 ? 8 : (kHeadDim % 64 == 0 ? 16 : 32);
    BOOL_SWITCH(params.seqused_q != nullptr, Varlen, [&] {
        if constexpr (kBlockM >= 16) {  // If kBlockM == 8 then the minimum number of splits is 32.
            if (params.num_splits <= 16) {
                run_flash_fwd_combine<kHeadDim, kBlockM, 4, false /*IsEvenK*/, Varlen, T, Tpartial>(params, stream);
                return;
            }
        }
        if (params.num_splits <= 32) {
            run_flash_fwd_combine<kHeadDim, kBlockM, 5, false /*IsEvenK*/, Varlen, T, Tpartial>(params, stream);
        } else if (params.num_splits <= 64) {
            run_flash_fwd_combine<kHeadDim, kBlockM, 6, false /*IsEvenK*/, Varlen, T, Tpartial>(params, stream);
        } else if (params.num_splits <= 128) {
            run_flash_fwd_combine<kHeadDim, kBlockM, 7, false /*IsEvenK*/, Varlen, T, Tpartial>(params, stream);
        } else {
            run_flash_fwd_combine<kHeadDim, kBlockM, 8, false /*IsEvenK*/, Varlen, T, Tpartial>(params, stream);
        }
    });
}
