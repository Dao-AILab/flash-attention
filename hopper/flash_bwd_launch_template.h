/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include "cute/tensor.hpp"

#include "cutlass/cluster_launch.hpp"
#include "cutlass/device_kernel.h"  // For device_kernel

#include "static_switch.h"
#include "flash.h"
#include "flash_bwd_preprocess_kernel.h"
#include "flash_bwd_postprocess_kernel.h"
#include "tile_scheduler_bwd.hpp"
#include "mainloop_bwd_sm90_tma_gmma_ws.hpp"
#include "epilogue_bwd_sm90_tma.hpp"
#include "flash_bwd_kernel.h"

using namespace cute;

template <int kHeadDim, int kBlockM, int kBlockN, typename Element, bool Is_causal, bool Varlen, bool Deterministic,
          bool dKV_swapAB, bool dQ_swapAB, int AtomLayoutMSdP=1, int AtomLayoutNdKV=2, int AtomLayoutMdQ=1>
void run_flash_bwd(Flash_bwd_params &params, cudaStream_t stream) {
    using TileShape_MK = cute::Shape<Int<kBlockM>, Int<kHeadDim>>;
    using ElementAccum = float;
    using PreprocessKernel = flash::FlashAttnBwdPreprocess<TileShape_MK, Element, ElementAccum, cutlass::arch::Sm90, /*Clear_dQaccum=*/true, Varlen>;
    int const total_q_padded_rounded = cute::round_up(params.total_q + params.b * 128, 128);
    typename PreprocessKernel::Arguments preprocess_args {
        static_cast<Element const*>(params.o_ptr),
        {!Varlen ? params.seqlen_q : params.total_q, params.d, params.h, !Varlen ? params.b : 1},  // shape_O
        {params.o_row_stride, _1{}, params.o_head_stride, !Varlen ? params.o_batch_stride : 0},  // stride_O
        static_cast<Element const*>(params.do_ptr),
        {params.do_row_stride, _1{}, params.do_head_stride, !Varlen ? params.do_batch_stride : 0},  // stride_dO
        static_cast<float*>(params.dsoftmax_sum),
        {!Varlen ? params.seqlen_q_rounded : total_q_padded_rounded, params.h, !Varlen ? params.b : 1},  // shape_dPsum
        {_1{}, !Varlen ? params.seqlen_q_rounded : total_q_padded_rounded, !Varlen ? params.h * params.seqlen_q_rounded : 0},  // stride_dPsum
        static_cast<float*>(params.softmax_lse_ptr),
        {_1{}, !Varlen ? params.seqlen_q : params.total_q, !Varlen ? params.h * params.seqlen_q : 0},  // stride_LSE
        static_cast<float*>(params.softmax_lse_log2_ptr),
        {_1{}, !Varlen ? params.seqlen_q_rounded : total_q_padded_rounded, !Varlen ? params.h * params.seqlen_q_rounded : 0},  // stride_LSE_log2
        static_cast<ElementAccum*>(params.dq_accum_ptr),
        {!Varlen ? params.seqlen_q_rounded : total_q_padded_rounded, params.d_rounded, params.h, !Varlen ? params.b : 1},  // shape_dQaccum
        {params.d_rounded, _1{}, params.d_rounded * (!Varlen ? params.seqlen_q_rounded : total_q_padded_rounded), !Varlen ? params.d_rounded * params.seqlen_q_rounded * params.h : 0},  // stride_dQ
        params.b,
        params.dq_semaphore,
        params.cu_seqlens_q
    };
    typename PreprocessKernel::Params preprocess_params = PreprocessKernel::to_underlying_arguments(preprocess_args);
    int num_m_block = cute::ceil_div(params.seqlen_q, kBlockM);
    dim3 grid_m(num_m_block, params.h, params.b);
    cutlass::device_kernel<PreprocessKernel><<<grid_m, PreprocessKernel::MaxThreadsPerBlock, PreprocessKernel::SharedStorageSize, stream>>>(preprocess_params);

    using TileShape_MNK = cute::Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
    using ClusterShape = cute::Shape<_1, Int<1>, _1>;
    static constexpr int Stages = 2;
    using CollectiveMainloop = flash::CollectiveMainloopBwd<Stages, ClusterShape, TileShape_MNK, Element, ElementAccum, cutlass::arch::Sm90,
            Is_causal, Varlen, Deterministic,
            dKV_swapAB, dQ_swapAB, AtomLayoutMSdP, AtomLayoutNdKV, AtomLayoutMdQ>;
    using CollectiveEpilogue = flash::CollectiveEpilogueBwd<TileShape_MNK, Element, CollectiveMainloop::NumMmaThreads, Varlen>;
    using Scheduler = flash::SingleTileSchedulerBwd;
    using AttnKernel = flash::FlashAttnBwd<CollectiveMainloop, CollectiveEpilogue, Scheduler>;

    typename CollectiveMainloop::Arguments mainloop_args {
        static_cast<Element const*>(params.q_ptr),
        {!Varlen ? params.seqlen_q : params.total_q, params.d, params.h, !Varlen ? params.b : 1},  // shape_Q
        {params.q_row_stride, _1{}, params.q_head_stride, !Varlen ? params.q_batch_stride : 0},  // stride_Q
        static_cast<Element const*>(params.k_ptr),
        {!Varlen ? params.seqlen_k : params.total_k, params.d, params.h_k, !Varlen ? params.b : 1},  // shape_K
        {params.k_row_stride, _1{}, params.k_head_stride, !Varlen ? params.k_batch_stride : 0},  // stride_K
        static_cast<Element const*>(params.v_ptr),
        {params.v_row_stride, _1{}, params.v_head_stride, !Varlen ? params.v_batch_stride : 0},  // stride_V
        static_cast<Element const*>(params.do_ptr),
        {params.do_row_stride, _1{}, params.do_head_stride, !Varlen ? params.do_batch_stride : 0},  // stride_dO
        static_cast<ElementAccum*>(params.dq_accum_ptr),
        // {params.seqlen_q_rounded, params.d_rounded, params.h, params.b},  // shape_dQaccum
        // {params.d_rounded, _1{}, params.d_rounded * params.seqlen_q_rounded, params.d_rounded * params.seqlen_q_rounded * params.h}, // stride_dQaccum
        {(!Varlen ? params.seqlen_q_rounded : total_q_padded_rounded) * (params.d_rounded / 32), 32, params.h, !Varlen ? params.b : 1},  // shape_dQaccum
        {32, _1{}, params.d_rounded * (!Varlen ? params.seqlen_q_rounded : total_q_padded_rounded), !Varlen ? params.d_rounded * params.seqlen_q_rounded * params.h : 0}, // stride_dQaccum
        static_cast<float*>(params.softmax_lse_log2_ptr),
        {!Varlen ? params.seqlen_q_rounded : total_q_padded_rounded, params.h, !Varlen ? params.b : 1},  // shape_LSE
        {_1{}, !Varlen ? params.seqlen_q_rounded : total_q_padded_rounded, !Varlen ? params.h * params.seqlen_q_rounded : 0},  // stride_LSE_log2
        static_cast<float*>(params.dsoftmax_sum),
        {_1{}, !Varlen ? params.seqlen_q_rounded : total_q_padded_rounded, !Varlen ? params.h * params.seqlen_q_rounded : 0},  // stride_dPsum
        params.scale_softmax,
        params.b,
        params.dq_semaphore,
        params.cu_seqlens_q, params.cu_seqlens_k,
    };
    typename CollectiveEpilogue::Arguments epilogue_args {
        static_cast<Element*>(params.dk_ptr),
        {!Varlen ? params.seqlen_k : params.total_k, params.d, params.h, !Varlen ? params.b : 1},  // shape_dK
        {params.dk_row_stride, _1{}, params.dk_head_stride, !Varlen ? params.dk_batch_stride : 0},  // stride_dK
        static_cast<Element*>(params.dv_ptr),
        {params.dv_row_stride, _1{}, params.dv_head_stride, !Varlen ? params.dv_batch_stride : 0},
        params.cu_seqlens_k
    };

    int num_blocks_n = cutlass::ceil_div(params.seqlen_k, get<1>(TileShape_MNK{}));
    num_blocks_n = cutlass::round_up(num_blocks_n, size<1>(ClusterShape{}));
    typename Scheduler::Arguments scheduler_args {
        num_blocks_n, params.h, params.b, params.tile_count_semaphore, params.cu_seqlens_k
    };

    int device;
    cudaGetDevice(&device);
    typename AttnKernel::Params kernel_params = AttnKernel::to_underlying_arguments({
        mainloop_args, epilogue_args, {device}, scheduler_args
    });

    // Get the ptr to kernel function.
    void const* kernel = (void const*) cutlass::device_kernel<AttnKernel>;
    int smem_size = AttnKernel::SharedStorageSize;
    // int smem_size_q = sizeof(decltype((typename AttnKernel::SharedStorage{}).mainloop.smem_q));
    // int smem_size_do = sizeof(decltype((typename AttnKernel::SharedStorage{}).mainloop.smem_do));
    // int smem_size_ds = sizeof(decltype((typename AttnKernel::SharedStorage{}).mainloop.smem_ds));
    // int smem_size_dqacc = sizeof(decltype((typename AttnKernel::SharedStorage{}).mainloop.smem_dqacc));
    // int smem_size_k = sizeof(decltype((typename AttnKernel::SharedStorage{}).mainloop.smem_k));
    // int smem_size_v = sizeof(decltype((typename AttnKernel::SharedStorage{}).mainloop.smem_v));
    // printf("smem_size = %d, q = %d, k = %d, v = %d, do = %d, ds = %d, dqacc = %d\n", smem_size, smem_size_q, smem_size_k, smem_size_v, smem_size_do, smem_size_ds, smem_size_dqacc);
    if (smem_size >= 48 * 1024) {
        CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }

    dim3 grid_dims = AttnKernel::get_grid_shape(kernel_params);
    dim3 block_dims = AttnKernel::get_block_shape();
    dim3 cluster_dims(size<0>(ClusterShape{}), size<1>(ClusterShape{}), size<2>(ClusterShape{}));
    cutlass::ClusterLaunchParams launch_params{grid_dims, block_dims, cluster_dims, smem_size, stream};
    cutlass::launch_kernel_on_cluster(launch_params, kernel, kernel_params);
    CHECK_CUDA_KERNEL_LAUNCH();

    using PostprocessKernel = flash::FlashAttnBwdPostprocessConvertdQ<TileShape_MK, Element, ElementAccum, cutlass::arch::Sm90,
        AttnKernel::CollectiveMainloop::kNThreadsdQ,
        typename AttnKernel::CollectiveMainloop::SmemLayoutdQaccumTMA,
        typename AttnKernel::CollectiveMainloop::TiledMmadQ,
        AttnKernel::CollectiveMainloop::dQ_swapAB
        >;
    typename PostprocessKernel::Arguments postprocess_args {
        static_cast<ElementAccum const*>(params.dq_accum_ptr),
        // {params.seqlen_q_rounded, params.d_rounded, params.h, params.b},  // shape_dQaccum
        // {params.d_rounded, _1{}, params.d_rounded * params.seqlen_q_rounded, params.d_rounded * params.seqlen_q_rounded * params.h},  // stride_dQaccum
        {(!Varlen ? params.seqlen_q_rounded : total_q_padded_rounded) * (params.d_rounded / 32), 32, params.h, !Varlen ? params.b : 1},  // shape_dQaccum
        {32, _1{}, params.d_rounded * (!Varlen ? params.seqlen_q_rounded : total_q_padded_rounded), !Varlen ? params.d_rounded * params.seqlen_q_rounded * params.h : 0}, // stride_dQaccum
        static_cast<Element*>(params.dq_ptr),
        {!Varlen ? params.seqlen_q : params.total_q, params.d, params.h, !Varlen ? params.b : 1},  // shape_dQ
        {params.dq_row_stride, _1{}, params.dq_head_stride, params.dq_batch_stride},  // stride_dQ
        params.scale_softmax,
        params.cu_seqlens_q
    };
    typename PostprocessKernel::Params postprocess_params = PostprocessKernel::to_underlying_arguments(postprocess_args);
    int num_m_block_postprocess = cute::ceil_div(params.seqlen_q, get<0>(TileShape_MK{}));
    dim3 grid_m_postprocess(num_m_block_postprocess, params.h, params.b);
    // Get the ptr to kernel function.
    auto postprocess_kernel = cutlass::device_kernel<PostprocessKernel>;
    int smem_size_postprocess = PostprocessKernel::SharedStorageSize;
    if (smem_size_postprocess >= 48 * 1024) {
        CHECK_CUDA(cudaFuncSetAttribute(postprocess_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }
    postprocess_kernel<<<grid_m_postprocess, PostprocessKernel::MaxThreadsPerBlock, smem_size_postprocess, stream>>>(postprocess_params);
    CHECK_CUDA_KERNEL_LAUNCH();

}


template<typename T>
void run_mha_bwd_hdim64(Flash_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 64;
    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        BOOL_SWITCH(params.cu_seqlens_q != nullptr || params.cu_seqlens_k != nullptr, Varlen, [&] {
            BOOL_SWITCH(params.deterministic, Deterministic, [&] {
                run_flash_bwd<Headdim, 128, 128, T, Is_causal, Varlen, Deterministic, false, false, 1, 2, 2>(params, stream);
            });
        });
    });
}

template<typename T>
void run_mha_bwd_hdim96(Flash_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 96;
    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        BOOL_SWITCH(params.cu_seqlens_q != nullptr || params.cu_seqlens_k != nullptr, Varlen, [&] {
            BOOL_SWITCH(params.deterministic, Deterministic, [&] {
                run_flash_bwd<Headdim, 64, 128, T, Is_causal, Varlen, Deterministic, false, false, 1, 2, 1>(params, stream);
            });
        });
    });
}

template<typename T>
void run_mha_bwd_hdim128(Flash_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        BOOL_SWITCH(params.cu_seqlens_q != nullptr || params.cu_seqlens_k != nullptr, Varlen, [&] {
            BOOL_SWITCH(params.deterministic, Deterministic, [&] {
                run_flash_bwd<Headdim, 64, 128, T, Is_causal, Varlen, Deterministic, false, false, 1, 2, 1>(params, stream);
            });
        });
    });
}
