/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include <ATen/cuda/CUDAContext.h>

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/cluster_launch.hpp"

#include "static_switch.h"
#include "flash.h"
#include "tile_scheduler.hpp"
#include "flash_fwd_kernel.h"
#include "kernel_traits.h"


template<typename Kernel_traits, bool Is_causal>
void run_flash_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    using Element = typename Kernel_traits::Element;
    using TileShape_MNK = typename Kernel_traits::TileShape_MNK;
    using ClusterShape = typename Kernel_traits::ClusterShape_MNK;

    // print(typename Kernel_traits::SmemLayoutVt{}); printf("\n"); print(typename Kernel_traits::SmemLayoutVt_tmp{});
    using CollectiveMainloop = flash::CollectiveMainloopFwd<Kernel_traits, Is_causal>;
    using CollectiveEpilogue = flash::CollectiveEpilogueFwd<Kernel_traits>;
    using Scheduler = std::conditional_t<!Is_causal,
        flash::StaticPersistentTileScheduler,
        flash::DynamicPersistentTileScheduler<Kernel_traits::kNThreads - cutlass::NumThreadsPerWarpGroup>>;
        // flash::SingleTileScheduler>;
    typename CollectiveMainloop::Params mainloop_params =
        CollectiveMainloop::to_underlying_arguments({
            static_cast<Element const*>(params.q_ptr),
            {params.seqlen_q, params.d, params.h, params.b},  // shape_Q
            {params.q_row_stride, _1{}, params.q_head_stride, params.q_batch_stride},  // stride_Q
            static_cast<Element const*>(params.k_ptr),
            {params.seqlen_k, params.d, params.h_k, params.b},  // shape_K
            {params.k_row_stride, _1{}, params.k_head_stride, params.k_batch_stride},  // stride_K
            static_cast<Element const*>(params.v_ptr),
            {params.v_row_stride, _1{}, params.v_head_stride, params.v_batch_stride},  // stride_V
            params.scale_softmax_log2
        });
    typename CollectiveEpilogue::Params epilogue_params =
        CollectiveEpilogue::to_underlying_arguments({
            static_cast<Element*>(params.o_ptr),
            {params.seqlen_q, params.d, params.h, params.b},  // shape_O
            {params.o_row_stride, _1{}, params.o_head_stride, params.o_batch_stride},  // stride_O
            static_cast<float*>(params.softmax_lse_ptr),
            {_1{}, params.seqlen_q, params.h * params.seqlen_q},  // stride_LSE
        });

    int num_blocks_m = cutlass::ceil_div(params.seqlen_q, Kernel_traits::kBlockM);
    num_blocks_m = cutlass::ceil_div(num_blocks_m, size<0>(ClusterShape{})) * size<0>(ClusterShape{});
    typename Scheduler::Arguments scheduler_args = {num_blocks_m, params.h, params.b, params.tile_count_semaphore};
    typename Scheduler::Params scheduler_params = Scheduler::to_underlying_arguments(scheduler_args);

    // Get the ptr to kernel function.
    void *kernel;
    kernel = (void *)flash::compute_attn_ws<Kernel_traits, Is_causal, Scheduler>;
    int smem_size = sizeof(typename Kernel_traits::SharedStorage);
    // int smem_size_q = sizeof(decltype((typename Kernel_traits::SharedStorage{}).smem_q));
    // int smem_size_k = sizeof(decltype((typename Kernel_traits::SharedStorage{}).smem_k));
    // int smem_size_v = sizeof(decltype((typename Kernel_traits::SharedStorage{}).smem_v));
    // printf("smem_size = %d, q = %d, k = %d, v = %d\n", smem_size, smem_size_q, smem_size_k, smem_size_v);
    if (smem_size >= 48 * 1024) {
       C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }

    int device;
    cudaGetDevice(&device);
    int multiprocessor_count;
    cudaError status_ = cudaDeviceGetAttribute(
        &multiprocessor_count, cudaDevAttrMultiProcessorCount, device);
    if (status_ != cudaSuccess) {
      C10_CUDA_CHECK(status_);
    }
    dim3 grid_dims = Scheduler::get_grid_dim(scheduler_args, multiprocessor_count);
    static constexpr int ctaSize = Kernel_traits::kNWarps * 32;
    dim3 block_dims(ctaSize);
    dim3 cluster_dims(size<0>(ClusterShape{}), size<1>(ClusterShape{}), size<2>(ClusterShape{}));
    cutlass::ClusterLaunchParams launch_params{grid_dims, block_dims, cluster_dims, smem_size, stream};
    cutlass::launch_kernel_on_cluster(launch_params, kernel, mainloop_params, epilogue_params, scheduler_params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<typename T>
void run_mha_fwd_hdim64(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 64;
    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 192, 128, 16, 2, false, 1, T>, Is_causal>(params, stream);
    });
}

template<typename T>
void run_mha_fwd_hdim128(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        // Only use Cluster if number of tiles along seqlen_q is even
        BOOL_SWITCH(cutlass::ceil_div(params.seqlen_q, 128) % 2 == 0, UseCluster, [&] {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, Is_causal ? 128 : 176, 12, 2, false, !Is_causal && UseCluster ? 2 : 1, T>, Is_causal>(params, stream);
        });
    });
}

template<typename T>
void run_mha_fwd_hdim256(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 256;
    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        // Only use Cluster if number of tiles along seqlen_q is even
        BOOL_SWITCH(cutlass::ceil_div(params.seqlen_q, 128) % 2 == 0, UseCluster, [&] {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 80, 12, 2, false, !Is_causal && UseCluster ? 2 : 1, T>, Is_causal>(params, stream);
        });
    });
}
