/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/cluster_launch.hpp"

#include "static_switch.h"
#include "flash.h"
#include "tile_scheduler.hpp"
#include "flash_fwd_kernel.h"
#include "kernel_traits.h"
#include "seq_len.h"
#include "utils.h"


template<typename Kernel_traits, bool Is_causal, typename Seqlen_traits>
void run_flash_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    using Element = typename Kernel_traits::Element;
    using OutputType = typename Kernel_traits::OutputType;
    using TileShape_MNK = typename Kernel_traits::TileShape_MNK;
    using ClusterShape = typename Kernel_traits::ClusterShape_MNK;

    // print(typename Kernel_traits::SmemLayoutVt{}); printf("\n"); print(typename Kernel_traits::SmemLayoutVt_tmp{});
    using CollectiveMainloop = flash::CollectiveMainloopFwd<Kernel_traits, Is_causal, Seqlen_traits>;
    using CollectiveEpilogue = flash::CollectiveEpilogueFwd<Kernel_traits, Seqlen_traits>;
    using Scheduler = std::conditional_t<
        Seqlen_traits::kUseVarSeqLen, 
        flash::SingleTileScheduler,
        std::conditional_t<!Is_causal,
            flash::StaticPersistentTileScheduler,
            flash::DynamicPersistentTileScheduler<Kernel_traits::kNThreads - cutlass::NumThreadsPerWarpGroup, Kernel_traits::NumProducerThreads>
    >>;
    // using Scheduler = flash::SingleTileScheduler;
    Seqlen_traits seqlen_traits_q(
        params.total_q, params.seqlen_q, params.cu_seqlens_q);
    Seqlen_traits seqlen_traits_k(
        params.total_k, params.seqlen_k, params.cu_seqlens_k, params.seqused_k);
    typename CollectiveMainloop::Params mainloop_params =
        CollectiveMainloop::to_underlying_arguments({
            static_cast<Element const*>(params.q_ptr),
            seqlen_traits_q.get_gmem_layout(
                params.seqlen_q, params.d, params.h, params.b, 
                params.q_row_stride, params.q_head_stride, params.q_batch_stride
            ),  // layout_Q
            static_cast<Element const*>(params.k_ptr),
            seqlen_traits_k.get_gmem_layout(
                params.seqlen_k, params.d, params.h_k, params.b, 
                params.k_row_stride, params.k_head_stride, params.k_batch_stride
            ),  // layout_K
            static_cast<Element const*>(params.v_ptr),
            seqlen_traits_k.get_gmem_layout(
                params.seqlen_k, params.d, params.h_k, params.b, 
                params.v_row_stride, params.v_head_stride, params.v_batch_stride
            ),  // layout_V
            params.scale_softmax_log2,
            params.descale_q_ptr,
            params.descale_k_ptr,
            params.descale_v_ptr
        });
    typename CollectiveEpilogue::Params epilogue_params =
        CollectiveEpilogue::to_underlying_arguments({
            static_cast<OutputType*>(params.o_ptr),
            seqlen_traits_q.get_gmem_layout(
                params.seqlen_q, params.d, params.h, params.b,
                params.o_row_stride, params.o_head_stride, params.o_batch_stride
            ),  // layout_O
            static_cast<float*>(params.softmax_lse_ptr),
            seqlen_traits_q.get_lse_gmem_layout(
                params.seqlen_q, params.h, params.b
            )  // layout_LSE
        });

    int num_blocks_m = cutlass::ceil_div(params.seqlen_q, Kernel_traits::kBlockM);
    num_blocks_m = cutlass::ceil_div(num_blocks_m, size<0>(ClusterShape{})) * size<0>(ClusterShape{});
    typename Scheduler::Arguments scheduler_args = {num_blocks_m, params.h, params.b, params.tile_count_semaphore};
    typename Scheduler::Params scheduler_params = Scheduler::to_underlying_arguments(scheduler_args);

    // Get the ptr to kernel function.
    void *kernel;
    if constexpr(cutlass::sizeof_bits_v<Element> == 8)
        kernel = (void *)flash::compute_attn_ws_fp8<Kernel_traits, Is_causal, Scheduler, Seqlen_traits>;
    else
        kernel = (void *)flash::compute_attn_ws<Kernel_traits, Is_causal, Scheduler, Seqlen_traits>;
    int smem_size = sizeof(typename Kernel_traits::SharedStorage);
    // int smem_size_q = sizeof(decltype((typename Kernel_traits::SharedStorage{}).smem_q));
    // int smem_size_k = sizeof(decltype((typename Kernel_traits::SharedStorage{}).smem_k));
    // int smem_size_v = sizeof(decltype((typename Kernel_traits::SharedStorage{}).smem_v));
    // int smem_size_o = sizeof(decltype((typename Kernel_traits::SharedStorage{}).smem_o));
    // printf("smem_size = %d, q = %d, k = %d, v = %d, o = %d.\n", smem_size, smem_size_q, smem_size_k, smem_size_v, smem_size_o);
    if (smem_size >= 48 * 1024) {
       CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }

    int device;
    cudaGetDevice(&device);
    int multiprocessor_count;
    CHECK_CUDA(cudaDeviceGetAttribute(&multiprocessor_count, cudaDevAttrMultiProcessorCount, device));
    dim3 grid_dims = Scheduler::get_grid_dim(scheduler_args, multiprocessor_count);
    static constexpr int ctaSize = Kernel_traits::kNWarps * 32;
    dim3 block_dims(ctaSize);
    dim3 cluster_dims(size<0>(ClusterShape{}), size<1>(ClusterShape{}), size<2>(ClusterShape{}));
    cutlass::ClusterLaunchParams launch_params{grid_dims, block_dims, cluster_dims, smem_size, stream};
    cutlass::launch_kernel_on_cluster(
        launch_params, kernel, mainloop_params, epilogue_params, 
        scheduler_params, seqlen_traits_q, seqlen_traits_k);
    CHECK_CUDA_KERNEL_LAUNCH();
}

template<typename T>
void run_mha_fwd_hdim64(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 64;
    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        SEQLEN_SWITCH(params.cu_seqlens_q, Seqlen_traits, [&] {
            run_flash_fwd<
                Flash_fwd_kernel_traits<Headdim, 192, 128, 16, 2, false, 1, T>, 
                Is_causal, Seqlen_traits
            >(params, stream);
        });
    });
}

template<typename T>
void run_mha_fwd_hdim128(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        SEQLEN_SWITCH(params.cu_seqlens_q, Seqlen_traits, [&] {
            // Only use Cluster if number of tiles along seqlen_q is even and not Is_causal
            BOOL_SWITCH(cutlass::ceil_div(params.seqlen_q, 128) % 2 == 0 && !Is_causal && !Seqlen_traits::kUseVarSeqLen, UseCluster, [&] {
                run_flash_fwd<
                    Flash_fwd_kernel_traits<Headdim, 128, Is_causal ? 128 : 176, 12, 2, false, UseCluster ? 2 : 1, T>, 
                    Is_causal, Seqlen_traits
                >(params, stream);
            });
        });
    });
}

template<typename T>
void run_mha_fwd_hdim256(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 256;
    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        SEQLEN_SWITCH(params.cu_seqlens_q, Seqlen_traits, [&] {
            // Only use Cluster if number of tiles along seqlen_q is even
            BOOL_SWITCH(cutlass::ceil_div(params.seqlen_q, 128) % 2 == 0 && !Is_causal && !Seqlen_traits::kUseVarSeqLen, UseCluster, [&] {
                run_flash_fwd<
                    Flash_fwd_kernel_traits<Headdim, 128, 80, 12, 2, false, UseCluster ? 2 : 1, T>, 
                    Is_causal, Seqlen_traits
                >(params, stream);
            });
        });
    });
}

template<typename T>
void run_mha_fwd_hdim64_fp8(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 64;
    constexpr static int kBlockM = 192;
    constexpr static int kBlockN = 128;
    constexpr static int kNWarps = 4 + kBlockM/16;
    constexpr static int kStages = 4;    
    using Seqlen_traits = flash::FixedSeqLenTraits;
    if(params.is_causal) {
        run_flash_fwd<Flash_fwd_kernel_traits_fp8<Headdim, kBlockM, kBlockN, kNWarps, kStages,
                        false, 1, T>, /*Is_causal=*/true, Seqlen_traits>(params, stream);
    } else {
        BOOL_SWITCH(cutlass::ceil_div(params.seqlen_q, kBlockM) % 2 == 0, UseCluster, [&] {
            run_flash_fwd<Flash_fwd_kernel_traits_fp8<Headdim, kBlockM, kBlockN, kNWarps, kStages,
                            false, UseCluster ? 2 : 1, T>, /*Is_causal=*/false, Seqlen_traits>(params, stream);
        });
    }
    // BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        // SEQLEN_SWITCH(params.cu_seqlens_q, Seqlen_traits, [&] {
            // Only use Cluster if number of tiles along seqlen_q is even
            // BOOL_SWITCH(cutlass::ceil_div(params.seqlen_q, kBlockM) % 2 == 0 && !Is_causal &&
            //             !Seqlen_traits::kUseVarSeqLen, UseCluster, [&] {
            //     run_flash_fwd<Flash_fwd_kernel_traits_fp8<Headdim, kBlockM, kBlockN, kNWarps, kStages,
            //                   false, UseCluster ? 2 : 1, T>, Is_causal, Seqlen_traits>(params, stream);            
            // });
        // });
    // });
}

template<typename T>
void run_mha_fwd_hdim128_fp8(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    constexpr static int kBlockM = 128;
    constexpr static int kBlockN = 256;
    constexpr static int kNWarps = 4 + kBlockM/16;
    constexpr static int kStages = 2;
    using Seqlen_traits = flash::FixedSeqLenTraits;
    if(params.is_causal) {
        run_flash_fwd<Flash_fwd_kernel_traits_fp8<Headdim, kBlockM, kBlockN, kNWarps, kStages,
                        false, 1, T>, /*Is_causal=*/true, Seqlen_traits>(params, stream);
    } else {
        BOOL_SWITCH(cutlass::ceil_div(params.seqlen_q, kBlockM) % 2 == 0, UseCluster, [&] {
            run_flash_fwd<Flash_fwd_kernel_traits_fp8<Headdim, kBlockM, kBlockN, kNWarps, kStages,
                            false, UseCluster ? 2 : 1, T>, /*Is_causal=*/false, Seqlen_traits>(params, stream);
        });
    }
    // BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        // SEQLEN_SWITCH(params.cu_seqlens_q, Seqlen_traits, [&] {
            // Only use Cluster if number of tiles along seqlen_q is even
            // BOOL_SWITCH(cutlass::ceil_div(params.seqlen_q, kBlockM) % 2 == 0 && !Is_causal &&
            //             !Seqlen_traits::kUseVarSeqLen, UseCluster, [&] {
            //     run_flash_fwd<Flash_fwd_kernel_traits_fp8<Headdim, kBlockM, kBlockN, kNWarps, kStages,
            //                   false, UseCluster ? 2 : 1, T>, Is_causal, Seqlen_traits>(params, stream);
            // });
        // });
    // });
}

template<typename T>
void run_mha_fwd_hdim256_fp8(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 256; 
    constexpr static int kBlockM = 128;
    constexpr static int kBlockN = 128;
    constexpr static int kNWarps = 4 + kBlockM/16;
    constexpr static int kStages = 2;
    using Seqlen_traits = flash::FixedSeqLenTraits;
    if(params.is_causal) {
        run_flash_fwd<Flash_fwd_kernel_traits_fp8<Headdim, kBlockM, kBlockN, kNWarps, kStages,
                        false, 1, T>, /*Is_causal=*/true, Seqlen_traits>(params, stream);
    } else {
        BOOL_SWITCH(cutlass::ceil_div(params.seqlen_q, kBlockM) % 2 == 0, UseCluster, [&] {
            run_flash_fwd<Flash_fwd_kernel_traits_fp8<Headdim, kBlockM, kBlockN, kNWarps, kStages,
                            false, UseCluster ? 2 : 1, T>, /*Is_causal=*/false, Seqlen_traits>(params, stream);
        });
    }
    // BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        // SEQLEN_SWITCH(params.cu_seqlens_q, Seqlen_traits, [&] {
            // Only use Cluster if number of tiles along seqlen_q is even
            // BOOL_SWITCH(cutlass::ceil_div(params.seqlen_q, kBlockM) % 2 == 0 && !Is_causal &&
            //             !Seqlen_traits::kUseVarSeqLen, UseCluster, [&] {
            //     run_flash_fwd<Flash_fwd_kernel_traits_fp8<Headdim, kBlockM, kBlockN, kNWarps, kStages,
            //                   false, UseCluster ? 2 : 1, T>, Is_causal, Seqlen_traits>(params, stream);
            // });
        // });
    // });
}
