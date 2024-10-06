/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"  // For device_kernel
#include <cutlass/kernel_hardware_info.h>
#include "cutlass/cluster_launch.hpp"

#include "static_switch.h"
#include "flash.h"
#include "tile_scheduler.hpp"
#include "flash_fwd_kernel.h"
#include "mainloop_fwd_sm90_tma_gmma_ws.hpp"
#include "epilogue_fwd_sm90_tma.hpp"


using namespace cute;

template <int kHeadDim, int kBlockM, int kBlockN, int Stages, int ClusterM, typename Element, typename ElementOut,
          bool Is_causal, bool Is_local, bool Has_softcap, bool Varlen, bool V_colmajor>
void run_flash_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    static_assert(!(Is_causal && Is_local), "Causal and Local cannot be enabled at the same time");
    static constexpr bool Is_FP8 = cute::is_same_v<Element, cutlass::float_e4m3_t> || cute::is_same_v<Element, cutlass::float_e5m2_t>;;
    static constexpr bool FP8_TransposeV = Is_FP8 && !V_colmajor;
    using TileShape_MNK = cute::Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
    using ClusterShape = cute::Shape<Int<ClusterM>, _1, _1>;
    using CollectiveMainloop = flash::CollectiveMainloopFwd<Stages, ClusterShape, TileShape_MNK, Element, float, cutlass::arch::Sm90, Is_causal, Is_local, Has_softcap, Varlen, V_colmajor>;
    using CollectiveEpilogue = flash::CollectiveEpilogueFwd<TileShape_MNK, ElementOut, CollectiveMainloop::NumMmaThreads, Varlen, FP8_TransposeV>;

    using Scheduler = std::conditional_t<Varlen,
        // flash::SingleTileScheduler<Varlen, kBlockM>,
        flash::VarlenDynamicPersistentTileScheduler<kBlockM, CollectiveMainloop::NumMmaThreads, CollectiveMainloop::NumProducerThreads>,
        std::conditional_t<!Is_causal && !Is_local,
            flash::StaticPersistentTileScheduler,
            flash::DynamicPersistentTileScheduler<CollectiveMainloop::NumMmaThreads, CollectiveMainloop::NumProducerThreads>>
            // flash::SingleTileScheduler<Varlen, kBlockM>>
    >;
    // using Scheduler = flash::SingleTileScheduler<Varlen, kBlockM>;
    using AttnKernel = std::conditional_t<!FP8_TransposeV,
        flash::FlashAttnFwd<CollectiveMainloop, CollectiveEpilogue, Scheduler>,
        flash::FlashAttnFwdFP8TransposeV<CollectiveMainloop, CollectiveEpilogue, Scheduler>
    >;

    typename CollectiveMainloop::StrideV v_strides =
        cute::conditional_return<!V_colmajor>(
            make_stride(params.v_row_stride, _1{}, params.v_head_stride, !Varlen ? params.v_batch_stride : 0),
            make_stride(_1{}, params.v_dim_stride, params.v_head_stride, !Varlen ? params.v_batch_stride : 0));
    // print(typename CollectiveMainloop::SmemLayoutVTma{}); printf("\n");
    // print(typename CollectiveMainloop::SmemLayoutVMma{}); printf("\n");
    typename CollectiveMainloop::Arguments mainloop_args {
        static_cast<Element const*>(params.q_ptr),
            {!Varlen ? params.seqlen_q : params.total_q, params.d, params.h, !Varlen ? params.b : 1},  // shape_Q
            {params.q_row_stride, _1{}, params.q_head_stride, !Varlen ? params.q_batch_stride : 0},  // stride_Q
            static_cast<Element const*>(params.k_ptr),
            {!Varlen ? params.seqlen_k : params.total_k, params.d, params.h_k, !Varlen ? params.b : 1},  // shape_K
            {params.k_row_stride, _1{}, params.k_head_stride, !Varlen ? params.k_batch_stride : 0},  // stride_K
            static_cast<Element const*>(params.v_ptr),
            v_strides,  // stride_V
        params.scale_softmax,
        params.q_scale_ptr, params.k_scale_ptr, params.v_scale_ptr,
        params.window_size_left, params.window_size_right,
        params.softcap,
        params.cu_seqlens_q, params.cu_seqlens_k,
        params.seqused_q, params.seqused_k,
    };
    typename CollectiveEpilogue::Arguments epilogue_args {
        static_cast<ElementOut*>(params.o_ptr),
        {!Varlen ? params.seqlen_q : params.total_q, params.d, params.h, !Varlen ? params.b : 1},  // shape_O
        {params.o_row_stride, _1{}, params.o_head_stride, !Varlen ? params.o_batch_stride : 0},  // stride_O
        static_cast<float*>(params.softmax_lse_ptr),
        {_1{}, !Varlen ? params.seqlen_q : params.total_q, !Varlen ? params.h * params.seqlen_q : 0},  // stride_LSE
        params.cu_seqlens_q, params.seqused_q
    };

    int num_blocks_m = cutlass::ceil_div(params.seqlen_q, get<0>(TileShape_MNK{}));
    num_blocks_m = cutlass::round_up(num_blocks_m, size<0>(ClusterShape{}));
    typename Scheduler::Arguments scheduler_args {
        num_blocks_m, params.h, params.b, params.tile_count_semaphore, params.cu_seqlens_q, params.seqused_q
    };

    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    typename AttnKernel::Params kernel_params = AttnKernel::to_underlying_arguments({
        mainloop_args, epilogue_args, {device}, scheduler_args
    });

    dim3 grid_dims = AttnKernel::get_grid_shape(kernel_params);
    dim3 block_dims = AttnKernel::get_block_shape();
    int smem_size = AttnKernel::SharedStorageSize;
    // int smem_size_q = sizeof(decltype((typename Kernel_traits::SharedStorage{}).smem_q));
    // int smem_size_k = sizeof(decltype((typename Kernel_traits::SharedStorage{}).smem_k));
    // int smem_size_v = sizeof(decltype((typename Kernel_traits::SharedStorage{}).smem_v));
    // printf("smem_size = %d, q = %d, k = %d, v = %d\n", smem_size, smem_size_q, smem_size_k, smem_size_v);
    // Get the ptr to kernel function.
    if constexpr (size(ClusterShape{}) > 1) {
        void const* kernel = (void const*) cutlass::device_kernel<AttnKernel>;
        if (smem_size >= 48 * 1024) {
            CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        }
        dim3 cluster_dims(size<0>(ClusterShape{}), size<1>(ClusterShape{}), size<2>(ClusterShape{}));
        cutlass::ClusterLaunchParams launch_params{grid_dims, block_dims, cluster_dims, smem_size, stream};
        cutlass::launch_kernel_on_cluster(launch_params, kernel, kernel_params);
    } else {
        auto kernel = cutlass::device_kernel<AttnKernel>;
        if (smem_size >= 48 * 1024) {
            CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        }
        kernel<<<grid_dims, block_dims, smem_size, stream>>>(kernel_params);
    }
    CHECK_CUDA_KERNEL_LAUNCH();
}

template<typename T, int kBlockM, int kBlockN, int kHeadDim, bool Is_causal, bool Is_local, bool Enable_cluster>
void run_mha_fwd_dispatch(Flash_fwd_params &params, cudaStream_t stream) {
    BOOL_SWITCH(params.cu_seqlens_q != nullptr || params.cu_seqlens_k != nullptr, Varlen, [&] {
        // Only use Cluster if number of tiles along seqlen_q is even and not varlen
        BOOL_SWITCH(cutlass::ceil_div(params.seqlen_q, kBlockM) % 2 == 0, UseCluster, [&] {
            BOOL_SWITCH(params.softcap > 0.0, Has_softcap, [&] {
                run_flash_fwd<kHeadDim, kBlockM, kBlockN, 2 /*Stages*/, !Is_causal && !Is_local && !Varlen && Enable_cluster && UseCluster ? 2 : 1, T, T, Is_causal, Is_local, Has_softcap, Varlen, false /*V_colmajor*/>(params, stream);
            });
        });
    });
}

template<typename T>
void run_mha_fwd_hdim64(Flash_fwd_params &params, cudaStream_t stream) {
    CAUSAL_LOCAL_SWITCH(params.is_causal, params.is_local, Is_causal, Is_local, [&] {
        run_mha_fwd_dispatch<T, 192, 128, 64, Is_causal, Is_local, false /*Enable_cluster*/>(params, stream);
    });
}

template<typename T>
void run_mha_fwd_hdim96(Flash_fwd_params &params, cudaStream_t stream) {
    CAUSAL_LOCAL_SWITCH(params.is_causal, params.is_local, Is_causal, Is_local, [&] {
        run_mha_fwd_dispatch<T, 128, Is_causal || Is_local ? 128 : 160, 96, Is_causal, Is_local, true /*Enable_cluster*/>(params, stream);
    });
}

template<typename T>
void run_mha_fwd_hdim128(Flash_fwd_params &params, cudaStream_t stream) {
    CAUSAL_LOCAL_SWITCH(params.is_causal, params.is_local, Is_causal, Is_local, [&] {
        run_mha_fwd_dispatch<T, 128, Is_causal || Is_local ? 128 : 176, 128, Is_causal, Is_local, true /*Enable_cluster*/>(params, stream);
    });
}

template<typename T>
void run_mha_fwd_hdim192(Flash_fwd_params &params, cudaStream_t stream) {
    CAUSAL_LOCAL_SWITCH(params.is_causal, params.is_local, Is_causal, Is_local, [&] {
        run_mha_fwd_dispatch<T, 128, 96, 192, Is_causal, Is_local, true /*Enable_cluster*/>(params, stream);
    });
}

template<typename T>
void run_mha_fwd_hdim256(Flash_fwd_params &params, cudaStream_t stream) {
    CAUSAL_LOCAL_SWITCH(params.is_causal, params.is_local, Is_causal, Is_local, [&] {
        run_mha_fwd_dispatch<T, 128, 80, 256, Is_causal, Is_local, true /*Enable_cluster*/>(params, stream);

    });
}

template<typename T, int kBlockM, int kBlockN, int kHeadDim, int kStages,
         bool Is_causal, bool Is_local, bool V_colmajor, bool Enable_cluster>
void run_mha_fwd_fp8_dispatch(Flash_fwd_params &params, cudaStream_t stream) {
    BOOL_SWITCH(params.cu_seqlens_q != nullptr || params.cu_seqlens_k != nullptr, Varlen, [&] {
        // Only use Cluster if number of tiles along seqlen_q is even and not varlen
        BOOL_SWITCH(cutlass::ceil_div(params.seqlen_q, kBlockM) % 2 == 0, UseCluster, [&] {
            run_flash_fwd<kHeadDim, kBlockM, kBlockN, kStages, !Is_causal && !Is_local && !Varlen && Enable_cluster && UseCluster ? 2 : 1, T, cutlass::bfloat16_t, Is_causal, Is_local, false /*Has_softcap*/, Varlen, V_colmajor && !Varlen>(params, stream);
        });
    });
}

template<typename T>
void run_mha_fwd_fp8_hdim64(Flash_fwd_params &params, cudaStream_t stream) {
    CAUSAL_LOCAL_SWITCH(params.is_causal, params.is_local, Is_causal, Is_local, [&] {
        BOOL_SWITCH(params.v_dim_stride != 1, V_colmajor, [&] {
            run_mha_fwd_fp8_dispatch<T, 192, 160, 64, 3, Is_causal, Is_local, V_colmajor, false /*Enable_cluster*/>(params, stream);
        });
    });
}

template<typename T>
void run_mha_fwd_fp8_hdim96(Flash_fwd_params &params, cudaStream_t stream) {
    CAUSAL_LOCAL_SWITCH(params.is_causal, params.is_local, Is_causal, Is_local, [&] {
        BOOL_SWITCH(params.v_dim_stride != 1, V_colmajor, [&] {
            run_mha_fwd_fp8_dispatch<T, 192, 128, 96, 3, Is_causal, Is_local, V_colmajor, false /*Enable_cluster*/>(params, stream);
        });
    });
}


template<typename T>
void run_mha_fwd_fp8_hdim128(Flash_fwd_params &params, cudaStream_t stream) {
    CAUSAL_LOCAL_SWITCH(params.is_causal, params.is_local, Is_causal, Is_local, [&] {
        BOOL_SWITCH(params.v_dim_stride != 1, V_colmajor, [&] {
            run_mha_fwd_fp8_dispatch<T, 128, V_colmajor ? 192 : 224, 128, 2, Is_causal, Is_local, V_colmajor, true /*Enable_cluster*/>(params, stream);
        });
    });
}

template<typename T>
void run_mha_fwd_fp8_hdim192(Flash_fwd_params &params, cudaStream_t stream) {
    CAUSAL_LOCAL_SWITCH(params.is_causal, params.is_local, Is_causal, Is_local, [&] {
        BOOL_SWITCH(params.v_dim_stride != 1, V_colmajor, [&] {
            run_mha_fwd_fp8_dispatch<T, 128, 160, 192, 2, Is_causal, Is_local, V_colmajor, true /*Enable_cluster*/>(params, stream);
        });
    });
}

template<typename T>
void run_mha_fwd_fp8_hdim256(Flash_fwd_params &params, cudaStream_t stream) {
    CAUSAL_LOCAL_SWITCH(params.is_causal, params.is_local, Is_causal, Is_local, [&] {
        BOOL_SWITCH(params.v_dim_stride != 1, V_colmajor, [&] {
            run_mha_fwd_fp8_dispatch<T, 128, 128, 256, 2, Is_causal, Is_local, V_colmajor, true /*Enable_cluster*/>(params, stream);
        });
    });
}
