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
#include "tile_size.h"
#include "tile_scheduler.hpp"
#include "flash_fwd_kernel.h"
#include "mainloop_fwd_sm90_tma_gmma_ws.hpp"
#include "epilogue_fwd_sm90_tma.hpp"


using namespace cute;

template <int kHeadDim, int kBlockM, int kBlockN, int kStages, int ClusterM, typename Element, typename ElementOut,
          bool Is_causal, bool Is_local, bool Has_softcap, bool Varlen, bool PagedKV, bool AppendKV,
          bool Mma1_is_RS, bool IntraWGOverlap, bool PackGQA, bool Split, bool V_colmajor>
void run_flash_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    static_assert(!(Is_causal && Is_local), "Causal and Local cannot be enabled at the same time");
    static_assert(!(AppendKV && V_colmajor), "AppendKV and V_colmajor cannot be enabled at the same time");
    static_assert(!(AppendKV && !Varlen), "AppendKV requires Varlen");
    static constexpr bool Is_FP8 = cute::is_same_v<Element, cutlass::float_e4m3_t> || cute::is_same_v<Element, cutlass::float_e5m2_t>;
    static constexpr bool FP8_TransposeV = Is_FP8 && !V_colmajor;
    using TileShape_MNK = cute::Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
    using ClusterShape = cute::Shape<Int<ClusterM>, _1, _1>;
    using CollectiveMainloop = flash::CollectiveMainloopFwd<kStages, ClusterShape, TileShape_MNK, Element, float, cutlass::arch::Sm90, Is_causal, Is_local, Has_softcap, Varlen, PagedKV, AppendKV, Mma1_is_RS, IntraWGOverlap, PackGQA, Split, V_colmajor>;
    using CollectiveEpilogue = flash::CollectiveEpilogueFwd<TileShape_MNK, ClusterShape, ElementOut, CollectiveMainloop::NumMmaThreads, Varlen, PackGQA, FP8_TransposeV>;

    using SchedulerPersistent = std::conditional_t<Varlen,
        flash::VarlenDynamicPersistentTileScheduler<kBlockM, CollectiveMainloop::NumMmaThreads, CollectiveMainloop::NumProducerThreads, Split, PackGQA>,
        std::conditional_t<!Is_causal && !Is_local,
            flash::StaticPersistentTileScheduler<Split>,
            flash::DynamicPersistentTileScheduler<CollectiveMainloop::NumMmaThreads, CollectiveMainloop::NumProducerThreads, Split, PackGQA>
        >
    >;
    using SchedulerSingleTile = flash::SingleTileScheduler<Varlen, Split, PackGQA, kBlockM>;
    // If Split, PagedKV, or AppendKV then we probably don't have enough work for PersistentScheduler to be useful.
    using Scheduler = std::conditional_t<Split || PagedKV || AppendKV, SchedulerSingleTile, SchedulerPersistent>;
    using AttnKernel = flash::FlashAttnFwd<CollectiveMainloop, CollectiveEpilogue, Scheduler>;

    bool const is_varlen_q = params.cu_seqlens_q;
    bool const is_varlen_k = params.cu_seqlens_k;
    bool const is_varlen_k_new = params.cu_seqlens_knew;
    int seqlen_q = !is_varlen_q ? params.seqlen_q : params.total_q;
    int batch_q = !is_varlen_q ? params.b : 1;
    int batch_k = !is_varlen_k ? (params.kv_batch_idx ? params.b_k : params.b) : 1;
    typename CollectiveMainloop::StrideV v_strides =
        cute::conditional_return<!V_colmajor>(
            make_stride(params.v_row_stride, _1{}, params.v_head_stride, !is_varlen_k ? params.v_batch_stride : 0),
            make_stride(_1{}, params.v_dim_stride, params.v_head_stride, !is_varlen_k ? params.v_batch_stride : 0));
    typename CollectiveMainloop::Arguments mainloop_args {
        static_cast<Element const*>(params.q_ptr),
        {seqlen_q, params.d, params.h, batch_q},  // shape_Q
        {params.q_row_stride, _1{}, params.q_head_stride, !is_varlen_q ? params.q_batch_stride : 0},  // stride_Q
        static_cast<Element*>(params.k_ptr),
        {!PagedKV ? (!is_varlen_k ? params.seqlen_k : params.total_k) : params.page_size,
         params.d, params.h_k, !PagedKV ? batch_k : params.num_pages},  // shape_K
        {params.k_row_stride, _1{}, params.k_head_stride, !is_varlen_k ? params.k_batch_stride : 0},  // stride_K
        static_cast<Element*>(params.v_ptr),
        v_strides,  // stride_V
        static_cast<Element const*>(params.knew_ptr),
        {!is_varlen_k_new ? params.seqlen_knew : params.total_knew, params.d, params.h_k, !is_varlen_k_new ? params.b : 1},  // shape_K_new
        {params.knew_row_stride, _1{}, params.knew_head_stride, !is_varlen_k_new ? params.knew_batch_stride : 0},  // stride_K_new
        static_cast<Element const*>(params.vnew_ptr),
        {params.vnew_row_stride, _1{}, params.vnew_head_stride, !is_varlen_k_new ? params.vnew_batch_stride : 0}, // stride_V_new
        static_cast<Element const*>(params.rotary_cos_ptr),
        {params.seqlen_k, params.rotary_dim / 2},  // shape_rotary, the seqlen shape doesn't matter
        {params.rotary_dim / 2, _1{}},  // stride_rotary_cos
        static_cast<Element const*>(params.rotary_sin_ptr),
        {params.rotary_dim / 2, _1{}},  // stride_rotary_sin
        params.is_rotary_interleaved,
        params.page_table,
        // if page_size is not set, avoid dividing by zero
        {params.kv_batch_idx ? params.b_k : params.b, !PagedKV ? 0 : params.seqlen_k / params.page_size}, // shape_page_table
        {params.page_table_batch_stride, _1{}},  // stride_page_table
        params.scale_softmax,
        params.q_descale_ptr, params.k_descale_ptr, params.v_descale_ptr,
        {params.q_descale_batch_stride, params.q_descale_head_stride},
        {params.k_descale_batch_stride, params.k_descale_head_stride},
        {params.v_descale_batch_stride, params.v_descale_head_stride},
        params.window_size_left, params.window_size_right, params.sink_token_length,
        params.softcap,
        params.num_splits,
        params.kv_batch_idx,
        params.cu_seqlens_q, params.cu_seqlens_k, params.cu_seqlens_knew,
        params.seqused_q, params.seqused_k,
        params.leftpad_k,
    };
    typename CollectiveEpilogue::Arguments epilogue_args {
        static_cast<ElementOut*>(!Split ? params.o_ptr : params.oaccum_ptr),
        {seqlen_q, params.d, params.h, batch_q, params.num_splits},  // shape_O
        {!Split ? params.o_row_stride : params.oaccum_row_stride,
         _1{},
         !Split ? params.o_head_stride : params.oaccum_head_stride,
         !is_varlen_q ? (!Split ? params.o_batch_stride : params.oaccum_batch_stride) : 0,
         !Split ? 0 : params.oaccum_split_stride},  // stride_O
        static_cast<float*>(!Split ? params.softmax_lse_ptr : params.softmax_lseaccum_ptr),
        {_1{}, seqlen_q, !is_varlen_q ? params.h * seqlen_q : 0, !Split ? 0 : params.h * seqlen_q * batch_q},  // stride_LSE
        params.h_k,
        params.cu_seqlens_q, params.seqused_q
    };

    int qhead_per_khead = !PackGQA ? 1 : cutlass::ceil_div(params.h, params.h_k);
    int num_blocks_m = cutlass::ceil_div(params.seqlen_q * qhead_per_khead, get<0>(TileShape_MNK{}));
    num_blocks_m = cutlass::round_up(num_blocks_m, size<0>(ClusterShape{}));
    typename flash::TileSchedulerArguments scheduler_args {
        num_blocks_m, !PackGQA ? params.h : params.h_k, params.b, params.num_splits,
        params.h / params.h_k,
        params.seqlen_q,
        params.seqlen_k, params.d, sizeof(Element),
        params.tile_count_semaphore, params.cu_seqlens_q, params.seqused_q
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

template<typename T, int kBlockM, int kBlockN, int kHeadDim, int kStages,
         bool Is_causal, bool Is_local, bool Has_softcap, bool PagedKV, bool Mma1_is_RS, bool IntraWGOverlap,
         bool Split, bool V_colmajor, bool Enable_cluster>
void run_mha_fwd_dispatch(Flash_fwd_params &params, cudaStream_t stream) {
    static constexpr bool Is_FP8 = cute::is_same_v<T, cutlass::float_e4m3_t> || cute::is_same_v<T, cutlass::float_e5m2_t>;
    using T_out = std::conditional_t<!Split, std::conditional_t<!Is_FP8, T, cutlass::bfloat16_t>, float>;
    VARLEN_SWITCH(params.cu_seqlens_q || params.cu_seqlens_k || params.seqused_q || params.seqused_k || params.leftpad_k, Varlen, [&] {
        APPENDKV_SWITCH(params.knew_ptr, AppendKV, [&] {
            PACKGQA_SWITCH(params.pack_gqa, PackGQA, [&] {
                // Only use Cluster if number of tiles along seqlen_q is even and not varlen
                CLUSTER_SWITCH(cutlass::ceil_div(params.seqlen_q * (!PackGQA ? 1 : params.h / params.h_k), kBlockM) % 2 == 0, Use_cluster, [&] {
                    static constexpr int ClusterM = !Varlen && Enable_cluster && Use_cluster ? 2 : 1;
                    run_flash_fwd<kHeadDim, kBlockM, kBlockN, kStages, ClusterM, T, T_out, Is_causal, Is_local, Has_softcap, Varlen, PagedKV, AppendKV && Varlen, Mma1_is_RS, IntraWGOverlap, PackGQA, Split, V_colmajor>(params, stream);
                });
            });
        });
    });
}

template<typename T, int kHeadDim, bool Split, bool PagedKV>
void run_mha_fwd_16b(Flash_fwd_params &params, cudaStream_t stream) {
    CAUSAL_LOCAL_SWITCH(params.is_causal, params.is_local, Is_causal, Is_local, [&] {
        SOFTCAP_SWITCH(params.softcap > 0.0, Has_softcap, [&] {
            // Can't use structured binding since it's not compatible with constexpr
            static constexpr std::tuple<int, int, bool, bool> kBlockMN_RS_IntraWGOverlap = tile_size_fwd(kHeadDim, Is_causal, Is_local, sizeof(T) /*element_size*/, false /*V_colmajor*/, PagedKV, Has_softcap);
            static constexpr bool Enable_cluster = kHeadDim >= 128 && !Is_causal && !Is_local && !Split && !PagedKV;
            run_mha_fwd_dispatch<T, std::get<0>(kBlockMN_RS_IntraWGOverlap), std::get<1>(kBlockMN_RS_IntraWGOverlap), kHeadDim, 2,
                Is_causal, Is_local, Has_softcap, PagedKV, std::get<2>(kBlockMN_RS_IntraWGOverlap), std::get<3>(kBlockMN_RS_IntraWGOverlap), Split, false /*V_colmajor*/, Enable_cluster>(params, stream);
        });
    });
}

template<typename T, int kHeadDim, bool Split, bool PagedKV>
void run_mha_fwd_8b(Flash_fwd_params &params, cudaStream_t stream) {
    CAUSAL_LOCAL_SWITCH(params.is_causal, params.is_local, Is_causal, Is_local, [&] {
        VCOLMAJOR_SWITCH(params.v_dim_stride != 1, V_colmajor, [&] {
            SOFTCAP_SWITCH(params.softcap > 0.0, Has_softcap, [&] {
                // Can't use structured binding since it's not compatible with constexpr
                static constexpr std::tuple<int, int, bool, bool> kBlockMN_RS_IntraWGOverlap = tile_size_fwd(kHeadDim, Is_causal, Is_local, sizeof(T) /*element_size*/, V_colmajor /*V_colmajor*/, PagedKV, Has_softcap);
                static constexpr bool Enable_cluster = kHeadDim == 192 && !Is_causal && !Is_local && !Split && !PagedKV;
                run_mha_fwd_dispatch<T, std::get<0>(kBlockMN_RS_IntraWGOverlap), std::get<1>(kBlockMN_RS_IntraWGOverlap), kHeadDim, 2,
                    Is_causal, Is_local, Has_softcap, PagedKV, std::get<2>(kBlockMN_RS_IntraWGOverlap), std::get<3>(kBlockMN_RS_IntraWGOverlap), Split, V_colmajor, Enable_cluster>(params, stream);
            });
        });
    });
}
