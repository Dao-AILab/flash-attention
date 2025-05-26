/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"  // For device_kernel
#include <cutlass/kernel_hardware_info.h>
#include "cutlass/cluster_launch.hpp"
#include "cutlass/kernel_launch.h"

#include "static_switch.h"
#include "flash.h"
#include "tile_size.h"
#include "tile_scheduler.hpp"
#include "flash_fwd_kernel_sm90.h"
#include "flash_fwd_kernel_sm80.h"
#include "mainloop_fwd_sm90_tma_gmma_ws.hpp"
#include "mainloop_fwd_sm80.hpp"
#include "epilogue_fwd.hpp"

using namespace cute;

template <int Arch, int kHeadDim, int kHeadDimV, int ClusterM, typename Element, typename ElementOut,
          bool Is_causal, bool Is_local, bool Has_softcap, bool Varlen, bool PagedKVNonTMA, bool AppendKV, bool HasQv,
          bool PackGQA, bool Split, bool V_colmajor>
void run_flash_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    static_assert(!(Is_causal && Is_local), "Causal and Local cannot be enabled at the same time");
    static_assert(!(AppendKV && V_colmajor), "AppendKV and V_colmajor cannot be enabled at the same time");
    static_assert(!(AppendKV && !Varlen), "AppendKV requires Varlen");
    static constexpr bool Is_FP8 = cute::is_same_v<Element, cutlass::float_e4m3_t> || cute::is_same_v<Element, cutlass::float_e5m2_t>;
    static constexpr bool FP8_TransposeV = Is_FP8 && !V_colmajor;
    using ArchTag = std::conditional_t<Arch >= 90, cutlass::arch::Sm90, cutlass::arch::Sm80>;

    // Can't use structured binding since it's not compatible with constexpr
    static constexpr std::tuple<int, int, bool, bool> kBlockMN_RS_IntraWGOverlap = tile_size_fwd_sm90(kHeadDim, kHeadDimV, Is_causal, Is_local, sizeof(Element) /*element_size*/, V_colmajor, PagedKVNonTMA, Has_softcap);
    static constexpr std::tuple<int, int, int, int, bool> kBlockMN_kNWarps_Stages_RS = tile_size_fwd_sm8x(Arch == 86 || Arch == 89, kHeadDim, kHeadDimV, Is_causal, Is_local, sizeof(Element) /*element_size*/, PagedKVNonTMA, Varlen && Split, Has_softcap, AppendKV);
    static constexpr int kBlockM = Arch >= 90 ? std::get<0>(kBlockMN_RS_IntraWGOverlap) : std::get<0>(kBlockMN_kNWarps_Stages_RS);
    static constexpr int kBlockN = Arch >= 90 ? std::get<1>(kBlockMN_RS_IntraWGOverlap) : std::get<1>(kBlockMN_kNWarps_Stages_RS);
    static constexpr bool MmaPV_is_RS = std::get<2>(kBlockMN_RS_IntraWGOverlap);
    static constexpr bool IntraWGOverlap = std::get<3>(kBlockMN_RS_IntraWGOverlap);
    static constexpr int kNWarps = std::get<2>(kBlockMN_kNWarps_Stages_RS);
    static constexpr int kStages = Arch >= 90 ? 2 : std::get<3>(kBlockMN_kNWarps_Stages_RS);
    static constexpr bool Q_in_regs = Arch >= 90 ? false : std::get<4>(kBlockMN_kNWarps_Stages_RS);

    using TileShape_MNK = cute::Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
    using TileShape_MNK_PV = cute::Shape<Int<kBlockM>, Int<kHeadDimV>, Int<kBlockN>>;
    using ClusterShape = cute::Shape<Int<ClusterM>, _1, _1>;
    using CollectiveMainloop = std::conditional_t<
        Arch >= 90,
        flash::CollectiveMainloopFwdSm90<kStages, ClusterShape, TileShape_MNK, kHeadDimV, Element, float, cutlass::arch::Sm90, Is_causal, Is_local, Has_softcap, Varlen, PagedKVNonTMA, AppendKV, HasQv, MmaPV_is_RS, IntraWGOverlap, PackGQA, Split, V_colmajor>,
        flash::CollectiveMainloopFwdSm80<kNWarps, kStages, Q_in_regs, TileShape_MNK, kHeadDimV, Element, float, cutlass::arch::Sm80, Is_causal, Is_local, Has_softcap, Varlen, PagedKVNonTMA, AppendKV, PackGQA, Split>
    >;
    using CollectiveEpilogue = flash::CollectiveEpilogueFwd<TileShape_MNK_PV, ClusterShape, ElementOut, ArchTag, CollectiveMainloop::NumMmaThreads, Varlen, PackGQA, Split, FP8_TransposeV>;

    static constexpr int NumProducerThreads = Arch >= 90 ? CollectiveMainloop::NumProducerThreads : CollectiveMainloop::NumMmaThreads;
    using SchedulerPersistent = std::conditional_t<Varlen,
        flash::VarlenDynamicPersistentTileScheduler<kBlockM, CollectiveMainloop::NumMmaThreads, NumProducerThreads, Split, PackGQA, Arch >= 90 /*WarpSpecialized*/>,
        std::conditional_t<!Is_causal && !Is_local,
            flash::StaticPersistentTileScheduler<Split>,
            flash::DynamicPersistentTileScheduler<CollectiveMainloop::NumMmaThreads, NumProducerThreads, Split, PackGQA, Arch >= 90 /*WarpSpecialized*/>
        >
    >;
    using SchedulerSingleTile = flash::SingleTileScheduler<Varlen, Split, PackGQA, kBlockM>;
    // If Split then we probably don't have enough work for PersistentScheduler to be useful.
    // However, if Varlen (e.g., during decode where we have max_seqlens), using PersistentScheduler is better
    // since we'll avoid launching a bunch of thread blocks that immediately exit.
    // On Sm80, noncausal persistent seems a bit slower.
    static constexpr bool UsePersistentScheduler = Arch >= 90 ? !(Split && !Varlen) : ((Is_causal && !Varlen) || (Varlen && Split));
    using Scheduler = std::conditional_t<!UsePersistentScheduler, SchedulerSingleTile, SchedulerPersistent>;
    using AttnKernel = std::conditional_t<
        Arch >= 90,
        flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<CollectiveMainloop, CollectiveEpilogue, Scheduler>>,
        flash::enable_sm80_to_sm89<flash::FlashAttnFwdSm80<CollectiveMainloop, CollectiveEpilogue, Scheduler>>
    >;

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
        {!params.page_table ? (!is_varlen_k ? params.seqlen_k : params.total_k) : params.page_size,
         params.d, params.h_k, !params.page_table ? batch_k : params.num_pages},  // shape_K
        {params.k_row_stride, _1{}, params.k_head_stride, !is_varlen_k ? params.k_batch_stride : 0},  // stride_K
        static_cast<Element*>(params.v_ptr),
        params.dv,  // headdim_v
        v_strides,  // stride_V
        static_cast<Element const*>(params.knew_ptr),
        {!is_varlen_k_new ? params.seqlen_knew : params.total_knew, params.d, params.h_k, !is_varlen_k_new ? params.b : 1},  // shape_K_new
        {params.knew_row_stride, _1{}, params.knew_head_stride, !is_varlen_k_new ? params.knew_batch_stride : 0},  // stride_K_new
        static_cast<Element const*>(params.vnew_ptr),
        {params.vnew_row_stride, _1{}, params.vnew_head_stride, !is_varlen_k_new ? params.vnew_batch_stride : 0}, // stride_V_new
        static_cast<Element const*>(params.qv_ptr),
        {params.qv_row_stride, _1{}, params.qv_head_stride, !is_varlen_q ? params.qv_batch_stride : 0},  // stride_Qv
        static_cast<Element const*>(params.rotary_cos_ptr),
        {params.seqlen_k, params.rotary_dim / 2},  // shape_rotary, the seqlen shape doesn't matter
        {params.rotary_dim / 2, _1{}},  // stride_rotary_cos
        static_cast<Element const*>(params.rotary_sin_ptr),
        {params.rotary_dim / 2, _1{}},  // stride_rotary_sin
        params.is_rotary_interleaved,
        params.page_table,
        // if page_size is not set, avoid dividing by zero
        {params.kv_batch_idx ? params.b_k : params.b, !params.page_table ? 0 : params.seqlen_k / params.page_size}, // shape_page_table
        {params.page_table_batch_stride, _1{}},  // stride_page_table
        params.scale_softmax,
        params.q_descale_ptr, params.k_descale_ptr, params.v_descale_ptr,
        {params.q_descale_batch_stride, params.q_descale_head_stride},
        {params.k_descale_batch_stride, params.k_descale_head_stride},
        {params.v_descale_batch_stride, params.v_descale_head_stride},
        params.window_size_left, params.window_size_right, params.attention_chunk,
        params.softcap,
        params.num_splits,
        params.kv_batch_idx,
        params.cu_seqlens_q, params.cu_seqlens_k, params.cu_seqlens_knew,
        params.seqused_q, params.seqused_k,
        params.leftpad_k, params.seqlens_rotary
    };
    typename CollectiveEpilogue::Arguments epilogue_args {
        static_cast<ElementOut*>(params.o_ptr),
        {seqlen_q, params.dv, params.h, batch_q, params.num_splits},  // shape_O
        {params.o_row_stride, _1{}, params.o_head_stride, !is_varlen_q ? params.o_batch_stride : 0, 0}, // stride_O
        static_cast<float*>(params.oaccum_ptr),
        {params.oaccum_row_stride, _1{}, params.oaccum_head_stride, !is_varlen_q ? params.oaccum_batch_stride : 0, params.oaccum_split_stride}, // stride_O_partial
        static_cast<float*>(params.softmax_lse_ptr),
        {_1{}, seqlen_q, !is_varlen_q ? params.h * seqlen_q : 0, 0},  // stride_LSE
        static_cast<float*>(params.softmax_lseaccum_ptr),
        {_1{}, seqlen_q, !is_varlen_q ? params.h * seqlen_q : 0, params.h * seqlen_q * batch_q},  // stride_LSE_partial
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
        params.seqlen_k, params.d, params.dv, sizeof(Element),
        params.tile_count_semaphore, params.cu_seqlens_q, params.seqused_q,
        // params.num_m_blocks_ptr,
        params.num_splits_dynamic_ptr,
    };

    if (Varlen && params.num_splits_dynamic_ptr && !params.skip_scheduler_metadata_computation) {
        prepare_varlen_num_blocks(params, stream, PackGQA, kBlockM, kBlockN, Arch >= 90 /*enable_pdl*/);
        CHECK_CUDA_KERNEL_LAUNCH();
    }

    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    typename AttnKernel::Params kernel_params = AttnKernel::to_underlying_arguments({
        mainloop_args, epilogue_args, {device, params.num_sm}, scheduler_args
    });

    dim3 grid_dims = AttnKernel::get_grid_shape(kernel_params);
    dim3 block_dims = AttnKernel::get_block_shape();
    int smem_size = AttnKernel::SharedStorageSize;
    // int smem_size_q = sizeof(decltype((typename CollectiveMainloop::TensorStorage{}).smem_q));
    // int smem_size_k = sizeof(decltype((typename CollectiveMainloop::TensorStorage{}).smem_k));
    // int smem_size_v = sizeof(decltype((typename CollectiveMainloop::TensorStorage{}).smem_v));
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
        // kernel<<<grid_dims, block_dims, smem_size, stream>>>(kernel_params);
        cutlass::kernel_launch<AttnKernel>(grid_dims, block_dims, smem_size, stream, kernel_params,
                                           Arch >= 90 && Varlen && params.num_splits_dynamic_ptr && !params.skip_scheduler_metadata_computation /*launch_with_pdl*/);
    }
    CHECK_CUDA_KERNEL_LAUNCH();
}

template<int Arch, typename T, int kHeadDim, int kHeadDimV, bool Split, bool PagedKVNonTMA, bool Has_softcap, bool PackGQA>
void run_mha_fwd_(Flash_fwd_params &params, cudaStream_t stream) {
    static_assert(sizeof(T) == 2 || sizeof(T) == 1, "Only 16bit and 8bit are supported");
    static constexpr bool Is_FP8 = cute::is_same_v<T, cutlass::float_e4m3_t> || cute::is_same_v<T, cutlass::float_e5m2_t>;
    using T_out = std::conditional_t<!Is_FP8, T, cutlass::bfloat16_t>;
    CAUSAL_LOCAL_SWITCH(params.is_causal, params.is_local, Is_causal, Is_local, [&] {
        VCOLMAJOR_SWITCH(params.v_dim_stride != 1, V_colmajor_, [&] {
            static constexpr bool V_colmajor = V_colmajor_ && sizeof(T) == 1;
            VARLEN_SWITCH(params.cu_seqlens_q || params.cu_seqlens_k || params.seqused_q || params.seqused_k || params.leftpad_k, Varlen, [&] {
                // Only needed here to decide if we should use cluster
                static constexpr int kBlockM = Arch >= 90 ? std::get<0>(tile_size_fwd_sm90(kHeadDim, kHeadDimV, Is_causal, Is_local, sizeof(T) /*element_size*/, V_colmajor, PagedKVNonTMA, Has_softcap)) : 128;

                static constexpr bool Enable_cluster = Arch == 90 && (sizeof(T) == 2 ? (kHeadDim >= 128) : (kHeadDim == 192)) && !Is_causal && !Is_local && !Split && !PagedKVNonTMA && !Varlen;
                BOOL_SWITCH(params.qv_ptr, HasQV_, [&] {
                    static constexpr bool HasQv = HasQV_ && Arch == 90 && !Is_FP8 && kHeadDim == 64 && kHeadDimV >= 256;
                    APPENDKV_SWITCH(params.knew_ptr, AppendKV, [&] {
                        // Only use Cluster if number of tiles along seqlen_q is even and not varlen
                        CLUSTER_SWITCH(cutlass::ceil_div(params.seqlen_q * (!PackGQA ? 1 : params.h / params.h_k), kBlockM) % 2 == 0, Use_cluster, [&] {
                            static constexpr int ClusterM = Enable_cluster && Use_cluster ? 2 : 1;
                            run_flash_fwd<Arch, kHeadDim, kHeadDimV, ClusterM, T, T_out, Is_causal, Is_local, Has_softcap, Varlen, PagedKVNonTMA, AppendKV && Varlen, HasQv, PackGQA, Split, V_colmajor>(params, stream);
                        });
                    });
                });
            });
        });
    });
}
