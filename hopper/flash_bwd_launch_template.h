/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include "cute/tensor.hpp"

#include "cutlass/device_kernel.h"  // For device_kernel
#include "cutlass/kernel_launch.h"  // For kernel_launch
#include "cutlass/cluster_launch.hpp"  // For ClusterLauncher

#include "static_switch.h"
#include "flash.h"
#include "flash_bwd_preprocess_kernel.h"
#include "flash_bwd_postprocess_kernel.h"
#include "tile_scheduler.hpp"
#include "mainloop_bwd_sm90_tma_gmma_ws.hpp"
#include "mainloop_bwd_sm80.hpp"
#include "epilogue_bwd.hpp"
#include "flash_bwd_kernel_sm90.h"
#include "flash_bwd_kernel_sm80.h"

using namespace cute;

template <int Arch, int kHeadDim, int kBlockM, int kBlockN, typename Element,
          bool Is_causal, bool Is_local, bool Has_softcap, bool Varlen, bool Deterministic, bool GQA,
          int Stages_dO=2, int Stages_dS_or_QSm80=2,
          bool SdP_swapAB=true, bool dKV_swapAB=false, bool dQ_swapAB=false,
          int NumMmaWarpGroups=2, int AtomLayoutMSdP=1, int AtomLayoutNdKV=2, int AtomLayoutMdQ=1,
          bool V_in_regs=false>
void run_flash_bwd(Flash_bwd_params &params, cudaStream_t stream) {
    static_assert(!(Is_causal && Is_local), "Is_causal and Is_local cannot be true at the same time.");
    using ElementAccum = float;
    using ArchTag = std::conditional_t<Arch >= 90, cutlass::arch::Sm90, cutlass::arch::Sm80>;

    int const total_q_padded_rounded = cute::round_up(params.total_q + params.b * kBlockM, kBlockM);
    int const total_k_padded_rounded = cute::round_up(params.total_k + params.b * kBlockN, kBlockN);
    bool const is_varlen_q = params.cu_seqlens_q;
    bool const is_varlen_k = params.cu_seqlens_k;
    int seqlen_q = !is_varlen_q ? params.seqlen_q : params.total_q;
    int seqlen_k = !is_varlen_k ? params.seqlen_k : params.total_k;
    int seqlen_q_rounded = !is_varlen_q ? params.seqlen_q_rounded : total_q_padded_rounded;
    int seqlen_k_rounded = !is_varlen_k ? params.seqlen_k_rounded : total_k_padded_rounded;
    int batch_q = !is_varlen_q ? params.b : 1;
    int batch_k = !is_varlen_k ? params.b : 1;

    using TileShape_MK = cute::Shape<Int<kBlockM>, Int<kHeadDim>>;
    using PreprocessKernel = flash::FlashAttnBwdPreprocess<TileShape_MK, Element, ElementAccum, ArchTag, /*Clear_dQaccum=*/true, Varlen>;
    typename PreprocessKernel::Arguments preprocess_args {
        static_cast<Element const*>(params.o_ptr),
        {seqlen_q, params.dv, params.h, batch_q},  // shape_O
        {params.o_row_stride, _1{}, params.o_head_stride, !is_varlen_q ? params.o_batch_stride : 0},  // stride_O
        static_cast<Element const*>(params.do_ptr),
        {params.do_row_stride, _1{}, params.do_head_stride, !is_varlen_q ? params.do_batch_stride : 0},  // stride_dO
        static_cast<float*>(params.dsoftmax_sum),
        {seqlen_q_rounded, params.h, batch_q},  // shape_dPsum
        {_1{}, seqlen_q_rounded, !is_varlen_q ? params.h * params.seqlen_q_rounded : 0},  // stride_dPsum
        static_cast<float*>(params.softmax_lse_ptr),
        {_1{}, seqlen_q, !is_varlen_q ? params.h * params.seqlen_q : 0},  // stride_LSE
        static_cast<float*>(params.softmax_lse_log2_ptr),
        {_1{}, seqlen_q_rounded, !is_varlen_q ? params.h * params.seqlen_q_rounded : 0},  // stride_LSE_log2
        static_cast<ElementAccum*>(params.dq_accum_ptr),
        {seqlen_q_rounded * params.d_rounded, params.h, batch_q},  // shape_dQaccum
        {_1{}, seqlen_q_rounded * params.d_rounded, !is_varlen_q ? params.d_rounded * seqlen_q_rounded * params.h : 0},  // stride_dQaccum
        params.b,
        params.dq_semaphore,
        params.cu_seqlens_q,
        params.seqused_q
    };
    typename PreprocessKernel::Params preprocess_params = PreprocessKernel::to_underlying_arguments(preprocess_args);
    int num_m_block = cute::ceil_div(params.seqlen_q, kBlockM);
    dim3 grid_m(num_m_block, params.h, params.b);
    cutlass::kernel_launch<PreprocessKernel>(grid_m, PreprocessKernel::MaxThreadsPerBlock, PreprocessKernel::SharedStorageSize, stream, preprocess_params, false /*launch_with_pdl*/);
    CHECK_CUDA_KERNEL_LAUNCH();

    using TileShape_MNK = cute::Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
    using ClusterShape = cute::Shape<_1, Int<1>, _1>;  // Currently doesn't not support cluster
    // Stages_dS_or_QSm80 is Stages_dS if Sm90 and Stages if Sm80
    static constexpr int Stages = Arch >= 90 ? 2 : Stages_dS_or_QSm80;
    static constexpr int Stages_dS = Arch >= 90 ? Stages_dS_or_QSm80 : 1;
    using CollectiveMainloop = std::conditional_t<
        Arch >= 90,
        flash::CollectiveMainloopBwdSm90<Stages, Stages_dO, Stages_dS, ClusterShape, TileShape_MNK, Element, ElementAccum, cutlass::arch::Sm90,
            Is_causal, Is_local, Has_softcap, Varlen, Deterministic,
            SdP_swapAB, dKV_swapAB, dQ_swapAB, NumMmaWarpGroups, AtomLayoutMSdP, AtomLayoutNdKV, AtomLayoutMdQ, V_in_regs>,
        flash::CollectiveMainloopBwdSm80<Stages, Stages_dO, TileShape_MNK, Element, ElementAccum, cutlass::arch::Sm80,
            Is_causal, Is_local, Has_softcap, Varlen, Deterministic,
            SdP_swapAB, dKV_swapAB, dQ_swapAB, NumMmaWarpGroups, AtomLayoutMSdP, AtomLayoutNdKV, AtomLayoutMdQ, V_in_regs>
    >;
    using CollectiveEpilogue = std::conditional_t<
        !GQA,
        flash::CollectiveEpilogueBwd<TileShape_MNK, Element, ArchTag, CollectiveMainloop::NumMmaThreads, Varlen, dKV_swapAB, NumMmaWarpGroups * (Arch >= 90 ? 1 : cutlass::NumWarpsPerWarpGroup) / AtomLayoutNdKV>,
        flash::CollectiveEpilogueBwdGQA<TileShape_MNK, ElementAccum, ArchTag, CollectiveMainloop::NumMmaThreads, Varlen, Deterministic>
    >;
    using Scheduler = std::conditional_t<
        Is_causal && !Varlen,
        flash::SingleTileBwdLPTScheduler,
        flash::SingleTileScheduler<Varlen, false /*Split*/, false /*PackGQA*/, kBlockN>
    >;
    using AttnKernel = std::conditional_t<
        Arch >= 90,
        flash::enable_sm90_or_later<flash::FlashAttnBwdSm90<CollectiveMainloop, CollectiveEpilogue, Scheduler>>,
        flash::enable_sm80_to_sm89<flash::FlashAttnBwdSm80<CollectiveMainloop, CollectiveEpilogue, Scheduler>>
    >;

    typename CollectiveMainloop::Arguments mainloop_args {
        static_cast<Element const*>(params.q_ptr),
        {seqlen_q, params.d, params.h, batch_q},  // shape_Q
        {params.q_row_stride, _1{}, params.q_head_stride, !is_varlen_q ? params.q_batch_stride : 0},  // stride_Q
        static_cast<Element const*>(params.k_ptr),
        {seqlen_k, params.d, params.h_k, batch_k},  // shape_K
        {params.k_row_stride, _1{}, params.k_head_stride, !is_varlen_k ? params.k_batch_stride : 0},  // stride_K
        static_cast<Element const*>(params.v_ptr),
        {seqlen_k, params.dv, params.h_k, batch_k},  // shape_V
        {params.v_row_stride, _1{}, params.v_head_stride, !is_varlen_k ? params.v_batch_stride : 0},  // stride_V
        static_cast<Element const*>(params.do_ptr),
        {seqlen_q, params.dv, params.h, batch_q},  // shape_dO
        {params.do_row_stride, _1{}, params.do_head_stride, !is_varlen_q ? params.do_batch_stride : 0},  // stride_dO
        static_cast<ElementAccum*>(params.dq_accum_ptr),
        {seqlen_q_rounded * params.d_rounded, params.h, batch_q},  // shape_dQaccum
        {_1{}, seqlen_q_rounded * params.d_rounded, !is_varlen_q ? params.d_rounded * params.seqlen_q_rounded * params.h : 0}, // stride_dQaccum
        static_cast<float*>(params.softmax_lse_log2_ptr),
        {seqlen_q_rounded, params.h, batch_q},  // shape_LSE
        {_1{}, seqlen_q_rounded, !is_varlen_q ? params.h * params.seqlen_q_rounded : 0},  // stride_LSE_log2
        static_cast<float*>(params.dsoftmax_sum),
        {_1{}, seqlen_q_rounded, !is_varlen_q ? params.h * params.seqlen_q_rounded : 0},  // stride_dPsum
        params.scale_softmax,
        params.window_size_left, params.window_size_right, 0 /*attention_chunk*/,
        params.softcap,
        params.b,
        params.dq_semaphore,
        params.cu_seqlens_q, params.cu_seqlens_k,
        params.seqused_q, params.seqused_k
    };
    // The case work with GQA is ugly but idk how to fix it.
    typename CollectiveEpilogue::Arguments epilogue_args {
        static_cast<typename CollectiveEpilogue::Element*>(!GQA ? params.dk_ptr : params.dk_accum_ptr),
        [&] {
            if constexpr (!GQA) {
                return typename CollectiveEpilogue::ShapedKV {seqlen_k, params.d, params.h, batch_k};  // shape_dK
            } else {
                return typename CollectiveEpilogue::ShapedKV {seqlen_k_rounded * params.d_rounded, params.h_k, batch_k};  // shape_dKaccum
            }
        }(),
        [&] {
            if constexpr (!GQA) {
                return typename CollectiveEpilogue::StridedKV {params.dk_row_stride, _1{}, params.dk_head_stride, !is_varlen_k ? params.dk_batch_stride : 0};  // stride_dK
            } else {
                return typename CollectiveEpilogue::StridedKV {_1{}, params.d_rounded * seqlen_k_rounded, !is_varlen_k ? params.h_k * params.d_rounded * params.seqlen_k_rounded : 0};  // stride_dKaccum
            }
        }(),
        static_cast<typename CollectiveEpilogue::Element*>(!GQA ? params.dv_ptr : params.dv_accum_ptr),
        [&] {
            if constexpr (!GQA) {
                return typename CollectiveEpilogue::ShapedKV {seqlen_k, params.dv, params.h, batch_k};  // shape_dV
            } else {
                return typename CollectiveEpilogue::ShapedKV {seqlen_k_rounded * params.dv_rounded, params.h_k, batch_k};  // shape_dVaccum
            }
        }(),
        [&] {
            if constexpr (!GQA) {
                return typename CollectiveEpilogue::StridedKV {params.dv_row_stride, _1{}, params.dv_head_stride, !is_varlen_k ? params.dv_batch_stride : 0};  // stride_dV
            } else {
                return typename CollectiveEpilogue::StridedKV {_1{}, params.dv_rounded * seqlen_k_rounded, !is_varlen_k ? params.h_k * params.dv_rounded * params.seqlen_k_rounded : 0};  // stride_dVaccum
            }
        }(),
        params.h,
        params.dk_semaphore,
        params.dv_semaphore,
        params.cu_seqlens_k,
        params.seqused_k,
    };

    int num_blocks_n = cutlass::ceil_div(params.seqlen_k, get<1>(TileShape_MNK{}));
    num_blocks_n = cutlass::round_up(num_blocks_n, size<1>(ClusterShape{}));
    typename flash::TileSchedulerArguments scheduler_args {
        num_blocks_n, params.h, params.b, 1 /*num_splits*/,
        params.h / params.h_k,
        params.seqlen_k,
        params.seqlen_q, params.d, params.dv, sizeof(Element),
        params.tile_count_semaphore, params.cu_seqlens_k, params.seqused_k
    };

    int device;
    cudaGetDevice(&device);
    typename AttnKernel::Params kernel_params = AttnKernel::to_underlying_arguments({
        mainloop_args, epilogue_args, {device, params.num_sm}, scheduler_args
    });

    dim3 grid_dims = AttnKernel::get_grid_shape(kernel_params);
    dim3 block_dims = AttnKernel::get_block_shape();
    int smem_size = AttnKernel::SharedStorageSize;
    // int smem_size_q = sizeof(decltype((typename CollectiveMainloop::TensorStorage{}).smem_q));
    // int smem_size_do = sizeof(decltype((typename CollectiveMainloop::TensorStorage{}).smem_do));
    // int smem_size_ds = sizeof(decltype((typename CollectiveMainloop::TensorStorage{}).smem_ds));
    // int smem_size_dqacc = [&] {
    //     if constexpr (Arch >= 90) {
    //         return sizeof(decltype((typename CollectiveMainloop::TensorStorage{}).smem_dqacc));
    //     } else {
    //         return 0;
    //     }
    // }();
    // int smem_size_k = sizeof(decltype((typename CollectiveMainloop::TensorStorage{}).smem_k));
    // int smem_size_v = sizeof(decltype((typename CollectiveMainloop::TensorStorage{}).smem_v));
    // int smem_size_lse = sizeof(decltype((typename CollectiveMainloop::TensorStorage{}).smem_lse));
    // int smem_size_dpsum = sizeof(decltype((typename CollectiveMainloop::TensorStorage{}).smem_dpsum));
    // printf("smem_size = %d, q = %d, k = %d, v = %d, do = %d, ds = %d, dqacc = %d, lse = %d, dpsum = %d\n", smem_size, smem_size_q, smem_size_k, smem_size_v, smem_size_do, smem_size_ds, smem_size_dqacc, smem_size_lse, smem_size_dpsum);
    if constexpr (size(ClusterShape{}) > 1) {
        void const* kernel = (void const*) cutlass::device_kernel<AttnKernel>;
        if (smem_size >= 48 * 1024) {
            CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        }
        dim3 cluster_dims(size<0>(ClusterShape{}), size<1>(ClusterShape{}), size<2>(ClusterShape{}));
        cutlass::ClusterLauncher::launch(
            grid_dims, cluster_dims, block_dims, smem_size, stream, kernel, kernel_params, false /*launch_with_pdl*/);
    } else {
        if (smem_size >= 48 * 1024) {
            CHECK_CUDA(cudaFuncSetAttribute(cutlass::device_kernel<AttnKernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        }
        cutlass::kernel_launch<AttnKernel>(grid_dims, block_dims, smem_size, stream, kernel_params, false /*launch_with_pdl*/);
    }
    CHECK_CUDA_KERNEL_LAUNCH();

    using PostprocessKernel = flash::FlashAttnBwdPostprocessConvertdQ<TileShape_MK, Element, ElementAccum, ArchTag,
        AttnKernel::CollectiveMainloop::NumMmaThreads,
        typename AttnKernel::CollectiveMainloop::TiledMmadQ,
        AttnKernel::CollectiveMainloop::dQ_swapAB
        >;
    typename PostprocessKernel::Arguments postprocess_args {
        static_cast<ElementAccum const*>(params.dq_accum_ptr),
        {seqlen_q_rounded * params.d_rounded, params.h, batch_q},  // shape_dQaccum
        {_1{}, seqlen_q_rounded * params.d_rounded, !is_varlen_q ? params.d_rounded * params.seqlen_q_rounded * params.h : 0}, // stride_dQaccum
        static_cast<Element*>(params.dq_ptr),
        {seqlen_q, params.d, params.h, batch_q},  // shape_dQ
        {params.dq_row_stride, _1{}, params.dq_head_stride, params.dq_batch_stride},  // stride_dQ
        params.scale_softmax,
        params.cu_seqlens_q,
        params.seqused_q
    };
    typename PostprocessKernel::Params postprocess_params = PostprocessKernel::to_underlying_arguments(postprocess_args);
    int num_m_block_postprocess = cute::ceil_div(params.seqlen_q, get<0>(TileShape_MK{}));
    dim3 grid_m_postprocess(num_m_block_postprocess, params.h, params.b);
    int smem_size_postprocess = PostprocessKernel::SharedStorageSize;
    if (smem_size_postprocess >= 48 * 1024) {
        CHECK_CUDA(cudaFuncSetAttribute(cutlass::device_kernel<PostprocessKernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_postprocess));
    }
    cutlass::kernel_launch<PostprocessKernel>(grid_m_postprocess, PostprocessKernel::MaxThreadsPerBlock, smem_size_postprocess, stream, postprocess_params, false /*launch_with_pdl*/);
    CHECK_CUDA_KERNEL_LAUNCH();

    if constexpr (GQA) {
        using TileShape_NK = cute::Shape<Int<kBlockN>, Int<kHeadDim>>;
        using PostprocessKerneldKV = flash::FlashAttnBwdPostprocessConvertdQ<TileShape_NK, Element, ElementAccum, ArchTag,
            AttnKernel::CollectiveEpilogue::NumEpilogueThreads,
            typename AttnKernel::CollectiveMainloop::TiledMmadKV,
            AttnKernel::CollectiveMainloop::dKV_swapAB
            >;
        typename PostprocessKerneldKV::Arguments postprocess_dK_args {
            static_cast<ElementAccum const*>(params.dk_accum_ptr),
            {seqlen_k_rounded * params.d_rounded, params.h_k, batch_k},  // shape_dKaccum
            {_1{}, seqlen_k_rounded * params.d_rounded, !is_varlen_k ? params.d_rounded * params.seqlen_k_rounded * params.h_k : 0},  // stride_dKaccum
            static_cast<Element*>(params.dk_ptr),
            {seqlen_k, params.d, params.h_k, batch_k},  // shape_dK
            {params.dk_row_stride, _1{}, params.dk_head_stride, params.dk_batch_stride},  // stride_dK
            1.f,
            params.cu_seqlens_k,
            params.seqused_k
        };
        typename PostprocessKerneldKV::Params postprocess_dK_params = PostprocessKerneldKV::to_underlying_arguments(postprocess_dK_args);
        typename PostprocessKerneldKV::Arguments postprocess_dV_args {
            static_cast<ElementAccum const*>(params.dv_accum_ptr),
            {seqlen_k_rounded * params.dv_rounded, params.h_k, batch_k},  // shape_dVaccum
            {_1{}, seqlen_k_rounded * params.dv_rounded, !is_varlen_k ? params.dv_rounded * params.seqlen_k_rounded * params.h_k : 0},  // stride_dVaccum
            static_cast<Element*>(params.dv_ptr),
            {seqlen_k, params.dv, params.h_k, batch_k},  // shape_dV
            {params.dv_row_stride, _1{}, params.dv_head_stride, params.dv_batch_stride},  // stride_dV
            1.f,
            params.cu_seqlens_k,
            params.seqused_k
        };
        typename PostprocessKerneldKV::Params postprocess_dV_params = PostprocessKerneldKV::to_underlying_arguments(postprocess_dV_args);
        int num_n_block_postprocess = cute::ceil_div(params.seqlen_k, get<0>(TileShape_NK{}));
        dim3 grid_n_postprocess(num_n_block_postprocess, params.h_k, params.b);
        int smem_size_postprocess = PostprocessKerneldKV::SharedStorageSize;
        if (smem_size_postprocess >= 48 * 1024) {
            CHECK_CUDA(cudaFuncSetAttribute(cutlass::device_kernel<PostprocessKerneldKV>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_postprocess));
        }
        cutlass::kernel_launch<PostprocessKerneldKV>(grid_n_postprocess, PostprocessKerneldKV::MaxThreadsPerBlock, smem_size_postprocess, stream, postprocess_dK_params, false /*launch_with_pdl*/);
        CHECK_CUDA_KERNEL_LAUNCH();
        cutlass::kernel_launch<PostprocessKerneldKV>(grid_n_postprocess, PostprocessKerneldKV::MaxThreadsPerBlock, smem_size_postprocess, stream, postprocess_dV_params, false /*launch_with_pdl*/);
        CHECK_CUDA_KERNEL_LAUNCH();
    }

}

template<int Arch, typename T, int kBlockM, int kBlockN, int kHeadDim, bool Is_causal, bool Is_local, bool Has_softcap,
         int Stages_dO=2, int Stages_dS_or_QSm80=2,
         bool SdP_swapAB=true, bool dKV_swapAB=false, bool dQ_swapAB=false,
         int NumMmaWarpGroups=2, int AtomLayoutMSdP=1, int AtomLayoutNdKV=2, int AtomLayoutMdQ=1,
         bool V_in_regs=false>
void run_mha_bwd_dispatch(Flash_bwd_params &params, cudaStream_t stream) {
    VARLEN_SWITCH(params.cu_seqlens_q != nullptr || params.cu_seqlens_k != nullptr, Varlen, [&] {
        BOOL_SWITCH(params.h != params.h_k, GQA, [&] {
//             BOOL_SWITCH(params.deterministic, Deterministic, [&] {
            // run_flash_bwd<kHeadDim, kBlockM, kBlockN, T, Is_causal, Is_local, Has_softcap, Varlen, false, GQA, Stages_dO, Stages_dS_or_QSm80, SdP_swapAB, dKV_swapAB, dQ_swapAB, NumMmaWarpGroups, AtomLayoutMSdP, AtomLayoutNdKV, AtomLayoutMdQ>(params, stream);
            run_flash_bwd<Arch, kHeadDim, kBlockM, kBlockN, T, Is_causal, Is_local, Has_softcap, Varlen /*Varlen*/, false /*Deterministic*/, GQA, Stages_dO, Stages_dS_or_QSm80, SdP_swapAB, dKV_swapAB, dQ_swapAB, NumMmaWarpGroups, AtomLayoutMSdP, AtomLayoutNdKV, AtomLayoutMdQ, V_in_regs>(params, stream);
//             });
        });
    });
}


template<int Arch, typename T, bool Has_softcap>
void run_mha_bwd_hdim64(Flash_bwd_params &params, cudaStream_t stream) {
    CAUSAL_LOCAL_SWITCH(params.is_causal, params.is_local, Is_causal, Is_local, [&] {
        if constexpr (Arch >= 90) {
            if constexpr (Is_causal && Has_softcap) {
                // register spill with 128 x 128
                run_mha_bwd_dispatch<Arch, T, 96, 128, 64, Is_causal, Is_local, Has_softcap, 2, 2, true, false, true, 2, 1, 2, 2, false>(params, stream);
            } else {
                // With ShuffleStats we no longer have register spilling when Has_softcap and using 128 x 128 block.
                run_mha_bwd_dispatch<Arch, T, 128, 128, 64, Is_causal, Is_local, Has_softcap, 2, 2, true, false, false, 2, 1, 2, 2, false>(params, stream);
            }
        } else if constexpr (Arch == 86 || Arch == 89) {
            run_mha_bwd_dispatch<Arch, T, 64, 128, 64, Is_causal, Is_local, Has_softcap, 2, 2, false, false, false, 2, 2, 4, 2, true>(params, stream);
            // run_mha_bwd_dispatch<Arch, T, 96, 96, 64, Is_causal, Is_local, Has_softcap, 1, 2, false, true, true, 2, 2, 4, 4, false>(params, stream);
            // run_mha_bwd_dispatch<Arch, T, 80, 128, 64, Is_causal, Is_local, Has_softcap, 1, 2, true, false, true, 2, 2, 4, 2, true>(params, stream);
            // run_mha_bwd_dispatch<Arch, T, 96, 128, 64, Is_causal, Is_local, Has_softcap, 1, 2, true, false, true, 2, 1, 8, 4, false>(params, stream);
        } else {
            run_mha_bwd_dispatch<Arch, T, 128, 128, 64, Is_causal, Is_local, Has_softcap, 2, 2, false, false, false, 2, 4, 4, 4, false>(params, stream);
        }
    });
}

template<int Arch, typename T, bool Has_softcap>
void run_mha_bwd_hdim96(Flash_bwd_params &params, cudaStream_t stream) {
    CAUSAL_LOCAL_SWITCH(params.is_causal, params.is_local, Is_causal, Is_local, [&] {
        if constexpr (Arch >= 90) {
            run_mha_bwd_dispatch<Arch, T, 64, 128, 96, Is_causal, Is_local, Has_softcap, 2, 2, true, false, false, 2, 1, 2, 1, true>(params, stream);
        } else if constexpr (Arch == 86 || Arch == 89) {
            run_mha_bwd_dispatch<Arch, T, 64, 128, 96, Is_causal, Is_local, Has_softcap, 1, 2, false, false, false, 2, 2, 4, 2, true>(params, stream);
        } else {
            run_mha_bwd_dispatch<Arch, T, 64, 128, 96, Is_causal, Is_local, Has_softcap, 2, 2, false, false, false, 2, 2, 4, 2, false>(params, stream);
        }
    });
}

template<int Arch, typename T, bool Has_softcap>
void run_mha_bwd_hdim128(Flash_bwd_params &params, cudaStream_t stream) {
    CAUSAL_LOCAL_SWITCH(params.is_causal, params.is_local, Is_causal, Is_local, [&] {
        if constexpr (Arch >= 90) {
            if constexpr (Is_causal || Is_local || Has_softcap) {
                run_mha_bwd_dispatch<Arch, T, 64, 128, 128, Is_causal, Is_local, Has_softcap, 2, 2, true, false, false, 2, 1, 2, 1, false>(params, stream);
            } else {
                run_mha_bwd_dispatch<Arch, T, 80, 128, 128, Is_causal, Is_local, Has_softcap, 2, 2, true, false, true, 2, 1, 2, 1, false>(params, stream);
            }
        } else if constexpr (Arch == 86 || Arch == 89) {
            run_mha_bwd_dispatch<Arch, T, 64, 96, 128, Is_causal, Is_local, Has_softcap, 1, 2, false, false, false, 2, 2, 2, 2, true>(params, stream);
        } else {
            run_mha_bwd_dispatch<Arch, T, 64, 128, 128, Is_causal, Is_local, Has_softcap, 2, 2, false, false, false, 2, 2, 2, 2, false>(params, stream);
        }
    });
}

template<int Arch, typename T, bool Has_softcap>
void run_mha_bwd_hdim192(Flash_bwd_params &params, cudaStream_t stream) {
    CAUSAL_LOCAL_SWITCH(params.is_causal, params.is_local, Is_causal, Is_local, [&] {
        if constexpr (Arch >= 90) {
            run_mha_bwd_dispatch<Arch, T, 64, 96, 192, Is_causal, Is_local, Has_softcap, 1, 1, false, true, false, 3, 1, 1, 1, false>(params, stream);
        } else if constexpr (Arch == 86 || Arch == 89) {
            run_mha_bwd_dispatch<Arch, T, 64, 64, 192, Is_causal, Is_local, Has_softcap, 1, 1, false, false, false, 2, 2, 2, 2, true>(params, stream);
        } else {
            run_mha_bwd_dispatch<Arch, T, 64, 80, 192, Is_causal, Is_local, Has_softcap, 1, 2, false, true, false, 2, 4, 2, 2, false>(params, stream);
        }
    });
}

template<int Arch, typename T, bool Has_softcap>
void run_mha_bwd_hdim256(Flash_bwd_params &params, cudaStream_t stream) {
    CAUSAL_LOCAL_SWITCH(params.is_causal, params.is_local, Is_causal, Is_local, [&] {
        if constexpr (Arch >= 90) {
            run_mha_bwd_dispatch<Arch, T, 64, 80, 256, Is_causal, Is_local, Has_softcap, 1, 1, false, true, true, 2, 1, 1, 1, false>(params, stream);
        } else if constexpr (Arch == 86 || Arch == 89) {
            run_mha_bwd_dispatch<Arch, T, 32, 64, 256, Is_causal, Is_local, Has_softcap, 1, 1, false, false, false, 2, 2, 2, 1, true>(params, stream);
            // run_mha_bwd_dispatch<Arch, T, 64, 32, 256, Is_causal, Is_local, Has_softcap, 1, 1, false, false, false, 2, 4, 1, 2, true>(params, stream);
        } else {
            run_mha_bwd_dispatch<Arch, T, 64, 64, 256, Is_causal, Is_local, Has_softcap, 1, 1, false, false, false, 2, 4, 2, 2, false>(params, stream);
        }
    });
}
