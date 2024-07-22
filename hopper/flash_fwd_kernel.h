/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include "cute/tensor.hpp"

#include <cutlass/cutlass.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include "cutlass/pipeline/pipeline.hpp"

#include "flash.h"
#include "utils.h"
#include "softmax.h"
#include "tile_scheduler.hpp"
#include "mainloop_fwd_sm90_tma_gmma_ws.hpp"
#include "epilogue_fwd_sm90_tma.hpp"

namespace flash {

using namespace cute;

template <typename Ktraits, bool Is_causal, typename TileScheduler, typename Seqlen_traits>
__global__ void __launch_bounds__(Ktraits::kNWarps * cutlass::NumThreadsPerWarp, 1)
    compute_attn_ws(CUTE_GRID_CONSTANT typename CollectiveMainloopFwd<Ktraits, Is_causal, Seqlen_traits>::Params const mainloop_params,
                    CUTE_GRID_CONSTANT typename CollectiveEpilogueFwd<Ktraits, Seqlen_traits>::Params const epilogue_params,
                    CUTE_GRID_CONSTANT typename TileScheduler::Params const scheduler_params,
                    Seqlen_traits seqlen_traits_q, Seqlen_traits seqlen_traits_k
                    ) {

    using Element = typename Ktraits::Element;
    using ElementAccum = typename Ktraits::ElementAccum;
    using SoftType = ElementAccum;
    using TileShape_MNK = typename Ktraits::TileShape_MNK;
    using ClusterShape = typename Ktraits::ClusterShape_MNK;

    static_assert(Ktraits::Is_WS);
    static constexpr bool Is_WS = Ktraits::Is_WS;

    static constexpr int NumMmaThreads = size(typename Ktraits::TiledMma0{});
    static constexpr int NumCopyThreads = !Is_WS ? 0 : cutlass::NumThreadsPerWarpGroup;
    static constexpr int kBlockM = Ktraits::kBlockM;
    // static constexpr int kBlockN = Ktraits::kBlockN;
    // constexpr int kHeadDim = Ktraits::kHeadDim;

    using CollectiveMainloop = CollectiveMainloopFwd<Ktraits, Is_causal, Seqlen_traits>;
    using CollectiveEpilogue = CollectiveEpilogueFwd<Ktraits, Seqlen_traits>;

    using MainloopPipeline = typename Ktraits::MainloopPipeline;
    using PipelineParams = typename MainloopPipeline::Params;
    using PipelineState = typename MainloopPipeline::PipelineState;

    extern __shared__ char shared_memory[];
    auto &shared_storage = *reinterpret_cast<typename Ktraits::SharedStorage*>(shared_memory);

    int const lane_predicate = cute::elect_one_sync();
    int const warp_idx = cutlass::canonical_warp_idx_sync();

    // Issue Tma Descriptor Prefetch from a single thread
    if (warp_idx == 0 && lane_predicate) {
        CollectiveMainloop::prefetch_tma_descriptors(mainloop_params);
        CollectiveEpilogue::prefetch_tma_descriptors(epilogue_params);
    }

    // Obtain warp index
    int const warp_group_thread_idx = threadIdx.x % cutlass::NumThreadsPerWarpGroup;

    PipelineParams pipeline_params;
    pipeline_params.transaction_bytes = CollectiveMainloop::TmaTransactionBytesK;
    int warp_group_idx = cutlass::canonical_warp_group_idx();
    pipeline_params.role = warp_group_idx == 0
        ? MainloopPipeline::ThreadCategory::Producer
        : MainloopPipeline::ThreadCategory::Consumer;
    pipeline_params.is_leader = warp_group_thread_idx == 0;
    pipeline_params.num_consumers = NumMmaThreads;

    if (warp_idx == 0 && lane_predicate) {
        shared_storage.barrier_Q.init(1 /*numThreads*/);
        shared_storage.barrier_O.init(size(ClusterShape{}) /*numThreads*/);
    }
    // We're counting on pipeline_k to call cutlass::arch::fence_barrier_init();
    MainloopPipeline pipeline_k(shared_storage.pipeline_k, pipeline_params, ClusterShape{});
    MainloopPipeline pipeline_v(shared_storage.pipeline_v, pipeline_params, ClusterShape{});

    CollectiveMainloop collective_mainloop;
    CollectiveEpilogue collective_epilogue;

    // We need this to guarantee that the Pipeline init is visible to all producers and consumer blocks in the Cluster
    if constexpr (size(ClusterShape{}) > 1) {
        cute::cluster_arrive_relaxed();
        cute::cluster_wait();
    } else {
        __syncthreads();
    }

    static_assert(Ktraits::kNWarps == 12 || Ktraits::kNWarps == 16);
    if (warp_group_idx == 0) {  // Producer
        cutlass::arch::warpgroup_reg_dealloc<Ktraits::kNWarps == 12 ? 24 : 32>();
        // cutlass::arch::warpgroup_reg_dealloc<56>();

        int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
        if (warp_idx_in_warpgroup == 0) {  // Load Q, K, V
            PipelineState smem_pipe_write_k = cutlass::make_producer_start_state<MainloopPipeline>();
            PipelineState smem_pipe_write_v = cutlass::make_producer_start_state<MainloopPipeline>();

            int work_idx = 0;

            TileScheduler scheduler(&shared_storage.tile_count_semaphore);
            for (auto work_tile_info = scheduler.get_initial_work();
                 work_tile_info.is_valid(scheduler_params);
                 work_tile_info = scheduler.template get_next_work</*IsProducer=*/true>(scheduler_params, work_tile_info)) {
                auto block_coord = work_tile_info.get_block_coord(scheduler_params);
                auto [m_block, bidh, bidb] = block_coord;

                seqlen_traits_q.init(bidb);
                seqlen_traits_k.init(bidb);
                if (m_block * kBlockM >= seqlen_traits_q.actual_seq_len) {
                    continue;
                }
                int n_block_max = collective_mainloop.get_n_block_max(
                    mainloop_params, m_block, seqlen_traits_q, seqlen_traits_k);
                if (Is_causal && n_block_max <= 0) {
                    scheduler.prefetch_next_work(scheduler_params, work_tile_info);
                    scheduler.broadcast_next_work(work_tile_info);
                    continue;
                }
                collective_mainloop.load(mainloop_params, pipeline_k, pipeline_v, smem_pipe_write_k, smem_pipe_write_v,
                                         shared_storage, scheduler, scheduler_params, work_tile_info, block_coord, work_idx,
                                         seqlen_traits_q, seqlen_traits_k);
                ++work_idx;
            }
            collective_mainloop.load_tail(pipeline_k, pipeline_v, smem_pipe_write_k, smem_pipe_write_v);
        }
    } else {  // Consumer
        cutlass::arch::warpgroup_reg_alloc<Ktraits::kNWarps == 12 ? 240 : 160>();
        // cutlass::arch::warpgroup_reg_alloc<Ktraits::kNWarps == 12 ? 224 : 160>();

        TileScheduler scheduler(&shared_storage.tile_count_semaphore);
        // Initialize matmul objects.
        typename Ktraits::TiledMma1 tiled_mma1;

        PipelineState smem_pipe_read_k, smem_pipe_read_v;
        // We don't need separate variables smem_pipe_release_k and smem_pipe_release_v
        // (like in Cutlass's gemm) because the read and release pipeline states are always the same.

        collective_mainloop.mma_init();
        scheduler.init_consumer();

        int work_idx = 0;
        CUTLASS_PRAGMA_NO_UNROLL
        for (auto work_tile_info = scheduler.get_initial_work();
             work_tile_info.is_valid(scheduler_params);
             work_tile_info = scheduler.template get_next_work</*IsProducer=*/false>(scheduler_params, work_tile_info)) {
            // Attention output (GEMM-II) accumulator.
            Tensor tOrO = partition_fragment_C(tiled_mma1, select<0, 2>(TileShape_MNK{}));
            flash::Softmax<2 * (2 * kBlockM / NumMmaThreads)> softmax;

            auto block_coord = work_tile_info.get_block_coord(scheduler_params);
            auto [m_block, bidh, bidb] = block_coord;

            seqlen_traits_q.init(bidb);
            seqlen_traits_k.init(bidb);
            if (m_block * kBlockM >= seqlen_traits_q.actual_seq_len) {
                continue;
            }
            int n_block_max = collective_mainloop.get_n_block_max(
                mainloop_params, m_block, seqlen_traits_q, seqlen_traits_k);
            if (Is_causal && n_block_max <= 0) {  // We exit early and write 0 to gO and -inf to gLSE.
                collective_epilogue.store_zero(epilogue_params, shared_storage, threadIdx.x - NumCopyThreads, block_coord, seqlen_traits_q);
                continue;
            }

            collective_mainloop.mma(mainloop_params, pipeline_k, pipeline_v, smem_pipe_read_k, smem_pipe_read_v,
                                    tOrO, softmax, n_block_max, threadIdx.x - NumCopyThreads, work_idx, m_block, shared_storage,
                                    seqlen_traits_q, seqlen_traits_k);
                                    // tOrO, softmax, n_block_max, threadIdx.x - NumCopyThreads + (work_idx >> 30), work_idx, shared_storage);
            collective_epilogue.store(epilogue_params, tOrO, softmax.row_sum, shared_storage, tiled_mma1,
                                      threadIdx.x - NumCopyThreads, block_coord, seqlen_traits_q);

            ++work_idx;
        }
        collective_epilogue.store_tail();
    }

}

template <typename Ktraits, bool Is_causal, typename TileScheduler>
__global__ void __launch_bounds__(Ktraits::kNWarps * cutlass::NumThreadsPerWarp, 1)
    compute_attn_ws_fp8(CUTE_GRID_CONSTANT typename CollectiveMainloopFwd<Ktraits, Is_causal>::Params const mainloop_params,
                        CUTE_GRID_CONSTANT typename CollectiveEpilogueFwd<Ktraits>::Params const epilogue_params,
                        CUTE_GRID_CONSTANT typename TileScheduler::Params const scheduler_params
                        ) {

    using Element = typename Ktraits::Element;
    static_assert(cutlass::sizeof_bits_v<Element> == 8);
    using ElementAccum = typename Ktraits::ElementAccum;
    using SoftType = ElementAccum;
    using TileShape_MNK = typename Ktraits::TileShape_MNK;
    using ClusterShape = typename Ktraits::ClusterShape_MNK;

    static_assert(Ktraits::Is_WS);
    static constexpr bool Is_WS = Ktraits::Is_WS;

    static constexpr int NumMmaThreads = size(typename Ktraits::TiledMma0{});
    static constexpr int NumCopyThreads = !Is_WS ? 0 : cutlass::NumThreadsPerWarpGroup;
    static constexpr int kBlockM = Ktraits::kBlockM;
    // static constexpr int kBlockN = Ktraits::kBlockN;
    // constexpr int kHeadDim = Ktraits::kHeadDim;

    using CollectiveMainloop = CollectiveMainloopFwd<Ktraits, Is_causal>;
    using CollectiveEpilogue = CollectiveEpilogueFwd<Ktraits>;

    using MainloopPipeline = typename Ktraits::MainloopPipeline;
    using MainloopPipelineVt = typename Ktraits::MainloopPipelineNoTMA;
    using PipelineParams = typename MainloopPipeline::Params;
    using PipelineParamsVt = typename MainloopPipelineVt::Params;
    using PipelineState = typename MainloopPipeline::PipelineState;

    extern __shared__ char shared_memory[];
    auto &shared_storage = *reinterpret_cast<typename Ktraits::SharedStorage*>(shared_memory);

    int const lane_predicate = cute::elect_one_sync();
    int const warp_idx = cutlass::canonical_warp_idx_sync();

    // Issue Tma Descriptor Prefetch from a single thread
    if (warp_idx == 0 && lane_predicate) {
        CollectiveMainloop::prefetch_tma_descriptors(mainloop_params);
        CollectiveEpilogue::prefetch_tma_descriptors(epilogue_params);
    }

    // Obtain warp index
    int const warp_group_thread_idx = threadIdx.x % cutlass::NumThreadsPerWarpGroup;

    // additional pipeline to synchronize out-of-place smem transpose of V
    PipelineParamsVt pipeline_params_vt;
    pipeline_params_vt.producer_arv_count = NumCopyThreads;
    pipeline_params_vt.consumer_arv_count = NumMmaThreads;
    MainloopPipelineVt pipeline_vt(shared_storage.pipeline_vt, pipeline_params_vt);
    
    PipelineParams pipeline_params;
    pipeline_params.transaction_bytes = CollectiveMainloop::TmaTransactionBytesK;
    int warp_group_idx = cutlass::canonical_warp_group_idx();
    pipeline_params.role = warp_group_idx == 0
        ? MainloopPipeline::ThreadCategory::Producer
        : MainloopPipeline::ThreadCategory::Consumer;
    pipeline_params.is_leader = warp_group_thread_idx == 0;
    pipeline_params.num_consumers = NumMmaThreads;

    if (warp_idx == 0 && lane_predicate) {
        shared_storage.barrier_Q.init(1 /*numThreads*/);
#ifndef NO_UNION
    #ifndef NEW_FP8_EPI_BARRIER
        shared_storage.barrier_O.init(size(ClusterShape{}) /*numThreads*/);
    #endif
#endif
    }
    // We're counting on pipeline_k to call cutlass::arch::fence_barrier_init();
    MainloopPipeline pipeline_k(shared_storage.pipeline_k, pipeline_params, ClusterShape{});
    // pipeline_v has producer warpgroup for its consumer in fp8 kernel
    pipeline_params.num_consumers = NumCopyThreads;
    pipeline_params.role = MainloopPipeline::ThreadCategory::ProducerConsumer;
    MainloopPipeline pipeline_v(shared_storage.pipeline_v, pipeline_params, ClusterShape{});

    CollectiveMainloop collective_mainloop;
    CollectiveEpilogue collective_epilogue;

    // We need this to guarantee that the Pipeline init is visible to all producers and consumer blocks in the Cluster
    if constexpr (size(ClusterShape{}) > 1) {
        cute::cluster_arrive_relaxed();
        cute::cluster_wait();
    } else {
        __syncthreads();
    }

    static_assert(Ktraits::kNWarps == 12 || Ktraits::kNWarps == 16);
    if (warp_group_idx == 0) {  // Producer
        cutlass::arch::warpgroup_reg_dealloc<Ktraits::kNWarps == 12 ? 40 : 32>();
        
               
    #ifdef USE_TRI_MMA_FP8
        PipelineState smem_pipe_write_k  = cutlass::make_producer_start_state<MainloopPipeline>(); 
        PipelineState smem_pipe_write_v  = cutlass::make_producer_start_state<MainloopPipeline>(); 
        PipelineState smem_pipe_read_v;
    #else
        PipelineState smem_pipe_write = cutlass::make_producer_start_state<MainloopPipeline>(); 
        PipelineState smem_pipe_read;
    #endif

        int work_idx = 0;

        TileScheduler scheduler(&shared_storage.tile_count_semaphore);
        for (auto work_tile_info = scheduler.get_initial_work();
                work_tile_info.is_valid(scheduler_params);
                work_tile_info = scheduler.template get_next_work</*IsProducer=*/true>(scheduler_params, work_tile_info)) {
            auto block_coord = work_tile_info.get_block_coord(scheduler_params);
            auto [m_block, bidh, bidb] = block_coord;

            int n_block_max = collective_mainloop.get_n_block_max(mainloop_params, m_block);
            if (Is_causal && n_block_max <= 0) {
                scheduler.prefetch_next_work(scheduler_params, work_tile_info);
                scheduler.broadcast_next_work(work_tile_info);
                // TODO: remove this
                cutlass::arch::NamedBarrier::sync(NumCopyThreads, static_cast<int>(FwdNamedBarriers::ProducerWG) /*id*/);
                continue;
            }                        
        #ifdef USE_TRI_MMA_FP8
            collective_mainloop.load_fp8_ver1(
                mainloop_params, pipeline_k, pipeline_v, pipeline_vt,
                smem_pipe_write_k, smem_pipe_write_v, smem_pipe_read_v, shared_storage,
                scheduler, scheduler_params, work_tile_info, block_coord, work_idx);            
        #else
            collective_mainloop.load_fp8(
                mainloop_params, pipeline_k, pipeline_v, pipeline_vt,
                smem_pipe_write, smem_pipe_read, shared_storage,
                scheduler, scheduler_params, work_tile_info, block_coord, work_idx);                                  
        #endif
            ++work_idx;
            // need to sync producer warpgroup
            // TODO: remove this
            // if (Is_causal)
            //     cutlass::arch::NamedBarrier::sync(NumCopyThreads, static_cast<int>(FwdNamedBarriers::ProducerWG) /*id*/);
        }
    #ifdef USE_TRI_MMA_FP8
        collective_mainloop.load_tail(pipeline_k, pipeline_v, smem_pipe_write_k, smem_pipe_write_v);
    #else
        collective_mainloop.load_tail_one_write(pipeline_k, pipeline_v, smem_pipe_write);
    #endif
        
    } else {  // Consumer
        cutlass::arch::warpgroup_reg_alloc<Ktraits::kNWarps == 12 ? 232 : 160>();        

        TileScheduler scheduler(&shared_storage.tile_count_semaphore);
        // Initialize matmul objects.
        typename Ktraits::TiledMma1 tiled_mma1;
    #ifdef USE_TRI_MMA_FP8
        PipelineState smem_pipe_read_k, smem_pipe_read_vt;        
    #else
        PipelineState smem_pipe_read;
        // PipelineState smem_pipe_release;
    #endif                

        collective_mainloop.mma_init_fp8();
        scheduler.init_consumer();

        int work_idx = 0;
        CUTLASS_PRAGMA_NO_UNROLL
        for (auto work_tile_info = scheduler.get_initial_work();
             work_tile_info.is_valid(scheduler_params);
             work_tile_info = scheduler.template get_next_work</*IsProducer=*/false>(scheduler_params, work_tile_info)) {
            // Attention output (GEMM-II) accumulator.
            Tensor tOrO = partition_fragment_C(tiled_mma1, select<0, 2>(TileShape_MNK{}));
            flash::Softmax<2 * (2 * kBlockM / NumMmaThreads)> softmax;

            auto block_coord = work_tile_info.get_block_coord(scheduler_params);
            auto [m_block, bidh, bidb] = block_coord;

            int n_block_max = collective_mainloop.get_n_block_max(mainloop_params, m_block);
            if (Is_causal && n_block_max <= 0) {  // We exit early and write 0 to gO and -inf to gLSE.
                collective_epilogue.store_zero(epilogue_params, threadIdx.x - NumCopyThreads, block_coord);
                continue;
            }
                        
        #ifdef USE_TRI_MMA_FP8
            collective_mainloop.mma_fp8_ver1(
                mainloop_params, pipeline_k, pipeline_vt,
                smem_pipe_read_k, smem_pipe_read_vt,
                tOrO, softmax, n_block_max,
                threadIdx.x - NumCopyThreads, work_idx, m_block,
                shared_storage);              
        #else
            // collective_mainloop.mma_fp8(
            //     mainloop_params, pipeline_k, pipeline_vt, smem_pipe_read,
            //     smem_pipe_release, tOrO, softmax, n_block_max,
            //     threadIdx.x - NumCopyThreads, work_idx, m_block,
            //     shared_storage);
            collective_mainloop.mma_fp8_ver2(
                mainloop_params, pipeline_k, pipeline_vt, smem_pipe_read,
                tOrO, softmax, n_block_max,
                threadIdx.x - NumCopyThreads, work_idx, m_block,
                shared_storage);  
        #endif

        #ifdef COLUMN_PERMUTE
            collective_epilogue.store_fp8(epilogue_params, tOrO, softmax.row_sum, shared_storage, tiled_mma1,
                                      threadIdx.x - NumCopyThreads, block_coord);                
            // collective_epilogue.store(epilogue_params, tOrO, softmax.row_sum, shared_storage, tiled_mma1,
            //                           threadIdx.x - NumCopyThreads, block_coord);
        #else
            collective_epilogue.store(epilogue_params, tOrO, softmax.row_sum, shared_storage, tiled_mma1,
                                      threadIdx.x - NumCopyThreads, block_coord);                
        #endif
            ++work_idx;
        }
        collective_epilogue.store_tail();
    }

}

} // namespace flash
