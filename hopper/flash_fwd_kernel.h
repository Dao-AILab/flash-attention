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

template <typename Ktraits, bool Is_causal, typename TileScheduler>
__global__ void __launch_bounds__(Ktraits::kNWarps * cutlass::NumThreadsPerWarp, 1)
    compute_attn_ws(CUTE_GRID_CONSTANT Flash_fwd_params const params,
                    CUTE_GRID_CONSTANT typename CollectiveMainloopFwd<Ktraits, Is_causal>::Params const mainloop_params,
                    CUTE_GRID_CONSTANT typename CollectiveEpilogueFwd<Ktraits>::Params const epilogue_params,
                    CUTE_GRID_CONSTANT typename TileScheduler::Params const scheduler_params
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

    using CollectiveMainloop = CollectiveMainloopFwd<Ktraits, Is_causal>;
    using CollectiveEpilogue = CollectiveEpilogueFwd<Ktraits>;

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
        // StaticPersistentTileScheduler scheduler{params.m_block_divmod, params.head_divmod, params.total_blocks};
        // auto work_tile_info = scheduler.get_current_work();
        TileScheduler scheduler;

        int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
        if (warp_idx_in_warpgroup == 0) {  // Load Q, K, V
            PipelineState smem_pipe_write_k = cutlass::make_producer_start_state<MainloopPipeline>();
            PipelineState smem_pipe_write_v = cutlass::make_producer_start_state<MainloopPipeline>();

            int work_idx = 0;

            // auto get_tile_count = [&] () {
            //     cutlass::arch::NamedBarrier::sync(NumMmaThreads + 2 * cutlass::NumThreadsPerWarp, 10 /*id*/);
            //     return shared_storage.tile_count_semaphore;
            // };

            // while (work_tile_info.is_valid()) {
            // for (int tile_count = blockIdx.x; tile_count < params.total_blocks; tile_count = get_tile_count()) {
            // for (int tile_count_semaphore = blockIdx.x; tile_count_semaphore < params.total_blocks; tile_count_semaphore = __shfl_sync(0xffffffff, tile_count_semaphore, 0)) {
            for (auto work_tile_info = scheduler.get_initial_work(); work_tile_info.is_valid(scheduler_params); work_tile_info = scheduler.get_next_work(scheduler_params, work_tile_info)) {
                int tile_count_semaphore = 0;
                collective_mainloop.load(params, mainloop_params, scheduler_params, pipeline_k, pipeline_v, smem_pipe_write_k, smem_pipe_write_v,
                                         shared_storage, work_tile_info, work_idx, tile_count_semaphore);
                // ++work_idx;
                // work_tile_info = scheduler.fetch_next_work();
            }
            collective_mainloop.load_tail(pipeline_k, pipeline_v, smem_pipe_write_k, smem_pipe_write_v);
        }
    } else {  // Consumer
        cutlass::arch::warpgroup_reg_alloc<Ktraits::kNWarps == 12 ? 240 : 160>();
        // cutlass::arch::warpgroup_reg_alloc<Ktraits::kNWarps == 12 ? 224 : 160>();

        // Initialize matmul objects.
        typename Ktraits::TiledMma1 tiled_mma1;

        TileScheduler scheduler{};

        PipelineState smem_pipe_read_k, smem_pipe_read_v;
        // We don't need separate variables smem_pip_release_k and smem_pipe_release_v
        // (like in Cutlass's gemm) because the read and release pipeline states are always the same.

        auto get_tile_count = [&] () {
            // cutlass::arch::NamedBarrier::sync(NumMmaThreads + 2 * cutlass::NumThreadsPerWarp, 10 /*id*/);
            cutlass::arch::NamedBarrier::sync(NumMmaThreads + cutlass::NumThreadsPerWarp, 10 /*id*/);
            return shared_storage.tile_count_semaphore;
        };

        collective_mainloop.mma_init();

        int work_idx = 0;
        CUTLASS_PRAGMA_NO_UNROLL
        // for (int work_idx = 0; work_idx * gridDim.x + blockIdx.x < params.total_blocks; ++work_idx) {
        // for (int tile_count_semaphore = blockIdx.x, work_idx = 0; tile_count_semaphore < params.total_blocks; tile_count_semaphore = get_tile_count()) {
        for (auto work_tile_info = scheduler.get_initial_work(); work_tile_info.is_valid(scheduler_params); work_tile_info = scheduler.get_next_work(scheduler_params, work_tile_info)) {
            // Attention output (GEMM-II) accumulator.
            Tensor tOrO = partition_fragment_C(tiled_mma1, select<0, 2>(TileShape_MNK{}));
            flash::Softmax<2 * (2 * kBlockM / NumMmaThreads)> softmax;

            // int m_block;
            // int bidh, bidb;
            // // bidb = params.head_divmod.divmod(bidh, params.m_block_divmod.divmod(m_block, work_idx * gridDim.x + blockIdx.x));
            // bidb = params.head_divmod.divmod(bidh, params.m_block_divmod.divmod(m_block, tile_count_semaphore));
            // cute::tuple<int32_t, int32_t, int32_t> block_coord = {m_block, bidh, bidb};
            auto block_coord = work_tile_info.get_block_coord(scheduler_params);
            auto [m_block, bidh, bidb] = block_coord;

            int n_block_max = collective_mainloop.get_n_block_max(mainloop_params, m_block);
            if (Is_causal && n_block_max <= 0) {  // We exit early and write 0 to gO and -inf to gLSE.
                // Need sync to avoid the case where the producer issues 2 arrives before the consumer can issue 1 wait
                cutlass::arch::NamedBarrier::arrive(NumMmaThreads + cutlass::NumThreadsPerWarp, 7 /*id*/);
                collective_epilogue.store_zero(epilogue_params, threadIdx.x - NumCopyThreads, block_coord);
                continue;
            }

            collective_mainloop.mma(mainloop_params, pipeline_k, pipeline_v, smem_pipe_read_k, smem_pipe_read_v,
                                    tOrO, softmax, n_block_max, threadIdx.x - NumCopyThreads, work_idx, m_block, shared_storage);
                                    // tOrO, softmax, n_block_max, threadIdx.x - NumCopyThreads + (work_idx >> 30), work_idx, shared_storage);
                                    // tOrO, softmax, n_block_max, threadIdx.x - NumCopyThreads, 0, shared_storage);
            collective_epilogue.store(epilogue_params, tOrO, softmax.row_sum, shared_storage, tiled_mma1,
                                      threadIdx.x - NumCopyThreads, block_coord);

            ++work_idx;
            // work_tile_info = scheduler.fetch_next_work();
        }
        collective_epilogue.store_tail();
    }
}

} // namespace flash
