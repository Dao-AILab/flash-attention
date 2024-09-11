
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
#include <cutlass/kernel_hardware_info.h>
#include "cutlass/pipeline/pipeline.hpp"

#include "utils.h"
#include "tile_scheduler_bwd.hpp"
#include "mainloop_bwd_sm90_tma_gmma_ws.hpp"
#include "epilogue_bwd_sm90_tma.hpp"

namespace flash {

using namespace cute;

template <class CollectiveMainloop_, class CollectiveEpilogue_, class TileScheduler_>
class FlashAttnBwd {

public:

    // Type Aliases
    static constexpr bool Is_causal = CollectiveMainloop_::Is_causal;
    static_assert(CollectiveMainloop_::Varlen == CollectiveEpilogue_::Varlen);
    static constexpr bool Varlen = CollectiveMainloop_::Varlen;

    // Mainloop derived types
    using CollectiveMainloop = CollectiveMainloop_;
    using TileShape_MNK = typename CollectiveMainloop::TileShape_MNK;
    using TiledMmaSdP = typename CollectiveMainloop::TiledMmaSdP;
    using TiledMmadKV = typename CollectiveMainloop::TiledMmadKV;
    using ArchTag = typename CollectiveMainloop::ArchTag;
    using ClusterShape = typename CollectiveMainloop::ClusterShape;
    using MainloopArguments = typename CollectiveMainloop::Arguments;
    using MainloopParams = typename CollectiveMainloop::Params;
    static constexpr bool dKV_swapAB = CollectiveMainloop::dKV_swapAB;

    // Epilogue derived types
    using CollectiveEpilogue = CollectiveEpilogue_;
    using EpilogueArguments = typename CollectiveEpilogue::Arguments;
    using EpilogueParams = typename CollectiveEpilogue::Params;

    static_assert(ArchTag::kMinComputeCapability >= 90);

    using TileScheduler = TileScheduler_;
    using TileSchedulerArguments = typename TileScheduler::Arguments;
    using TileSchedulerParams = typename TileScheduler::Params;

    static constexpr uint32_t NumLoadWarpGroups = 1;
    static constexpr uint32_t NumMmaWarpGroups = CUTE_STATIC_V(size(TiledMmaSdP{})) / cutlass::NumThreadsPerWarpGroup;
    static constexpr uint32_t MaxThreadsPerBlock = CUTE_STATIC_V(size(TiledMmaSdP{})) + (NumLoadWarpGroups * cutlass::NumThreadsPerWarpGroup);
    static constexpr uint32_t MinBlocksPerMultiprocessor = 1;
    static_assert(NumMmaWarpGroups == 2);

    /// Register requirement for Load and Math WGs
    static constexpr uint32_t LoadRegisterRequirement = 24;
    static constexpr uint32_t MmaRegisterRequirement = 240;
    // If you want to print from the producer warp, you'd need to increase the number of registers
    // Otherwise you'll get CUDA error.
    // static constexpr uint32_t LoadRegisterRequirement = 56;
    // static constexpr uint32_t MmaRegisterRequirement = 224;

    // Kernel level shared memory storage
    struct SharedStorage {
        struct {
            union {
                typename CollectiveMainloop::TensorStorage mainloop;
                typename CollectiveEpilogue::TensorStorage epilogue;
            };
        };

        struct {
            alignas(16) cutlass::arch::ClusterTransactionBarrier barrier_KV;
            alignas(16) cutlass::arch::ClusterBarrier barrier_dKV;
            alignas(16) typename CollectiveMainloop::MainloopPipeline::SharedStorage pipeline_q;
            alignas(16) typename CollectiveMainloop::MainloopPipeline::SharedStorage pipeline_do;
            alignas(16) typename TileScheduler::SharedStorage smem_scheduler;
        };

    };

    static constexpr int SharedStorageSize = sizeof(SharedStorage);

    // Device side arguments
    struct Arguments {
        MainloopArguments mainloop{};
        EpilogueArguments epilogue{};
        cutlass::KernelHardwareInfo hw_info{};
        TileSchedulerArguments scheduler{};
    };

    // Kernel entry point API
    struct Params {
        MainloopParams mainloop{};
        EpilogueParams epilogue{};
        cutlass::KernelHardwareInfo hw_info{};
        TileSchedulerParams scheduler{};
    };

    //
    // Methods
    //

    // Convert to underlying arguments. In this case, a simple copy for the aliased type.
    static
    Params
    to_underlying_arguments(Arguments const& args) {
        CUTLASS_TRACE_HOST("to_underlying_arguments():");

        // Get SM count if needed, otherwise use user supplied SM count
        int sm_count = args.hw_info.sm_count;
        if (sm_count <= 0) {
            CUTLASS_TRACE_HOST("  WARNING: Arguments do not include a valid SM count.\n"
                "  For optimal performance, populate the arguments KernelHardwareInfo struct with the SM count.");
            sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(args.hw_info.device_id);
        }

        CUTLASS_TRACE_HOST("to_underlying_arguments(): Setting persistent grid SM count to " << sm_count);

        cutlass::KernelHardwareInfo hw_info{args.hw_info.device_id, sm_count};
        return {
            CollectiveMainloop::to_underlying_arguments(args.mainloop),
            CollectiveEpilogue::to_underlying_arguments(args.epilogue),
            hw_info,
            TileScheduler::to_underlying_arguments(args.scheduler)
        };
    }

    // Computes the kernel launch grid shape based on runtime parameters
    static dim3
    get_grid_shape(Params const& params) {
        return TileScheduler::get_grid_shape(params.scheduler, params.hw_info.sm_count);
    }

    static dim3
    get_block_shape() {
        return dim3(MaxThreadsPerBlock, 1, 1);
    }


    CUTLASS_DEVICE
    void
    operator()(Params const& params, char* smem_buf) {

        static constexpr int NumMmaThreads = NumMmaWarpGroups * cutlass::NumThreadsPerWarpGroup;
        static constexpr int NumCopyThreads = NumLoadWarpGroups * cutlass::NumThreadsPerWarpGroup;
        static constexpr int kBlockM = get<0>(TileShape_MNK{});

        using MainloopPipeline = typename CollectiveMainloop::MainloopPipeline;
        using PipelineParams = typename MainloopPipeline::Params;
        using PipelineState = typename MainloopPipeline::PipelineState;

        SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

        int const lane_predicate = cute::elect_one_sync();
        int const warp_idx = cutlass::canonical_warp_idx_sync();

        // Issue Tma Descriptor Prefetch from a single thread
        if (warp_idx == 0 && lane_predicate) {
            CollectiveMainloop::prefetch_tma_descriptors(params.mainloop);
            CollectiveEpilogue::prefetch_tma_descriptors(params.epilogue);
        }

        // Obtain warp index
        int const warp_group_thread_idx = threadIdx.x % cutlass::NumThreadsPerWarpGroup;

        PipelineParams pipeline_params;
        pipeline_params.transaction_bytes = CollectiveMainloop::TmaTransactionBytesQ + CollectiveMainloop::TmaTransactionBytesLSE;
        int warp_group_idx = cutlass::canonical_warp_group_idx();
        pipeline_params.role = warp_group_idx == 0
            ? MainloopPipeline::ThreadCategory::Producer
            : MainloopPipeline::ThreadCategory::Consumer;
        pipeline_params.is_leader = warp_group_thread_idx == 0;
        pipeline_params.num_consumers = NumMmaThreads;

        if (warp_idx == 0 && lane_predicate) {
            shared_storage.barrier_KV.init(1 /*numThreads*/);
            // shared_storage.barrier_dKV.init(size(ClusterShape{}) /*numThreads*/);
        }
        // We're counting on pipeline_q to call cutlass::arch::fence_barrier_init();
        MainloopPipeline pipeline_q(shared_storage.pipeline_q, pipeline_params, ClusterShape{});
        MainloopPipeline pipeline_do(shared_storage.pipeline_do, pipeline_params, ClusterShape{});

        CollectiveMainloop collective_mainloop;
        CollectiveEpilogue collective_epilogue;

        // We need this to guarantee that the Pipeline init is visible to all producers and consumer blocks in the Cluster
        if constexpr (size(ClusterShape{}) > 1) {
            cute::cluster_arrive_relaxed();
            cute::cluster_wait();
        } else {
            __syncthreads();
        }

        if (warp_group_idx == 0) {  // Producer
            cutlass::arch::warpgroup_reg_dealloc<LoadRegisterRequirement>();

            int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
            if (warp_idx_in_warpgroup == 0) {  // Load K, V, and do TMA on Q and dO
                PipelineState smem_pipe_write = cutlass::make_producer_start_state<MainloopPipeline>();

                int work_idx = 0;

                TileScheduler scheduler(reinterpret_cast<typename TileScheduler::SharedStorage*>(&shared_storage.smem_scheduler));
                for (auto work_tile_info = scheduler.template get_initial_work</*IsProducer=*/true>(params.scheduler);
                     work_tile_info.is_valid(params.scheduler);
                     work_tile_info = scheduler.template get_next_work</*IsProducer=*/true>(params.scheduler, work_tile_info)) {
                    auto block_coord = work_tile_info.get_block_coord(params.scheduler);
                    auto [n_block, bidh, bidb] = block_coord;
                    if constexpr (Varlen) {
                        if (n_block * kBlockM >= collective_mainloop.get_seqlen_k(params.mainloop, bidb)) {
                            scheduler.prefetch_next_work(params.scheduler, work_tile_info);
                            continue;
                        }
                    }
                    if constexpr (Is_causal) {
                        int const m_block_min = collective_mainloop.get_m_block_min(params.mainloop, n_block, bidb);
                        int const m_block_max = cute::ceil_div(collective_mainloop.get_seqlen_q(params.mainloop, bidb), kBlockM);
                        if (m_block_min >= m_block_max) {
                            scheduler.prefetch_next_work(params.scheduler, work_tile_info);
                            continue;
                        }
                    }
                    auto scheduler_prefetch = [&scheduler, &params, &work_tile_info]() {
                        scheduler.prefetch_next_work(params.scheduler, work_tile_info);
                    };
                    collective_mainloop.load(params.mainloop, pipeline_q, pipeline_do, smem_pipe_write,
                                             shared_storage, scheduler_prefetch, block_coord, work_idx);
                    ++work_idx;
                }
                collective_mainloop.load_tail(pipeline_q, pipeline_do, smem_pipe_write);
            } else if (warp_idx_in_warpgroup == 1) {
                TileScheduler scheduler(reinterpret_cast<typename TileScheduler::SharedStorage*>(&shared_storage.smem_scheduler));
                for (auto work_tile_info = scheduler.template get_initial_work</*IsProducer=*/false>(params.scheduler);
                     work_tile_info.is_valid(params.scheduler);
                     work_tile_info = scheduler.template get_next_work</*IsProducer=*/false>(params.scheduler, work_tile_info)) {
                    auto block_coord = work_tile_info.get_block_coord(params.scheduler);
                    auto [n_block, bidh, bidb] = block_coord;
                    if constexpr (Varlen) {
                        if (n_block * kBlockM >= collective_mainloop.get_seqlen_k(params.mainloop, bidb)) { continue; }
                    }
                    if constexpr (Is_causal) {
                        int const m_block_min = collective_mainloop.get_m_block_min(params.mainloop, n_block, bidb);
                        int const m_block_max = cute::ceil_div(collective_mainloop.get_seqlen_q(params.mainloop, bidb), kBlockM);
                        if (m_block_min >= m_block_max) { continue; }
                    }
                    collective_mainloop.store_dq(params.mainloop, shared_storage, block_coord);
                }
            }
        } else {  // Consumer
            cutlass::arch::warpgroup_reg_alloc<MmaRegisterRequirement>();

            TileScheduler scheduler(reinterpret_cast<typename TileScheduler::SharedStorage*>(&shared_storage.smem_scheduler));
            // Initialize matmul objects.
            TiledMmadKV tiled_mma_dKV;

            PipelineState smem_pipe_read;

            collective_mainloop.mma_init();
            scheduler.init_consumer();

            int work_idx = 0;
            CUTLASS_PRAGMA_NO_UNROLL
            for (auto work_tile_info = scheduler.template get_initial_work</*IsProducer=*/false>(params.scheduler);
                 work_tile_info.is_valid(params.scheduler);
                 work_tile_info = scheduler.template get_next_work</*IsProducer=*/false>(params.scheduler, work_tile_info)) {
                auto block_coord = work_tile_info.get_block_coord(params.scheduler);
                auto [n_block, bidh, bidb] = block_coord;
                if constexpr (Varlen) {
                    if (n_block * kBlockM >= collective_mainloop.get_seqlen_k(params.mainloop, bidb)) { continue; }
                }
                if constexpr (Is_causal) {
                    int const m_block_min = collective_mainloop.get_m_block_min(params.mainloop, n_block, bidb);
                    int const m_block_max = cute::ceil_div(collective_mainloop.get_seqlen_q(params.mainloop, bidb), kBlockM);
                    if (m_block_min >= m_block_max) {  // We exit early and write 0 to dK and dV
                        collective_epilogue.store_zero(params.epilogue, threadIdx.x - NumCopyThreads, block_coord);
                        continue;
                    }
                }

                // dK and dV output accumulator.
                Tensor tdKrdK = partition_fragment_C(tiled_mma_dKV, select<!dKV_swapAB ? 1 : 2, !dKV_swapAB? 2 : 1>(TileShape_MNK{}));
                Tensor tdVrdV = partition_fragment_C(tiled_mma_dKV, select<!dKV_swapAB ? 1 : 2, !dKV_swapAB? 2 : 1>(TileShape_MNK{}));
                collective_mainloop.mma(params.mainloop, pipeline_q, pipeline_do, smem_pipe_read,
                                        tdKrdK, tdVrdV, threadIdx.x - NumCopyThreads, work_idx, block_coord, shared_storage);
                collective_epilogue.store(params.epilogue, tdKrdK, tdVrdV, shared_storage, tiled_mma_dKV,
                                          threadIdx.x - NumCopyThreads, block_coord);

                ++work_idx;
            }
            collective_epilogue.store_tail();
        }

    }

};

} // namespace flash
