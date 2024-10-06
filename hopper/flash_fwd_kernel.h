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
#include "softmax.h"

namespace flash {

using namespace cute;

template <class CollectiveMainloop_, class CollectiveEpilogue_, class TileScheduler_>
class FlashAttnFwd {

public:

    // Type Aliases
    static constexpr bool Is_causal = CollectiveMainloop_::Is_causal;
    static constexpr bool Is_local = CollectiveMainloop_::Is_local;
    static_assert(CollectiveMainloop_::Varlen == CollectiveEpilogue_::Varlen);
    static constexpr bool Varlen = CollectiveMainloop_::Varlen;
    static constexpr bool Is_FP8 = CollectiveMainloop_::Is_FP8;

    // Mainloop derived types
    using CollectiveMainloop = CollectiveMainloop_;
    using TileShape_MNK = typename CollectiveMainloop::TileShape_MNK;
    using TiledMma0 = typename CollectiveMainloop::TiledMma0;
    using TiledMma1 = typename CollectiveMainloop::TiledMma1;
    using ArchTag = typename CollectiveMainloop::ArchTag;
    using ClusterShape = typename CollectiveMainloop::ClusterShape;
    using MainloopArguments = typename CollectiveMainloop::Arguments;
    using MainloopParams = typename CollectiveMainloop::Params;

    // Epilogue derived types
    using CollectiveEpilogue = CollectiveEpilogue_;
    using EpilogueArguments = typename CollectiveEpilogue::Arguments;
    using EpilogueParams = typename CollectiveEpilogue::Params;

    static_assert(ArchTag::kMinComputeCapability >= 90);

    using TileScheduler = TileScheduler_;
    using TileSchedulerArguments = typename TileScheduler::Arguments;
    using TileSchedulerParams = typename TileScheduler::Params;

    static constexpr uint32_t NumLoadWarpGroups = 1;
    static constexpr uint32_t NumMmaWarpGroups = CUTE_STATIC_V(size(TiledMma0{})) / cutlass::NumThreadsPerWarpGroup;
    static constexpr uint32_t MaxThreadsPerBlock = CUTE_STATIC_V(size(TiledMma0{})) + (NumLoadWarpGroups * cutlass::NumThreadsPerWarpGroup);
    static constexpr uint32_t MinBlocksPerMultiprocessor = 1;
    static_assert(NumMmaWarpGroups == 2 || NumMmaWarpGroups == 3);

    /// Register requirement for Load and Math WGs
    static constexpr uint32_t LoadRegisterRequirement = NumMmaWarpGroups == 2 ? 24 : 32;
    static constexpr uint32_t MmaRegisterRequirement = NumMmaWarpGroups == 2 ? 240 : 160;
    // If you want to print from the producer warp, you'd need to increase the number of registers
    // Otherwise you'll get CUDA error.
    // static constexpr uint32_t LoadRegisterRequirement = 40;
    // static constexpr uint32_t MmaRegisterRequirement = NumMmaWarpGroups == 2 ? 232 : 152;

    // Kernel level shared memory storage
    struct SharedStorage {
        struct TensorStorage : cute::aligned_struct<128> {
            union {
                typename CollectiveMainloop::TensorStorage mainloop;
                // We want smem_o to line up with the start of smem_v
                typename CollectiveEpilogue::TensorStorage epilogue;
                static_assert(cute::cosize_v<typename CollectiveEpilogue::SmemLayoutO> * sizeof(typename CollectiveEpilogue::Element)
                              <= cute::cosize_v<typename CollectiveMainloop::SmemLayoutVt> * sizeof(typename CollectiveMainloop::Element));
            };
        } tensors;

        struct PipelineStorage : cute::aligned_struct<16> {
            alignas(16) cutlass::arch::ClusterTransactionBarrier barrier_Q;
            alignas(16) cutlass::arch::ClusterBarrier barrier_O;
            alignas(16) typename CollectiveMainloop::MainloopPipelineK::SharedStorage pipeline_k;
            alignas(16) typename CollectiveMainloop::MainloopPipelineV::SharedStorage pipeline_v;
            alignas(16) typename TileScheduler::SharedStorage smem_scheduler;
        } pipelines;

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

        using MainloopPipelineK = typename CollectiveMainloop::MainloopPipelineK;
        using MainloopPipelineV = typename CollectiveMainloop::MainloopPipelineV;
        using PipelineState = typename CollectiveMainloop::PipelineState;
        using PipelineParamsK = typename MainloopPipelineK::Params;
        using PipelineParamsV = typename MainloopPipelineV::Params;

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
        int warp_group_idx = cutlass::canonical_warp_group_idx();

        PipelineParamsK pipeline_params_k;
        pipeline_params_k.transaction_bytes = CollectiveMainloop::TmaTransactionBytesK;
        pipeline_params_k.role = warp_group_idx == 0
            ? MainloopPipelineK::ThreadCategory::Producer
            : MainloopPipelineK::ThreadCategory::Consumer;
        pipeline_params_k.is_leader = warp_group_thread_idx == 0;
        pipeline_params_k.num_consumers = NumMmaThreads;

        // PipelineParamsV pipeline_params_v;
        // pipeline_params_v.transaction_bytes = CollectiveMainloop::TmaTransactionBytesV;
        // pipeline_params_v.role = warp_group_idx == 0
        //     ? MainloopPipelineV::ThreadCategory::Producer
        //     : MainloopPipelineV::ThreadCategory::Consumer;
        // pipeline_params_v.is_leader = warp_group_thread_idx == 0;
        // pipeline_params_v.num_consumers = NumMmaThreads;

        if (warp_idx == 0 && lane_predicate) {
            shared_storage.pipelines.barrier_Q.init(1 /*numThreads*/);
            shared_storage.pipelines.barrier_O.init(size(ClusterShape{}) /*numThreads*/);
        }
        // We're counting on pipeline_k to call cutlass::arch::fence_barrier_init();
        MainloopPipelineK pipeline_k(shared_storage.pipelines.pipeline_k, pipeline_params_k, ClusterShape{});
        // MainloopPipelineV pipeline_v(shared_storage.pipelines.pipeline_v, pipeline_params_v, ClusterShape{});
        static_assert(is_same_v<PipelineParamsK, PipelineParamsV>);
        MainloopPipelineV pipeline_v(shared_storage.pipelines.pipeline_v, pipeline_params_k, ClusterShape{});

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
            if (warp_idx_in_warpgroup == 0) {  // Load Q, K, V
                PipelineState smem_pipe_write = cutlass::make_producer_start_state<MainloopPipelineK>();

                int work_idx = 0;

                TileScheduler scheduler(reinterpret_cast<typename TileScheduler::SharedStorage*>(&shared_storage.pipelines.smem_scheduler));
                for (auto work_tile_info = scheduler.template get_initial_work</*IsProducerWarp=*/true>(params.scheduler);
                     work_tile_info.is_valid(params.scheduler);
                     work_tile_info = scheduler.template get_next_work</*IsProducerWarp=*/true>(params.scheduler, work_tile_info)) {
                    auto block_coord = work_tile_info.get_block_coord(params.scheduler);
                    auto [m_block, bidh, bidb] = block_coord;

                    // With Varlen it's possible to have n_block_max == 0. Loading K can cause illegal memory access.
                    if constexpr (Is_causal || Is_local || Varlen) {
                        int n_block_max = collective_mainloop.get_n_block_max(params.mainloop, m_block, bidb);
                        int n_block_min = collective_mainloop.get_n_block_min(params.mainloop, m_block, bidb);
                        if (n_block_max <= n_block_min) {
                            scheduler.prefetch_next_work(params.scheduler, work_tile_info);
                            continue;
                        }
                    }
                    auto scheduler_prefetch = [&scheduler, &params, &work_tile_info]() {
                        scheduler.prefetch_next_work(params.scheduler, work_tile_info);
                    };
                    collective_mainloop.load(params.mainloop, pipeline_k, pipeline_v, smem_pipe_write,
                                            shared_storage, scheduler_prefetch, block_coord, work_idx);
                    ++work_idx;
                }
                collective_mainloop.load_tail(pipeline_k, pipeline_v, smem_pipe_write);
            }
        } else {  // Consumer
            cutlass::arch::warpgroup_reg_alloc<MmaRegisterRequirement>();

            TileScheduler scheduler(reinterpret_cast<typename TileScheduler::SharedStorage*>(&shared_storage.pipelines.smem_scheduler));
            // Initialize matmul objects.
            TiledMma1 tiled_mma1;

            PipelineState smem_pipe_read;
            // We don't need separate variables smem_pipe_release_k and smem_pipe_release_v
            // (like in Cutlass's gemm) because the read and release pipeline states are always the same.

            collective_mainloop.mma_init();
            scheduler.init_consumer();

            int work_idx = 0;
            CUTLASS_PRAGMA_NO_UNROLL
            for (auto work_tile_info = scheduler.template get_initial_work</*IsProducerWarp=*/false>(params.scheduler);
                 work_tile_info.is_valid(params.scheduler);
                 work_tile_info = scheduler.template get_next_work</*IsProducerWarp=*/false>(params.scheduler, work_tile_info)) {
                // Attention output (GEMM-II) accumulator.
                Tensor tOrO = partition_fragment_C(tiled_mma1, select<0, 2>(TileShape_MNK{}));
                float softmax_scale_log2 = params.mainloop.softmax_scale_log2;
                if constexpr (Is_FP8) {
                    float const q_scale = params.mainloop.ptr_q_scale == nullptr ? 1.0f : *params.mainloop.ptr_q_scale;
                    float const k_scale = params.mainloop.ptr_k_scale == nullptr ? 1.0f : *params.mainloop.ptr_k_scale;
                    softmax_scale_log2 *= q_scale * k_scale;
                }
                flash::Softmax<2 * (2 * kBlockM / NumMmaThreads), /*Max_offset=*/!Is_FP8 ? 0 : 8> softmax(softmax_scale_log2);

                auto block_coord = work_tile_info.get_block_coord(params.scheduler);
                auto [m_block, bidh, bidb] = block_coord;
                if constexpr (Is_causal || Is_local || Varlen) {
                    int n_block_max = collective_mainloop.get_n_block_max(params.mainloop, m_block, bidb);
                    int n_block_min = collective_mainloop.get_n_block_min(params.mainloop, m_block, bidb);
                    if (n_block_max <= n_block_min) {  // We exit early and write 0 to gO and -inf to gLSE.
                        collective_epilogue.store_zero(params.epilogue, threadIdx.x - NumCopyThreads, block_coord);
                        continue;
                    }
                }

                collective_mainloop.mma(params.mainloop, pipeline_k, pipeline_v, smem_pipe_read,
                                        tOrO, softmax, threadIdx.x - NumCopyThreads, work_idx, block_coord, shared_storage);
                                        // tOrO, softmax, n_block_max, threadIdx.x - NumCopyThreads + (work_idx >> 30), work_idx, shared_storage);
                collective_epilogue.store(params.epilogue, tOrO, softmax.row_sum, shared_storage, tiled_mma1,
                                          threadIdx.x - NumCopyThreads, block_coord);

                ++work_idx;
            }
            collective_epilogue.store_tail();
        }

    }

};

template <class CollectiveMainloop_, class CollectiveEpilogue_, class TileScheduler_,
          class Base=FlashAttnFwd<CollectiveMainloop_, CollectiveEpilogue_, TileScheduler_>>
class FlashAttnFwdFP8TransposeV : public Base {

public:

    using CollectiveMainloop = CollectiveMainloop_;
    using CollectiveEpilogue = CollectiveEpilogue_;
    using TileScheduler = TileScheduler_;

    // Type Aliases
    static constexpr bool Is_causal = CollectiveMainloop::Is_causal;
    static constexpr bool Is_local = CollectiveMainloop_::Is_local;
    using TileShape_MNK = typename Base::TileShape_MNK;
    using ClusterShape = typename Base::ClusterShape;
    using TiledMma1 = typename Base::TiledMma1;
    using Params = typename Base::Params;
    static constexpr bool Varlen = CollectiveMainloop::Varlen;

    static constexpr uint32_t NumLoadWarpGroups = Base::NumLoadWarpGroups;
    static constexpr uint32_t NumMmaWarpGroups = Base::NumMmaWarpGroups;
    /// Register requirement for Load and Math WGs
    static constexpr uint32_t LoadRegisterRequirement = NumMmaWarpGroups == 2 ? 24 : 32;
    static constexpr uint32_t MmaRegisterRequirement = NumMmaWarpGroups == 2 ? 240 : 160;
    // If you want to print from the producer warp, you'd need to increase the number of registers
    // Otherwise you'll get CUDA error.
    // static constexpr uint32_t LoadRegisterRequirement = 56;
    // static constexpr uint32_t MmaRegisterRequirement = NumMmaWarpGroups == 2 ? 224 : 152;

    // Kernel level shared memory storage
    struct SharedStorage {
        struct TensorStorage : cute::aligned_struct<128> {
            union {
                typename CollectiveMainloop::TensorStorage mainloop;
                // We want smem_o to line up with the start of smem_v
                typename CollectiveEpilogue::TensorStorage epilogue;
                static_assert(cute::cosize_v<typename CollectiveEpilogue::SmemLayoutO> <= cute::cosize_v<typename CollectiveMainloop::SmemLayoutVt>);
            };
        } tensors;

        struct PipelineStorage : cute::aligned_struct<16> {
            alignas(16) cutlass::arch::ClusterTransactionBarrier barrier_Q;
            alignas(16) cutlass::arch::ClusterBarrier barrier_O;
            alignas(16) typename CollectiveMainloop::MainloopPipelineK::SharedStorage pipeline_k;
            alignas(16) typename CollectiveMainloop::MainloopPipelineV::SharedStorage pipeline_v;
            alignas(16) typename CollectiveMainloop::MainloopPipelineVt::SharedStorage pipeline_vt;
            alignas(16) typename TileScheduler::SharedStorage smem_scheduler;
        } pipelines;

    };

    static constexpr int SharedStorageSize = sizeof(SharedStorage);

    CUTLASS_DEVICE
    void
    operator()(Params const& params, char* smem_buf) {

        static constexpr int NumMmaThreads = NumMmaWarpGroups * cutlass::NumThreadsPerWarpGroup;
        static constexpr int NumCopyThreads = NumLoadWarpGroups * cutlass::NumThreadsPerWarpGroup;
        static constexpr int kBlockM = get<0>(TileShape_MNK{});

        using MainloopPipelineK = typename CollectiveMainloop::MainloopPipelineK;
        using MainloopPipelineV = typename CollectiveMainloop::MainloopPipelineV;
        using MainloopPipelineVt = typename CollectiveMainloop::MainloopPipelineVt;
        using PipelineStateK = typename MainloopPipelineK::PipelineState;
        using PipelineStateV = typename MainloopPipelineV::PipelineState;
        using PipelineParamsK = typename MainloopPipelineK::Params;
        using PipelineParamsV = typename MainloopPipelineV::Params;
        using PipelineParamsVt = typename MainloopPipelineVt::Params;

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
        int warp_group_idx = cutlass::canonical_warp_group_idx();
        int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);

        PipelineParamsK pipeline_params_k;
        pipeline_params_k.transaction_bytes = CollectiveMainloop::TmaTransactionBytesK;
        pipeline_params_k.role = warp_group_idx == 0
            ? MainloopPipelineK::ThreadCategory::Producer
            : MainloopPipelineK::ThreadCategory::Consumer;
        pipeline_params_k.is_leader = warp_group_thread_idx == 0;
        pipeline_params_k.num_consumers = NumMmaThreads;
        // Technically for pipeline_params_vt, warp0 of WG0 is the producer and all of WG0 are consumers.
        // However, the thread role isn't used in the pipeline implementation.

        PipelineParamsV pipeline_params_v;
        pipeline_params_v.role = warp_group_idx == 0
            ? MainloopPipelineV::ThreadCategory::Producer
            : MainloopPipelineV::ThreadCategory::Consumer;
        pipeline_params_v.producer_arv_count = NumCopyThreads;
        pipeline_params_v.consumer_arv_count = NumMmaThreads;

        if (warp_idx == 0 && lane_predicate) {
            shared_storage.pipelines.barrier_Q.init(1 /*numThreads*/);
            shared_storage.pipelines.barrier_O.init(size(ClusterShape{}) /*numThreads*/);
        }
        // We're counting on pipeline_k to call cutlass::arch::fence_barrier_init();
        MainloopPipelineK pipeline_k(shared_storage.pipelines.pipeline_k, pipeline_params_k, ClusterShape{});
        MainloopPipelineV pipeline_v(shared_storage.pipelines.pipeline_v, pipeline_params_v);
        static_assert(is_same_v<MainloopPipelineK, MainloopPipelineVt>);
        pipeline_params_k.num_consumers = NumCopyThreads; // TMA_V is only consumed by the producer WG
        MainloopPipelineVt pipeline_vt(shared_storage.pipelines.pipeline_vt, pipeline_params_k, ClusterShape{});

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

            PipelineStateK smem_pipe_write = cutlass::make_producer_start_state<MainloopPipelineK>();
            int work_idx = 0;

            TileScheduler scheduler(reinterpret_cast<typename TileScheduler::SharedStorage*>(&shared_storage.pipelines.smem_scheduler));
            int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
            if (warp_idx_in_warpgroup != 0) { scheduler.init_consumer(); }

            for (auto work_tile_info = warp_idx_in_warpgroup == 0 ? scheduler.template get_initial_work</*IsProducerWarp=*/true>(params.scheduler) : scheduler.template get_initial_work</*IsProducerWarp=*/false>(params.scheduler);
                 work_tile_info.is_valid(params.scheduler);
                 work_tile_info = warp_idx_in_warpgroup == 0 ? scheduler.template get_next_work</*IsProducerWarp=*/true>(params.scheduler, work_tile_info) : scheduler.template get_next_work</*IsProducerWarp=*/false>(params.scheduler, work_tile_info)) {
                auto block_coord = work_tile_info.get_block_coord(params.scheduler);
                auto [m_block, bidh, bidb] = block_coord;

                // With Varlen it's possible to have n_block_max == 0. Loading K can cause illegal memory access.
                if constexpr (Is_causal || Is_local || Varlen) {
                    int n_block_max = collective_mainloop.get_n_block_max(params.mainloop, m_block, bidb);
                    int n_block_min = collective_mainloop.get_n_block_min(params.mainloop, m_block, bidb);
                    if (n_block_max <= n_block_min) {
                        scheduler.prefetch_next_work(params.scheduler, work_tile_info);
                        continue;
                    }
                }
                auto scheduler_prefetch = [&scheduler, &params, &work_tile_info]() {
                    scheduler.prefetch_next_work(params.scheduler, work_tile_info);
                };
                collective_mainloop.load_fp8_transpose_V(params.mainloop, pipeline_k, pipeline_v, pipeline_vt,
                                                         smem_pipe_write, shared_storage, scheduler_prefetch, block_coord, work_idx);
                ++work_idx;
            }
            collective_mainloop.load_tail(pipeline_k, pipeline_v, pipeline_vt, smem_pipe_write);
        } else {  // Consumer
            cutlass::arch::warpgroup_reg_alloc<MmaRegisterRequirement>();

            TileScheduler scheduler(reinterpret_cast<typename TileScheduler::SharedStorage*>(&shared_storage.pipelines.smem_scheduler));
            // Initialize matmul objects.
            TiledMma1 tiled_mma1;

            PipelineStateK smem_pipe_read_k;
            // We don't need separate variables smem_pipe_release_k and smem_pipe_release_v
            // (like in Cutlass's gemm) because the read and release pipeline states are always the same.

            collective_mainloop.mma_init();
            scheduler.init_consumer();

            int work_idx = 0;
            CUTLASS_PRAGMA_NO_UNROLL
            for (auto work_tile_info = scheduler.template get_initial_work</*IsProducerWarp=*/false>(params.scheduler);
                 work_tile_info.is_valid(params.scheduler);
                 work_tile_info = scheduler.template get_next_work</*IsProducerWarp=*/false>(params.scheduler, work_tile_info)) {
                // Attention output (GEMM-II) accumulator.
                Tensor tOrO = partition_fragment_C(tiled_mma1, select<0, 2>(TileShape_MNK{}));
                float const q_scale = params.mainloop.ptr_q_scale == nullptr ? 1.0f : *params.mainloop.ptr_q_scale;
                float const k_scale = params.mainloop.ptr_k_scale == nullptr ? 1.0f : *params.mainloop.ptr_k_scale;
                flash::Softmax<2 * (2 * kBlockM / NumMmaThreads), /*Max_offset=*/8> softmax(params.mainloop.softmax_scale_log2 * q_scale * k_scale);

                auto block_coord = work_tile_info.get_block_coord(params.scheduler);
                auto [m_block, bidh, bidb] = block_coord;
                if constexpr (Is_causal || Is_local || Varlen) {
                    int n_block_max = collective_mainloop.get_n_block_max(params.mainloop, m_block, bidb);
                    int n_block_min = collective_mainloop.get_n_block_min(params.mainloop, m_block, bidb);
                    if (n_block_max <= n_block_min) {  // We exit early and write 0 to gO and -inf to gLSE.
                        collective_epilogue.store_zero(params.epilogue, threadIdx.x - NumCopyThreads, block_coord);
                        continue;
                    }
                }

                collective_mainloop.mma(params.mainloop, pipeline_k, pipeline_v, smem_pipe_read_k,
                                        tOrO, softmax, threadIdx.x - NumCopyThreads, work_idx, block_coord, shared_storage);
                collective_epilogue.store(params.epilogue, tOrO, softmax.row_sum, shared_storage, tiled_mma1,
                                          threadIdx.x - NumCopyThreads, block_coord);

                ++work_idx;
            }
            collective_epilogue.store_tail();
        }

    }

};

} // namespace flash
