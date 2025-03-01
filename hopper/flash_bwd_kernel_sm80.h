/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include "cute/tensor.hpp"

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/kernel_hardware_info.h>

#include "utils.h"

namespace flash {

using namespace cute;

template <class CollectiveMainloop_, class CollectiveEpilogue_, class TileScheduler_>
class FlashAttnBwdSm80 {

public:

    // Type Aliases
    static constexpr bool Is_causal = CollectiveMainloop_::Is_causal;
    static constexpr bool Is_local = CollectiveMainloop_::Is_local;
    static_assert(CollectiveMainloop_::Varlen == CollectiveEpilogue_::Varlen);
    static constexpr bool Varlen = CollectiveMainloop_::Varlen;

    // Mainloop derived types
    using CollectiveMainloop = CollectiveMainloop_;
    using TileShape_MNK = typename CollectiveMainloop::TileShape_MNK;
    using TiledMmaSdP = typename CollectiveMainloop::TiledMmaSdP;
    using TiledMmadKV = typename CollectiveMainloop::TiledMmadKV;
    using ArchTag = typename CollectiveMainloop::ArchTag;
    using MainloopArguments = typename CollectiveMainloop::Arguments;
    using MainloopParams = typename CollectiveMainloop::Params;
    static constexpr bool dKV_swapAB = CollectiveMainloop::dKV_swapAB;

    // Epilogue derived types
    using CollectiveEpilogue = CollectiveEpilogue_;
    using EpilogueArguments = typename CollectiveEpilogue::Arguments;
    using EpilogueParams = typename CollectiveEpilogue::Params;

    static_assert(ArchTag::kMinComputeCapability >= 80);

    using TileScheduler = TileScheduler_;
    using TileSchedulerArguments = typename flash::TileSchedulerArguments;
    using TileSchedulerParams = typename TileScheduler::Params;

    static constexpr uint32_t NumThreads = CUTE_STATIC_V(size(TiledMmaSdP{}));
    static constexpr uint32_t MaxThreadsPerBlock = CUTE_STATIC_V(size(TiledMmaSdP{}));
    static constexpr uint32_t MinBlocksPerMultiprocessor = 1;

    // Kernel level shared memory storage
    struct SharedStorage {
        struct TensorStorage : cute::aligned_struct<128> {
            union {
                typename CollectiveMainloop::TensorStorage mainloop;
                typename CollectiveEpilogue::TensorStorage epilogue;
            };
        } tensors;

        alignas(16) typename TileScheduler::SharedStorage smem_scheduler;

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

        static constexpr int kBlockM = get<0>(TileShape_MNK{});
        static constexpr int kBlockN = get<1>(TileShape_MNK{});

        SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

        CollectiveMainloop mainloop;
        CollectiveEpilogue epilogue;

        TileScheduler scheduler(reinterpret_cast<typename TileScheduler::SharedStorage*>(&shared_storage.smem_scheduler));
        // Initialize matmul objects.
        TiledMmadKV tiled_mma_dKV;

        scheduler.init_consumer();

        int warp_idx = cutlass::canonical_warp_idx_sync();
        CUTLASS_PRAGMA_NO_UNROLL
        for (auto work_tile_info = warp_idx == 0 ? scheduler.template get_initial_work</*IsProducerWarp=*/true>(params.scheduler) : scheduler.template get_initial_work</*IsProducerWarp=*/false>(params.scheduler);
             work_tile_info.is_valid(params.scheduler);
             work_tile_info = warp_idx == 0 ? scheduler.template get_next_work</*IsProducerWarp=*/true>(params.scheduler, work_tile_info) : scheduler.template get_next_work</*IsProducerWarp=*/false>(params.scheduler, work_tile_info)) {

            auto block_coord_ = work_tile_info.get_block_coord(params.scheduler);
            auto [n_block, bidh, bidb, _ /*split_idx*/] = block_coord_;
            cute::tuple<int32_t, int32_t, int32_t> block_coord = {n_block, bidh, bidb};

            // dK and dV output accumulator.
            Tensor tdKrdK = partition_fragment_C(tiled_mma_dKV, select<!dKV_swapAB ? 1 : 2, !dKV_swapAB? 2 : 1>(TileShape_MNK{}));
            Tensor tdVrdV = partition_fragment_C(tiled_mma_dKV, select<!dKV_swapAB ? 1 : 2, !dKV_swapAB? 2 : 1>(TileShape_MNK{}));
            bool tile_valid = mainloop.mma(params.mainloop, tdKrdK, tdVrdV, threadIdx.x,
                                           block_coord, shared_storage);
            scheduler.prefetch_next_work(params.scheduler, work_tile_info);
            if (tile_valid) {
                epilogue.store(params.epilogue, tdKrdK, tdVrdV, shared_storage, tiled_mma_dKV,
                               threadIdx.x, block_coord);
            } else {
                epilogue.store_zero(params.epilogue, threadIdx.x, block_coord);
            }
        }

    }

};

} // namespace flash
