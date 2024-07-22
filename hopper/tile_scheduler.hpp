/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include "cutlass/fast_math.h"
#include "cutlass/arch/barrier.h"

#include "named_barrier.hpp"

namespace flash {

///////////////////////////////////////////////////////////////////////////////

struct SingleTileScheduler {

public:

    // Host side kernel arguments
    struct Arguments {
        int const num_blocks_m, num_head, num_batch;
        int* const tile_count_semaphore = nullptr;
    };

    // Device side kernel params
    struct Params {};

    static Params
    to_underlying_arguments(Arguments const& args) {
        return {};
    }

    static dim3
    get_grid_dim(Arguments const& args, int num_sm) {
        return {uint32_t(args.num_blocks_m), uint32_t(args.num_head), uint32_t(args.num_batch)};
    }

    struct WorkTileInfo {
        int M_idx = 0;
        int H_idx = 0;
        int B_idx = 0;
        bool is_valid_tile = false;

        CUTLASS_DEVICE
        bool
        is_valid(Params const& params) const {
            return is_valid_tile;
        }

        CUTLASS_DEVICE
        cute::tuple<int32_t, int32_t, int32_t>
        get_block_coord(Params const& params) const {
            return {M_idx, H_idx, B_idx};
        }

    };

    CUTLASS_DEVICE
    SingleTileScheduler(int* tile_count_smem_) { }

    CUTLASS_DEVICE
    WorkTileInfo
    get_initial_work() const {
        return {int(blockIdx.x), int(blockIdx.y), int(blockIdx.z), true};
    }

    CUTLASS_DEVICE
    void
    init_consumer() const {}

    CUTLASS_DEVICE
    void
    prefetch_next_work(Params const& params, WorkTileInfo& current_work) const {}

    CUTLASS_DEVICE
    void
    broadcast_next_work(WorkTileInfo& current_work) const {}

    template<bool IsProducer=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_next_work(Params const& params, WorkTileInfo const& current_work) const {
        return {-1, -1, -1, false};
    }

};

///////////////////////////////////////////////////////////////////////////////

class StaticPersistentTileScheduler {

public:

    // Host side kernel arguments
    struct Arguments {
        int const num_blocks_m, num_head, num_batch;
        int* const tile_count_semaphore = nullptr;
    };

    // Device side kernel params
    struct Params {
        int total_blocks;
        cutlass::FastDivmod m_block_divmod, head_divmod;
    };

    static Params
    to_underlying_arguments(Arguments const& args) {
        return {args.num_blocks_m * args.num_head * args.num_batch,
                cutlass::FastDivmod(args.num_blocks_m), cutlass::FastDivmod(args.num_head)};
    }

    static dim3
    get_grid_dim(Arguments const& args, int num_sm) {
        return {uint32_t(num_sm)};
    }

    struct WorkTileInfo {
        int tile_idx;

        CUTLASS_DEVICE
        bool
        is_valid(Params const& params) const {
            return tile_idx < params.total_blocks;
        }

        CUTLASS_DEVICE
        cute::tuple<int32_t, int32_t, int32_t>
        get_block_coord(Params const& params) const {
            int m_block, bidh, bidb;
            bidb = params.head_divmod.divmod(bidh, params.m_block_divmod.divmod(m_block, tile_idx));
            return {m_block, bidh, bidb};
        }

    };

    CUTLASS_DEVICE
    StaticPersistentTileScheduler(int* tile_count_smem_) {};

    CUTLASS_DEVICE
    WorkTileInfo
    get_initial_work() const {
        return {int(blockIdx.x)};
    }

    CUTLASS_DEVICE
    void
    init_consumer() const {}

    CUTLASS_DEVICE
    void
    prefetch_next_work(Params const& params, WorkTileInfo& current_work) const {}

    CUTLASS_DEVICE
    void
    broadcast_next_work(WorkTileInfo& current_work) const {}

    template<bool IsProducer=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_next_work(Params const& params, WorkTileInfo const& current_work) const {
        return {current_work.tile_idx + int(gridDim.x)};
    }

};

template<int NumMmaThreads=2 * cutlass::NumThreadsPerWarpGroup, int NumProducerThreads = cutlass::NumThreadsPerWarp>
class DynamicPersistentTileScheduler {

protected:
    int* const tile_count_smem;

public:

    // Host side kernel arguments
    struct Arguments {
        int const num_blocks_m, num_head, num_batch;
        int* const tile_count_semaphore;
    };

    // Device side kernel params
    struct Params {
        int const total_blocks;
        cutlass::FastDivmod const m_block_divmod, head_divmod;
        int* const tile_count_semaphore;
    };

    static Params
    to_underlying_arguments(Arguments const& args) {
        return {args.num_blocks_m * args.num_head * args.num_batch,
                cutlass::FastDivmod(args.num_blocks_m), cutlass::FastDivmod(args.num_head),
                args.tile_count_semaphore};
    }

    static dim3
    get_grid_dim(Arguments const& args, int num_sm) {
        return {uint32_t(num_sm)};
    }

    struct WorkTileInfo {
        int tile_idx;

        CUTLASS_DEVICE
        bool
        is_valid(Params const& params) const {
            return tile_idx < params.total_blocks;
        }

        CUTLASS_DEVICE
        cute::tuple<int32_t, int32_t, int32_t>
        get_block_coord(Params const& params) const {
            int m_block, bidh, bidb;
            bidb = params.head_divmod.divmod(bidh, params.m_block_divmod.divmod(m_block, tile_idx));
            return {m_block, bidh, bidb};
        }

    };

    CUTLASS_DEVICE
    DynamicPersistentTileScheduler(int* tile_count_smem_) : tile_count_smem(tile_count_smem_) {};

    CUTLASS_DEVICE
    WorkTileInfo
    get_initial_work() const {
        return {int(blockIdx.x)};
    }

    CUTLASS_DEVICE
    void
    init_consumer() const {
        cutlass::arch::NamedBarrier::arrive(NumMmaThreads + NumProducerThreads, static_cast<int>(FwdNamedBarriers::TileCountSmemEmpty) /*id*/);
    }

    CUTLASS_DEVICE
    void
    prefetch_next_work(Params const& params, WorkTileInfo& current_work) const {
        if (threadIdx.x % NumProducerThreads == 0) {
            current_work.tile_idx = atomicAdd(params.tile_count_semaphore, 1) + int(gridDim.x);
        }
    }

    CUTLASS_DEVICE
    void
    broadcast_next_work(WorkTileInfo& current_work) const {
        cutlass::arch::NamedBarrier::sync(NumMmaThreads + NumProducerThreads, static_cast<int>(FwdNamedBarriers::TileCountSmemEmpty) /*id*/);
        if (threadIdx.x % NumProducerThreads == 0) {
            *tile_count_smem = current_work.tile_idx;
        }
        cutlass::arch::NamedBarrier::arrive(NumMmaThreads + NumProducerThreads, static_cast<int>(FwdNamedBarriers::TileCountSmemFull) /*id*/);
    }

    template<bool IsProducer=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_next_work(Params const& params, WorkTileInfo const& current_work) const {
        if constexpr (IsProducer && NumProducerThreads == cutlass::NumThreadsPerWarp) {
            // thread 0 already has the right tile_idx, just need to broadcast to the rest of the producer threads (warp 0)
            return {__shfl_sync(0xffffffff, current_work.tile_idx, 0 /*lane*/)};
        } else if constexpr (IsProducer && NumProducerThreads == cutlass::NumThreadsPerWarpGroup) {
            // TODO: investigate optimal synchronize
            int tile_idx = *tile_count_smem;
            return {tile_idx};
        } else {
            cutlass::arch::NamedBarrier::sync(NumMmaThreads + NumProducerThreads, static_cast<int>(FwdNamedBarriers::TileCountSmemFull) /*id*/);
            int tile_idx = *tile_count_smem;
            cutlass::arch::NamedBarrier::arrive(NumMmaThreads + NumProducerThreads, static_cast<int>(FwdNamedBarriers::TileCountSmemEmpty) /*id*/);
            return {tile_idx};
        }
    }

};

} // flash