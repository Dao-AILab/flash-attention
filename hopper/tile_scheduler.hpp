/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include "cutlass/fast_math.h"
#include "cutlass/arch/barrier.h"

#include "named_barrier.hpp"

namespace flash {

///////////////////////////////////////////////////////////////////////////////

template<bool Varlen=false, int kBlock=128>
class SingleTileScheduler {

public:

    using SharedStorage = int;

    // Host side kernel arguments
    struct Arguments {
        int const num_blocks, num_head, num_batch;
        int* const tile_count_semaphore = nullptr;
        int* const cu_seqlens = nullptr;
        int* const seqused = nullptr;
    };

    // Device side kernel params
    struct Params {
        int const num_blocks, num_head, num_batch;
        int* const cu_seqlens;
        int* const seqused;
    };

    static Params
    to_underlying_arguments(Arguments const& args) {
        return {args.num_blocks, args.num_head, args.num_batch,
                !Varlen ? nullptr : args.cu_seqlens, !Varlen ? nullptr : args.seqused};
    }

    static dim3
    get_grid_shape(Params const& params, int num_sm) {
        return {uint32_t(params.num_blocks), uint32_t(params.num_head), uint32_t(params.num_batch)};
    }

    struct WorkTileInfo {
        int block_idx = 0;
        int bidh = 0;
        int bidb = 0;
        bool is_valid_tile = false;

        CUTLASS_DEVICE
        bool
        is_valid(Params const& params) const {
            return is_valid_tile;
        }

        CUTLASS_DEVICE
        cute::tuple<int32_t, int32_t, int32_t>
        get_block_coord(Params const& params) const {
            return {block_idx, bidh, bidb};
        }

    };

    CUTLASS_DEVICE
    SingleTileScheduler(SharedStorage* const smem_scheduler) { }

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_initial_work(Params const& params) const {
        WorkTileInfo work_info {int(blockIdx.x), int(blockIdx.y), int(blockIdx.z), true};
        if constexpr (Varlen) {
            work_info.is_valid_tile = work_info.block_idx * kBlock < (params.seqused ? params.seqused[work_info.bidb] : params.cu_seqlens[work_info.bidb + 1] - params.cu_seqlens[work_info.bidb]);
        }
        return work_info;
    }

    CUTLASS_DEVICE
    void
    init_consumer() const {}

    CUTLASS_DEVICE
    void
    prefetch_next_work(Params const& params, WorkTileInfo& current_work) const {}

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_next_work(Params const& params, WorkTileInfo const& current_work) const {
        return {-1, -1, -1, false};
    }

};

///////////////////////////////////////////////////////////////////////////////

class StaticPersistentTileScheduler {

public:

    using SharedStorage = int;

    // Host side kernel arguments
    struct Arguments {
        int const num_blocks, num_head, num_batch;
        int* const tile_count_semaphore = nullptr;
        int* const cu_seqlens = nullptr;
        int* const seqused = nullptr;
    };

    // Device side kernel params
    struct Params {
        int total_blocks;
        cutlass::FastDivmod m_block_divmod, head_divmod;
    };

    static Params
    to_underlying_arguments(Arguments const& args) {
        return {args.num_blocks * args.num_head * args.num_batch,
                cutlass::FastDivmod(args.num_blocks), cutlass::FastDivmod(args.num_head)};
    }

    static dim3
    get_grid_shape(Params const& params, int num_sm) {
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
            int block, bidh, bidb;
            bidb = params.head_divmod.divmod(bidh, params.m_block_divmod.divmod(block, tile_idx));
            return {block, bidh, bidb};
        }

    };

    CUTLASS_DEVICE
    StaticPersistentTileScheduler(SharedStorage* const smem_scheduler) {};

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_initial_work(Params const& params) const {
        return {int(blockIdx.x)};
    }

    CUTLASS_DEVICE
    void
    init_consumer() const {}

    CUTLASS_DEVICE
    void
    prefetch_next_work(Params const& params, WorkTileInfo& current_work) const {}

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_next_work(Params const& params, WorkTileInfo const& current_work) const {
        return {current_work.tile_idx + int(gridDim.x)};
    }

};

template<int NumMmaThreads=2 * cutlass::NumThreadsPerWarpGroup, int NumProducerThreads=cutlass::NumThreadsPerWarp>
class DynamicPersistentTileScheduler {

public:
    using SharedStorage = int;

protected:
    SharedStorage* const tile_count_smem;

public:

    // Host side kernel arguments
    struct Arguments {
        int const num_blocks, num_head, num_batch;
        int* const tile_count_semaphore;
        int* const cu_seqlens = nullptr;
        int* const seqused = nullptr;
    };

    // Device side kernel params
    struct Params {
        int const total_blocks;
        cutlass::FastDivmod const m_block_divmod, head_divmod;
        int* const tile_count_semaphore;
    };

    static Params
    to_underlying_arguments(Arguments const& args) {
        return {args.num_blocks * args.num_head * args.num_batch,
                cutlass::FastDivmod(args.num_blocks), cutlass::FastDivmod(args.num_head),
                args.tile_count_semaphore};
    }

    static dim3
    get_grid_shape(Params const& params, int num_sm) {
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
            int block, bidh, bidb;
            bidb = params.head_divmod.divmod(bidh, params.m_block_divmod.divmod(block, tile_idx));
            return {block, bidh, bidb};
        }

    };

    CUTLASS_DEVICE
    DynamicPersistentTileScheduler(SharedStorage* const smem_scheduler) : tile_count_smem(smem_scheduler) {};

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_initial_work(Params const& params) const {
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

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_next_work(Params const& params, WorkTileInfo const& current_work) const {
        if constexpr (IsProducerWarp) {
            // thread 0 already has the right tile_idx, just need to broadcast to the rest of warp 0
            int new_tile_idx = __shfl_sync(0xffffffff, current_work.tile_idx, 0 /*lane*/);
            cutlass::arch::NamedBarrier::sync(NumMmaThreads + NumProducerThreads, static_cast<int>(FwdNamedBarriers::TileCountSmemEmpty) /*id*/);
            if (threadIdx.x % NumProducerThreads == 0) {
                *tile_count_smem = current_work.tile_idx;
            }
            cutlass::arch::NamedBarrier::arrive(NumMmaThreads + NumProducerThreads, static_cast<int>(FwdNamedBarriers::TileCountSmemFull) /*id*/);
            return {new_tile_idx};
        } else {
            cutlass::arch::NamedBarrier::sync(NumMmaThreads + NumProducerThreads, static_cast<int>(FwdNamedBarriers::TileCountSmemFull) /*id*/);
            int tile_idx = *tile_count_smem;
            cutlass::arch::NamedBarrier::arrive(NumMmaThreads + NumProducerThreads, static_cast<int>(FwdNamedBarriers::TileCountSmemEmpty) /*id*/);
            return {tile_idx};
        }
    }

};

template<int kBlock, int NumMmaThreads=2 * cutlass::NumThreadsPerWarpGroup, int NumProducerThreads=cutlass::NumThreadsPerWarp>
class VarlenDynamicPersistentTileScheduler {

public:
    using SharedStorage = int4;

protected:
    SharedStorage* const work_info_smem;

public:

    // Host side kernel arguments
    struct Arguments {
        int const num_blocks, num_head, num_batch;
        int* const tile_count_semaphore;
        int* const cu_seqlens;
        int* const seqused;
    };

    // Device side kernel params
    struct Params {
        int num_head, num_batch;
        int* const tile_count_semaphore;
        int* const cu_seqlens;
        int* const seqused;
    };

    static Params
    to_underlying_arguments(Arguments const& args) {
        return {args.num_head, args.num_batch,
                args.tile_count_semaphore, args.cu_seqlens, args.seqused};
    }

    static dim3
    get_grid_shape(Params const& params, int num_sm) {
        return {uint32_t(num_sm)};
    }

    struct WorkTileInfo {
        int tile_idx, block, bidh, bidb;

        CUTLASS_DEVICE
        bool
        is_valid(Params const& params) const {
            // if (blockIdx.x >= 0 && (threadIdx.x == 128 || threadIdx.x == 0)) { printf("blockIdx.x = %d, threadIdx.x = %d, checking valid, bidb = %d, params.num_batch = %d\n", blockIdx.x, threadIdx.x, bidb, params.num_batch); }
            return bidb < params.num_batch;
        }

        CUTLASS_DEVICE
        cute::tuple<int32_t, int32_t, int32_t>
        get_block_coord(Params const& params) const {
            return {block, bidh, bidb};
        }
    };

    CUTLASS_DEVICE
    VarlenDynamicPersistentTileScheduler(SharedStorage* const smem_scheduler) : work_info_smem(smem_scheduler) {};

    CUTLASS_DEVICE
    WorkTileInfo
    tile_idx_to_work_tile(Params const& params, int next_tile_idx, WorkTileInfo const& current_work) const {
        auto prefix_sum = [](int val) {
            int lane = threadIdx.x % cutlass::NumThreadsPerWarp;
            CUTLASS_PRAGMA_UNROLL
            for (int i = 1; i < cutlass::NumThreadsPerWarp; i <<= 1) {
                int32_t partial_sum = __shfl_up_sync(0xffffffff, val, i);
                if (lane >= i) { val += partial_sum; }
            }
            return val;
        };

        auto get_num_m_blocks = [&](int bidb) {
            int lane = threadIdx.x % cutlass::NumThreadsPerWarp;
            int seqlen;
            if (params.seqused) {
                seqlen = lane + bidb < params.num_batch ? params.seqused[lane + bidb] : 0;
            } else {
                int cur_cu_seqlen = lane + bidb <= params.num_batch ? params.cu_seqlens[lane + bidb] : 0;
                int next_cu_seqlen = __shfl_down_sync(0xffffffff, cur_cu_seqlen, 1);
                seqlen = next_cu_seqlen - cur_cu_seqlen;
            }
            return lane + bidb < params.num_batch && lane < cutlass::NumThreadsPerWarp - 1
                ? cute::ceil_div(seqlen, kBlock) : 0;
        };

        int num_m_blocks = get_num_m_blocks(current_work.bidb);  // Different for each lane
        // Cumulative number of blocks for the next 31 batches
        int num_m_blocks_cumulative = prefix_sum(num_m_blocks);
        // Total number of blocks for the next 31 batches
        int m_blocks_in_group = __shfl_sync(0xffffffff, num_m_blocks_cumulative, cutlass::NumThreadsPerWarp - 1);
        int group_end_tile = current_work.tile_idx - current_work.block - current_work.bidh * __shfl_sync(0xffffffff, num_m_blocks, 0 /*lane*/) + m_blocks_in_group * params.num_head;  // Same for all lanes
        int bidb = current_work.bidb;
        // if (blockIdx.x <= 9 && threadIdx.x == 128) {
        //     printf("Before while, blockIdx.x = %d, threadIdx.x = %d, bidb = %d, num_m_blocks = %d, next_tile_idx = %d, group_end_tile = %d, mh_blocks_in_group = %d\n", blockIdx.x, threadIdx.x, bidb, num_m_blocks, next_tile_idx, group_end_tile, mh_blocks_in_group);
        // }
        while (group_end_tile <= next_tile_idx) {
            bidb += cutlass::NumThreadsPerWarp - 1;
            if (bidb >= params.num_batch) {
                // if (blockIdx.x <= 9 && threadIdx.x == 128) {
                //     printf("Returning early, blockIdx.x = %d, threadIdx.x = %d, bidb = %d, num_m_blocks = %d, next_tile_idx = %d, group_end_tile = %d, mh_blocks_in_group = %d\n", blockIdx.x, threadIdx.x, bidb, num_m_blocks, next_tile_idx, group_end_tile, mh_blocks_in_group);
                // }
                return {next_tile_idx, 0, 0, params.num_batch};
            }
            num_m_blocks = get_num_m_blocks(bidb);
            num_m_blocks_cumulative = prefix_sum(num_m_blocks);
            m_blocks_in_group = __shfl_sync(0xffffffff, num_m_blocks_cumulative, cutlass::NumThreadsPerWarp - 1);
            group_end_tile += m_blocks_in_group * params.num_head;
            // if (blockIdx.x <= 9 && threadIdx.x == 128) {
            //     printf("Bottom of while, blockIdx.x = %d, threadIdx.x = %d, bidb = %d, num_m_blocks = %d, next_tile_idx = %d, group_end_tile = %d, mh_blocks_in_group = %d\n", blockIdx.x, threadIdx.x, bidb, num_m_blocks, next_tile_idx, group_end_tile, mh_blocks_in_group);
            // }
        }
        int group_start_tile = group_end_tile - m_blocks_in_group * params.num_head;
        // The next problem to process is the first one that does not have ending tile position
        // that is greater than or equal to tile index.
        int batch_idx_in_group = __popc(__ballot_sync(0xffffffff, group_start_tile + num_m_blocks_cumulative * params.num_head <= next_tile_idx));
        bidb += batch_idx_in_group;
        num_m_blocks = __shfl_sync(0xffffffff, num_m_blocks, batch_idx_in_group);
        int mh_block = next_tile_idx - group_start_tile - (batch_idx_in_group == 0 ? 0 : __shfl_sync(0xffffffff, num_m_blocks_cumulative, batch_idx_in_group - 1)) * params.num_head;
        int bidh = mh_block / num_m_blocks;
        int block = mh_block - bidh * num_m_blocks;
        // if (blockIdx.x <= 9 && threadIdx.x == 128) {
        //     printf("blockIdx.x = %d, threadIdx.x = %d, num_mh_blocks = %d, batch_idx_in_group = %d, bidb = %d, num_m_blocks = %d, next_tile_idx = %d, group_end_tile = %d, mh_blocks_in_group = %d, mh_block = %d, bidh = %d, block = %d\n", blockIdx.x, threadIdx.x, num_mh_blocks, batch_idx_in_group, bidb, num_m_blocks, next_tile_idx, group_end_tile, mh_blocks_in_group, mh_block, bidh, block);
        // }
        return {next_tile_idx, block, bidh, bidb};
    }

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_initial_work(Params const& params) const {
        if constexpr (IsProducerWarp) {
            WorkTileInfo work_info = tile_idx_to_work_tile(params, int(blockIdx.x), {0, 0, 0, 0});
            if (threadIdx.x % cutlass::NumThreadsPerWarp == 0) {
                *work_info_smem = make_int4(work_info.tile_idx, work_info.block, work_info.bidh, work_info.bidb);
            }
            cutlass::arch::NamedBarrier::arrive(NumMmaThreads + NumProducerThreads, static_cast<int>(FwdNamedBarriers::TileCountSmemFull) /*id*/);
            return work_info;
        } else {
            return get_next_work<false>(params, {0, 0, 0, 0});
        }
    }

    CUTLASS_DEVICE
    void
    init_consumer() const {
        // Don't arrive at the TileCountSmemEmpty barrier here, because get_initial_work will do that
    }

    CUTLASS_DEVICE
    void
    prefetch_next_work(Params const& params, WorkTileInfo& current_work) const {
        if (threadIdx.x % NumProducerThreads == 0) {
            current_work.tile_idx = atomicAdd(params.tile_count_semaphore, 1) + int(gridDim.x);
        }
    }

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_next_work(Params const& params, WorkTileInfo const& current_work) const {
        if constexpr (IsProducerWarp) {
            // thread 0 has the next tile_idx, just need to broadcast to the rest of warp 0
            int new_tile_idx = __shfl_sync(0xffffffff, current_work.tile_idx, 0 /*lane*/);
            WorkTileInfo work_info = {__shfl_sync(0xffffffff, current_work.tile_idx, 1 /*lane*/), current_work.block, current_work.bidh, current_work.bidb};
            work_info = tile_idx_to_work_tile(params, new_tile_idx, work_info);
            cutlass::arch::NamedBarrier::sync(NumMmaThreads + NumProducerThreads, static_cast<int>(FwdNamedBarriers::TileCountSmemEmpty) /*id*/);
            if (threadIdx.x % cutlass::NumThreadsPerWarp == 0) {
                *work_info_smem = make_int4(work_info.tile_idx, work_info.block, work_info.bidh, work_info.bidb);
            }
            cutlass::arch::NamedBarrier::arrive(NumMmaThreads + NumProducerThreads, static_cast<int>(FwdNamedBarriers::TileCountSmemFull) /*id*/);
            return work_info;
        } else {
            cutlass::arch::NamedBarrier::sync(NumMmaThreads + NumProducerThreads, static_cast<int>(FwdNamedBarriers::TileCountSmemFull) /*id*/);
            int4 work_info = *work_info_smem;
            cutlass::arch::NamedBarrier::arrive(NumMmaThreads + NumProducerThreads, static_cast<int>(FwdNamedBarriers::TileCountSmemEmpty) /*id*/);
            return WorkTileInfo{work_info.x, work_info.y, work_info.z, work_info.w};
        }
    }

};

} // flash
