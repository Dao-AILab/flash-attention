/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include "cutlass/fast_math.h"
#include "cutlass/arch/barrier.h"

#include "named_barrier.hpp"

namespace flash {

///////////////////////////////////////////////////////////////////////////////

class SingleTileSchedulerBwd {

public:

    using SharedStorage = int;

    // Host side kernel arguments
    struct Arguments {
        int const num_blocks_m, num_head, num_batch;
        int* const tile_count_semaphore = nullptr;
        int* const cu_seqlens = nullptr;
    };

    // Device side kernel params
    struct Params {
        int const num_blocks_m, num_head, num_batch;
    };

    static Params
    to_underlying_arguments(Arguments const& args) {
        return {args.num_blocks_m, args.num_head, args.num_batch};
    }

    static dim3
    get_grid_shape(Params const& params, int num_sm) {
        return {uint32_t(params.num_blocks_m), uint32_t(params.num_head), uint32_t(params.num_batch)};
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
    SingleTileSchedulerBwd(SharedStorage* const smem_scheduler) { }

    template<bool IsProducer=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_initial_work(Params const& params) const {
        return {int(blockIdx.x), int(blockIdx.y), int(blockIdx.z), true};
    }

    CUTLASS_DEVICE
    void
    init_consumer() const {}

    CUTLASS_DEVICE
    void
    prefetch_next_work(Params const& params, WorkTileInfo& current_work) const {}

    template<bool IsProducer=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_next_work(Params const& params, WorkTileInfo const& current_work) const {
        return {-1, -1, -1, false};
    }

};


} // namespace flash
