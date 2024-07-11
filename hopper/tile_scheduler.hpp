/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include "cutlass/fast_math.h"

namespace flash {

///////////////////////////////////////////////////////////////////////////////

class StaticPersistentTileSchedulerOld {
  //
  // Data members
  //

private:
  int current_work_linear_idx_;
  cutlass::FastDivmod const &m_block_divmod, &head_divmod;
  int const total_blocks;

public:
  struct WorkTileInfo {
    int M_idx = 0;
    int H_idx = 0;
    int B_idx = 0;
    bool is_valid_tile = false;

    CUTLASS_HOST_DEVICE
    bool
    is_valid() const {
      return is_valid_tile;
    }

    CUTLASS_HOST_DEVICE
    static WorkTileInfo
    invalid_work_tile() {
      return {-1, -1, -1, false};
    }

  };

public:

  CUTLASS_DEVICE explicit StaticPersistentTileSchedulerOld(cutlass::FastDivmod const &m_block_divmod_,
                                                        cutlass::FastDivmod const &head_divmod_,
                                                        int const total_blocks_) :
    m_block_divmod(m_block_divmod_), head_divmod(head_divmod_), total_blocks(total_blocks_) {

    // MSVC requires protecting use of CUDA-specific nonstandard syntax,
    // like blockIdx and gridDim, with __CUDA_ARCH__.
#if defined(__CUDA_ARCH__)
    // current_work_linear_idx_ = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    current_work_linear_idx_ = blockIdx.x;
#else
    CUTLASS_ASSERT(false && "This line should never be reached");
#endif
  }

  CUTLASS_DEVICE
  WorkTileInfo
  get_current_work() const {
    return get_current_work_for_linear_idx(current_work_linear_idx_);
  }

  CUTLASS_DEVICE
  WorkTileInfo
  get_current_work_for_linear_idx(int linear_idx) const {
    if (linear_idx >= total_blocks) {
      return WorkTileInfo::invalid_work_tile();
    }

    // Map worker's linear index into the CTA tiled problem shape to the corresponding MHB indices
    int M_idx, H_idx, B_idx;
    int quotient = m_block_divmod.divmod(M_idx, linear_idx);
    B_idx = head_divmod.divmod(H_idx, quotient);
    return {M_idx, H_idx, B_idx, true};
  }

  CUTLASS_DEVICE
  void
  // advance_to_next_work(int advance_count = 1) {
  advance_to_next_work() {
    // current_work_linear_idx_ += int(gridDim.x * gridDim.y * gridDim.z);
    current_work_linear_idx_ += int(gridDim.x);
  }

  CUTLASS_DEVICE
  WorkTileInfo
  fetch_next_work() {
    WorkTileInfo new_work_tile_info;
    advance_to_next_work();
    new_work_tile_info = get_current_work();
    return new_work_tile_info;
  }

};

///////////////////////////////////////////////////////////////////////////////

class SingleTileScheduler {

public:

    // Host side kernel arguments
    struct Arguments {
        int const num_blocks_m, num_head, num_batch;
        int const* tile_count_semaphore = nullptr;
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

        CUTLASS_DEVICE
        WorkTileInfo
        get_next_work(Params const& params) const {
            return {-1, -1, -1, false};
        }

    };

    CUTLASS_DEVICE
    WorkTileInfo
    get_initial_work() const {
        return {int(blockIdx.x), int(blockIdx.y), int(blockIdx.z), true};
    }

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
        int const* tile_count_semaphore = nullptr;
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
    WorkTileInfo
    get_initial_work() const {
        return {int(blockIdx.x)};
    }

    CUTLASS_DEVICE
    WorkTileInfo
    get_next_work(Params const& params, WorkTileInfo const& current_work) const {
        return {current_work.tile_idx + int(gridDim.x)};
    }

};

class DynamicPersistentTileScheduler {

public:

    // Host side kernel arguments
    struct Arguments {
        int const num_blocks_m, num_head, num_batch;
        int const* tile_count_semaphore;
    };

    // Device side kernel params
    struct Params {
        int const total_blocks;
        cutlass::FastDivmod const m_block_divmod, head_divmod;
        int const* tile_count_semaphore;
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

    using WorkTileInfo = StaticPersistentTileScheduler::WorkTileInfo;
    // struct WorkTileInfo {
    //     int tile_idx;

    //     CUTLASS_DEVICE
    //     bool
    //     is_valid(Params const& params) const {
    //         return tile_idx < params.total_blocks;
    //     }

    //     CUTLASS_DEVICE
    //     cute::tuple<int32_t, int32_t, int32_t>
    //     get_block_coord(Params const& params) const {
    //         int m_block, bidh, bidb;
    //         bidb = params.head_divmod.divmod(bidh, params.m_block_divmod.divmod(m_block, tile_idx));
    //         return {m_block, bidh, bidb};
    //     }

    // };

    CUTLASS_DEVICE
    WorkTileInfo
    get_initial_work() const {
        return {int(blockIdx.x)};
    }

    CUTLASS_DEVICE
    WorkTileInfo
    get_next_work(Params const& params, WorkTileInfo const& current_work) const {
        return {current_work.tile_idx + int(gridDim.x)};
    }

};

} // flash
