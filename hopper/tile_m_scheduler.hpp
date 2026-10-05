#pragma once

#include "cutlass/fast_math.h"
#include <cuda/std/iterator>

/*
Scheduler and dependency helpers for the execution order of q tiles (m_block)
*/

namespace flash {

// targeting full mask, so m_min must be 0 and m_max must be the same across all kv tiles
// should be used with shift_dependency
// e.g. q_tile: 6 active_kv_tile: 3, schedule:
// | 0    | 4    | 2    |
// | 1    | 0    | 3    |
// | 2    | 1    | 0    |
// | 3    | 2    | 1    |
// | 4    | 3    | 2    |
class ShiftScheduler {
public:
    int cur_step;
    const int start_q_idx;
    const int total_steps;

    using iterator_category = cuda::std::forward_iterator_tag;
    using value_type = int;
    using difference_type = cuda::std::ptrdiff_t;
    using pointer = const int*;
    using reference = const int&;

    CUTLASS_DEVICE ShiftScheduler(int m_min, int m_max, int active_kv_idx)
        : cur_step(0), start_q_idx(active_kv_idx), total_steps(m_max) {
        (void)m_min;
    }

    CUTLASS_DEVICE bool valid() const { return cur_step < total_steps; }

    CUTLASS_DEVICE ShiftScheduler& operator++() {
        ++cur_step;
        return *this;
    }

    CUTLASS_DEVICE value_type operator*() const {
        int tmp = start_q_idx + cur_step;
        if (tmp >= total_steps) tmp -= total_steps;
        return tmp; // wrap around
    }
};

// dependency order for ShiftScheduler
// e.g. q_tile: 6 active_kv_tile: 3,
// schedule:
// | 0    | 4    | 2    |
// | 1    | 0    | 3    |
// | 2    | 1    | 0    |
// | 3    | 2    | 1    |
// | 4    | 3    | 2    |
// dependency:
// | 0    | 2    | 1    |
// | 1    | 0    | 2    |
// | 2    | 1    | 0    |
// | 2    | 1    | 0    |
// | 2    | 1    | 0    |
CUTLASS_DEVICE int shift_dependency(int q_id, int active_kv_id, int active_kv_tiles, int executed_kv_tiles) {
    int effective_row = q_id < active_kv_tiles ? q_id : int(active_kv_tiles - 1);
    int tmp = effective_row - active_kv_id;
    if (tmp < 0) tmp += active_kv_tiles;
    return executed_kv_tiles + tmp;
}

} // namespace flash
