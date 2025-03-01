/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include "cutlass/arch/barrier.h"

namespace flash {

////////////////////////////////////////////////////////////////////////////////////////////////////

// cutlass::arch::NamedBarrier::sync/arrive are only enabled Sm90 even though they work
// for Sm80 as well. We reimplement them here, enabled for both Sm90 and Sm80.

CUTLASS_DEVICE
static void named_barrier_sync(uint32_t num_threads, uint32_t barrier_id_) {
    static constexpr uint32_t ReservedNamedBarrierCount = static_cast<uint32_t>(cutlass::arch::ReservedNamedBarriers::FirstUserBarrier);
    uint32_t barrier_id = barrier_id_ + ReservedNamedBarrierCount;
    asm volatile("bar.sync %0, %1;" : : "r"(barrier_id), "r"(num_threads));
    cutlass::arch::synclog_emit_named_barrier_arrive_and_wait(__LINE__, num_threads, barrier_id);
}

CUTLASS_DEVICE
static void named_barrier_sync(uint32_t num_threads, cutlass::arch::ReservedNamedBarriers reserved_named_barriers) {
    uint32_t barrier_id = static_cast<uint32_t>(reserved_named_barriers);
    asm volatile("bar.sync %0, %1;" : : "r"(barrier_id), "r"(num_threads));
    cutlass::arch::synclog_emit_named_barrier_arrive_and_wait(__LINE__, num_threads, barrier_id);
}

CUTLASS_DEVICE
static void named_barrier_arrive(uint32_t num_threads, uint32_t barrier_id_) {
    static constexpr uint32_t ReservedNamedBarrierCount = static_cast<uint32_t>(cutlass::arch::ReservedNamedBarriers::FirstUserBarrier);
    uint32_t barrier_id = barrier_id_ + ReservedNamedBarrierCount;
    cutlass::arch::synclog_emit_named_barrier_arrive(__LINE__, num_threads, barrier_id);
    asm volatile("bar.arrive %0, %1;" : : "r"(barrier_id), "r"(num_threads));
}

CUTLASS_DEVICE
static void named_barrier_arrive(uint32_t num_threads, cutlass::arch::ReservedNamedBarriers reserved_named_barriers) {
    uint32_t barrier_id = static_cast<uint32_t>(reserved_named_barriers);
    cutlass::arch::synclog_emit_named_barrier_arrive(__LINE__, num_threads, barrier_id);
    asm volatile("bar.arrive %0, %1;" : : "r"(barrier_id), "r"(num_threads));
}


////////////////////////////////////////////////////////////////////////////////////////////////////
// Enumerates the reserved named barriers to avoid potential conflicts

enum class FwdNamedBarriers {
    QueryEmpty = 0,
    ProducerWG = 1,
    TileCountSmemEmpty = 2,
    TileCountSmemFull = 3,
    WarpSchedulerWG1 = 4,
    WarpSchedulerWG2 = 5,
    WarpSchedulerWG3 = 6,
    AppendKV = 7,
    QueryRotated = 8,
    PFull = 9,
    PEmpty = 6,  // HACK: PEmpty is only used when we don't have 3 WGs
};

enum class BwdNamedBarriers {
    KVEmpty = 0,
    PdS = 1,
    // This needs to match FwdNamedBarriers::TileCountSmemEmpty since TileScheduler uses it
    TileCountSmemEmpty = 2,
    TileCountSmemFull = 3,
    dQEmptyWG1 = 4,
    dQEmptyWG2 = 5,
    dQEmptyWG3 = 6,
    dQFullWG1 = 7,
    dQFullWG2 = 8,
    dQFullWG3 = 9,
};

} // flash
