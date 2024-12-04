/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include "cutlass/arch/barrier.h"

namespace flash {

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
