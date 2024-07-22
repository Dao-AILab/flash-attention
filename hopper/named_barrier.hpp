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
    ValueEmpty = 1,
    TileCountSmemEmpty = 2,
    TileCountSmemFull = 3,
    WarpSchedulerWG1 = 4,
    WarpSchedulerWG2 = 5,
    WarpSchedulerWG3 = 6,
    ProducerWG = 7

// #ifdef NEW_FP8_EPI_BARRIER
//     OutEmpty = 7
// #endif
};

} // flash