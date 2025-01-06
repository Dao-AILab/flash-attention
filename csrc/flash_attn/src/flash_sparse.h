#pragma once

#include "flash.h"

namespace FLASH_NAMESPACE {

struct Flash_fwd_params_sparse : public Flash_fwd_params {
    // For sparse attention
    const int* block_count;
    const int* block_offset;
    const int* column_count;
    const int* column_index;
    int NUM_ROWS;
    int NNZ_S;
    int NNZ_V;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, int Headdim, bool Is_causal> void run_mha_fwd_sparse_(Flash_fwd_params_sparse &params, cudaStream_t stream);

}  // namespace FLASH_NAMESPACE