// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

// The num_splits==1 blocksize-aligned tree is instantiated in its own translation unit
// (flash_fwd_split_align_*.cu) so it compiles in parallel; declare it extern so
// the dispatch below references it instead of re-instantiating.
extern template void run_mha_fwd_splitkv_align<cutlass::half_t, 192, true>(Flash_fwd_params &params, cudaStream_t stream);

template void run_mha_fwd_splitkv_dispatch<cutlass::half_t, 192, true>(Flash_fwd_params &params, cudaStream_t stream);

} // namespace FLASH_NAMESPACE