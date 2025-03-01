// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

template void run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 32, false>(Flash_fwd_params &params, cudaStream_t stream);

} // namespace FLASH_NAMESPACE