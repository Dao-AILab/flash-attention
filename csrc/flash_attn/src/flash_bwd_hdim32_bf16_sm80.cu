// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#include "flash_bwd_launch_template.h"

template<>
void run_mha_bwd_<cutlass::bfloat16_t, 32, false>(Flash_bwd_params &params, cudaStream_t stream) {
    run_mha_bwd_hdim32<cutlass::bfloat16_t, false>(params, stream);
}
