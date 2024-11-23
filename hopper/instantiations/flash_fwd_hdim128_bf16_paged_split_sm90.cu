// Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_<cutlass::bfloat16_t, 128, true, true>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim_16b<cutlass::bfloat16_t, 128, true, true>(params, stream);
}
