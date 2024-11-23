// Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_<cutlass::float_e4m3_t, 192, true, false>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_fp8_hdim192<cutlass::float_e4m3_t, true, false>(params, stream);
}
