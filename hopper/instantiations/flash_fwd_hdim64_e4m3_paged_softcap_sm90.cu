// Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
// Splitting the different template instantiations to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#include "flash_fwd_launch_template.h"

template void run_mha_fwd_<cutlass::float_e4m3_t, 64, false, true, true, false>(Flash_fwd_params &params, cudaStream_t stream);
