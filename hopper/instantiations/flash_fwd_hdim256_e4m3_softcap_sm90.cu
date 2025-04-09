// Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
// Splitting the different template instantiations to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#include "flash_fwd_launch_template.h"

#ifndef FLASHATTENTION_DISABLE_HDIM256
template void run_mha_fwd_<90, cutlass::float_e4m3_t, 256, 256, false, false, true, false>(Flash_fwd_params &params, cudaStream_t stream);
#endif
