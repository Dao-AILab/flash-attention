// Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
// Splitting the different template instantiations to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#include "flash_fwd_launch_template.h"

#ifndef FLASHATTENTION_DISABLE_SM8x
#ifndef FLASHATTENTION_DISABLE_HDIM128
template void run_mha_fwd_<80, cutlass::bfloat16_t, 128, 128, true, true, true, true>(Flash_fwd_params &params, cudaStream_t stream);
template void run_mha_fwd_<86, cutlass::bfloat16_t, 128, 128, true, true, true, true>(Flash_fwd_params &params, cudaStream_t stream);
#endif
#endif
