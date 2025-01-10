// Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
// Splitting the different template instantiations to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#include "flash_bwd_launch_template.h"

#ifndef FLASHATTENTION_DISABLE_SM8x
#ifndef FLASHATTENTION_DISABLE_HDIM64
template<>
void run_mha_bwd_<80, cutlass::half_t, 64, false>(Flash_bwd_params &params, cudaStream_t stream) {
    run_mha_bwd_hdim64<80, cutlass::half_t, false>(params, stream);
}
template<>
void run_mha_bwd_<86, cutlass::half_t, 64, false>(Flash_bwd_params &params, cudaStream_t stream) {
    run_mha_bwd_hdim64<86, cutlass::half_t, false>(params, stream);
}
#endif
#endif
