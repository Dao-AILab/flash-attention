// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#include "flash_fwd_qkdim192_vdim128_sm80.h"

template<>
void run_mha_fwd_<cutlass::half_t, 192, 128, false>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_qkdim192_vdim128<cutlass::half_t, false>(params, stream);
}