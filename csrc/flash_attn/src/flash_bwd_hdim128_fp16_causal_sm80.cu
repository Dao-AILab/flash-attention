// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#include "flash_bwd_launch_template.h"

template<>
void run_mha_bwd_<cutlass::half_t, 128, true>(Flash_bwd_params &params, cudaStream_t stream) {
    run_mha_bwd_hdim128<cutlass::half_t, true>(params, stream);
}
