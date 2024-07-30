// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.

#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_<cutlass::float_e4m3_t, 256>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim256_fp8<cutlass::float_e4m3_t>(params, stream);
}
