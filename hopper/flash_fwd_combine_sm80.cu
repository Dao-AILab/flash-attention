// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.

#include "flash_fwd_combine_launch_template.h"

template void run_mha_fwd_combine_<float, 64>(Flash_fwd_params &params, cudaStream_t stream);
template void run_mha_fwd_combine_<float, 128>(Flash_fwd_params &params, cudaStream_t stream);
template void run_mha_fwd_combine_<float, 256>(Flash_fwd_params &params, cudaStream_t stream);

template void run_mha_fwd_combine_<cutlass::half_t, 64>(Flash_fwd_params &params, cudaStream_t stream);
template void run_mha_fwd_combine_<cutlass::half_t, 128>(Flash_fwd_params &params, cudaStream_t stream);
template void run_mha_fwd_combine_<cutlass::half_t, 256>(Flash_fwd_params &params, cudaStream_t stream);

template void run_mha_fwd_combine_<cutlass::bfloat16_t, 64>(Flash_fwd_params &params, cudaStream_t stream);
template void run_mha_fwd_combine_<cutlass::bfloat16_t, 128>(Flash_fwd_params &params, cudaStream_t stream);
template void run_mha_fwd_combine_<cutlass::bfloat16_t, 256>(Flash_fwd_params &params, cudaStream_t stream);
