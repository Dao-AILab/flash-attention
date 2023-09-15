// Copyright (c) 2023, Tri Dao.

// Splitting the different head dimensions to different files to speed up compilation.

#include "flash_bwd_launch_template.h"

// template<>
// void run_mha_bwd_<cutlass::half_t, 96>(Flash_bwd_params &params, cudaStream_t stream, const bool configure) {
//     using elem_type = cutlass::half_t;
//     if (params.h == params.h_k) {
//         // run_flash_bwd<Flash_bwd_kernel_traits<96, 64, 128, 8, 2, 4, 4, true, false, elem_type>>(params, stream, configure);
//         // This is very slightly faster
//         run_flash_bwd<Flash_bwd_kernel_traits<96, 64, 128, 8, 2, 4, 4, false, false, elem_type>>(params, stream, configure);
//     } else {
//         run_flash_bwd_seqq_parallel<Flash_bwd_kernel_traits<96, 128, 64, 8, 4, 4, 4, false, false, elem_type>>(params, stream, configure);
//     }
// }

template<>
void run_mha_bwd_<cutlass::half_t, 96>(Flash_bwd_params &params, cudaStream_t stream, const bool configure) {
    run_mha_bwd_hdim96<cutlass::half_t>(params, stream, configure);
}