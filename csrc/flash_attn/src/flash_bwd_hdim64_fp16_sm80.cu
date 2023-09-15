// Copyright (c) 2023, Tri Dao.

// Splitting the different head dimensions to different files to speed up compilation.

#include "flash_bwd_launch_template.h"

// template<>
// void run_mha_bwd_<cutlass::half_t, 64>(Flash_bwd_params &params, cudaStream_t stream, const bool configure) {
//     using elem_type = cutlass::half_t;
//     // Changing AtomLayoutMdQ from 2 to 4 takes the same time
//     // run_flash_bwd<Flash_bwd_kernel_traits<64, 64, 128, 8, 2, 4, 2, false, false, elem_type>>(params, stream, configure);
//     // run_flash_bwd<Flash_bwd_kernel_traits<64, 64, 128, 8, 2, 4, 2, true, false, elem_type>>(params, stream, configure);
//     // run_flash_bwd<Flash_bwd_kernel_traits<64, 128, 128, 8, 2, 4, 4, false, false, elem_type>>(params, stream, configure);
//     // This is slightly faster. We want to split M more so we need fewer registers to store LSE.
//     run_flash_bwd<Flash_bwd_kernel_traits<64, 128, 128, 8, 4, 4, 4, false, false, elem_type>>(params, stream, configure);
//     // run_flash_bwd<Flash_bwd_kernel_traits<64, 128, 64, 8, 4, 2, 4, true, false, elem_type>>(params, stream, configure);
//     // run_flash_bwd<Flash_bwd_kernel_traits<64, 64, 64, 4, 2, 2, 2, true, false, elem_type>>(params, stream, configure);
//     // run_flash_bwd<Flash_bwd_kernel_traits<64, 32, 128, 4, 1, 4, 1, false, false, elem_type>>(params, stream, configure);
//     // run_flash_bwd<Flash_bwd_kernel_traits<64, 16, 128, 4, 1, 4, 1, false, false, elem_type>>(params, stream, configure);
//     // M=128, N=64 is quite slow, I think because we need to read/write dQaccum twice as many times
//     // run_flash_bwd<Flash_bwd_kernel_traits<64, 128, 64, 8, 2, 2, 2, false, elem_type>>(params, stream, configure);
//     // run_flash_bwd<Flash_bwd_kernel_traits<64, 128, 64, 8, false, elem_type>>(params, stream, configure);
//     // run_flash_bwd<Flash_bwd_kernel_traits<64, 16, 256, 8, false, elem_type>>(params, stream, configure);
//     // run_flash_bwd<Flash_bwd_kernel_traits<64, 128, 128, 8, false, elem_type>>(params, stream, configure);
//     // run_flash_bwd<Flash_bwd_kernel_traits<64, 64, 64, 4, false, elem_type>>(params, stream, configure);

//     // run_flash_bwd<Flash_bwd_kernel_traits<64, 256, 64, 8, 8, 4, 8, false, false, elem_type>>(params, stream, configure);
//     // run_flash_bwd<Flash_bwd_kernel_traits<64, 256, 64, 8, 8, 4, 4, false, false, elem_type>>(params, stream, configure);
//     // run_flash_bwd<Flash_bwd_kernel_traits<64, 128, 64, 4, 4, 2, 4, false, false, elem_type>>(params, stream, configure);
// }

template<>
void run_mha_bwd_<cutlass::half_t, 64>(Flash_bwd_params &params, cudaStream_t stream, const bool configure) {
    run_mha_bwd_hdim64<cutlass::half_t>(params, stream, configure);
}