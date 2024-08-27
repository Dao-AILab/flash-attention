#include "flash_bwd_launch_template.h"

template<typename T, bool Is_causal>
void run_mha_bwd_qkdim96_vdim192(Flash_bwd_params &params, cudaStream_t stream) {
    constexpr static int QKHeaddim = 96;
    constexpr static int VHeaddim = 192;
    int device;
    cudaGetDevice(&device);
    int max_smem_per_block;
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (status_ != cudaSuccess) {
      C10_CUDA_CHECK(status_);
    }
    // printf("max_smem_per_block = %d\n", max_smem_per_block);
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        constexpr static int Br = 64;
        constexpr static int Bc = 128;
        constexpr static int  smem_size = 2 *(Br * QKHeaddim * 2 /*Q with double buffer*/ +  Br * VHeaddim /* dO*/ + Bc * QKHeaddim /*K, dK*/ + Bc * VHeaddim /*V, dV*/ + 
                Br * Bc * 2 /*dS, P*/);
        if (max_smem_per_block >= 116 * 1024) {
            if constexpr(!Is_dropout) {  // 92KB
                run_flash_bwd<Flash_bwd_kernel_traits<QKHeaddim, VHeaddim, 64, 128, 8, 2, 4, 4, true, false, T>, Is_dropout, Is_causal>(params, stream);
            } else {  // 116 KB
                // This is faster for dropout since we don't have many registers to spare
                run_flash_bwd<Flash_bwd_kernel_traits<QKHeaddim, VHeaddim, 64, 128, 8, 2, 4, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            }
        } else {
            run_flash_bwd<Flash_bwd_kernel_traits<QKHeaddim, VHeaddim, 64, 128, 8, 2, 4, 4, true, false, T>, Is_dropout, Is_causal>(params, stream);
        }
    });
}

