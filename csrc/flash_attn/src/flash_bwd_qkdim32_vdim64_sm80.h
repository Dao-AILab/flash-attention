#include "flash_bwd_launch_template.h"

template<typename T, bool Is_causal>
void run_mha_bwd_qkdim32_vdim64(Flash_bwd_params &params, cudaStream_t stream) {
    constexpr static int QKHeaddim = 32;
    constexpr static int VHeaddim = 64;
    int device;
    cudaGetDevice(&device);
    int max_smem_per_block;
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (status_ != cudaSuccess) {
      C10_CUDA_CHECK(status_);
    }
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        constexpr static int Br = 128;
        constexpr static int Bc = 128;
        constexpr static int  smem_size = 2 *(Br * QKHeaddim * 2 /*Q with double buffer*/ +  Br * VHeaddim /* dO*/ + Bc * QKHeaddim /*K, dK*/ + Bc * VHeaddim /*V, dV*/ + 
                Br * Bc * 2 /*dS, P*/);
        // if (max_smem_per_block >= 2 * ((3 * 128 + 2 * 128) * Headdim + 2 * 128 * 128)) { // 104 KB
        if (max_smem_per_block >= 104 * 1024) {  // 104 KB
            if constexpr(!Is_dropout) {  // We can afford more registers to keep V in registers
                run_flash_bwd<Flash_bwd_kernel_traits<QKHeaddim, VHeaddim, 128, 128, 8, 4, 4, 4, true, false, T>, Is_dropout, Is_causal>(params, stream);
            } else {
                run_flash_bwd<Flash_bwd_kernel_traits<QKHeaddim, VHeaddim, 128, 128, 8, 4, 4, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            }
        } else {  // 96 KB
            run_flash_bwd<Flash_bwd_kernel_traits<QKHeaddim, VHeaddim, 128, 128, 8, 4, 4, 4, true, false, T>, Is_dropout, Is_causal>(params, stream);
        }
    });
}

