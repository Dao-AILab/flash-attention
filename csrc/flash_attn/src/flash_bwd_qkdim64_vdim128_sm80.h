#include "flash_bwd_launch_template.h"

template<typename T, bool Is_causal>
void run_mha_bwd_qkdim64_vdim128(Flash_bwd_params &params, cudaStream_t stream) {
    constexpr static int QKHeaddim = 64;
    constexpr static int VHeaddim = 128;
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
        // Changing AtomLayoutMdQ from 2 to 4 takes the same time
        // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, Headdim, 64, 128, 8, 2, 4, 2, false, false, T>>(params, stream);
        // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, Headdim, 64, 128, 8, 2, 4, 2, true, false, T>>(params, stream);
        // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, Headdim, 128, 128, 8, 2, 4, 4, false, false, T>>(params, stream);
        // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, Headdim, 128, 64, 8, 4, 2, 4, false, false, T>, Is_dropout>(params, stream);
        // This is slightly faster. We want to split M more so we need fewer registers to store LSE.
        constexpr static int Br = 64;
        constexpr static int Bc = 128;
        constexpr static int  smem_size = 2 *(Br * QKHeaddim * 2 /*Q with double buffer*/ +  Br * VHeaddim /* dO*/ + Bc * QKHeaddim /*K, dK*/ + Bc * VHeaddim /*V, dV*/ + 
                Br * Bc * 2 /*dS, P*/);
        // printf("smem_size = %d\n", smem_size);
        // printf("max_smem_per_block = %d\n", max_smem_per_block);

        if (max_smem_per_block >= 144 * 1024) {
            run_flash_bwd<Flash_bwd_kernel_traits<QKHeaddim, VHeaddim, 64, 128, 8, 2, 4, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            // A100 shared memory spill
            // run_flash_bwd<Flash_bwd_kernel_traits<QKHeaddim, VHeaddim, 128, 128, 8, 4, 4, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            // This has a lot of register spilling
            // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, Headdim, 128, 128, 8, 4, 4, 4, true, false, T>, Is_dropout>(params, stream);
        } else {
            // if (params.h == params.h_k) {
                // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, Headdim, 64, 128, 8, 2, 4, 4, false, false, T>, Is_dropout>(params, stream);
            run_flash_bwd<Flash_bwd_kernel_traits<QKHeaddim,VHeaddim, 64, 128, 8, 2, 4, 4, true, false, T>, Is_dropout, Is_causal>(params, stream);
                // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, Headdim, 128, 64, 8, 4, 2, 4, false, false, T>, Is_dropout>(params, stream);
                // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, Headdim, 128, 64, 8, 4, 2, 4, true, false, T>, Is_dropout>(params, stream);
            // } else {
            // }
        }
    });
    // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, Headdim, 128, 64, 8, 4, 2, 4, true, false, T>>(params, stream);
    // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, Headdim, 64, 64, 4, 2, 2, 2, true, false, T>>(params, stream);
    // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, Headdim, 32, 128, 4, 1, 4, 1, false, false, T>>(params, stream);
    // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, Headdim, 16, 128, 4, 1, 4, 1, false, false, T>>(params, stream);
    // M=128, N=64 is quite slow, I think because we need to read/write dQaccum twice as many times
    // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, Headdim, 128, 64, 8, 2, 2, 2, false, T>>(params, stream);
    // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, Headdim, 128, 64, 8, false, T>>(params, stream);
    // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, Headdim, 64, 64, 4, false, T>>(params, stream);

    // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, Headdim, 128, 64, 4, 4, 2, 4, false, false, T>>(params, stream);
}

