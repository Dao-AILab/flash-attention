#include "flash_bwd_launch_template.h"

template<typename T, bool Is_causal>
void run_mha_bwd_qkdim128_vdim256(Flash_bwd_params &params, cudaStream_t stream) {
    constexpr static int QKHeaddim = 128;
    constexpr static int VHeaddim = 256;
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
        constexpr static int Bc = 64;
        constexpr static int  smem_size = 2 *(Br * QKHeaddim * 2 /*Q with double buffer*/ +  Br * VHeaddim /* dO*/ + Bc * QKHeaddim /*K, dK*/ + Bc * VHeaddim /*V, dV*/ + 
                Br * Bc * 2 /*dS, P*/);
        // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, Headdim, 32, 128, 8, 2, 2, 2, false, false, T>>(params, stream);
        // This is faster, in the case of sequence-parallel bwd (where we need fewer registers).
        // Out of these three, the 2nd one is slightly faster (2% faster than the first). Idk why.
        // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, Headdim, 64, 128, 8, 2, 2, 2, false, false, T>>(params, stream);
        if (max_smem_per_block >= 144 * 1024) {
            run_flash_bwd<Flash_bwd_kernel_traits<QKHeaddim, VHeaddim, 64, 64, 8, 4, 2, 2, false, false, T>, Is_dropout, Is_causal>(params, stream);
            // A100 shared memory spill
            // run_flash_bwd<Flash_bwd_kernel_traits<QKHeaddim, VHeaddim, 64, 128, 8, 2, 4, 2, false, false, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_bwd_seqk_parallel<Flash_bwd_kernel_traits<Headdim, Headdim, 128, 128, 8, 4, 4, 4, false, false, T>, Is_dropout>(params, stream);
            // run_flash_bwd_seqk_parallel<Flash_bwd_kernel_traits<Headdim, Headdim, 128, 128, 8, 4, 4, 4, false, true, T>, Is_dropout>(params, stream);
            // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, Headdim, 64, 128, 8, 2, 4, 2, true, false, T>, Is_dropout>(params, stream);
            // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, Headdim, 128, 64, 8, 4, 2, 2, false, false, T>, Is_dropout>(params, stream);
            // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, Headdim, 128, 64, 8, 4, 2, 2, true, false, T>, Is_dropout>(params, stream);
        } else {
            // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, Headdim, 64, 64, 8, 4, 2, 2, false, false, T>, Is_dropout>(params, stream);
            run_flash_bwd<Flash_bwd_kernel_traits<QKHeaddim, VHeaddim, 64, 64, 8, 4, 2, 2, true, false, T>, Is_dropout, Is_causal>(params, stream);
        }
        // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, Headdim, 64, 128, 8, 2, 4, 4, false, false, T>>(params, stream);

        // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, Headdim, 128, 64, 8, 4, 4, 4, false, false, T>>(params, stream);
    });
}
