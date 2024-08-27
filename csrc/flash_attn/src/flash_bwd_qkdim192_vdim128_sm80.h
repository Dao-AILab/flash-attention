#include "flash_bwd_launch_template.h"

template<typename T, bool Is_causal>
void run_mha_bwd_qkdim192_vdim128(Flash_bwd_params &params, cudaStream_t stream) {
    constexpr static int QKHeaddim = 192;
    constexpr static int VHeaddim = 128;
    int device;
    cudaGetDevice(&device);
    int max_smem_per_block;
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (status_ != cudaSuccess) {
      C10_CUDA_CHECK(status_);
    }
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        if (max_smem_per_block >= 136 * 1024) {
            run_flash_bwd<Flash_bwd_kernel_traits<QKHeaddim, VHeaddim, 64, 64, 8, 4, 2, 2, false, false, T>, Is_dropout, Is_causal>(params, stream);
        } else {
            run_flash_bwd<Flash_bwd_kernel_traits<QKHeaddim, VHeaddim, 64, 64, 8, 4, 2, 2, true, true, T>, Is_dropout, Is_causal>(params, stream);
        }
    });
}