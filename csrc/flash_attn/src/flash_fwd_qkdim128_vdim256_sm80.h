#include "flash_fwd_launch_template.h"

template<typename T, bool Is_causal>
void run_mha_fwd_qkdim128_vdim256(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int QKHeaddim = 128;
    constexpr static int VHeaddim = 256;
    auto dprops = at::cuda::getCurrentDeviceProperties();
    bool is_sm8x = dprops->major == 8 && dprops->minor > 0;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        if constexpr(!Is_dropout) {
            // For sm86 or sm89, 64 x 64 is the fastest for causal (because it's square),
            // and 128 x 32 (48 KB smem) is the fastest for non-causal since we get 2 CTAs per SM.
            if (is_sm8x) {
                if constexpr(!Is_causal) {
                    run_flash_fwd<Flash_fwd_kernel_traits<QKHeaddim, VHeaddim, 128, 32, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
                } else {
                    run_flash_fwd<Flash_fwd_kernel_traits<QKHeaddim, VHeaddim, 64, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
                }
            } else {
                run_flash_fwd<Flash_fwd_kernel_traits<QKHeaddim, VHeaddim, 128, 64, 8, false, false, T>, Is_dropout, Is_causal>(params, stream);
                // slow on A100
                // run_flash_fwd<Flash_fwd_kernel_traits<QKHeaddim, VHeaddim, 128, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            }
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, Headdim, 128, 64, 4, true, false, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, Headdim, 128, 64, 4, true, true, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, Headdim, 64, 128, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            // Using 8 warps (128 x 128 and 256 x 64) is 28% slower for seqlen=2k
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, Headdim, 128, 128, 8, false, false, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, Headdim, 128, 64, 8, false, false, T>, Is_dropout, Is_causal>(params, stream);
            // 1st ones are good for H100, A100
            // 2nd one is good for A6000 bc we get slightly better occupancy
        } else {
            run_flash_fwd<Flash_fwd_kernel_traits<QKHeaddim, VHeaddim, 128, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            // A100 RuntimeError: CUDA error: an illegal memory access was encountered
            // run_flash_fwd<Flash_fwd_kernel_traits<QKHeaddim, VHeaddim, 128, 32, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, Headdim, 64, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, Headdim, 128, 32, 4, true, false, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, Headdim, 128, 32, 4, true, true, T>, Is_dropout, Is_causal>(params, stream);
        }
    });
}
