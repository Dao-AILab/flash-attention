#include "flash_fwd_launch_template.h"

template<typename T, bool Is_causal>
void run_mha_fwd_qkdim96_vdim192(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int QKHeaddim = 96;
    constexpr static int VHeaddim = 192;
    auto dprops = at::cuda::getCurrentDeviceProperties();
    bool is_sm8x = dprops->major == 8 && dprops->minor > 0;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        // For sm86 or sm89, 64 x 64 is the fastest for causal (because it's square),
        if (is_sm8x) {
            if constexpr(!Is_causal) {
                run_flash_fwd<Flash_fwd_kernel_traits<QKHeaddim, VHeaddim, 128, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            } else {
                run_flash_fwd<Flash_fwd_kernel_traits<QKHeaddim, VHeaddim, 64, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            }
        } else {
            run_flash_fwd<Flash_fwd_kernel_traits<QKHeaddim, VHeaddim, 128, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
        }
        // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, Headdim, 128, 64, 4, true, false, T>, Is_dropout, Is_causal>(params, stream);
        // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, Headdim, 128, 64, 4, true, true, T>, Is_dropout, Is_causal>(params, stream);
        // These two are always slower
        // run_flash_fwd<Flash_fwd_kernel_traits<96, 128, 128, 4, true, T>>(params, stream);
        // run_flash_fwd<Flash_fwd_kernel_traits<96, 64, 128, 4, true, T>>(params, stream);
    });
}
