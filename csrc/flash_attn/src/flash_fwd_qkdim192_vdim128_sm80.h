#include "flash_fwd_launch_template.h"

template<typename T, bool Is_causal>
void run_mha_fwd_qkdim192_vdim128(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int QKHeaddim = 192;
    constexpr static int VHeaddim = 128;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        if constexpr(!Is_dropout) {
            run_flash_fwd<Flash_fwd_kernel_traits<QKHeaddim, VHeaddim, 128, 64, 8, false, false, T>, Is_dropout, Is_causal>(params, stream);
        } else {
            run_flash_fwd<Flash_fwd_kernel_traits<QKHeaddim, VHeaddim, 64, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
        }
        // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, Headdim, 64, 32, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
        // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, Headdim, 128, 32, 8, false, false, T>, Is_dropout, Is_causal>(params, stream);
        // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, Headdim, 128, 64, 4, false, T>>(params, stream);
        // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, Headdim, 64, 128, 4, false, T>>(params, stream);
        // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, Headdim, 128, 128, 8, false, T>>(params, stream);
    });
}
