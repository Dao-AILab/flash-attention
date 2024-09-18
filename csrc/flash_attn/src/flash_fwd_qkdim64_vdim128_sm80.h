#include "flash_fwd_launch_template.h"

template<typename T, bool Is_causal>
void run_mha_fwd_qkdim64_vdim128(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int QKHeaddim = 64;
    constexpr static int VHeaddim = 128;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        if constexpr(!Is_dropout) {
            // Using 8 warps is 18% slower for seqlen=2k, 2 warps is 5% slower
            // Using block size (64 x 256) is 27% slower for seqlen=2k
            // Using block size (256 x 64) is 85% slower for seqlen=2k, because of register spilling
            run_flash_fwd<Flash_fwd_kernel_traits<QKHeaddim, VHeaddim, 128, 128, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, Headdim, 128, 64, 4, true, false, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, Headdim, 128, 64, 4, true, true, T>, Is_dropout, Is_causal>(params, stream);
        } else {
            run_flash_fwd<Flash_fwd_kernel_traits<QKHeaddim, VHeaddim, 128, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, Headdim, 128, 64, 4, true, true, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, Headdim, 128, 64, 4, true, false, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, Headdim, 128, 128, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
        }
    });
}
