
#include "flash_fwd_launch_template.h"

template<typename T, bool Is_causal>
void run_mha_fwd_qkdim32_vdim64(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int QKHeaddim = 32;
    constexpr static int VHeaddim = 64;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        run_flash_fwd<Flash_fwd_kernel_traits<QKHeaddim, VHeaddim, 128, 128, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
    });
}