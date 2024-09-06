#include "flash_fwd_launch_template.h"

#define False false
#define True true

template<typename T, bool Is_causal>
void run_mha_fwd_qkdim/*{Kd}*/_vdim/*{D}*/(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int QKHeaddim = /*{Kd}*/;
    constexpr static int VHeaddim = /*{D}*/;
    constexpr static int Br = /*{Br}*/;
    constexpr static int Bc = /*{Bc}*/;
    constexpr static int Nwarps = /*{Nwarps}*/;
    constexpr static bool IsQinRegs = /*{isQinRegs}*/;
    constexpr static bool SharedQKSmem = /*{SharedQKSmem}*/;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        run_flash_fwd<Flash_fwd_kernel_traits<QKHeaddim, VHeaddim, Br, Bc, Nwarps, IsQinRegs, SharedQKSmem, T>, Is_dropout, Is_causal>(params, stream);
    });
}