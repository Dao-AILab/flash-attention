#include "flash_fwd.h"

template<>
void run_mha_fwd_<std::conditional_t</*{is_bf16}*/,cutlass::bfloat16_t,cutlass::half_t>, /*{Kd}*/ , /*{D}*/, /*{is_causal}*/>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_qkdim/*{Kd}*/_vdim/*{D}*/<std::conditional_t</*{is_bf16}*/,cutlass::bfloat16_t,cutlass::half_t>, /*{is_causal}*/>(params, stream);
}
