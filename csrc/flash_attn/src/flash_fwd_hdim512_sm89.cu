#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

void run_mha_fwd_hdim512(Flash_fwd_params &params, cudaStream_t stream) {
    FP16_SWITCH(!params.is_bf16, [&] {
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
            // Keeping Q in registers lets Q and K share the same 64 KiB buffer.
            using Traits = Flash_fwd_kernel_traits<512, 64, 32, 4, true, true, elem_type>;
            constexpr int smem_size = Traits::kSmemSize;
            static_assert(smem_size <= 99 * 1024);
            auto kernel = &flash_fwd_kernel<Traits, false, Is_causal, false, false,
                                           false, true, false, false>;
            C10_CUDA_CHECK(cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
            dim3 grid((params.seqlen_q + Traits::kBlockM - 1) / Traits::kBlockM,
                      params.b, params.h);
            kernel<<<grid, Traits::kNThreads, smem_size, stream>>>(params);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
    });
}

}  // namespace FLASH_NAMESPACE
