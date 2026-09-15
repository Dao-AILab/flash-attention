#include "flash_bwd_launch_template.h"

namespace FLASH_NAMESPACE {

template<typename T>
struct Flash_bwd_d512_traits : Flash_bwd_kernel_traits<512, 16, 32, 2, 1, 1, 1, true, true, T> {
    // The smaller tile uses 64 threads. Clear dQ accumulators with all 64 threads,
    // instead of the 256-thread layout used by the existing backward tiles.
    using GmemTiledCopydQaccum = decltype(make_tiled_copy(
        Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, float>{},
        Layout<Shape<_4, _16>, Stride<_16, _1>>{}, Layout<Shape<_1, _4>>{}));
};

void run_mha_bwd_hdim512(Flash_bwd_params &params, cudaStream_t stream) {
#ifndef FLASHATTENTION_DISABLE_BACKWARD
    FP16_SWITCH(!params.is_bf16, [&] {
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
            // A single Q/dO buffer and register-resident V limit shared memory to 96 KiB.
            using Traits = Flash_bwd_d512_traits<elem_type>;
            constexpr int smem_size = Traits::kSmemSize1colblock;
            static_assert(smem_size <= 99 * 1024);
            dim3 grid_m((params.seqlen_q + Traits::kBlockM - 1) / Traits::kBlockM,
                        params.b, params.h);
            int nblocks = (params.seqlen_k + Traits::kBlockN - 1) / Traits::kBlockN;
            if (params.deterministic) {
                nblocks = (get_num_sm(get_current_device()) + params.b * params.h - 1)
                          / (params.b * params.h);
                flash_bwd_dot_do_o_kernel<false, Traits><<<grid_m, Traits::kNThreads, 0, stream>>>(params);
            } else {
                flash_bwd_dot_do_o_kernel<true, Traits><<<grid_m, Traits::kNThreads, 0, stream>>>(params);
            }
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            auto kernel = &flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<
                Traits, false, Is_causal, false, false, false, true, false>;
            C10_CUDA_CHECK(cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
            kernel<<<dim3(nblocks, params.b, params.h), Traits::kNThreads, smem_size, stream>>>(params);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            flash_bwd_convert_dq_kernel<Traits><<<grid_m, Traits::kNThreads, Traits::kSmemdQSize, stream>>>(
                params, params.deterministic ? nblocks : 1);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
    });
#endif
}

}  // namespace FLASH_NAMESPACE
