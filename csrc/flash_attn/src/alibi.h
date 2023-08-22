#include <cmath>

#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/array.h>

#include "utils.h"

namespace flash {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Engine, typename Layout>
inline __device__ void apply_alibi(Tensor<Engine, Layout> &tensor,
                                   const uint32_t col_idx_offset_,
                                   const uint32_t h_idx,
                                   const float softmax_scale) {
    // head_idx == bidh == blockIdx.z so num_heads == gridDim.z
    // TODO: compute alibi_start & alibi_ratio outside for tensor parallelism
    const uint32_t num_heads = gridDim.z;
    const float alibi_start = powf(2.f, -powf(2.f, -(log2f((float)num_heads)-3.f)));
    const float alibi_ratio = alibi_start;
    const float alibi_slope = alibi_start * powf(alibi_ratio, (float)h_idx);
    // tensor has shape (ncol=(2, MMA_M), nrow=(2, MMA_N))
    static_assert(Layout::rank == 2, "Only support 2D Tensor");
    const uint32_t lane_id = threadIdx.x % 32;
    const uint32_t col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;
    #pragma unroll
    for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
        const uint32_t col_idx_base = col_idx_offset + nj * 8;
        #pragma unroll
        for (int j = 0; j < size<1, 0>(tensor); ++j) {
            const uint32_t col_idx = col_idx_base + j;
            // tensor isn't scaled by softmax_scale so divide alibi with softmax_scale
            const float alibi = (alibi_slope * col_idx) / softmax_scale;
            // Without the "make_coord" we get wrong results
            #pragma unroll
            for (int mi = 0; mi < size<0>(tensor); ++mi) {
                tensor(mi, make_coord(j, nj)) = tensor(mi, make_coord(j, nj)) + alibi;
            }
        }
    }
}

// Apply the exp to all the elements with alibi.
template <bool Scale_max=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
inline __device__ void scale_apply_exp2_alibi(Tensor<Engine0, Layout0> &tensor, 
                                              Tensor<Engine1, Layout1> const &max, 
                                              const float scale, 
                                              const uint32_t col_idx_offset_,
                                              const uint32_t head_idx) {
    // head_idx == bidh == blockIdx.z so num_heads == gridDim.z
    // TODO: I'll compute alibi_start & alibi_ratio outside later
    const uint32_t num_heads = gridDim.z;
    const float alibi_start = powf(2.0f, -powf(2.0f, -(log2f((float)num_heads)-3.0f)));
    const float alibi_ratio = alibi_start;
    const float alibi_slope = alibi_start * powf(alibi_ratio, (float)head_idx);
    // tensor has shape (ncol=(2, MMA_M), nrow=(2, MMA_N))
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
    const uint32_t lane_id = threadIdx.x % 32;
    const uint32_t col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        // If max is -inf, then all elements must have been -inf (possibly due to masking).
        // We don't want (-inf - (-inf)) since that would give NaN.
        // If we don't have float around M_LOG2E the multiplication is done in fp64.
        const float max_scaled = max(mi) == -INFINITY ? 0.f : max(mi) * (Scale_max ? scale : float(M_LOG2E));
        #pragma unroll
        for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
            const uint32_t col_idx_base = col_idx_offset + nj * 8;
            #pragma unroll
            for (int j = 0; j < size<1, 0>(tensor); ++j) {
                // Make alibi * log_2(e)
                const uint32_t col_idx = col_idx_base + j;
                const float alibi = alibi_slope * col_idx * float(M_LOG2E);
                // Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
                // max * log_2(e)) This allows the compiler to use the ffma
                // instruction instead of fadd and fmul separately.
                tensor(mi, make_coord(j, nj)) = exp2f(tensor(mi, make_coord(j, nj)) * scale + alibi - max_scaled);
            }
        }
    }
}

}  // namespace flash