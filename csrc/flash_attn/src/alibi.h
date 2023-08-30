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
                                   const uint32_t head_idx,
                                   const uint32_t num_heads,
                                   const float softmax_scale,
                                   const float alibi_start,
                                   const float alibi_ratio) {
    // TODO: compute alibi_start & alibi_ratio outside for tensor parallelism
    // const float alibi_start = powf(2.f, -powf(2.f, -(log2f((float)num_heads)-3.f)));
    // const float alibi_ratio = alibi_start;
    const float alibi_slope = alibi_start * powf(alibi_ratio, (float)head_idx);
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
            // tensor isn't scaled by softmax_scale so unscale alibi with softmax_scale
            const float alibi = (alibi_slope * col_idx) / softmax_scale;
            // Without the "make_coord" we get wrong results
            #pragma unroll
            for (int mi = 0; mi < size<0>(tensor); ++mi) {
                tensor(mi, make_coord(j, nj)) = tensor(mi, make_coord(j, nj)) + alibi;
            }
        }
    }
}

}  // namespace flash