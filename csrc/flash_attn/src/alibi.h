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
                                   const int col_idx_offset_,
                                   const int max_seqlen_k, 
                                   const int row_idx_offset_,
                                   const int max_seqlen_q, 
                                   const int warp_row_stride,
                                   const int head_idx,
                                   const float softmax_scale,
                                   const float alibi_slope) {
    // tensor has shape (ncol=(2, MMA_M), nrow=(2, MMA_N))
    static_assert(Layout::rank == 2, "Only support 2D Tensor");
    const int lane_id = threadIdx.x % 32;
    const int row_idx_offset = row_idx_offset_;
    const int col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;
    const float alibi_slope_unscaled = alibi_slope / softmax_scale;
    #pragma unroll
    for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
        const int row_idx_base = row_idx_offset + mi * warp_row_stride;
        #pragma unroll
        for (int i = 0; i < size<0, 0>(tensor); ++i) {
            const int row_idx = row_idx_base + i * 8;
            #pragma unroll
            for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                const int col_idx_base = col_idx_offset + nj * 8;
                #pragma unroll
                for (int j = 0; j < size<1, 0>(tensor); ++j) {
                    const int col_idx = col_idx_base + j;
                    const float alibi = alibi_slope_unscaled * col_idx;
                    if (col_idx < max_seqlen_k && row_idx < max_seqlen_q) {
                        tensor(make_coord(i, mi), make_coord(j, nj)) += alibi;
                    }
                }
            }
        }
    }
}

}  // namespace flash