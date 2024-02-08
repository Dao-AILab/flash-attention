#pragma once

#include <cmath>

#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "utils.h"

namespace flash {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Is_even_MN, int kBlockM, int kBlockN>
struct AttnBias {

    const int max_seqlen_k, max_seqlen_q;

    __forceinline__ __device__ AttnBias(const int max_seqlen_k, const int max_seqlen_q)
        : max_seqlen_k(max_seqlen_k)
        , max_seqlen_q(max_seqlen_q) {
    };

    template <typename Engine0, typename Layout0, typename Engine1, typename Layout1>
    __forceinline__ __device__ void apply_attn_bias(Tensor<Engine0, Layout0> &tensor,
                                    Tensor<Engine1, Layout1> &bias,
                                    const int col_idx_offset_,
                                    const int row_idx_offset,
                                    const int warp_row_stride,
                                    const float softmax_scale) {
        // tensor has shape (ncol=(2, MMA_M), nrow=(2, MMA_N))
        //static_assert(Layout::rank == 2, "Only support 2D Tensor");
        const int lane_id = threadIdx.x % 32;
        const int col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;

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
                        if (Is_even_MN || (col_idx < max_seqlen_k && row_idx < max_seqlen_q)) {
                            tensor(make_coord(i, mi), make_coord(j, nj)) += bias(make_coord(i, mi), make_coord(j, nj)) / softmax_scale;
                        }
                    }
                }
            }
        }
    }

    template<typename Tensor0, typename Tensor1,
         typename TiledCopy, typename ThrCopy>
    __forceinline__ __device__ void copy_ds(Tensor0 &tBgdS, Tensor1 &tBsdS,
                                TiledCopy gmem_tiled_copy_dS, ThrCopy gmem_thr_copy_dS,
                                const int m_block, const int n_block,
                                const bool use_atomic_add) {

        if (Is_even_MN) {
            cute::copy(gmem_tiled_copy_dS, tBsdS, tBgdS);
        } else { // Is_even_MN
            Tensor cdS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});    // (BLK_M,BLK_N) -> (blk_m,blk_n)
            Tensor tdScdS = gmem_thr_copy_dS.partition_D(cdS);

            #pragma unroll
            for (int m = 0; m < size<1>(tdScdS); ++m) {
                if (Is_even_MN || get<0>(tdScdS(0, m, 0)) < max_seqlen_q - m_block * kBlockM) {
                    #pragma unroll
                    for (int n = 0; n < size<2>(tdScdS); ++n) {
                        if (Is_even_MN || get<1>(tdScdS(0, 0, n)) < max_seqlen_k - n_block * kBlockN) {
                            cute::copy(gmem_tiled_copy_dS, tBsdS(_, m, n), tBgdS(_, m, n));
                        }
                    }
                }
            }
        } // Is_even_MN
    }
};

} // namespace flash
