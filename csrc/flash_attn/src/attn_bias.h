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

__forceinline__ __device__ void atomicAdd2(cutlass::bfloat16_t *address_, cutlass::bfloat16_t *val) {
    atomicAdd(reinterpret_cast<__nv_bfloat162*>(address_), *reinterpret_cast<__nv_bfloat162*>(val));
};

__forceinline__ __device__ void atomicAdd2(cutlass::half_t *address_, cutlass::half_t *val) {
    atomicAdd(reinterpret_cast< __half2*>(address_), *reinterpret_cast< __half2*>(val));
};

/*template<bool Is_even_MN,
            typename Engine0, typename Layout0,
            typename Engine1, typename Layout1,
            typename Engine2, typename Layout2,
            typename TiledCopy>
__forceinline__ __device__ void copy_mn(Tensor<Engine0, Layout0> &src,
                                        Tensor<Engine1, Layout1> &dst,
                                        Tensor<Engine2, Layout2> &identity_MN,
                                        TiledCopy tiled_copy,
                                        const int max_M, const int max_N) {
    #pragma unroll
    for (int m = 0; m < size<1>(identity_MN); ++m) {
        if (get<0>(identity_MN(0, m, 0)) < max_M) {
            #pragma unroll
            for (int n = 0; n < size<2>(identity_MN); ++n) {
                if (get<1>(identity_MN(0, 0, n)) < max_N) {
                    cute::copy(tiled_copy, src(_, m, n), dst(_, m, n));
                }
            }
        }
    }

}*/

template <bool Is_even_MN, int kBlockM, int kBlockN>
struct AttnBias {

    const int max_seqlen_n, max_seqlen_m;

    __forceinline__ __device__ AttnBias(const int max_seqlen_n, const int max_seqlen_m)
        : max_seqlen_n(max_seqlen_n)
        , max_seqlen_m(max_seqlen_m) {
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
                        if (Is_even_MN || (col_idx < max_seqlen_n && row_idx < max_seqlen_m)) {
                            tensor(make_coord(i, mi), make_coord(j, nj)) += bias(make_coord(i, mi), make_coord(j, nj)) / softmax_scale;
                        }
                    }
                }
            }
        }
    }

    /*template <class Shape, class Stride>
    __device__ void print3D(Layout<Shape,Stride> const& layout)
    {
        for (int k = 0; k < size<0>(layout); ++k) {
            for (int m = 0; m < size<1>(layout); ++m) {
                for (int n = 0; n < size<2>(layout); ++n) {
                    printf("%3d  ", (int)layout(k,m,n));
                }
                printf(" | ");
            }
            printf("\n");
        }
    }*/

    template<typename Tensor0, typename Tensor1, typename Tensor2,
         typename TiledCopy, typename ThrCopy>
    __forceinline__ __device__ void copy_ds(Tensor0 &tBgdS, Tensor1 &tBsdS, Tensor2 &cdS,
                                TiledCopy gmem_tiled_copy_dS, ThrCopy gmem_thr_copy_dS,
                                const int m_block, const int n_block,
                                const bool use_atomic_add) {

        if (Is_even_MN) {
            if (use_atomic_add) {
                // idk why the following return a memory error when KBlockM != kBlockN
                #pragma unroll
                for (int i = 0; i < size(tBgdS); i=i+2) {
                    atomicAdd2(&tBgdS(i), &tBsdS(i));
                }
            } else {
                cute::copy(gmem_tiled_copy_dS, tBsdS, tBgdS);
            }
        } else { // Is_even_MN
            Tensor tdScdS = gmem_thr_copy_dS.partition_D(cdS);

            if (use_atomic_add) {
                #pragma unroll
                for (int m = 0; m < size<1>(tdScdS); ++m) {
                    if (get<0>(tdScdS(0, m, 0)) < max_seqlen_m - m_block * kBlockM) {
                        #pragma unroll
                        for (int n = 0; n < size<2>(tdScdS); ++n) {
                            if (get<1>(tdScdS(0, 0, n)) < max_seqlen_n - n_block * kBlockN) {
                                #pragma unroll
                                for (int k = 0; k < size<0>(tdScdS); k=k+2) {
                                    atomicAdd2(&tBgdS(k, m, n), &tBsdS(k, m, n));
                                }
                            }
                        }
                    }
                }
            } else {
                #pragma unroll
                for (int m = 0; m < size<1>(tdScdS); ++m) {
                    if (get<0>(tdScdS(0, m, 0)) < max_seqlen_m - m_block * kBlockM) {
                        #pragma unroll
                        for (int n = 0; n < size<2>(tdScdS); ++n) {
                            if (get<1>(tdScdS(0, 0, n)) < max_seqlen_n - n_block * kBlockN) {
                                cute::copy(gmem_tiled_copy_dS, tBsdS(_, m, n), tBgdS(_, m, n));
                            }
                        }
                    }
                }
            }
        } // Is_even_MN
    }
};

} // namespace flash
