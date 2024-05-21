#include <cmath>

#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/array.h>

#include "utils.h"

namespace flash {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Is_causal, int kBlockM, int kBlockN, int kNThreads, bool store_bucket>
struct RPE {

    const int max_distance;
    const int total_num_buckets;
    const int num_heads;
    const float scale_softmax;

    __forceinline__ __device__ RPE(const int num_buckets, const int max_distance,
                                   const int num_heads, const float scale_softmax)
        : max_distance(max_distance)
        , total_num_buckets(num_buckets)
        , num_heads(num_heads)
        , scale_softmax(scale_softmax) {
    };

    __forceinline__ __device__ void load_rpe(float *grpe,
                                      float *srpe,
                                      const int m_block,
                                      int n_block,
                                      const int tidx,
                                      int *rel_bucket
                                      ) {
        const int num_buckets = Is_causal ? total_num_buckets : total_num_buckets / 2;
        const int max_exact = num_buckets / 2;
        const int rel_pos_base = n_block * kBlockN - m_block * kBlockM;

        #pragma unroll
        for(int i = 0; (tidx + i) < kBlockM + kBlockN - 1; i += kNThreads) {

            int relative_position = rel_pos_base + tidx + i - (kBlockM-1);
            int relative_bucket = 0;

            if constexpr (!Is_causal) {
                if (relative_position > 0) {
                    relative_bucket = relative_bucket + num_buckets;
                } else {
                    relative_position = abs(relative_position);
                }
            } else {
                relative_position = -min(relative_position, 0);
            }

            if (relative_position > max_exact) {
                const int log_bucket = max_exact + int(logf(float(relative_position) / max_exact) / logf(float(max_distance) / max_exact) * (num_buckets - max_exact));
                relative_bucket = relative_bucket + min(log_bucket, num_buckets - 1);
            } else {
                relative_bucket = relative_bucket + relative_position;
            }

            if (store_bucket) {
                rel_bucket[tidx + i] = relative_bucket;
            }

            srpe[tidx + i] = grpe[relative_bucket] / scale_softmax;
        }
    }

    template <typename Engine, typename Layout>
    __forceinline__ __device__ void apply_rpe(Tensor<Engine, Layout> &tensor,
                                      float *rpe_weights,
                                      const int n_block,
                                      int m_block,
                                      const int col_idx_offset_,
                                      const int row_idx_offset,
                                      const int warp_row_stride) {
        // tensor has shape (ncol=(2, MMA_M), nrow=(2, MMA_N))
        static_assert(Layout::rank == 2, "Only support 2D Tensor");
        const int lane_id = threadIdx.x % 32;
        const int col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;
        const int row_block_offset = n_block * kBlockN;
        const int col_block_offset = m_block * kBlockM;
        const int rpe_max_length = kBlockM + kBlockN - 1;
        const int rpe_offset = kBlockM - 1;

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
                        const int adj_relative_position = std::min((col_idx - col_block_offset) - (row_idx - row_block_offset) + rpe_offset, rpe_max_length);
                        tensor(make_coord(i, mi), make_coord(j, nj)) += rpe_weights[adj_relative_position];
                    }
                }
            }
        }
    }

    template <typename Engine, typename Layout>
    __forceinline__ __device__ void store_ds(Tensor<Engine, Layout> &tensor,
                                      float *gmem_drpe_weights,
                                      int *smem_relative_position,
                                      const int max_seqlen_k,
                                      const int max_seqlen_q,
                                      const int n_block,
                                      int m_block,
                                      const int col_idx_offset_,
                                      const int row_idx_offset,
                                      const int warp_row_stride) {
        // tensor has shape (ncol=(2, MMA_M), nrow=(2, MMA_N))
        static_assert(Layout::rank == 2, "Only support 2D Tensor");
        const int lane_id = threadIdx.x % 32;
        const int col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;
        const int row_block_offset = n_block * kBlockN;
        const int col_block_offset = m_block * kBlockM;
        const int rpe_offset = kBlockM - 1;
        const int rpe_max_length = kBlockM + kBlockN - 1;

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
                        if ((col_idx < max_seqlen_k) && (row_idx < max_seqlen_q)) {
                            const int adj_relative_position = std::min((col_idx - col_block_offset) - (row_idx - row_block_offset) + rpe_offset, rpe_max_length);
                            int bucket = smem_relative_position[adj_relative_position];
                            float ds_val = (float)tensor(make_coord(i, mi), make_coord(j, nj));

                            atomicAdd(gmem_drpe_weights + bucket, ds_val);
                        }
                    }
                }
            }
        }
    }

};

}  // namespace flash
