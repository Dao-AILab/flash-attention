#include <cmath>

#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/array.h>

#include "utils.h"

namespace flash {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Is_causal, int kBlockM, int kBlockN, int kNThreads>
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
                                      const int bidh,
                                      const int tidx
                                      ) {
        const int num_buckets = Is_causal ? total_num_buckets : total_num_buckets / 2;
        const int max_exact = num_buckets / 2;
        const int rel_pos_base = n_block * kBlockN - m_block * kBlockM;

        #pragma unroll
        for(int i = 0; i < kBlockM + kBlockN - 1; i += kNThreads) {

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

            srpe[tidx + i] = grpe[relative_bucket] / scale_softmax;
        }
    }

};

}  // namespace flash
