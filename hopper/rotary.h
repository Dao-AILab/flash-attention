/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar,
 *Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cute/tensor.hpp>

#include "utils.h"

namespace flash {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Engine1, typename Layout1, typename Engine2, typename Layout2, typename Engine3, typename Layout3>
CUTLASS_DEVICE void
apply_rotary_interleaved(Tensor<Engine1, Layout1> &tKrK,
                         Tensor<Engine2, Layout2> const &tRrCos,
                         Tensor<Engine2, Layout2> const &tRrSin,
                         Tensor<Engine3, Layout3> const &tKcK,
                         int const max_MN, int const rotary_dim) {
    CUTE_STATIC_ASSERT_V(rank(tKrK) == _3{});
    CUTE_STATIC_ASSERT_V(rank(tRrCos) == _3{});
    CUTE_STATIC_ASSERT_V(rank(tRrSin) == _3{});
    CUTE_STATIC_ASSERT_V(size<1>(tKrK) == size<1>(tRrCos));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(tKrK) == size<2>(tRrCos));                     // MMA_K
    CUTE_STATIC_ASSERT_V(size<1>(tKrK) == size<1>(tRrSin));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(tKrK) == size<2>(tRrSin));                     // MMA_K
    CUTE_STATIC_ASSERT_V(size<0>(tRrCos) == size<0>(tRrSin));
    static_assert(decltype(size<0>(tKrK))::value == decltype(size<0>(tRrCos))::value * 2);
    static_assert(decltype(size<0>(tRrCos))::value % 2 == 0);  // Since we do fast conversion from fp16/bf16 to fp32
    Tensor tRpR = make_tensor<bool>(make_shape(size<2>(tKrK)));
    #pragma unroll
    for (int k = 0; k < size(tRpR); ++k) { tRpR(k) = get<1>(tKcK(_0{}, _0{}, k)) < rotary_dim; }
    #pragma unroll
    for (int m = 0; m < size<1>(tKrK); ++m) {
        if (get<0>(tKcK(_0{}, m, _0{})) < max_MN) {
            #pragma unroll
            for (int k = 0; k < size<2>(tKrK); ++k) {
                if (tRpR(k)) {
                    Tensor K_fp32 = convert_type_safe<float>(tKrK(_, m, k));
                    Tensor cos_fp32 = convert_type_safe<float>(tRrCos(_, m, k));
                    Tensor sin_fp32 = convert_type_safe<float>(tRrSin(_, m, k));
                    #pragma unroll
                    for (int i = 0; i < size<0>(tKrK) / 2; ++i) {
                        // if (sin_fp32(i) != 0.f) { printf("tidx = %d, bidx = %d, bidy = %d, bidz = %d, sin_fp32(i) = %f, m = %d, k = %d,  i =%d\n", threadIdx.x, blockIdx.x, blockIdx.y, blockIdx.z, sin_fp32(i), m, k, i);}
                        float real = K_fp32[2 * i] * cos_fp32[i] - K_fp32[2 * i + 1] * sin_fp32[i];
                        float imag = K_fp32[2 * i] * sin_fp32[i] + K_fp32[2 * i + 1] * cos_fp32[i];
                        K_fp32[2 * i] = real;
                        K_fp32[2 * i + 1] = imag;
                    }
                    cute::copy(convert_type_safe<Engine1::value_type>(K_fp32), tKrK(_, m, k));
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////


} // namespace flash
