/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cmath>

#include <cute/tensor.hpp>

#include <cutlass/numeric_types.h>

#include "utils.h"

#include "cutlass/fast_math.h"

namespace flash {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void thread_reduce_(Tensor<Engine0, Layout0> const &tensor, Tensor<Engine1, Layout1> &summary, Operator &op) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(summary) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); mi++) {
        summary(mi) = zero_init ? tensor(mi, 0) : op(summary(mi), tensor(mi, 0));
        #pragma unroll
        for (int ni = 1; ni < size<1>(tensor); ni++) {
            summary(mi) = op(summary(mi), tensor(mi, ni));
        }
    }
}

template<typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void quad_allreduce_(Tensor<Engine0, Layout0> &dst, Tensor<Engine1, Layout1> &src, Operator &op) {
    CUTE_STATIC_ASSERT_V(size(dst) == size(src));
    #pragma unroll
    for (int i = 0; i < size(dst); i++){
        dst(i) = Allreduce<4>::run(src(i), op);
    }
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void reduce_(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &summary, Operator &op) {
    thread_reduce_<zero_init>(tensor, summary, op);
    quad_allreduce_(summary, summary, op);
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void reduce_max(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &max){
    MaxOp<float> max_op;
    reduce_<zero_init>(tensor, max, max_op);
}

template<bool zero_init=true, bool warp_reduce=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void reduce_sum(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &sum){
    SumOp<float> sum_op;
    thread_reduce_<zero_init>(tensor, sum, sum_op);
    if constexpr (warp_reduce) { quad_allreduce_(sum, sum, sum_op); }
}

__forceinline__ __device__ __half2 half_exp(__half2 x) {
    uint32_t tmp_out, tmp_in;
    tmp_in = reinterpret_cast<uint32_t&>(x);
    asm ("ex2.approx.f16x2 %0, %1;\n"
      : "=r"(tmp_out)
      : "r"(tmp_in));
    __half2 out = reinterpret_cast<__half2&>(tmp_out);
    return out;
}

// Apply the exp to all the elements.
template <bool zero_init=false, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void max_scale_exp2_sum(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> &max, Tensor<Engine1, Layout1> &sum, const float scale) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor"); static_assert(Layout1::rank == 1, "Only support 1D Tensor"); CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        MaxOp<float> max_op;
        max(mi) = zero_init ? tensor(mi, 0) : max_op(max(mi), tensor(mi, 0));
        #pragma unroll
        for (int ni = 1; ni < size<1>(tensor); ni++) {
            max(mi) = max_op(max(mi), tensor(mi, ni));
        }
        max(mi) = Allreduce<4>::run(max(mi), max_op);
        // If max is -inf, then all elements must have been -inf (possibly due to masking).
        // We don't want (-inf - (-inf)) since that would give NaN.
        const float max_scaled = max(mi) == -INFINITY ? 0.f : max(mi) * scale;
        sum(mi) = 0;
        #pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ++ni)  {
            // Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
            // max * log_2(e)) This allows the compiler to use the ffma
            // instruction instead of fadd and fmul separately.
            tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
            sum(mi) += tensor(mi, ni);
        }
    }
}

// Apply the exp to all the elements.
template <bool Scale_max=true, bool Check_inf=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void scale_apply_exp2(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> const &max, const float scale) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        // If max is -inf, then all elements must have been -inf (possibly due to masking).
        // We don't want (-inf - (-inf)) since that would give NaN.
        // If we don't have float around M_LOG2E the multiplication is done in fp64.
        const float max_scaled = Check_inf
            ? (max(mi) == -INFINITY ? 0.f : (max(mi) * (Scale_max ? scale : float(M_LOG2E))))
            : (max(mi) * (Scale_max ? scale : float(M_LOG2E)));
        #pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ++ni)  {
            // Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
            // max * log_2(e)) This allows the compiler to use the ffma
            // instruction instead of fadd and fmul separately.
            tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int kNRows>
struct Softmax { 
    constexpr static float max_offset = 8.0f;

    using TensorT = decltype(make_tensor<float>(Shape<Int<kNRows>>{}));
    TensorT row_max, row_sum;

    CUTLASS_DEVICE Softmax() {};

    template<bool Is_first, bool Check_inf=false, typename Tensor0>
    __forceinline__ __device__ TensorT max(Tensor0 &acc_s, float softmax_scale_log2) {
        // Reshape acc_s from ((2, 2, V), MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, V, MMA_N))
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        static_assert(decltype(size<0>(scores))::value == kNRows);
        TensorT scores_scale;
        if constexpr (Is_first) {
            flash::template reduce_max</*zero_init=*/true>(scores, row_max);
            cute::fill(scores_scale, 1.f);
        } else {
            Tensor scores_max_prev = make_fragment_like(row_max);
            cute::copy(row_max, scores_max_prev);
            flash::template reduce_max</*zero_init=*/false>(scores, row_max);
            #pragma unroll
            for (int mi = 0; mi < size(row_max); ++mi) {
                float scores_max_cur = !Check_inf
                    ? row_max(mi)
                    : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));
                scores_scale(mi) = exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
                row_sum(mi) *= scores_scale(mi);
            }
        }
        return scores_scale;
    };

    template<bool Is_first, bool Check_inf=false, typename Tensor0>
    __forceinline__ __device__ TensorT online_softmax(Tensor0 &acc_s, float softmax_scale_log2) {
        // Reshape acc_s from ((2, 2, V), MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, V, MMA_N))
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        static_assert(decltype(size<0>(scores))::value == kNRows);
        TensorT scores_scale;
        if constexpr (Is_first) {
            flash::template reduce_max</*zero_init=*/true>(scores, row_max);
            flash::template scale_apply_exp2(scores, row_max, softmax_scale_log2);
            flash::reduce_sum</*zero_init=*/true, /*warp_reduce=*/false>(scores, row_sum);
            cute::fill(scores_scale, 1.f);
            // if (cute::thread0()) { print_tensor(scores); printf("\n scale = %f\n", softmax_scale_log2); print_tensor(row_sum); }
        } else {
            // Tensor scores_max_prev = make_fragment_like(row_max);
            // cute::copy(row_max, scores_max_prev);
            // flash::template reduce_max</*zero_init=*/false>(scores, row_max);
            // // if (cute::thread0()) { print_tensor(scores); printf("\n"); print_tensor(row_max); printf("\n"); }
            // #pragma unroll
            // for (int mi = 0; mi < size(row_max); ++mi) {
            //     float scores_max_cur = !Check_inf
            //         ? row_max(mi)
            //         : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));
            //     scores_scale(mi) = exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
            //     row_sum(mi) *= scores_scale(mi);
            // }
            flash::template scale_apply_exp2</*Scale_max=*/true, Check_inf>(scores, row_max, softmax_scale_log2);
            // We don't do the reduce across threads here since we don't need to use the row_sum.
            // We do that reduce at the end when we need to normalize the softmax.
            flash::reduce_sum</*zero_init=*/false, /*warp_reduce=*/false>(scores, row_sum);
        }
        return scores_scale;
    };
    
    template<bool Is_dropout=false, bool Split=false, typename Tensor0>
    __forceinline__ __device__ TensorT finalize(Tensor0 &acc_s, float softmax_scale_log2, float rp_dropout=1.0) {
        // Reshape acc_s from ((2, 2, V), MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, V, MMA_N))
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        static_assert(decltype(size<0>(scores))::value == kNRows);
        SumOp<float> sum_op;
        quad_allreduce_(row_sum, row_sum, sum_op);
        TensorT scores_scale;
        #pragma unroll
        for (int mi = 0; mi < size(row_max); ++mi) {
            float sum = row_sum(mi);
            float inv_sum = (sum == 0.f || sum != sum) ? 0.f : 1.f / sum;
            row_sum(mi) = (sum == 0.f || sum != sum) ? (Split ? -INFINITY : INFINITY) : row_max(mi) * (softmax_scale_log2 * float(M_LN2)) + __logf(sum);
            scores_scale(mi) = !Is_dropout ? inv_sum : inv_sum * rp_dropout;
        }
        return scores_scale;
    };

    template<typename Tensor1>
    __forceinline__ __device__ void rescale_o(Tensor1 &acc_o, TensorT const &scores_scale) {
        // Reshape acc_o from (MMA=4, MMA_M, MMA_K) to (nrow=(2, MMA_M), ncol=(2, MMA_K))
        Tensor acc_o_rowcol = make_tensor(acc_o.data(), flash::convert_layout_acc_rowcol(acc_o.layout()));
        static_assert(decltype(size<0>(acc_o_rowcol))::value == kNRows);
        #pragma unroll
        for (int mi = 0; mi < size(row_max); ++mi) {
            #pragma unroll
            for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scores_scale(mi); }
        }
    };

    // combined online softmax method with arbitrary predication
    template <bool is_first = false, bool check_infinity = false,
              typename FragmentS, typename FragmentO, typename PredicateFn>
    __forceinline__ __device__ void
    online_softmax_and_rescale_o(FragmentS &accum_s, FragmentO &accum_o,
        float softmax_scale_log2, const PredicateFn &predicateFn) {

        using FragValType = typename FragmentS::value_type;
        using FragValTypeO = typename FragmentO::value_type;
        using SoftType = float;
        
        auto VT = shape<0>(accum_s); // number of vector elements per tile, e.g. (2,2,X)
        auto MT = shape<1>(accum_s); // number of tiles along M.
        auto NT = shape<2>(accum_s); // number of tiles along N.
        static_assert(get<0>(VT) == 2);
        static_assert(get<1>(VT) == 2);
        static_assert(NT == 1, "We suppose KBLKSIZE <= 256.");

        auto VT_O = shape<0>(accum_o);
        auto MT_O = shape<1>(accum_o);
        auto KT = shape<2>(accum_o);
        static_assert(size<0>(VT_O) == 2);
        static_assert(size<1>(VT_O) == 2);
        static_assert(MT_O == MT);
        static_assert(KT == 1, "We suppose HEADDIM <= 256.");

        MaxOp<SoftType> maxOp;        

        auto data = accum_s.data();
        auto data_o = accum_o.data();

        auto maxPredicate = [&](SoftType &max_ref, const int &n) {
            if (!predicateFn(n))
                data[n] = -INFINITY;
            max_ref = cutlass::fast_max(max_ref, SoftType(data[n]));
        };

        auto scaleExpSum = [&](SoftType const &max_ref, SoftType &sum_ref,
                                const int &n) {
            data[n] = exp2f(SoftType(data[n]) * softmax_scale_log2 - max_ref);
            sum_ref += data[n];
        };

        const auto inverse_scale = SoftType(1.0) / softmax_scale_log2;

        CUTLASS_PRAGMA_UNROLL
        for (int rowIdx = 0; rowIdx < MT * 2; rowIdx += 2) {
            auto max0 = row_max(rowIdx) * inverse_scale;
            auto max1 = row_max(rowIdx + 1) * inverse_scale;

            // Traverse 2-rows + 2-cols (2x2) simultaneously.
            CUTLASS_PRAGMA_UNROLL
            for (int n = 0; n < 4 * NT * size<2>(VT); n += 4) {
                maxPredicate(max0, n);
                maxPredicate(max0, n + 1);
                maxPredicate(max1, n + 2);
                maxPredicate(max1, n + 3);
            }
            const auto max_quad_0 = Allreduce<4>::run(max0, maxOp) * softmax_scale_log2;
            const auto max_quad_1 = Allreduce<4>::run(max1, maxOp) * softmax_scale_log2;

            if constexpr (!is_first) {

                // Need to avoid NaN from subtracting -infinity
                // Can condition on previous row max
                const auto rescale0 = check_infinity ? ((row_max(rowIdx) == -INFINITY)
                                            ? SoftType(0.0)
                                            : exp2f(row_max(rowIdx) - max_quad_0))
                                            : exp2f(row_max(rowIdx) - max_quad_0);
                row_sum(rowIdx) *= rescale0;

                const auto rescale1 = check_infinity ? ((row_max(rowIdx + 1) == -INFINITY)
                                            ? SoftType(0.0)
                                            : exp2f(row_max(rowIdx + 1) - max_quad_1))
                                            : exp2f(row_max(rowIdx + 1) - max_quad_1);
                row_sum(rowIdx + 1) *= rescale1;

                CUTLASS_PRAGMA_UNROLL
                for (int no = 0; no < 4 * KT * size<2>(VT_O); no += 4) {
                    data_o[no] = FragValTypeO(SoftType(data_o[no]) * rescale0);
                    data_o[no + 1] = FragValTypeO(SoftType(data_o[no + 1]) * rescale0);
                    data_o[no + 2] = FragValTypeO(SoftType(data_o[no + 2]) * rescale1);
                    data_o[no + 3] = FragValTypeO(SoftType(data_o[no + 3]) * rescale1);
                }
            }

            row_max(rowIdx) = max_quad_0;
            row_max(rowIdx + 1) = max_quad_1;

            auto sum0 = SoftType(0.0);
            auto sum1 = SoftType(0.0);

            // Need to avoid NaN from subtracting -infinity
            const auto miRow0 = check_infinity ? (row_max(rowIdx) == -INFINITY
                                    ? SoftType(0.0)
                                    : row_max(rowIdx) - SoftType(max_offset))
                                    : row_max(rowIdx) - SoftType(max_offset);
            const auto miRow1 = check_infinity ? (row_max(rowIdx + 1) == -INFINITY
                                    ? SoftType(0.0)
                                    : row_max(rowIdx + 1) - SoftType(max_offset))
                                    : row_max(rowIdx + 1) - SoftType(max_offset);

            CUTLASS_PRAGMA_UNROLL
            for (int n = 0; n < 4 * NT * size<2>(VT); n += 4) {
                scaleExpSum(miRow0, sum0, n);
                scaleExpSum(miRow0, sum0, n + 1);
                scaleExpSum(miRow1, sum1, n + 2);
                scaleExpSum(miRow1, sum1, n + 3);
            }

            row_sum(rowIdx) += sum0;
            row_sum(rowIdx + 1) += sum1;            
        }
    }

    template <bool is_first = false, bool check_infinity = true,
              typename FragmentS, typename FragmentO>
    __forceinline__ __device__ void
    online_softmax_and_rescale_o(FragmentS &accum_s, FragmentO &accum_o,
                                 float softmax_scale_log2) {
        online_softmax_and_rescale_o<is_first, check_infinity>(
            accum_s, accum_o, softmax_scale_log2, TrivialPredTensor{});
    }

};

}  // namespace flash
