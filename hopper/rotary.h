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

template <int kBlockMN, int kHeadDim, int NumThreads, typename Element, bool FixedPosition=false>
struct Rotary {

    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "Headdim must be a multiple of kGmemElemsPerLoad");
    // We want each "row" to have 64 elements (128 bytes, i.e. 1 cache line). E.g. if hdim=128, we want each
    // thread to have 4 loads in the M direction and 2 vectorized load in the K direction.
    // In the case of PackGQA, this reduces the number of times we need to call divmod.
    static constexpr int kBlockKGmem = kHeadDim % 128 == 0 ? (sizeof(Element) == 2 ? 64 : 128) : (kHeadDim % 64 == 0 ? 64 : 32);
    static constexpr int kGmemThreadsPerRow = kBlockKGmem / kGmemElemsPerLoad;
    static_assert(NumThreads % kGmemThreadsPerRow == 0, "NumThreads must be a multiple of kGmemThreadsPerRow");
    // We assume threads loading the same row are in the same warp. This is for an optimization in PagedKV where
    // these threads share the same page table entry and share the work of computing pointers to paged K and paged V.
    static_assert(cutlass::NumThreadsPerWarp % kGmemThreadsPerRow == 0, "kGmemThreadsPerRow must divide NumThreadsPerWarp");
    using GmemLayoutAtom = Layout<Shape <Int<NumThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                  Stride<Int<kGmemThreadsPerRow>, _1>>;
    using GmemTiledCopyQK = decltype(
        make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));  // Val layout, 8 or 16 vals per store
    using GmemTiledCopyRotary = decltype(
        make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<64>, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoad / 2>>>{}));  // Val layout, 4 or 8 vals per store

    using ShapeRotary = cute::Shape<int32_t, int32_t>;  // (seqlen_ro, rotary_dim // 2)
    using StrideRotary = cute::Stride<int64_t, _1>;

    using GmemThrCopyRotary = decltype(GmemTiledCopyRotary{}.get_thread_slice(int(0)));
    using TensortRcR = decltype(GmemTiledCopyRotary{}.get_thread_slice(int(0)).partition_D(cute::make_identity_tensor(Shape<Int<kBlockMN>, Int<kHeadDim / 2>>{})));
    using TensortRpR = decltype(make_tensor<bool>(make_shape(size<2>(TensortRcR{}))));
    using TensormR = decltype(make_tensor(
        make_gmem_ptr((Element const*)nullptr),
        ShapeRotary{},
        make_stride(cute::conditional_return<FixedPosition>(_0{}, int64_t(0)), _1{})));
    using TensortRgR = decltype(
        GmemTiledCopyRotary{}.get_thread_slice(int(0)).partition_S(make_tensor(
            make_gmem_ptr((Element const*)nullptr),
            make_shape(Int<kBlockMN>{}, Int<kHeadDim / 2>{}, int(0)),
            make_stride(cute::conditional_return<FixedPosition>(_0{}, int64_t(0)), _1{}, cute::conditional_return<FixedPosition>(_0{}, int64_t(0))))));

    GmemTiledCopyRotary gmem_tiled_copy_rotary;
    int const rotary_dim;
    int const thread_idx;
    int const max_seqlen;
    GmemThrCopyRotary const gmem_thr_copy_rotary;
    TensortRpR tRpR;
    TensormR mCos, mSin;
    TensortRgR tRgCos, tRgSin;

    CUTLASS_DEVICE
    Rotary(Element const* const ptr_rotary_cos, ShapeRotary const &shape_rotary, StrideRotary const &stride_rotary_cos_,
           Element const* const ptr_rotary_sin, StrideRotary const &stride_rotary_sin_,
           int const thread_idx, int const max_seqlen, int const start_idx)
        : rotary_dim(get<1>(shape_rotary) * 2)
        , thread_idx(thread_idx)
        , max_seqlen(max_seqlen)
        , gmem_thr_copy_rotary(gmem_tiled_copy_rotary.get_thread_slice(thread_idx))

    {
        auto stride_rotary_cos = make_stride(cute::conditional_return<!FixedPosition>(get<0>(stride_rotary_cos_), _0{}), get<1>(stride_rotary_cos_));
        auto stride_rotary_sin = make_stride(cute::conditional_return<!FixedPosition>(get<0>(stride_rotary_sin_), _0{}), get<1>(stride_rotary_sin_));
        mCos = make_tensor(make_gmem_ptr(ptr_rotary_cos + start_idx * get<0>(stride_rotary_cos_)), shape_rotary, stride_rotary_cos);
        mSin = make_tensor(make_gmem_ptr(ptr_rotary_sin + start_idx * get<0>(stride_rotary_sin_)), shape_rotary, stride_rotary_sin);

        Tensor gCos = local_tile(mCos, Shape<Int<kBlockMN>, Int<kHeadDim / 2>>{}, make_coord(_, _0{}));  // (MN, K / 2, _)
        Tensor gSin = local_tile(mSin, Shape<Int<kBlockMN>, Int<kHeadDim / 2>>{}, make_coord(_, _0{}));  // (MN, K / 2, _)

        tRgCos = gmem_thr_copy_rotary.partition_S(gCos);
        tRgSin = gmem_thr_copy_rotary.partition_S(gSin);
        Tensor cR = cute::make_identity_tensor(Shape<Int<kBlockMN>, Int<kHeadDim / 2>>{});  // (BLK_N,BLK_K / 2)
        Tensor tRcR = gmem_thr_copy_rotary.partition_D(cR);
        tRpR = make_tensor<bool>(make_shape(size<2>(tRgCos)));
        #pragma unroll
        for (int k = 0; k < size(tRpR); ++k) { tRpR(k) = get<1>(tRcR(_0{}, _0{}, k)) < get<1>(shape_rotary); }
    };

    CUTLASS_DEVICE
    auto load_cos_sin(int const block) {
        Tensor cR = cute::make_identity_tensor(Shape<Int<kBlockMN>, Int<kHeadDim / 2>>{});  // (BLK_N,BLK_K / 2)
        Tensor tRcR = gmem_thr_copy_rotary.partition_D(cR);
        // make_tensor_like, not make_fragment_like. If the row_stride is _0{} we want to keep it that way
        Tensor tRrCos = make_tensor_like(tRgCos(_, _, _, _0{}));
        Tensor tRrSin = make_tensor_like(tRgSin(_, _, _, _0{}));
        if (rotary_dim <= 0) { return std::make_pair(tRrCos, tRrSin); }
        // If FixedPosition, only copy the first row as we only need the cos/sin for position cache_seqlens
        #pragma unroll
        for (int m = 0; m < (!FixedPosition ? size<1>(tRgCos) : 1); ++m) {
            if (get<0>(tRcR(_0{}, m, _0{})) < max_seqlen - block * kBlockMN) {
                #pragma unroll
                for (int k = 0; k < size<2>(tRgCos); ++k) {
                    if (tRpR(k)) {
                        cute::copy(gmem_tiled_copy_rotary, tRgCos(_, m, k, block), tRrCos(_, m, k));
                        cute::copy(gmem_tiled_copy_rotary, tRgSin(_, m, k, block), tRrSin(_, m, k));
                    }
                }
            }
        }
        return std::make_pair(tRrCos, tRrSin);
    }

    CUTLASS_DEVICE
    auto load_cos_sin_packgqa(int const block, cutlass::FastDivmod const &qhead_per_khead_divmod) {
        Tensor cR = cute::make_identity_tensor(Shape<Int<kBlockMN>, Int<kHeadDim / 2>>{});  // (BLK_N,BLK_K / 2)
        Tensor tRcR = gmem_thr_copy_rotary.partition_D(cR);
        // make_tensor_like, not make_fragment_like. If the row_stride is _0{} we want to keep it that way
        Tensor tRrCos = make_tensor_like(tRgCos(_, _, _, _0{}));
        Tensor tRrSin = make_tensor_like(tRgSin(_, _, _, _0{}));
        if (rotary_dim <= 0) { return std::make_pair(tRrCos, tRrSin); }
        int const qhead_per_khead = qhead_per_khead_divmod.divisor;
        #pragma unroll
        for (int m = 0; m < (!FixedPosition ? size<1>(tRgCos) : 1); ++m) {
            int const idx = block * kBlockMN + get<0>(tRcR(_0{}, m, _0{}));
            if (idx < max_seqlen * qhead_per_khead) {
                int const row = qhead_per_khead_divmod.divide(idx);
                Tensor mCos_copy = cute::tiled_divide(make_tensor(&mCos(row, _0{}), Shape<Int<kHeadDim / 2>>{}),
                                                      Shape<Int<kGmemElemsPerLoad / 2>>{});
                Tensor mSin_copy = cute::tiled_divide(make_tensor(&mSin(row, _0{}), Shape<Int<kHeadDim / 2>>{}),
                                                      Shape<Int<kGmemElemsPerLoad / 2>>{});
                #pragma unroll
                for (int k = 0; k < size<2>(tRgCos); ++k) {
                    int const ki = get<1>(tRcR(_0{}, _0{}, k)) / (kGmemElemsPerLoad / 2);
                    if (tRpR(k)) {
                        cute::copy(gmem_tiled_copy_rotary, mCos_copy(_, ki), tRrCos(_, m, k));
                        cute::copy(gmem_tiled_copy_rotary, mSin_copy(_, ki), tRrSin(_, m, k));
                    }
                }
            }
        }
        return std::make_pair(tRrCos, tRrSin);
    }

    template <typename TensorsQ, typename TensortRrR>
    CUTLASS_DEVICE
    void
    apply_Q(TensorsQ &sQ,  // (kBlockM, kHeadDim)
            TensortRrR const &tRrCos,   // (kBlockM, kHeadDim / 2) split across threads according to GmemThrCopyRotary
            TensortRrR const &tRrSin,   // (kBlockM, kHeadDim / 2) split across threads according to GmemThrCopyRotary
            int const m_block, int const qhead_per_khead=1)
    {
        if (rotary_dim <= 0) { return; }

        GmemTiledCopyQK gmem_tiled_copy_q;
        auto gmem_thr_copy_q = gmem_tiled_copy_q.get_thread_slice(thread_idx);
        Tensor tQsQ = gmem_thr_copy_q.partition_S(sQ);
        Tensor tQcQ = gmem_thr_copy_q.partition_S(cute::make_identity_tensor(Shape<Int<kBlockMN>, Int<kHeadDim>>{}));
        Tensor tQrQ = make_fragment_like(tQsQ);

        CUTE_STATIC_ASSERT_V(rank(tQrQ) == _3{});
        CUTE_STATIC_ASSERT_V(rank(tRrCos) == _3{});
        CUTE_STATIC_ASSERT_V(rank(tRrSin) == _3{});
        CUTE_STATIC_ASSERT_V(size<1>(tQrQ) == size<1>(tRrCos));
        CUTE_STATIC_ASSERT_V(size<2>(tQrQ) == size<2>(tRrCos));
        CUTE_STATIC_ASSERT_V(size<1>(tQrQ) == size<1>(tRrSin));
        CUTE_STATIC_ASSERT_V(size<2>(tQrQ) == size<2>(tRrSin));
        CUTE_STATIC_ASSERT_V(size<0>(tRrCos) == size<0>(tRrSin));
        static_assert(decltype(size<0>(tQrQ))::value == decltype(size<0>(tRrCos))::value * 2);
        static_assert(decltype(size<0>(tRrCos))::value % 2 == 0);  // Since we do fast conversion from fp16/bf16 to fp32

        #pragma unroll
        for (int m = 0; m < size<1>(tQsQ); ++m) {
            if (get<0>(tQcQ(_0{}, m, _0{})) < std::min(max_seqlen * qhead_per_khead - m_block * kBlockMN, kBlockMN)) {
                #pragma unroll
                for (int k = 0; k < size<2>(tQsQ); ++k) {
                    if (tRpR(k)) {
                        cute::copy(gmem_tiled_copy_q, tQsQ(_, m, k), tQrQ(_, m, k));
                        Tensor Q_fp32 = convert_type_safe<float>(tQrQ(_, m, k));
                        Tensor cos_fp32 = convert_type_safe<float>(tRrCos(_, m, k));
                        Tensor sin_fp32 = convert_type_safe<float>(tRrSin(_, m, k));
                        #pragma unroll
                        for (int i = 0; i < size<0>(tQsQ) / 2; ++i) {
                            float real = Q_fp32[2 * i] * cos_fp32[i] - Q_fp32[2 * i + 1] * sin_fp32[i];
                            float imag = Q_fp32[2 * i] * sin_fp32[i] + Q_fp32[2 * i + 1] * cos_fp32[i];
                            Q_fp32[2 * i] = real;
                            Q_fp32[2 * i + 1] = imag;
                        }
                        cute::copy(gmem_tiled_copy_q, convert_type_safe<TensorsQ::value_type>(Q_fp32), tQsQ(_, m, k));
                    }
                }
            }
        }
    };


};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace flash
