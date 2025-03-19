/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cute/tensor.hpp>

#include "utils.h"

namespace flash {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Engine1, typename Layout1, typename Engine2, typename Layout2>
CUTLASS_DEVICE void
apply_rotary_interleaved(Tensor<Engine1, Layout1> &rK,
                         Tensor<Engine2, Layout2> const &rCos,
                         Tensor<Engine2, Layout2> const &rSin) {
    CUTE_STATIC_ASSERT_V(rank(rK) == _1{});
    CUTE_STATIC_ASSERT_V(rank(rCos) == _1{});
    CUTE_STATIC_ASSERT_V(rank(rSin) == _1{});
    CUTE_STATIC_ASSERT_V(size<0>(rCos) == size<0>(rSin));
    static_assert(decltype(size<0>(rK))::value == decltype(size<0>(rCos))::value * 2);
    static_assert(decltype(size<0>(rCos))::value % 2 == 0);  // Since we do fast conversion from fp16/bf16 to fp32
    Tensor K_fp32 = make_tensor_like<float>(rK);
    convert_type_out(rK, K_fp32);
    Tensor cos_fp32 = make_tensor_like<float>(rCos);
    convert_type_out(rCos, cos_fp32);
    Tensor sin_fp32 = make_tensor_like<float>(rSin);
    convert_type_out(rSin, sin_fp32);
    #pragma unroll
    for (int i = 0; i < size<0>(K_fp32) / 2; ++i) {
        float real = K_fp32[2 * i] * cos_fp32[i] - K_fp32[2 * i + 1] * sin_fp32[i];
        float imag = K_fp32[2 * i] * sin_fp32[i] + K_fp32[2 * i + 1] * cos_fp32[i];
        K_fp32[2 * i] = real;
        K_fp32[2 * i + 1] = imag;
    }
    convert_type_out(K_fp32, rK);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Engine1, typename Layout1, typename Engine2, typename Layout2>
CUTLASS_DEVICE void
apply_rotary_contiguous(Tensor<Engine1, Layout1> &rK_left,
                        Tensor<Engine1, Layout1> &rK_right,
                        Tensor<Engine2, Layout2> const &rCos,
                        Tensor<Engine2, Layout2> const &rSin) {
    CUTE_STATIC_ASSERT_V(rank(rK_left) == _1{});
    CUTE_STATIC_ASSERT_V(rank(rK_right) == _1{});
    CUTE_STATIC_ASSERT_V(rank(rCos) == _1{});
    CUTE_STATIC_ASSERT_V(rank(rSin) == _1{});
    CUTE_STATIC_ASSERT_V(size<0>(rK_left) == size<0>(rK_right));
    CUTE_STATIC_ASSERT_V(size<0>(rK_left) == size<0>(rCos));
    CUTE_STATIC_ASSERT_V(size<0>(rCos) == size<0>(rSin));
    static_assert(decltype(size<0>(rCos))::value % 2 == 0);  // Since we do fast conversion from fp16/bf16 to fp32
    Tensor K_left_fp32 = make_tensor_like<float>(rK_left);
    convert_type_out(rK_left, K_left_fp32);
    Tensor K_right_fp32 = make_tensor_like<float>(rK_right);
    convert_type_out(rK_right, K_right_fp32);
    Tensor cos_fp32 = make_tensor_like<float>(rCos);
    convert_type_out(rCos, cos_fp32);
    Tensor sin_fp32 = make_tensor_like<float>(rSin);
    convert_type_out(rSin, sin_fp32);
    #pragma unroll
    for (int i = 0; i < size<0>(K_left_fp32); ++i) {
        float real = K_left_fp32[i] * cos_fp32[i] - K_right_fp32[i] * sin_fp32[i];
        float imag = K_left_fp32[i] * sin_fp32[i] + K_right_fp32[i] * cos_fp32[i];
        K_left_fp32[i] = real;
        K_right_fp32[i] = imag;
    }
    convert_type_out(K_left_fp32, rK_left);
    convert_type_out(K_right_fp32, rK_right);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int kBlockMN, int kHeadDim, int NumThreads, typename Element, bool FixedPosition=false>
struct Rotary {

    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "Headdim must be a multiple of kGmemElemsPerLoad");
    // We want each "row" to have 64 elements (128 bytes, i.e. 1 cache line). E.g. if hdim=128, we want each
    // thread to have 4 loads in the M direction and 2 vectorized load in the K direction.
    // We want each thread to have at least 2 loads in the K direction since in the case of non-interleaved
    // rotary (combining elements at indices 0 and rotary_dim/2, 1 and rotary_dim/2+1, etc), each thread will
    // load twice from the same row.
    static constexpr int kBytePerHalfRow = kHeadDim / 2 * sizeof(Element);
    static constexpr int kBlockKGmem = (kBytePerHalfRow % 128 == 0 ? 128 : (kBytePerHalfRow % 64 == 0 ? 64 : 32)) / sizeof(Element);
    static constexpr int kGmemThreadsPerRow = kBlockKGmem / kGmemElemsPerLoad;
    static_assert(NumThreads % kGmemThreadsPerRow == 0, "NumThreads must be a multiple of kGmemThreadsPerRow");
    // We assume threads loading the same row are in the same warp.
    static_assert(cutlass::NumThreadsPerWarp % kGmemThreadsPerRow == 0, "kGmemThreadsPerRow must divide NumThreadsPerWarp");

    using LayoutAtom = Layout<Shape <Int<NumThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                  Stride<Int<kGmemThreadsPerRow>, _1>>;
    using TiledCopyQK = decltype(
        make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
                        LayoutAtom{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));  // Val layout, 8 or 16 vals per store
    using GmemTiledCopyRotary = decltype(
        make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<64>, Element>{},
                        LayoutAtom{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoad / 2>>>{}));  // Val layout, 4 or 8 vals per store
    using GmemTiledCopyRotaryCont = decltype(
        make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
                        LayoutAtom{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));  // Val layout, 8 or 16 vals per store

    using ShapeRotary = cute::Shape<int32_t, int32_t>;  // (seqlen_ro, rotary_dim // 2)
    using StrideRotary = cute::Stride<int64_t, _1>;

    using GmemThrCopyRotary = decltype(GmemTiledCopyRotary{}.get_thread_slice(int(0)));
    using GmemThrCopyRotaryCont = decltype(GmemTiledCopyRotaryCont{}.get_thread_slice(int(0)));
    using TensortRcR = decltype(GmemTiledCopyRotary{}.get_thread_slice(int(0)).partition_D(cute::make_identity_tensor(Shape<Int<kBlockMN>, Int<kHeadDim / 2>>{})));
    using TensortRpR = decltype(make_tensor<bool>(make_shape(size<2>(TensortRcR{}))));
    using TensortRcRCont = decltype(GmemTiledCopyRotaryCont{}.get_thread_slice(int(0)).partition_D(cute::make_identity_tensor(Shape<Int<kBlockMN>, Int<kHeadDim / 2>>{})));
    using TensortRpRCont = decltype(make_tensor<bool>(make_shape(size<2>(TensortRcRCont{}))));
    using TensormR = decltype(make_tensor(
        make_gmem_ptr((Element const*)nullptr),
        ShapeRotary{},
        make_stride(cute::conditional_return<FixedPosition>(_0{}, int64_t(0)), _1{})));
    using TensortRgR = decltype(
        GmemTiledCopyRotary{}.get_thread_slice(int(0)).partition_S(make_tensor(
            make_gmem_ptr((Element const*)nullptr),
            make_shape(Int<kBlockMN>{}, Int<kHeadDim / 2>{}, int(0)),
            make_stride(cute::conditional_return<FixedPosition>(_0{}, int64_t(0)), _1{}, cute::conditional_return<FixedPosition>(_0{}, int64_t(0))))));
    using TensortRgRCont = decltype(
        GmemTiledCopyRotaryCont{}.get_thread_slice(int(0)).partition_S(make_tensor(
            make_gmem_ptr((Element const*)nullptr),
            make_shape(Int<kBlockMN>{}, Int<kHeadDim / 2>{}, int(0)),
            make_stride(cute::conditional_return<FixedPosition>(_0{}, int64_t(0)), _1{}, cute::conditional_return<FixedPosition>(_0{}, int64_t(0))))));

    GmemTiledCopyRotary gmem_tiled_copy_rotary;
    GmemTiledCopyRotaryCont gmem_tiled_copy_rotary_cont;
    bool const is_rotary_interleaved;
    int const rotary_dim;
    int const thread_idx;
    int const max_seqlen;
    GmemThrCopyRotary const gmem_thr_copy_rotary;
    GmemThrCopyRotaryCont const gmem_thr_copy_rotary_cont;
    TensortRpR tRpR;
    TensortRpRCont tRpRCont;
    TensormR mCos, mSin;
    TensortRgR tRgCos, tRgSin;
    TensortRgRCont tRgCosCont, tRgSinCont;

    CUTLASS_DEVICE
    Rotary(Element const* const ptr_rotary_cos, ShapeRotary const &shape_rotary, StrideRotary const &stride_rotary_cos_,
           Element const* const ptr_rotary_sin, StrideRotary const &stride_rotary_sin_,
           bool const is_rotary_interleaved, int const thread_idx, int const max_seqlen, int const start_idx)
        : is_rotary_interleaved(is_rotary_interleaved)
        , rotary_dim(get<1>(shape_rotary) * 2)
        , thread_idx(thread_idx)
        , max_seqlen(max_seqlen)
        , gmem_thr_copy_rotary(gmem_tiled_copy_rotary.get_thread_slice(thread_idx))
        , gmem_thr_copy_rotary_cont(gmem_tiled_copy_rotary_cont.get_thread_slice(thread_idx))

    {
        auto stride_rotary_cos = make_stride(cute::conditional_return<!FixedPosition>(get<0>(stride_rotary_cos_), _0{}), get<1>(stride_rotary_cos_));
        auto stride_rotary_sin = make_stride(cute::conditional_return<!FixedPosition>(get<0>(stride_rotary_sin_), _0{}), get<1>(stride_rotary_sin_));
        mCos = make_tensor(make_gmem_ptr(ptr_rotary_cos + start_idx * get<0>(stride_rotary_cos_)), shape_rotary, stride_rotary_cos);
        mSin = make_tensor(make_gmem_ptr(ptr_rotary_sin + start_idx * get<0>(stride_rotary_sin_)), shape_rotary, stride_rotary_sin);
        Tensor gCos = local_tile(mCos, Shape<Int<kBlockMN>, Int<kHeadDim / 2>>{}, make_coord(_, _0{}));  // (MN, K / 2, _)
        Tensor gSin = local_tile(mSin, Shape<Int<kBlockMN>, Int<kHeadDim / 2>>{}, make_coord(_, _0{}));  // (MN, K / 2, _)
        tRgCos = gmem_thr_copy_rotary.partition_S(gCos);
        tRgSin = gmem_thr_copy_rotary.partition_S(gSin);
        tRgCosCont = gmem_thr_copy_rotary_cont.partition_S(gCos);
        tRgSinCont = gmem_thr_copy_rotary_cont.partition_S(gSin);
        Tensor cR = cute::make_identity_tensor(Shape<Int<kBlockMN>, Int<kHeadDim / 2>>{});  // (BLK_N,BLK_K / 2)
        Tensor tRcR = gmem_thr_copy_rotary.partition_D(cR);
        tRpR = make_tensor<bool>(make_shape(size<2>(tRcR)));
        #pragma unroll
        for (int k = 0; k < size(tRpR); ++k) { tRpR(k) = get<1>(tRcR(_0{}, _0{}, k)) < get<1>(shape_rotary); }
        Tensor tRcRCont = gmem_thr_copy_rotary_cont.partition_D(cR);
        tRpRCont = make_tensor<bool>(make_shape(size<2>(tRcRCont)));
        #pragma unroll
        for (int k = 0; k < size(tRpRCont); ++k) { tRpRCont(k) = get<1>(tRcRCont(_0{}, _0{}, k)) < get<1>(shape_rotary); }
    };

    template <bool kInterleaved=true>
    CUTLASS_DEVICE
    auto load_cos_sin(int const block) {
        using GmemTiledCopyRo = std::conditional_t<kInterleaved, GmemTiledCopyRotary, GmemTiledCopyRotaryCont>;
        auto gmem_thr_copy_ro = cute::conditional_return<kInterleaved>(gmem_thr_copy_rotary, gmem_thr_copy_rotary_cont);
        Tensor tRpRCur = cute::conditional_return<kInterleaved>(tRpR, tRpRCont);
        Tensor tRgCosCur = cute::conditional_return<kInterleaved>(tRgCos, tRgCosCont)(_, _, _, block);
        Tensor tRgSinCur = cute::conditional_return<kInterleaved>(tRgSin, tRgSinCont)(_, _, _, block);
        // make_tensor_like, not make_fragment_like. If the row_stride is _0{} we want to keep it that way
        Tensor tRrCos = make_tensor_like(tRgCosCur);
        Tensor tRrSin = make_tensor_like(tRgSinCur);
        Tensor cR = cute::make_identity_tensor(Shape<Int<kBlockMN>, Int<kHeadDim / 2>>{});  // (BLK_N,BLK_K / 2)
        Tensor tRcR = gmem_thr_copy_ro.partition_D(cR);
        // If FixedPosition, only copy the first row as we only need the cos/sin for position cache_seqlens
        #pragma unroll
        for (int m = 0; m < (!FixedPosition ? size<1>(tRrCos) : 1); ++m) {
            if (get<0>(tRcR(_0{}, m, _0{})) < std::min(max_seqlen - block * kBlockMN, kBlockMN)) {
                #pragma unroll
                for (int k = 0; k < size<2>(tRrCos); ++k) {
                    if (tRpRCur(k)) {
                        cute::copy(GmemTiledCopyRo{}, tRgCosCur(_, m, k), tRrCos(_, m, k));
                        cute::copy(GmemTiledCopyRo{}, tRgSinCur(_, m, k), tRrSin(_, m, k));
                    }
                }
            }
        }
        return cute::make_tuple(tRrCos, tRrSin);;
    }

    template <bool kInterleaved=true>
    CUTLASS_DEVICE
    auto load_cos_sin_packgqa(int const block, cutlass::FastDivmod const &qhead_per_khead_divmod) {
        static constexpr int kGmemElemsPerLoadCur = kInterleaved ? kGmemElemsPerLoad / 2 : kGmemElemsPerLoad;
        using GmemTiledCopyRo = std::conditional_t<kInterleaved, GmemTiledCopyRotary, GmemTiledCopyRotaryCont>;
        auto gmem_thr_copy_ro = cute::conditional_return<kInterleaved>(gmem_thr_copy_rotary, gmem_thr_copy_rotary_cont);
        Tensor tRpRCur = cute::conditional_return<kInterleaved>(tRpR, tRpRCont);
        // make_tensor_like, not make_fragment_like. If the row_stride is _0{} we want to keep it that way
        Tensor tRrCos = make_tensor_like(cute::conditional_return<kInterleaved>(tRgCos, tRgCosCont)(_, _, _, _0{}));
        Tensor tRrSin = make_tensor_like(cute::conditional_return<kInterleaved>(tRgSin, tRgSinCont)(_, _, _, _0{}));
        int const qhead_per_khead = qhead_per_khead_divmod.divisor;
        Tensor cR = cute::make_identity_tensor(Shape<Int<kBlockMN>, Int<kHeadDim / 2>>{});  // (BLK_N,BLK_K / 2)
        Tensor tRcR = gmem_thr_copy_ro.partition_D(cR);

        // The main bottleneck here is actually instruction cache misses.

        // Similar to PagedKVNonTMA, it's expensive to compute the pointers.
        // We split the work among threads loading the same row, then __shfl_sync the pointers.
        static constexpr int NumPtrPerThread = cute::ceil_div(CUTE_STATIC_V(cute::size<1>(tRrCos)), kGmemThreadsPerRow);
        Tensor tPrCosPtr = make_tensor<Element const*>(Shape<Int<NumPtrPerThread>>{});
        Tensor tPrSinPtr = make_tensor<Element const*>(Shape<Int<NumPtrPerThread>>{});
        #pragma unroll
        for (int i = 0; i < NumPtrPerThread; ++i) {
            int const row = i * NumThreads + get<0>(tRcR(_0{}, thread_idx % kGmemThreadsPerRow, _0{}));
            int const idx = block * kBlockMN + row;
            int row_actual = qhead_per_khead_divmod.divide(idx);
            tPrCosPtr[i] = &mCos(row_actual, _0{});
            tPrSinPtr[i] = &mSin(row_actual, _0{});
        }

        #pragma unroll
        for (int m = 0; m < (!FixedPosition ? size<1>(tRgCos) : 1); ++m) {
            int const idx = block * kBlockMN + get<0>(tRcR(_0{}, m, _0{}));
            Element const* cos_ptr = reinterpret_cast<Element const*>(__shfl_sync(0xffffffff, reinterpret_cast<uint64_t>(tPrCosPtr(m / kGmemThreadsPerRow)), m % kGmemThreadsPerRow, kGmemThreadsPerRow));
            Element const* sin_ptr = reinterpret_cast<Element const*>(__shfl_sync(0xffffffff, reinterpret_cast<uint64_t>(tPrSinPtr(m / kGmemThreadsPerRow)), m % kGmemThreadsPerRow, kGmemThreadsPerRow));
            if (idx < max_seqlen * qhead_per_khead) {
                Tensor mCos_copy = cute::tiled_divide(make_tensor(make_gmem_ptr(cos_ptr), Shape<Int<kHeadDim / 2>>{}),
                                                    Shape<Int<kGmemElemsPerLoadCur>>{});
                Tensor mSin_copy = cute::tiled_divide(make_tensor(make_gmem_ptr(sin_ptr), Shape<Int<kHeadDim / 2>>{}),
                                                    Shape<Int<kGmemElemsPerLoadCur>>{});
                #pragma unroll
                for (int k = 0; k < size<2>(tRgCos); ++k) {
                    int const ki = get<1>(tRcR(_0{}, _0{}, k)) / (kGmemElemsPerLoadCur);
                    if (tRpRCur(k)) {
                        cute::copy(GmemTiledCopyRo{}, mCos_copy(_, ki), tRrCos(_, m, k));
                        cute::copy(GmemTiledCopyRo{}, mSin_copy(_, ki), tRrSin(_, m, k));
                    }
                }
            }
        }
        return cute::make_tuple(tRrCos, tRrSin);
    }

    template <typename TensorsQ, typename TensortRrR>
    CUTLASS_DEVICE
    void
    apply_Q_interleaved(TensorsQ &sQ,  // (kBlockM, kHeadDim)
                        TensortRrR const &tRrCos,   // (kBlockM, kHeadDim / 2) split according to GmemThrCopyRotary
                        TensortRrR const &tRrSin,   // (kBlockM, kHeadDim / 2) split according to GmemThrCopyRotary
                        int const m_block, int const qhead_per_khead=1)
    {
        TiledCopyQK tiled_copy_q;
        auto gmem_thr_copy_q = tiled_copy_q.get_thread_slice(thread_idx);
        Tensor tQsQ = gmem_thr_copy_q.partition_S(sQ);
        Tensor tQcQ = gmem_thr_copy_q.partition_S(cute::make_identity_tensor(Shape<Int<kBlockMN>, Int<kHeadDim>>{}));

        CUTE_STATIC_ASSERT_V(rank(tQsQ) == _3{});
        CUTE_STATIC_ASSERT_V(rank(tRrCos) == _3{});
        CUTE_STATIC_ASSERT_V(rank(tRrSin) == _3{});
        CUTE_STATIC_ASSERT_V(size<1>(tQsQ) == size<1>(tRrCos));
        CUTE_STATIC_ASSERT_V(size<2>(tQsQ) == size<2>(tRrCos));
        CUTE_STATIC_ASSERT_V(size<1>(tQsQ) == size<1>(tRrSin));
        CUTE_STATIC_ASSERT_V(size<2>(tQsQ) == size<2>(tRrSin));
        CUTE_STATIC_ASSERT_V(size<0>(tRrCos) == size<0>(tRrSin));
        static_assert(decltype(size<0>(tQsQ))::value == decltype(size<0>(tRrCos))::value * 2);
        static_assert(decltype(size<0>(tRrCos))::value % 2 == 0);  // Since we do fast conversion from fp16/bf16 to fp32

        #pragma unroll
        for (int m = 0; m < size<1>(tQsQ); ++m) {
            if (get<0>(tQcQ(_0{}, m, _0{})) < std::min(max_seqlen * qhead_per_khead - m_block * kBlockMN, kBlockMN)) {
                #pragma unroll
                for (int k = 0; k < size<2>(tQsQ); ++k) {
                    if (tRpR(k)) {
                        Tensor rQ = make_fragment_like(tQsQ(_, m, k));
                        cute::copy(tiled_copy_q, tQsQ(_, m, k), rQ);
                        apply_rotary_interleaved(rQ, tRrCos(_, m, k), tRrSin(_, m, k));
                        cute::copy(tiled_copy_q, rQ, tQsQ(_, m, k));
                    }
                }
            }
        }
    };

    template <typename TensorsQ, typename TensortRrR>
    CUTLASS_DEVICE
    void
    apply_Q_contiguous(TensorsQ &sQ,  // (kBlockM, kHeadDim)
                       TensortRrR const &tRrCosCont, // (kBlockM, kHeadDim / 2) split according to GmemThrCopyRotaryCont
                       TensortRrR const &tRrSinCont, // (kBlockM, kHeadDim / 2) split according to GmemThrCopyRotaryCont
                       int const m_block, int const qhead_per_khead=1)
    {
        TiledCopyQK tiled_copy_q;
        auto gmem_thr_copy_q = tiled_copy_q.get_thread_slice(thread_idx);
        Tensor sQ_copy = cute::tiled_divide(sQ, Shape<_1, Int<kGmemElemsPerLoad>>{});
        Tensor tQcQ = gmem_thr_copy_q.partition_S(cute::make_identity_tensor(Shape<Int<kBlockMN>, Int<kHeadDim / 2>>{}));

        CUTE_STATIC_ASSERT_V(rank(tQcQ) == _3{});
        CUTE_STATIC_ASSERT_V(rank(tRrCosCont) == _3{});
        CUTE_STATIC_ASSERT_V(rank(tRrSinCont) == _3{});
        CUTE_STATIC_ASSERT_V(size<1>(tQcQ) == size<1>(tRrCosCont));
        CUTE_STATIC_ASSERT_V(size<2>(tQcQ) == size<2>(tRrCosCont));
        CUTE_STATIC_ASSERT_V(size<1>(tQcQ) == size<1>(tRrSinCont));
        CUTE_STATIC_ASSERT_V(size<2>(tQcQ) == size<2>(tRrSinCont));
        CUTE_STATIC_ASSERT_V(size<0>(tRrCosCont) == size<0>(tRrSinCont));
        CUTE_STATIC_ASSERT_V(size<0>(tQcQ) == size<0>(tRrCosCont));
        static_assert(decltype(size<0>(tRrCosCont))::value % 2 == 0);  // Since we do fast conversion from fp16/bf16 to fp32

        #pragma unroll
        for (int m = 0; m < size<1>(tQcQ); ++m) {
            int const row = get<0>(tQcQ(_0{}, m, _0{}));
            if (row < std::min(max_seqlen * qhead_per_khead - m_block * kBlockMN, kBlockMN)) {
                #pragma unroll
                for (int k = 0; k < size<2>(tQcQ); ++k) {
                    int const col = get<1>(tQcQ(_0{}, _0{}, k));
                    if (col < rotary_dim / 2) {
                        int const col_idx_left = col / kGmemElemsPerLoad;
                        int const col_idx_right = col / kGmemElemsPerLoad + rotary_dim / (2 * kGmemElemsPerLoad);
                        Tensor rQ_left = make_fragment_like(sQ_copy(_, row, col_idx_left));
                        cute::copy(tiled_copy_q, sQ_copy(_, row, col_idx_left), rQ_left);
                        Tensor rQ_right = make_fragment_like(rQ_left);
                        cute::copy(tiled_copy_q, sQ_copy(_, row, col_idx_right), rQ_right);
                        apply_rotary_contiguous(rQ_left, rQ_right, tRrCosCont(_, m, k), tRrSinCont(_, m, k));
                        cute::copy(tiled_copy_q, rQ_left, sQ_copy(_, row, col_idx_left));
                        cute::copy(tiled_copy_q, rQ_right, sQ_copy(_, row, col_idx_right));
                    }
                }
            }
        }
    };

    template <bool PagedKVNonTMA=false, typename TensorsK, typename TensorgK, typename TensorpK, typename TensortRrR, typename TensorKPtr>
    CUTLASS_DEVICE
    void
    apply_K_interleaved(TensorsK const &sK,  // (kBlockN, kHeadDim)
                        TensorgK &gK,  // (kBlockN, kHeadDim)
                        TensorpK const &tKpK,  // (kBlockN, kHeadDim) split according to ThrCopyKV
                        TensortRrR const &tRrCos,   // (kBlockN, kHeadDim/2) split according to GmemThrCopyRotary
                        TensortRrR const &tRrSin,   // (kBlockN, kHeadDim/2) split according to GmemThrCopyRotary
                        TensorKPtr const &tPrKPtr,
                        int const n_block)
    {
        TiledCopyQK tiled_copy_k;
        auto gmem_thr_copy_q = tiled_copy_k.get_thread_slice(thread_idx);
        Tensor tKsK = gmem_thr_copy_q.partition_S(sK);
        Tensor tKgK = gmem_thr_copy_q.partition_S(gK);
        Tensor tKcK = gmem_thr_copy_q.partition_S(cute::make_identity_tensor(Shape<Int<kBlockMN>, Int<kHeadDim>>{}));

        CUTE_STATIC_ASSERT_V(rank(tKsK) == _3{});
        CUTE_STATIC_ASSERT_V(rank(tRrCos) == _3{});
        CUTE_STATIC_ASSERT_V(rank(tRrSin) == _3{});
        CUTE_STATIC_ASSERT_V(size<1>(tKsK) == size<1>(tRrCos));
        CUTE_STATIC_ASSERT_V(size<2>(tKsK) == size<2>(tRrCos));
        CUTE_STATIC_ASSERT_V(size<1>(tKsK) == size<1>(tRrSin));
        CUTE_STATIC_ASSERT_V(size<2>(tKsK) == size<2>(tRrSin));
        CUTE_STATIC_ASSERT_V(size<0>(tRrCos) == size<0>(tRrSin));
        static_assert(decltype(size<0>(tKsK))::value == decltype(size<0>(tRrCos))::value * 2);
        static_assert(decltype(size<0>(tRrCos))::value % 2 == 0);  // Since we do fast conversion from fp16/bf16 to fp32
        if constexpr (PagedKVNonTMA) {
            static_assert(decltype(size(tPrKPtr))::value == cute::ceil_div(size<1>(tKcK), kGmemThreadsPerRow));
        }

        #pragma unroll
        for (int m = 0; m < size<1>(tKsK); ++m) {
            int const row = get<0>(tKcK(_0{}, m, _0{}));
            auto mK_cur_copy = [&] {
                if constexpr (PagedKVNonTMA) {
                    Element* k_ptr = reinterpret_cast<Element*>(__shfl_sync(0xffffffff, reinterpret_cast<uint64_t>(tPrKPtr(m / kGmemThreadsPerRow)), (m % kGmemThreadsPerRow), kGmemThreadsPerRow));
                    Tensor mK_cur = make_tensor(make_gmem_ptr(k_ptr), Shape<Int<kHeadDim>>{});
                    return cute::tiled_divide(mK_cur, Shape<Int<kGmemElemsPerLoad>>{});
                } else {
                    return nullptr;
                }
            }();
            if (row < std::min(max_seqlen - n_block * kBlockMN, kBlockMN)) {
                #pragma unroll
                for (int k = 0; k < size<2>(tKsK); ++k) {
                    if (tKpK(k)) {
                        Tensor rK = make_fragment_like(tKsK(_, m, k));
                        cute::copy(tiled_copy_k, tKsK(_, m, k), rK);
                        if (tRpR(k)) { apply_rotary_interleaved(rK, tRrCos(_, m, k), tRrSin(_, m, k)); }
                        if constexpr (!PagedKVNonTMA) {
                            cute::copy(tiled_copy_k, rK, tKgK(_, m, k));
                        } else {
                            int const ki = get<1>(tKcK(_0{}, _0{}, k)) / kGmemElemsPerLoad;
                            cute::copy(tiled_copy_k, rK, mK_cur_copy(_, ki));
                        }
                    }
                }
            }
        }
    };

    template <bool PagedKVNonTMA=false, typename TensorsK, typename TensorgK, typename TensorpK, typename TensortRrR, typename TensorKPtr>
    CUTLASS_DEVICE
    void
    apply_K_contiguous(TensorsK const &sK,  // (kBlockN, kHeadDim)
                       TensorgK &gK,  // (kBlockN, kHeadDim)
                       TensorpK const &tKpK,  // (kBlockN, kHeadDim) split according to ThrCopyKV
                       TensortRrR const &tRrCosCont,   // (kBlockN, kHeadDim/2) split according to GmemThrCopyRotaryCont
                       TensortRrR const &tRrSinCont,   // (kBlockN, kHeadDim/2) split according to GmemThrCopyRotaryCont
                       TensorKPtr const &tPrKPtr,
                       int const n_block, int const max_k)
    {
        TiledCopyQK tiled_copy_k;
        auto gmem_thr_copy_q = tiled_copy_k.get_thread_slice(thread_idx);
        Tensor sK_copy = cute::tiled_divide(sK, Shape<_1, Int<kGmemElemsPerLoad>>{});
        Tensor gK_copy = cute::tiled_divide(gK, Shape<_1, Int<kGmemElemsPerLoad>>{});
        Tensor tKcK = gmem_thr_copy_q.partition_S(cute::make_identity_tensor(Shape<Int<kBlockMN>, Int<kHeadDim / 2>>{}));

        CUTE_STATIC_ASSERT_V(rank(tKcK) == _3{});
        CUTE_STATIC_ASSERT_V(rank(tRrCosCont) == _3{});
        CUTE_STATIC_ASSERT_V(rank(tRrSinCont) == _3{});
        CUTE_STATIC_ASSERT_V(size<1>(tKcK) == size<1>(tRrCosCont));
        CUTE_STATIC_ASSERT_V(size<2>(tKcK) == size<2>(tRrCosCont));
        CUTE_STATIC_ASSERT_V(size<1>(tKcK) == size<1>(tRrSinCont));
        CUTE_STATIC_ASSERT_V(size<2>(tKcK) == size<2>(tRrSinCont));
        CUTE_STATIC_ASSERT_V(size<0>(tRrCosCont) == size<0>(tRrSinCont));
        CUTE_STATIC_ASSERT_V(size<0>(tKcK) == size<0>(tRrCosCont));
        static_assert(decltype(size<0>(tRrCosCont))::value % 2 == 0);  // Since we do fast conversion from fp16/bf16 to fp32
        if constexpr (PagedKVNonTMA) {
            static_assert(decltype(size(tPrKPtr))::value == cute::ceil_div(size<1>(tKcK), kGmemThreadsPerRow));
        }

        const int ro_dim_vec = rotary_dim / kGmemElemsPerLoad;
        const int non_ro_dim_vec = (max_k - rotary_dim) / kGmemElemsPerLoad;
        #pragma unroll
        for (int m = 0; m < size<1>(tKcK); ++m) {
            int const row = get<0>(tKcK(_0{}, m, _0{}));
            Tensor gK_cur_copy = [&] {
                if constexpr (PagedKVNonTMA) {
                    Element* k_ptr = reinterpret_cast<Element*>(__shfl_sync(0xffffffff, reinterpret_cast<uint64_t>(tPrKPtr(m / kGmemThreadsPerRow)), (m % kGmemThreadsPerRow), kGmemThreadsPerRow));
                    Tensor mK_cur = make_tensor(make_gmem_ptr(k_ptr), Shape<Int<kHeadDim>>{});
                    return cute::tiled_divide(mK_cur, Shape<Int<kGmemElemsPerLoad>>{});
                } else {
                    return gK_copy(_, row, _);
                }
            }();
            if (row < std::min(max_seqlen - n_block * kBlockMN, kBlockMN)) {
                #pragma unroll
                for (int k = 0; k < size<2>(tKcK); ++k) {
                    if (tKpK(k)) {
                        int const col = get<1>(tKcK(_0{}, _0{}, k));
                        bool rotate = col < rotary_dim / 2;
                        int const col_idx_left = rotate ? col / kGmemElemsPerLoad : (col + rotary_dim / 2) / kGmemElemsPerLoad;
                        int const col_idx_right = col_idx_left + (rotate ? ro_dim_vec / 2 : non_ro_dim_vec / 2);
                        Tensor rK_left = make_fragment_like(sK_copy(_, row, col_idx_left));
                        cute::copy(tiled_copy_k, sK_copy(_, row, col_idx_left), rK_left);
                        Tensor rK_right = make_fragment_like(rK_left);
                        cute::copy(tiled_copy_k, sK_copy(_, row, col_idx_right), rK_right);
                        if (rotate) {
                            apply_rotary_contiguous(rK_left, rK_right, tRrCosCont(_, m, k), tRrSinCont(_, m, k));
                        }
                        cute::copy(tiled_copy_k, rK_left, gK_cur_copy(_, col_idx_left));
                        if (col_idx_right * kGmemElemsPerLoad < max_k) {
                            cute::copy(tiled_copy_k, rK_right, gK_cur_copy(_, col_idx_right));
                        }
                    }
                }
            }
        }
    };

};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace flash
