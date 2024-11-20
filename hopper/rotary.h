/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
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
                    Tensor K_fp32 = convert_type<float>(tKrK(_, m, k));
                    Tensor cos_fp32 = convert_type<float>(tRrCos(_, m, k));
                    Tensor sin_fp32 = convert_type<float>(tRrSin(_, m, k));
                    #pragma unroll
                    for (int i = 0; i < size<0>(tKrK) / 2; ++i) {
                        float real = K_fp32[2 * i] * cos_fp32[i] - K_fp32[2 * i + 1] * sin_fp32[i];
                        float imag = K_fp32[2 * i] * sin_fp32[i] + K_fp32[2 * i + 1] * cos_fp32[i];
                        K_fp32[2 * i] = real;
                        K_fp32[2 * i + 1] = imag;
                    }
                    cute::copy(convert_type<Engine1::value_type>(K_fp32), tKrK(_, m, k));
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
    // We assume threads loading the same row are in the same warp.
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
    using GmemTiledCopyRotaryCont = decltype(
        make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));  // Val layout, 8 or 16 vals per store

    using ShapeRotary = cute::Shape<int32_t, int32_t>;  // (seqlen_ro, rotary_dim // 2)
    using StrideRotary = cute::Stride<int64_t, _1>;

    using GmemThrCopyRotary = decltype(GmemTiledCopyRotary{}.get_thread_slice(int(0)));
    using GmemThrCopyRotaryCont = decltype(GmemTiledCopyRotaryCont{}.get_thread_slice(int(0)));
    using TensortRcR = decltype(GmemTiledCopyRotary{}.get_thread_slice(int(0)).partition_D(cute::make_identity_tensor(Shape<Int<kBlockMN>, Int<kHeadDim / 2>>{})));
    using TensortRcRCont = decltype(GmemTiledCopyRotaryCont{}.get_thread_slice(int(0)).partition_D(cute::make_identity_tensor(Shape<Int<kBlockMN>, Int<kHeadDim>>{})));
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
    using TensortRgRCont = decltype(
        GmemTiledCopyRotaryCont{}.get_thread_slice(int(0)).partition_S(make_tensor(
            make_gmem_ptr((Element const*)nullptr),
            make_shape(Int<kBlockMN>{}, Int<kHeadDim>{}, int(0)),
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
        Tensor gCosCont = local_tile(mCos, Shape<Int<kBlockMN>, Int<kHeadDim>>{}, make_coord(_, _0{}));  // (MN, K, _)
        Tensor gSinCont = local_tile(mSin, Shape<Int<kBlockMN>, Int<kHeadDim>>{}, make_coord(_, _0{}));  // (MN, K, _)

        tRgCos = gmem_thr_copy_rotary.partition_S(gCos);
        tRgSin = gmem_thr_copy_rotary.partition_S(gSin);
        tRgCosCont = gmem_thr_copy_rotary_cont.partition_S(gCosCont);
        tRgSinCont = gmem_thr_copy_rotary_cont.partition_S(gSinCont);
        Tensor cR = cute::make_identity_tensor(Shape<Int<kBlockMN>, Int<kHeadDim / 2>>{});  // (BLK_N,BLK_K / 2)
        Tensor tRcR = gmem_thr_copy_rotary.partition_D(cR);
        tRpR = make_tensor<bool>(make_shape(size<2>(tRgCos)));
        #pragma unroll
        for (int k = 0; k < size(tRpR); ++k) { tRpR(k) = get<1>(tRcR(_0{}, _0{}, k)) < get<1>(shape_rotary); }
    };

    CUTLASS_DEVICE
    auto load_cos_sin(int const block) {
        // make_tensor_like, not make_fragment_like. If the row_stride is _0{} we want to keep it that way
        Tensor tRrCos = make_tensor_like(tRgCos(_, _, _, _0{}));
        Tensor tRrSin = make_tensor_like(tRgSin(_, _, _, _0{}));
        Tensor tRrCosCont = make_tensor_like(tRgCosCont(_, _, _, _0{}));
        Tensor tRrSinCont = make_tensor_like(tRgSinCont(_, _, _, _0{}));
        if (rotary_dim <= 0) { return cute::make_tuple(tRrCos, tRrSin, tRrCosCont, tRrSinCont); }
        if (is_rotary_interleaved) {
            Tensor cR = cute::make_identity_tensor(Shape<Int<kBlockMN>, Int<kHeadDim / 2>>{});  // (BLK_N,BLK_K / 2)
            Tensor tRcR = gmem_thr_copy_rotary.partition_D(cR);
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
        } else {
            Tensor cR = cute::make_identity_tensor(Shape<Int<kBlockMN>, Int<kHeadDim>>{});  // (BLK_N,BLK_K / 2)
            Tensor tRcR = gmem_thr_copy_rotary_cont.partition_D(cR);
            #pragma unroll
            for (int m = 0; m < (!FixedPosition ? size<1>(tRgCosCont) : 1); ++m) {
                if (get<0>(tRcR(_0{}, m, _0{})) < max_seqlen - block * kBlockMN) {
                    #pragma unroll
                    for (int k = 0; k < size<2>(tRgCosCont); ++k) {
                        if (get<1>(tRcR(_0{}, _0{}, k)) < rotary_dim) {
                            const bool is_left = get<1>(tRcR(_0{}, _0{}, k)) < rotary_dim / 2;
                            Tensor gCos = domain_offset(make_coord(is_left ? 0 : -rotary_dim / 2), tRgCosCont(_, m, k, block));
                            Tensor gSin = domain_offset(make_coord(is_left ? 0 : -rotary_dim / 2), tRgSinCont(_, m, k, block));
                            // Tensor gCos = make_tensor(tRgCosCont(_, m, k, block).data() + (is_left ? 0 : -rotary_dim / 2), tRgCosCont(_, m, k, block).layout());
                            // Tensor gSin = make_tensor(tRgSinCont(_, m, k, block).data() + (is_left ? 0 : -rotary_dim / 2), tRgSinCont(_, m, k, block).layout());
                            // if (thread_idx == 0) { printf("is_left = %d, k_coord = %d\n", is_left, get<1>(tRcR(_0{}, _0{}, k))); print(tRgCosCont(_, m, k, block)); printf("\n"); print(gCos); printf("\n");}
                            cute::copy(gmem_tiled_copy_rotary_cont, gCos, tRrCosCont(_, m, k));
                            cute::copy(gmem_tiled_copy_rotary_cont, gSin, tRrSinCont(_, m, k));
                            // if (thread_idx == 0) { print_tensor(tRrCosCont(_, m, k));}
                        }
                    }
                }
            }
        }
        return cute::make_tuple(tRrCos, tRrSin, tRrCosCont, tRrSinCont);;
    }

    CUTLASS_DEVICE
    auto load_cos_sin_packgqa(int const block, cutlass::FastDivmod const &qhead_per_khead_divmod) {
        // make_tensor_like, not make_fragment_like. If the row_stride is _0{} we want to keep it that way
        Tensor tRrCos = make_tensor_like(tRgCos(_, _, _, _0{}));
        Tensor tRrSin = make_tensor_like(tRgSin(_, _, _, _0{}));
        Tensor tRrCosCont = make_tensor_like(tRgCosCont(_, _, _, _0{}));
        Tensor tRrSinCont = make_tensor_like(tRgSinCont(_, _, _, _0{}));
        if (rotary_dim <= 0) { return cute::make_tuple(tRrCos, tRrSin, tRrCosCont, tRrSinCont); }
        int const qhead_per_khead = qhead_per_khead_divmod.divisor;
        if (is_rotary_interleaved) {
            Tensor cR = cute::make_identity_tensor(Shape<Int<kBlockMN>, Int<kHeadDim / 2>>{});  // (BLK_N,BLK_K / 2)
            Tensor tRcR = gmem_thr_copy_rotary.partition_D(cR);
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
        } else {
            Tensor cR = cute::make_identity_tensor(Shape<Int<kBlockMN>, Int<kHeadDim>>{});  // (BLK_N,BLK_K / 2)
            Tensor tRcR = gmem_thr_copy_rotary_cont.partition_D(cR);
            #pragma unroll
            for (int m = 0; m < (!FixedPosition ? size<1>(tRgCos) : 1); ++m) {
                int const idx = block * kBlockMN + get<0>(tRcR(_0{}, m, _0{}));
                if (idx < max_seqlen * qhead_per_khead) {
                    int const row = qhead_per_khead_divmod.divide(idx);
                    Tensor mCos_copy = cute::tiled_divide(make_tensor(&mCos(row, _0{}), Shape<Int<kHeadDim>>{}),
                                                        Shape<Int<kGmemElemsPerLoad>>{});
                    Tensor mSin_copy = cute::tiled_divide(make_tensor(&mSin(row, _0{}), Shape<Int<kHeadDim>>{}),
                                                        Shape<Int<kGmemElemsPerLoad>>{});
                    #pragma unroll
                    for (int k = 0; k < size<2>(tRgCos); ++k) {
                        int const ki = get<1>(tRcR(_0{}, _0{}, k)) / kGmemElemsPerLoad;
                        if (get<1>(tRcR(_0{}, _0{}, k)) < rotary_dim) {
                            const bool is_left = get<1>(tRcR(_0{}, _0{}, k)) < rotary_dim / 2;
                            Tensor gCos = domain_offset(make_coord(is_left ? 0 : -rotary_dim / 2), mCos_copy(_, ki));
                            Tensor gSin = domain_offset(make_coord(is_left ? 0 : -rotary_dim / 2), mSin_copy(_, ki));
                            cute::copy(gmem_tiled_copy_rotary_cont, gCos, tRrCosCont(_, m, k));
                            cute::copy(gmem_tiled_copy_rotary_cont, gSin, tRrSinCont(_, m, k));
                        }
                    }
                }
            }
        }
        // if (thread_idx == 0) { print_tensor(tRrCosCont); print_tensor(tRrSinCont); }
        return cute::make_tuple(tRrCos, tRrSin, tRrCosCont, tRrSinCont);
    }

    template <typename TensorsQ, typename TensortRrR>
    CUTLASS_DEVICE
    void
    apply_Q_interleaved(TensorsQ &sQ,  // (kBlockM, kHeadDim)
                        TensortRrR const &tRrCos,   // (kBlockM, kHeadDim / 2) split according to GmemThrCopyRotary
                        TensortRrR const &tRrSin,   // (kBlockM, kHeadDim / 2) split according to GmemThrCopyRotary
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
                        Tensor Q_fp32 = convert_type<float>(tQrQ(_, m, k));
                        Tensor cos_fp32 = convert_type<float>(tRrCos(_, m, k));
                        Tensor sin_fp32 = convert_type<float>(tRrSin(_, m, k));
                        #pragma unroll
                        for (int i = 0; i < size<0>(tQsQ) / 2; ++i) {
                            float real = Q_fp32[2 * i] * cos_fp32[i] - Q_fp32[2 * i + 1] * sin_fp32[i];
                            float imag = Q_fp32[2 * i] * sin_fp32[i] + Q_fp32[2 * i + 1] * cos_fp32[i];
                            Q_fp32[2 * i] = real;
                            Q_fp32[2 * i + 1] = imag;
                        }
                        cute::copy(gmem_tiled_copy_q, convert_type<TensorsQ::value_type>(Q_fp32), tQsQ(_, m, k));
                    }
                }
            }
        }
    };

    template <typename TensorsQ, typename TensortRrR>
    CUTLASS_DEVICE
    void
    apply_Q_contiguous(TensorsQ &sQ,  // (kBlockM, kHeadDim)
                       TensortRrR const &tRrCosCont,   // (kBlockM, kHeadDim) split according to GmemThrCopyRotaryCont
                       TensortRrR const &tRrSinCont,   // (kBlockM, kHeadDim) split according to GmemThrCopyRotaryCont
                       int const m_block, int const qhead_per_khead=1)
    {
        if (rotary_dim <= 0) { return; }

        GmemTiledCopyQK gmem_tiled_copy_q;
        auto gmem_thr_copy_q = gmem_tiled_copy_q.get_thread_slice(thread_idx);
        Tensor tQsQ = gmem_thr_copy_q.partition_S(sQ);
        Tensor tQcQ = gmem_thr_copy_q.partition_S(cute::make_identity_tensor(Shape<Int<kBlockMN>, Int<kHeadDim>>{}));
        Tensor tQrQ = make_fragment_like(tQsQ);

        CUTE_STATIC_ASSERT_V(rank(tQrQ) == _3{});
        CUTE_STATIC_ASSERT_V(rank(tRrCosCont) == _3{});
        CUTE_STATIC_ASSERT_V(rank(tRrSinCont) == _3{});
        CUTE_STATIC_ASSERT_V(size<1>(tQrQ) == size<1>(tRrCosCont));
        CUTE_STATIC_ASSERT_V(size<2>(tQrQ) == size<2>(tRrCosCont));
        CUTE_STATIC_ASSERT_V(size<1>(tQrQ) == size<1>(tRrSinCont));
        CUTE_STATIC_ASSERT_V(size<2>(tQrQ) == size<2>(tRrSinCont));
        CUTE_STATIC_ASSERT_V(size<0>(tRrCosCont) == size<0>(tRrSinCont));
        static_assert(decltype(size<0>(tQrQ))::value == decltype(size<0>(tRrCosCont))::value);
        static_assert(decltype(size<0>(tRrCosCont))::value % 2 == 0);  // Since we do fast conversion from fp16/bf16 to fp32

        // Tensor membermask_k = make_tensor<uint32_t>(make_shape(size<2>(tQsQ)));
        // #pragma unroll
        // for (int k = 0; k < size<2>(tQsQ); ++k) {
        //     membermask_k(k) = __ballot_sync(0xffffffff, get<1>(tQcQ(_0{}, _0{}, k)) < rotary_dim);
        // }
        #pragma unroll
        for (int m = 0; m < size<1>(tQsQ); ++m) {
            int const row = get<0>(tQcQ(_0{}, m, _0{}));
            // bool valid_m = row < std::min(max_seqlen * qhead_per_khead - m_block * kBlockMN, kBlockMN);
            // uint32_t membermask_m = __ballot_sync(0xffffffff, valid_m);
            if (row < std::min(max_seqlen * qhead_per_khead - m_block * kBlockMN, kBlockMN)) {
            // if (valid_m) {
                #pragma unroll
                for (int k = 0; k < size<2>(tQsQ); ++k) {
                    int const col = get<1>(tQcQ(_0{}, _0{}, k));
                    if (col < rotary_dim) {
                        cute::copy(gmem_tiled_copy_q, tQsQ(_, m, k), tQrQ(_, m, k));
                        const bool is_left = col < rotary_dim / 2;
                        Tensor tQrQ_other = make_fragment_like(tQrQ(_, m, k));
                        Tensor sQ_other = make_tensor(make_smem_ptr(&sQ(row, col + (is_left ? rotary_dim / 2 : -rotary_dim / 2))), Shape<Shape<_1, Int<kGmemElemsPerLoad>>>{});
                        cute::copy(gmem_tiled_copy_q, sQ_other, tQrQ_other);
                        // if (thread_idx == 0) { printf("is_left = %d, row = %d, col = %d\n", is_left, row, col); print(tQsQ(_, m, k)); printf("\n"); print(sQ_other); printf("\n"); print_tensor(tQrQ(_, m, k)); print_tensor(tQrQ_other); }
                        Tensor Q_fp32 = convert_type<float>(tQrQ(_, m, k));
                        Tensor Q_other_fp32 = convert_type<float>(tQrQ_other);
                        Tensor cos_fp32 = convert_type<float>(tRrCosCont(_, m, k));
                        Tensor sin_fp32 = convert_type<float>(tRrSinCont(_, m, k));
                        #pragma unroll
                        for (int i = 0; i < size<0>(Q_fp32); ++i) {
                            Q_fp32(i) = Q_fp32(i) * cos_fp32(i) + Q_other_fp32(i) * (is_left ? -sin_fp32(i) : sin_fp32(i));
                        }
                        // Don't write to smem here since we still need the original Q for the next k
                        cute::copy(convert_type<TensorsQ::value_type>(Q_fp32), tQrQ(_, m, k));
                    }
                }
            }
        }
        __syncwarp();  // All smem reads are within the same warp
        #pragma unroll
        for (int m = 0; m < size<1>(tQsQ); ++m) {
            if (get<0>(tQcQ(_0{}, m, _0{})) < std::min(max_seqlen * qhead_per_khead - m_block * kBlockMN, kBlockMN)) {
                #pragma unroll
                for (int k = 0; k < size<2>(tQsQ); ++k) {
                    if (tRpR(k)) {
                        cute::copy(gmem_tiled_copy_q, tQrQ(_, m, k), tQsQ(_, m, k));
                    }
                }
            }
        }
        // if (thread_idx == 0) { print_tensor(tRrCosCont); print_tensor(tRrSinCont); print_tensor(tQrQ); }
    };

    template <typename TensorsK, typename TensorrK, typename TensorpK, typename TensortRrR>
    CUTLASS_DEVICE
    void
    apply_K_interleaved(TensorsK const &sK,  // (kBlockN, kHeadDim)
                        TensorrK &tKrK,  // (kBlockN, kHeadDim) split according to GmemThrCopyKV
                        TensorpK const &tKpK,  // (kBlockN, kHeadDim) split according to GmemThrCopyKV
                        TensortRrR const &tRrCos,   // (kBlockN, kHeadDim/2) split according to GmemThrCopyRotary
                        TensortRrR const &tRrSin,   // (kBlockN, kHeadDim/2) split according to GmemThrCopyRotary
                        int const n_block)
    {
        GmemTiledCopyQK gmem_tiled_copy_k;
        auto gmem_thr_copy_q = gmem_tiled_copy_k.get_thread_slice(thread_idx);
        Tensor tKsK = gmem_thr_copy_q.partition_S(sK);
        Tensor tKcK = gmem_thr_copy_q.partition_S(cute::make_identity_tensor(Shape<Int<kBlockMN>, Int<kHeadDim>>{}));

        CUTE_STATIC_ASSERT_V(rank(tKrK) == _3{});
        CUTE_STATIC_ASSERT_V(rank(tRrCos) == _3{});
        CUTE_STATIC_ASSERT_V(rank(tRrSin) == _3{});
        CUTE_STATIC_ASSERT_V(size<1>(tKrK) == size<1>(tRrCos));
        CUTE_STATIC_ASSERT_V(size<2>(tKrK) == size<2>(tRrCos));
        CUTE_STATIC_ASSERT_V(size<1>(tKrK) == size<1>(tRrSin));
        CUTE_STATIC_ASSERT_V(size<2>(tKrK) == size<2>(tRrSin));
        CUTE_STATIC_ASSERT_V(size<0>(tRrCos) == size<0>(tRrSin));
        static_assert(decltype(size<0>(tKrK))::value == decltype(size<0>(tRrCos))::value * 2);
        static_assert(decltype(size<0>(tRrCos))::value % 2 == 0);  // Since we do fast conversion from fp16/bf16 to fp32

        Tensor tRpR = make_tensor<bool>(make_shape(size<2>(tKrK)));
        #pragma unroll
        for (int k = 0; k < size(tRpR); ++k) { tRpR(k) = get<1>(tKcK(_0{}, _0{}, k)) < rotary_dim; }

        #pragma unroll
        for (int m = 0; m < size<1>(tKsK); ++m) {
            if (get<0>(tKcK(_0{}, m, _0{})) < std::min(max_seqlen - n_block * kBlockMN, kBlockMN)) {
                #pragma unroll
                for (int k = 0; k < size<2>(tKsK); ++k) {
                    if (tKpK(k)) {
                        cute::copy(gmem_tiled_copy_k, tKsK(_, m, k), tKrK(_, m, k));
                        if (tRpR(k)) {
                            Tensor K_fp32 = convert_type<float>(tKrK(_, m, k));
                            Tensor cos_fp32 = convert_type<float>(tRrCos(_, m, k));
                            Tensor sin_fp32 = convert_type<float>(tRrSin(_, m, k));
                            #pragma unroll
                            for (int i = 0; i < size<0>(tKrK) / 2; ++i) {
                                float real = K_fp32[2 * i] * cos_fp32[i] - K_fp32[2 * i + 1] * sin_fp32[i];
                                float imag = K_fp32[2 * i] * sin_fp32[i] + K_fp32[2 * i + 1] * cos_fp32[i];
                                K_fp32[2 * i] = real;
                                K_fp32[2 * i + 1] = imag;
                            }
                            cute::copy(convert_type<TensorsK::value_type>(K_fp32), tKrK(_, m, k));
                        }
                    }
                }
            }
        }
        // if (thread_idx == 0) { print_tensor(tRrCos); print_tensor(tRrSin); print_tensor(tKrK); }
    };

    template <typename TensorsK, typename TensorrK, typename TensorpK, typename TensortRrR>
    CUTLASS_DEVICE
    void
    apply_K_contiguous(TensorsK const &sK,  // (kBlockN, kHeadDim)
                       TensorrK &tKrK,  // (kBlockN, kHeadDim) split according to GmemThrCopyKV
                       TensorpK const &tKpK,  // (kBlockN, kHeadDim) split according to GmemThrCopyKV
                       TensortRrR const &tRrCosCont,   // (kBlockN, kHeadDim) split according to GmemThrCopyRotaryCont
                       TensortRrR const &tRrSinCont,   // (kBlockN, kHeadDim) split according to GmemThrCopyRotaryCont
                       int const n_block)
    {
        GmemTiledCopyQK gmem_tiled_copy_k;
        auto gmem_thr_copy_q = gmem_tiled_copy_k.get_thread_slice(thread_idx);
        Tensor tKsK = gmem_thr_copy_q.partition_S(sK);
        Tensor tKcK = gmem_thr_copy_q.partition_S(cute::make_identity_tensor(Shape<Int<kBlockMN>, Int<kHeadDim>>{}));

        CUTE_STATIC_ASSERT_V(rank(tKrK) == _3{});
        CUTE_STATIC_ASSERT_V(rank(tRrCosCont) == _3{});
        CUTE_STATIC_ASSERT_V(rank(tRrSinCont) == _3{});
        CUTE_STATIC_ASSERT_V(size<1>(tKrK) == size<1>(tRrCosCont));
        CUTE_STATIC_ASSERT_V(size<2>(tKrK) == size<2>(tRrCosCont));
        CUTE_STATIC_ASSERT_V(size<1>(tKrK) == size<1>(tRrSinCont));
        CUTE_STATIC_ASSERT_V(size<2>(tKrK) == size<2>(tRrSinCont));
        CUTE_STATIC_ASSERT_V(size<0>(tRrCosCont) == size<0>(tRrSinCont));
        static_assert(decltype(size<0>(tKrK))::value == decltype(size<0>(tRrCosCont))::value);
        static_assert(decltype(size<0>(tRrCosCont))::value % 2 == 0);  // Since we do fast conversion from fp16/bf16 to fp32

        #pragma unroll
        for (int m = 0; m < size<1>(tKsK); ++m) {
            int const row = get<0>(tKcK(_0{}, m, _0{}));
            if (row < std::min(max_seqlen - n_block * kBlockMN, kBlockMN)) {
                #pragma unroll
                for (int k = 0; k < size<2>(tKsK); ++k) {
                    if (tKpK(k)) {
                        cute::copy(gmem_tiled_copy_k, tKsK(_, m, k), tKrK(_, m, k));
                        int const col = get<1>(tKcK(_0{}, _0{}, k));
                        if (col < rotary_dim) {
                            const bool is_left = col < rotary_dim / 2;
                            Tensor tKrK_other = make_fragment_like(tKrK(_, m, k));
                            Tensor sK_other = make_tensor(make_smem_ptr(&sK(row, col + (is_left ? rotary_dim / 2 : -rotary_dim / 2))), Shape<Shape<_1, Int<kGmemElemsPerLoad>>>{});
                            cute::copy(gmem_tiled_copy_k, sK_other, tKrK_other);
                            // if (thread_idx == 0) { printf("is_left = %d, row = %d, col = %d\n", is_left, row, col); print(tKsK(_, m, k)); printf("\n"); print(sK_other); printf("\n"); print_tensor(tKrK(_, m, k)); print_tensor(tKrK_other); }
                            Tensor K_fp32 = convert_type<float>(tKrK(_, m, k));
                            Tensor K_other_fp32 = convert_type<float>(tKrK_other);
                            Tensor cos_fp32 = convert_type<float>(tRrCosCont(_, m, k));
                            Tensor sin_fp32 = convert_type<float>(tRrSinCont(_, m, k));
                            #pragma unroll
                            for (int i = 0; i < size<0>(K_fp32); ++i) {
                                K_fp32(i) = K_fp32(i) * cos_fp32(i) + K_other_fp32(i) * (is_left ? -sin_fp32(i) : sin_fp32(i));
                            }
                            cute::copy(convert_type<TensorsK::value_type>(K_fp32), tKrK(_, m, k));
                        }
                    }
                }
            }
        }
        // if (thread_idx == 0) { print_tensor(tRrCosCont); print_tensor(tRrSinCont); print_tensor(tKrK); }
    };

};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace flash
