/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cute/tensor.hpp>

#include "cutlass/fast_math.h"  // For cutlass::FastDivmod

#include "utils.h"

namespace flash {

using namespace cute;

template <int kBlockM, int kHeadDim, int NumThreads, typename Element>
struct PackGQAManager {
    // We use CpAsync for Q, since TMA doesn't work there
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static constexpr int kGmemElemsPerStore = kGmemElemsPerLoad;
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "Headdim must be a multiple of kGmemElemsPerLoad");
    // We want each "row" to have 64 elements (128 bytes, i.e. 1 cache line). E.g. if hdim=128, we want each
    // thread to have 4 loads in the M direction and 2 vectorized load in the K direction.
    // In the case of PackGQA, this reduces the number of times we need to call divmod.
    static constexpr int kBytePerRow = kHeadDim * sizeof(Element);
    static constexpr int kBlockKGmem = (kBytePerRow % 128 == 0 ? 128 : (kBytePerRow % 64 == 0 ? 64 : 32)) / sizeof(Element);
    static constexpr int kGmemThreadsPerRow = kBlockKGmem / kGmemElemsPerLoad;
    static_assert(NumThreads % kGmemThreadsPerRow == 0, "NumThreads must be a multiple of kGmemThreadsPerRow");
    // We assume threads loading the same row are in the same warp. This is for an optimization in PagedKV where
    // these threads share the same page table entry and share the work of computing pointers to paged K and paged V.
    static_assert(cutlass::NumThreadsPerWarp % kGmemThreadsPerRow == 0, "kGmemThreadsPerRow must divide NumThreadsPerWarp");
    using GmemCopyAtomCpAsync = cute::Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<uint128_t>, Element>;
    using GmemLayoutAtom = Layout<Shape <Int<NumThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                  Stride<Int<kGmemThreadsPerRow>, _1>>;
    using GmemTiledCopyQCpAsync = decltype(
        make_tiled_copy(GmemCopyAtomCpAsync{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));  // Val layout, 8 or 16 vals per load

    // Was trying to have each WG loading Q to the rows in sQ that only that WG needs so that we only need
    // to sync within each WG, but didn't seem to be any faster.
    // using GmemLayoutAtomWG = Layout<Shape <Int<128 / kGmemThreadsPerRow>, Int<NumThreads / 128>, Int<kGmemThreadsPerRow> >,
    //     Stride<Int<kGmemThreadsPerRow>, _128, _1>>;
    // using GmemTiledCopyQCpAsyncWG = decltype(
    //     make_tiled_copy(GmemCopyAtomCpAsync{},
    //                     GmemLayoutAtomNew{},
    //                     Layout<Shape<_1, _1, Int<kGmemElemsPerLoad>>>{}));  // Val layout, 8 or 16 vals per load

    using GmemTiledCopyO = decltype(
        make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, Int<kGmemElemsPerStore>>>{}));  // Val layout, 8 or 16 vals per store

    template <int NumThreadsPerRow=kGmemThreadsPerRow, typename Engine, typename Layout, typename TensorC>
    CUTLASS_DEVICE
    static auto
    compute_ptr(Tensor<Engine, Layout> &tensor, TensorC const &tRows,
                cutlass::FastDivmod const &qhead_per_khead_divmod, int const thread_idx, int const m_block) {
        // tensor of shape ((qhead_per_khead, seqlen_q))
        static constexpr int NumPtrPerThread = cute::ceil_div(CUTE_STATIC_V(cute::size(tRows)), NumThreadsPerRow);
        using TensorType = typename Engine::value_type;
        Tensor tPrPtr = make_tensor<TensorType const*>(Shape<Int<NumPtrPerThread>>{});
        #pragma unroll
        for (int i = 0; i < NumPtrPerThread; ++i) {
            int const row = i * NumThreads + get<0>(tRows(thread_idx % NumThreadsPerRow));
            int const idx = m_block * kBlockM + row;
            int m_idx, h_idx;
            m_idx = qhead_per_khead_divmod.divmod(h_idx, idx);
            tPrPtr[i] = &tensor(make_coord(make_coord(h_idx, m_idx)));
        }
        return tPrPtr;
    }


    template <typename TensormQ, typename TensorsQ>
    CUTLASS_DEVICE
    static void
    load_Q(TensormQ const &mQ,  // ((qhead_per_khead, seqlen_q), headdim)
           TensorsQ &sQ,  // (kBlockM, kHeadDim)
           cutlass::FastDivmod const &qhead_per_khead_divmod,
           int const thread_idx, int const seqlen_q, int const m_block
          )
    {
        GmemTiledCopyQCpAsync gmem_tiled_copy_Q_cp_async;
        // GmemTiledCopyQCpAsyncNew gmem_tiled_copy_Q_cp_async;
        auto gmem_thr_copy_Q_cp_async = gmem_tiled_copy_Q_cp_async.get_thread_slice(thread_idx);
        Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        Tensor tQcQ = gmem_thr_copy_Q_cp_async.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
        Tensor tQsQ = gmem_thr_copy_Q_cp_async.partition_D(sQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
        // Tensor tQcQ_ = gmem_thr_copy_Q_cp_async.partition_S(cute::flat_divide(cQ, _64{}));       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
        // Tensor tQsQ_ = gmem_thr_copy_Q_cp_async.partition_D(cute::flat_divide(sQ, _64{}));       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
        // Tensor tQcQ = group_modes<1, rank(tQcQ_) - 1>(tQcQ_);
        // Tensor tQsQ = group_modes<1, rank(tQsQ_) - 1>(tQsQ_);
        Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(_0{}, _0{}, k)) < size<1>(mQ); }

        // Similar to loading K and V when PagedKV, it's expensive to compute the pointers for Q.
        // We split the work among threads loading the same row of Q, then __shfl_sync the pointers.
        Tensor mQ_0 = mQ(_, _0{});
        Tensor tQcQ_row = tQcQ(_0{}, _, _0{});
        Tensor tPrQPtr = compute_ptr(mQ_0, tQcQ_row, qhead_per_khead_divmod, thread_idx, m_block);
        int const qhead_per_khead = qhead_per_khead_divmod.divisor;
        #pragma unroll
        for (int m = 0; m < size<1>(tQsQ); ++m) {
            int idx = m_block * kBlockM + get<0>(tQcQ(_0{}, m, _0{}));
            Element const* q_ptr = reinterpret_cast<Element const*>(__shfl_sync(0xffffffff, reinterpret_cast<uint64_t>(tPrQPtr(m / kGmemThreadsPerRow)), m % kGmemThreadsPerRow, kGmemThreadsPerRow));
            if (idx < seqlen_q * qhead_per_khead) {
                // if (thread_idx == 0) { printf("m: %d, m_idx: %d, h_idx: %d, q_ptr = %p, q_ptr_og = %p\n", m, m_idx, h_idx, q_ptr, &mQ_copy(0, make_coord(h_idx, m_idx), 0));}
                Tensor mQ_cur = make_tensor(make_gmem_ptr(q_ptr), Shape<Int<kHeadDim>>{});
                Tensor mQ_cur_copy = cute::tiled_divide(mQ_cur, Shape<Int<kGmemElemsPerLoad>>{});
                #pragma unroll
                for (int k = 0; k < size<2>(tQsQ); ++k) {
                    int ki = get<1>(tQcQ(_0{}, _0{}, k)) / kGmemElemsPerLoad;
                    // the "tiled_copy.with(tQpQ(k))"" will fill in zero for columns where tQpQ(k) is false
                    // TODO: check this
                    cute::copy(gmem_tiled_copy_Q_cp_async.with(tQpQ(k)), mQ_cur_copy(_, ki), tQsQ(_, m, k));
                }
            } // Don't need to fill in 0s for sQ since we're not gonna write the output to gmem for those rows
        }
    };

    template <typename TensormLSE, typename TensorsLSE, typename TiledMma>
    CUTLASS_DEVICE
    static void
    store_LSE(TensormLSE &mLSE,  // ((qhead_per_khead, seqlen_q))
              TensorsLSE const &tLSErLSE,  // (kBlockM) split across threads according to tiled_mma
              TiledMma tiled_mma,
              cutlass::FastDivmod const &qhead_per_khead_divmod,
              int const thread_idx, int const seqlen_o, int const m_block
             )
    {
        Tensor caccO = cute::make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
        auto thread_mma = tiled_mma.get_thread_slice(thread_idx);
        Tensor taccOcO = thread_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
        Tensor taccOcO_row = make_tensor(taccOcO.data(), flash::convert_layout_acc_rowcol(taccOcO.layout()))(_, _0{});
        CUTE_STATIC_ASSERT_V(size(tLSErLSE) == size(taccOcO_row));                     // MMA_M

        // If PackGQA, we split the work of compute divmod among threads in the same row
        static constexpr int kMmaThreadsPerRow = size<0, 0>(typename TiledMma::AtomLayoutC_TV{});
        static_assert(cutlass::NumThreadsPerWarp % kMmaThreadsPerRow == 0);
        static_assert(CUTE_STATIC_V(size(tLSErLSE)) <= kMmaThreadsPerRow);
        static_assert(CUTE_STATIC_V(size(taccOcO_row)) <= kMmaThreadsPerRow);

        Tensor tPrLSEPtr = compute_ptr<kMmaThreadsPerRow>(mLSE, taccOcO_row, qhead_per_khead_divmod, thread_idx, m_block);
        static_assert(CUTE_STATIC_V(size(tPrLSEPtr)) == 1);
        int const qhead_per_khead = qhead_per_khead_divmod.divisor;
        #pragma unroll
        for (int mi = 0; mi < size(tLSErLSE); ++mi) {
            int const row = m_block * kBlockM + get<0>(taccOcO_row(mi));
            float* ptr_LSE_cur = reinterpret_cast<float*>(__shfl_sync(0xffffffff, reinterpret_cast<uint64_t>(tPrLSEPtr[0]), mi % kMmaThreadsPerRow, kMmaThreadsPerRow));
            if (get<1>(taccOcO_row(_0{})) == 0 && row < seqlen_o * qhead_per_khead) {
                *ptr_LSE_cur = tLSErLSE(mi);
            }
        }
    };

    template <typename TensormO, typename TensorrO>
    CUTLASS_DEVICE
    static void
    store_O(TensormO &mO,  // ((qhead_per_khead, seqlen_o), headdim)
            TensorrO const &tOrO,  // (kBlockM, kHeadDim) split across threads according to gmem_tiled_copy_O
            cutlass::FastDivmod const &qhead_per_khead_divmod,
            int const thread_idx, int const seqlen_o, int const m_block
          )
    {
        GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(thread_idx);
        Tensor cO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOcO)));
        #pragma unroll
        for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(_0{}, _0{}, k)) < size<1>(mO); }

        // Similar to loading K and V when PagedKV, it's expensive to compute the pointers for O.
        // We split the work among threads loading the same row of O, then __shfl_sync the pointers.
        Tensor mO_0 = mO(_, _0{});
        Tensor tOcO_row = tOcO(_0{}, _, _0{});
        Tensor tPrOPtr = compute_ptr(mO_0, tOcO_row, qhead_per_khead_divmod, thread_idx, m_block);
        int const qhead_per_khead = qhead_per_khead_divmod.divisor;
        #pragma unroll
        for (int m = 0; m < size<1>(tOrO); ++m) {
            int idx = m_block * kBlockM + get<0>(tOcO(_0{}, m, _0{}));
            Element* o_ptr = reinterpret_cast<Element*>(__shfl_sync(0xffffffff, reinterpret_cast<uint64_t>(tPrOPtr(m / kGmemThreadsPerRow)), m % kGmemThreadsPerRow, kGmemThreadsPerRow));
            if (idx < seqlen_o * qhead_per_khead) {
                Tensor mO_cur = make_tensor(make_gmem_ptr(o_ptr), Shape<Int<kHeadDim>>{});
                Tensor mO_cur_copy = cute::tiled_divide(mO_cur, Shape<Int<kGmemElemsPerStore>>{});
                #pragma unroll
                for (int k = 0; k < size<2>(tOrO); ++k) {
                    int ki = get<1>(tOcO(_0{}, _0{}, k)) / kGmemElemsPerStore;
                    if (tOpO(k)) {
                        cute::copy(gmem_tiled_copy_O, tOrO(_, m, k), mO_cur_copy(_, ki));
                    }
                }
            }
        }
    };

    template <typename TensormO, typename TensorrO, typename TiledMma>
    CUTLASS_DEVICE
    static void
    store_O_direct(TensormO &mO,  // ((qhead_per_khead, seqlen_o), headdim)
                   TensorrO const &tOrO,  // (kBlockM, kHeadDim) split across threads according to tiled_mma
                   TiledMma tiled_mma,
                   cutlass::FastDivmod const &qhead_per_khead_divmod,
                   int const thread_idx, int const seqlen_o, int const m_block
                 )
    {
        static constexpr int kGmemElemsPerStoreDirect = 2;
        cute::Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element> gmem_copy_direct;
        // Reshape acc from ((2, 2, V), MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, V, MMA_N))
        Tensor tOrO_rowcol = make_tensor(tOrO.data(), flash::convert_layout_acc_rowcol(tOrO.layout()));
        Tensor tOrO_copy = cute::tiled_divide(tOrO_rowcol, Shape<_1, Int<kGmemElemsPerStoreDirect>>{});

        Tensor caccO = cute::make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
        auto thread_mma = tiled_mma.get_thread_slice(thread_idx);
        Tensor taccOcO = thread_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
        Tensor taccOcO_rowcol = make_tensor(taccOcO.data(), flash::convert_layout_acc_rowcol(taccOcO.layout()));
        Tensor taccOcO_row = taccOcO_rowcol(_, _0{});
        Tensor taccOcO_col = taccOcO_rowcol(_0{}, _);

        // If PackGQA, we split the work of compute divmod among threads in the same row
        static constexpr int kMmaThreadsPerRow = size<0, 0>(typename TiledMma::AtomLayoutC_TV{});
        static_assert(cutlass::NumThreadsPerWarp % kMmaThreadsPerRow == 0);
        static_assert(CUTE_STATIC_V(size(taccOcO_row)) <= kMmaThreadsPerRow);

        // Similar to loading K and V when PagedKV, it's expensive to compute the pointers for O.
        // We split the work among threads loading the same row of O, then __shfl_sync the pointers.
        Tensor mO_0 = mO(_, _0{});
        Tensor tPrOPtr = compute_ptr<kMmaThreadsPerRow>(mO_0, taccOcO_row, qhead_per_khead_divmod, thread_idx, m_block);
        static_assert(CUTE_STATIC_V(size(tPrOPtr)) == 1);

        int const qhead_per_khead = qhead_per_khead_divmod.divisor;
        #pragma unroll
        for (int m = 0; m < size<1>(tOrO_copy); ++m) {
            int row = m_block * kBlockM + get<0>(taccOcO_row(m));
            Element* o_ptr = reinterpret_cast<Element*>(__shfl_sync(0xffffffff, reinterpret_cast<uint64_t>(tPrOPtr[0]), m % kMmaThreadsPerRow, kMmaThreadsPerRow));
            if (row < seqlen_o * qhead_per_khead) {
                Tensor mO_cur = make_tensor(make_gmem_ptr(o_ptr), Shape<Int<kHeadDim>>{});
                Tensor mO_cur_copy = cute::tiled_divide(mO_cur, Shape<Int<kGmemElemsPerStoreDirect>>{});
                #pragma unroll
                for (int k = 0; k < size<2>(tOrO_copy); ++k) {
                    int col = get<1>(taccOcO_col(k * kGmemElemsPerStoreDirect));
                    if (col < size<1>(mO)) {
                        cute::copy(gmem_copy_direct, tOrO_copy(_, m, k), mO_cur_copy(_, col / kGmemElemsPerStoreDirect));
                    }
                }
            }
        }
    };

};

} // namespace flash
