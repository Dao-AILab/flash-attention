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
    using GmemCopyAtomCpAsync = cute::Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, Element>;
    using GmemLayoutAtomKVCpAsync = Layout<Shape <Int<NumThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                           Stride<Int<kGmemThreadsPerRow>, _1>>;
    using GmemTiledCopyQCpAsync = decltype(
        make_tiled_copy(GmemCopyAtomCpAsync{},
                        GmemLayoutAtomKVCpAsync{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));  // Val layout, 8 or 16 vals per load

     using ShapeQPacked = cute::Shape<cute::Shape<int32_t, int32_t>, int32_t, int32_t, int32_t>;
    using StrideQPacked = cute::Stride<cute::Stride<int64_t, int64_t>, _1, int64_t, int64_t>;

    template <typename TensormQ, typename TensorsQ>
    CUTLASS_DEVICE
    static void
    load_Q(TensormQ &&mQ,  // ((qhead_per_khead, seqlen_q), headdim)
           TensorsQ &&sQ,  // (kBlockM, kHeadDim)
           cutlass::FastDivmod const &qhead_per_khead_divmod,
           int const thread_idx, int const seqlen_q, int const m_block
          )
    {
        GmemTiledCopyQCpAsync gmem_tiled_copy_Q_cp_async;
        auto gmem_thr_copy_Q_cp_async = gmem_tiled_copy_Q_cp_async.get_thread_slice(thread_idx);
        Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        Tensor tQcQ = gmem_thr_copy_Q_cp_async.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
        Tensor tQsQ = gmem_thr_copy_Q_cp_async.partition_D(sQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
        Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(_0{}, _0{}, k)) < size<1>(mQ); }

        // Similar to loading K and V when PagedKV, it's expensive to compute the pointers for Q.
        // We split the work among threads loading the same row of Q, then __shfl_sync the pointers.
        static constexpr int kQPtrPerThread = cute::ceil_div(size<1>(tQsQ), kGmemThreadsPerRow);
        Tensor tPrQPtr = make_tensor<Element const*>(Shape<Int<kQPtrPerThread>>{});
        #pragma unroll
        for (int i = 0; i < kQPtrPerThread; ++i) {
            int const row = i * NumThreads + (thread_idx % kGmemThreadsPerRow) * (NumThreads / kGmemThreadsPerRow) + (thread_idx / kGmemThreadsPerRow);
            int const idx = m_block * kBlockM + row;
            int m_idx, h_idx;
            m_idx = qhead_per_khead_divmod.divmod(h_idx, idx);
            tPrQPtr[i] = &mQ(make_coord(h_idx, m_idx), _0{});
        }
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

};

} // namespace flash
