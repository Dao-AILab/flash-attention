/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cute/tensor.hpp>

#include "cutlass/fast_math.h"  // For cutlass::FastDivmod

#include "utils.h"

namespace flash {

using namespace cute;

template <int kBlockN, int kHeadDim, int NumThreads, typename Element, bool KV_Same_Iter=false>
struct PagedKVManager {
    // If KV_Same_Iter=false, then we do load_page_table(0), load_K(0), load_page_table(1), load_K(1), load_V(0),
    // load_page_table(2), load_K(2), load_V(1), etc.
    // So we need to compute the V pointers for the previous iteration.

    // We use CpAsync for K and V if PagedKV, since TMA doesn't work there
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
    using GmemTiledCopyKVCpAsync = decltype(
        make_tiled_copy(GmemCopyAtomCpAsync{},
                        GmemLayoutAtomKVCpAsync{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));  // Val layout, 8 or 16 vals per load

    using ShapeKV = cute::Shape<int32_t, int32_t, int32_t, int32_t>;  // (seqlen, d, head, batch)
    using StrideKV = cute::Stride<int64_t, _1, int64_t, int64_t>;
    using ShapePageTable = cute::Shape<int32_t, int32_t>;  // (batch, max_num_pages_per_seq)
    using StridePageTable = cute::Stride<int64_t, _1>;

    using TensorPageTable = decltype(make_tensor(make_gmem_ptr(static_cast<int const*>(nullptr)), ShapePageTable{}, StridePageTable{})(int(0), _));
    using TensorKV = decltype(make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeKV{}, StrideKV{})(_, _, int(0), _));
    using GmemThrCopyKVCpAsync = decltype(GmemTiledCopyKVCpAsync{}.get_thread_slice(int(0)));
    using TensortKcK = decltype(GmemTiledCopyKVCpAsync{}.get_thread_slice(int(0)).partition_D(cute::make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDim>>{})));
    using TensortKpK = decltype(make_tensor<bool>(make_shape(size<1>(TensortKcK{}), size<2>(TensortKcK{})), Stride<_0, _1>{}));

    // For PagedKV, it's expensive the calculate the pointers to K and V for each page table entry,
    // since those require int64_t arithmetic. We optimize by having threads split this work.
    // Typically there are 8 threads loading per row (e.g. hdim 64 and 128), and there are 11 rows
    // that each thread needs to load for the case of hdim 128 and kBlockN = 176.
    // So each of those 8 threads will calculate the K_ptr and V_ptr for 11 / 8 = 2 rows.
    // We then use __shfl_sync to broadcast the pointers to the other threads in the warp.
    static constexpr int kPageEntryPerThread = cute::ceil_div(size<1>(TensortKcK{}), kGmemThreadsPerRow);
    using TensorPageOffset = decltype(make_tensor<cute::tuple<int, int>>(Shape<Int<kPageEntryPerThread>>{}));
    using TensorKVPtr = decltype(make_tensor<Element const*>(Shape<Int<kPageEntryPerThread>>{}));

    cutlass::FastDivmod const &page_size_divmod;
    int const thread_idx;
    int const seqlen_k;
    int const leftpad_k;
    TensorPageTable mPageTable;
    TensorKV mK_paged, mV_paged;
    TensortKpK tKpK;
    TensorPageOffset tPrPageOffset;
    TensorKVPtr tPrVPtr;
    GmemTiledCopyKVCpAsync gmem_tiled_copy_kv;
    GmemThrCopyKVCpAsync const gmem_thr_copy_kv;


    CUTLASS_DEVICE
    PagedKVManager(int const* const ptr_page_table,
                   ShapePageTable const &shape_pagetable, StridePageTable const &stride_pagetable,
                   Element const* const ptr_K, ShapeKV const &shape_K, StrideKV const &stride_K,
                   Element const* const ptr_V, StrideKV const &stride_V,
                   cutlass::FastDivmod const &page_size_divmod,
                   int const bidb, int const bidh, int const thread_idx, int const seqlen_k, int const leftpad_k
                   )
        : page_size_divmod(page_size_divmod)
        , thread_idx(thread_idx)
        , seqlen_k(seqlen_k)
        , leftpad_k(leftpad_k)
        , gmem_thr_copy_kv(gmem_tiled_copy_kv.get_thread_slice(thread_idx))

    {
        mPageTable = make_tensor(make_gmem_ptr(ptr_page_table), shape_pagetable, stride_pagetable)(bidb, _);
        mK_paged = make_tensor(make_gmem_ptr(ptr_K), shape_K, stride_K)(_, _, bidh, _);
        mV_paged = make_tensor(make_gmem_ptr(ptr_V), shape_K, stride_V)(_, _, bidh, _);
        tKpK = make_tensor<bool>(make_shape(size<1>(TensortKcK{}), size<2>(TensortKcK{})), Stride<_0, _1>{});

        Tensor cK = cute::make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDim>>{});  // (BLK_N,BLK_K) -> (blk_n,blk_k)
        Tensor tKcK = gmem_thr_copy_kv.partition_S(cK);
        #pragma unroll
        for (int k = 0; k < size<1>(tKpK); ++k) { tKpK(_0{}, k) = get<1>(tKcK(_0{}, _0{}, k)) < get<1>(shape_K); }
    };

    template <bool Seqlenk_mask=false, bool First_iter=false>
    CUTLASS_DEVICE
    void load_page_table(const int n_block) {
        // The uncoalesced gmem load is intentional. This is so that each thread only loads the page table entries
        // it needs, and we don't need any sync between warps.
        // Assuming 8 threads per row, and 176 rows, then the rows from 0 to 175 are loaded by
        // threads 0, 8, 16, ..., 120, 1, 9, ..., 121, 2, 10, ..., 122, etc.
        #pragma unroll
        for (int i = 0; i < kPageEntryPerThread; ++i) {
            int const row = i * NumThreads + (thread_idx % kGmemThreadsPerRow) * (NumThreads / kGmemThreadsPerRow) + (thread_idx / kGmemThreadsPerRow);
            int const row_idx = n_block * kBlockN + row;
            int page_idx, page_offset;
            page_idx = page_size_divmod.divmod(page_offset, row_idx + leftpad_k);
            // Add the condition (i + 1) * NumThreads <= kBlockN since that is an upper bound of row
            // and is known at compile time. It avoids branching when e.g., kBlockN = 176 and i = 0.
            int const page = ((i + 1) * NumThreads <= kBlockN || row < kBlockN) && (!Seqlenk_mask || row_idx < seqlen_k) ? mPageTable[page_idx] : 0;
            tPrPageOffset[i] = {page, page_offset};
            // if (cute::thread0()) { printf("row = %d, page_idx = %d, page_offset = %d, page = %d, leftpad_k = %d, seqlen_k = %d\n", row, page_idx, page_offset, page, leftpad_k, seqlen_k); }
        }
        if constexpr (First_iter && !KV_Same_Iter) { compute_V_ptr(); }
    };

    CUTLASS_DEVICE
    void compute_V_ptr() {
        #pragma unroll
        for (int i = 0; i < kPageEntryPerThread; ++i) {
            auto [page, page_offset] = tPrPageOffset[i];
            tPrVPtr[i] = &mV_paged(page_offset, _0{}, page);
        }
    };

    template <bool Seqlenk_mask=false, typename TensorK>
    CUTLASS_DEVICE
    void load_K(const int n_block, TensorK &&sK) {
        Tensor tPrKPtr = make_tensor<Element const*>(Shape<Int<kPageEntryPerThread>>{});
        #pragma unroll
        for (int i = 0; i < kPageEntryPerThread; ++i) {
            auto [page, page_offset] = tPrPageOffset[i];
            tPrKPtr[i] = &mK_paged(page_offset, _0{}, page);
        }

        // Only for index calculation, since all the indices of thread 0 are known at compile time
        auto gmem_thr0_copy_kv = gmem_tiled_copy_kv.get_thread_slice(_0{});
        Tensor tKsK = gmem_thr_copy_kv.partition_D(sK);
        Tensor cK = cute::make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDim>>{});  // (BLK_N,BLK_K) -> (blk_n,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tKcK = gmem_thr_copy_kv.partition_S(cK);
        Tensor t0KcK = gmem_thr0_copy_kv.partition_S(cK);

        // We want to use the row indices of thread0 to compare, since that is known at compile time.
        // So we subtract the limit by the first row index of this thread (get<0>(tKcK(_0{}, _0{}, _0{})))
        int const seqlenk_row_limit = seqlen_k - n_block * kBlockN - get<0>(tKcK(_0{}, _0{}, _0{}));
        #pragma unroll
        for (int m = 0; m < size<1>(tKsK); ++m) {
            bool const should_load = !Seqlenk_mask || get<0>(t0KcK(_0{}, m, _0{})) < seqlenk_row_limit;
            Element const* k_ptr = reinterpret_cast<Element const*>(__shfl_sync(0xffffffff, reinterpret_cast<uint64_t>(tPrKPtr(m / kGmemThreadsPerRow)), (m % kGmemThreadsPerRow), kGmemThreadsPerRow));
            Tensor mK_paged_cur = make_tensor(make_gmem_ptr(k_ptr), Shape<Int<kHeadDim>>{});
            Tensor mK_paged_cur_copy = cute::tiled_divide(mK_paged_cur, Shape<Int<kGmemElemsPerLoad>>{});
            if (should_load) {
                #pragma unroll
                for (int k = 0; k < size<2>(tKsK); ++k) {
                    int const ki = get<1>(tKcK(_0{}, _0{}, k)) / kGmemElemsPerLoad;
                    cute::copy(gmem_tiled_copy_kv.with(tKpK(_0{}, k)), mK_paged_cur_copy(_, ki), tKsK(_, m, k));
                }
            }  // Don't need to clear out the rest of the smem since we'll mask out the scores anyway
        }
    };

    template <bool Seqlenk_mask=false, typename TensorV>
    CUTLASS_DEVICE
    void load_V(const int n_block, TensorV &&sV) {
        if constexpr (KV_Same_Iter) { compute_V_ptr(); }
        // Only for index calculation, since all the indices of thread 0 are known at compile time
        auto gmem_thr0_copy_kv = gmem_tiled_copy_kv.get_thread_slice(_0{});
        Tensor tVsV = gmem_thr_copy_kv.partition_D(sV);
        Tensor cK = cute::make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDim>>{});  // (BLK_N,BLK_K) -> (blk_n,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tKcK = gmem_thr_copy_kv.partition_S(cK);
        Tensor t0KcK = gmem_thr0_copy_kv.partition_S(cK);

        int const seqlenk_row_limit = seqlen_k - n_block * kBlockN - get<0>(tKcK(_0{}, _0{}, _0{}));
        #pragma unroll
        for (int m = 0; m < size<1>(tVsV); ++m) {
            // Faster to rely on the cp.async to clear smem that are out of bound,
            // rather than calling cute::clear directly.
            bool should_load = !Seqlenk_mask || get<0>(t0KcK(_0{}, m, _0{})) < seqlenk_row_limit;
            Element const* v_ptr = reinterpret_cast<Element const*>(__shfl_sync(0xffffffff, reinterpret_cast<uint64_t>(tPrVPtr(m / kGmemThreadsPerRow)), m % kGmemThreadsPerRow, kGmemThreadsPerRow));
            Tensor mV_paged_cur = make_tensor(make_gmem_ptr(v_ptr), Shape<Int<kHeadDim>>{});
            Tensor mV_paged_cur_copy = cute::tiled_divide(mV_paged_cur, Shape<Int<kGmemElemsPerLoad>>{});
            #pragma unroll
            for (int k = 0; k < size<2>(tVsV); ++k) {
                int const ki = get<1>(tKcK(_0{}, _0{}, k)) / kGmemElemsPerLoad;
                cute::copy(gmem_tiled_copy_kv.with(tKpK(_0{}, k) && should_load), mV_paged_cur_copy(_, ki), tVsV(_, m, k));
            }
        }
        if constexpr (!KV_Same_Iter) { compute_V_ptr(); }
    };


};

} // namespace flash
