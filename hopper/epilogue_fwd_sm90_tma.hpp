/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/fast_math.h>  // For FastDivMod
#include "cute/tensor.hpp"

#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"

#include "named_barrier.hpp"
#include "utils.h"

namespace flash {

using namespace cute;

template <class TileShape_MNK_, class Element_, int NumEpilogueThreads_, bool Varlen_, bool GQAPack_, bool FP8PermuteCol=false>
struct CollectiveEpilogueFwd {

    using TileShape_MNK = TileShape_MNK_;
    using Element = Element_;
    static constexpr int NumEpilogueThreads = NumEpilogueThreads_;
    static constexpr bool Varlen = Varlen_;
    static constexpr bool PackGQA = GQAPack_;

    static constexpr int kHeadDim = get<2>(TileShape_MNK{});
    static constexpr int kBlockM = get<0>(TileShape_MNK{});

    using GmemTiledCopyOTMA = cute::SM90_TMA_STORE;

    // These are for storing the output tensor without TMA (e.g., for setting output to zero)
    static constexpr int kGmemElemsPerStore = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerStore == 0, "Headdim must be a multiple of kGmemElemsPerStore");
    // We want each "row" to have 64 elements (128 bytes, i.e. 1 cache line). We want each thread to have 4 elements
    // in the M direction and 2 elements in the K direction. In the case of PackGQA, this reduces the number of times
    // we need to call divmod.
    static constexpr int kBlockKGmem = kHeadDim % 128 == 0 ? (128 / sizeof(Element)) : (kHeadDim % 64 == 0 ? 64 : 32);
    // static constexpr int kBlockKGmem = kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);
    // static constexpr int kGmemThreadsPerRow = cutlass::gcd(kHeadDim / kGmemElemsPerStore, NumEpilogueThreads);
    static constexpr int kGmemThreadsPerRow = kBlockKGmem / kGmemElemsPerStore;
    // If PackGQA, we split the work of compute O_ptr among threads in the same row, so we need this to within a warp
    static_assert(cutlass::NumThreadsPerWarp % kGmemThreadsPerRow == 0);
    static_assert(NumEpilogueThreads % kGmemThreadsPerRow == 0, "NumEpilogueThreads must be a multiple of kGmemThreadsPerRow");
    using GmemLayoutAtom = Layout<Shape <Int<NumEpilogueThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                  Stride<Int<kGmemThreadsPerRow>, _1>>;
    using GmemTiledCopyO = decltype(
        make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, Int<kGmemElemsPerStore>>>{}));  // Val layout, 8 or 16 vals per store

    static constexpr bool Use_smem = sizeof(Element) <= 2;

    using SmemLayoutAtomO = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutO = decltype(tile_to_shape(SmemLayoutAtomO{}, select<0, 2>(TileShape_MNK{})));

    using ShapeO = cute::Shape<int32_t, int32_t, int32_t, int32_t, int32_t>;  // (seqlen_q, d, head, batch, num_splits)
    using StrideO = cute::Stride<int64_t, _1, int64_t, int64_t, int64_t>;
    using StrideLSE = cute::Stride<_1, int64_t, int64_t, int64_t>;            // (seqlen_q, head, batch, num_splits)
    // ((qhead_per_khead, seqlen_q), d, nheads_kv, batch, num_splits)
    using ShapeOPacked = std::conditional_t<!PackGQA, ShapeO, cute::Shape<cute::Shape<int32_t, int32_t>, int32_t, int32_t, int32_t, int32_t>>;
    using StrideOPacked = std::conditional_t<!PackGQA, StrideO, cute::Stride<cute::Stride<int64_t, int64_t>, _1, int64_t, int64_t, int64_t>>;
    // ((qhead_per_khead, seqlen_q), nheads_kv, batch, num_splits)
    using ShapeLSEPacked = std::conditional_t<!PackGQA, cute::Shape<int32_t, int32_t, int32_t, int32_t>, cute::Shape<cute::Shape<int32_t, int32_t>, int32_t, int32_t, int32_t>>;
    using StrideLSEPacked = std::conditional_t<!PackGQA, StrideLSE, cute::Stride<cute::Stride<int64_t, _1>, int64_t, int64_t, int64_t>>;

    // cute::SM90_U32x4_STSM_N if Element size is 2 bytes (fp16, bf16)
    using CopyOpR2S = decltype(cutlass::epilogue::collective::detail::sm90_get_smem_store_op_for_accumulator<StrideO, Element>());
    using SmemCopyAtomO = Copy_Atom<CopyOpR2S, Element>;

    // static constexpr size_t SmemAlignmentO = cutlass::detail::alignment_for_swizzle(SmemLayoutO{});
    // static_assert(SmemAlignmentO >= 128, "Require at least 128B alignment");
    // struct TensorStorage : cute::aligned_struct<SmemAlignmentO> {
    //     cute::array_aligned<Element, Use_smem ? cute::cosize_v<SmemLayoutO> : 0, SmemAlignmentO> smem_o;
    // };
    struct TensorStorage : cute::aligned_struct<128> {
        cute::array_aligned<Element, Use_smem ? cute::cosize_v<SmemLayoutO> : 0> smem_o;
    };

    using TMA_O = decltype(make_tma_copy(
        GmemTiledCopyOTMA{},
        make_tensor(make_gmem_ptr(static_cast<Element*>(nullptr)), ShapeO{}, StrideO{}),
        SmemLayoutO{},
        select<0, 2>(TileShape_MNK{}),
        _1{}));  // no mcast for O

    // Host side kernel arguments
    struct Arguments {
        Element* ptr_O;
        ShapeO const shape_O;
        StrideO const stride_O;
        float* ptr_LSE;
        StrideLSE const stride_LSE;
        int32_t const nheads_kv;
        int const* cu_seqlens = nullptr;
        int const* seqused = nullptr;
    };

    // Device side kernel params
    struct Params {
        Element* ptr_O;
        ShapeO const shape_O;
        StrideO const stride_O;
        ShapeOPacked const shape_O_packed;
        StrideOPacked const stride_O_packed;
        float* ptr_LSE;
        StrideLSE const stride_LSE;
        ShapeLSEPacked const shape_LSE_packed;
        StrideLSEPacked const stride_LSE_packed;
        cutlass::FastDivmod qhead_per_khead_divmod;
        TMA_O tma_store_O;
        int const* cu_seqlens = nullptr;
        int const* seqused = nullptr;
    };

    static Params
    to_underlying_arguments(Arguments const& args) {
        Tensor mO = make_tensor(make_gmem_ptr(args.ptr_O), args.shape_O, args.stride_O);
        TMA_O tma_store_O = make_tma_copy(
            GmemTiledCopyOTMA{},
            mO,
            SmemLayoutO{},
            select<0, 2>(TileShape_MNK{}),
            _1{}); // no mcast for O
        // If PackGQA, reshape O to be ((qhead_per_khead, seqlen_q), head_size, nhead_k, batch_size, num_splits)
        int const qhead_per_khead = !PackGQA ? 1 : cute::ceil_div(get<2>(args.shape_O), args.nheads_kv);
        auto const shape_O_packed = cute::conditional_return<!PackGQA>(
            args.shape_O,
            make_shape(make_shape(qhead_per_khead, get<0>(args.shape_O)), get<1>(args.shape_O), args.nheads_kv, get<3>(args.shape_O), get<4>(args.shape_O))
        );
        auto const stride_O_packed = cute::conditional_return<!PackGQA>(
            args.stride_O,
            make_stride(make_stride(get<2>(args.stride_O), get<0>(args.stride_O)), get<1>(args.stride_O), get<2>(args.stride_O) * qhead_per_khead, get<3>(args.stride_O), get<4>(args.stride_O))
        );
        // If PackGQA, Reshape LSE to be ((qhead_per_khead, seqlen_q), nhead_k, batch_size, num_splits)
        auto const shape_LSE_packed = cute::conditional_return<!PackGQA>(
            select<0, 2, 3, 4>(args.shape_O),
            make_shape(make_shape(qhead_per_khead, get<0>(args.shape_O)), args.nheads_kv, get<3>(args.shape_O), get<4>(args.shape_O))
        );
        auto const stride_LSE_packed = cute::conditional_return<!PackGQA>(
            args.stride_LSE,
            make_stride(make_stride(get<1>(args.stride_LSE), get<0>(args.stride_LSE)), get<1>(args.stride_LSE) * qhead_per_khead, get<2>(args.stride_LSE), get<3>(args.stride_LSE))
        );
        return {args.ptr_O, args.shape_O, args.stride_O, shape_O_packed, stride_O_packed,
                args.ptr_LSE, args.stride_LSE, shape_LSE_packed, stride_LSE_packed,
                cutlass::FastDivmod(qhead_per_khead),
                tma_store_O, args.cu_seqlens, args.seqused};
    }

    /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
    CUTLASS_DEVICE
    static void prefetch_tma_descriptors(Params const& params) {
        if constexpr (!Varlen && Use_smem && !PackGQA) {
            cute::prefetch_tma_descriptor(params.tma_store_O.get_tma_descriptor());
        }
    }

    template <typename SharedStorage, typename FrgTensorO, typename FrgTensorLSE, typename TiledMma>
    CUTLASS_DEVICE void
    store(Params const& params,
          FrgTensorO const& tOrO,
          FrgTensorLSE const& lse,
          SharedStorage& shared_storage,
          TiledMma tiled_mma,
          int thread_idx,
          cute::tuple<int32_t, int32_t, int32_t, int32_t> const& block_coord
          ) {

        auto [m_block, bidh, bidb, split_idx] = block_coord;
        Tensor sO = make_tensor(make_smem_ptr(shared_storage.tensors.epilogue.smem_o.data()), SmemLayoutO{});
        // Tensor sO_pi = cute::as_position_independent_swizzle_tensor(sO);

        // Tensor tOrO_out = flash::convert_type<Element>(tOrO);
        Tensor tOrO_out = flash::convert_type_safe<Element>(tOrO);
        if constexpr (FP8PermuteCol && (sizeof(Element) == 2 || sizeof(Element) == 4)) { flash::permute_output_fp8_Vcolmajor(tOrO_out); }

        // Make sure all WGs have finished reading V
        // Technically we don't need this if we're not using smem, but the mainloop makes the assumption that
        // all epilogue threads sync at least once during the epilogue (so that we can start loading Q with
        // cp.async if we need).
        cutlass::arch::NamedBarrier::sync(NumEpilogueThreads, static_cast<int>(FwdNamedBarriers::ValueEmpty) /*id*/);

        // Step 1: Write O from rmem -> smem
        if constexpr (Use_smem) {
            auto smem_tiled_copy_O = make_tiled_copy_C(SmemCopyAtomO{}, tiled_mma);
            auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(thread_idx);
            Tensor taccOrO = smem_thr_copy_O.retile_S(tOrO_out);        // ((Atom,AtomNum), MMA_M, MMA_N)
            Tensor taccOsO = smem_thr_copy_O.partition_D(sO);     // ((Atom,AtomNum),PIPE_M,PIPE_N)
            // Tensor taccOsO = smem_thr_copy_O.partition_D(sO_pi);     // ((Atom,AtomNum),PIPE_M,PIPE_N)
            cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);
            if constexpr (!Varlen && !PackGQA) {
                cutlass::arch::fence_view_async_shared(); // ensure smem writes are visible to TMA
                cutlass::arch::NamedBarrier::arrive(NumEpilogueThreads + cutlass::NumThreadsPerWarp,
                                                    cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
            } else {
                cutlass::arch::NamedBarrier::sync(NumEpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
            }
        }

        bool is_varlen = Varlen && params.cu_seqlens;
        int offset_o = !is_varlen ? 0 : params.cu_seqlens[bidb];
        int seqlen_o = !Varlen ? size<0>(params.shape_O) : (params.seqused ? params.seqused[bidb] : (params.cu_seqlens ? params.cu_seqlens[bidb + 1] - offset_o : size<0>(params.shape_O)));

        // Step 2: Write LSE from rmem -> gmem
        Tensor caccO = cute::make_identity_tensor(select<0, 2>(TileShape_MNK{}));
        auto thread_mma = tiled_mma.get_thread_slice(thread_idx);
        Tensor taccOcO = thread_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
        static_assert(decltype(size<0, 0>(taccOcO))::value == 2);
        static_assert(decltype(size<0, 1>(taccOcO))::value == 2);
        // taccOcO has shape ((2, 2, V), MMA_M, MMA_K), we only take only the row indices.
        Tensor taccOcO_row = taccOcO(make_coord(_0{}, _, _0{}), _, _0{});
        CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));                     // MMA_M

        int qhead_per_khead = !PackGQA ? 1 : params.qhead_per_khead_divmod.divisor;
        // If PackGQA, we split the work of compute divmod among threads in the same row
        static constexpr int kMmaThreadsPerRow = size<0, 0>(typename TiledMma::AtomLayoutC_TV{});
        static_assert(cutlass::NumThreadsPerWarp % kMmaThreadsPerRow == 0);
        static_assert(CUTE_STATIC_V(size(lse)) <= kMmaThreadsPerRow);
        static_assert(CUTE_STATIC_V(size(taccOcO_row)) <= kMmaThreadsPerRow);
        int mma_m_idx, mma_h_idx;
        // Might get OOB but it's ok since we'll check it later
        if constexpr (PackGQA) {
            mma_m_idx = params.qhead_per_khead_divmod.divmod(mma_h_idx, m_block * kBlockM + get<0>(taccOcO_row(thread_idx % kMmaThreadsPerRow)));
        }

        Tensor mLSE = make_tensor(make_gmem_ptr(params.ptr_LSE + offset_o * get<0>(params.stride_LSE)), params.shape_LSE_packed, params.stride_LSE_packed)(_, bidh, !is_varlen ? bidb : 0, split_idx);
        // if (thread_idx == 0) { printf("Before LSE write, m_block: %d, bidh: %d, bidb: %d, split_idx: %d, offset_o: %d, seqlen_o: %d\n", m_block, bidh, bidb, split_idx, offset_o, seqlen_o); print(mLSE); printf("\n"); }
        float* ptr_LSE;
        if constexpr (PackGQA) { ptr_LSE = &mLSE(make_coord(make_coord(mma_h_idx, mma_m_idx))); }
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            int const row = m_block * kBlockM + get<0>(taccOcO_row(mi));
            if constexpr (!PackGQA) {
                if (get<1>(taccOcO_row(_0{})) == 0 && row < seqlen_o) { mLSE(row) = lse(mi); }
            } else {
                float* ptr_LSE_cur = reinterpret_cast<float*>(__shfl_sync(0xffffffff, reinterpret_cast<uint64_t>(ptr_LSE), mi % kMmaThreadsPerRow, kMmaThreadsPerRow));
                if (get<1>(taccOcO_row(_0{})) == 0 && row < seqlen_o * qhead_per_khead) {
                    // int m_idx, h_idx;
                    // m_idx = params.qhead_per_khead_divmod.divmod(h_idx, row);
                    // mLSE shape shape ((qhead_per_khead, seqlen_q)) and it's unhappy with just 1 "make_coord"
                    // mLSE(make_coord(make_coord(h_idx, m_idx))) = lse(mi);
                    *ptr_LSE_cur = lse(mi);
                }
            }
        }

        // Step 3: Write O from smem -> gmem
        if constexpr (!Varlen && Use_smem && !PackGQA) {
            Tensor mO = params.tma_store_O.get_tma_tensor(params.shape_O)(_, _, bidh, bidb, split_idx);
            Tensor gO = local_tile(mO, select<0, 2>(TileShape_MNK{}), make_coord(m_block, _0{}));  // (M, K)
            auto block_tma_O = params.tma_store_O.get_slice(_0{});
            Tensor tOgO = block_tma_O.partition_D(gO);  // (TMA, TMA_M, TMA_K)
            Tensor tOsO = block_tma_O.partition_S(sO); // (TMA, TMA_M, TMA_K)
            int warp_idx_sync = __shfl_sync(0xffffffff, thread_idx / cutlass::NumThreadsPerWarp, 0);
            if (warp_idx_sync == NumEpilogueThreads / cutlass::NumThreadsPerWarp - 1) {
                cutlass::arch::NamedBarrier::sync(NumEpilogueThreads + cutlass::NumThreadsPerWarp,
                                                  cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
                if (cute::elect_one_sync()) {
                    cute::copy(params.tma_store_O, tOsO, tOgO);
                    tma_store_arrive();
                }
            }
        } else {  // Don't use TMA since we don't want to overwrite the output of another sequence
            Tensor mO = make_tensor(make_gmem_ptr(params.ptr_O + offset_o * get<0>(params.stride_O)), params.shape_O_packed, params.stride_O_packed)(_, _, bidh, !is_varlen ? bidb : 0, split_idx);
            // if (thread_idx == 0) { printf("Before O write, m_block: %d, bidh: %d, bidb: %d, split_idx: %d, offset_o: %d, seqlen_o: %d, mO_addr = %p, addr diff = %d\n", m_block, bidh, bidb, split_idx, offset_o, seqlen_o, mO.data(), reinterpret_cast<int>(&mO(0)) - reinterpret_cast<int>(params.ptr_O)); }
            if constexpr (Use_smem) {
                GmemTiledCopyO gmem_tiled_copy_O;
                auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(thread_idx);
                Tensor tOsO = gmem_thr_copy_O.partition_S(sO);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
                // Tensor tOsO = gmem_thr_copy_O.partition_S(sO_pi);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
                Tensor tOrO = make_fragment_like(tOsO);
                cute::copy(gmem_tiled_copy_O, tOsO, tOrO);
                // Signal to the last warp that we're done reading from sO
                cutlass::arch::NamedBarrier::arrive(NumEpilogueThreads + cutlass::NumThreadsPerWarp,
                                                    cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
                // Construct identity layout for sO
                Tensor cO = cute::make_identity_tensor(select<0, 2>(TileShape_MNK{}));  // (BLK_M,BLK_K) -> (blk_m,blk_k)
                // Repeat the partitioning with identity layouts
                Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
                Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOsO)));
                #pragma unroll
                for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(_0{}, _0{}, k)) < get<1>(params.shape_O); }
                if constexpr (!PackGQA) {
                    Tensor gO = local_tile(mO, select<0, 2>(TileShape_MNK{}), make_coord(m_block, _0{}));  // (M, K)
                    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
                    // Clear_OOB_K must be false since we don't want to write zeros to gmem
                    flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
                        gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, seqlen_o - m_block * kBlockM
                    );
                } else {
                    // If PackGQA, we split the work of compute O_ptr among threads in the same row
                    static constexpr int kOPtrPerThread = cute::ceil_div(size<1>(tOcO), kGmemThreadsPerRow);
                    Tensor tPrOPtr = make_tensor<Element*>(Shape<Int<kOPtrPerThread>>{});
                    #pragma unroll
                    for (int i = 0; i < kOPtrPerThread; ++i) {
                        int const row = i * NumEpilogueThreads + (thread_idx % kGmemThreadsPerRow) * (NumEpilogueThreads / kGmemThreadsPerRow) + (thread_idx / kGmemThreadsPerRow);
                        int const idx = m_block * kBlockM + row;
                        int m_idx, h_idx;
                        m_idx = params.qhead_per_khead_divmod.divmod(h_idx, idx);
                        tPrOPtr[i] = &mO(make_coord(h_idx, m_idx), _0{});
                        // if (thread_idx < 8) { printf("thread_idx: %d, i: %d, row: %d, idx: %d, m_idx: %d, h_idx: %d\n", thread_idx, i, row, idx, m_idx, h_idx); }
                    }

                    // Tensor mO_copy = cute::tiled_divide(mO, Shape<_1, Int<kGmemElemsPerStore>>{});
                    // if (threadIdx.x == 128) { print(mO); printf("\n"); print(mO_copy); printf("\n"); print(tOrO); printf("\n"); print(sO_pi); printf("\n"); print(tOsO); printf("\n"); }
                    #pragma unroll
                    for (int m = 0; m < size<1>(tOrO); ++m) {
                        int idx = m_block * kBlockM + get<0>(tOcO(_0{}, m, _0{}));
                        Element* o_ptr_cur = reinterpret_cast<Element*>(__shfl_sync(0xffffffff, reinterpret_cast<uint64_t>(tPrOPtr[m / kGmemThreadsPerRow]), m % kGmemThreadsPerRow, kGmemThreadsPerRow));
                        if (idx < seqlen_o * qhead_per_khead) {
                            // int m_idx, h_idx;
                            // m_idx = params.qhead_per_khead_divmod.divmod(h_idx, idx);
                            Tensor mO_cur = make_tensor(make_gmem_ptr(o_ptr_cur), Shape<Int<kHeadDim>>{});
                            Tensor mO_cur_copy = cute::tiled_divide(mO_cur, Shape<Int<kGmemElemsPerStore>>{});
                            #pragma unroll
                            for (int k = 0; k < size<2>(tOrO); ++k) {
                                int ki = get<1>(tOcO(_0{}, _0{}, k)) / kGmemElemsPerStore;
                                if (tOpO(k)) {
                                    // cute::copy(gmem_tiled_copy_O, tOrO(_, m, k), mO_copy(_, make_coord(h_idx, m_idx), ki));
                                    cute::copy(gmem_tiled_copy_O, tOrO(_, m, k), mO_cur_copy(_, ki));
                                }
                            }
                        }
                    }
                }
                // Last warp needs to wait for everyone to finish reading from sO, which it is the warp
                // that will arrive on barrier_O in the mma of the next iteration.
                int warp_idx_sync = __shfl_sync(0xffffffff, thread_idx / cutlass::NumThreadsPerWarp, 0);
                if (warp_idx_sync == NumEpilogueThreads / cutlass::NumThreadsPerWarp - 1) {
                    cutlass::arch::NamedBarrier::sync(NumEpilogueThreads + cutlass::NumThreadsPerWarp,
                                                      cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
                }
            } else {
                static constexpr int kGmemElemsPerStoreDirect = 2;
                cute::Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element> gmem_copy_direct;
                // Reshape acc from ((2, 2, V), MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, V, MMA_N))
                Tensor tOrO_rowcol = make_tensor(tOrO_out.data(), flash::convert_layout_acc_rowcol(tOrO.layout()));
                Tensor tOrO_copy = cute::tiled_divide(tOrO_rowcol, Shape<_1, Int<kGmemElemsPerStoreDirect>>{});
                Tensor mO_copy = cute::tiled_divide(mO, Shape<_1, Int<kGmemElemsPerStoreDirect>>{});
                // taccOcO has shape ((2, 2, V), MMA_M, MMA_K), we only take only the row indices.
                Tensor taccOcO_col = taccOcO(make_coord(_, _0{}, _), _0{}, _);
                Element* ptr_O;
                // Split the work of computing O_ptr among threads in the same row
                if constexpr (PackGQA) { ptr_O = &mO(make_coord(mma_h_idx, mma_m_idx), _0{}); }
                #pragma unroll
                for (int m = 0; m < size(taccOcO_row); ++m) {
                    int row = get<0>(taccOcO_row(m)) + m_block * kBlockM;
                    if constexpr (!PackGQA) {
                        if (row < seqlen_o) {
                            #pragma unroll
                            for (int k = 0; k < size(taccOcO_col) / kGmemElemsPerStoreDirect; ++k) {
                                int col = get<1>(taccOcO_col(k * kGmemElemsPerStoreDirect));
                                if (col < get<1>(params.shape_O)) {
                                    cute::copy(gmem_copy_direct,
                                            tOrO_copy(_, m, k), mO_copy(_, row, col / kGmemElemsPerStoreDirect));
                                }
                            }
                        }
                    } else {
                        Element* o_ptr_cur = reinterpret_cast<Element*>(__shfl_sync(0xffffffff, reinterpret_cast<uint64_t>(ptr_O), m % kMmaThreadsPerRow, kMmaThreadsPerRow));
                        if (row < seqlen_o * qhead_per_khead) {
                            // int m_idx, h_idx;
                            // if constexpr (PackGQA) { m_idx = params.qhead_per_khead_divmod.divmod(h_idx, row); }
                            // auto row_coord = cute::conditional_return<!PackGQA>(row, make_coord(h_idx, m_idx));
                            Tensor mO_cur = make_tensor(make_gmem_ptr(o_ptr_cur), Shape<Int<kHeadDim>>{});
                            Tensor mO_cur_copy = cute::tiled_divide(mO_cur, Shape<Int<kGmemElemsPerStoreDirect>>{});
                            #pragma unroll
                            for (int k = 0; k < size(taccOcO_col) / kGmemElemsPerStoreDirect; ++k) {
                                int col = get<1>(taccOcO_col(k * kGmemElemsPerStoreDirect));
                                if (col < get<1>(params.shape_O)) {
                                    cute::copy(gmem_copy_direct,
                                            tOrO_copy(_, m, k), mO_cur_copy(_, col / kGmemElemsPerStoreDirect));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    CUTLASS_DEVICE void
    store_tail() {
        if constexpr (!Varlen && Use_smem && !PackGQA) { tma_store_wait<0>(); }
    }

    // Write 0 to output and -inf to LSE
    CUTLASS_DEVICE void
    store_zero(
         Params const& params,
         int thread_idx,
         cute::tuple<int32_t, int32_t, int32_t, int32_t> const& block_coord
         ) {
        static constexpr int kBlockM = get<0>(TileShape_MNK{});
        auto [m_block, bidh, bidb, split_idx] = block_coord;
        bool const is_varlen = Varlen && params.cu_seqlens;
        int offset_o = !is_varlen ? 0 : params.cu_seqlens[bidb];
        int seqlen_o = !Varlen ? size<0>(params.shape_O) : (params.seqused ? params.seqused[bidb] : (params.cu_seqlens ? params.cu_seqlens[bidb + 1] - offset_o : size<0>(params.shape_O)));
        int qhead_per_khead = !PackGQA ? 1 : params.qhead_per_khead_divmod.divisor;
        Tensor mO = make_tensor(make_gmem_ptr(params.ptr_O + offset_o * get<0>(params.stride_O)), params.shape_O_packed, params.stride_O_packed)(_, _, bidh, !is_varlen ? bidb : 0, split_idx);
        Tensor mLSE = make_tensor(make_gmem_ptr(params.ptr_LSE + offset_o * get<0>(params.stride_LSE)), params.shape_LSE_packed, params.stride_LSE_packed)(_, bidh, !is_varlen ? bidb : 0, split_idx);
        Tensor gLSE = local_tile(mLSE, Shape<Int<kBlockM>>{}, make_coord(m_block));

        GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(thread_idx);
        // Construct identity layout for gO
        Tensor cO = cute::make_identity_tensor(select<0, 2>(TileShape_MNK{}));  // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOcO)));
        #pragma unroll
        for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(_0{}, _0{}, k)) < get<1>(params.shape_O); }
        if constexpr (!PackGQA) {
            Tensor gO = local_tile(mO, select<0, 2>(TileShape_MNK{}), make_coord(m_block, _0{}));  // (M, K)
            Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
            Tensor tOrO = make_fragment_like(tOgO);
            cute::clear(tOrO);
            // Clear_OOB_K must be false since we don't want to write zeros to gmem
            flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
                gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, seqlen_o - m_block * kBlockM
            );
        } else {
            // If PackGQA, we split the work of compute O_ptr among threads in the same row
            // TODO: check correctness
            static constexpr int kOPtrPerThread = cute::ceil_div(size<1>(tOcO), kGmemThreadsPerRow);
            Tensor tPrOPtr = make_tensor<Element*>(Shape<Int<kOPtrPerThread>>{});
            #pragma unroll
            for (int i = 0; i < kOPtrPerThread; ++i) {
                int const row = i * NumEpilogueThreads + (thread_idx % kGmemThreadsPerRow) * (NumEpilogueThreads / kGmemThreadsPerRow) + (thread_idx / kGmemThreadsPerRow);
                int const idx = m_block * kBlockM + row;
                int m_idx, h_idx;
                m_idx = params.qhead_per_khead_divmod.divmod(h_idx, idx);
                tPrOPtr[i] = &mO(make_coord(h_idx, m_idx), _0{});
            }
            // Tensor mO_copy = cute::tiled_divide(mO, Shape<_1, Int<kGmemElemsPerStore>>{});
            Tensor tOrO_zero = make_fragment_like<Element>(Shape<_1, Int<kGmemElemsPerStore>>{});
            clear(tOrO_zero);
            #pragma unroll
            for (int m = 0; m < size<1>(tOcO); ++m) {
                int idx = m_block * kBlockM + get<0>(tOcO(_0{}, m, _0{}));
                Element* o_ptr_cur = reinterpret_cast<Element*>(__shfl_sync(0xffffffff, reinterpret_cast<uint64_t>(tPrOPtr[m / kGmemThreadsPerRow]), m % kGmemThreadsPerRow, kGmemThreadsPerRow));
                if (idx < seqlen_o * qhead_per_khead) {
                    // int m_idx, h_idx;
                    // m_idx = params.qhead_per_khead_divmod.divmod(h_idx, idx);
                    Tensor mO_cur = make_tensor(make_gmem_ptr(o_ptr_cur), Shape<Int<kHeadDim>>{});
                    Tensor mO_cur_copy = cute::tiled_divide(mO_cur, Shape<Int<kGmemElemsPerStore>>{});
                    #pragma unroll
                    for (int k = 0; k < size<2>(tOcO); ++k) {
                        if (tOpO(k)) {
                            int ki = get<1>(tOcO(_0{}, _0{}, k)) / kGmemElemsPerStore;
                            // cute::copy(gmem_tiled_copy_O, tOrO_zero, mO_copy(_, make_coord(h_idx, m_idx), ki));
                            cute::copy(gmem_tiled_copy_O, tOrO_zero, mO_cur_copy(_, ki));
                        }
                    }
                }
            }
        }

        static_assert(kBlockM <= NumEpilogueThreads);
        if (thread_idx < kBlockM) {
            const int row = m_block * kBlockM + thread_idx;
            if constexpr (!PackGQA) {
                if (row < seqlen_o) { mLSE(row) = -INFINITY; }
            } else {
                if (row < seqlen_o * qhead_per_khead) {
                    int m_idx, h_idx;
                    m_idx = params.qhead_per_khead_divmod.divmod(h_idx, row);
                    // mLSE shape shape ((qhead_per_khead, seqlen_q)) and it's unhappy with just 1 "make_coord"
                    mLSE(make_coord(make_coord(h_idx, m_idx))) = -INFINITY;
                }
            }
        }
    }

};

} // namespace flash
