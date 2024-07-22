/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cutlass/cutlass.h>
#include "cute/tensor.hpp"

#include "cutlass/gemm/collective/collective_builder.hpp"

#include "named_barrier.hpp"
#include "utils.h"

namespace flash {

using namespace cute;

// template <int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, typename Element_>
template <typename Ktraits>
struct CollectiveEpilogueFwd {

    using Element = typename Ktraits::Element;
    static constexpr int kBlockM = Ktraits::kBlockM;
    static constexpr int kBlockN = Ktraits::kBlockN;
    static constexpr int kHeadDim = Ktraits::kHeadDim;
    // using Element = Element_;
    // static constexpr int kBlockM = kBlockM_;
    // static constexpr int kBlockN = kBlockN_;
    // static constexpr int kHeadDim = kHeadDim_;
    using TileShape_MNK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;

    // static constexpr int kNWarps = kNWarps_;
    static constexpr int kNWarps = Ktraits::kNWarps;
    static constexpr int kNThreads = kNWarps * cutlass::NumThreadsPerWarp;
    static constexpr bool Is_WS = kNWarps >= 12;

    static constexpr int NumCopyThreads = !Is_WS ? 0 : cutlass::NumThreadsPerWarpGroup;
    static constexpr int NumMmaThreads = kNThreads - NumCopyThreads;

    using GmemTiledCopyOTMA = cute::SM90_TMA_STORE;

    // These are for storing the output tensor without TMA (e.g., for setting output to zero)
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");
    static constexpr int kGmemThreadsPerRow = kHeadDim / kGmemElemsPerLoad;
    static_assert(NumMmaThreads % kGmemThreadsPerRow == 0, "NumMmaThreads must be a multiple of kGmemThreadsPerRow");
    using GmemLayoutAtom = Layout<Shape <Int<NumMmaThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                  Stride<Int<kGmemThreadsPerRow>, _1>>;
    using GmemTiledCopyO = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));  // Val layout, 8 or 16 vals per store

    using SmemLayoutAtomO = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutO = decltype(tile_to_shape(SmemLayoutAtomO{}, select<0, 2>(TileShape_MNK{})));

    using SmemCopyAtomO = Copy_Atom<cute::SM90_U32x4_STSM_N, Element>;
    using SharedStorage = cute::array_aligned<Element, cute::cosize_v<SmemLayoutO>>;

    using ShapeO = cute::Shape<int32_t, int32_t, int32_t, int32_t>;  // (seqlen_q, d, head, batch)
    using StrideO = cute::Stride<int64_t, _1, int64_t, int64_t>;
    using StrideLSE = cute::Stride<_1, int64_t, int64_t>;            // (seqlen_q, head, batch)

    using TMA_O = decltype(make_tma_copy(
        GmemTiledCopyOTMA{},
        make_tensor(make_gmem_ptr(static_cast<Element*>(nullptr)), repeat_like(StrideO{}, int32_t(0)), StrideO{}),
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
    };

    // Device side kernel params
    struct Params {
        Element* ptr_O;
        ShapeO const shape_O;
        StrideO const stride_O;
        float* ptr_LSE;
        StrideLSE const stride_LSE;
        TMA_O tma_store_O;
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
        return {args.ptr_O, args.shape_O, args.stride_O, args.ptr_LSE, args.stride_LSE, tma_store_O};
    }

    /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
    CUTLASS_DEVICE
    static void prefetch_tma_descriptors(Params const& epilogue_params) {
        cute::prefetch_tma_descriptor(epilogue_params.tma_store_O.get_tma_descriptor());
    }

    template <typename SharedStorage, typename FrgTensorO, typename FrgTensorLSE, typename TiledMma>
    CUTLASS_DEVICE void
    store(Params const& epilogue_params,
          FrgTensorO const& tOrO,
          FrgTensorLSE const& lse,
          SharedStorage& shared_storage,
          TiledMma tiled_mma,
          int thread_idx,
          cute::tuple<int32_t, int32_t, int32_t> const& block_coord
          ) {

        auto [m_block, bidh, bidb] = block_coord;
        Tensor sO = make_tensor(make_smem_ptr(shared_storage.smem_o.data()), SmemLayoutO{});
        auto smem_tiled_copy_O = make_tiled_copy_C(SmemCopyAtomO{}, tiled_mma);
        auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(thread_idx);

        Tensor tOrO_out = flash::convert_type<Element>(tOrO);
        Tensor taccOrO = smem_thr_copy_O.retile_S(tOrO_out);        // ((Atom,AtomNum), MMA_M, MMA_N)
        Tensor taccOsO = smem_thr_copy_O.partition_D(sO);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

        // Make sure all WGs have finished reading V
        cutlass::arch::NamedBarrier::sync(NumMmaThreads, static_cast<int>(FwdNamedBarriers::ValueEmpty) /*id*/);
        cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);
        cutlass::arch::fence_view_async_shared(); // ensure smem writes are visible to TMA
        cutlass::arch::NamedBarrier::arrive(NumMmaThreads + cutlass::NumThreadsPerWarp,
                                            cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);

        Tensor mO = epilogue_params.tma_store_O.get_tma_tensor(epilogue_params.shape_O);
        Tensor gO = local_tile(mO(_, _, bidh, bidb), select<0, 2>(TileShape_MNK{}), make_coord(m_block, _0{}));  // (M, K)
        auto block_tma_O = epilogue_params.tma_store_O.get_slice(_0{});
        Tensor tOgO = block_tma_O.partition_D(gO);  // (TMA, TMA_M, TMA_K)
        Tensor tOsO = block_tma_O.partition_S(sO); // (TMA, TMA_M, TMA_K)

        auto shape_LSE = select<0, 2, 3>(epilogue_params.shape_O);
        Tensor mLSE = make_tensor(make_gmem_ptr(epilogue_params.ptr_LSE), shape_LSE, epilogue_params.stride_LSE);
        Tensor gLSE = local_tile(mLSE(_, bidh, bidb), Shape<Int<kBlockM>>{}, make_coord(m_block));

        Tensor caccO = cute::make_identity_tensor(select<0, 2>(TileShape_MNK{}));
        auto thread_mma = tiled_mma.get_thread_slice(thread_idx);
        Tensor taccOcO = thread_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
        static_assert(decltype(size<0, 0>(taccOcO))::value == 2);
        static_assert(decltype(size<0, 1>(taccOcO))::value == 2);
        // taccOcO has shape ((2, 2, V), MMA_M, MMA_K), we only take only the row indices.
        Tensor taccOcO_row = taccOcO(make_coord(_0{}, _, _0{}), _, _0{});
        CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));                     // MMA_M
        if (get<1>(taccOcO_row(_0{})) == 0) {
            #pragma unroll
            for (int mi = 0; mi < size(lse); ++mi) {
                const int row = get<0>(taccOcO_row(mi));
                if (row < get<0>(shape_LSE) - m_block * kBlockM) { gLSE(row) = lse(mi); }
            }
        }

        if (cutlass::canonical_warp_idx_sync() == kNWarps - 1) {
            cutlass::arch::NamedBarrier::sync(NumMmaThreads + cutlass::NumThreadsPerWarp,
                                              cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
            int const lane_predicate = cute::elect_one_sync();
            if (lane_predicate) {
                cute::copy(epilogue_params.tma_store_O, tOsO, tOgO);
                tma_store_arrive();
            }
        }
    }

    CUTLASS_DEVICE void
    store_tail() {
        tma_store_wait<0>();
    }

    // Write 0 to output and -inf to LSE
    CUTLASS_DEVICE void
    store_zero(
         Params const& epilogue_params,
         int thread_idx,
         cute::tuple<int32_t, int32_t, int32_t> const& block_coord
         ) {
        auto [m_block, bidh, bidb] = block_coord;
        Tensor mO = make_tensor(make_gmem_ptr(epilogue_params.ptr_O), epilogue_params.shape_O, epilogue_params.stride_O);
        Tensor gO = local_tile(mO(_, _, bidh, bidb), select<0, 2>(TileShape_MNK{}), make_coord(m_block, _0{}));  // (M, K)
        auto shape_LSE = select<0, 2, 3>(epilogue_params.shape_O);
        Tensor mLSE = make_tensor(make_gmem_ptr(epilogue_params.ptr_LSE), shape_LSE, epilogue_params.stride_LSE);
        Tensor gLSE = local_tile(mLSE(_, bidh, bidb), Shape<Int<kBlockM>>{}, make_coord(m_block));

        GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(thread_idx);
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
        Tensor tOrO = make_fragment_like(tOgO);
        clear(tOrO);
        // Construct identity layout for sO
        Tensor cO = cute::make_identity_tensor(select<0, 2>(TileShape_MNK{}));  // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
        #pragma unroll
        for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(_0{}, _0{}, k)) < get<1>(epilogue_params.shape_O); }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, get<0>(epilogue_params.shape_O) - m_block * kBlockM
        );
        static_assert(kBlockM <= NumMmaThreads);
        if (thread_idx < get<0>(shape_LSE) - m_block * kBlockM) { gLSE(thread_idx) = INFINITY; }
    }

};

} // namespace flash
