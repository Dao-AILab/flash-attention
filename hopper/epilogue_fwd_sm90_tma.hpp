/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cutlass/cutlass.h>
#include "cute/tensor.hpp"

#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"

#include "named_barrier.hpp"
#include "utils.h"

namespace flash {

using namespace cute;

template <class TileShape_MNK_, class Element_, int NumEpilogueThreads_, bool Varlen_, bool FP8PermuteCol=false>
struct CollectiveEpilogueFwd {

    using TileShape_MNK = TileShape_MNK_;
    using Element = Element_;
    static constexpr int NumEpilogueThreads = NumEpilogueThreads_;
    static constexpr bool Varlen = Varlen_;

    static constexpr int kHeadDim = get<2>(TileShape_MNK{});
    static constexpr int kBlockM = get<0>(TileShape_MNK{});

    using GmemTiledCopyOTMA = cute::SM90_TMA_STORE;

    // These are for storing the output tensor without TMA (e.g., for setting output to zero)
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(get<2>(TileShape_MNK{}) % kGmemElemsPerLoad == 0, "Headdim must be a multiple of kGmemElemsPerLoad");
    static constexpr int kGmemThreadsPerRow = cutlass::gcd(kHeadDim / kGmemElemsPerLoad, NumEpilogueThreads);
    static_assert(NumEpilogueThreads % kGmemThreadsPerRow == 0, "NumEpilogueThreads must be a multiple of kGmemThreadsPerRow");
    using GmemLayoutAtom = Layout<Shape <Int<NumEpilogueThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                  Stride<Int<kGmemThreadsPerRow>, _1>>;
    using GmemTiledCopyO = decltype(
        make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));  // Val layout, 8 or 16 vals per store

    using SmemLayoutAtomO = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutO = decltype(tile_to_shape(SmemLayoutAtomO{}, select<0, 2>(TileShape_MNK{})));

    using ShapeO = cute::Shape<int32_t, int32_t, int32_t, int32_t>;  // (seqlen_q, d, head, batch)
    using StrideO = cute::Stride<int64_t, _1, int64_t, int64_t>;
    using StrideLSE = cute::Stride<_1, int64_t, int64_t>;            // (seqlen_q, head, batch)

    // cute::SM90_U32x4_STSM_N if Element size is 2 bytes (fp16, bf16)
    using CopyOpR2S = decltype(cutlass::epilogue::collective::detail::sm90_get_smem_store_op_for_accumulator<StrideO, Element>());
    using SmemCopyAtomO = Copy_Atom<CopyOpR2S, Element>;


    struct TensorStorage : cute::aligned_struct<128> {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutO>> smem_o;
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
        int const* cu_seqlens = nullptr;
        int const* seqused = nullptr;
    };

    // Device side kernel params
    struct Params {
        Element* ptr_O;
        ShapeO const shape_O;
        StrideO const stride_O;
        float* ptr_LSE;
        StrideLSE const stride_LSE;
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
        if constexpr (Varlen) {
            assert(args.cu_seqlens != nullptr);
        }
        return {args.ptr_O, args.shape_O, args.stride_O, args.ptr_LSE, args.stride_LSE, tma_store_O, args.cu_seqlens, args.seqused};
    }

    /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
    CUTLASS_DEVICE
    static void prefetch_tma_descriptors(Params const& params) {
        if constexpr (!Varlen) {
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
          cute::tuple<int32_t, int32_t, int32_t> const& block_coord
          ) {

        auto [m_block, bidh, bidb] = block_coord;
        Tensor sO = make_tensor(make_smem_ptr(shared_storage.tensors.epilogue.smem_o.data()), SmemLayoutO{});

        // Tensor tOrO_out = flash::convert_type<Element>(tOrO);
        Tensor tOrO_out = flash::convert_type_safe<Element>(tOrO);
        if constexpr (FP8PermuteCol) { flash::permute_output_fp8_fp16(tOrO_out); }
        // Make sure all WGs have finished reading V
        cutlass::arch::NamedBarrier::sync(NumEpilogueThreads, static_cast<int>(FwdNamedBarriers::ValueEmpty) /*id*/);

        auto smem_tiled_copy_O = make_tiled_copy_C(SmemCopyAtomO{}, tiled_mma);
        auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(thread_idx);
        Tensor taccOrO = smem_thr_copy_O.retile_S(tOrO_out);        // ((Atom,AtomNum), MMA_M, MMA_N)
        Tensor taccOsO = smem_thr_copy_O.partition_D(sO);     // ((Atom,AtomNum),PIPE_M,PIPE_N)
        cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

        if constexpr (!Varlen) {
            cutlass::arch::fence_view_async_shared(); // ensure smem writes are visible to TMA
            cutlass::arch::NamedBarrier::arrive(NumEpilogueThreads + cutlass::NumThreadsPerWarp,
                                                cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
        } else {
            cutlass::arch::NamedBarrier::sync(NumEpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
        }

        int offset_o = !Varlen ? 0 : params.cu_seqlens[bidb];
        int seqlen_o = !Varlen ? get<0>(params.shape_O) : (params.seqused ? params.seqused[bidb] : params.cu_seqlens[bidb + 1] - offset_o);

        auto shape_LSE = select<0, 2, 3>(params.shape_O);
        Tensor mLSE = make_tensor(make_gmem_ptr(params.ptr_LSE), shape_LSE, params.stride_LSE)(_, bidh, !Varlen ? bidb : 0);
        Tensor gLSE = local_tile(cute::domain_offset(make_coord(offset_o), mLSE), Shape<Int<kBlockM>>{}, make_coord(m_block));
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
                if (row < seqlen_o - m_block * kBlockM) { gLSE(row) = lse(mi); }
            }
        }

        if constexpr (!Varlen) {
            Tensor mO = params.tma_store_O.get_tma_tensor(params.shape_O);
            Tensor gO = local_tile(mO(_, _, bidh, bidb), select<0, 2>(TileShape_MNK{}), make_coord(m_block, _0{}));  // (M, K)
            auto block_tma_O = params.tma_store_O.get_slice(_0{});
            Tensor tOgO = block_tma_O.partition_D(gO);  // (TMA, TMA_M, TMA_K)
            Tensor tOsO = block_tma_O.partition_S(sO); // (TMA, TMA_M, TMA_K)
            int warp_idx_sync = __shfl_sync(0xffffffff, thread_idx / cutlass::NumThreadsPerWarp, 0);
            if (warp_idx_sync == NumEpilogueThreads / cutlass::NumThreadsPerWarp - 1) {
                cutlass::arch::NamedBarrier::sync(NumEpilogueThreads + cutlass::NumThreadsPerWarp,
                                                  cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
                int lane_predicate = cute::elect_one_sync();
                if (lane_predicate) {
                    cute::copy(params.tma_store_O, tOsO, tOgO);
                    tma_store_arrive();
                }
            }
        } else {  // Don't use TMA since we don't want to overwrite the output of another sequence
            Tensor mO = make_tensor(make_gmem_ptr(params.ptr_O), params.shape_O, params.stride_O)(_, _, bidh, params.cu_seqlens == nullptr ? bidb : 0);
            Tensor gO = local_tile(cute::domain_offset(make_coord(offset_o, _0{}), mO), select<0, 2>(TileShape_MNK{}), make_coord(m_block, _0{}));  // (M, K)
            GmemTiledCopyO gmem_tiled_copy_O;
            auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(thread_idx);
            Tensor tOsO = gmem_thr_copy_O.partition_S(sO);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
            Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
            Tensor tOrO = make_fragment_like(tOsO);
            cute::copy(gmem_tiled_copy_O, tOsO, tOrO);
            // Construct identity layout for sO
            Tensor cO = cute::make_identity_tensor(select<0, 2>(TileShape_MNK{}));  // (BLK_M,BLK_K) -> (blk_m,blk_k)
            // Repeat the partitioning with identity layouts
            Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
            Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(_0{}, _0{}, k)) < get<1>(params.shape_O); }
            // Clear_OOB_K must be false since we don't want to write zeros to gmem
            flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
                gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, seqlen_o - m_block * kBlockM
            );
        }
    }

    CUTLASS_DEVICE void
    store_tail() {
        if constexpr (!Varlen) { tma_store_wait<0>(); }
    }

    // Write 0 to output and -inf to LSE
    CUTLASS_DEVICE void
    store_zero(
         Params const& params,
         int thread_idx,
         cute::tuple<int32_t, int32_t, int32_t> const& block_coord
         ) {
        static constexpr int kBlockM = get<0>(TileShape_MNK{});
        auto [m_block, bidh, bidb] = block_coord;
        int offset_o = !Varlen ? 0 : params.cu_seqlens[bidb];
        int seqlen_o = !Varlen ? get<0>(params.shape_O) : (params.seqused ? params.seqused[bidb] : params.cu_seqlens[bidb + 1] - offset_o);
        Tensor mO = make_tensor(make_gmem_ptr(params.ptr_O), params.shape_O, params.stride_O)(_, _, bidh, !Varlen ? bidb : 0);
        Tensor gO = local_tile(cute::domain_offset(make_coord(offset_o, _0{}), mO), select<0, 2>(TileShape_MNK{}), make_coord(m_block, _0{}));  // (M, K)
        auto shape_LSE = select<0, 2, 3>(params.shape_O);
        Tensor mLSE = make_tensor(make_gmem_ptr(params.ptr_LSE), shape_LSE, params.stride_LSE)(_, bidh, !Varlen ? bidb : 0);
        Tensor gLSE = local_tile(cute::domain_offset(make_coord(offset_o), mLSE), Shape<Int<kBlockM>>{}, make_coord(m_block));

        GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(thread_idx);
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
        Tensor tOrO = make_fragment_like(tOgO);
        clear(tOrO);
        // Construct identity layout for gO
        Tensor cO = cute::make_identity_tensor(select<0, 2>(TileShape_MNK{}));  // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
        #pragma unroll
        for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(_0{}, _0{}, k)) < get<1>(params.shape_O); }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, seqlen_o - m_block * kBlockM
        );
        static_assert(kBlockM <= NumEpilogueThreads);
        if (thread_idx < seqlen_o - m_block * kBlockM && thread_idx < kBlockM) { gLSE(thread_idx) = -INFINITY; }
    }

};

} // namespace flash
