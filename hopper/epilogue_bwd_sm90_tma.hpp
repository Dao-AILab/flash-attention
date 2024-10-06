/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/barrier.h>
#include "cute/tensor.hpp"

#include "cutlass/gemm/collective/collective_builder.hpp"

#include "named_barrier.hpp"
#include "utils.h"

namespace flash {

using namespace cute;

template <class TileShape_MNK_, class Element_, int NumEpilogueThreads_, bool Varlen_,
          bool dKV_swapAB_, int AtomLayoutKdKV=1>
struct CollectiveEpilogueBwd {

    using TileShape_MNK = TileShape_MNK_;
    using Element = Element_;
    static constexpr int NumEpilogueThreads = NumEpilogueThreads_;
    static constexpr bool Varlen = Varlen_;
    static constexpr bool dKV_swapAB = dKV_swapAB_;

    using GmemTiledCopydKVTMA = cute::SM90_TMA_STORE;

    // These are for storing the output tensor without TMA (e.g., for setting output to zero)
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(get<2>(TileShape_MNK{}) % kGmemElemsPerLoad == 0, "Headdim must be a multiple of kGmemElemsPerLoad");
    static constexpr int kHeadDim = get<2>(TileShape_MNK{});
    static constexpr int kGmemThreadsPerRow = cutlass::gcd(kHeadDim / kGmemElemsPerLoad, NumEpilogueThreads);
    static_assert(NumEpilogueThreads % kGmemThreadsPerRow == 0, "NumEpilogueThreads must be a multiple of kGmemThreadsPerRow");
    using GmemLayoutAtom = Layout<Shape <Int<NumEpilogueThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                  Stride<Int<kGmemThreadsPerRow>, _1>>;
    using GmemTiledCopydKV = decltype(
        make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));  // Val layout, 8 or 16 vals per store

    using SmemLayoutAtomdKVTMA = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
                                          // TODO: do we have to change this if dKV_swapAB is true?
                                          decltype(cute::get<1>(TileShape_MNK{})), Int<CUTE_STATIC_V(cute::get<2>(TileShape_MNK{})) / AtomLayoutKdKV>>());
    using SmemLayoutdKVTMA = decltype(tile_to_shape(SmemLayoutAtomdKVTMA{}, select<1, 2>(TileShape_MNK{})));
    using SmemLayoutdKVtTMA =
        decltype(cute::composition(SmemLayoutdKVTMA{},
                                   make_layout(make_shape(get<2>(TileShape_MNK{}), get<1>(TileShape_MNK{})),
                                               make_stride(decltype(get<1>(TileShape_MNK{})){}, _1{}))));

    // If we don't use TMA
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : (kHeadDim % 32 == 0 ? 32 : 16);
    static constexpr int kSwizzle = kBlockKSmem == 64 ? 3 : (kBlockKSmem == 32 ? 2 : 1);
    using SmemLayoutAtomdKVSTG =
        decltype(composition(Swizzle<kSwizzle, 3, 3>{},
                             Layout<Shape<Int<8>, Int<kBlockKSmem>>,
                             Stride<Int<kBlockKSmem>, _1>>{}));

    using SmemLayoutAtomdKV = std::conditional_t<!Varlen, SmemLayoutAtomdKVTMA, SmemLayoutAtomdKVSTG>;
    using SmemLayoutdKV = decltype(tile_to_shape(SmemLayoutAtomdKV{}, select<1, 2>(TileShape_MNK{})));
    using SmemLayoutdKVt =
        decltype(cute::composition(SmemLayoutdKV{},
                                   make_layout(make_shape(get<2>(TileShape_MNK{}), get<1>(TileShape_MNK{})),
                                               make_stride(decltype(get<1>(TileShape_MNK{})){}, _1{}))));

    using SmemCopyAtomdKV = Copy_Atom<
        std::conditional_t<!dKV_swapAB, cute::SM90_U32x4_STSM_N, cute::SM90_U16x8_STSM_T>,
            Element>;

    static constexpr size_t SmemAlignmentdKV = cutlass::detail::alignment_for_swizzle(SmemLayoutdKV{});
    static_assert(SmemAlignmentdKV >= 128, "Require at least 128B alignment");

    struct TensorStorage : cute::aligned_struct<SmemAlignmentdKV> {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutdKV>, SmemAlignmentdKV> smem_dk;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutdKV>, SmemAlignmentdKV> smem_dv;
    };

    using ShapedKV = cute::Shape<int32_t, int32_t, int32_t, int32_t>;  // (seqlen_q, d, head, batch)
    using StridedKV = cute::Stride<int64_t, _1, int64_t, int64_t>;

    using TMA_dKV = decltype(make_tma_copy(
        GmemTiledCopydKVTMA{},
        make_tensor(make_gmem_ptr(static_cast<Element*>(nullptr)), ShapedKV{}, StridedKV{}),
        SmemLayoutdKVTMA{},
        select<1, 2>(TileShape_MNK{}),
        _1{}));  // no mcast for dKV

    // Host side kernel arguments
    struct Arguments {
        Element* ptr_dK;
        ShapedKV const shape_dK;
        StridedKV const stride_dK;
        Element* ptr_dV;
        StridedKV const stride_dV;
        int const num_heads_q;
        int* dk_semaphore;
        int* dv_semaphore;
        int const* cu_seqlens;
        int const* seqused;
    };

    // Device side kernel params
    struct Params {
        Element* ptr_dK;
        ShapedKV const shape_dK;
        StridedKV const stride_dK;
        Element* ptr_dV;
        StridedKV const stride_dV;
        TMA_dKV tma_store_dK, tma_store_dV;
        int const* cu_seqlens = nullptr;
        int const* seqused = nullptr;
    };

    static Params
    to_underlying_arguments(Arguments const& args) {
        if constexpr (Varlen) {
            assert (args.cu_seqlens != nullptr);
        }
        Tensor mdK = make_tensor(make_gmem_ptr(args.ptr_dK), args.shape_dK, args.stride_dK);
        Tensor mdV = make_tensor(make_gmem_ptr(args.ptr_dV), args.shape_dK, args.stride_dV);
        TMA_dKV tma_store_dK = make_tma_copy(
            GmemTiledCopydKVTMA{},
            mdK,
            SmemLayoutdKVTMA{},
            select<1, 2>(TileShape_MNK{}),
            _1{}); // no mcast for dKV
        TMA_dKV tma_store_dV = make_tma_copy(
            GmemTiledCopydKVTMA{},
            mdV,
            SmemLayoutdKVTMA{},
            select<1, 2>(TileShape_MNK{}),
            _1{}); // no mcast for dKV
        return {args.ptr_dK, args.shape_dK, args.stride_dK, args.ptr_dV, args.stride_dV,
                tma_store_dK, tma_store_dV, args.cu_seqlens, args.seqused};
    }

    /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
    CUTLASS_DEVICE
    static void prefetch_tma_descriptors(Params const& params) {
        if constexpr (!Varlen) {
            cute::prefetch_tma_descriptor(params.tma_store_dK.get_tma_descriptor());
            cute::prefetch_tma_descriptor(params.tma_store_dV.get_tma_descriptor());
        }
    }

    template <typename SharedStorage, typename FrgTensorO, typename TiledMma>
    CUTLASS_DEVICE void
    store(Params const& params,
          FrgTensorO const& tdKrdK,
          FrgTensorO const& tdVrdV,
          SharedStorage& shared_storage,
          TiledMma tiled_mma,
          int thread_idx,
          cute::tuple<int32_t, int32_t, int32_t> const& block_coord
          ) {

        auto [n_block, bidh, bidb] = block_coord;
        Tensor sdK = cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.epilogue.smem_dk.data()), SmemLayoutdKV{}));
        Tensor sdV = cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.epilogue.smem_dv.data()), SmemLayoutdKV{}));
        Tensor sdKt = cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.epilogue.smem_dk.data()), SmemLayoutdKVt{}));
        Tensor sdVt = cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.epilogue.smem_dv.data()), SmemLayoutdKVt{}));
        auto smem_tiled_copy_dKV = make_tiled_copy_C(SmemCopyAtomdKV{}, tiled_mma);
        auto smem_thr_copy_dKV = smem_tiled_copy_dKV.get_thread_slice(thread_idx);

        Tensor tdVrdV_out = flash::convert_type<Element>(tdVrdV);
        Tensor tdKrdK_out = flash::convert_type<Element>(tdKrdK);
        Tensor taccdKrdK = smem_thr_copy_dKV.retile_S(tdKrdK_out);        // ((Atom,AtomNum), MMA_M, MMA_N)
        Tensor taccdVrdV = smem_thr_copy_dKV.retile_S(tdVrdV_out);        // ((Atom,AtomNum), MMA_M, MMA_N)
        // if (blockIdx.x == 0 && threadIdx.x == 128) { print(smem_thr_copy_dKV); print(sdK); printf("\n"); print(sdKt); printf("\n"); }
        Tensor taccdKsdK = smem_thr_copy_dKV.partition_D(cute::conditional_return<!dKV_swapAB>(sdK, sdKt));     // ((Atom,AtomNum),PIPE_M,PIPE_N)
        Tensor taccdVsdV = smem_thr_copy_dKV.partition_D(cute::conditional_return<!dKV_swapAB>(sdV, sdVt));     // ((Atom,AtomNum),PIPE_M,PIPE_N)

        // Make sure all WGs have finished reading K and V
        cutlass::arch::NamedBarrier::sync(NumEpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
        cute::copy(smem_tiled_copy_dKV, taccdVrdV, taccdVsdV);
        cute::copy(smem_tiled_copy_dKV, taccdKrdK, taccdKsdK);
        if constexpr (!Varlen) {
            cutlass::arch::fence_view_async_shared(); // ensure smem writes are visible to TMA
            cutlass::arch::NamedBarrier::arrive(NumEpilogueThreads + cutlass::NumThreadsPerWarp,
                                                cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);

            Tensor mdK = params.tma_store_dK.get_tma_tensor(params.shape_dK);
            Tensor mdV = params.tma_store_dV.get_tma_tensor(params.shape_dK);
            Tensor gdK = local_tile(mdK(_, _, bidh, bidb), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{}));  // (M, K)
            Tensor gdV = local_tile(mdV(_, _, bidh, bidb), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{}));  // (M, K)
            auto block_tma_dK = params.tma_store_dK.get_slice(_0{});
            auto block_tma_dV = params.tma_store_dV.get_slice(_0{});
            Tensor tdKgdK = block_tma_dK.partition_D(gdK);  // (TMA, TMA_M, TMA_K)
            Tensor tdKsdK = block_tma_dK.partition_S(sdK); // (TMA, TMA_M, TMA_K)
            Tensor tdVgdV = block_tma_dV.partition_D(gdV);  // (TMA, TMA_M, TMA_K)
            Tensor tdVsdV = block_tma_dV.partition_S(sdV); // (TMA, TMA_M, TMA_K)
            int warp_idx_sync = __shfl_sync(0xffffffff, thread_idx / cutlass::NumThreadsPerWarp, 0);
            if (warp_idx_sync == NumEpilogueThreads / cutlass::NumThreadsPerWarp - 1) {
                cutlass::arch::NamedBarrier::sync(NumEpilogueThreads + cutlass::NumThreadsPerWarp,
                                                cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
                int const lane_predicate = cute::elect_one_sync();
                if (lane_predicate) {
                    cute::copy(params.tma_store_dV, tdVsdV, tdVgdV);
                    cute::copy(params.tma_store_dK, tdKsdK, tdKgdK);
                    tma_store_arrive();
                }
            }
            tma_store_wait<0>();
            // // Tell warp 0 that smem_k and smem_v are ready
            // cutlass::arch::NamedBarrier::arrive(NumEpilogueThreads + cutlass::NumThreadsPerWarp, static_cast<int>(BwdNamedBarriers::KVEmpty) /*id*/);

        } else {
            cutlass::arch::NamedBarrier::sync(NumEpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
            int const offset = !Varlen ? 0 : params.cu_seqlens[bidb];
            int const seqlen = !Varlen ? get<0>(params.shape_dK) : (params.seqused ? params.seqused[bidb] : params.cu_seqlens[bidb + 1] - params.cu_seqlens[bidb]);

            Tensor mdK = make_tensor(make_gmem_ptr(params.ptr_dK), params.shape_dK, params.stride_dK)(_, _, bidh, !Varlen ? bidb : 0);
            Tensor gdK = local_tile(cute::domain_offset(make_coord(offset, _0{}), mdK), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{}));  // (M, K)
            Tensor mdV = make_tensor(make_gmem_ptr(params.ptr_dV), params.shape_dK, params.stride_dV)(_, _, bidh, !Varlen ? bidb : 0);
            Tensor gdV = local_tile(cute::domain_offset(make_coord(offset, _0{}), mdV), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{}));  // (M, K)

            GmemTiledCopydKV gmem_tiled_copy_dKV;
            auto gmem_thr_copy_dKV = gmem_tiled_copy_dKV.get_thread_slice(thread_idx);
            Tensor tdKVgdV = gmem_thr_copy_dKV.partition_D(gdV);
            Tensor tdKVsdV = gmem_thr_copy_dKV.partition_S(sdV); // (TMA, TMA_M, TMA_K)
            Tensor tdKVgdK = gmem_thr_copy_dKV.partition_D(gdK);
            Tensor tdKVsdK = gmem_thr_copy_dKV.partition_S(sdK); // (TMA, TMA_M, TMA_K)
            Tensor tdKVrdV = make_fragment_like(tdKVgdV);
            Tensor tdKVrdK = make_fragment_like(tdKVgdK);
            cute::copy(gmem_tiled_copy_dKV, tdKVsdV, tdKVrdV);
            cute::copy(gmem_tiled_copy_dKV, tdKVsdK, tdKVrdK);
            // Construct identity layout for gdKV
            Tensor cdKV = cute::make_identity_tensor(select<1, 2>(TileShape_MNK{}));  // (BLK_M,BLK_K) -> (blk_m,blk_k)
            // Repeat the partitioning with identity layouts
            Tensor tdKVcdKV = gmem_thr_copy_dKV.partition_D(cdKV);
            Tensor tdKVpdKV = make_tensor<bool>(make_shape(size<2>(tdKVgdV)));
            #pragma unroll
            for (int k = 0; k < size(tdKVpdKV); ++k) { tdKVpdKV(k) = get<1>(tdKVcdKV(_0{}, _0{}, k)) < get<1>(params.shape_dK); }
            static constexpr int kBlockN = get<1>(TileShape_MNK{});
            // Clear_OOB_K must be false since we don't want to write zeros to gmem
            flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
                gmem_tiled_copy_dKV, tdKVrdV, tdKVgdV, tdKVcdKV, tdKVpdKV, seqlen - n_block * kBlockN
            );
            flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
                gmem_tiled_copy_dKV, tdKVrdK, tdKVgdK, tdKVcdKV, tdKVpdKV, seqlen - n_block * kBlockN
            );
        }
    }

    CUTLASS_DEVICE void
    store_tail() {
        // if constexpr (!Varlen) { tma_store_wait<0>(); }
    }

    // Write 0 to dK and dV
    CUTLASS_DEVICE void
    store_zero(
         Params const& params,
         int thread_idx,
         cute::tuple<int32_t, int32_t, int32_t> const& block_coord
         ) {
        static constexpr int kBlockN = get<1>(TileShape_MNK{});
        auto [n_block, bidh, bidb] = block_coord;
        int const offset = !Varlen ? 0 : params.cu_seqlens[bidb];
        int const seqlen = !Varlen ? get<0>(params.shape_dK) : (params.seqused ? params.seqused[bidb] : params.cu_seqlens[bidb + 1] - offset);

        Tensor mdK = make_tensor(make_gmem_ptr(params.ptr_dK), params.shape_dK, params.stride_dK)(_, _, bidh, !Varlen ? bidb : 0);
        Tensor gdK = local_tile(cute::domain_offset(make_coord(offset, _0{}), mdK), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{}));  // (M, K)
        Tensor mdV = make_tensor(make_gmem_ptr(params.ptr_dV), params.shape_dK, params.stride_dV)(_, _, bidh, !Varlen ? bidb : 0);
        Tensor gdV = local_tile(cute::domain_offset(make_coord(offset, _0{}), mdV), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{}));  // (M, K)

        GmemTiledCopydKV gmem_tiled_copy_dKV;
        auto gmem_thr_copy_dKV = gmem_tiled_copy_dKV.get_thread_slice(thread_idx);
        Tensor tdKVgdK = gmem_thr_copy_dKV.partition_D(gdK);
        Tensor tdKVgdV = gmem_thr_copy_dKV.partition_D(gdV);
        Tensor tdKVrdKV = make_fragment_like(tdKVgdK);
        clear(tdKVrdKV);
        // Construct identity layout for gdKV
        Tensor cdKV = cute::make_identity_tensor(select<1, 2>(TileShape_MNK{}));  // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tdKVcdKV = gmem_thr_copy_dKV.partition_D(cdKV);
        Tensor tdKVpdKV = make_tensor<bool>(make_shape(size<2>(tdKVgdK)));
        #pragma unroll
        for (int k = 0; k < size(tdKVpdKV); ++k) { tdKVpdKV(k) = get<1>(tdKVcdKV(_0{}, _0{}, k)) < get<1>(params.shape_dK); }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_dKV, tdKVrdKV, tdKVgdK, tdKVcdKV, tdKVpdKV, seqlen - n_block * kBlockN
        );
        flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_dKV, tdKVrdKV, tdKVgdV, tdKVcdKV, tdKVpdKV, seqlen - n_block * kBlockN
        );
    }

};

template <class TileShape_MNK_, class ElementAccum, int NumEpilogueThreads_, bool Varlen_, bool Deterministic>
struct CollectiveEpilogueBwdGQA {

    using TileShape_MNK = TileShape_MNK_;
    using Element = ElementAccum;
    static constexpr int NumEpilogueThreads = NumEpilogueThreads_;
    static constexpr bool Varlen = Varlen_;

    using GmemTiledCopydKVTMA = cute::SM90_TMA_REDUCE_ADD;

    static constexpr int kBlockN = get<1>(TileShape_MNK{});
    static constexpr int kHeadDim = get<2>(TileShape_MNK{});
    using SmemLayoutAtomdKVaccumTMA =
        decltype(composition(Swizzle<0, 4, 3>{},  // We don't want any swizzle
                             Layout<Shape<Int<8>, Int<kHeadDim>>,
                             Stride<Int<kHeadDim>, _1>>{}));
    using SmemLayoutdKVaccumTMA = decltype(tile_to_shape(SmemLayoutAtomdKVaccumTMA{}, select<1, 2>(TileShape_MNK{})));
    // Thread layout, 256 threads per row
    using R2SLayoutAtomdKVaccum = Layout<Shape<Int<NumEpilogueThreads>>, Stride<_1>>;
    using R2STiledCopydKVaccum = decltype(make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccum>{}, R2SLayoutAtomdKVaccum{},
                                                         Layout<Shape < _4>>{}));  // Val layout, 4 vals per store
    using SmemLayoutdKVaccum = Layout<Shape<Int<kBlockN * kHeadDim>>, Stride<_1>>;
    static_assert(size(SmemLayoutdKVaccumTMA{}) == size(SmemLayoutdKVaccum{}), "SmemLayoutdKVaccumTMA and SmemLayoutdKVaccum must have the same size");

    struct TensorStorage : cute::aligned_struct<128> {
        cute::array_aligned<ElementAccum, cute::cosize_v<SmemLayoutdKVaccumTMA>> smem_dkv;
    };

    using ShapedKV = cute::Shape<int32_t, int32_t, int32_t, int32_t>;  // (seqlen_q, d, head, batch)
    using StridedKV = cute::Stride<int64_t, _1, int64_t, int64_t>;

    using TMA_add_dKV = decltype(make_tma_copy(
        GmemTiledCopydKVTMA{},
        make_tensor(make_gmem_ptr(static_cast<ElementAccum*>(nullptr)), ShapedKV{}, StridedKV{}),
        SmemLayoutdKVaccumTMA{},
        select<1, 2>(TileShape_MNK{}),
        _1{}));  // no mcast for dKV

    // Host side kernel arguments
    struct Arguments {
        ElementAccum* ptr_dKaccum;
        ShapedKV const shape_dKaccum;
        StridedKV const stride_dKaccum;
        ElementAccum* ptr_dVaccum;
        StridedKV const stride_dVaccum;
        int num_heads_q;
        int* dk_semaphore;
        int* dv_semaphore;
        int const* cu_seqlens;
        int const* seqused;
    };

    // Device side kernel params
    struct Params {
        ShapedKV const shape_dKaccum;
        TMA_add_dKV tma_add_dK, tma_add_dV;
        cutlass::FastDivmod qhead_per_khead_divmod;
        int* dk_semaphore;
        int* dv_semaphore;
        int const* cu_seqlens = nullptr;
        int const* seqused = nullptr;
    };

    static Params
    to_underlying_arguments(Arguments const& args) {
        if constexpr (Varlen) {
            assert (args.cu_seqlens != nullptr);
        }
        Tensor mdKaccum = make_tensor(make_gmem_ptr(args.ptr_dKaccum), args.shape_dKaccum, args.stride_dKaccum);
        Tensor mdVaccum = make_tensor(make_gmem_ptr(args.ptr_dVaccum), args.shape_dKaccum, args.stride_dVaccum);
        TMA_add_dKV tma_add_dK = make_tma_copy(
            GmemTiledCopydKVTMA{},
            mdKaccum,
            SmemLayoutdKVaccumTMA{},
            select<1, 2>(TileShape_MNK{}),
            _1{}); // no mcast for dKV
        TMA_add_dKV tma_add_dV = make_tma_copy(
            GmemTiledCopydKVTMA{},
            mdVaccum,
            SmemLayoutdKVaccumTMA{},
            select<1, 2>(TileShape_MNK{}),
            _1{}); // no mcast for dKV
        if constexpr (Deterministic) {
            assert(args.dk_semaphore != nullptr);
            assert(args.dv_semaphore != nullptr);
        }
        if constexpr (Varlen) {
            assert(args.cu_seqlens != nullptr);
        }
        return {args.shape_dKaccum, tma_add_dK, tma_add_dV,
                cutlass::FastDivmod(cute::ceil_div(args.num_heads_q, get<2>(args.shape_dKaccum))),
                args.dk_semaphore, args.dv_semaphore,
                args.cu_seqlens, args.seqused};
    }

    /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
    CUTLASS_DEVICE
    static void prefetch_tma_descriptors(Params const& params) {
        cute::prefetch_tma_descriptor(params.tma_add_dK.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_add_dV.get_tma_descriptor());
    }

    template <typename SharedStorage, typename FrgTensorO, typename TiledMma>
    CUTLASS_DEVICE void
    store(Params const& params,
          FrgTensorO const& tdKrdK,
          FrgTensorO const& tdVrdV,
          SharedStorage& shared_storage,
          TiledMma tiled_mma,
          int thread_idx,
          cute::tuple<int32_t, int32_t, int32_t> const& block_coord
          ) {

        auto [n_block, bidh, bidb] = block_coord;
        int bidh_idx_in_group;
        // int bidh_kv = params.qhead_per_khead_divmod.divide(bidh);
        int bidh_kv = params.qhead_per_khead_divmod.divmod(bidh_idx_in_group, bidh);
        Tensor sdKV = make_tensor(make_smem_ptr(shared_storage.tensors.epilogue.smem_dkv.data()), SmemLayoutdKVaccum{});
        Tensor sdKVTMA = make_tensor(make_smem_ptr(shared_storage.tensors.epilogue.smem_dkv.data()), SmemLayoutdKVaccumTMA{});

        int const offset_padded = !Varlen ? 0 : (params.cu_seqlens[bidb] + bidb * kBlockN) / kBlockN * kBlockN;
        Tensor mdKaccum = params.tma_add_dK.get_tma_tensor(params.shape_dKaccum)(_, _, bidh_kv, !Varlen ? bidb : 0);
        Tensor mdVaccum = params.tma_add_dV.get_tma_tensor(params.shape_dKaccum)(_, _, bidh_kv, !Varlen ? bidb : 0);
        Tensor gdKaccum = local_tile(domain_offset(make_coord(offset_padded, _0{}), mdKaccum), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{}));  // (M, K)
        Tensor gdVaccum = local_tile(domain_offset(make_coord(offset_padded, _0{}), mdVaccum), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{}));  // (M, K)
        auto block_tma_dK = params.tma_add_dK.get_slice(_0{});
        auto block_tma_dV = params.tma_add_dV.get_slice(_0{});
        Tensor tdKgdK = block_tma_dK.partition_D(gdKaccum);  // (TMA, TMA_M, TMA_K)
        Tensor tdKsdK = block_tma_dK.partition_S(sdKVTMA); // (TMA, TMA_M, TMA_K)
        Tensor tdVgdV = block_tma_dV.partition_D(gdVaccum);  // (TMA, TMA_M, TMA_K)
        Tensor tdVsdV = block_tma_dV.partition_S(sdKVTMA); // (TMA, TMA_M, TMA_K)

        R2STiledCopydKVaccum r2s_tiled_copy_dKVaccum;
        auto r2s_thr_copy_dKVaccum = r2s_tiled_copy_dKVaccum.get_thread_slice(thread_idx);
        Tensor tdKVsdKVaccum = r2s_thr_copy_dKVaccum.partition_D(sdKV);

        Tensor taccdKVrdV = r2s_thr_copy_dKVaccum.retile_S(tdVrdV); // ((Atom,AtomNum), MMA_M, MMA_N)
        cute::copy(r2s_tiled_copy_dKVaccum, taccdKVrdV, tdKVsdKVaccum);

        // int const num_batch = params.num_batch;
        int const num_batch = get<3>(params.shape_dKaccum);
        int const num_head_kv = get<2>(params.shape_dKaccum);
        int *lock_ptr = !Deterministic ? nullptr : params.dv_semaphore + bidb * num_head_kv + bidh_kv;
        using Barrier = cutlass::GenericBarrier<cutlass::detail::SyncwarpSync>;

        // if (thread_idx == 0) { printf("blockIdx.x = %d, blockIdx.y = %d, blockIdx.z = %d, bidb = %d, bidh_kv = %d, lock_ptr = %p, dv_semaphore = %p, num_batch = %d, num_head_kv = %d, n_block = %d, bihd_idx_in_group = %d\n", blockIdx.x, blockIdx.y, blockIdx.z, bidb, bidh_kv, lock_ptr, params.dv_semaphore, num_batch, num_head_kv, n_block, bidh_idx_in_group);}

        if constexpr (Deterministic) {
            Barrier::wait_eq(lock_ptr, thread_idx, n_block * num_batch * num_head_kv, bidh_idx_in_group);
        }
        // if (thread_idx == 0) { printf("After barrier blockIdx.x = %d, blockIdx.y = %d, blockIdx.z = %d, bidb = %d, bidh_kv = %d, lock_ptr = %p, dv_semaphore = %p\n", blockIdx.x, blockIdx.y, blockIdx.z, bidb, bidh_kv, lock_ptr, params.dv_semaphore);}
        cutlass::arch::fence_view_async_shared();
        cutlass::arch::NamedBarrier::sync(NumEpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
        if (thread_idx == 0) {
            cute::copy(params.tma_add_dV, tdVsdV, tdVgdV);
            tma_store_arrive();
        }
        tma_store_wait<0>();
        if constexpr (Deterministic) {
            Barrier::arrive_inc(lock_ptr, thread_idx, n_block * num_batch * num_head_kv);
        }
        cutlass::arch::NamedBarrier::sync(NumEpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);

        Tensor taccdKVrdK = r2s_thr_copy_dKVaccum.retile_S(tdKrdK); // ((Atom,AtomNum), MMA_M, MMA_N)
        cute::copy(r2s_tiled_copy_dKVaccum, taccdKVrdK, tdKVsdKVaccum);
        lock_ptr = !Deterministic ? nullptr : params.dk_semaphore + bidb * num_head_kv + bidh_kv;
        // if (thread_idx == 0) { printf("blockIdx.x = %d, blockIdx.y = %d, blockIdx.z = %d, bidb = %d, bidh_kv = %d, lock_ptr = %p, dk_semaphore = %p, num_batch = %d, num_head_kv = %d, n_block = %d, bihd_idx_in_group = %d\n", blockIdx.x, blockIdx.y, blockIdx.z, bidb, bidh_kv, lock_ptr, params.dk_semaphore, num_batch, num_head_kv, n_block, bidh_idx_in_group);}

        if constexpr (Deterministic) {
            Barrier::wait_eq(lock_ptr, thread_idx, n_block * num_batch * num_head_kv, bidh_idx_in_group);
        }
        // if (thread_idx == 0) { printf("After barrier blockIdx.x = %d, blockIdx.y = %d, blockIdx.z = %d, bidb = %d, bidh_kv = %d, lock_ptr = %p, dk_semaphore = %p\n", blockIdx.x, blockIdx.y, blockIdx.z, bidb, bidh_kv, lock_ptr, params.dk_semaphore);}
        cutlass::arch::fence_view_async_shared();
        cutlass::arch::NamedBarrier::sync(NumEpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
        if (thread_idx == 0) {
            cute::copy(params.tma_add_dK, tdKsdK, tdKgdK);
            tma_store_arrive();
        }
        tma_store_wait<0>();
        if constexpr (Deterministic) {
            Barrier::arrive_inc(lock_ptr, thread_idx, n_block * num_batch * num_head_kv);
        }
    }

    CUTLASS_DEVICE void
    store_tail() {
    }

    // Write 0 to dK and dV
    CUTLASS_DEVICE void
    store_zero(
         Params const& params,
         int thread_idx,
         cute::tuple<int32_t, int32_t, int32_t> const& block_coord
         ) {
        // Don't need to do anything since dKaccum and dVaccum are already zero-initialized
    }

};

} // namespace flash
