/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/barrier.h"
#include "cute/tensor.hpp"

#include "cutlass/gemm/collective/builders/sm90_common.inl"

#include "seqlen.h"
#include "named_barrier.hpp"
#include "utils.h"

namespace flash {

using namespace cute;

template <class TileShape_MNK_, class Element_, class ArchTag_,
          int NumEpilogueThreads_, bool Varlen_, bool dKV_swapAB_, int AtomLayoutKdKV=1>
struct CollectiveEpilogueBwd {

    using TileShape_MNK = TileShape_MNK_;
    using Element = Element_;
    using ArchTag = ArchTag_;
    static constexpr int NumEpilogueThreads = NumEpilogueThreads_;
    static constexpr bool Varlen = Varlen_;
    static constexpr bool dKV_swapAB = dKV_swapAB_;
    static constexpr bool Use_TMA = !Varlen && ArchTag::kMinComputeCapability >= 90;

    static_assert(ArchTag::kMinComputeCapability >= 80);

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

    using SmemLayoutAtomdKV = std::conditional_t<Use_TMA, SmemLayoutAtomdKVTMA, SmemLayoutAtomdKVSTG>;
    using SmemLayoutdKV = decltype(tile_to_shape(SmemLayoutAtomdKV{}, select<1, 2>(TileShape_MNK{})));
    using SmemLayoutdKVt =
        decltype(cute::composition(SmemLayoutdKV{},
                                   make_layout(make_shape(get<2>(TileShape_MNK{}), get<1>(TileShape_MNK{})),
                                               make_stride(decltype(get<1>(TileShape_MNK{})){}, _1{}))));

    using SmemCopyAtomdKV = Copy_Atom<
        std::conditional_t<
            ArchTag::kMinComputeCapability >= 90,
            std::conditional_t<!dKV_swapAB, cute::SM90_U32x4_STSM_N, cute::SM90_U16x8_STSM_T>,
            AutoVectorizingCopyWithAssumedAlignment<128>
        >,
        Element>;

    static constexpr size_t SmemAlignmentdKV = ArchTag::kMinComputeCapability >= 90 ? cutlass::detail::alignment_for_swizzle(SmemLayoutdKV{}) : 128;
    static_assert(SmemAlignmentdKV >= 128, "Require at least 128B alignment");

    struct TensorStorage : cute::aligned_struct<SmemAlignmentdKV> {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutdKV>, SmemAlignmentdKV> smem_dk;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutdKV>, SmemAlignmentdKV> smem_dv;
    };

    using ShapedKV = cute::Shape<int32_t, int32_t, int32_t, int32_t>;  // (seqlen_k, d, head, batch)
    using StridedKV = cute::Stride<int64_t, _1, int64_t, int64_t>;

    using TMA_dKV = std::conditional_t<
        Use_TMA,
        decltype(make_tma_copy(
            GmemTiledCopydKVTMA{},
            make_tensor(make_gmem_ptr(static_cast<Element*>(nullptr)), ShapedKV{}, StridedKV{}),
            SmemLayoutdKVTMA{},
            select<1, 2>(TileShape_MNK{}),
            _1{})),  // no mcast for dKV
        std::nullptr_t
        >;

    // Host side kernel arguments
    struct Arguments {
        Element* ptr_dK;
        ShapedKV const shape_dK;
        StridedKV const stride_dK;
        Element* ptr_dV;
        ShapedKV const shape_dV;
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
        ShapedKV const shape_dV;
        StridedKV const stride_dV;
        TMA_dKV tma_store_dK, tma_store_dV;
        int const* cu_seqlens = nullptr;
        int const* seqused = nullptr;
    };

    static Params
    to_underlying_arguments(Arguments const& args) {
        Tensor mdK = make_tensor(make_gmem_ptr(args.ptr_dK), args.shape_dK, args.stride_dK);
        Tensor mdV = make_tensor(make_gmem_ptr(args.ptr_dV), args.shape_dV, args.stride_dV);
        TMA_dKV tma_store_dK = [&] {
            if constexpr (Use_TMA) {
                return make_tma_copy(GmemTiledCopydKVTMA{}, mdK, SmemLayoutdKVTMA{}, select<1, 2>(TileShape_MNK{}), _1{}); // no mcast for dKV
            } else {
                return nullptr;
            }
        }();
        TMA_dKV tma_store_dV = [&] {
            if constexpr (Use_TMA) {
                return make_tma_copy(GmemTiledCopydKVTMA{}, mdV, SmemLayoutdKVTMA{}, select<1, 2>(TileShape_MNK{}), _1{}); // no mcast for dKV
            } else {
                return nullptr;
            }
        }();
        return {args.ptr_dK, args.shape_dK, args.stride_dK, args.ptr_dV, args.shape_dV, args.stride_dV,
                tma_store_dK, tma_store_dV, args.cu_seqlens, args.seqused};
    }

    /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
    CUTLASS_DEVICE
    static void prefetch_tma_descriptors(Params const& params) {
        if constexpr (Use_TMA) {
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

        Tensor tdVrdV_out = make_tensor_like<Element>(tdVrdV);
        flash::convert_type_out(tdVrdV, tdVrdV_out);
        Tensor tdKrdK_out = make_tensor_like<Element>(tdKrdK);
        flash::convert_type_out(tdKrdK, tdKrdK_out);
        Tensor taccdKrdK = smem_thr_copy_dKV.retile_S(tdKrdK_out);        // ((Atom,AtomNum), MMA_M, MMA_N)
        Tensor taccdVrdV = smem_thr_copy_dKV.retile_S(tdVrdV_out);        // ((Atom,AtomNum), MMA_M, MMA_N)
        // if (blockIdx.x == 0 && threadIdx.x == 128) { print(smem_thr_copy_dKV); print(sdK); printf("\n"); print(sdKt); printf("\n"); }
        Tensor taccdKsdK = smem_thr_copy_dKV.partition_D(cute::conditional_return<!dKV_swapAB>(sdK, sdKt));     // ((Atom,AtomNum),PIPE_M,PIPE_N)
        Tensor taccdVsdV = smem_thr_copy_dKV.partition_D(cute::conditional_return<!dKV_swapAB>(sdV, sdVt));     // ((Atom,AtomNum),PIPE_M,PIPE_N)

        // Make sure all WGs have finished reading K and V
        flash::named_barrier_sync(NumEpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
        cute::copy(smem_tiled_copy_dKV, taccdVrdV, taccdVsdV);
        cute::copy(smem_tiled_copy_dKV, taccdKrdK, taccdKsdK);
        if constexpr (Use_TMA) {
            cutlass::arch::fence_view_async_shared(); // ensure smem writes are visible to TMA
            cutlass::arch::NamedBarrier::arrive(NumEpilogueThreads + cutlass::NumThreadsPerWarp,
                                                cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);

            Tensor mdK = params.tma_store_dK.get_tma_tensor(params.shape_dK);
            Tensor mdV = params.tma_store_dV.get_tma_tensor(params.shape_dV);
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
                if (cute::elect_one_sync()) {
                    cute::copy(params.tma_store_dV, tdVsdV, tdVgdV);
                    cute::copy(params.tma_store_dK, tdKsdK, tdKgdK);
                    tma_store_arrive();
                }
            }
            tma_store_wait<0>();
            // // Tell warp 0 that smem_k and smem_v are ready
            // cutlass::arch::NamedBarrier::arrive(NumEpilogueThreads + cutlass::NumThreadsPerWarp, static_cast<uint32_t>(BwdNamedBarriers::KVEmpty) /*id*/);

        } else {
            flash::named_barrier_sync(NumEpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
            static constexpr int kBlockN = get<1>(TileShape_MNK{});
            flash::SeqlenInfo<Varlen, kBlockN> seqlen_info{bidb, size<0>(params.shape_dK), params.cu_seqlens, params.seqused};
            bool const is_varlen = Varlen && params.cu_seqlens;
            Tensor mdK = make_tensor(make_gmem_ptr(params.ptr_dK), params.shape_dK, params.stride_dK)(_, _, bidh, !is_varlen ? bidb : 0);
            Tensor gdK = local_tile(cute::domain_offset(make_coord(seqlen_info.offset, _0{}), mdK), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{}));  // (M, K)
            Tensor mdV = make_tensor(make_gmem_ptr(params.ptr_dV), params.shape_dV, params.stride_dV)(_, _, bidh, !is_varlen ? bidb : 0);
            Tensor gdV = local_tile(cute::domain_offset(make_coord(seqlen_info.offset, _0{}), mdV), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{}));  // (M, K)

            GmemTiledCopydKV gmem_tiled_copy_dKV;
            auto gmem_thr_copy_dKV = gmem_tiled_copy_dKV.get_thread_slice(thread_idx);
            Tensor tdKVgdV = gmem_thr_copy_dKV.partition_D(gdV);
            Tensor tdKVsdV = gmem_thr_copy_dKV.partition_S(sdV); // (TMA, TMA_M, TMA_K)
            Tensor tdKVgdK = gmem_thr_copy_dKV.partition_D(gdK);
            Tensor tdKVsdK = gmem_thr_copy_dKV.partition_S(sdK); // (TMA, TMA_M, TMA_K)
            Tensor tdKVrdV = make_fragment_like(tdKVgdV);
            Tensor tdKVrdK = make_fragment_like(tdKVgdK);
            Tensor cdKV = cute::make_identity_tensor(select<1, 2>(TileShape_MNK{}));  // (BLK_N,BLK_K) -> (blk_n,blk_k)
            // Repeat the partitioning with identity layouts
            Tensor tdKVcdKV = gmem_thr_copy_dKV.partition_D(cdKV);
            Tensor tdKVpdV = make_tensor<bool>(make_shape(size<2>(tdKVgdV)));
            Tensor tdKVpdK = make_tensor<bool>(make_shape(size<2>(tdKVgdK)));
            #pragma unroll
            for (int k = 0; k < size(tdKVpdV); ++k) { tdKVpdV(k) = get<1>(tdKVcdKV(_0{}, _0{}, k)) < get<1>(params.shape_dV); }
            #pragma unroll
            for (int k = 0; k < size(tdKVpdK); ++k) { tdKVpdK(k) = get<1>(tdKVcdKV(_0{}, _0{}, k)) < get<1>(params.shape_dK); }
            // Need to check OOB when reading from smem if kBlockN isn't evenly tiled
            static constexpr bool EvenN = kBlockN % CUTE_STATIC_V(size<0>(GmemLayoutAtom{})) == 0;
            flash::copy</*Is_even_MN=*/EvenN, /*Is_even_K=*/true, /*Clear_OOB_MN=*/false>(
                gmem_tiled_copy_dKV, tdKVsdV, tdKVrdV, tdKVcdKV, tdKVpdV, kBlockN);
            flash::copy</*Is_even_MN=*/EvenN, /*Is_even_K=*/true, /*Clear_OOB_MN=*/false>(
                gmem_tiled_copy_dKV, tdKVsdK, tdKVrdK, tdKVcdKV, tdKVpdK, kBlockN);
            // // Tell warp 0 that smem_k and smem_v are ready
            // cutlass::arch::fence_view_async_shared(); // ensure smem reads are done before next TMA to smem_k/v
            // flash::named_barrier_arrive(NumEpilogueThreads + cutlass::NumThreadsPerWarp, static_cast<uint32_t>(BwdNamedBarriers::KVEmpty) /*id*/);
            // Construct identity layout for gdKV
            // Clear_OOB_K must be false since we don't want to write zeros to gmem
            flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
                gmem_tiled_copy_dKV, tdKVrdV, tdKVgdV, tdKVcdKV, tdKVpdV, std::min(seqlen_info.seqlen - n_block * kBlockN, kBlockN)
            );
            flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
                gmem_tiled_copy_dKV, tdKVrdK, tdKVgdK, tdKVcdKV, tdKVpdK, std::min(seqlen_info.seqlen - n_block * kBlockN, kBlockN)
            );
        }
    }

    CUTLASS_DEVICE void
    store_tail() {
        // if constexpr (Use_TMA) { tma_store_wait<0>(); }
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
        flash::SeqlenInfo<Varlen, kBlockN> seqlen_info{bidb, size<0>(params.shape_dK), params.cu_seqlens, params.seqused};
        bool const is_varlen = Varlen && params.cu_seqlens;
        Tensor mdK = make_tensor(make_gmem_ptr(params.ptr_dK), params.shape_dK, params.stride_dK)(_, _, bidh, !is_varlen ? bidb : 0);
        Tensor gdK = local_tile(cute::domain_offset(make_coord(seqlen_info.offset, _0{}), mdK), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{}));  // (M, K)
        Tensor mdV = make_tensor(make_gmem_ptr(params.ptr_dV), params.shape_dV, params.stride_dV)(_, _, bidh, !is_varlen ? bidb : 0);
        Tensor gdV = local_tile(cute::domain_offset(make_coord(seqlen_info.offset, _0{}), mdV), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{}));  // (M, K)

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
        Tensor tdKVpdK = make_tensor<bool>(make_shape(size<2>(tdKVgdK)));
        Tensor tdKVpdV = make_tensor<bool>(make_shape(size<2>(tdKVgdV)));
        #pragma unroll
        for (int k = 0; k < size(tdKVpdK); ++k) { tdKVpdK(k) = get<1>(tdKVcdKV(_0{}, _0{}, k)) < get<1>(params.shape_dK); }
        #pragma unroll
        for (int k = 0; k < size(tdKVpdV); ++k) { tdKVpdV(k) = get<1>(tdKVcdKV(_0{}, _0{}, k)) < get<1>(params.shape_dV); }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_dKV, tdKVrdKV, tdKVgdK, tdKVcdKV, tdKVpdK, seqlen_info.seqlen - n_block * kBlockN
        );
        flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_dKV, tdKVrdKV, tdKVgdV, tdKVcdKV, tdKVpdV, seqlen_info.seqlen - n_block * kBlockN
        );
    }

};

template <class TileShape_MNK_, class ElementAccum, class ArchTag_,
          int NumEpilogueThreads_, bool Varlen_, bool Deterministic>
struct CollectiveEpilogueBwdGQA {

    using TileShape_MNK = TileShape_MNK_;
    using Element = ElementAccum;
    using ArchTag = ArchTag_;
    static constexpr int NumEpilogueThreads = NumEpilogueThreads_;
    static constexpr bool Varlen = Varlen_;
    static constexpr bool Use_TMA = ArchTag::kMinComputeCapability >= 90;

    static_assert(ArchTag::kMinComputeCapability >= 80);

    static constexpr int kBlockN = get<1>(TileShape_MNK{});
    static constexpr int kHeadDim = get<2>(TileShape_MNK{});
    static_assert(NumEpilogueThreads % cutlass::NumThreadsPerWarp == 0, "NumEpilogueThreads must be a multiple of NumThreadsPerWarp");
    static constexpr int NumWarpGroups = NumEpilogueThreads / cutlass::NumThreadsPerWarpGroup;
    // Thread layout, 256 or 384 threads per row
    // We split into NumWarpGroups so that we can use the same postprocessing kernel as dQ
    using R2SLayoutAtomdKVaccum = Layout<Shape<Int<cutlass::NumThreadsPerWarpGroup>, Int<NumWarpGroups>>>;
    using R2STiledCopydKVaccum = decltype(make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccum>{}, R2SLayoutAtomdKVaccum{},
                                                         Layout<Shape < _4>>{}));  // Val layout, 4 vals per store
    // For Sm80
    using R2GLayoutAtomdKVaccum = Layout<Shape<Int<NumEpilogueThreads>>>;
    using R2GTiledCopydKVaccum = decltype(make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccum>{}, R2GLayoutAtomdKVaccum{},
                                                         Layout<Shape < _1>>{}));  // Val layout, 1 vals per store

    using SmemLayoutdKVaccum = Layout<Shape<Int<kBlockN * kHeadDim / NumWarpGroups>, Int<NumWarpGroups>>>;
    using SmemLayoutdKVaccumFlat = Layout<Shape<Int<kBlockN * kHeadDim>>>;

    // Strangely without this SmemAlignment, the total smem for hdim 128 (80 x 128) is 228KB even though we
    // only need 227KB. We use the same alignment as the non-GQA epilogue to avoid this issue.
    static constexpr int SmemAlignment = kHeadDim % 64 == 0 ? 1024 : (kHeadDim % 32 == 0 ? 512 : 256);
    struct TensorStorageTMA : cute::aligned_struct<SmemAlignment> {
        cute::array_aligned<ElementAccum, cute::cosize_v<SmemLayoutdKVaccum>, SmemAlignment> smem_dkv;
    };
    struct TensorStorageSTG {
        cute::array<ElementAccum, 0> smem_dkv;
    };
    using TensorStorage = std::conditional_t<Use_TMA, TensorStorageTMA, TensorStorageSTG>;

    using ShapedKV = cute::Shape<int32_t, int32_t, int32_t>;  // (seqlen_k_rounded * d, head, batch)
    using StridedKV = cute::Stride<_1, int64_t, int64_t>;

    // Host side kernel arguments
    struct Arguments {
        ElementAccum* ptr_dKaccum;
        ShapedKV const shape_dKaccum;
        StridedKV const stride_dKaccum;
        ElementAccum* ptr_dVaccum;
        ShapedKV const shape_dVaccum;
        StridedKV const stride_dVaccum;
        int num_heads_q;
        int* dk_semaphore;
        int* dv_semaphore;
        int const* cu_seqlens;
        int const* seqused;
    };

    // Device side kernel params
    struct Params {
        ElementAccum* ptr_dKaccum;
        ShapedKV const shape_dKaccum;
        StridedKV const stride_dKaccum;
        ElementAccum* ptr_dVaccum;
        ShapedKV const shape_dVaccum;
        StridedKV const stride_dVaccum;
        cutlass::FastDivmod qhead_per_khead_divmod;
        int* dk_semaphore;
        int* dv_semaphore;
        int const* cu_seqlens = nullptr;
        int const* seqused = nullptr;
    };

    static Params
    to_underlying_arguments(Arguments const& args) {
        if constexpr (Deterministic) {
            assert(args.dk_semaphore != nullptr);
            assert(args.dv_semaphore != nullptr);
        }
        return {args.ptr_dKaccum, args.shape_dKaccum, args.stride_dKaccum, args.ptr_dVaccum, args.shape_dVaccum, args.stride_dVaccum,
                cutlass::FastDivmod(cute::ceil_div(args.num_heads_q, get<1>(args.shape_dKaccum))),
                args.dk_semaphore, args.dv_semaphore,
                args.cu_seqlens, args.seqused};
    }

    /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
    CUTLASS_DEVICE
    static void prefetch_tma_descriptors(Params const& params) {
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
        int bidh_kv = params.qhead_per_khead_divmod.divmod(bidh_idx_in_group, bidh);
        Tensor sdKV = make_tensor(make_smem_ptr(shared_storage.tensors.epilogue.smem_dkv.data()), SmemLayoutdKVaccum{});
        Tensor sdKV_flat = make_tensor(make_smem_ptr(shared_storage.tensors.epilogue.smem_dkv.data()), SmemLayoutdKVaccumFlat{});
        static constexpr int dKV_TMA_num_bytes = CUTE_STATIC_V(size(sdKV_flat)) * sizeof(ElementAccum);

        flash::SeqlenInfo<Varlen, kBlockN> seqlen_info{bidb, size<0>(params.shape_dKaccum), params.cu_seqlens, params.seqused};
        bool const is_varlen = Varlen && params.cu_seqlens;
        Tensor mdKaccum = make_tensor(make_gmem_ptr(params.ptr_dKaccum), params.shape_dKaccum, params.stride_dKaccum)(_, bidh_kv, !is_varlen ? bidb : 0);
        Tensor mdVaccum = make_tensor(make_gmem_ptr(params.ptr_dVaccum), params.shape_dVaccum, params.stride_dVaccum)(_, bidh_kv, !is_varlen ? bidb : 0);
        Tensor gdKaccum = local_tile(domain_offset(make_coord(seqlen_info.offset_padded * kHeadDim), mdKaccum), Shape<Int<kBlockN * kHeadDim>>{}, make_coord(n_block));  // (M * K)
        Tensor gdVaccum = local_tile(domain_offset(make_coord(seqlen_info.offset_padded * kHeadDim), mdVaccum), Shape<Int<kBlockN * kHeadDim>>{}, make_coord(n_block));  // (M * K)

        R2STiledCopydKVaccum r2s_tiled_copy_dKVaccum;
        auto r2s_thr_copy_dKVaccum = r2s_tiled_copy_dKVaccum.get_thread_slice(thread_idx);
        Tensor tdKVsdKVaccum = r2s_thr_copy_dKVaccum.partition_D(sdKV);

        // Only used if !Use_TMA
        R2GTiledCopydKVaccum r2g_tiled_copy_dKVaccum;
        auto r2g_thr_copy_dKVaccum = r2g_tiled_copy_dKVaccum.get_thread_slice(thread_idx);

        // Make sure all WGs have finished reading K and V, otherwise we get racy dQ
        // because smem_q could be changed.
        flash::named_barrier_sync(NumEpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
        if constexpr (Use_TMA) {
            Tensor taccdKVrdV = r2s_thr_copy_dKVaccum.retile_S(tdVrdV); // ((Atom,AtomNum), MMA_M, MMA_N)
            cute::copy(r2s_tiled_copy_dKVaccum, taccdKVrdV, tdKVsdKVaccum);
        }

        // int const num_batch = params.num_batch;
        int const num_batch = get<2>(params.shape_dKaccum);
        int const num_head_kv = get<1>(params.shape_dKaccum);
        int *lock_ptr = !Deterministic ? nullptr : params.dv_semaphore + bidb * num_head_kv + bidh_kv;
        using Barrier = cutlass::GenericBarrier<cutlass::detail::SyncwarpSync>;

        // if (thread_idx == 0) { printf("blockIdx.x = %d, blockIdx.y = %d, blockIdx.z = %d, bidb = %d, bidh_kv = %d, lock_ptr = %p, dv_semaphore = %p, num_batch = %d, num_head_kv = %d, n_block = %d, bihd_idx_in_group = %d\n", blockIdx.x, blockIdx.y, blockIdx.z, bidb, bidh_kv, lock_ptr, params.dv_semaphore, num_batch, num_head_kv, n_block, bidh_idx_in_group);}

        if constexpr (Deterministic) {
            Barrier::wait_eq(lock_ptr, thread_idx, n_block * num_batch * num_head_kv, bidh_idx_in_group);
        }
        // if (thread_idx == 0) { printf("After barrier blockIdx.x = %d, blockIdx.y = %d, blockIdx.z = %d, bidb = %d, bidh_kv = %d, lock_ptr = %p, dv_semaphore = %p\n", blockIdx.x, blockIdx.y, blockIdx.z, bidb, bidh_kv, lock_ptr, params.dv_semaphore);}
        if constexpr (Use_TMA) {
            cutlass::arch::fence_view_async_shared();
            cutlass::arch::NamedBarrier::sync(NumEpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
            if (thread_idx == 0) {
                SM90_BULK_REDUCE_ADD::copy(raw_pointer_cast(sdKV_flat.data()), raw_pointer_cast(gdVaccum.data()), dKV_TMA_num_bytes, static_cast<uint64_t>(TMA::CacheHintSm90::EVICT_LAST));
                tma_store_arrive();
                tma_store_wait<0>();
            }
        } else {
            Tensor tdVrdV_atomic = r2g_thr_copy_dKVaccum.retile_S(tdVrdV);
            Tensor tdVgdV_atomic = r2g_thr_copy_dKVaccum.partition_D(gdVaccum);
            static_assert(CUTE_STATIC_V(size(tdVrdV_atomic)) == CUTE_STATIC_V(size(tdVgdV_atomic)));
            #pragma unroll
            for (int i = 0; i < size(tdVrdV_atomic); ++i) { atomicAdd(&tdVgdV_atomic(i), tdVrdV_atomic(i)); }
        }
        if constexpr (Deterministic) {
            Barrier::arrive_inc(lock_ptr, thread_idx, n_block * num_batch * num_head_kv);
        }

        if constexpr (Use_TMA) {
            cutlass::arch::NamedBarrier::sync(NumEpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
            Tensor taccdKVrdK = r2s_thr_copy_dKVaccum.retile_S(tdKrdK); // ((Atom,AtomNum), MMA_M, MMA_N)
            cute::copy(r2s_tiled_copy_dKVaccum, taccdKVrdK, tdKVsdKVaccum);
        }
        lock_ptr = !Deterministic ? nullptr : params.dk_semaphore + bidb * num_head_kv + bidh_kv;
        // if (thread_idx == 0) { printf("blockIdx.x = %d, blockIdx.y = %d, blockIdx.z = %d, bidb = %d, bidh_kv = %d, lock_ptr = %p, dk_semaphore = %p, num_batch = %d, num_head_kv = %d, n_block = %d, bihd_idx_in_group = %d\n", blockIdx.x, blockIdx.y, blockIdx.z, bidb, bidh_kv, lock_ptr, params.dk_semaphore, num_batch, num_head_kv, n_block, bidh_idx_in_group);}

        if constexpr (Deterministic) {
            Barrier::wait_eq(lock_ptr, thread_idx, n_block * num_batch * num_head_kv, bidh_idx_in_group);
        }
        // if (thread_idx == 0) { printf("After barrier blockIdx.x = %d, blockIdx.y = %d, blockIdx.z = %d, bidb = %d, bidh_kv = %d, lock_ptr = %p, dk_semaphore = %p\n", blockIdx.x, blockIdx.y, blockIdx.z, bidb, bidh_kv, lock_ptr, params.dk_semaphore);}
        if constexpr (Use_TMA) {
            cutlass::arch::fence_view_async_shared();
            cutlass::arch::NamedBarrier::sync(NumEpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
            if (thread_idx == 0) {
                SM90_BULK_REDUCE_ADD::copy(raw_pointer_cast(sdKV_flat.data()), raw_pointer_cast(gdKaccum.data()), dKV_TMA_num_bytes, static_cast<uint64_t>(TMA::CacheHintSm90::EVICT_LAST));
                tma_store_arrive();
                tma_store_wait<0>();
            }
        } else {
            Tensor tdKrdK_atomic = r2g_thr_copy_dKVaccum.retile_S(tdKrdK);
            Tensor tdKgdK_atomic = r2g_thr_copy_dKVaccum.partition_D(gdKaccum);
            static_assert(CUTE_STATIC_V(size(tdKrdK_atomic)) == CUTE_STATIC_V(size(tdKgdK_atomic)));
            #pragma unroll
            for (int i = 0; i < size(tdKrdK_atomic); ++i) { atomicAdd(&tdKgdK_atomic(i), tdKrdK_atomic(i)); }
        }
        if constexpr (Deterministic) {
            Barrier::arrive_inc(lock_ptr, thread_idx, n_block * num_batch * num_head_kv);
        }
        // // Tell warp 0 that smem_k and smem_v are ready
        // flash::named_barrier_arrive(NumEpilogueThreads + cutlass::NumThreadsPerWarp, static_cast<uint32_t>(BwdNamedBarriers::KVEmpty) /*id*/);
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
