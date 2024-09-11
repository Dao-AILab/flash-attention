/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include "cute/tensor.hpp"

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include "cutlass/arch/barrier.h"

#include "utils.h"

namespace flash {

using namespace cute;

template <class TileShape_MK_, class Element, class ElementAccum, class ArchTag_, int kNThreads, class SmemLayoutdQaccumTMA,
          class TiledMma, bool dQ_swapAB>
class FlashAttnBwdPostprocessConvertdQ {

public:

    // Type Aliases
    using TileShape_MK = TileShape_MK_;
    using ArchTag = ArchTag_;

    static_assert(ArchTag::kMinComputeCapability >= 90);

    static constexpr uint32_t MaxThreadsPerBlock = kNThreads;
    static constexpr uint32_t MinBlocksPerMultiprocessor = 2;

    static constexpr int kHeadDim = get<1>(TileShape_MK{});
    using R2SLayoutAtomdQaccum = Layout<Shape<Int<kNThreads>>, Stride<_1>>;
    using R2STiledCopydQaccum = decltype(make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{}, R2SLayoutAtomdQaccum{},
                                                         Layout<Shape < _4>>{}));  // Val layout, 4 vals per read
    static constexpr int SmemdQaccumSize = size(TileShape_MK{});
    static_assert(size(TileShape_MK{}) == size(SmemLayoutdQaccumTMA{}), "TileShape_MK and SmemLayoutdQaccumTMA must have the same size");
    using SmemLayoutdQaccum = Layout<Shape<Int<SmemdQaccumSize>>, Stride<_1>>;

    // We can't just use kHeadDim here. E.g. if MMA shape is 64 x 96 but split across 2 WGs,
    // then setting kBlockKSmem to 32 will cause "Static shape_div failure".
    // We want to treat it as 64 x 48, so kBlockKSmem should be 16.
    static constexpr int MmaShapeN = get<1>(typename TiledMma::AtomShape_MNK{});
    static constexpr int kBlockKSmem = MmaShapeN % 64 == 0 ? 64 : (MmaShapeN % 32 == 0 ? 32 : 16);
    static constexpr int kSwizzle = kBlockKSmem == 64 ? 3 : (kBlockKSmem == 32 ? 2 : 1);
    using SmemLayoutAtomdQ =
        decltype(composition(Swizzle<kSwizzle, 3, 3>{},
                 Layout<Shape<Int<8>, Int<kBlockKSmem>>,
                 Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutdQ = decltype(tile_to_shape(SmemLayoutAtomdQ{}, TileShape_MK{}));
    using SmemLayoutdQt =
        decltype(cute::composition(SmemLayoutdQ{},
                                   make_layout(make_shape(get<1>(TileShape_MK{}), get<0>(TileShape_MK{})),
                                               make_stride(Int<get<0>(TileShape_MK{})>{}, _1{}))));

    using SmemCopyAtomdQ = Copy_Atom<
        std::conditional_t<!dQ_swapAB, cute::SM90_U32x4_STSM_N, cute::SM90_U16x8_STSM_T>,
        Element>;

    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "Headdim must be a multiple of kGmemElemsPerLoad");
    static constexpr int kGmemThreadsPerRow = cutlass::gcd(kHeadDim / kGmemElemsPerLoad, int(MaxThreadsPerBlock));
    static_assert(MaxThreadsPerBlock % kGmemThreadsPerRow == 0, "MaxThreadsPerBlock must be a multiple of kGmemThreadsPerRow");
    using GmemLayoutAtom = Layout<Shape <Int<MaxThreadsPerBlock / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                  Stride<Int<kGmemThreadsPerRow>, _1>>;
    using GmemTiledCopy = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));  // Val layout, 8 or 16 vals per load

    using GmemTiledCopydQaccum = cute::SM90_TMA_LOAD;

    struct SharedStorage : cute::aligned_struct<128> {
        cute::array_aligned<ElementAccum, cute::cosize_v<SmemLayoutdQaccumTMA>, 1024> smem_dqacc;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutdQ>> smem_dq;
        alignas(16) cutlass::arch::ClusterTransactionBarrier barrier_dQaccum;
    };

    static constexpr int SharedStorageSize = sizeof(SharedStorage);

    using ShapedQ = cute::Shape<int32_t, int32_t, int32_t, int32_t>;   // (seqlen_q, d, head, batch)
    using StridedQ = cute::Stride<int64_t, _1, int64_t, int64_t>;

    using TMA_dQaccum = decltype(make_tma_copy(
        GmemTiledCopydQaccum{},
        make_tensor(make_gmem_ptr(static_cast<ElementAccum*>(nullptr)), ShapedQ{}, StridedQ{}),
        SmemLayoutdQaccumTMA{},
        SmemLayoutdQaccumTMA{}.shape(),
        _1{})); // no mcast for dQ

    // Device side arguments
    struct Arguments {
        ElementAccum const* ptr_dQaccum;
        ShapedQ const shape_dQaccum;
        StridedQ const stride_dQaccum;
        Element* ptr_dQ;
        ShapedQ const shape_dQ;
        StridedQ const stride_dQ;
        float const softmax_scale;
        int const* cu_seqlens = nullptr;
    };

    // Kernel entry point API
    struct Params {
        TMA_dQaccum tma_load_dQaccum;
        ShapedQ const shape_dQaccum;
        Element* ptr_dQ;
        ShapedQ const shape_dQ;
        StridedQ const stride_dQ;
        float const softmax_scale;
        int const* cu_seqlens = nullptr;
    };

    // Convert to underlying arguments. In this case, a simple copy for the aliased type.
    static
    Params
    to_underlying_arguments(Arguments const& args) {
        Tensor mdQaccum = make_tensor(make_gmem_ptr(args.ptr_dQaccum), args.shape_dQaccum, args.stride_dQaccum);
        TMA_dQaccum tma_load_dQaccum = make_tma_copy(
            GmemTiledCopydQaccum{},
            mdQaccum,
            SmemLayoutdQaccumTMA{},
            SmemLayoutdQaccumTMA{}.shape(),
            _1{}); // no mcast for dQaccum
        return {
            tma_load_dQaccum,
            args.shape_dQaccum,
            args.ptr_dQ,
            args.shape_dQ,
            args.stride_dQ,
            args.softmax_scale,
            args.cu_seqlens
        };
    }

    CUTLASS_DEVICE
    void
    operator()(Params const& params, char* smem_buf) {

        static constexpr int kBlockM = get<0>(TileShape_MK{});
        SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

        Tensor sdQaccumTMA = make_tensor(make_smem_ptr(shared_storage.smem_dqacc.data()), SmemLayoutdQaccumTMA{});
        // Tensor sdQaccumTMAnoswizzle = make_tensor(make_smem_ptr(shared_storage.smem_dqacc.data()), SmemLayoutdQaccumTMANoSwizzle{});
        Tensor sdQaccum = make_tensor(make_smem_ptr(shared_storage.smem_dqacc.data()), SmemLayoutdQaccum{});
        Tensor sdQ = make_tensor(make_smem_ptr(shared_storage.smem_dq.data()), SmemLayoutdQ{});
        Tensor sdQt = make_tensor(make_smem_ptr(shared_storage.smem_dq.data()), SmemLayoutdQt{});

        int const thread_idx = threadIdx.x;
        int const m_block = blockIdx.x;
        int const bidh = blockIdx.y;
        int const bidb = blockIdx.z;

        bool const is_varlen = params.cu_seqlens != nullptr;
        int const seqlen = !is_varlen ? get<0>(params.shape_dQ) : params.cu_seqlens[bidb + 1] - params.cu_seqlens[bidb];
        if (is_varlen && m_block * kBlockM >= seqlen) { return; }

        int lane_predicate = cute::elect_one_sync();
        int warp_idx = cutlass::canonical_warp_idx_sync();
        // Issue Tma Descriptor Prefetch from a single thread
        if (warp_idx == 0 && lane_predicate) {
            cute::prefetch_tma_descriptor(params.tma_load_dQaccum.get_tma_descriptor());
            shared_storage.barrier_dQaccum.init(1 /*numThreads*/);
        }
        __syncthreads();

        // Step 1: TMA to load dQaccum from gmem to smem
        // We reshaped dQaccum to have last dimension 32, so the offset needs to be multiplied by kHeadDim / 32
        int const offset_padded = !is_varlen ? 0 : ((params.cu_seqlens[bidb] + bidb * 128) / 128 * 128) * (kHeadDim / get<1>(SmemLayoutdQaccumTMA{}.shape()));
        Tensor mdQaccum = params.tma_load_dQaccum.get_tma_tensor(params.shape_dQaccum)(_, _, bidh, !is_varlen ? bidb : 0);
        Tensor gdQaccum = local_tile(domain_offset(make_coord(offset_padded, _0{}), mdQaccum), SmemLayoutdQaccumTMA{}.shape(), make_coord(m_block, _0{}));  // (M, K)
        auto block_tma_dQ = params.tma_load_dQaccum.get_slice(_0{});
        Tensor tdQgdQaccumTMA = block_tma_dQ.partition_D(gdQaccum);  // (TMA, TMA_M, TMA_K)
        Tensor tdQsdQaccumTMA = block_tma_dQ.partition_S(sdQaccumTMA); // (TMA, TMA_M, TMA_K)
        static constexpr uint32_t TmaTransactionBytesdQaccum = static_cast<uint32_t>(size(SmemLayoutdQaccumTMA{}) * cute::sizeof_bits_v<ElementAccum> / 8);
        if (warp_idx == 0 && lane_predicate) {
            shared_storage.barrier_dQaccum.arrive_and_expect_tx(TmaTransactionBytesdQaccum);
            copy(params.tma_load_dQaccum.with(reinterpret_cast<cutlass::arch::ClusterTransactionBarrier::ValueType&>(shared_storage.barrier_dQaccum), 0 /*mcast_mask*/), tdQgdQaccumTMA, tdQsdQaccumTMA);
        }
        shared_storage.barrier_dQaccum.wait(0);

        // __syncthreads(); if (cute::thread0()) { print_tensor(sdQaccumTMA); }
        // __syncthreads(); if (cute::thread0()) { print_tensor(sdQaccumTMAnoswizzle); }
        // __syncthreads(); if (cute::thread0()) { print_tensor(sdQaccum); }

        // Step 2: Load dQaccum from smem to register, then convert fp32 -> fp16/bf16
        R2STiledCopydQaccum s2r_tiled_copy_dQaccum;
        auto s2r_thr_copy_dQaccum = s2r_tiled_copy_dQaccum.get_thread_slice(thread_idx);
        Tensor tdQsdQaccum = s2r_thr_copy_dQaccum.partition_S(sdQaccum);
        TiledMma tiled_mma_dQ;
        Tensor taccdQrdQaccum = partition_fragment_C(tiled_mma_dQ, select<!dQ_swapAB ? 0 : 1, !dQ_swapAB ? 1 : 0>(TileShape_MK{}));
        // if (cute::thread0()) { print(tiled_mma_dQ); printf("\n"); }
        // if (cute::thread0()) { print(tdQsdQaccum); }
        // if (cute::thread0()) { print(taccdQrdQaccum); }
        CUTE_STATIC_ASSERT_V(size(taccdQrdQaccum) == size(tdQsdQaccum));
        Tensor tdQrdQaccum = s2r_thr_copy_dQaccum.retile_D(taccdQrdQaccum);
        cute::copy(s2r_tiled_copy_dQaccum, tdQsdQaccum, tdQrdQaccum);
        #pragma unroll
        for (int i = 0; i < size(taccdQrdQaccum); ++i) { taccdQrdQaccum(i) *= params.softmax_scale; }
        // Convert tdQrdQ from fp32 to fp16
        Tensor rdQ = flash::convert_type<Element>(taccdQrdQaccum);

        // Step 3: Copy dQ from register to smem
        auto smem_tiled_copy_dQ = make_tiled_copy_C(SmemCopyAtomdQ{}, tiled_mma_dQ);
        auto smem_thr_copy_dQ = smem_tiled_copy_dQ.get_thread_slice(thread_idx);
        Tensor taccdQrdQ = smem_thr_copy_dQ.retile_S(rdQ);  // ((Atom,AtomNum), MMA_N, MMA_N)
        // if (cute::thread0()) { print(smem_tiled_copy_dQ); }
        // if (cute::thread0()) { print(smem_thr_copy_dQ); }
        // if (cute::thread0()) { print(sdQ); }
        if constexpr (!dQ_swapAB) {
            Tensor taccdQsdQ = smem_thr_copy_dQ.partition_D(sdQ);  // ((Atom,AtomNum),PIPE_M,PIPE_N)
            cute::copy(smem_tiled_copy_dQ, taccdQrdQ, taccdQsdQ);
        } else {
            Tensor taccdQsdQt = smem_thr_copy_dQ.partition_D(sdQt);  // ((Atom,AtomNum),PIPE_M,PIPE_N)
            cute::copy(smem_tiled_copy_dQ, taccdQrdQ, taccdQsdQt);
        }
        __syncthreads();

        // Step 4: Copy dQ from smem to register to prepare for coalesced write to gmem
        int const offset = !is_varlen ? 0 : params.cu_seqlens[bidb];
        Tensor mdQ = make_tensor(make_gmem_ptr(params.ptr_dQ), params.shape_dQ, params.stride_dQ)(_, _, bidh, !is_varlen ? bidb : 0);
        Tensor gdQ = local_tile(domain_offset(make_coord(offset, _0{}), mdQ), TileShape_MK{}, make_coord(m_block, _0{}));  // (M, K)
        GmemTiledCopy gmem_tiled_copy_dQ;
        auto gmem_thr_copy_dQ = gmem_tiled_copy_dQ.get_thread_slice(thread_idx);
        Tensor tdQsdQ = gmem_thr_copy_dQ.partition_S(sdQ);    // ((Atom,AtomNum),ATOM_M,ATOM_N)
        Tensor tdQgdQ = gmem_thr_copy_dQ.partition_D(gdQ);

        Tensor tdQrdQ = make_fragment_like(tdQsdQ);
        cute::copy(gmem_tiled_copy_dQ, tdQsdQ, tdQrdQ);

        // Step 5: Copy dQ from register to gmem
        // Construct identity layout for gdQ
        Tensor cdQ = cute::make_identity_tensor(TileShape_MK{});  // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tdQcdQ = gmem_thr_copy_dQ.partition_D(cdQ);
        Tensor tdQpdQ = make_tensor<bool>(make_shape(size<2>(tdQgdQ)));
        #pragma unroll
        for (int k = 0; k < size(tdQpdQ); ++k) { tdQpdQ(k) = get<1>(tdQcdQ(_0{}, _0{}, k)) < get<1>(params.shape_dQ); }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_dQ, tdQrdQ, tdQgdQ, tdQcdQ, tdQpdQ, seqlen - m_block * kBlockM
        );
    }

};

} // namespace flash
