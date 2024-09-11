/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/barrier.h>
#include "cutlass/pipeline/pipeline.hpp"

#include "cute/tensor.hpp"

#include "cutlass/gemm/collective/collective_builder.hpp"

#include "named_barrier.hpp"
#include "softmax.h"
#include "utils.h"

namespace flash {

using namespace cute;

template <int Stages, class ClusterShape_, class TileShape_MNK_, class Element_, class ElementAccum_, class ArchTag_,
        bool Is_causal_, bool Varlen_, bool Deterministic,
        bool dKV_swapAB_, bool dQ_swapAB_,
        int AtomLayoutMSdP=1, int AtomLayoutNdKV=2, int AtomLayoutMdQ=1>
struct CollectiveMainloopBwd {

    static constexpr int kStages = Stages;
    using ClusterShape = ClusterShape_;
    using TileShape_MNK = TileShape_MNK_;
    using Element = Element_;
    using ElementAccum = ElementAccum_;
    using ArchTag = ArchTag_;
    static constexpr bool Is_causal = Is_causal_;
    static constexpr bool Varlen = Varlen_;
    static constexpr bool SdP_swapAB = true;
    static constexpr bool dKV_swapAB = dKV_swapAB_;
    static constexpr bool dQ_swapAB = dQ_swapAB_;
    static_assert(!(SdP_swapAB && dKV_swapAB));  // If SdP_swapAB, then we don't swap for dKV

    static constexpr int kBlockM = get<0>(TileShape_MNK{});
    static constexpr int kBlockN = get<1>(TileShape_MNK{});
    static constexpr int kHeadDim = get<2>(TileShape_MNK{});

    static constexpr int NumdQWarpGroups = 2;
    static constexpr int kNThreadsdQ = NumdQWarpGroups * cutlass::NumThreadsPerWarpGroup;

    static_assert(ArchTag::kMinComputeCapability >= 90);
    static_assert(get<0>(ClusterShape{}) == 1 && get<2>(ClusterShape{}) == 1);

    static constexpr bool Mma_dQ_is_RS = AtomLayoutMSdP == 2 && AtomLayoutMdQ == 2 && !SdP_swapAB && !dQ_swapAB;  // If dQ_swapAB we can't use RS
    using TileShapeAtomSdP = std::conditional_t<
        !SdP_swapAB,
        Shape<Int<kBlockM>, Int<kBlockN / (2 / AtomLayoutMSdP)>, Int<kHeadDim>>,
        Shape<Int<kBlockN>, Int<kBlockM / AtomLayoutMSdP>, Int<kHeadDim>>
    >;
    using AtomLayoutSdP = std::conditional_t<
        !SdP_swapAB,
        Layout<Shape<Int<AtomLayoutMSdP>, Int<2 / AtomLayoutMSdP>, _1>>,
        Layout<Shape<Int<2 / AtomLayoutMSdP>, Int<AtomLayoutMSdP>, _1>>
    >;
    using TiledMmaSdP = decltype(cute::make_tiled_mma(
        cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShapeAtomSdP>(),
        AtomLayoutSdP{}));

    using TileShapeAtomdKV = std::conditional_t<
        !dKV_swapAB,
        Shape<Int<kBlockN>, Int<kHeadDim / (2 / AtomLayoutNdKV)>, Int<kBlockM>>,
        Shape<Int<kHeadDim>, Int<kBlockN / AtomLayoutNdKV>, Int<kBlockM>>
    >;
    using AtomLayoutdKV = std::conditional_t<
        !dKV_swapAB,
        Layout<Shape<Int<AtomLayoutNdKV>, Int<2 / AtomLayoutNdKV>, _1>>,
        Layout<Shape<Int<2 / AtomLayoutNdKV>, Int<AtomLayoutNdKV>, _1>>
    >;
    using TiledMmadKV = decltype(cute::make_tiled_mma(
        std::conditional_t<
            !SdP_swapAB,
            decltype(cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShapeAtomdKV, GMMA::Major::MN, GMMA::Major::MN>()),
            decltype(cute::GMMA::rs_op_selector<Element, Element, ElementAccum, TileShapeAtomdKV, GMMA::Major::K, GMMA::Major::MN>())
        >{},
        AtomLayoutdKV{}));

    using TileShapeAtomdQ = std::conditional_t<
        !dQ_swapAB,
        Shape<Int<kBlockM>, Int<kHeadDim / (NumdQWarpGroups / AtomLayoutMdQ)>, Int<kBlockN>>,
        Shape<Int<kHeadDim>, Int<kBlockM / AtomLayoutMdQ>, Int<kBlockN>>
    >;
    using AtomLayoutdQ = std::conditional_t<
        !dQ_swapAB,
        Layout<Shape<Int<AtomLayoutMdQ>, Int<NumdQWarpGroups / AtomLayoutMdQ>, _1>>,
        Layout<Shape<Int<NumdQWarpGroups / AtomLayoutMdQ>, Int<AtomLayoutMdQ>, _1>>
    >;
    static constexpr GMMA::Major MmadQMajorA = !dQ_swapAB ? GMMA::Major::K : GMMA::Major::MN;
    static constexpr GMMA::Major MmadQMajorB = !dQ_swapAB ? GMMA::Major::MN : GMMA::Major::K;
    using TiledMmadQ = decltype(cute::make_tiled_mma(
        std::conditional_t<
            !dQ_swapAB,
            std::conditional_t<
                Mma_dQ_is_RS,
                decltype(cute::GMMA::rs_op_selector<Element, Element, ElementAccum, TileShapeAtomdQ, GMMA::Major::K, GMMA::Major::MN>()),
                decltype(cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShapeAtomdQ, GMMA::Major::K, GMMA::Major::MN>())
            >,
            decltype(cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShapeAtomdQ, GMMA::Major::MN, GMMA::Major::K>())
        >{},
        AtomLayoutdQ{}));

    using SmemLayoutAtomQ = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
                                     Int<kBlockM>, Int<dKV_swapAB ? kHeadDim : kHeadDim / (2 / AtomLayoutNdKV)>>());
    using SmemLayoutQ =
        decltype(tile_to_shape(SmemLayoutAtomQ{},
                 make_shape(shape<0>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{})));
    using SmemLayoutdO = SmemLayoutQ;

    using SmemLayoutAtomK = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
                                     Int<kBlockN>, Int<dQ_swapAB ? kHeadDim : kHeadDim / (NumdQWarpGroups / AtomLayoutMdQ)>>());
    using SmemLayoutK = decltype(tile_to_shape(SmemLayoutAtomK{}, select<1, 2>(TileShape_MNK{})));

    using SmemLayoutAtomV = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutV = decltype(tile_to_shape(SmemLayoutAtomV{}, select<1, 2>(TileShape_MNK{})));

    using SmemLayoutAtomP = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<1>(TileShape_MNK{}))>());
    using SmemLayoutP = decltype(tile_to_shape(SmemLayoutAtomP{}, select<0, 1>(TileShape_MNK{})));
    using SmemLayoutAtomdS = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<1>(TileShape_MNK{}))>());
    using SmemLayoutdS = decltype(tile_to_shape(SmemLayoutAtomdS{}, make_shape(Int<kBlockM>{}, Int<kBlockN>{}, Int<kStages>{})));

    // Need stride to be multiple of 32, otherwise we get error (misaligned address) when doing TMA if e.g. kBlockM=80
    using SmemLayoutLSE = cute::Layout<cute::Shape<Int<kBlockM>, Int<kStages>>, cute::Stride<_1, Int<cute::round_up(kBlockM, 32)>>>;
    using SmemLayoutLSEMma = cute::Layout<cute::Shape<Int<kBlockN>, Int<kBlockM>, Int<kStages>>, cute::Stride<_0, _1, Int<cute::round_up(kBlockM, 32)>>>;

    // Note this is the transpose in terms of the view, not in terms of memory.
    using SmemLayoutQt =
        decltype(cute::composition(SmemLayoutQ{},
                                   make_layout(make_shape(get<2>(TileShape_MNK{}), get<0>(TileShape_MNK{}), Int<kStages>{}),
                                               make_stride(Int<kBlockM>{}, _1{}, Int<kBlockM * kHeadDim>{}))));
    using SmemLayoutdOt =
        decltype(cute::composition(SmemLayoutdO{},
                                   make_layout(make_shape(get<2>(TileShape_MNK{}), get<0>(TileShape_MNK{}), Int<kStages>{}),
                                               make_stride(Int<kBlockM>{}, _1{}, Int<kBlockM * kHeadDim>{}))));
    using SmemLayoutKt =
        decltype(cute::composition(SmemLayoutK{},
                                   make_layout(make_shape(get<2>(TileShape_MNK{}), get<1>(TileShape_MNK{})),
                                               make_stride(Int<kBlockN>{}, _1{}))));
    using SmemLayoutPt =
        decltype(cute::composition(SmemLayoutP{},
                                   make_layout(make_shape(get<1>(TileShape_MNK{}), get<0>(TileShape_MNK{})),
                                               make_stride(Int<kBlockM>{}, _1{}))));
    using SmemLayoutdSt =
        decltype(cute::composition(SmemLayoutdS{},
                                   make_layout(make_shape(Int<kBlockN>{}, Int<kBlockM>{}, Int<kStages>{}),
                                               make_stride(Int<kBlockM>{}, _1{}, Int<kBlockM * kBlockN>{}))));

    // Thread layout, 256 threads per row
    using R2SLayoutAtomdQaccum = Layout<Shape<Int<kNThreadsdQ>>, Stride<_1>>;
    using R2STiledCopydQaccum = decltype(make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{}, R2SLayoutAtomdQaccum{},
                                                         Layout<Shape < _4>>{}));  // Val layout, 4 vals per store
    using SmemLayoutdQaccum = Layout<Shape<Int<kBlockM * kHeadDim>>, Stride<_1>>;
    // We want dQaccum smem to have last dimension 32, so that we only need to do 1 TMA instruction.
    // The layout Layout_K_SW128_Atom<ElementAccum> has 32 elements per row.
    // // TMA limit is that each dimension in smem must be <= 256.
    // static constexpr int ElemsPerRowTMA = (kBlockM * kHeadDim) / 32 <= 256 ? 32 : 64;
    static constexpr int ElemsPerRowTMA = 32;  // If we change this, we'll also need to change the dQ shape in host.
    static_assert((kBlockM * kHeadDim) % ElemsPerRowTMA == 0);
    using TileShape_dQaccum = cute::Shape<Int<(kBlockM * kHeadDim) / ElemsPerRowTMA>, Int<ElemsPerRowTMA>>;
    // using TileShape_dQaccum = cute::Shape<Int<kBlockM>, Int<kHeadDim>>;
    using SmemLayoutdQaccumTMA =
        decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<ElementAccum>{}, TileShape_dQaccum{}));
    using SmemLayoutdQaccumTMANoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutdQaccumTMA{}));

    using SmemCopyAtomPdS = Copy_Atom<
        std::conditional_t<!SdP_swapAB, cute::SM90_U32x4_STSM_N, cute::SM90_U16x8_STSM_T>,
            Element>;
    using SmemCopyAtomdKV = Copy_Atom<
        std::conditional_t<!dKV_swapAB, cute::SM90_U32x4_STSM_N, cute::SM90_U16x8_STSM_T>,
            Element>;

    using GmemTiledCopyQdO = decltype(cutlass::gemm::collective::detail::sm90_cluster_shape_to_tma_atom(shape<1>(ClusterShape{})));
    using GmemTiledCopyKV = cute::SM90_TMA_LOAD;
    using GmemTiledCopydQaccum = cute::SM90_TMA_REDUCE_ADD;
    using GmemTiledCopyLSE = cute::SM90_TMA_LOAD;

    using ShapeQKV = cute::Shape<int32_t, int32_t, int32_t, int32_t>;  // (seqlen, d, head, batch)
    using StrideQKV = cute::Stride<int64_t, _1, int64_t, int64_t>;
    using ShapeLSE = cute::Shape<int32_t, int32_t, int32_t>;  // (seqlen, head, batch)
    using StrideLSE = cute::Stride<_1, int64_t, int64_t>;  // (seqlen, head, batch)

    using TMA_QdO = decltype(make_tma_copy(
        GmemTiledCopyQdO{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQKV{}, StrideQKV{}),
        take<0, 2>(SmemLayoutQ{}),
        select<0, 2>(TileShape_MNK{}),
        size<1>(ClusterShape{}))); // mcast along N mode for this M load, if any

    using TMA_K = decltype(make_tma_copy(
        GmemTiledCopyKV{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQKV{}, StrideQKV{}),
        SmemLayoutK{},
        select<1, 2>(TileShape_MNK{}),
        _1{})); // no mcast for KV

    using TMA_V = decltype(make_tma_copy(
        GmemTiledCopyKV{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQKV{}, StrideQKV{}),
        SmemLayoutV{},
        select<1, 2>(TileShape_MNK{}),
        _1{})); // no mcast for KV

    using TMA_add_dQ = decltype(make_tma_copy(
        GmemTiledCopydQaccum{},
        make_tensor(make_gmem_ptr(static_cast<ElementAccum*>(nullptr)), ShapeQKV{}, StrideQKV{}),
        SmemLayoutdQaccumTMA{},
        TileShape_dQaccum{},
        _1{})); // no mcast for dQ

    using TMA_LSE = decltype(make_tma_copy(
        GmemTiledCopyLSE{},
        make_tensor(make_gmem_ptr(static_cast<ElementAccum const*>(nullptr)), ShapeLSE{}, StrideLSE{}),
        select<0>(SmemLayoutLSE{}),
        select<0>(TileShape_MNK{}),
        _1{})); // no mcast for LSE

    static constexpr int NumMmaThreads = size(TiledMmaSdP{});

    using MainloopPipeline = typename cutlass::PipelineTmaAsync<kStages>;
    using PipelineState = typename MainloopPipeline::PipelineState;

    // Set the bytes transferred in this TMA transaction (may involve multiple issues)
    static constexpr uint32_t TmaTransactionBytesQ = static_cast<uint32_t>(size(take<0, 2>(SmemLayoutQ{})) * cutlass::sizeof_bits_v<Element> / 8);
    static constexpr uint32_t TmaTransactionBytesK = static_cast<uint32_t>(size(SmemLayoutK{}) * cutlass::sizeof_bits_v<Element> / 8);
    static constexpr uint32_t TmaTransactionBytesV = static_cast<uint32_t>(size(SmemLayoutV{}) * cutlass::sizeof_bits_v<Element> / 8);
    static constexpr uint32_t TmaTransactionBytesLSE = static_cast<uint32_t>(size(select<0>(SmemLayoutLSE{})) * cutlass::sizeof_bits_v<ElementAccum> / 8);

    struct TensorStorage : cute::aligned_struct<1024> {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>> smem_k;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>> smem_v;
        // It's important that smem_dqacc is aligned to 1024 bytes for the TMA, so that the 1st row
        // has no swizzle.
        // If the address is only 128 bytes aligned, it's possible that the 1st row has swizzle
        // and when we read it back in the postprocess kernel, the swizzle will not match.
        cute::array_aligned<ElementAccum, cute::cosize_v<SmemLayoutdQaccum>, 1024> smem_dqacc;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> smem_q;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutdO>> smem_do;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutdS>> smem_ds;
        cute::array_aligned<ElementAccum, cute::cosize_v<SmemLayoutLSE>, 128> smem_lse;
        cute::array_aligned<ElementAccum, cute::cosize_v<SmemLayoutLSE>, 128> smem_dpsum;
    };

    static constexpr int SharedStorageQdOSize = sizeof(decltype((TensorStorage{}).smem_q)) + sizeof(decltype((TensorStorage{}).smem_do)) + sizeof(decltype((TensorStorage{}).smem_ds)) + sizeof(decltype((TensorStorage{}).smem_dqacc));

    // Host side kernel arguments
    struct Arguments {
        Element const* ptr_Q;
        ShapeQKV const shape_Q;
        StrideQKV const stride_Q;
        Element const* ptr_K;
        ShapeQKV const shape_K;
        StrideQKV const stride_K;
        Element const* ptr_V;
        StrideQKV const stride_V;
        Element const* ptr_dO;
        StrideQKV const stride_dO;
        ElementAccum* ptr_dQaccum;
        ShapeQKV const shape_dQaccum;
        StrideQKV const stride_dQaccum;
        float const* ptr_LSE_log2;
        ShapeLSE const shape_LSE;
        StrideLSE const stride_LSE_log2;
        float const* ptr_dPsum;
        StrideLSE const stride_dPsum;
        float const softmax_scale;
        int num_batch;
        int* dq_semaphore;
        int const* cu_seqlens_q = nullptr;
        int const* cu_seqlens_k = nullptr;
    };

    // Device side kernel params
    struct Params {
        ShapeQKV const shape_Q;
        ShapeQKV const shape_K;
        ShapeQKV const shape_dQaccum;
        cutlass::FastDivmod qhead_per_khead_divmod;
        TMA_QdO tma_load_Q, tma_load_dO;
        TMA_K tma_load_K;
        TMA_V tma_load_V;
        TMA_add_dQ tma_add_dQ;
        TMA_LSE tma_load_LSE, tma_load_dPsum;
        float const* ptr_LSE_log2;
        ShapeLSE const shape_LSE;
        StrideLSE const stride_LSE_log2;
        float const* ptr_dPsum;
        StrideLSE const stride_dPsum;
        float const softmax_scale;
        float const softmax_scale_log2;
        int num_batch;
        int* dq_semaphore;
        int const* cu_seqlens_q = nullptr;
        int const* cu_seqlens_k = nullptr;
    };

    static Params
    to_underlying_arguments(Arguments const& args) {
        Tensor mQ = make_tensor(make_gmem_ptr(args.ptr_Q), args.shape_Q, args.stride_Q);
        TMA_QdO tma_load_Q = make_tma_copy(
            GmemTiledCopyQdO{},
            mQ,
            SmemLayoutQ{}(_, _, _0{}),
            select<0, 2>(TileShape_MNK{}),
            size<1>(ClusterShape{})); // mcast along N mode for this M load, if any
        Tensor mdO = make_tensor(make_gmem_ptr(args.ptr_dO), args.shape_Q, args.stride_dO);
        TMA_QdO tma_load_dO = make_tma_copy(
            GmemTiledCopyQdO{},
            mdO,
            SmemLayoutdO{}(_, _, _0{}),
            select<0, 2>(TileShape_MNK{}),
            size<1>(ClusterShape{})); // mcast along N mode for this M load, if any
        Tensor mK = make_tensor(make_gmem_ptr(args.ptr_K), args.shape_K, args.stride_K);
        TMA_K tma_load_K = make_tma_copy(
            GmemTiledCopyKV{},
            mK,
            SmemLayoutK{},
            select<1, 2>(TileShape_MNK{}),
            _1{}); // no mcast for KV
        Tensor mV = make_tensor(make_gmem_ptr(args.ptr_V), args.shape_K, args.stride_V);
        TMA_V tma_load_V = make_tma_copy(
            GmemTiledCopyKV{},
            mV,
            SmemLayoutV{},
            select<1, 2>(TileShape_MNK{}),
            _1{}); // no mcast for KV
        Tensor mdQaccum = make_tensor(make_gmem_ptr(args.ptr_dQaccum), args.shape_dQaccum, args.stride_dQaccum);
        TMA_add_dQ tma_add_dQ = make_tma_copy(
            GmemTiledCopydQaccum{},
            mdQaccum,
            SmemLayoutdQaccumTMA{},
            TileShape_dQaccum{},
            _1{}); // no mcast for dQaccum
        Tensor mLSE = make_tensor(make_gmem_ptr(args.ptr_LSE_log2), args.shape_LSE, args.stride_LSE_log2);
        TMA_LSE tma_load_LSE = make_tma_copy(
            GmemTiledCopyLSE{},
            mLSE,
            select<0>(SmemLayoutLSE{}),
            select<0>(TileShape_MNK{}),
            _1{}); // no mcast for LSE
        Tensor mdPsum = make_tensor(make_gmem_ptr(args.ptr_dPsum), args.shape_LSE, args.stride_dPsum);
        TMA_LSE tma_load_dPsum = make_tma_copy(
            GmemTiledCopyLSE{},
            mdPsum,
            select<0>(SmemLayoutLSE{}),
            select<0>(TileShape_MNK{}),
            _1{}); // no mcast for dPsum
        if constexpr (Deterministic) { assert(args.dq_semaphore != nullptr); }
        return {args.shape_Q, args.shape_K, args.shape_dQaccum,
                cutlass::FastDivmod(cute::ceil_div(get<2>(args.shape_Q), get<2>(args.shape_K))),
                tma_load_Q, tma_load_dO, tma_load_K, tma_load_V, tma_add_dQ, tma_load_LSE, tma_load_dPsum,
                args.ptr_LSE_log2, args.shape_LSE, args.stride_LSE_log2, args.ptr_dPsum, args.stride_dPsum,
                args.softmax_scale, float(args.softmax_scale * M_LOG2E),
                args.num_batch, args.dq_semaphore, args.cu_seqlens_q, args.cu_seqlens_k};
    }

    /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
    CUTLASS_DEVICE
    static void prefetch_tma_descriptors(Params const& params) {
        cute::prefetch_tma_descriptor(params.tma_load_Q.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_dO.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_K.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_V.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_LSE.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_dPsum.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_add_dQ.get_tma_descriptor());
    }

    CUTLASS_DEVICE
    int get_seqlen_q(Params const& params, int bidb) {
        if constexpr (!Varlen) {
            return get<0>(params.shape_Q);
        } else {
            return params.cu_seqlens_q == nullptr
                ? get<0>(params.shape_Q)
                : params.cu_seqlens_q[bidb + 1] - params.cu_seqlens_q[bidb];
        }
    }

    CUTLASS_DEVICE
    int get_seqlen_k(Params const& params, int bidb) {
        if constexpr (!Varlen) {
            return get<0>(params.shape_K);
        } else {
            return params.cu_seqlens_k == nullptr
                ? get<0>(params.shape_K)
                : params.cu_seqlens_k[bidb + 1] - params.cu_seqlens_k[bidb];
        }
    }

    CUTLASS_DEVICE
    int get_m_block_min(Params const& params, int n_block, int bidb) {
        if constexpr (Is_causal) {
            int const seqlen_q = get_seqlen_q(params, bidb);
            int const seqlen_k = get_seqlen_k(params, bidb);
            return std::max(0, (n_block * kBlockN + seqlen_q - seqlen_k) / kBlockM);
        } else {
            return 0;
        }
    }

    template <typename SchedulerPrefetch, typename SharedStorage>
    CUTLASS_DEVICE void
    load(Params const& params,
         MainloopPipeline pipeline_q,
         MainloopPipeline pipeline_do,
         PipelineState& smem_pipe_write,
         SharedStorage &shared_storage,
         SchedulerPrefetch const& scheduler_prefetch,
         cute::tuple<int32_t, int32_t, int32_t> block_coord,
         int work_idx
         ) {

        Tensor sQ = make_tensor(make_smem_ptr(shared_storage.mainloop.smem_q.data()), SmemLayoutQ{});
        Tensor sdO = make_tensor(make_smem_ptr(shared_storage.mainloop.smem_do.data()), SmemLayoutdO{});
        Tensor sK = make_tensor(make_smem_ptr(shared_storage.mainloop.smem_k.data()), SmemLayoutK{});
        Tensor sV = make_tensor(make_smem_ptr(shared_storage.mainloop.smem_v.data()), SmemLayoutV{});
        Tensor sLSE = make_tensor(make_smem_ptr(shared_storage.mainloop.smem_lse.data()), SmemLayoutLSE{});
        Tensor sdPsum = make_tensor(make_smem_ptr(shared_storage.mainloop.smem_dpsum.data()), SmemLayoutLSE{});

        auto [n_block, bidh, bidb] = block_coord;
        int bidh_kv = params.qhead_per_khead_divmod.divide(bidh);

        // Prepare the TMA loads
        uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();
        constexpr uint32_t cluster_shape_x = get<0>(ClusterShape());
        uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x, block_rank_in_cluster / cluster_shape_x};
        bool const is_varlen_q = Varlen && params.cu_seqlens_q != nullptr;
        bool const is_varlen_k = Varlen && params.cu_seqlens_k != nullptr;
        Tensor mQ = params.tma_load_Q.get_tma_tensor(params.shape_Q)(_, _, bidh, !is_varlen_q ? bidb : 0);
        Tensor mdO = params.tma_load_dO.get_tma_tensor(params.shape_Q)(_, _, bidh, !is_varlen_q ? bidb : 0);
        Tensor mK = params.tma_load_K.get_tma_tensor(params.shape_K)(_, _, bidh_kv, !is_varlen_k ? bidb : 0);
        Tensor mV = params.tma_load_V.get_tma_tensor(params.shape_K)(_, _, bidh_kv, !is_varlen_k ? bidb : 0);
        Tensor mLSE = params.tma_load_LSE.get_tma_tensor(params.shape_LSE)(_, bidh, !is_varlen_q ? bidb : 0);
        Tensor mdPsum = params.tma_load_dPsum.get_tma_tensor(params.shape_LSE)(_, bidh, !is_varlen_q ? bidb : 0);

        int const offset_q = !is_varlen_q ? 0 : params.cu_seqlens_q[bidb];
        int const offset_k = !is_varlen_k ? 0 : params.cu_seqlens_k[bidb];
        int const offset_padded = !is_varlen_q ? 0 : (params.cu_seqlens_q[bidb] + bidb * 128) / 128 * 128;
        Tensor gQ = local_tile(domain_offset(make_coord(offset_q, _0{}), mQ), select<0, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (M, K, _)
        Tensor gdO = local_tile(domain_offset(make_coord(offset_q, _0{}), mdO), select<0, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (M, K, _)
        Tensor gK = local_tile(domain_offset(make_coord(offset_k, _0{}), mK), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{}));  // (N, K)
        Tensor gV = local_tile(domain_offset(make_coord(offset_k, _0{}), mV), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{}));  // (N, K)
        Tensor gLSE = local_tile(domain_offset(make_coord(offset_padded), mLSE), select<0>(TileShape_MNK{}), make_coord(_));  // (M, _)
        Tensor gdPsum = local_tile(domain_offset(make_coord(offset_padded), mdPsum), select<0>(TileShape_MNK{}), make_coord(_));  // (M, _)

        Tensor sK_x = make_tensor(sK.data(), make_layout(sK.layout(), Layout<_1>{}));
        Tensor gK_x = make_tensor(gK.data(), make_layout(gK.layout(), Layout<_1>{}));
        Tensor sV_x = make_tensor(sV.data(), make_layout(sV.layout(), Layout<_1>{}));
        Tensor gV_x = make_tensor(gV.data(), make_layout(gV.layout(), Layout<_1>{}));
        auto [tQgQ, tQsQ] = tma_partition(params.tma_load_Q, block_rank_in_cluster, Layout<ClusterShape>{},
                                          group_modes<0, 2>(sQ), group_modes<0, 2>(gQ));  // (TMA, k), (TMA, PIPE)
        auto [tdOgdO, tdOsdO] = tma_partition(params.tma_load_dO, block_rank_in_cluster, Layout<ClusterShape>{},
                                          group_modes<0, 2>(sdO), group_modes<0, 2>(gdO));  // (TMA, k), (TMA, PIPE)
        auto [tKgK, tKsK] = tma_partition(params.tma_load_K, _0{}, Layout<_1>{},
                                          group_modes<0, 2>(sK_x), group_modes<0, 2>(gK_x));  // (TMA), (TMA)
        auto [tVgV, tVsV] = tma_partition(params.tma_load_V, _0{}, Layout<_1>{},
                                          group_modes<0, 2>(sV_x), group_modes<0, 2>(gV_x));  // (TMA), (TMA)
        auto [tLSEgLSE, tLSEsLSE] = tma_partition(params.tma_load_LSE, _0{}, Layout<_1>{},
                                                  sLSE, gLSE);  // (TMA, k), (TMA, PIPE)
        auto [tLSEgdPsum, tLSEsdPsum] = tma_partition(params.tma_load_dPsum, _0{}, Layout<_1>{},
                                                  sdPsum, gdPsum);  // (TMA, k), (TMA, PIPE)

        uint16_t mcast_mask_qdo = 0;
        if constexpr (cute::is_same_v<GmemTiledCopyQdO, SM90_TMA_LOAD_MULTICAST>) {
            auto block_layout = Layout<ClusterShape>{}; // (m,n) -> block_id
            for (int n = 0; n < size<1>(block_layout); ++n) {
                mcast_mask_qdo |= (uint16_t(1) << block_layout(n, cluster_local_block_id.x, _0{}));
            }
        }

        int m_block_max = cute::ceil_div(get_seqlen_q(params, bidb), get<0>(TileShape_MNK{}));
        int m_block_min = get_m_block_min(params, n_block, bidb);
        int m_block = m_block_min;

        int lane_predicate = cute::elect_one_sync();

        // // Wait for the MMA warpgroups to say that smem_q is ready
        // cutlass::arch::NamedBarrier::sync(NumMmaThreads + cutlass::NumThreadsPerWarp, static_cast<int>(BwdNamedBarriers::QueryEmpty) /*id*/);

        if (lane_predicate) {
            // Copy K tile and V tile from GMEM to SMEM.
            shared_storage.barrier_KV.arrive_and_expect_tx(TmaTransactionBytesK + TmaTransactionBytesV);
            copy(params.tma_load_K.with(reinterpret_cast<cutlass::arch::ClusterTransactionBarrier::ValueType&>(shared_storage.barrier_KV), 0 /*mcast_mask*/), tKgK, tKsK);
            copy(params.tma_load_V.with(reinterpret_cast<cutlass::arch::ClusterTransactionBarrier::ValueType&>(shared_storage.barrier_KV), 0 /*mcast_mask*/), tVgV, tVsV);

            pipeline_q.producer_acquire(smem_pipe_write);
            copy(params.tma_load_Q.with(*pipeline_q.producer_get_barrier(smem_pipe_write), mcast_mask_qdo), tQgQ(_, m_block), tQsQ(_, smem_pipe_write.index()));
            copy(params.tma_load_LSE.with(*pipeline_q.producer_get_barrier(smem_pipe_write), 0), tLSEgLSE(_, m_block), tLSEsLSE(_, smem_pipe_write.index()));
            #pragma unroll 2
            for (; m_block < m_block_max - 1; ++m_block) {
                pipeline_do.producer_acquire(smem_pipe_write);
                copy(params.tma_load_dO.with(*pipeline_do.producer_get_barrier(smem_pipe_write), mcast_mask_qdo), tdOgdO(_, m_block), tdOsdO(_, smem_pipe_write.index()));
                copy(params.tma_load_dPsum.with(*pipeline_do.producer_get_barrier(smem_pipe_write), 0), tLSEgdPsum(_, m_block), tLSEsdPsum(_, smem_pipe_write.index()));
                ++smem_pipe_write;
                pipeline_q.producer_acquire(smem_pipe_write);
                copy(params.tma_load_Q.with(*pipeline_q.producer_get_barrier(smem_pipe_write), mcast_mask_qdo), tQgQ(_, m_block + 1), tQsQ(_, smem_pipe_write.index()));
                copy(params.tma_load_LSE.with(*pipeline_q.producer_get_barrier(smem_pipe_write), 0), tLSEgLSE(_, m_block + 1), tLSEsLSE(_, smem_pipe_write.index()));
            }
        }
        scheduler_prefetch();
        if (lane_predicate) {
            pipeline_do.producer_acquire(smem_pipe_write);
            copy(params.tma_load_dO.with(*pipeline_do.producer_get_barrier(smem_pipe_write), mcast_mask_qdo), tdOgdO(_, m_block), tdOsdO(_, smem_pipe_write.index()));
            copy(params.tma_load_dPsum.with(*pipeline_do.producer_get_barrier(smem_pipe_write), 0), tLSEgdPsum(_, m_block), tLSEsdPsum(_, smem_pipe_write.index()));
            ++smem_pipe_write;
        }
    }

    /// Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
    CUTLASS_DEVICE void
    load_tail(MainloopPipeline pipeline_q, MainloopPipeline pipeline_do,
              PipelineState& smem_pipe_write) {
        // Need to copy since pipeline_q.producer_tail(smem_pipe_write) will increment smem_pipe_write
        PipelineState smem_pipe_write_do = smem_pipe_write;
        int lane_predicate = cute::elect_one_sync();
        // Issue the epilogue waits
        if (lane_predicate) {
            /* This helps avoid early exit of blocks in Cluster
            * Waits for all stages to either be released (all Consumer UNLOCKs), or if the stage was never used
            * then would just be acquired since the phase was still inverted from make_producer_start_state
            */
            pipeline_q.producer_tail(smem_pipe_write);
            pipeline_do.producer_tail(smem_pipe_write_do);
        }
    }

    template <typename SharedStorage>
    CUTLASS_DEVICE void
    store_dq(Params const& params,
             SharedStorage &shared_storage,
             cute::tuple<int32_t, int32_t, int32_t> block_coord
             ) {

        Tensor sdQ = make_tensor(make_smem_ptr(shared_storage.mainloop.smem_dqacc.data()), SmemLayoutdQaccumTMA{});
        Tensor sdQnoswizzle = make_tensor(make_smem_ptr(shared_storage.mainloop.smem_dqacc.data()), SmemLayoutdQaccumTMANoSwizzle{});
        auto [n_block, bidh, bidb] = block_coord;

        bool const is_varlen_q = Varlen && params.cu_seqlens_q != nullptr;
        // We reshaped dQaccum to have last dimension 32, so the offset needs to be multiplied by kHeadDim / 32
        int const offset_padded = !is_varlen_q ? 0 : ((params.cu_seqlens_q[bidb] + bidb * 128) / 128 * 128) * (kHeadDim / ElemsPerRowTMA);
        // Prepare the TMA loads
        Tensor mdQaccum = params.tma_add_dQ.get_tma_tensor(params.shape_dQaccum)(_, _, bidh, !is_varlen_q ? bidb : 0);
        Tensor gdQaccum = local_tile(domain_offset(make_coord(offset_padded, _0{}), mdQaccum), TileShape_dQaccum{}, make_coord(_, _0{}));  // (M, K, _)
        auto block_tma_dQ = params.tma_add_dQ.get_slice(_0{});
        Tensor tdQgdQ = block_tma_dQ.partition_D(gdQaccum);  // (TMA, TMA_M, TMA_K)
        Tensor tdQsdQ = block_tma_dQ.partition_S(sdQ); // (TMA, TMA_M, TMA_K)

        int m_block_max = cute::ceil_div(get_seqlen_q(params, bidb), get<0>(TileShape_MNK{}));
        int m_block_min = get_m_block_min(params, n_block, bidb);
        int m_block = m_block_min;
        int const num_batch = params.num_batch;
        int const num_head = get<2>(params.shape_Q);
        int *lock_ptr = !Deterministic ? nullptr : params.dq_semaphore + bidb * num_head + bidh;
        using Barrier = cutlass::GenericBarrier<cutlass::detail::SyncwarpSync>;
        int lane_predicate = cute::elect_one_sync();
        #pragma unroll 2
        for (; m_block < m_block_max; ++m_block) {
            if constexpr (Deterministic) {
                Barrier::wait_eq(lock_ptr, threadIdx.x % cutlass::NumThreadsPerWarp, m_block * num_batch * num_head, n_block);
            }
            cutlass::arch::NamedBarrier::sync(kNThreadsdQ + cutlass::NumThreadsPerWarp, static_cast<int>(BwdNamedBarriers::dQFull) /*id*/);  // sdQ full, to be written to gmem
            if (lane_predicate) {
                cute::copy(params.tma_add_dQ, tdQsdQ, tdQgdQ(_, _, _, m_block));
                tma_store_arrive();
            }
            tma_store_wait<0>();
            if constexpr (Deterministic) {
                Barrier::arrive_inc(lock_ptr, threadIdx.x % cutlass::NumThreadsPerWarp, m_block * num_batch * num_head);
            }
            cutlass::arch::NamedBarrier::arrive(kNThreadsdQ + cutlass::NumThreadsPerWarp, static_cast<int>(BwdNamedBarriers::dQEmpty) /*id*/);  // sdQ empty, ready to be written to
        }
    }

    CUTLASS_DEVICE void
    mma_init() {
        // // Tell producer (warp 0) that smem_q is ready
        // cutlass::arch::NamedBarrier::arrive(NumMmaThreads + cutlass::NumThreadsPerWarp, static_cast<int>(BwdNamedBarriers::QueryEmpty) /*id*/);
        int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
        if (cutlass::canonical_warp_group_idx() == 1 && warp_idx_in_warpgroup == 0) {
            cutlass::arch::NamedBarrier::arrive(kNThreadsdQ + cutlass::NumThreadsPerWarp, static_cast<int>(BwdNamedBarriers::dQEmpty) /*id*/);  // sdQ empty, ready to be written to
        }
    }

    template <typename SharedStorage, typename FrgTensordKV>
    CUTLASS_DEVICE void
    mma(Params const& params,
        MainloopPipeline pipeline_q,
        MainloopPipeline pipeline_do,
        PipelineState& smem_pipe_read,
        FrgTensordKV& tdKrdK,
        FrgTensordKV& tdVrdV,
        int thread_idx,
        int work_idx,
        cute::tuple<int32_t, int32_t, int32_t> block_coord,
        SharedStorage& shared_storage
        ) {
        static_assert(is_rmem<FrgTensordKV>::value, "dK and dV tensor must be rmem resident.");

        Tensor sQ = make_tensor(make_smem_ptr(shared_storage.mainloop.smem_q.data()), SmemLayoutQ{});
        Tensor sdO = make_tensor(make_smem_ptr(shared_storage.mainloop.smem_do.data()), SmemLayoutdO{});
        Tensor sK = make_tensor(make_smem_ptr(shared_storage.mainloop.smem_k.data()), SmemLayoutK{});
        Tensor sV = make_tensor(make_smem_ptr(shared_storage.mainloop.smem_v.data()), SmemLayoutV{});
        Tensor sQt = make_tensor(make_smem_ptr(shared_storage.mainloop.smem_q.data()), SmemLayoutQt{});
        Tensor sdOt = make_tensor(make_smem_ptr(shared_storage.mainloop.smem_do.data()), SmemLayoutdOt{});
        Tensor sKt = make_tensor(make_smem_ptr(shared_storage.mainloop.smem_k.data()), SmemLayoutKt{});
        Tensor sdS = make_tensor(make_smem_ptr(shared_storage.mainloop.smem_ds.data()), SmemLayoutdS{});
        Tensor sdSt = make_tensor(make_smem_ptr(shared_storage.mainloop.smem_ds.data()), SmemLayoutdSt{});
        Tensor sdQ = make_tensor(make_smem_ptr(shared_storage.mainloop.smem_dqacc.data()), SmemLayoutdQaccum{});
        Tensor sLSEMma = make_tensor(make_smem_ptr(shared_storage.mainloop.smem_lse.data()), SmemLayoutLSEMma{});
        Tensor sdPsumMma = make_tensor(make_smem_ptr(shared_storage.mainloop.smem_dpsum.data()), SmemLayoutLSEMma{});

        static_assert(stride<0>(typename TiledMmaSdP::ALayout{}) == 0 and
                      stride<0>(typename TiledMmaSdP::BLayout{}) == 0 and
                      size<0>(typename TiledMmaSdP::ALayout{}) == cutlass::NumThreadsPerWarpGroup and
                      size<0>(typename TiledMmaSdP::BLayout{}) == cutlass::NumThreadsPerWarpGroup,
                      "Stride of the first mode must be 0 and the size of the mode must be NumThreadsPerWarpGroup");
        constexpr int MmaWarpGroups = NumMmaThreads / cutlass::NumThreadsPerWarpGroup;
        Layout warp_group_thread_layout = make_layout(make_shape(Int<MmaWarpGroups>{}),
                                                      make_stride(Int<cutlass::NumThreadsPerWarpGroup>{}));
        Layout warp_group_thread_layout_dq = make_layout(make_shape(Int<NumdQWarpGroups>{}),
                                                      make_stride(Int<cutlass::NumThreadsPerWarpGroup>{}));

        int warp_group_idx = __shfl_sync(0xFFFFFFFF, thread_idx / cutlass::NumThreadsPerWarpGroup, 0);
        TiledMmaSdP tiled_mma_SdP;
        TiledMmadKV tiled_mma_dKV;
        TiledMmadQ tiled_mma_dQ;
        static_assert(!dKV_swapAB);

        auto wg_mma_SdP = tiled_mma_SdP.get_slice(warp_group_thread_layout(warp_group_idx));
        auto thread_mma_SdP = tiled_mma_SdP.get_thread_slice(thread_idx);
        auto wg_mma_dKV = tiled_mma_dKV.get_slice(warp_group_thread_layout(warp_group_idx));
        auto wg_mma_dQ = tiled_mma_dQ.get_slice(!Varlen ? warp_group_thread_layout_dq(NumdQWarpGroups == 2 ? warp_group_idx : 0) : thread_idx);
        // auto wg_mma_dQ = tiled_mma_dQ.get_thread_slice(thread_idx);

        auto smem_tiled_copy_PdS = make_tiled_copy_C(SmemCopyAtomPdS{}, tiled_mma_SdP);
        auto smem_thr_copy_PdS = smem_tiled_copy_PdS.get_thread_slice(thread_idx);
        Tensor tdSsdS = smem_thr_copy_PdS.partition_D(sdSt);      // ((Atom,AtomNum),PIPE_M,PIPE_N)

        R2STiledCopydQaccum r2s_tiled_copy_dQaccum;
        // auto r2s_thr_copy_dQaccum = r2s_tiled_copy_dQaccum.get_thread_slice(thread_idx);
        auto r2s_thr_copy_dQaccum = r2s_tiled_copy_dQaccum.get_thread_slice(NumdQWarpGroups == 2 ? thread_idx : thread_idx % cutlass::NumThreadsPerWarpGroup);
        Tensor tdQsdQaccum = r2s_thr_copy_dQaccum.partition_D(sdQ);

        // Allocate "fragments/descriptors"
        Tensor tSrQ = wg_mma_SdP.partition_fragment_B(sQ);
        Tensor tSrK = wg_mma_SdP.partition_fragment_A(sK);
        Tensor tdPrdO = wg_mma_SdP.partition_fragment_B(sdO);
        Tensor tdPrV = wg_mma_SdP.partition_fragment_A(sV);
        Tensor tdVrdO = wg_mma_dKV.partition_fragment_B(sdOt);
        Tensor tdKrQ = wg_mma_dKV.partition_fragment_B(sQt);

        int n_block = get<0>(block_coord);
        int bidh = get<1>(block_coord);
        int bidb = get<2>(block_coord);
        int const seqlen_q = get_seqlen_q(params, bidb);
        int const seqlen_k = get_seqlen_k(params, bidb);

        int m_block_max = cute::ceil_div(get_seqlen_q(params, bidb), get<0>(TileShape_MNK{}));
        int m_block_min = get_m_block_min(params, n_block, bidb);
        int m_block = m_block_min;

        // thread_mma_SdP.partition_C(sLSEMma) has shape ((2, 2, V), MMA_M, MMA_N, PIPE), we only take the row indices.
        Tensor tLSEsLSE = thread_mma_SdP.partition_C(sLSEMma)(make_coord(_, _0{}, _), _0{}, _0{}, _);  // (2, V, PIPE)
        Tensor tLSEsdPsum = thread_mma_SdP.partition_C(sdPsumMma)(make_coord(_, _0{}, _), _0{}, _0{}, _);


        clear(tdKrdK);
        clear(tdVrdV);
        // tiled_mma_dKV.accumulate_ = GMMA::ScaleOut::Zero;

        cutlass::ConsumerToken barrier_token = static_cast<cutlass::BarrierStatus>(shared_storage.barrier_KV.try_wait(work_idx % 2));
        if (barrier_token == cutlass::BarrierStatus::WaitAgain) { shared_storage.barrier_KV.wait(work_idx % 2); }

        auto consumer_wait = [](auto& pipeline, auto& smem_pipe_read) {
            auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
            pipeline.consumer_wait(smem_pipe_read, barrier_token);
        };

        auto compute_dQ = [&]() {
            static_assert(!Mma_dQ_is_RS);
            // SMEM fence to make sure sP is written before it's read by WGMMA
            cutlass::arch::fence_view_async_shared();
            cutlass::arch::NamedBarrier::sync(kNThreadsdQ + cutlass::NumThreadsPerWarp, static_cast<int>(BwdNamedBarriers::dQEmpty) /*id*/);  // sdQ empty, ready to be written to
            Tensor tdQrdQ = partition_fragment_C(tiled_mma_dQ, select<!dQ_swapAB ? 0 : 2, !dQ_swapAB ? 2 : 0>(TileShape_MNK{}));
            if constexpr (!dQ_swapAB) {
                Tensor tdQrdS = wg_mma_dQ.partition_fragment_A(sdS);
                Tensor tdQrK = wg_mma_dQ.partition_fragment_B(sKt);
                flash::gemm</*zero_init=*/true, /*wg_wait=*/1>(tiled_mma_dQ, tdQrdS(_, _, _, smem_pipe_read.index()), tdQrK, tdQrdQ);
            } else {
                Tensor tdQrdS = wg_mma_dQ.partition_fragment_B(sdS);
                Tensor tdQrK = wg_mma_dQ.partition_fragment_A(sKt);
                flash::gemm</*zero_init=*/true, /*wg_wait=*/1>(tiled_mma_dQ, tdQrK, tdQrdS(_, _, _, smem_pipe_read.index()), tdQrdQ);
            }
            pipeline_q.consumer_release(smem_pipe_read);  // release Q
            warpgroup_wait<0>();
            Tensor taccdQrdQ = r2s_thr_copy_dQaccum.retile_S(tdQrdQ);        // ((Atom,AtomNum), MMA_M, MMA_N)
            cute::copy(r2s_tiled_copy_dQaccum, taccdQrdQ, tdQsdQaccum);
            cutlass::arch::fence_view_async_shared();
            cutlass::arch::NamedBarrier::arrive(kNThreadsdQ + cutlass::NumThreadsPerWarp, static_cast<int>(BwdNamedBarriers::dQFull) /*id*/);  // sdQ full, to be written to gmem
        };

        // We have separate iterations with causal masking. Not necessary for hdim 128 but for hdim 64
        // this helps quite a bit to not have to do causal masking for most of the iterations.
        if constexpr (Is_causal) {
            static constexpr int n_masking_steps = cute::ceil_div(kBlockN, kBlockM) + 1;
            CUTLASS_PRAGMA_NO_UNROLL
            for (; m_block < std::min(m_block_max, m_block_min + n_masking_steps); ++m_block) {
                Tensor tSrS = partition_fragment_C(tiled_mma_SdP, select<1, 0>(TileShape_MNK{}));
                pipeline_q.consumer_wait(smem_pipe_read);
                flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma_SdP, tSrK, tSrQ(_, _, _, smem_pipe_read.index()), tSrS);
                Tensor tLSErLSE = make_fragment_like(tLSEsLSE(_, _, _0{}));
                cute::copy(tLSEsLSE(_, _, smem_pipe_read.index()), tLSErLSE);

                Tensor tdPrdP = partition_fragment_C(tiled_mma_SdP, select<1, 0>(TileShape_MNK{}));
                pipeline_do.consumer_wait(smem_pipe_read);
                flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma_SdP, tdPrV, tdPrdO(_, _, _, smem_pipe_read.index()), tdPrdP);
                warpgroup_wait<1>();
                Tensor cS = cute::make_identity_tensor(select<1, 0>(TileShape_MNK{}));
                Tensor taccScS = thread_mma_SdP.partition_C(cS);
                int causal_row_offset = 1 + seqlen_k - n_block * kBlockN - seqlen_q + m_block * kBlockM;
                #pragma unroll
                for (int i = 0; i < size(tSrS); ++i) {
                    if (int(get<0>(taccScS(i))) >= std::min(int(get<1>(taccScS(i))) + causal_row_offset,
                                                            seqlen_k - n_block * kBlockN)) {
                        tSrS(i) = -INFINITY;
                    }
                }
                // Reshape tSrS from ((2, 2, V), MMA_N, MMA_M) to (nrow=(2, V, MMA_M), ncol=(2, MMA_N))
                Tensor scores = make_tensor(tSrS.data(), flash::convert_layout_acc_transposed_rowcol(tSrS.layout()));
                flash::scale_apply_exp2</*Scale_max=*/false, /*Check_inf=*/false>(scores, group_modes<0, 2>(tLSErLSE), params.softmax_scale_log2);

                Tensor tLSErdPsum = make_fragment_like(tLSEsdPsum(_, _, _0{}));
                cute::copy(tLSEsdPsum(_, _, smem_pipe_read.index()), tLSErdPsum);

                // Convert scores from fp32 to fp16/bf16
                Tensor rP = flash::convert_type<Element>(tSrS);

                warpgroup_wait<0>();
                // Reshape tdPrdP from ((2, 2, V), MMA_N, MMA_M) to (nrow=(2, V, MMA_M), ncol=(2, MMA_N))
                Tensor dS = make_tensor(tdPrdP.data(), scores.layout());
                for (int mi = 0; mi < size<0>(dS); ++mi) {
                    #pragma unroll
                    for (int ni = 0; ni < size<1>(dS); ++ni) { dS(mi, ni) = scores(mi, ni) * (dS(mi, ni) - tLSErdPsum(mi)); }
                }
                Tensor rdS = flash::convert_type<Element>(tdPrdP);

                // Because of double buffering on dS, we don't need to sync here.
                // Otherwise we might have WG1 writing to dS before WG2 is done reading from it during MmadQ.
                // But because both WGs have to sync at the end of the loop and double buffering, this race condition
                // is not possible.
                Tensor tdSadS = smem_thr_copy_PdS.retile_S(rdS);     // ((Atom,AtomNum), MMA_N, MMA_N)
                cute::copy(smem_tiled_copy_PdS, tdSadS, tdSsdS(_, _, _, smem_pipe_read.index()));

                Tensor tdVrP = make_tensor(rP.data(), convert_layout_acc_Aregs<TiledMmadKV>(tSrS.layout()));
                flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(tiled_mma_dKV, tdVrP, tdVrdO(_, _, _, smem_pipe_read.index()), tdVrdV);

                Tensor tdKrdS = make_tensor(rdS.data(), convert_layout_acc_Aregs<TiledMmadKV>(tdPrdP.layout()));
                flash::gemm</*zero_init=*/false, /*wg_wait=*/1>(tiled_mma_dKV, tdKrdS, tdKrQ(_, _, _, smem_pipe_read.index()), tdKrdK);
                pipeline_do.consumer_release(smem_pipe_read);  // release dO

                compute_dQ();
                ++smem_pipe_read;
            }
        }

        CUTLASS_PRAGMA_NO_UNROLL
        for (; m_block < m_block_max; ++m_block) {
            Tensor tSrS = partition_fragment_C(tiled_mma_SdP, select<1, 0>(TileShape_MNK{}));
            pipeline_q.consumer_wait(smem_pipe_read);
            flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma_SdP, tSrK, tSrQ(_, _, _, smem_pipe_read.index()), tSrS);
            Tensor tLSErLSE = make_fragment_like(tLSEsLSE(_, _, _0{}));
            cute::copy(tLSEsLSE(_, _, smem_pipe_read.index()), tLSErLSE);

            Tensor tdPrdP = partition_fragment_C(tiled_mma_SdP, select<1, 0>(TileShape_MNK{}));
            pipeline_do.consumer_wait(smem_pipe_read);
            flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma_SdP, tdPrV, tdPrdO(_, _, _, smem_pipe_read.index()), tdPrdP);
            warpgroup_wait<1>();
            Tensor cS = cute::make_identity_tensor(select<1, 0>(TileShape_MNK{}));
            Tensor taccScS = thread_mma_SdP.partition_C(cS);
            #pragma unroll
            for (int i = 0; i < size(tSrS); ++i) {
                if (int(get<0>(taccScS(i))) >= int(seqlen_k - n_block * kBlockN)) { tSrS(i) = -INFINITY; }
            }
            // Reshape tSrS from ((2, 2, V), MMA_N, MMA_M) to (nrow=(2, V, MMA_M), ncol=(2, MMA_N))
            Tensor scores = make_tensor(tSrS.data(), flash::convert_layout_acc_transposed_rowcol(tSrS.layout()));
            // if (blockIdx.x == 0 && threadIdx.x == 128) { print_tensor(tLSErLSE); }
            // if (blockIdx.x == 0 && threadIdx.x == 128) { print_tensor(scores); }
            flash::scale_apply_exp2</*Scale_max=*/false, /*Check_inf=*/false>(scores, group_modes<0, 2>(tLSErLSE), params.softmax_scale_log2);
            // if (blockIdx.x == 0 && threadIdx.x == 128) { print_tensor(scores); }

            Tensor tLSErdPsum = make_fragment_like(tLSEsdPsum(_, _, _0{}));
            cute::copy(tLSEsdPsum(_, _, smem_pipe_read.index()), tLSErdPsum);

            // Convert scores from fp32 to fp16/bf16
            Tensor rP = flash::convert_type<Element>(tSrS);

            warpgroup_wait<0>();
            // Reshape tdPrdP from ((2, 2, V), MMA_N, MMA_M) to (nrow=(2, V, MMA_M), ncol=(2, MMA_N))
            Tensor dS = make_tensor(tdPrdP.data(), scores.layout());
            #pragma unroll
            for (int mi = 0; mi < size<0>(dS); ++mi) {
                #pragma unroll
                for (int ni = 0; ni < size<1>(dS); ++ni) { dS(mi, ni) = scores(mi, ni) * (dS(mi, ni) - tLSErdPsum(mi)); }
            }
            // if (blockIdx.x == 0 && threadIdx.x == 128) { print_tensor(dS); }
            Tensor rdS = flash::convert_type<Element>(tdPrdP);

            Tensor tdSadS = smem_thr_copy_PdS.retile_S(rdS);     // ((Atom,AtomNum), MMA_N, MMA_N)
            cute::copy(smem_tiled_copy_PdS, tdSadS, tdSsdS(_, _, _, smem_pipe_read.index()));

            Tensor tdVrP = make_tensor(rP.data(), convert_layout_acc_Aregs<TiledMmadKV>(tSrS.layout()));
            flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(tiled_mma_dKV, tdVrP, tdVrdO(_, _, _, smem_pipe_read.index()), tdVrdV);

            Tensor tdKrdS = make_tensor(rdS.data(), convert_layout_acc_Aregs<TiledMmadKV>(tdPrdP.layout()));
            flash::gemm</*zero_init=*/false, /*wg_wait=*/1>(tiled_mma_dKV, tdKrdS, tdKrQ(_, _, _, smem_pipe_read.index()), tdKrdK);
            pipeline_do.consumer_release(smem_pipe_read);  // release dO

            compute_dQ();
            ++smem_pipe_read;
        }
        // if (blockIdx.x == 0 && threadIdx.x == 128) { print_tensor(tdVrdV); }
        #pragma unroll
        for (int i = 0; i < size(tdKrdK); ++i) { tdKrdK(i) *= params.softmax_scale; }
    }

};

} // namespace flash

