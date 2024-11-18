/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include "cutlass/pipeline/pipeline.hpp"

#include "cute/tensor.hpp"

#include "cutlass/gemm/collective/collective_builder.hpp"

#include "named_barrier.hpp"
#include "mask.h"
#include "pack_gqa.h"
#include "paged_kv.h"
#include "rotary.h"
#include "utils.h"

namespace flash {

using namespace cute;

template <int Stages, class ClusterShape_, class TileShape_MNK_, class Element_, class ElementAccum_, class ArchTag_,
          bool Is_causal_, bool Is_local_, bool Has_softcap_, bool Varlen_, bool PagedKV_, bool AppendKV_, bool PackGQA_, bool Split_, bool V_colmajor_>
struct CollectiveMainloopFwd {

    static constexpr int kStages = Stages;
    using ClusterShape = ClusterShape_;
    using TileShape_MNK = TileShape_MNK_;
    using Element = Element_;
    using ElementAccum = ElementAccum_;
    using ArchTag = ArchTag_;
    static constexpr bool Is_FP8 = cute::is_same_v<Element, cutlass::float_e4m3_t> || cute::is_same_v<Element, cutlass::float_e5m2_t>;;
    static constexpr bool Is_causal = Is_causal_;
    static constexpr bool Is_local = Is_local_;
    static constexpr bool Has_softcap = Has_softcap_;
    static constexpr bool Varlen = Varlen_;
    static constexpr bool PagedKV = PagedKV_;
    static constexpr bool AppendKV = AppendKV_;
    static constexpr bool PackGQA = PackGQA_;
    static constexpr bool Split = Split_;
    static constexpr bool V_colmajor = V_colmajor_;
    static constexpr bool Transpose_V = Is_FP8 && !V_colmajor;
    static constexpr bool Use_TMA_Q = !PackGQA;
    static constexpr bool Use_TMA_KV = !PagedKV;
    static_assert(Use_TMA_KV || size(ClusterShape{}) == 1, "If not using TMA for KV, ClusterShape must be 1");
    static_assert(Use_TMA_KV || !V_colmajor, "If not using TMA for KV, V_colmajor is not supported");

    static_assert(ArchTag::kMinComputeCapability >= 90);

    static constexpr cute::GMMA::Major MmaMajorV = !Is_FP8 && !V_colmajor ? GMMA::Major::MN : GMMA::Major::K;
    static constexpr cute::GMMA::Major TmaMajorV = !V_colmajor ? GMMA::Major::MN : GMMA::Major::K;

    static constexpr int kBlockM = get<0>(TileShape_MNK{});
    static constexpr int kBlockN = get<1>(TileShape_MNK{});
    static constexpr int kHeadDim = get<2>(TileShape_MNK{});

    using AtomLayoutMNK = Layout<Shape<Int<kBlockM / 64>, _1, _1>>;
    using TiledMma0 = decltype(cute::make_tiled_mma(
        cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShape_MNK>(),
        AtomLayoutMNK{}));
    using TiledMma1 = decltype(cute::make_tiled_mma(
        cute::GMMA::rs_op_selector<Element, Element, ElementAccum, decltype(select<0, 2, 1>(TileShape_MNK{})),
                                   GMMA::Major::K, MmaMajorV>(),
        AtomLayoutMNK{}));

    static constexpr int NumMmaThreads = size(TiledMma0{});
    static constexpr int NumProducerThreads = !Transpose_V && Use_TMA_KV ? cutlass::NumThreadsPerWarp : cutlass::NumThreadsPerWarpGroup;
    static_assert(NumMmaThreads % cutlass::NumThreadsPerWarpGroup == 0);
    static constexpr int NumMmaWarpGroups = NumMmaThreads / cutlass::NumThreadsPerWarpGroup;
    static_assert(NumMmaWarpGroups == 1 || NumMmaWarpGroups == 2 || NumMmaWarpGroups == 3);


    using SmemLayoutAtomQ = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQ{}, select<0, 2>(TileShape_MNK{})));

    using SmemLayoutAtomK = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutK = decltype(tile_to_shape(
        SmemLayoutAtomK{},
        make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{})));

    using SmemLayoutAtomVt = decltype(cutlass::gemm::collective::detail::ss_smem_selector<TmaMajorV, Element,
        decltype(cute::get<2>(TileShape_MNK{})), decltype(cute::get<1>(TileShape_MNK{}))>());
    using SmemLayoutVt = decltype(tile_to_shape(
        SmemLayoutAtomVt{},
        make_shape(shape<2>(TileShape_MNK{}), shape<1>(TileShape_MNK{}), Int<kStages>{}),
        std::conditional_t<TmaMajorV == GMMA::Major::K, cute::Step<_1, _2, _3>, cute::Step<_2, _1, _3>>{}));

    using SmemLayoutAtomVtMma = decltype(cutlass::gemm::collective::detail::ss_smem_selector<MmaMajorV, Element,
        decltype(cute::get<2>(TileShape_MNK{})), decltype(cute::get<1>(TileShape_MNK{}))>());
    using SmemLayoutVtMma = decltype(tile_to_shape(
        SmemLayoutAtomVtMma{},
        make_shape(shape<2>(TileShape_MNK{}), shape<1>(TileShape_MNK{}), Int<kStages>{}),
        std::conditional_t<MmaMajorV == GMMA::Major::K, cute::Step<_1, _2, _3>, cute::Step<_2, _1, _3>>{}));

    // Only used if we're using cp.async to load V
    using SmemLayoutAtomVCpAsync = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutVCpAsync = decltype(tile_to_shape(
        SmemLayoutAtomVCpAsync{},
        make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{})));

    // Use LDSM.T and STSM to transpose V in the case of FP8 and V being row-major.
    // For FP16/BF16 we don't do any transposing.
    static_assert(!Transpose_V || (kHeadDim % 32 == 0 && kBlockN % 32 == 0));
    static constexpr bool kHeadDim_multiple_64 = kHeadDim % 64 == 0;
    // Either kHeadDim is a multiple of 64 (in which case we use a block size of 64 x 32 for the transpose),
    // or we need kBlockN to be a multiple of 64 (in which case we use a block size of 32 x 64 for the transpose).
    static_assert(!Transpose_V || (kHeadDim_multiple_64 || kBlockN % 64 == 0));
    using LDSM_thread_shape  = std::conditional_t<kHeadDim_multiple_64, Shape<_32, _4, _1, _1>, Shape<_16, _4, _1, _2>>;
    using LDSM_thread_stride = std::conditional_t<kHeadDim_multiple_64, Stride<_4, _1, _0, _0>, Stride<_4, _1, _0, _64>>;
    using LDSM_value_shape = Shape<_2, _2, _1, _4>;
    using LDSM_value_stride = Stride<_1, _2, _16, _4>;
    using LDSM_divide_shape = std::conditional_t<kHeadDim_multiple_64, Shape<_64, _8>, Shape<_32, _8>>;
    using S2RTiledCopyVt = decltype(make_tiled_copy(
        Copy_Atom<SM75_U16x8_LDSM_T, Element>{}, Layout<LDSM_thread_shape, LDSM_thread_stride>{},
        Layout<LDSM_value_shape, LDSM_value_stride>{}));

    using STSM_thread_shape  = std::conditional_t<kHeadDim_multiple_64, Shape<_8, _4, _4, _1>, Shape<_8, _4, _2, _2>>;
    using STSM_thread_stride = std::conditional_t<kHeadDim_multiple_64, Stride<_4, _1, _32, _0>, Stride<_4, _1, _32, _64>>;
    using STSM_value_shape = Shape<_1, _4, _2, _2>;
    using STSM_value_stride = Stride<_0, _1, _4, _8>;
    using STSM_divide_shape = Shape<_8, _16>;
    // These will not permute the columns of V (the kHeadDim dimension) but incur bank conflicts
    // so a little slower (e.g. 1150 TFLOPS for hdim 256 instead of 1200 TFLOPS).
    // Instead we will permute the cols of V, and un-permute the cols of O in the epilogue.
    // using STSM_value_shape = Shape<_2, _4, _1, _2>;
    // using STSM_value_stride = Stride<_4, _1, _0, _8>;
    // using STSM_divide_shape = Shape<_16, _16>;
    using R2STiledCopyV = decltype(make_tiled_copy(
        Copy_Atom<SM90_U32x4_STSM_N, Element>{}, Layout<STSM_thread_shape, STSM_thread_stride>{},
        Layout<STSM_value_shape, STSM_value_stride>{}));

    using GmemTiledCopyQ = cute::SM90_TMA_LOAD;
    using GmemTiledCopyKV = decltype(cutlass::gemm::collective::detail::sm90_cluster_shape_to_tma_atom(shape<0>(ClusterShape{})));

    // We use CpAsync for Q load if PackGQA, since TMA doesn't work there
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "Headdim must be a multiple of kGmemElemsPerLoad");
    // We want each "row" to have 64 elements (128 bytes, i.e. 1 cache line). E.g. if hdim=128, we want each
    // thread to have 4 loads in the M direction and 2 vectorized load in the K direction.
    // In the case of PackGQA, this reduces the number of times we need to call divmod.
    static constexpr int kBlockKGmem = kHeadDim % 128 == 0 ? (sizeof(Element) == 2 ? 64 : 128) : (kHeadDim % 64 == 0 ? 64 : 32);
    static constexpr int kGmemThreadsPerRow = kBlockKGmem / kGmemElemsPerLoad;
    static_assert(NumMmaThreads % kGmemThreadsPerRow == 0, "NumMmaThreads must be a multiple of kGmemThreadsPerRow");
    // We assume threads loading the same row are in the same warp. This is for an optimization in PagedKV where
    // these threads share the same page table entry and share the work of computing pointers to paged K and paged V.
    static_assert(cutlass::NumThreadsPerWarp % kGmemThreadsPerRow == 0, "kGmemThreadsPerRow must divide NumThreadsPerWarp");
    using GmemLayoutAtom = Layout<Shape <Int<NumMmaThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                  Stride<Int<kGmemThreadsPerRow>, _1>>;
    // If AppendKV, we'll be loading Q for rotary, and we assume divisibility to avoid predication
    static_assert(!AppendKV || kBlockM % CUTE_STATIC_V(shape<0>(GmemLayoutAtom{})) == 0, "kBlockM must be a multiple of NumMmaThreads / kGmemThreadsPerRow");
    using GmemTiledCopyAppendKV = decltype(
        make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));  // Val layout, 8 or 16 vals per store

    using GmemTiledCopyRotary = decltype(
        make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<64>, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoad / 2>>>{}));  // Val layout, 4 or 8 vals per store

    using ShapeQKV = cute::Shape<int32_t, int32_t, int32_t, int32_t>;  // (seqlen, d, head, batch)
    using StrideQK = cute::Stride<int64_t, _1, int64_t, int64_t>;
    using StrideV = std::conditional_t<!V_colmajor, StrideQK, cute::Stride<_1, int64_t, int64_t, int64_t>>;
    // ((qhead_per_khead, seqlen_q), d, nheads_kv, batch, num_splits)
    using ShapeQPacked = std::conditional_t<!PackGQA, ShapeQKV, cute::Shape<cute::Shape<int32_t, int32_t>, int32_t, int32_t, int32_t>>;
    using StrideQPacked = std::conditional_t<!PackGQA, StrideQK, cute::Stride<cute::Stride<int64_t, int64_t>, _1, int64_t, int64_t>>;
    using ShapePageTable = cute::Shape<int32_t, int32_t>;  // (batch, max_num_pages_per_seq)
    using StridePageTable = cute::Stride<int64_t, _1>;
    using ShapeRotary = cute::Shape<int32_t, int32_t>;  // (seqlen_ro, rotary_dim // 2)
    using StrideRotary = cute::Stride<int64_t, _1>;

    using TMA_Q = decltype(make_tma_copy_A_sm90(
        GmemTiledCopyQ{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQKV{}, StrideQK{}),
        SmemLayoutQ{},
        TileShape_MNK{},
        ClusterShape{}));

    using TMA_K = decltype(make_tma_copy_B_sm90(
        GmemTiledCopyKV{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQKV{}, StrideQK{}),
        take<0, 2>(SmemLayoutK{}),
        TileShape_MNK{},
        ClusterShape{})); // mcast along M mode for this N load, if any

    using TMA_V = decltype(make_tma_copy(
        GmemTiledCopyKV{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQKV{}, select<1, 0, 2, 3>(StrideV{})),
        take<0, 2>(SmemLayoutVt{}),
        select<2, 1>(TileShape_MNK{}),
        size<0>(ClusterShape{}))); // mcast along M mode for this N load, if any

    // Set the bytes transferred in this TMA transaction (may involve multiple issues)
    static constexpr uint32_t TmaTransactionBytesQ = static_cast<uint32_t>(size(SmemLayoutQ{}) * cutlass::sizeof_bits_v<Element> / 8);
    static constexpr uint32_t TmaTransactionBytesK = static_cast<uint32_t>(size(take<0, 2>(SmemLayoutK{})) * cutlass::sizeof_bits_v<Element> / 8);
    static constexpr uint32_t TmaTransactionBytesV = static_cast<uint32_t>(size(take<0, 2>(SmemLayoutVt{})) * cutlass::sizeof_bits_v<Element> / 8);
    static_assert(TmaTransactionBytesK == TmaTransactionBytesV);

    using MainloopPipelineK = std::conditional_t<Use_TMA_KV, typename cutlass::PipelineTmaAsync<kStages>, typename cutlass::PipelineAsync<kStages>>;
    using MainloopPipelineV = std::conditional_t<!Transpose_V && Use_TMA_KV, typename cutlass::PipelineTmaAsync<kStages>, typename cutlass::PipelineAsync<kStages>>;
    using MainloopPipelineVt = std::conditional_t<Use_TMA_KV, typename cutlass::PipelineTmaAsync<kStages>, typename cutlass::PipelineAsync<kStages>>;
    // We always use TMA for K_new and V_new
    using MainloopPipelineKVNew = typename cutlass::PipelineTmaAsync<kStages>;
    using PipelineState = cutlass::PipelineState<kStages>;

    // If PackGQA, we use cp.async (instead of TMA) to load Q, so we want smem_q to be aligned
    // and have sQ being position_independent_swizzle_tensor.
    // If !Use_TMA_KV, we use cp.async (instead of TMA) to load K & V, so we want smem_k and smem_v to be aligned.
    static constexpr size_t SmemAlignmentQ = Use_TMA_Q && !AppendKV ? 128 : cutlass::detail::alignment_for_swizzle(SmemLayoutQ{});
    static constexpr size_t SmemAlignmentK = Use_TMA_KV && !AppendKV ? 128 : cutlass::detail::alignment_for_swizzle(SmemLayoutK{});
    static constexpr size_t SmemAlignmentVtNoTranspose = cutlass::detail::alignment_for_swizzle(SmemLayoutVt{});
    static_assert(SmemAlignmentQ >= 128 and SmemAlignmentK >= 128 && SmemAlignmentVtNoTranspose >= 128, "Require at least 128B alignment");

    struct TensorStorageNoTranspose : cute::aligned_struct<cute::max(SmemAlignmentQ, SmemAlignmentK, SmemAlignmentVtNoTranspose)> {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutVt>, SmemAlignmentVtNoTranspose> smem_v;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>, SmemAlignmentQ> smem_q;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>, SmemAlignmentK> smem_k;
    };

    static constexpr size_t SmemAlignmentVt = cutlass::detail::alignment_for_swizzle(SmemLayoutVt{});
    static constexpr size_t SmemAlignmentV = cutlass::detail::alignment_for_swizzle(SmemLayoutVtMma{});
    static_assert(SmemAlignmentVt >= 128 and SmemAlignmentV >= 128, "Require at least 128B alignment");
    struct TensorStorageTransposeV : cute::aligned_struct<cute::max(SmemAlignmentQ, SmemAlignmentK, SmemAlignmentV)> {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutVtMma>, SmemAlignmentV> smem_v;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutVt>, SmemAlignmentVt> smem_vt;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>, SmemAlignmentQ> smem_q;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>, SmemAlignmentK> smem_k;
    };

    using TensorStorage = std::conditional_t<!Transpose_V, TensorStorageNoTranspose, TensorStorageTransposeV>;

    // These are tuned for speed. They don't affect correctness.
    static constexpr bool UseSchedulerBarrier = (NumMmaWarpGroups >= 2) && (!Is_FP8 ? kHeadDim <= 128 : kHeadDim >= 128);
    static constexpr bool RescaleOBeforeGemm = kHeadDim > 128 && (!Is_FP8 || V_colmajor);

    // Host side kernel arguments
    struct Arguments {
        Element const* const ptr_Q;
        ShapeQKV const shape_Q;
        StrideQK const stride_Q;
        Element* const ptr_K;  // Not Element const* since we might append to KV cache in-place
        ShapeQKV const shape_K;
        StrideQK const stride_K;
        Element* const ptr_V;
        StrideV const stride_V;
        Element const* const ptr_K_new;
        ShapeQKV const shape_K_new;
        StrideQK const stride_K_new;
        Element const* const ptr_V_new;
        StrideV const stride_V_new;
        Element const* const ptr_rotary_cos;
        ShapeRotary const shape_rotary;
        StrideRotary const stride_rotary_cos;
        Element const* const ptr_rotary_sin;
        StrideRotary const stride_rotary_sin;
        int const* const ptr_pagetable;
        ShapePageTable const shape_pagetable;
        StridePageTable const stride_pagetable;
        float const softmax_scale;
        float const* ptr_q_descale = nullptr, *ptr_k_descale = nullptr, *ptr_v_descale = nullptr;
        int const window_size_left = -1, window_size_right = -1, sink_token_length = 0;
        float const softcap_val;
        int const num_splits;
        int const* const kv_batch_idx = nullptr;
        int const* const cu_seqlens_q = nullptr;
        int const* const cu_seqlens_k = nullptr;
        int const* const cu_seqlens_k_new = nullptr;
        int const* const seqused_q = nullptr;
        int const* const seqused_k = nullptr;
        int const* const leftpad_k = nullptr;
    };

    // Device side kernel params
    struct Params {
        Element const* const ptr_Q;
        ShapeQKV const shape_Q;
        StrideQK const stride_Q;
        ShapeQPacked const shape_Q_packed;
        StrideQPacked const stride_Q_packed;
        Element* const ptr_K;
        ShapeQKV const shape_K;
        StrideQK const stride_K;
        Element* const ptr_V;
        StrideV const stride_V;
        Element const* const ptr_K_new;
        ShapeQKV const shape_K_new;
        StrideQK const stride_K_new;
        Element const* const ptr_V_new;
        StrideV const stride_V_new;
        Element const* const ptr_rotary_cos;
        ShapeRotary const shape_rotary;
        StrideRotary const stride_rotary_cos;
        Element const* const ptr_rotary_sin;
        StrideRotary const stride_rotary_sin;
        int const* const ptr_pagetable;
        ShapePageTable const shape_pagetable;
        StridePageTable const stride_pagetable;
        cutlass::FastDivmod page_size_divmod;
        cutlass::FastDivmod qhead_per_khead_divmod;
        TMA_Q tma_load_Q;
        TMA_K tma_load_K;
        TMA_V tma_load_V;
        TMA_K tma_load_K_new;
        TMA_V tma_load_V_new;
        float const softmax_scale_log2;
        float const* ptr_q_descale = nullptr, *ptr_k_descale = nullptr, *ptr_v_descale = nullptr;
        float const softcap_val;
        int const window_size_left, window_size_right, sink_token_length;
        int const num_splits;
        int const* const kv_batch_idx = nullptr;
        int const* const cu_seqlens_q = nullptr;
        int const* const cu_seqlens_k = nullptr;
        int const* const cu_seqlens_k_new = nullptr;
        int const* const seqused_q = nullptr;
        int const* const seqused_k = nullptr;
        int const* const leftpad_k = nullptr;
    };

    static Params
    to_underlying_arguments(Arguments const& args) {
        Tensor mQ = make_tensor(make_gmem_ptr(args.ptr_Q), args.shape_Q, args.stride_Q);
        TMA_Q tma_load_Q = make_tma_copy_A_sm90(
            GmemTiledCopyQ{},
            mQ,
            SmemLayoutQ{},
            TileShape_MNK{},
            ClusterShape{}); // no mcast for Q
        Tensor mK = make_tensor(make_gmem_ptr(args.ptr_K), args.shape_K, args.stride_K);
        TMA_K tma_load_K = make_tma_copy_B_sm90(
            GmemTiledCopyKV{},
            mK,
            take<0, 2>(SmemLayoutK{}),
            TileShape_MNK{},
            ClusterShape{}); // mcast along M mode for this N load, if any
        Tensor mV = make_tensor(make_gmem_ptr(args.ptr_V), select<1, 0, 2, 3>(args.shape_K), select<1, 0, 2, 3>(args.stride_V));
        TMA_V tma_load_V = make_tma_copy(
            GmemTiledCopyKV{},
            mV,
            take<0, 2>(SmemLayoutVt{}),
            select<2, 1>(TileShape_MNK{}),
            size<0>(ClusterShape{})); // mcast along M mode for this N load, if any
        Tensor mKnew = make_tensor(make_gmem_ptr(args.ptr_K_new), args.shape_K_new, args.stride_K_new);
        TMA_K tma_load_K_new = make_tma_copy_B_sm90(
            GmemTiledCopyKV{},
            cute::conditional_return<AppendKV>(mKnew, mK),
            take<0, 2>(SmemLayoutK{}),
            TileShape_MNK{},
            ClusterShape{}); // mcast along M mode for this N load, if any
        Tensor mVnew = make_tensor(make_gmem_ptr(args.ptr_V_new), select<1, 0, 2, 3>(args.shape_K_new), select<1, 0, 2, 3>(args.stride_V_new));
        TMA_V tma_load_V_new = make_tma_copy(
            GmemTiledCopyKV{},
            cute::conditional_return<AppendKV>(mVnew, mV),
            take<0, 2>(SmemLayoutVt{}),
            select<2, 1>(TileShape_MNK{}),
            size<0>(ClusterShape{})); // mcast along M mode for this N load, if any
        // If PackGQA, reshape Q to be ((qhead_per_khead, seqlen_q), head_size, nhead_k, batch_size)
        int const qhead_per_khead = !PackGQA ? 1 : cute::ceil_div(get<2>(args.shape_Q), get<2>(args.shape_K));
        auto const shape_Q_packed = cute::conditional_return<!PackGQA>(
            args.shape_Q,
            make_shape(make_shape(qhead_per_khead, get<0>(args.shape_Q)), get<1>(args.shape_Q), get<2>(args.shape_K), get<3>(args.shape_Q))
        );
        auto const stride_Q_packed = cute::conditional_return<!PackGQA>(
            args.stride_Q,
            make_stride(make_stride(get<2>(args.stride_Q), get<0>(args.stride_Q)), get<1>(args.stride_Q), get<2>(args.stride_Q) * qhead_per_khead, get<3>(args.stride_Q))
        );
        if (get<1>(args.shape_rotary) > 0) {
            assert(args.ptr_rotary_cos != nullptr && args.ptr_rotary_sin != nullptr);
        }
        // If there's tanh softcapping, we do tanh(scores * softmax_scale / softcap_val) * softcap_val.
        // Right after this, we multiply by log2(e) before applying exp2.
        // To reduce the number of instructions, we instead pre-multiply softmax_scale / softcap_val
        // (assigning it to params.softcap_val) and pre-multiply softcap_val * log2(e)
        // (assigning it to params.softmax_scale_log2).
        return {args.ptr_Q, args.shape_Q, args.stride_Q, shape_Q_packed, stride_Q_packed,
                args.ptr_K, args.shape_K, args.stride_K, args.ptr_V, args.stride_V,
                args.ptr_K_new, args.shape_K_new, args.stride_K_new, args.ptr_V_new, args.stride_V_new,
                args.ptr_rotary_cos, args.shape_rotary, args.stride_rotary_cos,
                args.ptr_rotary_sin, args.stride_rotary_sin,
                args.ptr_pagetable, args.shape_pagetable, args.stride_pagetable,
                cutlass::FastDivmod(int(get<0>(args.shape_K))),
                cutlass::FastDivmod(cute::ceil_div(get<2>(args.shape_Q), get<2>(args.shape_K))),
                tma_load_Q, tma_load_K, tma_load_V, tma_load_K_new, tma_load_V_new,
                !Has_softcap ? float(args.softmax_scale * M_LOG2E) : float(args.softcap_val * M_LOG2E),
                args.ptr_q_descale, args.ptr_k_descale, args.ptr_v_descale,
                !Has_softcap ? 0.f : args.softmax_scale / args.softcap_val,
                args.window_size_left, args.window_size_right, args.sink_token_length,
                !Split ? 1 : args.num_splits,
                args.kv_batch_idx,
                args.cu_seqlens_q, args.cu_seqlens_k, args.cu_seqlens_k_new,
                args.seqused_q, args.seqused_k, args.leftpad_k};
    }

    /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
    CUTLASS_DEVICE
    static void prefetch_tma_descriptors(Params const& params) {
        if constexpr (Use_TMA_Q) {
            cute::prefetch_tma_descriptor(params.tma_load_Q.get_tma_descriptor());
        }
        if constexpr (Use_TMA_KV) {
            cute::prefetch_tma_descriptor(params.tma_load_K.get_tma_descriptor());
            cute::prefetch_tma_descriptor(params.tma_load_V.get_tma_descriptor());
        }
        if constexpr (AppendKV) {
            cute::prefetch_tma_descriptor(params.tma_load_K_new.get_tma_descriptor());
            cute::prefetch_tma_descriptor(params.tma_load_V_new.get_tma_descriptor());
        }
    }

    CUTLASS_DEVICE
    int get_seqlen_q(Params const& params, int bidb) {
        if constexpr (!Varlen) {
            return size<0>(params.shape_Q);
        } else {
            return params.seqused_q ? params.seqused_q[bidb] : (params.cu_seqlens_q ? params.cu_seqlens_q[bidb + 1] - params.cu_seqlens_q[bidb] : size<0>(params.shape_Q));
        }
    }

    template<bool Seqlen_before_append_KV=false>
    CUTLASS_DEVICE
    int get_seqlen_k(Params const& params, int bidb) {
        int const seqlen_k_novarlen = !PagedKV ? size<0>(params.shape_K) : size<0>(params.shape_K) * size<1>(params.shape_pagetable);
        if constexpr (!Varlen) {
            return seqlen_k_novarlen;
        } else {
            int const leftpad = params.leftpad_k ? params.leftpad_k[bidb] : 0;
            int seqlen_k = (params.seqused_k ? params.seqused_k[bidb] : (params.cu_seqlens_k ? params.cu_seqlens_k[bidb + 1] - params.cu_seqlens_k[bidb] : seqlen_k_novarlen)) - leftpad;
            if constexpr (AppendKV && !Seqlen_before_append_KV) { seqlen_k += get_seqlen_k_new(params, bidb); }
            return seqlen_k;
        }
    }

    CUTLASS_DEVICE
    int get_seqlen_k_new(Params const& params, int bidb) {
        // If there's K_new then it must be varlen
        return params.cu_seqlens_k_new ? params.cu_seqlens_k_new[bidb + 1] - params.cu_seqlens_k_new[bidb] : size<0>(params.shape_K_new);
    }

    CUTLASS_DEVICE
    cute::tuple<int, int> get_n_block_min_max(Params const& params, int m_block, int bidb, int split_idx=0, int num_splits=1) {
        static constexpr int kBlockM = get<0>(TileShape_MNK{});
        static constexpr int kBlockN = get<1>(TileShape_MNK{});
        int const seqlen_k = get_seqlen_k(params, bidb);
        int const seqlen_q = get_seqlen_q(params, bidb);
        int n_block_max = cute::ceil_div(seqlen_k, kBlockN);
        if constexpr (Is_causal || Is_local) {
            int m_idx_max = (m_block + 1) * kBlockM;
            // TODO: check off-by-1 error
            if (PackGQA) { m_idx_max = params.qhead_per_khead_divmod.divide(m_idx_max - 1) + 1 ; }
            n_block_max = std::min(n_block_max,
                                   cute::ceil_div(m_idx_max + seqlen_k - seqlen_q + params.window_size_right, kBlockN));
        }
        int n_block_min = 0;
        if constexpr (Is_local) {
            int m_idx_min = m_block * kBlockM;
            if (PackGQA) { m_idx_min = params.qhead_per_khead_divmod.divide(m_idx_min); }
            n_block_min = std::max(int(0), (m_idx_min + seqlen_k - seqlen_q - params.window_size_left) / kBlockN);
        }
        // if (threadIdx.x == 128) { printf("Inside, bid.x = %d, bid.y = %d, bid.z = %d, split_idx = %d, n_block_min: %d, n_block_max: %d\n", blockIdx.x, blockIdx.y, blockIdx.z, split_idx, n_block_min, n_block_max); }
        if constexpr (Split) {
            int num_n_blocks_per_split = n_block_max <= n_block_min ? 0 : cute::ceil_div(n_block_max - n_block_min, num_splits);
            n_block_min = n_block_min + split_idx * num_n_blocks_per_split;
            n_block_max = std::min(n_block_min + num_n_blocks_per_split, n_block_max);
        }
        // if (threadIdx.x == 128) { printf("After split, inside, bid.y = %d, bid.z = %d, split_idx = %d, n_block_min: %d, n_block_max: %d\n", blockIdx.y, blockIdx.z, split_idx, n_block_min, n_block_max); }
        return {n_block_min, n_block_max};
    }

    template <typename SchedulerPrefetch, typename SharedStorage>
    CUTLASS_DEVICE void
    load(Params const& params,
         MainloopPipelineK pipeline_k,
         MainloopPipelineV pipeline_v,
         MainloopPipelineVt pipeline_vt,
         PipelineState& smem_pipe_write,
         SharedStorage &shared_storage,
         SchedulerPrefetch const& scheduler_prefetch,
         cute::tuple<int32_t, int32_t, int32_t, int32_t> block_coord,
         int work_idx
         ) {

        Tensor sQ = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()), SmemLayoutQ{});
        Tensor sK = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()), SmemLayoutK{});
        Tensor sK_pi = as_position_independent_swizzle_tensor(sK);
        // as_position_independent_swizzle_tensor makes address calculation easier when we do LDSM & STSM to transpose.
        // But it requires smem_vt and smem_v to be aligned to e.g 512 bytes.
        Tensor sVt = [&] {
            if constexpr (!Transpose_V) {
                return make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutVt{});
            } else {
                return cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_vt.data()), SmemLayoutVt{}));
            }
        }();
        // Only used if Transpose_V
        Tensor sV = cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutVtMma{}));
        // Only used if we're using cp.async to load V
        Tensor sVcpasync = [&] {
            if constexpr (!Transpose_V) {
                return cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutVCpAsync{}));
            } else {
                return cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_vt.data()), SmemLayoutVCpAsync{}));
            }
        }();

        auto [m_block, bidh, bidb, split_idx] = block_coord;
        int const thread_idx = threadIdx.x % NumProducerThreads;
        int const bidh_kv = !PackGQA ? params.qhead_per_khead_divmod.divide(bidh) : bidh;
        int const bidb_kv = params.kv_batch_idx == nullptr ? bidb : params.kv_batch_idx[bidb];

        // Prepare the TMA loads
        uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();
        constexpr uint32_t cluster_shape_x = get<0>(ClusterShape());
        uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x, block_rank_in_cluster / cluster_shape_x};

        bool const is_varlen_q = Varlen && params.cu_seqlens_q;
        bool const is_varlen_k = Varlen && params.cu_seqlens_k;
        Tensor mQ = params.tma_load_Q.get_tma_tensor(params.shape_Q)(_, _, bidh, !is_varlen_q ? bidb : 0);
        Tensor mK_TMA = params.tma_load_K.get_tma_tensor(params.shape_K)(_, _, bidh_kv, !is_varlen_k ? bidb_kv : 0);
        Tensor mVt_TMA = params.tma_load_V.get_tma_tensor(select<1, 0, 2, 3>(params.shape_K))(_, _, bidh_kv, !is_varlen_k ? bidb_kv : 0);

        Tensor gQ = local_tile(domain_offset(make_coord(!is_varlen_q ? 0 : params.cu_seqlens_q[bidb], _0{}), mQ), select<0, 2>(TileShape_MNK{}), make_coord(m_block, _0{}));  // (M, K)
        int const leftpad_k = Varlen && params.leftpad_k ? params.leftpad_k[bidb] : 0;
        // if (cute::thread0()) { printf("Varlen = %d, params.leftpad_k = %p, leftpad_k = %d\n", Varlen, params.leftpad_k, leftpad_k); }
        int const offset_k = !Varlen ? 0 : (params.cu_seqlens_k ? params.cu_seqlens_k[bidb] : 0) + leftpad_k;
        Tensor gK_TMA = local_tile(domain_offset(make_coord(offset_k, _0{}), mK_TMA), select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (N, K, _)
        Tensor gVt_TMA = local_tile(domain_offset(make_coord(_0{}, offset_k), mVt_TMA), select<2, 1>(TileShape_MNK{}), make_coord(_0{}, _));  // (K, N, _)

        auto block_tma_Q = params.tma_load_Q.get_slice(_0{});
        Tensor tQgQ = group_modes<0, 3>(block_tma_Q.partition_S(gQ));  // (TMA)
        Tensor tQsQ = group_modes<0, 3>(block_tma_Q.partition_D(sQ));  // (TMA)
        // tma_partition doesn't handle position_independent_swizzle_tensor correctly, so we need to do it manually
        auto block_tma_K = params.tma_load_K.get_slice(cluster_local_block_id.x);
        Tensor tKgK_TMA = group_modes<0, 3>(block_tma_K.partition_S(gK_TMA));  // (TMA, k)
        Tensor tKsK_TMA = group_modes<0, 3>(block_tma_K.partition_D(sK));  // (TMA, PIPE)
        auto block_tma_V = params.tma_load_V.get_slice(cluster_local_block_id.x);
        Tensor tVgVt_TMA = group_modes<0, 3>(block_tma_V.partition_S(gVt_TMA));  // (TMA, k)
        Tensor tVsVt_TMA = group_modes<0, 3>(block_tma_V.partition_D(sVt));  // (TMA, PIPE)

        PagedKVManager<get<1>(TileShape_MNK{}), get<2>(TileShape_MNK{}), NumProducerThreads, Element, Transpose_V> paged_kv_manager(
            params.ptr_pagetable, params.shape_pagetable, params.stride_pagetable,
            params.ptr_K, params.shape_K, params.stride_K,
            params.ptr_V, params.stride_V,
            params.page_size_divmod, bidb_kv, bidh_kv, thread_idx, get_seqlen_k(params, bidb), leftpad_k
        );

        // Set up for transposing V, only used if Transpose_V
        S2RTiledCopyVt s2r_tiled_copy_vt;
        R2STiledCopyV r2s_tiled_copy_v;
        auto s2r_thr_copy_vt = s2r_tiled_copy_vt.get_thread_slice(thread_idx);
        auto r2s_thr_copy_v = r2s_tiled_copy_v.get_thread_slice(thread_idx);
        // flat_divide(sVt, LDSM_divide_shape{}):  (64, 8, kHeadDim / 64, kBlockN / 8, kStages)
        Tensor tTranssVt_ = s2r_thr_copy_vt.partition_S(flat_divide(sVt, LDSM_divide_shape{}));  // ((16, 1), 1, 1, kHeadDim / 64, kBlockN / 32, kStages)
        // flat_divide(sV, STSM_divide_shape{}):  (8, 16, kHeadDim / 8, (4, kBlockN / 64), kStages)
        Tensor tTranssV_ = r2s_thr_copy_v.partition_D(flat_divide(sV, STSM_divide_shape{}));  // ((16, 1), 1, 1, kHeadDim / 64, (2, kBlockN / 64), kStages)
        CUTE_STATIC_ASSERT_V(rank(tTranssVt_) == rank(tTranssV_));
        CUTE_STATIC_ASSERT_V(size<0>(tTranssVt_) == size<0>(tTranssV_));
        CUTE_STATIC_ASSERT_V(size<1>(tTranssVt_) == size<1>(tTranssV_));
        CUTE_STATIC_ASSERT_V(size<2>(tTranssVt_) == size<2>(tTranssV_));
        CUTE_STATIC_ASSERT_V(size<3>(tTranssVt_) == size<3>(tTranssV_));
        CUTE_STATIC_ASSERT_V(size<4>(tTranssVt_) == size<4>(tTranssV_));
        // Faster to have 2 LDSM.T, byte permute, STSM for better ILP
        static constexpr int Transpose_ILP = (size<2>(tTranssVt_) * size<3>(tTranssVt_)) % 2 == 0 ? 2 : 1;
        Tensor tTranssVt = logical_divide(group_modes<1, rank(tTranssVt_) - 1>(tTranssVt_), Shape<Underscore, Int<Transpose_ILP>>{});  // ((16, 1), (2, kHeadDim / 64 * kBlockN / 32 / 2), kStages)
        Tensor tTranssV = logical_divide(group_modes<1, rank(tTranssV_) - 1>(tTranssV_), Shape<Underscore, Int<Transpose_ILP>>{});  // ((16, 1), (2, kHeadDim / 64 * kBlockN / 32 / 2), kStages)
        auto transpose_V = [&](int stage) {
            if constexpr (Transpose_V) {
                #pragma unroll
                for (int i = 0; i < size<1, 1>(tTranssVt); ++i) {
                    Tensor tTransrV = make_fragment_like(tTranssV(_, make_coord(_, _0{}), _0{}));
                    static_assert(size<0>(tTransrV) == 16);
                    Tensor tTransrV_64 = recast<uint2>(tTransrV);
                    cute::copy(s2r_tiled_copy_vt, tTranssVt(_, make_coord(_, i), stage), tTransrV);
                    #pragma unroll
                    for (int j = 0; j < size(tTransrV_64); ++j) {
                        uint32_t upper = tTransrV_64[j].x;
                        uint32_t lower = tTransrV_64[j].y;
                        tTransrV_64[j].x = __byte_perm(upper, lower, 0x6420);
                        tTransrV_64[j].y = __byte_perm(upper, lower, 0x7531);
                    }
                    cute::copy(r2s_tiled_copy_v, tTransrV, tTranssV(_, make_coord(_, i), stage));
                }
            }
        };

        uint16_t mcast_mask_kv = 0;
        if constexpr (cute::is_same_v<GmemTiledCopyKV, SM90_TMA_LOAD_MULTICAST>) {
            auto block_layout = Layout<ClusterShape>{}; // (m,n) -> block_id
            for (int m = 0; m < size<0>(block_layout); ++m) {
                mcast_mask_kv |= (uint16_t(1) << block_layout(m, cluster_local_block_id.y, _0{}));
            }
        }

        auto load_K = [&] (int const n_block, auto const& smem_pipe_write, auto need_seqlenk_masking_type) {
            pipeline_k.producer_acquire(smem_pipe_write);
            if constexpr (Use_TMA_KV) {
                copy(params.tma_load_K.with(*pipeline_k.producer_get_barrier(smem_pipe_write), mcast_mask_kv, TMA::CacheHintSm90::EVICT_LAST),
                    tKgK_TMA(_, n_block), tKsK_TMA(_, smem_pipe_write.index()));
            } else {
                constexpr bool Seqlenk_mask = decltype(need_seqlenk_masking_type)::value;
                paged_kv_manager.template load_K<Seqlenk_mask>(n_block, sK_pi(_, _, smem_pipe_write.index()));
                pipeline_k.producer_commit(smem_pipe_write, cutlass::arch::cpasync_barrier_arrive);
            }
        };

        auto load_V = [&] (int const n_block, auto const& smem_pipe_write, auto need_seqlenk_masking_type) {
            auto pipeline_v_load = cute::conditional_return<!Transpose_V>(pipeline_v, pipeline_vt);
            pipeline_v_load.producer_acquire(smem_pipe_write);
            if constexpr (Use_TMA_KV) {
                copy(params.tma_load_V.with(*pipeline_v_load.producer_get_barrier(smem_pipe_write), mcast_mask_kv, TMA::CacheHintSm90::EVICT_LAST),
                    tVgVt_TMA(_, n_block), tVsVt_TMA(_, smem_pipe_write.index()));
            } else {
                constexpr bool Seqlenk_mask = decltype(need_seqlenk_masking_type)::value;
                paged_kv_manager.template load_V<Seqlenk_mask>(n_block, sVcpasync(_, _, smem_pipe_write.index()));
                pipeline_v_load.producer_commit(smem_pipe_write, cutlass::arch::cpasync_barrier_arrive);
            }
        };

        auto copy_Vt_to_V = [&] (auto const& smem_pipe_write) {
            // Instead of maintaining smem_pipe_read as a separate variable, we can just use smem_pipe_write,
            // and exploit the invariance that smem_pipe_write.phase() == smem_pipe_read.phase() ^ 1.
            // This saves 1 or 2 registers.
            PipelineState smem_pipe_read{smem_pipe_write.index(), smem_pipe_write.phase() ^ 1, smem_pipe_write.count()};
            pipeline_vt.consumer_wait(smem_pipe_read);
            pipeline_v.producer_acquire(smem_pipe_write);
            transpose_V(smem_pipe_write.index());
            // SMEM fence to make sure V is transposed before math
            cutlass::arch::fence_view_async_shared();
            pipeline_v.producer_commit(smem_pipe_write);
            // Very important: PipelineTmaAsync::consumer_release assumes that the warpgroup is synchronized
            // before calling. Without this we get race conditions.
            cutlass::arch::NamedBarrier::sync(cutlass::NumThreadsPerWarpGroup, static_cast<int>(FwdNamedBarriers::ProducerWG) /*id*/);
            pipeline_vt.consumer_release(smem_pipe_read);
        };

        int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
        // If this is true, we're guaranteed that only the first warp will execute this function
        static constexpr bool SingleProducerWarp = NumProducerThreads == cutlass::NumThreadsPerWarp;
        bool should_load_KV = !Use_TMA_KV || ((SingleProducerWarp || warp_idx_in_warpgroup == 0) && cute::elect_one_sync());

        auto [n_block_min, n_block_max] = get_n_block_min_max(params, m_block, bidb, split_idx, params.num_splits);
        int n_block = n_block_max - 1;

        if (should_load_KV) {
            if constexpr (PagedKV) {
                paged_kv_manager.template load_page_table<true /*Seqlenk_mask*/, true /*First_iter*/>(n_block);
            }
            if constexpr (Transpose_V) { load_V(n_block, smem_pipe_write, cute::bool_constant<true>{} /*Seqlenk_mask*/); }
            // if (thread_idx == 0) { printf("Producer: main load, before load_K, index = %d\n", smem_pipe_write.index());}
            load_K(n_block, smem_pipe_write, cute::bool_constant<true>{} /*Seqlenk_mask*/);
            // if (thread_idx == 0) { printf("Producer: main load, after loaf K, index = %d\n", smem_pipe_write.index());}
        }

        // If !Use_TMA_Q, we use the MMA WGs to load Q with cp.async
        if constexpr (Use_TMA_Q) {
            // Wait for the MMA warpgroups to signal that smem_q is ready
            if (SingleProducerWarp || warp_idx_in_warpgroup == 0) {
                cutlass::arch::NamedBarrier::sync(NumMmaThreads + cutlass::NumThreadsPerWarp, static_cast<int>(FwdNamedBarriers::QueryEmpty) /*id*/);
            }

            if ((SingleProducerWarp || warp_idx_in_warpgroup == 0) && cute::elect_one_sync()) {
                shared_storage.pipelines.barrier_Q.arrive_and_expect_tx(TmaTransactionBytesQ);
                copy(params.tma_load_Q.with(reinterpret_cast<typename cutlass::arch::ClusterTransactionBarrier::ValueType&>(shared_storage.pipelines.barrier_Q), 0 /*mcast_mask*/, !Split ? TMA::CacheHintSm90::EVICT_FIRST : TMA::CacheHintSm90::EVICT_LAST),
                    tQgQ, tQsQ);
            }
        }

        // Wait for warp 1 to signal that smem_v are ready and V can be copied from gmem
        // Need ClusterBarrier, not just NamedBarrier. Otherwise we might have CTA 0 finishing the
        // TMA store on O first, call TMA multicast load on V, before CTA 1 can finishing TMA store on O.
        // if (thread_idx == 0) { printf("Producer: main load, before barrier_O\n");}
        shared_storage.pipelines.barrier_O.wait((work_idx + 1) % 2);
        // if (thread_idx == 0) { printf("Producer: main load, after barrier_O\n");}

        int n_block_prev = n_block;
        --n_block;
        #pragma unroll (!Transpose_V && Use_TMA_KV ? 2 : 1)
        for (; n_block >= n_block_min; --n_block) {
            PipelineState smem_pipe_write_v = smem_pipe_write; // copy the state, write_v is always 1 step behind
            ++smem_pipe_write;
            if (should_load_KV) {
                if constexpr (PagedKV) {
                    paged_kv_manager.template load_page_table<false /*Seqlenk_mask*/>(n_block);
                }
                if constexpr (Transpose_V) { load_V(n_block, smem_pipe_write, cute::bool_constant<false>{} /*Seqlenk_mask*/); }
                load_K(n_block, smem_pipe_write, cute::bool_constant<false>{} /*Seqlenk_mask*/);
                if constexpr (!Transpose_V) { load_V(n_block_prev, smem_pipe_write_v, cute::bool_constant<true>{} /*Seqlenk_mask*/); }
            }
            n_block_prev = n_block;
            if constexpr (Transpose_V) { copy_Vt_to_V(smem_pipe_write_v); }
        }
        if (Is_local) {
            static constexpr int kBlockN = get<1>(TileShape_MNK{});
            int n_block_sink_max = cute::ceil_div(params.sink_token_length, kBlockN);
            #pragma unroll 1
            for (n_block = std::min(n_block, n_block_sink_max - 1); n_block >= 0; --n_block) {
                PipelineState smem_pipe_write_v = smem_pipe_write; // copy the state, write_v is always 1 step behind
                ++smem_pipe_write;
                if (should_load_KV) {
                    if constexpr (PagedKV) {
                        paged_kv_manager.template load_page_table<false /*Seqlenk_mask*/>(n_block);
                    }
                    if constexpr (Transpose_V) { load_V(n_block, smem_pipe_write, cute::bool_constant<false>{} /*Seqlenk_mask*/); }
                    load_K(n_block, smem_pipe_write, cute::bool_constant<false>{} /*Seqlenk_mask*/);
                    if constexpr (!Transpose_V) { load_V(n_block_prev, smem_pipe_write_v, cute::bool_constant<true>{} /*Seqlenk_mask*/); }
                }
                n_block_prev = n_block;
                if constexpr (Transpose_V) { copy_Vt_to_V(smem_pipe_write_v); }
            }
        }
        scheduler_prefetch();
        if (should_load_KV) {
            if constexpr (!Transpose_V) {
                // if (thread_idx == 0) { printf("Producer: main load, before load_V, index = %d\n", smem_pipe_write.index());}
                load_V(n_block_prev, smem_pipe_write, cute::bool_constant<true>{} /*Seqlenk_mask*/);
                // if (thread_idx == 0) { printf("Producer: main load, after load V, index = %d\n", smem_pipe_write.index());}
            }
        }
        if constexpr (Transpose_V) { copy_Vt_to_V(smem_pipe_write); }
        ++smem_pipe_write;
        // At the end, all threads have the correct smem_pipe_write.
    }

    CUTLASS_DEVICE void
    load_tail(MainloopPipelineK pipeline_k, MainloopPipelineV pipeline_v, MainloopPipelineVt pipeline_vt,
              PipelineState& smem_pipe_write) {
        int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
        // Issue the epilogue waits
        // TODO: check if this should be called by 1 thread or more
        if (warp_idx_in_warpgroup == 0 && cute::elect_one_sync()) {
            /* This helps avoid early exit of blocks in Cluster
            *  Waits for all stages to either be released (all Consumer UNLOCKs), or if the stage was never used
            *  then would just be acquired since the phase was still inverted from make_producer_start_state
            */
            pipeline_k.producer_tail(smem_pipe_write);
            pipeline_v.producer_tail(smem_pipe_write);
            if constexpr (Transpose_V) { pipeline_vt.producer_tail(smem_pipe_write); }
        }
    }

    CUTLASS_DEVICE void
    warp_scheduler_barrier_sync() {
        if constexpr (UseSchedulerBarrier) {
            cutlass::arch::NamedBarrier::sync(2 * cutlass::NumThreadsPerWarpGroup, static_cast<int>(FwdNamedBarriers::WarpSchedulerWG1) - 1 + cutlass::canonical_warp_group_idx() /*id*/);
        }
    }

    CUTLASS_DEVICE void
    warp_scheduler_barrier_arrive() {
        if constexpr (UseSchedulerBarrier) {
            static_assert(NumMmaWarpGroups == 2 || NumMmaWarpGroups == 3);
            int const cur_WG = cutlass::canonical_warp_group_idx();
            int const next_WG = NumMmaWarpGroups == 2
                ? 3 - cur_WG
                : (cur_WG <= NumMmaWarpGroups - 1 ? cur_WG + 1 : cur_WG + 1 - NumMmaWarpGroups);
            cutlass::arch::NamedBarrier::arrive(2 * cutlass::NumThreadsPerWarpGroup, static_cast<int>(FwdNamedBarriers::WarpSchedulerWG1) - 1 + next_WG /*id*/);
        }
    }

    CUTLASS_DEVICE void
    mma_init() {
        // Tell producer (warp 0) that smem_q is ready
        if constexpr (Use_TMA_Q) {
            cutlass::arch::NamedBarrier::arrive(NumMmaThreads + cutlass::NumThreadsPerWarp, static_cast<int>(FwdNamedBarriers::QueryEmpty) /*id*/);
        }
        if constexpr (UseSchedulerBarrier) {
            // We have NamedBarrier for up to 3 WGs
            static_assert(NumMmaWarpGroups == 2 || NumMmaWarpGroups == 3);
            // WG1 needs the very first signal to start
            if (cutlass::canonical_warp_group_idx() == 1) {
                cutlass::arch::NamedBarrier::arrive(2 * cutlass::NumThreadsPerWarpGroup, static_cast<int>(FwdNamedBarriers::WarpSchedulerWG1) /*id*/);
            }
        }
    }

    template <typename SharedStorage, typename FrgTensorO, typename Softmax>
    CUTLASS_DEVICE void
    mma(Params const& params,
        MainloopPipelineK pipeline_k,
        MainloopPipelineV pipeline_v,
        PipelineState& smem_pipe_read,
        FrgTensorO& tOrO,
        Softmax& softmax,
        int thread_idx,
        int work_idx,
        cute::tuple<int32_t, int32_t, int32_t, int32_t> block_coord,
        SharedStorage& shared_storage
        ) {
        static_assert(is_rmem<FrgTensorO>::value, "O tensor must be rmem resident.");

        static constexpr int kBlockM = get<0>(TileShape_MNK{});
        static constexpr int kBlockN = get<1>(TileShape_MNK{});

        Tensor sQ = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()), SmemLayoutQ{});
        Tensor sK = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()), SmemLayoutK{});
        Tensor sV = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutVtMma{});

        static_assert(stride<0>(typename TiledMma0::ALayout{}) == 0 and
                      stride<0>(typename TiledMma0::BLayout{}) == 0 and
                      size<0>(typename TiledMma0::ALayout{}) == cutlass::NumThreadsPerWarpGroup and
                      size<0>(typename TiledMma0::BLayout{}) == cutlass::NumThreadsPerWarpGroup,
              "Stride of the first mode must be 0 and the size of the mode must be NumThreadsPerWarpGroup");
        constexpr int MmaWarpGroups = size(TiledMma0{}) / cutlass::NumThreadsPerWarpGroup;
        Layout warp_group_thread_layout = make_layout(make_shape(Int<MmaWarpGroups>{}),
                                                      make_stride(Int<cutlass::NumThreadsPerWarpGroup>{}));

        int warp_group_idx = __shfl_sync(0xFFFFFFFF, thread_idx / cutlass::NumThreadsPerWarpGroup, 0);
        TiledMma0 tiled_mma0;
        TiledMma1 tiled_mma1;
        auto wg_mma0 = tiled_mma0.get_slice(warp_group_thread_layout(warp_group_idx));
        auto wg_mma1 = tiled_mma1.get_slice(warp_group_thread_layout(warp_group_idx));

        // Allocate "fragments/descriptors"
        Tensor tSrQ = wg_mma0.partition_fragment_A(sQ);
        Tensor tSrK = wg_mma0.partition_fragment_B(sK);
        Tensor tOrV = wg_mma1.partition_fragment_B(sV);

        auto consumer_wait = [](auto& pipeline, auto& smem_pipe_read) {
            auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
            pipeline.consumer_wait(smem_pipe_read, barrier_token);
        };

        // clear(tOrO);
        tiled_mma1.accumulate_ = GMMA::ScaleOut::Zero;

        // can't use auto [m_block, ...] = block_coord since structured binding cannot be captured in lambda
        int m_block = get<0>(block_coord);
        int bidb = get<2>(block_coord);
        int split_idx = get<3>(block_coord);
        int const seqlen_q = get_seqlen_q(params, bidb);
        int const seqlen_k = get_seqlen_k(params, bidb);
        auto [n_block_min, n_block_max] = get_n_block_min_max(params, m_block, bidb, split_idx, params.num_splits);
        int n_block = n_block_max - 1;

        flash::Mask<kBlockM, kBlockN, PackGQA, TiledMma0> mask(
            thread_idx, seqlen_q, seqlen_k, params.window_size_left, params.window_size_right, params.sink_token_length,
            params.qhead_per_khead_divmod
        );

        float softcap_val = params.softcap_val;
        if constexpr (Has_softcap && Is_FP8) {
            float const q_descale = params.ptr_q_descale == nullptr ? 1.0f : *params.ptr_q_descale;
            float const k_descale = params.ptr_k_descale == nullptr ? 1.0f : *params.ptr_k_descale;
            softcap_val *= q_descale * k_descale;
        }
        // Softcapping needs to happen before masking since if we apply after masking, softcapping can turn
        // -inf to e.g. -50.0, which can affect the attention softmax.
        auto scoremod_premask_fn = [&](auto& tSrS) {
            if constexpr (Has_softcap) { flash::apply_softcap(tSrS, softcap_val); }
        };

        auto &barrier_Q = shared_storage.pipelines.barrier_Q;
        if constexpr (!Use_TMA_Q) {  // Use Mma threads to load Q with cp.async
            // If persistent, we don't need to wait for the previous work_idx to finish and signal QueryEmpty
            // since we assume that all MMA threads sync in the epilogue before writing to smem_o.
            // So any thread gets there, all threads must have finished the previous MMA and at least started
            // writing to smem_o.
            int const bidh = get<1>(block_coord);
            bool const is_varlen_q = Varlen && params.cu_seqlens_q != nullptr;
            int const offset_q = !is_varlen_q ? 0 : params.cu_seqlens_q[bidb];
            Tensor mQ = make_tensor(make_gmem_ptr(params.ptr_Q + offset_q * get<0>(params.stride_Q)), params.shape_Q_packed, params.stride_Q_packed)(_, _, bidh, !is_varlen_q ? bidb : 0);
            Tensor sQ_pi = cute::as_position_independent_swizzle_tensor(sQ);
            using PackGQAt = flash::PackGQAManager<get<0>(TileShape_MNK{}), get<2>(TileShape_MNK{}), NumMmaThreads, Element>;
            PackGQAt::load_Q(mQ, sQ_pi, params.qhead_per_khead_divmod, thread_idx, seqlen_q, m_block);
            cutlass::arch::cpasync_barrier_arrive(reinterpret_cast<uint64_t*>(&barrier_Q));
            barrier_Q.arrive();
        }

        auto [rotary, tRrCos, tRrSin] = [&] {
            if constexpr (AppendKV) {
                int const leftpad_k = Varlen && params.leftpad_k ? params.leftpad_k[bidb] : 0;
                int const seqlen_k_og = get_seqlen_k<true /*Seqlen_before_append_KV*/>(params, bidb);
                int const offset_rotary = seqlen_k_og + leftpad_k;
                using Rotary_t = Rotary<kBlockM, kHeadDim, NumMmaThreads, Element, !(Is_causal || Is_local) /*FixedPosition*/>;
                Rotary_t rotary(params.ptr_rotary_cos, params.shape_rotary, params.stride_rotary_cos,
                                params.ptr_rotary_sin, params.stride_rotary_sin,
                                thread_idx, seqlen_q, offset_rotary);
                auto [tRrCos, tRrSin] = cute::conditional_return<!PackGQA>(rotary.load_cos_sin(m_block), rotary.load_cos_sin_packgqa(m_block, params.qhead_per_khead_divmod));
                return cute::make_tuple(rotary, tRrCos, tRrSin);
            } else {
                return cute::make_tuple(nullptr, nullptr, nullptr);
            }
        }();

        typename cutlass::ConsumerToken barrier_token = static_cast<cutlass::BarrierStatus>(barrier_Q.try_wait(work_idx % 2));
        if (barrier_token == cutlass::BarrierStatus::WaitAgain) { barrier_Q.wait(work_idx % 2); }

        if constexpr (AppendKV) {
            if (get<1>(params.shape_rotary) > 0) {  // Apply rotary to Q
                Tensor sQ_pi = cute::as_position_independent_swizzle_tensor(sQ);
                rotary.apply_Q(sQ_pi, tRrCos, tRrSin, m_block,
                               !PackGQA ? 1 : params.qhead_per_khead_divmod.divisor /*qhead_per_khead*/);
                // SMEM fence to make sure the rotated Q is visible to GMMA
                cutlass::arch::fence_view_async_shared();
                cutlass::arch::NamedBarrier::sync(NumMmaThreads, static_cast<int>(FwdNamedBarriers::QueryRotated) /*id*/);
            }
        }

        // TODO: check the case where n_block_max <= n_block_min but there are sink tokens
        if constexpr (true) {
            Tensor tSrS = partition_fragment_C(tiled_mma0, select<0, 1>(TileShape_MNK{}));
            consumer_wait(pipeline_k, smem_pipe_read);
            // if (thread_idx == 0) { print_tensor(sK(_, _, _0{})); }
            warp_scheduler_barrier_sync();
            flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma0, tSrQ, tSrK(_, _, _, smem_pipe_read.index()), tSrS);
            warp_scheduler_barrier_arrive();
            warpgroup_wait<0>();
            pipeline_k.consumer_release(smem_pipe_read);
            scoremod_premask_fn(tSrS);
            mask.template apply<true /*Seqlenk_mask*/, Is_causal, Is_local>(tSrS, m_block, n_block);

            Tensor scores_scale = softmax.template max_get_scale</*Is_first=*/true, /*Check_inf=*/true>(tSrS);
            softmax.template online_softmax</*Is_first=*/true, /*Check_inf=*/true>(tSrS);
            if constexpr (Is_FP8 && !V_colmajor) { flash::permute_Cregs_fp8(tSrS); }
            Tensor tOrP = make_tensor(convert_type<Element>(tSrS).data(), flash::convert_layout_acc_Aregs<TiledMma1>(tSrS.layout()));
            if constexpr (Is_FP8 && V_colmajor) { flash::permute_Aregs_fp8(tOrP); }

            // Each step does gemm0 for iter n_block - 1, gemm1 for iter n_block, and softmax for iter n_block - 1.
            auto fwd_step = [&](int const n_block, auto mask_fn, auto is_first_iter_type, auto check_inf_type) {
                static constexpr bool Is_first_iter = decltype(is_first_iter_type)::value;
                static constexpr bool Check_inf = decltype(check_inf_type)::value;
                PipelineState smem_pipe_read_v(smem_pipe_read.index(), smem_pipe_read.phase(), smem_pipe_read.count());
                ++smem_pipe_read;
                Tensor tSrS = partition_fragment_C(tiled_mma0, select<0, 1>(TileShape_MNK{}));
                consumer_wait(pipeline_k, smem_pipe_read);
                warp_scheduler_barrier_sync();
                flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma0, tSrQ, tSrK(_, _, _, smem_pipe_read.index()), tSrS);
                if constexpr (RescaleOBeforeGemm && !Is_first_iter) { softmax.rescale_o(tOrO, scores_scale); }
                consumer_wait(pipeline_v, smem_pipe_read_v);
                flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(tiled_mma1, tOrP, tOrV(_, _, _, smem_pipe_read_v.index()), tOrO);
                warp_scheduler_barrier_arrive();
                warpgroup_wait<1>();
                // if (thread_idx == 0) { print_tensor(sK(_, _, smem_pipe_read.index())); }
                // Tensor scores = make_tensor(tSrS.data(), flash::convert_layout_acc_rowcol(tSrS.layout()));
                // if (thread_idx == 0 && m_block == 0) { print_tensor(scores); }
                pipeline_k.consumer_release(smem_pipe_read);  // release K
                scoremod_premask_fn(tSrS);
                mask_fn(tSrS, n_block - 1);
                cute::copy(softmax.template max_get_scale</*Is_first=*/false, Check_inf>(tSrS), scores_scale);
                softmax.template online_softmax</*Is_first=*/false, Check_inf>(tSrS);
                warpgroup_wait<0>();
                pipeline_v.consumer_release(smem_pipe_read_v);  // release V
                if constexpr (Is_FP8 && !V_colmajor) { flash::permute_Cregs_fp8(tSrS); }
                cute::copy(make_tensor(convert_type<Element>(tSrS).data(), tOrP.layout()), tOrP);
                if constexpr (Is_FP8 && V_colmajor) { flash::permute_Aregs_fp8(tOrP); }
                if constexpr (!RescaleOBeforeGemm) { softmax.rescale_o(tOrO, scores_scale); }
            };

            if constexpr (Is_causal || Is_local) { // Separate iterations with causal or local masking
                auto mask_fn = [&](auto& tSrS, int n_block) { mask.template apply<false /*Seqlenk_mask*/, Is_causal, Is_local>(tSrS, m_block, n_block); };
                constexpr int n_masking_steps = cute::ceil_div(kBlockM, kBlockN) + 1;
                #pragma unroll
                for (int masking_step = 0; masking_step < n_masking_steps - 1 && n_block > n_block_min; ++masking_step, --n_block) {
                    if (masking_step == 0) {
                        fwd_step(n_block, mask_fn, cute::bool_constant<true>{} /*is_first_iter*/, cute::bool_constant<true>{} /*check_inf*/);
                    } else {
                        fwd_step(n_block, mask_fn, cute::bool_constant<false>{} /*is_first_iter*/, cute::bool_constant<true>{} /*check_inf*/);
                    }
                }
            }

            static constexpr int n_local_left_steps = !Is_local ? 0 : cute::ceil_div(kBlockM, kBlockN) + 1;
            auto no_mask_fn = [](auto& tSrS, int n_block) { };
            #pragma unroll 1
            for (; n_block > n_block_min + n_local_left_steps; --n_block) {
                fwd_step(n_block, no_mask_fn, cute::bool_constant<false>{} /*is_first_iter*/, cute::bool_constant<false>{} /*check_inf*/);
            }
            // Separate masking iterations on the left for local attention
            if constexpr (Is_local) {
                auto local_mask_fn = [&](auto& tSrS, int n_block) { mask.template apply<false /*Seqlenk_mask*/, false /*Causal_mask*/, Is_local>(tSrS, m_block, n_block); };
                #pragma unroll 1
                for (; n_block > n_block_min; --n_block) {
                    fwd_step(n_block, local_mask_fn, cute::bool_constant<false>{} /*is_first_iter*/, cute::bool_constant<Is_local>{} /*check_inf*/);
                }
                int n_block_sink_max = cute::ceil_div(params.sink_token_length, kBlockN);
                #pragma unroll 1
                // Masking is done on (n_block - 1) so n_block is 1 more than the index of the block that needs masking
                for (n_block = std::min(n_block, n_block_sink_max); n_block > 0; --n_block) {
                    fwd_step(n_block, local_mask_fn, cute::bool_constant<false>{} /*is_first_iter*/, cute::bool_constant<Is_local>{} /*check_inf*/);
                }
            }
            // Tell warp 0 that smem_q is ready
            if constexpr (Use_TMA_Q) {  // If !Use_TMA_Q, we don't use the producer WG to load Q
                cutlass::arch::NamedBarrier::arrive(NumMmaThreads + cutlass::NumThreadsPerWarp, static_cast<int>(FwdNamedBarriers::QueryEmpty) /*id*/);
            }
            if constexpr (RescaleOBeforeGemm) { softmax.rescale_o(tOrO, scores_scale); }
            consumer_wait(pipeline_v, smem_pipe_read);
            flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(tiled_mma1, tOrP, tOrV(_, _, _, smem_pipe_read.index()), tOrO);
            cute::copy(softmax.finalize(!Is_FP8 || params.ptr_v_descale == nullptr ? 1.f : *params.ptr_v_descale), scores_scale);
            warpgroup_wait<0>();
            pipeline_v.consumer_release(smem_pipe_read);  // release V, otherwise producers will hang
            softmax.rescale_o(tOrO, scores_scale);
            if constexpr (Is_FP8 && !V_colmajor) { flash::permute_output_fp8(tOrO); }
            ++smem_pipe_read;

        } else {
            // WIP: a version without intra-warpgroup overlap, for benchmarking / didactic purposes

            auto fwd_step = [&](int const n_block, auto mask_fn, auto is_first_iter_type, auto check_inf_type) {
                static constexpr bool Is_first_iter = decltype(is_first_iter_type)::value;
                static constexpr bool Check_inf = decltype(check_inf_type)::value;
                Tensor tSrS = partition_fragment_C(tiled_mma0, select<0, 1>(TileShape_MNK{}));
                consumer_wait(pipeline_k, smem_pipe_read);
                flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma0, tSrQ, tSrK(_, _, _, smem_pipe_read.index()), tSrS);
                warpgroup_wait<0>();
                pipeline_k.consumer_release(smem_pipe_read);  // release K
                scoremod_premask_fn(tSrS);
                mask_fn(tSrS, n_block - 1);
                Tensor scores_scale = softmax.template max_get_scale</*Is_first=*/false>(tSrS);
                warp_scheduler_barrier_sync();
                softmax.template online_softmax</*Is_first=*/false>(tSrS);
                if constexpr (Is_FP8 && !V_colmajor) { flash::permute_Cregs_fp8(tSrS); }
                Tensor tOrP = make_tensor(convert_type<Element>(tSrS).data(), flash::convert_layout_acc_Aregs<TiledMma1>(tSrS.layout()));
                if constexpr (Is_FP8 && V_colmajor) { flash::permute_Aregs_fp8(tOrP); }
                warp_scheduler_barrier_arrive();
                softmax.rescale_o(tOrO, scores_scale);
                consumer_wait(pipeline_v, smem_pipe_read);
                flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(tiled_mma1, tOrP, tOrV(_, _, _, smem_pipe_read.index()), tOrO);
                warpgroup_wait<0>();
                pipeline_v.consumer_release(smem_pipe_read);  // release V
                ++smem_pipe_read;
            };
            // fwd_step(n_block, mask.template apply<true /*Seqlenk_mask*/, Is_causal, Is_local>, cute::bool_constant<true>{} /*is_first_iter*/, cute::bool_constant<true>{} /*check_inf*/);
            // --n_block;
            // #pragma unroll 1
            // for (; n_block >= n_block_min; --n_block) {
            //     fwd_step(n_block, mask.template apply<false /*Seqlenk_mask*/, Is_causal, Is_local>, cute::bool_constant<false>{} /*is_first_iter*/, cute::bool_constant<true>{} /*check_inf*/);
            // }
            // Tell warp 0 that smem_q is ready
            if constexpr (Use_TMA_Q) {  // If !Use_TMA_Q, we don't use the producer WG to load Q
                cutlass::arch::NamedBarrier::arrive(NumMmaThreads + cutlass::NumThreadsPerWarp, static_cast<int>(FwdNamedBarriers::QueryEmpty) /*id*/);
            }
            Tensor scores_scale = softmax.finalize();
            softmax.rescale_o(tOrO, scores_scale);
        }
    }

    CUTLASS_DEVICE
    cute::tuple<int, int> get_n_block_k_new_min_max(Params const& params, int m_block, int bidb, int split_idx=0, int num_splits=1) {
        static constexpr int kBlockN = get<1>(TileShape_MNK{});
        int const seqlen_k_new = get_seqlen_k_new(params, bidb);
        int const seqlen_k_og = get_seqlen_k<true /*Seqlen_before_append_KV*/>(params, bidb);
        auto [n_block_min, n_block_max] = get_n_block_min_max(params, m_block, bidb, split_idx, num_splits);
        int const idx_k_new_min = std::max(n_block_min * kBlockN - seqlen_k_og, 0);
        int const idx_k_new_max = std::min(n_block_max * kBlockN - seqlen_k_og, seqlen_k_new);
        int const n_block_new_min = idx_k_new_min / kBlockN;
        int const n_block_new_max = idx_k_new_max > idx_k_new_min ? cute::ceil_div(idx_k_new_max, kBlockN) : n_block_new_min;
        // if (threadIdx.x == 128 && m_block == 0) { printf("bidb = %d, seqlen_k_new = %d, seqlen_k_og = %d, n_block_min = %d, n_block_max = %d, idx_k_new_min = %d, idx_k_new_max = %d, n_block_new_min = %d, n_block_new_max = %d\n", bidb, seqlen_k_new, seqlen_k_og, n_block_min, n_block_max, idx_k_new_min, idx_k_new_max, n_block_new_min, n_block_new_max);}
        return {n_block_new_min, n_block_new_max};
    }

    template <typename SharedStorage>
    CUTLASS_DEVICE void
    load_kv_new(Params const& params,
         MainloopPipelineKVNew pipeline_k_new,
         MainloopPipelineKVNew pipeline_v_new,
         PipelineState& smem_pipe_write,
         SharedStorage &shared_storage,
         cute::tuple<int32_t, int32_t, int32_t, int32_t> block_coord,
         int work_idx
         ) {

        Tensor sK = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()), SmemLayoutK{});
        Tensor sVt = [&] {
            if constexpr (!Transpose_V) {
                return make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutVt{});
            } else {
                return make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_vt.data()), SmemLayoutVt{});
            }
        }();

        auto [m_block, bidh, bidb, split_idx] = block_coord;
        auto [n_block_new_min, n_block_new_max] = get_n_block_k_new_min_max(params, m_block, bidb, split_idx, params.num_splits);
        if (n_block_new_max <= n_block_new_min) { return; }

        int const thread_idx = threadIdx.x % NumProducerThreads;
        int const bidh_kv = !PackGQA ? params.qhead_per_khead_divmod.divide(bidh) : bidh;

        // Prepare the TMA loads
        uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();
        constexpr uint32_t cluster_shape_x = get<0>(ClusterShape());
        uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x, block_rank_in_cluster / cluster_shape_x};

        bool const is_varlen_k_new = Varlen && params.cu_seqlens_k_new;
        Tensor mKnew_TMA = params.tma_load_K_new.get_tma_tensor(params.shape_K_new)(_, _, bidh_kv, !is_varlen_k_new ? bidb : 0);
        Tensor mVnewt_TMA = params.tma_load_V_new.get_tma_tensor(select<1, 0, 2, 3>(params.shape_K_new))(_, _, bidh_kv, !is_varlen_k_new ? bidb : 0);

        int const offset_k_new = !Varlen ? 0 : (params.cu_seqlens_k_new ? params.cu_seqlens_k_new[bidb] : 0);
        Tensor gKnew_TMA = local_tile(domain_offset(make_coord(offset_k_new, _0{}), mKnew_TMA), select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (N, K, _)
        Tensor gVnewt_TMA = local_tile(domain_offset(make_coord(_0{}, offset_k_new), mVnewt_TMA), select<2, 1>(TileShape_MNK{}), make_coord(_0{}, _));  // (K, N, _)

        auto block_tma_K_new = params.tma_load_K_new.get_slice(cluster_local_block_id.x);
        Tensor tKgKnew_TMA = group_modes<0, 3>(block_tma_K_new.partition_S(gKnew_TMA));  // (TMA, k)
        Tensor tKsK_TMA = group_modes<0, 3>(block_tma_K_new.partition_D(sK));  // (TMA, PIPE)
        auto block_tma_V_new = params.tma_load_V_new.get_slice(cluster_local_block_id.x);
        Tensor tVgVnewt_TMA = group_modes<0, 3>(block_tma_V_new.partition_S(gVnewt_TMA));  // (TMA, k)
        Tensor tVsVt_TMA = group_modes<0, 3>(block_tma_V_new.partition_D(sVt));  // (TMA, PIPE)

        uint16_t mcast_mask_kv = 0;
        if constexpr (cute::is_same_v<GmemTiledCopyKV, SM90_TMA_LOAD_MULTICAST>) {
            auto block_layout = Layout<ClusterShape>{}; // (m,n) -> block_id
            for (int m = 0; m < size<0>(block_layout); ++m) {
                mcast_mask_kv |= (uint16_t(1) << block_layout(m, cluster_local_block_id.y, _0{}));
            }
        }

        auto load_K_new = [&] (int const n_block, auto const& smem_pipe_write) {
            pipeline_k_new.producer_acquire(smem_pipe_write);
            copy(params.tma_load_K_new.with(*pipeline_k_new.producer_get_barrier(smem_pipe_write), mcast_mask_kv, TMA::CacheHintSm90::EVICT_FIRST),
                tKgKnew_TMA(_, n_block), tKsK_TMA(_, smem_pipe_write.index()));
        };

        auto load_V_new = [&] (int const n_block, auto const& smem_pipe_write) {
            pipeline_v_new.producer_acquire(smem_pipe_write);
            copy(params.tma_load_V_new.with(*pipeline_v_new.producer_get_barrier(smem_pipe_write), mcast_mask_kv, TMA::CacheHintSm90::EVICT_FIRST),
                tVgVnewt_TMA(_, n_block), tVsVt_TMA(_, smem_pipe_write.index()));
        };

        int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
        // If this is true, we're guaranteed that only the first warp will execute this function
        static constexpr bool SingleProducerWarp = NumProducerThreads == cutlass::NumThreadsPerWarp;
        bool should_load_KV = (SingleProducerWarp || warp_idx_in_warpgroup == 0) && cute::elect_one_sync();

        int n_block = n_block_new_max - 1;
        if (should_load_KV) { load_K_new(n_block, smem_pipe_write); }
        // if (thread_idx == 0) { printf("Producer: Done loading K, n_block = %d, n_block_new_min = %d\n", n_block, n_block_new_min); }
        // Wait for warp 1 to signal that smem_v are ready and V can be copied from gmem
        if constexpr (!Transpose_V) { shared_storage.pipelines.barrier_O.wait((work_idx + 1) % 2); }
        if (should_load_KV) { load_V_new(n_block, smem_pipe_write); }
        // if (thread_idx == 0) { printf("Producer: Done loading V, n_block = %d, n_block_new_min = %d\n", n_block, n_block_new_min); }
        ++smem_pipe_write;
        --n_block;
        // if (thread_idx == 0) { printf("Producer: before for loop\n"); }
        #pragma unroll 1
        for (; n_block >= n_block_new_min; --n_block) {
            if (should_load_KV) {
                load_K_new(n_block, smem_pipe_write);
                // if (thread_idx == 0) { printf("Producer: Done loading K, n_block = %d, n_block_new_min = %d\n", n_block, n_block_new_min); }
                load_V_new(n_block, smem_pipe_write);
                // if (thread_idx == 0) { printf("Producer: Done loading V, n_block = %d, n_block_new_min = %d\n", n_block, n_block_new_min); }
            }
            ++smem_pipe_write;
        }
        // if (thread_idx == 0) { printf("Producer: after for loop\n"); }
        // At the end, all threads have the correct smem_pipe_write.
    }

    template <typename SharedStorage>
    CUTLASS_DEVICE void
    store_kv_new(Params const& params,
                 MainloopPipelineKVNew pipeline_k_new,
                 MainloopPipelineKVNew pipeline_v_new,
                 PipelineState& smem_pipe_read,
                 int const thread_idx,
                 SharedStorage &shared_storage,
                 cute::tuple<int32_t, int32_t, int32_t, int32_t> block_coord
    ) {

        // as_position_independent_swizzle_tensor makes address calculation easier
        Tensor sK = cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()), SmemLayoutK{}));
        // We want to use SmemLayoutVCpAsync to have shape (kBlockN, kHeadDim) instead of (kHeadDim, kBlockN)
        Tensor sV = [&] {
            if constexpr (!Transpose_V) {
                return cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutVCpAsync{}));
            } else {
                return cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_vt.data()), SmemLayoutVCpAsync{}));
            }
        }();

        auto [m_block, bidh, bidb, split_idx] = block_coord;
        auto [n_block_new_min, n_block_new_max] = get_n_block_k_new_min_max(params, m_block, bidb, split_idx, params.num_splits);
        if (n_block_new_max <= n_block_new_min) { return; }

        int const bidh_kv = !PackGQA ? params.qhead_per_khead_divmod.divide(bidh) : bidh;
        int const bidb_kv = params.kv_batch_idx == nullptr ? bidb : params.kv_batch_idx[bidb];

        bool const is_varlen_k = Varlen && params.cu_seqlens_k;
        Tensor mK = make_tensor(make_gmem_ptr(params.ptr_K), params.shape_K, params.stride_K)(_, _, bidh_kv, !is_varlen_k ? bidb_kv : 0);
        Tensor mV = make_tensor(make_gmem_ptr(params.ptr_V), params.shape_K, params.stride_V)(_, _, bidh_kv, !is_varlen_k ? bidb_kv : 0);

        int const leftpad_k = Varlen && params.leftpad_k ? params.leftpad_k[bidb] : 0;
        int const seqlen_k_og = get_seqlen_k<true /*Seqlen_before_append_KV*/>(params, bidb);

        int const offset_k = (!Varlen ? 0 : (params.cu_seqlens_k ? params.cu_seqlens_k[bidb] : 0) + leftpad_k) + seqlen_k_og;
        Tensor gK = local_tile(domain_offset(make_coord(offset_k, _0{}), mK), select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (N, K, _)
        Tensor gV = local_tile(domain_offset(make_coord(offset_k, _0{}), mV), select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (N, K, _)

        static constexpr int kBlockN = get<1>(TileShape_MNK{});
        static constexpr int kHeadDim = get<2>(TileShape_MNK{});
        int const offset_rotary = seqlen_k_og + leftpad_k;
        int const seqlen_k_new = get_seqlen_k_new(params, bidb);
        int const rotary_dim = get<1>(params.shape_rotary) * 2;
        using Rotary_t = Rotary<kBlockN, kHeadDim, NumMmaThreads, Element>;
        Rotary_t rotary(params.ptr_rotary_cos, params.shape_rotary, params.stride_rotary_cos,
                        params.ptr_rotary_sin, params.stride_rotary_sin,
                        thread_idx, seqlen_k_new, offset_rotary);

        PagedKVManager<get<1>(TileShape_MNK{}), get<2>(TileShape_MNK{}), NumMmaThreads, Element, true /*KV_Same_Iter*/> paged_kv_manager(
            params.ptr_pagetable, params.shape_pagetable, params.stride_pagetable,
            params.ptr_K, params.shape_K, params.stride_K,
            params.ptr_V, params.stride_V,
            params.page_size_divmod, bidb_kv, bidh_kv, thread_idx, seqlen_k_new, offset_k
            // passing offset_k instead of leftpad_k will move the PageTable pointer to the right position
        );

        if constexpr (UseSchedulerBarrier) {
            // WG1 already got the very first signal from mma_init(), but we'll be using the same NamedBarrier.
            // So we'll need to "cancel it out" here and then re-signal it at the end.
            if (cutlass::canonical_warp_group_idx() == 1) {
                cutlass::arch::NamedBarrier::sync(2 * cutlass::NumThreadsPerWarpGroup, static_cast<int>(FwdNamedBarriers::WarpSchedulerWG1) /*id*/);
            }
        }

        GmemTiledCopyAppendKV gmem_tiled_copy_kv;
        auto gmem_thr_copy_kv = gmem_tiled_copy_kv.get_thread_slice(thread_idx);
        Tensor tKsK = gmem_thr_copy_kv.partition_S(sK);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
        Tensor tKgK = gmem_thr_copy_kv.partition_D(gK);
        Tensor tVsV = gmem_thr_copy_kv.partition_S(sV);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
        Tensor tVgV = gmem_thr_copy_kv.partition_D(gV);
        Tensor cK = cute::make_identity_tensor(select<1, 2>(TileShape_MNK{}));  // (BLK_N,BLK_K) -> (blk_n,blk_k)
        Tensor tKcK = gmem_thr_copy_kv.partition_D(cK);
        Tensor tKpK = make_tensor<bool>(make_shape(size<2>(tKsK)));
        #pragma unroll
        for (int k = 0; k < size(tKpK); ++k) { tKpK(k) = get<1>(tKcK(_0{}, _0{}, k)) < get<1>(params.shape_K); }

        auto store_K = [&] (int const n_block, auto const& smem_pipe_read) {
            auto [tRrCos, tRrSin] = rotary.load_cos_sin(n_block);
            pipeline_k_new.consumer_wait(smem_pipe_read);
            Tensor tKrK = make_fragment_like(tKsK(_, _, _, _0{}));
            int const n_limit = std::min(seqlen_k_new - n_block * kBlockN, kBlockN);
            flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/true, /*Clear_OOB_MN=*/false>(
                gmem_tiled_copy_kv, tKsK(_, _, _, smem_pipe_read.index()), tKrK, tKcK, tKpK, n_limit
            );
            // Without this sync I'm getting race condition when seqlen_k is large
            cutlass::arch::fence_view_async_shared();
            // Very important: PipelineTmaAsync::consumer_release assumes that the warpgroup is synchronized
            // before calling.
            cutlass::arch::NamedBarrier::sync(cutlass::NumThreadsPerWarpGroup, static_cast<int>(FwdNamedBarriers::WarpSchedulerWG1) - 1 + cutlass::canonical_warp_group_idx() /*id*/);
            pipeline_k_new.consumer_release(smem_pipe_read);
            // if (thread_idx == 0) { print_tensor(sK(_, _, smem_pipe_read.index())); }
            // if (thread_idx == 0) { print(tKsK); printf("\n"); print(tKrK); printf("\n"); print(tKgK); printf("\n");}
            // if (thread_idx == 0) { print_tensor(tKpK); printf("\n"); printf("seqlen_limit = %d\n", seqlen_k_new - n_block * kBlockN);}
            if (rotary_dim > 0) { flash::apply_rotary_interleaved(tKrK, tRrCos, tRrSin, tKcK, n_limit, rotary_dim); }

            if constexpr (!PagedKV) {
                Tensor tKgK_cur = tKgK(_, _, _, n_block);
                // Clear_OOB_K must be false since we don't want to write zeros to gmem
                flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
                    gmem_tiled_copy_kv, tKrK, tKgK_cur, tKcK, tKpK, std::min(seqlen_k_new - n_block * kBlockN, kBlockN)
                );
            } else {
                paged_kv_manager.template store_K<true /*Seqlenk_mask*/>(n_block, tKrK);
            }
        };

        auto store_V = [&] (int const n_block, auto const& smem_pipe_read) {
            pipeline_v_new.consumer_wait(smem_pipe_read);
            Tensor tVrV = make_fragment_like(tVsV(_, _, _, _0{}));
            int const n_limit = std::min(seqlen_k_new - n_block * kBlockN, kBlockN);
            flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/true, /*Clear_OOB_MN=*/false>(
                gmem_tiled_copy_kv, tVsV(_, _, _, smem_pipe_read.index()), tVrV, tKcK, tKpK, n_limit
            );
            cutlass::arch::fence_view_async_shared();
            cutlass::arch::NamedBarrier::sync(cutlass::NumThreadsPerWarpGroup, static_cast<int>(FwdNamedBarriers::WarpSchedulerWG1) - 1 + cutlass::canonical_warp_group_idx() /*id*/);
            pipeline_v_new.consumer_release(smem_pipe_read);
            // if (thread_idx == 0) { print_tensor(sV(_, _, smem_pipe_read.index())); }
            if constexpr (!PagedKV) {
                Tensor tVgV_cur = tVgV(_, _, _, n_block);
                // Clear_OOB_K must be false since we don't want to write zeros to gmem
                flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
                    gmem_tiled_copy_kv, tVrV, tVgV_cur, tKcK, tKpK, n_limit);
            } else {
                paged_kv_manager.template store_V<true /*Seqlenk_mask*/>(n_block, tVrV);
            }
        };

        #pragma unroll 1
        for (int n_block = n_block_new_max - 1; n_block >= n_block_new_min; --n_block) {
            if constexpr (PagedKV) { paged_kv_manager.template load_page_table<true /*Seqlenk_mask*/>(n_block); }
            store_K(n_block, smem_pipe_read);
            // if (thread_idx == 0) { printf("Done storing K, n_block = %d, n_block_new_min = %d\n", n_block, n_block_new_min); }
            store_V(n_block, smem_pipe_read);
            // if (thread_idx == 0) { printf("Done storing V, n_block = %d, n_block_new_min = %d\n", n_block, n_block_new_min); }
            ++smem_pipe_read;
        }
        // if (thread_idx == 0) { printf("After for loop\n"); }

        // Re-signaling the NamedBarrier that we "canceled out"
        if constexpr (UseSchedulerBarrier) {
            if (cutlass::canonical_warp_group_idx() == 1) {
                cutlass::arch::NamedBarrier::arrive(2 * cutlass::NumThreadsPerWarpGroup, static_cast<int>(FwdNamedBarriers::WarpSchedulerWG1) /*id*/);
            }
        }

    }

};

} // namespace flash
