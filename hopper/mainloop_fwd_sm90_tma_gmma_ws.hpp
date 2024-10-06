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
#include "utils.h"

namespace flash {

using namespace cute;

template <int Stages, class ClusterShape_, class TileShape_MNK_, class Element_, class ElementAccum_, class ArchTag_,
        bool Is_causal_, bool Is_local_, bool Has_softcap_, bool Varlen_, bool V_colmajor_>
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
    static constexpr bool V_colmajor = V_colmajor_;
    static constexpr bool Transpose_V = Is_FP8 && !V_colmajor;
    static constexpr int kHeadDim = get<2>(TileShape_MNK{});

    static_assert(ArchTag::kMinComputeCapability >= 90);
    static_assert(get<1>(ClusterShape{}) == 1 && get<2>(ClusterShape{}) == 1);

    static constexpr cute::GMMA::Major MmaMajorV = !Is_FP8 && !V_colmajor ? GMMA::Major::MN : GMMA::Major::K;
    static constexpr cute::GMMA::Major TmaMajorV = !V_colmajor ? GMMA::Major::MN : GMMA::Major::K;

    using AtomLayoutMNK = Layout<Shape<Int<get<0>(TileShape_MNK{}) / 64>, _1, _1>>;
    using TiledMma0 = decltype(cute::make_tiled_mma(
        cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShape_MNK>(),
        AtomLayoutMNK{}));
    using TiledMma1 = decltype(cute::make_tiled_mma(
        cute::GMMA::rs_op_selector<Element, Element, ElementAccum, decltype(select<0, 2, 1>(TileShape_MNK{})),
                                   GMMA::Major::K, MmaMajorV>(),
        AtomLayoutMNK{}));

    static constexpr int NumMmaThreads = size(TiledMma0{});
    static constexpr int NumProducerThreads = !Transpose_V ? cutlass::NumThreadsPerWarp : cutlass::NumThreadsPerWarpGroup;

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

    // Use LDSM.T and STSM to transpose V in the case of FP8 and V being row-major.
    // For FP16/BF16 we don't do any transposing.
    static_assert(!Transpose_V || (kHeadDim % 32 == 0 && CUTE_STATIC_V(get<1>(TileShape_MNK{})) % 32 == 0));
    static constexpr bool kHeadDim_multiple_64 = kHeadDim % 64 == 0;
    // Either kHeadDim is a multiple of 64 (in which case we use a block size of 64 x 32 for the transpose),
    // or we need kBlockN to be a multiple of 64 (in which case we use a block size of 32 x 64 for the transpose).
    static_assert(!Transpose_V || (kHeadDim_multiple_64 || CUTE_STATIC_V(get<1>(TileShape_MNK{})) % 64 == 0));
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

    using ShapeQKV = cute::Shape<int32_t, int32_t, int32_t, int32_t>;  // (seqlen, d, head, batch)
    using StrideQK = cute::Stride<int64_t, _1, int64_t, int64_t>;
    using StrideV = std::conditional_t<!V_colmajor, StrideQK, cute::Stride<_1, int64_t, int64_t, int64_t>>;

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

    using MainloopPipelineK = typename cutlass::PipelineTmaAsync<kStages>;
    using MainloopPipelineV = std::conditional_t<!Transpose_V, typename cutlass::PipelineTmaAsync<kStages>, typename cutlass::PipelineAsync<kStages>>;
    using MainloopPipelineVt = typename cutlass::PipelineTmaAsync<kStages>;
    using PipelineState = cutlass::PipelineState<kStages>;

    struct TensorStorageNoTranspose : cute::aligned_struct<128> {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutVt>> smem_v;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> smem_q;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>> smem_k;
    };

    static constexpr size_t SmemAlignmentVt = cutlass::detail::alignment_for_swizzle(SmemLayoutVt{});
    static constexpr size_t SmemAlignmentV = cutlass::detail::alignment_for_swizzle(SmemLayoutVtMma{});
    static_assert(SmemAlignmentVt >= 128 and SmemAlignmentV >= 128, "Require at least 128B alignment");

    struct TensorStorageTransposeV : cute::aligned_struct<cute::max(SmemAlignmentVt, SmemAlignmentV)> {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutVtMma>, SmemAlignmentV> smem_v;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutVt>, SmemAlignmentVt> smem_vt;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> smem_q;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>> smem_k;
    };

    using TensorStorage = std::conditional_t<!Transpose_V, TensorStorageNoTranspose, TensorStorageTransposeV>;

    // These are tuned for speed. They don't affect correctness.
    static constexpr bool UseSchedulerBarrier = !Is_FP8 ? kHeadDim <= 128 : kHeadDim >= 128;
    static constexpr bool RescaleOBeforeGemm = kHeadDim > 128 && (!Is_FP8 || V_colmajor);

    // Host side kernel arguments
    struct Arguments {
        Element const* ptr_Q;
        ShapeQKV const shape_Q;
        StrideQK const stride_Q;
        Element const* ptr_K;
        ShapeQKV const shape_K;
        StrideQK const stride_K;
        Element const* ptr_V;
        StrideV const stride_V;
        float const softmax_scale;
        float const* ptr_q_scale = nullptr, *ptr_k_scale = nullptr, *ptr_v_scale = nullptr;
        int const window_size_left = -1, window_size_right = -1;
        float const softcap_val;
        int const* cu_seqlens_q = nullptr;
        int const* cu_seqlens_k = nullptr;
        int const* seqused_q = nullptr;
        int const* seqused_k = nullptr;
    };

    // Device side kernel params
    struct Params {
        ShapeQKV const shape_Q;
        ShapeQKV const shape_K;
        cutlass::FastDivmod qhead_per_khead_divmod;
        TMA_Q tma_load_Q;
        TMA_K tma_load_K;
        TMA_V tma_load_V;
        float const softmax_scale_log2;
        float const* ptr_q_scale = nullptr, *ptr_k_scale = nullptr, *ptr_v_scale = nullptr;
        float const softcap_val;
        int const window_size_left, window_size_right;
        int const* cu_seqlens_q = nullptr;
        int const* cu_seqlens_k = nullptr;
        int const* seqused_q = nullptr;
        int const* seqused_k = nullptr;
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
        Tensor mVt = make_tensor(make_gmem_ptr(args.ptr_V), select<1, 0, 2, 3>(args.shape_K), select<1, 0, 2, 3>(args.stride_V));
        TMA_V tma_load_V = make_tma_copy(
            GmemTiledCopyKV{},
            mVt,
            take<0, 2>(SmemLayoutVt{}),
            select<2, 1>(TileShape_MNK{}),
            size<0>(ClusterShape{})); // mcast along M mode for this N load, if any
        if constexpr (Varlen) {
            assert(args.cu_seqlens_q != nullptr && args.cu_seqlens_k != nullptr);
        }
        // If there's tanh softcapping, we do tanh(scores * softmax_scale / softcap_val) * softcap_val.
        // Right after this, we multiply by log2(e) before applying exp2.
        // To reduce the number of instructions, we instead pre-multiply softmax_scale / softcap_val
        // (assigning it to params.softcap_val) and pre-multiply softcap_val * log2(e)
        // (assigning it to params.softmax_scale_log2).
        // TODO: this currently doesn't work with FP8 scaling
        return {args.shape_Q, args.shape_K,
                cutlass::FastDivmod(cute::ceil_div(get<2>(args.shape_Q), get<2>(args.shape_K))),
                tma_load_Q, tma_load_K, tma_load_V,
                !Has_softcap ? float(args.softmax_scale * M_LOG2E) : float(args.softcap_val * M_LOG2E),
                args.ptr_q_scale, args.ptr_k_scale, args.ptr_v_scale,
                !Has_softcap ? 0.f : args.softmax_scale / args.softcap_val,
                args.window_size_left, args.window_size_right,
                args.cu_seqlens_q, args.cu_seqlens_k, args.seqused_q, args.seqused_k};
    }

    /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
    CUTLASS_DEVICE
    static void prefetch_tma_descriptors(Params const& params) {
        cute::prefetch_tma_descriptor(params.tma_load_Q.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_K.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_V.get_tma_descriptor());
    }

    CUTLASS_DEVICE
    int get_seqlen_q(Params const& params, int bidb) {
        if constexpr (!Varlen) {
            return get<0>(params.shape_Q);
        } else {
            return params.seqused_q ? params.seqused_q[bidb] : params.cu_seqlens_q[bidb + 1] - params.cu_seqlens_q[bidb];
        }
    }

    CUTLASS_DEVICE
    int get_seqlen_k(Params const& params, int bidb) {
        if constexpr (!Varlen) {
            return get<0>(params.shape_K);
        } else {
            return params.seqused_k ? params.seqused_k[bidb] : params.cu_seqlens_k[bidb + 1] - params.cu_seqlens_k[bidb];
        }
    }

    CUTLASS_DEVICE
    int get_n_block_max(Params const& params, int m_block, int bidb) {
        static constexpr int kBlockM = get<0>(TileShape_MNK{});
        static constexpr int kBlockN = get<1>(TileShape_MNK{});
        int const seqlen_k = get_seqlen_k(params, bidb);
        int n_block_max = cute::ceil_div(seqlen_k, kBlockN);
        if constexpr (Is_causal || Is_local) {
            int const seqlen_q = get_seqlen_q(params, bidb);
            n_block_max = std::min(n_block_max,
                                   cute::ceil_div((m_block + 1) * kBlockM + seqlen_k - seqlen_q + params.window_size_right, kBlockN));
        }
        return n_block_max;
    }

    CUTLASS_DEVICE
    int get_n_block_min(Params const& params, int m_block, int bidb) {
        if (!Is_local) {
            return 0;
        } else {
            static constexpr int kBlockM = get<0>(TileShape_MNK{});
            static constexpr int kBlockN = get<1>(TileShape_MNK{});
            int const seqlen_k = get_seqlen_k(params, bidb);
            int const seqlen_q = get_seqlen_q(params, bidb);
            return std::max(int(0), (m_block * kBlockM + seqlen_k - seqlen_q - params.window_size_left) / kBlockN);
        }
    }

    template <typename SchedulerPrefetch, typename SharedStorage>
    CUTLASS_DEVICE void
    load(Params const& params,
         MainloopPipelineK pipeline_k,
         MainloopPipelineV pipeline_v,
         PipelineState& smem_pipe_write,
         SharedStorage &shared_storage,
         SchedulerPrefetch const& scheduler_prefetch,
         cute::tuple<int32_t, int32_t, int32_t> block_coord,
         int work_idx
         ) {

        Tensor sQ = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()), SmemLayoutQ{});
        Tensor sK = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()), SmemLayoutK{});
        Tensor sVt = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutVt{});

        auto [m_block, bidh, bidb] = block_coord;
        int bidh_kv = params.qhead_per_khead_divmod.divide(bidh);

        // Prepare the TMA loads
        uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();
        constexpr uint32_t cluster_shape_x = get<0>(ClusterShape());
        uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x, block_rank_in_cluster / cluster_shape_x};
        Tensor mQ = params.tma_load_Q.get_tma_tensor(params.shape_Q)(_, _, bidh, !Varlen ? bidb : 0);
        Tensor mK = params.tma_load_K.get_tma_tensor(params.shape_K)(_, _, bidh_kv, !Varlen ? bidb : 0);
        Tensor mVt = params.tma_load_V.get_tma_tensor(select<1, 0, 2, 3>(params.shape_K))(_, _, bidh_kv, !Varlen ? bidb : 0);

        Tensor gQ = local_tile(domain_offset(make_coord(!Varlen ? 0 : params.cu_seqlens_q[bidb], _0{}), mQ), select<0, 2>(TileShape_MNK{}), make_coord(m_block, _0{}));  // (M, K)
        Tensor gK = local_tile(domain_offset(make_coord(!Varlen ? 0 : params.cu_seqlens_k[bidb], _0{}), mK), select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (N, K, _)
        Tensor gVt = local_tile(domain_offset(make_coord(_0{}, !Varlen ? 0 : params.cu_seqlens_k[bidb]), mVt), select<2, 1>(TileShape_MNK{}), make_coord(_0{}, _));  // (K, N, _)

        Tensor sQ_x = make_tensor(sQ.data(), make_layout(sQ.layout(), Layout<_1>{}));
        Tensor gQ_x = make_tensor(gQ.data(), make_layout(gQ.layout(), Layout<_1>{}));
        auto [tQgQ, tQsQ] = tma_partition(params.tma_load_Q, _0{}, Layout<_1>{},
                                          group_modes<0, 2>(sQ_x), group_modes<0, 2>(gQ_x));  // (TMA), (TMA)
        auto [tKgK, tKsK] = tma_partition(params.tma_load_K, block_rank_in_cluster, Layout<ClusterShape>{},
                                          group_modes<0, 2>(sK), group_modes<0, 2>(gK));  // (TMA, k), (TMA, PIPE)
        auto [tVgVt, tVsVt] = tma_partition(params.tma_load_V, block_rank_in_cluster, Layout<ClusterShape>{},
                                          group_modes<0, 2>(sVt), group_modes<0, 2>(gVt));  // (TMA, k), (TMA, PIPE)

        uint16_t mcast_mask_kv = 0;
        if constexpr (cute::is_same_v<GmemTiledCopyKV, SM90_TMA_LOAD_MULTICAST>) {
            auto block_layout = Layout<ClusterShape>{}; // (m,n) -> block_id
            for (int m = 0; m < size<0>(block_layout); ++m) {
                mcast_mask_kv |= (uint16_t(1) << block_layout(m, cluster_local_block_id.y, _0{}));
            }
        }

        int n_block_max = get_n_block_max(params, m_block, bidb);
        int n_block_min = get_n_block_min(params, m_block, bidb);
        int n_block = n_block_max - 1;

        int lane_predicate = cute::elect_one_sync();
        if (lane_predicate) {
            pipeline_k.producer_acquire(smem_pipe_write);
            if constexpr (size(ClusterShape{}) == 1) {
                copy(params.tma_load_K.with(*pipeline_k.producer_get_barrier(smem_pipe_write), mcast_mask_kv, TMA::CacheHintSm90::EVICT_LAST),
                    tKgK(_, n_block), tKsK(_, smem_pipe_write.index()));
            } else {
                copy(params.tma_load_K.with(*pipeline_k.producer_get_barrier(smem_pipe_write), mcast_mask_kv),
                    tKgK(_, n_block), tKsK(_, smem_pipe_write.index()));
            }
        }

        // Wait for the MMA warpgroups to say that smem_q is ready
        cutlass::arch::NamedBarrier::sync(NumMmaThreads + cutlass::NumThreadsPerWarp, static_cast<int>(FwdNamedBarriers::QueryEmpty) /*id*/);

        if (lane_predicate) {
            shared_storage.pipelines.barrier_Q.arrive_and_expect_tx(TmaTransactionBytesQ);
            copy(params.tma_load_Q.with(reinterpret_cast<typename cutlass::arch::ClusterTransactionBarrier::ValueType&>(shared_storage.pipelines.barrier_Q), 0 /*mcast_mask*/, TMA::CacheHintSm90::EVICT_FIRST),
                 tQgQ, tQsQ);
        }

        // Wait for warp 1 to signal that smem_v are ready and V can be copied from gmem
        // Need ClusterBarrier, not just NamedBarrier. Otherwise we might have CTA 0 finishing the
        // TMA store on O first, call TMA multicast load on V, before CTA 1 can finishing TMA store on O.
        shared_storage.pipelines.barrier_O.wait((work_idx + 1) % 2);

        if (lane_predicate) {
            // CUTLASS_PRAGMA_NO_UNROLL
            #pragma unroll 2
            for (; n_block > n_block_min; --n_block) {
                PipelineState smem_pipe_write_v = smem_pipe_write; // copy the state, write_v is always 1 step behind
                ++smem_pipe_write;
                pipeline_k.producer_acquire(smem_pipe_write);
                if constexpr (size(ClusterShape{}) == 1) {
                    copy(params.tma_load_K.with(*pipeline_k.producer_get_barrier(smem_pipe_write), mcast_mask_kv, TMA::CacheHintSm90::EVICT_LAST),
                        tKgK(_, n_block - 1), tKsK(_, smem_pipe_write.index()));
                } else {
                    copy(params.tma_load_K.with(*pipeline_k.producer_get_barrier(smem_pipe_write), mcast_mask_kv),
                        tKgK(_, n_block - 1), tKsK(_, smem_pipe_write.index()));
                }
                pipeline_v.producer_acquire(smem_pipe_write_v);
                if constexpr (size(ClusterShape{}) == 1) {
                    copy(params.tma_load_V.with(*pipeline_v.producer_get_barrier(smem_pipe_write_v), mcast_mask_kv, TMA::CacheHintSm90::EVICT_LAST),
                        tVgVt(_, n_block), tVsVt(_, smem_pipe_write_v.index()));
                } else {
                    copy(params.tma_load_V.with(*pipeline_v.producer_get_barrier(smem_pipe_write_v), mcast_mask_kv),
                        tVgVt(_, n_block), tVsVt(_, smem_pipe_write_v.index()));
                }
            }
        }
        scheduler_prefetch();
        if (lane_predicate) {
            pipeline_v.producer_acquire(smem_pipe_write);
            if constexpr (size(ClusterShape{}) == 1) {
                copy(params.tma_load_V.with(*pipeline_v.producer_get_barrier(smem_pipe_write), mcast_mask_kv, TMA::CacheHintSm90::EVICT_LAST),
                    tVgVt(_, n_block), tVsVt(_, smem_pipe_write.index()));
            } else {
                copy(params.tma_load_V.with(*pipeline_v.producer_get_barrier(smem_pipe_write), mcast_mask_kv),
                    tVgVt(_, n_block), tVsVt(_, smem_pipe_write.index()));
            }
            ++smem_pipe_write;
        }
    }

    template <typename SchedulerPrefetch, typename SharedStorage>
    CUTLASS_DEVICE void
    load_fp8_transpose_V(
         Params const& params,
         MainloopPipelineK pipeline_k,
         MainloopPipelineV pipeline_v,
         MainloopPipelineVt pipeline_vt,
         PipelineState& smem_pipe_write,
         SharedStorage &shared_storage,
         SchedulerPrefetch const& scheduler_prefetch,
         cute::tuple<int32_t, int32_t, int32_t> block_coord,
         int work_idx
         ) {

        Tensor sQ = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()), SmemLayoutQ{});
        Tensor sK = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()), SmemLayoutK{});
        // as_position_independent_swizzle_tensor makes address calculation easier when we do LDSM & STSM to transpose.
        // But it requires smem_vt and smem_v to be aligned to e.g 512 bytes.
        Tensor sVt = cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_vt.data()), SmemLayoutVt{}));
        Tensor sV = cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutVtMma{}));

        auto [m_block, bidh, bidb] = block_coord;
        int bidh_kv = params.qhead_per_khead_divmod.divide(bidh);

        // Prepare the TMA loads
        uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();
        constexpr uint32_t cluster_shape_x = get<0>(ClusterShape());
        uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x, block_rank_in_cluster / cluster_shape_x};
        Tensor mQ = params.tma_load_Q.get_tma_tensor(params.shape_Q)(_, _, bidh, !Varlen ? bidb : 0);
        Tensor mK = params.tma_load_K.get_tma_tensor(params.shape_K)(_, _, bidh_kv, !Varlen ? bidb : 0);
        Tensor mVt = params.tma_load_V.get_tma_tensor(select<1, 0, 2, 3>(params.shape_K))(_, _, bidh_kv, !Varlen ? bidb : 0);

        Tensor gQ = local_tile(domain_offset(make_coord(!Varlen ? 0 : params.cu_seqlens_q[bidb], _0{}), mQ), select<0, 2>(TileShape_MNK{}), make_coord(m_block, _0{}));  // (M, K)
        Tensor gK = local_tile(domain_offset(make_coord(!Varlen ? 0 : params.cu_seqlens_k[bidb], _0{}), mK), select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (N, K, _)
        Tensor gVt = local_tile(domain_offset(make_coord(_0{}, !Varlen ? 0 : params.cu_seqlens_k[bidb]), mVt), select<2, 1>(TileShape_MNK{}), make_coord(_0{}, _));  // (K, N, _)

        auto block_tma_Q = params.tma_load_Q.get_slice(_0{});
        Tensor tQgQ = group_modes<0, 3>(block_tma_Q.partition_S(gQ));  // (TMA)
        Tensor tQsQ = group_modes<0, 3>(block_tma_Q.partition_D(sQ));  // (TMA)
        // tma_partition doesn't handle position_independent_swizzle_tensor correctly, so we need to do it manually
        auto block_tma_K = params.tma_load_K.get_slice(cluster_local_block_id.x);
        Tensor tKgK = group_modes<0, 3>(block_tma_K.partition_S(gK));  // (TMA, k)
        Tensor tKsK = group_modes<0, 3>(block_tma_K.partition_D(sK));  // (TMA, PIPE)
        auto block_tma_V = params.tma_load_V.get_slice(cluster_local_block_id.x);
        Tensor tVgVt = group_modes<0, 3>(block_tma_V.partition_S(gVt));  // (TMA, k)
        Tensor tVsVt = group_modes<0, 3>(block_tma_V.partition_D(sVt));  // (TMA, PIPE)

        uint16_t mcast_mask_kv = 0;
        if constexpr (cute::is_same_v<GmemTiledCopyKV, SM90_TMA_LOAD_MULTICAST>) {
            auto block_layout = Layout<ClusterShape>{}; // (m,n) -> block_id
            for (int m = 0; m < size<0>(block_layout); ++m) {
                mcast_mask_kv |= (uint16_t(1) << block_layout(m, cluster_local_block_id.y, _0{}));
            }
        }

        // Set up for transposing V
        S2RTiledCopyVt s2r_tiled_copy_vt;
        R2STiledCopyV r2s_tiled_copy_v;
        auto s2r_thr_copy_vt = s2r_tiled_copy_vt.get_thread_slice(threadIdx.x % NumProducerThreads);
        auto r2s_thr_copy_v = r2s_tiled_copy_v.get_thread_slice(threadIdx.x % NumProducerThreads);
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
        };

        int n_block_max = get_n_block_max(params, m_block, bidb);
        int n_block_min = get_n_block_min(params, m_block, bidb);
        int n_block = n_block_max - 1;

        int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
        int lane_predicate = cute::elect_one_sync();
        if (warp_idx_in_warpgroup == 0) {
            if (lane_predicate) {
                pipeline_vt.producer_acquire(smem_pipe_write);
                if constexpr (size(ClusterShape{}) == 1) {
                    copy(params.tma_load_V.with(*pipeline_vt.producer_get_barrier(smem_pipe_write), mcast_mask_kv, TMA::CacheHintSm90::EVICT_LAST),
                        tVgVt(_, n_block), tVsVt(_, smem_pipe_write.index()));
                } else {
                    copy(params.tma_load_V.with(*pipeline_vt.producer_get_barrier(smem_pipe_write), mcast_mask_kv),
                        tVgVt(_, n_block), tVsVt(_, smem_pipe_write.index()));
                }
                pipeline_k.producer_acquire(smem_pipe_write);
                if constexpr (size(ClusterShape{}) == 1) {
                    copy(params.tma_load_K.with(*pipeline_k.producer_get_barrier(smem_pipe_write), mcast_mask_kv, TMA::CacheHintSm90::EVICT_LAST),
                        tKgK(_, n_block), tKsK(_, smem_pipe_write.index()));
                } else {
                    copy(params.tma_load_K.with(*pipeline_k.producer_get_barrier(smem_pipe_write), mcast_mask_kv),
                        tKgK(_, n_block), tKsK(_, smem_pipe_write.index()));
                }
            }
            // Wait for the MMA warpgroups to say that smem_q is ready
            cutlass::arch::NamedBarrier::sync(NumMmaThreads + cutlass::NumThreadsPerWarp, static_cast<int>(FwdNamedBarriers::QueryEmpty) /*id*/);

            if (lane_predicate) {
                shared_storage.pipelines.barrier_Q.arrive_and_expect_tx(TmaTransactionBytesQ);
                copy(params.tma_load_Q.with(reinterpret_cast<typename cutlass::arch::ClusterTransactionBarrier::ValueType&>(shared_storage.pipelines.barrier_Q), 0 /*mcast_mask*/, TMA::CacheHintSm90::EVICT_FIRST),
                    tQgQ, tQsQ);
            }
        }
        --n_block;

        // Wait for warp 1 to signal that smem_v are ready and V can be copied from gmem
        // Need ClusterBarrier, not just NamedBarrier. Otherwise we might have CTA 0 finishing the
        // TMA store on O first, call TMA multicast load on V, before CTA 1 can finishing TMA store on O.
        // if (blockIdx.x == 0 && threadIdx.x % 32 == 0) { printf("tidx = %d, Producer: before barrier_O.wait\n", threadIdx.x); }
        shared_storage.pipelines.barrier_O.wait((work_idx + 1) % 2);

        // CUTLASS_PRAGMA_NO_UNROLL
        #pragma unroll 1
        for (; n_block >= n_block_min; --n_block) {
            PipelineState smem_pipe_write_v = smem_pipe_write; // copy the state, write_v is always 1 step behind
            ++smem_pipe_write;
            if (warp_idx_in_warpgroup == 0 && lane_predicate) {
                pipeline_vt.producer_acquire(smem_pipe_write);
                if constexpr (size(ClusterShape{}) == 1) {
                    copy(params.tma_load_V.with(*pipeline_vt.producer_get_barrier(smem_pipe_write), mcast_mask_kv, TMA::CacheHintSm90::EVICT_LAST),
                        tVgVt(_, n_block), tVsVt(_, smem_pipe_write.index()));
                } else {
                    copy(params.tma_load_V.with(*pipeline_vt.producer_get_barrier(smem_pipe_write), mcast_mask_kv),
                        tVgVt(_, n_block), tVsVt(_, smem_pipe_write.index()));
                }
                pipeline_k.producer_acquire(smem_pipe_write);
                if constexpr (size(ClusterShape{}) == 1) {
                    copy(params.tma_load_K.with(*pipeline_k.producer_get_barrier(smem_pipe_write), mcast_mask_kv, TMA::CacheHintSm90::EVICT_LAST),
                        tKgK(_, n_block), tKsK(_, smem_pipe_write.index()));
                } else {
                    copy(params.tma_load_K.with(*pipeline_k.producer_get_barrier(smem_pipe_write), mcast_mask_kv),
                        tKgK(_, n_block), tKsK(_, smem_pipe_write.index()));
                }
            }
            // Instead of maintaining smem_pipe_read_v as a separate variable, we can just use smem_pipe_write_v,
            // and exploit the invariance that smem_pipe_write_v.phase() == smem_pipe_read_v.phase() ^ 1.
            // This saves 1 or 2 registers.
            PipelineState smem_pipe_read_v{smem_pipe_write_v.index(), smem_pipe_write_v.phase() ^ 1, smem_pipe_write_v.count()};
            pipeline_vt.consumer_wait(smem_pipe_read_v);
            pipeline_v.producer_acquire(smem_pipe_write_v);
            transpose_V(smem_pipe_write_v.index());
            // SMEM fence to make sure V is transposed before math
            cutlass::arch::fence_view_async_shared();
            pipeline_v.producer_commit(smem_pipe_write_v);
            // PipelineTmaAsync::consumer_release assumes that the warpgroup is synchronized before calling
            cutlass::arch::NamedBarrier::sync(cutlass::NumThreadsPerWarpGroup, static_cast<int>(FwdNamedBarriers::ProducerWG) /*id*/);
            pipeline_vt.consumer_release(smem_pipe_read_v);
        }
        scheduler_prefetch();
        PipelineState smem_pipe_read_v{smem_pipe_write.index(), smem_pipe_write.phase() ^ 1, smem_pipe_write.count()};
        pipeline_vt.consumer_wait(smem_pipe_read_v);
        pipeline_v.producer_acquire(smem_pipe_write);
        transpose_V(smem_pipe_write.index());
        // SMEM fence to make sure V is transposed before math
        cutlass::arch::fence_view_async_shared();
        pipeline_v.producer_commit(smem_pipe_write);
        cutlass::arch::NamedBarrier::sync(cutlass::NumThreadsPerWarpGroup, static_cast<int>(FwdNamedBarriers::ProducerWG) /*id*/);
        pipeline_vt.consumer_release(smem_pipe_read_v);
        ++smem_pipe_write;
    }

    /// Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
    CUTLASS_DEVICE void
    load_tail(MainloopPipelineK pipeline_k, MainloopPipelineV pipeline_v, PipelineState& smem_pipe_write) {
        int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
        int lane_predicate = cute::elect_one_sync();
        // Issue the epilogue waits
        if (warp_idx_in_warpgroup == 0 && lane_predicate) {
            /* This helps avoid early exit of blocks in Cluster
             * Waits for all stages to either be released (all Consumer UNLOCKs), or if the stage was never used
             * then would just be acquired since the phase was still inverted from make_producer_start_state
             */
            pipeline_k.producer_tail(smem_pipe_write);
            pipeline_v.producer_tail(smem_pipe_write);
        }
    }

    CUTLASS_DEVICE void
    load_tail(MainloopPipelineK pipeline_k, MainloopPipelineV pipeline_v, MainloopPipelineVt pipeline_vt,
              PipelineState& smem_pipe_write) {
        int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
        int lane_predicate = cute::elect_one_sync();
        // Issue the epilogue waits
        if (warp_idx_in_warpgroup == 0 && lane_predicate) {
            /* This helps avoid early exit of blocks in Cluster
            *  Waits for all stages to either be released (all Consumer UNLOCKs), or if the stage was never used
            *  then would just be acquired since the phase was still inverted from make_producer_start_state
            */
            pipeline_k.producer_tail(smem_pipe_write);
            pipeline_v.producer_tail(smem_pipe_write);
            pipeline_vt.producer_tail(smem_pipe_write);
        }
    }

    CUTLASS_DEVICE void
    warp_scheduler_barrier_sync() {
        if constexpr (UseSchedulerBarrier) {
            cutlass::arch::NamedBarrier::sync(NumMmaThreads, static_cast<int>(FwdNamedBarriers::WarpSchedulerWG1) - 1 + cutlass::canonical_warp_group_idx() /*id*/);
        }
    }

    CUTLASS_DEVICE void
    warp_scheduler_barrier_arrive() {
        if constexpr (!UseSchedulerBarrier) { return; }
        static_assert(NumMmaThreads == 2 * cutlass::NumThreadsPerWarpGroup || NumMmaThreads == 3 * cutlass::NumThreadsPerWarpGroup);
        if constexpr (NumMmaThreads == 2 * cutlass::NumThreadsPerWarpGroup) {
            cutlass::arch::NamedBarrier::arrive(NumMmaThreads, static_cast<int>(FwdNamedBarriers::WarpSchedulerWG1) - 1 + (3 - cutlass::canonical_warp_group_idx()) /*id*/);
        } else {
            cutlass::arch::NamedBarrier::arrive(NumMmaThreads, static_cast<int>(FwdNamedBarriers::WarpSchedulerWG1) - 1 + (cutlass::canonical_warp_group_idx() <= 2 ? cutlass::canonical_warp_group_idx() + 1 : cutlass::canonical_warp_group_idx() + 1 - 3)  /*id*/);
            cutlass::arch::NamedBarrier::arrive(NumMmaThreads, static_cast<int>(FwdNamedBarriers::WarpSchedulerWG1) - 1 + (cutlass::canonical_warp_group_idx() <= 1 ? cutlass::canonical_warp_group_idx() + 2 : cutlass::canonical_warp_group_idx() + 2 - 3)  /*id*/);
        }
    }

    CUTLASS_DEVICE void
    mma_init() {
        // Tell producer (warp 0) that smem_q is ready
        cutlass::arch::NamedBarrier::arrive(NumMmaThreads + cutlass::NumThreadsPerWarp, static_cast<int>(FwdNamedBarriers::QueryEmpty) /*id*/);
        if constexpr (!UseSchedulerBarrier) { return; }
        static_assert(NumMmaThreads == 2 * cutlass::NumThreadsPerWarpGroup || NumMmaThreads == 3 * cutlass::NumThreadsPerWarpGroup);
        if (cutlass::canonical_warp_group_idx() > 1) {
            cutlass::arch::NamedBarrier::arrive(NumMmaThreads, static_cast<int>(FwdNamedBarriers::WarpSchedulerWG1) - 1 + 1 /*id*/);
        }
        if constexpr (NumMmaThreads == 3 * cutlass::NumThreadsPerWarpGroup) {
            if (cutlass::canonical_warp_group_idx() > 2) {
                cutlass::arch::NamedBarrier::arrive(NumMmaThreads, static_cast<int>(FwdNamedBarriers::WarpSchedulerWG1) - 1 + 2 /*id*/);
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
        cute::tuple<int32_t, int32_t, int32_t> block_coord,
        SharedStorage& shared_storage
        ) {
        static_assert(is_rmem<FrgTensorO>::value, "O tensor must be rmem resident.");

        static constexpr int kBlockM = get<0>(TileShape_MNK{});
        static constexpr int kBlockN = get<1>(TileShape_MNK{});

        Tensor sQ = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()), SmemLayoutQ{});
        Tensor sK = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()), SmemLayoutK{});
        Tensor sVt = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutVtMma{});

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
        auto thread_mma0 = tiled_mma0.get_thread_slice(thread_idx);
        auto wg_mma1 = tiled_mma1.get_slice(warp_group_thread_layout(warp_group_idx));

        // Allocate "fragments/descriptors"
        Tensor tSrQ = wg_mma0.partition_fragment_A(sQ);
        Tensor tSrK = wg_mma0.partition_fragment_B(sK);
        Tensor tOrV = wg_mma1.partition_fragment_B(sVt);

        auto consumer_wait = [](auto& pipeline, auto& smem_pipe_read) {
            auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
            pipeline.consumer_wait(smem_pipe_read, barrier_token);
        };

        // clear(tOrO);
        tiled_mma1.accumulate_ = GMMA::ScaleOut::Zero;

        int m_block = get<0>(block_coord);
        int bidb = get<2>(block_coord);
        int const seqlen_q = get_seqlen_q(params, bidb);
        int const seqlen_k = get_seqlen_k(params, bidb);
        int n_block_max = get_n_block_max(params, m_block, bidb);
        int n_block_min = get_n_block_min(params, m_block, bidb);
        int n_block = n_block_max - 1;

        auto causal_local_mask_fn = [&](auto& tSrS, int const n_block, auto need_seqlenk_masking_type, auto is_causal_type, auto is_local_type) {
            constexpr bool Need_seqlenk_masking = decltype(need_seqlenk_masking_type)::value;
            constexpr bool Is_causal = decltype(is_causal_type)::value;
            constexpr bool Is_local = decltype(is_local_type)::value;
            Tensor cS = cute::make_identity_tensor(select<0, 1>(TileShape_MNK{}));
            Tensor tScS = thread_mma0.partition_C(cS);
            if constexpr (!Is_causal && !Is_local) {
                if constexpr (Need_seqlenk_masking) {  // Just masking based on col
                    #pragma unroll
                    for (int i = 0; i < size(tSrS); ++i) {
                        if (int(get<1>(tScS(i))) >= int(seqlen_k - n_block * kBlockN)) { tSrS(i) = -INFINITY; }
                    }
                }
            } else {  // mask based on both row and col
                int causal_row_offset = 1 + seqlen_k - n_block * kBlockN - seqlen_q + m_block * kBlockM;
                if constexpr (Is_causal) {
                    #pragma unroll
                    for (int i = 0; i < size(tSrS); ++i) {
                        // using std::min is faster than doing col >= limit0 or col >= limit1
                        // Need to cast get<1>(tScS(i)) to (signed) int since by default it's unsigned, and the
                        // right hand side can be negative and might be converted to a very large unsigned integer.
                        int col_limit_right = !Need_seqlenk_masking
                            ? int(get<0>(tScS(i))) + causal_row_offset
                            : std::min(int(get<0>(tScS(i))) + causal_row_offset, seqlen_k - n_block * kBlockN);
                        if (int(get<1>(tScS(i))) >= col_limit_right) { tSrS(i) = -INFINITY; }
                    }
                } else {
                    int local_row_offset_right = causal_row_offset + params.window_size_right;
                    int local_row_offset_left = causal_row_offset - 1 - params.window_size_left;
                    #pragma unroll
                    for (int i = 0; i < size(tSrS); ++i) {
                        int col_limit_right = !Need_seqlenk_masking
                            ? int(get<0>(tScS(i))) + local_row_offset_right
                            : __viaddmin_s32(int(get<0>(tScS(i))), local_row_offset_right, seqlen_k - n_block * kBlockN);
                        int col_limit_left = int(get<0>(tScS(i))) + local_row_offset_left;
                        if (int(get<1>(tScS(i))) >= col_limit_right || int(get<1>(tScS(i))) < col_limit_left) {
                            tSrS(i) = -INFINITY;
                        }
                    }
                }
            }
        };

        typename cutlass::ConsumerToken barrier_token = static_cast<cutlass::BarrierStatus>(shared_storage.pipelines.barrier_Q.try_wait(work_idx % 2));
        if (barrier_token == cutlass::BarrierStatus::WaitAgain) { shared_storage.pipelines.barrier_Q.wait(work_idx % 2); }

        if constexpr (true) {
            Tensor tSrS = partition_fragment_C(tiled_mma0, select<0, 1>(TileShape_MNK{}));
            // if (blockIdx.x == 0 && threadIdx.x == 128) { print(tSrS); }
            consumer_wait(pipeline_k, smem_pipe_read);
            warp_scheduler_barrier_sync();
            flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma0, tSrQ, tSrK(_, _, _, smem_pipe_read.index()), tSrS);
            warp_scheduler_barrier_arrive();
            if (work_idx != 0) {
                int lane_predicate = cute::elect_one_sync();
                int warp_idx_sync = __shfl_sync(0xffffffff, thread_idx / cutlass::NumThreadsPerWarp, 0);
                if (warp_idx_sync == NumMmaThreads / cutlass::NumThreadsPerWarp - 1 && lane_predicate) {
                    if constexpr (!Varlen) { tma_store_wait<0>(); }
                    #pragma unroll
                    for (uint32_t cta_id = 0; cta_id < size(ClusterShape{}); ++cta_id) {
                        shared_storage.pipelines.barrier_O.arrive(cta_id, lane_predicate);
                    }
                }
            }
            warpgroup_wait<0>();
            pipeline_k.consumer_release(smem_pipe_read);
            // This needs to happen before masking since if we apply after masking, softcapping can turn
            // -inf to e.g. -50.0, which can affect the attention softmax.
            if constexpr (Has_softcap) { flash::apply_softcap(tSrS, params.softcap_val); }

            causal_local_mask_fn(tSrS, n_block, cute::bool_constant<true>{} /*need_seqlenk_masking*/, cute::bool_constant<Is_causal>{}, cute::bool_constant<Is_local>{});

            Tensor scores_scale = softmax.template max_get_scale</*Is_first=*/true, /*Check_inf=*/true>(tSrS);
            softmax.template online_softmax</*Is_first=*/true, /*Check_inf=*/true>(tSrS);
            if constexpr (Is_FP8 && !V_colmajor) { flash::permute_Cregs_fp8(tSrS); }
            Tensor tOrP = make_tensor(convert_type<Element>(tSrS).data(), convert_layout_acc_Aregs<TiledMma1>(tSrS.layout()));
            if constexpr (Is_FP8 && V_colmajor) { flash::permute_Aregs_fp8(tOrP); }

            // Each step does gemm0 for iter n_block - 1, gemm1 for iter n_block, and softmax for iter n_block - 1.
            auto fwd_step = [&](int n_block, auto mask_fn, auto is_first_iter_type, auto check_inf_type) {
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
                pipeline_k.consumer_release(smem_pipe_read);  // release K
                if constexpr (Has_softcap) { flash::apply_softcap(tSrS, params.softcap_val); }
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
                auto mask_fn = [&](auto& tSrS, int n_block) { causal_local_mask_fn(tSrS, n_block, cute::bool_constant<false>{} /*need_seqlenk_masking*/, cute::bool_constant<Is_causal>{}, cute::bool_constant<Is_local>{}); };
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
                auto local_mask_fn = [&](auto& tSrS, int n_block) { causal_local_mask_fn(tSrS, n_block, cute::bool_constant<false>{} /*need_seqlenk_masking*/, cute::bool_constant<false>{} /*is_causal*/, cute::bool_constant<Is_local>{}); };
                #pragma unroll 1
                for (; n_block > n_block_min; --n_block) {
                    fwd_step(n_block, local_mask_fn, cute::bool_constant<false>{} /*is_first_iter*/, cute::bool_constant<Is_local>{} /*check_inf*/);
                }
            }
            // Tell warp 0 that smem_q is ready
            cutlass::arch::NamedBarrier::arrive(NumMmaThreads + cutlass::NumThreadsPerWarp, static_cast<int>(FwdNamedBarriers::QueryEmpty) /*id*/);
            if constexpr (RescaleOBeforeGemm) { softmax.rescale_o(tOrO, scores_scale); }
            consumer_wait(pipeline_v, smem_pipe_read);
            flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(tiled_mma1, tOrP, tOrV(_, _, _, smem_pipe_read.index()), tOrO);
            cute::copy(softmax.finalize(!Is_FP8 || params.ptr_v_scale == nullptr ? 1.f : *params.ptr_v_scale), scores_scale);
            warpgroup_wait<0>();
            pipeline_v.consumer_release(smem_pipe_read);  // release V, otherwise producers will hang
            softmax.rescale_o(tOrO, scores_scale);
            if constexpr (Is_FP8 && !V_colmajor) { flash::permute_output_fp8(tOrO); }
            ++smem_pipe_read;

        } else {
            // WIP

            if (work_idx != 0) {
                int lane_predicate = cute::elect_one_sync();
                int warp_idx_sync = __shfl_sync(0xffffffff, thread_idx / cutlass::NumThreadsPerWarp, 0);
                if (warp_idx_sync == NumMmaThreads / cutlass::NumThreadsPerWarp - 1 && lane_predicate) {
                    if constexpr (!Varlen) { tma_store_wait<0>(); }
                    #pragma unroll
                    for (uint32_t cta_id = 0; cta_id < size(ClusterShape{}); ++cta_id) {
                        shared_storage.pipelines.barrier_O.arrive(cta_id, lane_predicate);
                    }
                }
            }

            #pragma unroll 1
            for (; n_block >= 0; --n_block) {
                Tensor tSrS = partition_fragment_C(tiled_mma0, select<0, 1>(TileShape_MNK{}));
                consumer_wait(pipeline_k, smem_pipe_read);
                flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma0, tSrQ, tSrK(_, _, _, smem_pipe_read.index()), tSrS);
                warpgroup_wait<0>();
                pipeline_k.consumer_release(smem_pipe_read);  // release K
                Tensor scores_scale = softmax.template max_get_scale</*Is_first=*/false>(tSrS);
                warp_scheduler_barrier_sync();
                softmax.template online_softmax</*Is_first=*/false>(tSrS);
                Tensor tOrP = make_tensor(convert_type<Element>(tSrS).data(), convert_layout_acc_Aregs<TiledMma1>(tSrS.layout()));
                warp_scheduler_barrier_arrive();
                if constexpr (Is_FP8) { flash::permute_Aregs_fp8(tOrP); }
                softmax.rescale_o(tOrO, scores_scale);
                consumer_wait(pipeline_v, smem_pipe_read);
                flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(tiled_mma1, tOrP, tOrV(_, _, _, smem_pipe_read.index()), tOrO);
                warpgroup_wait<0>();
                pipeline_v.consumer_release(smem_pipe_read);  // release V
                ++smem_pipe_read;
            }
            // Tell warp 0 that smem_q is ready
            cutlass::arch::NamedBarrier::arrive(NumMmaThreads + cutlass::NumThreadsPerWarp, static_cast<int>(FwdNamedBarriers::QueryEmpty) /*id*/);
            Tensor scores_scale = softmax.finalize();
            softmax.rescale_o(tOrO, scores_scale);

        }

    }

};

} // namespace flash

