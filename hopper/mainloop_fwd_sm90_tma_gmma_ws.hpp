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

template <typename Ktraits, bool Is_causal>
struct CollectiveMainloopFwd {

    using Element = typename Ktraits::Element;
    using TileShape_MNK = typename Ktraits::TileShape_MNK;
    using ClusterShape = typename Ktraits::ClusterShape_MNK;

    static constexpr int kStages = Ktraits::kStages;
    static constexpr int kHeadDim = Ktraits::kHeadDim;

    using GmemTiledCopyQ = cute::SM90_TMA_LOAD;
    using GmemTiledCopyKV = decltype(cutlass::gemm::collective::detail::sm90_cluster_shape_to_tma_atom(shape<0>(ClusterShape{})));

    using SmemLayoutAtomQ = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQ{}, select<0, 2>(TileShape_MNK{})));

    using SmemLayoutAtomK = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutK =
        decltype(tile_to_shape(SmemLayoutAtomK{},
                 make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{})));
    using SmemLayoutV = SmemLayoutK;
    // Note this is the transpose in terms of the view, not in terms of memory.
    using SmemLayoutVt =
        decltype(cute::composition(SmemLayoutV{},
                                   make_layout(make_shape(get<2>(TileShape_MNK{}), get<1>(TileShape_MNK{}), Int<kStages>{}),
                                               make_stride(get<1>(TileShape_MNK{}), _1{}, Int<size(SmemLayoutV{}(_, _, _0{}))>{}))));
    // using SmemLayoutAtomVt = cute::GMMA::Layout_MN_SW128_Atom<Element>;
    // using SmemLayoutVt =
    //     decltype(tile_to_shape(SmemLayoutAtomVt{},
    //                            make_shape(shape<2>(TileShape_MNK{}), shape<1>(TileShape_MNK{}), Int<kStages>{}),
    //                            Step<_2, _1, _3>{}));  // This gives correct results, without Step it's wrong
    // using SmemLayoutAtomVt = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::MN, Element,
    //     decltype(cute::get<2>(TileShape_MNK{})), decltype(cute::get<1>(TileShape_MNK{}))>());
    // using SmemLayoutVt =
    //     decltype(tile_to_shape(SmemLayoutAtomVt{},
    //              make_shape(shape<2>(TileShape_MNK{}), shape<1>(TileShape_MNK{}), Int<kStages>{})));
    // using SmemLayoutAtomVTMA = cute::GMMA::Layout_K_SW128_Atom<Element>;
    // using SmemLayoutVTMA =
    //     decltype(tile_to_shape(SmemLayoutAtomVTMA{},
    //                            make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{})));

    using ShapeQKV = cute::Shape<int32_t, int32_t, int32_t, int32_t>;  // (seqlen, d, head, batch)
    using StrideQKV = cute::Stride<int64_t, _1, int64_t, int64_t>;

    using TMA_Q = decltype(make_tma_copy(
        GmemTiledCopyQ{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), repeat_like(StrideQKV{}, int32_t(0)), StrideQKV{}),
        SmemLayoutQ{},
        select<0, 2>(TileShape_MNK{}),
        _1{}));  // no mcast for Q

    using TMA_KV = decltype(make_tma_copy(
        GmemTiledCopyKV{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), repeat_like(StrideQKV{}, int32_t(0)), StrideQKV{}),
        take<0, 2>(SmemLayoutK{}),
        select<1, 2>(TileShape_MNK{}),
        size<0>(ClusterShape{}))); // mcast along M mode for this N load, if any

    static constexpr int NumMmaThreads = size(typename Ktraits::TiledMma0{});
    using MainloopPipeline = typename Ktraits::MainloopPipeline;
    using PipelineParams = typename MainloopPipeline::Params;
    using PipelineState = typename MainloopPipeline::PipelineState;

    // Set the bytes transferred in this TMA transaction (may involve multiple issues)
    static constexpr uint32_t TmaTransactionBytesQ = static_cast<uint32_t>(size(SmemLayoutQ{}) * cutlass::sizeof_bits_v<Element> / 8);
    static constexpr uint32_t TmaTransactionBytesK = static_cast<uint32_t>(size(take<0, 2>(SmemLayoutK{})) * cutlass::sizeof_bits_v<Element> / 8);

    static constexpr bool UseSchedulerBarrier = kHeadDim <= 128;

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
        float const softmax_scale_log2;
    };

    // Device side kernel params
    struct Params {
        ShapeQKV const shape_Q;
        ShapeQKV const shape_K;
        cutlass::FastDivmod qhead_per_khead_divmod;
        TMA_Q tma_load_Q;
        TMA_KV tma_load_K, tma_load_V;
        float const softmax_scale_log2;
    };


    static Params
    to_underlying_arguments(Arguments const& args) {
        Tensor mQ = make_tensor(make_gmem_ptr(args.ptr_Q), args.shape_Q, args.stride_Q);
        TMA_Q tma_load_Q = make_tma_copy(
            GmemTiledCopyQ{},
            mQ,
            SmemLayoutQ{},
            select<0, 2>(TileShape_MNK{}),
            _1{}); // no mcast for Q
        Tensor mK = make_tensor(make_gmem_ptr(args.ptr_K), args.shape_K, args.stride_K);
        TMA_KV tma_load_K = make_tma_copy(
            GmemTiledCopyKV{},
            mK,
            SmemLayoutK{}(_, _, _0{}),
            select<1, 2>(TileShape_MNK{}),
            size<0>(ClusterShape{})); // mcast along M mode for this N load, if any
        Tensor mV = make_tensor(make_gmem_ptr(args.ptr_V), args.shape_K, args.stride_V);
        TMA_KV tma_load_V = make_tma_copy(
            GmemTiledCopyKV{},
            mV,
            SmemLayoutV{}(_, _, _0{}),
            select<1, 2>(TileShape_MNK{}),
            size<0>(ClusterShape{})); // mcast along M mode for this N load, if any
        return {args.shape_Q, args.shape_K,
                cutlass::FastDivmod(cute::ceil_div(get<2>(args.shape_Q), get<2>(args.shape_K))),
                tma_load_Q, tma_load_K, tma_load_V,
                args.softmax_scale_log2};
    }

    /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
    CUTLASS_DEVICE
    static void prefetch_tma_descriptors(Params const& mainloop_params) {
        cute::prefetch_tma_descriptor(mainloop_params.tma_load_Q.get_tma_descriptor());
        cute::prefetch_tma_descriptor(mainloop_params.tma_load_K.get_tma_descriptor());
        cute::prefetch_tma_descriptor(mainloop_params.tma_load_V.get_tma_descriptor());
    }

    CUTLASS_DEVICE
    int get_n_block_max(Params const& mainloop_params, int m_block) {
        static constexpr int kBlockM = get<0>(TileShape_MNK{});
        static constexpr int kBlockN = get<1>(TileShape_MNK{});
        int const seqlen_q = get<0>(mainloop_params.shape_Q);
        int const seqlen_k = get<0>(mainloop_params.shape_K);
        int n_block_max = cute::ceil_div(seqlen_k, kBlockN);
        if constexpr (Is_causal) {
            n_block_max = std::min(n_block_max,
                                   cute::ceil_div((m_block + 1) * kBlockM + seqlen_k - seqlen_q, kBlockN));
        }
        return n_block_max;
    }

    template <typename Scheduler, typename SharedStorage>
    CUTLASS_DEVICE void
    load(Params const& mainloop_params,
         MainloopPipeline pipeline_k,
         MainloopPipeline pipeline_v,
         PipelineState& smem_pipe_write_k,
         PipelineState& smem_pipe_write_v,
         SharedStorage &shared_storage,
         Scheduler& scheduler,
         typename Scheduler::Params const& scheduler_params,
         typename Scheduler::WorkTileInfo& work_tile_info,
         cute::tuple<int32_t, int32_t, int32_t> block_coord,
         int work_idx
         ) {

        Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{});
        Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{});
        Tensor sV = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutV{});

        Tensor mQ = mainloop_params.tma_load_Q.get_tma_tensor(mainloop_params.shape_Q);
        Tensor mK = mainloop_params.tma_load_K.get_tma_tensor(mainloop_params.shape_K);
        Tensor mV = mainloop_params.tma_load_V.get_tma_tensor(mainloop_params.shape_K);

        auto [m_block, bidh, bidb] = block_coord;
        int bidh_kv = mainloop_params.qhead_per_khead_divmod.divide(bidh);

        // Prepare the TMA loads
        uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();
        constexpr uint32_t cluster_shape_x = get<0>(ClusterShape());
        uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x, block_rank_in_cluster / cluster_shape_x};
        Tensor gQ = local_tile(mQ(_, _, bidh, bidb), select<0, 2>(TileShape_MNK{}), make_coord(m_block, _0{}));  // (M, K)
        Tensor gK = local_tile(mK(_, _, bidh_kv, bidb), select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (N, K, _)
        Tensor gV = local_tile(mV(_, _, bidh_kv, bidb), select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (N, K, _)

        Tensor sQ_x = make_tensor(sQ.data(), make_layout(sQ.layout(), Layout<_1>{}));
        Tensor gQ_x = make_tensor(gQ.data(), make_layout(gQ.layout(), Layout<_1>{}));
        auto [tQgQ, tQsQ] = tma_partition(mainloop_params.tma_load_Q, _0{}, Layout<_1>{},
                                          group_modes<0, 2>(sQ_x), group_modes<0, 2>(gQ_x));  // (TMA), (TMA)
        auto [tKgK, tKsK] = tma_partition(mainloop_params.tma_load_K, block_rank_in_cluster, Layout<ClusterShape>{},
                                          group_modes<0, 2>(sK), group_modes<0, 2>(gK));  // (TMA, k), (TMA, PIPE)
        auto [tVgV, tVsV] = tma_partition(mainloop_params.tma_load_V, block_rank_in_cluster, Layout<ClusterShape>{},
                                          group_modes<0, 2>(sV), group_modes<0, 2>(gV));  // (TMA, k), (TMA, PIPE)

        uint16_t mcast_mask_kv = 0;
        if constexpr (cute::is_same_v<GmemTiledCopyKV, SM90_TMA_LOAD_MULTICAST>) {
            auto block_layout = Layout<ClusterShape>{}; // (m,n) -> block_id
            for (int m = 0; m < size<0>(block_layout); ++m) {
                mcast_mask_kv |= (uint16_t(1) << block_layout(m, cluster_local_block_id.y, _0{}));
            }
        }

        int n_block_max = get_n_block_max(mainloop_params, m_block);
        int n_block = n_block_max - 1;

        int lane_predicate = cute::elect_one_sync();
        if (lane_predicate) {
            pipeline_k.producer_acquire(smem_pipe_write_k);
            copy(mainloop_params.tma_load_K.with(*pipeline_k.producer_get_barrier(smem_pipe_write_k), mcast_mask_kv),
                tKgK(_, n_block), tKsK(_, smem_pipe_write_k.index()));
            ++smem_pipe_write_k;
        }

        // Wait for the MMA warpgroups to say that smem_q is ready
        cutlass::arch::NamedBarrier::sync(NumMmaThreads + cutlass::NumThreadsPerWarp, static_cast<int>(FwdNamedBarriers::QueryEmpty) /*id*/);

        if (lane_predicate) {
            shared_storage.barrier_Q.arrive_and_expect_tx(TmaTransactionBytesQ);
            copy(mainloop_params.tma_load_Q.with(reinterpret_cast<cutlass::arch::ClusterTransactionBarrier::ValueType&>(shared_storage.barrier_Q), 0 /*mcast_mask*/), tQgQ, tQsQ);
        }

        // Wait for warp 1 to signal that smem_v are ready and V can be copied from gmem
        // Need ClusterBarrier, not just NamedBarrier. Otherwise we might have CTA 0 finishing the
        // TMA store on O first, call TMA multicast load on V, before CTA 1 can finishing TMA store on O.
        shared_storage.barrier_O.wait((work_idx + 1) % 2);

        if (lane_predicate) {
            // CUTLASS_PRAGMA_NO_UNROLL
            #pragma unroll 2
            for (; n_block > 0; --n_block) {
                pipeline_k.producer_acquire(smem_pipe_write_k);
                copy(mainloop_params.tma_load_K.with(*pipeline_k.producer_get_barrier(smem_pipe_write_k), mcast_mask_kv),
                    tKgK(_, n_block - 1), tKsK(_, smem_pipe_write_k.index()));
                ++smem_pipe_write_k;
                pipeline_v.producer_acquire(smem_pipe_write_v);
                copy(mainloop_params.tma_load_V.with(*pipeline_v.producer_get_barrier(smem_pipe_write_v), mcast_mask_kv),
                    tVgV(_, n_block), tVsV(_, smem_pipe_write_v.index()));
                ++smem_pipe_write_v;
            }
        }
        scheduler.prefetch_next_work(scheduler_params, work_tile_info);
        if (lane_predicate) {
            pipeline_v.producer_acquire(smem_pipe_write_v);
            copy(mainloop_params.tma_load_V.with(*pipeline_v.producer_get_barrier(smem_pipe_write_v), mcast_mask_kv),
                tVgV(_, n_block), tVsV(_, smem_pipe_write_v.index()));
            ++smem_pipe_write_v;
        }
        scheduler.broadcast_next_work(work_tile_info);
    }

    /// Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
    CUTLASS_DEVICE void
    load_tail(MainloopPipeline pipeline_k, MainloopPipeline pipeline_v,
              PipelineState& smem_pipe_write_k, PipelineState& smem_pipe_write_v) {
        int lane_predicate = cute::elect_one_sync();
        // Issue the epilogue waits
        if (lane_predicate) {
          /* This helps avoid early exit of blocks in Cluster
          * Waits for all stages to either be released (all Consumer UNLOCKs), or if the stage was never used
          * then would just be acquired since the phase was still inverted from make_producer_start_state
          */
          pipeline_k.producer_tail(smem_pipe_write_k);
          pipeline_v.producer_tail(smem_pipe_write_v);
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
    mma(Params const& mainloop_params,
        MainloopPipeline pipeline_k,
        MainloopPipeline pipeline_v,
        PipelineState& smem_pipe_read_k,
        PipelineState& smem_pipe_read_v,
        FrgTensorO& tOrO,
        Softmax& softmax,
        int n_block_count,
        int thread_idx,
        int work_idx,
        int m_block,
        SharedStorage& shared_storage
        ) {
        static_assert(is_rmem<FrgTensorO>::value, "O tensor must be rmem resident.");

        static constexpr int kBlockM = get<0>(TileShape_MNK{});
        static constexpr int kBlockN = get<1>(TileShape_MNK{});

        Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{});
        Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{});
        Tensor sVt = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutVt{});

        typename Ktraits::TiledMma0 tiled_mma0;
        typename Ktraits::TiledMma1 tiled_mma1;
        auto threadMma0 = tiled_mma0.get_thread_slice(thread_idx);
        auto threadMma1 = tiled_mma1.get_thread_slice(thread_idx);

        // Allocate "fragments/descriptors" for first matmul.
        Tensor tSrQ = threadMma0.partition_fragment_A(sQ);
        Tensor tSrK = threadMma0.partition_fragment_B(sK);
        // Allocate "fragments/descriptors" for second matmul.
        // Note: S becomes P.
        Tensor tOrV = threadMma1.partition_fragment_B(sVt);

        auto consumer_wait = [](auto& pipeline, auto& smem_pipe_read) {
            auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
            pipeline.consumer_wait(smem_pipe_read, barrier_token);
        };

        tiled_mma1.accumulate_ = GMMA::ScaleOut::Zero;
        int const seqlen_q = get<0>(mainloop_params.shape_Q);
        int const seqlen_k = get<0>(mainloop_params.shape_K);
        int n_block = n_block_count - 1;

        cutlass::ConsumerToken barrier_token = static_cast<cutlass::BarrierStatus>(shared_storage.barrier_Q.try_wait(work_idx % 2));
        if (barrier_token == cutlass::BarrierStatus::WaitAgain) { shared_storage.barrier_Q.wait(work_idx % 2); }

        Tensor tSrS = partition_fragment_C(tiled_mma0, select<0, 1>(TileShape_MNK{}));
        consumer_wait(pipeline_k, smem_pipe_read_k);
        warp_scheduler_barrier_sync();
        flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma0, tSrQ, tSrK(_, _, _, smem_pipe_read_k.index()), tSrS);
        warp_scheduler_barrier_arrive();
        if (work_idx != 0) {
            int lane_predicate = cute::elect_one_sync();
            if (cutlass::canonical_warp_idx_sync() == Ktraits::kNWarps - 1 && lane_predicate) {
                tma_store_wait<0>();
                #pragma unroll
                for (uint32_t cta_id = 0; cta_id < size(ClusterShape{}); ++cta_id) {
                    shared_storage.barrier_O.arrive(cta_id, lane_predicate);
                }
            }
        }
        warpgroup_wait<0>();
        pipeline_k.consumer_release(smem_pipe_read_k);
        ++smem_pipe_read_k;

        auto col_limit_causal = [&](int row, int n_block) {
            return row + 1 + seqlen_k - n_block * kBlockN - seqlen_q + m_block * kBlockM;
        };
        {
            Tensor cS = cute::make_identity_tensor(select<0, 1>(TileShape_MNK{}));
            Tensor tScS = threadMma0.partition_C(cS);
            #pragma unroll
            for (int i = 0; i < size(tSrS); ++i) {
                if constexpr (!Is_causal) {  // Just masking based on col
                    if (int(get<1>(tScS(i))) >= int(seqlen_k - n_block * kBlockN)) { tSrS(i) = -INFINITY; }
                } else {  // mask based on both row and col
                    // using std::min is faster than doing col >= limit0 or col >= limit1
                    // Need to cast get<1>(tScS(i)) to (signed) int since by default it's unsigned, and the
                    // right hand side can be negative and might be converted to a very large unsigned integer.
                    if (int(get<1>(tScS(i))) >= std::min(seqlen_k - n_block * kBlockN,
                                                        col_limit_causal(int(get<0>(tScS(i))), n_block))) {
                        tSrS(i) = -INFINITY;
                    }
                }
            }
        }

        softmax.template online_softmax</*Is_first=*/true>(tSrS, mainloop_params.softmax_scale_log2);
        Tensor tOrP = make_tensor(convert_type<Element>(tSrS).data(), convert_layout_acc_Aregs<typename Ktraits::TiledMma1>(tSrS.layout()));
        Tensor scores_scale = make_fragment_like(softmax.row_max);
        clear(scores_scale);

        constexpr int n_masking_steps = !Is_causal ? 1 : cute::ceil_div(kBlockM, kBlockN) + 1;
        // Only go through these if Is_causal, since n_masking_steps = 1 when !Is_causal
        #pragma unroll
        for (int masking_step = 0; masking_step < n_masking_steps - 1 && n_block > 0; ++masking_step, --n_block) {
            Tensor tSrS = partition_fragment_C(tiled_mma0, select<0, 1>(TileShape_MNK{}));
            consumer_wait(pipeline_k, smem_pipe_read_k);
            warp_scheduler_barrier_sync();
            flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma0, tSrQ, tSrK(_, _, _, smem_pipe_read_k.index()), tSrS);
            if (masking_step > 0) { softmax.rescale_o(tOrO, scores_scale); }
            consumer_wait(pipeline_v, smem_pipe_read_v);
            flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(tiled_mma1, tOrP, tOrV(_, _, _, smem_pipe_read_v.index()), tOrO);
            warp_scheduler_barrier_arrive();
            warpgroup_wait<1>();
            pipeline_k.consumer_release(smem_pipe_read_k);  // release K
            Tensor cS = cute::make_identity_tensor(select<0, 1>(TileShape_MNK{}));
            Tensor tScS = threadMma0.partition_C(cS);
            #pragma unroll
            for (int i = 0; i < size(tSrS); ++i) {
                if (int(get<1>(tScS(i))) >= col_limit_causal(int(get<0>(tScS(i))), n_block - 1)) {
                    tSrS(i) = -INFINITY;
                }
            }
            cute::copy(softmax.template max</*Is_first=*/false, /*Check_inf=*/true>(tSrS, mainloop_params.softmax_scale_log2), scores_scale);
            softmax.template online_softmax</*Is_first=*/false, /*Check_inf=*/true>(tSrS, mainloop_params.softmax_scale_log2);
            warpgroup_wait<0>();
            pipeline_v.consumer_release(smem_pipe_read_v);  // release V
            ++smem_pipe_read_k;
            ++smem_pipe_read_v;
            cute::copy(make_tensor(convert_type<Element>(tSrS).data(), convert_layout_acc_Aregs<typename Ktraits::TiledMma1>(tSrS.layout())), tOrP);
        }

        #pragma unroll 1
        for (; n_block > 0; --n_block) {
            Tensor tSrS = partition_fragment_C(tiled_mma0, select<0, 1>(TileShape_MNK{}));
            consumer_wait(pipeline_k, smem_pipe_read_k);
            warp_scheduler_barrier_sync();
            flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma0, tSrQ, tSrK(_, _, _, smem_pipe_read_k.index()), tSrS);
            softmax.rescale_o(tOrO, scores_scale);
            consumer_wait(pipeline_v, smem_pipe_read_v);
            flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(tiled_mma1, tOrP, tOrV(_, _, _, smem_pipe_read_v.index()), tOrO);
            warp_scheduler_barrier_arrive();
            warpgroup_wait<1>();
            pipeline_k.consumer_release(smem_pipe_read_k);  // release K
            // auto scores_scale = softmax.template max</*Is_first=*/false>(tSrS);
            cute::copy(softmax.template max</*Is_first=*/false>(tSrS, mainloop_params.softmax_scale_log2), scores_scale);
            softmax.template online_softmax</*Is_first=*/false>(tSrS, mainloop_params.softmax_scale_log2);
            warpgroup_wait<0>();
            pipeline_v.consumer_release(smem_pipe_read_v);  // release V
            ++smem_pipe_read_k;
            ++smem_pipe_read_v;
            // softmax.rescale_o(tOrO, scores_scale);
            cute::copy(make_tensor(convert_type<Element>(tSrS).data(), convert_layout_acc_Aregs<typename Ktraits::TiledMma1>(tSrS.layout())), tOrP);
        }
        // Tell warp 0 that smem_q is ready
        cutlass::arch::NamedBarrier::arrive(NumMmaThreads + cutlass::NumThreadsPerWarp, static_cast<int>(FwdNamedBarriers::QueryEmpty) /*id*/);
        softmax.rescale_o(tOrO, scores_scale);
        consumer_wait(pipeline_v, smem_pipe_read_v);
        flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(tiled_mma1, tOrP, tOrV(_, _, _, smem_pipe_read_v.index()), tOrO);
        cute::copy(softmax.template finalize</*Check_inf=*/Is_causal>(tSrS, mainloop_params.softmax_scale_log2), scores_scale);
        warpgroup_wait<0>();
        pipeline_v.consumer_release(smem_pipe_read_v);  // release V, otherwise producers will hang
        ++smem_pipe_read_v;

        softmax.rescale_o(tOrO, scores_scale);
        return;
    }

};

} // namespace flash

