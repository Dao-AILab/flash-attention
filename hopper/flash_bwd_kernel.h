/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once


#include "cute/tensor.hpp"

#include <cutlass/cutlass.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/array.h>
#include <cutlass/barrier.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include "cutlass/pipeline/pipeline.hpp"

#include "flash.h"
#include "utils.h"
#include "softmax.h"

namespace flash {

using namespace cute;

template <typename Ktraits, bool Is_causal, typename TiledCopyQ, typename TiledCopydO,
          typename TiledCopyK, typename TiledCopyV, typename TiledCopydK, typename TiledCopydV>
__global__ void __launch_bounds__(Ktraits::kNWarps * cutlass::NumThreadsPerWarp, 1)
    compute_dqkv(CUTE_GRID_CONSTANT Flash_bwd_params const params,
                 CUTE_GRID_CONSTANT TiledCopyQ const tma_load_Q,
                 CUTE_GRID_CONSTANT TiledCopydO const tma_load_dO,
                 CUTE_GRID_CONSTANT TiledCopyK const tma_load_K,
                 CUTE_GRID_CONSTANT TiledCopyV const tma_load_V,
                 CUTE_GRID_CONSTANT TiledCopydK const tma_store_dK,
                 CUTE_GRID_CONSTANT TiledCopydV const tma_store_dV) {

    using Element = typename Ktraits::Element;
    using ElementAccum = typename Ktraits::ElementAccum;
    using SoftType = ElementAccum;
    using index_t = typename Ktraits::index_t;
    using TileShape_MNK = typename Ktraits::TileShape_MNK;
    using ClusterShape = typename Ktraits::ClusterShape_MNK;

    static constexpr int kNThreads = Ktraits::kNThreads;
    // static constexpr int NumMmaThreads = size(typename Ktraits::TiledMmaSdP{});
    static constexpr int NumMmaThreads = Ktraits::kNThreads;
    static constexpr int kBlockM = Ktraits::kBlockM;
    // static constexpr int kBlockN = Ktraits::kBlockN;
    // constexpr int kHeadDim = Ktraits::kHeadDim;
    static constexpr int kStages = Ktraits::kStages;

    static constexpr bool SdP_swapAB = Ktraits::SdP_swapAB;
    static constexpr bool dKV_swapAB = Ktraits::dKV_swapAB;
    static constexpr bool dQ_swapAB = Ktraits::dQ_swapAB;

    static constexpr bool Mma_dQ_is_RS = Ktraits::Mma_dQ_is_RS;
    if constexpr (dQ_swapAB) { static_assert(!Mma_dQ_is_RS); }

    using MainloopPipeline = typename Ktraits::MainloopPipeline;
    using PipelineParams = typename MainloopPipeline::Params;
    using PipelineState = typename MainloopPipeline::PipelineState;

    extern __shared__ char shared_memory[];
    auto &shared_storage = *reinterpret_cast<typename Ktraits::SharedStorage*>(shared_memory);

    int const n_block = blockIdx.x;
    int const bidb = blockIdx.z;  // The block index for the batch.
    int const bidh = blockIdx.y;  // The block index for the head.

    int lane_predicate = cute::elect_one_sync();
    int warp_idx = cutlass::canonical_warp_idx_sync();

    // Issue Tma Descriptor Prefetch from a single thread
    if (warp_idx == 0 && lane_predicate) {
        cute::prefetch_tma_descriptor(tma_load_Q.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_load_dO.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_load_K.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_load_V.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_store_dK.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_store_dV.get_tma_descriptor());
    }

    Tensor mQ = tma_load_Q.get_tma_tensor(make_shape(params.seqlen_q, params.d, params.h, params.b));
    Tensor mdO = tma_load_dO.get_tma_tensor(make_shape(params.seqlen_q, params.d, params.h, params.b));
    Tensor mK = tma_load_K.get_tma_tensor(make_shape(params.seqlen_k, params.d, params.h, params.b));
    Tensor mV = tma_load_V.get_tma_tensor(make_shape(params.seqlen_k, params.d, params.h, params.b));
    Tensor mLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum*>(params.softmax_lse_ptr)),
                              make_shape(params.b, params.h, params.seqlen_q),
                              make_stride(params.h * params.seqlen_q, params.seqlen_q, _1{}));
    Tensor mdPsum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum*>(params.dsoftmax_sum)),
                                make_shape(params.b, params.h, params.seqlen_q),
                                make_stride(params.h * params.seqlen_q_rounded, params.seqlen_q_rounded, _1{}));
    Tensor mdQaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum*>(params.dq_accum_ptr)),
                                  make_shape(params.seqlen_q, params.d, params.h, params.b),
                                  make_stride(params.d * params.h, _1{}, params.d, params.d * params.h * params.seqlen_q_rounded));


    Tensor gQ = local_tile(mQ(_, _, bidh, bidb), select<0, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (M, K, _)
    Tensor gdO = local_tile(mdO(_, _, bidh, bidb), select<0, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (M, K, _)
    Tensor gK = local_tile(mK(_, _, bidh, bidb), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{}));  // (N, K)
    Tensor gV = local_tile(mV(_, _, bidh, bidb), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{}));  // (N, K)
    Tensor gdQaccum = local_tile(mdQaccum(_, _, bidh, bidb), select<0, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (M, K, _)
    // if (cute::thread0()) { print(tma_load_K); printf("\n"); }
    // if (cute::thread0()) { print(mK); printf("\n"); print(gK); printf("\n"); }

    typename Ktraits::GmemTiledCopydQaccum gmem_tiled_copy_dQaccum;
    auto gmem_thr_copy_dQaccum = gmem_tiled_copy_dQaccum.get_thread_slice(threadIdx.x);
    Tensor tdQgdQaccum = gmem_thr_copy_dQaccum.partition_D(gdQaccum);

    // Construct SMEM tensors.
    Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), typename Ktraits::SmemLayoutQ{});
    Tensor sdO = make_tensor(make_smem_ptr(shared_storage.smem_do.data()), typename Ktraits::SmemLayoutdO{});
    Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), typename Ktraits::SmemLayoutK{});
    Tensor sV = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), typename Ktraits::SmemLayoutV{});
    Tensor sP = make_tensor(make_smem_ptr(shared_storage.smem_p.data()), typename Ktraits::SmemLayoutP{});
    Tensor sdS = make_tensor(make_smem_ptr(shared_storage.smem_ds.data()), typename Ktraits::SmemLayoutdS{});
    Tensor sQt = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), typename Ktraits::SmemLayoutQt{});
    Tensor sdOt = make_tensor(make_smem_ptr(shared_storage.smem_do.data()), typename Ktraits::SmemLayoutdOt{});
    Tensor sKt = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), typename Ktraits::SmemLayoutKt{});
    Tensor sPt = make_tensor(make_smem_ptr(shared_storage.smem_p.data()), typename Ktraits::SmemLayoutPt{});
    Tensor sdSt = make_tensor(make_smem_ptr(shared_storage.smem_ds.data()), typename Ktraits::SmemLayoutdSt{});

    // Prepare the TMA loads
    uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();
    constexpr uint32_t cluster_shape_x = get<0>(ClusterShape());
    uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x, block_rank_in_cluster / cluster_shape_x};
    auto block_tma_Q = tma_load_Q.get_slice(cluster_local_block_id.y);
    auto block_tma_dO = tma_load_dO.get_slice(cluster_local_block_id.y);
    auto block_tma_K = tma_load_K.get_slice(_0{});
    auto block_tma_V = tma_load_V.get_slice(_0{});

    Tensor tQgQ = block_tma_Q.partition_S(gQ);  // (TMA, TMA_M, TMA_K, k)
    Tensor tQsQ = block_tma_Q.partition_D(sQ);  // (TMA, TMA_M, TMA_K, PIPE)
    Tensor tdOgdO = block_tma_dO.partition_S(gdO);  // (TMA, TMA_M, TMA_K, k)
    Tensor tdOsdO = block_tma_dO.partition_D(sdO);  // (TMA, TMA_M, TMA_K, PIPE)
    Tensor tKgK = block_tma_K.partition_S(gK);  // (TMA, TMA_N, TMA_K)
    Tensor tKsK = block_tma_K.partition_D(sK);  // (TMA, TMA_N, TMA_K)
    Tensor tVgV = block_tma_V.partition_S(gV);  // (TMA, TMA_N, TMA_K)
    Tensor tVsV = block_tma_V.partition_D(sV); // (TMA, TMA_N, TMA_K)
    // if (cute::thread0()) { print(tQgQ); printf("\n"); print(tQsQ); printf("\n"); }
    // if (cute::thread0()) { print(tKgK); printf("\n"); print(tKsK); printf("\n"); }

    // Set the bytes transferred in this TMA transaction (may involve multiple issues)
    constexpr uint32_t TmaTransactionBytesQ = static_cast<uint32_t>(size<0>(sQ) * size<1>(sQ) * cutlass::sizeof_bits_v<Element> / 8);
    constexpr uint32_t TmaTransactionBytesdO = static_cast<uint32_t>(size<0>(sdO) * size<1>(sdO) * cutlass::sizeof_bits_v<Element> / 8);
    static_assert(TmaTransactionBytesQ == TmaTransactionBytesdO);
    constexpr uint32_t TmaTransactionBytesK = static_cast<uint32_t>(size<0>(sK) * size<1>(sK) * cutlass::sizeof_bits_v<Element> / 8);
    constexpr uint32_t TmaTransactionBytesV = static_cast<uint32_t>(size<0>(sV) * size<1>(sV) * cutlass::sizeof_bits_v<Element> / 8);
    static_assert(TmaTransactionBytesK == TmaTransactionBytesV);

    // Obtain warp index
    int thread_idx = int(threadIdx.x);
    int warp_group_thread_idx = thread_idx % cutlass::NumThreadsPerWarpGroup;
    // int const warp_group_thread_idx = threadIdx.x % cutlass::NumThreadsPerWarpGroup;

    PipelineParams pipeline_params;
    pipeline_params.transaction_bytes = TmaTransactionBytesQ;
    pipeline_params.role = MainloopPipeline::ThreadCategory::ProducerConsumer;
    pipeline_params.is_leader = warp_group_thread_idx == 0;
    pipeline_params.num_consumers = NumMmaThreads;

    if (warp_idx == 0 && lane_predicate) {
        shared_storage.barrier_K.init(1 /*numThreads*/);
        shared_storage.barrier_V.init(1 /*numThreads*/);
    }
    // cutlass::arch::fence_barrier_init();
    // We're counting on pipeline_q to call fence_barrier_init();
    MainloopPipeline pipeline_q(shared_storage.pipeline_q, pipeline_params, ClusterShape{});
    MainloopPipeline pipeline_do(shared_storage.pipeline_do, pipeline_params, ClusterShape{});

    // We need this to guarantee that the Pipeline init is visible
    // To all producers and consumer blocks in the Cluster
    if constexpr (size(ClusterShape{}) > 1) {
        cute::cluster_arrive_relaxed();
        cute::cluster_wait();
    } else {
        __syncthreads();
    }

    // State variables used for iterating the circular buffer
    // smem_pipe_read / release is used by the consumer of SMEM data - i.e MMA
    // smem_pipe_write is used by the producer of SMEM data - i.e TMA
    PipelineState smem_pipe_read_q, smem_pipe_read_do;
    PipelineState smem_pipe_release_q, smem_pipe_release_do;
    PipelineState smem_pipe_write_q = cutlass::make_producer_start_state<MainloopPipeline>();
    PipelineState smem_pipe_write_do = cutlass::make_producer_start_state<MainloopPipeline>();

    // Copy K tile and V tile from GMEM to SMEM.
    if (warp_idx == 0 && lane_predicate) {
        shared_storage.barrier_K.arrive_and_expect_tx(TmaTransactionBytesK);
        copy(tma_load_K.with(reinterpret_cast<cutlass::arch::ClusterTransactionBarrier::ValueType&>(shared_storage.barrier_K), 0 /*mcast_mask*/), tKgK, tKsK);
        shared_storage.barrier_V.arrive_and_expect_tx(TmaTransactionBytesV);
        copy(tma_load_V.with(reinterpret_cast<cutlass::arch::ClusterTransactionBarrier::ValueType&>(shared_storage.barrier_V), 0 /*mcast_mask*/), tVgV, tVsV);
    }
    // if (cute::thread0()) { print_tensor(sQ); printf("\n"); } __syncthreads();

    int m_block = cute::ceil_div(params.seqlen_q, kBlockM) - 1;

    uint16_t mcast_mask_qdo = 0;
    if constexpr (cute::is_same_v<typename Ktraits::GmemTiledCopyQdO, SM90_TMA_LOAD_MULTICAST>) {
        auto block_layout = Layout<ClusterShape>{}; // (m,n) -> block_id
        for (int n = 0; n < size<1>(block_layout); ++n) {
            mcast_mask_qdo |= (uint16_t(1) << block_layout(n, cluster_local_block_id.x, _0{}));
        }
    }
    // Issue TmaLoads (Prologue fetches)
    if (warp_idx == 0 && lane_predicate) {
        // Issue the prologue loads
        CUTLASS_PRAGMA_UNROLL
        for (int stage = 0; stage < kStages && stage <= m_block; ++stage) {
            pipeline_q.producer_acquire(smem_pipe_write_q);
            copy(tma_load_Q.with(*pipeline_q.producer_get_barrier(smem_pipe_write_q), mcast_mask_qdo), tQgQ(_, _, _, m_block - stage), tQsQ(_, _, _, stage));
            ++smem_pipe_write_q;
            pipeline_do.producer_acquire(smem_pipe_write_do);
            copy(tma_load_dO.with(*pipeline_do.producer_get_barrier(smem_pipe_write_do), mcast_mask_qdo), tdOgdO(_, _, _, m_block - stage), tdOsdO(_, _, _, stage));
            ++smem_pipe_write_do;
        }
    }

    Tensor gLSE = local_tile(mLSE(bidb, bidh, _), Shape<Int<kBlockM>>{}, make_coord(m_block));
    Tensor gdPsum = local_tile(mdPsum(bidb, bidh, _), Shape<Int<kBlockM>>{}, make_coord(m_block));

    // Initialize matmul objects.
    typename Ktraits::TiledMmaSdP tiledMmaSdP;
    auto threadMmaSdP = tiledMmaSdP.get_thread_slice(threadIdx.x);
    typename Ktraits::TiledMmadKV tiledMmadKV;
    auto threadMmadKV = tiledMmadKV.get_thread_slice(threadIdx.x);
    typename Ktraits::TiledMmadQ tiledMmadQ;
    auto threadMmadQ = tiledMmadQ.get_thread_slice(threadIdx.x);

    // Allocate accumulator
    Tensor tdKrdK = partition_fragment_C(tiledMmadKV, select<!dKV_swapAB ? 1 : 2, !dKV_swapAB? 2 : 1>(TileShape_MNK{}));
    Tensor tdVrdV = partition_fragment_C(tiledMmadKV, select<!dKV_swapAB ? 1 : 2, !dKV_swapAB? 2 : 1>(TileShape_MNK{}));

    auto smem_tiled_copy_PdS = make_tiled_copy_C(typename Ktraits::SmemCopyAtomPdS{}, tiledMmaSdP);
    auto smem_thr_copy_PdS = smem_tiled_copy_PdS.get_thread_slice(threadIdx.x);

    if constexpr (!SdP_swapAB) {
        Tensor tPsP = smem_thr_copy_PdS.partition_D(sP);      // ((Atom,AtomNum),PIPE_M,PIPE_N)
        Tensor tdSsdS = smem_thr_copy_PdS.partition_D(sdS);      // ((Atom,AtomNum),PIPE_M,PIPE_N)

        // Allocate "fragments/descriptors"
        Tensor tSrQ = threadMmaSdP.partition_fragment_A(sQ);
        Tensor tSrK = threadMmaSdP.partition_fragment_B(sK);
        Tensor tdPrdO = threadMmaSdP.partition_fragment_A(sdO);
        Tensor tdPrV = threadMmaSdP.partition_fragment_B(sV);

        Tensor caccS = make_identity_tensor(select<0, 1>(TileShape_MNK{}));    // (BLK_M,BLK_N) -> (blk_m,blk_n)
        Tensor taccScS = threadMmaSdP.partition_C(caccS);                           // (MMA,MMA_N,MMA_N)
        static_assert(decltype(size<0, 0>(taccScS))::value == 2);
        static_assert(decltype(size<0, 1>(taccScS))::value == 2);
        // taccScS has shape ((2, 2, V), MMA_M, MMA_N), we only take only the row indices.
        Tensor taccScS_row = taccScS(make_coord(_0{}, _, _0{}), _, _0{});
        Tensor lse = make_tensor<ElementAccum>(Shape<Int<decltype(size(taccScS_row))::value>>{});
        Tensor dP_sum = make_fragment_like(lse);
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccScS_row(mi));
            lse(mi) = row < params.seqlen_q - m_block * kBlockM ? gLSE(row) : INFINITY;
            dP_sum(mi) = row < params.seqlen_q - m_block * kBlockM ? gdPsum(row) : 0;
        }
        // if (cute::thread0()) { print_tensor(dP_sum); printf("\n"); }
        // We want LSE = inf if the row is OOB. In that case Q would be zero, K would be zero,
        // and scores would be zero. With LSE = 0, probs will be all 1's, and when we multiply
        // with V (which would be zero), we're fine. However, with ALiBi, we might modify these
        // scores, and probs can become NaN. Instead if we set LSE = inf for OOB rows, probs are always 0.

        clear(tdKrdK);
        clear(tdVrdV);

        shared_storage.barrier_K.wait(0);
        shared_storage.barrier_V.wait(0);
        __syncthreads();

        // #pragma unroll 2
        CUTLASS_PRAGMA_NO_UNROLL
        for (; m_block >= 0; --m_block) {
            Tensor tSrS = partition_fragment_C(tiledMmaSdP, select<0, 1>(TileShape_MNK{}));
            pipeline_q.consumer_wait(smem_pipe_read_q);
            __syncwarp();
            flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiledMmaSdP, tSrQ(_, _, _, smem_pipe_read_q.index()), tSrK, tSrS);
            Tensor tdPrdP = partition_fragment_C(tiledMmaSdP, select<0, 1>(TileShape_MNK{}));
            pipeline_do.consumer_wait(smem_pipe_read_do);
            __syncwarp();
            flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiledMmaSdP, tdPrdO(_, _, _, smem_pipe_read_do.index()), tdPrV, tdPrdP);

            warpgroup_wait<1>();
            // Reshape tSrS from ((2, 2, V), MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, V, MMA_N))
            Tensor scores = make_tensor(tSrS.data(), flash::convert_layout_acc_rowcol(tSrS.layout()));
            flash::scale_apply_exp2</*Scale=*/true, /*scale_max=*/false>(scores, lse, params.scale_softmax_log2);
            // if (cute::thread0()) { print_tensor(scores); printf("\n"); }
            // Convert scores from fp32 to fp16/bf16
            Tensor rP = flash::convert_type<Element>(tSrS);
            Tensor tPaP = smem_thr_copy_PdS.retile_S(rP);     // ((Atom,AtomNum), MMA_N, MMA_N)
            cute::copy(smem_tiled_copy_PdS, tPaP, tPsP);
            int const warp_group_idx = cutlass::canonical_warp_group_idx();
            cutlass::arch::NamedBarrier::arrive(kNThreads, warp_group_idx /*id*/);

            warpgroup_wait<0>();
            // Reshape tdPrdP from ((2, 2, V), MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, V, MMA_N))
            Tensor dS = make_tensor(tdPrdP.data(), scores.layout());
            // if (cute::thread0()) { print_tensor(dS); printf("\n"); }
            #pragma unroll
            for (int mi = 0; mi < size<0>(dS); ++mi) {
                #pragma unroll
                for (int ni = 0; ni < size<1>(dS); ++ni) { dS(mi, ni) = scores(mi, ni) * (dS(mi, ni) - dP_sum(mi)); }
            }
            Tensor rdS = flash::convert_type<Element>(tdPrdP);

            Tensor tdSadS = smem_thr_copy_PdS.retile_S(rdS);     // ((Atom,AtomNum), MMA_N, MMA_N)
            cute::copy(smem_tiled_copy_PdS, tdSadS, tdSsdS);
            // cutlass::arch::NamedBarrier::arrive(kNThreads, 1 /*id*/);
            cutlass::arch::NamedBarrier::arrive(kNThreads, 2 + warp_group_idx /*id*/);
            // if (cute::thread0()) { print_tensor(dS); printf("\n"); }

            if (m_block > 0) {
                gLSE.data() = gLSE.data() + (-int(kBlockM));
                gdPsum.data() = gdPsum.data() + (-int(kBlockM));
                #pragma unroll
                for (int mi = 0; mi < size(lse); ++mi) {
                    const int row = get<0>(taccScS_row(mi));
                    lse(mi) = gLSE(row);
                    dP_sum(mi) = gdPsum(row);
                }
            }

            Tensor tdQrdQ = partition_fragment_C(tiledMmadQ, select<!dQ_swapAB ? 0 : 2, !dQ_swapAB ? 2 : 0>(TileShape_MNK{}));
            if constexpr (Mma_dQ_is_RS) {
                static_assert(!dQ_swapAB);
                Tensor tdQrdS = make_tensor(rdS.data(), convert_layout_acc_Aregs<typename Ktraits::TiledMmadQ>(tdPrdP.layout()));
                Tensor tdQrK = threadMmadQ.partition_fragment_B(sKt);
                flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiledMmadQ, tdQrdS, tdQrK, tdQrdQ);
                // if (cute::thread0()) { print(tdQrdS); printf("\n"); print(tdQrK); printf("\n"); print(tdQrdQ); printf("\n"); }
            }

            // warpgroup_wait<0>();
            // Tensor dQ_tmp = make_tensor(tdQrdQ.data(), flash::convert_layout_acc_rowcol(tdQrdQ.layout()));
            // if (cute::thread0()) { print_tensor(dQ_tmp); printf("\n"); }
            // if (cute::thread0()) { print_tensor(sK); printf("\n"); }
            // if (cute::thread0()) { print_tensor(sKt); printf("\n"); } __syncthreads();

            // __syncthreads();  // Without this I'm getting race condition, I thought the barrier would be enough
            // SMEM fence to make sure sP is written before it's read by WGMMA
            cutlass::arch::fence_view_async_shared();
            cutlass::arch::NamedBarrier::sync(kNThreads, 1 - warp_group_idx /*id*/);
            if constexpr (!dKV_swapAB) {
                Tensor tdVrP = threadMmadKV.partition_fragment_A(sPt);
                Tensor tdVrdO = threadMmadKV.partition_fragment_B(sdOt);
                flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(tiledMmadKV, tdVrP, tdVrdO(_, _, _, smem_pipe_read_do.index()), tdVrdV);
            } else {
                Tensor tdVrP = threadMmadKV.partition_fragment_B(sPt);
                Tensor tdVrdO = threadMmadKV.partition_fragment_A(sdOt);
                flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(tiledMmadKV, tdVrdO(_, _, _, smem_pipe_read_do.index()), tdVrP, tdVrdV);
            }
            ++smem_pipe_read_do;

            // warpgroup_wait<0>();
            // Tensor dV_tmp = make_tensor(tdVrdV.data(), flash::convert_layout_acc_rowcol(tdVrdV.layout()));
            // if (cute::thread0()) { print_tensor(dV_tmp); printf("\n"); }
            cutlass::arch::fence_view_async_shared();
            cutlass::arch::NamedBarrier::sync(kNThreads, 2 + 1 - warp_group_idx /*id*/);
            if constexpr (!Mma_dQ_is_RS) {
                if constexpr (!dQ_swapAB) {
                    Tensor tdQrdS = threadMmadQ.partition_fragment_A(sdS);
                    Tensor tdQrK = threadMmadQ.partition_fragment_B(sKt);
                    flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiledMmadQ, tdQrdS, tdQrK, tdQrdQ);
                } else {
                    Tensor tdQrdS = threadMmadQ.partition_fragment_B(sdS);
                    Tensor tdQrK = threadMmadQ.partition_fragment_A(sKt);
                    flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiledMmadQ, tdQrK, tdQrdS, tdQrdQ);
                }
            }
            // warpgroup_wait<0>();
            // Tensor dQ_tmp = make_tensor(tdQrdQ.data(), flash::convert_layout_acc_rowcol(tdQrdQ.layout()));
            // if (cute::thread0()) { print_tensor(dQ_tmp); printf("\n"); }
            // if (blockIdx.x == 0 && threadIdx.x == 128) { print_tensor(dQ_tmp); printf("\n"); }

            if constexpr (!dKV_swapAB) {
                Tensor tdKrdS = threadMmadKV.partition_fragment_A(sdSt);
                Tensor tdKrQ = threadMmadKV.partition_fragment_B(sQt);
                flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(tiledMmadKV, tdKrdS, tdKrQ(_, _, _, smem_pipe_read_q.index()), tdKrdK);
            } else {
                Tensor tdKrdS = threadMmadKV.partition_fragment_B(sdSt);
                Tensor tdKrQ = threadMmadKV.partition_fragment_A(sQt);
                flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(tiledMmadKV, tdKrQ(_, _, _, smem_pipe_read_q.index()), tdKrdS, tdKrdK);
            }
            ++smem_pipe_read_q;
            // Tensor dK_tmp = make_tensor(tdKrdK.data(), flash::convert_layout_acc_rowcol(tdKrdK.layout()));
            // if (cute::thread0()) { print_tensor(dK_tmp); printf("\n"); }

            warpgroup_wait<Mma_dQ_is_RS ? 2 : 1>();
            // if (cute::thread0()) { print(tdQrdQ); printf("\n"); print(tdQgdQaccum); printf("\n"); }
            Tensor tdQrdQ_atomic = recast<float4>(tdQrdQ);
            Tensor tdQgdQaccum_atomic = recast<float4>(tdQgdQaccum(_, _, _, m_block));
            #pragma unroll
            for (int i = 0; i < size(tdQrdQ_atomic); ++i) { atomicAdd(&tdQgdQaccum_atomic(i), tdQrdQ_atomic(i)); }
            // for (int i = 0; i < size(tdQrdQ_atomic); ++i) { tdQgdQaccum_atomic(i) = tdQrdQ_atomic(i); }

            warpgroup_wait<0>();

            pipeline_do.consumer_release(smem_pipe_release_do);  // release V
            ++smem_pipe_release_do;
            pipeline_q.consumer_release(smem_pipe_release_q);  // release V
            ++smem_pipe_release_q;

            int const lane_predicate = cute::elect_one_sync();
            int const warp_idx = cutlass::canonical_warp_idx_sync();
            if (warp_idx == 0 && lane_predicate && m_block >= kStages) {
                pipeline_q.producer_acquire(smem_pipe_write_q);
                copy(tma_load_Q.with(*pipeline_q.producer_get_barrier(smem_pipe_write_q), mcast_mask_qdo), tQgQ(_, _, _, m_block - kStages), tQsQ(_, _, _, smem_pipe_write_q.index()));
                ++smem_pipe_write_q;
                pipeline_do.producer_acquire(smem_pipe_write_do);
                copy(tma_load_dO.with(*pipeline_do.producer_get_barrier(smem_pipe_write_do), mcast_mask_qdo), tdOgdO(_, _, _, m_block - kStages), tdOsdO(_, _, _, smem_pipe_write_do.index()));
                ++smem_pipe_write_do;
            }
        }

    } else {  // SdP_swapAB
        Tensor tdSsdS = smem_thr_copy_PdS.partition_D(sdSt);      // ((Atom,AtomNum),PIPE_M,PIPE_N)

        // Allocate "fragments/descriptors"
        Tensor tSrQ = threadMmaSdP.partition_fragment_B(sQ);
        Tensor tSrK = threadMmaSdP.partition_fragment_A(sK);
        Tensor tdPrdO = threadMmaSdP.partition_fragment_B(sdO);
        Tensor tdPrV = threadMmaSdP.partition_fragment_A(sV);

        Tensor caccS = make_identity_tensor(select<1, 0>(TileShape_MNK{}));    // (BLK_M,BLK_N) -> (blk_m,blk_n)
        Tensor taccScS = threadMmaSdP.partition_C(caccS);                           // (MMA,MMA_N,MMA_N)
        static_assert(decltype(size<0, 0>(taccScS))::value == 2);
        static_assert(decltype(size<0, 1>(taccScS))::value == 2);
        // taccScS has shape ((2, 2, V), MMA_M, MMA_N), we only take only the row indices.
        Tensor taccScS_row = taccScS(make_coord(_, _0{}, _), _0{}, _);
        Tensor lse = make_tensor<ElementAccum>(Shape<Int<decltype(size(taccScS_row))::value>>{});
        Tensor dP_sum = make_fragment_like(lse);
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<1>(taccScS_row(mi));
            lse(mi) = row < params.seqlen_q - m_block * kBlockM ? gLSE(row) : INFINITY;
            dP_sum(mi) = row < params.seqlen_q - m_block * kBlockM ? gdPsum(row) : 0;
        }
        // cute::fill(lse, 1);
        // cute::fill(dP_sum, 1);
        // if (cute::thread0()) { print_tensor(dP_sum); printf("\n"); }
        // We want LSE = inf if the row is OOB. In that case Q would be zero, K would be zero,
        // and scores would be zero. With LSE = 0, probs will be all 1's, and when we multiply
        // with V (which would be zero), we're fine. However, with ALiBi, we might modify these
        // scores, and probs can become NaN. Instead if we set LSE = inf for OOB rows, probs are always 0.

        clear(tdKrdK);
        clear(tdVrdV);

        shared_storage.barrier_K.wait(0);
        shared_storage.barrier_V.wait(0);

        // #pragma unroll 2
        CUTLASS_PRAGMA_NO_UNROLL
        for (; m_block >= 0; --m_block) {
            Tensor tSrS = partition_fragment_C(tiledMmaSdP, select<1, 0>(TileShape_MNK{}));
            pipeline_q.consumer_wait(smem_pipe_read_q);
            flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiledMmaSdP, tSrK, tSrQ(_, _, _, smem_pipe_read_q.index()), tSrS);
            Tensor tdPrdP = partition_fragment_C(tiledMmaSdP, select<1, 0>(TileShape_MNK{}));
            pipeline_do.consumer_wait(smem_pipe_read_do);
            flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiledMmaSdP, tdPrV, tdPrdO(_, _, _, smem_pipe_read_do.index()), tdPrdP);

            warpgroup_wait<1>();
            // Reshape tSrS from ((2, 2, V), MMA_N, MMA_M) to (nrow=(2, V, MMA_M), ncol=(2, MMA_N))
            Tensor scores = make_tensor(tSrS.data(), flash::convert_layout_acc_transposed_rowcol(tSrS.layout()));
            flash::scale_apply_exp2</*Scale=*/true, /*scale_max=*/false>(scores, lse, params.scale_softmax_log2);
            // if (cute::thread0()) { print_tensor(scores); printf("\n"); }

            // Convert scores from fp32 to fp16/bf16
            Tensor rP = flash::convert_type<Element>(tSrS);

            static_assert(!dKV_swapAB);
            Tensor tdVrP = make_tensor(rP.data(), convert_layout_acc_Aregs<typename Ktraits::TiledMmadKV>(tSrS.layout()));
            Tensor tdVrdO = threadMmadKV.partition_fragment_B(sdOt);
            flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(tiledMmadKV, tdVrP, tdVrdO(_, _, _, smem_pipe_read_do.index()), tdVrdV);
            ++smem_pipe_read_do;
            // warpgroup_wait<0>();
            // Tensor dV_tmp = make_tensor(tdVrdV.data(), flash::convert_layout_acc_rowcol(tdVrdV.layout()));
            // if (cute::thread0()) { print_tensor(dV_tmp); printf("\n"); }

            warpgroup_wait<1>();
            // Reshape tdPrdP from ((2, 2, V), MMA_N, MMA_M) to (nrow=(2, V, MMA_M), ncol=(2, MMA_N))
            Tensor dS = make_tensor(tdPrdP.data(), scores.layout());
            #pragma unroll
            for (int mi = 0; mi < size<0>(dS); ++mi) {
                #pragma unroll
                for (int ni = 0; ni < size<1>(dS); ++ni) { dS(mi, ni) = scores(mi, ni) * (dS(mi, ni) - dP_sum(mi)); }
            }
            // if (cute::thread0()) { print_tensor(dS); printf("\n"); }
            Tensor rdS = flash::convert_type<Element>(tdPrdP);

            Tensor tdSadS = smem_thr_copy_PdS.retile_S(rdS);     // ((Atom,AtomNum), MMA_N, MMA_N)
            cute::copy(smem_tiled_copy_PdS, tdSadS, tdSsdS);

            if (m_block > 0) {
                gLSE.data() = gLSE.data() + (-int(kBlockM));
                gdPsum.data() = gdPsum.data() + (-int(kBlockM));
                #pragma unroll
                for (int mi = 0; mi < size(lse); ++mi) {
                    const int row = get<1>(taccScS_row(mi));
                    lse(mi) = gLSE(row);
                    dP_sum(mi) = gdPsum(row);
                }
            }

            Tensor tdKrdS = make_tensor(rdS.data(), convert_layout_acc_Aregs<typename Ktraits::TiledMmadKV>(tdPrdP.layout()));
            Tensor tdKrQ = threadMmadKV.partition_fragment_B(sQt);
            flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(tiledMmadKV, tdKrdS, tdKrQ(_, _, _, smem_pipe_read_q.index()), tdKrdK);
            ++smem_pipe_read_q;
            // warpgroup_wait<0>();
            // Tensor dK_tmp = make_tensor(tdKrdK.data(), flash::convert_layout_acc_rowcol(tdKrdK.layout()));
            // if (cute::thread0()) { print_tensor(dK_tmp); printf("\n"); }

            // SMEM fence to make sure sP is written before it's read by WGMMA
            cutlass::arch::fence_view_async_shared();
            // cutlass::arch::NamedBarrier::sync(kNThreads, 0 /*id*/);
            __syncthreads();
            static_assert(!Mma_dQ_is_RS);
            Tensor tdQrdQ = partition_fragment_C(tiledMmadQ, select<!dQ_swapAB ? 0 : 2, !dQ_swapAB ? 2 : 0>(TileShape_MNK{}));
            if constexpr (!dQ_swapAB) {
                Tensor tdQrdS = threadMmadQ.partition_fragment_A(sdS);
                Tensor tdQrK = threadMmadQ.partition_fragment_B(sKt);
                flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiledMmadQ, tdQrdS, tdQrK, tdQrdQ);
            } else {
                Tensor tdQrdS = threadMmadQ.partition_fragment_B(sdS);
                Tensor tdQrK = threadMmadQ.partition_fragment_A(sKt);
                flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiledMmadQ, tdQrK, tdQrdS, tdQrdQ);
            }
            // warpgroup_wait<0>();
            // Tensor dQ_tmp = make_tensor(tdQrdQ.data(), flash::convert_layout_acc_rowcol(tdQrdQ.layout()));
            // if (cute::thread0()) { print_tensor(dQ_tmp); printf("\n"); }


            warpgroup_wait<0>();
            // if (cute::thread0()) { print(tdQrdQ); printf("\n"); print(tdQgdQaccum); printf("\n"); }
            Tensor tdQrdQ_atomic = recast<float4>(tdQrdQ);
            Tensor tdQgdQaccum_atomic = recast<float4>(tdQgdQaccum(_, _, _, m_block));
            #pragma unroll
            for (int i = 0; i < size(tdQrdQ_atomic); ++i) { atomicAdd(&tdQgdQaccum_atomic(i), tdQrdQ_atomic(i)); }
            // for (int i = 0; i < size(tdQrdQ_atomic); ++i) { tdQgdQaccum_atomic(i) = tdQrdQ_atomic(i); }

            pipeline_do.consumer_release(smem_pipe_release_do);  // release V
            ++smem_pipe_release_do;
            pipeline_q.consumer_release(smem_pipe_release_q);  // release V
            ++smem_pipe_release_q;

            int const lane_predicate = cute::elect_one_sync();
            int const warp_idx = cutlass::canonical_warp_idx_sync();
            if (warp_idx == 0 && lane_predicate && m_block >= kStages) {
                pipeline_q.producer_acquire(smem_pipe_write_q);
                copy(tma_load_Q.with(*pipeline_q.producer_get_barrier(smem_pipe_write_q), mcast_mask_qdo), tQgQ(_, _, _, m_block - kStages), tQsQ(_, _, _, smem_pipe_write_q.index()));
                ++smem_pipe_write_q;
                pipeline_do.producer_acquire(smem_pipe_write_do);
                copy(tma_load_dO.with(*pipeline_do.producer_get_barrier(smem_pipe_write_do), mcast_mask_qdo), tdOgdO(_, _, _, m_block - kStages), tdOsdO(_, _, _, smem_pipe_write_do.index()));
                ++smem_pipe_write_do;
            }
        }
    }

    // Epilogue

    #pragma unroll
    for (int i = 0; i < size(tdKrdK); ++i) { tdKrdK(i) *= params.scale_softmax; }

    Tensor tdKrdK_out = convert_type<Element>(tdKrdK);
    Tensor tdVrdV_out = convert_type<Element>(tdVrdV);

    Tensor sdK = make_tensor(make_smem_ptr(shared_storage.smem_dk.data()), typename Ktraits::SmemLayoutdK{});
    Tensor sdV = make_tensor(make_smem_ptr(shared_storage.smem_dv.data()), typename Ktraits::SmemLayoutdV{});
    Tensor sdKt = make_tensor(make_smem_ptr(shared_storage.smem_dk.data()), typename Ktraits::SmemLayoutdKt{});
    Tensor sdVt = make_tensor(make_smem_ptr(shared_storage.smem_dv.data()), typename Ktraits::SmemLayoutdVt{});

    auto smem_tiled_copy_dKV = make_tiled_copy_C(typename Ktraits::SmemCopyAtomdKV{}, tiledMmadKV);
    auto smem_thr_copy_dKV = smem_tiled_copy_dKV.get_thread_slice(threadIdx.x);
    Tensor taccdKrdK = smem_thr_copy_dKV.retile_S(tdKrdK_out);        // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor taccdVrdV = smem_thr_copy_dKV.retile_S(tdVrdV_out);        // ((Atom,AtomNum), MMA_M, MMA_N)

    __syncthreads();
    if constexpr (!dKV_swapAB) {
        Tensor taccdKsdK = smem_thr_copy_dKV.partition_D(sdK);     // ((Atom,AtomNum),PIPE_M,PIPE_N)
        Tensor taccdVsdV = smem_thr_copy_dKV.partition_D(sdV);     // ((Atom,AtomNum),PIPE_M,PIPE_N)
        cute::copy(smem_tiled_copy_dKV, taccdKrdK, taccdKsdK);
        cute::copy(smem_tiled_copy_dKV, taccdVrdV, taccdVsdV);
    } else {
        Tensor taccdKsdKt = smem_thr_copy_dKV.partition_D(sdKt);     // ((Atom,AtomNum),PIPE_M,PIPE_N)
        Tensor taccdVsdVt = smem_thr_copy_dKV.partition_D(sdVt);     // ((Atom,AtomNum),PIPE_M,PIPE_N)
        cute::copy(smem_tiled_copy_dKV, taccdKrdK, taccdKsdKt);
        cute::copy(smem_tiled_copy_dKV, taccdVrdV, taccdVsdVt);
    }
    cutlass::arch::fence_view_async_shared(); // ensure smem writes are visible to TMA

    Tensor mdK = tma_store_dK.get_tma_tensor(make_shape(params.seqlen_k, params.d, params.h, params.b));
    Tensor mdV = tma_store_dV.get_tma_tensor(make_shape(params.seqlen_k, params.d, params.h, params.b));
    Tensor gdK = local_tile(mdK(_, _, bidh, bidb), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{}));  // (M, K)
    Tensor gdV = local_tile(mdV(_, _, bidh, bidb), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{}));  // (M, K)
    auto block_tma_dK = tma_store_dK.get_slice(_0{});
    auto block_tma_dV = tma_store_dV.get_slice(_0{});
    Tensor tdKgdK = block_tma_dK.partition_D(gdK);  // (TMA, TMA_M, TMA_K)
    Tensor tdKsdK = block_tma_dK.partition_S(sdK); // (TMA, TMA_M, TMA_K)
    Tensor tdVgdV = block_tma_dV.partition_D(gdV);  // (TMA, TMA_M, TMA_K)
    Tensor tdVsdV = block_tma_dV.partition_S(sdV); // (TMA, TMA_M, TMA_K)

    __syncthreads(); // ensure all threads have issued their async fence

    lane_predicate = cute::elect_one_sync();
    warp_idx = cutlass::canonical_warp_idx_sync();
    if (warp_idx == 0 && lane_predicate) {
        cute::copy(tma_store_dV, tdVsdV, tdVgdV);
        cute::copy(tma_store_dK, tdKsdK, tdKgdK);
        tma_store_arrive();
    }
    tma_store_wait<0>();

    // To make sure remote SMEM doesn't get destroyed
    if constexpr (size(ClusterShape{}) > 1) {
        cute::cluster_arrive();
        cute::cluster_wait();
    }

}

template <typename Ktraits, bool Is_causal, typename TiledCopyQ, typename TiledCopydO,
          typename TiledCopyK, typename TiledCopyV, typename TiledCopydQ, typename TiledCopydK, typename TiledCopydV>
__global__ void __launch_bounds__(Ktraits::kNWarps * cutlass::NumThreadsPerWarp, 1)
    compute_dqkv_seqqpar(CUTE_GRID_CONSTANT Flash_bwd_params const params,
                         CUTE_GRID_CONSTANT TiledCopyQ const tma_load_Q,
                         CUTE_GRID_CONSTANT TiledCopydO const tma_load_dO,
                         CUTE_GRID_CONSTANT TiledCopyK const tma_load_K,
                         CUTE_GRID_CONSTANT TiledCopyV const tma_load_V,
                         CUTE_GRID_CONSTANT TiledCopydQ const tma_store_dQ,
                         CUTE_GRID_CONSTANT TiledCopydK const tma_store_dK,
                         CUTE_GRID_CONSTANT TiledCopydV const tma_store_dV) {

    using Element = typename Ktraits::Element;
    using ElementAccum = typename Ktraits::ElementAccum;
    using SoftType = ElementAccum;
    using index_t = typename Ktraits::index_t;
    using TileShape_MNK = typename Ktraits::TileShape_MNK;
    using ClusterShape = typename Ktraits::ClusterShape_MNK;

    static constexpr int kNThreads = Ktraits::kNThreads;
    static constexpr int NumMmaThreads = Ktraits::kNThreads;
    static constexpr int kBlockM = Ktraits::kBlockM;
    static constexpr int kBlockN = Ktraits::kBlockN;
    // constexpr int kHeadDim = Ktraits::kHeadDim;
    static constexpr int kStages = Ktraits::kStages;

    static constexpr bool SdP_swapAB = Ktraits::SdP_swapAB;
    static constexpr bool dKV_swapAB = Ktraits::dKV_swapAB;
    static constexpr bool dQ_swapAB = Ktraits::dQ_swapAB;

    static constexpr bool Mma_dQ_is_RS = Ktraits::Mma_dQ_is_RS;
    if constexpr (dQ_swapAB) { static_assert(!Mma_dQ_is_RS); }

    using MainloopPipeline = typename Ktraits::MainloopPipeline;
    using PipelineParams = typename MainloopPipeline::Params;
    using PipelineState = typename MainloopPipeline::PipelineState;

    extern __shared__ char shared_memory[];
    auto &shared_storage = *reinterpret_cast<typename Ktraits::SharedStorage*>(shared_memory);

    int const m_block = blockIdx.x;
    int const bidb = blockIdx.z;  // The block index for the batch.
    int const bidh = blockIdx.y;  // The block index for the head.

    int lane_predicate = cute::elect_one_sync();
    int warp_idx = cutlass::canonical_warp_idx_sync();

    // Issue Tma Descriptor Prefetch from a single thread
    if (warp_idx == 0 && lane_predicate) {
        cute::prefetch_tma_descriptor(tma_load_Q.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_load_dO.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_load_K.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_load_V.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_store_dQ.get_tma_descriptor());
    }

    Tensor mQ = tma_load_Q.get_tma_tensor(make_shape(params.seqlen_q, params.d, params.h, params.b));
    Tensor mdO = tma_load_dO.get_tma_tensor(make_shape(params.seqlen_q, params.d, params.h, params.b));
    Tensor mK = tma_load_K.get_tma_tensor(make_shape(params.seqlen_k, params.d, params.h, params.b));
    Tensor mV = tma_load_V.get_tma_tensor(make_shape(params.seqlen_k, params.d, params.h, params.b));
    Tensor mLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum*>(params.softmax_lse_ptr)),
                              make_shape(params.b, params.h, params.seqlen_q),
                              make_stride(params.h * params.seqlen_q, params.seqlen_q, _1{}));
    Tensor mdPsum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum*>(params.dsoftmax_sum)),
                                make_shape(params.b, params.h, params.seqlen_q),
                                make_stride(params.h * params.seqlen_q_rounded, params.seqlen_q_rounded, _1{}));
    Tensor mdKaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum*>(params.dk_accum_ptr)),
                                  make_shape(params.seqlen_k, params.d, params.h, params.b),
                                  make_stride(params.d * params.h, _1{}, params.d, params.d * params.h * params.seqlen_k));
    Tensor mdVaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum*>(params.dv_accum_ptr)),
                                  make_shape(params.seqlen_k, params.d, params.h, params.b),
                                  make_stride(params.d * params.h, _1{}, params.d, params.d * params.h * params.seqlen_k));


    Tensor gQ = local_tile(mQ(_, _, bidh, bidb), select<0, 2>(TileShape_MNK{}), make_coord(m_block, _0{}));  // (M, K)
    Tensor gdO = local_tile(mdO(_, _, bidh, bidb), select<0, 2>(TileShape_MNK{}), make_coord(m_block, _0{}));  // (M, K)
    Tensor gK = local_tile(mK(_, _, bidh, bidb), select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (N, K, _)
    Tensor gV = local_tile(mV(_, _, bidh, bidb), select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (N, K, _)
    Tensor gdKaccum = local_tile(mdKaccum(_, _, bidh, bidb), select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (N, K, _)
    Tensor gdVaccum = local_tile(mdVaccum(_, _, bidh, bidb), select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (N, K, _)

    typename Ktraits::GmemTiledCopydQaccum gmem_tiled_copy_dKVaccum;
    auto gmem_thr_copy_dKVaccum = gmem_tiled_copy_dKVaccum.get_thread_slice(threadIdx.x);
    Tensor tdKgdKaccum = gmem_thr_copy_dKVaccum.partition_D(gdKaccum);
    Tensor tdVgdVaccum = gmem_thr_copy_dKVaccum.partition_D(gdVaccum);

    // Construct SMEM tensors.
    Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), typename Ktraits::SmemLayoutQ{});
    Tensor sdO = make_tensor(make_smem_ptr(shared_storage.smem_do.data()), typename Ktraits::SmemLayoutdO{});
    Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), typename Ktraits::SmemLayoutK{});
    Tensor sV = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), typename Ktraits::SmemLayoutV{});
    Tensor sP = make_tensor(make_smem_ptr(shared_storage.smem_p.data()), typename Ktraits::SmemLayoutP{});
    Tensor sdS = make_tensor(make_smem_ptr(shared_storage.smem_ds.data()), typename Ktraits::SmemLayoutdS{});
    Tensor sQt = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), typename Ktraits::SmemLayoutQt{});
    Tensor sdOt = make_tensor(make_smem_ptr(shared_storage.smem_do.data()), typename Ktraits::SmemLayoutdOt{});
    Tensor sKt = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), typename Ktraits::SmemLayoutKt{});
    Tensor sPt = make_tensor(make_smem_ptr(shared_storage.smem_p.data()), typename Ktraits::SmemLayoutPt{});
    Tensor sdSt = make_tensor(make_smem_ptr(shared_storage.smem_ds.data()), typename Ktraits::SmemLayoutdSt{});

    // Prepare the TMA loads
    uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();
    constexpr uint32_t cluster_shape_x = get<0>(ClusterShape());
    uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x, block_rank_in_cluster / cluster_shape_x};
    auto block_tma_Q = tma_load_Q.get_slice(_0{});
    auto block_tma_dO = tma_load_dO.get_slice(_0{});
    auto block_tma_K = tma_load_K.get_slice(cluster_local_block_id.x);
    auto block_tma_V = tma_load_V.get_slice(cluster_local_block_id.x);

    Tensor tQgQ = block_tma_Q.partition_S(gQ);  // (TMA, TMA_M, TMA_K)
    Tensor tQsQ = block_tma_Q.partition_D(sQ);  // (TMA, TMA_M, TMA_K)
    Tensor tdOgdO = block_tma_dO.partition_S(gdO);  // (TMA, TMA_M, TMA_K)
    Tensor tdOsdO = block_tma_dO.partition_D(sdO);  // (TMA, TMA_M, TMA_K)
    Tensor tKgK = block_tma_K.partition_S(gK);  // (TMA, TMA_N, TMA_K, k)
    Tensor tKsK = block_tma_K.partition_D(sK);  // (TMA, TMA_N, TMA_K, PIPE)
    Tensor tVgV = block_tma_V.partition_S(gV);  // (TMA, TMA_N, TMA_K, k)
    Tensor tVsV = block_tma_V.partition_D(sV); // (TMA, TMA_N, TMA_K, PIPE)

    // Set the bytes transferred in this TMA transaction (may involve multiple issues)
    constexpr uint32_t TmaTransactionBytesQ = static_cast<uint32_t>(size<0>(sQ) * size<1>(sQ) * cutlass::sizeof_bits_v<Element> / 8);
    constexpr uint32_t TmaTransactionBytesdO = static_cast<uint32_t>(size<0>(sdO) * size<1>(sdO) * cutlass::sizeof_bits_v<Element> / 8);
    static_assert(TmaTransactionBytesQ == TmaTransactionBytesdO);
    constexpr uint32_t TmaTransactionBytesK = static_cast<uint32_t>(size<0>(sK) * size<1>(sK) * cutlass::sizeof_bits_v<Element> / 8);
    constexpr uint32_t TmaTransactionBytesV = static_cast<uint32_t>(size<0>(sV) * size<1>(sV) * cutlass::sizeof_bits_v<Element> / 8);
    static_assert(TmaTransactionBytesK == TmaTransactionBytesV);

    // Obtain warp index
    int thread_idx = int(threadIdx.x);
    int warp_group_thread_idx = thread_idx % cutlass::NumThreadsPerWarpGroup;
    // int const warp_group_thread_idx = threadIdx.x % cutlass::NumThreadsPerWarpGroup;

    PipelineParams pipeline_params;
    pipeline_params.transaction_bytes = TmaTransactionBytesK;
    pipeline_params.role = MainloopPipeline::ThreadCategory::ProducerConsumer;
    pipeline_params.is_leader = warp_group_thread_idx == 0;
    pipeline_params.num_consumers = NumMmaThreads;

    if (warp_idx == 0 && lane_predicate) {
        shared_storage.barrier_Q.init(1 /*numThreads*/);
        shared_storage.barrier_dO.init(1 /*numThreads*/);
    }
    // cutlass::arch::fence_barrier_init();
    // We're counting on pipeline_k to call fence_barrier_init();
    MainloopPipeline pipeline_k(shared_storage.pipeline_k, pipeline_params, ClusterShape{});
    MainloopPipeline pipeline_v(shared_storage.pipeline_v, pipeline_params, ClusterShape{});

    // We need this to guarantee that the Pipeline init is visible
    // To all producers and consumer blocks in the Cluster
    if constexpr (size(ClusterShape{}) > 1) {
        cute::cluster_arrive_relaxed();
        cute::cluster_wait();
    } else {
        __syncthreads();
    }

    // State variables used for iterating the circular buffer
    // smem_pipe_read / release is used by the consumer of SMEM data - i.e MMA
    // smem_pipe_write is used by the producer of SMEM data - i.e TMA
    PipelineState smem_pipe_read_k, smem_pipe_read_v;
    PipelineState smem_pipe_release_k, smem_pipe_release_v;
    PipelineState smem_pipe_write_k = cutlass::make_producer_start_state<MainloopPipeline>();
    PipelineState smem_pipe_write_v = cutlass::make_producer_start_state<MainloopPipeline>();

    // Copy K tile and V tile from GMEM to SMEM.
    if (warp_idx == 0 && lane_predicate) {
        shared_storage.barrier_Q.arrive_and_expect_tx(TmaTransactionBytesQ);
        copy(tma_load_Q.with(reinterpret_cast<cutlass::arch::ClusterTransactionBarrier::ValueType&>(shared_storage.barrier_Q), 0 /*mcast_mask*/), tQgQ, tQsQ);
        shared_storage.barrier_dO.arrive_and_expect_tx(TmaTransactionBytesdO);
        copy(tma_load_dO.with(reinterpret_cast<cutlass::arch::ClusterTransactionBarrier::ValueType&>(shared_storage.barrier_dO), 0 /*mcast_mask*/), tdOgdO, tdOsdO);
    }
    // if (cute::thread0()) { print_tensor(sQ); printf("\n"); } __syncthreads();

    int n_block = cute::ceil_div(params.seqlen_k, kBlockN) - 1;

    uint16_t mcast_mask_kv = 0;
    if constexpr (cute::is_same_v<typename Ktraits::GmemTiledCopyKV, SM90_TMA_LOAD_MULTICAST>) {
        auto block_layout = Layout<ClusterShape>{}; // (m,n) -> block_id
        for (int m = 0; m < size<0>(block_layout); ++m) {
            mcast_mask_kv |= (uint16_t(1) << block_layout(m, cluster_local_block_id.y, _0{}));
        }
    }
    // Issue TmaLoads (Prologue fetches)
    if (warp_idx == 0 && lane_predicate) {
        // Issue the prologue loads
        CUTLASS_PRAGMA_UNROLL
        for (int stage = 0; stage < kStages && stage <= n_block; ++stage) {
            pipeline_k.producer_acquire(smem_pipe_write_k);
            copy(tma_load_K.with(*pipeline_k.producer_get_barrier(smem_pipe_write_k), mcast_mask_kv), tKgK(_, _, _, n_block - stage), tKsK(_, _, _, stage));
            ++smem_pipe_write_k;
            pipeline_v.producer_acquire(smem_pipe_write_v);
            copy(tma_load_V.with(*pipeline_v.producer_get_barrier(smem_pipe_write_v), mcast_mask_kv), tVgV(_, _, _, n_block - stage), tVsV(_, _, _, stage));
            ++smem_pipe_write_v;
        }
    }

    Tensor gLSE = local_tile(mLSE(bidb, bidh, _), Shape<Int<kBlockM>>{}, make_coord(m_block));
    Tensor gdPsum = local_tile(mdPsum(bidb, bidh, _), Shape<Int<kBlockM>>{}, make_coord(m_block));

    // Initialize matmul objects.
    typename Ktraits::TiledMmaSdP tiledMmaSdP;
    auto threadMmaSdP = tiledMmaSdP.get_thread_slice(threadIdx.x);
    typename Ktraits::TiledMmadKV tiledMmadKV;
    auto threadMmadKV = tiledMmadKV.get_thread_slice(threadIdx.x);
    typename Ktraits::TiledMmadQ tiledMmadQ;
    auto threadMmadQ = tiledMmadQ.get_thread_slice(threadIdx.x);

    // Allocate accumulator
    Tensor tdQrdQ = partition_fragment_C(tiledMmadQ, select<!dQ_swapAB ? 0 : 2, !dQ_swapAB ? 2 : 0>(TileShape_MNK{}));
    clear(tdQrdQ);

    auto smem_tiled_copy_PdS = make_tiled_copy_C(typename Ktraits::SmemCopyAtomPdS{}, tiledMmaSdP);
    auto smem_thr_copy_PdS = smem_tiled_copy_PdS.get_thread_slice(threadIdx.x);

    if constexpr (!SdP_swapAB) {
        Tensor tPsP = smem_thr_copy_PdS.partition_D(sP);      // ((Atom,AtomNum),PIPE_M,PIPE_N)
        Tensor tdSsdS = smem_thr_copy_PdS.partition_D(sdS);      // ((Atom,AtomNum),PIPE_M,PIPE_N)

        // Allocate "fragments/descriptors"
        Tensor tSrQ = threadMmaSdP.partition_fragment_A(sQ);
        Tensor tSrK = threadMmaSdP.partition_fragment_B(sK);
        Tensor tdPrdO = threadMmaSdP.partition_fragment_A(sdO);
        Tensor tdPrV = threadMmaSdP.partition_fragment_B(sV);

        Tensor caccS = make_identity_tensor(select<0, 1>(TileShape_MNK{}));    // (BLK_M,BLK_N) -> (blk_m,blk_n)
        Tensor taccScS = threadMmaSdP.partition_C(caccS);                           // (MMA,MMA_N,MMA_N)
        static_assert(decltype(size<0, 0>(taccScS))::value == 2);
        static_assert(decltype(size<0, 1>(taccScS))::value == 2);
        // taccScS has shape ((2, 2, V), MMA_M, MMA_N), we only take only the row indices.
        Tensor taccScS_row = taccScS(make_coord(_0{}, _, _0{}), _, _0{});
        Tensor lse = make_tensor<ElementAccum>(Shape<Int<decltype(size(taccScS_row))::value>>{});
        Tensor dP_sum = make_fragment_like(lse);
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccScS_row(mi));
            lse(mi) = row < params.seqlen_q - m_block * kBlockM ? gLSE(row) : INFINITY;
            dP_sum(mi) = row < params.seqlen_q - m_block * kBlockM ? gdPsum(row) : 0;
        }
        // if (cute::thread0()) { print_tensor(lse); printf("\n"); }
        // if (cute::thread0()) { print_tensor(dP_sum); printf("\n"); }
        // We want LSE = inf if the row is OOB. In that case Q would be zero, K would be zero,
        // and scores would be zero. With LSE = 0, probs will be all 1's, and when we multiply
        // with V (which would be zero), we're fine. However, with ALiBi, we might modify these
        // scores, and probs can become NaN. Instead if we set LSE = inf for OOB rows, probs are always 0.

        shared_storage.barrier_Q.wait(0);
        shared_storage.barrier_dO.wait(0);

        // #pragma unroll 2
        CUTLASS_PRAGMA_NO_UNROLL
        for (; n_block >= 0; --n_block) {
            // Otherwise we might have WG0 still wating on NamedBarrier but WG1 already
            // started the next iteration and start flipping the same NamedBarrier.
            __syncthreads();
            Tensor tSrS = partition_fragment_C(tiledMmaSdP, select<0, 1>(TileShape_MNK{}));
            pipeline_k.consumer_wait(smem_pipe_read_k);
            flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiledMmaSdP, tSrQ, tSrK(_, _, _, smem_pipe_read_k.index()), tSrS);
            Tensor tdPrdP = partition_fragment_C(tiledMmaSdP, select<0, 1>(TileShape_MNK{}));
            pipeline_v.consumer_wait(smem_pipe_read_v);
            flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiledMmaSdP, tdPrdO, tdPrV(_, _, _, smem_pipe_read_v.index()), tdPrdP);
            ++smem_pipe_read_v;

            warpgroup_wait<1>();
            // Reshape tSrS from ((2, 2, V), MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, V, MMA_N))
            Tensor scores = make_tensor(tSrS.data(), flash::convert_layout_acc_rowcol(tSrS.layout()));
            flash::scale_apply_exp2</*Scale=*/true, /*scale_max=*/false>(scores, lse, params.scale_softmax_log2);
            // if (cute::thread0()) { print_tensor(scores); printf("\n"); }
            // Convert scores from fp32 to fp16/bf16
            Tensor rP = flash::convert_type<Element>(tSrS);
            Tensor tPaP = smem_thr_copy_PdS.retile_S(rP);     // ((Atom,AtomNum), MMA_N, MMA_N)
            cute::copy(smem_tiled_copy_PdS, tPaP, tPsP);
            int const warp_group_idx = cutlass::canonical_warp_group_idx();
            cutlass::arch::NamedBarrier::arrive(kNThreads, warp_group_idx /*id*/);

            warpgroup_wait<0>();
            // Reshape tdPrdP from ((2, 2, V), MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, V, MMA_N))
            Tensor dS = make_tensor(tdPrdP.data(), scores.layout());
            // if (cute::thread0()) { print_tensor(dS); printf("\n"); }
            #pragma unroll
            for (int mi = 0; mi < size<0>(dS); ++mi) {
                #pragma unroll
                for (int ni = 0; ni < size<1>(dS); ++ni) { dS(mi, ni) = scores(mi, ni) * (dS(mi, ni) - dP_sum(mi)); }
            }
            Tensor rdS = flash::convert_type<Element>(tdPrdP);

            Tensor tdSadS = smem_thr_copy_PdS.retile_S(rdS);     // ((Atom,AtomNum), MMA_N, MMA_N)
            cute::copy(smem_tiled_copy_PdS, tdSadS, tdSsdS);
            cutlass::arch::NamedBarrier::arrive(kNThreads, 2 + warp_group_idx /*id*/);
            // if (cute::thread0()) { print_tensor(dS); printf("\n"); }

            if constexpr (Mma_dQ_is_RS) {
                static_assert(!dQ_swapAB);
                Tensor tdQrdS = make_tensor(rdS.data(), convert_layout_acc_Aregs<typename Ktraits::TiledMmadQ>(tdPrdP.layout()));
                Tensor tdQrK = threadMmadQ.partition_fragment_B(sKt);
                flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(tiledMmadQ, tdQrdS, tdQrK(_, _, _, smem_pipe_read_k.index()), tdQrdQ);
                // if (cute::thread0()) { print(tdQrdS); printf("\n"); print(tdQrK); printf("\n"); print(tdQrdQ); printf("\n"); }
            }

            // warpgroup_wait<0>();
            // Tensor dQ_tmp = make_tensor(tdQrdQ.data(), flash::convert_layout_acc_rowcol(tdQrdQ.layout()));
            // if (cute::thread0()) { print_tensor(dQ_tmp); printf("\n"); }
            // if (cute::thread0()) { print_tensor(sK); printf("\n"); }
            // if (cute::thread0()) { print_tensor(sKt); printf("\n"); } __syncthreads();

            // if (cute::thread0()) { printf("before barrier sync 0\n"); }
            // SMEM fence to make sure sP is written before it's read by WGMMA
            cutlass::arch::fence_view_async_shared();
            cutlass::arch::NamedBarrier::sync(kNThreads, 1 - warp_group_idx /*id*/);
            // if (cute::thread0()) { printf("after barrier sync 0\n"); }
            Tensor tdVrdV = partition_fragment_C(tiledMmadKV, select<!dKV_swapAB ? 1 : 2, !dKV_swapAB? 2 : 1>(TileShape_MNK{}));

            if constexpr (!dKV_swapAB) {
                Tensor tdVrP = threadMmadKV.partition_fragment_A(sPt);
                Tensor tdVrdO = threadMmadKV.partition_fragment_B(sdOt);
                flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiledMmadKV, tdVrP, tdVrdO, tdVrdV);
            } else {
                Tensor tdVrP = threadMmadKV.partition_fragment_B(sPt);
                Tensor tdVrdO = threadMmadKV.partition_fragment_A(sdOt);
                flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiledMmadKV, tdVrdO, tdVrP, tdVrdV);
            }

            // warpgroup_wait<0>();
            // Tensor dV_tmp = make_tensor(tdVrdV.data(), flash::convert_layout_acc_rowcol(tdVrdV.layout()));
            // if (cute::thread0()) { print_tensor(dV_tmp); printf("\n"); }
            cutlass::arch::fence_view_async_shared();
            cutlass::arch::NamedBarrier::sync(kNThreads, 2 + 1 - warp_group_idx /*id*/);
            if constexpr (!Mma_dQ_is_RS) {
                if constexpr (!dQ_swapAB) {
                    Tensor tdQrdS = threadMmadQ.partition_fragment_A(sdS);
                    Tensor tdQrK = threadMmadQ.partition_fragment_B(sKt);
                    flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(tiledMmadQ, tdQrdS, tdQrK(_, _, _, smem_pipe_read_k.index()), tdQrdQ);
                } else {
                    Tensor tdQrdS = threadMmadQ.partition_fragment_B(sdS);
                    Tensor tdQrK = threadMmadQ.partition_fragment_A(sKt);
                    flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(tiledMmadQ, tdQrK(_, _, _, smem_pipe_read_k.index()), tdQrdS, tdQrdQ);
                }
            }
            ++smem_pipe_read_k;
            // warpgroup_wait<0>();
            // Tensor dQ_tmp = make_tensor(tdQrdQ.data(), flash::convert_layout_acc_rowcol(tdQrdQ.layout()));
            // if (cute::thread0()) { print_tensor(dQ_tmp); printf("\n"); }
            // if (blockIdx.x == 0 && threadIdx.x == 128) { print_tensor(dQ_tmp); printf("\n"); }

            Tensor tdKrdK = partition_fragment_C(tiledMmadKV, select<!dKV_swapAB ? 1 : 2, !dKV_swapAB? 2 : 1>(TileShape_MNK{}));
            if constexpr (!dKV_swapAB) {
                Tensor tdKrdS = threadMmadKV.partition_fragment_A(sdSt);
                Tensor tdKrQ = threadMmadKV.partition_fragment_B(sQt);
                flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiledMmadKV, tdKrdS, tdKrQ, tdKrdK);
            } else {
                Tensor tdKrdS = threadMmadKV.partition_fragment_B(sdSt);
                Tensor tdKrQ = threadMmadKV.partition_fragment_A(sQt);
                flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiledMmadKV, tdKrQ, tdKrdS, tdKrdK);
            }
            // warpgroup_wait<0>();
            // Tensor dK_tmp = make_tensor(tdKrdK.data(), flash::convert_layout_acc_rowcol(tdKrdK.layout()));
            // if (cute::thread0()) { print_tensor(dK_tmp); printf("\n"); }

            warpgroup_wait<Mma_dQ_is_RS ? 1 : 2>();
            // if (cute::thread0()) { print(tdQrdQ); printf("\n"); print(tdQgdQaccum); printf("\n"); }
            Tensor tdVrdV_atomic = recast<float4>(tdVrdV);
            Tensor tdVgdVaccum_atomic = recast<float4>(tdVgdVaccum(_, _, _, n_block));
            #pragma unroll
            for (int i = 0; i < size(tdVrdV_atomic); ++i) { atomicAdd(&tdVgdVaccum_atomic(i), tdVrdV_atomic(i)); }
            // for (int i = 0; i < size(tdVrdV_atomic); ++i) { tdVgdVaccum_atomic(i) = tdVrdV_atomic(i); }

            warpgroup_wait<0>();
            Tensor tdKrdK_atomic = recast<float4>(tdKrdK);
            Tensor tdKgdKaccum_atomic = recast<float4>(tdKgdKaccum(_, _, _, n_block));
            #pragma unroll
            for (int i = 0; i < size(tdKrdK_atomic); ++i) { atomicAdd(&tdKgdKaccum_atomic(i), tdKrdK_atomic(i)); }

            pipeline_v.consumer_release(smem_pipe_release_v);  // release V
            ++smem_pipe_release_v;
            pipeline_k.consumer_release(smem_pipe_release_k);  // release V
            ++smem_pipe_release_k;

            int const lane_predicate = cute::elect_one_sync();
            int const warp_idx = cutlass::canonical_warp_idx_sync();
            if (warp_idx == 0 && lane_predicate && n_block >= kStages) {
                pipeline_k.producer_acquire(smem_pipe_write_k);
                copy(tma_load_K.with(*pipeline_k.producer_get_barrier(smem_pipe_write_k), mcast_mask_kv), tKgK(_, _, _, n_block - kStages), tKsK(_, _, _, smem_pipe_write_k.index()));
                ++smem_pipe_write_k;
                pipeline_v.producer_acquire(smem_pipe_write_v);
                copy(tma_load_V.with(*pipeline_v.producer_get_barrier(smem_pipe_write_v), mcast_mask_kv), tVgV(_, _, _, n_block - kStages), tVsV(_, _, _, smem_pipe_write_v.index()));
                ++smem_pipe_write_v;
            }
        }

    } else {  // SdP_swapAB
        Tensor tdSsdS = smem_thr_copy_PdS.partition_D(sdSt);      // ((Atom,AtomNum),PIPE_M,PIPE_N)

        // Allocate "fragments/descriptors"
        Tensor tSrQ = threadMmaSdP.partition_fragment_B(sQ);
        Tensor tSrK = threadMmaSdP.partition_fragment_A(sK);
        Tensor tdPrdO = threadMmaSdP.partition_fragment_B(sdO);
        Tensor tdPrV = threadMmaSdP.partition_fragment_A(sV);

        Tensor caccS = make_identity_tensor(select<1, 0>(TileShape_MNK{}));    // (BLK_M,BLK_N) -> (blk_m,blk_n)
        Tensor taccScS = threadMmaSdP.partition_C(caccS);                           // (MMA,MMA_N,MMA_N)
        static_assert(decltype(size<0, 0>(taccScS))::value == 2);
        static_assert(decltype(size<0, 1>(taccScS))::value == 2);
        // taccScS has shape ((2, 2, V), MMA_M, MMA_N), we only take only the row indices.
        Tensor taccScS_row = taccScS(make_coord(_, _0{}, _), _0{}, _);
        Tensor lse = make_tensor<ElementAccum>(Shape<Int<decltype(size(taccScS_row))::value>>{});
        Tensor dP_sum = make_fragment_like(lse);
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<1>(taccScS_row(mi));
            lse(mi) = row < params.seqlen_q - m_block * kBlockM ? gLSE(row) : INFINITY;
            dP_sum(mi) = row < params.seqlen_q - m_block * kBlockM ? gdPsum(row) : 0;
        }
        // if (cute::thread0()) { print_tensor(taccScS_row); printf("\n"); }
        // cute::fill(lse, 1);
        // cute::fill(dP_sum, 1);
        // if (cute::thread0()) { print_tensor(dP_sum); printf("\n"); }
        // We want LSE = inf if the row is OOB. In that case Q would be zero, K would be zero,
        // and scores would be zero. With LSE = 0, probs will be all 1's, and when we multiply
        // with V (which would be zero), we're fine. However, with ALiBi, we might modify these
        // scores, and probs can become NaN. Instead if we set LSE = inf for OOB rows, probs are always 0.

        clear(tdQrdQ);

        shared_storage.barrier_Q.wait(0);
        shared_storage.barrier_dO.wait(0);

        // #pragma unroll 2
        CUTLASS_PRAGMA_NO_UNROLL
        for (; n_block >= 0; --n_block) {
            Tensor tSrS = partition_fragment_C(tiledMmaSdP, select<1, 0>(TileShape_MNK{}));
            pipeline_k.consumer_wait(smem_pipe_read_k);
            flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiledMmaSdP, tSrK(_, _, _, smem_pipe_read_k.index()), tSrQ, tSrS);
            Tensor tdPrdP = partition_fragment_C(tiledMmaSdP, select<1, 0>(TileShape_MNK{}));
            pipeline_v.consumer_wait(smem_pipe_read_v);
            flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiledMmaSdP, tdPrV(_, _, _, smem_pipe_read_v.index()), tdPrdO, tdPrdP);
            ++smem_pipe_read_v;

            warpgroup_wait<1>();
            // Reshape tSrS from ((2, 2, V), MMA_N, MMA_M) to (nrow=(2, V, MMA_M), ncol=(2, MMA_N))
            Tensor scores = make_tensor(tSrS.data(), flash::convert_layout_acc_transposed_rowcol(tSrS.layout()));
            // if (cute::thread0()) { print_tensor(lse); printf("\n"); }
            flash::scale_apply_exp2</*Scale=*/true, /*scale_max=*/false>(scores, lse, params.scale_softmax_log2);
            // if (cute::thread0()) { print_tensor(scores); printf("\n"); }

            // Convert scores from fp32 to fp16/bf16
            Tensor rP = flash::convert_type<Element>(tSrS);

            static_assert(!dKV_swapAB);
            Tensor tdVrdV = partition_fragment_C(tiledMmadKV, select<1, 2>(TileShape_MNK{}));
            Tensor tdVrP = make_tensor(rP.data(), convert_layout_acc_Aregs<typename Ktraits::TiledMmadKV>(tSrS.layout()));
            Tensor tdVrdO = threadMmadKV.partition_fragment_B(sdOt);
            flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiledMmadKV, tdVrP, tdVrdO, tdVrdV);
            // warpgroup_wait<0>();
            // Tensor dV_tmp = make_tensor(tdVrdV.data(), flash::convert_layout_acc_rowcol(tdVrdV.layout()));
            // if (cute::thread0()) { print_tensor(dV_tmp); printf("\n"); }

            warpgroup_wait<1>();
            // Reshape tdPrdP from ((2, 2, V), MMA_N, MMA_M) to (nrow=(2, V, MMA_M), ncol=(2, MMA_N))
            Tensor dS = make_tensor(tdPrdP.data(), scores.layout());
            #pragma unroll
            for (int mi = 0; mi < size<0>(dS); ++mi) {
                #pragma unroll
                for (int ni = 0; ni < size<1>(dS); ++ni) { dS(mi, ni) = scores(mi, ni) * (dS(mi, ni) - dP_sum(mi)); }
            }
            // if (cute::thread0()) { print_tensor(dS); printf("\n"); }
            Tensor rdS = flash::convert_type<Element>(tdPrdP);

            Tensor tdSadS = smem_thr_copy_PdS.retile_S(rdS);     // ((Atom,AtomNum), MMA_N, MMA_N)
            cute::copy(smem_tiled_copy_PdS, tdSadS, tdSsdS);

            Tensor tdKrdK = partition_fragment_C(tiledMmadKV, select<1, 2>(TileShape_MNK{}));
            Tensor tdKrdS = make_tensor(rdS.data(), convert_layout_acc_Aregs<typename Ktraits::TiledMmadKV>(tdPrdP.layout()));
            Tensor tdKrQ = threadMmadKV.partition_fragment_B(sQt);
            flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiledMmadKV, tdKrdS, tdKrQ, tdKrdK);
            // warpgroup_wait<0>();
            // Tensor dK_tmp = make_tensor(tdKrdK.data(), flash::convert_layout_acc_rowcol(tdKrdK.layout()));
            // if (cute::thread0()) { print_tensor(dK_tmp); printf("\n"); }

            warpgroup_wait<1>();
            // if (cute::thread0()) { print(tdQrdQ); printf("\n"); print(tdQgdQaccum); printf("\n"); }
            Tensor tdVrdV_atomic = recast<float4>(tdVrdV);
            Tensor tdVgdVaccum_atomic = recast<float4>(tdVgdVaccum(_, _, _, n_block));
            #pragma unroll
            for (int i = 0; i < size(tdVrdV_atomic); ++i) { atomicAdd(&tdVgdVaccum_atomic(i), tdVrdV_atomic(i)); }
            // for (int i = 0; i < size(tdVrdV_atomic); ++i) { tdVgdVaccum_atomic(i) = tdVrdV_atomic(i); }

            // SMEM fence to make sure sP is written before it's read by WGMMA
            cutlass::arch::fence_view_async_shared();
            // cutlass::arch::NamedBarrier::sync(kNThreads, 0 /*id*/);
            __syncthreads();
            static_assert(!Mma_dQ_is_RS);
            if constexpr (!dQ_swapAB) {
                Tensor tdQrdS = threadMmadQ.partition_fragment_A(sdS);
                Tensor tdQrK = threadMmadQ.partition_fragment_B(sKt);
                flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(tiledMmadQ, tdQrdS, tdQrK(_, _, _, smem_pipe_read_k.index()), tdQrdQ);
            } else {
                Tensor tdQrdS = threadMmadQ.partition_fragment_B(sdS);
                Tensor tdQrK = threadMmadQ.partition_fragment_A(sKt);
                flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(tiledMmadQ, tdQrK(_, _, _, smem_pipe_read_k.index()), tdQrdS, tdQrdQ);
            }
            ++smem_pipe_read_k;
            // warpgroup_wait<0>();
            // Tensor dQ_tmp = make_tensor(tdQrdQ.data(), flash::convert_layout_acc_rowcol(tdQrdQ.layout()));
            // if (cute::thread0()) { print_tensor(dQ_tmp); printf("\n"); }

            warpgroup_wait<1>();
            // if (cute::thread0()) { print(tdQrdQ); printf("\n"); print(tdQgdQaccum); printf("\n"); }
            Tensor tdKrdK_atomic = recast<float4>(tdKrdK);
            Tensor tdKgdKaccum_atomic = recast<float4>(tdKgdKaccum(_, _, _, n_block));
            #pragma unroll
            for (int i = 0; i < size(tdKrdK_atomic); ++i) { atomicAdd(&tdKgdKaccum_atomic(i), tdKrdK_atomic(i)); }
            // for (int i = 0; i < size(tdVrdV_atomic); ++i) { tdVgdVaccum_atomic(i) = tdVrdV_atomic(i); }

            warpgroup_wait<0>();

            pipeline_v.consumer_release(smem_pipe_release_v);  // release V
            ++smem_pipe_release_v;
            pipeline_k.consumer_release(smem_pipe_release_k);  // release V
            ++smem_pipe_release_k;

            int const lane_predicate = cute::elect_one_sync();
            int const warp_idx = cutlass::canonical_warp_idx_sync();
            if (warp_idx == 0 && lane_predicate && n_block >= kStages) {
                pipeline_k.producer_acquire(smem_pipe_write_k);
                copy(tma_load_K.with(*pipeline_k.producer_get_barrier(smem_pipe_write_k), mcast_mask_kv), tKgK(_, _, _, n_block - kStages), tKsK(_, _, _, smem_pipe_write_k.index()));
                ++smem_pipe_write_k;
                pipeline_v.producer_acquire(smem_pipe_write_v);
                copy(tma_load_V.with(*pipeline_v.producer_get_barrier(smem_pipe_write_v), mcast_mask_kv), tVgV(_, _, _, n_block - kStages), tVsV(_, _, _, smem_pipe_write_v.index()));
                ++smem_pipe_write_v;
            }
        }
    }

    // Epilogue

    #pragma unroll
    for (int i = 0; i < size(tdQrdQ); ++i) { tdQrdQ(i) *= params.scale_softmax; }
    // if (cute::thread0()) { print_tensor(tdQrdQ); }

    Tensor tdQrdQ_out = convert_type<Element>(tdQrdQ);

    Tensor sdQ = make_tensor(make_smem_ptr(shared_storage.smem_dq.data()), typename Ktraits::SmemLayoutdQ{});
    Tensor sdQt = make_tensor(make_smem_ptr(shared_storage.smem_dq.data()), typename Ktraits::SmemLayoutdQt{});

    auto smem_tiled_copy_dQ = make_tiled_copy_C(typename Ktraits::SmemCopyAtomdQ{}, tiledMmadQ);
    auto smem_thr_copy_dQ = smem_tiled_copy_dQ.get_thread_slice(threadIdx.x);
    Tensor taccdQrdQ = smem_thr_copy_dQ.retile_S(tdQrdQ_out);        // ((Atom,AtomNum), MMA_M, MMA_N)

    __syncthreads();
    if constexpr (!dQ_swapAB) {
        Tensor taccdQsdQ = smem_thr_copy_dQ.partition_D(sdQ);     // ((Atom,AtomNum),PIPE_M,PIPE_N)
        cute::copy(smem_tiled_copy_dQ, taccdQrdQ, taccdQsdQ);
    } else {
        Tensor taccdQsdQt = smem_thr_copy_dQ.partition_D(sdQt);     // ((Atom,AtomNum),PIPE_M,PIPE_N)
        cute::copy(smem_tiled_copy_dQ, taccdQrdQ, taccdQsdQt);
    }
    cutlass::arch::fence_view_async_shared(); // ensure smem writes are visible to TMA

    Tensor mdQ = tma_store_dQ.get_tma_tensor(make_shape(params.seqlen_q, params.d, params.h, params.b));
    Tensor gdQ = local_tile(mdQ(_, _, bidh, bidb), select<0, 2>(TileShape_MNK{}), make_coord(m_block, _0{}));  // (M, K)
    auto block_tma_dQ = tma_store_dQ.get_slice(_0{});
    Tensor tdQgdQ = block_tma_dQ.partition_D(gdQ);  // (TMA, TMA_M, TMA_K)
    Tensor tdQsdQ = block_tma_dQ.partition_S(sdQ); // (TMA, TMA_M, TMA_K)

    __syncthreads(); // ensure all threads have issued their async fence
    // if (cute::thread0()) { print_tensor(sdQ); }

    lane_predicate = cute::elect_one_sync();
    warp_idx = cutlass::canonical_warp_idx_sync();
    if (warp_idx == 0 && lane_predicate) {
        cute::copy(tma_store_dQ, tdQsdQ, tdQgdQ);
        tma_store_arrive();
    }
    tma_store_wait<0>();

    // To make sure remote SMEM doesn't get destroyed
    if constexpr (size(ClusterShape{}) > 1) {
        cute::cluster_arrive();
        cute::cluster_wait();
    }

}

template <typename Ktraits, bool Is_causal, typename TiledCopyQ, typename TiledCopydO,
        typename TiledCopyK, typename TiledCopyV, typename TiledCopydK, typename TiledCopydV, typename TiledCopydQ, typename TiledCopyAdddQ>
__global__ void __launch_bounds__(Ktraits::kNWarps * cutlass::NumThreadsPerWarp, 1)
    compute_dqkv_ws(CUTE_GRID_CONSTANT Flash_bwd_params const params,
                    CUTE_GRID_CONSTANT TiledCopyQ const tma_load_Q,
                    CUTE_GRID_CONSTANT TiledCopydO const tma_load_dO,
                    CUTE_GRID_CONSTANT TiledCopyK const tma_load_K,
                    CUTE_GRID_CONSTANT TiledCopyV const tma_load_V,
                    CUTE_GRID_CONSTANT TiledCopydK const tma_store_dK,
                    CUTE_GRID_CONSTANT TiledCopydV const tma_store_dV,
                    CUTE_GRID_CONSTANT TiledCopydQ const tma_store_dQ,
                    CUTE_GRID_CONSTANT TiledCopyAdddQ const tma_reduce_add_dQ) {

    using Element = typename Ktraits::Element;
    using ElementAccum = typename Ktraits::ElementAccum;
    using SoftType = ElementAccum;
    using index_t = typename Ktraits::index_t;
    using TileShape_MNK = typename Ktraits::TileShape_MNK;
    using ClusterShape = typename Ktraits::ClusterShape_MNK;

    static_assert(Ktraits::Is_WS);

    // static constexpr int kNThreads = Ktraits::kNThreads;
    // static constexpr int NumMmaThreads = size(typename Ktraits::TiledMmaSdP{});
    static constexpr int NumMmaThreads = 256;
    static constexpr int NumCopyThreads = cutlass::NumThreadsPerWarpGroup;
    static constexpr int kBlockM = Ktraits::kBlockM;
    static constexpr int kNThreadsdQ = Ktraits::kNThreadsdQ;
    // static constexpr int kBlockN = Ktraits::kBlockN;
    // constexpr int kHeadDim = Ktraits::kHeadDim;
    // static constexpr int kStages = Ktraits::kStages;

    static constexpr bool SdP_swapAB = Ktraits::SdP_swapAB;
    static constexpr bool dKV_swapAB = Ktraits::dKV_swapAB;
    static constexpr bool dQ_swapAB = Ktraits::dQ_swapAB;

    if constexpr (SdP_swapAB) { static_assert(!dKV_swapAB); }

    static constexpr bool Mma_dQ_is_RS = Ktraits::Mma_dQ_is_RS;
    if constexpr (dQ_swapAB) { static_assert(!Mma_dQ_is_RS); }

    using MainloopPipeline = typename Ktraits::MainloopPipeline;
    using PipelineParams = typename MainloopPipeline::Params;
    using PipelineState = typename MainloopPipeline::PipelineState;

    extern __shared__ char shared_memory[];
    auto &shared_storage = *reinterpret_cast<typename Ktraits::SharedStorage*>(shared_memory);

    int lane_predicate = cute::elect_one_sync();
    int warp_idx = cutlass::canonical_warp_idx_sync();

    // Issue Tma Descriptor Prefetch from a single thread
    if (warp_idx == 0 && lane_predicate) {
        cute::prefetch_tma_descriptor(tma_load_Q.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_load_dO.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_load_K.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_load_V.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_store_dK.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_store_dV.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_store_dQ.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_reduce_add_dQ.get_tma_descriptor());
    }

    // Construct SMEM tensors.
    Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), typename Ktraits::SmemLayoutQ{});
    Tensor sdO = make_tensor(make_smem_ptr(shared_storage.smem_do.data()), typename Ktraits::SmemLayoutdO{});
    Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), typename Ktraits::SmemLayoutK{});
    Tensor sV = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), typename Ktraits::SmemLayoutV{});
    Tensor sP = make_tensor(make_smem_ptr(shared_storage.smem_p.data()), typename Ktraits::SmemLayoutP{});
    Tensor sdS = make_tensor(make_smem_ptr(shared_storage.smem_ds.data()), typename Ktraits::SmemLayoutdS{});
    Tensor sQt = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), typename Ktraits::SmemLayoutQt{});
    Tensor sdOt = make_tensor(make_smem_ptr(shared_storage.smem_do.data()), typename Ktraits::SmemLayoutdOt{});
    Tensor sKt = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), typename Ktraits::SmemLayoutKt{});
    Tensor sPt = make_tensor(make_smem_ptr(shared_storage.smem_p.data()), typename Ktraits::SmemLayoutPt{});
    Tensor sdSt = make_tensor(make_smem_ptr(shared_storage.smem_ds.data()), typename Ktraits::SmemLayoutdSt{});
    Tensor sdQ = make_tensor(make_smem_ptr(shared_storage.smem_dqacc.data()), typename Ktraits::SmemLayoutdQacc{});
    Tensor sdQ2 = make_tensor(make_smem_ptr(shared_storage.smem_dqacc.data()), typename Ktraits::SmemLayoutdQacc2{});
    Tensor sdQt = make_tensor(make_smem_ptr(shared_storage.smem_dqacc.data()), typename Ktraits::SmemLayoutdQacct{});

    // Set the bytes transferred in this TMA transaction (may involve multiple issues)
    constexpr uint32_t TmaTransactionBytesQ = static_cast<uint32_t>(size<0>(sQ) * size<1>(sQ) * cutlass::sizeof_bits_v<Element> / 8);
    constexpr uint32_t TmaTransactionBytesdO = static_cast<uint32_t>(size<0>(sdO) * size<1>(sdO) * cutlass::sizeof_bits_v<Element> / 8);
    static_assert(TmaTransactionBytesQ == TmaTransactionBytesdO);
    constexpr uint32_t TmaTransactionBytesK = static_cast<uint32_t>(size<0>(sK) * size<1>(sK) * cutlass::sizeof_bits_v<Element> / 8);
    constexpr uint32_t TmaTransactionBytesV = static_cast<uint32_t>(size<0>(sV) * size<1>(sV) * cutlass::sizeof_bits_v<Element> / 8);
    static_assert(TmaTransactionBytesK == TmaTransactionBytesV);

    // Obtain warp index
    int thread_idx = int(threadIdx.x);
    int warp_group_thread_idx = thread_idx % cutlass::NumThreadsPerWarpGroup;
    // int const warp_group_thread_idx = threadIdx.x % cutlass::NumThreadsPerWarpGroup;

    PipelineParams pipeline_params;
    pipeline_params.transaction_bytes = TmaTransactionBytesQ;
    int warp_group_idx = cutlass::canonical_warp_group_idx();
    if (warp_group_idx == 0) {
        pipeline_params.role = MainloopPipeline::ThreadCategory::Producer;
    } else {
        pipeline_params.role = MainloopPipeline::ThreadCategory::Consumer;
    }
    pipeline_params.is_leader = warp_group_thread_idx == 0;
    pipeline_params.num_consumers = NumMmaThreads;

    if (warp_idx == 0 && lane_predicate) {
        shared_storage.barrier_K.init(1 /*numThreads*/);
        shared_storage.barrier_V.init(1 /*numThreads*/);
    }
    // cutlass::arch::fence_barrier_init();
    // We're counting on pipeline_q to call fence_barrier_init();
    MainloopPipeline pipeline_q(shared_storage.pipeline_q, pipeline_params, ClusterShape{});
    MainloopPipeline pipeline_do(shared_storage.pipeline_do, pipeline_params, ClusterShape{});

    // We need this to guarantee that the Pipeline init is visible
    // To all producers and consumer blocks in the Cluster
    if constexpr (size(ClusterShape{}) > 1) {
        cute::cluster_arrive_relaxed();
        cute::cluster_wait();
    } else {
        __syncthreads();
    }

    if (warp_group_idx == 0) {  // Producer
        // method in cutlass/arch/reg_reconfig.h
        // calls setmaxnreg.dec.sync.aligned.u32
        cutlass::arch::warpgroup_reg_dealloc<24>();

        int const n_block = blockIdx.x;
        int const bidb = blockIdx.z;  // The block index for the batch.
        int const bidh = blockIdx.y;  // The block index for the head.

        int m_block = cute::ceil_div(params.seqlen_q, kBlockM) - 1;

        int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
        int lane_predicate = cute::elect_one_sync();
        // if (warp_idx_in_warpgroup == 0 && lane_predicate) {
        if (warp_idx_in_warpgroup == 0) {  // Load K, and do TMA on Q and dO
            Tensor mQ = tma_load_Q.get_tma_tensor(make_shape(params.seqlen_q, params.d, params.h, params.b));
            Tensor mdO = tma_load_dO.get_tma_tensor(make_shape(params.seqlen_q, params.d, params.h, params.b));
            Tensor mK = tma_load_K.get_tma_tensor(make_shape(params.seqlen_k, params.d, params.h, params.b));
            Tensor gQ = local_tile(mQ(_, _, bidh, bidb), select<0, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (M, K, _)
            Tensor gdO = local_tile(mdO(_, _, bidh, bidb), select<0, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (M, K, _)
            Tensor gK = local_tile(mK(_, _, bidh, bidb), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{}));  // (N, K)
            // Prepare the TMA loads
            uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();
            constexpr uint32_t cluster_shape_x = get<0>(ClusterShape());
            uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x, block_rank_in_cluster / cluster_shape_x};
            auto block_tma_Q = tma_load_Q.get_slice(cluster_local_block_id.y);
            auto block_tma_dO = tma_load_dO.get_slice(cluster_local_block_id.y);
            auto block_tma_K = tma_load_K.get_slice(_0{});
            Tensor tQgQ = block_tma_Q.partition_S(gQ);  // (TMA, TMA_M, TMA_K, k)
            Tensor tQsQ = block_tma_Q.partition_D(sQ);  // (TMA, TMA_M, TMA_K, PIPE)
            Tensor tdOgdO = block_tma_dO.partition_S(gdO);  // (TMA, TMA_M, TMA_K, k)
            Tensor tdOsdO = block_tma_dO.partition_D(sdO);  // (TMA, TMA_M, TMA_K, PIPE)
            Tensor tKgK = block_tma_K.partition_S(gK);  // (TMA, TMA_N, TMA_K)
            Tensor tKsK = block_tma_K.partition_D(sK);  // (TMA, TMA_N, TMA_K)

            PipelineState smem_pipe_write_q = cutlass::make_producer_start_state<MainloopPipeline>();
            PipelineState smem_pipe_write_do = cutlass::make_producer_start_state<MainloopPipeline>();

            uint16_t mcast_mask_qdo = 0;
            if constexpr (cute::is_same_v<typename Ktraits::GmemTiledCopyQdO, SM90_TMA_LOAD_MULTICAST>) {
                auto block_layout = Layout<ClusterShape>{}; // (m,n) -> block_id
                for (int n = 0; n < size<1>(block_layout); ++n) {
                    mcast_mask_qdo |= (uint16_t(1) << block_layout(n, cluster_local_block_id.x, _0{}));
                }
            }

            if (lane_predicate) {
                // Copy K tile and V tile from GMEM to SMEM.
                shared_storage.barrier_K.arrive_and_expect_tx(TmaTransactionBytesK);
                copy(tma_load_K.with(reinterpret_cast<cutlass::arch::ClusterTransactionBarrier::ValueType&>(shared_storage.barrier_K), 0 /*mcast_mask*/), tKgK, tKsK);

                #pragma unroll 2
                for (; m_block >= 0; --m_block) {
                    pipeline_q.producer_acquire(smem_pipe_write_q);
                    copy(tma_load_Q.with(*pipeline_q.producer_get_barrier(smem_pipe_write_q), mcast_mask_qdo), tQgQ(_, _, _, m_block), tQsQ(_, _, _, smem_pipe_write_q.index()));
                    ++smem_pipe_write_q;
                    pipeline_do.producer_acquire(smem_pipe_write_do);
                    copy(tma_load_dO.with(*pipeline_do.producer_get_barrier(smem_pipe_write_do), mcast_mask_qdo), tdOgdO(_, _, _, m_block), tdOsdO(_, _, _, smem_pipe_write_do.index()));
                    ++smem_pipe_write_do;
                }

                // Tail loop
                /* This helps avoid early exit of blocks in Cluster
                * Waits for all stages to either be released (all
                * Consumer UNLOCKs), or if the stage was never used
                * then would just be acquired since the phase was
                * still inverted from make_producer_start_state
                */
                pipeline_q.producer_tail(smem_pipe_write_q);
                pipeline_do.producer_tail(smem_pipe_write_do);
            }
        } else if (warp_idx_in_warpgroup == 1) {  // Load V, and do TMA_REDUCE_ADD on dQ
            Tensor mV = tma_load_V.get_tma_tensor(make_shape(params.seqlen_k, params.d, params.h, params.b));
            Tensor gV = local_tile(mV(_, _, bidh, bidb), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{}));  // (N, K)
            auto block_tma_V = tma_load_V.get_slice(_0{});
            Tensor tVgV = block_tma_V.partition_S(gV);  // (TMA, TMA_N, TMA_K)
            Tensor tVsV = block_tma_V.partition_D(sV); // (TMA, TMA_N, TMA_K)
            if (lane_predicate) {
                shared_storage.barrier_V.arrive_and_expect_tx(TmaTransactionBytesV);
                copy(tma_load_V.with(reinterpret_cast<cutlass::arch::ClusterTransactionBarrier::ValueType&>(shared_storage.barrier_V), 0 /*mcast_mask*/), tVgV, tVsV);
            }

            Tensor mdQaccum = tma_store_dQ.get_tma_tensor(make_shape(params.seqlen_q, params.d, params.h, params.b));
            Tensor gdQaccum = local_tile(mdQaccum(_, _, bidh, bidb), select<0, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (M, K, _)
            auto block_tma_dQ = tma_store_dQ.get_slice(_0{});
            Tensor tdQgdQ = block_tma_dQ.partition_D(gdQaccum);  // (TMA, TMA_M, TMA_K)
            Tensor tdQsdQ = block_tma_dQ.partition_S(sdQ); // (TMA, TMA_M, TMA_K)
            Tensor tdQsdQ2 = block_tma_dQ.partition_S(sdQ2); // (TMA, TMA_M, TMA_K, 2)
            int *lock_ptr = params.dq_semaphore + bidb * params.h + bidh;
            using Barrier = cutlass::GenericBarrier<cutlass::detail::SyncwarpSync>;
            // cutlass::arch::NamedBarrier::arrive(kNThreadsdQ + 32, 1 /*id*/);  // sdQ empty, ready to be written to
            cutlass::arch::NamedBarrier::arrive(kNThreadsdQ + 32, 0 /*id*/);  // sdQ empty, ready to be written to
            // cutlass::arch::NamedBarrier::arrive(kNThreadsdQ + 32, 0 + (m_block + 1) % 2 /*id*/);  // sdQ empty, ready to be written to
            // cutlass::arch::NamedBarrier::arrive(kNThreadsdQ + 32, 0 + m_block % 2 /*id*/);  // sdQ empty, ready to be written to
            // if (n_block == 0) {  // Use TMA_STORE
            if (false) {  // Use TMA_STORE
                #pragma unroll 2
                for (; m_block >= 0; --m_block) {
                    cutlass::arch::NamedBarrier::sync(kNThreadsdQ + 32, 2 /*id*/);  // sdQ full, to be written to gmem
                    // cutlass::arch::NamedBarrier::sync(kNThreadsdQ + 32, 2 + m_block % 2 /*id*/);  // sdQ full, to be written to gmem
                    if (lane_predicate) {
                        cute::copy(tma_store_dQ, tdQsdQ, tdQgdQ(_, _, _, m_block));
                        // cute::copy(tma_store_dQ, tdQsdQ2(_, _, _, m_block % 2), tdQgdQ(_, _, _, m_block));
                        tma_store_arrive();
                    }
                    tma_store_wait<0>();
                    Barrier::arrive_inc(lock_ptr, threadIdx.x % 32, m_block * params.b * params.h);
                    cutlass::arch::NamedBarrier::arrive(kNThreadsdQ + 32, 0 /*id*/);  // sdQ empty, ready to be written to
                    // cutlass::arch::NamedBarrier::arrive(kNThreadsdQ + 32, 0 + m_block % 2 /*id*/);  // sdQ empty, ready to be written to
                }
            } else {  // Use TMA_REDUCE_ADD
                #pragma unroll 2
                for (; m_block >= 0; --m_block) {
                    // Barrier::wait_eq(lock_ptr, threadIdx.x % 32, m_block * params.b * params.h, n_block);
                    // Barrier::wait_lt(lock_ptr, threadIdx.x % 32, m_block * params.b * params.h, 1);
                    cutlass::arch::NamedBarrier::sync(kNThreadsdQ + 32, 2 /*id*/);  // sdQ full, to be written to gmem
                    // cutlass::arch::NamedBarrier::sync(kNThreadsdQ + 32, 2 + m_block % 2 /*id*/);  // sdQ full, to be written to gmem
                    if (lane_predicate) {
                        cute::copy(tma_reduce_add_dQ, tdQsdQ, tdQgdQ(_, _, _, m_block));
                        // cute::copy(tma_reduce_add_dQ, tdQsdQ2(_, _, _, m_block % 2), tdQgdQ(_, _, _, m_block));
                        tma_store_arrive();
                    }
                    tma_store_wait<0>();
                    // Barrier::arrive_inc(lock_ptr, threadIdx.x % 32, m_block * params.b * params.h);
                    // cutlass::arch::NamedBarrier::arrive(kNThreadsdQ + 32, 0 + m_block % 2 /*id*/);  // sdQ empty, ready to be written to
                    cutlass::arch::NamedBarrier::arrive(kNThreadsdQ + 32, 0 /*id*/);  // sdQ empty, ready to be written to
                }
            }
        // } else if (warp_idx_in_warpgroup == 2) {  // Load LSE and dPSum
            // Tensor mLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum*>(params.softmax_lse_ptr)),
            //                           make_shape(params.b, params.h, params.seqlen_q),
            //                           make_stride(params.h * params.seqlen_q, params.seqlen_q, _1{}));
            // Tensor mdPsum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum*>(params.dsoftmax_sum)),
            //                             make_shape(params.b, params.h, params.seqlen_q),
            //                             make_stride(params.h * params.seqlen_q, params.seqlen_q, _1{}));
            // Tensor gLSE = local_tile(mLSE(bidb, bidh, _), Shape<Int<kBlockM>>{}, make_coord(_));  // (M, _)
            // Tensor gdPsum = local_tile(mdPsum(bidb, bidh, _), Shape<Int<kBlockM>>{}, make_coord(_));  // (M, _)
            // Tensor sLSE = make_tensor(make_smem_ptr(shared_storage.smem_lse.data()), Shape<Int<kBlockM>>{});
            // Tensor sdPsum = make_tensor(make_smem_ptr(shared_storage.smem_dpsum.data()), Shape<Int<kBlockM>>{});
            // #pragma unroll 2
            // for (; m_block >= 0; --m_block) {
            //     cutlass::arch::NamedBarrier::sync(NumMmaThreads + 32, 3 /*id*/);  // sLSE and sdPsum are empty
            //     #pragma unroll
            //     for (int i = 0; i < cute::ceil_div(kBlockM, 32); ++i) {
            //         int idx = threadIdx.x % 32 + i * 32;
            //         sLSE(idx) = idx < params.seqlen_q - m_block * kBlockM ? gLSE(idx, m_block) : INFINITY;
            //         sdPsum(idx) = idx < params.seqlen_q - m_block * kBlockM ? gdPsum(idx, m_block) : 0;
            //     }
            //     // sLSE and sdPsum are ready for WG 1
            //     cutlass::arch::NamedBarrier::arrive(128 + 32, 3 + 1 /*id*/);
            //     // sLSE and sdPsum are ready for WG 2
            //     cutlass::arch::NamedBarrier::arrive(128 + 32, 3 + 2 /*id*/);
            // }
        }


    } else {  // Consumers
        // method in cutlass/arch/reg_reconfig.h
        // calls setmaxnreg.inc.sync.aligned.u32
        cutlass::arch::warpgroup_reg_alloc<240>();

        // State variables used for iterating the circular buffer
        // smem_pipe_read / release is used by the consumer of SMEM data - i.e MMA
        // smem_pipe_write is used by the producer of SMEM data - i.e TMA
        PipelineState smem_pipe_read_q, smem_pipe_read_do;
        PipelineState smem_pipe_release_q, smem_pipe_release_do;

        int m_block = cute::ceil_div(params.seqlen_q, kBlockM) - 1;
        const int m_block_max = m_block;

        int bidb = blockIdx.z;  // The block index for the batch.
        int bidh = blockIdx.y;  // The block index for the head.

        Tensor mLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum*>(params.softmax_lse_ptr)),
                                  make_shape(params.b, params.h, params.seqlen_q),
                                  make_stride(params.h * params.seqlen_q, params.seqlen_q, _1{}));
        Tensor mdPsum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum*>(params.dsoftmax_sum)),
                                    make_shape(params.b, params.h, params.seqlen_q),
                                    make_stride(params.h * params.seqlen_q_rounded, params.seqlen_q_rounded, _1{}));
        Tensor gLSE = local_tile(mLSE(bidb, bidh, _), Shape<Int<kBlockM>>{}, make_coord(m_block));
        Tensor gdPsum = local_tile(mdPsum(bidb, bidh, _), Shape<Int<kBlockM>>{}, make_coord(m_block));

        Tensor sLSE = make_tensor(make_smem_ptr(shared_storage.smem_lse.data()), Shape<Int<kBlockM>>{});
        Tensor sdPsum = make_tensor(make_smem_ptr(shared_storage.smem_dpsum.data()), Shape<Int<kBlockM>>{});

        typename Ktraits::RmemTiledCopydQacc rmem_tiled_copy_dQaccum;
        // auto rmem_thr_copy_dQaccum = rmem_tiled_copy_dQaccum.get_thread_slice((threadIdx.x - NumCopyThreads) % kNThreadsdQ);
        auto rmem_thr_copy_dQaccum = rmem_tiled_copy_dQaccum.get_thread_slice(threadIdx.x - NumCopyThreads);
        Tensor tdQsdQaccum = rmem_thr_copy_dQaccum.partition_D(sdQ);
        Tensor tdQsdQaccum2 = rmem_thr_copy_dQaccum.partition_D(sdQ2);

        // Initialize matmul objects.
        typename Ktraits::TiledMmaSdP tiledMmaSdP;
        auto threadMmaSdP = tiledMmaSdP.get_thread_slice(threadIdx.x - NumCopyThreads);
        typename Ktraits::TiledMmadKV tiledMmadKV;
        auto threadMmadKV = tiledMmadKV.get_thread_slice(threadIdx.x - NumCopyThreads);
        typename Ktraits::TiledMmadQ tiledMmadQ;
        // auto threadMmadQ = tiledMmadQ.get_thread_slice((threadIdx.x - NumCopyThreads) % kNThreadsdQ);
        auto threadMmadQ = tiledMmadQ.get_thread_slice(threadIdx.x - NumCopyThreads);

        // Allocate accumulator
        Tensor tdKrdK = partition_fragment_C(tiledMmadKV, select<!dKV_swapAB ? 1 : 2, !dKV_swapAB? 2 : 1>(TileShape_MNK{}));
        Tensor tdVrdV = partition_fragment_C(tiledMmadKV, select<!dKV_swapAB ? 1 : 2, !dKV_swapAB? 2 : 1>(TileShape_MNK{}));

        auto smem_tiled_copy_PdS = make_tiled_copy_C(typename Ktraits::SmemCopyAtomPdS{}, tiledMmaSdP);
        auto smem_thr_copy_PdS = smem_tiled_copy_PdS.get_thread_slice(threadIdx.x - NumCopyThreads);
        // auto smem_tiled_copy_dQ = make_tiled_copy_C(typename Ktraits::SmemCopyAtomdQ{}, tiledMmadQ);
        // auto smem_tiled_copy_dQ = make_tiled_copy_C(Copy_Atom<cute::SM90_U32x4_STSM_N, ElementAccum>{}, tiledMmadQ);
        // auto smem_tiled_copy_dQ = make_tiled_copy_C(Copy_Atom<DefaultCopy, ElementAccum>{}, tiledMmadQ);
        // auto smem_thr_copy_dQ = smem_tiled_copy_dQ.get_thread_slice(threadIdx.x - NumCopyThreads);

        Tensor tdSsdS = smem_thr_copy_PdS.partition_D(sdSt);      // ((Atom,AtomNum),PIPE_M,PIPE_N)

        if constexpr (SdP_swapAB) {

            // Allocate "fragments/descriptors"
            Tensor tSrQ = threadMmaSdP.partition_fragment_B(sQ);
            Tensor tSrK = threadMmaSdP.partition_fragment_A(sK);
            Tensor tdPrdO = threadMmaSdP.partition_fragment_B(sdO);
            Tensor tdPrV = threadMmaSdP.partition_fragment_A(sV);

            Tensor caccS = make_identity_tensor(select<1, 0>(TileShape_MNK{}));    // (BLK_M,BLK_N) -> (blk_m,blk_n)
            Tensor taccScS = threadMmaSdP.partition_C(caccS);                           // (MMA,MMA_N,MMA_N)
            static_assert(decltype(size<0, 0>(taccScS))::value == 2);
            static_assert(decltype(size<0, 1>(taccScS))::value == 2);
            // taccScS has shape ((2, 2, V), MMA_M, MMA_N), we only take only the row indices.
            Tensor taccScS_row = taccScS(make_coord(_, _0{}, _), _0{}, _);
            static constexpr int kStatsPerThread = cute::ceil_div(decltype(size(taccScS_row))::value, 8);
            static constexpr bool kStatsDivisibleBy8 = decltype(size(taccScS_row))::value % 8 == 0;
            Tensor lse = make_tensor<ElementAccum>(Shape<Int<decltype(size(taccScS_row))::value>>{});
            // Tensor lse = make_tensor<ElementAccum>(Shape<Int<kStatsPerThread>>{});
            Tensor dP_sum = make_fragment_like(lse);
            #pragma unroll
            for (int mi = 0; mi < size(lse); ++mi) {
                const int row = get<1>(taccScS_row(mi));
                lse(mi) = row < params.seqlen_q - m_block * kBlockM ? gLSE(row) : INFINITY;
                dP_sum(mi) = row < params.seqlen_q - m_block * kBlockM ? gdPsum(row) : 0;
            }
            // #pragma unroll
            // for (int mi = 0; mi < size(lse); ++mi) {
                // const int row_idx = mi * 8 + (threadIdx.x % 32) / 4;
                // const int row = kStatsDivisibleBy8 || row_idx < size(taccScS_row) ? get<1>(taccScS_row(row_idx)) : 0;
                // lse(mi) = row < params.seqlen_q - m_block * kBlockM ? gLSE(row) : INFINITY;
                // dP_sum(mi) = row < params.seqlen_q - m_block * kBlockM ? gdPsum(row) : 0;
            // }
            // if (blockIdx.x == 0 && blockIdx.z == 1 && threadIdx.x == 128) { print_tensor(dP_sum); printf("\n"); }
            // Trying to spread LSE and dPSum across threads in a warp but it's slower
            // const int row_idx = mi * 8 + (threadIdx.x % 32) / 4;
            // const int row = get<1>(taccScS_row(row_idx));  // TODO: what if row_idx is outside the range?
            // cute::fill(lse, 1);
            // cute::fill(dP_sum, 1);
            // if (cute::thread0()) { print_tensor(dP_sum); printf("\n"); }
            // We want LSE = inf if the row is OOB. In that case Q would be zero, K would be zero,
            // and scores would be zero. With LSE = 0, probs will be all 1's, and when we multiply
            // with V (which would be zero), we're fine. However, with ALiBi, we might modify these
            // scores, and probs can become NaN. Instead if we set LSE = inf for OOB rows, probs are always 0.

            // cutlass::arch::NamedBarrier::arrive(NumMmaThreads + 32, 3 /*id*/);  // sLSE and sdPsum are empty

            clear(tdKrdK);
            clear(tdVrdV);

            shared_storage.barrier_K.wait(0);
            shared_storage.barrier_V.wait(0);

            // #pragma unroll 2
            CUTLASS_PRAGMA_NO_UNROLL
            for (; m_block >= 0; --m_block) {

                // Putting this dQ block at the beginning of the loop gives an extra 10 TFLOPs
                // It does make the code uglier, idk if it's worth it.
                if (m_block < m_block_max) {
                    // SMEM fence to make sure sP is written before it's read by WGMMA
                    cutlass::arch::fence_view_async_shared();
                    // dS is already written to smem, and the smem for dQ is empty (from warp 1 doing TMA_REDUCE_ADD)
                    // int warp_group_idx = cutlass::canonical_warp_group_idx();
                    // if (warp_group_idx == 1 + (m_block + 1) % 2) {
                    //     // cutlass::arch::NamedBarrier::arrive(NumMmaThreads + 32, 0 + (m_block + 1) % 2 /*id*/);
                    //     cutlass::arch::NamedBarrier::arrive(NumMmaThreads, 4);
                    // } else {
                    //     // cutlass::arch::NamedBarrier::sync(NumMmaThreads + 32, 0 + (m_block + 1) % 2 /*id*/);
                    //     cutlass::arch::NamedBarrier::sync(NumMmaThreads, 4);
                    //     static_assert(!Mma_dQ_is_RS);
                    //     Tensor tdQrdQ = partition_fragment_C(tiledMmadQ, select<!dQ_swapAB ? 0 : 2, !dQ_swapAB ? 2 : 0>(TileShape_MNK{}));
                    //     if constexpr (!dQ_swapAB) {
                    //         Tensor tdQrdS = threadMmadQ.partition_fragment_A(sdS);
                    //         Tensor tdQrK = threadMmadQ.partition_fragment_B(sKt);
                    //         flash::gemm</*zero_init=*/true, /*wg_wait=*/0>(tiledMmadQ, tdQrdS, tdQrK, tdQrdQ);
                    //     } else {
                    //         Tensor tdQrdS = threadMmadQ.partition_fragment_B(sdS);
                    //         Tensor tdQrK = threadMmadQ.partition_fragment_A(sKt);
                    //         flash::gemm</*zero_init=*/true, /*wg_wait=*/0>(tiledMmadQ, tdQrK, tdQrdS, tdQrdQ);
                    //     }
                    //     Tensor taccdQrdQ = rmem_thr_copy_dQaccum.retile_S(tdQrdQ);        // ((Atom,AtomNum), MMA_M, MMA_N)
                    //     cutlass::arch::NamedBarrier::sync(NumMmaThreads / 2 + 32, 0 + (m_block + 1) % 2 /*id*/);
                    //     cute::copy(rmem_tiled_copy_dQaccum, taccdQrdQ, tdQsdQaccum2(_, _, _, (m_block + 1) % 2));
                    //     cutlass::arch::fence_view_async_shared();
                    //     cutlass::arch::NamedBarrier::arrive(NumMmaThreads / 2 + 32, 2 + (m_block + 1) % 2 /*id*/);  // sdQ ready to be written to gmem
                    // }
                    // cutlass::arch::NamedBarrier::sync(NumMmaThreads + 32, 0 + (m_block + 1) % 2 /*id*/);
                    cutlass::arch::NamedBarrier::sync(NumMmaThreads + 32, 0 /*id*/);
                    static_assert(!Mma_dQ_is_RS);
                    Tensor tdQrdQ = partition_fragment_C(tiledMmadQ, select<!dQ_swapAB ? 0 : 2, !dQ_swapAB ? 2 : 0>(TileShape_MNK{}));
                    // Tensor taccdQrdQ = smem_thr_copy_dQ.retile_S(tdQrdQ);        // ((Atom,AtomNum), MMA_M, MMA_N)
                    if constexpr (!dQ_swapAB) {
                        Tensor tdQrdS = threadMmadQ.partition_fragment_A(sdS);
                        Tensor tdQrK = threadMmadQ.partition_fragment_B(sKt);
                        flash::gemm</*zero_init=*/true, /*wg_wait=*/0>(tiledMmadQ, tdQrdS, tdQrK, tdQrdQ);
                    } else {
                        Tensor tdQrdS = threadMmadQ.partition_fragment_B(sdS);
                        Tensor tdQrK = threadMmadQ.partition_fragment_A(sKt);
                        flash::gemm</*zero_init=*/true, /*wg_wait=*/0>(tiledMmadQ, tdQrK, tdQrdS, tdQrdQ);
                    }
                    // Tensor taccdQsdQt = smem_thr_copy_dQ.partition_D(sdQt);     // ((Atom,AtomNum),PIPE_M,PIPE_N)
                    // cute::copy(smem_tiled_copy_dQ, taccdQrdQ, taccdQsdQt);
                    // Tensor taccdQsdQ = smem_thr_copy_dQ.partition_D(sdQ);     // ((Atom,AtomNum),PIPE_M,PIPE_N)
                    // cute::copy(smem_tiled_copy_dQ, taccdQrdQ, taccdQsdQ);
                    Tensor taccdQrdQ = rmem_thr_copy_dQaccum.retile_S(tdQrdQ);        // ((Atom,AtomNum), MMA_M, MMA_N)
                    // cutlass::arch::NamedBarrier::sync(NumMmaThreads + 32, 1 /*id*/);  // sdQ empty, ready to be written to
                    cute::copy(rmem_tiled_copy_dQaccum, taccdQrdQ, tdQsdQaccum);
                    // cute::copy(rmem_tiled_copy_dQaccum, taccdQrdQ, tdQsdQaccum2(_, _, _, (m_block + 1) % 2));
                    cutlass::arch::fence_view_async_shared();
                    cutlass::arch::NamedBarrier::arrive(NumMmaThreads + 32, 2 /*id*/);  // sdQ ready to be written to gmem
                    // cutlass::arch::NamedBarrier::arrive(NumMmaThreads + 32, 2 + (m_block + 1) % 2 /*id*/);  // sdQ ready to be written to gmem
                }

                Tensor tSrS = partition_fragment_C(tiledMmaSdP, select<1, 0>(TileShape_MNK{}));
                pipeline_q.consumer_wait(smem_pipe_read_q);
                flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiledMmaSdP, tSrK, tSrQ(_, _, _, smem_pipe_read_q.index()), tSrS);
                Tensor tdPrdP = partition_fragment_C(tiledMmaSdP, select<1, 0>(TileShape_MNK{}));
                pipeline_do.consumer_wait(smem_pipe_read_do);
                flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiledMmaSdP, tdPrV, tdPrdO(_, _, _, smem_pipe_read_do.index()), tdPrdP);

                // sLSE and sdPsum are done loading for WG 1 or 2
                // cutlass::arch::NamedBarrier::sync(128 + 32, 3 + cutlass::canonical_warp_group_idx() /*id*/);
                // Tensor lse = make_tensor<ElementAccum>(Shape<Int<decltype(size(taccScS_row))::value>>{});
                // #pragma unroll
                // for (int mi = 0; mi < size(lse); ++mi) { lse(mi) = sLSE(get<1>(taccScS_row(mi))); }
                warpgroup_wait<1>();
                // Reshape tSrS from ((2, 2, V), MMA_N, MMA_M) to (nrow=(2, V, MMA_M), ncol=(2, MMA_N))
                Tensor scores = make_tensor(tSrS.data(), flash::convert_layout_acc_transposed_rowcol(tSrS.layout()));
                flash::scale_apply_exp2</*Scale=*/true, /*scale_max=*/false>(scores, lse, params.scale_softmax_log2);
                // #pragma unroll
                // for (int mi = 0; mi < size<0>(lse); ++mi) { lse(mi) *= float(M_LOG2E); }
                // #pragma unroll
                // for (int mi = 0; mi < size<0>(scores); ++mi) {
                //     // const float lse_scaled = lse(mi) * float(M_LOG2E);
                //     const float lse_scaled = __shfl_sync(0xffffffff, lse(mi / 8), (mi % 8) * 4 + (threadIdx.x % 4));
                //     // const float lse_scaled = __shfl_xor_sync(0xffffffff, lse(mi / 8), 1 << (mi % 4)) * float(M_LOG2E);
                //     // const float lse_scaled = lse(mi);
                //     #pragma unroll
                //     for (int ni = 0; ni < size<1>(scores); ++ni) {
                //         scores(mi, ni) = exp2f(scores(mi, ni) * params.scale_softmax_log2 - lse_scaled);
                //     }
                // }
                // if (blockIdx.x == 0 && blockIdx.z == 1 && threadIdx.x == 128) { print_tensor(scores); printf("\n"); }
                // Tensor dP_sum = make_fragment_like(lse);
                // sLSE and sdPsum are done loading for WG 1 or 2
                // cutlass::arch::NamedBarrier::sync(128 + 32, 3 + cutlass::canonical_warp_group_idx() /*id*/);
                // #pragma unroll
                // for (int mi = 0; mi < size(dP_sum); ++mi) { dP_sum(mi) = sdPsum(get<1>(taccScS_row(mi))); }


                // Convert scores from fp32 to fp16/bf16
                Tensor rP = flash::convert_type<Element>(tSrS);

                warpgroup_wait<0>();
                // Reshape tdPrdP from ((2, 2, V), MMA_N, MMA_M) to (nrow=(2, V, MMA_M), ncol=(2, MMA_N))
                Tensor dS = make_tensor(tdPrdP.data(), scores.layout());
                // if (blockIdx.x == 0 && blockIdx.z == 1 && threadIdx.x == 128) { print_tensor(dS); printf("\n"); }
                // if (blockIdx.x == 0 && blockIdx.z == 1 && threadIdx.x == 128) { print_tensor(dP_sum); printf("\n"); }
                #pragma unroll
                for (int mi = 0; mi < size<0>(dS); ++mi) {
                    // const float dP_sum_cur = __shfl_sync(0xffffffff, dP_sum(mi / 8), (mi % 8) * 4 + (threadIdx.x % 4));
                    // const float dP_sum_cur = __shfl_xor_sync(0xffffffff, dP_sum(mi / 8), 1 << (mi % 4));
                    #pragma unroll
                    for (int ni = 0; ni < size<1>(dS); ++ni) { dS(mi, ni) = scores(mi, ni) * (dS(mi, ni) - dP_sum(mi)); }
                    // for (int ni = 0; ni < size<1>(dS); ++ni) { dS(mi, ni) = scores(mi, ni) * (dS(mi, ni) - dP_sum_cur); }
                }
                // sLSE and sdPsum are done processing, can load for the next iteration
                // cutlass::arch::NamedBarrier::arrive(NumMmaThreads + 32, 3 /*id*/);
                // if (blockIdx.x == 0 && blockIdx.z == 1 && threadIdx.x == 128) { print_tensor(dS); printf("\n"); }
                Tensor rdS = flash::convert_type<Element>(tdPrdP);

                Tensor tdSadS = smem_thr_copy_PdS.retile_S(rdS);     // ((Atom,AtomNum), MMA_N, MMA_N)
                cute::copy(smem_tiled_copy_PdS, tdSadS, tdSsdS);

                if (m_block > 0) {
                    gLSE.data() = gLSE.data() + (-int(kBlockM));
                    gdPsum.data() = gdPsum.data() + (-int(kBlockM));
                }
                // #pragma unroll
                // for (int mi = 0; mi < size(lse); ++mi) {
                //     // const int row = get<1>(taccScS_row(mi));
                //     const int row_idx = mi * 8 + (threadIdx.x % 32) / 4;
                //     const int row = kStatsDivisibleBy8 || row_idx < size(taccScS_row) ? get<1>(taccScS_row(row_idx)) : 0;
                //     lse(mi) = gLSE(row);
                //     dP_sum(mi) = gdPsum(row);
                // }
                Tensor lse_float2 = recast<float2>(lse);
                Tensor dP_sum_float2 = recast<float2>(dP_sum);
                #pragma unroll
                for (int mi = 0; mi < size(lse) / 2; ++mi) {
                    const int row = get<1>(taccScS_row(mi * 2));
                    lse_float2(mi) = *reinterpret_cast<float2*>(&(gLSE(row)));
                    dP_sum_float2(mi) = *reinterpret_cast<float2*>(&(gdPsum(row)));
                }

                static_assert(!dKV_swapAB);
                Tensor tdVrP = make_tensor(rP.data(), convert_layout_acc_Aregs<typename Ktraits::TiledMmadKV>(tSrS.layout()));
                Tensor tdVrdO = threadMmadKV.partition_fragment_B(sdOt);
                flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(tiledMmadKV, tdVrP, tdVrdO(_, _, _, smem_pipe_read_do.index()), tdVrdV);
                ++smem_pipe_read_do;
                // warpgroup_wait<0>();
                // Tensor dV_tmp = make_tensor(tdVrdV.data(), flash::convert_layout_acc_rowcol(tdVrdV.layout()));
                // if (blockIdx.x == 0 && threadIdx.x == 128) { print_tensor(dV_tmp); printf("\n"); }

                Tensor tdKrdS = make_tensor(rdS.data(), convert_layout_acc_Aregs<typename Ktraits::TiledMmadKV>(tdPrdP.layout()));
                Tensor tdKrQ = threadMmadKV.partition_fragment_B(sQt);
                flash::gemm</*zero_init=*/false, /*wg_wait=*/0>(tiledMmadKV, tdKrdS, tdKrQ(_, _, _, smem_pipe_read_q.index()), tdKrdK);
                ++smem_pipe_read_q;
                // warpgroup_wait<0>();
                // Tensor dK_tmp = make_tensor(tdKrdK.data(), flash::convert_layout_acc_rowcol(tdKrdK.layout()));
                // if (blockIdx.x == 0 && threadIdx.x == 128) { print_tensor(dK_tmp); printf("\n"); }

                pipeline_do.consumer_release(smem_pipe_release_do);  // release V
                ++smem_pipe_release_do;
                pipeline_q.consumer_release(smem_pipe_release_q);  // release V
                ++smem_pipe_release_q;

                // warpgroup_wait<0>();
                // Tensor dQ_tmp = make_tensor(tdQrdQ.data(), flash::convert_layout_acc_rowcol(tdQrdQ.layout()));
                // if (cute::thread0()) { print_tensor(dQ_tmp); printf("\n"); }

            }

            {
                // SMEM fence to make sure sP is written before it's read by WGMMA
                cutlass::arch::fence_view_async_shared();
                // dS is already written to smem, and the smem for dQ is empty (from warp 1 doing TMA_REDUCE_ADD)
                // int warp_group_idx = cutlass::canonical_warp_group_idx();
                // if (warp_group_idx == 1 + (m_block + 1) % 2) {
                //     // cutlass::arch::NamedBarrier::arrive(NumMmaThreads + 32, 0 + (m_block + 1) % 2 /*id*/);
                //     cutlass::arch::NamedBarrier::arrive(NumMmaThreads, 4);
                // } else {
                //     // cutlass::arch::NamedBarrier::sync(NumMmaThreads + 32, 0 + (m_block + 1) % 2 /*id*/);
                //     cutlass::arch::NamedBarrier::sync(NumMmaThreads, 4);
                //     static_assert(!Mma_dQ_is_RS);
                //     Tensor tdQrdQ = partition_fragment_C(tiledMmadQ, select<!dQ_swapAB ? 0 : 2, !dQ_swapAB ? 2 : 0>(TileShape_MNK{}));
                //     if constexpr (!dQ_swapAB) {
                //         Tensor tdQrdS = threadMmadQ.partition_fragment_A(sdS);
                //         Tensor tdQrK = threadMmadQ.partition_fragment_B(sKt);
                //         flash::gemm</*zero_init=*/true, /*wg_wait=*/0>(tiledMmadQ, tdQrdS, tdQrK, tdQrdQ);
                //     } else {
                //         Tensor tdQrdS = threadMmadQ.partition_fragment_B(sdS);
                //         Tensor tdQrK = threadMmadQ.partition_fragment_A(sKt);
                //         flash::gemm</*zero_init=*/true, /*wg_wait=*/0>(tiledMmadQ, tdQrK, tdQrdS, tdQrdQ);
                //     }
                //     Tensor taccdQrdQ = rmem_thr_copy_dQaccum.retile_S(tdQrdQ);        // ((Atom,AtomNum), MMA_M, MMA_N)
                //     cutlass::arch::NamedBarrier::sync(NumMmaThreads / 2 + 32, 0 + (m_block + 1) % 2 /*id*/);
                //     cute::copy(rmem_tiled_copy_dQaccum, taccdQrdQ, tdQsdQaccum2(_, _, _, (m_block + 1) % 2));
                //     cutlass::arch::fence_view_async_shared();
                //     cutlass::arch::NamedBarrier::arrive(NumMmaThreads / 2 + 32, 2 + (m_block + 1) % 2 /*id*/);  // sdQ ready to be written to gmem
                //     Tensor dQ_tmp = make_tensor(tdQrdQ.data(), flash::convert_layout_acc_rowcol(tdQrdQ.layout()));
                //     // if (blockIdx.x == 0 && threadIdx.x == 128) { print_tensor(dQ_tmp); printf("\n"); }
                //     // if (blockIdx.x == 0 && threadIdx.x == 128) { print(taccdQrdQ); printf("\n"); print(tdQsdQaccum2); printf("\n"); }
                // }
                cutlass::arch::NamedBarrier::sync(NumMmaThreads + 32, 0 /*id*/);
                // cutlass::arch::NamedBarrier::sync(NumMmaThreads + 32, 0 + 0 % 2 /*id*/);
                // cutlass::arch::NamedBarrier::sync(NumMmaThreads, 0 /*id*/);
                static_assert(!Mma_dQ_is_RS);
                Tensor tdQrdQ = partition_fragment_C(tiledMmadQ, select<!dQ_swapAB ? 0 : 2, !dQ_swapAB ? 2 : 0>(TileShape_MNK{}));
                if constexpr (!dQ_swapAB) {
                    Tensor tdQrdS = threadMmadQ.partition_fragment_A(sdS);
                    Tensor tdQrK = threadMmadQ.partition_fragment_B(sKt);
                    flash::gemm</*zero_init=*/true, /*wg_wait=*/0>(tiledMmadQ, tdQrdS, tdQrK, tdQrdQ);
                } else {
                    Tensor tdQrdS = threadMmadQ.partition_fragment_B(sdS);
                    Tensor tdQrK = threadMmadQ.partition_fragment_A(sKt);
                    flash::gemm</*zero_init=*/true, /*wg_wait=*/0>(tiledMmadQ, tdQrK, tdQrdS, tdQrdQ);
                }
                // Tensor dQ_tmp = make_tensor(tdQrdQ.data(), flash::convert_layout_acc_rowcol(tdQrdQ.layout()));
                // if (blockIdx.x == 0 && threadIdx.x == 128) { print_tensor(dQ_tmp); printf("\n"); }
                Tensor taccdQrdQ = rmem_thr_copy_dQaccum.retile_S(tdQrdQ);        // ((Atom,AtomNum), MMA_M, MMA_N)
                // cutlass::arch::NamedBarrier::sync(NumMmaThreads + 32, 1 /*id*/);  // sdQ empty, ready to be written to
                cute::copy(rmem_tiled_copy_dQaccum, taccdQrdQ, tdQsdQaccum);
                // cute::copy(rmem_tiled_copy_dQaccum, taccdQrdQ, tdQsdQaccum2(_, _, _, 0 % 2));
                cutlass::arch::fence_view_async_shared();
                cutlass::arch::NamedBarrier::arrive(NumMmaThreads + 32, 2 /*id*/);  // sdQ ready to be written to gmem
                // cutlass::arch::NamedBarrier::arrive(NumMmaThreads + 32, 2 + 0 % 2 /*id*/);  // sdQ ready to be written to gmem
                // if (blockIdx.x == 0 && threadIdx.x == 128) { print_tensor(sdQ); printf("\n"); }
            }

        } else {  // !SdP_swapAB
            Tensor tPsP = smem_thr_copy_PdS.partition_D(sP);      // ((Atom,AtomNum),PIPE_M,PIPE_N)
            Tensor tdSsdS = smem_thr_copy_PdS.partition_D(sdS);      // ((Atom,AtomNum),PIPE_M,PIPE_N)

            // Allocate "fragments/descriptors"
            Tensor tSrQ = threadMmaSdP.partition_fragment_A(sQ);
            Tensor tSrK = threadMmaSdP.partition_fragment_B(sK);
            Tensor tdPrdO = threadMmaSdP.partition_fragment_A(sdO);
            Tensor tdPrV = threadMmaSdP.partition_fragment_B(sV);

            Tensor caccS = make_identity_tensor(select<0, 1>(TileShape_MNK{}));    // (BLK_M,BLK_N) -> (blk_m,blk_n)
            Tensor taccScS = threadMmaSdP.partition_C(caccS);                           // (MMA,MMA_N,MMA_N)
            static_assert(decltype(size<0, 0>(taccScS))::value == 2);
            static_assert(decltype(size<0, 1>(taccScS))::value == 2);
            // taccScS has shape ((2, 2, V), MMA_M, MMA_N), we only take only the row indices.
            Tensor taccScS_row = taccScS(make_coord(_0{}, _, _0{}), _, _0{});
            Tensor lse = make_tensor<ElementAccum>(Shape<Int<decltype(size(taccScS_row))::value>>{});
            Tensor dP_sum = make_fragment_like(lse);
            #pragma unroll
            for (int mi = 0; mi < size(lse); ++mi) {
                const int row = get<0>(taccScS_row(mi));
                lse(mi) = row < params.seqlen_q - m_block * kBlockM ? gLSE(row) : INFINITY;
                dP_sum(mi) = row < params.seqlen_q - m_block * kBlockM ? gdPsum(row) : 0;
            }
            // if (cute::thread0()) { print_tensor(dP_sum); printf("\n"); }
            // We want LSE = inf if the row is OOB. In that case Q would be zero, K would be zero,
            // and scores would be zero. With LSE = 0, probs will be all 1's, and when we multiply
            // with V (which would be zero), we're fine. However, with ALiBi, we might modify these
            // scores, and probs can become NaN. Instead if we set LSE = inf for OOB rows, probs are always 0.

            clear(tdKrdK);
            clear(tdVrdV);

            shared_storage.barrier_K.wait(0);
            shared_storage.barrier_V.wait(0);

            CUTLASS_PRAGMA_NO_UNROLL
            for (; m_block >= 0; --m_block) {
                Tensor tSrS = partition_fragment_C(tiledMmaSdP, select<0, 1>(TileShape_MNK{}));
                pipeline_q.consumer_wait(smem_pipe_read_q);
                flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiledMmaSdP, tSrQ(_, _, _, smem_pipe_read_q.index()), tSrK, tSrS);
                Tensor tdPrdP = partition_fragment_C(tiledMmaSdP, select<0, 1>(TileShape_MNK{}));
                pipeline_do.consumer_wait(smem_pipe_read_do);
                // if (blockIdx.x == 0 && blockIdx.z == 0 && threadIdx.x == 128) { printf("After dO wait\n"); }
                flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiledMmaSdP, tdPrdO(_, _, _, smem_pipe_read_do.index()), tdPrV, tdPrdP);

                warpgroup_wait<1>();
                // Reshape tSrS from ((2, 2, V), MMA_N, MMA_M) to (nrow=(2, V, MMA_M), ncol=(2, MMA_N))
                Tensor scores = make_tensor(tSrS.data(), flash::convert_layout_acc_rowcol(tSrS.layout()));
                flash::scale_apply_exp2</*Scale=*/true, /*scale_max=*/false>(scores, lse, params.scale_softmax_log2);
                // if (blockIdx.x == 0 && blockIdx.z == 0 && threadIdx.x == 128) { print_tensor(scores); printf("\n"); }

                // Convert scores from fp32 to fp16/bf16
                Tensor rP = flash::convert_type<Element>(tSrS);
                Tensor tPaP = smem_thr_copy_PdS.retile_S(rP);     // ((Atom,AtomNum), MMA_N, MMA_N)
                cutlass::arch::NamedBarrier::sync(NumMmaThreads, 8 /*id*/);
                cute::copy(smem_tiled_copy_PdS, tPaP, tPsP);
                int const warp_group_idx = cutlass::canonical_warp_group_idx() - 1;
                cutlass::arch::NamedBarrier::arrive(NumMmaThreads, 4 + warp_group_idx /*id*/);
                // if (blockIdx.x == 0 && blockIdx.z == 0 && (threadIdx.x == 128 || threadIdx.x == 256)) { printf("After barrier arrive 4, tidx = %d\n", threadIdx.x); }

                warpgroup_wait<0>();
                // Reshape tdPrdP from ((2, 2, V), MMA_N, MMA_M) to (nrow=(2, V, MMA_M), ncol=(2, MMA_N))
                Tensor dS = make_tensor(tdPrdP.data(), scores.layout());
                // if (blockIdx.x == 0 && blockIdx.z == 1 && threadIdx.x == 128) { print_tensor(dS); printf("\n"); }
                // if (blockIdx.x == 0 && blockIdx.z == 1 && threadIdx.x == 128) { print_tensor(dP_sum); printf("\n"); }
                #pragma unroll
                for (int mi = 0; mi < size<0>(dS); ++mi) {
                    #pragma unroll
                    for (int ni = 0; ni < size<1>(dS); ++ni) { dS(mi, ni) = scores(mi, ni) * (dS(mi, ni) - dP_sum(mi)); }
                }
                // if (blockIdx.x == 0 && blockIdx.z == 0 && threadIdx.x == 128) { print_tensor(dS); printf("\n"); }
                Tensor rdS = flash::convert_type<Element>(tdPrdP);

                Tensor tdSadS = smem_thr_copy_PdS.retile_S(rdS);     // ((Atom,AtomNum), MMA_N, MMA_N)
                cute::copy(smem_tiled_copy_PdS, tdSadS, tdSsdS);
                cutlass::arch::NamedBarrier::arrive(NumMmaThreads, 6 + warp_group_idx /*id*/);
                // if (blockIdx.x == 0 && blockIdx.z == 0 && (threadIdx.x == 128 || threadIdx.x == 256)) { printf("After barrier arrive 6, tidx = %d\n", threadIdx.x); }

                if (m_block > 0) {
                    gLSE.data() = gLSE.data() + (-int(kBlockM));
                    gdPsum.data() = gdPsum.data() + (-int(kBlockM));
                }
                #pragma unroll
                for (int mi = 0; mi < size(lse); ++mi) {
                    const int row = get<1>(taccScS_row(mi));
                    lse(mi) = gLSE(row);
                    dP_sum(mi) = gdPsum(row);
                }

                Tensor tdQrdQ = partition_fragment_C(tiledMmadQ, select<!dQ_swapAB ? 0 : 2, !dQ_swapAB ? 2 : 0>(TileShape_MNK{}));
                if constexpr (Mma_dQ_is_RS) {
                    static_assert(!dQ_swapAB);
                    Tensor tdQrdS = make_tensor(rdS.data(), convert_layout_acc_Aregs<typename Ktraits::TiledMmadQ>(tdPrdP.layout()));
                    Tensor tdQrK = threadMmadQ.partition_fragment_B(sKt);
                    flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiledMmadQ, tdQrdS, tdQrK, tdQrdQ);
                    // if (cute::thread0()) { print(tdQrdS); printf("\n"); print(tdQrK); printf("\n"); print(tdQrdQ); printf("\n"); }
                }

                cutlass::arch::fence_view_async_shared();
                // if (blockIdx.x == 0 && blockIdx.z == 0 && (threadIdx.x == 128 || threadIdx.x == 256)) { printf("Before barrier sync 4, tidx = %d\n", threadIdx.x); }
                cutlass::arch::NamedBarrier::sync(NumMmaThreads, 4 + 1 - warp_group_idx /*id*/);
                // if (blockIdx.x == 0 && blockIdx.z == 0 && (threadIdx.x == 128 || threadIdx.x == 256)) { printf("After barrier sync 4, tidx = %d\n", threadIdx.x); }
                // if (blockIdx.x == 0 && blockIdx.z == 0 && (threadIdx.x == 128)) { print_tensor(sPt); printf("\n"); }
                // if (blockIdx.x == 0 && blockIdx.z == 0 && (threadIdx.x == 128)) { print_tensor(sdOt); printf("\n"); }
                if constexpr (!dKV_swapAB) {
                    Tensor tdVrP = threadMmadKV.partition_fragment_A(sPt);
                    Tensor tdVrdO = threadMmadKV.partition_fragment_B(sdOt);
                    flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(tiledMmadKV, tdVrP, tdVrdO(_, _, _, smem_pipe_read_do.index()), tdVrdV);
                } else {
                    Tensor tdVrP = threadMmadKV.partition_fragment_B(sPt);
                    Tensor tdVrdO = threadMmadKV.partition_fragment_A(sdOt);
                    flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(tiledMmadKV, tdVrdO(_, _, _, smem_pipe_read_do.index()), tdVrP, tdVrdV);
                }
                ++smem_pipe_read_do;
                // warpgroup_wait<0>();
                // Tensor dV_tmp = make_tensor(tdVrdV.data(), flash::convert_layout_acc_rowcol(tdVrdV.layout()));
                // if (blockIdx.x == 0 && threadIdx.x == 128) { print_tensor(dV_tmp); printf("\n"); }

                cutlass::arch::fence_view_async_shared();
                // if (blockIdx.x == 0 && blockIdx.z == 0 && (threadIdx.x == 128 || threadIdx.x == 256)) { printf("Before barrier sync 6, tidx = %d\n", threadIdx.x); }
                cutlass::arch::NamedBarrier::sync(NumMmaThreads, 6 + 1 - warp_group_idx /*id*/);
                // if (blockIdx.x == 0 && blockIdx.z == 0 && (threadIdx.x == 128 || threadIdx.x == 256)) { printf("After barrier sync 6, tidx = %d\n", threadIdx.x); }
                if constexpr (!dKV_swapAB) {
                    Tensor tdKrdS = threadMmadKV.partition_fragment_A(sdSt);
                    Tensor tdKrQ = threadMmadKV.partition_fragment_B(sQt);
                    flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(tiledMmadKV, tdKrdS, tdKrQ(_, _, _, smem_pipe_read_q.index()), tdKrdK);
                } else {
                    Tensor tdKrdS = threadMmadKV.partition_fragment_B(sdSt);
                    Tensor tdKrQ = threadMmadKV.partition_fragment_A(sQt);
                    flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(tiledMmadKV, tdKrQ(_, _, _, smem_pipe_read_q.index()), tdKrdS, tdKrdK);
                }
                ++smem_pipe_read_q;
                warpgroup_wait<0>();
                // Tensor dK_tmp = make_tensor(tdKrdK.data(), flash::convert_layout_acc_rowcol(tdKrdK.layout()));
                // if (blockIdx.x == 0 && threadIdx.x == 128) { print_tensor(dK_tmp); printf("\n"); }

                pipeline_do.consumer_release(smem_pipe_release_do);  // release V
                ++smem_pipe_release_do;
                pipeline_q.consumer_release(smem_pipe_release_q);  // release V
                ++smem_pipe_release_q;

                // warpgroup_wait<0>();
                // Tensor dQ_tmp = make_tensor(tdQrdQ.data(), flash::convert_layout_acc_rowcol(tdQrdQ.layout()));
                // if (cute::thread0()) { print_tensor(dQ_tmp); printf("\n"); }

                cutlass::arch::NamedBarrier::sync(NumMmaThreads, 8 /*id*/);
            }

        }

        // Epilogue

        Tensor sdK = make_tensor(make_smem_ptr(shared_storage.smem_dk.data()), typename Ktraits::SmemLayoutdK{});
        Tensor sdV = make_tensor(make_smem_ptr(shared_storage.smem_dv.data()), typename Ktraits::SmemLayoutdV{});
        Tensor sdKt = make_tensor(make_smem_ptr(shared_storage.smem_dk.data()), typename Ktraits::SmemLayoutdKt{});
        Tensor sdVt = make_tensor(make_smem_ptr(shared_storage.smem_dv.data()), typename Ktraits::SmemLayoutdVt{});

        auto smem_tiled_copy_dKV = make_tiled_copy_C(typename Ktraits::SmemCopyAtomdKV{}, tiledMmadKV);
        auto smem_thr_copy_dKV = smem_tiled_copy_dKV.get_thread_slice(threadIdx.x - NumCopyThreads);

        int n_block = blockIdx.x;
        bidb = blockIdx.z;  // The block index for the batch.
        bidh = blockIdx.y;  // The block index for the head.
        Tensor mdK = tma_store_dK.get_tma_tensor(make_shape(params.seqlen_k, params.d, params.h, params.b));
        Tensor mdV = tma_store_dV.get_tma_tensor(make_shape(params.seqlen_k, params.d, params.h, params.b));
        Tensor gdK = local_tile(mdK(_, _, bidh, bidb), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{}));  // (M, K)
        Tensor gdV = local_tile(mdV(_, _, bidh, bidb), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{}));  // (M, K)
        auto block_tma_dK = tma_store_dK.get_slice(_0{});
        auto block_tma_dV = tma_store_dV.get_slice(_0{});
        Tensor tdKgdK = block_tma_dK.partition_D(gdK);  // (TMA, TMA_M, TMA_K)
        Tensor tdKsdK = block_tma_dK.partition_S(sdK); // (TMA, TMA_M, TMA_K)
        Tensor tdVgdV = block_tma_dV.partition_D(gdV);  // (TMA, TMA_M, TMA_K)
        Tensor tdVsdV = block_tma_dV.partition_S(sdV); // (TMA, TMA_M, TMA_K)


        // Very slightly faster to do the smem write and TMA write for dV first, then do the same for dK,
        // Instead of doing both at the same time.
        Tensor tdVrdV_out = convert_type<Element>(tdVrdV);
        #pragma unroll
        for (int i = 0; i < size(tdKrdK); ++i) { tdKrdK(i) *= params.scale_softmax; }
        Tensor tdKrdK_out = convert_type<Element>(tdKrdK);

        Tensor taccdVrdV = smem_thr_copy_dKV.retile_S(tdVrdV_out);        // ((Atom,AtomNum), MMA_M, MMA_N)
        Tensor taccdKrdK = smem_thr_copy_dKV.retile_S(tdKrdK_out);        // ((Atom,AtomNum), MMA_M, MMA_N)

        //  Can't use __syncthreads() in WS code
        auto synchronize = [&] () { cutlass::arch::NamedBarrier::sync(NumMmaThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier); };
        synchronize();
        if constexpr (!dKV_swapAB) {
            Tensor taccdVsdV = smem_thr_copy_dKV.partition_D(sdV);     // ((Atom,AtomNum),PIPE_M,PIPE_N)
            cute::copy(smem_tiled_copy_dKV, taccdVrdV, taccdVsdV);
        } else {
            Tensor taccdVsdVt = smem_thr_copy_dKV.partition_D(sdVt);     // ((Atom,AtomNum),PIPE_M,PIPE_N)
            cute::copy(smem_tiled_copy_dKV, taccdVrdV, taccdVsdVt);
        }
        cutlass::arch::fence_view_async_shared(); // ensure smem writes are visible to TMA
        synchronize();

        lane_predicate = cute::elect_one_sync();
        warp_idx = cutlass::canonical_warp_idx_sync();
        if (warp_idx == NumCopyThreads / cutlass::NumThreadsPerWarp && lane_predicate) {
            cute::copy(tma_store_dV, tdVsdV, tdVgdV);
            tma_store_arrive();
        }

        if constexpr (!dKV_swapAB) {
            Tensor taccdKsdK = smem_thr_copy_dKV.partition_D(sdK);     // ((Atom,AtomNum),PIPE_M,PIPE_N)
            cute::copy(smem_tiled_copy_dKV, taccdKrdK, taccdKsdK);
        } else {
            Tensor taccdKsdKt = smem_thr_copy_dKV.partition_D(sdKt);     // ((Atom,AtomNum),PIPE_M,PIPE_N)
            cute::copy(smem_tiled_copy_dKV, taccdKrdK, taccdKsdKt);
        }
        cutlass::arch::fence_view_async_shared(); // ensure smem writes are visible to TMA
        synchronize();
        if (warp_idx == NumCopyThreads / cutlass::NumThreadsPerWarp && lane_predicate) {
            cute::copy(tma_store_dK, tdKsdK, tdKgdK);
            tma_store_arrive();
        }
        tma_store_wait<0>();
    }

}

} // namespace flash
