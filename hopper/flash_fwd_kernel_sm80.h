/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include "cute/tensor.hpp"

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/kernel_hardware_info.h>

#include "seqlen.h"
#include "utils.h"
#include "softmax.h"

namespace flash {

using namespace cute;

template <class CollectiveMainloop_, class CollectiveEpilogue_, class TileScheduler_>
class FlashAttnFwdSm80 {

public:

    // Type Aliases
    using CollectiveMainloop = CollectiveMainloop_;
    using CollectiveEpilogue = CollectiveEpilogue_;
    static constexpr bool Is_causal = CollectiveMainloop::Is_causal;
    static constexpr bool Is_local = CollectiveMainloop::Is_local;
    static_assert(CollectiveMainloop::Varlen == CollectiveEpilogue::Varlen);
    static constexpr bool Has_softcap = CollectiveMainloop::Has_softcap;
    static constexpr bool Varlen = CollectiveMainloop::Varlen;
    static constexpr bool PagedKV = CollectiveMainloop::PagedKV;
    static constexpr bool Split = CollectiveMainloop::Split;
    static constexpr bool Is_FP8 = CollectiveMainloop::Is_FP8;
    static constexpr bool Transpose_V = CollectiveMainloop::Transpose_V;
    static constexpr bool AppendKV = CollectiveMainloop::AppendKV;
    static constexpr bool PackGQA = CollectiveMainloop::PackGQA;
    static constexpr int NumProducerThreads = CollectiveMainloop::NumProducerThreads;
    using SeqlenInfo_t = typename CollectiveMainloop::SeqlenInfo_t;

    // Mainloop derived types
    using TileShape_MNK = typename CollectiveMainloop::TileShape_MNK;
    using TiledMma = typename CollectiveMainloop::TiledMma;
    using ArchTag = typename CollectiveMainloop::ArchTag;
    using MainloopArguments = typename CollectiveMainloop::Arguments;
    using MainloopParams = typename CollectiveMainloop::Params;

    // Epilogue derived types
    using EpilogueArguments = typename CollectiveEpilogue::Arguments;
    using EpilogueParams = typename CollectiveEpilogue::Params;

    static_assert(ArchTag::kMinComputeCapability >= 80);

    using TileScheduler = TileScheduler_;
    using TileSchedulerArguments = typename flash::TileSchedulerArguments;
    using TileSchedulerParams = typename TileScheduler::Params;

    static constexpr uint32_t NumThreads = CUTE_STATIC_V(size(TiledMma{}));
    static constexpr uint32_t MaxThreadsPerBlock = CUTE_STATIC_V(size(TiledMma{}));
    static constexpr uint32_t MinBlocksPerMultiprocessor = NumThreads == 128 ? 2 : 1;

    // Kernel level shared memory storage
    // We overlap the shared memory for the mainloop and epilogue. However, we only want smem_o to overlap with smem_v + smem_k and not smem_q
    // and nothing else, so we'll pad in case sizeof(smem_o) > sizeof(smem_v) + sizeof(smem_k).
    static constexpr int mainloop_smem_padding_ = int(sizeof(typename CollectiveEpilogue::TensorStorage))
        - int(sizeof(decltype((typename CollectiveMainloop::TensorStorage{}).smem_v)))
        - int(sizeof(decltype((typename CollectiveMainloop::TensorStorage{}).smem_k)));
    static constexpr int mainloop_smem_padding = mainloop_smem_padding_ < 0 ? 0 : mainloop_smem_padding_;
    struct SharedStorage {
        struct TensorStorage : cute::aligned_struct<128> {
            union {
                struct {
                    cute::array<uint32_t, mainloop_smem_padding / sizeof(uint32_t)> padding_;
                    typename CollectiveMainloop::TensorStorage mainloop;
                };
                // We want smem_o to line up with the start of smem_v
                typename CollectiveEpilogue::TensorStorage epilogue;
            };
        } tensors;

        alignas(16) typename TileScheduler::SharedStorage smem_scheduler;

    };

    static constexpr int SharedStorageSize = sizeof(SharedStorage);

    // Device side arguments
    struct Arguments {
        MainloopArguments mainloop{};
        EpilogueArguments epilogue{};
        cutlass::KernelHardwareInfo hw_info{};
        TileSchedulerArguments scheduler{};
    };

    // Kernel entry point API
    struct Params {
        MainloopParams mainloop{};
        EpilogueParams epilogue{};
        cutlass::KernelHardwareInfo hw_info{};
        TileSchedulerParams scheduler{};
    };

    //
    // Methods
    //

    // Convert to underlying arguments. In this case, a simple copy for the aliased type.
    static
    Params
    to_underlying_arguments(Arguments const& args) {
        CUTLASS_TRACE_HOST("to_underlying_arguments():");

        // Get SM count if needed, otherwise use user supplied SM count
        int sm_count = args.hw_info.sm_count;
        if (sm_count <= 0) {
            CUTLASS_TRACE_HOST("  WARNING: Arguments do not include a valid SM count.\n"
                "  For optimal performance, populate the arguments KernelHardwareInfo struct with the SM count.");
            sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(args.hw_info.device_id);
        }

        CUTLASS_TRACE_HOST("to_underlying_arguments(): Setting persistent grid SM count to " << sm_count);

        cutlass::KernelHardwareInfo hw_info{args.hw_info.device_id, sm_count};
        return {
            CollectiveMainloop::to_underlying_arguments(args.mainloop),
            CollectiveEpilogue::to_underlying_arguments(args.epilogue),
            hw_info,
            TileScheduler::to_underlying_arguments(args.scheduler)
        };
    }

    // Computes the kernel launch grid shape based on runtime parameters
    static dim3
    get_grid_shape(Params const& params) {
        return TileScheduler::get_grid_shape(params.scheduler, params.hw_info.sm_count * MinBlocksPerMultiprocessor);
    }

    static dim3
    get_block_shape() {
        return dim3(MaxThreadsPerBlock, 1, 1);
    }

    CUTLASS_DEVICE
    void
    operator()(Params const& params, char* smem_buf) {

        static constexpr int kBlockM = get<0>(TileShape_MNK{});

        SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

        CollectiveMainloop mainloop;
        CollectiveEpilogue epilogue;

        TileScheduler scheduler(reinterpret_cast<typename TileScheduler::SharedStorage*>(&shared_storage.smem_scheduler));
        // Initialize matmul objects.
        TiledMma tiled_mma;

        scheduler.init_consumer();

        int warp_idx = cutlass::canonical_warp_idx_sync();
        CUTLASS_PRAGMA_NO_UNROLL
        for (auto work_tile_info = warp_idx == 0 ? scheduler.template get_initial_work</*IsProducerWarp=*/true>(params.scheduler) : scheduler.template get_initial_work</*IsProducerWarp=*/false>(params.scheduler);
             work_tile_info.is_valid(params.scheduler);
             work_tile_info = warp_idx == 0 ? scheduler.template get_next_work</*IsProducerWarp=*/true>(params.scheduler, work_tile_info) : scheduler.template get_next_work</*IsProducerWarp=*/false>(params.scheduler, work_tile_info)) {
            // Attention output (GEMM-II) accumulator.
            Tensor tOrO = partition_fragment_C(tiled_mma, select<0, 2>(TileShape_MNK{}));
            float softmax_scale_log2 = params.mainloop.softmax_scale_log2;
            // If there's tanh softcap, the scaling will be done before tanh.
            auto block_coord = work_tile_info.get_block_coord(params.scheduler);
            int const bidb = get<2>(block_coord);
            if constexpr (Is_FP8 && !Has_softcap) {
                int const bidh = get<1>(block_coord);
                int const bidh_kv = !PackGQA ? params.mainloop.qhead_per_khead_divmod.divide(bidh) : bidh;
                float const q_descale = params.mainloop.ptr_q_descale == nullptr ? 1.0f : params.mainloop.ptr_q_descale[bidb * get<0>(params.mainloop.stride_q_descale) + bidh_kv * get<1>(params.mainloop.stride_q_descale)];
                float const k_descale = params.mainloop.ptr_k_descale == nullptr ? 1.0f : params.mainloop.ptr_k_descale[bidb * get<0>(params.mainloop.stride_k_descale) + bidh_kv * get<1>(params.mainloop.stride_k_descale)];
                softmax_scale_log2 *= q_descale * k_descale;
            }
            flash::Softmax<2 * (2 * kBlockM / NumThreads), /*Max_offset=*/!Is_FP8 ? 0 : 8> softmax(softmax_scale_log2);

            SeqlenInfo_t seqlen_info{
                bidb,
                get<0>(params.mainloop.shape_Q),
                !PagedKV ? size<0>(params.mainloop.shape_K) : size<0>(params.mainloop.shape_K) * size<1>(params.mainloop.shape_pagetable),
                get<0>(params.mainloop.shape_K_new),
                params.mainloop.cu_seqlens_q, params.mainloop.cu_seqlens_k, params.mainloop.cu_seqlens_k_new,
                params.mainloop.seqused_q, params.mainloop.seqused_k, params.mainloop.leftpad_k,
                params.mainloop.seqlens_rotary
            };
            if constexpr (AppendKV) {
                bool tile_new_valid = mainloop.store_kv_new(
                    params.mainloop, threadIdx.x, shared_storage, seqlen_info, block_coord);
                if (tile_new_valid) { __syncthreads(); }
            }
            bool tile_valid = mainloop.mma(
                params.mainloop, tOrO, softmax, threadIdx.x, seqlen_info, block_coord,
                shared_storage);
            scheduler.prefetch_next_work(params.scheduler, work_tile_info);
            if (tile_valid) {
                // if (threadIdx.x == 128) { printf("Before epilogue, bid.x = %d, bid.y = %d, bid.z = %d, m_block = %d, bidb = %d, split_idx = %d\n", blockIdx.x, blockIdx.y, blockIdx.z, m_block, bidb, split_idx); }
                epilogue.store(params.epilogue, tOrO, softmax.row_sum, shared_storage, tiled_mma,
                               threadIdx.x, block_coord);
            } else {
                // Write 0 to gO and -inf to gLSE.
                epilogue.store_zero(params.epilogue, threadIdx.x, block_coord);
            }
        }

    }

};

} // namespace flash
