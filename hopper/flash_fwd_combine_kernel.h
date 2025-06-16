/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include "cute/tensor.hpp"

#include <cutlass/cutlass.h>
#include <cutlass/arch/memory.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>

#include "cutlass/arch/grid_dependency_control.h"

#include "seqlen.h"
#include "utils.h"

namespace flash {

using namespace cute;

template <class TileShape_MK_, int kLogMaxSplits_, int kNThreads, int AlignmentLSE_,
          bool Is_even_K, bool Varlen, class Element, class ElementPartial, class ArchTag_>
class FlashAttnFwdCombine {

public:

    // Type Aliases
    using TileShape_MK = TileShape_MK_;
    using ArchTag = ArchTag_;
    static constexpr int kMaxSplits = 1 << kLogMaxSplits_;
    static constexpr int AlignmentLSE = std::min(AlignmentLSE_, int(128 / 8 / sizeof(float)));
    static_assert(AlignmentLSE >= 1);
    static constexpr int kStages = 4;

    static_assert(ArchTag::kMinComputeCapability >= 75);
    static constexpr bool Has_cp_async = ArchTag::kMinComputeCapability >= 80;

    static constexpr uint32_t MaxThreadsPerBlock = kNThreads;
    static constexpr uint32_t MinBlocksPerMultiprocessor = 2;

    static constexpr int kBlockM = get<0>(TileShape_MK{});
    static constexpr int kBlockK = get<1>(TileShape_MK{});

    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(ElementPartial);
    static_assert(kBlockK % kGmemElemsPerLoad == 0, "kBlockK must be a multiple of kGmemElemsPerLoad");
    static constexpr int kBlockKGmem = kBlockK % 128 == 0 ? 128 : (kBlockK % 64 == 0 ? 64 : 32);
    static constexpr int kGmemThreadsPerRow = kBlockKGmem / kGmemElemsPerLoad;
    static_assert(MaxThreadsPerBlock % kGmemThreadsPerRow == 0, "MaxThreadsPerBlock must be a multiple of kGmemThreadsPerRow");
    using GmemCopyAtom = std::conditional_t<
        Has_cp_async,
        cute::Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, ElementPartial>,
        cute::Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementPartial>
    >;
    using GmemLayoutAtom = Layout<Shape <Int<MaxThreadsPerBlock / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                  Stride<Int<kGmemThreadsPerRow>, _1>>;
    static_assert(kBlockM % CUTE_STATIC_V(shape<0>(GmemLayoutAtom{})) == 0);
    using GmemTiledCopyAccum = decltype(
        make_tiled_copy(GmemCopyAtom{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));  // Val layout, 4 vals per load
    using GmemTiledCopy = decltype(
        make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));  // Val layout, 4 vals per load

    using AlignmentTypeLSE = cute::uint_byte_t<static_cast<int>(sizeof(float)) * AlignmentLSE>;
    static constexpr int kGmemElemsPerLoadLSE = sizeof(AlignmentTypeLSE) / sizeof(float);
    static_assert(kBlockM % kGmemElemsPerLoadLSE == 0, "kBlockM must be a multiple of kGmemElemsPerLoadLSE");
    static_assert(kBlockM % 8 == 0, "kBlockM must be a multiple of 8");
    static constexpr int kBlockMSmem = kBlockM % 128 == 0 ? 128 : (kBlockM % 64 == 0 ? 64 : (kBlockM % 32 == 0 ? 32 : (kBlockM % 16 == 0 ? 16 : 8)));
    static constexpr int kGmemThreadsPerRowLSE = kBlockMSmem / kGmemElemsPerLoadLSE;
    static_assert(MaxThreadsPerBlock % kGmemThreadsPerRowLSE == 0, "MaxThreadsPerBlock must be a multiple of kGmemThreadsPerRowLSE");
    using GmemLayoutAtomLSE = Layout<Shape <Int<MaxThreadsPerBlock / kGmemThreadsPerRowLSE>, Int<kGmemThreadsPerRowLSE>>,
                                     Stride<Int<kGmemThreadsPerRowLSE>, _1>>;
    static_assert(kMaxSplits % CUTE_STATIC_V(shape<0>(GmemLayoutAtomLSE{})) == 0);
    using GmemCopyAtomLSE = std::conditional_t<
        Has_cp_async,
        cute::Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<AlignmentTypeLSE>, float>,
        cute::Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<AlignmentLSE * sizeof(float) * 8>, float>
    >;
    using GmemTiledCopyLSE = decltype(
        make_tiled_copy(GmemCopyAtomLSE{},
                        GmemLayoutAtomLSE{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoadLSE>>>{}));  // Val layout, 4 vals per load

    // Otherwise we get IMA when some threads access sLSE, as we're not doing any masking
    static_assert((kBlockM * kMaxSplits * AlignmentLSE) % kNThreads == 0, "kNThreads must divide kBlockM * kMaxSplits * AlignmentLSE");
    // This works for kBlockMSmem = 8, 16, 32, 64, 128, no bank conflicts
    using SmemLSESwizzle = std::conditional_t<
        kBlockMSmem == 8,
        Swizzle<5, 0, 5>,
        std::conditional_t<kBlockMSmem == 16, Swizzle<4, 0, 4>, Swizzle<3, 2, 3>>
    >;
    using SmemLayoutAtomLSE =
        decltype(composition(SmemLSESwizzle{},
                 Layout<Shape<Int<8>, Int<kBlockMSmem>>,
                 Stride<Int<kBlockMSmem>, _1>>{}));
    using SmemLayoutLSE = decltype(tile_to_shape(SmemLayoutAtomLSE{}, Shape<Int<kMaxSplits>, Int<kBlockM>>{}));

    using SmemLayoutO = Layout<Shape<Int<kBlockM>, Int<kBlockK>, Int<kStages>>,
                               Stride<Int<kBlockK>, _1, Int<kBlockM * kBlockK>>>;

    // We want each column (kMaxSplits) to be processed by threads in the same warp.
    // To reduce the number of shuffles, we want as few threads on the same column as possible.
    // E.g., if kBlockM is divisible by 64, and there are 256 threads, we want 4 threads (0, 1, 2, 4) per column
    // have have 64 such quads.
    static_assert(MaxThreadsPerBlock % kBlockMSmem == 0, "MaxThreadsPerBlock must be a multiple of kBlockMSmem");
    static constexpr int kSmemThreadsPerColLSEt = MaxThreadsPerBlock / kBlockMSmem;
    static_assert(cutlass::NumThreadsPerWarp % kSmemThreadsPerColLSEt == 0, "kSmemThreadsPerColLSEt must divide NumThreadsPerWarp");
    using S2RLayoutAtomLSE = Layout<Shape<Int<kSmemThreadsPerColLSEt>, Int<MaxThreadsPerBlock / kSmemThreadsPerColLSEt>>>;
    using S2RTiledCopyLSE = decltype(make_tiled_copy(cute::Copy_Atom<cute::DefaultCopy, float>{}, S2RLayoutAtomLSE{}, Layout<_1>{}));

    using ShapeOPartial = cute::Shape<int32_t, int32_t, int32_t, int32_t, int32_t>;  // (seqlen, d, num_splits, head, batch)
    using StrideOPartial = cute::Stride<int64_t, _1, int64_t, int64_t, int64_t>;
    using ShapeLSEPartial = cute::Shape<int32_t, int32_t, int32_t, int32_t>;  // (seqlen, num_splits, head, batch)
    using StrideLSEPartial = cute::Stride<_1, int64_t, int64_t, int64_t>;  // (seqlen, num_splits, head, batch)
    using ShapeO = cute::Shape<int32_t, int32_t, int32_t, int32_t>;  // (seqlen, d, head, batch)
    using StrideO = cute::Stride<int64_t, _1, int64_t, int64_t>;
    using ShapeLSE = cute::Shape<int32_t, int32_t, int32_t>;  // (seqlen, head, batch)
    using StrideLSE = cute::Stride<_1, int64_t, int64_t>;  // (seqlen, head, batch)

    struct BlockCoord {
        int block_m;
        int block_k;
        int bidb;
    };

    struct SharedStorage : cute::aligned_struct<128> {
        cute::array_aligned<float, cute::cosize_v<SmemLayoutLSE>> smem_lse_partial;
        cute::array_aligned<int, kBlockM> smem_max_valid_split;
        cute::array_aligned<ElementPartial, cute::cosize_v<SmemLayoutO>> smem_o_partial;
        BlockCoord block_coord;
    };

    static constexpr int SharedStorageSize = sizeof(SharedStorage);

    // Device side arguments
    struct Arguments {
        int b;
        ElementPartial const* const ptr_O_partial;
        ShapeOPartial const shape_O_partial;
        StrideOPartial const stride_O_partial;
        float const* const ptr_LSE_partial;
        ShapeLSEPartial const shape_LSE_partial;
        StrideLSEPartial const stride_LSE_partial;
        Element* const ptr_O;
        StrideO const stride_O;
        float* const ptr_LSE;
        StrideLSE const stride_LSE;
        int const* const cu_seqlens = nullptr;
        int const* const seqused = nullptr;
        int const* const num_splits_dynamic_ptr = nullptr;
        int* const semaphore_to_reset = nullptr;
    };

    // Kernel entry point API
    struct CollectiveParams {
        int b;
        ElementPartial const* const ptr_O_partial;
        ShapeOPartial const shape_O_partial;
        StrideOPartial const stride_O_partial;
        float const* const ptr_LSE_partial;
        ShapeLSEPartial const shape_LSE_partial;
        StrideLSEPartial const stride_LSE_partial;
        Element* const ptr_O;
        StrideO const stride_O;
        float* const ptr_LSE;
        StrideLSE const stride_LSE;
        cutlass::FastDivmod seqlen_divmod, head_divmod;
        int const* const cu_seqlens = nullptr;
        int const* const seqused = nullptr;
        int const* const num_splits_dynamic_ptr = nullptr;
        int* const semaphore_to_reset = nullptr;
    };

    // Convert to underlying arguments. In this case, a simple copy for the aliased type.
    static
    CollectiveParams
    to_underlying_arguments(Arguments const& args) {
        assert(get<1>(args.shape_LSE_partial) <= kMaxSplits);
        return {
            args.b,
            args.ptr_O_partial,
            args.shape_O_partial,
            args.stride_O_partial,
            args.ptr_LSE_partial,
            args.shape_LSE_partial,
            args.stride_LSE_partial,
            args.ptr_O,
            args.stride_O,
            args.ptr_LSE,
            args.stride_LSE,
            cutlass::FastDivmod(get<0>(args.shape_LSE_partial)), cutlass::FastDivmod(get<2>(args.shape_LSE_partial)),
            args.cu_seqlens,
            args.seqused,
            args.num_splits_dynamic_ptr,
            args.semaphore_to_reset
        };
    }

    struct SchedulerArguments {
        int b;
        int seqlen_q;
        int total_q;
        int num_heads;
        int dv;
        int const* cu_seqlens_q;
        int const* seqused_q;
    };

    struct StaticTileScheduler {
        struct Params {};
        static Params to_underlying_arguments(SchedulerArguments const& args) { return {}; }

        SharedStorage& shared_storage;
        CUTE_DEVICE StaticTileScheduler(SharedStorage& shared_storage): shared_storage(shared_storage) {}

        static dim3 get_grid_shape(SchedulerArguments const& args) {
            unsigned int num_blocks_k = cute::ceil_div(args.dv, kBlockK);
            unsigned int num_blocks_m = cute::ceil_div(args.seqlen_q * args.num_heads, kBlockM);
            return {num_blocks_m, num_blocks_k, static_cast<unsigned int>(args.b)};
        }

        CUTE_DEVICE BlockCoord get_block_coord(Params const& params) {
            int block_m = blockIdx.x;
            int block_k = blockIdx.y;
            int bidb = blockIdx.z;
            return {block_m, block_k, bidb};
        }
    };

    struct StaticVarlenTileScheduler {
        //
        // For varlen we have two Scheduling algos:
        //  1) STANDARD, same as StaticTileScheduler
        //  2) LINEARIZE_M_AND_BATCH, this flattens the tiled M dimension and
        //     batch dimension into a linear tile index. The grid is then a
        //     2D grid of (tile_id, k_block). We then map the linear tile id
        //     to (m_block, bidb) in the get_block_coord function. This mapping
        //     is non-trivial since each batch element can have a different
        //     number of m_blocks. This has overhead when computing the block
        //     coordinates, but it is more efficient when prefills and decodes
        //     are mixed since in that case the STANDARD scheduling algo will
        //     have a lot of empty (no work) blocks in the grid.
        //

        enum SchedulingAlgo {
            STANDARD,           // Same as StaticTileScheduler
            LINEARIZE_M_AND_BATCH,  // Linearize the M and batch dimensions into a single tile index
        };

        struct Params {
            int b;
            int num_heads;
            int const* const cu_seqlens_q;
            int const* const seqused_q;
            SchedulingAlgo algo;
        };

        SharedStorage& shared_storage;
        CUTE_DEVICE StaticVarlenTileScheduler(SharedStorage& shared_storage): shared_storage(shared_storage) {}

        static SchedulingAlgo choose_scheduling_algo(SchedulerArguments const& args) {
            // Choose the scheduling algorithm based on how dense the grid of tiles that
            // do actual work is. If the grid is more then 50% sparse, we linearize the M
            // and batch. If the grid is more than 50% dense, we use the standard scheduling
            // algorithm since its more efficient at calculating the block coordinates.
            // NOTE: in varlen case args.seqlen_q is the max seqlen_q across all batches
            // use lower bound to estimate when the density is more than 50%
            int lower_bound_on_non_empty_tiles = cute::ceil_div(args.total_q, kBlockM);
            int grid_size = args.b * cute::ceil_div(args.seqlen_q, kBlockM);
            return 2 * lower_bound_on_non_empty_tiles >= grid_size ? 
                SchedulingAlgo::STANDARD : 
                SchedulingAlgo::LINEARIZE_M_AND_BATCH;
        }

        static Params to_underlying_arguments(SchedulerArguments const& args) { 
            return {
                args.b,
                args.num_heads,
                args.cu_seqlens_q,
                args.seqused_q,
                choose_scheduling_algo(args)
            }; 
        }

        static dim3 get_grid_shape(SchedulerArguments const& args) {
            unsigned int num_blocks_k = cute::ceil_div(args.dv, kBlockK);

            switch (choose_scheduling_algo(args)) {
            case SchedulingAlgo::STANDARD: {
                unsigned int num_blocks_k = cute::ceil_div(args.dv, kBlockK);
                unsigned int num_blocks_m = cute::ceil_div(args.seqlen_q * args.num_heads, kBlockM);
                return {num_blocks_m, num_blocks_k, static_cast<unsigned int>(args.b)};
            }
            case SchedulingAlgo::LINEARIZE_M_AND_BATCH: {
                // rough worst case upper bound on the number of blocks required 
                //  (assuming each batch has an additional partial block)
                unsigned int num_blocks_m = cute::ceil_div(args.total_q * args.num_heads, kBlockM) + args.b;
                return {num_blocks_m, num_blocks_k, 1};
            }}

            // rough worst case upper bound on the number of blocks required 
            //  (assuming each batch has an additional partial block)
            unsigned int num_blocks_m = cute::ceil_div(args.total_q * args.num_heads, kBlockM) + args.b;
            return {num_blocks_m, num_blocks_k, 1};
        }

        CUTE_DEVICE BlockCoord get_block_coord_linearized_m_and_batch(Params const& params) {
            int num_heads = params.num_heads;
            int curr_tile_id = blockIdx.x;

            // Scan through the batches find the batch that contains the current
            // tile_id. Compute using only the first warp of the block.
            if (threadIdx.x < 32) {
                // We compute linearized tile index start and ends for each batch
                // in groups of 32 in parallel
                int group_start_bidb = -(cutlass::NumThreadsPerWarp);
                int group_end_bidb = 0;
                int group_end_tile_id = 0;
                int group_start_tile_id = 0;
                int group_total_num_tiles = 0;

                int local_num_m_blocks = 0;
                int local_num_m_blocks_cumulative = 0;

                do {
                    group_start_bidb += cutlass::NumThreadsPerWarp;
                    group_end_bidb += cutlass::NumThreadsPerWarp;

                    auto get_num_m_blocks = [&](int bidb) {
                        if (bidb >= params.b) return 0;
                        flash::SeqlenInfo<Varlen, kBlockM> seqlen_info{bidb, 0, params.cu_seqlens_q, params.seqused_q};
                        return cute::ceil_div(seqlen_info.seqlen * num_heads, Int<kBlockM>{}());
                    };

                    // Cumulative number of blocks for the next 31 batches
                    local_num_m_blocks = get_num_m_blocks(group_start_bidb + threadIdx.x);
                    local_num_m_blocks_cumulative = warp_prefix_sum(local_num_m_blocks);
                    // Total number of blocks for the next 32 batches
                    group_total_num_tiles = warp_shfl_get_last(local_num_m_blocks_cumulative);
                    
                    group_start_tile_id = group_end_tile_id;
                    group_end_tile_id += group_total_num_tiles;
                } while (curr_tile_id >= group_end_tile_id && group_end_bidb < params.b);
                
                int local_batch_end_tile_id = group_start_tile_id + local_num_m_blocks_cumulative;
                // Find the last batch idx in the group where `local_batch_end_tile_id <= curr_tile_id`
                // these values below are now common to all threads in the warp
                int batch_idx_in_group = warp_last_true_laneid(local_batch_end_tile_id <= curr_tile_id);
                int batch_num_m_blocks = warp_shfl_get(local_num_m_blocks, batch_idx_in_group);
                int batch_m_start_tile_id = group_start_tile_id + (batch_idx_in_group > 0 ? 
                    warp_shfl_get(local_num_m_blocks_cumulative, batch_idx_in_group - 1) : 0);
                
                int bidb = group_start_bidb + batch_idx_in_group;
                int block_m = curr_tile_id - batch_m_start_tile_id;
                // NOTE(lucas): not sure why this causes a block_k unused warning
                //  just inlined `blockIdx.y` to suppress the warning
                // int block_k = blockIdx.y;
                // shared_storage.block_coord = {block_m, block_k, bidb};
                BlockCoord block_coord{block_m, static_cast<int>(blockIdx.y), bidb};
                if (threadIdx.x == 0) { shared_storage.block_coord = block_coord; }
            }

            __syncthreads();
            return shared_storage.block_coord;
        }


        CUTE_DEVICE BlockCoord get_block_coord_standard(Params const& params) {
            int block_m = blockIdx.x;
            int block_k = blockIdx.y;
            int bidb = blockIdx.z;
            return {block_m, block_k, bidb};
        }

        CUTE_DEVICE BlockCoord get_block_coord(Params const& params) {
            switch (params.algo) {
                case SchedulingAlgo::STANDARD:
                    return get_block_coord_standard(params);
                case SchedulingAlgo::LINEARIZE_M_AND_BATCH:
                    return get_block_coord_linearized_m_and_batch(params);
            }
            return {0, 0, 0};  // Should never reach here
        }
    };

    using TileScheduler = std::conditional_t<
        Varlen,
        StaticVarlenTileScheduler,
        StaticTileScheduler
    >;

    using SchedulerParams = typename TileScheduler::Params;

    struct Params {
        CollectiveParams params;
        SchedulerParams scheduler_params;
    };

    CUTLASS_DEVICE
    void
    operator()(Params const& kernel_params, char* smem_buf) {
        CollectiveParams const& params = kernel_params.params;

        SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);
        TileScheduler tile_scheduler{shared_storage};

        Tensor sLSE = make_tensor(make_smem_ptr(shared_storage.smem_lse_partial.data()), SmemLayoutLSE{});
        Tensor sMaxValidSplit = make_tensor(make_smem_ptr(shared_storage.smem_max_valid_split.data()), Shape<Int<kBlockM>>{});
        Tensor sO = make_tensor(make_smem_ptr(shared_storage.smem_o_partial.data()), SmemLayoutO{});

        int const thread_idx = threadIdx.x;

        BlockCoord block_coord = tile_scheduler.get_block_coord(kernel_params.scheduler_params);

        int const m_block = block_coord.block_m;
        int const k_block = block_coord.block_k;
        int const batch = block_coord.bidb;

        if (params.semaphore_to_reset && threadIdx.x == 0 && blockIdx.x == gridDim.x - 1 && blockIdx.y == gridDim.y - 1 && blockIdx.z == gridDim.z - 1) {
            cutlass::arch::wait_on_dependent_grids();
            *params.semaphore_to_reset = 0;
        }

        flash::SeqlenInfo<Varlen, kBlockM> seqlen_info{batch, size<0>(params.shape_LSE_partial), params.cu_seqlens, params.seqused};
        int const offset = seqlen_info.offset;
        int const seqlen = seqlen_info.seqlen;
        int max_idx = seqlen * get<2>(params.shape_LSE_partial);

        bool block_coord_valid = 
            block_coord.block_m < cute::ceil_div(max_idx, Int<kBlockM>{}) &&
            block_coord.bidb < params.b;
        if (!block_coord_valid) { return; }

        int const num_splits = params.num_splits_dynamic_ptr ? params.num_splits_dynamic_ptr[batch] : get<1>(params.shape_LSE_partial);
        if (num_splits <= 1) { return; }

        cutlass::FastDivmod seqlen_divmod_dynamic(seqlen);

        // Step 1: load LSE_partial from gmem -> smem
        Tensor mLSEpartial = make_tensor(make_gmem_ptr(params.ptr_LSE_partial + offset * get<0>(params.stride_LSE_partial)),
                                         select<1, 0, 2, 3>(params.shape_LSE_partial),
                                         select<1, 0, 2, 3>(params.stride_LSE_partial))(_, _, _, !Varlen ? batch : 0);  // (num_splits, seqlen, head)
        Tensor mLSEpartial_copy = cute::tiled_divide(mLSEpartial, Shape<_1, Int<kGmemElemsPerLoadLSE>>{});
        GmemTiledCopyLSE gmem_tiled_copy_LSE;
        auto gmem_thr_copy_LSE = gmem_tiled_copy_LSE.get_thread_slice(thread_idx);
        Tensor tLSEsLSE = gmem_thr_copy_LSE.partition_D(sLSE);

        // Construct identity layout for sLSE
        Tensor cLSE = make_identity_tensor(make_shape(size<0>(sLSE), size<1>(sLSE)));    // (NUM_SPLITS, BLK_M) -> (num_splits, blk_m)
        // Repeat the partitioning with identity layouts
        Tensor tLSEcLSE = gmem_thr_copy_LSE.partition_S(cLSE);

        cutlass::arch::wait_on_dependent_grids();

        #pragma unroll
        for (int m = 0; m < size<2>(tLSEcLSE); ++m) {
            int mi = int(get<1>(tLSEcLSE(_0{}, _0{}, m)));
            int idx = m_block * kBlockM + mi;
            if (idx < max_idx) {
                int m_idx, bidh;
                if constexpr (!Varlen) {
                    bidh = params.seqlen_divmod.divmod(m_idx, idx);
                } else {
                    bidh = seqlen_divmod_dynamic.divmod(m_idx, idx);
                }
                Tensor mLSEpartial_cur_copy = mLSEpartial_copy(_, _, m_idx, bidh);
                #pragma unroll
                for (int s = 0; s < size<1>(tLSEcLSE); ++s) {
                    int si = get<0>(tLSEcLSE(_0{}, s, _0{}));
                    // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && thread_idx < 32) { printf("thread_idx = %d, m = %d, s = %d, addr = %p, bank = %d\n", thread_idx, m, s, reinterpret_cast<float *>(&(tLSEsLSE(_0{}, s, m))), reinterpret_cast<int>(&(tLSEsLSE(_0{}, s, m))) / 4 % 32);}
                    if (si < num_splits) {
                        cute::copy(gmem_tiled_copy_LSE, mLSEpartial_cur_copy(_, si), tLSEsLSE(_, s, m));
                    } else {
                        cute::fill(tLSEsLSE(_, s, m), -INFINITY);
                    }
                }
            } else {
                // We don't need to zero out the rest of the LSEs, as we will not write the output to gmem
                // cute::fill(tLSEsLSE(_, _, m), -INFINITY);
            }
        }
        if constexpr (Has_cp_async) { cute::cp_async_fence(); }

        // Step 2: Load O_partial from gmem -> smem for split = 0, 1, ..., kStages - 2.
        // We want these async loads to be in flight as we compute the LSE.
        GmemTiledCopyAccum gmem_tiled_copy_O_partial;
        auto gmem_thr_copy_O_partial = gmem_tiled_copy_O_partial.get_thread_slice(thread_idx);
        // Construct identity layout for gO
        Tensor cO = cute::make_identity_tensor(TileShape_MK{});  // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_O_partial.partition_D(cO);
        Tensor mOpartial = make_tensor(make_gmem_ptr(params.ptr_O_partial + offset * get<0>(params.stride_O_partial)),
                                       params.shape_O_partial, params.stride_O_partial)(_, _, _, _, !Varlen ? batch : 0);  // (seqlen, d, num_splits, head)

        // Precompute these values to avoid recomputing them in the loop
        Tensor tOmidx = make_tensor<int>(make_shape(size<1>(tOcO)));
        Tensor tObidh = make_tensor<int>(make_shape(size<1>(tOcO)));
        Tensor tOrOptr = make_tensor<ElementPartial const*>(make_shape(size<1>(tOcO)));
        #pragma unroll
        for (int m = 0; m < size<1>(tOcO); ++m) {
            int mi = get<0>(tOcO(_0{}, m, _0{}));
            int idx = m_block * kBlockM + mi;
            if constexpr (!Varlen) {
                tObidh(m) = params.seqlen_divmod.divmod(tOmidx(m), idx);
            } else {
                tObidh[m] = seqlen_divmod_dynamic.divmod(tOmidx(m), idx);
            }
            tOrOptr[m] = &mOpartial(tOmidx(m), k_block * kBlockK, _0{}, tObidh(m));
            if (idx >= max_idx) {
                tObidh[m] = -1;
            }
        }

        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOcO)));
        if constexpr (!(Is_even_K)) {
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(_0{}, _0{}, k)) < get<1>(params.shape_O_partial) - k_block * kBlockK; }
        }

        Tensor tOsOpartial = gmem_thr_copy_O_partial.partition_D(sO);

        auto load_O_partial = [&] (int split, int stage) {
            Tensor tOsOpartial_cur = tOsOpartial(_, _, _, stage);
            #pragma unroll
            for (int m = 0; m < size<1>(tOcO); ++m) {
                if (tObidh(m) >= 0)  {
                    Tensor mOpartial_cur = make_tensor(make_gmem_ptr(tOrOptr[m]), mOpartial(_0{}, _, _, _0{}).layout());
                    Tensor mOpartial_cur_copy = cute::tiled_divide(mOpartial_cur, Shape<Int<kGmemElemsPerLoad>>{});
                    #pragma unroll
                    for (int k = 0; k < size<2>(tOcO); ++k) {
                        int k_idx = get<1>(tOcO(_0{}, _0{}, k)) / kGmemElemsPerLoad;
                        if (Is_even_K || tOpO(k)) {
                            cute::copy(gmem_tiled_copy_O_partial, mOpartial_cur_copy(_, k_idx, split), tOsOpartial_cur(_, m, k));
                        }
                    }
                }
            }
        };

        for (int s = 0; s < kStages - 1; ++s) {
            if (s < num_splits) { load_O_partial(s, s); }
            if constexpr (Has_cp_async) { cute::cp_async_fence(); }
        }

        // Step 3: load and transpose LSE_partial from smem -> rmem
        if constexpr (Has_cp_async) { cutlass::arch::cp_async_wait<kStages - 1>(); }
        __syncthreads();

        S2RTiledCopyLSE s2r_tiled_copy_LSE;
        auto s2r_thr_copy_LSE = s2r_tiled_copy_LSE.get_thread_slice(thread_idx);
        Tensor ts2rsLSE = s2r_thr_copy_LSE.partition_S(sLSE);
        Tensor ts2rrLSE = make_fragment_like(ts2rsLSE);
        cute::copy(s2r_tiled_copy_LSE, ts2rsLSE, ts2rrLSE);

        // Step 4: compute the final LSE along the split dimension
        Tensor lse_sum = make_tensor<float>(make_shape(size<2>(ts2rrLSE)));
        Tensor ts2rcLSE = s2r_thr_copy_LSE.partition_D(cLSE);
        // We compute the max valid split for each row to short-circuit the computation later
        Tensor max_valid_split = make_tensor<int>(make_shape(size<2>(ts2rrLSE)));
        static_assert(CUTE_STATIC_V(size<0>(ts2rrLSE)) == 1);
        #pragma unroll
        for (int m = 0; m < size<2>(ts2rrLSE); ++m) {
            float lse_max = ts2rrLSE(_0{}, _0{}, m);
            #pragma unroll
            for (int s = 1; s < size<1>(ts2rrLSE); ++s) { lse_max = max(lse_max, ts2rrLSE(_0{}, s, m)); }
            MaxOp<float> max_op;
            lse_max = Allreduce<kSmemThreadsPerColLSEt>::run(lse_max, max_op);
            int max_valid_idx = -1;
            #pragma unroll
            for (int s = 0; s < size<1>(ts2rrLSE); ++s) {
                if (ts2rrLSE(_0{}, s, m) != -INFINITY) { max_valid_idx = get<0>(ts2rcLSE(_0{}, s, _0{})); }
            }
            MaxOp<int> max_int_op;
            max_valid_split[m] = Allreduce<kSmemThreadsPerColLSEt>::run(max_valid_idx, max_int_op);
            float lse_max_cur = lse_max == -INFINITY ? 0.0f : lse_max;  // In case all local LSEs are -inf
            float lse_sum_cur = 0.f;
            #pragma unroll
            for (int s = 0; s < size<1>(ts2rrLSE); ++s) {
                float scale = expf(ts2rrLSE(_0{}, s, m) - lse_max_cur);
                lse_sum_cur += scale;
                // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && thread_idx < 32) { printf("thread_idx = %d, m = %d, s = %d, addr = %p, bank = %d\n", thread_idx, m, s, reinterpret_cast<float *>(&(ts2rsLSE(_0{}, s, m))), reinterpret_cast<int>(&(ts2rsLSE(_0{}, s, m))) / 4 % 32);}
                // ts2rsLSE(_0{}, m, s) = scale;
                ts2rrLSE(_0{}, s, m) = scale;
            }
            SumOp<float> sum_op;
            lse_sum_cur = Allreduce<kSmemThreadsPerColLSEt>::run(lse_sum_cur, sum_op);
            lse_sum(m) = logf(lse_sum_cur) + lse_max;
            float inv_sum = (lse_sum_cur == 0.f || lse_sum_cur != lse_sum_cur) ? 0.f : 1.f / lse_sum_cur;
            #pragma unroll
            for (int s = 0; s < size<1>(ts2rrLSE); ++s) { ts2rrLSE(_0{}, s, m) *= inv_sum; }
        }
        // Store the scales exp(lse - lse_logsum) back to smem
        cute::copy(s2r_tiled_copy_LSE, ts2rrLSE, ts2rsLSE);

        // Store max_valid_split to smem
        #pragma unroll
        for (int m = 0; m < size<2>(ts2rrLSE); ++m) {
            if (get<0>(ts2rcLSE(_0{}, _0{}, m)) == 0) {  // Only the thread responsible for s=0 writes to smem
                int mi = int(get<1>(ts2rcLSE(_0{}, _0{}, m)));
                if (mi < kBlockM) { sMaxValidSplit[mi] = max_valid_split[m]; }
            }
        }

        // Step 5: store final LSE back to gmem
        if (k_block == 0) {
            auto shape_LSE = select<0, 2, 3>(params.shape_LSE_partial);
            Tensor mLSE = make_tensor(make_gmem_ptr(params.ptr_LSE + offset * get<0>(params.stride_LSE)), shape_LSE, params.stride_LSE)(_, _, !Varlen ? batch : 0);
            #pragma unroll
            for (int m = 0; m < size<2>(ts2rrLSE); ++m) {
                if (get<0>(ts2rcLSE(_0{}, _0{}, m)) == 0) {  // Only the thread responsible for s=0 writes to gmem
                    int mi = int(get<1>(ts2rcLSE(_0{}, _0{}, m)));
                    int idx = m_block * kBlockM + mi;
                    if (idx < max_idx) {
                        int m_idx, bidh;
                        if constexpr (!Varlen) {
                            bidh = params.seqlen_divmod.divmod(m_idx, idx);
                        } else {
                            bidh = seqlen_divmod_dynamic.divmod(m_idx, idx);
                        }
                        // printf("thread_idx = %d, m = %d, mi = %d, idx = %d, m_idx = %d, bidh = %d, bidb = %d, lse_sum = %f\n", thread_idx, m, mi, idx, m_idx, bidh, bidb, lse_sum(m));
                        mLSE(m_idx, bidh) = lse_sum(m);
                    }
                }
            }
        }

        // Step 6: read O_partial from gmem -> smem -> rmem and accumulate the final O
        __syncthreads();
        int thr_max_valid_split = sMaxValidSplit[get<0>(tOcO(_0{}, _0{}, _0{}))];
        #pragma unroll
        for (int m = 1; m < size<1>(tOcO); ++m) { thr_max_valid_split = max(thr_max_valid_split, sMaxValidSplit[get<0>(tOcO(_0{}, m, _0{}))]); }
        Layout tOrOpartial_layout = gmem_thr_copy_O_partial.partition_S(make_tensor<ElementPartial>(TileShape_MK{})).layout();
        Tensor tOrOpartial = make_fragment_like<ElementPartial>(tOrOpartial_layout);
        Tensor tOrO = make_fragment_like<float>(tOrOpartial);
        clear(tOrO);
        int stage_load = kStages - 1, stage_compute = 0;
        #pragma unroll 4 // Already tuned for speed
        for (int s = 0; s <= thr_max_valid_split; ++s) {
            Tensor scale = make_tensor<float>(make_shape(size<1>(tOrOpartial)));
            #pragma unroll
            for (int m = 0; m < size<1>(tOrOpartial); ++m) { scale(m) = sLSE(s, get<0>(tOcO(_0{}, m, _0{}))); }

            if (s + kStages - 1 <= thr_max_valid_split) { load_O_partial(s + kStages - 1, stage_load); }
            if constexpr (Has_cp_async) { cute::cp_async_fence(); }
            stage_load = stage_load < kStages - 1 ? stage_load + 1 : 0;
            if constexpr (Has_cp_async) { cutlass::arch::cp_async_wait<kStages - 1>(); }
            // We don't need __syncthreads() because each thread is just reading its own data from smem
            cute::copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementPartial>{},
                       tOsOpartial(_, _, _, stage_compute), tOrOpartial);
            stage_compute = stage_compute < kStages - 1 ? stage_compute + 1 : 0;

            #pragma unroll
            for (int m = 0; m < size<1>(tOrOpartial); ++m) {
                if (tObidh(m) >= 0 && scale(m) > 0.f) {
                    #pragma unroll
                    for (int k = 0; k < size<2>(tOrOpartial); ++k) {
                        if (Is_even_K || tOpO(k)) {
                            Tensor rOpartial = make_tensor_like<float>(tOrOpartial(_, m, k));
                            flash::convert_type_out(tOrOpartial(_, m, k), rOpartial);
                            #pragma unroll
                            for (int i = 0; i < size<0>(tOrOpartial); ++i) {
                                tOrO(i, m, k) += scale(m) * rOpartial[i];
                            }
                        }
                    }
                }
            }
        }

        // Step 7: Write the final O to gmem
        Tensor rO = make_tensor_like<Element>(tOrO);
        flash::convert_type_out(tOrO, rO);
        auto shape_O = make_shape(get<0>(params.shape_O_partial), get<1>(params.shape_O_partial) - k_block * kBlockK, get<3>(params.shape_O_partial), get<4>(params.shape_O_partial));
        Tensor mO = make_tensor(make_gmem_ptr(params.ptr_O + offset * get<0>(params.stride_O) + k_block * kBlockK * get<1>(params.stride_O)),
                                shape_O, params.stride_O)(_, _, _, !Varlen ? batch : 0);
        Tensor mO_copy = cute::tiled_divide(mO, Shape<_1, Int<kGmemElemsPerLoad>>{});
        GmemTiledCopy gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(thread_idx);

        #pragma unroll
        for (int m = 0; m < size<1>(tOcO); ++m) {
            if (tObidh(m) >= 0)  {
                #pragma unroll
                for (int k = 0; k < size<2>(tOcO); ++k) {
                    int k_idx = get<1>(tOcO(_0{}, _0{}, k)) / kGmemElemsPerLoad;
                    if (Is_even_K || tOpO(k)) {
                        cute::copy(gmem_tiled_copy_O, rO(_, m, k), mO_copy(_, tOmidx(m), k_idx, tObidh(m)));
                    }
                }
            }
        }

    }

};

} // namespace flash
