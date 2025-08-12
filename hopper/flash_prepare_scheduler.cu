/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#include <cub/cub.cuh>
#include "cutlass/fast_math.h"
#include "cutlass/barrier.h"
#include "cutlass/arch/barrier.h"

#include "cutlass/arch/grid_dependency_control.h"

#include "flash.h"

#include "static_switch.h"

namespace flash {

// needs (tensor size = batches):
// 1. batch_idx_ptr: virtual_batch_idx -> batch_idx
// 2. num_m_blocks_ptr: virtual_batch_idx -> num_m_blocks[batch_idx]
// 3. num_splits_dynamic_ptr: virtual_batch_idx -> num_splits[batch_idx]

// Custom comparison functor for descending order
template <typename DataType>
struct CustomMore
{
    __device__ bool operator()(const DataType &lhs, const DataType &rhs)
    {
        return lhs > rhs;
    }
};

// Specialization for int2
template <>
struct CustomMore<int2> {
    __device__ bool operator()(const int2& lhs, const int2& rhs) const {
        return lhs.x > rhs.x;
    }
};

// Specialization for int4
template <>
struct CustomMore<int4> {
    __device__ bool operator()(const int4& lhs, const int4& rhs) const {
        return lhs.x > rhs.x;
    }
};

template <bool Sort>
__global__ void prepare_varlen_num_blocks_kernel(
        int seqlen_q_static, int seqlen_k_static, int seqlen_k_new_static,
        int const* const cu_seqlens_q, int const* const cu_seqlens_k, int const* const cu_seqlens_k_new,
        int const* const seqused_q, int const* const seqused_k, int const* const leftpad_k_ptr,
        int num_batch, int num_head, int qhead_per_khead, int num_sm, int num_splits_static,
        cutlass::FastDivmod blockm_divmod, cutlass::FastDivmod blockn_divmod,
        int* const tile_count_semaphore,
        int* const num_m_blocks_ptr, // virtual_batch_idx -> num_m_blocks[batch_idx]
        int* const num_splits_dynamic_ptr,
        int* const batch_idx_ptr,
        bool enable_pdl) {

    static constexpr int kNumBatchPerWarp = cutlass::NumThreadsPerWarp - 1;
    static constexpr int kSmemSize = 1;
    static constexpr int BLOCK_THREADS = 1024;
    static constexpr int ITEMS_PER_THREAD = 1;
    static_assert(BLOCK_THREADS * ITEMS_PER_THREAD == 1024);
    using BlockMergeSort = cub::BlockMergeSort<int4, BLOCK_THREADS, ITEMS_PER_THREAD>;

    // Assume that there's only one block in the grid
    __shared__ int total_blocks_smem[kSmemSize];

    // Allocate shared memory for BlockMergeSort operations
    __shared__ typename BlockMergeSort::TempStorage temp_storage;

    // There's only 1 block in the grid, so might as well start launching the main attn kernel
    if (enable_pdl) { cutlass::arch::launch_dependent_grids(); }

    if (threadIdx.x < kSmemSize) { total_blocks_smem[threadIdx.x] = 0; }
    __syncthreads();

    if (threadIdx.x == 0 && tile_count_semaphore) { *tile_count_semaphore = 0; }

    int lane = threadIdx.x % cutlass::NumThreadsPerWarp;

    auto get_num_m_blocks = [&](int batch_idx) {
        int seqlen;
        if (seqused_q) {
            seqlen = batch_idx < num_batch ? seqused_q[batch_idx] : 0;
        } else if (cu_seqlens_q) {
            int cur_cu_seqlen = batch_idx <= num_batch ? cu_seqlens_q[batch_idx] : 0;
            int next_cu_seqlen = __shfl_down_sync(0xffffffff, cur_cu_seqlen, 1);
            seqlen = next_cu_seqlen - cur_cu_seqlen;
        } else {
            seqlen = seqlen_q_static;
        }
        seqlen *= qhead_per_khead;
        return batch_idx < num_batch && lane < kNumBatchPerWarp
            ? blockm_divmod.div(seqlen + blockm_divmod.divisor - 1) : 0;
    };

    auto get_num_n_blocks = [&](int batch_idx) {
        int leftpad_k = batch_idx < num_batch && leftpad_k_ptr != nullptr ? leftpad_k_ptr[batch_idx] : 0;
        int seqlen;
        if (seqused_k) {
            seqlen = batch_idx < num_batch ? seqused_k[batch_idx] : 0;
        } else if (cu_seqlens_k) {
            int cur_cu_seqlen = batch_idx <= num_batch ? cu_seqlens_k[batch_idx] : 0;
            int next_cu_seqlen = __shfl_down_sync(0xffffffff, cur_cu_seqlen, 1);
            seqlen = next_cu_seqlen - cur_cu_seqlen;
        } else {
            seqlen = seqlen_k_static;
        }
        int seqlen_new;
        if (cu_seqlens_k_new) {
            int cur_cu_seqlen_new = batch_idx <= num_batch ? cu_seqlens_k_new[batch_idx] : 0;
            int next_cu_seqlen_new = __shfl_down_sync(0xffffffff, cur_cu_seqlen_new, 1);
            seqlen_new = next_cu_seqlen_new - cur_cu_seqlen_new;
        } else {
            seqlen_new = seqlen_k_new_static;
        }
        // if (threadIdx.x == 0) { printf("seqlen = %d, seqlen_new = %d, leftpad_k = %d\n", seqlen, seqlen_new, leftpad_k); }
        seqlen = seqlen - leftpad_k + seqlen_new;
        return batch_idx < num_batch && lane < kNumBatchPerWarp
            ? blockn_divmod.div(seqlen + blockn_divmod.divisor - 1) : 0;
    };

    int warp_idx = threadIdx.x / cutlass::NumThreadsPerWarp;
    int bidb_start = kNumBatchPerWarp * warp_idx;
    int batch_idx = lane + bidb_start;
    int num_m_blocks = get_num_m_blocks(batch_idx);
    int num_n_blocks = get_num_n_blocks(batch_idx);

    int total_blocks = num_m_blocks * num_n_blocks;
    // Warp sum
    #pragma unroll
    for (int i = cutlass::NumThreadsPerWarp / 2; i >= 1; i /= 2) {
        total_blocks += __shfl_down_sync(0xffffffff, total_blocks, i);
    }
    if (lane == 0) { atomicAdd(total_blocks_smem, total_blocks); }
    __syncthreads();
    total_blocks = total_blocks_smem[0];
    // 10% margin
    int blocks_per_sm = static_cast<int>(ceilf(float(total_blocks) * 1.1f * float(num_head) / float(num_sm)));
    // blocks_per_sm = std::max(1, blocks_per_sm);  // 1 is the minimum number of blocks per SM
    int num_splits_dynamic = std::max(std::min((num_n_blocks + blocks_per_sm - 1) / blocks_per_sm, num_splits_static), 1);

    if constexpr(Sort) {
        num_n_blocks = cute::ceil_div(num_n_blocks, num_splits_dynamic); // num_n_blocks per batch accounting for splits

        // At this point, I have total blocks in shared memory, and each thread has its num_m_blocks, num_n_blocks, and num_splits
        // thread batch_idx = bidb_start + lane, lane 31 is a dummy.

        // Goal: sort batches by num_n_blocks
        if(lane == kNumBatchPerWarp) {
            num_n_blocks = -1; // sort last
        }
        int4 thread_keys[ITEMS_PER_THREAD]; // 1 item per thread for now
        thread_keys[0] = make_int4(num_n_blocks, num_m_blocks, num_splits_dynamic, batch_idx);

        BlockMergeSort(temp_storage).Sort(thread_keys, CustomMore<int4>());

        if (threadIdx.x < num_batch) {
            num_m_blocks_ptr[threadIdx.x] = thread_keys[0].y;
            num_splits_dynamic_ptr[threadIdx.x] = thread_keys[0].z;
            batch_idx_ptr[threadIdx.x] = thread_keys[0].w;
        }
    } else {
        if (bidb_start + lane < num_batch && lane < kNumBatchPerWarp) {
            num_splits_dynamic_ptr[bidb_start + lane] = num_splits_dynamic;
            // printf("idx = %d, num_m_blocks = %d, num_n_blocks = %d, num_split_static = %d, num_splits_dynamic = %d\n", bidb_start + lane, num_m_blocks_ptr[bidb_start + lane], num_n_blocks, num_splits_static, num_splits_dynamic);
        }
    }
}

} // flash

void prepare_varlen_num_blocks(Flash_fwd_params &params, cudaStream_t stream, bool packgqa,
                               int blockM, int blockN, bool enable_pdl, bool sort_batches) {
    // Only support batch <= 992 (32 warps, each with 31 batches)
    int qhead_per_khead = !packgqa ? 1 : cutlass::ceil_div(params.h, params.h_k);
    printf("sort_batches = %d.\n", sort_batches);
    BOOL_SWITCH(sort_batches, Sort, [&] {
        if constexpr(Sort) {
            printf("Sorting it!.\n");
        } else {
            printf("No sorting.\n");
        }
        flash::prepare_varlen_num_blocks_kernel<Sort><<<1 /*grid*/, 1024 /*block*/, 0, stream>>>(
            params.seqlen_q, params.seqlen_k, params.seqlen_knew,
            params.cu_seqlens_q, params.cu_seqlens_k, params.cu_seqlens_knew,
            params.seqused_q, params.seqused_k, params.leftpad_k,
            params.b, !packgqa ? params.h : params.h_k, qhead_per_khead, params.num_sm, params.num_splits,
            cutlass::FastDivmod(blockM), cutlass::FastDivmod(blockN),
            params.tile_count_semaphore,
            params.num_m_blocks_ptr,
            params.num_splits_dynamic_ptr,
            params.batch_idx_ptr,
            enable_pdl);
    });
}
