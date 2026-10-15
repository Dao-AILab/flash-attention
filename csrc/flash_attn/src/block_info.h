/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include "namespace_config.h"
namespace FLASH_NAMESPACE {

////////////////////////////////////////////////////////////////////////////////////////////////////

// The blocks of kBlock rows of a varlen batch of `total` rows in `num_batch` sequences, numbered
// in its padded block space: sequence b's blocks start at cu_seqlens[b] / kBlock + b. Each sequence
// leaves at most one gap after its last block, so the space spans this many blocks however long
// the longest sequence is, where a grid of num_batch * ceil_div(max_seqlen, kBlock) gives every
// (sequence, block) pair a CTA, most of which find their sequence already ended.
inline int varlen_num_blocks(int total, int num_batch, int block) {
    return (total + block - 1) / block + num_batch;
}

// The `block_idx`-th block of the padded block space (varlen_num_blocks) as (bidb, block within the
// sequence), or (-1, 0) in a gap or past the batch: a binary search over cu_seqlens for the last
// sequence starting at or before it.
template <int kBlock>
__forceinline__ __device__ cute::tuple<int, int> varlen_block_coord(const int block_idx, const int num_batch,
                                                                    const int *cu_seqlens, const int *seqused) {
    int bidb = 0, last = num_batch - 1;
    while (bidb < last) {
        const int middle = (bidb + last + 1) / 2;
        if (cu_seqlens[middle] / kBlock + middle <= block_idx) { bidb = middle; } else { last = middle - 1; }
    }
    const int block = block_idx - (cu_seqlens[bidb] / kBlock + bidb);
    const int seqlen = seqused ? seqused[bidb] : cu_seqlens[bidb + 1] - cu_seqlens[bidb];
    return block < (seqlen + kBlock - 1) / kBlock ? cute::make_tuple(bidb, block) : cute::make_tuple(-1, 0);
}

template<bool Varlen=true>
struct BlockInfo {

    template<typename Params>
    __device__ BlockInfo(const Params &params, const int bidb)
        : sum_s_q(!Varlen || params.cu_seqlens_q == nullptr ? -1 : params.cu_seqlens_q[bidb])
        , sum_s_k(!Varlen || params.cu_seqlens_k == nullptr || !params.is_seqlens_k_cumulative ? -1 : params.cu_seqlens_k[bidb])
        , actual_seqlen_q(!Varlen || params.cu_seqlens_q == nullptr ? params.seqlen_q : params.cu_seqlens_q[bidb + 1] - sum_s_q)
        // If is_seqlens_k_cumulative, then seqlen_k is cu_seqlens_k[bidb + 1] - cu_seqlens_k[bidb].
        // Otherwise it's cu_seqlens_k[bidb], i.e., we use cu_seqlens_k to store the sequence lengths of K.
        , leftpad_k(params.leftpad_k == nullptr ? 0 : params.leftpad_k[bidb])
        , seqlen_k_cache((!Varlen || params.cu_seqlens_k == nullptr ? params.seqlen_k : (params.is_seqlens_k_cumulative ? params.cu_seqlens_k[bidb + 1] - sum_s_k : params.cu_seqlens_k[bidb])) - leftpad_k)
        , actual_seqlen_k(params.seqused_k ? params.seqused_k[bidb] - leftpad_k : seqlen_k_cache + (params.knew_ptr == nullptr ? 0 : params.seqlen_knew))
        {
        }

    template <typename index_t>
    __forceinline__ __device__ index_t q_offset(const index_t batch_stride, const index_t row_stride, const int bidb) const {
        return sum_s_q == -1 ? bidb * batch_stride : uint32_t(sum_s_q) * row_stride;
    }

    template <typename index_t>
    __forceinline__ __device__ index_t k_offset(const index_t batch_stride, const index_t row_stride, const int bidb) const {
        return sum_s_k == -1 ? bidb * batch_stride + leftpad_k * row_stride : uint32_t(sum_s_k + leftpad_k) * row_stride;
    }

    const int sum_s_q;
    const int sum_s_k;
    const int actual_seqlen_q;
    // We have to have seqlen_k_cache declared before actual_seqlen_k, otherwise actual_seqlen_k is set to 0.
    const int leftpad_k;
    const int seqlen_k_cache;
    const int actual_seqlen_k;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace FLASH_NAMESPACE
