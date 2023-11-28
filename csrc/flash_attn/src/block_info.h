/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

namespace flash {

////////////////////////////////////////////////////////////////////////////////////////////////////

template<bool Varlen=true>
struct BlockInfo {

    template<typename Params>
    __device__ BlockInfo(const Params &params, const int bidb)
        : sum_s_q(!Varlen || params.cu_seqlens_q == nullptr ? -1 : params.cu_seqlens_q[bidb])
        , sum_s_k(!Varlen || params.cu_seqlens_k == nullptr ? -1 : params.cu_seqlens_k[bidb])
        , sum_s_knew(!Varlen || params.cu_seqlens_knew == nullptr ? -1 : params.cu_seqlens_knew[bidb])
        , actual_seqlen_q(!Varlen || params.cu_seqlens_q == nullptr ? params.seqlen_q : params.cu_seqlens_q[bidb + 1] - sum_s_q)
        , seqlen_k_cache(!Varlen || (params.cu_seqlens_k == nullptr && params.k_cache_seqlens == nullptr) ? params.seqlen_k : (params.k_cache_seqlens == nullptr ? params.cu_seqlens_k[bidb + 1] - sum_s_k : params.k_cache_seqlens[bidb]))
        , actual_seqlen_k(params.seqused_k ? params.seqused_k[bidb] : seqlen_k_cache + (params.knew_ptr == nullptr ? 0 : (params.cu_seqlens_knew == nullptr? params.seqlen_knew : params.cu_seqlens_knew[bidb + 1] - sum_s_knew)))
        {
        }

    template <typename index_t>
    inline __device__ index_t q_offset(const index_t batch_stride, const index_t row_stride, const int bidb) const {
        return sum_s_q == -1 ? bidb * batch_stride : uint32_t(sum_s_q) * row_stride;
    }

    template <typename index_t>
    inline __device__ index_t k_offset(const index_t batch_stride, const index_t row_stride, const int bidb) const {
        return sum_s_k == -1 ? bidb * batch_stride : uint32_t(sum_s_k) * row_stride;
    }

    template <typename index_t>
    inline __device__ index_t knew_offset(const index_t batch_stride, const index_t row_stride, const int bidb) const {
        // Used only in KV cache functions and when knew_ptr is not null.
        return sum_s_knew == -1 ? bidb * batch_stride : uint32_t(sum_s_knew) * row_stride;
    }

    const int sum_s_q;
    const int sum_s_k;
    const int sum_s_knew;
    const int actual_seqlen_q;
    // We have to have seqlen_k_cache declared before actual_seqlen_k, otherwise actual_seqlen_k is set to 0.
    const int seqlen_k_cache;
    const int actual_seqlen_k;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace flash
