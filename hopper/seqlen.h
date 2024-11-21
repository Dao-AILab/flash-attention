/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

namespace flash {

// We consolidate all the info related to sequence length here. This is so that we can do all
// the gmem reads once at the beginning of each tile, rather than having to repeat these reads
// to compute various things like n_block_min, n_block_max, etc.
template <bool Varlen, bool AppendKV>
struct SeqlenInfo {

    static_assert(!(AppendKV && !Varlen), "AppendKV is only supported with Varlen");

    int const leftpad_k;
    int const offset_q, offset_k, offset_k_new;
    int const seqlen_q, seqlen_k_og, seqlen_k_new, seqlen_k;

    CUTLASS_DEVICE
    SeqlenInfo(int const bidb, int const shape_Q_0, int const shape_K_0, int const shape_K_new_0,
               int const* const cu_seqlens_q, int const* const cu_seqlens_k, int const* const cu_seqlens_k_new,
               int const* const seqused_q, int const* const seqused_k, int const* const ptr_leftpad_k
               )
        : leftpad_k(ptr_leftpad_k ? ptr_leftpad_k[bidb] : 0)
        , offset_q(!Varlen || cu_seqlens_q == nullptr ? 0 : cu_seqlens_q[bidb])
        , offset_k(!Varlen ? 0 : (cu_seqlens_k ? cu_seqlens_k[bidb] : 0) + leftpad_k)
        , offset_k_new(!AppendKV || cu_seqlens_k_new == nullptr ? 0 : cu_seqlens_k_new[bidb])
        , seqlen_q(!Varlen
                   ? shape_Q_0
                   : (seqused_q ? seqused_q[bidb] : (cu_seqlens_q ? cu_seqlens_q[bidb + 1] - cu_seqlens_q[bidb] : shape_Q_0)))
        , seqlen_k_og(!Varlen
                      ? shape_K_0
                      : (seqused_k ? seqused_k[bidb] : (cu_seqlens_k ? cu_seqlens_k[bidb + 1] - cu_seqlens_k[bidb] : shape_K_0)) - leftpad_k)
        , seqlen_k_new(!AppendKV
                       ? 0
                       : (cu_seqlens_k_new ? cu_seqlens_k_new[bidb + 1] - cu_seqlens_k_new[bidb] : shape_K_new_0))
        , seqlen_k(!AppendKV ? seqlen_k_og : seqlen_k_og + seqlen_k_new)
    {
    }

};

} // namespace flash
