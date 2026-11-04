/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

namespace flash {

// We consolidate all the info related to sequence length here. This is so that we can do all
// the gmem reads once at the beginning of each tile, rather than having to repeat these reads
// to compute various things like n_block_min, n_block_max, etc.

template <bool Varlen, int kBlock>
struct SeqlenInfo {

    int const offset, offset_padded;
    int const seqlen;

    CUTLASS_DEVICE
    SeqlenInfo(int const bidb, int const seqlen_static, int const* const cu_seqlens, int const* const seqused)
        : offset(!Varlen || cu_seqlens == nullptr ? 0 : cu_seqlens[bidb])
        , offset_padded(!Varlen || cu_seqlens == nullptr ? 0 : (cu_seqlens[bidb] + bidb * kBlock) / kBlock * kBlock)
        , seqlen(!Varlen
                 ? seqlen_static
                 : (seqused ? seqused[bidb] : (cu_seqlens ? cu_seqlens[bidb + 1] - cu_seqlens[bidb] : seqlen_static)))
    {
    }

};

// Map a block index in the linearized padded layout back to the sequence that
// owns it, plus the block index within that sequence.
//
// The padded layout gives sequence i the half-open block range
//   [ (cu_seqlens[i] + i * kBlock) / kBlock, (cu_seqlens[i+1] + (i+1) * kBlock) / kBlock )
// i.e. SeqlenInfo::offset_padded / kBlock. Because cu_seqlens is non-decreasing,
// consecutive starts differ by at least one block, so the starts are strictly
// increasing and a binary search recovers the owner. Every sequence owns at
// least one block, including empty ones (which the caller then skips).
//
// This lets a kernel launch ceil_div(total + num_batch * kBlock, kBlock) blocks,
// the exact extent of the padded layout, instead of a grid sized
// ceil_div(max_seqlen, kBlock) * num_batch, whose cost grows with the longest
// sequence and with sequences that hold no work at all.
template <int kBlock>
CUTLASS_DEVICE cute::tuple<int, int>
padded_block_to_seq(int const block_linear, int const num_batch, int const* const cu_seqlens) {
    auto start_block = [&](int const bidb) { return (cu_seqlens[bidb] + bidb * kBlock) / kBlock; };
    // Last sequence whose padded range starts at or before block_linear.
    int lo = 0, hi = num_batch;
    while (lo + 1 < hi) {
        int const mid = lo + (hi - lo) / 2;
        if (start_block(mid) <= block_linear) { lo = mid; } else { hi = mid; }
    }
    return {lo, block_linear - start_block(lo)};
}

template <bool Varlen, int kBlockM>
struct SeqlenInfoQK {

    int const offset_q, offset_k, offset_q_padded;
    int const seqlen_q, seqlen_k;

    CUTLASS_DEVICE
    SeqlenInfoQK(int const bidb, int const seqlen_q_static, int const seqlen_k_static,
                 int const* const cu_seqlens_q, int const* const cu_seqlens_k,
                 int const* const seqused_q, int const* const seqused_k
                 )
        : offset_q(!Varlen || cu_seqlens_q == nullptr ? 0 : cu_seqlens_q[bidb])
        , offset_k(!Varlen || cu_seqlens_k == nullptr ? 0 : cu_seqlens_k[bidb])
        // If varlen, the layout for dPSum, LSE_log2, and dQaccum is that we pad each sequence in the batch
        // by an extra kBlockM, so that the write for each sequence doesn't touch the next sequence.
        // Sequence i starts at cu_seqlens[i] + i * kBlockM and ends at cu_seqlens[i + 1] + i * kBlockM
        // However, the start must align to multiples of kBlockM.
        , offset_q_padded(!Varlen || cu_seqlens_q == nullptr ? 0 : (cu_seqlens_q[bidb] + bidb * kBlockM) / kBlockM * kBlockM)
        , seqlen_q(!Varlen
                   ? seqlen_q_static
                   : (seqused_q ? seqused_q[bidb] : (cu_seqlens_q ? cu_seqlens_q[bidb + 1] - cu_seqlens_q[bidb] : seqlen_q_static)))
        , seqlen_k(!Varlen
                   ? seqlen_k_static
                   : (seqused_k ? seqused_k[bidb] : (cu_seqlens_k ? cu_seqlens_k[bidb + 1] - cu_seqlens_k[bidb] : seqlen_k_static)))
    {
    }

};

template <bool Varlen, bool AppendKV>
struct SeqlenInfoQKNewK {

    static_assert(!(AppendKV && !Varlen), "AppendKV is only supported with Varlen");

    int const leftpad_k;
    int const offset_q, offset_k, offset_k_new;
    int const seqlen_q, seqlen_k_og, seqlen_k_new, seqlen_k, seqlen_rotary;

    CUTLASS_DEVICE
    SeqlenInfoQKNewK(int const bidb, int const seqlen_q_static, int const seqlen_k_static, int const shape_K_new_0,
                     int const* const cu_seqlens_q, int const* const cu_seqlens_k, int const* const cu_seqlens_k_new,
                     int const* const seqused_q, int const* const seqused_k, int const* const ptr_leftpad_k,
                     int const* const seqlens_rotary
                     )
        : leftpad_k(ptr_leftpad_k ? ptr_leftpad_k[bidb] : 0)
        , offset_q(!Varlen || cu_seqlens_q == nullptr ? 0 : cu_seqlens_q[bidb])
        , offset_k(!Varlen ? 0 : (cu_seqlens_k ? cu_seqlens_k[bidb] : 0) + leftpad_k)
        , offset_k_new(!AppendKV || cu_seqlens_k_new == nullptr ? 0 : cu_seqlens_k_new[bidb])
        , seqlen_q(!Varlen
                   ? seqlen_q_static
                   : (seqused_q ? seqused_q[bidb] : (cu_seqlens_q ? cu_seqlens_q[bidb + 1] - cu_seqlens_q[bidb] : seqlen_q_static)))
        , seqlen_k_og(!Varlen
                      ? seqlen_k_static
                      : (seqused_k ? seqused_k[bidb] : (cu_seqlens_k ? cu_seqlens_k[bidb + 1] - cu_seqlens_k[bidb] : seqlen_k_static)) - leftpad_k)
        , seqlen_k_new(!AppendKV
                       ? 0
                       : (cu_seqlens_k_new ? cu_seqlens_k_new[bidb + 1] - cu_seqlens_k_new[bidb] : shape_K_new_0))
        , seqlen_k(!AppendKV ? seqlen_k_og : seqlen_k_og + seqlen_k_new)
        , seqlen_rotary(!AppendKV || !seqlens_rotary ? seqlen_k_og + leftpad_k : seqlens_rotary[bidb])
    {
    }

};

} // namespace flash
