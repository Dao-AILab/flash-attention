/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

namespace flash {

// The blocks of kBlock rows of a varlen batch of `total` rows in `num_batch` sequences, numbered
// in its padded block space: sequence b's blocks start at cu_seqlens[b] / kBlock + b, the rows the
// backward pads its accumulators by (SeqlenInfo::offset_padded). Each sequence leaves at most one
// gap after its last block, so the space spans this many blocks however long the longest sequence
// is, where a grid of num_batch * ceil_div(max_seqlen, kBlock) gives every (sequence, block) pair
// a CTA, most of which find their sequence already ended when most sequences are short or empty.
inline int varlen_num_blocks(int total, int num_batch, int block) {
    return (total + block - 1) / block + num_batch;
}

// The `block_idx`-th block of the padded block space (varlen_num_blocks) as (bidb, block within
// the sequence), or (-1, 0) in a gap or past the batch: a binary search over cu_seqlens for the
// last sequence starting at or before it. Without cu_seqlens, each sequence holds seqlen_static
// rows' blocks.
template <int kBlock>
CUTLASS_DEVICE
cute::tuple<int, int> varlen_block_coord(int const block_idx, int const num_batch, int const* const cu_seqlens,
                                         int const* const seqused, int const seqlen_static) {
    int bidb, block, seqlen;
    if (cu_seqlens) {
        int last = num_batch - 1;
        bidb = 0;
        while (bidb < last) {
            int const middle = (bidb + last + 1) / 2;
            if (cu_seqlens[middle] / kBlock + middle <= block_idx) { bidb = middle; } else { last = middle - 1; }
        }
        block = block_idx - (cu_seqlens[bidb] / kBlock + bidb);
        seqlen = seqused ? seqused[bidb] : cu_seqlens[bidb + 1] - cu_seqlens[bidb];
    } else {
        int const blocks = cute::ceil_div(seqlen_static, kBlock);
        bidb = block_idx / blocks;
        block = block_idx % blocks;
        seqlen = bidb < num_batch ? (seqused ? seqused[bidb] : seqlen_static) : 0;
    }
    return block < cute::ceil_div(seqlen, kBlock) ? cute::make_tuple(bidb, block) : cute::make_tuple(-1, 0);
}

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
