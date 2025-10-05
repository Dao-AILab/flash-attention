/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

namespace flash {

template <class SeqlenInfo_t, int kBlockM, int kBlockN, bool Is_causal, bool Is_local, bool PackGQA=false, bool Split=false>
struct BlockMN {

    static
    CUTLASS_DEVICE
    cute::tuple<int, int, int> get_n_block_min_max(
            SeqlenInfo_t const& seqlen_info,
            int const m_block, int const bidb, int const split_idx, int const num_splits,
            int const window_size_left, int const window_size_right,
            cutlass::FastDivmod const& qhead_per_khead_divmod) {

        int seqlen_k = seqlen_info.seqlen_k;
        int const seqlen_q = seqlen_info.seqlen_q;
        int n_offset = 0;

        // If local, calculate n_offset and update seqlen_k
        if constexpr (Is_local) {
            int m_idx_min = m_block * kBlockM;
            if (PackGQA) { m_idx_min = qhead_per_khead_divmod.divide(m_idx_min); }
            // unlike previously, we don't divide by kBlockN because we want offset for seqlen_k
            n_offset = std::max(int(0), m_idx_min + seqlen_k - seqlen_q - window_size_left);
            // Subtract n_offset from seqlen_k for subsequent calculations such as n_block_max
            // This is the actual seqlen_k processed for this m_block
            seqlen_k -= n_offset;
        }

        int n_block_max = cute::ceil_div(seqlen_k, kBlockN);
        if constexpr (Is_causal || Is_local) {
            int m_idx_max = (m_block + 1) * kBlockM;
            // TODO: check off-by-1 error
            if (PackGQA) { m_idx_max = qhead_per_khead_divmod.divide(m_idx_max - 1) + 1 ; }
            // If local, blocking (m_idx_max - m_idx_min + window_size_right + window_size_left)  
            // when cp is not enabled, tot_seqlen_k is equal to seqlen_k, and cp_world_size is 1.
            // cp_world_size is guaranteed to be greater than 0
            n_block_max = std::min(n_block_max,
                                    cute::ceil_div(
                                    cute::ceil_div(m_idx_max + seqlen_info.tot_seqlen_k - seqlen_q + window_size_right - seqlen_info.cp_rank,
                                                  seqlen_info.cp_world_size),
                                    kBlockN));
        }
        // Now, only adjust n_block_min if split
        int n_block_min = 0;
        
        // if (threadIdx.x == 128) { printf("Inside, bid.x = %d, bid.y = %d, bid.z = %d, split_idx = %d, n_block_min: %d, n_block_max: %d\n", blockIdx.x, blockIdx.y, blockIdx.z, split_idx, n_block_min, n_block_max); }
        if constexpr (Split) {
            uint32_t num_splits_dynamic_u = reinterpret_cast<uint32_t const&>(split_idx) >> 16; // first 16 bits are for num_splits
            int num_splits_dynamic = reinterpret_cast<int&>(num_splits_dynamic_u);
            int split_idx_actual = split_idx & 0x0000FFFF;
            int num_splits_actual = num_splits_dynamic > 0 ? num_splits_dynamic : num_splits;
            int num_n_blocks_per_split = n_block_max <= n_block_min ? 0 : cute::ceil_div(n_block_max - n_block_min, num_splits_actual);
            n_block_min = n_block_min + split_idx_actual * num_n_blocks_per_split;
            n_block_max = std::min(n_block_min + num_n_blocks_per_split, n_block_max);
            // if (threadIdx.x == 128) { printf("Inside, bid.x = %d, bid.y = %d, bid.z = %d, split_idx = %d, num_splits_dynamic = %d, num_splits_actual = %d, num_n_blocks_per_split = %d, n_block_min: %d, n_block_max: %d\n", blockIdx.x, blockIdx.y, blockIdx.z, split_idx, num_splits_dynamic, num_splits_actual, num_n_blocks_per_split, n_block_min, n_block_max); }
        }
        // if (threadIdx.x == 128) { printf("After split, inside, bid.y = %d, bid.z = %d, split_idx = %d, n_block_min: %d, n_block_max: %d\n", blockIdx.y, blockIdx.z, split_idx, n_block_min, n_block_max); }

        // Return n_offset to add to KV gmem pointers and use in masks
        return {n_block_min, n_block_max, n_offset};
    }

    static
    CUTLASS_DEVICE
    cute::tuple<int, int> get_n_block_k_new_min_max(
            SeqlenInfo_t const& seqlen_info,
            int const m_block, int const bidb, int const split_idx, int const num_splits,
            int const window_size_left, int const window_size_right,
            cutlass::FastDivmod const& qhead_per_khead_divmod) {
        // TODO: check logic with n_offset
        auto [n_block_min, n_block_max, n_offset] = get_n_block_min_max(
            seqlen_info, m_block, bidb, split_idx, num_splits,
            window_size_left, window_size_right, qhead_per_khead_divmod);
        int const idx_k_new_min = std::max(n_block_min * kBlockN + n_offset - seqlen_info.seqlen_k_og, 0);
        int const idx_k_new_max = std::min(n_block_max * kBlockN + n_offset - seqlen_info.seqlen_k_og, seqlen_info.seqlen_k_new);
        int const n_block_new_min = idx_k_new_min / kBlockN;
        int const n_block_new_max = idx_k_new_max > idx_k_new_min ? cute::ceil_div(idx_k_new_max, kBlockN) : n_block_new_min;
        // if (threadIdx.x == 128 && m_block == 0) { printf("bidb = %d, seqlen_k_new = %d, seqlen_k_og = %d, n_block_min = %d, n_block_max = %d, idx_k_new_min = %d, idx_k_new_max = %d, n_block_new_min = %d, n_block_new_max = %d\n", bidb, seqlen_k_new, seqlen_k_og, n_block_min, n_block_max, idx_k_new_min, idx_k_new_max, n_block_new_min, n_block_new_max);}
        return {n_block_new_min, n_block_new_max};
    }

    static
    CUTLASS_DEVICE
    cute::tuple<int, int> get_m_block_min_max(
            SeqlenInfo_t const& seqlen_info,
            int const n_block, int const bidb,
            int const window_size_left, int const window_size_right, int const sink_token_length) {

        int const seqlen_q = seqlen_info.seqlen_q;
        int const seqlen_k = seqlen_info.seqlen_k;
        int m_block_max = cute::ceil_div(seqlen_q, kBlockM);
        if constexpr (Is_local) {
            if (n_block >= cute::ceil_div(sink_token_length, kBlockN)) {
                m_block_max = std::min(m_block_max, cute::ceil_div((n_block + 1) * kBlockN + seqlen_q - seqlen_k + window_size_left, kBlockM));
            }
        }
        int m_block_min = 0;
        if constexpr (Is_causal || Is_local) {
            m_block_min = std::max(m_block_min, (n_block * kBlockN + seqlen_q - seqlen_k - window_size_right) / kBlockM);
        }
        return {m_block_min, m_block_max};
    }

};

} // namespace flash
