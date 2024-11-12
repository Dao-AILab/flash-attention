/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cute/tensor.hpp>

#include "cutlass/fast_math.h"  // For cutlass::FastDivmod

#include "utils.h"

namespace flash {

using namespace cute;

template <int kBlockM, int kBlockN, bool PackGQA=false>
struct Mask {

    int const seqlen_q, seqlen_k;
    int const window_size_left, window_size_right, sink_token_length;
    cutlass::FastDivmod const qhead_per_khead_divmod;

    CUTLASS_DEVICE
    Mask(const int seqlen_q, const int seqlen_k,
         const int window_size_left, const int window_size_right, const int sink_token_length,
         cutlass::FastDivmod const &qhead_per_khead_divmod)
        : seqlen_q(seqlen_q)
        , seqlen_k(seqlen_k)
        , window_size_left(window_size_left)
        , window_size_right(window_size_right)
        , sink_token_length(sink_token_length)
        , qhead_per_khead_divmod(qhead_per_khead_divmod)
        {
    };

    // Causal_mask: whether this particular iteration needs causal masking
    template <bool Seqlenk_mask=false, bool Causal_mask=false, bool Local_mask=false,
        typename Engine, typename Layout, typename TiledMma>
    CUTLASS_DEVICE
    void apply(Tensor<Engine, Layout> &tSrS,
               int const thread_idx,
               const int m_block, const int n_block,
               TiledMma &tiled_mma) {
        static_assert(!(Causal_mask && Local_mask), "Cannot be both causal and local");
        static_assert(Layout::rank == 3, "Only support 3D Tensor");
        if (!Seqlenk_mask && !Causal_mask && !Local_mask) { return; }

        auto thread_mma = tiled_mma.get_thread_slice(thread_idx);
        auto thread0_mma = tiled_mma.get_thread_slice(_0{});

        Tensor cS = cute::make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
        Tensor tScS = thread_mma.partition_C(cS);
        Tensor tSrS_rowcol = make_tensor(tSrS.data(), flash::convert_layout_acc_rowcol(tSrS.layout()));
        Tensor tScS_rowcol = make_tensor(tScS.data(), flash::convert_layout_acc_rowcol(tScS.layout()));
        Tensor t0ScS = thread0_mma.partition_C(cS);
        Tensor t0ScS_rowcol = make_tensor(t0ScS.data(), flash::convert_layout_acc_rowcol(t0ScS.layout()));
        // We want to use the col indices of thread0 to compare, since that is known at compile time.
        // So we subtract the limit by the first col index of this thread (get<1>(tScS_rowcol(_0{}, _0{})))
        int const thread_col_offset = get<1>(tScS_rowcol(_0{}, _0{}));
        int const seqlenk_col_limit = seqlen_k - n_block * kBlockN - thread_col_offset;
        if constexpr (!Causal_mask && !Local_mask) {
            if constexpr (Seqlenk_mask) {  // Just masking based on col
                #pragma unroll
                for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                    if (int(get<1>(t0ScS_rowcol(_0{}, n))) >= seqlenk_col_limit) {
                        #pragma unroll
                        for (int m = 0; m < size<0>(tSrS_rowcol); ++m) { tSrS_rowcol(m, n) = -INFINITY; }
                    }
                }
            }
        } else {  // mask based on both row and col
            // If PackGQA, we split the work of compute divmod among threads in the same row
            static constexpr int kMmaThreadsPerRow = size<0, 0>(typename TiledMma::AtomLayoutC_TV{});
            static_assert(cutlass::NumThreadsPerWarp % kMmaThreadsPerRow == 0);
            static_assert(CUTE_STATIC_V(size<0>(tSrS_rowcol)) <= kMmaThreadsPerRow);
            int mma_m_idx;
            // Might get OOB but it's ok since we'll check it later
            if constexpr (PackGQA) {
                mma_m_idx = qhead_per_khead_divmod.divide(m_block * kBlockM + get<0>(tScS_rowcol(thread_idx % kMmaThreadsPerRow, _0{})));
            }
            int causal_row_offset = 1 + seqlen_k - n_block * kBlockN - seqlen_q - thread_col_offset;
            if constexpr (Causal_mask) {
                #pragma unroll
                for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
                    int row_idx = !PackGQA
                        ? get<0>(tScS_rowcol(m, _0{})) + m_block * kBlockM
                        : row_idx = __shfl_sync(0xffffffff, mma_m_idx, m % kMmaThreadsPerRow, kMmaThreadsPerRow);
                    int col_limit_right = !Seqlenk_mask
                        ? row_idx + causal_row_offset
                        // : std::min(row_idx + causal_row_offset, seqlenk_col_limit);
                        : __viaddmin_s32(row_idx, causal_row_offset, seqlenk_col_limit);
                        // Slightly slower for hdim 64 and slightly faster for hdim128
                    #pragma unroll
                    for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                        if (int(get<1>(t0ScS_rowcol(_0{}, n))) >= col_limit_right) { tSrS_rowcol(m, n) = -INFINITY; }
                    }
                }
            } else {
                int local_row_offset_right = causal_row_offset + window_size_right;
                int local_row_offset_left = causal_row_offset - 1 - window_size_left;
                int col_limit_sink = sink_token_length - n_block * kBlockN;
                #pragma unroll
                for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
                    int row_idx = !PackGQA
                        ? get<0>(tScS_rowcol(m, _0{})) + m_block * kBlockM
                        : row_idx = __shfl_sync(0xffffffff, mma_m_idx, m % kMmaThreadsPerRow, kMmaThreadsPerRow);
                    int col_limit_right = !Seqlenk_mask
                        ? row_idx + local_row_offset_right
                        // : std::min(row_idx, seqlenk_col_limit);
                        : __viaddmin_s32(row_idx, local_row_offset_right, seqlenk_col_limit);
                    int col_limit_left = row_idx + local_row_offset_left;
                    #pragma unroll
                    for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                        int col_idx = int(get<1>(t0ScS_rowcol(m, n)));
                        if (col_idx >= col_limit_right || (col_idx < col_limit_left && col_idx >= col_limit_sink)) { tSrS_rowcol(m, n) = -INFINITY; }
                    }
                }
            }
        }
    };

};

} // namespace flash
