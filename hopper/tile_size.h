/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include <tuple>

// Return {kBlockM, kBlockN, IntraWGOverlap}
constexpr std::tuple<int, int, bool> tile_size_fwd(int headdim, bool is_causal, bool is_local, int element_size=2,
                                                   bool v_colmajor=false, bool paged_kv=false, bool softcap=false) {
    if (element_size == 2) {
        if (headdim <= 64) {
            return {192, 128, true};
            // Good for long seqlen (>= 4k) but suffers from tile quantization at short seqlen
            // return {192, is_causal || is_local ? 192 : 176, false};
        } else if (headdim <= 96) {
            if (is_causal || is_local) {
                return {128, is_local ? 160 : 192, true};
            } else {
                return {192, 144, false};
            }
        } else if (headdim <= 128) {
            return {128, is_causal || is_local ? 128 : 176, true};
            // {128, 192, false} and {192, 128, false} are quite good too
            // 128 x 192 hits the limit of smem
        } else if (headdim <= 192) {
            return {128, paged_kv || is_local ? 96 : 112, true};  // 128 x 112 hits the limit of smem
        } else {
            return {128, is_local ? 64 : 80, true};  // 128 x 80 hits the limit of smem
        }
    } else {
        if (headdim <= 64) {
            return {192, 160, true};
        } else if (headdim <= 96) {
            return {192, 128, true};
        } else if (headdim <= 128) {
            return {128, v_colmajor || (paged_kv && is_local) ? 192 : 224, true};
        } else if (headdim <= 192) {
            return {128, paged_kv && is_local ? 128 : 160, true};
        } else {
            return {128, is_local ? 64 : 128, !paged_kv};  // PagedKV uses more registers so we disabled IntraWGOverlap
        }
    }
}

inline bool should_pack_gqa(bool varlen_q, bool is_causal_or_local, int seqlen_q, int qhead_per_khead, int blockM) {
    // If varlen, we don't actually know seqlen_q but only max_seqlen_q.
    // If causal, PackGQA always seems faster
    if (varlen_q || is_causal_or_local) return true;
    // Heuristic: PackGQA is a bit slower but can help if seqlen_q is small or not near a multiple of kBlockM
    auto round_up = [](int a, int b) { return (a + b - 1) / b * b; };
    float nopack_gqa_efficiency = float(seqlen_q) / float(round_up(seqlen_q, blockM));
    float pack_gqa_efficiency = float(seqlen_q * qhead_per_khead) / float(round_up(seqlen_q * qhead_per_khead, blockM));
    return nopack_gqa_efficiency < 0.95 * pack_gqa_efficiency;
};
