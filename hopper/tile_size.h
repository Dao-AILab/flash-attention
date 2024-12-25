/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include <tuple>

// Return {kBlockM, kBlockN, Mma1_is_RS, IntraWGOverlap}
constexpr std::tuple<int, int, bool, bool> tile_size_fwd_sm90(
        int headdim, bool is_causal, bool is_local, int element_size=2,
        bool v_colmajor=false, bool paged_kv=false, bool softcap=false) {
    if (element_size == 2) {
        if (headdim <= 64) {
            return {192, 128, true, true};
            // Good for long seqlen (>= 4k) but suffers from tile quantization at short seqlen
            // return {192, is_causal || is_local ? 192 : 176, true, false};
        } else if (headdim <= 96) {
            return {192, is_local ? 128 : 144, false, true};
        } else if (headdim <= 128) {
            return {128, is_causal || is_local || paged_kv ? 128 : 176, true, true};
            // {128, 192, false, false} and {192, 128, false, true} are quite good too
            // 128 x 192 hits the limit of smem if Mma1_is_RS, 128 x 144 hits the limit if !Mma1_is_RS
        } else if (headdim <= 192) {
            return {128, paged_kv || is_local ? 96 : 112, true, true};  // 128 x 112 hits the limit of smem
        } else {
            return {128, is_local ? 64 : 80, true, true};  // 128 x 80 hits the limit of smem
        }
    } else {
        if (headdim <= 64) {
            return {192, 160, true, true};
        } else if (headdim <= 96) {
            return {192, 128, true, true};
        } else if (headdim <= 128) {
            return {128, paged_kv ? 160 : (v_colmajor || (softcap && is_local) ? 192 : 224), true, true};
        } else if (headdim <= 192) {
            return {128, (paged_kv || softcap) && is_local ? 128 : 160, true, true};
        } else {
            return {128, is_local ? 64 : 128, true, !paged_kv};  // PagedKV uses more registers so we disabled IntraWGOverlap
        }
    }
}

// Return {kBlockM, kBlockN, kNWarps, kStages, Q_in_regs}
constexpr std::tuple<int, int, int, int, bool> tile_size_fwd_sm80(
        int headdim, bool is_causal, bool is_local, int element_size=2,
        bool paged_kv=false, bool softcap=false) {
    if (element_size == 2) {
        if (headdim <= 64) {
            return {128, 128, 4, 2, true};
        } else if (headdim <= 96) {
            return {128, 64, 4, 2, true};
        } else if (headdim <= 128) {
            return {128, 128, 8, 2, true};
            // return {128, 64, 4, 1, false};  // a bit of spill but very good perf
        } else if (headdim <= 192) {
            return {128, 96, 8, 2, true};
        } else {
            return {128, 96, 8, 1, false};
        }
    } else {
        // Placeholder for now
        return {128, 64, 8, 2, false};
    }
}
