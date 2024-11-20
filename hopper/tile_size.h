/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include <tuple>

constexpr std::tuple<int, int> tile_size_fwd(int headdim, bool is_causal_or_local,
                                             int element_size=2, bool v_colmajor=false) {
    if (element_size == 2) {
        if (headdim <= 64) {
            return {192, 128};
        } else if (headdim <= 96) {
            return {128, is_causal_or_local ? 128 : 160};
        } else if (headdim <= 128) {
            return {128, is_causal_or_local ? 128 : 176};
        } else if (headdim <= 192) {
            return {128, 96};
        } else {
            return {128, 80};
        }
    } else {
        if (headdim <= 64) {
            return {192, 160};
        } else if (headdim <= 96) {
            return {192, 128};
        } else if (headdim <= 128) {
            return {128, v_colmajor ? 192 : 224};
        } else if (headdim <= 192) {
            return {128, 160};
        } else {
            return {128, 128};
        }
    }
}
