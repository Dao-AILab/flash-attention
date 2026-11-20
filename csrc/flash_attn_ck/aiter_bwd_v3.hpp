/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include "fmha_bwd.hpp"

#include <cstddef>
#include <functional>
#include <string>

namespace flash {

// Runs the backward with aiter's hand-written assembly kernels (third_party/aiter, fmha_v3_bwd)
// when one covers the case exactly, and returns false otherwise so the caller runs fmha_bwd.
// Only called for fp16/bf16 without dropout, ALiBi or deterministic mode. dk/dv in `args` are
// nhead_q-wide, as for fmha_bwd. workspace_alloc must return device memory that stays valid
// until the enqueued kernels have run, zero-filled when asked.
bool run_aiter_bwd_v3(const fmha_bwd_args &args,
                      const std::string &dtype,
                      const ck_tile::stream_config &stream_config,
                      const std::function<void *(size_t bytes, bool zero_init)> &workspace_alloc);

} // namespace flash
