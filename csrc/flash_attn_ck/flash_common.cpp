/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#include "flash_common.hpp"

namespace flash {
int override_num_splits_if_necessary(int batch,
                                     int nhead,
                                     int nhead_k,
                                     int max_seqlen_q,
                                     int max_seqlen_k,
                                     int hdim_v,
                                     float p_drop,
                                     int num_splits)
{
    int device;
    auto status = hipGetDevice(&device);
    if(status != hipSuccess)
        return num_splits;

    hipDeviceProp_t props{};
    status = hipGetDeviceProperties(&props, device);
    if(status != hipSuccess)
        return num_splits;

    // TODO - tile size should match the TileFmhaShape, hardcode for now
    const int kM0 = 128;
    const int kN0 = 128;
    // Upper bound on splits the CK splitkv combine kernel can correctly reduce (see below).
    const int kMaxSplits = 8;

    const int num_m_blocks = (max_seqlen_q + kM0 - 1) / kM0;
    // num_n_blocks counts the KV-sequence blocks that a split is carved out of, so it
    // must come from seqlen_k -- cf. set_params_splitkv() in csrc/flash_attn/flash_api.cpp.
    // It used to be computed as ceildiv(hdim_v, kN1) with kN1 == hdim_v, i.e. a constant 1,
    // which clamped max_splits to 1 in num_splits_heuristic_ck() and made the heuristic
    // incapable of ever returning more than a single split.
    const int num_n_blocks = (max_seqlen_k + kN0 - 1) / kN0;

    // The splitkv kernel packs the query heads that share a KV head into one workgroup, so
    // the grid is batch * nhead_k * num_m_blocks. Passing nhead here overstated the available
    // parallelism by the GQA ratio and suppressed splitting exactly where it pays most
    // (MQA/GQA decode). Measured on MI350X: at a fixed batch * nhead of 4096, the speedup from
    // splitting tracks batch * nhead_k -- 1.02x at 4096, 1.95x at 128, 3.21x at 64.
    // The CK splitkv combine kernel only produces correct results up to 8 splits: measured on
    // gfx950 / CK c56c6750, num_splits 1..8 match an fp32 reference to ~5e-3 relative error and
    // num_splits >= 9 diverge (rel err 0.28 .. 2.0), independent of batch, seqlen_k and causal.
    // Until that is lifted, never hand the kernel a split count it cannot combine.
    if(num_splits < 1 && p_drop == 0.0f)
        return num_splits_heuristic_ck(batch * nhead_k * num_m_blocks,
                                       props.multiProcessorCount * 2,
                                       num_n_blocks,
                                       kMaxSplits);

    return num_splits;
}

} // namespace flash
