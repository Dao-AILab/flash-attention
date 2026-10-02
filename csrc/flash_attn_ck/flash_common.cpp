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
    // The CK combine kernel only reduces up to 8 splits correctly.
    const int kMaxSplits = 8;

    const int num_m_blocks = (max_seqlen_q + kM0 - 1) / kM0;
    // Splits are carved from the KV sequence -- cf. set_params_splitkv() on the CUDA side.
    const int num_n_blocks = (max_seqlen_k + kN0 - 1) / kN0;

    // Query heads sharing a KV head share a workgroup, so the grid is nhead_k-wide, not nhead.
    if(num_splits < 1 && p_drop == 0.0f)
        return num_splits_heuristic_ck(batch * nhead_k * num_m_blocks,
                                       props.multiProcessorCount * 2,
                                       num_n_blocks,
                                       kMaxSplits);

    return num_splits;
}

} // namespace flash
