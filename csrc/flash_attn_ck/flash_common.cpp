/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#include "flash_common.hpp"

namespace flash {
int override_num_splits_if_necessary(int batch, int nhead, int max_seqlen_q, int hdim_v, float p_drop, int num_splits)
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
    const int kN1 = hdim_v;

    const int num_m_blocks = (max_seqlen_q + kM0 - 1) / kM0;
    const int num_n_blocks = (hdim_v + kN1 - 1) / kN1;

    if(num_splits < 1 && p_drop == 0.0f)
        return num_splits_heuristic_ck(
            batch * nhead * num_m_blocks, props.multiProcessorCount * 2, num_n_blocks, 128);

    return num_splits;
}

} // namespace flash
