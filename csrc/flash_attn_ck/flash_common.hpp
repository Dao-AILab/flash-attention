/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

// Include these 2 headers instead of torch/extension.h since we don't need all of the torch headers.
#include <torch/python.h>
#include <torch/nn/functional.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#ifdef OLD_GENERATOR_PATH
#include <ATen/CUDAGeneratorImpl.h>
#else
#include <ATen/cuda/CUDAGeneratorImpl.h>
#endif


#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

namespace flash {
inline __global__ void ParsePhiloxCudaState(at::PhiloxCudaState arg, uint64_t* rng_state)
{
    // Imitate from PyTorch
    // https://github.com/pytorch/pytorch/blob/8b61daaf7349e9102117e1aeefaa51666d887547/aten/src/ATen/cuda/detail/UnpackRaw.cuh#L17
    if (arg.captured_) {
        rng_state[0] = static_cast<uint64_t>(*arg.seed_.ptr);
        rng_state[1] = static_cast<uint64_t>(*(arg.offset_.ptr) + arg.offset_intragraph_);
    } else {
        rng_state[0] = arg.seed_.val;
        rng_state[1] = arg.offset_.val;
    }
}

inline int num_splits_heuristic_ck(int batch_nheads_mblocks, int num_SMs, [[maybe_unused]] int num_n_blocks, int max_splits) {
    // If we have enough to almost fill the SMs, then just use 1 split
    for (int num_splits = 1; num_splits <= max_splits; num_splits *= 2) {
        if (num_SMs < batch_nheads_mblocks * (num_splits * 2)) {
            return num_splits;
        }
    }

    return max_splits;
}

int override_num_splits_if_necessary(int batch, int nhead, int max_seqlen_q, int hdim_v, float p_drop, int num_splits);

} // namespace flash
