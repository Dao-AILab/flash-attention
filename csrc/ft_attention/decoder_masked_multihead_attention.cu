// Adapted from from FasterTransformer v5.2.1
// https://github.com/NVIDIA/FasterTransformer/blob/release/v5.2.1_tag/src/fastertransformer/kernels/decoder_masked_multihead_attention/decoder_masked_multihead_attention_128.cu
/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "decoder_masked_multihead_attention.h"
#include "decoder_masked_multihead_attention_utils.h"
#include "cuda_bf16_wrapper.h"
#include <assert.h>
#include <float.h>
#include <type_traits>

#include "decoder_masked_multihead_attention_template.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////

#define MMHA_LAUNCH_KERNEL(T, Dh, Dh_MAX, THDS_PER_KEY, THDS_PER_VALUE, THDS_PER_BLOCK, DO_CROSS_ATTENTION, stream)    \
    size_t smem_sz = mmha::smem_size_in_bytes<T, DO_CROSS_ATTENTION>(params, THDS_PER_VALUE, THDS_PER_BLOCK);          \
    auto kernel = mmha::masked_multihead_attention_kernel<T, Dh, Dh_MAX, THDS_PER_KEY, THDS_PER_VALUE,                 \
                                                          THDS_PER_BLOCK, DO_CROSS_ATTENTION>;                         \
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_sz);                                \
    dim3 grid(params.nnz_head_idx == nullptr ? params.num_heads : params.nnz_heads, params.batch_size);                \
    kernel<<<grid, THDS_PER_BLOCK, smem_sz, stream>>>(params)

////////////////////////////////////////////////////////////////////////////////////////////////////

// !!! Specialize the launcher for Cross attention
template<typename T, int Dh, int Dh_MAX, typename KERNEL_PARAMS_TYPE>
void mmha_launch_kernel(const KERNEL_PARAMS_TYPE& params, const cudaStream_t& stream)
{
    constexpr int  THREADS_PER_VALUE  = Dh_MAX * sizeof(T) / 16;
    constexpr bool DO_CROSS_ATTENTION = std::is_same<KERNEL_PARAMS_TYPE, Cross_multihead_attention_params<T>>::value;
    int            tlength            = (DO_CROSS_ATTENTION) ? params.memory_max_len : params.timestep;
    // printf("tlength, CROSS_ATTENTION = %d, %d\n", tlength, DO_CROSS_ATTENTION);
    if (tlength < 32) {
        MMHA_LAUNCH_KERNEL(T, Dh, Dh_MAX, 4, THREADS_PER_VALUE, 64, DO_CROSS_ATTENTION, stream);
    }
    else if (tlength < 2048) {
        MMHA_LAUNCH_KERNEL(T, Dh, Dh_MAX, 2, THREADS_PER_VALUE, 128, DO_CROSS_ATTENTION, stream);
    }
    else {
        MMHA_LAUNCH_KERNEL(T, Dh, Dh_MAX, 1, THREADS_PER_VALUE, 256, DO_CROSS_ATTENTION, stream);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#undef MMHA_LAUNCH_KERNEL

template<typename T, typename KERNEL_PARAMS_TYPE>
void multihead_attention_(const KERNEL_PARAMS_TYPE& params, const cudaStream_t& stream)
{
    switch (params.hidden_size_per_head) {
        case 32:
            mmha_launch_kernel<T, 32, 32, KERNEL_PARAMS_TYPE>(params, stream);
            break;
        case 48:
            mmha_launch_kernel<T, 48, 64, KERNEL_PARAMS_TYPE>(params, stream);
            break;
        case 64:
            mmha_launch_kernel<T, 64, 64, KERNEL_PARAMS_TYPE>(params, stream);
            break;
        case 80:
            mmha_launch_kernel<T, 80, 128, KERNEL_PARAMS_TYPE>(params, stream);
            break;
        case 96:
            mmha_launch_kernel<T, 96, 128, KERNEL_PARAMS_TYPE>(params, stream);
            break;
        case 128:
            mmha_launch_kernel<T, 128, 128, KERNEL_PARAMS_TYPE>(params, stream);
            break;
        case 160:
            mmha_launch_kernel<T, 160, 256, KERNEL_PARAMS_TYPE>(params, stream);
            break;
        case 192:
            mmha_launch_kernel<T, 192, 256, KERNEL_PARAMS_TYPE>(params, stream);
            break;
        case 224:
            mmha_launch_kernel<T, 224, 256, KERNEL_PARAMS_TYPE>(params, stream);
            break;
        case 256:
            mmha_launch_kernel<T, 256, 256, KERNEL_PARAMS_TYPE>(params, stream);
            break;
        default:
            assert(false);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void masked_multihead_attention(const Masked_multihead_attention_params<float>& params, const cudaStream_t& stream)
{
    multihead_attention_<float, Masked_multihead_attention_params<float>>(params, stream);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void masked_multihead_attention(const Masked_multihead_attention_params<uint16_t>& params, const cudaStream_t& stream)
{
    multihead_attention_<uint16_t, Masked_multihead_attention_params<uint16_t>>(params, stream);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef ENABLE_BF16
void masked_multihead_attention(const Masked_multihead_attention_params<__nv_bfloat16>& params,
                                const cudaStream_t&                                     stream)
{
    multihead_attention_<__nv_bfloat16, Masked_multihead_attention_params<__nv_bfloat16>>(params, stream);
}
#endif
////////////////////////////////////////////////////////////////////////////////////////////////////

void cross_multihead_attention(const Cross_multihead_attention_params<float>& params, const cudaStream_t& stream)
{
    multihead_attention_<float, Cross_multihead_attention_params<float>>(params, stream);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void cross_multihead_attention(const Cross_multihead_attention_params<uint16_t>& params, const cudaStream_t& stream)
{
    multihead_attention_<uint16_t, Cross_multihead_attention_params<uint16_t>>(params, stream);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef ENABLE_BF16
void cross_multihead_attention(const Cross_multihead_attention_params<__nv_bfloat16>& params,
                               const cudaStream_t&                                    stream)
{
    multihead_attention_<__nv_bfloat16, Cross_multihead_attention_params<__nv_bfloat16>>(params, stream);
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////
