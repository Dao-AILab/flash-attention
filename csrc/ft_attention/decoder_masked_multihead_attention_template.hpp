// Downloaded from from FasterTransformer v5.2.1
// https://github.com/NVIDIA/FasterTransformer/blob/release/v5.2.1_tag/src/fastertransformer/kernels/decoder_masked_multihead_attention/decoder_masked_multihead_attention_template.hpp
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
#pragma once

#include "decoder_masked_multihead_attention.h"
#include "decoder_masked_multihead_attention_utils.h"
#include "cuda_bf16_wrapper.h"
#include "cuda_bf16_fallbacks.cuh"
#include <assert.h>
#include <float.h>
#include <type_traits>

// #define MMHA_USE_HMMA_FOR_REDUCTION

// Below are knobs to extend FP32 accumulation for higher FP16 accuracy

// Does not seem to affect the accuracy that much
#define MMHA_USE_FP32_ACUM_FOR_FMA

// Seems to slightly improve the accuracy
#define MMHA_USE_FP32_ACUM_FOR_OUT

#if 0 && defined(MMHA_USE_FP32_ACUM_FOR_OUT)
 // Does not seem to improve the accuracy
 //#define MMHA_USE_FP32_ACUM_FOR_LOGITS
#endif

namespace mmha {

////////////////////////////////////////////////////////////////////////////////////////////////////

//
// We use the following terminology to describe the different dimensions.
//
// B:  Batch size (number of sequences),
// L:  Sequence length,
// D:  Hidden dimension,
// H:  Number of heads,
// Dh: Hidden dimension per head - Dh = D / H.
//
// The different kernels assign a threadblock for B x H pair. The grid has size (1, B, H). We use
// 64, 128 and 256 threads per block.
//
// Each threadblock loads Dh values from Q and its associated bias. The kernels run a loop to
// compute Q * K^T where K is loaded from a cache buffer -- except for the current timestep. The
// cache buffer helps with memory accesses and contains keys with bias.
//
// The layout of the cache buffer for the keys is [B, H, Dh/x, L, x] where x == 8 for FP16 and
// x == 4 for FP32 where the fastest moving dimension (contiguous data) is the rightmost one. The
// values for x are chosen to create chunks of 16 bytes.
//
// The different kernels use 1, 2 or 4 threads per key (THREADS_PER_KEY). The size of the LDGs
// depends on the number of threads per key. Each thread sums Dh / THREADS_PER_KEY elements. At
// the end of each iteration of the Q * K^T loop, we perform a reduction between lanes using an
// HMMA instruction (Tensor Core). Each Q * K^T valuey is stored in shared memory in FP32.
//
// After that loop, a parallel softmax is computed across the different Q * K^T values stored in
// shared memory.
//
// The kernel ends with a loop over the values in V. We use THREADS_PER_VALUE to control how many
// timesteps are computed by loop iteration. As with the keys, the values are read from a cache
// except for the current timestep. The layout of the cache buffer for the values is much simpler
// as it is [B, H, L, Dh].
//

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, int Dh>
struct Qk_vec_ {
};

template<>
struct Qk_vec_<float, 32> {
    using Type = float;
};
template<>
struct Qk_vec_<float, 64> {
    using Type = float2;
};
template<>
struct Qk_vec_<float, 128> {
    using Type = float4;
};
template<>
struct Qk_vec_<float, 256> {
    using Type = float4;
};
template<>
struct Qk_vec_<uint16_t, 32> {
    using Type = uint32_t;
};
template<>
struct Qk_vec_<uint16_t, 64> {
    using Type = uint32_t;
};
template<>
struct Qk_vec_<uint16_t, 128> {
    using Type = uint2;
};
template<>
struct Qk_vec_<uint16_t, 256> {
    using Type = uint4;
};
#ifdef ENABLE_BF16
template<>
struct Qk_vec_<__nv_bfloat16, 32> {
    using Type = __nv_bfloat162;
};
template<>
struct Qk_vec_<__nv_bfloat16, 64> {
    using Type = __nv_bfloat162;
};
template<>
struct Qk_vec_<__nv_bfloat16, 128> {
    using Type = bf16_4_t;
};
template<>
struct Qk_vec_<__nv_bfloat16, 256> {
    using Type = bf16_8_t;
};
#endif  // ENABLE_BF16
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, int THREADS_PER_KEY>
struct K_vec_ {
};

template<>
struct K_vec_<float, 4> {
    using Type = float;
};
template<>
struct K_vec_<float, 2> {
    using Type = float2;
};
template<>
struct K_vec_<float, 1> {
    using Type = float4;
};
template<>
struct K_vec_<uint16_t, 4> {
    using Type = uint32_t;
};
template<>
struct K_vec_<uint16_t, 2> {
    using Type = uint2;
};
template<>
struct K_vec_<uint16_t, 1> {
    using Type = uint4;
};
#ifdef ENABLE_BF16
template<>
struct K_vec_<__nv_bfloat16, 4> {
    using Type = __nv_bfloat162;
};
template<>
struct K_vec_<__nv_bfloat16, 2> {
    using Type = bf16_4_t;
};
template<>
struct K_vec_<__nv_bfloat16, 1> {
    using Type = bf16_8_t;
};
#endif  // ENABLE_BF16
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, int V_VEC_SIZE>
struct V_vec_ {
};

template<>
struct V_vec_<float, 1> {
    using Type = float;
};
template<>
struct V_vec_<float, 2> {
    using Type = float2;
};
template<>
struct V_vec_<float, 4> {
    using Type = float4;
};
template<>
struct V_vec_<uint16_t, 2> {
    using Type = uint32_t;
};
template<>
struct V_vec_<uint16_t, 4> {
    using Type = uint2;
};
template<>
struct V_vec_<uint16_t, 8> {
    using Type = uint4;
};
#ifdef ENABLE_BF16
template<>
struct V_vec_<__nv_bfloat16, 2> {
    using Type = __nv_bfloat162;
};
template<>
struct V_vec_<__nv_bfloat16, 4> {
    using Type = bf16_4_t;
};
template<>
struct V_vec_<__nv_bfloat16, 8> {
    using Type = bf16_8_t;
};
#endif  // ENABLE_BF16
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef MMHA_USE_FP32_ACUM_FOR_FMA
template<typename T>
struct Qk_vec_acum_fp32_ {
};

template<>
struct Qk_vec_acum_fp32_<float> {
    using Type = float;
};
template<>
struct Qk_vec_acum_fp32_<float2> {
    using Type = float2;
};
template<>
struct Qk_vec_acum_fp32_<float4> {
    using Type = float4;
};
// template<> struct Qk_vec_acum_fp32_<uint16_t> { using Type = float;        };
template<>
struct Qk_vec_acum_fp32_<uint32_t> {
    using Type = float2;
};
template<>
struct Qk_vec_acum_fp32_<uint2> {
    using Type = Float4_;
};
template<>
struct Qk_vec_acum_fp32_<uint4> {
    using Type = Float8_;
};
template<>
struct Qk_vec_acum_fp32_<__nv_bfloat16> {
    using Type = float;
};
template<>
struct Qk_vec_acum_fp32_<__nv_bfloat162> {
    using Type = float2;
};
template<>
struct Qk_vec_acum_fp32_<bf16_4_t> {
    using Type = Float4_;
};
template<>
struct Qk_vec_acum_fp32_<bf16_8_t> {
    using Type = Float8_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct K_vec_acum_fp32_ {
};

template<>
struct K_vec_acum_fp32_<float> {
    using Type = float;
};
template<>
struct K_vec_acum_fp32_<float2> {
    using Type = float2;
};
template<>
struct K_vec_acum_fp32_<float4> {
    using Type = float4;
};
template<>
struct K_vec_acum_fp32_<uint32_t> {
    using Type = float2;
};
template<>
struct K_vec_acum_fp32_<uint2> {
    using Type = Float4_;
};
template<>
struct K_vec_acum_fp32_<uint4> {
    using Type = Float8_;
};
template<>
struct K_vec_acum_fp32_<__nv_bfloat16> {
    using Type = float;
};
template<>
struct K_vec_acum_fp32_<__nv_bfloat162> {
    using Type = float2;
};
template<>
struct K_vec_acum_fp32_<bf16_4_t> {
    using Type = Float4_;
};
template<>
struct K_vec_acum_fp32_<bf16_8_t> {
    using Type = Float8_;
};
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef MMHA_USE_FP32_ACUM_FOR_OUT
template<typename T>
struct V_vec_acum_fp32_ {
};

template<>
struct V_vec_acum_fp32_<float> {
    using Type = float;
};
template<>
struct V_vec_acum_fp32_<float2> {
    using Type = float2;
};
template<>
struct V_vec_acum_fp32_<float4> {
    using Type = float4;
};
template<>
struct V_vec_acum_fp32_<uint32_t> {
    using Type = float2;
};
template<>
struct V_vec_acum_fp32_<uint2> {
    using Type = Float4_;
};
template<>
struct V_vec_acum_fp32_<uint4> {
    using Type = Float8_;
};
#ifdef ENABLE_BF16
template<>
struct V_vec_acum_fp32_<__nv_bfloat162> {
    using Type = float2;
};
template<>
struct V_vec_acum_fp32_<bf16_4_t> {
    using Type = Float4_;
};
template<>
struct V_vec_acum_fp32_<bf16_8_t> {
    using Type = Float8_;
};
#endif  // ENABLE_BF16
#endif
////////////////////////////////////////////////////////////////////////////////////////////////////

template<int THREADS_PER_KEY, typename K_vec, int N>
inline __device__ float qk_dot_(const K_vec (&q)[N], const K_vec (&k)[N])
{
#ifdef MMHA_USE_FP32_ACUM_FOR_FMA
    using K_vec_acum = typename K_vec_acum_fp32_<K_vec>::Type;
#else
    using K_vec_acum = K_vec;
#endif
    // Compute the parallel products for Q*K^T (treat vector lanes separately).
    K_vec_acum qk_vec = mul<K_vec_acum, K_vec, K_vec>(q[0], k[0]);
#pragma unroll
    for (int ii = 1; ii < N; ++ii) {
        qk_vec = fma(q[ii], k[ii], qk_vec);
    }

    // Finalize the reduction across lanes.
    float qk = sum(qk_vec);
#pragma unroll
    for (int mask = THREADS_PER_KEY / 2; mask >= 1; mask /= 2) {
        qk += __shfl_xor_sync(uint32_t(-1), qk, mask);
    }
    return qk;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, int THREADS_PER_KEY>
struct Qk_dot {
    template<typename K_vec, int N>
    static inline __device__ float dot(const K_vec (&q)[N], const K_vec (&k)[N])
    {
        return qk_dot_<THREADS_PER_KEY>(q, k);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float4 hmma_fp32(const uint2& a, uint32_t b)
{
    float4 c;
    float  zero = 0.f;
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 \n"
                 "    {%0, %1, %2, %3}, \n"
                 "    {%4, %5}, \n"
                 "    {%6}, \n"
                 "    {%7, %7, %7, %7}; \n"

                 : "=f"(c.x), "=f"(c.y), "=f"(c.z), "=f"(c.w)
                 : "r"(a.x) "r"(a.y), "r"(b), "f"(zero));
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int N>
inline __device__ float qk_hmma_dot_(const uint32_t (&q)[N], const uint32_t (&k)[N])
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
#ifdef MMHA_USE_FP32_ACUM_FOR_FMA
    using K_vec_acum = typename K_vec_acum_fp32_<uint32_t>::Type;
#else
    using K_vec_acum = uint32_t;
#endif
    K_vec_acum qk_vec = mul<K_vec_acum, uint32_t, uint32_t>(q[0], k[0]);
#pragma unroll
    for (int ii = 1; ii < N; ++ii) {
        qk_vec = fma(q[ii], k[ii], qk_vec);
    }
#ifdef MMHA_USE_FP32_ACUM_FOR_FMA
    uint32_t qk_vec_ = float2_to_half2(qk_vec);
    return hmma_fp32(make_uint2(qk_vec_, 0u), 0x3c003c00u).x;
#else
    return hmma_fp32(make_uint2(qk_vec, 0u), 0x3c003c00u).x;
#endif
#else
    return 0.f;
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Qk_dot<uint16_t, 4> {
    template<int N>
    static inline __device__ float dot(const uint32_t (&q)[N], const uint32_t (&k)[N])
    {
#if __CUDA_ARCH__ >= 750 && defined(MMHA_USE_HMMA_FOR_REDUCTION)
        return qk_hmma_dot_(q, k);
#else
        return qk_dot_<4>(q, k);
#endif  // defined MMHA_USE_HMMA_FOR_REDUCTION
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int WARPS_PER_BLOCK, int WARP_SIZE = 32>
inline __device__ float block_sum(float* red_smem, float sum)
{

    // Decompose the thread index into warp / lane.
    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;

// Compute the sum per warp.
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
        sum += __shfl_xor_sync(uint32_t(-1), sum, mask);
    }

    // Warp leaders store the data to shared memory.
    if (lane == 0) {
        red_smem[warp] = sum;
    }

    // Make sure the data is in shared memory.
    __syncthreads();

    // The warps compute the final sums.
    if (lane < WARPS_PER_BLOCK) {
        sum = red_smem[lane];
    }

// Parallel reduction inside the warp.
#pragma unroll
    for (int mask = WARPS_PER_BLOCK / 2; mask >= 1; mask /= 2) {
        sum += __shfl_xor_sync(uint32_t(-1), sum, mask);
    }

    // Broadcast to other threads.
    return __shfl_sync(uint32_t(-1), sum, 0);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_float(float& dst, float src)
{
    dst = src;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_float(uint16_t& dst, float src)
{
    dst = float_to_half(src);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_float(uint32_t& dst, float2 src)
{
    dst = float2_to_half2(src);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef ENABLE_BF16
inline __device__ void convert_from_float(__nv_bfloat16& dst, float src)
{
    dst = __float2bfloat16(src);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_float(__nv_bfloat162& dst, float2 src)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    dst = __float22bfloat162_rn(src);
#else
    dst   = __floats2bfloat162_rn(src.x, src.y);
#endif
}
#endif  // ENABLE_BF16
////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_float(uint2& dst, Float4_ src)
{
    dst.x = float2_to_half2(src.x);
    dst.y = float2_to_half2(src.y);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_float(uint2& dst, float4 src)
{
    convert_from_float(dst, Float4_{make_float2(src.x, src.y), make_float2(src.z, src.w)});
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_float(uint4& dst, Float8_ src)
{
    dst.x = float2_to_half2(src.x);
    dst.y = float2_to_half2(src.y);
    dst.z = float2_to_half2(src.z);
    dst.w = float2_to_half2(src.w);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef ENABLE_BF16
inline __device__ void convert_from_float(bf16_4_t& dst, Float4_ src)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    dst.x = __float22bfloat162_rn(src.x);
    dst.y = __float22bfloat162_rn(src.y);
#else
    dst.x = __floats2bfloat162_rn(src.x.x, src.x.y);
    dst.y = __floats2bfloat162_rn(src.y.x, src.y.y);
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_float(bf16_4_t& dst, float4 src)
{
    convert_from_float(dst, Float4_{make_float2(src.x, src.y), make_float2(src.z, src.w)});
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_float(bf16_8_t& dst, Float8_ src)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    dst.x = __float22bfloat162_rn(src.x);
    dst.y = __float22bfloat162_rn(src.y);
    dst.z = __float22bfloat162_rn(src.z);
    dst.w = __float22bfloat162_rn(src.w);
#else
    dst.x = __floats2bfloat162_rn(src.x.x, src.x.y);
    dst.y = __floats2bfloat162_rn(src.y.x, src.y.y);
    dst.z = __floats2bfloat162_rn(src.z.x, src.z.y);
    dst.w = __floats2bfloat162_rn(src.w.x, src.w.y);
#endif
}
#endif  // ENABLE_BF16

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_float(float2& dst, float2 src)
{
    dst = src;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_float(float4& dst, float4 src)
{
    dst = src;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float convert_to_float(float4 u)
{
    return u.x;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float convert_to_float(uint4 u)
{
    float2 tmp = half2_to_float2(u.x);
    return tmp.x;
}

#if defined(MMHA_USE_FP32_ACUM_FOR_LOGITS)

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float cast_to_float(float u)
{
    return u;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float2 cast_to_float(float2 u)
{
    return u;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float4 cast_to_float(float4 u)
{
    return u;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float4_ cast_to_float(Float4_ u)
{
    return u;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float8_ cast_to_float(Float8_ u)
{
    return u;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float2 cast_to_float(uint32_t u)
{
    return half2_to_float2(u);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float4_ cast_to_float(uint2 u)
{
    Float4_ tmp;
    tmp.x = half2_to_float2(u.x);
    tmp.y = half2_to_float2(u.y);
    return tmp;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float8_ cast_to_float(uint4 u)
{
    Float8_ tmp;
    tmp.x = half2_to_float2(u.x);
    tmp.y = half2_to_float2(u.y);
    tmp.z = half2_to_float2(u.z);
    tmp.w = half2_to_float2(u.w);
    return tmp;
}

#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float float_from_int8(int8_t u)
{
    return u;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float2 float_from_int8(int16_t u)
{
    union {
        int16_t int16;
        int8_t  int8[2];
    };
    int16 = u;
    return make_float2(int8[0], int8[1]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float4 float_from_int8(int32_t u)
{
    union {
        int32_t int32;
        int8_t  int8[4];
    };
    int32 = u;
    return make_float4(int8[0], int8[1], int8[2], int8[3]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// clang-format off
inline __device__ Float8_ float_from_int8(int64_t u)
{
    union {
        int64_t int64;
        int16_t int16[4];
    };
    int64 = u;
    return Float8_ {float_from_int8(int16[0]),
                    float_from_int8(int16[1]),
                    float_from_int8(int16[2]),
                    float_from_int8(int16[3])};
}
// clang-format on

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ int8_t cast_to_int8(float val)
{
    union {
        int8_t  int8[2];
        int16_t int16;
    };
    asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=h"(int16) : "f"(val));
    return int8[0];
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ int32_t cast_to_int8(float4 val)
{
    union {
        int8_t  int8[4];
        int32_t int32;
    };
    int8[0] = cast_to_int8(val.x);
    int8[1] = cast_to_int8(val.y);
    int8[2] = cast_to_int8(val.z);
    int8[3] = cast_to_int8(val.w);
    return int32;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ int64_t cast_to_int8(Float8_ val)
{
    union {
        int8_t  int8[8];
        int64_t int64;
    };
    int8[0] = cast_to_int8(val.x.x);
    int8[1] = cast_to_int8(val.x.y);
    int8[2] = cast_to_int8(val.y.x);
    int8[3] = cast_to_int8(val.y.y);
    int8[4] = cast_to_int8(val.z.x);
    int8[5] = cast_to_int8(val.z.y);
    int8[6] = cast_to_int8(val.w.x);
    int8[7] = cast_to_int8(val.w.y);
    return int64;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
inline __device__ __host__ T div_up(T m, T n)
{
    return (m + n - 1) / n;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, bool DO_CROSS_ATTENTION>
inline size_t smem_size_in_bytes(const Multihead_attention_params<T, DO_CROSS_ATTENTION>& params,
                                 int                                                      threads_per_value,
                                 int                                                      threads_per_block)
{
    // The amount of shared memory needed to store the Q*K^T values in float.
    const int max_timesteps = min(params.timestep, params.memory_max_len);
    size_t qk_sz = (DO_CROSS_ATTENTION) ? div_up(params.memory_max_len + 1, 4) * 16 : div_up(max_timesteps + 1, 4) * 16;

    // The extra memory needed if we are not using floats for the final logits.
    size_t logits_sz = 0;
#ifndef MMHA_USE_FP32_ACUM_FOR_LOGITS
    if (sizeof(T) != 4) {
        // TDOD
        logits_sz = (DO_CROSS_ATTENTION) ? div_up(params.memory_max_len + 1, 4) * 4 * sizeof(T) :
                                           div_up(max_timesteps + 1, 4) * 4 * sizeof(T);
    }
#endif

    // The total size needed during softmax.
    size_t softmax_sz = qk_sz + logits_sz;

    // The number of partial rows to reduce in the final reduction.
    int rows_per_red = threads_per_block / threads_per_value;
    // The amount of storage needed to finalize the outputs.
    size_t red_sz = rows_per_red * params.hidden_size_per_head * sizeof(T) / 2;

    size_t transpose_rotary_size = 0;
    if (params.rotary_embedding_dim > 0 && params.neox_rotary_style) {
        transpose_rotary_size = 2 * params.rotary_embedding_dim * sizeof(T);
    }

    // The max.
    return max(max(softmax_sz, red_sz), transpose_rotary_size);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ constexpr uint32_t shfl_mask(int threads)
{
    return threads == 32 ? uint32_t(-1) : (1u << threads) - 1u;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The type of the inputs. Supported types: float and half.
    typename T,
    // The hidden dimension per head.
    int Dh,
    int Dh_MAX,
    // The number of threads per key.
    int THREADS_PER_KEY,
    // The number of threads per value.
    int THREADS_PER_VALUE,
    // The number of threads in a threadblock.
    int  THREADS_PER_BLOCK,
    bool DO_CROSS_ATTENTION>
__global__ void masked_multihead_attention_kernel(Multihead_attention_params<T, DO_CROSS_ATTENTION> params)
{

    // Make sure the hidden dimension per head is a multiple of the number of threads per key.
    static_assert(Dh_MAX % THREADS_PER_KEY == 0, "");
    // Make sure the hidden dimension per head is a multiple of the number of threads per value.
    static_assert(Dh_MAX % THREADS_PER_VALUE == 0, "");

    // The size of a warp.
    constexpr int WARP_SIZE = 32;
    // The number of warps in a threadblock.
    constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / WARP_SIZE;

    // Use smem_size_in_bytes (above) to determine the amount of shared memory.
    extern __shared__ char smem_[];

    // The shared memory for the Q*K^T values and partial logits in softmax.
    float* qk_smem = reinterpret_cast<float*>(smem_);

    // The shared memory for the logits. For FP32, that's the same buffer as qk_smem.
    char* logits_smem_ = smem_;
#ifndef MMHA_USE_FP32_ACUM_FOR_LOGITS
    if (sizeof(T) != 4) {
        // TODO - change to tlength
        const int max_timesteps = min(params.timestep, params.memory_max_len);
        logits_smem_ +=
            (DO_CROSS_ATTENTION) ? div_up(params.memory_max_len + 1, 4) * 16 : div_up(max_timesteps + 1, 4) * 16;
    }
    T* logits_smem = reinterpret_cast<T*>(logits_smem_);
#else
    float* logits_smem = reinterpret_cast<float*>(logits_smem_);
#endif

    // The shared memory to do the final reduction for the output values. Reuse qk_smem.
    T* out_smem = reinterpret_cast<T*>(smem_);

    // The shared memory buffers for the block-wide reductions. One for max, one for sum.
    __shared__ float red_smem[WARPS_PER_BLOCK * 2];

    // A vector of Q or K elements for the current timestep.
    using Qk_vec = typename Qk_vec_<T, Dh_MAX>::Type;

    // Use alignment for safely casting the shared buffers as Qk_vec.
    // Shared memory to store Q inputs.
    __shared__ __align__(sizeof(Qk_vec)) T q_smem[Dh_MAX];

    // This is one of the reasons we should have a separate kernel for cross attention
    __shared__ __align__(sizeof(Qk_vec)) T bias_smem[DO_CROSS_ATTENTION ? Dh_MAX : 1];

    // A vector of Q or K elements for the current timestep.
    using Qk_vec = typename Qk_vec_<T, Dh_MAX>::Type;
    // The number of elements per vector.
    constexpr int QK_VEC_SIZE = sizeof(Qk_vec) / sizeof(T);
    // Make sure the hidden size per head is a multiple of the vector size.
    static_assert(Dh_MAX % QK_VEC_SIZE == 0, "");
    // We will use block wide reduction if needed
    // static_assert(Dh_MAX / QK_VEC_SIZE <= WARP_SIZE, "");
    // The number of vectors per warp.
    constexpr int QK_VECS_PER_WARP = Dh_MAX / QK_VEC_SIZE;

    // The layout of the cache is [B, H, Dh/x, L, x] with x == 4/8 for FP32/FP16. Since each thread
    // owns x elements, we have to decompose the linear index into chunks of x values and the posi-
    // tion of the thread in that chunk.

    // The number of elements in a chunk of 16B (that's the x in the above formula).
    constexpr int QK_ELTS_IN_16B = 16 / sizeof(T);
    // The number of K vectors in 16B.
    constexpr int QK_VECS_IN_16B = 16 / sizeof(Qk_vec);

    // The batch/beam idx
    const int bi = blockIdx.y;
    if (params.finished != nullptr && params.finished[bi] == true) {
        return;
    }
    // The beam idx
    const int beami = bi % params.beam_width;
    // The "beam-aware" batch idx
    const int bbi = bi / params.beam_width;
    // The head.
    // const int hi = blockIdx.x;
    const int hi = params.nnz_head_idx == nullptr ? blockIdx.x : params.nnz_head_idx[blockIdx.x];
    const int hi_kv = hi / params.num_heads_q_kv_ratio;
    // Combine the batch and the head indices.
    const int bhi = bi * params.num_heads + hi;
    const int bhi_kv = bi * params.num_heads_kv + hi_kv;
    // Combine the "beam-aware" batch idx and the head indices.
    const int bbhi = bbi * params.beam_width * params.num_heads_kv + hi_kv;
    // The thread in the block.
    const int tidx = threadIdx.x;

    const bool handle_kv = !DO_CROSS_ATTENTION || (DO_CROSS_ATTENTION && params.timestep == 0);

    // While doing the product Q*K^T for the different keys we track the max.
    float qk_max = -FLT_MAX;

    float qk = 0.0F;

    int q_base_offset = (params.stride_q == 0) ? bhi * Dh : bi * params.stride_q + hi * Dh;
    int k_base_offset = (params.stride_k == 0) ? bhi_kv * Dh : bi * params.stride_k + hi_kv * Dh;
    int v_base_offset = (params.stride_v == 0) ? bhi_kv * Dh : bi * params.stride_v + hi_kv * Dh;

    const size_t bi_seq_len_offset = bi * params.memory_max_len;

    // int tlength = (DO_CROSS_ATTENTION)? params.memory_length_per_sample[bi] - 1 : params.timestep;
    int       tlength      = (DO_CROSS_ATTENTION) ? params.memory_length_per_sample[bi] - 1 :
                             (params.length_per_sample == nullptr) ?
                                                    params.timestep :
                                                    params.length_per_sample[bi] + params.max_prefix_prompt_length;
    const int first_step   = max(0, tlength + 1 - params.memory_max_len);
    const int tlength_circ = tlength % params.memory_max_len;

    // First QK_VECS_PER_WARP load Q and K + the bias values for the current timestep.
    const bool is_masked = tidx >= QK_VECS_PER_WARP;

    // The offset in the Q and K buffer also accounts for the batch.
    int q_offset = q_base_offset + tidx * QK_VEC_SIZE;
    int k_offset = k_base_offset + tidx * QK_VEC_SIZE;
    // The offset in the bias buffer.
    int q_bias_offset = hi * Dh + tidx * QK_VEC_SIZE;
    int k_bias_offset = hi_kv * Dh + tidx * QK_VEC_SIZE;

    const bool do_ia3      = handle_kv && params.ia3_tasks != nullptr;
    const int  ia3_task_id = do_ia3 ? params.ia3_tasks[bbi] : 0;

    // Trigger the loads from the Q and K buffers.
    Qk_vec q;
    zero(q);
    if (!is_masked && (Dh == Dh_MAX || tidx * QK_VEC_SIZE < Dh)) {
        if (params.int8_mode == 2) {
            using Packed_Int8_t  = typename packed_type<int8_t, num_elems<Qk_vec>::value>::type;
            using Packed_Float_t = typename packed_type<float, num_elems<Qk_vec>::value>::type;
            const auto q_scaling = params.qkv_scale_out[0];
            const auto q_quant =
                *reinterpret_cast<const Packed_Int8_t*>(&reinterpret_cast<const int8_t*>(params.q)[q_offset]);

            convert_from_float(q, mul<Packed_Float_t, float>(q_scaling, float_from_int8(q_quant)));
        }
        else {
            q = *reinterpret_cast<const Qk_vec*>(&params.q[q_offset]);
        }
    }

    Qk_vec k;
    zero(k);
    if (DO_CROSS_ATTENTION) {
        // The 16B chunk written by the thread.
        int co = tidx / QK_VECS_IN_16B;
        // The position of the thread in that 16B chunk.
        int ci = tidx % QK_VECS_IN_16B * QK_VEC_SIZE;

        // Two chunks are separated by L * x elements. A thread write QK_VEC_SIZE elements.
        int offset = bhi_kv * params.memory_max_len * Dh + co * params.memory_max_len * QK_ELTS_IN_16B +
                     // params.timestep*QK_ELTS_IN_16B +
                     tlength * QK_ELTS_IN_16B + ci;
        k = !is_masked && (Dh == Dh_MAX || tidx * QK_VEC_SIZE < Dh) ?
                *reinterpret_cast<const Qk_vec*>(&params.k_cache[offset]) :
                k;
    }
    else {
        if (!is_masked && (Dh == Dh_MAX || tidx * QK_VEC_SIZE < Dh)) {
            if (params.int8_mode == 2) {
                using Packed_Int8_t  = typename packed_type<int8_t, num_elems<Qk_vec>::value>::type;
                using Packed_Float_t = typename packed_type<float, num_elems<Qk_vec>::value>::type;
                const auto k_scaling = params.qkv_scale_out[1];
                const auto k_quant =
                    *reinterpret_cast<const Packed_Int8_t*>(&reinterpret_cast<const int8_t*>(params.k)[k_offset]);

                convert_from_float(k, mul<Packed_Float_t, float>(k_scaling, float_from_int8(k_quant)));
            }
            else {
                k = *reinterpret_cast<const Qk_vec*>(&params.k[k_offset]);
            }
        }
    }

    // Trigger the loads from the Q and K bias buffers.
    Qk_vec q_bias;
    zero(q_bias);
    q_bias = (!is_masked && Dh == Dh_MAX || tidx * QK_VEC_SIZE < Dh) && params.q_bias != nullptr ?
                 *reinterpret_cast<const Qk_vec*>(&params.q_bias[q_bias_offset]) :
                 q_bias;

    Qk_vec k_bias;
    zero(k_bias);
    if (handle_kv) {
        k_bias = !is_masked && (Dh == Dh_MAX || tidx * QK_VEC_SIZE < Dh) && params.k_bias != nullptr ?
                     *reinterpret_cast<const Qk_vec*>(&params.k_bias[k_bias_offset]) :
                     k_bias;
    }

    // Computes the Q/K values with bias.
    q = add(q, q_bias);
    if (handle_kv) {
        k = add(k, k_bias);
    }
    if (do_ia3 && !is_masked) {
        k = mul<Qk_vec, Qk_vec, Qk_vec>(
            k,
            *reinterpret_cast<const Qk_vec*>(
                &params.ia3_key_weights[(ia3_task_id * params.num_heads + hi) * Dh + tidx * QK_VEC_SIZE]));
    }

    // Padded len
    const int padd_len = (params.total_padding_tokens == nullptr) ? 0 : params.total_padding_tokens[bi];
    if (params.rotary_embedding_dim > 0 && !params.neox_rotary_style) {
        if (handle_kv) {
            if (params.rotary_cos == nullptr) {
                apply_rotary_embedding(q, k, tidx, params.rotary_embedding_dim, tlength - padd_len, params.rotary_base);
            } else {
                apply_rotary_embedding(q, k, tidx, params.rotary_embedding_dim, tlength - padd_len,
                                       params.rotary_cos + bi * params.rotary_embedding_dim / 2,
                                       params.rotary_sin + bi * params.rotary_embedding_dim / 2);
            }
        }
        else {
            if (params.rotary_cos == nullptr) {
                apply_rotary_embedding(q, tidx, params.rotary_embedding_dim, tlength - padd_len, params.rotary_base);
            } else {
                apply_rotary_embedding(q, tidx, params.rotary_embedding_dim, tlength - padd_len,
                                       params.rotary_cos + bi * params.rotary_embedding_dim / 2,
                                       params.rotary_sin + bi * params.rotary_embedding_dim / 2);
            }
        }
    }
    else if (params.rotary_embedding_dim > 0 && params.neox_rotary_style) {
        const bool do_rotary = !is_masked && QK_VEC_SIZE * tidx < params.rotary_embedding_dim;

        T* q_smem = reinterpret_cast<T*>(smem_);
        T* k_smem = q_smem + params.rotary_embedding_dim;

        const int half_rotary_dim = params.rotary_embedding_dim / 2;
        const int half_idx        = (tidx * QK_VEC_SIZE) / half_rotary_dim;
        const int intra_half_idx  = (tidx * QK_VEC_SIZE) % half_rotary_dim;
        const int smem_pitch      = half_rotary_dim;  // TODO: adjust for bank conflicts

        assert(half_rotary_dim % QK_VEC_SIZE == 0);

        if (do_rotary) {
            *reinterpret_cast<Qk_vec*>(q_smem + half_idx * smem_pitch + intra_half_idx) = q;

            if (handle_kv) {
                *reinterpret_cast<Qk_vec*>(k_smem + half_idx * smem_pitch + intra_half_idx) = k;
            }
        }

        __syncthreads();

        const int     transpose_idx = half_idx * (half_rotary_dim / 2) + intra_half_idx / 2;
        constexpr int tidx_factor   = (QK_VEC_SIZE > 1) ? QK_VEC_SIZE / 2 : 1;
        if (do_rotary) {
            mmha::vec_from_smem_transpose(q, q_smem, transpose_idx, smem_pitch);

            if (handle_kv) {
                mmha::vec_from_smem_transpose(k, k_smem, transpose_idx, smem_pitch);

                if (params.rotary_cos == nullptr) {
                    mmha::apply_rotary_embedding(
                        q, k, transpose_idx / tidx_factor, params.rotary_embedding_dim, tlength - padd_len, params.rotary_base);
                } else {
                    mmha::apply_rotary_embedding(
                        q, k, transpose_idx / tidx_factor, params.rotary_embedding_dim, tlength - padd_len,
                        params.rotary_cos + bi * params.rotary_embedding_dim / 2,
                        params.rotary_sin + bi * params.rotary_embedding_dim / 2);
                }

                mmha::write_smem_transpose(k, k_smem, transpose_idx, smem_pitch);
            }
            else {
                if (params.rotary_cos == nullptr) {
                    mmha::apply_rotary_embedding(
                        q, transpose_idx / tidx_factor, params.rotary_embedding_dim, tlength, params.rotary_base);
                } else {
                    mmha::apply_rotary_embedding(
                        q, transpose_idx / tidx_factor, params.rotary_embedding_dim, tlength,
                        params.rotary_cos + bi * params.rotary_embedding_dim / 2,
                        params.rotary_sin + bi * params.rotary_embedding_dim / 2);
                }
            }
            mmha::write_smem_transpose(q, q_smem, transpose_idx, smem_pitch);
        }

        __syncthreads();

        if (do_rotary) {
            q = *reinterpret_cast<Qk_vec*>(q_smem + half_idx * smem_pitch + intra_half_idx);
            if (handle_kv) {
                k = *reinterpret_cast<Qk_vec*>(k_smem + half_idx * smem_pitch + intra_half_idx);
            }
        }

        __syncthreads();
    }

    if (!is_masked) {
        // Store the Q values to shared memory.
        *reinterpret_cast<Qk_vec*>(&q_smem[tidx * QK_VEC_SIZE]) = q;

        // Store Dh values of k_bias into smem, since will need to add later
        // if params.timestep == 0
        if (DO_CROSS_ATTENTION && params.timestep == 0) {
            *reinterpret_cast<Qk_vec*>(&bias_smem[tidx * QK_VEC_SIZE]) = k_bias;
        }

        // Write the K values to the global memory cache.
        //
        // NOTE: The stores are uncoalesced as we have multiple chunks of 16B spread across the memory
        // system. We designed it this way as it allows much better memory loads (and there are many
        // more loads) + the stores are really "write and forget" since we won't need the ack before
        // the end of the kernel. There's plenty of time for the transactions to complete.

        // The 16B chunk written by the thread.
        int co = tidx / QK_VECS_IN_16B;
        // The position of the thread in that 16B chunk.
        int ci = tidx % QK_VECS_IN_16B * QK_VEC_SIZE;

        // Two chunks are separated by L * x elements. A thread write QK_VEC_SIZE elements.
        int offset = bhi_kv * params.memory_max_len * Dh + co * params.memory_max_len * QK_ELTS_IN_16B +
                     // params.timestep*QK_ELTS_IN_16B +
                     tlength_circ * QK_ELTS_IN_16B + ci;

        if (handle_kv && hi % params.num_heads_q_kv_ratio == 0) {
            // Trigger the stores to global memory.
            if (Dh == Dh_MAX || co < Dh / QK_ELTS_IN_16B) {
                *reinterpret_cast<Qk_vec*>(&params.k_cache[offset]) = k;
            }
        }

        // Compute \sum_i Q[i] * K^T[i] for the current timestep.
#ifdef MMHA_USE_FP32_ACUM_FOR_FMA
        using Qk_vec_acum = typename Qk_vec_acum_fp32_<Qk_vec>::Type;
#else
        using Qk_vec_acum = Qk_vec;
#endif
        qk = dot<Qk_vec_acum, Qk_vec>(q, k);
        if (QK_VECS_PER_WARP <= WARP_SIZE) {
#pragma unroll
            for (int mask = QK_VECS_PER_WARP / 2; mask >= 1; mask /= 2) {
                qk += __shfl_xor_sync(shfl_mask(QK_VECS_PER_WARP), qk, mask);
            }
        }
    }

    if (QK_VECS_PER_WARP > WARP_SIZE) {
        constexpr int WARPS_PER_RED = (QK_VECS_PER_WARP + WARP_SIZE - 1) / WARP_SIZE;
        qk                          = block_sum<WARPS_PER_RED>(&red_smem[WARPS_PER_RED], qk);
    }

    // Store that value in shared memory. Keep the Q*K^T value in register for softmax.
    if (tidx == 0) {
        // Normalize qk.
        qk *= params.inv_sqrt_dh;
        if (params.relative_attention_bias != nullptr) {
            qk = add(qk,
                     params.relative_attention_bias[hi * params.relative_attention_bias_stride
                                                        * params.relative_attention_bias_stride
                                                    + (tlength - padd_len) * params.relative_attention_bias_stride
                                                    + (tlength - padd_len)]);
        }
        // We don't need to apply the linear position bias here since qi - ki = 0 yields the position bias 0.

        qk_max                        = qk;
        qk_smem[tlength - first_step] = qk;
        // qk_smem[params.timestep] = qk;
    }

    // Make sure the data is in shared memory.
    __syncthreads();

    // The type of queries and keys for the math in the Q*K^T product.
    using K_vec = typename K_vec_<T, THREADS_PER_KEY>::Type;
    // The number of elements per vector.
    constexpr int K_VEC_SIZE = sizeof(K_vec) / sizeof(T);
    // Make sure the hidden size per head is a multiple of the vector size.
    static_assert(Dh_MAX % K_VEC_SIZE == 0, "");
    // The number of elements per thread.
    constexpr int K_ELTS_PER_THREAD = Dh_MAX / THREADS_PER_KEY;
    // The number of vectors per thread.
    constexpr int K_VECS_PER_THREAD = K_ELTS_PER_THREAD / K_VEC_SIZE;

    // The position the first key loaded by each thread from the cache buffer (for this B * H).
    int ko = tidx / THREADS_PER_KEY;
    // The position of the thread in the chunk of keys.
    int ki = tidx % THREADS_PER_KEY * K_VEC_SIZE;

    static_assert(Dh_MAX == THREADS_PER_KEY * K_VEC_SIZE * K_VECS_PER_THREAD);

    // Load the Q values from shared memory. The values are reused during the loop on K.
    K_vec q_vec[K_VECS_PER_THREAD];
#pragma unroll
    for (int ii = 0; ii < K_VECS_PER_THREAD; ++ii) {
        q_vec[ii] = *reinterpret_cast<const K_vec*>(&q_smem[ki + ii * THREADS_PER_KEY * K_VEC_SIZE]);
    }

    K_vec k_bias_vec[DO_CROSS_ATTENTION ? K_VECS_PER_THREAD : 1];
    if (DO_CROSS_ATTENTION && params.timestep == 0) {
#pragma unroll
        for (int ii = 0; ii < K_VECS_PER_THREAD; ++ii) {
            k_bias_vec[ii] = *reinterpret_cast<const K_vec*>(&bias_smem[ki + ii * THREADS_PER_KEY * K_VEC_SIZE]);
        }
    }

    // The number of timesteps loaded per iteration.
    constexpr int K_PER_ITER = THREADS_PER_BLOCK / THREADS_PER_KEY;
    // The number of keys per warp.
    constexpr int K_PER_WARP = WARP_SIZE / THREADS_PER_KEY;

    // The base pointer for the key in the cache buffer.
    T* k_cache = &params.k_cache[bhi_kv * params.memory_max_len * Dh + ki];
    // Base pointer for the beam's batch, before offsetting with indirection buffer
    T* k_cache_batch = &params.k_cache[bbhi * params.memory_max_len * Dh + ki];

    // Pick a number of keys to make sure all the threads of a warp enter (due to shfl_sync).
    // int ti_end = div_up(params.timestep, K_PER_WARP) * K_PER_WARP;
    int ti_end = div_up(tlength - first_step, K_PER_WARP) * K_PER_WARP + first_step;

    // prefix prompt length if has
    const int prefix_prompt_length = (params.prefix_prompt_lengths == nullptr) ? 0 : params.prefix_prompt_lengths[bi];

    // Iterate over the keys/timesteps to compute the various (Q*K^T)_{ti} values.
    const bool has_beams    = params.cache_indir != nullptr;
    const int* beam_indices = has_beams ? &params.cache_indir[bi_seq_len_offset] : nullptr;

    for (int ti = first_step + ko; ti < ti_end; ti += K_PER_ITER) {
        const int ti_circ = ti % params.memory_max_len;

        // The keys loaded from the key cache.
        K_vec k[K_VECS_PER_THREAD];
        K_vec k_vec_zero;
        zero(k_vec_zero);
#pragma unroll
        for (int ii = 0; ii < K_VECS_PER_THREAD; ++ii) {
            int jj = ii * params.memory_max_len + ti_circ;
            // if( ti < params.timestep ) {
            const bool within_bounds = (Dh == Dh_MAX || jj * QK_ELTS_IN_16B < Dh * params.memory_max_len);
            if (ti < tlength) {
                if (!within_bounds) {
                    k[ii] = k_vec_zero;
                }
                else {
                    if (has_beams) {
                        const int beam_offset = beam_indices[ti_circ] * params.num_heads * params.memory_max_len * Dh;
                        k[ii] = *reinterpret_cast<const K_vec*>(&k_cache_batch[beam_offset + jj * QK_ELTS_IN_16B]);
                    }
                    else {
                        k[ii] = *reinterpret_cast<const K_vec*>(&k_cache_batch[jj * QK_ELTS_IN_16B]);
                    }
                }
                // add bias and update k_cache
                if (DO_CROSS_ATTENTION && params.timestep == 0) {
                    k[ii] = add(k[ii], k_bias_vec[ii]);

                    if (do_ia3) {
                        k[ii] = mul<K_vec, K_vec, K_vec>(
                            k[ii],
                            *reinterpret_cast<const K_vec*>(
                                &params.ia3_key_weights[(ia3_task_id * params.num_heads + hi) * Dh + ki
                                                        + ii * THREADS_PER_KEY * K_VEC_SIZE]));
                    }

                    if (Dh == Dh_MAX || jj * QK_ELTS_IN_16B < Dh * params.memory_max_len) {
                        *reinterpret_cast<K_vec*>(&k_cache[jj * QK_ELTS_IN_16B]) = k[ii];
                    }
                }
            }
        }

        // Perform the dot product and normalize qk.
        //
        // WARNING: ALL THE THREADS OF A WARP MUST ENTER!!!
        float qk      = Qk_dot<T, THREADS_PER_KEY>::dot(q_vec, k) * params.inv_sqrt_dh;
        bool  is_mask = (params.masked_tokens != nullptr) && params.masked_tokens[bi_seq_len_offset + ti];

        // Store the product to shared memory. There's one qk value per timestep. Update the max.
        // if( ti < params.timestep && tidx % THREADS_PER_KEY == 0 ) {
        if (ti < tlength && tidx % THREADS_PER_KEY == 0) {
            if (params.relative_attention_bias != nullptr) {
                qk = add(qk,
                         params.relative_attention_bias[hi * params.relative_attention_bias_stride
                                                            * params.relative_attention_bias_stride
                                                        + tlength * params.relative_attention_bias_stride + ti]);
            }
            if (params.linear_bias_slopes != nullptr) {
                // Apply the linear position bias: (ki - qi) * slope[hi].
                // The padding token locates between the input context and the generated tokens.
                // We need to remove the number of padding tokens in the distance computation.
                //   ti   : 0 1 2 3 4 5 6 7 8 9(tlength)
                //   token: i i i i p p p o o o where i=input, p=pad, o=output.
                // e.g. ti = 2, dist = (9 - 3) - 2 = 4.
                int   max_context_length = params.max_prefix_prompt_length + params.max_input_length;
                float dist               = (ti < max_context_length ? ti + padd_len : ti) - tlength;

                qk += mul<float, T, float>(params.linear_bias_slopes[hi], dist);
            }
            qk_max                   = is_mask ? qk_max : fmaxf(qk_max, qk);
            qk_smem[ti - first_step] = qk;
        }
    }

// Perform the final reduction to compute the max inside each warp.
//
// NOTE: In a group of THREADS_PER_KEY threads, the leader already has the max value for the
// group so it's not needed to run the reduction inside the group (again).
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask >= THREADS_PER_KEY; mask /= 2) {
        qk_max = fmaxf(qk_max, __shfl_xor_sync(uint32_t(-1), qk_max, mask));
    }

    // Decompose the thread index into warp and lane.
    const int warp = tidx / WARP_SIZE;
    const int lane = tidx % WARP_SIZE;

    // The warp leader writes the max to shared memory.
    if (lane == 0) {
        red_smem[warp] = qk_max;
    }

    // Make sure the products are in shared memory.
    __syncthreads();

    // The warps finalize the reduction.
    qk_max = lane < WARPS_PER_BLOCK ? red_smem[lane] : -FLT_MAX;
#pragma unroll
    for (int mask = WARPS_PER_BLOCK / 2; mask >= 1; mask /= 2) {
        qk_max = fmaxf(qk_max, __shfl_xor_sync(uint32_t(-1), qk_max, mask));
    }

    // Broadcast to all the threads in the warp.
    qk_max = __shfl_sync(uint32_t(-1), qk_max, 0);

    // Compute the logits and start the sum.
    float sum = 0.f;
    // for( int ti = tidx; ti <= params.timestep; ti += THREADS_PER_BLOCK ) {
    for (int ti = first_step + tidx; ti <= tlength; ti += THREADS_PER_BLOCK) {
        bool  is_mask = (params.masked_tokens != nullptr) && params.masked_tokens[bi_seq_len_offset + ti];
        float logit   = is_mask ? 0.f : __expf(qk_smem[ti - first_step] - qk_max);
        sum += logit;
        qk_smem[ti - first_step] = logit;
    }

    // Compute the sum.
    sum = block_sum<WARPS_PER_BLOCK>(&red_smem[WARPS_PER_BLOCK], sum);

    // Normalize the logits.
    float inv_sum = __fdividef(1.f, sum + 1.e-6f);
    // for( int ti = tidx; ti <= params.timestep; ti += THREADS_PER_BLOCK ) {
    const size_t cross_attention_out_offset =
        params.is_return_cross_attentions ?
            bhi * params.max_decoder_seq_len * params.memory_max_len + params.timestep * params.memory_max_len :
            0;
    for (int ti = first_step + tidx; ti <= tlength; ti += THREADS_PER_BLOCK) {
        float logit = qk_smem[ti - first_step] * inv_sum;
        if (params.is_return_cross_attentions) {
            params.cross_attention_out[cross_attention_out_offset + ti] = logit;
        }
        convert_from_float(logits_smem[ti - first_step], logit);
    }

    // Put Values part below so we leverage __syncthreads
    // from the previous step

    // The number of elements per vector.
    constexpr int V_VEC_SIZE = Dh_MAX / THREADS_PER_VALUE;
    // A vector of V elements for the current timestep.
    using V_vec = typename V_vec_<T, V_VEC_SIZE>::Type;

    // The value computed by this thread.
    int vo = tidx / THREADS_PER_VALUE;
    // The hidden dimensions computed by this particular thread.
    int vi = tidx % THREADS_PER_VALUE * V_VEC_SIZE;

    // The base pointer for the value in the cache buffer.
    T* v_cache = &params.v_cache[bhi_kv * params.memory_max_len * Dh + vi];
    // Base pointer for the beam's batch, before offsetting with indirection buffer
    T* v_cache_batch = &params.v_cache[bbhi * params.memory_max_len * Dh + vi];

    // The number of values processed per iteration of the loop.
    constexpr int V_PER_ITER = THREADS_PER_BLOCK / THREADS_PER_VALUE;

    // One group of threads computes the product(s) for the current timestep.
    V_vec v_bias;
    zero(v_bias);
    // if( vo == params.timestep % V_PER_ITER ) {
    if (Dh == Dh_MAX || vi < Dh) {
        if (handle_kv) {
            if (vo == tlength % V_PER_ITER) {
                // Trigger the loads from the V bias buffer.
                if (params.v_bias != nullptr) {
                    v_bias = *reinterpret_cast<const V_vec*>(&params.v_bias[hi_kv * Dh + vi]);
                }
                if (DO_CROSS_ATTENTION) {
                    *reinterpret_cast<V_vec*>(&bias_smem[vi]) = v_bias;
                }
            }
        }
    }

    // From previous, before values, step
    // Also make sure the logits are in shared memory.
    __syncthreads();

    // Values continued
#ifdef MMHA_USE_FP32_ACUM_FOR_OUT
    using V_vec_acum = typename V_vec_acum_fp32_<V_vec>::Type;
#else
    using V_vec_acum = V_vec;
#endif
    // The partial outputs computed by each thread.
    V_vec_acum out;
    zero(out);

    // Loop over the timesteps to compute the partial outputs.
    // for( int ti = vo; ti < params.timestep; ti += V_PER_ITER ) {
    if (Dh == Dh_MAX || vi < Dh) {
        for (int ti = first_step + vo; ti < tlength; ti += V_PER_ITER) {
            const int ti_circ = ti % params.memory_max_len;

            // Fetch offset based on cache_indir when beam sampling
            const int beam_src = (params.cache_indir != nullptr) ? params.cache_indir[bi_seq_len_offset + ti_circ] : 0;
            const int beam_offset = beam_src * params.num_heads * params.memory_max_len * Dh;
            // Load the values from the cache.
            V_vec v = *reinterpret_cast<const V_vec*>(&v_cache_batch[beam_offset + ti_circ * Dh]);
            if (DO_CROSS_ATTENTION && params.timestep == 0) {
                v = add(v, *reinterpret_cast<V_vec*>(&bias_smem[vi]));
                if (do_ia3) {
                    v = mul<V_vec, V_vec, V_vec>(
                        v,
                        *reinterpret_cast<const V_vec*>(
                            &params.ia3_value_weights[(ia3_task_id * params.num_heads + hi) * Dh + vi]));
                }
                *reinterpret_cast<V_vec*>(&v_cache[ti * Dh]) = v;
            }
            // Load the logits from shared memory.
#if defined(MMHA_USE_FP32_ACUM_FOR_LOGITS)
            float logit = logits_smem[ti - first_step];
            out         = fma(logit, cast_to_float(v), out);
#else
            T logit = logits_smem[ti - first_step];

            // Update the partial sums.
            out = fma(logit, v, out);
#endif
        }
    }

    // One group of threads computes the product(s) for the current timestep.
    // if( vo == params.timestep % V_PER_ITER ) {
    if (vo == tlength % V_PER_ITER && (Dh == Dh_MAX || vi < Dh)) {

        V_vec v;
        if (DO_CROSS_ATTENTION) {
            v = *reinterpret_cast<const V_vec*>(&v_cache[tlength * Dh]);
        }
        else {
            // Trigger the loads from the V buffer.
            const auto v_offset = v_base_offset + vi;
            if (params.int8_mode == 2) {
                using Packed_Int8_t  = typename packed_type<int8_t, num_elems<V_vec>::value>::type;
                using Packed_Float_t = typename packed_type<float, num_elems<V_vec>::value>::type;
                const auto v_scaling = params.qkv_scale_out[2];
                const auto v_quant =
                    *reinterpret_cast<const Packed_Int8_t*>(&reinterpret_cast<const int8_t*>(params.v)[v_offset]);

                convert_from_float(v, mul<Packed_Float_t, float>(v_scaling, float_from_int8(v_quant)));
            }
            else {
                v = *reinterpret_cast<const V_vec*>(&params.v[v_offset]);
            }
            // Trigger the loads from the V bias buffer.
            // V_vec v_bias = *reinterpret_cast<const V_vec*>(&params.v_bias[hi*Dh + vi]);
        }

        // Compute the V values with bias.
        if (handle_kv) {
            v = add(v, v_bias);

            if (do_ia3) {
                v = mul<V_vec, V_vec, V_vec>(
                    v,
                    *reinterpret_cast<const V_vec*>(
                        &params.ia3_value_weights[(ia3_task_id * params.num_heads + hi) * Dh + vi]));
            }

            // Store the values with bias back to global memory in the cache for V.
            if (hi % params.num_heads_q_kv_ratio == 0) {
                //*reinterpret_cast<V_vec*>(&v_cache[params.timestep*Dh]) = v;
                *reinterpret_cast<V_vec*>(&v_cache[tlength_circ * Dh]) = v;
            }
        }

        // Initialize the output value with the current timestep.
#if defined(MMHA_USE_FP32_ACUM_FOR_LOGITS)
        // out = fma(logits_smem[params.timestep], cast_to_float(v), out);
        out = fma(logits_smem[tlength - first_step], cast_to_float(v), out);
#else
        // out = fma(logits_smem[params.timestep], v, out);
        out = fma(logits_smem[tlength - first_step], v, out);
#endif
    }

    // Make sure we can start writing to shared memory.
    __syncthreads();

    // Run the final reduction amongst the different groups computing different partial outputs.
    if (Dh == Dh_MAX || vi < Dh) {
#pragma unroll
        for (int active_groups = V_PER_ITER; active_groups >= 2; active_groups /= 2) {

            // The midpoint in the number of active groups.
            int midpoint = active_groups / 2;

            // The upper part of active threads store to shared memory.
            if (vo >= midpoint && vo < active_groups && (Dh == Dh_MAX || vi < Dh)) {
#ifdef MMHA_USE_FP32_ACUM_FOR_OUT
                convert_from_float(*reinterpret_cast<V_vec*>(&out_smem[(vo - midpoint) * Dh + vi]), out);
#else
                *reinterpret_cast<V_vec*>(&out_smem[(vo - midpoint) * Dh + vi]) = out;
#endif
            }
            __syncthreads();

            // The bottom warps update their values.
            if (vo < midpoint && (Dh == Dh_MAX || vi < Dh)) {
                out = add(*reinterpret_cast<const V_vec*>(&out_smem[vo * Dh + vi]), out);
            }
            __syncthreads();
        }
    }

    // Output the final values.
    if (vo == 0 && (Dh == Dh_MAX || vi < Dh)) {
#ifdef MMHA_USE_FP32_ACUM_FOR_OUT
        if (params.int8_mode == 2) {
            using Packed_Int8_t = typename packed_type<int8_t, num_elems<V_vec_acum>::value>::type;
            out                 = mul<V_vec_acum, float>(*params.attention_out_scale, out);
            *reinterpret_cast<Packed_Int8_t*>(&(reinterpret_cast<int8_t*>(params.out)[bhi * Dh + vi])) =
                cast_to_int8(out);
        }
        else {
            convert_from_float(*reinterpret_cast<V_vec*>(&params.out[bhi * Dh + vi]), out);
        }
#else
        // TODO: support int8_mode?
        *reinterpret_cast<V_vec*>(&params.out[bhi * Dh + vi]) = out;
#endif
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace mmha

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, int Dh, int Dh_MAX, typename KERNEL_PARAMS_TYPE>
void mmha_launch_kernel(const KERNEL_PARAMS_TYPE& params, const cudaStream_t& stream);
