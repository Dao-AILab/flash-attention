// Downloaded from from FasterTransformer v5.2.1
// https://github.com/NVIDIA/FasterTransformer/blob/release/v5.2.1_tag/src/fastertransformer/kernels/decoder_masked_multihead_attention.h
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

#include "cuda_bf16_wrapper.h"
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

////////////////////////////////////////////////////////////////////////////////////////////////////

#define CHECK_CUDA(call)                                                                                               \
    do {                                                                                                               \
        cudaError_t status_ = call;                                                                                    \
        if (status_ != cudaSuccess) {                                                                                  \
            fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(status_));              \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)

////////////////////////////////////////////////////////////////////////////////////////////////////

// The structure of parameters for the masked multihead attention kernel.
//
// We use the following terminology to describe the different dimensions.
//
// B:  Batch size (number of sequences),
// L:  Sequence length,
// D:  Hidden dimension,
// H:  Number of heads,
// Dh: Hidden dimension per head - Dh = D / H.

template<typename T>
struct Multihead_attention_params_base {

    // The output buffer. Dimensions B x D.
    T* out = nullptr;

    // The input Qs and the associated bias. Dimensions B x D and D, resp.
    const T *q = nullptr, *q_bias = nullptr;
    // The input Ks and the associated bias. Dimensions B x D and D, resp.
    const T *k = nullptr, *k_bias = nullptr;
    // The input Vs and the associated bias. Dimensions B x D and D, resp.
    const T *v = nullptr, *v_bias = nullptr;

    // The cache for the Ks. The size must be at least B x L x D.
    T* k_cache = nullptr;
    // The cache for the Vs. The size must be at least B x L x D.
    T* v_cache = nullptr;
    // The indirections to use for cache when beam sampling.
    const int* cache_indir = nullptr;

    // Stride to handle the case when KQV is a single buffer
    int stride_q = 0;
    int stride_k = 0;
    int stride_v = 0;

    // The batch size.
    int batch_size = 0;
    // The beam width
    int beam_width = 0;
    // The sequence length.
    int memory_max_len = 0;
    // The number of heads (H).
    int num_heads = 0;
    int num_heads_kv = 0;
    int num_heads_q_kv_ratio = 0;
    // The hidden dimension per head (Dh).
    int hidden_size_per_head = 0;
    // The per-head latent space reserved for rotary embeddings.
    int  rotary_embedding_dim = 0;
    bool neox_rotary_style    = false;
    float rotary_base = 0.0f;
    // The maximum length of input sentences.
    int max_input_length = 0;
    // The current timestep. TODO(bhsueh) Check that do we only this param in cross attention?
    int timestep = 0;
    // The current timestep of each sentences (support different timestep for different sentences)

    // The 1.f / sqrt(Dh). Computed on the host.
    float inv_sqrt_dh = 0.0f;

    // Used when we have some input context like gpt
    const int* total_padding_tokens = nullptr;

    const bool* masked_tokens            = nullptr;
    const int*  prefix_prompt_lengths    = nullptr;
    int         max_prefix_prompt_length = 0;

    const T* relative_attention_bias        = nullptr;
    int      relative_attention_bias_stride = 0;
    // The slope per head of linear position bias to attention score (H).
    const T* linear_bias_slopes = nullptr;

    const T*   ia3_key_weights   = nullptr;
    const T*   ia3_value_weights = nullptr;
    const int* ia3_tasks         = nullptr;

    const float* qkv_scale_out       = nullptr;
    const float* attention_out_scale = nullptr;
    int          int8_mode           = 0;

    const T *rotary_cos = nullptr;
    const T *rotary_sin = nullptr;

    const int *nnz_head_idx = nullptr;
    int nnz_heads = 0;
};

template<typename T, bool CROSS_ATTENTION>
struct Multihead_attention_params: public Multihead_attention_params_base<T> {
    // output cross attentions
    float* cross_attention_out        = nullptr;
    int    max_decoder_seq_len        = 0;
    bool   is_return_cross_attentions = false;

    // allows to exist attention eary
    bool* finished = nullptr;

    // required in case of cross attention
    // will need it here till if constexpr in c++17
    int* memory_length_per_sample = nullptr;

    // required in case of masked attention with different length
    const int* length_per_sample = nullptr;
};

template<typename T>
struct Multihead_attention_params<T, true>: public Multihead_attention_params_base<T> {
    // output cross attentions
    float* cross_attention_out        = nullptr;
    int    max_decoder_seq_len        = 0;
    bool   is_return_cross_attentions = false;

    // allows to exist attention eary
    bool* finished = nullptr;

    // required in case of cross attention
    int* memory_length_per_sample = nullptr;

    // required in case of masked attention with different length
    const int* length_per_sample = nullptr;
};

template<class T>
using Masked_multihead_attention_params = Multihead_attention_params<T, false>;

template<class T>
using Cross_multihead_attention_params = Multihead_attention_params<T, true>;

template<typename T>
struct outputCrossAttentionParam {
    // max decoder output length
    int  max_decoder_seq_len        = 0;
    T*   cross_attention_out        = nullptr;
    bool is_return_cross_attentions = false;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

void masked_multihead_attention(const Masked_multihead_attention_params<float>& params, const cudaStream_t& stream);
void masked_multihead_attention(const Masked_multihead_attention_params<uint16_t>& params, const cudaStream_t& stream);
#ifdef ENABLE_BF16
void masked_multihead_attention(const Masked_multihead_attention_params<__nv_bfloat16>& params,
                                const cudaStream_t&                                     stream);
#endif
void cross_multihead_attention(const Cross_multihead_attention_params<float>& params, const cudaStream_t& stream);
void cross_multihead_attention(const Cross_multihead_attention_params<uint16_t>& params, const cudaStream_t& stream);
#ifdef ENABLE_BF16
void cross_multihead_attention(const Cross_multihead_attention_params<__nv_bfloat16>& params,
                               const cudaStream_t&                                    stream);
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////
