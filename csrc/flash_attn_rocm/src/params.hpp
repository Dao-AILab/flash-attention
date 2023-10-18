// BSD 3 Clause
// Copyright 2023 Advanced Micro Devices, Inc.
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include <vector>
#include <memory>

#include "utils.hpp"

// TODO: Use shared_ptr to use the same memory of BaseParams when calling forward/backward parameters
// Common argements used by both batched & grouped gemms
struct BaseParams {
  explicit BaseParams(const Index b,
                      const Index seqlen_q,
                      const Index seqlen_kv,
                      const Index seqlen_q_rounded, // TODO: remove me
                      const Index seqlen_k_rounded, // TODO: remove me
                      const Index h_q,
                      const Index h_kv,
                      const Index d,
                      const Index d_rounded, //TODO: remove me
                      const torch::Tensor &q,
                      const torch::Tensor &k,
                      const torch::Tensor &v,
                      torch::Tensor &out,
                      const float p_dropout,
                      const float softmax_scale,
                      const bool is_causal,
                      const bool z_permute)
    : b(b),
      seqlen_q(seqlen_q),
      seqlen_kv(seqlen_kv),
      seqlen_q_rounded(seqlen_q_rounded), // TODO: remove me
      seqlen_k_rounded(seqlen_k_rounded), // TODO: remove me
      h_q(h_q),
      h_kv(h_kv),
      d(d),
      d_rounded(d_rounded), //TODO: remove me
      p_dropout(p_dropout),
      softmax_scale(softmax_scale),
      is_bf16(q.dtype() == torch::kBFloat16),
      is_dropout(p_dropout > 0.0f),
      is_mnko_padding(false),
      is_causal(is_causal),
      z_permute(z_permute),
      q_seq_stride(q.stride(-3)),
      kv_seq_stride(k.stride(-3)),
      out_seq_stride(out.stride(-3)),
      q_head_stride(q.stride(-2)),
      kv_head_stride(k.stride(-2)),
      out_head_stride(out.stride(-2)) {

    TORCH_CHECK(p_dropout < 1.f);
    
    if(!is_mnko_padding && d <= 32) {
      is_mnko_padding = ((d % 32)==0 ? false : true);
    } else if(!is_mnko_padding && d <= 64) {
      is_mnko_padding = ((d % 64)==0 ? false : true);
    } else if(!is_mnko_padding && d <= 128) {
      is_mnko_padding = ((d % 128)==0 ? false : true);
    } else {
      std::cout << "Unsupported head dimension" << std::endl;
    }
  }  
  // The dimensions.
  Index b, seqlen_q, seqlen_kv, d, seqlen_q_rounded, seqlen_k_rounded, d_rounded;  // TODO: remove seqlen_q_rounded, seqlen_k_rounded, d_rounded;

  // The number of heads.
  Index h_q, h_kv;
  // In the case of multi-query and grouped-query attention (MQA/GQA), nheads_k could be
  // different from nheads (query).
  // int h_h_k_ratio; // precompute h / h_k

  // The scaling factors for the kernel.
  float softmax_scale;
  // float softmax_scale_log2;

  // The dropout probability (probability of keeping an activation).
  float p_dropout;
  // uint8_t p_dropout_in_uint8_t;

  // Scale factor of 1 / (1 - p_dropout).
  // float rp_dropout;
  // float scale_softmax_rp_dropout;

  // Random state.
  at::PhiloxCudaState philox_args;

  // seeds
  std::tuple<uint64_t, uint64_t> seeds;

  // Pointer to the RNG seed (idx 0) and offset (idx 1).
  uint64_t* rng_state;

  bool is_bf16;
  bool is_dropout;
  bool is_mnko_padding;
  bool is_causal;

  bool z_permute;

  Index q_seq_stride;
  Index kv_seq_stride;
  Index out_seq_stride;

  Index q_head_stride;
  Index kv_head_stride;
  Index out_head_stride;

  static inline const bool kIsUnitTestMode = get_env_("FLASH_ATTENTION_INTERNAL_UNIT_TEST_MODE");
  static inline const bool kIsDeterministic = get_env_("FLASH_ATTENTION_INTERNAL_DETERMINISTIC");
};

// Common Batched Arguments
struct BatchedParams : public BaseParams {
  explicit BatchedParams(const Index b,
                         const Index seqlen_q,
                         const Index seqlen_kv,
                         const Index seqlen_q_rounded, //TODO: remove me
                         const Index seqlen_k_rounded, //TODO: remove me
                         const Index h_q,
                         const Index h_kv,
                         const Index d,
                         const Index d_rounded, //TODO: remove me
                         const torch::Tensor &q,
                         const torch::Tensor &k,
                         const torch::Tensor &v,
                         torch::Tensor &out,
                         void* z_d,
                         void* softmax_lse_d,
                         float p_dropout,
                         float softmax_scale,
                         bool is_causal,
                         bool z_permute)
    : BaseParams(b,
                 seqlen_q,
                 seqlen_k,
                 seqlen_q_rounded, //TODO: remove me
                 seqlen_k_rounded, //TODO: remove me
                 h_q,
                 h_kv,
                 d,
                 d_rounded, //TODO: remove me
                 q,
                 k,
                 v,
                 out,
                 p_dropout,
                 softmax_scale,
                 is_causal,
                 z_permute),
      q_ptr(q.data_ptr()),
      k_ptr(k.data_ptr()),
      v_ptr(v.data_ptr()),
      out_ptr(out.data_ptr()),
      z_ptr(z_d),
      softmax_lse_ptr(softmax_lse_d),
      q_batch_stride(q.stride(0)),
      kv_batch_stride(k.stride(0)),
      out_batch_stride(out.stride(0)) {
    
    if(!is_mnko_padding && d <= 32) {
      is_mnko_padding = ((seqlen_q % 128)==0 && (seqlen_kv % 128)==0 ? false : true);
    } else if(!is_mnko_padding && d <= 64) {
      if(is_dropout) {
        is_mnko_padding = ((seqlen_q % 128)==0 && (seqlen_kv % 128)==0 ? false : true);
      } else {
        is_mnko_padding = ((seqlen_q % 128)==0 && (seqlen_kv % 256)==0 ? false : true);
      }
    } else if(!is_mnko_padding && d <= 128) {
      is_mnko_padding = ((seqlen_q % 128)==0 && (seqlen_kv % 128)==0 ? false : true);
    }

    // Q layout [b, seqlen_q, h_q, d]
    std::vector<Index> q_lengths{b, h_q, seqlen_q, d};
    std::vector<Index> q_strides{q_batch_stride, q_head_stride, q_seq_stride, 1};

    // K layout [b, seqlen_kv, h_kv, d]
    std::vector<Index> k_lengths{b, h_kv, seqlen_kv, d};
    std::vector<Index> k_strides{kv_batch_stride, kv_head_stride, kv_seq_stride, 1};

    // V layout [b, seqlen_kv, h_kv, d]
    std::vector<Index> v_lengths{b, h_kv, d, seqlen_kv};
    std::vector<Index> v_strides{kv_batch_stride, kv_head_stride, 1, kv_seq_stride};

    // Y layout [b, seqlen_q, h_q, d]
    std::vector<Index> out_lengths{b, h_q, seqlen_q, d};
    std::vector<Index> out_strides{out_batch_stride, out_head_stride, out_seq_stride, 1};

    std::vector<Index> z_lengths{b, h_q, seqlen_q, seqlen_kv};
    std::vector<Index> z_strides = 
        z_permute ? 
        std::vector<Index>{h*seqlen_q*seqlen_kv, seqlen_kv, h*seqlen_kv, 1} :
        // Z layout [b, seqlen_q, h_q, seqlen_kv]
        std::vector<Index>{h*seqlen_q*seqlen_kv, seqlen_q*seqlen_kv, seqlen_kv, 1};
        // Z layout [b, h_q, seqlen_q, seqlen_kv]

    // LSE layout [b, h, seqlen_q]
    std::vector<Index> lse_lengths{b, h_q, seqlen_q};
    // std::vector<Index> lse_strides{h_q*seqlen_q, seqlen_q, 1};
  }
  
  void* __restrict__ q_ptr;
  void* __restrict__ k_ptr;
  void* __restrict__ v_ptr;

  void* __restrict__ out_ptr;
  void* __restrict__ z_ptr;
  void* __restrict__ softmax_lse_ptr;

  int q_batch_stride;
  int kv_batch_stride;
  int out_batch_stride;

  std::vector<Index> q_lengths;
  std::vector<Index> q_strides;
  std::vector<Index> k_lengths;
  std::vector<Index> k_strides;
  std::vector<Index> v_lengths;
  std::vector<Index> v_strides;
  std::vector<Index> out_lengths;
  std::vector<Index> out_strides;
  std::vector<Index> z_lengths;  
  std::vector<Index> z_strides;
  std::vector<Index> lse_lengths;
  // std::vector<Index> lse_strides;
};

// Forward Batched Arguments
struct FlashFwdBatchedParams : public BatchedParams {
  explicit FlashFwdBatchedParams(const Index b,
                                 const Index seqlen_q,
                                 const Index seqlen_k,
                                 const Index seqlen_q_rounded,
                                 const Index seqlen_k_rounded,
                                 const Index h,
                                 const Index h_k,
                                 const Index d,
                                 const Index d_rounded,
                                 const torch::Tensor &q,
                                 const torch::Tensor &k,
                                 const torch::Tensor &v,
                                 torch::Tensor &out,
                                 void* z_d,
                                 void* softmax_lse_d,
                                 float p_dropout,
                                 float softmax_scale,
                                 bool is_causal)
    : BatchedParams(b,
                    seqlen_q,
                    seqlen_k,
                    seqlen_q_rounded,
                    seqlen_k_rounded,
                    h,
                    h_k,
                    d,
                    d_rounded,
                    q,
                    k,
                    v,
                    out,
                    z_d,
                    softmax_lse_d,
                    p_dropout,
                    softmax_scale,
                    is_causal,
                    false) {}
};

// Backward Batched Arguments
struct FlashBwdBatchedParams : public BatchedParams {
  explicit FlashBwdBatchedParams(const Index b,
                                 const Index seqlen_q,
                                 const Index seqlen_k,
                                 const Index seqlen_q_rounded,
                                 const Index seqlen_k_rounded,
                                 const Index h,
                                 const Index h_k,
                                 const Index d,
                                 const Index d_rounded,
                                 const torch::Tensor &q,
                                 const torch::Tensor &k,
                                 const torch::Tensor &v,
                                 torch::Tensor out,
                                 const torch::Tensor &dout,
                                 torch::Tensor &dq,
                                 torch::Tensor &dk,
                                 torch::Tensor &dv,
                                 void* z_d,
                                 void* softmax_lse_d,
                                 float p_dropout,
                                 float softmax_scale,
                                 bool is_causal)
    : BatchedParams(b,
                    seqlen_q,
                    seqlen_k,
                    seqlen_q_rounded,
                    seqlen_k_rounded,
                    h,
                    h_k,
                    d,
                    d_rounded,
                    q,
                    k,
                    v,
                    out,
                    z_d,
                    softmax_lse_d,
                    p_dropout,
                    softmax_scale,
                    is_causal,
                    true),
      dq_ptr(dq.data_ptr()),
      dk_ptr(dk.data_ptr()),
      dv_ptr(dv.data_ptr()),
      dout_ptr(dout.data_ptr()),
      d_ptr(torch::empty({b, static_cast<long>(h), seqlen_q}, 
            q.options().dtype(torch::kFloat32)).data_ptr()) {

    Index dkv_batch_stride = dk.stride(0);
    Index dkv_seq_stride = dk.stride(-3);
    Index dkv_head_stride = dk.stride(-2);
      
    // KGrad layout [b, seqlen_kv, h_q, d]
    std::vector<Index> dk_lengths{b, h_q, seqlen_kv, d};
    std::vector<Index> dk_strides{dkv_batch_stride, dkv_head_stride, dkv_seq_stride, 1};

    // VGrad layout [b, seqlen_kv, h_q, d]
    std::vector<Index> dv_lengths{b, h_q, d, seqlen_kv};
    std::vector<Index> dv_strides{dkv_batch_stride, dkv_head_stride, 1, dkv_seq_stride};  
  }

  void* __restrict__ dq_ptr;
  void* __restrict__ dk_ptr;
  void* __restrict__ dv_ptr;

  void* __restrict__ dout_ptr;
  void* __restrict__ d_ptr;
};

// Common Grouped Arguments
struct GroupedParams : public BaseParams {
  explicit GroupedParams(const Index b,
                         const Index seqlen_q,
                         const Index seqlen_k,
                         const Index seqlen_q_rounded,
                         const Index seqlen_k_rounded,
                         const Index h,
                         const Index h_k,
                         const Index d,
                         const Index d_rounded,
                         const at::Tensor &q,
                         const at::Tensor &k,
                         const at::Tensor &v,
                         torch::Tensor &out,
                         void* cu_seqlens_q_d,
                         void* cu_seqlens_k_d,
                         void* z_d,
                         void* softmax_lse_d,
                         float p_dropout,
                         float softmax_scale,
                         bool is_causal,
                         const bool z_permute)
    : BaseParams(b,
                 seqlen_q,
                 seqlen_k,
                 seqlen_q_rounded,
                 seqlen_k_rounded,
                 h,
                 h_k,
                 d,
                 d_rounded,
                 q,
                 k,
                 v,
                 out,
                 p_dropout,
                 softmax_scale,
                 is_causal,
                 z_permute),
      host_seqlens_q(std::vector<int>(b+1)),
      host_seqlens_k(std::vector<int>(b+1)) {
    
    char* q_ptr = reinterpret_cast<char*>(q.data_ptr());
    char* k_ptr = reinterpret_cast<char*>(k.data_ptr());
    char* v_ptr = reinterpret_cast<char*>(v.data_ptr());

    char* out_ptr = reinterpret_cast<char*>(out.data_ptr());
    char* z_ptr = reinterpret_cast<char*>(z_d);
    char* softmax_lse_ptr = reinterpret_cast<char*>(softmax_lse_d);  

    FMHA_CHECK_HIP(hipMemcpy(host_seqlens_q.data(),   
                             cu_seqlens_q_d, 
                             (b+1)*sizeof(int), 
                             hipMemcpyDeviceToHost));
    FMHA_CHECK_HIP(hipMemcpy(host_seqlens_k.data(), 
                             cu_seqlens_k_d, 
                             (b+1)*sizeof(int), 
                             hipMemcpyDeviceToHost));

    problem_descs.reserve(b);

    for (int i = 0; i < b; ++i) {
      int curr_seqlen_q = host_seqlens_q[i+1] - host_seqlens_q[i];
      int curr_seqlen_kv = host_seqlens_k[i+1] - host_seqlens_k[i];

      int curr_q_batch_stride = curr_seqlen_q * q_seq_stride;
      int curr_kv_batch_stride = curr_seqlen_kv * kv_seq_stride;
      int curr_out_batch_stride = curr_seqlen_q * out_seq_stride;

      if(!is_mnko_padding && d <= 32) {
        is_mnko_padding = ((curr_seqlen_q % 128)==0 && (curr_seqlen_kv % 128)==0 ? false : true);
      } else if(!is_mnko_padding && d <= 64) {
        if(is_dropout) {
          is_mnko_padding = ((curr_seqlen_q % 128)==0 && (curr_seqlen_kv % 128)==0 ? false : true);
        } else {
          is_mnko_padding = ((curr_seqlen_q % 128)==0 && (curr_seqlen_kv % 256)==0 ? false : true);
        }
      } else if(!is_mnko_padding && d <= 128) {
        is_mnko_padding = ((curr_seqlen_q % 128)==0 && (curr_seqlen_kv % 128)==0 ? false : true);
      }

      q_ptrs.push_back(reinterpret_cast<void*>(q_ptr));
      q_ptr = q_ptr + get_size_in_bytes(curr_q_batch_stride, q.dtype());

      k_ptrs.push_back(reinterpret_cast<void*>(k_ptr));
      k_ptr = k_ptr + get_size_in_bytes(curr_kv_batch_stride, k.dtype());

      v_ptrs.push_back(reinterpret_cast<void*>(v_ptr));     
      v_ptr = v_ptr + get_size_in_bytes(curr_kv_batch_stride, v.dtype());      

      out_ptrs.push_back(reinterpret_cast<void*>(out_ptr));
      out_ptr = out_ptr + get_size_in_bytes(out_seq_stride, out.dtype());

      softmax_lse_ptrs.push_back(reinterpret_cast<void*>(softmax_lse_ptr));
      softmax_lse_ptr = softmax_lse_ptr + get_size_in_bytes(h * seqlen_q, torch::kFloat32);

      if(z_d) {
        z_ptrs.push_back(reinterpret_cast<void*>(z_ptr + i * h_q * seqlen_q * seqlen_kv * sizeof(int)));
      }
      else{
        z_ptrs.push_back(nullptr);
      }

      // Q layout [b, seqlen_q, h, d]
      std::vector<Index> q_lengths{1, h_q, curr_seqlen_q, d};
      std::vector<Index> q_strides{curr_q_batch_stride, q_head_stride, q_seq_stride, 1};
 
      // K layout [b, seqlen_kv, h, d]
      std::vector<Index> k_lengths{1, h_kv, curr_seqlen_kv, d};
      std::vector<Index> k_strides{curr_kv_batch_stride, kv_head_stride, kv_seq_stride, 1};

      // V layout [b, seqlen_kv, h, d]
      std::vector<Index> v_lengths{1, h_kv, d, curr_seqlen_kv};
      std::vector<Index> v_strides{curr_kv_batch_stride, kv_head_stride, 1, kv_seq_stride};

      // Y layout [b, seqlen_q, h, O]
      std::vector<Index> out_lengths{1, h_q, curr_seqlen_q, d};
      std::vector<Index> out_strides{curr_out_batch_stride, out_head_stride, out_seq_stride, 1};

      std::vector<Index> z_lengths{1, h_q, curr_seqlen_q, curr_seqlen_kv};
      std::vector<Index> z_strides = 
          z_permute ? 
          std::vector<Index>{h_q*curr_seqlen_q*curr_seqlen_kv, curr_seqlen_kv, h_q*curr_seqlen_kv, 1} :
          // Z layout [b, seqlen_q, h, seqlen_kv]
          std::vector<Index>{h_q*curr_seqlen_q*curr_seqlen_kv, curr_seqlen_q*curr_seqlen_kv, curr_seqlen_kv, 1};
          // Z layout [b, h, seqlen_q, seqlen_kv]

      // LSE layout [b, h, seqlen_q]
      std::vector<Index> lse_lengths{1, h_q, curr_seqlen_q};
      std::vector<Index> lse_strides{h_q*curr_seqlen_q, curr_seqlen_q, 1};

      problem_descs.push_back({
          q_lengths,
          q_strides,
          k_lengths,
          k_strides,
          v_lengths,
          v_strides,
          out_lengths,
          out_strides,
          z_lengths,
          z_strides,
          dk_lengths,
          dk_strides,
          dv_lengths,
          dv_strides,
          lse_lengths,
          lse_strides,
      });
    }
  }

  std::vector<const void*> q_ptrs;
  std::vector<const void*> k_ptrs;
  std::vector<const void*> v_ptrs;

  std::vector<void*> out_ptrs;
  std::vector<void*> z_ptrs;
  std::vector<void*> softmax_lse_ptrs;

  std::vector<int> host_seqlens_q;
  std::vector<int> host_seqlens_k;

  std::vector<std::vector<Index>> q_lengths;
  std::vector<std::vector<Index>> q_strides;
  std::vector<std::vector<Index>> k_lengths;
  std::vector<std::vector<Index>> k_strides;
  std::vector<std::vector<Index>> v_lengths;
  std::vector<std::vector<Index>> v_strides;
  std::vector<Index><std::vector<Index>> out_lengths;
  std::vector<Index><std::vector<Index>> out_strides;
  std::vector<Index><std::vector<Index>> z_lengths;
  std::vector<Index><std::vector<Index>> z_strides;
  std::vector<Index><std::vector<Index>> dk_lengths;
  std::vector<Index><std::vector<Index>> dk_strides;
  std::vector<Index><std::vector<Index>> dv_lengths;
  std::vector<Index><std::vector<Index>> dv_strides;
  std::vector<Index><std::vector<Index>> lse_lengths;
  std::vector<Index><std::vector<Index>> lse_strides;
};

// Forward Grouped Arguments
struct FlashFwdGroupedParams : public GroupedParams {
  explicit FlashFwdGroupedParams(const Index b,
                                 const Index seqlen_q,
                                 const Index seqlen_k,
                                 const Index seqlen_q_rounded,
                                 const Index seqlen_k_rounded,
                                 const Index h,
                                 const Index h_k,
                                 const Index d,
                                 const Index d_rounded,
                                 const torch::Tensor &q,
                                 const torch::Tensor &k,
                                 const torch::Tensor &v,
                                 torch::Tensor &out,
                                 void* cu_seqlens_q_d,
                                 void* cu_seqlens_k_d,
                                 void* z_d,
                                 void* softmax_lse_d,
                                 float p_dropout,
                                 float softmax_scale,
                                 bool is_causal) 
    : GroupedParams(b,
                    seqlen_q,
                    seqlen_k,
                    seqlen_q_rounded,
                    seqlen_k_rounded,
                    h,
                    h_k,
                    d,
                    d_rounded,
                    q,
                    k,
                    v,
                    out,
                    cu_seqlens_q_d,
                    cu_seqlens_k_d,
                    z_d,
                    softmax_lse_d,
                    p_dropout,
                    softmax_scale,
                    is_causal,
                    false) {}
};

// Backward Grouped Arguments
struct FlashBwdGroupedParams : public GroupedParams {
  explicit FlashBwdGroupedParams(const Index b,
                                 const Index seqlen_q,
                                 const Index seqlen_k,
                                 const Index seqlen_q_rounded,
                                 const Index seqlen_k_rounded,
                                 const Index h,
                                 const Index h_k,
                                 const Index d,
                                 const Index d_rounded,
                                 const torch::Tensor &q,
                                 const torch::Tensor &k,
                                 const torch::Tensor &v,
                                 torch::Tensor out,
                                 const torch::Tensor &dout,
                                 torch::Tensor &dq,
                                 torch::Tensor &dk,
                                 torch::Tensor &dv,
                                 void* cu_seqlens_q_d,
                                 void* cu_seqlens_k_d,
                                 void* z_d,
                                 void* softmax_lse_d,
                                 float p_dropout,
                                 float softmax_scale,
                                 bool is_causal)
    : GroupedParams(b,
                    seqlen_q,
                    seqlen_k,
                    seqlen_q_rounded,
                    seqlen_k_rounded,
                    h,
                    h_k,
                    d,
                    d_rounded,
                    q,
                    k,
                    v,
                    out,
                    cu_seqlens_q_d,
                    cu_seqlens_k_d,
                    z_d,
                    softmax_lse_d,
                    p_dropout,
                    softmax_scale,
                    is_causal,
                    true),
      bwd_out_ptrs(std::vector<const void*>(out_ptrs.begin(), out_ptrs.end())),
      bwd_softmax_lse_ptrs(std::vector<const void*>(softmax_lse_ptrs.begin(), softmax_lse_ptrs.end())) {
    
    char* q_ptr = reinterpret_cast<char*>(q.data_ptr());  
    char* k_ptr = reinterpret_cast<char*>(k.data_ptr());
    char* v_ptr = reinterpret_cast<char*>(v.data_ptr());

    char* dq_ptr = reinterpret_cast<char*>(dq.data_ptr());
    char* dk_ptr = reinterpret_cast<char*>(dk.data_ptr());
    char* dv_ptr = reinterpret_cast<char*>(dv.data_ptr());
    char* dout_ptr = reinterpret_cast<char*>(dout.data_ptr());

    for (int i = 0; i < b; ++i) {
      int curr_seqlen_q = host_seqlens_q[i+1] - host_seqlens_q[i];
      int temp_q_stride = get_size_in_bytes(d * h * curr_seqlen_q, q.dtype());
      int temp_dq_stride = get_size_in_bytes(d * h * curr_seqlen_q, dq.dtype());
      int curr_seqlen_kv = host_seqlens_k[i+1] - host_seqlens_k[i];
      int temp_k_stride = get_size_in_bytes(d * h * curr_seqlen_kv, q.dtype());
      int temp_dk_stride = get_size_in_bytes(d * h * curr_seqlen_kv, dk.dtype());

      if(!is_mnko_padding && d <= 32) {
        is_mnko_padding = ((curr_seqlen_q % 128)==0 && (curr_seqlen_kv % 128)==0 ? false : true);
      }
      else if(!is_mnko_padding && d <= 64) {
        is_mnko_padding = ((curr_seqlen_q % 128)==0 && (curr_seqlen_kv % 128)==0 ? false : true);
      }
      else if(!is_mnko_padding && d <= 128) {
        is_mnko_padding = ((curr_seqlen_q % 64)==0 && (curr_seqlen_kv % 128)==0 ? false : true);
      }

      auto opts = q.options();
  
      dout_ptrs.push_back(reinterpret_cast<const void*>(dout_ptr));
      dout_ptr += temp_q_stride;

      torch::Tensor d_tensor;
      d_tensor = torch::empty({1, static_cast<long>(h), curr_seqlen_q}, opts.dtype(torch::kFloat32));
      d_ptrs.push_back(reinterpret_cast<void*>(d_tensor.data_ptr()));

      dq_ptrs.push_back(reinterpret_cast<void*>(dq_ptr));
      dq_ptr = dq_ptr + temp_dq_stride;

      dk_ptrs.push_back(reinterpret_cast<void*>(dk_ptr));
      dk_ptr = dk_ptr + temp_dk_stride;

      dv_ptrs.push_back(reinterpret_cast<void*>(dv_ptr));
      dv_ptr = dv_ptr + temp_dk_stride;
    }            
  }

  std::vector<void*> dq_ptrs;
  std::vector<void*> dk_ptrs;
  std::vector<void*> dv_ptrs;

  std::vector<const void*> bwd_out_ptrs;
  std::vector<const void*> bwd_softmax_lse_ptrs;

  std::vector<const void*> dout_ptrs;
  std::vector<void*> d_ptrs;
};
