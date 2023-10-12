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

// Common argements used by both batched & grouped gemms
struct BaseParams {
  explicit BaseParams(const Index b,
                      const Index seqlen_q,
                      const Index seqlen_k,
                      const Index seqlen_q_rounded,
                      const Index seqlen_k_rounded,
                      const Index h,
                      const Index h_k,
                      const Index d,
                      const Index d_rounded,
                      const at::Tensor q,
                      const at::Tensor k,
                      const at::Tensor v,
                      const float p_dropout,
                      const float softmax_scale,
                      const bool is_causal)
    : b(b),
      seqlen_q(seqlen_q),
      seqlen_k(seqlen_k),
      seqlen_q_rounded(seqlen_q_rounded),
      seqlen_k_rounded(seqlen_k_rounded),
      h(h),
      h_k(h_k),
      d(d),
      d_rounded(d_rounded),
      p_dropout(p_dropout),
      softmax_scale(softmax_scale),
      is_bf16(q.dtype() == torch::kBFloat16),
      is_dropout(p_dropout > 0.0f),
      is_mnko_padding(false),
      is_causal(is_causal) {

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

    if(q.is_contiguous() && k.is_contiguous() && v.is_contiguous()) {  
      q_stride_multiplier = 1;
      kv_stride_multiplier = 1;
    } else if(q.is_contiguous() && !k.is_contiguous() && !v.is_contiguous()) {
      q_stride_multiplier = 1;
      kv_stride_multiplier = 2;
    } else if(!q.is_contiguous() && !k.is_contiguous() && !v.is_contiguous()) {
      q_stride_multiplier = 3;
      kv_stride_multiplier = 3;
    } else {
      std::cout<< "Wrong matrix inputs." <<std::endl;
    }
  }

  // The dimensions.
  int b, seqlen_q, seqlen_k, d, seqlen_q_rounded, seqlen_k_rounded, d_rounded;

  // The number of heads.
  int h, h_k;
  // In the case of multi-query and grouped-query attention (MQA/GQA), nheads_k could be
  // different from nheads (query).
  int h_h_k_ratio; // precompute h / h_k,

  // The scaling factors for the kernel.
  float softmax_scale;
  // float softmax_scale_log2;

  int q_stride_multiplier;
  int kv_stride_multiplier;

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

  static const bool input_permute = true;
  static const bool output_permute = true;
  static const bool z_tensor_permute = false;

  static inline const bool kIsUnitTestMode = get_env_("FLASH_ATTENTION_INTERNAL_UNIT_TEST_MODE");
  static inline const bool kIsDeterministic = get_env_("FLASH_ATTENTION_INTERNAL_DETERMINISTIC");
};

// Common Batched Arguments
struct BatchedParams : public BaseParams {
  explicit BatchedParams(const Index b,
                         const Index seqlen_q,
                         const Index seqlen_k,
                         const Index seqlen_q_rounded,
                         const Index seqlen_k_rounded,
                         const Index h,
                         const Index h_k,
                         const Index d,
                         const Index d_rounded,
                         const at::Tensor q,
                         const at::Tensor k,
                         const at::Tensor v,
                         at::Tensor out,
                         void* z_d,
                         void* softmax_lse_d,
                         float p_dropout,
                         float softmax_scale,
                         bool is_causal)
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
                 p_dropout,
                 softmax_scale,
                 is_causal),
      q_ptr(q.data_ptr()),
      k_ptr(k.data_ptr()),
      z_ptr(z_d),
      v_ptr(v.data_ptr()),
      out_ptr(out.data_ptr()),
      softmax_lse_ptr(softmax_lse_d),
      q_gs_ms_ks_lengths({b, h, seqlen_q_rounded, d}),
      q_gs_ms_ks_strides(
            input_permute
          ? std::vector<Index>{seqlen_q_rounded * h * d * q_stride_multiplier, d, h * d * q_stride_multiplier, 1}    // A layout [b, seqlen_q_rounded, h, d]
          : std::vector<Index>{h * seqlen_q_rounded * d, seqlen_q_rounded * d, d, 1}),   // A layout [b, h, seqlen_q_rounded, d]
      k_gs_ns_ks_lengths({b, h, seqlen_k_rounded, d}),
      k_gs_ns_ks_strides(
            input_permute
          ? std::vector<Index>{seqlen_k_rounded * h * d * kv_stride_multiplier, d, h * d * kv_stride_multiplier, 1}  // B0 layout [b, seqlen_k_rounded, h, d]
          : std::vector<Index>{h * seqlen_k_rounded * d, seqlen_k_rounded * d, d, 1}), // B0 layout [b, h, seqlen_k_rounded, d]
      z_gs_ms_ns_lengths({b, h, seqlen_q_rounded, seqlen_k_rounded}),  
      z_gs_ms_ns_strides( 
            z_tensor_permute
          ? std::vector<Index>{seqlen_q_rounded * h * seqlen_k_rounded, seqlen_k_rounded, h * seqlen_k_rounded, 1}  // Z layout [b, seqlen_q_rounded, h, seqlen_k_rounded]
          : std::vector<Index>{h * seqlen_q_rounded * seqlen_k_rounded, seqlen_q_rounded * seqlen_k_rounded, seqlen_k_rounded, 1}), // Z layout [b, h, seqlen_q_rounded, seqlen_k_rounded]
      v_gs_gemm1ns_gemm1ks_lengths({b, h, d, seqlen_k_rounded}),
      v_gs_gemm1ns_gemm1ks_strides(
            input_permute
          ? std::vector<Index>{seqlen_k_rounded * h * d * kv_stride_multiplier, d, 1, h * d * kv_stride_multiplier}  // B1 layout [b, seqlen_k_rounded, h, d]
          : std::vector<Index>{h * seqlen_k_rounded * d, seqlen_k_rounded * d, 1, d}), // B1 layout [b, h, seqlen_k_rounded, d]
      out_gs_ms_gemm1ns_lengths({b, h, seqlen_q_rounded, d}),  
      out_gs_ms_gemm1ns_strides(
            output_permute
          ? std::vector<Index>{seqlen_q_rounded * h * d, d, h * d, 1}  // C layout [b, seqlen_q_rounded, h, d]
          : std::vector<Index>{h * seqlen_q_rounded * d, seqlen_q_rounded * d, d, 1}), // C layout [b, h, seqlen_q_rounded, d]
      lse_gs_ms_lengths({b, h, seqlen_q_rounded}),                       // LSE layout [b, h, seqlen_q_rounded]
      lse_gs_ms_strides({h * seqlen_q_rounded, seqlen_q_rounded, 1}) {}
  
  void* __restrict__ q_ptr;
  void* __restrict__ k_ptr;
  void* __restrict__ z_ptr;
  void* __restrict__ v_ptr;

  void* __restrict__ out_ptr;
  void* __restrict__ softmax_lse_ptr;

  std::vector<Index> q_gs_ms_ks_lengths;
  std::vector<Index> q_gs_ms_ks_strides;
  std::vector<Index> k_gs_ns_ks_lengths;
  std::vector<Index> k_gs_ns_ks_strides;
  std::vector<Index> z_gs_ms_ns_lengths;  
  std::vector<Index> z_gs_ms_ns_strides;
  std::vector<Index> v_gs_gemm1ns_gemm1ks_lengths;    // b1_gs_os_ns_lengths
  std::vector<Index> v_gs_gemm1ns_gemm1ks_strides;    // b1_gs_os_ns_strides
  std::vector<Index> out_gs_ms_gemm1ns_lengths;       // c_gs_ms_os_lengths
  std::vector<Index> out_gs_ms_gemm1ns_strides;       // c_gs_ms_os_strides
  std::vector<Index> lse_gs_ms_lengths;
  std::vector<Index> lse_gs_ms_strides;
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
                                 const at::Tensor q,
                                 const at::Tensor k,
                                 const at::Tensor v,
                                 at::Tensor out,
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
                    is_causal) {}
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
                                 const at::Tensor q,
                                 const at::Tensor k,
                                 const at::Tensor v,
                                 const at::Tensor out,
                                 const at::Tensor dout,
                                 at::Tensor dq,
                                 at::Tensor dk,
                                 at::Tensor dv,
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
                    is_causal),
      dq_ptr(dq.data_ptr()),
      dk_ptr(dk.data_ptr()),
      dv_ptr(dv.data_ptr()),
      dout_ptr(dout.data_ptr()),
      d_ptr(at::empty({b, static_cast<long>(h), seqlen_q_rounded}, 
                      q.options().dtype(at::kFloat)).data_ptr()) {}

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
                         const at::Tensor q,
                         const at::Tensor k,
                         const at::Tensor v,
                         at::Tensor out,
                         void* cu_seqlens_q_d,
                         void* cu_seqlens_k_d,
                         void* z_d,
                         void* softmax_lse_d,
                         float p_dropout,
                         float softmax_scale,
                         bool is_causal)
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
                 p_dropout,
                 softmax_scale,
                 is_causal),
      host_seqlens_q(std::vector<int>(b+1)),
      host_seqlens_k(std::vector<int>(b+1)) {
    
    char* q_ptr = reinterpret_cast<char*>(q.data_ptr());
    char* k_ptr = reinterpret_cast<char*>(k.data_ptr());
    char* z_ptr = reinterpret_cast<char*>(z_d);
    char* v_ptr = reinterpret_cast<char*>(v.data_ptr());

    char* out_ptr = reinterpret_cast<char*>(out.data_ptr());
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
      int temp_seqlen_q = host_seqlens_q[i+1] - host_seqlens_q[i];
      int temp_q_stride = get_size_in_bytes(d * h * temp_seqlen_q, q.dtype());
      int temp_seqlen_k = host_seqlens_k[i+1] - host_seqlens_k[i];
      int temp_k_stride = get_size_in_bytes(d * h * temp_seqlen_k, q.dtype());

      if(!is_mnko_padding && d <= 32) {
        is_mnko_padding = ((temp_seqlen_q % 128)==0 && (temp_seqlen_k % 128)==0 ? false : true);
      } else if(!is_mnko_padding && d <= 64) {
        if(is_dropout) {
          is_mnko_padding = ((temp_seqlen_q % 128)==0 && (temp_seqlen_k % 128)==0 ? false : true);
        } else {
          is_mnko_padding = ((temp_seqlen_q % 128)==0 && (temp_seqlen_k % 256)==0 ? false : true);
        }
      } else if(!is_mnko_padding && d <= 128) {
        is_mnko_padding = ((temp_seqlen_q % 128)==0 && (temp_seqlen_k % 128)==0 ? false : true);
      }

      q_ptrs.push_back(reinterpret_cast<void*>(q_ptr));
      q_ptr = q_ptr + temp_q_stride * q_stride_multiplier;

      k_ptrs.push_back(reinterpret_cast<void*>(k_ptr));
      k_ptr = k_ptr + temp_k_stride * kv_stride_multiplier;

      v_ptrs.push_back(reinterpret_cast<void*>(v_ptr));     
      v_ptr = v_ptr + temp_k_stride * kv_stride_multiplier;      

      out_ptrs.push_back(reinterpret_cast<void*>(out_ptr));
      out_ptr = out_ptr + temp_q_stride;

      softmax_lse_ptrs.push_back(reinterpret_cast<void*>(softmax_lse_ptr));
      int temp_lse_stride = get_size_in_bytes(h * seqlen_q, torch::kFloat32);
      softmax_lse_ptr = softmax_lse_ptr + temp_lse_stride;

      if(z_d) {
        z_ptrs.push_back(reinterpret_cast<void*>(z_ptr + i * h * seqlen_q * seqlen_k * sizeof(int)));
      }
      else{
        z_ptrs.push_back(nullptr);
      }

      int K  = d;
      int O  = d;
      int G0 = 1;
      int G1 = h;
      int M = host_seqlens_q[i + 1] - host_seqlens_q[i]; //seqlen Q
      int N = host_seqlens_k[i + 1] - host_seqlens_k[i]; //seqlen K

      std::vector<Index> q_gs_ms_ks_lengths{G0, G1, M, K};
      std::vector<Index> q_gs_ms_ks_strides =
          input_permute
              ? std::vector<Index>{M * G1 * K * q_stride_multiplier, K, G1 * K * q_stride_multiplier, 1}
              // Q layout [G0, M, G1, K]
              : std::vector<Index>{G1 * M * K, M * K, K, 1}; // Q layout [G0, G1, M, K]

      std::vector<Index> k_gs_ns_ks_lengths{G0, G1, N, K};
      std::vector<Index> k_gs_ns_ks_strides =
          input_permute
              ? std::vector<Index>{N * G1 * K * kv_stride_multiplier, K, G1 * K * kv_stride_multiplier, 1}
              // K layout [G0, N, G1, K]
              : std::vector<Index>{G1 * N * K, N * K, K, 1}; // K layout [G0, G1, N, K]

      std::vector<Index> z_gs_ms_ns_lengths{G0, G1, M, N};
      std::vector<Index> z_gs_ms_ns_strides = 
          input_permute
          ? std::vector<Index>{M * G1 * N, N, G1 * N, 1}
          // Z layout [G0, M, G1, N]
          : std::vector<Index>{G1 * M * N, M * N, N, 1}; // Z layout [G0, G1, M, N]

      std::vector<Index> v_gs_os_ns_lengths{G0, G1, O, N};
      std::vector<Index> v_gs_os_ns_strides =
          input_permute
              ? std::vector<Index>{N * G1 * O * kv_stride_multiplier, O, 1, G1 * O * kv_stride_multiplier}
              // V layout [G0, N, G1, O]
              : std::vector<Index>{G1 * N * O, N * O, 1, O}; // V layout [G0, G1, N, O]

      std::vector<Index> out_gs_ms_os_lengths{G0, G1, M, O};
      std::vector<Index> out_gs_ms_os_strides =
          output_permute
              ? std::vector<Index>{M * G1 * O, O, G1 * O, 1}
              // Y layout [G0, M, G1, O]
              : std::vector<Index>{G1 * M * O, M * O, O, 1}; // Y layout [G0, G1, M, O]

      std::vector<Index> lse_gs_ms_lengths{G0, G1, M};
      std::vector<Index> lse_gs_ms_strides{G1 * M, M, 1}; // LSE layout [G0, G1, M]

      problem_descs.push_back({
          q_gs_ms_ks_lengths,
          q_gs_ms_ks_strides,
          k_gs_ns_ks_lengths,
          k_gs_ns_ks_strides,
          z_gs_ms_ns_lengths,
          z_gs_ms_ns_strides,
          v_gs_os_ns_lengths,
          v_gs_os_ns_strides,
          out_gs_ms_os_lengths,
          out_gs_ms_os_strides,
          lse_gs_ms_lengths,
          lse_gs_ms_strides,
      });
    }
  }

  std::vector<const void*> q_ptrs;
  std::vector<const void*> k_ptrs;
  std::vector<void*> z_ptrs;
  std::vector<const void*> v_ptrs;

  std::vector<void*> out_ptrs;
  std::vector<void*> softmax_lse_ptrs;

  std::vector<int> host_seqlens_q;
  std::vector<int> host_seqlens_k;

  struct ProblemDesc {
    std::vector<Index> q_gs_ms_ks_lengths;
    std::vector<Index> q_gs_ms_ks_strides;
    std::vector<Index> k_gs_ns_ks_lengths;
    std::vector<Index> k_gs_ns_ks_strides;
    std::vector<Index> z_gs_ms_ns_lengths;  
    std::vector<Index> z_gs_ms_ns_strides;
    std::vector<Index> v_gs_gemm1ns_gemm1ks_lengths;    // b1_gs_os_ns_lengths
    std::vector<Index> v_gs_gemm1ns_gemm1ks_strides;    // b1_gs_os_ns_strides
    std::vector<Index> out_gs_ms_gemm1ns_lengths;       // c_gs_ms_os_lengths
    std::vector<Index> out_gs_ms_gemm1ns_strides;       // c_gs_ms_os_strides
    std::vector<Index> lse_gs_ms_lengths;
    std::vector<Index> lse_gs_ms_strides;
  };

  std::vector<ProblemDesc> problem_descs;
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
                                 const at::Tensor q,
                                 const at::Tensor k,
                                 const at::Tensor v,
                                 at::Tensor out,
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
                    is_causal) {
                        
  }
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
                                 const at::Tensor q,
                                 const at::Tensor k,
                                 const at::Tensor v,
                                 const at::Tensor out,
                                 const at::Tensor dout,
                                 at::Tensor dq,
                                 at::Tensor dk,
                                 at::Tensor dv,
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
                    is_causal),
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
      int temp_seqlen_q = host_seqlens_q[i+1] - host_seqlens_q[i];
      int temp_q_stride = get_size_in_bytes(d * h * temp_seqlen_q, q.dtype());
      int temp_dq_stride = get_size_in_bytes(d * h * temp_seqlen_q, dq.dtype());
      int temp_seqlen_k = host_seqlens_k[i+1] - host_seqlens_k[i];
      int temp_k_stride = get_size_in_bytes(d * h * temp_seqlen_k, q.dtype());
      int temp_dk_stride = get_size_in_bytes(d * h * temp_seqlen_k, dk.dtype());

      if(!is_mnko_padding && d <= 32) {
        is_mnko_padding = ((temp_seqlen_q % 128)==0 && (temp_seqlen_k % 128)==0 ? false : true);
      }
      else if(!is_mnko_padding && d <= 64) {
        is_mnko_padding = ((temp_seqlen_q % 128)==0 && (temp_seqlen_k % 128)==0 ? false : true);
      }
      else if(!is_mnko_padding && d <= 128) {
        is_mnko_padding = ((temp_seqlen_q % 64)==0 && (temp_seqlen_k % 128)==0 ? false : true);
      }

      auto opts = q.options();

      at::Tensor d_tensor;
      d_tensor = at::empty({1, static_cast<long>(h), temp_seqlen_q}, opts.dtype(at::kFloat));
      d_ptrs.push_back(reinterpret_cast<void*>(d_tensor.data_ptr()));

      // unit test mode
      if(kIsUnitTestMode) {
        q_ptrs.clear();
        k_ptrs.clear();
        v_ptrs.clear();

        if(q.is_contiguous()) {
          q_ptrs.push_back(reinterpret_cast<void*>(q_ptr));
          dq_ptrs.push_back(reinterpret_cast<void*>(dq_ptr));
          q_ptr = q_ptr + temp_q_stride;
          dq_ptr = dq_ptr + temp_dq_stride;
        } else {
          auto q_each_tmp = q.index({torch::indexing::Slice(host_seqlens_q[i], host_seqlens_q[i+1])}).contiguous();
          auto qgrad_each_tmp = dq.index({torch::indexing::Slice(host_seqlens_q[i], host_seqlens_q[i+1])}).contiguous();
          dq_tensors.push_back(qgrad_each_tmp);
          q_ptrs.push_back(reinterpret_cast<const void*>(q_each_tmp.data_ptr()));
          dq_ptrs.push_back(reinterpret_cast<void*>(qgrad_each_tmp.data_ptr()));
        }

        if(k.is_contiguous()) {
          k_ptrs.push_back(reinterpret_cast<void*>(k_ptr));
          dk_ptrs.push_back(reinterpret_cast<void*>(dk_ptr));
          k_ptr = k_ptr + temp_k_stride;
          dk_ptr = dk_ptr + temp_dk_stride;
        } else {
          auto k_each_tmp = k.index({torch::indexing::Slice(host_seqlens_k[i], host_seqlens_k[i+1])}).contiguous();
          auto kgrad_each_tmp = dk.index({torch::indexing::Slice(host_seqlens_k[i], host_seqlens_k[i+1])}).contiguous();
          dk_tensors.push_back(kgrad_each_tmp);
          k_ptrs.push_back(reinterpret_cast<const void*>(k_each_tmp.data_ptr()));
          dk_ptrs.push_back(reinterpret_cast<void*>(kgrad_each_tmp.data_ptr()));
        }

        if(v.is_contiguous()) {
          v_ptrs.push_back(reinterpret_cast<void*>(v_ptr)); 
          dv_ptrs.push_back(reinterpret_cast<void*>(dv_ptr));
          v_ptr = v_ptr + temp_k_stride;   
          dv_ptr = dv_ptr + temp_dk_stride;
        } else {
          auto v_each_tmp = v.index({torch::indexing::Slice(host_seqlens_k[i], host_seqlens_k[i+1])}).contiguous();
          auto vgrad_each_tmp = dv.index({torch::indexing::Slice(host_seqlens_k[i], host_seqlens_k[i+1])}).contiguous();
          dv_tensors.push_back(vgrad_each_tmp);
          v_ptrs.push_back(reinterpret_cast<const void*>(v_each_tmp.data_ptr()));
          dv_ptrs.push_back(reinterpret_cast<void*>(vgrad_each_tmp.data_ptr()));
        }
      // performance mode
      } else {
        dq_ptrs.push_back(reinterpret_cast<void*>(dq_ptr));
        dq_ptr = dq_ptr + temp_dq_stride * q_stride_multiplier;

        dk_ptrs.push_back(reinterpret_cast<void*>(dk_ptr));
        dk_ptr = dk_ptr + temp_dk_stride * kv_stride_multiplier;

        dv_ptrs.push_back(reinterpret_cast<void*>(dv_ptr));
        dv_ptr = dv_ptr + temp_dk_stride * kv_stride_multiplier;
      }

      dout_ptrs.push_back(reinterpret_cast<const void*>(dout_ptr));
      dout_ptr += temp_q_stride;
    }            
  }

  std::vector<void*> dq_ptrs;
  std::vector<void*> dk_ptrs;
  std::vector<void*> dv_ptrs;

  std::vector<const void*> bwd_out_ptrs;
  std::vector<const void*> bwd_softmax_lse_ptrs;

  std::vector<const void*> dout_ptrs;
  std::vector<void*> d_ptrs;

  std::vector<at::Tensor> dq_tensors;
  std::vector<at::Tensor> dk_tensors;
  std::vector<at::Tensor> dv_tensors;
};