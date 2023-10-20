// BSD 3 Clause
// Copyright 2023 Advanced Micro Devices, Inc.
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

#include "flash_runner.hpp"

std::vector<torch::Tensor>
mha_fwd(const torch::Tensor &q,                         // batch_size x seqlen_q x num_heads_q x head_size
        const torch::Tensor &k,                         // batch_size x seqlen_kv x num_heads_kv x head_size
        const torch::Tensor &v,                         // batch_size x seqlen_kv x num_heads_kv x head_size
        c10::optional<torch::Tensor> &out_,             // batch_size x seqlen_q x num_heads_q x head_size
        const float p_dropout,
        const float softmax_scale,
        const bool is_causal,
        const bool return_softmax,
        c10::optional<at::Generator> gen_) {
  auto dprops = at::cuda::getCurrentDeviceProperties();
  bool is_gfx90x = dprops->major == 9 && dprops->minor == 0;
  bool is_gfx94x = dprops->major == 9 && dprops->minor == 4;
  TORCH_CHECK(is_gfx90x || is_gfx94x, "FlashAttention only supports AMD MI200 GPUs or newer.");

  auto q_dtype = q.dtype();
  TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
              "FlashAttention only support fp16 and bf16 data type");
  if (q_dtype == torch::kBFloat16) {
    TORCH_CHECK(is_gfx90x || is_gfx94x, "bfloat16 is only supported on AMD MI200 GPUs or newer");
  }
  TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
  TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");

  TORCH_CHECK(q.is_cuda(), "Input tensor must be on ROCm device");
  TORCH_CHECK(k.is_cuda(), "Input tensor must be on ROCm device");
  TORCH_CHECK(v.is_cuda(), "Input tensor must be on ROCm device");

  TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");

  const auto sizes = q.sizes();

  const int batch_size = sizes[0];
  const int seqlen_q = sizes[1];
  const int num_heads_q = sizes[2];
  const int head_size_og = sizes[3];
  const int seqlen_kv = k.size(1);
  const int num_heads_kv = k.size(2);
  TORCH_CHECK(batch_size > 0, "batch size must be postive");
  TORCH_CHECK(head_size_og <= 128, "FlashAttention forward only supports head dimension at most 128");
  TORCH_CHECK(num_heads_q % num_heads_kv == 0, "Number of heads in key/value must divide number of heads in query");

  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  const int head_size = round_multiple(head_size_og, 8);
  TORCH_CHECK(head_size == round_multiple(head_size_og, 8), "head_size must be head_size_og rounded to a multiple of 8");

  CHECK_SHAPE(q, batch_size, seqlen_q, num_heads_q, head_size_og);
  CHECK_SHAPE(k, batch_size, seqlen_kv, num_heads_kv, head_size_og);
  CHECK_SHAPE(v, batch_size, seqlen_kv, num_heads_kv, head_size_og);

  torch::Tensor q_padded, k_padded, v_padded;
  if (head_size_og % 8 != 0) {
    q_padded = torch::nn::functional::pad(q, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
    k_padded = torch::nn::functional::pad(k, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
    v_padded = torch::nn::functional::pad(v, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
  } else {
    q_padded = q;
    k_padded = k;
    v_padded = v;
  }

  torch::Tensor out;
  if (out_.has_value()) {
    out = out_.value();
    TORCH_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
    TORCH_CHECK(out.is_cuda(), "Output tensor must be on ROCm device");
    TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
    CHECK_SHAPE(out, batch_size, seqlen_q, num_heads_q, head_size_og);
    if (head_size_og % 8 != 0) { out = torch::empty_like(q_padded); }
  } else {
    out = torch::empty_like(q_padded);
  }

  // Otherwise the kernel will be launched from cuda:0 device
  // Cast to char to avoid compiler warning about narrowing
  at::cuda::HIPGuard device_guard{(char)q.get_device()};

  auto opts = q.options();

  auto softmax_lse = torch::empty({batch_size, num_heads_q, seqlen_q}, opts.dtype(torch::kFloat32));
  torch::Tensor z;
  // Only return softmax if there's dropout to reduce compilation time
  if (return_softmax) {
    // TORCH_CHECK(p_dropout > 0.0f, "return_softmax is only supported when p_dropout > 0.0");
    z = torch::empty({ batch_size, num_heads_q, seqlen_q, seqlen_kv }, opts.dtype(torch::kInt32));
  }

  FlashFwdBatchedParams params(batch_size,
                               seqlen_q,
                               seqlen_kv,
                               num_heads_q,
                               num_heads_kv,
                               head_size,
                               q_padded,
                               k_padded,
                               v_padded,
                               out,
                               return_softmax ? z.data_ptr() : nullptr,
                               softmax_lse.data_ptr(),
                               p_dropout,
                               softmax_scale,
                               is_causal);

  // number of times random will be generated per thread, to offset philox counter in thc random
  // state
  // We use a custom RNG that increases the offset by batch_size * nheads * 32.
  int64_t counter_offset = params.b * params.h_q * 32;
  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  auto rng_state = torch::empty({2}, options.dtype(torch::kInt64));
  // Forward kernel will populate memory with the seed and offset.
  params.rng_state = reinterpret_cast<uint64_t*>(rng_state.data_ptr());

  if (params.is_dropout) {
    auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
        gen_, at::cuda::detail::getDefaultCUDAGenerator());
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    params.philox_args = gen->philox_cuda_state(counter_offset);
    auto seeds = unpack(params.philox_args);
    params.seeds = seeds;
  }

  auto stream = at::cuda::getCurrentHIPStream().stream();
  auto flash_runner = std::make_unique<FlashRunner>();
  flash_runner->Run<FlashFwdBatchedParams>(params, stream);

  torch::Tensor out_padded = out;
  if (head_size_og % 8 != 0) {
    out = out.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
    if (out_.has_value()) { out_.value().copy_(out); }
  }

  return { out, q_padded, k_padded, v_padded, out_padded, softmax_lse, z, rng_state };
}

std::vector<torch::Tensor>
mha_varlen_fwd(const torch::Tensor &q,  // total_q x num_heads_q x head_size, total_q := \sum_{i=0}^{b} s_i
               const torch::Tensor &k,  // total_k x num_heads_kv x head_size, total_k := \sum_{i=0}^{b} s_i
               const torch::Tensor &v,  // total_k x num_heads_kv x head_size, total_k := \sum_{i=0}^{b} s_i
               c10::optional<torch::Tensor> &out_, // total_q x num_heads_q x head_size, total_k := \sum_{i=0}^{b} s_i
               const torch::Tensor &cu_seqlens_q,  // b+1
               const torch::Tensor &cu_seqlens_k,  // b+1
               const int max_seqlen_q,
               const int max_seqlen_k,
               const float p_dropout,
               const float softmax_scale,
               const bool zero_tensors,
               const bool is_causal,
               const bool return_softmax, // in rocm ,this will return the random number matrix when doing dropout
               c10::optional<at::Generator> gen_) {
  auto dprops = at::cuda::getCurrentDeviceProperties();
  bool is_gfx90x = dprops->major == 9 && dprops->minor == 0;
  bool is_gfx94x = dprops->major == 9 && dprops->minor == 4;
  TORCH_CHECK(is_gfx90x || is_gfx94x, "FlashAttention only supports AMD MI200 GPUs or newer.");

  auto q_dtype = q.dtype();
  TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
              "FlashAttention only support fp16 and bf16 data type");
  if (q_dtype == torch::kBFloat16) {
    TORCH_CHECK(is_gfx90x || is_gfx94x, "bfloat16 is only supported on AMD MI200 GPUs or newer");
  }
  TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
  TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
  TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must have dtype int32");
  TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must have dtype int32");

  TORCH_CHECK(q.is_cuda(), "Input tensor must be on ROCm device");
  TORCH_CHECK(k.is_cuda(), "Input tensor must be on ROCm device");
  TORCH_CHECK(v.is_cuda(), "Input tensor must be on ROCm device");
  TORCH_CHECK(cu_seqlens_q.is_cuda(), "cu_seqlens_q must be on ROCm device");
  TORCH_CHECK(cu_seqlens_k.is_cuda(), "cu_seqlens_k must be on ROCm device");

  TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(cu_seqlens_q.is_contiguous(), "cu_seqlens_q must be contiguous");
  TORCH_CHECK(cu_seqlens_k.is_contiguous(), "cu_seqlens_k must be contiguous");

  const auto sizes = q.sizes();

  const int total_q = sizes[0];
  const int batch_size = cu_seqlens_q.numel() - 1;
  const int num_heads_q = sizes[1];
  const int head_size_og = sizes[2];
  const int total_k = k.size(0);
  const int num_heads_kv = k.size(1);
  TORCH_CHECK(batch_size > 0, "batch size must be positive");
  TORCH_CHECK(head_size_og <= 128, "FlashAttention forward only supports head dimension at most 128");
  TORCH_CHECK(num_heads_q % num_heads_kv == 0, "Number of heads in key/value must divide number of heads in query");

  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  const int head_size = round_multiple(head_size_og, 8);
  TORCH_CHECK(head_size == round_multiple(head_size_og, 8), "head_size must be head_size_og rounded to a multiple of 8");

  CHECK_SHAPE(q, total_q, num_heads_q, head_size_og);
  CHECK_SHAPE(k, total_k, num_heads_kv, head_size_og);
  CHECK_SHAPE(v, total_k, num_heads_kv, head_size_og);
  CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
  CHECK_SHAPE(cu_seqlens_k, batch_size + 1);

  torch::Tensor q_padded, k_padded, v_padded;
  if (head_size_og % 8 != 0) {
    q_padded = torch::nn::functional::pad(q, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
    k_padded = torch::nn::functional::pad(k, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
    v_padded = torch::nn::functional::pad(v, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
  } else {
    q_padded = q;
    k_padded = k;
    v_padded = v;
  }

  torch::Tensor out;
  if (out_.has_value()) {
    out = out_.value();
    TORCH_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
    TORCH_CHECK(out.is_cuda(), "Output tensor must be on ROCm device");
    TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
    CHECK_SHAPE(out, total_q, num_heads_q, head_size_og);
    if (head_size_og % 8 != 0) { out = torch::empty_like(q_padded); }
  } else {
    out = torch::empty_like(q_padded);
  }

  // Otherwise the kernel will be launched from cuda:0 device
  // Cast to char to avoid compiler warning about narrowing
  at::cuda::HIPGuard device_guard{(char)q.get_device()};

  auto opts = q.options();

  auto softmax_lse = torch::empty({batch_size, num_heads_q, max_seqlen_q}, opts.dtype(torch::kFloat32));
  torch::Tensor z;
  if (return_softmax) {
    // TORCH_CHECK(p_dropout > 0.0f, "return_softmax is only supported when p_dropout > 0.0");
    z = torch::empty({ batch_size, num_heads_q, max_seqlen_q, max_seqlen_k }, opts.dtype(torch::kInt32));
  }
  
  if (zero_tensors) {
    out.zero_();
    softmax_lse.fill_(-std::numeric_limits<float>::infinity());
    if (return_softmax) {z.zero_();}
  }

  FlashFwdGroupedParams params(batch_size,
                               max_seqlen_q, 
                               max_seqlen_k,
                               num_heads_q, 
                               num_heads_kv,
                               head_size,
                               q_padded, 
                               k_padded, 
                               v_padded, 
                               out,
                               cu_seqlens_q.data_ptr(),
                               cu_seqlens_k.data_ptr(),
                               return_softmax ? z.data_ptr() : nullptr,
                               softmax_lse.data_ptr(),
                               p_dropout,
                               softmax_scale,
                               is_causal);

  // number of times random will be generated per thread, to offset philox counter in thc random
  // state
  // We use a custom RNG that increases the offset by batch_size * nheads * 32.
  int64_t counter_offset = params.b * params.h_q * 32;
  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  auto rng_state = torch::empty({2}, options.dtype(torch::kInt64));
  // Forward kernel will populate memory with the seed and offset.
  params.rng_state = reinterpret_cast<uint64_t*>(rng_state.data_ptr());

  if (params.is_dropout)  {
    auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
        gen_, at::cuda::detail::getDefaultCUDAGenerator());
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    params.philox_args = gen->philox_cuda_state(counter_offset);
    auto seeds = unpack(params.philox_args);
    params.seeds = seeds;
  }

  auto stream = at::cuda::getCurrentHIPStream().stream();
  auto flash_runner = std::make_unique<FlashRunner>();
  flash_runner->Run<FlashFwdGroupedParams>(params, stream);

  torch::Tensor out_padded = out;
  if (head_size_og % 8 != 0) {
    out = out.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
    if (out_.has_value()) { out_.value().copy_(out); }
  }

  return { out, q_padded, k_padded, v_padded, out_padded, softmax_lse, z, rng_state };
}

std::vector<torch::Tensor>
mha_bwd(const torch::Tensor &dout,  // batch_size x seqlen_q x num_heads_q, x head_size_og
        const torch::Tensor &q,   // batch_size x seqlen_q x num_heads_q x head_size
        const torch::Tensor &k,   // batch_size x seqlen_kv x num_heads_kv x head_size
        const torch::Tensor &v,   // batch_size x seqlen_kv x num_heads_kv x head_size
        const torch::Tensor &out,   // batch_size x seqlen_q x num_heads_q x head_size
        const torch::Tensor &softmax_lse,     // b x h x seqlen_q
        c10::optional<torch::Tensor> &dq_,   // batch_size x seqlen_q x num_heads_q x head_size
        c10::optional<torch::Tensor> &dk_,   // batch_size x seqlen_kv x num_heads_kv x head_size
        c10::optional<torch::Tensor> &dv_,   // batch_size x seqlen_kv x num_heads_kv x head_size
        const float p_dropout,         // probability to drop
        const float softmax_scale,
        const bool is_causal,
        c10::optional<at::Generator> gen_,
        c10::optional<torch::Tensor> &rng_state) {
  auto dprops = at::cuda::getCurrentDeviceProperties();
  bool is_gfx90x = dprops->major == 9 && dprops->minor == 0;
  bool is_gfx94x = dprops->major == 9 && dprops->minor == 4;
  TORCH_CHECK(is_gfx90x || is_gfx94x, "FlashAttention only supports AMD MI200 GPUs or newer.");

  auto q_dtype = q.dtype();
  TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
              "FlashAttention only support fp16 and bf16 data type");
  if (q_dtype == torch::kBFloat16) {
    TORCH_CHECK(is_gfx90x || is_gfx94x, "bfloat16 is only supported on AMD MI200 GPUs or newer");
  }
  TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
  TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
  TORCH_CHECK(out.dtype() == q_dtype, "query and out must have the same dtype");
  TORCH_CHECK(dout.dtype() == q_dtype, "query and dout must have the same dtype");

  TORCH_CHECK(q.is_cuda(), "Input tensor must be on ROCm device");
  TORCH_CHECK(k.is_cuda(), "Input tensor must be on ROCm device");
  TORCH_CHECK(v.is_cuda(), "Input tensor must be on ROCm device");
  TORCH_CHECK(out.is_cuda(), "out tensor must be on ROCm device");
  TORCH_CHECK(dout.is_cuda(), "dout tensor must be on ROCm device");
  TORCH_CHECK(softmax_lse.is_cuda(), "softmax_lse tensor must be on ROCm device");

  TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(out.stride(-1) == 1, "out tensor must have contiguous last dimension");
  TORCH_CHECK(dout.stride(-1) == 1, "dout tensor must have contiguous last dimension");

  const auto sizes = q.sizes();

  const int batch_size = sizes[0];
  const int seqlen_q = sizes[1];
  const int num_heads_q = sizes[2];
  const int head_size_og = dout.size(3);
  const int head_size = sizes[3];
  const int seqlen_kv = k.size(1);
  const int num_heads_kv = k.size(2);
  TORCH_CHECK(batch_size > 0, "batch size must be positive");
  TORCH_CHECK(head_size % 8 == 0, "head_size should be a multiple of 8");
  TORCH_CHECK(head_size <= 128, "FlashAttention backward only supports head dimension at most 128");
  TORCH_CHECK(num_heads_q % num_heads_kv == 0, "Number of heads in key/value must divide number of heads in query");

  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  TORCH_CHECK(head_size == round_multiple(head_size_og, 8), "head_size must be head_size_og rounded to a multiple of 8");

  CHECK_SHAPE(q, batch_size, seqlen_q, num_heads_q, head_size);
  CHECK_SHAPE(k, batch_size, seqlen_kv, num_heads_kv, head_size);
  CHECK_SHAPE(v, batch_size, seqlen_kv, num_heads_kv, head_size);
  CHECK_SHAPE(out, batch_size, seqlen_q, num_heads_q, head_size);
  CHECK_SHAPE(dout, batch_size, seqlen_q, num_heads_q, head_size_og);

  torch::Tensor dq, dk, dv;
  if (dq_.has_value()) {
    dq = dq_.value();
    TORCH_CHECK(dq.dtype() == q_dtype, "dq must have the same dtype as q");
    TORCH_CHECK(dq.is_cuda(), "dq must be on ROCm device");
    TORCH_CHECK(dq.stride(-1) == 1, "dq must have contiguous last dimension");
    CHECK_SHAPE(dq, batch_size, seqlen_q, num_heads_q, head_size);
  } else {
    dq = torch::empty_like(q);
  }
  if (dk_.has_value()) {
    dk = dk_.value();
    TORCH_CHECK(dk.dtype() == q_dtype, "dk must have the same dtype as q");
    TORCH_CHECK(dk.is_cuda(), "dk must be on ROCm device");
    TORCH_CHECK(dk.stride(-1) == 1, "dk must have contiguous last dimension");
    CHECK_SHAPE(dk, batch_size, seqlen_kv, num_heads_kv, head_size);
  } else {
    dk = torch::empty_like(k);
  }
  if (dv_.has_value()) {
    dv = dv_.value();
    TORCH_CHECK(dv.dtype() == q_dtype, "dv must have the same dtype as q");
    TORCH_CHECK(dv.is_cuda(), "dv must be on ROCm device");
    TORCH_CHECK(dv.stride(-1) == 1, "dv must have contiguous last dimension");
    CHECK_SHAPE(dv, batch_size, seqlen_kv, num_heads_kv, head_size);
  } else {
    dv = torch::empty_like(k);
  }

  torch::Tensor dout_padded;
  if (head_size_og % 8 != 0) {
    dout_padded = torch::nn::functional::pad(dout, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
  } else {
    dout_padded = dout;
  }

  // Otherwise the kernel will be launched from cuda:0 device
  // Cast to char to avoid compiler warning about narrowing
  at::cuda::HIPGuard device_guard{(char)q.get_device()};

  auto opts = q.options();
  auto softmax_d = torch::empty({batch_size, num_heads_q, seqlen_q}, opts.dtype(torch::kFloat32));

  dq.zero_();

  FlashBwdBatchedParams params(batch_size,
                               seqlen_q, 
                               seqlen_kv,
                               num_heads_q, 
                               num_heads_kv,
                               head_size,
                               q, 
                               k, 
                               v, 
                               out,
                               dout_padded, 
                               dq, 
                               dk, 
                               dv,
                               nullptr,
                               softmax_lse.data_ptr(),
                               p_dropout,
                               softmax_scale,
                               is_causal);

  auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
      gen_, at::cuda::detail::getDefaultCUDAGenerator());

  // We use a custom RNG that increases the offset by batch_size * nheads * 32.
  int64_t counter_offset = params.b * params.h_q * 32;
      
  if (rng_state.has_value()) {
    params.rng_state = reinterpret_cast<uint64_t*>(rng_state.value().data_ptr());
    params.seeds = std::tuple<uint64_t, uint64_t>(params.rng_state[0], params.rng_state[1]);
  } else if(params.is_dropout) {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    params.philox_args = gen->philox_cuda_state(counter_offset);
    auto seeds = unpack(params.philox_args);
    params.rng_state[0] = std::get<0>(seeds);
    params.rng_state[1] = std::get<1>(seeds);
  }

  auto stream = at::cuda::getCurrentHIPStream().stream();
  auto flash_runner = std::make_unique<FlashRunner>();
  flash_runner->Run<FlashBwdBatchedParams>(params, stream);

  return { dq, dk, dv, softmax_d };
}

std::vector<torch::Tensor>
mha_varlen_bwd(const torch::Tensor &dout,  // total_q x num_heads_q, x head_size
               const torch::Tensor &q,   // total_q x num_heads_q x head_size, total_q := \sum_{i=0}^{b} s_i
               const torch::Tensor &k,   // total_k x num_heads_kv x head_size, total_k := \sum_{i=0}^{b} s_i
               const torch::Tensor &v,   // total_k x num_heads_kv x head_size, total_k := \sum_{i=0}^{b} s_i
               const torch::Tensor &out,   // total_q x num_heads_q x head_size
               const torch::Tensor &softmax_lse,     // b x h x s   softmax logsumexp
               c10::optional<torch::Tensor> &dq_,   // total_q x num_heads_q x head_size, total_q := \sum_{i=0}^{b} s_i
               c10::optional<torch::Tensor> &dk_,   // total_k x num_heads_kv x head_size, total_k := \sum_{i=0}^{b} s_i
               c10::optional<torch::Tensor> &dv_,   // total_k x num_heads_kv x head_size, total_k := \sum_{i=0}^{b} s_i
               const torch::Tensor &cu_seqlens_q,  // b+1
               const torch::Tensor &cu_seqlens_k,  // b+1
               const int max_seqlen_q,
               const int max_seqlen_k,          // max sequence length to choose the kernel
               const float p_dropout,         // probability to drop
               const float softmax_scale,
               const bool zero_tensors,
               const bool is_causal,
               c10::optional<at::Generator> gen_,
               c10::optional<torch::Tensor> &rng_state) {
  auto dprops = at::cuda::getCurrentDeviceProperties();
  bool is_gfx90x = dprops->major == 9 && dprops->minor == 0;
  bool is_gfx94x = dprops->major == 9 && dprops->minor == 4;
  TORCH_CHECK(is_gfx90x || is_gfx94x, "FlashAttention only supports AMD MI200 GPUs or newer.");

  auto q_dtype = q.dtype();
  TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
              "FlashAttention only support fp16 and bf16 data type");
  if (q_dtype == torch::kBFloat16) {
    TORCH_CHECK(is_gfx90x || is_gfx94x, "bfloat16 is only supported on AMD MI200 GPUs or newer");
  }
  TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
  TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
  TORCH_CHECK(out.dtype() == q_dtype, "query and out must have the same dtype");
  TORCH_CHECK(dout.dtype() == q_dtype, "query and dout must have the same dtype");
  TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must have dtype int32");
  TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must have dtype int32");

  TORCH_CHECK(q.is_cuda(), "Input tensor must be on ROCm device");
  TORCH_CHECK(k.is_cuda(), "Input tensor must be on ROCm device");
  TORCH_CHECK(v.is_cuda(), "Input tensor must be on ROCm device");
  TORCH_CHECK(out.is_cuda(), "out tensor must be on ROCm device");
  TORCH_CHECK(dout.is_cuda(), "dout tensor must be on ROCm device");
  TORCH_CHECK(softmax_lse.is_cuda(), "softmax_lse tensor must be on ROCm device");
  TORCH_CHECK(cu_seqlens_q.is_cuda(), "cu_seqlens_q must be on ROCm device");
  TORCH_CHECK(cu_seqlens_k.is_cuda(), "cu_seqlens_k must be on ROCm device");

  TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(out.stride(-1) == 1, "out tensor must have contiguous last dimension");
  TORCH_CHECK(dout.stride(-1) == 1, "dout tensor must have contiguous last dimension");
  TORCH_CHECK(cu_seqlens_q.is_contiguous(), "cu_seqlens_q must be contiguous");
  TORCH_CHECK(cu_seqlens_k.is_contiguous(), "cu_seqlens_k must be contiguous");

  const auto sizes = q.sizes();

  const int total_q = sizes[0];
  const int batch_size = cu_seqlens_q.numel() - 1;
  const int num_heads_q = sizes[1];
  const int head_size_og = dout.size(2);
  const int head_size = sizes[2];
  const int total_k = k.size(0);
  const int num_heads_kv = k.size(1);
  TORCH_CHECK(batch_size > 0, "batch size must be positive");
  TORCH_CHECK(head_size % 8 == 0, "head_size should be a multiple of 8");
  TORCH_CHECK(head_size <= 128, "FlashAttention backward only supports head dimension at most 128");
  TORCH_CHECK(num_heads_q % num_heads_kv == 0, "Number of heads in key/value must divide number of heads in query");

  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  TORCH_CHECK(head_size == round_multiple(head_size_og, 8), "head_size must be head_size_og rounded to a multiple of 8");

  CHECK_SHAPE(q, total_q, num_heads_q, head_size);
  CHECK_SHAPE(k, total_k, num_heads_kv, head_size);
  CHECK_SHAPE(v, total_k, num_heads_kv, head_size);
  CHECK_SHAPE(out, total_q, num_heads_q, head_size);
  CHECK_SHAPE(dout, total_q, num_heads_q, head_size_og);
  CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
  CHECK_SHAPE(cu_seqlens_k, batch_size + 1);

  torch::Tensor dq, dk, dv;
  if (dq_.has_value()) {
    dq = dq_.value();
    TORCH_CHECK(dq.dtype() == q_dtype, "dq must have the same dtype as q");
    TORCH_CHECK(dq.is_cuda(), "dq must be on ROCm device");
    TORCH_CHECK(dq.stride(-1) == 1, "dq must have contiguous last dimension");
    CHECK_SHAPE(dq, total_q, num_heads_q, head_size);
  } else {
    dq = torch::empty_like(q);
  }

  if (dk_.has_value()) {
    dk = dk_.value();
    TORCH_CHECK(dk.dtype() == q_dtype, "dk must have the same dtype as q");
    TORCH_CHECK(dk.is_cuda(), "dk must be on ROCm device");
    TORCH_CHECK(dk.stride(-1) == 1, "dk must have contiguous last dimension");
    CHECK_SHAPE(dk, total_k, num_heads_kv, head_size);
  } else {
    dk = torch::empty_like(k);
  }

  if (dv_.has_value()) {
    dv = dv_.value();
    TORCH_CHECK(dv.dtype() == q_dtype, "dv must have the same dtype as q");
    TORCH_CHECK(dv.is_cuda(), "dv must be on ROCm device");
    TORCH_CHECK(dv.stride(-1) == 1, "dv must have contiguous last dimension");
    CHECK_SHAPE(dv, total_k, num_heads_kv, head_size);
  } else {
    dv = torch::empty_like(k);
  }

  torch::Tensor dout_padded;
  if (head_size_og % 8 != 0) {
    dout_padded = torch::nn::functional::pad(dout, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
  } else {
    dout_padded = dout;
  }

  // Otherwise the kernel will be launched from cuda:0 device
  // Cast to char to avoid compiler warning about narrowing
  at::cuda::HIPGuard device_guard{(char)q.get_device()};

  auto opts = q.options();
  auto softmax_d = torch::empty({batch_size, num_heads_q, max_seqlen_q}, opts.dtype(torch::kFloat32));
  
  // CK need zeroed tensors
  dq.zero_();
  dk.zero_();
  dv.zero_();

  FlashBwdGroupedParams params(batch_size,
                               max_seqlen_q, 
                               max_seqlen_k,
                               num_heads_q, 
                               num_heads_kv,
                               head_size, 
                               q, 
                               k, 
                               v, 
                               out,
                               dout_padded, 
                               dq, 
                               dk, 
                               dv,
                               cu_seqlens_q.data_ptr(),
                               cu_seqlens_k.data_ptr(),
                               nullptr,
                               softmax_lse.data_ptr(),
                               p_dropout,
                               softmax_scale,
                               is_causal);

  auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
      gen_, at::cuda::detail::getDefaultCUDAGenerator());

  // We use a custom RNG that increases the offset by batch_size * nheads * 32.
  int64_t counter_offset = params.b * params.h_q * 32;

  if (rng_state.has_value()) {
    params.rng_state = reinterpret_cast<uint64_t*>(rng_state.value().data_ptr());
    params.seeds = std::tuple<uint64_t, uint64_t>(params.rng_state[0], params.rng_state[1]);
  } else if(params.is_dropout) {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    params.philox_args = gen->philox_cuda_state(counter_offset);
    auto seeds = unpack(params.philox_args);
    params.rng_state[0] = std::get<0>(seeds);
    params.rng_state[1] = std::get<1>(seeds);
  }

  auto stream = at::cuda::getCurrentHIPStream().stream();
  auto flash_runner = std::make_unique<FlashRunner>();
  flash_runner->Run<FlashBwdGroupedParams>(params, stream);

  return { dq, dk, dv, softmax_d };
}


#ifdef BUILD_PYTHON_PACKAGE
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashAttention";
    m.def("fwd", &mha_fwd, "Forward pass");
    m.def("varlen_fwd", &mha_varlen_fwd, "Forward pass (variable length)");
    m.def("bwd", &mha_bwd, "Backward pass");
    m.def("varlen_bwd", &mha_varlen_bwd, "Backward pass (variable length)");
}
#endif

#ifndef BUILD_PYTHON_PACKAGE
//main function to test with the API
bool fwd_test(bool do_verification){
    int batch_size = 64;
    int nheads = 16;
    int seqlen = 256;
    int n = 1024;
    int d = n / nheads; //head_size//64

    //initialize the tensors
    torch::Tensor q_host = at::rand({batch_size, seqlen, nheads, d}, torch::kFloat16);//torch::kBFloat16;at::kHalf
    torch::Tensor k_host = at::rand({batch_size, seqlen, nheads, d}, torch::kFloat16);
    torch::Tensor v_host = at::rand({batch_size, seqlen, nheads, d}, torch::kFloat16);

    torch::Tensor q = q_host.to(at::kCUDA);
    torch::Tensor k = k_host.to(at::kCUDA);
    torch::Tensor v = v_host.to(at::kCUDA);

    //initialize the output tensor
    torch::Tensor out_host = at::empty({batch_size, seqlen, nheads, d}, torch::kFloat16);
    c10::optional<torch::Tensor> out = out_host.to(at::kCUDA);

    //initialize seqlens vector (size is b+1)
    std::vector<int> cu_seqlens_q_vec;
    std::vector<int> cu_seqlens_k_vec;

    for (int i = 0 ; i < batch_size + 1; i++){
      cu_seqlens_q_vec.push_back(i * seqlen);
      cu_seqlens_k_vec.push_back(i * seqlen);
    }

    torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kInt32);
    torch::Tensor cu_seqlens_q = at::from_blob(cu_seqlens_q_vec.data(),{batch_size + 1},opts).clone().to(at::kCUDA);
    torch::Tensor cu_seqlens_k = at::from_blob(cu_seqlens_k_vec.data(),{batch_size + 1},opts).clone().to(at::kCUDA);

    int max_seqlen_q_ = seqlen;
    int max_seqlen_k_ = seqlen;

    //dropout parameters
    float p_drop                    = 0.17;
    float p_dropout                 = 1 - p_drop;
    uint16_t p_dropout_in_16bits    = uint16_t(std::floor(p_dropout * 65535.0));
    float rp_dropout                = 1.0 / p_dropout;
    const unsigned long long seed   = 1;
    const unsigned long long offset = 0;
    
    //other parameters
    float softmax_scale = 0.125;
    bool zero_tensors = true;
    bool is_causal = false;
    bool return_softmax = true;

    c10::optional<at::Generator> gen_ = c10::nullopt;

    auto results =
    mha_fwd(q,
            k,
            v,
            out,
            p_drop,
            softmax_scale,
            is_causal,
            return_softmax,
            gen_);

    using FP16 = ck::half_t;
    using BF16 = ck::bhalf_t;
    using F32 = float;
    using U16 = unsigned short;

    using PassThrough = ck::tensor_operation::element_wise::PassThrough;

    using ADataType        = BF16;
    using B0DataType       = BF16;
    using B1DataType       = BF16;
    using AccDataType      = F32;
    using CShuffleDataType = F32;
    using CDataType        = BF16;
    using ZDataType        = U16;
    using LSEDataType      = F32;
    using Acc0BiasDataType = ck::Tuple<>;
    using Acc1BiasDataType = ck::Tuple<>;

    static constexpr ck::index_t NumDimG = 2;
    static constexpr ck::index_t NumDimM = 1;
    static constexpr ck::index_t NumDimN = 1;
    static constexpr ck::index_t NumDimK = 1;
    static constexpr ck::index_t NumDimO = 1;

    using AElementOp    = PassThrough;
    using B0ElementOp   = PassThrough;
    using Acc0ElementOp = ck::tensor_operation::element_wise::Scale;
    using B1ElementOp   = PassThrough;
    using CElementOp    = PassThrough;

    // Ref Gemm0: fp16 in, fp32 out
    using ReferenceGemm0Instance = ck::tensor_operation::host::ReferenceBatchedGemm<ADataType,
                                                                                    B0DataType,
                                                                                    AccDataType,
                                                                                    AccDataType,
                                                                                    AElementOp,
                                                                                    B0ElementOp,
                                                                                    Acc0ElementOp>;

    // Ref Softmax: fp32 in, fp16 out
    using ReferenceSoftmaxInstance =
        ck::tensor_operation::host::ReferenceSoftmax<AccDataType, ADataType, AccDataType>;

    // Ref Gemm1: fp16 in, fp16 out
    using ReferenceGemm1Instance = ck::tensor_operation::host::ReferenceBatchedGemm<ADataType,
                                                                                    B1DataType,
                                                                                    CDataType,
                                                                                    AccDataType,
                                                                                    AElementOp,
                                                                                    B1ElementOp,
                                                                                    CElementOp>;
    
    // Ref dropout
    using ReferenceDropoutInstance =
        ck::tensor_operation::host::ReferenceDropout<ZDataType, ADataType, ADataType>;

    bool pass = true;
    if(do_verification)
    {
        q_host = q_host.view({ batch_size, seqlen, nheads, d }); //64 256 16 64
        k_host = k_host.view({ batch_size, seqlen, nheads, d });
        v_host = v_host.view({ batch_size, seqlen, nheads, d });

        const int M   = seqlen;   // seqlen Q
        const int N   = seqlen;   // seqlen K
        const int K   = d;        // head_dim
        const int O   = d;        // head_dim
        const int G0  = 1;        // G0 = batch_size
        const int G1  = nheads;   // num_heads_q

        std::vector<Tensor<ADataType>>   a_tensors;
        std::vector<Tensor<B0DataType>>  b0_tensors;
        std::vector<Tensor<B1DataType>>  b1_tensors;
        std::vector<Tensor<CDataType>>   c_tensors;
        std::vector<Tensor<ZDataType>>   z_tensors;
        std::vector<Tensor<LSEDataType>> lse_tensors;

        auto a_element_op    = AElementOp{};
        auto b0_element_op   = B0ElementOp{};
        auto acc0_element_op = Acc0ElementOp{softmax_scale};
        auto b1_element_op   = B1ElementOp{};
        auto c_element_op    = CElementOp{};

        for(std::size_t i = 0; i < batch_size; i++)
        {

            std::vector<ck::index_t> a_gs_ms_ks_lengths{G0, G1, M, K};
            std::vector<ck::index_t> a_gs_ms_ks_strides ={M * G1 * K, K, G1 * K, 1};

            std::vector<ck::index_t> b0_gs_ns_ks_lengths{G0, G1, N, K};
            std::vector<ck::index_t> b0_gs_ns_ks_strides ={N * G1 * K, K, G1 * K, 1};

            std::vector<ck::index_t> b1_gs_os_ns_lengths{G0, G1, O, N};
            std::vector<ck::index_t> b1_gs_os_ns_strides ={N * G1 * O, O, 1, G1 * O};

            std::vector<ck::index_t> c_gs_ms_os_lengths{G0, G1, M, O};
            std::vector<ck::index_t> c_gs_ms_os_strides ={M * G1 * O, O, G1 * O, 1};

            std::vector<ck::index_t> z_gs_ms_ns_lengths{G0, G1, M, N};
            std::vector<ck::index_t> z_gs_ms_ns_strides ={M * G1 * N, N, G1 * N, 1}; // Z layout [G0, M, G1, N]

            std::vector<ck::index_t> lse_gs_ms_lengths{G0, G1, M};
            std::vector<ck::index_t> lse_gs_ms_strides =
                std::vector<ck::index_t>{G1 * M, M, 1}; // LSE layout [G0, G1, M]

            // C_m_o = A_m_k * B0_k_n * B1_n_o
            Tensor<ADataType> a_gs_ms_ks(a_gs_ms_ks_lengths, a_gs_ms_ks_strides);
            Tensor<B0DataType> b0_gs_ns_ks(b0_gs_ns_ks_lengths, b0_gs_ns_ks_strides);
            Tensor<B1DataType> b1_gs_os_ns(b1_gs_os_ns_lengths, b1_gs_os_ns_strides);
            Tensor<CDataType> c_gs_ms_os_device_result(c_gs_ms_os_lengths, c_gs_ms_os_strides);
            Tensor<ZDataType> z_gs_ms_ns(z_gs_ms_ns_lengths, z_gs_ms_ns_strides);
            Tensor<LSEDataType> lse_gs_ms_device_result(lse_gs_ms_lengths, lse_gs_ms_strides);

            void* q_h_ptr_f = q_host[i].data_ptr();
            void* k_h_ptr_f = k_host[i].data_ptr();
            void* v_h_ptr_f = v_host[i].data_ptr();

            ADataType* q_h_ptr = reinterpret_cast<ADataType*>(q_h_ptr_f);
            B0DataType* k_h_ptr = reinterpret_cast<B0DataType*>(k_h_ptr_f);
            B1DataType* v_h_ptr = reinterpret_cast<B1DataType*>(v_h_ptr_f);

            std::vector<ADataType> a_vector(q_h_ptr, q_h_ptr + q_host[i].numel()); //transfer tensor into vector
            a_gs_ms_ks.mData.assign(a_vector.begin(), a_vector.end());

            std::vector<B0DataType> b0_vector(k_h_ptr, k_h_ptr + k_host[i].numel()); //transfer tensor into vector
            b0_gs_ns_ks.mData.assign(b0_vector.begin(), b0_vector.end());

            std::vector<B1DataType> b1_vector(v_h_ptr, v_h_ptr + v_host[i].numel()); //transfer tensor into vector
            b1_gs_os_ns.mData.assign(b1_vector.begin(), b1_vector.end());

            a_tensors.push_back(a_gs_ms_ks);
            b0_tensors.push_back(b0_gs_ns_ks);
            b1_tensors.push_back(b1_gs_os_ns);
            c_tensors.push_back(c_gs_ms_os_device_result);
            z_tensors.push_back(z_gs_ms_ns);
            lse_tensors.push_back(lse_gs_ms_device_result);

        }

        torch::Tensor out_device_result = results[4].to(torch::kCPU).view({batch_size, seqlen, nheads, d});
        torch::Tensor lse_device_result = results[5].to(torch::kCPU);
        torch::Tensor z_device_result = results[6].to(torch::kCPU);

        for(std::size_t i = 0; i < batch_size; i++)
        {
            const auto& a_gs_ms_ks         = a_tensors[i];
            const auto& b0_gs_ns_ks        = b0_tensors[i];
            const auto& b1_gs_os_ns        = b1_tensors[i];
            auto& c_gs_ms_os_device_result = c_tensors[i];
            auto& z_gs_ms_ns_device_result = z_tensors[i];
            auto& lse_gs_ms_device_result = lse_tensors[i];
            //auto& c_gs_ms_os_device_buf    = *c_tensors_device[i];

            //torch::Tensor out_device_result = out.to(torch::kCPU).view({batch_size, seqlen, nheads, d});
            void* out_host_ptr_f = out_device_result[i].data_ptr();
            CDataType* out_host_ptr = reinterpret_cast<CDataType*>(out_host_ptr_f);
            std::vector<CDataType> result_vector(out_host_ptr, out_host_ptr + out_device_result[i].numel()); //transfer tensor into vector
            c_gs_ms_os_device_result.mData.assign(result_vector.begin(), result_vector.end());

            void* lse_host_ptr_f = lse_device_result[i].data_ptr();
            LSEDataType* lse_host_ptr = reinterpret_cast<LSEDataType*>(lse_host_ptr_f);
            std::vector<LSEDataType> result_lse_vector(lse_host_ptr, lse_host_ptr + lse_device_result[i].numel()); //transfer tensor into vector
            lse_gs_ms_device_result.mData.assign(result_lse_vector.begin(), result_lse_vector.end());

            void* z_host_ptr_f = z_device_result[i].data_ptr();
            ZDataType* z_host_ptr = reinterpret_cast<ZDataType*>(z_host_ptr_f);
            std::vector<ZDataType> result_z_vector(z_host_ptr, z_host_ptr + z_device_result[i].numel()); //transfer tensor into vector
            z_gs_ms_ns_device_result.mData.assign(result_z_vector.begin(), result_z_vector.end());

            //c_gs_ms_os_device_buf.FromDevice(c_gs_ms_os_device_result.mData.data());//

            Tensor<ADataType> a_g_m_k({G0 * G1, M, K});
            Tensor<B0DataType> b0_g_k_n({G0 * G1, K, N});
            Tensor<B1DataType> b1_g_n_o({G0 * G1, N, O});
            Tensor<AccDataType> acc0_g_m_n({G0 * G1, M, N});        // scratch object after gemm0
            Tensor<ADataType> a1_g_m_n({G0 * G1, M, N});            // scratch object after softmax
            Tensor<ADataType> a1_g_m_n_drop({G0 * G1, M, N});
            Tensor<CDataType> c_g_m_o_host_result({G0 * G1, M, O}); // scratch object after gemm1
            Tensor<ZDataType> z_g_m_n({G0 * G1, M, N});
            Tensor<LSEDataType> lse_g_m_host_result({G0 * G1, M}); // scratch object after gemm1

            std::vector<ck::index_t> c_gs_ms_os_lengths{G0, G1, M, O};
            std::vector<ck::index_t> c_gs_ms_os_strides{M * G1 * O, O, G1 * O, 1};
            std::vector<ck::index_t> lse_gs_ms_lengths{G0, G1, M};
            std::vector<ck::index_t> lse_gs_ms_strides{M * G1, M, 1};

            Tensor<CDataType> c_gs_ms_os_host_result(c_gs_ms_os_lengths, c_gs_ms_os_strides);
            Tensor<LSEDataType> lse_gs_ms_host_result(lse_gs_ms_lengths, lse_gs_ms_strides);

            // permute
            a_gs_ms_ks.ForEach([&](auto& self, auto idx) {
                a_g_m_k(idx[0] * G1 + idx[1], idx[2], idx[3]) = self(idx);
            });
            b0_gs_ns_ks.ForEach([&](auto& self, auto idx) {
                b0_g_k_n(idx[0] * G1 + idx[1], idx[3], idx[2]) = self(idx);
            });
            b1_gs_os_ns.ForEach([&](auto& self, auto idx) {
                b1_g_n_o(idx[0] * G1 + idx[1], idx[3], idx[2]) = self(idx);
            });

            z_gs_ms_ns_device_result.ForEach([&](auto& self, auto idx) {
                z_g_m_n(idx[0] * G1 + idx[1], idx[2], idx[3]) = self(idx);
            });

            // gemm 0
            auto ref_gemm0          = ReferenceGemm0Instance{};
            auto ref_gemm0_invoker  = ref_gemm0.MakeInvoker();
            auto ref_gemm0_argument = ref_gemm0.MakeArgument(
                a_g_m_k, b0_g_k_n, acc0_g_m_n, a_element_op, b0_element_op, acc0_element_op);

            ref_gemm0_invoker.Run(ref_gemm0_argument);

            //// masking
            //const auto mask = DeviceGemmInstance::C0MatrixMask(N);
            //acc0_g_m_n.ForEach([&](auto& self, auto idx) {
            //    if(mask.IsMaskedElement(idx[1], idx[2]))
            //        self(idx) = -ck::NumericLimits<float>::Infinity();
            //});

            // softmax
            auto ref_softmax          = ReferenceSoftmaxInstance{};
            auto ref_softmax_invoker  = ref_softmax.MakeInvoker();
            auto ref_softmax_argument = ref_softmax.MakeArgument(acc0_g_m_n, a1_g_m_n, 1, 0, {2}, &lse_g_m_host_result);

            ref_softmax_invoker.Run(ref_softmax_argument);

            //printf("print z_g_m_n \n");
            //z_g_m_n.ForEach([&](auto& self, auto idx) {printf("%u ", self(idx));});

            // dropout after softmax
            auto ref_dropout         = ReferenceDropoutInstance{};
            auto ref_dropout_invoker = ref_dropout.MakeInvoker();
            auto ref_dropout_argment = ref_dropout.MakeArgument(
                z_g_m_n, a1_g_m_n, a1_g_m_n_drop, p_dropout_in_16bits, rp_dropout);
            ref_dropout_invoker.Run(ref_dropout_argment);

            // gemm 1
            auto ref_gemm1          = ReferenceGemm1Instance{};
            auto ref_gemm1_invoker  = ref_gemm1.MakeInvoker();
            auto ref_gemm1_argument = ref_gemm1.MakeArgument(a1_g_m_n,
                                                             b1_g_n_o,
                                                             c_g_m_o_host_result,
                                                             PassThrough{},
                                                             b1_element_op,
                                                             c_element_op);

            ref_gemm1_invoker.Run(ref_gemm1_argument);

            // permute
            c_gs_ms_os_host_result.ForEach([&](auto& self, auto idx) {
                const size_t& g0 = idx[0];
                const size_t& g1 = idx[1];

                const size_t g = g0 * G1 + g1;

                self(idx) = c_g_m_o_host_result(g, idx[2], idx[3]);
            });


            lse_gs_ms_host_result.ForEach([&](auto& self, auto idx) {
                const size_t& g0 = idx[0];
                const size_t& g1 = idx[1];

                const size_t g = g0 * G1 + g1;

                self(idx) = lse_g_m_host_result(g, idx[2]);
            });

            double rtol = 1e-2;
            double atol = 1e-2;

            return ck::utils::check_err(c_gs_ms_os_device_result.mData, c_gs_ms_os_host_result.mData, "Error: Incorrect results!",
                                    rtol,
                                    atol);
        }
    }
    return true;
}


bool bwd_test(bool do_verification){
    int batch_size = 64;
    int nheads = 16;
    int seqlen = 256;
    int n = 1024;
    int d = n / nheads; //head_size//64

    //initialize the tensors
    torch::Tensor q_host = at::rand({batch_size, seqlen, nheads, d}, torch::kFloat16);//torch::kBFloat16;at::kHalf
    torch::Tensor k_host = at::rand({batch_size, seqlen, nheads, d}, torch::kFloat16);
    torch::Tensor v_host = at::rand({batch_size, seqlen, nheads, d}, torch::kFloat16);
    torch::Tensor y_host = at::empty({batch_size, seqlen, nheads, d}, torch::kFloat16);
    torch::Tensor z_host = at::empty({batch_size, nheads, seqlen, seqlen}, torch::kInt32);
    torch::Tensor lse_host = at::empty({batch_size, nheads, seqlen}, torch::kFloat32);

    torch::Tensor ygrad_host = at::rand({batch_size, seqlen, nheads, d}, torch::kFloat16);
    torch::Tensor qgrad_host = at::empty({batch_size, seqlen, nheads, d}, torch::kFloat16);
    torch::Tensor kgrad_host = at::empty({batch_size, seqlen, nheads, d}, torch::kFloat16);
    torch::Tensor vgrad_host = at::empty({batch_size, seqlen, nheads, d}, torch::kFloat16);

    torch::Tensor q = q_host.to(at::kCUDA);
    torch::Tensor k = k_host.to(at::kCUDA);
    torch::Tensor v = v_host.to(at::kCUDA);
    c10::optional<torch::Tensor> y = y_host.to(at::kCUDA);
    torch::Tensor lse = lse_host.to(at::kCUDA);
    c10::optional<torch::Tensor> qgrad = qgrad_host.to(at::kCUDA);
    c10::optional<torch::Tensor> vgrad = vgrad_host.to(at::kCUDA);
    c10::optional<torch::Tensor> kgrad = kgrad_host.to(at::kCUDA);
    torch::Tensor ygrad = ygrad_host.to(at::kCUDA);

    //initialize seqlens vector (size is b+1)
    std::vector<int> cu_seqlens_q_vec;
    std::vector<int> cu_seqlens_k_vec;

    for (int i = 0 ; i < batch_size + 1; i++){
      cu_seqlens_q_vec.push_back(i * seqlen);
      cu_seqlens_k_vec.push_back(i * seqlen);
    }

    torch::TensorOptions opts=torch::TensorOptions().dtype(torch::kInt32);
    torch::Tensor cu_seqlens_q=at::from_blob(cu_seqlens_q_vec.data(),{batch_size + 1},opts).clone().to(at::kCUDA);
    torch::Tensor cu_seqlens_k=at::from_blob(cu_seqlens_k_vec.data(),{batch_size + 1},opts).clone().to(at::kCUDA);

    int max_seqlen_q_ = seqlen;
    int max_seqlen_k_ = seqlen;
    
    //other parameters
    float p_dropout = 0;
    float p_dropout2                = 1 - p_dropout;
    uint16_t p_dropout_in_16bits    = uint16_t(std::floor(p_dropout2 * 65535.0));
    float rp_dropout                = 1.0 / p_dropout2;
    const unsigned long long seed   = 1;
    const unsigned long long offset = 0;           
    float softmax_scale = 1/sqrt(d);  
    bool zero_tensors = true;    
    bool is_causal = false;
    bool return_softmax = false;
    c10::optional<at::Generator> gen_ = c10::nullopt;
    c10::optional<torch::Tensor> rng_state = c10::nullopt;
    auto results = mha_fwd(q,   
                  k,   
                  v,   
                  y, 
                  p_dropout,
                  softmax_scale,
                  is_causal,
                  return_softmax,
                  gen_);
    mha_bwd(ygrad,
            q,   
            k,   
            v,   
            results[4],
            lse = results[5],
            qgrad,
            kgrad,
            vgrad,
            p_dropout,
            softmax_scale,
            is_causal,
            gen_,
            rng_state);
    using F16 = ck::half_t;
    using BF16 = ck::bhalf_t;
    using F32 = float;
    using U16 = unsigned short;

    using PassThrough = ck::tensor_operation::element_wise::PassThrough;
    using Scale       = ck::tensor_operation::element_wise::Scale;

    using QKVElementOp = PassThrough;
    using YElementOp   = PassThrough;

    using DataType         = F16;
    using ZDataType        = U16;
    using AccDataType      = F32;
    using ShuffleDataType  = F32;
    using LSEDataType      = F32;
    using Acc0BiasDataType = ck::Tuple<>;
    using Acc1BiasDataType = ck::Tuple<>;

    static constexpr ck::index_t NumDimG = 2;
    static constexpr ck::index_t NumDimM = 1;
    static constexpr ck::index_t NumDimN = 1;
    static constexpr ck::index_t NumDimK = 1;
    static constexpr ck::index_t NumDimO = 1;
    // Ref Gemm0: S = alpha * Q * K^T
    // fp16 in, fp32 out
    using ReferenceGemm0Instance = ck::tensor_operation::host::ReferenceBatchedGemm<DataType,
                                                                                    DataType,
                                                                                    AccDataType,
                                                                                    AccDataType,
                                                                                    PassThrough,
                                                                                    PassThrough,
                                                                                    Scale>;

    // Ref Softmax: P = Softmax(S)
    // fp32 in, fp16 out
    using ReferenceSoftmaxInstance =
        ck::tensor_operation::host::ReferenceSoftmax<AccDataType, DataType, AccDataType>;

    // Ref Gemm1: Y = P * V
    // fp16 in, fp16 out
    using ReferenceGemm1Instance = ck::tensor_operation::host::ReferenceBatchedGemm<DataType,
                                                                                    DataType,
                                                                                    DataType,
                                                                                    AccDataType,
                                                                                    PassThrough,
                                                                                    PassThrough,
                                                                                    PassThrough>;

    // Ref Gemm for backward pass
    // fp16 in, fp16 out
    using ReferenceGemmGradInstance = ck::tensor_operation::host::ReferenceBatchedGemm<DataType,
                                                                                    DataType,
                                                                                    DataType,
                                                                                    AccDataType,
                                                                                    PassThrough,
                                                                                    PassThrough,
                                                                                    Scale>;

    using ReferenceDropoutInstance =
        ck::tensor_operation::host::ReferenceDropout<ushort, DataType, DataType>;
                                                                                    
    if(do_verification){
        bool input_permute = false;
        bool output_permute = false;
        auto run_attention_fwd_host = []<typename TensorQ,
          typename TensorK,
          typename TensorV,
          typename TensorS,
          typename TensorP,
          typename TensorZ,
          typename TensorY,
          typename TensorLSE = TensorP>(const TensorQ& q_g_m_k,
                                        const TensorK& k_g_n_k,
                                        const TensorV& v_g_n_o,
                                        const float alpha,
                                        TensorS& s_g_m_n,
                                        TensorP& p_g_m_n,
                                        TensorY& y_g_m_o,
                                        TensorLSE& lse_g_m,
                                        TensorP& p_drop_g_m_n,
                                        TensorZ& z_g_m_n,
                                        ushort p_dropout_in_16bits,
                                        float rp_dropout)
        {
            // S = alpha * Q * K^T
            auto k_g_k_n            = k_g_n_k.Transpose({0, 2, 1});
            auto ref_gemm0          = ReferenceGemm0Instance{};
            auto ref_gemm0_invoker  = ref_gemm0.MakeInvoker();
            auto ref_gemm0_argument = ref_gemm0.MakeArgument(
                q_g_m_k, k_g_k_n, s_g_m_n, PassThrough{}, PassThrough{}, Scale{alpha});

            ref_gemm0_invoker.Run(ref_gemm0_argument);

            // masking
        // #if USING_MASK
        //     auto N          = s_g_m_n.GetLengths()[2];
        //     const auto mask = DeviceGemmInstance::C0MatrixMask(N);
        //     s_g_m_n.ForEach([&](auto& self, auto idx) {
        //         if(mask.IsMaskedElement(idx[1], idx[2]))
        //             self(idx) = -ck::NumericLimits<float>::Infinity();
        //     });
        // #endif

            // P = Softmax(S)
            auto ref_softmax          = ReferenceSoftmaxInstance{};
            auto ref_softmax_invoker  = ref_softmax.MakeInvoker();
            auto ref_softmax_argument = ref_softmax.MakeArgument(s_g_m_n, p_g_m_n, 1, 0, {2}, &lse_g_m);

            ref_softmax_invoker.Run(ref_softmax_argument);

            // P_dropped
            auto ref_dropout         = ReferenceDropoutInstance{};
            auto ref_dropout_invoker = ref_dropout.MakeInvoker();
            auto ref_dropout_argment =
                ref_dropout.MakeArgument(z_g_m_n, p_g_m_n, p_drop_g_m_n, p_dropout_in_16bits, rp_dropout);
            ref_dropout_invoker.Run(ref_dropout_argment);

            // Y = P_dropout * V
            auto ref_gemm1          = ReferenceGemm1Instance{};
            auto ref_gemm1_invoker  = ref_gemm1.MakeInvoker();
            auto ref_gemm1_argument = ref_gemm1.MakeArgument(
                p_drop_g_m_n, v_g_n_o, y_g_m_o, PassThrough{}, PassThrough{}, PassThrough{});

            ref_gemm1_invoker.Run(ref_gemm1_argument);
        };
        q_host = q_host.view({ batch_size, seqlen, nheads, d }); //64 256 16 64
        k_host = k_host.view({ batch_size, seqlen, nheads, d });
        v_host = v_host.view({ batch_size, seqlen, nheads, d });
        z_host = z_host.view({ batch_size, nheads, seqlen, seqlen });
        ygrad_host = ygrad_host.view({ batch_size, seqlen, nheads, d });

        const int M   = seqlen;   //seqlen Q
        const int N   = seqlen;   //seqlen K
        const int K   = d;        //head_dim
        const int O   = d;        //head_dim
        const int G0  = 1;        // G0 = batch_size
        const int G1  = nheads;   // num_heads_q

        auto a_element_op    = QKVElementOp{};
        auto b0_element_op   = QKVElementOp{};
        auto acc0_element_op = Scale{softmax_scale};
        auto b1_element_op   = QKVElementOp{};
        auto c_element_op    = YElementOp{};
        qgrad_host = qgrad->to(torch::kCPU).view({batch_size, seqlen, nheads, d});
        kgrad_host = kgrad->to(torch::kCPU).view({batch_size, seqlen, nheads, d});
        vgrad_host = vgrad->to(torch::kCPU).view({batch_size, seqlen, nheads, d});
        lse_host = lse.to(torch::kCPU);
        y_host = y->to(torch::kCPU).view({batch_size, seqlen, nheads, d});

        for(std::size_t i=0; i<batch_size; i++){
            std::vector<ck::index_t> q_gs_ms_ks_lengths{G0, G1, M, K};
            std::vector<ck::index_t> q_gs_ms_ks_strides =
                input_permute
                    ? std::vector<ck::index_t>{M * G1 * K, K, G1 * K, 1} // Q layout [G0, M, G1, K]
                    : std::vector<ck::index_t>{G1 * M * K, M * K, K, 1}; // Q layout [G0, G1, M, K]

            std::vector<ck::index_t> k_gs_ns_ks_lengths{G0, G1, N, K};
            std::vector<ck::index_t> k_gs_ns_ks_strides =
                input_permute
                    ? std::vector<ck::index_t>{N * G1 * K, K, G1 * K, 1} // K layout [G0, N, G1, K]
                    : std::vector<ck::index_t>{G1 * N * K, N * K, K, 1}; // K layout [G0, G1, N, K]

            std::vector<ck::index_t> v_gs_os_ns_lengths{G0, G1, O, N};
            std::vector<ck::index_t> v_gs_os_ns_strides =
                input_permute
                    ? std::vector<ck::index_t>{N * G1 * O, O, 1, G1 * O} // V layout [G0, N, G1, O]
                    : std::vector<ck::index_t>{G1 * N * O, N * O, 1, O}; // V layout [G0, G1, N, O]

            std::vector<ck::index_t> y_gs_ms_os_lengths{G0, G1, M, O};
            std::vector<ck::index_t> y_gs_ms_os_strides =
                output_permute
                    ? std::vector<ck::index_t>{M * G1 * O, O, G1 * O, 1} // Y layout [G0, M, G1, O]
                    : std::vector<ck::index_t>{G1 * M * O, M * O, O, 1}; // Y layout [G0, G1, M, O]

            std::vector<ck::index_t> z_gs_ms_ns_lengths{G0, G1, M, N};
            std::vector<ck::index_t> z_gs_ms_ns_strides =
                input_permute
                    ? std::vector<ck::index_t>{M * G1 * N, N, G1 * N, 1} // Z layout [G0, M, G1, N]
                    : std::vector<ck::index_t>{G1 * M * N, M * N, N, 1}; // Z layout [G0, G1, M, N]
            // The softmax stat log-sum-exp (LSE) is used to speed up softmax calculation in backward
            // pass Pi = exp(Si) / sum(exp(S0) + exp(S1) + ...)
            //    = exp(Si) / exp(log(sum(exp() + ...)))
            //    = exp(Si - log(sum(exp() + ...)))
            //               ^^^^^^^^^^^^^^^^^^^^^
            //                       LSE
            std::vector<ck::index_t> lse_gs_ms_lengths{G0, G1, M};
            std::vector<ck::index_t> lse_gs_ms_strides{G1 * M, M, 1}; // LSE layout [G0, G1, M]

            Tensor<DataType> q_gs_ms_ks(q_gs_ms_ks_lengths, q_gs_ms_ks_strides);
            Tensor<DataType> k_gs_ns_ks(k_gs_ns_ks_lengths, k_gs_ns_ks_strides);
            Tensor<DataType> v_gs_os_ns(v_gs_os_ns_lengths, v_gs_os_ns_strides);
            Tensor<DataType> y_gs_ms_os(y_gs_ms_os_lengths, y_gs_ms_os_strides);
            Tensor<ZDataType> z_gs_ms_ns(z_gs_ms_ns_lengths, z_gs_ms_ns_strides);
            // Tensor<DataType> y_gs_ms_os_device(y_gs_ms_os_lengths, y_gs_ms_os_strides);
            Tensor<DataType> ygrad_gs_ms_os(y_gs_ms_os_lengths, y_gs_ms_os_strides);
            Tensor<LSEDataType> lse_gs_ms(lse_gs_ms_lengths, lse_gs_ms_strides);
            // Tensor<LSEDataType> lse_gs_ms_device(lse_gs_ms_lengths, lse_gs_ms_strides);
            Tensor<DataType> qgrad_gs_ms_ks(q_gs_ms_ks_lengths, q_gs_ms_ks_strides);
            Tensor<DataType> kgrad_gs_ns_ks(k_gs_ns_ks_lengths, k_gs_ns_ks_strides);
            Tensor<DataType> vgrad_gs_os_ns(v_gs_os_ns_lengths, v_gs_os_ns_strides);

            DataType* q_h_ptr = reinterpret_cast<DataType*>(q_host[i].data_ptr());
            DataType* k_h_ptr = reinterpret_cast<DataType*>(k_host[i].data_ptr());
            DataType* v_h_ptr = reinterpret_cast<DataType*>(v_host[i].data_ptr());
            DataType* y_h_ptr = reinterpret_cast<DataType*>(y_host[i].data_ptr());
            ZDataType* z_h_ptr = reinterpret_cast<ZDataType*>(z_host[i].data_ptr());
            LSEDataType* lse_h_ptr = reinterpret_cast<LSEDataType*>(lse_host[i].data_ptr());
            DataType* ygrad_h_ptr = reinterpret_cast<DataType*>(ygrad_host[i].data_ptr());
            DataType* qgrad_h_ptr = reinterpret_cast<DataType*>(qgrad_host[i].data_ptr());
            DataType* kgrad_h_ptr = reinterpret_cast<DataType*>(kgrad_host[i].data_ptr());
            DataType* vgrad_h_ptr = reinterpret_cast<DataType*>(vgrad_host[i].data_ptr());

            std::vector<DataType> q_vector(q_h_ptr, q_h_ptr + q_host[i].numel()); 
            q_gs_ms_ks.mData.assign(q_vector.begin(), q_vector.end());
            std::vector<DataType> k_vector(k_h_ptr, k_h_ptr + k_host[i].numel()); 
            k_gs_ns_ks.mData.assign(k_vector.begin(), k_vector.end());
            std::vector<DataType> v_vector(v_h_ptr, v_h_ptr + v_host[i].numel()); 
            v_gs_os_ns.mData.assign(v_vector.begin(), v_vector.end());
            std::vector<ZDataType> z_vector(z_h_ptr, z_h_ptr + z_host[i].numel()); 
            z_gs_ms_ns.mData.assign(z_vector.begin(), z_vector.end());
            std::vector<DataType> y_vector(y_h_ptr, y_h_ptr + y_host[i].numel()); 
            y_gs_ms_os.mData.assign(y_vector.begin(), y_vector.end());

            std::vector<DataType> lse_vector(lse_h_ptr, lse_h_ptr + lse_host[i].numel()); 
            lse_gs_ms.mData.assign(lse_vector.begin(), lse_vector.end());
            std::vector<DataType> ygrad_vector(ygrad_h_ptr, ygrad_h_ptr + ygrad_host[i].numel()); 
            ygrad_gs_ms_os.mData.assign(ygrad_vector.begin(), ygrad_vector.end());
            std::vector<DataType> qgrad_vector(qgrad_h_ptr, qgrad_h_ptr + qgrad_host[i].numel()); 
            qgrad_gs_ms_ks.mData.assign(qgrad_vector.begin(), qgrad_vector.end());
            std::vector<DataType> kgrad_vector(kgrad_h_ptr, kgrad_h_ptr + kgrad_host[i].numel()); 
            kgrad_gs_ns_ks.mData.assign(kgrad_vector.begin(), kgrad_vector.end());
            std::vector<DataType> vgrad_vector(vgrad_h_ptr, vgrad_h_ptr + vgrad_host[i].numel()); 
            vgrad_gs_os_ns.mData.assign(vgrad_vector.begin(), vgrad_vector.end());

            int BatchCount = G0 * G1;
            Tensor<DataType> q_g_m_k({BatchCount, M, K});
            Tensor<DataType> k_g_n_k({BatchCount, N, K});
            Tensor<DataType> v_g_n_o({BatchCount, N, O});
            Tensor<ZDataType> z_g_m_n({BatchCount, M, N});
            Tensor<AccDataType> s_g_m_n({BatchCount, M, N});
            Tensor<DataType> p_g_m_n({BatchCount, M, N});
            Tensor<DataType> y_g_m_o({BatchCount, M, O});
            Tensor<LSEDataType> lse_g_m({BatchCount, M});            
            Tensor<DataType> ygrad_g_m_o({BatchCount, M, O});
            Tensor<DataType> p_drop_g_m_n({BatchCount, M, N});

            q_gs_ms_ks.ForEach(
                [&](auto& self, auto idx) { q_g_m_k(idx[0] * G1 + idx[1], idx[2], idx[3]) = self(idx); });
            k_gs_ns_ks.ForEach(
                [&](auto& self, auto idx) { k_g_n_k(idx[0] * G1 + idx[1], idx[2], idx[3]) = self(idx); });
            v_gs_os_ns.ForEach(
                [&](auto& self, auto idx) { v_g_n_o(idx[0] * G1 + idx[1], idx[3], idx[2]) = self(idx); });
            z_gs_ms_ns.ForEach(
                [&](auto& self, auto idx) { z_g_m_n(idx[0] * G1 + idx[1], idx[2], idx[3]) = self(idx); });

            run_attention_fwd_host(q_g_m_k, k_g_n_k, v_g_n_o, softmax_scale, s_g_m_n, p_g_m_n, y_g_m_o, lse_g_m, p_drop_g_m_n, z_g_m_n, p_dropout_in_16bits, rp_dropout);
            std::cout << "Checking lse:\n";
            ck::utils::check_err(lse_g_m.mData,
                                 lse_gs_ms.mData,
                                 "error",
                                 1e-2,
                                 1e-2);
            ygrad_gs_ms_os.ForEach([&](auto& self, auto idx) {
                ygrad_g_m_o(idx[0] * G1 + idx[1], idx[2], idx[3]) = self(idx);
            });
            Tensor<DataType> pgrad_g_m_n({BatchCount, M, N});
            Tensor<DataType> sgrad_g_m_n({BatchCount, M, N});
            Tensor<DataType> qgrad_g_m_k({BatchCount, M, K});
            Tensor<DataType> kgrad_g_n_k({BatchCount, N, K});
            Tensor<DataType> vgrad_g_n_o({BatchCount, N, O});

            auto ref_gemm_grad         = ReferenceGemmGradInstance{};
            auto ref_gemm_grad_invoker = ref_gemm_grad.MakeInvoker();
            using RefGemmGradArg       = ReferenceGemmGradInstance::Argument;
            // dP = dY * V^T
            auto v_g_o_n = v_g_n_o.Transpose({0, 2, 1});
            ref_gemm_grad_invoker.Run(RefGemmGradArg{
                ygrad_g_m_o, v_g_o_n, pgrad_g_m_n, PassThrough{}, PassThrough{}, Scale{1.f}});
            sgrad_g_m_n.ForEach([&](auto& self, auto idx_gmn) {
                float ygrad_dot_y = 0;
                for(int o = 0; o < O; o++)
                {
                    auto idx_gmo = idx_gmn;
                    idx_gmo[2]   = o;
                    ygrad_dot_y += ygrad_g_m_o(idx_gmo) * y_g_m_o(idx_gmo);
                }
                self(idx_gmn) = p_g_m_n(idx_gmn) * (pgrad_g_m_n(idx_gmn) - ygrad_dot_y);
            });
            auto p_g_n_m = p_g_m_n.Transpose({0, 2, 1});
            ref_gemm_grad_invoker.Run(RefGemmGradArg{
                p_g_n_m, ygrad_g_m_o, vgrad_g_n_o, PassThrough{}, PassThrough{}, Scale{1.f}});
            ref_gemm_grad_invoker.Run(RefGemmGradArg{
                sgrad_g_m_n, k_g_n_k, qgrad_g_m_k, PassThrough{}, PassThrough{}, Scale{softmax_scale}});
            auto sgrad_g_n_m = sgrad_g_m_n.Transpose({0, 2, 1});
            ref_gemm_grad_invoker.Run(RefGemmGradArg{
                sgrad_g_n_m, q_g_m_k, kgrad_g_n_k, PassThrough{}, PassThrough{}, Scale{softmax_scale}});

            Tensor<DataType> qgrad_gs_ms_ks_host_result(qgrad_gs_ms_ks.GetLengths(), qgrad_gs_ms_ks.GetStrides());
            Tensor<DataType> kgrad_gs_ns_ks_host_result(kgrad_gs_ns_ks.GetLengths(), kgrad_gs_ns_ks.GetStrides());
            Tensor<DataType> vgrad_gs_os_ns_host_result(vgrad_gs_os_ns.GetLengths(), vgrad_gs_os_ns.GetStrides());

            // permute
            qgrad_gs_ms_ks_host_result.ForEach([&](auto& self, auto idx) {
                const size_t& g0 = idx[0];
                const size_t& g1 = idx[1];

                const size_t g = g0 * G1 + g1;

                self(idx) = qgrad_g_m_k(g, idx[2], idx[3]);
            });
            kgrad_gs_ns_ks_host_result.ForEach([&](auto& self, auto idx) {
                const size_t& g0 = idx[0];
                const size_t& g1 = idx[1];

                const size_t g = g0 * G1 + g1;

                self(idx) = kgrad_g_n_k(g, idx[2], idx[3]);
            });
            vgrad_gs_os_ns_host_result.ForEach([&](auto& self, auto idx) {
                const size_t& g0 = idx[0];
                const size_t& g1 = idx[1];

                const size_t g = g0 * G1 + g1;

                self(idx) = vgrad_g_n_o(g, idx[3], idx[2]);
            });
            bool pass = true;
            std::cout << "Checking qgrad:\n";
            pass &= ck::utils::check_err(qgrad_gs_ms_ks.mData,
                                         qgrad_gs_ms_ks_host_result.mData,
                                         "error",
                                         1e-2,
                                         1e-2);
            std::cout << "Checking kgrad:\n";
            pass &= ck::utils::check_err(kgrad_gs_ns_ks.mData,
                                         kgrad_gs_ns_ks_host_result.mData,
                                         "error",
                                         1e-2,
                                         1e-2);
            std::cout << "Checking vgrad:\n";
            pass &= ck::utils::check_err(vgrad_gs_os_ns.mData,
                                         vgrad_gs_os_ns_host_result.mData,
                                         "error",
                                         1e-2,
                                         1e-2);
            return pass;
        }
    }
    return true;    
}


int main(){
    bool pass = true;
    bool do_verification = true; // whether do verification
    pass &= fwd_test(do_verification);
    std::cout << "Forward finished!" <<std::endl;
    pass &= bwd_test(do_verification);
    std::cout << "Backward finished!" <<std::endl;
    if(do_verification){
        if(pass)
            std::cout << "Verification passed!" <<std::endl;
        else
            std::cout << "Verification failed!" <<std::endl;
    }
    return pass ? 0 : 1;
}
#endif