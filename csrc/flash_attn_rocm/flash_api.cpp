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
    // dq = dq.contiguous();
    // dk = dk.contiguous();
    // dv = dv.contiguous();
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