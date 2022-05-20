/******************************************************************************
 * Copyright (c) 2011-2021, NVIDIA CORPORATION.  All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "fmha.h"

void set_params(Fused_multihead_attention_fprop_params &params,
                // sizes
                const size_t b,
                const size_t s,
                const size_t h,
                const size_t d,
                // device pointers
                void *qkv_packed_d,
                void *cu_seqlens_d,
                void *o_packed_d,
                void *o_tmp_d,
                void *do_packed_d,
                void *s_d,
                void *softmax_lse_d,
                void *dsoftmax_sum_d,
                float p_dropout,
                float softmax_scale,
                bool is_causal) {

    Data_type acc_type = DATA_TYPE_FP32;
    Data_type data_type = DATA_TYPE_FP16;

    // Reset the parameters
    memset(&params, 0, sizeof(params));

    // Set the pointers and strides.
    params.qkv_ptr = qkv_packed_d;
    params.qkv_stride_in_elts = h * 3 * d;
    params.qkv_stride_in_bytes = get_size_in_bytes(h * 3 * d, data_type);
    params.o_ptr = o_packed_d;
    params.o_stride_in_elts = h * d;
    params.o_stride_in_bytes = get_size_in_bytes(h * d, data_type);
    params.do_ptr = do_packed_d;
    params.o_tmp_ptr = o_tmp_d;

    params.cu_seqlens = static_cast<int *>(cu_seqlens_d);

    // S = softmax(P)
    params.s_ptr = s_d;
    params.s_stride_in_bytes = get_size_in_bytes(b * h * s, data_type);

    // Softmax sum
    params.softmax_lse_ptr = softmax_lse_d;
    params.dsoftmax_sum = dsoftmax_sum_d;

    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.s = s;
    params.d = d;

    // Set the different scale values.
    // const float scale_bmm1 = 1.f / sqrtf(d);
    const float scale_bmm1 = softmax_scale;
    constexpr float scale_softmax = 1.f;
    constexpr float scale_bmm2 = 1.f;

    params.scale_bmm1f = scale_bmm1;
    set_alpha(params.scale_bmm1, scale_bmm1, data_type);
    set_alpha(params.scale_softmax, scale_softmax, acc_type);
    set_alpha(params.scale_bmm2, scale_bmm2, data_type);

    // Set this to probability of keeping an element to simplify things.
    params.p_dropout = 1.f - p_dropout;
    // Convert p from float to int so we don't have to convert the random uint to float to compare.
    // [Minor] We want to round down since when we do the comparison we use <= instead <
    params.p_dropout_in_uint = uint32_t(std::floor(params.p_dropout * 4294967295.0));
    params.p_dropout_in_uint16_t = uint16_t(std::floor(params.p_dropout * 65535.0));
    params.rp_dropout = 1.f / params.p_dropout;
    TORCH_CHECK(p_dropout < 1.f);
    set_alpha(params.scale_dropout, params.rp_dropout, data_type);

    params.is_causal = is_causal;
}

std::vector<at::Tensor> 
mha_fwd(const at::Tensor &qkv,         // total x num_heads x 3 x head_size, total := \sum_{i=0}^{b} s_i
        const at::Tensor &cu_seqlens,  // b+1
        const float p_dropout,
        const int max_seq_len,
        const float softmax_scale,
        const bool zero_tensors,
        const bool is_causal,
        const bool return_softmax,
        c10::optional<at::Generator> gen_) {

    auto dprops = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(dprops->major == 8 && dprops->minor >= 0);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    bool is_dropout = p_dropout > 0.0;
    Launch_params<Fused_multihead_attention_fprop_params> launch_params(dprops, stream, is_dropout, return_softmax);

    TORCH_CHECK(qkv.is_cuda())
    TORCH_CHECK(cu_seqlens.is_cuda())

    TORCH_CHECK(qkv.is_contiguous())
    TORCH_CHECK(cu_seqlens.is_contiguous())

    TORCH_CHECK(cu_seqlens.dim() == 1);
    TORCH_CHECK(qkv.dim() == 4);

    const auto sizes = qkv.sizes();

    TORCH_CHECK(sizes[THREE_DIM] == 3);

    const int batch_size = cu_seqlens.numel() - 1;
    const int total = sizes[TOTAL_DIM];
    const int num_heads = sizes[H_DIM];
    const int head_size = sizes[D_DIM];
    TORCH_CHECK(batch_size > 0);
    TORCH_CHECK(head_size == 16 || head_size == 32 || head_size == 64 || head_size == 128);

    // int base_N = head_size == 16 ? 512 : (head_size == 128 ? 128 : 256);
    int base_N = head_size == 128 ? 128 : 256;
    // int base_N = 256;
    int seq_len = 512;
    if( max_seq_len <= 128 ) {
        seq_len = 128;
    } else if( max_seq_len <= 256 ) {
        seq_len = 256;
    } else {
        seq_len = ((max_seq_len + base_N - 1) / base_N) * base_N;
    }
    bool loop = seq_len > base_N;

    auto opts = qkv.options();

    auto ctx = torch::empty({ total, num_heads, head_size }, opts);

    at::Tensor o_tmp;
    if (loop) {
        o_tmp = torch::empty({total, num_heads, head_size}, opts.dtype(at::kFloat));
    }

    auto softmax_lse = torch::empty({batch_size, num_heads, seq_len}, opts.dtype(at::kFloat));
    // auto softmax_lse = torch::full({batch_size, num_heads, seq_len}, -std::numeric_limits<float>::infinity(), opts.dtype(at::kFloat));

    at::Tensor s;
    if (return_softmax) {
        s = torch::empty({ batch_size, num_heads, seq_len, seq_len }, opts);
        // s = torch::ones({ batch_size, num_heads, seq_len, seq_len }, opts) * 10000.0;
    }

    if( zero_tensors ) {
        ctx.zero_();
        softmax_lse.fill_(-std::numeric_limits<float>::infinity());
        if (loop) { o_tmp.zero_(); }
        if (return_softmax) {s.zero_();}
    }

    auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
        gen_, at::cuda::detail::getDefaultCUDAGenerator());


    set_params(launch_params.params,
               batch_size,
               seq_len,
               num_heads,
               head_size,
               qkv.data_ptr(),
               cu_seqlens.data_ptr(),
               ctx.data_ptr(),
               loop ? o_tmp.data_ptr() : nullptr,
               nullptr,
               return_softmax ? s.data_ptr() : nullptr,
               softmax_lse.data_ptr(),
               nullptr,
               p_dropout,
               softmax_scale,
               is_causal);

    run_fmha_fp16_sm80(launch_params, /*configure=*/ true);
    // number of times random will be generated per thread, to offset philox counter in thc random
    // state
    int64_t counter_offset = launch_params.elts_per_thread;
    at::PhiloxCudaState rng_engine_inputs;

    if( is_dropout ) {
        // See Note [Acquire lock when using random generators]
        std::lock_guard<std::mutex> lock(gen->mutex_);
        launch_params.params.philox_args = gen->philox_cuda_state(counter_offset);
    }

    run_fmha_fp16_sm80(launch_params, /*configure=*/false);

    std::vector<at::Tensor> result = {ctx, softmax_lse};
    if (return_softmax) {result.push_back(s);}
    return result;
}


std::vector<at::Tensor>
mha_bwd(const at::Tensor &dout,  // total x num_heads, x head_size
        const at::Tensor &qkv,   // total x num_heads x 3 x head_size, total := \sum_{i=0}^{b} s_i
        const at::Tensor &out,   // total x num_heads x head_size
        at::Tensor &softmax,     // b x h x s x s softmax and dmask - will be overwritten with dP
        const at::Tensor &softmax_lse,     // b x h x s softmax logsumexp
        const at::Tensor &cu_seqlens,  // b+1
        const float p_dropout,         // probability to drop
        const float softmax_scale,
        const int max_seq_len,          // max sequence length to choose the kernel
        const bool zero_tensors,
        const bool is_causal,
        c10::optional<at::Generator> gen_
) {
    auto dprops = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(dprops->major == 8 && dprops->minor >= 0);
    auto launch = &run_fmha_dgrad_fp16_sm80;

    bool is_dropout = p_dropout > 0.0;
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK(qkv.dtype() == torch::kFloat16);
    TORCH_CHECK(dout.dtype() == torch::kFloat16);
    TORCH_CHECK(softmax.dtype() == torch::kFloat16);
    TORCH_CHECK(cu_seqlens.dtype() == torch::kInt32);

    TORCH_CHECK(qkv.is_cuda());
    TORCH_CHECK(cu_seqlens.is_cuda());

    TORCH_CHECK(qkv.is_contiguous());
    TORCH_CHECK(cu_seqlens.is_contiguous());

    TORCH_CHECK(cu_seqlens.dim() == 1);
    TORCH_CHECK(qkv.dim() == 4);

    const auto sizes = qkv.sizes();

    TORCH_CHECK(sizes[THREE_DIM] == 3);

    const int batch_size = cu_seqlens.numel() - 1;
    const int total = sizes[TOTAL_DIM];
    const int num_heads = sizes[H_DIM];
    const int head_size = sizes[D_DIM];
    TORCH_CHECK(batch_size > 0);
    TORCH_CHECK(head_size == 16 || head_size == 32 || head_size == 64 || head_size == 128);

    // int base_N = head_size == 16 ? 512 : (head_size == 128 ? 128 : 256);
    int base_N = head_size == 128 ? 128 : 256;
    int seq_len = 512;
    if( max_seq_len <= 128 ) {
        seq_len = 128;
    } else if( max_seq_len <= 256 ) {
        seq_len = 256;
    } else {
        seq_len = ((max_seq_len + base_N - 1) / base_N) * base_N;
    }
    bool loop = seq_len > base_N;

    auto dqkv = torch::empty_like(qkv);
    auto opts = qkv.options();
    // auto softmax_lse =
    //     torch::empty({batch_size, num_heads, seq_len}, opts.dtype(at::kFloat));
    auto softmax_d = torch::empty({batch_size, num_heads, seq_len}, opts.dtype(at::kFloat));
    // softmax.zero_();
    // torch::nn::init::ones_(softmax);
    // torch::nn::init::ones_(dqkv);
    at::Tensor dq_tmp;
    if (loop) {
        dq_tmp = torch::empty({total, num_heads, head_size}, opts.dtype(at::kFloat));
    }

    if( zero_tensors ) {
        dqkv.zero_();
        softmax_d.zero_();
        if (loop) { dq_tmp.zero_(); }
    }

    Fused_multihead_attention_fprop_params params;

    set_params(params,
               batch_size,
               seq_len,
               num_heads,
               head_size,
               qkv.data_ptr(),
               cu_seqlens.data_ptr(),
               out.data_ptr(),
               loop ? dq_tmp.data_ptr() : nullptr,
               dout.data_ptr(),
               softmax.data_ptr(),  // softmax gets overwritten by dP!
               softmax_lse.data_ptr(),
               softmax_d.data_ptr(),
               p_dropout,
               softmax_scale,
               is_causal);

    auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
        gen_, at::cuda::detail::getDefaultCUDAGenerator());

    // We're gonna reset the rng state in Python after this kernel, so the counter offset
    // here doesn't matter at all. We just choose an arbitrary number;
    int64_t counter_offset = 4;

    if( is_dropout ) {
        // See Note [Acquire lock when using random generators]
        std::lock_guard<std::mutex> lock(gen->mutex_);
        params.philox_args = gen->philox_cuda_state(counter_offset);
    }

    Data_type acc_type = DATA_TYPE_FP32;
    params.dqkv_ptr = dqkv.data_ptr();

    launch(params, stream);
    return { dqkv, softmax, softmax_d };
    // std::vector<at::Tensor> result = {dqkv, softmax, softmax_d};
    // if (loop) {
    //   result.push_back(dq_tmp);
    // }
    // return result;
}

std::vector<at::Tensor>
mha_fwd_block(const at::Tensor &qkv,         // total x num_heads x 3 x head_size, total := \sum_{i=0}^{b} s_i
              const at::Tensor &cu_seqlens,  // b+1
              const at::Tensor &blockmask,   // (seqlen / 256, seqlen / 16)
              const float p_dropout,
              const int max_seq_len,
              const float softmax_scale,
              const bool is_causal,
              const bool return_softmax,
              c10::optional<at::Generator> gen_) {

    auto dprops = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(dprops->major == 8 && dprops->minor >= 0);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    bool is_dropout = p_dropout > 0.0;
    Launch_params<Fused_multihead_attention_fprop_params> launch_params(dprops, stream, is_dropout, return_softmax);

    bool loop = false;
    int seq_len = 256;
    if( max_seq_len > 256 ) {
        seq_len = ((max_seq_len + 256 - 1) / 256) * 256;
        loop = true;
    }

    TORCH_CHECK(qkv.is_cuda())
    TORCH_CHECK(cu_seqlens.is_cuda())
    TORCH_CHECK(blockmask.is_cuda())

    TORCH_CHECK(qkv.is_contiguous())
    TORCH_CHECK(cu_seqlens.is_contiguous())
    TORCH_CHECK(blockmask.is_contiguous())

    TORCH_CHECK(cu_seqlens.dim() == 1);
    TORCH_CHECK(qkv.dim() == 4);
    TORCH_CHECK(blockmask.dim() == 2);

    const auto sizes = qkv.sizes();

    TORCH_CHECK(sizes[THREE_DIM] == 3);

    const int batch_size = cu_seqlens.numel() - 1;
    const int total = sizes[TOTAL_DIM];
    const int num_heads = sizes[H_DIM];
    const int head_size = sizes[D_DIM];
    TORCH_CHECK(batch_size > 0);
    TORCH_CHECK(head_size == 16 || head_size == 32 || head_size == 64);
    auto opts = qkv.options();

    auto ctx = torch::zeros({ total, num_heads, head_size }, opts);

    at::Tensor o_tmp;
    if (loop) {
        // o_tmp = torch::zeros({total, num_heads, head_size}, opts.dtype(at::kFloat));
        o_tmp = torch::empty({total, num_heads, head_size}, opts.dtype(at::kFloat));
    }

    // auto softmax_lse = torch::full({batch_size, num_heads, seq_len}, -std::numeric_limits<float>::infinity(), opts.dtype(at::kFloat));
    auto softmax_lse = torch::empty({batch_size, num_heads, seq_len}, opts.dtype(at::kFloat));

    at::Tensor s;
    if (return_softmax) {
        s = torch::zeros({ batch_size, num_heads, seq_len, seq_len }, opts);
    }

    auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
        gen_, at::cuda::detail::getDefaultCUDAGenerator());


    set_params(launch_params.params,
               batch_size,
               seq_len,
               num_heads,
               head_size,
               qkv.data_ptr(),
               cu_seqlens.data_ptr(),
               ctx.data_ptr(),
               loop ? o_tmp.data_ptr() : nullptr,
               nullptr,
               return_softmax ? s.data_ptr() : nullptr,
               softmax_lse.data_ptr(),
               nullptr,
               p_dropout,
               softmax_scale,
               is_causal);
    launch_params.params.blockmask = static_cast<int *>(blockmask.data_ptr());

    run_fmha_block_fp16_sm80(launch_params, /*configure=*/ true);
    // number of times random will be generated per thread, to offset philox counter in thc random
    // state
    int64_t counter_offset = launch_params.elts_per_thread;
    at::PhiloxCudaState rng_engine_inputs;

    if( is_dropout ) {
        // See Note [Acquire lock when using random generators]
        std::lock_guard<std::mutex> lock(gen->mutex_);
        launch_params.params.philox_args = gen->philox_cuda_state(counter_offset);
    }

    run_fmha_block_fp16_sm80(launch_params, /*configure=*/false);

    std::vector<at::Tensor> result = {ctx, softmax_lse};
    if (return_softmax) {result.push_back(s);}
    return result;
}

std::vector<at::Tensor>
mha_bwd_block(const at::Tensor &dout,  // total x num_heads, x head_size
              const at::Tensor &qkv,   // total x num_heads x 3 x head_size, total := \sum_{i=0}^{b} s_i
              const at::Tensor &out,   // total x num_heads x head_size
              at::Tensor &softmax,     // b x h x s x s softmax and dmask - will be overwritten with dP
              const at::Tensor &softmax_lse,     // b x h x s softmax logsumexp
              const at::Tensor &cu_seqlens,  // b+1
              const at::Tensor &blockmask,   // (seqlen / 256, seqlen / 16)
              const float p_dropout,         // probability to drop
              const float softmax_scale,
              const int max_seq_len,          // max sequence length to choose the kernel
              const bool is_causal,
              c10::optional<at::Generator> gen_
) {
    auto dprops = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(dprops->major == 8 && dprops->minor >= 0);
    bool loop = false;
    int seq_len = 256;
    auto launch = &run_fmha_block_dgrad_fp16_sm80;
    if (max_seq_len > 256) {
        seq_len = ((max_seq_len + 256 - 1) / 256) * 256;
        loop = true;
    }

    bool is_dropout = p_dropout > 0.0;
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK(qkv.dtype() == torch::kFloat16);
    TORCH_CHECK(dout.dtype() == torch::kFloat16);
    TORCH_CHECK(softmax.dtype() == torch::kFloat16);
    TORCH_CHECK(cu_seqlens.dtype() == torch::kInt32);
    TORCH_CHECK(blockmask.dtype() == torch::kInt32);

    TORCH_CHECK(qkv.is_cuda());
    TORCH_CHECK(cu_seqlens.is_cuda());
    TORCH_CHECK(blockmask.is_cuda());

    TORCH_CHECK(qkv.is_contiguous());
    TORCH_CHECK(cu_seqlens.is_contiguous());
    TORCH_CHECK(blockmask.is_contiguous());

    TORCH_CHECK(cu_seqlens.dim() == 1);
    TORCH_CHECK(qkv.dim() == 4);
    TORCH_CHECK(blockmask.dim() == 2);

    const auto sizes = qkv.sizes();

    TORCH_CHECK(sizes[THREE_DIM] == 3);

    const int batch_size = cu_seqlens.numel() - 1;
    const int total = sizes[TOTAL_DIM];
    const int num_heads = sizes[H_DIM];
    const int head_size = sizes[D_DIM];
    TORCH_CHECK(batch_size > 0);
    TORCH_CHECK(head_size == 16 || head_size == 32 || head_size == 64);

    auto dqkv = torch::zeros_like(qkv);
    auto opts = qkv.options();
    auto softmax_d = torch::empty({batch_size, num_heads, seq_len}, opts.dtype(at::kFloat));
    at::Tensor dq_tmp;
    if (loop) {
        // dq_tmp = torch::zeros({total, num_heads, head_size}, opts.dtype(at::kFloat));
        dq_tmp = torch::empty({total, num_heads, head_size}, opts.dtype(at::kFloat));
    }

    Fused_multihead_attention_fprop_params params;

    set_params(params,
               batch_size,
               seq_len,
               num_heads,
               head_size,
               qkv.data_ptr(),
               cu_seqlens.data_ptr(),
               out.data_ptr(),
               loop ? dq_tmp.data_ptr() : nullptr,
               dout.data_ptr(),
               softmax.data_ptr(),  // softmax gets overwritten by dP!
               softmax_lse.data_ptr(),
               softmax_d.data_ptr(),
               p_dropout,
               softmax_scale,
               is_causal);
    params.blockmask = static_cast<int *>(blockmask.data_ptr());

    auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
        gen_, at::cuda::detail::getDefaultCUDAGenerator());

    // We're gonna reset the rng state in Python after this kernel, so the counter offset
    // here doesn't matter at all. We just choose an arbitrary number;
    int64_t counter_offset = 4;

    if( is_dropout ) {
        // See Note [Acquire lock when using random generators]
        std::lock_guard<std::mutex> lock(gen->mutex_);
        params.philox_args = gen->philox_cuda_state(counter_offset);
    }

    Data_type acc_type = DATA_TYPE_FP32;
    params.dqkv_ptr = dqkv.data_ptr();

    launch(params, stream);
    return { dqkv, softmax, softmax_d };
    // std::vector<at::Tensor> result = {dqkv, softmax, softmax_d};
    // if (loop) {
    //   result.push_back(dq_tmp);
    // }
    // return result;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Fused Multi-head Self-attention";
    m.def("fwd", &mha_fwd, "Forward pass");
    m.def("bwd", &mha_bwd, "Backward pass");
    m.def("fwd_block", &mha_fwd_block, "Forward pass (blocksparse)");
    m.def("bwd_block", &mha_bwd_block, "Backward pass (blocksparse)");
}
