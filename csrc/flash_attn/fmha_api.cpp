/******************************************************************************
 * Copyright (c) 2022, Tri Dao.
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
#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>

#include "fmha.h"


#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")


void set_params_fprop(FMHA_fprop_params &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t h,
                      const size_t d,
                      // device pointers
                      const at::Tensor q,
                      const at::Tensor k,
                      const at::Tensor v,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      void *o_packed_d,
                      void *o_tmp_d,
                      void *s_d,
                      void *softmax_lse_d,
                      float p_dropout,
                      float softmax_scale,
                      bool is_causal,
                      void *attn_mask,
                      void *attn_bias,
                      int bias_mod_size,
                      int mask_head_mod_size,
                      int mask_seq_mod_size
                      ) {

    Data_type acc_type = DATA_TYPE_FP32;
    Data_type data_type = !(q.dtype() == torch::kBFloat16) ? DATA_TYPE_FP16 : DATA_TYPE_BF16;

    // Reset the parameters
    memset(&params, 0, sizeof(params));

    params.is_bf16 = q.dtype() == torch::kBFloat16;

    // Set the pointers and strides.
    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.v_ptr = v.data_ptr();
    params.q_row_stride_in_elts = q.stride(0);
    params.k_row_stride_in_elts = k.stride(0);
    params.v_row_stride_in_elts = v.stride(0);
    params.q_head_stride_in_elts = q.stride(1);
    params.k_head_stride_in_elts = k.stride(1);
    params.v_head_stride_in_elts = v.stride(1);
    params.o_ptr = o_packed_d;
    params.o_row_stride_in_elts = h * d;
    params.o_head_stride_in_elts = d;
    params.o_tmp_ptr = o_tmp_d;

    params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q_d);
    params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k_d);

    // S = softmax(P)
    params.s_ptr = s_d;
    params.s_stride_in_bytes = get_size_in_bytes(b * h * seqlen_k, data_type);

    // Softmax sum
    params.softmax_lse_ptr = softmax_lse_d;

    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.d = d;

    // attn mask & bias
    params.attn_mask_ptr = attn_mask;
    params.attn_bias_ptr = attn_bias;
    params.bias_mod_size = bias_mod_size;
    params.mask_head_mod_size = mask_head_mod_size;
    params.mask_seq_mod_size = mask_seq_mod_size;

    // Set the different scale values.
    // const float scale_bmm1 = 1.f / sqrtf(d);
    const float scale_bmm1 = softmax_scale;

    params.scale_bmm1f = scale_bmm1;
    set_alpha(params.scale_bmm1, scale_bmm1, data_type);

    // Set this to probability of keeping an element to simplify things.
    params.p_dropout = 1.f - p_dropout;
    // Convert p from float to int so we don't have to convert the random uint to float to compare.
    // [Minor] We want to round down since when we do the comparison we use <= instead of <
    params.p_dropout_in_uint = uint32_t(std::floor(params.p_dropout * 4294967295.0));
    params.p_dropout_in_uint16_t = uint16_t(std::floor(params.p_dropout * 65535.0));
    params.rp_dropout = 1.f / params.p_dropout;
    params.scale_bmm1_rp_dropout = params.rp_dropout * params.scale_bmm1f;
    TORCH_CHECK(p_dropout < 1.f);
    set_alpha(params.scale_dropout, params.rp_dropout, data_type);

    params.is_causal = is_causal;
}

void set_params_dgrad(FMHA_dgrad_params &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t h,
                      const size_t d,
                      // device pointers
                      const at::Tensor q,
                      const at::Tensor k,
                      const at::Tensor v,
                      at::Tensor dq,
                      at::Tensor dk,
                      at::Tensor dv,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      void *o_packed_d,
                      void *dq_tmp_d,
                      void *do_packed_d,
                      void *softmax_lse_d,
                      void *dsoftmax_sum_d,
                      float p_dropout,
                      float softmax_scale,
                      bool is_causal,
                      void *attn_mask,
                      void *attn_bias,
                      void *attn_ds,
                      int bias_mod_size,
                      int mask_head_mod_size,
                      int mask_seq_mod_size) {

    set_params_fprop(params,
                     b, seqlen_q, seqlen_k, h, d,
                     q, k, v,
                     cu_seqlens_q_d,
                     cu_seqlens_k_d,
                     o_packed_d,
                     dq_tmp_d,  // Reusing the o_tmp_ptr variable to store dq_tmp
                     nullptr,
                     softmax_lse_d,
                     p_dropout,
                     softmax_scale,
                     is_causal,
                     attn_mask,
                     attn_bias,
                     bias_mod_size,
                     mask_head_mod_size,
                     mask_seq_mod_size);

    // Set the pointers and strides.
    params.dq_ptr = dq.data_ptr();
    params.dk_ptr = dk.data_ptr();
    params.dv_ptr = dv.data_ptr();
    params.dq_row_stride_in_elts = dq.stride(0);
    params.dk_row_stride_in_elts = dk.stride(0);
    params.dv_row_stride_in_elts = dv.stride(0);
    params.dq_head_stride_in_elts = dq.stride(1);
    params.dk_head_stride_in_elts = dk.stride(1);
    params.dv_head_stride_in_elts = dv.stride(1);
    params.do_ptr = do_packed_d;

    // Softmax sum
    params.dsoftmax_sum = dsoftmax_sum_d;
    params.attn_ds_ptr = attn_ds;
}

std::vector<at::Tensor>
mha_fwd(const at::Tensor &q,         // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
        const at::Tensor &k,         // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
        const at::Tensor &v,         // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
        const at::Tensor &cu_seqlens_q,  // b+1
        const at::Tensor &cu_seqlens_k,  // b+1
        const int max_seqlen_q_,
        const int max_seqlen_k_,
        const float p_dropout,
        const float softmax_scale,
        const bool zero_tensors,
        const bool is_causal,
        const bool return_softmax,
        c10::optional<at::Generator> gen_,
        const c10::optional<at::Tensor> &attn_mask, // attn_mask
        const c10::optional<at::Tensor> &attn_bias // attn bias
        ) {

    auto dprops = at::cuda::getCurrentDeviceProperties();
    bool is_sm75 = dprops->major == 7 && dprops->minor == 5;
    bool is_sm80 = dprops->major == 8 && dprops->minor == 0;
    bool is_sm8x = dprops->major == 8 && dprops->minor >= 0;
    TORCH_CHECK(is_sm8x || is_sm75);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    bool is_dropout = p_dropout > 0.0;
    Launch_params<FMHA_fprop_params> launch_params(dprops, stream, is_dropout, return_softmax);

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || (is_sm8x && q_dtype == torch::kBFloat16));
    TORCH_CHECK(k.dtype() == q_dtype);
    TORCH_CHECK(v.dtype() == q_dtype);
    TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32);
    TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32);

    TORCH_CHECK(q.is_cuda());
    TORCH_CHECK(k.is_cuda());
    TORCH_CHECK(v.is_cuda());
    TORCH_CHECK(cu_seqlens_q.is_cuda());
    TORCH_CHECK(cu_seqlens_k.is_cuda());

    TORCH_CHECK(q.stride(-1) == 1);
    TORCH_CHECK(k.stride(-1) == 1);
    TORCH_CHECK(v.stride(-1) == 1);
    TORCH_CHECK(cu_seqlens_q.is_contiguous());
    TORCH_CHECK(cu_seqlens_k.is_contiguous());

    const auto sizes = q.sizes();

    const int batch_size = cu_seqlens_q.numel() - 1;
    const int total_q = sizes[TOTAL_DIM];
    const int num_heads = sizes[H_DIM];
    const int head_size = sizes[D_DIM];
    const int total_k = k.size(TOTAL_DIM);
    TORCH_CHECK(batch_size > 0);
    TORCH_CHECK(head_size == 16 || head_size == 32 || head_size == 64 || head_size == 128);

    CHECK_SHAPE(q, total_q, num_heads, head_size);
    CHECK_SHAPE(k, total_k, num_heads, head_size);
    CHECK_SHAPE(v, total_k, num_heads, head_size);
    CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
    CHECK_SHAPE(cu_seqlens_k, batch_size + 1);

    int bias_mod_size = 0;
    if (attn_bias.has_value()) {
        TORCH_CHECK(attn_bias.value().is_cuda());
        TORCH_CHECK(attn_bias.value().dtype() == q_dtype);
        TORCH_CHECK(attn_bias.value().is_contiguous());

        const auto bias_sizes = attn_bias->sizes();
        // last two dimension
        bias_mod_size = bias_sizes[0];
        TORCH_CHECK(bias_sizes[1] == num_heads);
    }

    int mask_head_mod_size = 0;
    int mask_seq_mod_size = 0;
    if (attn_mask.has_value()) {
        TORCH_CHECK(attn_mask.value().is_cuda());
        TORCH_CHECK(attn_mask.value().dtype() == q_dtype);
        TORCH_CHECK(attn_mask.value().is_contiguous());

        const auto mask_sizes = attn_mask->sizes();
        // last two dimension
        mask_head_mod_size = mask_sizes[1];
        mask_seq_mod_size = mask_sizes[2];
        TORCH_CHECK(mask_sizes[1] == 1 || mask_sizes[1] == num_heads);
        TORCH_CHECK(mask_sizes[2] == 1 || mask_sizes[2] == max_seqlen_q_);
    }

    int blocksize_c = ((head_size == 128 && (is_dropout || !is_sm80)) || (is_sm75 && head_size == 64 && is_dropout)) ? 128 : 256;
    // Need to round max_seqlen_k to multiples of blocksize_c
    int max_seqlen_k = ((max_seqlen_k_ + blocksize_c - 1) / blocksize_c) * blocksize_c;
    if( max_seqlen_k_ <= 128 ) {
        max_seqlen_k = 128;
    } else if( max_seqlen_k_ <= 256 ) {
        max_seqlen_k = 256;
    }
    int max_seqlen_q = ((max_seqlen_q_ + 16 - 1) / 16) * 16;
    bool loop = max_seqlen_k > blocksize_c;

    auto opts = q.options();

    auto o = torch::empty({ total_q, num_heads, head_size }, opts);

    at::Tensor o_tmp;
    if (loop) { o_tmp = torch::empty({total_q, num_heads, head_size}, opts.dtype(at::kFloat)); }

    auto softmax_lse = torch::empty({batch_size, num_heads, max_seqlen_q}, opts.dtype(at::kFloat));
    // auto softmax_lse = torch::full({batch_size, num_heads, max_seqlen_k}, -std::numeric_limits<float>::infinity(), opts.dtype(at::kFloat));

    at::Tensor s;
    if (return_softmax) { s = torch::empty({ batch_size, num_heads, max_seqlen_q, max_seqlen_k }, opts); }

    if( zero_tensors ) {
        o.zero_();
        softmax_lse.fill_(-std::numeric_limits<float>::infinity());
        if (return_softmax) {s.zero_();}
    }

    auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
        gen_, at::cuda::detail::getDefaultCUDAGenerator());

    set_params_fprop(launch_params.params,
                     batch_size,
                     max_seqlen_q,
                     max_seqlen_k,
                     num_heads,
                     head_size,
                     q, k, v,
                     cu_seqlens_q.data_ptr(),
                     cu_seqlens_k.data_ptr(),
                     o.data_ptr(),
                     loop ? o_tmp.data_ptr() : nullptr,
                     return_softmax ? s.data_ptr() : nullptr,
                     softmax_lse.data_ptr(),
                     p_dropout,
                     softmax_scale,
                     is_causal,
                     attn_mask ? attn_mask->data_ptr() : nullptr,
                     attn_bias ? attn_bias->data_ptr() : nullptr,
                     bias_mod_size,
                     mask_head_mod_size,
                     mask_seq_mod_size
                     );

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

    std::vector<at::Tensor> result = {o, softmax_lse};
    if (return_softmax) {result.push_back(s);}
    return result;
}


std::vector<at::Tensor>
mha_bwd(const at::Tensor &dout,  // total_q x num_heads, x head_size
        const at::Tensor &q,   // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
        const at::Tensor &k,   // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
        const at::Tensor &v,   // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
        const at::Tensor &out,   // total_q x num_heads x head_size
        const at::Tensor &softmax_lse_,     // b x h x s softmax logsumexp
        at::Tensor &dq,   // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
        at::Tensor &dk,   // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
        at::Tensor &dv,   // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
        const at::Tensor &cu_seqlens_q,  // b+1
        const at::Tensor &cu_seqlens_k,  // b+1
        const int max_seqlen_q_,
        const int max_seqlen_k_,          // max sequence length to choose the kernel
        const float p_dropout,         // probability to drop
        const float softmax_scale,
        const bool zero_tensors,
        const bool is_causal,
        c10::optional<at::Generator> gen_,
        const c10::optional<at::Tensor> &attn_mask, // attn_mask
        const c10::optional<at::Tensor> &attn_bias // attn bias
) {
    auto dprops = at::cuda::getCurrentDeviceProperties();
    bool is_sm75 = dprops->major == 7 && dprops->minor == 5;
    bool is_sm80 = dprops->major == 8 && dprops->minor == 0;
    bool is_sm8x = dprops->major == 8 && dprops->minor >= 0;
    TORCH_CHECK(is_sm8x || is_sm75);
    auto launch = &run_fmha_dgrad_fp16_sm80;

    bool is_dropout = p_dropout > 0.0;
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || (is_sm8x && q_dtype == torch::kBFloat16));
    TORCH_CHECK(k.dtype() == q_dtype);
    TORCH_CHECK(v.dtype() == q_dtype);
    TORCH_CHECK(out.dtype() == q_dtype);
    TORCH_CHECK(dout.dtype() == q_dtype);
    TORCH_CHECK(dq.dtype() == q_dtype);
    TORCH_CHECK(dk.dtype() == q_dtype);
    TORCH_CHECK(dv.dtype() == q_dtype);
    TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32);
    TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32);

    TORCH_CHECK(q.is_cuda());
    TORCH_CHECK(k.is_cuda());
    TORCH_CHECK(v.is_cuda());
    TORCH_CHECK(out.is_cuda());
    TORCH_CHECK(dout.is_cuda());
    TORCH_CHECK(softmax_lse_.is_cuda());
    TORCH_CHECK(cu_seqlens_q.is_cuda());
    TORCH_CHECK(cu_seqlens_k.is_cuda());

    TORCH_CHECK(q.stride(-1) == 1);
    TORCH_CHECK(k.stride(-1) == 1);
    TORCH_CHECK(v.stride(-1) == 1);
    TORCH_CHECK(out.is_contiguous());
    TORCH_CHECK(dout.is_contiguous());
    TORCH_CHECK(dq.stride(-1) == 1);
    TORCH_CHECK(dk.stride(-1) == 1);
    TORCH_CHECK(dv.stride(-1) == 1);
    TORCH_CHECK(cu_seqlens_q.is_contiguous());
    TORCH_CHECK(cu_seqlens_k.is_contiguous());

    const auto sizes = q.sizes();

    const int batch_size = cu_seqlens_q.numel() - 1;
    const int total_q = sizes[TOTAL_DIM];
    const int num_heads = sizes[H_DIM];
    const int head_size = sizes[D_DIM];
    const int total_k = k.size(TOTAL_DIM);
    TORCH_CHECK(batch_size > 0);
    TORCH_CHECK(head_size == 16 || head_size == 32 || head_size == 64 || head_size == 128);
    if (head_size == 128) {  // TODO: eventually we should support SM86 and SM70 with d=128 as well
        TORCH_CHECK(is_sm80);
    }

    CHECK_SHAPE(q, total_q, num_heads, head_size);
    CHECK_SHAPE(k, total_k, num_heads, head_size);
    CHECK_SHAPE(v, total_k, num_heads, head_size);
    CHECK_SHAPE(out, total_q, num_heads, head_size);
    CHECK_SHAPE(dout, total_q, num_heads, head_size);
    CHECK_SHAPE(dq, total_q, num_heads, head_size);
    CHECK_SHAPE(dk, total_k, num_heads, head_size);
    CHECK_SHAPE(dv, total_k, num_heads, head_size);
    CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
    CHECK_SHAPE(cu_seqlens_k, batch_size + 1);

    int bias_mod_size = 0;
    if (attn_bias.has_value()) {
        TORCH_CHECK(attn_bias.value().is_cuda());
        TORCH_CHECK(attn_bias.value().dtype() == q_dtype);
        TORCH_CHECK(attn_bias.value().is_contiguous());
        // check attn_bias shape
        const auto bias_sizes = attn_bias->sizes();
        // last two dimension
        bias_mod_size = bias_sizes[0];
        TORCH_CHECK(bias_sizes[1] == num_heads);
    }

    int mask_head_mod_size = 0;
    int mask_seq_mod_size = 0;
    if (attn_mask.has_value()) {
        TORCH_CHECK(attn_mask.value().is_cuda());
        TORCH_CHECK(attn_mask.value().dtype() == q_dtype);
        TORCH_CHECK(attn_mask.value().is_contiguous());

        const auto mask_sizes = attn_mask->sizes();
        // last two dimension
        mask_head_mod_size = mask_sizes[1];
        mask_seq_mod_size = mask_sizes[2];
        TORCH_CHECK(mask_sizes[1] == 1 || mask_sizes[1] == num_heads);
        TORCH_CHECK(mask_sizes[2] == 1 || mask_sizes[2] == max_seqlen_q_);
    }

    auto opts = q.options();
    at::Tensor ds;
    if (attn_bias.has_value()) {
        ds = torch::empty({batch_size, num_heads, max_seqlen_q_, max_seqlen_k_}, opts.dtype(q_dtype));
        ds.zero_();
        TORCH_CHECK(ds.is_contiguous());
    }

    int blocksize_c = (head_size == 128 || (is_sm75 && head_size == 64)) ? 128 : 256;
    int max_seqlen_k = ((max_seqlen_k_ + blocksize_c - 1) / blocksize_c) * blocksize_c;
    if( max_seqlen_k_ <= 128 ) {
        max_seqlen_k = 128;
    } else if( max_seqlen_k_ <= 256 ) {
        max_seqlen_k = 256;
    }
    int max_seqlen_q = ((max_seqlen_q_ + 16 - 1) / 16) * 16;
    bool loop = max_seqlen_k > blocksize_c;

    // It's possible the softmax_lse_ from the fwd has a different length since blocksize_c could be different.
    auto softmax_lse = softmax_lse_.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, max_seqlen_q)}).contiguous();

    
    auto softmax_d = torch::empty({batch_size, num_heads, max_seqlen_q}, opts.dtype(at::kFloat));
    at::Tensor dq_tmp;
    if (loop) { dq_tmp = torch::empty({total_q, num_heads, head_size}, opts.dtype(at::kFloat)); }

    if( zero_tensors ) {
        dq.zero_();
        dk.zero_();
        dv.zero_();
        softmax_d.zero_();
    }

    FMHA_dgrad_params params;

    set_params_dgrad(params,
                     batch_size,
                     max_seqlen_q,
                     max_seqlen_k,
                     num_heads,
                     head_size,
                     q, k, v,
                     dq, dk, dv,
                     cu_seqlens_q.data_ptr(),
                     cu_seqlens_k.data_ptr(),
                     out.data_ptr(),
                     loop ? dq_tmp.data_ptr() : nullptr,
                     dout.data_ptr(),
                     softmax_lse.data_ptr(),
                     softmax_d.data_ptr(),
                     p_dropout,
                     softmax_scale,
                     is_causal,
                     attn_mask ? attn_mask->data_ptr() : nullptr,
                     attn_bias ? attn_bias->data_ptr() : nullptr,
                     attn_bias ? ds.data_ptr() : nullptr,
                     bias_mod_size,
                     mask_head_mod_size,
                     mask_seq_mod_size);
                    // used for dbias

    auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
        gen_, at::cuda::detail::getDefaultCUDAGenerator());

    // We're gonna reset the rng state in Python after this kernel, so the counter offset
    // here doesn't matter at all. We just choose an arbitrary number.
    int64_t counter_offset = 4;

    if( is_dropout ) {
        // See Note [Acquire lock when using random generators]
        std::lock_guard<std::mutex> lock(gen->mutex_);
        params.philox_args = gen->philox_cuda_state(counter_offset);
    }

    launch(params, stream);

    std::vector<at::Tensor> result = { softmax_d };
    at::Tensor dbias;
    if (attn_bias.has_value()) {
        // compare block reduce
        auto size = attn_bias->sizes();
        dbias = ds.reshape({ -1, size[0], size[1], size[2], size[3] }).sum({ 0 });
        result.push_back( dbias );
    }
    return result;
}

std::vector<at::Tensor>
mha_fwd_block(const at::Tensor &q,         // total_q x num_heads x head_size, total := \sum_{i=0}^{b} s_i
              const at::Tensor &k,         // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
              const at::Tensor &v,         // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
              const at::Tensor &cu_seqlens_q,  // b+1
              const at::Tensor &cu_seqlens_k,  // b+1
              const at::Tensor &blockmask,   // (seqlen / 256, seqlen / 16)
              const int max_seqlen_q_,
              const int max_seqlen_k_,
              const float p_dropout,
              const float softmax_scale,
              const bool is_causal,
              const bool return_softmax,
              c10::optional<at::Generator> gen_) {

    auto dprops = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(dprops->major == 8 && dprops->minor >= 0);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    bool is_dropout = p_dropout > 0.0;
    Launch_params<FMHA_fprop_params> launch_params(dprops, stream, is_dropout, return_softmax);

    TORCH_CHECK(q.dtype() == torch::kFloat16);
    TORCH_CHECK(k.dtype() == torch::kFloat16);
    TORCH_CHECK(v.dtype() == torch::kFloat16);
    TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32);
    TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32);
    TORCH_CHECK(blockmask.dtype() == torch::kInt32);

    TORCH_CHECK(q.is_cuda());
    TORCH_CHECK(k.is_cuda());
    TORCH_CHECK(v.is_cuda());
    TORCH_CHECK(cu_seqlens_q.is_cuda());
    TORCH_CHECK(cu_seqlens_k.is_cuda());
    TORCH_CHECK(blockmask.is_cuda())

    TORCH_CHECK(q.stride(-1) == 1);
    TORCH_CHECK(k.stride(-1) == 1);
    TORCH_CHECK(v.stride(-1) == 1);
    TORCH_CHECK(cu_seqlens_k.is_contiguous());
    TORCH_CHECK(cu_seqlens_k.is_contiguous());
    TORCH_CHECK(blockmask.is_contiguous())

    const auto sizes = q.sizes();

    const int batch_size = cu_seqlens_q.numel() - 1;
    const int total_q = sizes[TOTAL_DIM];
    const int num_heads = sizes[H_DIM];
    const int head_size = sizes[D_DIM];
    const int total_k = k.size(TOTAL_DIM);
    TORCH_CHECK(batch_size > 0);
    TORCH_CHECK(head_size == 16 || head_size == 32 || head_size == 64 || head_size == 128);

    CHECK_SHAPE(q, total_q, num_heads, head_size);
    CHECK_SHAPE(k, total_k, num_heads, head_size);
    CHECK_SHAPE(v, total_k, num_heads, head_size);
    CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
    CHECK_SHAPE(cu_seqlens_k, batch_size + 1);

    int max_seqlen_k = ((max_seqlen_k_ + 256 - 1) / 256) * 256;
    if( max_seqlen_k <= 256 ) {
        max_seqlen_k = 256;
    }
    int max_seqlen_q = ((max_seqlen_q_ + 16 - 1) / 16) * 16;
    bool loop = max_seqlen_k > 256;
    CHECK_SHAPE(blockmask, max_seqlen_k / 256, max_seqlen_q / 16);

    auto opts = q.options();

    auto o = torch::zeros({ total_q, num_heads, head_size }, opts);

    at::Tensor o_tmp;
    if (loop) {
        // o_tmp = torch::zeros({total, num_heads, head_size}, opts.dtype(at::kFloat));
        o_tmp = torch::empty({total_q, num_heads, head_size}, opts.dtype(at::kFloat));
    }

    // auto softmax_lse = torch::full({batch_size, num_heads, max_seqlen_k}, -std::numeric_limits<float>::infinity(), opts.dtype(at::kFloat));
    auto softmax_lse = torch::empty({batch_size, num_heads, max_seqlen_q}, opts.dtype(at::kFloat));

    at::Tensor s;
    if (return_softmax) {
        s = torch::zeros({ batch_size, num_heads, max_seqlen_q, max_seqlen_k }, opts);
    }

    auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
        gen_, at::cuda::detail::getDefaultCUDAGenerator());

    set_params_fprop(launch_params.params,
                     batch_size,
                     max_seqlen_q,
                     max_seqlen_k,
                     num_heads,
                     head_size,
                     q, k, v,
                     cu_seqlens_q.data_ptr(),
                     cu_seqlens_k.data_ptr(),
                     o.data_ptr(),
                     loop ? o_tmp.data_ptr() : nullptr,
                     return_softmax ? s.data_ptr() : nullptr,
                     softmax_lse.data_ptr(),
                     p_dropout,
                     softmax_scale,
                     is_causal,
                     nullptr, // attn_mask
                     nullptr, // attn_bias
                     0, // bias_mod_size
                     0, // mask_head_mod_size
                     0); // mask_seq_mod_size
                    //  TODO: add mask / bias
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

    std::vector<at::Tensor> result = {o, softmax_lse};
    if (return_softmax) {result.push_back(s);}
    return result;
}

std::vector<at::Tensor>
mha_bwd_block(const at::Tensor &dout,  // total x num_heads, x head_size
              const at::Tensor &q,   // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
              const at::Tensor &k,   // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
              const at::Tensor &v,   // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
              const at::Tensor &out,   // total_q x num_heads x head_size
              const at::Tensor &softmax_lse_,     // b x h x s softmax logsumexp
              at::Tensor &dq,   // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
              at::Tensor &dk,   // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
              at::Tensor &dv,   // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
              const at::Tensor &cu_seqlens_q,  // b+1
              const at::Tensor &cu_seqlens_k,  // b+1
              const at::Tensor &blockmask,   // (seqlen / 256, seqlen / 16)
              const int max_seqlen_q_,
              const int max_seqlen_k_,          // max sequence length to choose the kernel
              const float p_dropout,         // probability to drop
              const float softmax_scale,
              const bool is_causal,
              c10::optional<at::Generator> gen_
) {
    auto dprops = at::cuda::getCurrentDeviceProperties();
    bool is_sm80 = dprops->major == 8 && dprops->minor == 0;
    bool is_sm8x = dprops->major == 8 && dprops->minor >= 0;
    TORCH_CHECK(dprops->major == 8 && dprops->minor >= 0);
    auto launch = &run_fmha_block_dgrad_fp16_sm80;

    bool is_dropout = p_dropout > 0.0;
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK(q.dtype() == torch::kFloat16);
    TORCH_CHECK(k.dtype() == torch::kFloat16);
    TORCH_CHECK(v.dtype() == torch::kFloat16);
    TORCH_CHECK(out.dtype() == torch::kFloat16);
    TORCH_CHECK(dout.dtype() == torch::kFloat16);
    TORCH_CHECK(dq.dtype() == torch::kFloat16);
    TORCH_CHECK(dk.dtype() == torch::kFloat16);
    TORCH_CHECK(dv.dtype() == torch::kFloat16);
    TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32);
    TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32);
    TORCH_CHECK(blockmask.dtype() == torch::kInt32);

    TORCH_CHECK(q.is_cuda());
    TORCH_CHECK(k.is_cuda());
    TORCH_CHECK(v.is_cuda());
    TORCH_CHECK(out.is_cuda());
    TORCH_CHECK(dout.is_cuda());
    TORCH_CHECK(softmax_lse_.is_cuda());
    TORCH_CHECK(cu_seqlens_q.is_cuda());
    TORCH_CHECK(cu_seqlens_k.is_cuda());
    TORCH_CHECK(blockmask.is_cuda());

    TORCH_CHECK(q.stride(-1) == 1);
    TORCH_CHECK(k.stride(-1) == 1);
    TORCH_CHECK(v.stride(-1) == 1);
    TORCH_CHECK(out.is_contiguous());
    TORCH_CHECK(dout.is_contiguous());
    TORCH_CHECK(dq.stride(-1) == 1);
    TORCH_CHECK(dk.stride(-1) == 1);
    TORCH_CHECK(dv.stride(-1) == 1);
    TORCH_CHECK(cu_seqlens_q.is_contiguous());
    TORCH_CHECK(cu_seqlens_k.is_contiguous());
    TORCH_CHECK(blockmask.is_contiguous());

    const auto sizes = q.sizes();

    const int batch_size = cu_seqlens_q.numel() - 1;
    const int total_q = sizes[TOTAL_DIM];
    const int num_heads = sizes[H_DIM];
    const int head_size = sizes[D_DIM];
    const int total_k = k.size(TOTAL_DIM);
    TORCH_CHECK(batch_size > 0);
    TORCH_CHECK(head_size == 16 || head_size == 32 || head_size == 64 || head_size == 128);
    if (head_size == 128) {  // TODO: eventually we should support SM86 and SM70 with d=128 as well
        TORCH_CHECK(is_sm80);
    }

    CHECK_SHAPE(q, total_q, num_heads, head_size);
    CHECK_SHAPE(k, total_k, num_heads, head_size);
    CHECK_SHAPE(v, total_k, num_heads, head_size);
    CHECK_SHAPE(out, total_q, num_heads, head_size);
    CHECK_SHAPE(dout, total_q, num_heads, head_size);
    CHECK_SHAPE(dq, total_q, num_heads, head_size);
    CHECK_SHAPE(dk, total_k, num_heads, head_size);
    CHECK_SHAPE(dv, total_k, num_heads, head_size);
    CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
    CHECK_SHAPE(cu_seqlens_k, batch_size + 1);

    int max_seqlen_k = ((max_seqlen_k_ + 256 - 1) / 256) * 256;
    if( max_seqlen_k <= 256 ) {
        max_seqlen_k = 256;
    }
    int max_seqlen_q = ((max_seqlen_q_ + 16 - 1) / 16) * 16;
    bool loop = max_seqlen_k > 256;
    CHECK_SHAPE(blockmask, max_seqlen_k / 256, max_seqlen_q / 16);

    // It's possible the softmax_lse_ from the fwd has a different length since blocksize_c could be different.
    auto softmax_lse = softmax_lse_.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, max_seqlen_q)}).contiguous();

    auto opts = q.options();
    auto softmax_d = torch::empty({batch_size, num_heads, max_seqlen_q}, opts.dtype(at::kFloat));
    at::Tensor dq_tmp;
    if (loop) {
        // dq_tmp = torch::zeros({total, num_heads, head_size}, opts.dtype(at::kFloat));
        dq_tmp = torch::empty({total_q, num_heads, head_size}, opts.dtype(at::kFloat));
    }

    FMHA_dgrad_params params;

    set_params_dgrad(params,
                     batch_size,
                     max_seqlen_q,
                     max_seqlen_k,
                     num_heads,
                     head_size,
                     q, k, v,
                     dq, dk, dv,
                     cu_seqlens_q.data_ptr(),
                     cu_seqlens_k.data_ptr(),
                     out.data_ptr(),
                     loop ? dq_tmp.data_ptr() : nullptr,
                     dout.data_ptr(),
                     softmax_lse.data_ptr(),
                     softmax_d.data_ptr(),
                     p_dropout,
                     softmax_scale,
                     is_causal,
                     nullptr, // attn_mask
                     nullptr, // attn_bias
                     nullptr, // attn_ds
                     0, // bias_mod_size
                     0, // mask_head_mod_size
                     0); // mask_seq_mod_size
                    //  TODO: add support bias / mask
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

    launch(params, stream);
    return { dq, dk, dv, softmax_d };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Fused Multi-head Self-attention";
    m.def("fwd", &mha_fwd, "Forward pass");
    m.def("bwd", &mha_bwd, "Backward pass");
    m.def("fwd_block", &mha_fwd_block, "Forward pass (blocksparse)");
    m.def("bwd_block", &mha_bwd_block, "Backward pass (blocksparse)");
}