// BSD 3 Clause
// Copyright 2023 Advanced Micro Devices, Inc.
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "fmha.h"

#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")


void set_params_fprop(FmhaFpropParams &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t h,
                      const size_t d,
                      // device pointers
                      const at::Tensor& q,
                      const at::Tensor& k,
                      const at::Tensor& v,
                      at::Tensor& out,
                      const at::Tensor& cu_seqlens_q,
                      const at::Tensor& cu_seqlens_k,
                      void *o_tmp_d,
                      void *s_d,
                      void *softmax_lse_d,
                      float p_dropout,
                      float softmax_scale,
                      bool is_causal,
                      bool is_deterministic) {

    auto acc_type = torch::kFloat32;
    auto data_type = q.dtype();

    // Reset the parameters
    memset(&params, 0, sizeof(params));

    params.is_bf16 = (q.dtype() == at::kBFloat16);

    // S = softmax(P)     //TO DO
    // params.s_ptr = s_d;
    // params.s_stride_in_bytes = get_size_in_bytes(b * h * seqlen_k, data_type);

    // Set the dimensions.
    params.b = b;                 // batch_size
    params.h = h;                 // num_heads
    params.seqlen_q = seqlen_q;   // seqlen q
    params.seqlen_k = seqlen_k;   // seqlen k
    params.d = d;                 // head_dim
    if(cu_seqlens_q.device().type() == c10::kCUDA){
        params.host_seqlens_q = std::vector<int>(params.b+1);
        params.host_seqlens_k = std::vector<int>(params.b+1);
        FMHA_CHECK_HIP(hipMemcpy(params.host_seqlens_q.data(), cu_seqlens_q.data_ptr(), (params.b+1)*sizeof(int), hipMemcpyDeviceToHost));
        FMHA_CHECK_HIP(hipMemcpy(params.host_seqlens_k.data(), cu_seqlens_k.data_ptr(), (params.b+1)*sizeof(int), hipMemcpyDeviceToHost));
    }else{
        params.host_seqlens_q = std::vector<int>(static_cast<int*>(cu_seqlens_q.data_ptr()), static_cast<int*>(cu_seqlens_q.data_ptr())+params.b+1);
        params.host_seqlens_k = std::vector<int>(static_cast<int*>(cu_seqlens_k.data_ptr()), static_cast<int*>(cu_seqlens_k.data_ptr())+params.b+1);
    }

    char* q_ptr = reinterpret_cast<char*>(q.data_ptr());
    char* k_ptr = reinterpret_cast<char*>(k.data_ptr());
    char* v_ptr = reinterpret_cast<char*>(v.data_ptr());

    char* out_ptr = reinterpret_cast<char*>(out.data_ptr());
    char* lse_ptr = reinterpret_cast<char*>(softmax_lse_d);
    char* s_ptr = reinterpret_cast<char*>(s_d);

    for (int i = 0; i < b; i++){
        int temp_seqlen_q = params.host_seqlens_q[i+1] - params.host_seqlens_q[i];
        int temp_q_stride = get_size_in_bytes(d * h * temp_seqlen_q, data_type);
        int temp_seqlen_k = params.host_seqlens_k[i+1] - params.host_seqlens_k[i];
        int temp_k_stride = get_size_in_bytes(d * h * temp_seqlen_k, data_type);
        if(q.is_contiguous()){
            params.q_ptr.push_back(reinterpret_cast<void*>(q_ptr));
            q_ptr = q_ptr + temp_q_stride;
        }else{
            auto q_each_tmp = q.index({torch::indexing::Slice(params.host_seqlens_q[i], params.host_seqlens_q[i+1])}).contiguous();
            params.q_tensors.push_back(q_each_tmp);
            params.q_ptr.push_back(reinterpret_cast<void*>(q_each_tmp.data_ptr()));          
        }
        if(k.is_contiguous()){
            params.k_ptr.push_back(reinterpret_cast<void*>(k_ptr));
            k_ptr = k_ptr + temp_k_stride;
        }else{
            auto k_each_tmp = k.index({torch::indexing::Slice(params.host_seqlens_k[i], params.host_seqlens_k[i+1])}).contiguous();
            params.k_tensors.push_back(k_each_tmp);
            params.k_ptr.push_back(reinterpret_cast<void*>(k_each_tmp.data_ptr()));
        }

        if(v.is_contiguous()){
            params.v_ptr.push_back(reinterpret_cast<void*>(v_ptr));     
            v_ptr = v_ptr + temp_k_stride;
        }else{
            auto v_each_tmp = v.index({torch::indexing::Slice(params.host_seqlens_k[i], params.host_seqlens_k[i+1])}).contiguous();
            params.v_tensors.push_back(v_each_tmp);
            params.v_ptr.push_back(reinterpret_cast<void*>(v_each_tmp.data_ptr()));
        }
        
        params.o_ptr.push_back(reinterpret_cast<void*>(out_ptr));
        out_ptr = out_ptr + temp_q_stride;

        params.softmax_lse_ptr.push_back(reinterpret_cast<void*>(lse_ptr));
        int temp_lse_stride = get_size_in_bytes(h * seqlen_q, acc_type);
        lse_ptr = lse_ptr + temp_lse_stride;

        if(s_d){
            params.s_ptr.push_back(reinterpret_cast<void*>(s_ptr + i * h * seqlen_q * seqlen_k * sizeof(int)));
        }
        else{
            params.s_ptr.push_back(nullptr);
        }
    }

    // Set the different scale values.
    // const float scale_bmm1 = 1.f / sqrtf(d);
    params.scale_bmm1f = softmax_scale;

    // Set this to probability of keeping an element to simplify things.
    params.p_dropout = p_dropout;
    params.is_causal = is_causal;
    params.is_deterministic = is_deterministic;
}

void set_params_dgrad(FmhaDgradParams &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t h,
                      const size_t d,
                      // device pointers
                      const at::Tensor& q,
                      const at::Tensor& k,
                      const at::Tensor& v,
                      const at::Tensor& y,
                      const at::Tensor& ygrad,
                      at::Tensor &dq,
                      at::Tensor &dk,
                      at::Tensor &dv,
                      const at::Tensor& cu_seqlens_q,
                      const at::Tensor& cu_seqlens_k,
                      void *s_d,
                      void *softmax_lse_d,
                      float p_dropout,
                      float softmax_scale,
                      bool is_causal,
                      bool is_deterministic,
                      bool is_performance_mode) {

    auto acc_type = torch::kFloat32;
    auto data_type = q.dtype();

    // Reset the parameters
    memset(&params, 0, sizeof(params));

    params.is_bf16 = q.dtype() == at::kBFloat16;

    // params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q_d);
    // params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k_d);

    // S = softmax(P)
    // params.s_ptr = s_d;
    // params.s_stride_in_bytes = get_size_in_bytes(b * h * seqlen_k, data_type);

    // Softmax sum
    // params.softmax_lse_ptr = softmax_lse_d;

    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.d = d;
    if(cu_seqlens_q.device().type()==c10::kCUDA){
        params.host_seqlens_q = std::vector<int>(params.b+1);
        params.host_seqlens_k = std::vector<int>(params.b+1);

        FMHA_CHECK_HIP(hipMemcpy(params.host_seqlens_q.data(), cu_seqlens_q.data_ptr(), (params.b+1)*sizeof(int), hipMemcpyDeviceToHost));
        FMHA_CHECK_HIP(hipMemcpy(params.host_seqlens_k.data(), cu_seqlens_k.data_ptr(), (params.b+1)*sizeof(int), hipMemcpyDeviceToHost));
    }else{
        params.host_seqlens_q = std::vector<int>(static_cast<int*>(cu_seqlens_q.data_ptr()), static_cast<int*>(cu_seqlens_q.data_ptr())+params.b+1);
        params.host_seqlens_k = std::vector<int>(static_cast<int*>(cu_seqlens_k.data_ptr()), static_cast<int*>(cu_seqlens_k.data_ptr())+params.b+1);
    }

    char* q_ptr = reinterpret_cast<char*>(q.data_ptr());
    char* k_ptr = reinterpret_cast<char*>(k.data_ptr());
    char* v_ptr = reinterpret_cast<char*>(v.data_ptr());

    char* dq_ptr = reinterpret_cast<char*>(dq.data_ptr());
    char* dk_ptr = reinterpret_cast<char*>(dk.data_ptr());
    char* dv_ptr = reinterpret_cast<char*>(dv.data_ptr());

    char* y_ptr = reinterpret_cast<char*>(y.data_ptr());
    char* lse_ptr = reinterpret_cast<char*>(softmax_lse_d);
    char* ygrad_ptr = reinterpret_cast<char*>(ygrad.data_ptr());
    
    for (int i = 0; i < b; i++){
        int temp_seqlen_q = params.host_seqlens_q[i+1] - params.host_seqlens_q[i];
        int temp_q_stride = get_size_in_bytes(d * h * temp_seqlen_q, data_type);
        int temp_dq_stride = get_size_in_bytes(d * h * temp_seqlen_q, dq.dtype());
        int temp_seqlen_k = params.host_seqlens_k[i+1] - params.host_seqlens_k[i];
        int temp_k_stride = get_size_in_bytes(d * h * temp_seqlen_k, data_type);
        int temp_dk_stride = get_size_in_bytes(d * h * temp_seqlen_k, dk.dtype());
        if(q.is_contiguous()){
            params.q_ptr.push_back(reinterpret_cast<void*>(q_ptr));
            params.qgrad_ptr.push_back(reinterpret_cast<void*>(dq_ptr));
            q_ptr = q_ptr + temp_q_stride;
            dq_ptr = dq_ptr + temp_dq_stride;
        }else{
            auto q_each_tmp = q.index({torch::indexing::Slice(params.host_seqlens_q[i], params.host_seqlens_q[i+1])}).contiguous();
            auto qgrad_each_tmp = dq.index({torch::indexing::Slice(params.host_seqlens_q[i], params.host_seqlens_q[i+1])}).contiguous();
            params.q_tensors.push_back(q_each_tmp);
            params.qgrad_tensors.push_back(qgrad_each_tmp);
            params.q_ptr.push_back(reinterpret_cast<const void*>(q_each_tmp.data_ptr()));
            params.qgrad_ptr.push_back(reinterpret_cast<void*>(qgrad_each_tmp.data_ptr()));
        }
        if(k.is_contiguous()){
            params.k_ptr.push_back(reinterpret_cast<void*>(k_ptr));
            params.kgrad_ptr.push_back(reinterpret_cast<void*>(dk_ptr));
            k_ptr = k_ptr + temp_k_stride;
            dk_ptr = dk_ptr + temp_dk_stride;
        }else{
            auto k_each_tmp = k.index({torch::indexing::Slice(params.host_seqlens_k[i], params.host_seqlens_k[i+1])}).contiguous();
            auto kgrad_each_tmp = dk.index({torch::indexing::Slice(params.host_seqlens_k[i], params.host_seqlens_k[i+1])}).contiguous();
            params.k_tensors.push_back(k_each_tmp);
            params.kgrad_tensors.push_back(kgrad_each_tmp);
            params.k_ptr.push_back(reinterpret_cast<const void*>(k_each_tmp.data_ptr()));
            params.kgrad_ptr.push_back(reinterpret_cast<void*>(kgrad_each_tmp.data_ptr()));
        }
        if(v.is_contiguous()){
            params.v_ptr.push_back(reinterpret_cast<void*>(v_ptr)); 
            params.vgrad_ptr.push_back(reinterpret_cast<void*>(dv_ptr));
            v_ptr = v_ptr + temp_k_stride;   
            dv_ptr = dv_ptr + temp_dk_stride;
        }else{
            auto v_each_tmp = v.index({torch::indexing::Slice(params.host_seqlens_k[i], params.host_seqlens_k[i+1])}).contiguous();
            auto vgrad_each_tmp = dv.index({torch::indexing::Slice(params.host_seqlens_k[i], params.host_seqlens_k[i+1])}).contiguous();
            params.v_tensors.push_back(v_each_tmp);
            params.vgrad_tensors.push_back(vgrad_each_tmp);
            params.v_ptr.push_back(reinterpret_cast<const void*>(v_each_tmp.data_ptr()));
            params.vgrad_ptr.push_back(reinterpret_cast<void*>(vgrad_each_tmp.data_ptr()));
        }

        params.z_ptr.push_back(nullptr);
        params.y_ptr.push_back(reinterpret_cast<const void*>(y_ptr));
        params.lse_ptr.push_back(reinterpret_cast<const void*>(lse_ptr));
        params.ygrad_ptr.push_back(reinterpret_cast<const void*>(ygrad_ptr));

        int temp_lse_stride = get_size_in_bytes(h * seqlen_q, acc_type);
        y_ptr += temp_q_stride;
        ygrad_ptr += temp_q_stride;
        lse_ptr += temp_lse_stride;
    }

    // Set the different scale values.
    // const float scale_bmm1 = 1.f / sqrtf(d);
    params.scale_bmm1f = softmax_scale;
    //set_alpha(params.scale_bmm1, scale_bmm1, data_type);

    // Set this to probability of keeping an element to simplify things.
    params.p_dropout = p_dropout;
    params.is_causal = is_causal;
    params.is_deterministic = is_deterministic;
    params.is_performance_mode = is_performance_mode;
}

std::vector<at::Tensor>
mha_fwd(const at::Tensor &q,
        const at::Tensor &k,
        const at::Tensor &v,
        at::Tensor &out,
        const at::Tensor &cu_seqlens_q,
        const at::Tensor &cu_seqlens_k,
        const int max_seqlen_q,
        const int max_seqlen_k,
        const float p_dropout,
        const float softmax_scale,
        const bool zero_tensors,
        const bool is_causal,
        const bool is_deterministic,
        const bool return_softmax, // in rocm ,this will return the random number matrix when doing dropout
        const int num_splits,      // num_splits is not used in rocm
        c10::optional<at::Generator> gen_) {
    auto dprops = at::cuda::getCurrentDeviceProperties();
    auto stream = at::cuda::getCurrentHIPStream().stream();
    bool is_dropout = p_dropout > 0.0;
    LaunchParams<FmhaFpropParams> launch_params(dprops, stream, is_dropout, return_softmax);

    auto q_dtype = q.dtype();

    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16);
    TORCH_CHECK(k.dtype() == q_dtype);
    TORCH_CHECK(v.dtype() == q_dtype);
    TORCH_CHECK(out.dtype() == q_dtype);
    TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32);
    TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32);

    TORCH_CHECK(q.is_cuda());
    TORCH_CHECK(k.is_cuda());
    TORCH_CHECK(v.is_cuda());
    TORCH_CHECK(out.is_cuda());
    // TORCH_CHECK(cu_seqlens_q.is_cuda());
    // TORCH_CHECK(cu_seqlens_k.is_cuda());

    TORCH_CHECK(q.stride(-1) == 1);
    TORCH_CHECK(k.stride(-1) == 1);
    TORCH_CHECK(v.stride(-1) == 1);
    TORCH_CHECK(out.stride(-1) == 1);
    TORCH_CHECK(cu_seqlens_q.is_contiguous());
    TORCH_CHECK(cu_seqlens_k.is_contiguous());

    const auto sizes = q.sizes();

    const int batch_size = cu_seqlens_q.numel() - 1;
    const int total_q = sizes[TOTAL_DIM];
    const int num_heads = sizes[H_DIM];
    const int head_size = sizes[D_DIM];
    const int total_k = k.size(TOTAL_DIM);

    TORCH_CHECK(batch_size > 0);
    TORCH_CHECK((head_size % 8 == 0) && (head_size <= 128));

    CHECK_SHAPE(q, total_q, num_heads, head_size);
    CHECK_SHAPE(k, total_k, num_heads, head_size);
    CHECK_SHAPE(v, total_k, num_heads, head_size);
    CHECK_SHAPE(out, total_q, num_heads, head_size);
    CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
    CHECK_SHAPE(cu_seqlens_k, batch_size + 1);

    at::cuda::HIPGuard device_guard{(char)q.get_device()};
    // bool loop = false;

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    // at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    auto opts = q.options();

    auto softmax_lse = at::empty({batch_size, num_heads, max_seqlen_q}, opts.dtype(at::kFloat));
    // auto softmax_lse = torch::full({batch_size, num_heads, max_seqlen_k}, -std::numeric_limits<float>::infinity(), opts.dtype(at::kFloat));
    at::Tensor s;
    if (return_softmax) { s = torch::empty({ batch_size, num_heads, max_seqlen_q, max_seqlen_k }, opts.dtype(at::kInt)); }
    if (zero_tensors) {
        out.zero_();
        //softmax_lse.zero_();
        softmax_lse.fill_(-std::numeric_limits<float>::infinity());
        if (return_softmax) { s.zero_(); }
    }

    auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
        gen_, at::cuda::detail::getDefaultCUDAGenerator());

    set_params_fprop(launch_params.params,
                     batch_size,
                     max_seqlen_q,
                     max_seqlen_k,
                     num_heads,
                     head_size,
                     q, k, v, out,
                     cu_seqlens_q,
                     cu_seqlens_k,
                     nullptr,
                     return_softmax ? s.data_ptr() : nullptr,
                     //return_softmax ? z_device_buf.GetDeviceBuffer() : nullptr,
                     softmax_lse.data_ptr(),
                     p_dropout,
                     softmax_scale,
                     is_causal,
                     is_deterministic);

    // number of times random will be generated per thread, to offset philox counter in thc random
    // state
    // We use a custom RNG that increases the offset by batch_size * nheads * 32.
    

    // at::PhiloxCudaState rng_engine_inputs;

    if( is_dropout ) {
        // See Note [Acquire lock when using random generators]
        int64_t counter_offset = launch_params.params.b * launch_params.params.h * 32;
        std::lock_guard<std::mutex> lock(gen->mutex_);
        launch_params.params.philox_args = gen->philox_cuda_state(counter_offset);
    }

    run_fmha_fp16_bf16_gfx90a(launch_params);

    std::vector<at::Tensor> result = {softmax_lse};

    if (return_softmax) {
        result.push_back(s);
    }
    return result;
}


std::vector<at::Tensor>
mha_bwd(const at::Tensor &dout,  // total_q x num_heads, x head_size
        const at::Tensor &q,   // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
        const at::Tensor &k,   // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
        const at::Tensor &v,   // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
        const at::Tensor &out,   // total_q x num_heads x head_size
        const at::Tensor &softmax_lse,     // b x h x s softmax logsumexp
        at::Tensor &dq,   // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
        at::Tensor &dk,   // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
        at::Tensor &dv,   // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
        const at::Tensor &cu_seqlens_q,  // b+1
        const at::Tensor &cu_seqlens_k,  // b+1
        const int max_seqlen_q,
        const int max_seqlen_k,          // max sequence length to choose the kernel
        const float p_dropout,         // probability to drop
        const float softmax_scale,
        const bool zero_tensors,
        const bool is_causal,
        const bool is_deterministic,
        const bool is_performance_mode,
        const int num_splits,
        c10::optional<at::Generator> gen_
) {
    auto dprops = at::cuda::getCurrentDeviceProperties();

    bool is_dropout = p_dropout > 0.0;
    auto stream = at::cuda::getCurrentHIPStream().stream();
    LaunchParams<FmhaDgradParams> launch_params(dprops, stream, is_dropout, false);

    auto q_dtype = q.dtype();

    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16);
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
    TORCH_CHECK(softmax_lse.is_cuda());
    // TORCH_CHECK(cu_seqlens_q.is_cuda());
    // TORCH_CHECK(cu_seqlens_k.is_cuda());

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
    TORCH_CHECK((head_size % 8 == 0) && (head_size <= 128));

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

    // int blocksize_c = (head_size > 64 || (head_size > 32)) ? 128 : 256;
    at::cuda::HIPGuard device_guard{(char)q.get_device()};
    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    // at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    // It's possible the softmax_lse_ from the fwd has a different length since blocksize_c could be different.
    // auto softmax_lse = softmax_lse_.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, max_seqlen_q)}).contiguous();

    // at::Tensor softmax_d = at::empty(dq.sizes(), dq.options()).contiguous();
    at::Tensor softmax_d;

    if (zero_tensors) {
        dq.zero_();
        dk.zero_();
        dv.zero_();
        // softmax_d.zero_();
    }
    
    auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
        gen_, at::cuda::detail::getDefaultCUDAGenerator());

    if(!is_performance_mode){
        at::Tensor dq_tmp = at::empty(dq.sizes(), dq.options().dtype(at::kFloat)).contiguous();
        at::Tensor dk_tmp = at::empty(dk.sizes(), dk.options().dtype(at::kFloat)).contiguous();
        at::Tensor dv_tmp = at::empty(dv.sizes(), dv.options().dtype(at::kFloat)).contiguous();
        dq_tmp.zero_();
        dk_tmp.zero_();
        dv_tmp.zero_();
        set_params_dgrad(launch_params.params,
                        batch_size,
                        max_seqlen_q,
                        max_seqlen_k,
                        num_heads,
                        head_size,
                        q, k, v, out,
                        dout, dq_tmp, dk_tmp, dv_tmp,
                        cu_seqlens_q,
                        cu_seqlens_k,
                        nullptr,
                        softmax_lse.data_ptr(),
                        p_dropout,
                        softmax_scale,
                        is_causal,
                        is_deterministic,
                        is_performance_mode);
        
        if( is_dropout ) {
            // See Note [Acquire lock when using random generators]
            int64_t counter_offset = launch_params.params.b * launch_params.params.h * 32;
            std::lock_guard<std::mutex> lock(gen->mutex_);
            launch_params.params.philox_args = gen->philox_cuda_state(counter_offset);
        }

        run_fmha_dgrad_fp16_bf16_gfx90a(launch_params);
        if(!q.is_contiguous()){
            dq_tmp.copy_(torch::cat(launch_params.params.qgrad_tensors, 0).contiguous(), true);
        }
        if(!k.is_contiguous()){
            dk_tmp.copy_(torch::cat(launch_params.params.kgrad_tensors, 0).contiguous(), true);
        }
        if(!v.is_contiguous()){
            dv_tmp.copy_(torch::cat(launch_params.params.vgrad_tensors, 0).contiguous(), true);
        }

        dq.copy_(dq_tmp, true);
        dk.copy_(dk_tmp, true);
        dv.copy_(dv_tmp, true);
    }else{
        set_params_dgrad(launch_params.params,
                         batch_size,
                         max_seqlen_q,
                         max_seqlen_k,
                         num_heads,
                         head_size,
                         q, k, v, out,
                         dout, dq, dk, dv,
                         cu_seqlens_q,
                         cu_seqlens_k,
                         nullptr,
                         softmax_lse.data_ptr(),
                         p_dropout,
                         softmax_scale,
                         is_causal,
                         is_deterministic,
                         is_performance_mode);
        
        if( is_dropout ) {
            // See Note [Acquire lock when using random generators]
            int64_t counter_offset = launch_params.params.b * launch_params.params.h * 32;
            std::lock_guard<std::mutex> lock(gen->mutex_);
            launch_params.params.philox_args = gen->philox_cuda_state(counter_offset);
        }

        run_fmha_dgrad_fp16_bf16_gfx90a(launch_params);

        if(!q.is_contiguous()){
            dq.copy_(torch::cat(launch_params.params.qgrad_tensors, 0), true);
        }
        if(!k.is_contiguous()){
            dk.copy_(torch::cat(launch_params.params.kgrad_tensors, 0), true);
        }
        if(!v.is_contiguous()){
            dv.copy_(torch::cat(launch_params.params.vgrad_tensors, 0), true);
        }
    }
    return { dq, dk, dv, softmax_d };
}


#ifdef BUILD_PYTHON_PACKAGE
    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.doc() = "Fused Multi-head Self-attention";
        m.def("fwd", &mha_fwd, "Forward pass");
        m.def("bwd", &mha_bwd, "Backward pass");
        // m.def("fwd_block", &mha_fwd_block, "Forward pass (blocksparse)");
        // m.def("bwd_block", &mha_bwd_block, "Backward pass (blocksparse)");
    }
#endif

//main function to test with the API
bool fwd_test(bool do_verification){
    int batch_size = 64;
    int nheads = 16;
    int seqlen = 256;
    int n = 1024;
    int d = n / nheads; //head_size//64

    //initialize the tensors
    at::Tensor q_host = at::rand({batch_size*seqlen, nheads, d}, torch::kFloat16);//torch::kBFloat16;at::kHalf
    at::Tensor k_host = at::rand({batch_size*seqlen, nheads, d}, torch::kFloat16);
    at::Tensor v_host = at::rand({batch_size*seqlen, nheads, d}, torch::kFloat16);

    at::Tensor q = q_host.to(at::kCUDA);
    at::Tensor k = k_host.to(at::kCUDA);
    at::Tensor v = v_host.to(at::kCUDA);

    //initialize the output tensor
    at::Tensor out_host = at::empty({batch_size*seqlen, nheads, d}, torch::kFloat16);
    at::Tensor out = out_host.to(at::kCUDA);

    //initialize seqlens vector (size is b+1)
    std::vector<int> cu_seqlens_q_vec;
    std::vector<int> cu_seqlens_k_vec;

    for (int i = 0 ; i < batch_size + 1; i++){
      cu_seqlens_q_vec.push_back(i * seqlen);
      cu_seqlens_k_vec.push_back(i * seqlen);
    }

    at::TensorOptions opts = at::TensorOptions().dtype(at::kInt);
    at::Tensor cu_seqlens_q = at::from_blob(cu_seqlens_q_vec.data(),{batch_size + 1},opts).clone().to(at::kCUDA);
    at::Tensor cu_seqlens_k = at::from_blob(cu_seqlens_k_vec.data(),{batch_size + 1},opts).clone().to(at::kCUDA);

    int max_seqlen_q_ = seqlen;
    int max_seqlen_k_ = seqlen;

    //dropout parameters
    float p_drop                    = 0.2;
    float p_dropout                 = 1 - p_drop;
    uint16_t p_dropout_in_16bits    = uint16_t(std::floor(p_dropout * 65535.0));
    float rp_dropout                = 1.0 / p_dropout;
    const unsigned long long seed   = 1;
    const unsigned long long offset = 0;
    
    //other parameters
    float softmax_scale = 0.125;
    bool zero_tensors = true;
    bool is_causal = false;
    bool is_deterministic = true;
    bool return_softmax = true;
    int num_splits = 0;

    c10::optional<at::Generator> gen_ = c10::nullopt;

    auto result =
    mha_fwd(q,
            k,
            v,
            out,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q_,
            max_seqlen_k_,
            p_drop,
            softmax_scale,
            zero_tensors,
            is_causal,
            is_deterministic,
            return_softmax,
            num_splits,
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
        const int G1  = nheads;   // num_heads

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

        at::Tensor out_device_result = out.to(torch::kCPU).view({batch_size, seqlen, nheads, d});
        at::Tensor lse_device_result = result[0].to(torch::kCPU);
        at::Tensor z_device_result = result[1].to(torch::kCPU);

        for(std::size_t i = 0; i < batch_size; i++)
        {
            const auto& a_gs_ms_ks         = a_tensors[i];
            const auto& b0_gs_ns_ks        = b0_tensors[i];
            const auto& b1_gs_os_ns        = b1_tensors[i];
            auto& c_gs_ms_os_device_result = c_tensors[i];
            auto& z_gs_ms_ns_device_result = z_tensors[i];
            auto& lse_gs_ms_device_result = lse_tensors[i];
            //auto& c_gs_ms_os_device_buf    = *c_tensors_device[i];

            //at::Tensor out_device_result = out.to(torch::kCPU).view({batch_size, seqlen, nheads, d});
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
    int batch_size = 2;
    int nheads = 16;
    int seqlen = 256;
    int n = 1024;
    int d = n / nheads; //head_size//64

    //initialize the tensors
    at::Tensor q_host = at::rand({batch_size*seqlen, nheads, d}, torch::kFloat16);//torch::kBFloat16;at::kHalf
    at::Tensor k_host = at::rand({batch_size*seqlen, nheads, d}, torch::kFloat16);
    at::Tensor v_host = at::rand({batch_size*seqlen, nheads, d}, torch::kFloat16);
    at::Tensor y_host = at::empty({batch_size*seqlen, nheads, d}, torch::kFloat16);
    at::Tensor z_host = at::empty({batch_size*nheads, seqlen, seqlen}, torch::kInt32);
    at::Tensor lse_host = at::empty({batch_size, nheads, seqlen}, torch::kFloat32);

    at::Tensor ygrad_host = at::rand({batch_size*seqlen, nheads, d}, torch::kFloat16);
    at::Tensor qgrad_host = at::empty({batch_size*seqlen, nheads, d}, torch::kFloat16);
    at::Tensor kgrad_host = at::empty({batch_size*seqlen, nheads, d}, torch::kFloat16);
    at::Tensor vgrad_host = at::empty({batch_size*seqlen, nheads, d}, torch::kFloat16);

    at::Tensor q = q_host.to(at::kCUDA);
    at::Tensor k = k_host.to(at::kCUDA);
    at::Tensor v = v_host.to(at::kCUDA);
    at::Tensor y = y_host.to(at::kCUDA);
    at::Tensor lse = lse_host.to(at::kCUDA);
    at::Tensor qgrad = qgrad_host.to(at::kCUDA);
    at::Tensor vgrad = vgrad_host.to(at::kCUDA);
    at::Tensor kgrad = kgrad_host.to(at::kCUDA);
    at::Tensor ygrad = ygrad_host.to(at::kCUDA);

    //initialize seqlens vector (size is b+1)
    std::vector<int> cu_seqlens_q_vec;
    std::vector<int> cu_seqlens_k_vec;

    for (int i = 0 ; i < batch_size + 1; i++){
      cu_seqlens_q_vec.push_back(i * seqlen);
      cu_seqlens_k_vec.push_back(i * seqlen);
    }

    at::TensorOptions opts=at::TensorOptions().dtype(at::kInt);
    at::Tensor cu_seqlens_q=at::from_blob(cu_seqlens_q_vec.data(),{batch_size + 1},opts).clone().to(at::kCUDA);
    at::Tensor cu_seqlens_k=at::from_blob(cu_seqlens_k_vec.data(),{batch_size + 1},opts).clone().to(at::kCUDA);

    int max_seqlen_q_ = seqlen;
    int max_seqlen_k_ = seqlen;
    
    //other parameters
    float p_dropout = 0.0;
    float p_dropout2                = 1 - p_dropout;
    uint16_t p_dropout_in_16bits    = uint16_t(std::floor(p_dropout2 * 65535.0));
    float rp_dropout                = 1.0 / p_dropout2;
    const unsigned long long seed   = 1;
    const unsigned long long offset = 0;           
    float softmax_scale = 1/sqrt(d);  
    bool zero_tensors = true;    
    bool is_causal = false;     
    bool is_deterministic = true;
    bool is_performance_mode = true;  
    bool return_softmax = false;  
    int num_splits = 0;    
    c10::optional<at::Generator> gen_ = c10::nullopt;
    lse = mha_fwd(q,   
                  k,   
                  v,   
                  y, 
                  cu_seqlens_q, 
                  cu_seqlens_k, 
                  max_seqlen_q_,
                  max_seqlen_k_,
                  p_dropout,
                  softmax_scale,
                  zero_tensors,
                  is_causal,
                  is_deterministic,
                  return_softmax,
                  num_splits,
                  gen_)[0];
    mha_bwd(ygrad,
            q,   
            k,   
            v,   
            y,
            lse,
            qgrad,
            kgrad,
            vgrad,
            cu_seqlens_q, 
            cu_seqlens_k, 
            max_seqlen_q_,
            max_seqlen_k_,
            p_dropout,
            softmax_scale,
            zero_tensors,
            is_causal,
            is_deterministic,
            is_performance_mode,
            num_splits,
            gen_);
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
        bool input_permute = true;
        bool output_permute = true;
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
        const int G1  = nheads;   // num_heads

        auto a_element_op    = QKVElementOp{};
        auto b0_element_op   = QKVElementOp{};
        auto acc0_element_op = Scale{softmax_scale};
        auto b1_element_op   = QKVElementOp{};
        auto c_element_op    = YElementOp{};
        qgrad_host = qgrad.to(torch::kCPU).view({batch_size, seqlen, nheads, d});
        kgrad_host = kgrad.to(torch::kCPU).view({batch_size, seqlen, nheads, d});
        vgrad_host = vgrad.to(torch::kCPU).view({batch_size, seqlen, nheads, d});
        lse_host = lse.to(torch::kCPU);
        y_host = y.to(torch::kCPU).view({batch_size, seqlen, nheads, d});

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
