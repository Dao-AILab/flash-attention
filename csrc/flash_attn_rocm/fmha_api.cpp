#include <ATen/ATen.h>
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/HIPGeneratorImpl.h>
#include <c10/hip/HIPGuard.h>
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
                      at::Tensor out,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      void *o_tmp_d,
                      void *s_d,
                      void *softmax_lse_d,
                      float p_dropout,
                      float softmax_scale,
                      bool is_causal,
                      int num_splits) {

    Data_type acc_type = DATA_TYPE_FP32;
    Data_type data_type = !(q.dtype() == at::kBFloat16) ? DATA_TYPE_FP16 : DATA_TYPE_BF16;

    // Reset the parameters
    memset(&params, 0, sizeof(params));

    params.is_bf16 = q.dtype() == at::kBFloat16;

    params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q_d);
    params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k_d);

    // S = softmax(P)     //TO DO
    // params.s_ptr = s_d;
    // params.s_stride_in_bytes = get_size_in_bytes(b * h * seqlen_k, data_type);

    // Set the dimensions.
    params.b = b;                 // batch_size
    params.h = h;                 // num_heads
    params.seqlen_q = seqlen_q;   // seqlen q
    params.seqlen_k = seqlen_k;   // seqlen k
    params.d = d;                 // head_dim

    params.host_seqlens_q = (int*)malloc((params.b+1)*sizeof(int));
    params.host_seqlens_k = (int*)malloc((params.b+1)*sizeof(int));
    FMHA_CHECK_HIP(hipMemcpy(params.host_seqlens_q, params.cu_seqlens_q, (params.b+1)*sizeof(int), hipMemcpyDeviceToHost));
    FMHA_CHECK_HIP(hipMemcpy(params.host_seqlens_k, params.cu_seqlens_k, (params.b+1)*sizeof(int), hipMemcpyDeviceToHost));

    char* q_ptr = reinterpret_cast<char*>(q.data_ptr());
    char* k_ptr = reinterpret_cast<char*>(k.data_ptr());
    char* v_ptr = reinterpret_cast<char*>(v.data_ptr());
    char* out_ptr = reinterpret_cast<char*>(out.data_ptr());
    char* lse_ptr = reinterpret_cast<char*>(softmax_lse_d);

    //std::cout << "multiply" << params.seqlen_q * params.h * params.d<< std::endl;

    //std::cout << " q.data_ptr() " << q.data_ptr() << std::endl;
    //std::cout << " q_.data_ptr() " << q_.data_ptr() << std::endl;
    //std::cout << " q_[0].data_ptr() " << q_[0].data_ptr() << std::endl;
    //std::cout << " q_[1].data_ptr() " << q_[1].data_ptr() << std::endl;
    //std::cout << " new q[1] " << reinterpret_cast<void*>(q_ptr + params.seqlen_q * params.h * params.d * 2) << std::endl;
    //std::cout << " q_[0][0][0][0].data_ptr() " << q_[0][0][0][0].data_ptr() << std::endl;
    //std::cout << " q_[0][0][0][1].data_ptr() " << q_[0][0][0][1].data_ptr() << std::endl;
    //std::cout << " q_[0][0][1][0].data_ptr() " << q_[0][0][1][0].data_ptr() << std::endl;
    //std::cout << " q_[0][1][0][0].data_ptr() " << q_[0][1][0][0].data_ptr() << std::endl;
    //std::cout << " q_[1][0][0][0].data_ptr() " << q_[1][0][0][0].data_ptr() << std::endl;

    for (int i = 0; i < b; i++){
        params.q_ptr.push_back(reinterpret_cast<void*>(q_ptr));
        params.k_ptr.push_back(reinterpret_cast<void*>(k_ptr));
        params.v_ptr.push_back(reinterpret_cast<void*>(v_ptr));
        params.o_ptr.push_back(reinterpret_cast<void*>(out_ptr));
        int temp_seqlen_q = params.host_seqlens_q[i+1] - params.host_seqlens_q[i];
        int temp_seqlen_k = params.host_seqlens_k[i+1] - params.host_seqlens_k[i];
        int temp_q_stride = get_size_in_bytes(d * h * temp_seqlen_q, data_type);
        int temp_k_stride = get_size_in_bytes(d * h * temp_seqlen_k, data_type);
        q_ptr = q_ptr     + temp_q_stride;
        k_ptr = k_ptr     + temp_k_stride;
        v_ptr = v_ptr     + temp_k_stride;
        out_ptr = out_ptr + temp_q_stride;

        params.softmax_lse_ptr.push_back(reinterpret_cast<void*>(lse_ptr));
        int temp_lse_stride = get_size_in_bytes(h * seqlen_q, acc_type);
        lse_ptr = lse_ptr + temp_lse_stride;
    }

    // Set the different scale values.
    // const float scale_bmm1 = 1.f / sqrtf(d);
    const float scale_bmm1 = softmax_scale;

    params.scale_bmm1f = scale_bmm1;

    // Set this to probability of keeping an element to simplify things.
    params.p_dropout = p_dropout;

    params.is_causal = is_causal;
    params.num_splits = num_splits;
}

std::vector<at::Tensor>
mha_fwd(const at::Tensor &q,
        const at::Tensor &k,
        const at::Tensor &v,
        at::Tensor &out,
        const at::Tensor &cu_seqlens_q,
        const at::Tensor &cu_seqlens_k,
        const int max_seqlen_q_,
        const int max_seqlen_k_,
        const float p_dropout,
        const float softmax_scale,
        const bool zero_tensors,
        const bool is_causal,
        const bool return_softmax, // TO DO
        const int num_splits,      // num_splits is not used in rocm
        c10::optional<at::Generator> gen_) {

    auto dprops = at::cuda::getCurrentDeviceProperties();
    auto stream = at::cuda::getCurrentHIPStream().stream();
    bool is_dropout = p_dropout > 0.0;
    Launch_params<FMHA_fprop_params> launch_params(dprops, stream, is_dropout, return_softmax);

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
    TORCH_CHECK(cu_seqlens_q.is_cuda());
    TORCH_CHECK(cu_seqlens_k.is_cuda());

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

    int blocksize_c = head_size > 64 ? 128 : 256;
    // Need to round max_seqlen_k to multiples of blocksize_c
    int max_seqlen_k = ((max_seqlen_k_ + blocksize_c - 1) / blocksize_c) * blocksize_c;
    if( max_seqlen_k_ <= 128 ) {
        max_seqlen_k = 128;
    } else if( max_seqlen_k_ <= 256 ) {
        max_seqlen_k = 256;
    }
    int max_seqlen_q = ((max_seqlen_q_ + 16 - 1) / 16) * 16;
    // bool loop = false;

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    // at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    auto opts = q.options();

    auto softmax_lse = at::empty({batch_size, num_heads, max_seqlen_q}, opts.dtype(at::kFloat));
    // auto softmax_lse = torch::full({batch_size, num_heads, max_seqlen_k}, -std::numeric_limits<float>::infinity(), opts.dtype(at::kFloat));

    at::Tensor s;
    if (return_softmax) { s = at::empty({ batch_size, num_heads, max_seqlen_q, max_seqlen_k }, opts); }

    if( zero_tensors ) {
        out.zero_();
        softmax_lse.fill_(-std::numeric_limits<float>::infinity()).to(at::kCUDA);
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
                     q, k, v, out,
                     cu_seqlens_q.data_ptr(),
                     cu_seqlens_k.data_ptr(),
                     nullptr,
                     return_softmax ? s.data_ptr() : nullptr,
                     softmax_lse.data_ptr(),
                     p_dropout,
                     softmax_scale,
                     is_causal,
                     num_splits);

    // number of times random will be generated per thread, to offset philox counter in thc random
    // state
    // We use a custom RNG that increases the offset by batch_size * nheads * 32.
    int64_t counter_offset = launch_params.params.b * launch_params.params.h * 32;

    // at::PhiloxCudaState rng_engine_inputs;

    if( is_dropout ) {
        // See Note [Acquire lock when using random generators]
        std::lock_guard<std::mutex> lock(gen->mutex_);
        launch_params.params.philox_args = gen->philox_cuda_state(counter_offset);
    }

    run_fmha_fp16_bf16_gfx90a(launch_params);

    //at::Tensor softmax_lse_result = softmax_lse.to(torch::kCPU);

    std::vector<at::Tensor> result = {softmax_lse};
    if (return_softmax) {result.push_back(s);}
    return result;
}

//Commented functions yet to be supported.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Fused Multi-head Self-attention";
    m.def("fwd", &mha_fwd, "Forward pass");
    // m.def("bwd", &mha_bwd, "Backward pass");
    // m.def("fwd_block", &mha_fwd_block, "Forward pass (blocksparse)");
    // m.def("bwd_block", &mha_bwd_block, "Backward pass (blocksparse)");
}


//main function to test with the API
int main(){

    bool do_verification = true; // whether do verification

    int batch_size = 64;
    int nheads = 16;
    int seqlen = 256;
    int n = 1024;
    int d = n / nheads; //head_size//64

    //initialize the tensors
    at::Tensor q_host = at::rand({batch_size*seqlen, nheads, d}, torch::kBFloat16);//torch::kBFloat16;at::kHalf
    at::Tensor k_host = at::rand({batch_size*seqlen, nheads, d}, torch::kBFloat16);
    at::Tensor v_host = at::rand({batch_size*seqlen, nheads, d}, torch::kBFloat16);

    at::Tensor q = q_host.to(at::kCUDA);
    at::Tensor k = k_host.to(at::kCUDA);
    at::Tensor v = v_host.to(at::kCUDA);

    //initialize the output tensor
    at::Tensor out_host = at::empty({batch_size*seqlen, nheads, d},torch::kBFloat16);
    at::Tensor out = out_host.to(at::kCUDA);

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

    int max_seqlen_q_ = 256;
    int max_seqlen_k_ = 256;

    //other parameters
    float p_dropout = 0;
    float softmax_scale = 0.125;
    bool zero_tensors = true;
    bool is_causal = false;
    bool return_softmax = false; // TO DO
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
            p_dropout,
            softmax_scale,
            zero_tensors,
            is_causal,
            return_softmax,
            num_splits,
            gen_);


    using FP16 = ck::half_t;
    using BF16 = ck::bhalf_t;
    using F32 = float;

    using PassThrough = ck::tensor_operation::element_wise::PassThrough;

    using ADataType        = BF16;
    using B0DataType       = BF16;
    using B1DataType       = BF16;
    using AccDataType      = F32;
    using CShuffleDataType = F32;
    using CDataType        = BF16;
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

        std::vector<Tensor<ADataType>>  a_tensors;
        std::vector<Tensor<B0DataType>> b0_tensors;
        std::vector<Tensor<B1DataType>> b1_tensors;
        std::vector<Tensor<CDataType>>  c_tensors;
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

            std::vector<ck::index_t> lse_gs_ms_lengths{G0, G1, M};
            std::vector<ck::index_t> lse_gs_ms_strides =
                std::vector<ck::index_t>{G1 * M, M, 1}; // LSE layout [G0, G1, M]

            // C_m_o = A_m_k * B0_k_n * B1_n_o
            Tensor<ADataType> a_gs_ms_ks(a_gs_ms_ks_lengths, a_gs_ms_ks_strides);
            Tensor<B0DataType> b0_gs_ns_ks(b0_gs_ns_ks_lengths, b0_gs_ns_ks_strides);
            Tensor<B1DataType> b1_gs_os_ns(b1_gs_os_ns_lengths, b1_gs_os_ns_strides);
            Tensor<CDataType> c_gs_ms_os_device_result(c_gs_ms_os_lengths, c_gs_ms_os_strides);
            Tensor<LSEDataType> lse_gs_ms_device_result(lse_gs_ms_lengths, lse_gs_ms_strides);

            void* q_h_ptr_f = q_host[i].data_ptr();
            void* k_h_ptr_f = k_host[i].data_ptr();
            void* v_h_ptr_f = v_host[i].data_ptr();

            ADataType* q_h_ptr = reinterpret_cast<ADataType*>(q_h_ptr_f);
            B0DataType* k_h_ptr = reinterpret_cast<B0DataType*>(k_h_ptr_f);
            B1DataType* v_h_ptr = reinterpret_cast<B1DataType*>(v_h_ptr_f);

            //std::cout << "q_host[i].numel() " << q_host[i].numel() << std::endl;

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
            lse_tensors.push_back(lse_gs_ms_device_result);

        }

        at::Tensor out_device_result = out.to(torch::kCPU).view({batch_size, seqlen, nheads, d});
        at::Tensor lse_device_result = result[0].to(torch::kCPU);

        for(std::size_t i = 0; i < batch_size; i++)
        {
            const auto& a_gs_ms_ks         = a_tensors[i];
            const auto& b0_gs_ns_ks        = b0_tensors[i];
            const auto& b1_gs_os_ns        = b1_tensors[i];
            auto& c_gs_ms_os_device_result = c_tensors[i];
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

            //c_gs_ms_os_device_buf.FromDevice(c_gs_ms_os_device_result.mData.data());//

            Tensor<ADataType> a_g_m_k({G0 * G1, M, K});
            Tensor<B0DataType> b0_g_k_n({G0 * G1, K, N});
            Tensor<B1DataType> b1_g_n_o({G0 * G1, N, O});
            Tensor<AccDataType> acc0_g_m_n({G0 * G1, M, N});        // scratch object after gemm0
            Tensor<ADataType> a1_g_m_n({G0 * G1, M, N});            // scratch object after softmax
            Tensor<CDataType> c_g_m_o_host_result({G0 * G1, M, O}); // scratch object after gemm1
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

            bool pass_ =
                ck::utils::check_err(c_gs_ms_os_device_result.mData,
                                     c_gs_ms_os_host_result.mData,
                                     "Error: Incorrect results!",
                                     rtol,
                                     atol) &&
                ck::utils::check_err(lse_gs_ms_device_result.mData,
                                     lse_gs_ms_host_result.mData,
                                     "Error: Incorrect results lse!",
                                     rtol,
                                     atol);
            pass &= pass_;

            //for (int j = 0; j < 4 ; j++){
            //    std::cout << "data at j is "
            //    << ck::type_convert<float>(c_gs_ms_os_device_result.mData[j])
            //    << " , "
            //    << ck::type_convert<float>(c_gs_ms_os_host_result.mData[j])
            //    <<std::endl;
            //}

            //for (int j = 0; j < 16 ; j++){
            //    std::cout << "lse data at " << j << " is "
            //    << ck::type_convert<float>(lse_gs_ms_device_result.mData[j])
            //    << " , "
            //    << ck::type_convert<float>(lse_gs_ms_host_result.mData[j])
            //   <<std::endl;
            //}

        }

        if(pass)
        std::cout << "Verification passed!" <<std::endl;
        else
        std::cout << "Verification failed!" <<std::endl;
    }

    return pass ? 0 : 1;


}
