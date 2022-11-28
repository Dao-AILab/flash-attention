
//#include <cuda_fp16.h>
//#include <cuda_bf16.h>

#include "fmha.h"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

struct SimpleDeviceMem
{
    SimpleDeviceMem() = delete;

    SimpleDeviceMem(std::size_t mem_size) : p_mem_{}
    {
        (void)hipMalloc(static_cast<void**>(&p_mem_), mem_size);
    }

    void* GetDeviceBuffer() { return p_mem_; }

    ~SimpleDeviceMem() { (void)hipFree(p_mem_); }

    void* p_mem_;
};

void run_fmha_fp16_gfx90a(Launch_params<FMHA_fprop_params> &launch_params) {

    //TODO : Find out and choose proper instances parameters for different problem sizes

    using F16 = ck::half_t;
    using F32 = float;

    using PassThrough = ck::tensor_operation::element_wise::PassThrough;

    using ADataType        = F16;
    using B0DataType       = F16;
    using B1DataType       = F16;
    using AccDataType      = F32;
    using CShuffleDataType = F32;
    using CDataType        = F16;
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

    static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::MNKOPadding;
    static constexpr auto MaskingSpec =
        ck::tensor_operation::device::MaskingSpecialization::MaskDisabled;

    static constexpr auto TensorSpecA  = ck::tensor_operation::device::TensorSpecialization::Default;
    static constexpr auto TensorSpecB0 = ck::tensor_operation::device::TensorSpecialization::Default;
    static constexpr auto TensorSpecB1 = ck::tensor_operation::device::TensorSpecialization::Default;
    static constexpr auto TensorSpecC  = ck::tensor_operation::device::TensorSpecialization::Default;
    
    //init the instance with parameters
    using DeviceGemmInstance =
        ck::tensor_operation::device::DeviceGroupedGemmSoftmaxGemmPermute_Xdl_CShuffle<
            NumDimG,
            NumDimM,
            NumDimN,
            NumDimK,
            NumDimO,
            ADataType,
            B0DataType,
            B1DataType,
            CDataType,
            Acc0BiasDataType,
            Acc1BiasDataType,
            AccDataType,
            CShuffleDataType,
            AElementOp,
            B0ElementOp,
            Acc0ElementOp,
            B1ElementOp,
            CElementOp,
            GemmSpec,
            TensorSpecA,
            TensorSpecB0,
            TensorSpecB1,
            TensorSpecC,
            1,
            256,
            128,         // MPerBlock
            128,         // NPerBlock
            32,          // KPerBlock
            64,          // Gemm1NPerBlock
            32,          // Gemm1KPerBlock
            8,           // AK1
            8,           // BK1
            2,           // B1K1
            32,          // MPerXDL
            32,          // NPerXDL
            1,           // MXdlPerWave
            4,           // NXdlPerWave
            2,           // Gemm1NXdlPerWave
            S<4, 64, 1>, // ABlockTransfer
            S<1, 0, 2>,
            S<1, 0, 2>,
            2,
            8,
            8,
            true,
            S<4, 64, 1>, // BBlockTransfer
            S<1, 0, 2>,
            S<1, 0, 2>,
            2,
            8,
            8,
            true,
            S<16, 16, 1>, // B1BlockTransfer
            S<0, 2, 1>,
            S<0, 2, 1>,
            1,
            4,
            2,
            false,
            1,              // CShuffleMXdlPerWavePerShuffle
            2,              // CShuffleNXdlPerWavePerShuffle
            S<1, 32, 1, 8>, // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
            8,              // CShuffleBlockTransferScalarPerVector_NPerBlock
            MaskingSpec>;   // MaskingSpecialization
        
    bool do_verification = false;
    bool time_kernel     = true;

    bool input_permute  = true;
    bool output_permute = true;

    auto a_element_op    = AElementOp{};
    auto b0_element_op   = B0ElementOp{};
    auto acc0_element_op = Acc0ElementOp{alpha};
    auto b1_element_op   = B1ElementOp{};
    auto c_element_op    = CElementOp{};

    void* p_a = launch_params.params.q_ptr;
    void* p_b0 = launch_params.params.k_ptr;
    void* p_b1 = launch_params.params.v_ptr;
    void* p_c = launch_params.params.o_ptr;

    std::vector<DeviceGemmInstance::ProblemDesc> problem_descs;

    int batch_size = launch_params.params.b;
    int num_heads = launch_params.params.h;
    int head_dim = launch_params.params.d;

    int* host_seqlens_q;
    int* host_seqlens_k;
    host_seqlens_q = (int*)malloc((params.b+1)*sizeof(int));
    host_seqlens_k = (int*)malloc((params.b+1)*sizeof(int));
    hipMemcpy(host_seqlens_q, params.cu_seqlens_q, (params.b+1)*sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(host_seqlens_k, params.cu_seqlens_k, (params.b+1)*sizeof(int), hipMemcpyDeviceToHost);

    for(size_t i = 0; i < (batch_size + 1); i++){
        int M     = host_seqlens_q[i + 1] - host_seqlens_q[i]; //seqlen Q
        int N     = host_seqlens_k[i + 1] - host_seqlens_k[i]; //seqlen K
        int K     = head_dim;
        int O     = head_dim;
        int G0 = 1; // G0 = batch_size
        int G1 = num_heads;

        std::vector<ck::index_t> a_gs_ms_ks_lengths{G0, G1, M, K};
        std::vector<ck::index_t> a_gs_ms_ks_strides =
            input_permute
                ? std::vector<ck::index_t>{M * G1 * K, K, G1 * K, 1} // A layout [G0, M, G1, K]
                : std::vector<ck::index_t>{G1 * M * K, M * K, K, 1}; // A layout [G0, G1, M, K]

        std::vector<ck::index_t> b0_gs_ns_ks_lengths{G0, G1, N, K};
        std::vector<ck::index_t> b0_gs_ns_ks_strides =
            input_permute
                ? std::vector<ck::index_t>{N * G1 * K, K, G1 * K, 1} // B0 layout [G0, N, G1, K]
                : std::vector<ck::index_t>{G1 * N * K, N * K, K, 1}; // B0 layout [G0, G1, N, K]

        std::vector<ck::index_t> b1_gs_os_ns_lengths{G0, G1, O, N};
        std::vector<ck::index_t> b1_gs_os_ns_strides =
            input_permute
                ? std::vector<ck::index_t>{N * G1 * O, O, 1, G1 * O} // B1 layout [G0, N, G1, O]
                : std::vector<ck::index_t>{G1 * N * O, N * O, 1, O}; // B1 layout [G0, G1, N, O]

        std::vector<ck::index_t> c_gs_ms_os_lengths{G0, G1, M, O};
        std::vector<ck::index_t> c_gs_ms_os_strides =
            output_permute
                ? std::vector<ck::index_t>{M * G1 * O, O, G1 * O, 1} // C layout [G0, M, G1, O]
                : std::vector<ck::index_t>{G1 * M * O, M * O, O, 1}; // C layout [G0, G1, M, O]

        problem_descs.push_back({a_gs_ms_ks_lengths,
                                 a_gs_ms_ks_strides,
                                 b0_gs_ns_ks_lengths,
                                 b0_gs_ns_ks_strides,
                                 b1_gs_os_ns_lengths,
                                 b1_gs_os_ns_strides,
                                 c_gs_ms_os_lengths,
                                 c_gs_ms_os_strides,
                                 {},   // acc0_biases_gs_ms_ns_lengths
                                 {},   // acc0_biases_gs_ms_ns_strides
                                 {},   // acc1_biases_gs_ms_os_lengths
                                 {}}); // acc1_biases_gs_ms_os_strides
                                 
    }

    // do GEMM
    auto gemm     = DeviceGemmInstance{};
    auto invoker  = gemm.MakeInvoker();
    auto argument = gemm.MakeArgument(p_a,
                                      p_b0,
                                      p_b1,
                                      p_c,
                                      {},
                                      {},
                                      problem_descs,
                                      a_element_op,
                                      b0_element_op,
                                      acc0_element_op,
                                      b1_element_op,
                                      c_element_op);

    // specify workspace for problem_desc
    SimpleDeviceMem problem_desc_workspace(gemm.GetWorkSpaceSize(&argument));

    gemm.SetWorkSpacePointer(&argument, problem_desc_workspace.GetDeviceBuffer());

    if(!gemm.IsSupportedArgument(argument))
    {
        std::cout << gemm.GetTypeString() << " does not support this problem" << std::endl;

        return 0;
    }

    float ave_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel});


}