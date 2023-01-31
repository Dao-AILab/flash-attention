#include "fmha.h"
#include "fp16_switch.h"

#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <numeric>

#define FLASH_ATTN_IMPLENTATION 0

template <ck::index_t... Is> using S = ck::Sequence<Is...>;
using MaskingSpecialization =
    ck::tensor_operation::device::MaskingSpecialization;

static constexpr auto MaskingSpec_default = MaskingSpecialization::MaskDisabled;
static constexpr auto MaskingSpec_causal =
    MaskingSpecialization::MaskOutUpperTriangle;

struct SimpleDeviceMem {
  SimpleDeviceMem() = delete;
  SimpleDeviceMem(std::size_t mem_size) : p_mem_{} {
    (void)hipMalloc(static_cast<void **>(&p_mem_), mem_size);
  }
  void *GetDeviceBuffer() { return p_mem_; }
  ~SimpleDeviceMem() { (void)hipFree(p_mem_); }

  void *p_mem_;
};

template <typename InputType, ck::index_t MPerBlock, ck::index_t NPerBlock,
          ck::index_t KPerBlock, ck::index_t Gemm1NPerBlock,
          ck::index_t MPerXDL, ck::index_t NPerXDL, ck::index_t NXdlPerWave,
          ck::index_t Gemm1NXdlPerWave, typename ABlockTransfer,
          bool ABlockLdsExtraM, typename BBlockTransfer, bool B0BlockLdsExtraN,
          typename B1BlockTransfer, ck::index_t CShuffleNXdlPerWavePerShuffle,
          typename CShuffleBlockTransferClusterLengths,
          MaskingSpecialization MaskingSpec>
void run_fmha_dgrad_fp16_bf16_gfx90a_loop_(
    Launch_params<FMHA_dgrad_params> &launch_params) {
  using F16 = ck::half_t;
  using F32 = float;

  using PassThrough = ck::tensor_operation::element_wise::PassThrough;
  using Scale = ck::tensor_operation::element_wise::Scale;

  using QKVElementOp = PassThrough;
  using YElementOp = PassThrough;

  using DataType = F16;
  using AccDataType = F32;
  using ShuffleDataType = F32;
  using LSEDataType = F32;
  using Acc0BiasDataType = ck::Tuple<>;
  using Acc1BiasDataType = ck::Tuple<>;

  static constexpr ck::index_t NumDimG = 2;
  static constexpr ck::index_t NumDimM = 1;
  static constexpr ck::index_t NumDimN = 1;
  static constexpr ck::index_t NumDimK = 1;
  static constexpr ck::index_t NumDimO = 1;

  static constexpr auto GemmSpec =
      ck::tensor_operation::device::GemmSpecialization::MNKOPadding;

  static constexpr auto TensorSpecQ =
      ck::tensor_operation::device::TensorSpecialization::Default;
  static constexpr auto TensorSpecK =
      ck::tensor_operation::device::TensorSpecialization::Default;
  static constexpr auto TensorSpecV =
      ck::tensor_operation::device::TensorSpecialization::Default;
  static constexpr auto TensorSpecY =
      ck::tensor_operation::device::TensorSpecialization::Default;

  // init the instance with parameters
  #if FLASH_ATTN_IMPLENTATION
  using DeviceGemmInstance = 
      ck::tensor_operation::device::DeviceGroupedMultiheadAttentionBackward_Xdl_CShuffle<
          NumDimG, NumDimM, NumDimN, NumDimK, NumDimO, DataType, LSEDataType,
          Acc0BiasDataType, Acc1BiasDataType, AccDataType, ShuffleDataType,
          QKVElementOp, QKVElementOp, Scale, QKVElementOp, YElementOp, GemmSpec,
          TensorSpecQ, TensorSpecK, TensorSpecV, TensorSpecY, 1, 256,
          MPerBlock,        // MPerBlock
          NPerBlock,        // NPerBlock
          KPerBlock,        // KPerBlock
          Gemm1NPerBlock,   // Gemm1NPerBlock
          64,               // Gemm1KPerBlock
          8,                // AK1
          8,                // BK1
          2,                // B1K1
          MPerXDL,          // MPerXDL
          NPerXDL,          // NPerXDL
          1,                // MXdlPerWave
          NXdlPerWave,      // NXdlPerWave
          Gemm1NXdlPerWave, // Gemm1NXdlPerWave
          ABlockTransfer,   // ABlockTransfer
          S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8,
          ABlockLdsExtraM, // ABlockLdsExtraM
          BBlockTransfer,  // BBlockTransfer
          S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8,
          B0BlockLdsExtraN, // B0BlockLdsExtraN
          B1BlockTransfer,  // B1BlockTransfer
          S<0, 2, 1>, S<0, 2, 1>, 1, 4, 2, false,
          1,                             // CShuffleMXdlPerWavePerShuffle
          CShuffleNXdlPerWavePerShuffle, // CShuffleNXdlPerWavePerShuffle
          CShuffleBlockTransferClusterLengths, // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
          8,            // CShuffleBlockTransferScalarPerVector_NPerBlock
          MaskingSpec>; // MaskingSpecialization
  #else
  using DeviceGemmInstance = 
      ck::tensor_operation::device::DeviceGroupedMultiheadAttentionBackward_Xdl_CShuffle_V2<
          NumDimG, NumDimM, NumDimN, NumDimK, NumDimO, DataType, LSEDataType,
          Acc0BiasDataType, Acc1BiasDataType, AccDataType, ShuffleDataType,
          QKVElementOp, QKVElementOp, Scale, QKVElementOp, YElementOp, GemmSpec,
          TensorSpecQ, TensorSpecK, TensorSpecV, TensorSpecY, 1, 256,
          MPerBlock,        // MPerBlock
          NPerBlock,        // NPerBlock
          KPerBlock,        // KPerBlock
          Gemm1NPerBlock,   // Gemm1NPerBlock
          32,               // Gemm1KPerBlock
          8,                // AK1
          8,                // BK1
          2,                // B1K1
          MPerXDL,          // MPerXDL
          NPerXDL,          // NPerXDL
          1,                // MXdlPerWave
          NXdlPerWave,      // NXdlPerWave
          Gemm1NXdlPerWave, // Gemm1NXdlPerWave
          ABlockTransfer,   // ABlockTransfer
          S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8,
          ABlockLdsExtraM, // ABlockLdsExtraM
          BBlockTransfer,  // BBlockTransfer
          S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8,
          B0BlockLdsExtraN, // B0BlockLdsExtraN
          B1BlockTransfer,  // B1BlockTransfer
          S<0, 2, 1>, S<0, 2, 1>, 1, 4, 2, false,
          1,                             // CShuffleMXdlPerWavePerShuffle
          CShuffleNXdlPerWavePerShuffle, // CShuffleNXdlPerWavePerShuffle
          CShuffleBlockTransferClusterLengths, // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
          8,            // CShuffleBlockTransferScalarPerVector_NPerBlock
          MaskingSpec>; // MaskingSpecialization
  #endif

  bool time_kernel = false;

  bool input_permute = true;
  bool output_permute = true;

  float alpha = launch_params.params.scale_bmm1f;
  auto a_element_op = QKVElementOp{};
  auto b0_element_op = QKVElementOp{};
  auto acc0_element_op = Scale{alpha};
  auto b1_element_op = QKVElementOp{};
  auto c_element_op = YElementOp{};

  auto p_q = launch_params.params.q_ptr;
  auto p_k = launch_params.params.k_ptr;
  auto p_v = launch_params.params.v_ptr;
  auto p_y = launch_params.params.y_ptr;
  auto p_lse = launch_params.params.lse_ptr;
  auto p_ygrad = launch_params.params.ygrad_ptr;
  auto p_qgrad = launch_params.params.qgrad_ptr;
  auto p_kgrad = launch_params.params.kgrad_ptr;
  auto p_vgrad = launch_params.params.vgrad_ptr;

  std::vector<typename DeviceGemmInstance::ProblemDesc> problem_descs;

  int batch_size = launch_params.params.b;
  int num_heads = launch_params.params.h;
  int head_dim = launch_params.params.d;

  // int* host_seqlens_q;
  // int* host_seqlens_k;
  // host_seqlens_q = (int*)malloc((launch_params.params.b+1)*sizeof(int));
  // host_seqlens_k = (int*)malloc((launch_params.params.b+1)*sizeof(int));
  // FMHA_CHECK_HIP(hipMemcpy(host_seqlens_q, launch_params.params.cu_seqlens_q,
  // (launch_params.params.b+1)*sizeof(int), hipMemcpyDeviceToHost));
  // FMHA_CHECK_HIP(hipMemcpy(host_seqlens_k, launch_params.params.cu_seqlens_k,
  // (launch_params.params.b+1)*sizeof(int), hipMemcpyDeviceToHost));

  for (size_t i = 0; i < batch_size; i++) {
    int M = launch_params.params.host_seqlens_q[i + 1] -
            launch_params.params.host_seqlens_q[i]; // seqlen Q
    int N = launch_params.params.host_seqlens_k[i + 1] -
            launch_params.params.host_seqlens_k[i]; // seqlen K    
    int K = head_dim;
    int O = head_dim;
    int G0 = 1; // G0 = batch_size
    int G1 = num_heads;
    std::vector<ck::index_t> q_gs_ms_ks_lengths{G0, G1, M, K};
    std::vector<ck::index_t> q_gs_ms_ks_strides =
        input_permute ? std::vector<ck::index_t>{M * G1 * K, K, G1 * K, 1}
                      // A layout [G0, M, G1, K]
                      : std::vector<ck::index_t>{G1 * M * K, M * K, K,
                                                 1}; // A layout [G0, G1, M, K]

    std::vector<ck::index_t> k_gs_ns_ks_lengths{G0, G1, N, K};
    std::vector<ck::index_t> k_gs_ns_ks_strides =
        input_permute ? std::vector<ck::index_t>{N * G1 * K, K, G1 * K, 1}
                      // B0 layout [G0, N, G1, K]
                      : std::vector<ck::index_t>{G1 * N * K, N * K, K,
                                                 1}; // B0 layout [G0, G1, N, K]

    std::vector<ck::index_t> v_gs_os_ns_lengths{G0, G1, O, N};
    std::vector<ck::index_t> v_gs_os_ns_strides =
        input_permute ? std::vector<ck::index_t>{N * G1 * O, O, 1, G1 * O}
                      // B1 layout [G0, N, G1, O]
                      : std::vector<ck::index_t>{G1 * N * O, N * O, 1,
                                                 O}; // B1 layout [G0, G1, N, O]

    std::vector<ck::index_t> y_gs_ms_os_lengths{G0, G1, M, O};
    std::vector<ck::index_t> y_gs_ms_os_strides =
        output_permute ? std::vector<ck::index_t>{M * G1 * O, O, G1 * O, 1}
                       // C layout [G0, M, G1, O]
                       : std::vector<ck::index_t>{G1 * M * O, M * O, O,
                                                  1}; // C layout [G0, G1, M, O]

    std::vector<ck::index_t> lse_gs_ms_lengths{G0, G1, M};
    std::vector<ck::index_t> lse_gs_ms_strides{G1 * M, M,
                                               1}; // LSE layout [G0, G1, M]

    problem_descs.push_back({q_gs_ms_ks_lengths,
                             q_gs_ms_ks_strides,
                             k_gs_ns_ks_lengths,
                             k_gs_ns_ks_strides,
                             v_gs_os_ns_lengths,
                             v_gs_os_ns_strides,
                             y_gs_ms_os_lengths,
                             y_gs_ms_os_strides,
                             lse_gs_ms_lengths,
                             lse_gs_ms_strides,
                             {},   // acc0_biases_gs_ms_ns_lengths
                             {},   // acc0_biases_gs_ms_ns_strides
                             {},   // acc1_biases_gs_ms_os_lengths
                             {}}); // acc1_biases_gs_ms_os_strides
  }

  // do GEMM
  auto gemm = DeviceGemmInstance{};
  auto invoker = gemm.MakeInvoker();
  auto argument = gemm.MakeArgument(
      p_q, p_k, p_v, p_y, p_lse, p_ygrad, p_qgrad, p_kgrad, p_vgrad, {}, {},
      problem_descs, a_element_op, b0_element_op, acc0_element_op,
      b1_element_op, c_element_op);

  // specify workspace for problem_desc
  SimpleDeviceMem problem_desc_workspace(gemm.GetWorkSpaceSize(&argument));

  gemm.SetWorkSpacePointer(&argument, problem_desc_workspace.GetDeviceBuffer());

  if (!gemm.IsSupportedArgument(argument)) {
    std::cout << gemm.GetTypeString() << " does not support this problem"
              << std::endl;

    return;
  }

  float ave_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel});

  if (time_kernel) {
    std::cout << "time elpase is " << ave_time << " ms" << std::endl;
  }
}

void run_fmha_dgrad_fp16_bf16_gfx90a(
    Launch_params<FMHA_dgrad_params> &launch_params) {

  // ck::index_t MPerBlock,    ck::index_t NPerBlock, ck::index_t KPerBlock,
  // ck::index_t Gemm1NPerBlock, ck::index_t MPerXDL,      ck::index_t NPerXDL,
  // ck::index_t NXdlPerWave, ck::index_t Gemm1NXdlPerWave, typename
  // ABlockTransfer,  bool ABlockLdsExtraM,  typename BBlockTransfer, bool
  // B0BlockLdsExtraN, typename B1BlockTransfer, ck::index_t
  // CShuffleNXdlPerWavePerShuffle >

  FP16_SWITCH(launch_params.params.is_bf16, [&] {
    if (launch_params.params.is_causal) {
      if (launch_params.params.b <= 16) {
        if (launch_params.params.d <= 32) {
          if (launch_params.params.seqlen_k <= 128) {
            run_fmha_dgrad_fp16_bf16_gfx90a_loop_<
                elem_type, 128, 64, 32, 128, 32, 32, 2, 4, S<4, 64, 1>, true,
                S<4, 64, 1>, true, S<8, 32, 1>, 2, S<1, 32, 1, 8>,
                MaskingSpec_causal>(launch_params);
          } else { // if(launch_params.params.seqlen_k <= 256){
            run_fmha_dgrad_fp16_bf16_gfx90a_loop_<
                elem_type, 128, 128, 64, 128, 32, 32, 4, 4, S<8, 32, 1>, false,
                S<8, 32, 1>, false, S<8, 32, 1>, 2, S<1, 32, 1, 8>,
                MaskingSpec_causal>(launch_params);
          }
        } else { // if(launch_params.params.d <= 128){
          if (launch_params.params.seqlen_k <= 128) {
            run_fmha_dgrad_fp16_bf16_gfx90a_loop_<
                elem_type, 128, 128, 32, 64, 32, 32, 4, 2, S<4, 64, 1>, true,
                S<4, 64, 1>, true, S<16, 16, 1>, 2, S<1, 32, 1, 8>,
                MaskingSpec_causal>(launch_params);
          } else { // if(launch_params.params.seqlen_k <= 256){
            run_fmha_dgrad_fp16_bf16_gfx90a_loop_<
                elem_type, 64, 256, 32, 64, 16, 16, 16, 4, S<4, 64, 1>, true,
                S<4, 64, 1>, true, S<16, 16, 1>, 4, S<1, 32, 1, 8>,
                MaskingSpec_causal>(launch_params);
          }
        }
      } else {
        if (launch_params.params.seqlen_k <= 128) {
          if (launch_params.params.d > 32 && launch_params.params.d <= 64) {
            run_fmha_dgrad_fp16_bf16_gfx90a_loop_<
                elem_type, 128, 128, 32, 64, 32, 32, 4, 2, S<4, 64, 1>, true,
                S<4, 64, 1>, true, S<16, 16, 1>, 2, S<1, 32, 1, 8>,
                MaskingSpec_causal>(launch_params);
          } else {
            run_fmha_dgrad_fp16_bf16_gfx90a_loop_<
                elem_type, 128, 128, 64, 128, 32, 32, 4, 4, S<8, 32, 1>, false,
                S<8, 32, 1>, false, S<8, 32, 1>, 2, S<1, 32, 1, 8>,
                MaskingSpec_causal>(launch_params);
          }
        } else {
          if (launch_params.params.d <= 32) {
            run_fmha_dgrad_fp16_bf16_gfx90a_loop_<
                elem_type, 128, 128, 64, 128, 32, 32, 4, 4, S<8, 32, 1>, false,
                S<8, 32, 1>, false, S<8, 32, 1>, 2, S<1, 32, 1, 8>,
                MaskingSpec_causal>(launch_params);
          } else if (launch_params.params.d <= 64) {
            run_fmha_dgrad_fp16_bf16_gfx90a_loop_<
                elem_type, 128, 128, 32, 64, 32, 32, 4, 2, S<4, 64, 1>, true,
                S<4, 64, 1>, true, S<16, 16, 1>, 2, S<1, 32, 1, 8>,
                MaskingSpec_causal>(launch_params);
          } else { // if(launch_params.params.d <= 128){
            run_fmha_dgrad_fp16_bf16_gfx90a_loop_<
                elem_type, 64, 256, 32, 128, 16, 16, 16, 8, S<4, 64, 1>, true,
                S<4, 64, 1>, true, S<8, 32, 1>, 8, S<1, 16, 1, 16>,
                MaskingSpec_causal>(launch_params);
          }
        }
      }
    } else {
      if (launch_params.params.b <= 16) {
        if (launch_params.params.d <= 32) {
          if (launch_params.params.seqlen_k <= 128) {
            run_fmha_dgrad_fp16_bf16_gfx90a_loop_<
                elem_type, 128, 64, 32, 128, 32, 32, 2, 4, S<4, 64, 1>, true,
                S<4, 64, 1>, true, S<8, 32, 1>, 2, S<1, 32, 1, 8>,
                MaskingSpec_default>(launch_params);
          } else { // if(launch_params.params.seqlen_k <= 256){
            run_fmha_dgrad_fp16_bf16_gfx90a_loop_<
                elem_type, 128, 128, 64, 128, 32, 32, 4, 4, S<8, 32, 1>, false,
                S<8, 32, 1>, false, S<8, 32, 1>, 2, S<1, 32, 1, 8>,
                MaskingSpec_default>(launch_params);
          }
        } else if (launch_params.params.d <= 128) {
          if (launch_params.params.seqlen_k <= 128) {
            run_fmha_dgrad_fp16_bf16_gfx90a_loop_<
                elem_type, 128, 128, 32, 64, 32, 32, 4, 2, S<4, 64, 1>, true,
                S<4, 64, 1>, true, S<16, 16, 1>, 2, S<1, 32, 1, 8>,
                MaskingSpec_default>(launch_params);
          } else { // if(launch_params.params.seqlen_k <= 256){
            run_fmha_dgrad_fp16_bf16_gfx90a_loop_<
                elem_type, 64, 256, 32, 64, 16, 16, 16, 4, S<4, 64, 1>, true,
                S<4, 64, 1>, true, S<16, 16, 1>, 4, S<1, 32, 1, 8>,
                MaskingSpec_default>(launch_params);
          }
        }
      } else {
        if (launch_params.params.seqlen_k <= 128) {
          if (launch_params.params.d > 32 && launch_params.params.d <= 64) {
            run_fmha_dgrad_fp16_bf16_gfx90a_loop_<
                elem_type, 128, 128, 32, 64, 32, 32, 4, 2, S<4, 64, 1>, true,
                S<4, 64, 1>, true, S<16, 16, 1>, 2, S<1, 32, 1, 8>,
                MaskingSpec_default>(launch_params);
          } else {
            run_fmha_dgrad_fp16_bf16_gfx90a_loop_<
                elem_type, 128, 128, 64, 128, 32, 32, 4, 4, S<8, 32, 1>, false,
                S<8, 32, 1>, false, S<8, 32, 1>, 2, S<1, 32, 1, 8>,
                MaskingSpec_default>(launch_params);
          }
        } else {
          if (launch_params.params.d <= 32) {
            run_fmha_dgrad_fp16_bf16_gfx90a_loop_<
                elem_type, 128, 128, 64, 128, 32, 32, 4, 4, S<8, 32, 1>, false,
                S<8, 32, 1>, false, S<8, 32, 1>, 2, S<1, 32, 1, 8>,
                MaskingSpec_default>(launch_params);
          } else if (launch_params.params.d <= 64) {
            run_fmha_dgrad_fp16_bf16_gfx90a_loop_<
                elem_type, 128, 128, 32, 64, 32, 32, 4, 2, S<4, 64, 1>, true,
                S<4, 64, 1>, true, S<16, 16, 1>, 2, S<1, 32, 1, 8>,
                MaskingSpec_default>(launch_params);
          } else { // if(launch_params.params.d <= 128){
            run_fmha_dgrad_fp16_bf16_gfx90a_loop_<
                elem_type, 64, 256, 32, 128, 16, 16, 16, 8, S<4, 64, 1>, true,
                S<4, 64, 1>, true, S<8, 32, 1>, 8, S<1, 16, 1, 16>,
                MaskingSpec_default>(launch_params);
          }
        }
      }
    }
  });
}