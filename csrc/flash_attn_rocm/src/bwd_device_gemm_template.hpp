// BSD 3 Clause
// Copyright 2023 Advanced Micro Devices, Inc.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software without
// specific prior written permission. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT
// HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
// INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
// FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
// OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
// EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include "device_gemm_trait.hpp"

namespace bwd_device_gemm {
namespace device_op =
    ck::tensor_operation::device; // namespace alias for internal use
// type alias for DeviceGroupedMultiheadAttentionBackward_Xdl_CShuffle_V1 of
// head dim = 32
template <typename DeviceGemmTraits>
using DeviceGemmGroupedHeadDim32 = device_op::
    DeviceGroupedMultiheadAttentionBackward_Qloop_Xdl_CShuffle_Light_V1<
        DeviceGemmTraits::kNumDimG, DeviceGemmTraits::kNumDimM,
        DeviceGemmTraits::kNumDimN, DeviceGemmTraits::kNumDimK,
        DeviceGemmTraits::kNumDimO, typename DeviceGemmTraits::InputDataType,
        typename DeviceGemmTraits::OutputDataType,
        typename DeviceGemmTraits::GemmDataType,
        typename DeviceGemmTraits::ZDataType,
        typename DeviceGemmTraits::LSEDataType,
        typename DeviceGemmTraits::DDataType,
        typename DeviceGemmTraits::Acc0BiasDataType,
        typename DeviceGemmTraits::Acc1BiasDataType,
        typename DeviceGemmTraits::AccDataType,
        typename DeviceGemmTraits::ShuffleDataType,
        typename DeviceGemmTraits::QElementOp,
        typename DeviceGemmTraits::KElementOp,
        typename DeviceGemmTraits::Acc0ElementOp,
        typename DeviceGemmTraits::VElementOp,
        typename DeviceGemmTraits::OutElementOp, DeviceGemmTraits::kGemmSpec,
        DeviceGemmTraits::kTensorSpecQ, DeviceGemmTraits::kTensorSpecK,
        DeviceGemmTraits::kTensorSpecV, DeviceGemmTraits::kTensorSpecY, 1, 256,
        128,                                // MPerBlock
        128,                                // NPerBlock
        32,                                 // KPerBlock
        32,                                 // Gemm1NPerBlock
        32,                                 // Gemm1KPerBlock
        64,                                 // Gemm2KPerBlock
        8,                                  // AK1
        8,                                  // BK1
        2,                                  // B1K1
        32,                                 // MPerXDL
        32,                                 // NPerXDL
        4,                                  // MXdlPerWave
        1,                                  // NXdlPerWave
        1,                                  // Gemm1NXdlPerWave
        1,                                  // Gemm2NXdlPerWave
        32, device_gemm_trait::S<4, 64, 1>, // ABlockTransfer
        device_gemm_trait::S<1, 0, 2>, device_gemm_trait::S<1, 0, 2>, 2, 8, 8,
        true, device_gemm_trait::S<4, 64, 1>, // BBlockTransfer
        device_gemm_trait::S<1, 0, 2>, device_gemm_trait::S<1, 0, 2>, 2, 8, 8,
        true, 4,
        1, // CShuffleMXdlPerWavePerShuffle
        1, // CShuffleNXdlPerWavePerShuffle
        device_gemm_trait::S<
            1, 64, 1,
            4>, // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
        DeviceGemmTraits::
            kCShuffleBlockTransferScalarPerVectorNPerBlock, // c_shuffle_block_transfer_scalar_per_vector_n_per_block
        DeviceGemmTraits::kMaskingSpec, // MaskingSpec
        DeviceGemmTraits::kIsDeterministic>;

// type alias for DeviceGroupedMultiheadAttentionBackward_Xdl_CShuffle_V1 of
// head dim = 64
template <typename DeviceGemmTraits>
using DeviceGemmGroupedHeadDim64 = device_op::
    DeviceGroupedMultiheadAttentionBackward_Qloop_Xdl_CShuffle_Light_V1<
        DeviceGemmTraits::kNumDimG, DeviceGemmTraits::kNumDimM,
        DeviceGemmTraits::kNumDimN, DeviceGemmTraits::kNumDimK,
        DeviceGemmTraits::kNumDimO, typename DeviceGemmTraits::InputDataType,
        typename DeviceGemmTraits::OutputDataType,
        typename DeviceGemmTraits::GemmDataType,
        typename DeviceGemmTraits::ZDataType,
        typename DeviceGemmTraits::LSEDataType,
        typename DeviceGemmTraits::DDataType,
        typename DeviceGemmTraits::Acc0BiasDataType,
        typename DeviceGemmTraits::Acc1BiasDataType,
        typename DeviceGemmTraits::AccDataType,
        typename DeviceGemmTraits::ShuffleDataType,
        typename DeviceGemmTraits::QElementOp,
        typename DeviceGemmTraits::KElementOp,
        typename DeviceGemmTraits::Acc0ElementOp,
        typename DeviceGemmTraits::VElementOp,
        typename DeviceGemmTraits::OutElementOp, DeviceGemmTraits::kGemmSpec,
        DeviceGemmTraits::kTensorSpecQ, DeviceGemmTraits::kTensorSpecK,
        DeviceGemmTraits::kTensorSpecV, DeviceGemmTraits::kTensorSpecY, 1, 256,
        128,                                // MPerBlock
        128,                                // NPerBlock
        64,                                 // KPerBlock
        64,                                 // Gemm1NPerBlock
        32,                                 // Gemm1KPerBlock
        32,                                 // Gemm2KPerBlock
        8,                                  // AK1
        8,                                  // BK1
        2,                                  // B1K1
        32,                                 // MPerXDL
        32,                                 // NPerXDL
        4,                                  // MXdlPerWave
        1,                                  // NXdlPerWave
        2,                                  // Gemm1NXdlPerWave
        1,                                  // Gemm2NXdlPerWave
        64, device_gemm_trait::S<4, 64, 1>, // ABlockTransfer
        device_gemm_trait::S<1, 0, 2>, device_gemm_trait::S<1, 0, 2>, 2, 8, 8,
        true, device_gemm_trait::S<4, 64, 1>, // BBlockTransfer
        device_gemm_trait::S<1, 0, 2>, device_gemm_trait::S<1, 0, 2>, 2, 8, 8,
        true, 4,
        1, // CShuffleMXdlPerWavePerShuffle
        2, // CShuffleNXdlPerWavePerShuffle
        device_gemm_trait::S<
            1, 32, 1,
            8>, // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
        DeviceGemmTraits::
            kCShuffleBlockTransferScalarPerVectorNPerBlock, // c_shuffle_block_transfer_scalar_per_vector_n_per_block
        DeviceGemmTraits::kMaskingSpec, // MaskingSpecc
        DeviceGemmTraits::kIsDeterministic>;

// type alias for DeviceGroupedMultiheadAttentionBackward_Xdl_CShuffle_V2 of
// head dim = 128
template <typename DeviceGemmTraits>
using DeviceGemmGroupedHeadDim128 = device_op::
    DeviceGroupedMultiheadAttentionBackward_Qloop_Xdl_CShuffle_Light_V2<
        DeviceGemmTraits::kNumDimG, DeviceGemmTraits::kNumDimM,
        DeviceGemmTraits::kNumDimN, DeviceGemmTraits::kNumDimK,
        DeviceGemmTraits::kNumDimO, typename DeviceGemmTraits::InputDataType,
        typename DeviceGemmTraits::OutputDataType,
        typename DeviceGemmTraits::GemmDataType,
        typename DeviceGemmTraits::ZDataType,
        typename DeviceGemmTraits::LSEDataType,
        typename DeviceGemmTraits::DDataType,
        typename DeviceGemmTraits::Acc0BiasDataType,
        typename DeviceGemmTraits::Acc1BiasDataType,
        typename DeviceGemmTraits::AccDataType,
        typename DeviceGemmTraits::ShuffleDataType,
        typename DeviceGemmTraits::QElementOp,
        typename DeviceGemmTraits::KElementOp,
        typename DeviceGemmTraits::Acc0ElementOp,
        typename DeviceGemmTraits::VElementOp,
        typename DeviceGemmTraits::OutElementOp, DeviceGemmTraits::kGemmSpec,
        DeviceGemmTraits::kTensorSpecQ, DeviceGemmTraits::kTensorSpecK,
        DeviceGemmTraits::kTensorSpecV, DeviceGemmTraits::kTensorSpecY, 1, 256,
        64,                                 // MPerBlock
        128,                                // NPerBlock
        64,                                 // KPerBlock
        128,                                // Gemm1NPerBlock
        32,                                 // Gemm1KPerBlock
        128,                                // Gemm2KPerBlock
        8,                                  // AK1
        8,                                  // BK1
        2,                                  // B1K1
        32,                                 // MPerXDL
        32,                                 // NPerXDL
        2,                                  // MXdlPerWave
        1,                                  // NXdlPerWave
        4,                                  // Gemm1NXdlPerWave
        1,                                  // Gemm2NXdlPerWave
        64, device_gemm_trait::S<4, 64, 1>, // ABlockTransfer
        device_gemm_trait::S<1, 0, 2>, device_gemm_trait::S<1, 0, 2>, 2, 8, 8,
        true, device_gemm_trait::S<4, 64, 1>, // BBlockTransfer
        device_gemm_trait::S<1, 0, 2>, device_gemm_trait::S<1, 0, 2>, 2, 8, 8,
        true, 4, device_gemm_trait::S<8, 32, 1>, // B1BlockTransfer
        device_gemm_trait::S<0, 2, 1>, device_gemm_trait::S<0, 2, 1>, 1, 4, 2,
        false,
        1, // CShuffleMXdlPerWavePerShuffle
        4, // CShuffleNXdlPerWavePerShuffle
        device_gemm_trait::S<
            1, 32, 1,
            8>, // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
        DeviceGemmTraits::
            kCShuffleBlockTransferScalarPerVectorNPerBlock, // c_shuffle_block_transfer_scalar_per_vector_n_per_block
        DeviceGemmTraits::kMaskingSpec, // MaskingSpec
        DeviceGemmTraits::kIsDeterministic>;

// type alias for DeviceBatchedMultiheadAttentionBackward_Xdl_CShuffle_V1 of
// head dim = 32
template <typename DeviceGemmTraits>
using DeviceGemmBatchedHeadDim32 = device_op::
    DeviceBatchedMultiheadAttentionBackward_Qloop_Xdl_CShuffle_Light_V1<
        DeviceGemmTraits::kNumDimG, DeviceGemmTraits::kNumDimM,
        DeviceGemmTraits::kNumDimN, DeviceGemmTraits::kNumDimK,
        DeviceGemmTraits::kNumDimO, typename DeviceGemmTraits::InputDataType,
        typename DeviceGemmTraits::OutputDataType,
        typename DeviceGemmTraits::GemmDataType,
        typename DeviceGemmTraits::ZDataType,
        typename DeviceGemmTraits::LSEDataType,
        typename DeviceGemmTraits::DDataType,
        typename DeviceGemmTraits::Acc0BiasDataType,
        typename DeviceGemmTraits::Acc1BiasDataType,
        typename DeviceGemmTraits::AccDataType,
        typename DeviceGemmTraits::ShuffleDataType,
        typename DeviceGemmTraits::QElementOp,
        typename DeviceGemmTraits::KElementOp,
        typename DeviceGemmTraits::Acc0ElementOp,
        typename DeviceGemmTraits::VElementOp,
        typename DeviceGemmTraits::OutElementOp, DeviceGemmTraits::kGemmSpec,
        DeviceGemmTraits::kTensorSpecQ, DeviceGemmTraits::kTensorSpecK,
        DeviceGemmTraits::kTensorSpecV, DeviceGemmTraits::kTensorSpecY, 1, 256,
        128,                                // MPerBlock
        128,                                // NPerBlock
        32,                                 // KPerBlock
        32,                                 // Gemm1NPerBlock
        32,                                 // Gemm1KPerBlock
        64,                                 // Gemm2KPerBlock
        8,                                  // AK1
        8,                                  // BK1
        2,                                  // B1K1
        32,                                 // MPerXDL
        32,                                 // NPerXDL
        4,                                  // MXdlPerWave
        1,                                  // NXdlPerWave
        1,                                  // Gemm1NXdlPerWave
        1,                                  // Gemm2NXdlPerWave
        32, device_gemm_trait::S<4, 64, 1>, // ABlockTransfer
        device_gemm_trait::S<1, 0, 2>, device_gemm_trait::S<1, 0, 2>, 2, 8, 8,
        true, device_gemm_trait::S<4, 64, 1>, // BBlockTransfer
        device_gemm_trait::S<1, 0, 2>, device_gemm_trait::S<1, 0, 2>, 2, 8, 8,
        true, 4, 1, 1,
        device_gemm_trait::S<
            1, 64, 1,
            4>, // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
        DeviceGemmTraits::
            kCShuffleBlockTransferScalarPerVectorNPerBlock, // c_shuffle_block_transfer_scalar_per_vector_n_per_block
        DeviceGemmTraits::kMaskingSpec, // MaskingSpec
        DeviceGemmTraits::kIsDeterministic>;

// type alias for DeviceBatchedMultiheadAttentionBackward_Xdl_CShuffle_V1 of
// head dim = 64
template <typename DeviceGemmTraits>
using DeviceGemmBatchedHeadDim64 = device_op::
    DeviceBatchedMultiheadAttentionBackward_Qloop_Xdl_CShuffle_Light_V1<
        DeviceGemmTraits::kNumDimG, DeviceGemmTraits::kNumDimM,
        DeviceGemmTraits::kNumDimN, DeviceGemmTraits::kNumDimK,
        DeviceGemmTraits::kNumDimO, typename DeviceGemmTraits::InputDataType,
        typename DeviceGemmTraits::OutputDataType,
        typename DeviceGemmTraits::GemmDataType,
        typename DeviceGemmTraits::ZDataType,
        typename DeviceGemmTraits::LSEDataType,
        typename DeviceGemmTraits::DDataType,
        typename DeviceGemmTraits::Acc0BiasDataType,
        typename DeviceGemmTraits::Acc1BiasDataType,
        typename DeviceGemmTraits::AccDataType,
        typename DeviceGemmTraits::ShuffleDataType,
        typename DeviceGemmTraits::QElementOp,
        typename DeviceGemmTraits::KElementOp,
        typename DeviceGemmTraits::Acc0ElementOp,
        typename DeviceGemmTraits::VElementOp,
        typename DeviceGemmTraits::OutElementOp, DeviceGemmTraits::kGemmSpec,
        DeviceGemmTraits::kTensorSpecQ, DeviceGemmTraits::kTensorSpecK,
        DeviceGemmTraits::kTensorSpecV, DeviceGemmTraits::kTensorSpecY, 1, 256,
        128,                                // MPerBlock
        128,                                // NPerBlock
        64,                                 // KPerBlock
        64,                                 // Gemm1NPerBlock
        32,                                 // Gemm1KPerBlock
        32,                                 // Gemm2KPerBlock
        8,                                  // AK1
        8,                                  // BK1
        2,                                  // B1K1
        32,                                 // MPerXDL
        32,                                 // NPerXDL
        4,                                  // MXdlPerWave
        1,                                  // NXdlPerWave
        2,                                  // Gemm1NXdlPerWave
        1,                                  // Gemm2NXdlPerWave
        64, device_gemm_trait::S<4, 64, 1>, // ABlockTransfer
        device_gemm_trait::S<1, 0, 2>, device_gemm_trait::S<1, 0, 2>, 2, 8, 8,
        true, device_gemm_trait::S<4, 64, 1>, // BBlockTransfer
        device_gemm_trait::S<1, 0, 2>, device_gemm_trait::S<1, 0, 2>, 2, 8, 8,
        true, 4,
        1, // CShuffleMXdlPerWavePerShuffle
        2, // CShuffleNXdlPerWavePerShuffle
        device_gemm_trait::S<
            1, 32, 1,
            8>, // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
        DeviceGemmTraits::
            kCShuffleBlockTransferScalarPerVectorNPerBlock, // c_shuffle_block_transfer_scalar_per_vector_n_per_block
        DeviceGemmTraits::kMaskingSpec, // MaskingSpecc
        DeviceGemmTraits::kIsDeterministic>;

// type alias for DeviceBatchedMultiheadAttentionBackward_Xdl_CShuffle_V1 of
// head dim = 128
template <typename DeviceGemmTraits>
using DeviceGemmBatchedHeadDim128 = device_op::
    DeviceBatchedMultiheadAttentionBackward_Qloop_Xdl_CShuffle_Light_V2<
        DeviceGemmTraits::kNumDimG, DeviceGemmTraits::kNumDimM,
        DeviceGemmTraits::kNumDimN, DeviceGemmTraits::kNumDimK,
        DeviceGemmTraits::kNumDimO, typename DeviceGemmTraits::InputDataType,
        typename DeviceGemmTraits::OutputDataType,
        typename DeviceGemmTraits::GemmDataType,
        typename DeviceGemmTraits::ZDataType,
        typename DeviceGemmTraits::LSEDataType,
        typename DeviceGemmTraits::DDataType,
        typename DeviceGemmTraits::Acc0BiasDataType,
        typename DeviceGemmTraits::Acc1BiasDataType,
        typename DeviceGemmTraits::AccDataType,
        typename DeviceGemmTraits::ShuffleDataType,
        typename DeviceGemmTraits::QElementOp,
        typename DeviceGemmTraits::KElementOp,
        typename DeviceGemmTraits::Acc0ElementOp,
        typename DeviceGemmTraits::VElementOp,
        typename DeviceGemmTraits::OutElementOp, DeviceGemmTraits::kGemmSpec,
        DeviceGemmTraits::kTensorSpecQ, DeviceGemmTraits::kTensorSpecK,
        DeviceGemmTraits::kTensorSpecV, DeviceGemmTraits::kTensorSpecY, 1, 256,
        64,                                 // MPerBlock
        128,                                // NPerBlock
        64,                                 // KPerBlock
        128,                                // Gemm1NPerBlock
        32,                                 // Gemm1KPerBlock
        128,                                // Gemm2KPerBlock
        8,                                  // AK1
        8,                                  // BK1
        2,                                  // B1K1
        32,                                 // MPerXDL
        32,                                 // NPerXDL
        2,                                  // MXdlPerWave
        1,                                  // NXdlPerWave
        4,                                  // Gemm1NXdlPerWave
        1,                                  // Gemm2NXdlPerWave
        64, device_gemm_trait::S<4, 64, 1>, // ABlockTransfer
        device_gemm_trait::S<1, 0, 2>, device_gemm_trait::S<1, 0, 2>, 2, 8, 8,
        true, device_gemm_trait::S<4, 64, 1>, // BBlockTransfer
        device_gemm_trait::S<1, 0, 2>, device_gemm_trait::S<1, 0, 2>, 2, 8, 8,
        true, 4, device_gemm_trait::S<8, 32, 1>, // B1BlockTransfer
        device_gemm_trait::S<0, 2, 1>, device_gemm_trait::S<0, 2, 1>, 1, 4, 2,
        false,
        1, // CShuffleMXdlPerWavePerShuffle
        4, // CShuffleNXdlPerWavePerShuffle
        device_gemm_trait::S<
            1, 32, 1,
            8>, // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
        DeviceGemmTraits::
            kCShuffleBlockTransferScalarPerVectorNPerBlock, // c_shuffle_block_transfer_scalar_per_vector_n_per_block
        DeviceGemmTraits::kMaskingSpec, // MaskingSpec
        DeviceGemmTraits::kIsDeterministic>;
} // namespace bwd_device_gemm