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

//qloop head files
#include "ck/tensor_operation/gpu/device/impl/device_grouped_mha_bwd_xdl_cshuffle_qloop_v1.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_mha_bwd_xdl_cshuffle_qloop_v2.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_mha_fwd_xdl_cshuffle_v2.hpp"
//kloop head files
#include "ck/tensor_operation/gpu/device/impl/device_grouped_mha_bwd_xdl_cshuffle_kloop_v1.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_mha_bwd_xdl_cshuffle_kloop_v2.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_mha_fwd_xdl_cshuffle_v1.hpp"

namespace device_gemm_trait {
using Int32 = int;
using Int16 = unsigned short;
using Float32 = float;
using BFloat16 = ck::bhalf_t;
using Float16 = ck::half_t;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using Scale = ck::tensor_operation::element_wise::Scale;
using MaskingSpec = ck::tensor_operation::device::MaskingSpecialization;
using TensorSpec  = ck::tensor_operation::device::TensorSpecialization;
using GemmSpec  = ck::tensor_operation::device::GemmSpecialization;
using Index = ck::index_t;
using AElementOp  = PassThrough;
using B0ElementOp = PassThrough;
using Acc0ElementOp = Scale;
using B1ElementOp = PassThrough;
using CElementOp  = PassThrough;

template <ck::index_t... Is> 
using S = ck::Sequence<Is...>;

static constexpr bool kDeterministic = true;
static constexpr bool kNonDeterministic = false;       
static constexpr auto kMaskingSpecDefault = MaskingSpec::MaskDisabled;                                        
static constexpr auto kMaskingSpecCausal = MaskingSpec::MaskUpperTriangleFromTopLeft;

template <typename InputDataType_,
          MaskingSpec kMaskingSpec_,
          bool kIsDeterministic_>
struct Forward {
  using ADataType        = InputDataType_;
  using B0DataType       = InputDataType_;
  using B1DataType       = InputDataType_;
  using AccDataType      = Float32;
  using CShuffleDataType = Float32;
  using CDataType        = InputDataType_;
  using GemmDataType     = InputDataType_;
  using ZDataType        = Int32;
  using LSEDataType      = Float32;
  using Acc0BiasDataType = ck::Tuple<>;
  using Acc1BiasDataType = ck::Tuple<>;
  using AElementOp       = PassThrough;
  using B0ElementOp      = PassThrough;
  using Acc0ElementOp    = Scale;
  using B1ElementOp      = PassThrough;
  using CElementOp       = PassThrough;

  static constexpr Index kNumDimG = 2;
  static constexpr Index kNumDimM = 1;
  static constexpr Index kNumDimN = 1;
  static constexpr Index kNumDimK = 1;
  static constexpr Index kNumDimO = 1;

  static constexpr auto kGemmSpec = GemmSpec::MNKOPadding;

  static constexpr auto kTensorSpecA  = TensorSpec::Default;
  static constexpr auto kTensorSpecB0 = TensorSpec::Default;
  static constexpr auto kTensorSpecB1 = TensorSpec::Default;
  static constexpr auto kTensorSpecC  = TensorSpec::Default;

  static constexpr auto kMaskingSpec = kMaskingSpec_;
  static constexpr bool kIsDeterministic = kIsDeterministic_;
}; // device gemm traits forward

template <typename InputDataType_,
          typename OutputDataType_,
          typename GemmDataType_,
          typename ZDataType_,
          Index kCShuffleBlockTransferScalarPerVectorNPerBlock_,
          MaskingSpec kMaskingSpec_,
          bool kIsDeterministic_>
struct Backward {
  using InputDataType    = InputDataType_;
  using OutputDataType   = OutputDataType_;
  using GemmDataType     = GemmDataType_;
  using ZDataType        = ZDataType_;
  using QkvElementOp     = PassThrough;
  using YElementOp       = PassThrough;
  using AccDataType      = Float32;
  using ShuffleDataType  = Float32;
  using LSEDataType      = Float32;
  using Acc0BiasDataType = ck::Tuple<>;
  using Acc1BiasDataType = ck::Tuple<>;
  using Acc0ElementOp    = Scale;

  static constexpr Index kNumDimG = 2;
  static constexpr Index kNumDimM = 1;
  static constexpr Index kNumDimN = 1;
  static constexpr Index kNumDimK = 1;
  static constexpr Index kNumDimO = 1;      

  static constexpr Index kCShuffleBlockTransferScalarPerVectorNPerBlock = kCShuffleBlockTransferScalarPerVectorNPerBlock_;                                      

  static constexpr auto kGemmSpec = GemmSpec::MNKOPadding;

  static constexpr auto kTensorSpecQ = TensorSpec::Default;
  static constexpr auto kTensorSpecK = TensorSpec::Default;
  static constexpr auto kTensorSpecV = TensorSpec::Default;
  static constexpr auto kTensorSpecY = TensorSpec::Default;
  
  static constexpr auto kMaskingSpec = kMaskingSpec_;
  static constexpr bool kIsDeterministic = kIsDeterministic_;
}; // device gemm traits backward
} // namespace device_gemm_trait
