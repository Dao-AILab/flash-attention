// BSD 3 Clause
// Copyright 2023 Advanced Micro Devices, Inc.
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


#pragma once

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/torch.h>

#include "ck/ck.hpp"

#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/tensor_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_multihead_attention_backward_xdl_cshuffle_v1.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_multihead_attention_backward_xdl_cshuffle_v2.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_multihead_attention_forward_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_batched_gemm.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_softmax.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_dropout.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////

#define NEW_UNPACK (TORCH_VERSION_MAJOR * 10000 + TORCH_VERSION_MINOR * 100 + TORCH_VERSION_PATCH) > 11300


#define FMHA_CHECK_HIP( call )                                                                     \
    do {                                                                                           \
        hipError_t status_ = call;                                                                 \
        if( status_ != hipSuccess ) {                                                              \
            fprintf( stderr,                                                                       \
                     "HIP error (%s:%d): %s\n",                                                    \
                     __FILE__,                                                                     \
                     __LINE__,                                                                     \
                     hipGetErrorString( status_ ) );                                               \
            exit( 1 );                                                                             \
        }                                                                                          \
    } while( 0 )

////////////////////////////////////////////////////////////////////////////////////////////////////

enum DataType {kFloat16, kFloat32, kBFloat16, kInt32, kInt8};

////////////////////////////////////////////////////////////////////////////////////////////////////

//static inline void set_alpha( uint32_t &alpha, float norm, Data_type dtype ) {
//    if( dtype == DATA_TYPE_FP16 ) {
//        ck::half_t x = ck::type_convert<ck::half_t>( norm );
//        uint16_t h = reinterpret_cast<const uint16_t &>( x );
//        ushort2 h2 = { h, h };
//        alpha = reinterpret_cast<const uint32_t &>( h2 );
//    } else if( dtype == DATA_TYPE_BF16 ) {
//        ck::bhalf_t x = ck::type_convert<ck::bhalf_t>( norm );
//        uint16_t h = reinterpret_cast<const uint16_t &>( x );
//        ushort2 h2 = { h, h };
//        alpha = reinterpret_cast<const uint32_t &>( h2 );
//    } else if( dtype == DATA_TYPE_FP32 ) {
//        alpha = reinterpret_cast<const uint32_t &>( norm );
//    } else if( dtype == DATA_TYPE_INT32 ) {
//        int32_t inorm = static_cast<int32_t>( norm );
//        alpha = reinterpret_cast<const uint32_t &>( inorm );
//    } else {
//        assert( false );
//    }
//}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline size_t get_size_in_bytes( size_t n, auto dtype ) {
    if(dtype == torch::kFloat32){
        return n * 4;
    }else if(dtype == torch::kBFloat16){
        return n * 2;
    }else if(dtype == torch::kFloat16){
        return n * 2;
    }else if(dtype == torch::kInt32){
        return n * 4;
    }else if(dtype == torch::kInt8){
        return n;
    }
    return 0;
}


static std::tuple<uint64_t, uint64_t> unpack(at::PhiloxCudaState arg) {
  if (arg.captured_) {
    #if NEW_UNPACK
        return std::make_tuple(static_cast<uint64_t>(*arg.seed_.ptr), static_cast<uint64_t>(*(arg.offset_.ptr) + arg.offset_intragraph_));
    #else
        return std::make_tuple(arg.seed_, static_cast<uint64_t>(*(arg.offset_.ptr) + arg.offset_intragraph_));
    #endif
  } else {
    #if NEW_UNPACK
        return std::make_tuple(arg.seed_.val, arg.offset_.val);
    #else
        return std::make_tuple(arg.seed_, arg.offset_.val);
    #endif
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
