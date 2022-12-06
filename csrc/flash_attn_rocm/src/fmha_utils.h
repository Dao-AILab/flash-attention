
#pragma once

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "hip/hip_runtime.h"
//#include "ck/utility/data_type.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////

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

enum Data_type { DATA_TYPE_FP16, DATA_TYPE_BF16, DATA_TYPE_FP32, DATA_TYPE_INT32, DATA_TYPE_INT8 };

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

static inline size_t get_size_in_bytes( size_t n, Data_type dtype ) {
    switch( dtype ) {
    case DATA_TYPE_FP32:
        return n * 4;
    case DATA_TYPE_FP16:
        return n * 2;
    case DATA_TYPE_BF16:
        return n * 2;
    case DATA_TYPE_INT32:
        return n * 4;
    case DATA_TYPE_INT8:
        return n;
    default:
        assert( false );
        return 0;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

