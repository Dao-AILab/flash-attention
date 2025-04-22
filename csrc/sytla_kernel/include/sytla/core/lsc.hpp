#pragma once

#include "sytla/core/common.hpp"

namespace sytla {

extern "C" {
// Load message caching control
enum LSC_LDCC {
  LSC_LDCC_DEFAULT = 0,
  LSC_LDCC_L1UC_L3UC = 1, // Override to L1 uncached and L3 uncached
  LSC_LDCC_L1UC_L3C = 2,  // Override to L1 uncached and L3 cached
  LSC_LDCC_L1C_L3UC = 3,  // Override to L1 cached and L3 uncached
  LSC_LDCC_L1C_L3C = 4,   // Override to L1 cached and L3 cached
  LSC_LDCC_L1S_L3UC = 5,  // Override to L1 streaming load and L3 uncached
  LSC_LDCC_L1S_L3C = 6,   // Override to L1 streaming load and L3 cached
  LSC_LDCC_L1IAR_L3C = 7, // Override to L1 invalidate-after-read, and L3 cached
};
// Store message caching control (also used for atomics)
enum LSC_STCC {
  LSC_STCC_DEFAULT = 0,
  LSC_STCC_L1UC_L3UC = 1, // Override to L1 uncached and L3 uncached
  LSC_STCC_L1UC_L3WB = 2, // Override to L1 uncached and L3 written back
  LSC_STCC_L1WT_L3UC = 3, // Override to L1 written through and L3 uncached
  LSC_STCC_L1WT_L3WB = 4, // Override to L1 written through and L3 written back
  LSC_STCC_L1S_L3UC = 5,  // Override to L1 streaming and L3 uncached
  LSC_STCC_L1S_L3WB = 6,  // Override to L1 streaming and L3 written back
  LSC_STCC_L1WB_L3WB = 7, // Override to L1 written through and L3 written back
};
}

#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
namespace intrinsic {

//==-------------------------- SubGroupBlockLoad ---------------------------==//

template <typename RetType, bool kTranspose, bool kVnni, int kLscDataSize, int kBlockWidth,
          int kBlockHeight, int kArrLen>
struct SubGroupBlockLoad;

// Macro to generate functors for using intrinsic functions of loading
//   RET    : function's return type, such as short4
//   SUFFIX : function's suffix, such as u16_m4k16v1
//   TRANS  : whether transpose is required
//   VNNI   : whether VNNI trasform is required
//   LSC_DS : the size of data type used in the function
//   HEIGHT : the block height
//   WIDTH  : the block width
//   ALEN   : the array length
//
#define __SUB_GROUP_BLOCK_LOAD(RET, SUFFIX, TRANS, VNNI, LSC_DS, HEIGHT, WIDTH, ALEN)              \
  extern "C" {                                                                                     \
  extern SYCL_EXTERNAL RET __builtin_IB_subgroup_block_read_ap_##SUFFIX(int *addrPayload,          \
                                                                        const int immX,            \
                                                                        const int immY,            \
                                                                        enum LSC_LDCC cacheOpt);   \
  }                                                                                                \
  template <>                                                                                      \
  struct SubGroupBlockLoad<RET, TRANS, VNNI, LSC_DS, WIDTH, HEIGHT, ALEN> {                        \
    INLINE static void call(RET &ret, int *addrPayload, int immX, int immY, LSC_LDCC cacheOpt) {   \
      ret = __builtin_IB_subgroup_block_read_ap_##SUFFIX(addrPayload, immX, immY, cacheOpt);       \
    }                                                                                              \
  };

// intrinsics for loading
// 2Bytes -> short
__SUB_GROUP_BLOCK_LOAD(short, u16_m1k16v1, false, false, 2, 1, 16, 1)
__SUB_GROUP_BLOCK_LOAD(short2, u16_m2k16v1, false, false, 2, 2, 16, 1)
__SUB_GROUP_BLOCK_LOAD(short4, u16_m2k16v2, false, false, 2, 2, 16, 2)
__SUB_GROUP_BLOCK_LOAD(short4, u16_m4k16v1, false, false, 2, 4, 16, 1)
__SUB_GROUP_BLOCK_LOAD(short8, u16_m4k16v2, false, false, 2, 4, 16, 2)
__SUB_GROUP_BLOCK_LOAD(short8, u16_m8k16v1, false, false, 2, 8, 16, 1)
__SUB_GROUP_BLOCK_LOAD(short16, u16_m8k16v2, false, false, 2, 8, 16, 2)
__SUB_GROUP_BLOCK_LOAD(short16, u16_m16k16v1, false, false, 2, 16, 16, 1)
__SUB_GROUP_BLOCK_LOAD(short32, u16_m16k16v2, false, false, 2, 16, 16, 2)
__SUB_GROUP_BLOCK_LOAD(short32, u16_m32k16v1, false, false, 2, 32, 16, 1)
__SUB_GROUP_BLOCK_LOAD(short64, u16_m32k16v2, false, false, 2, 32, 16, 2)
// 4Bytes -> int
__SUB_GROUP_BLOCK_LOAD(int, u32_m1k16v1, false, false, 4, 1, 16, 1)
__SUB_GROUP_BLOCK_LOAD(int2, u32_m1k16v2, false, false, 4, 1, 16, 2)
__SUB_GROUP_BLOCK_LOAD(int2, u32_m2k16v1, false, false, 4, 2, 16, 1)
__SUB_GROUP_BLOCK_LOAD(int4, u32_m2k16v2, false, false, 4, 2, 16, 2)
__SUB_GROUP_BLOCK_LOAD(int4, u32_m4k16v1, false, false, 4, 4, 16, 1)
__SUB_GROUP_BLOCK_LOAD(int8, u32_m4k16v2, false, false, 4, 4, 16, 2)
__SUB_GROUP_BLOCK_LOAD(int8, u32_m8k16v1, false, false, 4, 8, 16, 1)
__SUB_GROUP_BLOCK_LOAD(int16, u32_m8k16v2, false, false, 4, 8, 16, 2)
__SUB_GROUP_BLOCK_LOAD(int16, u32_m16k16v1, false, false, 4, 16, 16, 1)
__SUB_GROUP_BLOCK_LOAD(int32, u32_m16k16v2, false, false, 4, 16, 16, 2)
__SUB_GROUP_BLOCK_LOAD(int32, u32_m32k16v1, false, false, 4, 32, 16, 1)
__SUB_GROUP_BLOCK_LOAD(int64, u32_m32k16v2, false, false, 4, 32, 16, 2)

// intrinsics for transformed loading
// 1Byte X m4 -> int
__SUB_GROUP_BLOCK_LOAD(int4, transform_u8_m16k16v1, false, true, 1, 16, 16, 1)
__SUB_GROUP_BLOCK_LOAD(int8, transform_u8_m16k16v2, false, true, 1, 16, 16, 2)
__SUB_GROUP_BLOCK_LOAD(int8, transform_u8_m32k16v1, false, true, 1, 32, 16, 1)
__SUB_GROUP_BLOCK_LOAD(int16, transform_u8_m32k16v2, false, true, 1, 32, 16, 2)
// 2Bytes X m2 -> int
__SUB_GROUP_BLOCK_LOAD(int4, transform_u16_m8k16v1, false, true, 2, 8, 16, 1)
__SUB_GROUP_BLOCK_LOAD(int8, transform_u16_m8k16v2, false, true, 2, 8, 16, 2)
__SUB_GROUP_BLOCK_LOAD(int8, transform_u16_m16k16v1, false, true, 2, 16, 16, 1)
__SUB_GROUP_BLOCK_LOAD(int16, transform_u16_m16k16v2, false, true, 2, 16, 16, 2)
__SUB_GROUP_BLOCK_LOAD(int16, transform_u16_m32k16v1, false, true, 2, 32, 16, 1)
__SUB_GROUP_BLOCK_LOAD(int32, transform_u16_m32k16v2, false, true, 2, 32, 16, 2)

// intrinsics for transposed loading
__SUB_GROUP_BLOCK_LOAD(int4, transpose_u32_m16k4v1, true, true, 4, 16, 4, 1)
__SUB_GROUP_BLOCK_LOAD(int8, transpose_u32_m16k8v1, true, true, 4, 16, 8, 1)
#undef __SUB_GROUP_BLOCK_LOAD

//==------------------------- SubGroupBlockStore ---------------------------==//

template <typename ValType, int kLscDataSize, int kBlockWidth, int kBlockHeight>
struct SubGroupBlockStore;

// Macro to generate functors for using intrinsic functions of storing
//   VALT   : function's input value type, such as short4
//   SUFFIX : function's suffix, such as u16_m4k16v1
//   LSC_DS : the size of data type used in the function
//   HEIGHT : the block height
//   WIDTH  : the block width
//
#define __SUB_GROUP_BLOCK_STORE(VALT, SUFFIX, LSC_DS, HEIGHT, WIDTH)                               \
  extern "C" {                                                                                     \
  extern SYCL_EXTERNAL void __builtin_IB_subgroup_block_write_ap_##SUFFIX(                         \
      int *addrPayload, const int immX, const int immY, VALT val, enum LSC_STCC cacheOpt);         \
  }                                                                                                \
  template <>                                                                                      \
  struct SubGroupBlockStore<VALT, LSC_DS, WIDTH, HEIGHT> {                                         \
    INLINE static void call(VALT &val, int *addrPayload, int immX, int immY, LSC_STCC cacheOpt) {  \
      __builtin_IB_subgroup_block_write_ap_##SUFFIX(addrPayload, immX, immY, val, cacheOpt);       \
    }                                                                                              \
  };

// intrinsics for storing
// short -> 1Byte
__SUB_GROUP_BLOCK_STORE(short, u8_m1k32v1, 1, 1, 32)
__SUB_GROUP_BLOCK_STORE(short2, u8_m2k32v1, 1, 2, 32)
__SUB_GROUP_BLOCK_STORE(short4, u8_m4k32v1, 1, 4, 32)
__SUB_GROUP_BLOCK_STORE(short8, u8_m8k32v1, 1, 8, 32)
// short -> 2Bytes
__SUB_GROUP_BLOCK_STORE(short, u16_m1k16v1, 2, 1, 16)
__SUB_GROUP_BLOCK_STORE(short2, u16_m2k16v1, 2, 2, 16)
__SUB_GROUP_BLOCK_STORE(short4, u16_m4k16v1, 2, 4, 16)
__SUB_GROUP_BLOCK_STORE(short8, u16_m8k16v1, 2, 8, 16)
// int -> 4Bytes
__SUB_GROUP_BLOCK_STORE(int, u32_m1k16v1, 4, 1, 16)
__SUB_GROUP_BLOCK_STORE(int2, u32_m2k16v1, 4, 2, 16)
__SUB_GROUP_BLOCK_STORE(int4, u32_m4k16v1, 4, 4, 16)
__SUB_GROUP_BLOCK_STORE(int8, u32_m8k16v1, 4, 8, 16)
#undef __SUB_GROUP_BLOCK_STORE

//==------------------------ SubGroupBlockPrefetch -------------------------==//

template <int kLscDataSize, int kBlockWidth, int kBlockHeight>
struct SubGroupBlockPrefetch;

// Macro to generate functors for using intrinsic functions of prefetching
//   SUFFIX : function's suffix, such as u8_m4k32v1
//   LSC_DS : the size of data type used in the function
//   HEIGHT : the block height
//   WIDTH  : the block width
//
#define __SUB_GROUP_BLOCK_PREFETCH(SUFFIX, LSC_DS, HEIGHT, WIDTH)                                  \
  extern "C" {                                                                                     \
  extern SYCL_EXTERNAL void __builtin_IB_subgroup_block_read_ap_prefetch_##SUFFIX(                 \
      int *addrPayload, const int immX, const int immY, enum LSC_LDCC cacheOpt);                   \
  }                                                                                                \
  template <>                                                                                      \
  struct SubGroupBlockPrefetch<LSC_DS, WIDTH, HEIGHT> {                                            \
    INLINE static void call(int *addrPayload, int immX, int immY, LSC_LDCC cacheOpt) {             \
      __builtin_IB_subgroup_block_read_ap_prefetch_##SUFFIX(addrPayload, immX, immY, cacheOpt);    \
    }                                                                                              \
  };

// intrinsics for prefetching
__SUB_GROUP_BLOCK_PREFETCH(u8_m1k32v1, 1, 1, 32)
__SUB_GROUP_BLOCK_PREFETCH(u8_m2k32v1, 1, 2, 32)
__SUB_GROUP_BLOCK_PREFETCH(u8_m4k32v1, 1, 4, 32)
__SUB_GROUP_BLOCK_PREFETCH(u8_m8k32v1, 1, 8, 32)
__SUB_GROUP_BLOCK_PREFETCH(u8_m8k16v1, 1, 8, 16)
__SUB_GROUP_BLOCK_PREFETCH(u8_m16k32v1, 1, 16, 32)
__SUB_GROUP_BLOCK_PREFETCH(u8_m32k32v1, 1, 32, 32)
__SUB_GROUP_BLOCK_PREFETCH(u8_m1k64v1, 1, 1, 64)
__SUB_GROUP_BLOCK_PREFETCH(u8_m2k64v1, 1, 2, 64)
__SUB_GROUP_BLOCK_PREFETCH(u8_m4k64v1, 1, 4, 64)
__SUB_GROUP_BLOCK_PREFETCH(u8_m8k64v1, 1, 8, 64)
__SUB_GROUP_BLOCK_PREFETCH(u8_m16k64v1, 1, 16, 64)
__SUB_GROUP_BLOCK_PREFETCH(u8_m32k64v1, 1, 32, 64)
#undef __SUB_GROUP_BLOCK_PREFETCH

} // namespace intrinsic
#endif

} // namespace sytla