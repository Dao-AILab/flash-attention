// Inspired by
// https://github.com/NVIDIA/DALI/blob/main/include/dali/core/static_switch.h
// and https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Dispatch.h

#pragma once

/// @param COND       - a boolean expression to switch by
/// @param CONST_NAME - a name given for the constexpr bool variable.
/// @param ...       - code to execute for true and false
///
/// Usage:
/// ```
/// BOOL_SWITCH(flag, BoolConst, [&] {
///     some_function<BoolConst>(...);
/// });
/// ```

#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
  [&] {                                         \
    if (COND) {                                 \
      constexpr static bool CONST_NAME = true;  \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static bool CONST_NAME = false; \
      return __VA_ARGS__();                     \
    }                                           \
  }()

#ifdef FLASHATTENTION_DISABLE_DROPOUT
  #define DROPOUT_SWITCH(COND, CONST_NAME, ...) \
  [&] {                                         \
    constexpr static bool CONST_NAME = false;   \
    return __VA_ARGS__();                       \
  }()
#else
  #define DROPOUT_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_ALIBI
  #define ALIBI_SWITCH(COND, CONST_NAME, ...)   \
  [&] {                                         \
    constexpr static bool CONST_NAME = false;   \
    return __VA_ARGS__();                       \
  }()
#else
  #define ALIBI_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_UNEVEN_K
  #define EVENK_SWITCH(COND, CONST_NAME, ...)   \
  [&] {                                         \
    constexpr static bool CONST_NAME = true;    \
    return __VA_ARGS__();                       \
  }()
#else
  #define EVENK_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_LOCAL
  #define LOCAL_SWITCH(COND, CONST_NAME, ...)   \
  [&] {                                         \
    constexpr static bool CONST_NAME = false;    \
    return __VA_ARGS__();                       \
  }()
#else
  #define LOCAL_SWITCH BOOL_SWITCH
#endif

#define FP16_SWITCH(COND, ...)               \
  [&] {                                      \
    if (COND) {                              \
      using elem_type = cutlass::half_t;     \
      return __VA_ARGS__();                  \
    } else {                                 \
      using elem_type = cutlass::bfloat16_t; \
      return __VA_ARGS__();                  \
    }                                        \
  }()

#define HEADDIM_SWITCH(HEADDIMQ, HEADDIMV, ...)   \
  [&] {                                       \
    if (HEADDIMQ <= 32) {                     \
      constexpr static int kHeadDimQ = 32;    \
      constexpr static int kHeadDimV = 32;    \
      return __VA_ARGS__();                   \
    } else if (HEADDIMQ <= 64) {              \
      constexpr static int kHeadDimQ = 64;    \
      constexpr static int kHeadDimV = 64;    \
      return __VA_ARGS__();                   \
    } else if (HEADDIMQ <= 96) {              \
      constexpr static int kHeadDimQ = 96;    \
      constexpr static int kHeadDimV = 96;    \
      return __VA_ARGS__();                   \
    } else if (HEADDIMQ <= 128) {             \
      constexpr static int kHeadDimQ = 128;   \
      constexpr static int kHeadDimV = 128;   \
      return __VA_ARGS__();                   \
    } else if (HEADDIMQ <= 160) {             \
      constexpr static int kHeadDimQ = 160;   \
      constexpr static int kHeadDimV = 160;   \
      return __VA_ARGS__();                   \
    } else if (HEADDIMQ <= 192) {             \
      constexpr static int kHeadDimQ = 192;   \
      if (HEADDIMV <= 128) {                  \
        constexpr static int kHeadDimV = 128; \
        return __VA_ARGS__();                 \
      } else if (HEADDIMV <= 192) {           \
        constexpr static int kHeadDimV = 192; \
        return __VA_ARGS__();                 \
      }                                       \
    } else if (HEADDIMQ <= 224) {             \
      constexpr static int kHeadDimQ = 224;   \
      constexpr static int kHeadDimV = 224;   \
      return __VA_ARGS__();                   \
    } else if (HEADDIMQ <= 256) {             \
      constexpr static int kHeadDimQ = 256;   \
      constexpr static int kHeadDimV = 256;   \
      return __VA_ARGS__();                   \
    }                                         \
  }()
