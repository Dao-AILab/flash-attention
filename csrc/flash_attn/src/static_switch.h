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

#ifdef FLASHATTENTION_DISABLE_SOFTCAP
  #define SOFTCAP_SWITCH(COND, CONST_NAME, ...)   \
  [&] {                                         \
    constexpr static bool CONST_NAME = false;    \
    return __VA_ARGS__();                       \
  }()
#else
  #define SOFTCAP_SWITCH BOOL_SWITCH
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


#define DTYPE(COND, cond, dtype, ...) \
  else if (COND == cond) {using elem_type = dtype; return __VA_ARGS__();}

#if defined(DTYPE_FP16)
#define FP16_SWITCH(COND, ...)  [&] { if(false){} DTYPE(COND, true, cutlass::half_t, __VA_ARGS__)}()

#elif defined(DTYPE_BF16)
#define FP16_SWITCH(COND, ...)  [&] { if(false){} DTYPE(COND, false, cutlass::bfloat16_t, __VA_ARGS__)}()

#else
#define FP16_SWITCH(COND, ...)                             \
  [&] {                                                    \
    if (false) {}                                          \
      DTYPE(COND, true, cutlass::half_t, __VA_ARGS__)      \
      DTYPE(COND, false, cutlass::bfloat16_t, __VA_ARGS__) \
  }()
#endif


#define HEAD(HEADDIM, dim, ...) \
    else if (HEADDIM <= dim) {constexpr static int kHeadDim = dim; return __VA_ARGS__();} \

#if defined(HEADDIM_32)
#define HEADDIM_SWITCH(HEADDIM, ...)  [&]{ if(false){} HEAD(HEADDIM, 32, __VA_ARGS__)}()

#elif defined(HEADDIM_64)
#define HEADDIM_SWITCH(HEADDIM, ...)  [&]{ if(false){} HEAD(HEADDIM, 64, __VA_ARGS__)}()

#elif defined(HEADDIM_96)
#define HEADDIM_SWITCH(HEADDIM, ...)  [&]{ if(false){} HEAD(HEADDIM, 96, __VA_ARGS__)}()

#elif defined(HEADDIM_128)
#define HEADDIM_SWITCH(HEADDIM, ...)  [&]{ if(false){} HEAD(HEADDIM, 128, __VA_ARGS__)}()

#elif defined(HEADDIM_160)
#define HEADDIM_SWITCH(HEADDIM, ...)  [&]{ if(false){} HEAD(HEADDIM, 160, __VA_ARGS__)}()

#elif defined(HEADDIM_192)
#define HEADDIM_SWITCH(HEADDIM, ...)  [&]{ if(false){} HEAD(HEADDIM, 192, __VA_ARGS__)}()

#elif defined(HEADDIM_256)
#define HEADDIM_SWITCH(HEADDIM, ...)  [&]{ if(false){} HEAD(HEADDIM, 256, __VA_ARGS__)}()

#else
#define HEADDIM_SWITCH(HEADDIM, ...)   \
  [&] { \
    if (false) {} \
    HEAD(HEADDIM, 32, __VA_ARGS__) \
    HEAD(HEADDIM, 64, __VA_ARGS__) \
    HEAD(HEADDIM, 96, __VA_ARGS__) \
    HEAD(HEADDIM, 128, __VA_ARGS__) \
    HEAD(HEADDIM, 160, __VA_ARGS__) \
    HEAD(HEADDIM, 192, __VA_ARGS__) \
    HEAD(HEADDIM, 256, __VA_ARGS__) \
  }()
#endif