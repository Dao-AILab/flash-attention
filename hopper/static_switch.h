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
//

#define BOOL_SWITCH(COND, CONST_NAME, ...)                                                       \
  [&] {                                                                                          \
    if (COND) {                                                                                  \
      constexpr static bool CONST_NAME = true;                                                   \
      return __VA_ARGS__();                                                                      \
    } else {                                                                                     \
      constexpr static bool CONST_NAME = false;                                                  \
      return __VA_ARGS__();                                                                      \
    }                                                                                            \
  }()

#ifdef FLASHATTENTION_DISABLE_LOCAL
  #define CAUSAL_LOCAL_SWITCH(CAUSAL_COND, LOCAL_COND, CAUSAL_CONST_NAME, LOCAL_CONST_NAME, ...) \
    [&] {                                                                                        \
      constexpr static bool LOCAL_CONST_NAME = false;                                            \
      if (CAUSAL_COND) {                                                                         \
        constexpr static bool CAUSAL_CONST_NAME = true;                                          \
        return __VA_ARGS__();                                                                    \
      } else {                                                                                   \
        constexpr static bool CAUSAL_CONST_NAME = false;                                         \
        return __VA_ARGS__();                                                                    \
      }                                                                                          \
    }()
#else
  #define CAUSAL_LOCAL_SWITCH(CAUSAL_COND, LOCAL_COND, CAUSAL_CONST_NAME, LOCAL_CONST_NAME, ...) \
    [&] {                                                                                        \
      if (CAUSAL_COND) {                                                                         \
        constexpr static bool CAUSAL_CONST_NAME = true;                                          \
        constexpr static bool LOCAL_CONST_NAME = false;                                          \
        return __VA_ARGS__();                                                                    \
      } else if (LOCAL_COND) {                                                                   \
        constexpr static bool CAUSAL_CONST_NAME = false;                                         \
        constexpr static bool LOCAL_CONST_NAME = true;                                           \
        return __VA_ARGS__();                                                                    \
      } else {                                                                                   \
        constexpr static bool CAUSAL_CONST_NAME = false;                                         \
        constexpr static bool LOCAL_CONST_NAME = false;                                          \
        return __VA_ARGS__();                                                                    \
      }                                                                                          \
    }()
#endif

#ifdef FLASHATTENTION_DISABLE_SOFTCAP
  #define SOFTCAP_SWITCH(COND, CONST_NAME, ...)                                                  \
  [&] {                                                                                          \
    constexpr static bool CONST_NAME = false;                                                    \
    return __VA_ARGS__();                                                                        \
  }()
#else
  #define SOFTCAP_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_PAGEDKV
  #define PAGEDKV_SWITCH(COND, CONST_NAME, ...)                                                  \
  [&] {                                                                                          \
    constexpr static bool CONST_NAME = false;                                                    \
    return __VA_ARGS__();                                                                        \
  }()
#else
  #define PAGEDKV_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_SPLIT
  #define SPLIT_SWITCH(COND, CONST_NAME, ...)                                                    \
  [&] {                                                                                          \
    constexpr static bool CONST_NAME = false;                                                    \
    return __VA_ARGS__();                                                                        \
  }()
#else
  #define SPLIT_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_APPENDKV
  #define APPENDKV_SWITCH(COND, CONST_NAME, ...)                                                 \
  [&] {                                                                                          \
    constexpr static bool CONST_NAME = false;                                                    \
    return __VA_ARGS__();                                                                        \
  }()
#else
  #define APPENDKV_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_PACKGQA
  #define PACKGQA_SWITCH(COND, CONST_NAME, ...)                                                  \
  [&] {                                                                                          \
    constexpr static bool CONST_NAME = false;                                                    \
    return __VA_ARGS__();                                                                        \
  }()
#else
  #define PACKGQA_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_VARLEN
  #define VARLEN_SWITCH(COND, CONST_NAME, ...)                                                   \
  [&] {                                                                                          \
    constexpr static bool CONST_NAME = false;                                                    \
    return __VA_ARGS__();                                                                        \
  }()
#else
  #define VARLEN_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_CLUSTER
  #define CLUSTER_SWITCH(COND, CONST_NAME, ...)                                                  \
  [&] {                                                                                          \
    constexpr static bool CONST_NAME = false;                                                    \
    return __VA_ARGS__();                                                                        \
  }()
#else
  #define CLUSTER_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_SM8x
  #define ARCH_SWITCH(ARCH, ARCH_NAME, ...)                                                      \
  [&] {                                                                                          \
    constexpr static int ARCH_NAME = 90;                                                         \
    return __VA_ARGS__();                                                                        \
  }()
#else
  #define ARCH_SWITCH(ARCH, ARCH_NAME, ...)                                                      \
  [&] {                                                                                          \
    if (ARCH == 86 || ARCH == 89) {                                                              \
      constexpr static int ARCH_NAME = 86;                                                       \
      return __VA_ARGS__();                                                                      \
    } else if (ARCH < 90) {                                                                      \
      constexpr static int ARCH_NAME = 80;                                                       \
      return __VA_ARGS__();                                                                      \
    } else {                                                                                     \
      constexpr static int ARCH_NAME = 90;                                                       \
      return __VA_ARGS__();                                                                      \
    }                                                                                            \
  }()
#endif

#ifndef FLASHATTENTION_ENABLE_VCOLMAJOR
  #define VCOLMAJOR_SWITCH(COND, CONST_NAME, ...)                                                \
  [&] {                                                                                          \
    constexpr static bool CONST_NAME = false;                                                    \
    return __VA_ARGS__();                                                                        \
  }()
#else
  #define VCOLMAJOR_SWITCH BOOL_SWITCH
#endif

#define HEADDIM_SWITCH(HEADDIM, ...)                                                             \
  [&] {                                                                                          \
    if (HEADDIM == 64) {                                                                         \
      constexpr static int kHeadSize = 64;                                                       \
      return __VA_ARGS__();                                                                      \
    } else if (HEADDIM == 96) {                                                                  \
      constexpr static int kHeadSize = 96;                                                       \
      return __VA_ARGS__();                                                                      \
    } else if (HEADDIM == 128) {                                                                 \
      constexpr static int kHeadSize = 128;                                                      \
      return __VA_ARGS__();                                                                      \
    } else if (HEADDIM == 96) {                                                                  \
      constexpr static int kHeadSize = 96;                                                       \
      return __VA_ARGS__();                                                                      \
    } else if (HEADDIM == 256) {                                                                 \
      constexpr static int kHeadSize = 256;                                                      \
      return __VA_ARGS__();                                                                      \
    }                                                                                            \
  }()

#define NUM_WARP_SWITCH(VALUE, CONST_NAME, ...)                                                  \
  [&] {                                                                                          \
    if (VALUE <= 1) {                                                                            \
      constexpr static int CONST_NAME = 1;                                                       \
      return __VA_ARGS__();                                                                      \
    } else if (VALUE <= 2) {                                                                     \
      constexpr static int CONST_NAME = 2;                                                       \
      return __VA_ARGS__();                                                                      \
    } else if (VALUE <= 4) {                                                                     \
      constexpr static int CONST_NAME = 4;                                                       \
      return __VA_ARGS__();                                                                      \
    } else if (VALUE <= 8) {                                                                     \
      constexpr static int CONST_NAME = 8;                                                       \
      return __VA_ARGS__();                                                                      \
    } else if (VALUE <= 16) {                                                                    \
      constexpr static int CONST_NAME = 16;                                                      \
      return __VA_ARGS__();                                                                      \
    } else {                                                                                     \
      constexpr static int CONST_NAME = 32;                                                      \
      return __VA_ARGS__();                                                                      \
    }                                                                                            \
  }()
