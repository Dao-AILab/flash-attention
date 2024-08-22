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

#define QKHEADDIM_VHEADDIM_SWITCH(QKHEADDIM, VHEADDIM, ...)   \
  [&] {                                                        \
    if (QKHEADDIM <= 32 && VHEADDIM <= 32) {                     \
      constexpr static int kQKHeadDim = 32;                    \
      constexpr static int kVHeadDim = 32;                   \
      return __VA_ARGS__();                                    \
    } else if (QKHEADDIM <= 32 && VHEADDIM <= 64) {             \
      constexpr static int kQKHeadDim = 32;                    \
      constexpr static int kVHeadDim = 64;                   \
      return __VA_ARGS__();                                    \
    } else if (QKHEADDIM <= 64 && VHEADDIM <= 64) {             \
      constexpr static int kQKHeadDim = 64;                    \
      constexpr static int kVHeadDim = 64;                   \
      return __VA_ARGS__();                                    \
    } else if (QKHEADDIM <= 64 && VHEADDIM <= 128) {           \
      constexpr static int kQKHeadDim = 64;                    \
      constexpr static int kVHeadDim = 128;                  \
      return __VA_ARGS__();                                    \
    } else if (QKHEADDIM <= 96 && VHEADDIM <= 96) {            \
      constexpr static int kQKHeadDim = 96;                    \
      constexpr static int kVHeadDim = 96;                   \
      return __VA_ARGS__();                                    \
    } else if (QKHEADDIM <= 96 && VHEADDIM <= 192) {           \
      constexpr static int kQKHeadDim = 96;                    \
      constexpr static int kVHeadDim = 192;                  \
      return __VA_ARGS__();                                    \
    } else if (QKHEADDIM <= 128 && VHEADDIM <= 128) {             \
      constexpr static int kQKHeadDim = 128;                   \
      constexpr static int kVHeadDim = 128;                  \
      return __VA_ARGS__();                                    \
    } else if (QKHEADDIM <= 128 && VHEADDIM <= 256) {           \
      constexpr static int kQKHeadDim = 128;                   \
      constexpr static int kVHeadDim = 256;                  \
      return __VA_ARGS__();                                    \
    } else if (QKHEADDIM <= 160 && VHEADDIM <= 160) {            \
      constexpr static int kQKHeadDim = 160;                  \
      constexpr static int kVHeadDim = 160;                  \
      return __VA_ARGS__();                                    \
    } else if (QKHEADDIM <= 192 && VHEADDIM <= 192) {          \
      constexpr static int kQKHeadDim = 192;                  \
      constexpr static int kVHeadDim = 192;                  \
      return __VA_ARGS__();                                    \
    } else if (QKHEADDIM <= 256 && VHEADDIM <= 256) {           \
      constexpr static int kQKHeadDim = 256;                  \
      constexpr static int kVHeadDim = 256;                  \
      return __VA_ARGS__();                                    \
    }                                                           \
  }()
