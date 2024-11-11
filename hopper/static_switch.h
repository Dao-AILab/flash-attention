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

#define BOOL_SWITCH(COND, CONST_NAME, ...)                                     \
  [&] {                                                                        \
    if (COND) {                                                                \
      constexpr static bool CONST_NAME = true;                                 \
      return __VA_ARGS__();                                                    \
    } else {                                                                   \
      constexpr static bool CONST_NAME = false;                                \
      return __VA_ARGS__();                                                    \
    }                                                                          \
  }()

#define PREC_SWITCH(PRECTYPE, NAME, ...)                                       \
  [&] {                                                                        \
    if (PRECTYPE == 3) {                                                       \
      using NAME = cutlass::float_e4m3_t;                                      \
      return __VA_ARGS__();                                                    \
    } else if (PRECTYPE == 2) {                                                \
      using NAME = cutlass::bfloat16_t;                                        \
      return __VA_ARGS__();                                                    \
    } else {                                                                   \
      using NAME = cutlass::half_t;                                            \
      return __VA_ARGS__();                                                    \
    }                                                                          \
  }()

#define HEADDIM_SWITCH(HEADDIM, CONST_NAME, ...)                               \
  [&] {                                                                        \
    if (HEADDIM == 64) {                                                       \
      constexpr static int CONST_NAME = 64;                                    \
      return __VA_ARGS__();                                                    \
    } else if (HEADDIM == 128) {                                               \
      constexpr static int CONST_NAME = 128;                                   \
      return __VA_ARGS__();                                                    \
    } else  {                                                                  \
      constexpr static int CONST_NAME = 256;                                   \
      return __VA_ARGS__();                                                    \
    }                                                                          \
  }()

#define SEQLEN_SWITCH(PARAMS, NAME, NAME_Q, ...)                               \
  [&] {                                                                        \
    const bool useSeqLen = PARAMS.cu_seqlens_q;                                \
    const bool usePagedKV = PARAMS.page_block_size>0;                          \
    if (useSeqLen) {                                                           \
      if (usePagedKV) {                                                        \
        using NAME = flash::PagedSeqLenTraits;                                 \
        using NAME_Q = flash::VarSeqLenTraits;                                 \
        return __VA_ARGS__();                                                  \
      } else {                                                                 \
        using NAME = flash::VarSeqLenTraits;                                   \
        using NAME_Q = flash::VarSeqLenTraits;                                 \
        return __VA_ARGS__();                                                  \
      }                                                                        \
    } else {                                                                   \
      using NAME = flash::FixedSeqLenTraits;                                   \
      using NAME_Q = flash::FixedSeqLenTraits;                                 \
      return __VA_ARGS__();                                                    \
    }                                                                          \
  }()

#define SEQLEN_SWITCH_FWD(VAR_SEQ_LEN_Q, SEQ_USED_K, NAME_Q, NAME_K, ...)      \
  [&] {                                                                        \
    bool useVarSeqLenQ = VAR_SEQ_LEN_Q;                                        \
    bool useSeqUsedK = SEQ_USED_K;                                             \
    if (useVarSeqLenQ) {                                                       \
      using NAME_Q = flash::VarSeqLenTraits;                                   \
      using NAME_K = flash::VarSeqLenTraits;                                   \
      return __VA_ARGS__();                                                    \
    } else if (useSeqUsedK) {                                                  \
      using NAME_Q = flash::FixedSeqLenTraits;                                 \
      using NAME_K = flash::FixedSeqLenTraitsDynamic;                          \
      return __VA_ARGS__();                                                    \
    } else {                                                                   \
      using NAME_Q = flash::FixedSeqLenTraits;                                 \
      using NAME_K = flash::FixedSeqLenTraits;                                 \
      return __VA_ARGS__();                                                    \
    }                                                                          \
  }()

#define QUERYHEAD_SWITCH(QUERYHEADS, CONST_NAME, ...)                          \
  [&] {                                                                        \
    if (QUERYHEADS <= 2) {                                                     \
      constexpr static int CONST_NAME = 2;                                     \
      return __VA_ARGS__();                                                    \
    } else if (QUERYHEADS <= 4) {                                              \
      constexpr static int CONST_NAME = 4;                                     \
      return __VA_ARGS__();                                                    \
    } else if (QUERYHEADS <= 8) {                                              \
      constexpr static int CONST_NAME = 8;                                     \
      return __VA_ARGS__();                                                    \
    } else if (QUERYHEADS <= 16) {                                             \
      constexpr static int CONST_NAME = 16;                                    \
      return __VA_ARGS__();                                                    \
    } else {                                                                   \
      constexpr static int CONST_NAME = 32;                                    \
      return __VA_ARGS__();                                                    \
    }                                                                          \
  }()

#define MMA_3WG_SWITCH(QLEN, CONST_NAME, ...)                                  \
  [&] {                                                                        \
    if (QLEN <= 64) {                                                          \
      constexpr static int CONST_NAME = 1;                                     \
      return __VA_ARGS__();                                                    \
    } else if (QLEN <= 128) {                                                  \
      constexpr static int CONST_NAME = 2;                                     \
      return __VA_ARGS__();                                                    \
    } else {                                                                   \
      constexpr static int CONST_NAME = 3;                                     \
      return __VA_ARGS__();                                                    \
    }                                                                          \
  }()

#define MMA_2WG_SWITCH(QLEN, CONST_NAME, ...)                                  \
  [&] {                                                                        \
    if (QLEN <= 64) {                                                          \
      constexpr static int CONST_NAME = 1;                                     \
      return __VA_ARGS__();                                                    \
    } else {                                                                   \
      constexpr static int CONST_NAME = 2;                                     \
      return __VA_ARGS__();                                                    \
    }                                                                          \
  }()

#define NUM_SPLITS_SWITCH(NUM_SPLITS, LOG_MAX_SPLITS, ...)                     \
  [&] {                                                                        \
    if (NUM_SPLITS <= 2) {                                                     \
      constexpr static int LOG_MAX_SPLITS = 1;                                 \
      return __VA_ARGS__();                                                    \
    } else if (NUM_SPLITS <= 4) {                                              \
      constexpr static int LOG_MAX_SPLITS = 2;                                 \
      return __VA_ARGS__();                                                    \
    } else if (NUM_SPLITS <= 8) {                                              \
      constexpr static int LOG_MAX_SPLITS = 3;                                 \
      return __VA_ARGS__();                                                    \
    } else if (NUM_SPLITS <= 16) {                                             \
      constexpr static int LOG_MAX_SPLITS = 4;                                 \
      return __VA_ARGS__();                                                    \
    } else if (NUM_SPLITS <= 32) {                                             \
      constexpr static int LOG_MAX_SPLITS = 5;                                 \
      return __VA_ARGS__();                                                    \
    } else if (NUM_SPLITS <= 64) {                                             \
      constexpr static int LOG_MAX_SPLITS = 6;                                 \
      return __VA_ARGS__();                                                    \
    } else {                                                                   \
      constexpr static int LOG_MAX_SPLITS = 7;                                 \
      return __VA_ARGS__();                                                    \
    }                                                                          \
  }()
