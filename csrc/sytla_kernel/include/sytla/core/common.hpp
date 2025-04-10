#pragma once

#include <sycl/sycl.hpp>
#include <type_traits>

namespace sytla {

//==---------------------------- Arch configure ----------------------------==//

// Supported architectures
enum class ArchTag : uint8_t { Xe };

template <ArchTag kArchTag = ArchTag::Xe>
struct ArchConfig;

template <>
struct ArchConfig<ArchTag::Xe> {
  static constexpr ArchTag kArchTag = ArchTag::Xe;
  static constexpr int kSubGroupSize = 16;
  static constexpr int kCacheLine = 64; // bytes
  static constexpr int kRegSize = 64;   // bytes
  // lsc load
  static constexpr int kMaxLoadHeight = 32;
  static constexpr int kMaxLoadWidthBytes = 64;
  static constexpr int kMaxTransLoadWidthBytes = 32;
  using TransLoadDataT = int;
  // lsc store
  static constexpr int kMaxStoreHeight = 8;
  static constexpr int kMaxStoreWidthBytes = 64;
};

//==-------------------------------- Macros --------------------------------==//

#define STR(x) #x
#define XSTR(x) STR(x)

#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
#define INLINE __attribute__((always_inline))
#else
#define INLINE inline
#endif

//==------------------------------ Data types ------------------------------==//

// Sycl data types
using sycl::half;
using sycl::ext::oneapi::bfloat16;

template <typename T>
struct native_type {
  using type = T;
};

template <typename T>
using native_type_t = typename native_type<T>::type;

#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
template <>
struct native_type<half> {
  using type = _Float16;
};
#else
template <>
struct native_type<half> {
  using type = short;
};
#endif

template <>
struct native_type<bfloat16> {
  using type = short;
};

//==------------------------------- int type -------------------------------==//

// clang-format off
template <int kDataSize> struct int_type;
template <> struct int_type<1> { using type = char;  };
template <> struct int_type<2> { using type = short; };
template <> struct int_type<4> { using type = int; };
template <> struct int_type<8> { using type = long; };
// clang-format on

template <int kDataSize>
using int_type_t = typename int_type<kDataSize>::type;

//==----------------------------- vector type ------------------------------==//

// Wrapper of clang vector type extension
template <typename T, int N>
struct vector_type {
  static_assert(N > 0, "zero-element vector not supported");
  // Element data type
  using DataT = native_type_t<std::remove_cv_t<T>>;
  // Number of elements in vector
  static constexpr int kLength = N;

  using type = std::conditional_t<N == 1, DataT, DataT __attribute__((ext_vector_type(N)))>;
};

template <typename T, int N>
using vector_type_t = typename vector_type<T, N>::type;

// Common aliases
#define GEN_VEC_TYPES(T)                                                                           \
  using T##2 = vector_type_t<T, 2>;                                                                \
  using T##4 = vector_type_t<T, 4>;                                                                \
  using T##8 = vector_type_t<T, 8>;                                                                \
  using T##16 = vector_type_t<T, 16>;                                                              \
  using T##32 = vector_type_t<T, 32>;                                                              \
  using T##64 = vector_type_t<T, 64>;

GEN_VEC_TYPES(half)
GEN_VEC_TYPES(float)
GEN_VEC_TYPES(short)
GEN_VEC_TYPES(int)
#undef GEN_VEC_TYPES

// Vector operations
template <typename DstT, typename SrcT, int N,
          typename = std::enable_if_t<sizeof(DstT) == sizeof(SrcT)>>
INLINE const vector_type_t<DstT, N> &recast(const vector_type_t<SrcT, N> &src) {
  if constexpr (std::is_same_v<DstT, SrcT>) {
    return src;
  } else {
    return reinterpret_cast<const vector_type_t<DstT, N> &>(src);
  }
}

template <typename DstT, typename SrcT, int N,
          typename = std::enable_if_t<sizeof(DstT) == sizeof(SrcT)>>
INLINE vector_type_t<DstT, N> &recast(vector_type_t<SrcT, N> &src) {
  if constexpr (std::is_same_v<DstT, SrcT>) {
    return src;
  } else {
    return reinterpret_cast<vector_type_t<DstT, N> &>(src);
  }
}

template <typename DstT, typename SrcT, int N>
INLINE vector_type_t<DstT, N> convert(const vector_type_t<SrcT, N> &src) {
  if constexpr (N == 1) {
    return static_cast<DstT>(src);
  } else {
    vector_type_t<DstT, N> dst;
#pragma unroll
    for (int i = 0; i < N; i++) {
      dst[i] = static_cast<DstT>(src[i]);
    }
    return dst;
  }
}

template <typename T, int N>
INLINE vector_type_t<T, N> max(const vector_type_t<T, N> &a, const vector_type_t<T, N> &b) {
  if constexpr (N == 1) {
    return a > b ? a : b;
  } else {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
    vector_type_t<T, N> ret = a;
#pragma unroll
    for (int i = 0; i < N; i++) {
      ret[i] = ret[i] > b[i] ? ret[i] : b[i];
    };
    return ret;
#else
    return a > b ? a : b;
#endif
  }
}

template <typename T, int N>
INLINE vector_type_t<T, N> sum(const vector_type_t<T, N> &a, const vector_type_t<T, N> &b) {
  return a + b;
}

//==----------------------------- debug utils ------------------------------==//

template <int T>
struct always_false : std::false_type {};

template <int T>
constexpr bool always_false_v = always_false<T>::value;

template <int T>
constexpr void PRINT() {
  static_assert(always_false_v<T>, "print info for debug");
}

template <typename T>
constexpr void PRINT() {
  static_assert(always_false_v<sizeof(T)>, "print info for debug");
}

INLINE void fence_sw() {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
  asm volatile("fence_sw\n");
#endif
}

} // namespace sytla
