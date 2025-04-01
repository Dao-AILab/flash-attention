#pragma once

#include "sytla/core/common.hpp"
#include <type_traits>

namespace sytla {

//==------------------------------- SubArray -------------------------------==//

#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
namespace intrinsic {

extern "C" {
extern SYCL_EXTERNAL float __attribute__((overloadable)) intel_sub_group_shuffle(float X, uint c);
}

} // namespace intrinsic
#endif

template <typename T_, int kLength_, ArchTag kArchTag_ = ArchTag::Xe>
struct SubArray {
  using ElementT = std::remove_cv_t<T_>;
  static constexpr int kLength = kLength_;
  static constexpr ArchTag kArchTag = kArchTag_;

  using Arch = ArchConfig<kArchTag>;
  static constexpr int kSubGroupSize = Arch::kSubGroupSize;
  static constexpr int kSize = (kLength + kSubGroupSize - 1) / kSubGroupSize;

  using VectorT = vector_type_t<ElementT, kSize>;
  using Storage = std::array<ElementT, kSize>;

  Storage data_;

  // Constructors
  INLINE SubArray() = default;
  INLINE SubArray(const SubArray &rhs) = default;
  INLINE SubArray(SubArray &&rhs) = default;
  INLINE SubArray &operator=(const SubArray &rhs) = default;

  INLINE SubArray(VectorT vec) { data_ = sycl::bit_cast<Storage>(vec); }

  INLINE VectorT vector() const { return sycl::bit_cast<VectorT>(data_); }

  // Returns the length of sub-group array
  INLINE static constexpr int len() noexcept { return kLength; }
  // Returns the storage size of vec
  INLINE static constexpr int size() noexcept { return kSize; }

  // Returns the element in the sub-group array at a given index
  INLINE ElementT operator()(int index) const {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
    int i = index / kSubGroupSize;
    int j = index % kSubGroupSize;
    return intrinsic::intel_sub_group_shuffle(data_[i], j);
#else
    return index;
#endif
  }

  // Returns the raw vec ref at a given index
  INLINE const ElementT &operator[](int index) const { return data_[index]; }

  INLINE ElementT &operator[](int index) { return data_[index]; }
};

// ==------------------------------- Operators ------------------------------==//

// #define __SUB_ARRAY_BINOP_SCALAR(OP)                                                               \
//   template <typename T, int N>                                                                     \
//   INLINE SubArray<T, N> operator OP(const SubArray<T, N> &lhs, const T & rhs) {                    \
//     return SubArray<T, N>{lhs.vector() OP rhs};                                                    \
//   }                                                                                                \
//   template <typename T, int N>                                                                     \
//   INLINE SubArray<T, N> operator OP(const T & lhs, const SubArray<T, N> &rhs) {                    \
//     return SubArray<T, N>{lhs OP rhs.vector()};                                                    \
//   }                                                                                                \
//   template <typename T, int N>                                                                     \
//   INLINE SubArray<T, N> operator OP(const SubArray<T, N> &lhs, const SubArray<T, N> &rhs) {        \
//     return SubArray<T, N>{lhs.vector() OP rhs.vector()};                                           \
//   }

#define __SUB_ARRAY_BINOP_SCALAR(OP)                                                               \
  template <typename Lhs, typename Rhs,                                                            \
            typename T = typename std::common_type_t<Lhs, Rhs>::ElementT,                          \
            int N = std::common_type_t<Lhs, Rhs>::kLength,                                         \
            typename enable = std::enable_if_t<std::is_same_v<Lhs, SubArray<T, N>> ||              \
                                               std::is_same_v<Rhs, SubArray<T, N>>>>               \
  INLINE auto operator OP(const Lhs &lhs, const Rhs &rhs) {                                        \
    auto get_vector = [](const auto &x) {                                                          \
      if constexpr (std::is_same_v<std::decay_t<decltype(x)>, SubArray<T, N>>) {                   \
        return x.vector();                                                                         \
      } else {                                                                                     \
        return x;                                                                                  \
      }                                                                                            \
    };                                                                                             \
    return SubArray<T, N>{get_vector(lhs) OP get_vector(rhs)};                                     \
  }

__SUB_ARRAY_BINOP_SCALAR(+)
__SUB_ARRAY_BINOP_SCALAR(-)
__SUB_ARRAY_BINOP_SCALAR(*)
__SUB_ARRAY_BINOP_SCALAR(/)
#undef __SUB_ARRAY_BINOP_SCALAR

template <typename T, int N>
INLINE SubArray<T, N> max(const SubArray<T, N> &lhs, const SubArray<T, N> &rhs) {
  SubArray<T, N> ret;
#pragma unroll
  for (int i = 0; i < SubArray<T, N>::size(); ++i) {
    ret[i] = lhs[i] > rhs[i] ? lhs[i] : rhs[i];
  }
  return ret;
}

template <typename T, int N>
INLINE SubArray<T, N> exp(const SubArray<T, N> &x) {
  SubArray<T, N> ret;
#pragma unroll
  for (int i = 0; i < SubArray<T, N>::size(); ++i) {
    ret[i] = sycl::exp(x[i]);
  }
  return ret;
}

template <typename T, int N>
INLINE SubArray<T, N> exp2(const SubArray<T, N> &x) {
  SubArray<T, N> ret;
#pragma unroll
  for (int i = 0; i < SubArray<T, N>::size(); ++i) {
    ret[i] = sycl::exp2(x[i]);
  }
  return ret;
}

} // namespace sytla