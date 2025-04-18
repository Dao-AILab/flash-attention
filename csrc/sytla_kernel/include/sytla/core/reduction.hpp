#pragma once

#include "sytla/core/sub_array.hpp"
#include "sytla/core/tile.hpp"

namespace sytla {

#if defined(__SYCL_DEVICE_ONLY__)
namespace inline_asm {

//==--------------------------- REDUCE_U32_8x16 ----------------------------==//

// clang-format off
#define REDUCE_U32_8x16(NAME, OP, DT, TT)                                                          \
  INLINE DT __##NAME##_##TT##_8x16(const vector_type_t<DT, 8> &src) {                                 \
    vector_type_t<DT, 8> tmp;                                                                         \
    DT res;                                                                                        \
    asm volatile("{\n"                                                                             \
                 STR(OP) " (M1,16) %0(0,0)<1> %1(0,0)<16;8,1> %1(0,8)<16;8,1>\n"                   \
                 STR(OP) " (M1,16) %0(1,0)<1> %1(2,0)<16;8,1> %1(2,8)<16;8,1>\n"                   \
                 STR(OP) " (M1,16) %0(2,0)<1> %1(4,0)<16;8,1> %1(4,8)<16;8,1>\n"                   \
                 STR(OP) " (M1,16) %0(3,0)<1> %1(6,0)<16;8,1> %1(6,8)<16;8,1>\n"                   \
                 "}\n"                                                                             \
                 : "=rw"(tmp) : "rw"(src));                                                        \
    asm volatile("{\n"                                                                             \
                 ".decl aliasDst v_type=G type=" STR(TT) " num_elts=16 align=GRF alias=<%0,0>\n"   \
                 STR(OP) " (M1,16) %1(4,0)<1> %1(0,0)<8;4,1> %1(0,4)<8;4,1>\n"                     \
                 STR(OP) " (M1,16) %1(5,0)<1> %1(2,0)<8;4,1> %1(2,4)<8;4,1>\n"                     \
                 STR(OP) " (M1,16) %1(6,0)<1> %1(4,0)<4;2,1> %1(4,2)<4;2,1>\n"                     \
                 STR(OP) " (M1,16) aliasDst(0,0)<1> %1(6,0)<2;1,0> %1(6,1)<2;1,0>\n"               \
                 "}\n"                                                                             \
                 : "=rw"(res) : "+rw"(tmp));                                                       \
    return res;                                                                                    \
  }

REDUCE_U32_8x16(max, max, float, f)
REDUCE_U32_8x16(sum, add, float, f)
#undef REDUCE_U32_8x16

//==--------------------------- REDUCE_U32_16x16 ---------------------------==//

#define REDUCE_U32_16x16(NAME, OP, DT, TT)                                                         \
  INLINE DT __##NAME##_##TT##_16x16(const vector_type_t<DT, 16> &src) {                               \
    vector_type_t<DT, 16> tmp;                                                                        \
    DT res;                                                                                        \
    asm volatile("{\n"                                                                             \
                 STR(OP) " (M1,16) %0(0,0)<1> %1(0, 0)<16;8,1> %1(0, 8)<16;8,1>\n"                 \
                 STR(OP) " (M1,16) %0(1,0)<1> %1(2, 0)<16;8,1> %1(2, 8)<16;8,1>\n"                 \
                 STR(OP) " (M1,16) %0(2,0)<1> %1(4, 0)<16;8,1> %1(4, 8)<16;8,1>\n"                 \
                 STR(OP) " (M1,16) %0(3,0)<1> %1(6, 0)<16;8,1> %1(6, 8)<16;8,1>\n"                 \
                 STR(OP) " (M1,16) %0(4,0)<1> %1(8, 0)<16;8,1> %1(8, 8)<16;8,1>\n"                 \
                 STR(OP) " (M1,16) %0(5,0)<1> %1(10,0)<16;8,1> %1(10,8)<16;8,1>\n"                 \
                 STR(OP) " (M1,16) %0(6,0)<1> %1(12,0)<16;8,1> %1(12,8)<16;8,1>\n"                 \
                 STR(OP) " (M1,16) %0(7,0)<1> %1(14,0)<16;8,1> %1(14,8)<16;8,1>\n"                 \
                 "}\n"                                                                             \
                 : "=rw"(tmp) : "rw"(src));                                                        \
    asm volatile("{\n"                                                                             \
                 ".decl aliasDst v_type=G type=" STR(TT) " num_elts=16 align=GRF alias=<%0,0>\n"   \
                 STR(OP) " (M1,16) %1(8, 0)<1> %1(0, 0)<8;4,1> %1(0, 4)<8;4,1>\n"                  \
                 STR(OP) " (M1,16) %1(9, 0)<1> %1(2, 0)<8;4,1> %1(2, 4)<8;4,1>\n"                  \
                 STR(OP) " (M1,16) %1(10,0)<1> %1(4, 0)<8;4,1> %1(4, 4)<8;4,1>\n"                  \
                 STR(OP) " (M1,16) %1(11,0)<1> %1(6, 0)<8;4,1> %1(6, 4)<8;4,1>\n"                  \
                 STR(OP) " (M1,16) %1(12,0)<1> %1(8, 0)<4;2,1> %1(8, 2)<4;2,1>\n"                  \
                 STR(OP) " (M1,16) %1(13,0)<1> %1(10,0)<4;2,1> %1(10,2)<4;2,1>\n"                  \
                 STR(OP) " (M1,16) aliasDst(0,0)<1> %1(12,0)<2;1,0> %1(12,1)<2;1,0>\n"             \
                 "}\n"                                                                             \
                 : "=rw"(res) : "+rw"(tmp));                                                       \
    return res;                                                                                    \
  }


REDUCE_U32_16x16(max, max, float, f)
REDUCE_U32_16x16(sum, add, float, f)
#undef REDUCE_U32_16x16
// clang-format on

} // namespace inline_asm
#endif

//==------------------------------ vec reduce ------------------------------==//

enum class ReduceOp : uint8_t { Max, Sum };

// Reduce float8 along the horizontal (kHorizontal = true)
template <ReduceOp kOp, bool kHorizontal, int N>
INLINE std::enable_if_t<kHorizontal && N == 8, float> reduce(const vector_type_t<float, N> &v) {
#if defined(__SYCL_DEVICE_ONLY__)
  if constexpr (kOp == ReduceOp::Max) {
    return inline_asm::__max_f_8x16(v);
  } else {
    return inline_asm::__sum_f_8x16(v);
  }
#else
  // throw an error in non-SYCL environments
  assert(false &&
         "Reduction for float8 is only supported in SYCL device code with inline assembly");
#endif
}

// Reduce float16 along the horizontal (kHorizontal = true)
template <ReduceOp kOp, bool kHorizontal, int N>
INLINE std::enable_if_t<kHorizontal && N == 16, float> reduce(const vector_type_t<float, N> &v) {
#if defined(__SYCL_DEVICE_ONLY__)
  if constexpr (kOp == ReduceOp::Max) {
    return inline_asm::__max_f_16x16(v);
  } else {
    return inline_asm::__sum_f_16x16(v);
  }
#else
  // throw an error in non-SYCL environments
  assert(false &&
         "Reduction for float16 is only supported in SYCL device code with inline assembly");
#endif
}

template <ReduceOp kOp, typename T, int N>
INLINE vector_type_t<T, N> reduce_helper(const vector_type_t<T, N> &a,
                                         const vector_type_t<T, N> &b) {
  if constexpr (kOp == ReduceOp::Max) {
    return max<T, N>(a, b);
  } else {
    return sum<T, N>(a, b);
  }
}

//==----------------------------- tile_reduce ------------------------------==//

// Check if the tile and layout is valid for reduction.
// Currently, only float8 and float16 blocks are supported.
template <typename T, typename Layout>
constexpr bool is_reducible_v =
    Layout::kElementSize == sizeof(T) && !Layout::kVnni && std::is_same_v<float, T>;

// Helper struct to get a SubArray type as the return type of tile reduction
template <typename T, typename Layout, bool kHorizontal>
using tile_reduce_return_t =
    std::conditional_t<kHorizontal, SubArray<T, Tile<T, Layout>::get_height(), Layout::kArchTag>,
                       SubArray<T, Tile<T, Layout>::get_width(), Layout::kArchTag>>;

// Reduction operation of tile
template <ReduceOp kOp, bool kHorizontal = true, typename T, typename Layout,
          typename = std::enable_if_t<kHorizontal && is_reducible_v<T, Layout>>>
INLINE tile_reduce_return_t<T, Layout, kHorizontal> tile_reduce(const Tile<T, Layout> &tile) {
  // Create layout for reducing
  using LayoutR = split_layout_t<T, Layout, 16>;
  static_assert(LayoutR::kLength == 8 || LayoutR::kLength == 16,
                "Unsupported tile layout for reduction");

  using ReturnT = tile_reduce_return_t<T, Layout, kHorizontal>;
  ReturnT ret;

#pragma unroll
  for (int y = 0; y < LayoutR::kReplicaY; y++) {
    if constexpr (LayoutR::kReplicaX > 1) {
      auto aggregate = tile.template block<true, LayoutR>(0, y);
#pragma unroll
      for (int x = 1; x < LayoutR::kReplicaX; x++) {
        auto &other = tile.template block<true, LayoutR>(x, y);
        aggregate = reduce_helper<kOp, T, LayoutR::kLength>(aggregate, other);
      }
      ret[y] = reduce<kOp, kHorizontal, LayoutR::kLength>(aggregate);
    } else {
      ret[y] = reduce<kOp, kHorizontal, LayoutR::kLength>(tile.template block<true, LayoutR>(0, y));
    }
  }

  return ret;
}

} // namespace sytla