/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include <array>
#include <algorithm>

#include <cutlass/cutlass.h>
#include <cute/layout.hpp>

namespace flash {

static constexpr int kMaxTileSize = 128;

template <bool UseVarSeqLen_, bool UseGQAPacking_> class SeqLenTraits {
public:
  static_assert(!(UseVarSeqLen_ && UseGQAPacking_),
    "Variable sequence length with GQA parallelization not implemented yet.");

  // Total number of queries / keys. Unpadded.
  int sum_s = 0;
  // seq len offsets.
  int *cu_seq_len = nullptr;
  // actual seq len array.
  int *seq_used = nullptr;
  // seq len of the current batch.
  int actual_seq_len = -1;

  // Whether this is for fixed-seq-len or var-seq-len.
  static constexpr bool UseVarSeqLen = UseVarSeqLen_;
  static constexpr bool UseGQAPacking = UseGQAPacking_;

  using ShapeT = std::conditional_t<
      UseVarSeqLen, 
      cute::Shape<int32_t, int32_t, int32_t>,
      std::conditional_t<
        UseGQAPacking,
        cute::Shape<int32_t, int32_t, int32_t, int32_t, int32_t>,
        cute::Shape<int32_t, int32_t, int32_t, int32_t>
      >
  >;
  using StrideT = std::conditional_t<
      UseVarSeqLen, 
      cute::Shape<int64_t, _1, int64_t>, 
      std::conditional_t<
        UseGQAPacking,
        cute::Shape<int64_t, int64_t, _1, int64_t, int64_t>,
        cute::Shape<int64_t, _1, int64_t, int64_t>
      >
  >;
  using LayoutT = cute::Layout<ShapeT, StrideT>;

  using ShapeLseT = std::conditional_t<
      UseVarSeqLen, 
      cute::Shape<int32_t, int32_t>, 
      cute::Shape<int32_t, int32_t, int32_t>
  >;
  using StrideLseT = std::conditional_t<
      UseVarSeqLen, 
      cute::Shape<int64_t, _1>, 
      cute::Shape<int64_t, int64_t, _1>
  >;
  using LayoutLseT = cute::Layout<ShapeLseT, StrideLseT>;

  // Not used for varseqlen
  using ShapeOAccumT = std::conditional_t<
    UseGQAPacking,
    cute::Shape<int32_t, int32_t, int32_t, int32_t, int32_t, int32_t>,
    cute::Shape<int32_t, int32_t, int32_t, int32_t, int32_t>
  >;
  using StrideOAccumT = std::conditional_t<
    UseGQAPacking,
    cute::Shape<int64_t, int64_t, _1, int64_t, int64_t, int64_t>,
    cute::Shape<int64_t, _1, int64_t, int64_t, int64_t>
  >;
  using LayoutOAccumT = cute::Layout<ShapeOAccumT, StrideOAccumT>;

  using ShapeLseAccumT = cute::Shape<int32_t, int32_t, int32_t, int32_t>;
  using StrideLseAccumT = cute::Shape<int64_t, int64_t, int64_t, _1>;
  using LayoutLseAccumT = cute::Layout<ShapeLseAccumT, StrideLseAccumT>;

  CUTLASS_HOST SeqLenTraits() {}

  CUTLASS_HOST SeqLenTraits(
      int sum_s, int max_seq_len, int *cu_seq_len = nullptr, int *seq_used = nullptr): 
      sum_s(sum_s), cu_seq_len(cu_seq_len), seq_used(seq_used), actual_seq_len(max_seq_len) {}

  CUTLASS_DEVICE void init(int bidb) {
    // TODO: add leftpad, seqlen_new for kv cache support
    if (seq_used) {
      actual_seq_len = seq_used[bidb];
    }
  }

  CUTLASS_DEVICE void init_no_guard(int bidb) {
    actual_seq_len = seq_used[bidb];
  }

  // Returns the layout of a tensor in MKHB format in global memory.
  // padded: only useful for var-seq-len for dq_accum and softmax_d.
  CUTLASS_HOST_DEVICE auto get_gmem_layout(
      int m, int k, int h, int b, 
      int64_t m_stride, int64_t h_stride, int64_t b_stride,
      bool padded = false) const {
    static_assert(!UseVarSeqLen, "Specialize default implementation for VarSeqLen.");
    // static_assert(!UseGQAPacking, "Specialize default implementation for UseGQAPacking.");
    return make_layout(make_shape(m, k, h, b),
                       make_stride(m_stride, cute::_1{}, h_stride, b_stride));
  }

  // Returns the layout of a tensor in MKHB format in global memory.
  // padded: only useful for var-seq-len for dq_accum and softmax_d.
  // Overload that separates h into h_k and h/h_k.
  CUTLASS_HOST_DEVICE auto get_gmem_layout(
      int m, int k, int h_k, int b, int h_h_k_ratio,
      int64_t m_stride, int64_t h_stride, int64_t b_stride,
      bool padded = false) const {
    static_assert(!UseVarSeqLen, "Specialize default implementation for VarSeqLen.");
    static_assert(!UseGQAPacking, "Specialize default implementation for UseGQAPacking.");
    return make_layout(make_shape(m, k, h_k * h_h_k_ratio, b),
                       make_stride(m_stride, cute::_1{}, h_stride, b_stride));    
  }

  // Returns the layout of a tensor in MKHBT format in global memory,
  // where T is number of splits.
  CUTLASS_HOST_DEVICE auto get_oaccum_gmem_layout(
      int m, int k, int h, int b, int num_splits,
      int64_t m_stride, int64_t h_stride, int64_t b_stride, int64_t split_stride,
      bool padded = false) const {
    return make_layout(make_shape(m, k, h, b, num_splits),
                       make_stride(m_stride, cute::_1{}, h_stride, b_stride, split_stride));
  }

  // Returns the layout of a tensor in MKHBT format in global memory,
  // where T is number of splits.
  // Overload that separates h into h_k and h/h_k.
  CUTLASS_HOST_DEVICE auto get_oaccum_gmem_layout(
      int m, int k, int h_k, int b, int h_h_k_ratio, int num_splits,
      int64_t m_stride, int64_t h_stride, int64_t b_stride, int64_t split_stride,
      bool padded = false) const {
    return make_layout(make_shape(m, k, h_k * h_h_k_ratio, b, num_splits),
                       make_stride(m_stride, cute::_1{}, h_stride, b_stride, split_stride));
  }

  // Returns the layout of lse tensor in BHM format in global memory.
  // padded: only useful for var-seq-len for dq_accum and softmax_d.
  CUTLASS_HOST_DEVICE auto get_lse_gmem_layout(
      int m, int h, int b, bool padded = false) const {
    static_assert(!UseVarSeqLen, "Specialize default implementation for VarSeqLen.");
    return make_layout(make_shape(b, h, m),
                       make_stride(int64_t(h * m), int64_t(m), cute::_1()));
  }

  // Returns the layout of lse tensor in TBHM format in global memory,
  // where T is number of splits.
  CUTLASS_HOST_DEVICE auto get_lseaccum_gmem_layout(
      int m, int h, int b, int num_splits, bool padded = false) const {
    return make_layout(make_shape(num_splits, b, h, m),
                       make_stride(int64_t(b * h * m), int64_t(h * m), int64_t(m), cute::_1()));
  }

  template <typename MTensor, typename Shape>
  CUTLASS_DEVICE auto get_local_tile_tensor(
      const MTensor &m_tensor, const Shape &tile_shape, 
      int bidh, int bidb, bool padded = false) const {
    auto g_tensor = local_tile(
      m_tensor(_, _, bidh, bidb), tile_shape, make_coord(_, _0{}));
    return g_tensor;
  }

  template <bool Is_split, typename MTensor, typename Shape>
  CUTLASS_DEVICE auto get_lse_local_tile_tensor(
      const MTensor &m_tensor, const Shape &tile_shape, 
      int bidh, int bidb, int n_split_idx, bool padded = false) const {
    // m_tensor has shape (B, H, M) or (splits, B, H, M)
    // Expect tile shape (bM)
    // Returns g_tensor of shape = (bM, ceil_div(M,bM))
    if constexpr(!Is_split) {
      auto g_tensor = local_tile(m_tensor(bidb, bidh, _), tile_shape, make_coord(_));
      return g_tensor;
    } else {
      auto g_tensor = local_tile(m_tensor(n_split_idx, bidb, bidh, _), tile_shape, make_coord(_));
      return g_tensor;
    }
  }

  template <bool Is_split, typename MTensor, typename Shape>
  CUTLASS_DEVICE auto get_o_local_tile_tensor(
      const MTensor &m_tensor, const Shape &tile_shape,
      int bidh, int bidb, int split_idx, bool padded = false) const {
    // static_assert(!UseVarSeqLen, "Don't use get_o_local_tile_tensor with VarSeqLen.");
    // m_tensor has shape (M, K, H, B) or (M, K, H, B, splits)
    // Expect tile shape (bM, K)
    // Returns g_tensor of shape = (bM, K, ceil_div(M,bM))
    if constexpr(!Is_split) {
      auto g_tensor = local_tile(
        m_tensor(_, _, bidh, bidb), tile_shape, make_coord(_, _0{}));
      return g_tensor;
    } else {
      auto g_tensor = local_tile(
        m_tensor(_, _, bidh, bidb, split_idx), tile_shape, make_coord(_, _0{}));
      return g_tensor;
    }
  }
  
};

using FixedSeqLenTraits = SeqLenTraits<false, false>;
using VarSeqLenTraits = SeqLenTraits<true, false>;
using FixedGQASeqLenTraits = SeqLenTraits<false, true>;

template <>
CUTLASS_DEVICE void VarSeqLenTraits::init(int bidb) {
  actual_seq_len = 
      seq_used ? seq_used[bidb] : (cu_seq_len[bidb + 1] - cu_seq_len[bidb]);
}

template <>
CUTLASS_DEVICE void FixedGQASeqLenTraits::init(int bidb) {
  // no op
}

// Returns the static layout of a var-seq-len tensor in global memory based on
// max_seq_len and max_batch_size.
// padded: only useful for var-seq-len for dq_accum and softmax_d.
// When padded is True, use B_M + kMaxTileSize * B as the total B_M.
template <>
CUTLASS_HOST_DEVICE auto VarSeqLenTraits::get_gmem_layout(
    int m, int k, int h, int b, 
    int64_t m_stride, int64_t h_stride, int64_t b_stride,
    bool padded) const {
  return make_layout(
    make_shape(sum_s + (padded ? kMaxTileSize * b : 0), k, h), 
    make_stride(m_stride, cute::_1{}, h_stride));
}

template <>
CUTLASS_HOST_DEVICE auto VarSeqLenTraits::get_gmem_layout(
    int m, int k, int h_k, int b, int h_h_k_ratio,
    int64_t m_stride, int64_t h_stride, int64_t b_stride,
    bool padded) const {
  return make_layout(
    make_shape(sum_s + (padded ? kMaxTileSize * b : 0), k, h_k * h_h_k_ratio), 
    make_stride(m_stride, cute::_1{}, h_stride));
}

// padded: only useful for var-seq-len for dq_accum and softmax_d.
// When padded is True, use B_M + kMaxTileSize * B as the total B_M.
template <>
CUTLASS_HOST_DEVICE auto VarSeqLenTraits::get_lse_gmem_layout(
    int m, int h, int b, bool padded) const {
  return make_layout(
    make_shape(h, sum_s + (padded ? kMaxTileSize * b : 0)), 
    make_stride(int64_t(sum_s + (padded ? kMaxTileSize * b : 0)), cute::_1()));
}

template <>
template <typename MTensor, typename Shape>
CUTLASS_DEVICE auto VarSeqLenTraits::get_local_tile_tensor(
    const MTensor &m_tensor, const Shape &tile_shape,
    int bidh, int bidb, bool padded) const {
  auto g_offset = local_tile(
      m_tensor(_, _, bidh), 
      cute::make_shape(1, get<1>(tile_shape)), 
      make_coord(cu_seq_len[bidb] + (padded ? kMaxTileSize * bidb : 0), _0{}));
  auto g_sequence = make_tensor(
      g_offset.data(), 
      make_layout(
        cute::make_shape(actual_seq_len, get<1>(tile_shape)), 
        g_offset.stride()
      ));
  auto g_tensor = local_tile(g_sequence, tile_shape, make_coord(_, _0{}));
  return g_tensor;
}

// TODO: restructure to not duplicate code
template <>
template <bool Is_split, typename MTensor, typename Shape>
CUTLASS_DEVICE auto VarSeqLenTraits::get_o_local_tile_tensor(
    const MTensor &m_tensor, const Shape &tile_shape,
    int bidh, int bidb, int n_split_idx, bool padded) const {
  static_assert(!Is_split, "Don't currently support split kv kernel with VarSeqLenTraits");
  auto g_offset = local_tile(
      m_tensor(_, _, bidh), 
      cute::make_shape(1, get<1>(tile_shape)), 
      make_coord(cu_seq_len[bidb] + (padded ? kMaxTileSize * bidb : 0), _0{}));
  auto g_sequence = make_tensor(
      g_offset.data(), 
      make_layout(
        cute::make_shape(actual_seq_len, get<1>(tile_shape)), 
        g_offset.stride()
      ));
  auto g_tensor = local_tile(g_sequence, tile_shape, make_coord(_, _0{}));
  return g_tensor;
}

template <>
template <bool Is_split, typename MTensor, typename Shape>
CUTLASS_DEVICE auto VarSeqLenTraits::get_lse_local_tile_tensor(
    const MTensor &m_tensor, const Shape &tile_shape,
    int bidh, int bidb, int n_split_idx, bool padded) const {
  static_assert(!Is_split, "Don't currently support split kv kernel with VarSeqLenTraits");
  auto g_offset = local_tile(
      m_tensor(bidh, _), cute::make_shape(_1{}), 
      make_coord(cu_seq_len[bidb] + (padded ? kMaxTileSize * bidb : 0)));
  auto g_sequence = make_tensor(
      g_offset.data(), 
      make_layout(cute::make_shape(actual_seq_len), cute::make_shape(_1{})));
  auto g_tensor = local_tile(g_sequence, tile_shape, make_coord(_));
  return g_tensor;
}

// Returns layout of QO tensor in (M,H/HK,K,HK,B) format in global memory.
template <>
CUTLASS_HOST_DEVICE auto FixedGQASeqLenTraits::get_gmem_layout(
    int m, int k, int h_k, int b, int h_h_k_ratio,
    int64_t m_stride, int64_t h_stride, int64_t b_stride, bool padded) const {
  return make_layout(make_shape(m, h_h_k_ratio, k, h_k, b),
                     make_stride(m_stride, h_stride, cute::_1{},
                                 h_stride * h_h_k_ratio, b_stride));
}

// Returns layout of Oaccum tensor in (M,H/HK,K,HK,B,T) format in global memory.
template <>
CUTLASS_HOST_DEVICE auto FixedGQASeqLenTraits::get_oaccum_gmem_layout(
    int m, int k, int h_k, int b, int h_h_k_ratio, int num_splits,
    int64_t m_stride, int64_t h_stride, int64_t b_stride, int64_t split_stride,
    bool padded) const {
  return make_layout(make_shape(m, h_h_k_ratio, k, h_k, b, num_splits),
                     make_stride(m_stride, h_stride, cute::_1{},
                                 h_stride * h_h_k_ratio, b_stride,
                                 split_stride));
}

template <>
template <typename MTensor, typename Shape>
CUTLASS_DEVICE auto FixedGQASeqLenTraits::get_local_tile_tensor(
    const MTensor &m_tensor, const Shape &tile_shape, 
    int bidh_kv, int bidb, bool padded) const {
  // m_tensor has shape (M, H/H_K, K, H_K, B)
  // Expect tile_shape (bM/bH, bH, K)
  // Returns g_tensor of shape (bM/bH, bH, K, ceil_div(M,bM/bH), ceil_div(H/H_K,bH))
  auto g_tensor = local_tile(
      m_tensor(_, _, _, bidh_kv, bidb), tile_shape, make_coord(_, _, _0{}));
  return g_tensor;
}

template <>
template <bool Is_split, typename MTensor, typename Shape>
CUTLASS_DEVICE auto FixedGQASeqLenTraits::get_o_local_tile_tensor(
    const MTensor &m_tensor, const Shape &tile_shape,
    int bidh_kv, int bidb, int split_idx, bool padded) const {
  // m_tensor has shape (M, H/H_K, K, H_K, B) or (M, H/H_K, K, H_K, B, splits)
  // Expect tile_shape (bM/bH, bH, K)
  // Returns g_tensor of shape (bM/bH, bH, K, ceil_div(M,bM/bH), ceil_div(H/H_K,bH))
  if constexpr(!Is_split) {
    auto g_tensor = local_tile(
      m_tensor(_, _, _, bidh_kv, bidb), tile_shape, make_coord(_, _, _0{}));
    return g_tensor;
  } else {
    auto g_tensor = local_tile(
      m_tensor(_, _, _, bidh_kv, bidb, split_idx), tile_shape, make_coord(_, _, _0{}));
    return g_tensor;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace flash
