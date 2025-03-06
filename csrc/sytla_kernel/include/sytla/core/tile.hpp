#pragma once

#include "sytla/core/common.hpp"
#include "sytla/core/layout.hpp"
#include "sytla/core/sub_array.hpp"

namespace sytla {

//==--------------------------------- Tile ---------------------------------==//

template <typename ValueT_, typename Layout_>
struct Tile {
  using ValueT = std::remove_cv_t<ValueT_>;
  using Layout = Layout_;
  using ElementT = typename Layout::ElementT;

  template <bool kUseValueType>
  using get_data_t = std::conditional_t<kUseValueType, ValueT, ElementT>;

  static constexpr int kSize = Layout::kSize;
  using Storage = typename Layout::Storage;
  Storage data_;

  // Constructors
  INLINE Tile() = default;

  INLINE Tile(const Tile &tile) : data_(tile.data_) {}

  INLINE Tile &operator=(const Tile &tile) {
    data_ = tile.data_;
    return *this;
  }

  // Assign data
  INLINE Tile(const Storage &data) : data_(data) {}

  INLINE Tile &operator=(const Storage &data) {
    data_ = data;
    return *this;
  }

  // Set value
  INLINE void set(ValueT value) {
    auto &data_ref = data();
    data_ref = value;
  }

  // Assign value
  INLINE Tile &operator=(ValueT value) {
    set(value);
    return *this;
  }

  // Set data to zero
  INLINE void zero() { set(ValueT(0)); }

  // Returns sub group size
  INLINE static constexpr int get_sg_size() { return Layout::ArchConfig::kSubGroupSize; }

  // Returns the tile's block width
  INLINE static constexpr int get_block_width() {
    return Layout::template get_block_width<ValueT>();
  }
  // Returns the tile's block height
  INLINE static constexpr int get_block_height() {
    return Layout::template get_block_height<ValueT>();
  }
  // Returns the tile's width
  INLINE static constexpr int get_width() { return Layout::template get_width<ValueT>(); }
  // Returns the tile's height
  INLINE static constexpr int get_height() { return Layout::template get_height<ValueT>(); }

  // Returns the storage data with/without value type
  template <bool kUseValueType = true,
            typename = std::enable_if_t<sizeof(get_data_t<kUseValueType>) == Layout::kElementSize>>
  INLINE const vector_type_t<get_data_t<kUseValueType>, kSize> &data() const {
    if constexpr (kUseValueType) {
      return reinterpret_cast<const vector_type_t<ValueT, kSize> &>(data_);
    } else {
      return data_;
    }
  }

  template <bool kUseValueType = true,
            typename = std::enable_if_t<sizeof(get_data_t<kUseValueType>) == Layout::kElementSize>>
  INLINE vector_type_t<get_data_t<kUseValueType>, kSize> &data() {
    if constexpr (kUseValueType) {
      return reinterpret_cast<vector_type_t<ValueT, kSize> &>(data_);
    } else {
      return data_;
    }
  }

  // Returns tile's block at given coordinate
  template <bool kUseValueType = true, typename LayoutI = Layout,
            typename = std::enable_if_t<is_compatible_layout_v<LayoutI, Layout>>>
  INLINE const Block<LayoutI, get_data_t<kUseValueType>> &block(int x, int y) const {
    using BlockT = Block<LayoutI, get_data_t<kUseValueType>>;
    constexpr int kReplicaX = LayoutI::kReplicaX;
    constexpr int kReplicaY = LayoutI::kReplicaY;

    const auto &blocks = reinterpret_cast<const BlockT(&)[kReplicaX][kReplicaY]>(data_);
    return blocks[x][y];
  }

  template <bool kUseValueType = true, typename LayoutI = Layout,
            typename = std::enable_if_t<is_compatible_layout_v<LayoutI, Layout>>>
  INLINE Block<LayoutI, get_data_t<kUseValueType>> &block(int x, int y) {
    using BlockT = Block<LayoutI, get_data_t<kUseValueType>>;
    constexpr int kReplicaX = LayoutI::kReplicaX;
    constexpr int kReplicaY = LayoutI::kReplicaY;

    auto &blocks = reinterpret_cast<BlockT(&)[kReplicaX][kReplicaY]>(data_);
    return blocks[x][y];
  }

  template <bool kUseValueType = true, typename LayoutI = Layout>
  INLINE auto block_as_array(int x, int y) -> get_data_t<kUseValueType> (&)[LayoutI::kLength] {
    using ArrayT = get_data_t<kUseValueType>[LayoutI::kLength];
    return reinterpret_cast<ArrayT &>(block<kUseValueType, LayoutI>(x, y));
  }

  template <bool kUseValueType = true, typename LayoutI = Layout>
  INLINE auto block_as_array(int x,
                             int y) const -> const get_data_t<kUseValueType> (&)[LayoutI::kLength] {
    using ArrayT = get_data_t<kUseValueType>[LayoutI::kLength];
    return reinterpret_cast<const ArrayT &>(block<kUseValueType, LayoutI>(x, y));
  }

  template <
      int kWidth = get_block_width(),
      typename = std::enable_if_t<(get_width() % kWidth == 0) && (kWidth % get_block_width() == 0)>>
  INLINE decltype(auto) column_slice(int x) const {
    constexpr int kNumColumnSlice = get_width() / kWidth;
    constexpr int kSliceSize = Layout::kReplicaY * Layout::kLength * kWidth / get_block_width();
    using ColumnT = vector_type_t<ElementT, kSliceSize>;
    const auto &columns = reinterpret_cast<const ColumnT(&)[kNumColumnSlice]>(data_);
    return columns[x];
  }

  template <bool kHorizontal = true, typename = std::enable_if_t<kHorizontal && !Layout::kVnni>>
  INLINE void broadcast_div(const SubArray<ValueT, get_height()> &arr) {
#pragma unroll
    for (int y = 0; y < Layout::kReplicaY; y++) {
      int hid = y * get_block_height();
#pragma unroll
      for (int x = 0; x < Layout::kReplicaX; x++) {
        auto &block_arr = block_as_array(x, y);
#pragma unroll
        for (int i = 0; i < Layout::kLength; i++) {
          block_arr[i] /= arr(i + hid);
        }
      }
    }
  }

  template <bool kHorizontal = true, typename = std::enable_if_t<kHorizontal && !Layout::kVnni>>
  INLINE void broadcast_mul(const SubArray<ValueT, get_height()> &arr) {
#pragma unroll
    for (int y = 0; y < Layout::kReplicaY; y++) {
      int hid = y * get_block_height();
#pragma unroll
      for (int x = 0; x < Layout::kReplicaX; x++) {
        auto &block_arr = block_as_array(x, y);
#pragma unroll
        for (int i = 0; i < Layout::kLength; i++) {
          block_arr[i] *= arr(i + hid);
        }
      }
    }
  }

  template <bool kHorizontal = true, typename = std::enable_if_t<kHorizontal && !Layout::kVnni>>
  INLINE void broadcast_add(const SubArray<ValueT, get_height()> &arr) {
#pragma unroll
    for (int y = 0; y < Layout::kReplicaY; y++) {
      int hid = y * get_block_height();
#pragma unroll
      for (int x = 0; x < Layout::kReplicaX; x++) {
        auto &block_arr = block_as_array(x, y);
#pragma unroll
        for (int i = 0; i < Layout::kLength; i++) {
          block_arr[i] += arr(i + hid);
        }
      }
    }
  }

  template <bool kHorizontal = true, typename = std::enable_if_t<kHorizontal && !Layout::kVnni>>
  INLINE void broadcast_sub(const SubArray<ValueT, get_height()> &arr) {
#pragma unroll
    for (int y = 0; y < Layout::kReplicaY; y++) {
      int hid = y * get_block_height();
#pragma unroll
      for (int x = 0; x < Layout::kReplicaX; x++) {
        auto &block_arr = block_as_array(x, y);
#pragma unroll
        for (int i = 0; i < Layout::kLength; i++) {
          block_arr[i] = block_arr[i] - arr(i + hid);
        }
      }
    }
  }

  template <typename T, typename = std::enable_if_t<std::is_same_v<T, ValueT> &&
                                                    sizeof(T) == Layout::kElementSize>>
  INLINE void operator*=(T value) {
#pragma unroll
    for (int y = 0; y < Layout::kReplicaY; y++) {
      int hid = y * get_block_height();
#pragma unroll
      for (int x = 0; x < Layout::kReplicaX; x++) {
        auto &block_arr = block_as_array(x, y);
#pragma unroll
        for (int i = 0; i < Layout::kLength; i++) {
          block_arr[i] = block_arr[i] * value;
        }
      }
    }
  }

// TODO: remove redundant mov caused by below code
#define __TILE_BINOP_SCALAR(OP)                                                                    \
  template <typename T, typename = std::enable_if_t<std::is_same_v<T, ValueT> &&                   \
                                                    sizeof(T) == Layout::kElementSize>>            \
  INLINE Tile<T, Layout> operator OP(T value) {                                                    \
    Tile<T, Layout> tile;                                                                          \
    tile.data() = data() OP value;                                                                 \
    return tile;                                                                                   \
  }

  __TILE_BINOP_SCALAR(+)
  __TILE_BINOP_SCALAR(-)
  __TILE_BINOP_SCALAR(*)
  __TILE_BINOP_SCALAR(/)
#undef __TILE_BINOP_SCALAR

  INLINE Tile<ValueT, Layout> operator+(Tile<ValueT, Layout> &other) {
    Tile<ValueT, Layout> dst;
#pragma unroll
    for (int x = 0; x < Layout_::kReplicaX; x++) {
#pragma unroll
      for (int y = 0; y < Layout_::kReplicaY; y++) {
        auto &dst_block = dst.block_as_array(x, y);
        auto &src_block = block_as_array(x, y);
        auto &other_block = other.block_as_array(x, y);

#pragma unroll
        for (int i = 0; i < Layout_::kLength; i++) {
          dst_block[i] = src_block[i] + other_block[i];
        }
      }
    }

    return dst;
  }
};

template <typename ValueT_, typename Layout_>
Tile<ValueT_, Layout_> exp(const Tile<ValueT_, Layout_> &src) {
  Tile<ValueT_, Layout_> dst;

#pragma unroll
  for (int x = 0; x < Layout_::kReplicaX; x++) {
#pragma unroll
    for (int y = 0; y < Layout_::kReplicaY; y++) {
      auto &dst_block = dst.block_as_array(x, y);
      auto &src_block = src.block_as_array(x, y);

#pragma unroll
      for (int i = 0; i < Layout_::kLength; i++) {
        dst_block[i] = sycl::exp(src_block[i]);
      }
    }
  }

  return dst;
}

template <typename ValueT_, typename Layout_>
Tile<ValueT_, Layout_> exp2(const Tile<ValueT_, Layout_> &src) {
  Tile<ValueT_, Layout_> dst(0);

#pragma unroll
  for (int x = 0; x < Layout_::kReplicaX; x++) {
#pragma unroll
    for (int y = 0; y < Layout_::kReplicaY; y++) {
      auto &dst_block = dst.block_as_array(x, y);
      auto &src_block = src.block_as_array(x, y);

#pragma unroll
      for (int i = 0; i < Layout_::kLength; i++) {
        dst_block[i] = sycl::exp2(src_block[i]);
      }
    }
  }

  return dst;
}

//==------------------------------- MakeTile -------------------------------==//

template <typename T, int kWidth, int kHeight, bool kVnni = false, ArchTag kArchTag = ArchTag::Xe>
struct MakeTile {
private:
  static_assert(sizeof(T) <= 4, "Only support data type with size <= 4 bytes");
  static constexpr int kSubGroupSize = ArchConfig<kArchTag>::kSubGroupSize;
  static constexpr int kMaxLoadHeight = ArchConfig<kArchTag>::kMaxLoadHeight;
  static constexpr int kElementSize = kVnni ? sizeof(int) : sizeof(uint16_t);

  static constexpr int kBlockWidth =
      kVnni ? kSubGroupSize : kSubGroupSize * kElementSize / sizeof(T);
  static constexpr int kReplicaX = kWidth / kBlockWidth;
  static_assert(kWidth % kBlockWidth == 0, "Width should be multiple of block width");

  template <int x, int limit>
  INLINE static constexpr int get_largest_factor_within_limit() {
    return (x <= limit) ? x : get_largest_factor_within_limit<x / 2, limit>();
  }

  static constexpr int kBlockHeight = get_largest_factor_within_limit<kHeight, kMaxLoadHeight>();
  static constexpr int kReplicaY = kHeight / kBlockHeight;

public:
  using type = Layout<kElementSize, kBlockHeight, kReplicaX, kReplicaY, kVnni, kArchTag>;
};

template <typename T, int kWidth, int kHeight, bool kVnni = false, ArchTag kArchTag = ArchTag::Xe>
using MakeTile_t = typename MakeTile<T, kWidth, kHeight, kVnni, kArchTag>::type;

template <typename T, int kWidth, int kHeight, bool kVnni = false, ArchTag kArchTag = ArchTag::Xe>
INLINE auto make_tile() {
  using Layout = MakeTile_t<T, kWidth, kHeight, kVnni, kArchTag>;
  return Tile<T, Layout>{};
}

template <typename DstT, typename SrcT, typename LayoutS>
INLINE Tile<DstT, val_layout_t<DstT, LayoutS>> convert(const Tile<SrcT, LayoutS> &src) {
  Tile<DstT, val_layout_t<DstT, LayoutS>> dst;

#pragma unroll
  for (int x = 0; x < LayoutS::kReplicaX; x++) {
#pragma unroll
    for (int y = 0; y < LayoutS::kReplicaY; y++) {
      auto &dst_block = dst.block_as_array(x, y);
      auto &src_block = src.block_as_array(x, y);

#pragma unroll
      for (int i = 0; i < LayoutS::kLength; i++) {
        dst_block[i] = static_cast<DstT>(src_block[i]);
      }
    }
  }

  return dst;
}

} // namespace sytla