#pragma once

#include "sytla/core/common.hpp"

namespace sytla {

//==-------------------------------- Layout --------------------------------==//

// Layout is used to illustrate the information of a tile stored in the sub_group's register.
//
// Layout contains the shape information of a tile stored as a vector. The tile is divided into
// equal blocks and each block contains the same number of vector elements. The blocks are
// arranged into 2D grid in column-major order.
template <int kElementSize_, int kLength_, int kReplicaX_ = 1, int kReplicaY_ = 1,
          bool kVnni_ = false, ArchTag kArchTag_ = ArchTag::Xe>
struct Layout {
  // Size of element in the storage
  static constexpr int kElementSize = kElementSize_;
  // Number of elements in one block
  static constexpr int kLength = kLength_;
  // Number of blocks in X direction
  static constexpr int kReplicaX = kReplicaX_;
  // Number of blocks in Y direction
  static constexpr int kReplicaY = kReplicaY_;
  // Whether it is VNNI arrangement
  static constexpr bool kVnni = kVnni_;
  // Architecture and its configure
  static constexpr ArchTag kArchTag = kArchTag_;
  using Arch = ArchConfig<kArchTag>;
  // Total number of elements
  static constexpr int kSize = kLength * kReplicaX * kReplicaY;
  static_assert((kSize & (kSize - 1)) == 0, "Size must be power of 2");

  // Element type and storage
  using ElementT = int_type_t<kElementSize>;
  using Storage = vector_type_t<ElementT, kSize>;

  // The Block type in Layout. The data type can be replaced by the actual value type.
  template <typename T = void>
  struct BlockType {
    using DataT = std::conditional_t<std::is_void_v<T>, ElementT, T>;
    static_assert(sizeof(DataT) == kElementSize, "Mismatched size of the data type");
    using type = vector_type_t<DataT, kLength>;
  };

  // Returns the width of the layout's block with actual value type.
  template <typename T>
  INLINE static constexpr int get_block_width() {
    constexpr int kSubGroupSize = Arch::kSubGroupSize;
    return kVnni ? kSubGroupSize : kSubGroupSize * kElementSize / sizeof(T);
  }

  // Returns the height of the layout's block with actual size of value type.
  template <typename T>
  INLINE static constexpr int get_block_height() {
    return kVnni ? kLength * kElementSize / sizeof(T) : kLength;
  }

  // Returns the total width of the layout with actual size of value type.
  template <typename T>
  INLINE static constexpr int get_width() {
    return kReplicaX * get_block_width<T>();
  }

  // Returns the total height of the layout with actual size of value type.
  template <typename T>
  INLINE static constexpr int get_height() {
    return kReplicaY * get_block_height<T>();
  }
};

// Alias for the BlockType of Layout
template <typename Layout, typename T = void>
using Block = typename Layout::template BlockType<T>::type;

// Split layout into new blocks along vertical direction with given kCutoff
template <typename ValueT, typename LayoutI, int kCutoff>
struct split_layout {
private:
  static constexpr bool kVnni = LayoutI::kVnni;
  static constexpr int kElementSize = LayoutI::kElementSize;

  static constexpr int kCutLength = kVnni ? kCutoff * sizeof(ValueT) / kElementSize : kCutoff;
  static_assert((kCutoff & (kCutoff - 1)) == 0 && kCutLength > 0,
                "Cutoff is not power of 2 or too small");

  static constexpr int kLengthI = LayoutI::kLength * LayoutI::kReplicaY;
  static constexpr int kLength = kLengthI > kCutLength ? kCutLength : kLengthI;
  static constexpr int kReplicaY = kLengthI / kLength;

public:
  using type =
      Layout<kElementSize, kLength, LayoutI::kReplicaX, kReplicaY, kVnni, LayoutI::kArchTag>;
};

template <typename ValueT, typename Layout, int kCutoff>
using split_layout_t = typename split_layout<ValueT, Layout, kCutoff>::type;

// Returns a new layout with a storage data size consistent with the provide value type and size
template <typename ValueT, typename LayoutI>
using val_layout_t = Layout<sizeof(ValueT), LayoutI::kLength, LayoutI::kReplicaX,
                            LayoutI::kReplicaY, LayoutI::kVnni, LayoutI::kArchTag>;

// Check if the two layouts are compatible, which means they can be applied to the same tile.
template <typename LayoutA, typename LayoutB>
constexpr bool is_compatible_layout_v = (LayoutA::kElementSize == LayoutB::kElementSize) &&
                                        (LayoutA::kSize == LayoutB::kSize) &&
                                        (LayoutA::kVnni == LayoutB::kVnni) &&
                                        (LayoutA::kArchTag == LayoutB::kArchTag);

} // namespace sytla