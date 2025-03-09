#pragma once

#include "sytla/core/lsc.hpp"
#include "sytla/core/tile.hpp"

namespace sytla {

#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
namespace intrinsic {

extern "C" {
// clang-format off
extern SYCL_EXTERNAL int* __builtin_IB_subgroup_createBlock2DAddressPayload(
    long base, int width_minus_one, int height_minus_one, int pitch_minus_one,
    int blockX, int blockY, int blockWidth, int blockHeight, int numBlocks);
extern SYCL_EXTERNAL int* __builtin_IB_subgroup_copyBlock2DAddressPayload(int* AP);
extern SYCL_EXTERNAL void __builtin_IB_subgroup_addBlock2DAddressPayloadBlockX(int* addrPayload, int blockX);
extern SYCL_EXTERNAL void __builtin_IB_subgroup_addBlock2DAddressPayloadBlockY(int* addrPayload, int blockY);
extern SYCL_EXTERNAL void __builtin_IB_subgroup_setBlock2DAddressPayloadBlockX(int* addrPayload, int blockX);
extern SYCL_EXTERNAL void __builtin_IB_subgroup_setBlock2DAddressPayloadBlockY(int* addrPayload, int blockY);
extern SYCL_EXTERNAL void __builtin_IB_subgroup_setBlock2DAddressPayloadWidth(int* addrPayload, int width_minus_one);
extern SYCL_EXTERNAL void __builtin_IB_subgroup_setBlock2DAddressPayloadHeigth(int* addrPayload, int height_minus_one);
extern SYCL_EXTERNAL void __builtin_IB_subgroup_setBlock2DAddressPayloadPitch(int* addrPayload, int pitch_minus_one);
// clang-format on
}

} // namespace intrinsic
#endif

//==----------------------------- BasePayload ------------------------------==//

// Base payload
//
// Wrapper of the real address payload for intrinsic lsc
template <typename PayloadInfo>
class BasePayload {
public:
  using ValueT = typename PayloadInfo::ValueT;
  static constexpr int kLscDataSize = PayloadInfo::kLscDataSize;

  // Block width/height/arrLength
  static constexpr int kBlockWidth = PayloadInfo::kBlockWidth;
  static constexpr int kBlockHeight = PayloadInfo::kBlockHeight;
  static constexpr int kArrLen = PayloadInfo::kArrLen;

  INLINE BasePayload() : addr_(nullptr){};

  INLINE BasePayload(ValueT *base, int width, int height, int pitch, int coord_x, int coord_y) {
    init(base, width, height, pitch, coord_x, coord_y);
  }

  INLINE BasePayload(const BasePayload &payload) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
    addr_ = intrinsic::__builtin_IB_subgroup_copyBlock2DAddressPayload(payload.addr());
#else
    addr_ = nullptr;
#endif
  }

  INLINE BasePayload &operator=(const BasePayload &payload) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
    addr_ = intrinsic::__builtin_IB_subgroup_copyBlock2DAddressPayload(payload.addr());
#else
    addr_ = nullptr;
#endif
    return *this;
  }

  INLINE void init(ValueT *base, int width, int height, int pitch, int coord_x, int coord_y) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
    long base_ = reinterpret_cast<long>(base);
    int width_1 = width * sizeof(ValueT) - 1;
    int height_1 = height - 1;
    int pitch_1 = pitch * sizeof(ValueT) - 1;
    int lsc_coord_x = coord_x * sizeof(ValueT) / kLscDataSize;

    addr_ = intrinsic::__builtin_IB_subgroup_createBlock2DAddressPayload(
        base_, width_1, height_1, pitch_1, lsc_coord_x, coord_y, kBlockWidth, kBlockHeight,
        kArrLen);
#else
    addr_ = nullptr;
#endif
  }

  INLINE void set_coord_x(int coord_x) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
    int lsc_coord_x = coord_x * sizeof(ValueT) / kLscDataSize;
    intrinsic::__builtin_IB_subgroup_setBlock2DAddressPayloadBlockX(addr_, lsc_coord_x);
#endif
  }

  INLINE void set_coord_y(int coord_y) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
    intrinsic::__builtin_IB_subgroup_setBlock2DAddressPayloadBlockY(addr_, coord_y);
#endif
  }

  INLINE void update_coord_x(int offset_x) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
    int lsc_offset_x = offset_x * sizeof(ValueT) / kLscDataSize;
    intrinsic::__builtin_IB_subgroup_addBlock2DAddressPayloadBlockX(addr_, lsc_offset_x);
#endif
  }

  INLINE void update_coord_y(int offset_y) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
    intrinsic::__builtin_IB_subgroup_addBlock2DAddressPayloadBlockY(addr_, offset_y);
#endif
  }

  INLINE static constexpr int get_block_width() {
    return kBlockWidth * kArrLen * kLscDataSize / sizeof(ValueT);
  }

  INLINE static constexpr int get_block_height() { return kBlockHeight; }

  INLINE int *addr() const  { return addr_; }

private:
  int *addr_;
};

//==----------------------------- LoadPayload ------------------------------==//

namespace detail {

// Mapping Tile to global memory for loading
template <typename Tile, bool kTranspose = false>
struct LoadPayloadInfo;

template <typename Tile>
struct LoadPayloadInfo<Tile, false> {
  using ValueT = typename Tile::ValueT;

private:
  using LayoutI = typename Tile::Layout;
  using Arch = typename LayoutI::Arch;

  using LscType = std::conditional_t<LayoutI::kVnni, ValueT, typename LayoutI::ElementT>;
  using LayoutM = split_layout_t<LscType, LayoutI, Arch::kMaxLoadHeight>;

  static constexpr int get_array_length() {
    constexpr int block_width = LayoutM::template get_block_width<LscType>();
    constexpr int block_height = LayoutM::template get_block_height<LscType>();
    constexpr int block_size = block_width * block_height;
    constexpr int width = LayoutI::template get_width<LscType>();
    constexpr int replica_x = LayoutI::kReplicaX;
    constexpr int elem_per_cl = Arch::kCacheLine / sizeof(LscType);
    constexpr int elem_per_reg = Arch::kRegSize / sizeof(LscType);

    constexpr bool is_aligned = (block_size % elem_per_reg == 0);

    if constexpr (is_aligned && (width % elem_per_cl == 0) &&
                  (elem_per_cl == 2 * block_width || elem_per_cl == 4 * block_width)) {
      return elem_per_cl / block_width;
    }

    if constexpr (is_aligned && width < elem_per_cl &&
                  (replica_x == 1 || replica_x == 2 || replica_x == 4)) {
      return replica_x;
    }
    return 1;
  }

public:
  static constexpr int kLscDataSize = sizeof(LscType);
  static constexpr int kArrLen = get_array_length();
  using LscLayout =
      Layout<LayoutM::kElementSize, LayoutM::kLength * kArrLen, LayoutM::kReplicaX / kArrLen,
             LayoutM::kReplicaY, LayoutM::kVnni, LayoutM::kArchTag>;
  static constexpr int kBlockWidth = LscLayout::template get_block_width<LscType>();
  static constexpr int kBlockHeight = LscLayout::template get_block_height<LscType>() / kArrLen;
};

template <typename Tile>
struct LoadPayloadInfo<Tile, true> {
  using ValueT = typename Tile::ValueT;

private:
  using LayoutI = typename Tile::Layout;
  using Arch = typename LayoutI::Arch;
  static_assert(LayoutI::kVnni, "Not supported for non-vnni transpose");

  using LscType = typename Arch::TransLoadDataT;
  static constexpr int kCutoff = Arch::kMaxTransLoadWidthBytes / sizeof(LscType);

public:
  static constexpr int kLscDataSize = sizeof(LscType);
  static constexpr int kArrLen = 1;
  using LscLayout = split_layout_t<LscType, LayoutI, kCutoff>;
  static constexpr int kBlockWidth = LscLayout::template get_block_height<LscType>();
  static constexpr int kBlockHeight = LscLayout::template get_block_width<LscType>();
};

} // namespace detail

// Load payload
//
// Payload for loading data from global memory
template <typename Tile_, bool kTranspose_>
class LoadPayload : public BasePayload<detail::LoadPayloadInfo<Tile_, kTranspose_>> {
public:
  using Info = detail::LoadPayloadInfo<Tile_, kTranspose_>;
  using Base = BasePayload<Info>;

  using Tile = Tile_;
  using ValueT = typename Info::ValueT;
  using LscLayout = typename Info::LscLayout;
  static constexpr bool kVnni = LscLayout::kVnni;
  static constexpr bool kTranspose = kTranspose_;

  static constexpr int kReplicaX = kTranspose ? LscLayout::kReplicaY : LscLayout::kReplicaX;
  static constexpr int kReplicaY = kTranspose ? LscLayout::kReplicaX : LscLayout::kReplicaY;

  INLINE LoadPayload() : Base() {}

  INLINE LoadPayload(ValueT *base, int width, int height, int pitch, int coord_x, int coord_y)
      : Base(base, width, height, pitch, coord_x, coord_y) {}

  INLINE LoadPayload(LoadPayload &payload) : Base(payload) {}

  INLINE LoadPayload &operator=(const LoadPayload &payload) {
    Base::operator=(payload);
    return *this;
  }

  INLINE static constexpr int get_width() { return kReplicaX * Base::get_block_width(); }
  INLINE static constexpr int get_height() { return kReplicaY * Base::get_block_height(); }

  // Load data from global memory to tile
  template <LSC_LDCC kCacheOpt = LSC_LDCC_L1C_L3C>
  INLINE void load_tile(Tile &tile) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
    constexpr int kLscDataSize = Base::kLscDataSize;
    constexpr int kBlockWidth = Base::kBlockWidth;
    constexpr int kBlockHeight = Base::kBlockHeight;
    constexpr int kArrLen = Base::kArrLen;
    using BlockLoad =
        intrinsic::SubGroupBlockLoad<Block<LscLayout>, kTranspose, kVnni, kLscDataSize, kBlockWidth,
                                     kBlockHeight, kArrLen>;

#pragma unroll
    for (int x = 0; x < kReplicaX; x++) {
      int immX = x * kBlockWidth * kArrLen;
#pragma unroll
      for (int y = 0; y < kReplicaY; y++) {
        int immY = y * kBlockHeight;

        if constexpr (kTranspose) {
          auto &vec = tile.template block<false, LscLayout>(y, x);
          BlockLoad::call(vec, this->addr(), immX, immY, kCacheOpt);
        } else {
          auto &vec = tile.template block<false, LscLayout>(x, y);
          BlockLoad::call(vec, this->addr(), immX, immY, kCacheOpt);
        }
      }
    }
#endif
  }
};

//==----------------------------- StorePayload -----------------------------==//

namespace detail {

// Mapping Tile to global memory for storing
template <typename Tile>
struct StorePayloadInfo {
  using ValueT = typename Tile::ValueT;

private:
  using LayoutI = typename Tile::Layout;
  static_assert(!LayoutI::kVnni, "Not supported for vnni layout");

public:
  // TODO(FixMe): storing with different block size results in redundant copying
  using Arch = typename LayoutI::Arch;
  using LscLayout = split_layout_t<ValueT, LayoutI, Arch::kMaxStoreHeight>;
  static constexpr int kLscDataSize = sizeof(ValueT);
  static constexpr int kBlockWidth = LscLayout::template get_block_width<ValueT>();
  static constexpr int kBlockHeight = LscLayout::template get_block_height<ValueT>();
  static constexpr int kArrLen = 1;
};

} // namespace detail

// Store payload
//
// Payload for storing data to global memory
template <typename Tile_>
class StorePayload : public BasePayload<detail::StorePayloadInfo<Tile_>> {
public:
  using Info = detail::StorePayloadInfo<Tile_>;
  using Base = BasePayload<Info>;

  using Tile = Tile_;
  using ValueT = typename Info::ValueT;
  using LscLayout = typename Info::LscLayout;
  static constexpr int kReplicaX = LscLayout::kReplicaX;
  static constexpr int kReplicaY = LscLayout::kReplicaY;

  INLINE StorePayload() : Base() {}

  INLINE StorePayload(ValueT *base, int width, int height, int pitch, int coord_x, int coord_y)
      : Base(base, width, height, pitch, coord_x, coord_y) {}

  INLINE StorePayload(const StorePayload &payload) : Base(payload) {}

  INLINE StorePayload &operator=(const StorePayload &payload) {
    Base::operator=(payload);
    return *this;
  }

  // Store tile data to device
  template <LSC_STCC kCacheOpt = LSC_STCC_L1WB_L3WB>
  INLINE void store_tile(Tile &tile) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
    constexpr int kLscDataSize = Base::kLscDataSize;
    constexpr int kBlockWidth = Base::kBlockWidth;
    constexpr int kBlockHeight = Base::kBlockHeight;
    using BlockStore =
        intrinsic::SubGroupBlockStore<Block<LscLayout>, kLscDataSize, kBlockWidth, kBlockHeight>;

#pragma unroll
    for (int x = 0; x < kReplicaX; x++) {
      int immX = x * kBlockWidth;
#pragma unroll
      for (int y = 0; y < kReplicaY; y++) {
        int immY = y * kBlockHeight;
        auto &vec = tile.template block<false, LscLayout>(x, y);
        BlockStore::call(vec, this->addr(), immX, immY, kCacheOpt);
      }
    }
#endif
  }
};

//==--------------------------- PrefetchPayload ----------------------------==//

namespace detail {

// Mapping LoadPayload to PrefetchPayload
template <typename LoadPayload, int kNumSubGroups>
struct PrefetchPayloadInfo {
  using ValueT = typename LoadPayload::ValueT;

private:
  using Arch = typename LoadPayload::LscLayout::Arch;
  static constexpr int kMaxPrefetchHeight = Arch::kMaxLoadHeight;
  static constexpr int kMaxPrefetchWidth = Arch::kMaxLoadWidthBytes / sizeof(ValueT);
  static constexpr int kWidth = LoadPayload::get_width();
  static constexpr int kHeight = LoadPayload::get_height();

  static constexpr int kBlockWidth_ = kWidth > kMaxPrefetchWidth ? kMaxPrefetchWidth : kWidth;
  static constexpr int kNumBlocksX = (kWidth + kBlockWidth_ - 1) / kBlockWidth_;

public:
  static constexpr int kLscDataSize = 1;
  static constexpr int kBlockWidth = kBlockWidth_ * sizeof(ValueT);
  static constexpr int kNumSgX = std::gcd(kNumBlocksX, kNumSubGroups);
  static constexpr int kReplicaX = kNumBlocksX / kNumSgX;

  static constexpr int kNumSgY = kNumSubGroups / kNumSgX;
  static constexpr int kHeightPerSG = (kHeight + kNumSgY - 1) / kNumSgY;
  static constexpr int kBlockHeight =
      kHeightPerSG > kMaxPrefetchHeight ? kMaxPrefetchHeight : kHeightPerSG;
  static constexpr int kReplicaY = (kHeightPerSG + kBlockHeight - 1) / kBlockHeight;
  static constexpr int kArrLen = 1;
};

} // namespace detail

// Prefetch payload
//
// Payload for prefetching data from global memory
template <typename LoadPayload_, int kNumSubGroups_ = 1>
class PrefetchPayload
    : public BasePayload<detail::PrefetchPayloadInfo<LoadPayload_, kNumSubGroups_>> {
public:
  using Info = detail::PrefetchPayloadInfo<LoadPayload_, kNumSubGroups_>;
  using Base = BasePayload<Info>;

  using ValueT = typename Info::ValueT;
  static constexpr int kNumSgX = Info::kNumSgX;
  static constexpr int kNumSgY = Info::kNumSgY;
  static constexpr int kReplicaX = Info::kReplicaX;
  static constexpr int kReplicaY = Info::kReplicaY;

  INLINE PrefetchPayload() : Base() {}

  INLINE PrefetchPayload(ValueT *base, int width, int height, int pitch, int coord_x, int coord_y,
                         int coop_id)
      : Base(base, width, height, pitch, compute_sg_coord_x(coop_id, coord_x),
             compute_sg_coord_y(coop_id, coord_y)) {}

  INLINE PrefetchPayload(const PrefetchPayload &payload) : Base(payload) {}

  INLINE PrefetchPayload &operator=(const PrefetchPayload &payload) {
    Base::operator=(payload);
    return *this;
  }

  INLINE void init(ValueT *base, int width, int height, int pitch, int coord_x, int coord_y,
                   int coop_id) {
    int sg_coord_x = compute_sg_coord_x(coop_id, coord_x);
    int sg_coord_y = compute_sg_coord_y(coop_id, coord_y);
    Base::init(base, width, height, pitch, sg_coord_x, sg_coord_y);
  }

  INLINE static constexpr int compute_sg_coord_x(int coop_id, int coord_x = 0) {
    int coop_idx = coop_id % kNumSgX;
    return coord_x + coop_idx * kReplicaX * Base::get_block_width();
  }

  INLINE static constexpr int compute_sg_coord_y(int coop_id, int coord_y = 0) {
    int coop_idy = coop_id / kNumSgX;
    return coord_y + coop_idy * kReplicaY * Base::get_block_height();
  }

  // Prefetch data from device to cache
  template <LSC_LDCC kCacheOpt = LSC_LDCC_L1C_L3C>
  INLINE void prefetch() {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
    constexpr int kLscDataSize = Base::kLscDataSize;
    constexpr int kBlockWidth = Base::kBlockWidth;
    constexpr int kBlockHeight = Base::kBlockHeight;
    using BlockPrefetch = intrinsic::SubGroupBlockPrefetch<kLscDataSize, kBlockWidth, kBlockHeight>;

#pragma unroll
    for (int x = 0; x < kReplicaX; x++) {
      int immX = x * kBlockWidth;
#pragma unroll
      for (int y = 0; y < kReplicaY; y++) {
        int immY = y * kBlockHeight;
        BlockPrefetch::call(this->addr(), immX, immY, kCacheOpt);
      }
    }
#endif
  }
};

//==----------------------------- LocalPayload -----------------------------==//

// Local payload
//
// Payload of shared local memory for loading/storing data
template <typename ValueT_>
class LocalPayload {
public:
  using ValueT = ValueT_;

  INLINE LocalPayload() : base_(nullptr) {}

  INLINE LocalPayload(ValueT *base, int width, int height, int pitch, int coord_x, int coord_y,
                      int lane_id)
      : base_(base), width_(width), height_(height), pitch_(pitch), coord_x_(coord_x),
        coord_y_(coord_y), lane_id_(lane_id) {}

  INLINE LocalPayload(const LocalPayload &payload) {
    base_ = payload.base_;
    width_ = payload.width_;
    height_ = payload.height_;
    pitch_ = payload.pitch_;
    coord_x_ = payload.coord_x_;
    coord_y_ = payload.coord_y_;
    lane_id_ = payload.lane_id_;
  }

  INLINE LocalPayload &operator=(const LocalPayload &payload) {
    base_ = payload.base_;
    width_ = payload.width_;
    height_ = payload.height_;
    pitch_ = payload.pitch_;
    coord_x_ = payload.coord_x_;
    coord_y_ = payload.coord_y_;
    lane_id_ = payload.lane_id_;
    return *this;
  }

  INLINE void init(ValueT *base, int width, int height, int pitch, int coord_x, int coord_y,
                   int lane_id) {
    base_ = base;
    width_ = width;
    height_ = height;
    pitch_ = pitch;
    coord_x_ = coord_x;
    coord_y_ = coord_y;
    lane_id_ = lane_id;
  }

  INLINE void set_coord_x(int coord_x) { coord_x_ = coord_x; }

  INLINE void set_coord_y(int coord_y) { coord_y_ = coord_y; }

  INLINE void set_lane_id(int lane_id) { lane_id_ = lane_id; }

  INLINE void update_coord_x(int offset_x) { coord_x_ += offset_x; }

  INLINE void update_coord_y(int offset_y) { coord_y_ += offset_y; }

  // Load data from local memory to tile
  template <typename T, typename Layout,
            typename = std::enable_if_t<std::is_same_v<T, ValueT> && !Layout::kVnni>>
  INLINE void load_tile(Tile<T, Layout> &tile) {
    using ElementT = typename Layout::ElementT;
    constexpr int kBlockWidth = Layout::template get_block_width<ElementT>();
    constexpr int kBlockHeight = Layout::template get_block_height<ElementT>();

    const ElementT *ptr = reinterpret_cast<const ElementT *>(base_);
    int offset_x = coord_x_ * sizeof(T) / sizeof(ElementT);
    int width = width_ * sizeof(T) / sizeof(ElementT);
    int pitch = pitch_ * sizeof(T) / sizeof(ElementT);

    for (int x = 0; x < Layout::kReplicaX; x++) {
      offset_x += lane_id_ + x * kBlockWidth;

      for (int y = 0; y < Layout::kReplicaY; y++) {
        int offset_y = coord_y_ + y * kBlockHeight;
        auto &arr = tile.template block<false>(x, y);

        for (int i = 0; i < Layout::kLength; i++, offset_y++) {
          // if (offset_y < height_ || offset_x < width) {
          int offset = offset_x + pitch * offset_y;
          arr[i] = ptr[offset];
          // }
        }
      }
    }
  }

  // Store data from tile to local memory
  template <typename T, typename Layout,
            typename = std::enable_if_t<std::is_same_v<T, ValueT> && !Layout::kVnni>>
  INLINE void store_tile(const Tile<T, Layout> &tile) {
    using ElementT = typename Layout::ElementT;
    constexpr int kBlockWidth = Layout::template get_block_width<ElementT>();
    constexpr int kBlockHeight = Layout::template get_block_height<ElementT>();

    ElementT *ptr = reinterpret_cast<ElementT *>(base_);
    int offset_x = coord_x_ * sizeof(T) / sizeof(ElementT);
    int width = width_ * sizeof(T) / sizeof(ElementT);
    int pitch = pitch_ * sizeof(T) / sizeof(ElementT);

    for (int x = 0; x < Layout::kReplicaX; x++) {
      offset_x += lane_id_ + x * kBlockWidth;

      for (int y = 0; y < Layout::kReplicaY; y++) {
        int offset_y = coord_y_ + y * kBlockHeight;
        auto &arr = tile.template block<false>(x, y);

        for (int i = 0; i < Layout::kLength; i++, offset_y++) {
          // if (offset_y < height_ || offset_x < width) {
          int offset = offset_x + pitch * offset_y;
          ptr[offset] = arr[i];
          // }
        }
      }
    }
  }

  // Store data from array to local memory
  template <typename T, int kLength, ArchTag kArchTag = ArchTag::Xe>
  INLINE void store_array(const SubArray<T, kLength, kArchTag> &arr) {}

private:
  ValueT *base_;
  int width_;
  int height_;
  int pitch_;
  int coord_x_;
  int coord_y_;
  int lane_id_;
};

//==-------------------------------- Utils ---------------------------------==//

// Create a payload for loading
template <bool kTranspose = false, typename T, typename Layout>
INLINE constexpr auto make_load_payload([[maybe_unused]] const Tile<T, Layout> &tile, T *base,
                                        int width, int height, int pitch, int coord_x = 0,
                                        int coord_y = 0) {
  using LoadPayload = LoadPayload<Tile<T, Layout>, kTranspose>;
  return LoadPayload{base, width, height, pitch, coord_x, coord_y};
}

// Create a payload for storing
template <typename T, typename LayoutI>
INLINE constexpr auto make_store_payload([[maybe_unused]] const Tile<T, LayoutI> &tile, T *base,
                                         int width, int height, int pitch, int coord_x = 0,
                                         int coord_y = 0) {
  using StorePayload = StorePayload<Tile<T, LayoutI>>;
  return StorePayload{base, width, height, pitch, coord_x, coord_y};
}

// Create a payload for prefetching
template <int kNumSubGroups = 1, typename LoadPayload>
INLINE constexpr auto make_prefetch_payload([[maybe_unused]] const LoadPayload &payload,
                                            int coop_id, typename LoadPayload::ValueT *base,
                                            int width, int height, int pitch, int coord_x = 0,
                                            int coord_y = 0) {
  using PrefetchPayload = PrefetchPayload<LoadPayload, kNumSubGroups>;
  return PrefetchPayload{base, width, height, pitch, coord_x, coord_y, coop_id};
}

// Create a local payload
template <typename T>
INLINE constexpr auto make_local_payload(T *base, int width, int height, int pitch, int coord_x = 0,
                                         int coord_y = 0, int lane_id = 0) {
  using LocalPayload = LocalPayload<T>;
  return LocalPayload{base, width, height, pitch, coord_x, coord_y, lane_id};
}

} // namespace sytla