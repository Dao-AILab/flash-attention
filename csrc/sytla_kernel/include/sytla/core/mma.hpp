#pragma once

#include "sytla/core/tile.hpp"

namespace sytla {

#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
namespace intrinsic {

extern "C" {
// clang-format off
extern SYCL_EXTERNAL half8 __builtin_IB_sub_group16_fdpas_hf_hf_hf_hf_8_8(half8 acc, short8 a, int8 b) __attribute__((const));
extern SYCL_EXTERNAL short8 __builtin_IB_sub_group16_fdpas_bf_bf_bf_bf_8_8(short8 acc, short8 a, int8 b) __attribute__((const));
extern SYCL_EXTERNAL float8 __builtin_IB_sub_group16_fdpas_f_f_hf_hf_8_8(float8 acc, short8 a, int8 b) __attribute__((const));
extern SYCL_EXTERNAL float8 __builtin_IB_sub_group16_fdpas_f_f_bf_bf_8_8(float8 acc, short8  a, int8 b) __attribute__((const));
// clang-format on
}

} // namespace intrinsic
#endif

//==------------------------------ MNK Shape -------------------------------==//

template <int M_, int N_, int K_>
struct ShapeMNK {
  static constexpr int M = M_;
  static constexpr int N = N_;
  static constexpr int K = K_;
};

template <typename ShapeA, typename ShapeB>
struct shape_multiply {
  using type = ShapeMNK<ShapeA::M * ShapeB::M, ShapeA::N * ShapeB::N, ShapeA::K * ShapeB::K>;
};

template <typename ShapeA, typename ShapeB>
using shape_multiply_t = typename shape_multiply<ShapeA, ShapeB>::type;

//==------------------------------- DpasMeta -------------------------------==//

// DpasMeta is used to carry information for DPAS intrinsics
template <typename ValueT_, typename ElementT_, int kSize_, bool kVnni_>
struct DpasMeta {
  using ValueT = ValueT_;
  using ElementT = ElementT_;
  static constexpr int kSize = kSize_;
  static constexpr bool kVnni = kVnni_;
};

template <typename DpasMeta, int kReplicaY, int kReplicaX, ArchTag kArchTag>
struct MetaToTile {
  using DstLayout = Layout<sizeof(typename DpasMeta::ElementT), DpasMeta::kSize, kReplicaX, kReplicaY,
                        DpasMeta::kVnni, kArchTag>;
  using type = Tile<typename DpasMeta::ValueT, DstLayout>;
};

//==---------------------------- MMA operations ----------------------------==//

struct XE2_8x16x16_hf_hf_hf_hf {
  static constexpr ArchTag kArchTag = ArchTag::Xe;
  using DpasShape = ShapeMNK<8, 16, 16>;

  // Meta info: ValueT, ElementT, kSize, kVnni
  using DpasMetaD = DpasMeta<half, half, 8, false>;
  using DpasMetaC = DpasMeta<half, half, 8, false>;
  using DpasMetaA = DpasMeta<half, short, 8, false>;
  using DpasMetaB = DpasMeta<half, int, 8, true>;

  INLINE static void mma(half8 &d, const half8 &c, const short8 &a, const int8 &b) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
    d = intrinsic::__builtin_IB_sub_group16_fdpas_hf_hf_hf_hf_8_8(c, a, b);
#endif
  }
};

struct XE2_8x16x16_bf_bf_bf_bf {
  static constexpr ArchTag kArchTag = ArchTag::Xe;
  using DpasShape = ShapeMNK<8, 16, 16>;

  // Meta info: ValueT, ElementT, kSize, kVnni
  using DpasMetaD = DpasMeta<bfloat16, short, 8, false>;
  using DpasMetaC = DpasMeta<bfloat16, short, 8, false>;
  using DpasMetaA = DpasMeta<bfloat16, short, 8, false>;
  using DpasMetaB = DpasMeta<bfloat16, int, 8, true>;

  INLINE static void mma(short8 &d, const short8 &c, const short8 &a, const int8 &b) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
    d = intrinsic::__builtin_IB_sub_group16_fdpas_bf_bf_bf_bf_8_8(c, a, b);
#endif
  }
};

struct XE2_8x16x16_f_f_hf_hf {
  static constexpr ArchTag kArchTag = ArchTag::Xe;
  using DpasShape = ShapeMNK<8, 16, 16>;

  // Meta info: ValueT, ElementT, kSize, kVnni
  using DpasMetaD = DpasMeta<float, float, 8, false>;
  using DpasMetaC = DpasMeta<float, float, 8, false>;
  using DpasMetaA = DpasMeta<half, short, 8, false>;
  using DpasMetaB = DpasMeta<half, int, 8, true>;

  INLINE static void mma(float8 &d, const float8 &c, const short8 &a, const int8 &b) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
    d = intrinsic::__builtin_IB_sub_group16_fdpas_f_f_hf_hf_8_8(c, a, b);
#endif
  }
};

struct XE2_8x16x16_f_f_bf_bf {
  static constexpr ArchTag kArchTag = ArchTag::Xe;
  using DpasShape = ShapeMNK<8, 16, 16>;

  // Meta info: ValueT, ElementT, kSize, kVnni
  using DpasMetaD = DpasMeta<float, float, 8, false>;
  using DpasMetaC = DpasMeta<float, float, 8, false>;
  using DpasMetaA = DpasMeta<bfloat16, short, 8, false>;
  using DpasMetaB = DpasMeta<bfloat16, int, 8, true>;

  INLINE static void mma(float8 &d, const float8 &c, const short8 &a, const int8 &b) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
    d = intrinsic::__builtin_IB_sub_group16_fdpas_f_f_bf_bf_8_8(c, a, b);
#endif
  }
};

//==----------------------------- SubGroupMma ------------------------------==//

template <typename MmaOperation, typename ReplicaMNK = ShapeMNK<1, 1, 1>>
struct SubGroupMma {
  static constexpr ArchTag kArchTag = MmaOperation::kArchTag;
  static constexpr int kSubGroupSize = ArchConfig<kArchTag>::kSubGroupSize;

  using DpasShape = typename MmaOperation::DpasShape;
  using MmaShape = shape_multiply_t<DpasShape, ReplicaMNK>;

  using DpasMetaD = typename MmaOperation::DpasMetaD;
  using DpasMetaC = typename MmaOperation::DpasMetaC;
  using DpasMetaA = typename MmaOperation::DpasMetaA;
  using DpasMetaB = typename MmaOperation::DpasMetaB;

  using TileD = typename MetaToTile<DpasMetaD, ReplicaMNK::M, ReplicaMNK::N, kArchTag>::type;
  using TileC = typename MetaToTile<DpasMetaC, ReplicaMNK::M, ReplicaMNK::N, kArchTag>::type;
  using TileA = typename MetaToTile<DpasMetaA, ReplicaMNK::M, ReplicaMNK::K, kArchTag>::type;
  using TileB = typename MetaToTile<DpasMetaB, ReplicaMNK::K, ReplicaMNK::N, kArchTag>::type;

  // Call MMA operation: TileD = TileC + TileA x TileB
  INLINE static void call(TileD &d, TileC &c, const TileA &a, const TileB &b) {
    using MatrixD = vector_type_t<typename DpasMetaD::ElementT, DpasMetaD::kSize>;
    using MatrixC = vector_type_t<typename DpasMetaC::ElementT, DpasMetaC::kSize>;
    using MatrixA = vector_type_t<typename DpasMetaA::ElementT, DpasMetaA::kSize>;
    using MatrixB = vector_type_t<typename DpasMetaB::ElementT, DpasMetaB::kSize>;
#pragma unroll
    for (int k = 0; k < ReplicaMNK::K; k++) {
#pragma unroll
      for (int n = 0; n < ReplicaMNK::N; n++) {
#pragma unroll
        for (int m = 0; m < ReplicaMNK::M; m++) {
          MatrixD &mat_d = reinterpret_cast<MatrixD &>(d.template block<false>(n, m));
          MatrixC &mat_c = reinterpret_cast<MatrixC &>(c.template block<false>(n, m));
          const MatrixA &mat_a = reinterpret_cast<const MatrixA &>(a.template block<false>(k, m));
          const MatrixB &mat_b = reinterpret_cast<const MatrixB &>(b.template block<false>(n, k));
          MmaOperation::mma(mat_d, mat_c, mat_a, mat_b);
        }
      }
    }
  }

  // Call MMA operation: TileC += TileA x TileB
  INLINE static void call(TileD &c, const TileA &a, const TileB &b) { call(c, c, a, b); }

  INLINE static TileA make_tile_a() { return TileA{}; }
  INLINE static TileB make_tile_b() { return TileB{}; }
  INLINE static TileC make_tile_c() { return TileC{}; }
  INLINE static TileD make_tile_d() { return TileD{}; }
};

} // namespace sytla