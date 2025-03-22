#pragma once

#include "sytla/core/core.hpp"

namespace sytla {
namespace flash {

namespace detail {

//
// flash_mma_traits
//

template <typename T, ArchTag kArchTag = ArchTag::Xe>
struct flash_mma_traits;

template <>
struct flash_mma_traits<half, ArchTag::Xe> {
  using MmaOperation = XE2_8x16x16_f_f_hf_hf;
  using AccumT = float;
};

template <>
struct flash_mma_traits<bfloat16, ArchTag::Xe> {
  using MmaOperation = XE2_8x16x16_f_f_bf_bf;
  using AccumT = float;
};

//
// flash_forward_traits
//

template <typename T, int kNumSg_, int kHeadDim_, int kSgStrideQ_, int kSgStrideKV_,
          int kAccumStride_, int kPrefetchStages_ = 3, ArchTag kArchTag = ArchTag::Xe>
struct flash_forward_traits;

template <typename T, int kNumSg_, int kHeadDim_, int kSgStrideQ_, int kSgStrideKV_,
          int kAccumStride_, int kPrefetchStages_>
struct flash_forward_traits<T, kNumSg_, kHeadDim_, kSgStrideQ_, kSgStrideKV_, kAccumStride_,
                            kPrefetchStages_, ArchTag::Xe> {
  static constexpr ArchTag kArchTag = ArchTag::Xe;
  using flash_mma_traits = detail::flash_mma_traits<T, kArchTag>;
  using ScalarT = T;
  using AccumT = typename flash_mma_traits::AccumT;
  using MmaOp = typename flash_mma_traits::MmaOperation;

  static constexpr int kNumSg = kNumSg_;
  static constexpr int kHeadDim = kHeadDim_;
  static constexpr int kSgStrideQ = kSgStrideQ_;
  static constexpr int kSgStrideKV = kSgStrideKV_;
  static constexpr int kAccumStride = kAccumStride_;
  static constexpr int kPrefetchStages =
      std::min(std::min(kHeadDim / kAccumStride, kSgStrideKV / kAccumStride), kPrefetchStages_);

  static_assert(kAccumStride % MmaOp::DpasShape::K == 0,
                "kAccumStride should be a multiply of DPAS_K");
  static_assert(kSgStrideQ % MmaOp::DpasShape::M == 0, "kSgStrideQ should be a multiply of DPAS_M");
  static_assert((kSgStrideKV % MmaOp::DpasShape::N == 0) && (kSgStrideKV % kAccumStride == 0) &&
                    (kSgStrideKV % kSgStrideQ == 0),
                "kSgStrideKV should be a multiply of DPAS_N, kAccumStride and kSgStrideQ");
  static_assert((kHeadDim % MmaOp::DpasShape::N == 0) && (kHeadDim % kAccumStride == 0),
                "kHeadDim should be a multiply of DPAS_N and kAccumStride");
  // Actually, due to register pressue, kSgStrideQ is limited to 8/16
  static_assert(kSgStrideKV % kSgStrideQ == 0, "kSgStrideKV should be a multiply of kSgStrideQ");
  using MmaQK =
      SubGroupMma<MmaOp,
                  ShapeMNK<kSgStrideQ / MmaOp::DpasShape::M, kSgStrideKV / MmaOp::DpasShape::N,
                           kAccumStride / MmaOp::DpasShape::K>>;
  using MmaPV =
      SubGroupMma<MmaOp, ShapeMNK<kSgStrideQ / MmaOp::DpasShape::M, kHeadDim / MmaOp::DpasShape::N,
                                  kAccumStride / MmaOp::DpasShape::K>>;
};

} // namespace detail

//
// FlashForwardTraits
//

template <typename ScalarT, int kHeadDim, bool kIsCausal>
struct FlashForwardTraits;

//
// HeadDim: 64
//

template <bool kIsCausal>
struct FlashForwardTraits<half, 64, kIsCausal>
    : detail::flash_forward_traits<half, /* kNumSg */ 8, /* kHeadDim */ 64,
                                   /* kSgStrideQ */ 8, /* kSgStrideKV */ 64, /* kAccumStride */ 32,
                                   /* kPrefetchStages */ 3> {};

template <bool kIsCausal>
struct FlashForwardTraits<bfloat16, 64, kIsCausal>
    : detail::flash_forward_traits<bfloat16, /* kNumSg */ 8, /* kHeadDim */ 64,
                                   /* kSgStrideQ */ 8, /* kSgStrideKV */ 64, /* kAccumStride */ 32,
                                   /* kPrefetchStages */ 3> {};

//
// HeadDim: 128
//

template <bool kIsCausal>
struct FlashForwardTraits<half, 128, kIsCausal>
    : detail::flash_forward_traits<half, /* kNumSg */ 8, /* kHeadDim */ 128,
                                   /* kSgStrideQ */ 8, /* kSgStrideKV */ 32, /* kAccumStride */ 16,
                                   /* kPrefetchStages */ 3> {};

template <bool kIsCausal>
struct FlashForwardTraits<bfloat16, 128, kIsCausal>
    : detail::flash_forward_traits<bfloat16, /* kNumSg */ 8, /* kHeadDim */ 128,
                                   /* kSgStrideQ */ 8, /* kSgStrideKV */ 32, /* kAccumStride */ 16,
                                   /* kPrefetchStages */ 3> {};

} // namespace flash
} // namespace sytla