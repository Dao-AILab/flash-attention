#pragma once

#include "sytla/kernel/flash_fwd_traits.hpp"

namespace sytla {
namespace flash {

template <typename KernelTraits, bool kIsCausal, typename = void>
struct FlashForward;

template <typename KernelTraits, bool kIsCausal>
struct FlashForward<KernelTraits, kIsCausal,
                    std::enable_if_t<KernelTraits::kArchTag == ArchTag::Xe>> {
private:
  using ScalarT = typename KernelTraits::ScalarT;
  using AccumT = typename KernelTraits::AccumT;

  static constexpr int kSubGroupSize = ArchConfig<KernelTraits::kArchTag>::kSubGroupSize;
  static constexpr float kNegInfinity = -std::numeric_limits<AccumT>::max();
  static constexpr float kLog2e = 1.442695e+00f; // 1/ln(2)
  static constexpr int kNumSg = KernelTraits::kNumSg;
  static constexpr int kHeadDim = KernelTraits::kHeadDim;
  static constexpr int kSgStrideQ = KernelTraits::kSgStrideQ;
  static constexpr int kSgStrideKV = KernelTraits::kSgStrideKV;
  static constexpr int kAccumStride = KernelTraits::kAccumStride;
  static constexpr int kPrefetchStages = KernelTraits::kPrefetchStages;

  static constexpr int kWgStrideQ = kSgStrideQ * kNumSg;
  static constexpr int kWgStrideKV = kSgStrideKV;

public:
  struct Arguments {
    Arguments(ScalarT *q_ptr, ScalarT *k_ptr, ScalarT *v_ptr, ScalarT *out_ptr,
              AccumT softmax_scale, int bs, int nheads, int head_dim, int seqlen_q, int seqlen_kv)
        : q_ptr(q_ptr), k_ptr(k_ptr), v_ptr(v_ptr), out_ptr(out_ptr),
          softmax_scale(softmax_scale * kLog2e), bs(bs), nheads(nheads), head_dim(head_dim),
          seqlen_q(seqlen_q), seqlen_kv(seqlen_kv) {}

    ScalarT *q_ptr;   // query [bs, seqlen_q, nheads, head_dim]
    ScalarT *k_ptr;   // key   [bs, seqlen_kv, nheads, head_dim]
    ScalarT *v_ptr;   // value [bs, seqlen_kv, nheads, head_dim]
    ScalarT *out_ptr; // out   [bs, seqlen_q, nheads, head_dim]
    AccumT softmax_scale;
    int bs;
    int nheads;
    int head_dim;
    int seqlen_q;
    int seqlen_kv;
  };

  FlashForward(Arguments args) : args_(args) {}

private:
  Arguments args_;

public:
  // Returns the 3D-nd_range of the kernel
  sycl::nd_range<3> get_nd_range() {
    sycl::range<3> local_range{kNumSg, 1, kSubGroupSize};
    size_t num_wg_q = (args_.seqlen_q + kWgStrideQ - 1) / kWgStrideQ;
    sycl::range<3> group_range{static_cast<size_t>(args_.bs), static_cast<size_t>(args_.nheads),
                               num_wg_q};
    return sycl::nd_range<3>{group_range * local_range, local_range};
  }

private:
  // S = Q x K'
  using MmaQK = typename KernelTraits::MmaQK;
  using TileS = typename MmaQK::TileC;

  using TileQ = typename MmaQK::TileA;
  using LoadPayloadQ = LoadPayload<TileQ, false>;
  using PrefetchPayloadQ = PrefetchPayload<LoadPayloadQ, 1>;

  using TileK = typename MmaQK::TileB;
  using LoadPayloadK = LoadPayload<TileK, /*Transposed*/ true>;
  using PrefetchPayloadK = PrefetchPayload<LoadPayloadK, kNumSg>;

  // O = P x V
  using MmaPV = typename KernelTraits::MmaPV;
  using TileO = typename MmaPV::TileC;

  using TileP = typename MmaPV::TileA; // a slice of TileS

  using TileV = typename MmaPV::TileB;
  using LoadPayloadV = LoadPayload<TileV, false>;
  using PrefetchPayloadV = PrefetchPayload<LoadPayloadV, kNumSg>;

  // Container used for intermediate data of SoftMax
  using SfmMax = SubArray<AccumT, kSgStrideQ, KernelTraits::kArchTag>;
  using TileSLayout = typename TileS::Layout;
  using SfmSum = Tile<AccumT, Layout<TileSLayout::kElementSize, TileSLayout::kLength, 1,
                                     TileSLayout::kReplicaY, false, KernelTraits::kArchTag>>;

  struct Context {
    // softmax
    SfmMax softmax_max;
    SfmSum softmax_sum;
    // query
    LoadPayloadQ ld_payload_q;
    PrefetchPayloadQ pf_payload_q;
    // key
    LoadPayloadK ld_payload_k;
    PrefetchPayloadK pf_payload_k;
    // value
    LoadPayloadV ld_payload_v;
    PrefetchPayloadV pf_payload_v;

    int start_x_;
    int start_q_;
    int lane_id;

    INLINE Context() : softmax_max(kNegInfinity), softmax_sum(0) {}

    INLINE void init(sycl::nd_item<3> &item, const Arguments &args) {
      int batch_id = item.get_group(0);
      int head_id = item.get_group(1);
      int q_wg_id = item.get_group(2);
      int sg_id = item.get_local_id(0);
      lane_id = item.get_local_id(2);

      start_q_ = q_wg_id * kWgStrideQ + sg_id * kSgStrideQ;

      start_x_ = head_id * args.head_dim;
      int width = start_x_ + args.head_dim;
      int pitch = args.nheads * args.head_dim;

      int start_y_q = batch_id * args.seqlen_q + q_wg_id * kWgStrideQ + sg_id * kSgStrideQ;
      int height_q = (batch_id + 1) * args.seqlen_q;
      ld_payload_q.init(args.q_ptr, width, height_q, pitch, start_x_, start_y_q);
      pf_payload_q.init(args.q_ptr, width, height_q, pitch, start_x_, start_y_q, 0);

      int start_y_kv = batch_id * args.seqlen_kv;
      int height_kv = start_y_kv + args.seqlen_kv;
      ld_payload_k.init(args.k_ptr, width, height_kv, pitch, start_x_, start_y_kv);
      pf_payload_k.init(args.k_ptr, width, height_kv, pitch, start_x_, start_y_kv, sg_id);
      ld_payload_v.init(args.v_ptr, width, height_kv, pitch, start_x_, start_y_kv);
      pf_payload_v.init(args.v_ptr, width, height_kv, pitch, start_x_, start_y_kv, sg_id);

#pragma unroll
      for (int i = 0; i < kPrefetchStages; i++) {
        pf_payload_q.prefetch();
        pf_payload_k.prefetch();
        pf_payload_q.update_coord_x(kAccumStride);
        pf_payload_k.update_coord_x(kAccumStride);
      }
    }

    INLINE void set_next_qk() {
      ld_payload_q.set_coord_x(start_x_);
      ld_payload_k.set_coord_x(start_x_);
      ld_payload_k.update_coord_y(kSgStrideKV);
      if constexpr (kPrefetchStages != 0) {
        pf_payload_q.set_coord_x(start_x_);
        pf_payload_k.set_coord_x(start_x_);
        pf_payload_k.update_coord_y(kSgStrideKV);
      }
    }
  };

  // S = Q x K'
  INLINE void gemm_qk(TileS &tile_s, Context &ctx) const {
    tile_s.zero();

    TileQ tile_q;
    TileK tile_k;

    constexpr int kLoopCount = kHeadDim / kAccumStride;
    constexpr int kSwitchCount = kLoopCount - kPrefetchStages;
    for (int i = 0; i < kSwitchCount; i++) {
      ctx.ld_payload_q.load_tile(tile_q);
      ctx.ld_payload_k.load_tile(tile_k);
      if constexpr (kPrefetchStages != 0) {
        ctx.pf_payload_q.prefetch();
        ctx.pf_payload_k.prefetch();
      }
      fence_sw();
      ctx.ld_payload_q.update_coord_x(kAccumStride);
      ctx.ld_payload_k.update_coord_x(kAccumStride);
      if constexpr (kPrefetchStages != 0) {
        ctx.pf_payload_q.update_coord_x(kAccumStride);
        ctx.pf_payload_k.update_coord_x(kAccumStride);
      }
      fence_sw();
      MmaQK::call(tile_s, tile_q, tile_k);
      fence_sw();
    }

    for (int i = kSwitchCount; i < kLoopCount; i++) {
      ctx.ld_payload_q.load_tile(tile_q);
      ctx.ld_payload_k.load_tile(tile_k);
      if constexpr (kPrefetchStages != 0) {
        ctx.pf_payload_v.prefetch();
      }
      fence_sw();
      ctx.ld_payload_q.update_coord_x(kAccumStride);
      ctx.ld_payload_k.update_coord_x(kAccumStride);
      if constexpr (kPrefetchStages != 0) {
        ctx.pf_payload_v.update_coord_y(kAccumStride);
      }
      fence_sw();
      MmaQK::call(tile_s, tile_q, tile_k);
      fence_sw();
    }
    tile_s *= (args_.softmax_scale);
    ctx.set_next_qk();
  }

  INLINE void padding_mask(TileS &tile_s, int remaining_kv, int lane_id) const {
    using LayoutS = typename TileS::Layout;

#pragma unroll
    for (int x = 0; x < LayoutS::kReplicaX; x++) {
      int curr_kv = x * TileS::get_block_width() + lane_id;

#pragma unroll
      for (int y = 0; y < LayoutS::kReplicaY; y++) {
        auto &block = tile_s.block(x, y);
        if (curr_kv >= remaining_kv) {
          block = kNegInfinity;
        }
      }
    }
  }

  INLINE void causal_mask(TileS &tile_s, int start_kv, Context &ctx) const {
    using LayoutS = typename TileS::Layout;
#pragma unroll
    for (int x = 0; x < LayoutS::kReplicaX; x++) {
      int curr_kv = start_kv + x * TileS::get_block_width() + ctx.lane_id;
#pragma unroll
      for (int y = 0; y < LayoutS::kReplicaY; y++) {
        auto &block_arr = tile_s.block_as_array(x, y);
#pragma unroll
        for (int i = 0; i < LayoutS::kLength; i++) {
          int curr_q = ctx.start_q_ + y * LayoutS::kLength + i;
          if (curr_kv > curr_q) {
            block_arr[i] = kNegInfinity;
          }
        }
      }
    }
  }

  INLINE SfmSum inter_x_block_sum(TileS &tile_s) const {
    using LayoutS = typename TileS::Layout;
    if constexpr (LayoutS::kReplicaX > 1) {
      SfmSum ret;

#pragma unroll
      for (int y = 0; y < LayoutS::kReplicaY; y++) {
        auto &dst_block = ret.block(0, 0);

        dst_block = tile_s.block(0, y);
#pragma unroll
        for (int x = 1; x < LayoutS::kReplicaX; x++) {
          auto &other = tile_s.block(x, y);
          dst_block = reduce_helper<ReduceOp::Sum, AccumT, LayoutS::kLength>(dst_block, other);
        }
      }
      return ret;

    } else {
      return tile_s;
    }
  }

  // S -- softmax --> P
  INLINE void softmax(TileS &tile_s, TileO &tile_out, Context &ctx) const {
    // compute new maximum
    SubArray cur_max = tile_reduce<ReduceOp::Max>(tile_s);
    SubArray new_max = max(cur_max, ctx.softmax_max);

    tile_s.broadcast_sub(new_max);
    tile_s = exp2(tile_s);

    SfmSum sum = inter_x_block_sum(tile_s);

    // compute new sum
    SubArray o_scale = exp2(ctx.softmax_max - new_max);
    ctx.softmax_sum.broadcast_mul(o_scale);

    SfmSum new_sum = ctx.softmax_sum + sum;

    // rescale output
    tile_out.broadcast_mul(o_scale);

    // update softmax_max and softmax_sum
    ctx.softmax_max = new_max;
    ctx.softmax_sum = new_sum;
  }

  // O += P x V
  INLINE void gemm_pv(TileS &tile_s, TileO &tile_out, Context &ctx) const {
    Tile tile_s_hf = convert<ScalarT>(tile_s);

    TileP tile_p;
    TileV tile_v;

    constexpr int kLoopCount = kSgStrideKV / kAccumStride;
    constexpr int kSwitchCount = kLoopCount - kPrefetchStages;
    for (int i = 0; i < kSwitchCount; i++) {
      tile_p = tile_s_hf.template column_slice<kAccumStride>(i);
      ctx.ld_payload_v.load_tile(tile_v);
      if constexpr (kPrefetchStages != 0) {
        ctx.pf_payload_v.prefetch();
      }
      fence_sw();
      ctx.ld_payload_v.update_coord_y(kAccumStride);
      if constexpr (kPrefetchStages != 0) {
        ctx.pf_payload_v.update_coord_y(kAccumStride);
      }
      fence_sw();
      MmaPV::call(tile_out, tile_out, tile_p, tile_v);
      fence_sw();
    }

    for (int i = kSwitchCount; i < kLoopCount; i++) {
      tile_p = tile_s_hf.template column_slice<kAccumStride>(i);
      ctx.ld_payload_v.load_tile(tile_v);
      if constexpr (kPrefetchStages != 0) {
        ctx.pf_payload_q.prefetch();
        ctx.pf_payload_k.prefetch();
      }
      fence_sw();
      ctx.ld_payload_v.update_coord_y(kAccumStride);
      if constexpr (kPrefetchStages != 0) {
        ctx.pf_payload_q.update_coord_x(kAccumStride);
        ctx.pf_payload_k.update_coord_x(kAccumStride);
      }
      fence_sw();
      MmaPV::call(tile_out, tile_p, tile_v);
      fence_sw();
    }
  }

  // output: [bs, seqlen_q, nheads, head_dim]
  INLINE void rescale_and_store(sycl::nd_item<3> &item, TileO &tile_out, const Context &ctx) const {
    SubArray sum = tile_reduce<ReduceOp::Sum>(ctx.softmax_sum);
    SubArray final_scale = 1.0f / sum;
    tile_out.broadcast_mul(final_scale);
    Tile tile_out_hf = convert<ScalarT>(tile_out);

    int batch_id = item.get_group(0);
    int head_id = item.get_group(1);
    int q_wg_id = item.get_group(2);
    int sg_id = item.get_local_id(0);

    int start_x = head_id * args_.head_dim;
    int width = start_x + args_.head_dim;
    int start_y = batch_id * args_.seqlen_q + q_wg_id * kWgStrideQ + sg_id * kSgStrideQ;
    int height = (batch_id + 1) * args_.seqlen_q;
    int pitch = args_.nheads * args_.head_dim;

    StorePayload st_payload_out =
        make_store_payload(tile_out_hf, args_.out_ptr, width, height, pitch, start_x, start_y);
    st_payload_out.store_tile(tile_out_hf);
  }

public:
  // NOLINTNEXTLINE
  [[intel::reqd_sub_group_size(kSubGroupSize)]] void operator()(sycl::nd_item<3> item) const {
    TileO tile_out;
    tile_out.zero();

    Context ctx;
    ctx.init(item, args_);

    int end_kv = args_.seqlen_kv;
    if constexpr (kIsCausal) {
      end_kv = std::min(end_kv, ctx.start_q_);
    }

    int n_steps = end_kv / kSgStrideKV;
    for (int i = 0; i < n_steps; i++) {
      TileS tile_s;
      gemm_qk(tile_s, ctx);
      softmax(tile_s, tile_out, ctx);
      gemm_pv(tile_s, tile_out, ctx);
    }

    int remaining_kv = end_kv - n_steps * kSgStrideKV;

    if (kIsCausal || (remaining_kv > 0)) {
      TileS tile_s;
      gemm_qk(tile_s, ctx);
      if constexpr (kIsCausal) {
        causal_mask(tile_s, n_steps * kSgStrideKV, ctx);
      } else {
        padding_mask(tile_s, remaining_kv, ctx.lane_id);
      }
      softmax(tile_s, tile_out, ctx);
      gemm_pv(tile_s, tile_out, ctx);
    }
    rescale_and_store(item, tile_out, ctx);
  }
};

} // namespace flash
} // namespace sytla