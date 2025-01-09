/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include<cutlass/pipeline/sm90_pipeline.hpp>

namespace cutlass {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

// As of Cutlass v3.6.0, if size(ClusterShape) == 1, PipelineTmaAsync has all threads
// signaling the barrier during consumer_release. This causes a perf regression in FA3
// forward pass (especially hdim 128 causal). We instead reimplement the version of
// PipelineTmaAsync before v3.6.0 where only 1 out of 128 threads signals the barrier.
//
// Assumption: params.num_consumers % NumThreadsPerWarpGroup == 0
template <int Stages_, class Base=cutlass::PipelineTmaAsync<Stages_>>
class PipelineTmaAsyncNoCluster: public Base {
public:
  using FullBarrier = typename Base::FullBarrier;
  using EmptyBarrier = typename Base::EmptyBarrier;
  static constexpr uint32_t Stages = Stages_;
  using PipelineState = typename Base::PipelineState;

  using SharedStorage = typename Base::SharedStorage;
  using ThreadCategory = typename Base::ThreadCategory;
  using Params = typename Base::Params;

  static
  CUTLASS_DEVICE
  void
  init_barriers(SharedStorage& storage, Params params) {
    int warp_idx = canonical_warp_idx_sync();
    bool is_initializing_warp = (warp_idx == 0);
    if (is_initializing_warp) {
      // Barrier FULL and EMPTY init
      constexpr int producer_arv_cnt = 1;
      uint32_t const num_consumer_warpgroups_per_cluster = params.num_consumers / NumThreadsPerWarpGroup;
      uint32_t const multicast_consumer_arrival_count = num_consumer_warpgroups_per_cluster;

      cutlass::arch::detail::initialize_barrier_array_pair_aligned<decltype(storage.full_barrier_), decltype(storage.empty_barrier_), Stages>(
          storage.full_barrier_, storage.empty_barrier_, producer_arv_cnt, multicast_consumer_arrival_count);
    }
    cutlass::arch::fence_barrier_init();
  }

  template<class ClusterShape, class InitBarriers, class InitMasks>
  CUTLASS_DEVICE
  PipelineTmaAsyncNoCluster(SharedStorage& storage, Params params, ClusterShape cluster_shape, InitBarriers = {}, InitMasks = {})
      : Base(storage, params, make_shape(_1{}, _1{}, _1{}) /*cluster_shape*/, cute::false_type{} /*init_barriers*/, cute::false_type{} /*init_masks*/)
      , empty_barrier_ptr_(&storage.empty_barrier_[0]) {

    int warp_idx = canonical_warp_idx_sync();
    int lane_predicate = cute::elect_one_sync();

    static_assert(cute::is_same_v<InitBarriers, cute::true_type> || cute::is_same_v<InitBarriers, cute::false_type>);
    static_assert(cute::is_same_v<InitMasks, cute::true_type> || cute::is_same_v<InitMasks, cute::false_type>);
    if constexpr (cute::is_same_v<InitBarriers, cute::true_type>) {
      init_barriers(storage, params);
    }

  }

  // Constructor
  template<class ClusterShape>
  CUTLASS_DEVICE
  PipelineTmaAsyncNoCluster(SharedStorage& storage, Params params, ClusterShape cluster_shape)
      : PipelineTmaAsyncNoCluster(storage, params, cluster_shape, cute::true_type{}, cute::true_type{}) { }

  template<class ClusterShape, class InitBarriers>
  CUTLASS_DEVICE
  PipelineTmaAsyncNoCluster(SharedStorage& storage, Params params, ClusterShape cluster_shape, InitBarriers = {})
      : PipelineTmaAsyncNoCluster(storage, params, cluster_shape, InitBarriers{}, cute::true_type{}) { }

  CUTLASS_DEVICE
  void consumer_release(PipelineState state) {
    consumer_release(state.index());
  }

private:
  EmptyBarrier* const empty_barrier_ptr_ = nullptr;

  // Consumer signalling Producer of completion
  // Ensures all blocks in the Same Row and Column get notifed.
  CUTLASS_DEVICE
  void consumer_release(uint32_t stage, uint32_t skip = false) {
    empty_barrier_ptr_[stage].arrive(0 /*dst_blockid_*/, uint32_t(threadIdx.x % cutlass::NumThreadsPerWarpGroup == 0) & (!skip) /*is_signaling_thread*/);
  }

};


////////////////////////////////////////////////////////////////////////////////////////////////////

} // end namespace cutlass
