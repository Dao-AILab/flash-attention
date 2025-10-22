# Copyright (c) 2025, Tri Dao.

# import math
from typing import Optional
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Int32, const_expr
from cutlass.cutlass_dsl import if_generate
from cutlass.pipeline import PipelineAsync, PipelineState, Agent, CooperativeGroup
from cutlass.pipeline import PipelineUserType, PipelineOp
from cutlass.pipeline import PipelineTmaAsync as PipelineTmaAsyncOg
from cutlass.pipeline import PipelineTmaUmma as PipelineTmaUmmaOg


# We deviate from cute-dsl implementation to use cute.arch.cluster_arrive_relaxed
def pipeline_init_wait(cta_layout_vmnk: Optional[cute.Layout] = None):
    """
    Fences the mbarrier init and syncs the threadblock or cluster
    """
    cute.arch.mbarrier_init_fence()

    if cta_layout_vmnk is None or cute.size(cta_layout_vmnk) == 1:
        # If not using clusters, sync the threadblock
        _sync(Agent.ThreadBlock)
    else:
        # If using clusters, sync the cluster
        _sync(Agent.ThreadBlockCluster)


def _sync(group: Agent):
    """
    Syncs all threads within an agent.
    """
    if group is Agent.Thread:
        raise NotImplementedError("Error: Not supported.")
    elif group is Agent.ThreadBlock:
        cute.arch.sync_threads()
    elif group is Agent.ThreadBlockCluster:
        cute.arch.cluster_arrive_relaxed()
        cute.arch.cluster_wait()
    else:
        assert False, (
            "Error: No explicit sync instruction exists. Please use barriers (named / mbarrier) instead."
        )


class PipelineStateSimple:
    """
    Pipeline state contains an index and phase bit corresponding to the current position in the circular buffer.
    Use a single Int32 to store both the index and phase bit, then we use divmod to get the
    index and phase. If stages is a power of 2, divmod turns into bit twiddling.
    """

    def __init__(self, stages: int, phase_index: Int32):
        # assert stages < 2**16
        # self._log_stages = int(math.log2(stages))
        # assert 1 << self._log_stages == stages, "Number of stages must be a power of 2."
        self._stages = stages
        self._phase_index = phase_index

    def clone(self) -> "PipelineStateSimple":
        return PipelineStateSimple(self.stages, self._phase_index)

    @property
    def stages(self) -> int:
        # return 1 << self._log_stages
        return self._stages

    @property
    def index(self) -> Int32:
        # return self._phase_index & 0xFFFF
        # return self._phase_index & ((1 << self._log_stages) - 1)
        if const_expr(self._stages == 1):
            return Int32(0)
        else:
            return self._phase_index % self._stages

    @property
    def phase(self) -> Int32:
        # return self._phase_index >> 16
        # PTX docs say that the phase parity needs to be 0 or 1, so by right we need to
        # take modulo 2. But in practice just passing the phase in without modulo works fine.
        # return (self._phase_index >> self._log_stages) % 2
        # return self._phase_index >> self._log_stages
        if const_expr(self._stages == 1):
            return self._phase_index
        else:
            return self._phase_index // self._stages

    def advance(self):
        if const_expr(self._stages == 1):
            self._phase_index ^= 1
        else:
            self._phase_index += 1

        # def then_body(phase_index):
        #     # XOR the phase bit and set the index to 0
        #     return (phase_index & 0xFFFF0000) ^ (1 << 16)

        # def else_body(phase_index):
        #     return phase_index

        # self._phase_index = if_generate(
        #     (self._phase_index & 0xFFFF) == self.stages,
        #     then_body,
        #     else_body,
        #     [self._phase_index],
        #     [Int32],
        # )

    def __extract_mlir_values__(self):
        phase_index = self._phase_index
        return [phase_index.ir_value()]

    def __new_from_mlir_values__(self, values):
        return PipelineStateSimple(self.stages, Int32(values[0]))


def make_pipeline_state(type: PipelineUserType, stages: int):
    """
    Creates a pipeline state. Producers are assumed to start with an empty buffer and have a flipped phase bit of 1.
    """
    if type is PipelineUserType.Producer:
        # return PipelineStateSimple(stages, Int32(1 << 16))
        return PipelineStateSimple(stages, Int32(stages))
    elif type is PipelineUserType.Consumer:
        return PipelineStateSimple(stages, Int32(0))
    else:
        assert False, "Error: invalid PipelineUserType specified for make_pipeline_state."


@dataclass(frozen=True)
class PipelineTmaAsync(PipelineTmaAsyncOg):
    """
    If size(ClusterShape) == 1, PipelineTmaAsync has all threads
    signaling the barrier during consumer_release. This causes a perf regression in FA3
    forward pass (especially hdim 128 causal). We instead implement a version of
    PipelineTmaAsync where only 1 out of 128 threads signals the barrier.

    Assumptions:
    (1) num_consumers % NumThreadsPerWarpGroup == 0
    (2) all 128 threads in the warp group are sync'ed right before calling consumer_release
    """

    @staticmethod
    def create(
        *,
        num_stages: int,
        producer_group: CooperativeGroup,
        consumer_group: CooperativeGroup,
        tx_count: int,
        barrier_storage: cute.Pointer = None,
        cta_layout_vmnk: Optional[cute.Layout] = None,
        tidx: Optional[Int32] = None,
        init_wait: cutlass.Constexpr[bool] = True,
    ):
        """
        This helper function computes any necessary attributes and returns an instance of PipelineTmaAsync.
        :param barrier_storage: Pointer to the smem address for this pipeline's mbarriers
        :type barrier_storage: cute.Pointer
        :param num_stages: Number of buffer stages for this pipeline
        :type num_stages: Int32
        :param producer_group: `CooperativeGroup` for the producer agent
        :type producer_group: CooperativeGroup
        :param consumer_group: `CooperativeGroup` for the consumer agent
        :type consumer_group: CooperativeGroup
        :param tx_count: Number of bytes expected to be written to the transaction barrier for one stage
        :type tx_count: int
        :param cta_layout_vmnk: Layout of the cluster shape
        :type cta_layout_vmnk: cute.Layout | None
        :param tidx: thread index to consumer async threads
        :type tidx: Int32 | None
        """
        if not isinstance(barrier_storage, cute.Pointer):
            raise ValueError(
                f"Expected barrier_storage to be a cute.Pointer, but got {type(barrier_storage)}"
            )

        producer_type = PipelineOp.TmaLoad
        consumer_type = PipelineOp.AsyncThread

        producer = (producer_type, producer_group)
        consumer = (consumer_type, consumer_group)

        sync_object_full = PipelineAsync._make_sync_object(
            barrier_storage.align(min_align=8), num_stages, producer, tx_count
        )
        sync_object_empty = PipelineAsync._make_sync_object(
            barrier_storage.align(min_align=8) + num_stages, num_stages, consumer
        )
        if tidx is None:
            tidx, _, _ = cute.arch.thread_idx()
        if cta_layout_vmnk is None:
            cta_layout_vmnk = cute.make_layout((1, 1, 1, 1))
        if const_expr(cta_layout_vmnk is None or cute.size(cta_layout_vmnk) == 1):
            dst_rank = None
            is_signalling_thread = tidx % 128 == 0
        else:
            (
                dst_rank,
                is_signalling_thread,
            ) = PipelineTmaAsync.init_empty_barrier_arrive_signal(cta_layout_vmnk, tidx)

        producer_mask = None

        if const_expr(init_wait):
            pipeline_init_wait()

        return PipelineTmaAsync(
            sync_object_full,
            sync_object_empty,
            num_stages,
            producer_mask,
            dst_rank,
            is_signalling_thread,
        )

    def producer_acquire(
        self,
        state: PipelineState,
        try_acquire_token: Optional[Boolean] = None,
        extra_tx_count: int = 0,
    ):
        """
        TMA producer commit conditionally waits on buffer empty and sets the transaction barrier for leader threadblocks.
        """
        if_generate(
            try_acquire_token is None or try_acquire_token == 0,
            lambda: self.sync_object_empty.wait(state.index, state.phase),
        )
        if const_expr(extra_tx_count == 0):
            self.sync_object_full.arrive(state.index, self.producer_mask)
        else:
            tx_count = self.sync_object_full.tx_count + extra_tx_count
            self.sync_object_full.arrive_and_expect_tx(state.index, tx_count)

    def consumer_release(self, state: PipelineState):
        """
        TMA consumer release conditionally signals the empty buffer to the producer.
        """
        # Only 1 thread per warp group signals the empty buffer.
        if self.consumer_mask is None:  # No cluster, 1 thread per warp group to signal
            if_generate(
                cute.arch.thread_idx()[0] % 128 == 0,
                lambda: self.sync_object_empty.arrive(state.index, self.consumer_mask),
            )
        else:
            if_generate(
                self.is_signalling_thread,
                lambda: self.sync_object_empty.arrive(state.index, self.consumer_mask),
            )


@dataclass(frozen=True)
class PipelineTmaUmma(PipelineTmaUmmaOg):
    @staticmethod
    def create(
        *,
        num_stages: int,
        producer_group: CooperativeGroup,
        consumer_group: CooperativeGroup,
        tx_count: int,
        barrier_storage: cute.Pointer = None,
        cta_layout_vmnk: Optional[cute.Layout] = None,
        init_wait: cutlass.Constexpr[bool] = True,
    ):
        """
        This helper function computes any necessary attributes and returns an instance of PipelineTmaUmma.
        :param barrier_storage: Pointer to the smem address for this pipeline's mbarriers
        :type barrier_storage: cute.Pointer
        :param num_stages: Number of buffer stages for this pipeline
        :type num_stages: Int32
        :param producer_group: `CooperativeGroup` for the producer agent
        :type producer_group: CooperativeGroup
        :param consumer_group: `CooperativeGroup` for the consumer agent
        :type consumer_group: CooperativeGroup
        :param tx_count: Number of bytes expected to be written to the transaction barrier for one stage
        :type tx_count: int
        :param cta_layout_vmnk: Layout of the cluster shape
        :type cta_layout_vmnk: cute.Layout | None
        """
        if not isinstance(barrier_storage, cute.Pointer):
            raise ValueError(
                f"Expected barrier_storage to be a cute.Pointer, but got {type(barrier_storage)}"
            )

        producer_type = PipelineOp.TmaLoad
        consumer_type = PipelineOp.TCGen05Mma

        producer = (producer_type, producer_group)
        consumer = (consumer_type, consumer_group)

        sync_object_full = PipelineAsync._make_sync_object(
            barrier_storage.align(min_align=8), num_stages, producer, tx_count
        )
        sync_object_empty = PipelineAsync._make_sync_object(
            barrier_storage.align(min_align=8) + num_stages, num_stages, consumer
        )

        if cta_layout_vmnk is None or cute.size(cta_layout_vmnk) == 1:
            # No mcast mask if not using clusters
            producer_mask = None
            # All threadblocks are leaders if not using clusters
            is_leader_cta = True
        else:
            producer_mask = PipelineTmaUmma._compute_mcast_arrival_mask(cta_layout_vmnk)
            is_leader_cta = PipelineTmaUmma._compute_is_leader_cta(cta_layout_vmnk)

        cta_group = (
            cute.nvgpu.tcgen05.CtaGroup.ONE
            if cta_layout_vmnk is None or cute.size(cta_layout_vmnk, mode=[0]) == 1
            else cute.nvgpu.tcgen05.CtaGroup.TWO
        )

        consumer_mask = producer_mask

        if const_expr(init_wait):
            pipeline_init_wait(cta_layout_vmnk)

        return PipelineTmaUmma(
            sync_object_full,
            sync_object_empty,
            num_stages,
            producer_mask,
            consumer_mask,
            is_leader_cta,
            cta_group,
        )

    def producer_acquire(
        self,
        state: PipelineState,
        try_acquire_token: Optional[Boolean] = None,
        extra_tx_count: int = 0,
    ):
        """
        TMA producer commit conditionally waits on buffer empty and sets the transaction barrier for leader threadblocks.
        """
        if_generate(
            try_acquire_token is None or try_acquire_token == 0,
            lambda: self.sync_object_empty.wait(state.index, state.phase),
        )
        if const_expr(extra_tx_count == 0):
            if_generate(
                self.is_leader_cta,
                lambda: self.sync_object_full.arrive(state.index, self.producer_mask),
            )
        else:
            tx_count = self.sync_object_full.tx_count + extra_tx_count
            if_generate(
                self.is_leader_cta,
                lambda: self.sync_object_full.arrive_and_expect_tx(state.index, tx_count),
            )
