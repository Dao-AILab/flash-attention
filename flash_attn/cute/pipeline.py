# Copyright (c) 2025, Tri Dao.

from typing import Optional
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Boolean, Int32, if_generate
from cutlass.utils import PipelineAsync, PipelineState, CooperativeGroup, pipeline_init_wait
from cutlass.utils.pipeline import _PipelineOp


@dataclass(frozen=True)
class PipelineTmaAsyncNoCluster(PipelineAsync):

    """
        If size(ClusterShape) == 1, PipelineTmaAsync has all threads
        signaling the barrier during consumer_release. This causes a perf regression in FA3
        forward pass (especially hdim 128 causal). We instead implement a version of
        PipelineTmaAsync where only 1 out of 128 threads signals the barrier.

        Assumption:
        (1) num_consumers % NumThreadsPerWarpGroup == 0
        (2) all 128 threads in the warp group are sync'ed right before calling consumer_release
    """

    @staticmethod
    def create(
        barrier_storage: cute.Pointer,
        num_stages: Int32,
        producer_group: CooperativeGroup,
        consumer_group: CooperativeGroup,
        tx_count: int,
        init_wait: bool = True,
    ):
        """
        This helper function computes any necessary attributes and returns an instance of PipelineTmaAsync.
        :param barrier_storage: Pointer to the smem address for this pipeline's mbarriers
        :type barrier_storage: cute.Pointer
        :param num_stages: Number of buffer stages for this pipeline
        :type num_stages: Int32
        :param producer_group: CooperativeGroup for the producer agent
        :type producer_group: CooperativeGroup
        :param consumer_group: CooperativeGroup for the consumer agent
        :type consumer_group: CooperativeGroup
        :param tx_count: Number of bytes expected to be written to the transaction barrier for one stage
        :type tx_count: int
        """
        producer_type = _PipelineOp.TmaLoad
        consumer_type = _PipelineOp.AsyncThread
        producer = (producer_type, producer_group)
        consumer = (consumer_type, consumer_group)
        sync_object_array_full = PipelineAsync._make_sync_object_array(
            barrier_storage.align(min_align=8), num_stages, producer, tx_count
        )
        sync_object_array_empty = PipelineAsync._make_sync_object_array(
            barrier_storage.align(min_align=8) + num_stages, num_stages, consumer
        )
        dst_rank = None
        producer_mask = None
        if init_wait:
            pipeline_init_wait()
        return PipelineTmaAsyncNoCluster(
            sync_object_array_full,
            sync_object_array_empty,
            num_stages,
            producer_mask,
            dst_rank,
        )

    def producer_acquire(
        self, state: PipelineState, try_acquire_token: Optional[Boolean] = None
    ):
        """
        TMA producer commit conditionally waits on buffer empty and sets the transaction barrier for leader threadblocks.
        """
        if_generate(
            try_acquire_token is None or try_acquire_token == 0,
            lambda: self.sync_object_array_empty.wait(state.index, state.phase),
        )
        self.sync_object_array_full.arrive(state.index, self.producer_mask)

    def producer_commit(self, state: PipelineState):
        """
        TMA producer commit is a NOP. The transaction barrier signals the commit upon completion of the TMA.
        """
        pass

    def consumer_release(self, state: PipelineState):
        """
        TMA consumer release conditionally signals the empty buffer to the producer.
        """
        # Only 1 thread per warp group signals the empty buffer.
        if_generate(
            cute.arch.thread_idx()[0] % 128 == 0,
            lambda: self.sync_object_array_empty.arrive(state.index, self.consumer_mask),
        )
