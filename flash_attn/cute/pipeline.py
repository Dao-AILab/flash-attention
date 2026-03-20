# Copyright (c) 2025, Tri Dao.

from typing import Optional
from dataclasses import dataclass

import cutlass.cute as cute
from cutlass import Boolean, Int32, const_expr
from cutlass.cutlass_dsl import if_generate, dsl_user_op
from cutlass.pipeline import PipelineState
from cutlass.pipeline import PipelineUserType
from cutlass.pipeline import NamedBarrier as NamedBarrierOg
from cutlass.pipeline import PipelineAsync as PipelineAsyncOg
from cutlass.pipeline import PipelineCpAsync as PipelineCpAsyncOg
from cutlass.pipeline import PipelineTmaAsync as PipelineTmaAsyncOg
from cutlass.pipeline import PipelineTmaUmma as PipelineTmaUmmaOg
from cutlass.pipeline import PipelineUmmaAsync as PipelineUmmaAsyncOg
from cutlass.pipeline import PipelineAsyncUmma as PipelineAsyncUmmaOg


def _override_create(parent_cls, child_cls):
    """Create a static factory that constructs parent_cls then re-classes to child_cls."""

    @staticmethod
    def create(*args, **kwargs):
        obj = parent_cls.create(*args, **kwargs)
        # Can't assign to __class__ directly since the dataclass is frozen
        object.__setattr__(obj, "__class__", child_cls)
        return obj

    return create


def _make_state(index: Int32, phase: Int32) -> PipelineState:
    """Construct a PipelineState from index and phase (count/stages unused by callers)."""
    return PipelineState(stages=0, count=Int32(0), index=index, phase=phase)


class PipelineStateSimple:
    """
    Pipeline state contains an index and phase bit corresponding to the current position in the circular buffer.
    Use a single Int32 to store both the index and phase bit, then we use divmod to get the
    index and phase. If stages is a power of 2, divmod turns into bit twiddling.
    """

    def __init__(self, stages: int, phase_index: Int32):
        self._stages = stages
        self._phase_index = phase_index

    def clone(self) -> "PipelineStateSimple":
        return PipelineStateSimple(self.stages, self._phase_index)

    @property
    def stages(self) -> int:
        return self._stages

    @property
    def index(self) -> Int32:
        if const_expr(self._stages == 1):
            return Int32(0)
        else:
            return self._phase_index % self._stages

    @property
    def phase(self) -> Int32:
        # PTX docs say that the phase parity needs to be 0 or 1, so by right we need to
        # take modulo 2. But in practice just passing the phase in without modulo works fine.
        if const_expr(self._stages == 1):
            return self._phase_index
        else:
            return self._phase_index // self._stages

    def advance(self):
        if const_expr(self._stages == 1):
            self._phase_index ^= 1
        else:
            self._phase_index += 1

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
        return PipelineStateSimple(stages, Int32(stages))
    elif type is PipelineUserType.Consumer:
        return PipelineStateSimple(stages, Int32(0))
    else:
        assert False, "Error: invalid PipelineUserType specified for make_pipeline_state."


# ── Mixin: _w_index / _w_index_phase variants that delegate to parent ───────
# Each parent class has PipelineState-based methods (producer_acquire, producer_commit,
# consumer_wait, consumer_release). The _w_index_phase variants just construct a
# PipelineState from (index, phase) and delegate.


class _PipelineIndexPhaseMixin:
    """Mixin providing _w_index_phase / _w_index methods that delegate to PipelineState-based parents."""

    @dsl_user_op
    def producer_acquire_w_index_phase(
        self,
        index: Int32,
        phase: Int32,
        try_acquire_token: Optional[Boolean] = None,
        *,
        loc=None,
        ip=None,
    ):
        state = _make_state(index, phase)
        # Call the parent's producer_acquire (which takes PipelineState)
        self.producer_acquire(state, try_acquire_token, loc=loc, ip=ip)

    @dsl_user_op
    def producer_commit_w_index(self, index: Int32, *, loc=None, ip=None):
        state = _make_state(index, Int32(0))
        self.producer_commit(state, loc=loc, ip=ip)

    @dsl_user_op
    def consumer_wait_w_index_phase(
        self,
        index: Int32,
        phase: Int32,
        try_wait_token: Optional[Boolean] = None,
        *,
        loc=None,
        ip=None,
    ):
        state = _make_state(index, phase)
        self.consumer_wait(state, try_wait_token, loc=loc, ip=ip)

    @dsl_user_op
    def consumer_release_w_index(self, index: Int32, *, loc=None, ip=None):
        state = _make_state(index, Int32(0))
        self.consumer_release(state, loc=loc, ip=ip)


# ── NamedBarrier ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class NamedBarrier(NamedBarrierOg):
    create = _override_create(NamedBarrierOg, None)  # patched below

    @dsl_user_op
    def arrive_w_index(self, index: Int32, *, loc=None, ip=None) -> None:
        """
        The aligned flavor of arrive is used when all threads in the CTA will execute the
        same instruction. See PTX documentation.
        """
        cute.arch.barrier_arrive(
            barrier_id=self.barrier_id + index,
            number_of_threads=self.num_threads,
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def arrive_and_wait_w_index(self, index: Int32, *, loc=None, ip=None) -> None:
        cute.arch.barrier(
            barrier_id=self.barrier_id + index,
            number_of_threads=self.num_threads,
            loc=loc,
            ip=ip,
        )


NamedBarrier.create = _override_create(NamedBarrierOg, NamedBarrier)


# ── PipelineAsync ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PipelineAsync(_PipelineIndexPhaseMixin, PipelineAsyncOg):
    pass


PipelineAsync.create = _override_create(PipelineAsyncOg, PipelineAsync)


# ── PipelineCpAsync ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PipelineCpAsync(_PipelineIndexPhaseMixin, PipelineCpAsyncOg):
    pass


PipelineCpAsync.create = _override_create(PipelineCpAsyncOg, PipelineCpAsync)


# ── PipelineTmaAsync ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PipelineTmaAsync(_PipelineIndexPhaseMixin, PipelineTmaAsyncOg):
    """Override producer_acquire to take in extra_tx_count parameter."""

    @dsl_user_op
    def producer_acquire(
        self,
        state: PipelineState,
        try_acquire_token: Optional[Boolean] = None,
        extra_tx_count: int = 0,
        *,
        loc=None,
        ip=None,
    ):
        """
        TMA producer commit conditionally waits on buffer empty and sets the transaction barrier for leader threadblocks.
        """
        if_generate(
            try_acquire_token is None or try_acquire_token == 0,
            lambda: self.sync_object_empty.wait(state.index, state.phase, loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
        if const_expr(extra_tx_count == 0):
            self.sync_object_full.arrive(state.index, self.producer_mask, loc=loc, ip=ip)
        else:
            tx_count = self.sync_object_full.tx_count + extra_tx_count
            self.sync_object_full.arrive_and_expect_tx(state.index, tx_count, loc=loc, ip=ip)


PipelineTmaAsync.create = _override_create(PipelineTmaAsyncOg, PipelineTmaAsync)


# ── PipelineTmaUmma ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PipelineTmaUmma(_PipelineIndexPhaseMixin, PipelineTmaUmmaOg):
    """Override producer_acquire to take in extra_tx_count parameter."""

    @dsl_user_op
    def producer_acquire(
        self,
        state: PipelineState,
        try_acquire_token: Optional[Boolean] = None,
        extra_tx_count: int = 0,
        *,
        loc=None,
        ip=None,
    ):
        """
        TMA producer commit conditionally waits on buffer empty and sets the transaction barrier for leader threadblocks.
        """
        if_generate(
            try_acquire_token is None or try_acquire_token == 0,
            lambda: self.sync_object_empty.wait(state.index, state.phase, loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
        if const_expr(extra_tx_count == 0):
            if_generate(
                self.is_leader_cta,
                lambda: self.sync_object_full.arrive(
                    state.index, self.producer_mask, loc=loc, ip=ip
                ),
                loc=loc,
                ip=ip,
            )
        else:
            tx_count = self.sync_object_full.tx_count + extra_tx_count
            if_generate(
                self.is_leader_cta,
                lambda: self.sync_object_full.arrive_and_expect_tx(
                    state.index, tx_count, loc=loc, ip=ip
                ),
                loc=loc,
                ip=ip,
            )


PipelineTmaUmma.create = _override_create(PipelineTmaUmmaOg, PipelineTmaUmma)


# ── PipelineUmmaAsync ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class PipelineUmmaAsync(_PipelineIndexPhaseMixin, PipelineUmmaAsyncOg):
    pass


PipelineUmmaAsync.create = _override_create(PipelineUmmaAsyncOg, PipelineUmmaAsync)


# ── PipelineAsyncUmma ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class PipelineAsyncUmma(_PipelineIndexPhaseMixin, PipelineAsyncUmmaOg):
    pass


PipelineAsyncUmma.create = _override_create(PipelineAsyncUmmaOg, PipelineAsyncUmma)
