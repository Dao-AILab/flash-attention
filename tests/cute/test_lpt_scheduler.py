"""Exercise the LPT scheduler's actual CuTe lowering on SM100/SM110.

The probe supplies CLC responses directly (including assigning scheduler._ctx
for fetched CLC work); attention tests separately exercise the hardware CLC
pipeline. Run this file in one real-GPU pass. The exhaustion test passes a null
metadata pointer and is also suitable for memcheck.
"""

import os
from dataclasses import dataclass
from functools import cache

import cutlass
import pytest
import torch
from cuda.bindings import driver as cuda
from cutlass import Int32, const_expr, cute
from cutlass.cute.runtime import from_dlpack
from quack.cute_dsl_utils import ParamsBase

from flash_attn.cute.tile_scheduler import (
    SchedulingMode,
    SingleTileLPTScheduler,
    TileSchedulerArguments,
    WorkTileInfo,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("FLASH_ATTENTION_FAKE_TENSOR", "0") == "1"
    or not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] not in (10, 11),
    reason="LPT coordinate probes require real SM100/SM110 execution",
)


@dataclass
class ProbeCLCState(ParamsBase):
    work: WorkTileInfo

    @cute.jit
    def initial_work_tile_info(self):
        return self.work

    @cute.jit
    def get_current_work(self):
        return self.work


@cute.jit
def store_work(out: cute.Tensor, work: WorkTileInfo, phase: Int32):
    x, y, _ = cute.arch.block_idx()
    for i in cutlass.range_constexpr(4):
        out[x, y, phase, i] = work.tile_idx[i]
    out[x, y, phase, 4] = Int32(work.is_valid_tile)


@cute.kernel
def probe_kernel(
    params: SingleTileLPTScheduler.Params,
    out: cute.Tensor,
    validity: cute.Tensor,
    exhausted_only: cutlass.Constexpr[bool],
):
    if cute.arch.thread_idx()[0] == 0:
        x, y, _ = cute.arch.block_idx()
        invalid = WorkTileInfo(
            (Int32(-123), Int32(-456), Int32(-789)), validity[0] != 0
        )
        ctx = None
        if const_expr(params.scheduling_mode == SchedulingMode.CLC):
            raw = WorkTileInfo(
                (x // params.cluster_shape_m * params.cluster_shape_m, y, Int32(0)),
                cutlass.Boolean(True),
            )
            ctx = ProbeCLCState(raw)
        scheduler = SingleTileLPTScheduler.create(params, ctx)
        if const_expr(not exhausted_only):
            store_work(out, scheduler.initial_work_tile_info(), Int32(0))
            if const_expr(params.scheduling_mode == SchedulingMode.CLC):
                # Fetch a different cluster/split, including section wraparound.
                next_x = (x // params.cluster_shape_m + 1) % params.total_blocks
                scheduler._ctx = ProbeCLCState(
                    WorkTileInfo(
                        (
                            next_x * params.cluster_shape_m,
                            (y + 1) % params.num_splits,
                            Int32(0),
                        ),
                        cutlass.Boolean(True),
                    )
                )
            store_work(out, scheduler.get_current_work(), Int32(1))
        if const_expr(params.scheduling_mode == SchedulingMode.CLC):
            scheduler._ctx = ProbeCLCState(invalid)
            exhausted = scheduler.get_current_work()
        else:
            exhausted = scheduler.advance_to_next_work()
        store_work(out, exhausted, Int32(2))


@cute.jit
def launch_probe(
    out: cute.Tensor,
    metadata: cute.Tensor | None,
    validity: cute.Tensor,
    num_block: Int32,
    num_head: Int32,
    num_batch: Int32,
    num_splits: Int32,
    mode: cutlass.Constexpr[SchedulingMode],
    cluster_m: cutlass.Constexpr[int],
    use_cluster_idx: cutlass.Constexpr[bool],
    lpt: cutlass.Constexpr[bool],
    split_kv: cutlass.Constexpr[bool],
    exhausted_only: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    args = TileSchedulerArguments(
        num_block,
        num_head,
        num_batch,
        num_splits,
        Int32(8192),
        Int32(128),
        Int32(128),
        Int32(0),
        (128, 128),
        cluster_shape_mn=(cluster_m, 1),
        use_cluster_idx=use_cluster_idx,
        lpt=lpt,
        is_split_kv=split_kv,
        num_splits_dynamic_ptr=metadata,
    )
    params = SingleTileLPTScheduler.to_underlying_arguments(args, scheduling_mode=mode)
    probe_kernel(params, out, validity, exhausted_only).launch(
        grid=SingleTileLPTScheduler.get_grid_shape(params),
        block=(32, 1, 1),
        cluster=(cluster_m, 1, 1) if const_expr(cluster_m > 1) else None,
        stream=stream,
    )


@cache
def compile_probe(mode, cluster_m, use_cluster_idx, lpt, split_kv, exhausted_only):
    out = torch.empty(2, 3, 3, 5, device="cuda", dtype=torch.int32)
    metadata = torch.empty(2, device="cuda", dtype=torch.int32) if split_kv else None
    validity = torch.empty(1, device="cuda", dtype=torch.int32)
    return cute.compile(
        launch_probe,
        from_dlpack(out).mark_layout_dynamic(),
        from_dlpack(metadata).mark_layout_dynamic() if metadata is not None else None,
        from_dlpack(validity).mark_layout_dynamic(),
        Int32(3),
        Int32(4),
        Int32(2),
        Int32(3),
        mode,
        cluster_m,
        use_cluster_idx,
        lpt,
        split_kv,
        exhausted_only,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


# STATIC 2CTA counts clusters. CLC also accepts the persistent CTA-count domain.
GEOMETRIES = [
    pytest.param(SchedulingMode.STATIC, 1, False, id="static-1cta"),
    pytest.param(SchedulingMode.STATIC, 2, True, id="static-2cta-clusters"),
    pytest.param(SchedulingMode.CLC, 1, False, id="clc-1cta"),
    pytest.param(SchedulingMode.CLC, 2, True, id="clc-2cta-clusters"),
    pytest.param(SchedulingMode.CLC, 2, False, id="clc-2cta-ctas"),
]


def ordered_tiles(m_clusters, heads, batch, lpt):
    """Oracle: enumerate head/batch sections rather than duplicating divmods."""
    hb = [(h, b) for b in range(batch) for h in range(heads)]
    blocks = list(range(m_clusters))
    if lpt:
        blocks.reverse()
    return [
        (m, h, b)
        for start in range(0, len(hb), 8)
        for m in blocks
        for h, b in hb[start : start + 8]
    ]


@pytest.mark.parametrize("mode,cluster_m,use_cluster_idx", GEOMETRIES)
@pytest.mark.parametrize("lpt", [False, True], ids=["forward", "lpt"])
@pytest.mark.parametrize("num_splits", [1, 3], ids=["split1", "split3"])
@pytest.mark.parametrize(
    "heads,batch", [(4, 2), (5, 3), (3, 1)], ids=["full", "full-residual", "residual"]
)
@pytest.mark.parametrize("num_block", [1, 3, 4], ids=["single", "odd", "even"])
def test_coordinate_mapping(
    mode, cluster_m, use_cluster_idx, lpt, num_splits, heads, batch, num_block
):
    m_clusters = num_block
    if mode == SchedulingMode.CLC and not use_cluster_idx:
        m_clusters = (num_block + cluster_m - 1) // cluster_m
    tiles = ordered_tiles(m_clusters, heads, batch, lpt)
    grid_x = len(tiles) * cluster_m
    out = torch.full((grid_x, num_splits, 3, 5), -999, device="cuda", dtype=torch.int32)
    counts = [1 + b % num_splits for b in range(batch)]
    metadata = (
        torch.tensor(counts, device="cuda", dtype=torch.int32)
        if num_splits > 1
        else None
    )
    compiled = compile_probe(
        mode, cluster_m, use_cluster_idx, lpt, num_splits > 1, False
    )
    validity = torch.zeros(1, device="cuda", dtype=torch.int32)
    compiled(out, metadata, validity, num_block, heads, batch, num_splits)
    actual = out.cpu()
    expected = torch.empty_like(actual[:, :, :2])
    for x in range(grid_x):
        for split in range(num_splits):
            for phase in range(2):
                tile = x // cluster_m
                selected_split = split
                if mode == SchedulingMode.CLC and phase == 1:
                    tile = (tile + 1) % len(tiles)
                    selected_split = (split + 1) % num_splits
                m, h, b = tiles[tile]
                if mode == SchedulingMode.CLC and not use_cluster_idx:
                    m = m * cluster_m + x % cluster_m
                packed_split = (
                    selected_split | (counts[b] << 16) if metadata is not None else 0
                )
                expected[x, split, phase] = torch.tensor([m, h, b, packed_split, 1])
    torch.testing.assert_close(actual[:, :, :2], expected, rtol=0, atol=0)
    assert (actual[:, :, 2, 4] == 0).all(), "Exhausted work must be invalid"
    assert (actual[:, :, 2, 3] < (1 << 16)).all(), (
        "Invalid work must not pack split metadata"
    )
    first = actual[:, :, 0, :4].reshape(-1, 4)
    if use_cluster_idx:
        first = actual[::cluster_m, :, 0, :4].reshape(-1, 4)
    assert len(torch.unique(first, dim=0)) == len(first), "Duplicate logical work"
    if cluster_m > 1:
        torch.testing.assert_close(actual[::2, :, :2, 1:4], actual[1::2, :, :2, 1:4])


@pytest.mark.parametrize("mode,cluster_m,use_cluster_idx", GEOMETRIES)
def test_exhaustion_does_not_read_metadata(mode, cluster_m, use_cluster_idx):
    # Keep validity device-driven so the compiler cannot fold away the guard.
    # A load from this zero-length tensor would dereference a null pointer.
    metadata = torch.empty(0, device="cuda", dtype=torch.int32)
    validity = torch.zeros(1, device="cuda", dtype=torch.int32)
    out = torch.full((8 * cluster_m, 3, 3, 5), -999, device="cuda", dtype=torch.int32)
    compiled = compile_probe(mode, cluster_m, use_cluster_idx, True, True, True)
    compiled(out, metadata, validity, 1, 4, 2, 3)
    actual = out.cpu()
    assert (actual[:, :, 2, 4] == 0).all()
    assert (actual[:, :, 2, 3] < (1 << 16)).all()
