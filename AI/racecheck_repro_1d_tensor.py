"""Minimal reproducer: cp.async.bulk.tensor.1d (descriptor TMA) passes racecheck.

Same pipeline as racecheck_repro_1d_bulk.py but uses make_tiled_tma_atom to
create a TMA descriptor, which generates cp.async.bulk.tensor.1d PTX.

  python AI/racecheck_repro_1d_tensor.py                                    # correctness
  CUTE_DSL_LINEINFO=1 compute-sanitizer --tool=racecheck python AI/racecheck_repro_1d_tensor.py # 0 hazards
"""
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack
from cutlass import Float32, Int32
import cutlass.pipeline
from cutlass.pipeline.sm90 import PipelineTmaAsync, make_pipeline_state
import cuda.bindings.driver as cuda
import torch

N_BLKS, TILE = 4, 128
N_STG = 2


@cute.kernel
def kernel(g_dst: cute.Tensor, tma_atom: cute.CopyAtom, tma_tensor: cute.Tensor):
    smem = cutlass.utils.SmemAllocator()
    s = smem.allocate_tensor(Float32, cute.make_layout((TILE, N_STG)), byte_alignment=128)
    s_mbar = smem.allocate_tensor(cutlass.Int64, cute.make_layout(2 * N_STG), byte_alignment=8)
    tidx, _, _ = cute.arch.thread_idx()
    warp, lane = tidx // 32, tidx % 32

    pipe = PipelineTmaAsync.create(
        barrier_storage=s_mbar.iterator, num_stages=N_STG,
        producer_group=cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread, 1),
        consumer_group=cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread, 1),
        tx_count=TILE * 4, defer_sync=False,
    )
    tma_s, tma_g = cpasync.tma_partition(
        tma_atom, Int32(0), cute.make_layout(1),
        cute.group_modes(s, 0, 1),
        cute.group_modes(cute.local_tile(tma_tensor, (TILE,), (None,)), 0, 1),
    )
    dst = cute.local_tile(g_dst, (TILE,), (None,))

    if warp == 0:
        with cute.arch.elect_one():
            cpasync.prefetch_descriptor(tma_atom)
    if warp == 0:
        ps = make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, N_STG)
        for blk in cutlass.range(N_BLKS, unroll=1):
            pipe.producer_acquire(ps)
            cute.copy(tma_atom, tma_g[None, blk], tma_s[None, ps.index],
                      tma_bar_ptr=pipe.producer_get_barrier(ps))
            ps.advance()
        pipe.producer_tail(ps)
    if warp == 1:
        cs = make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, N_STG)
        for blk in cutlass.range(N_BLKS, unroll=1):
            pipe.consumer_wait(cs)
            for i in cutlass.range_constexpr(TILE // 32):
                dst[lane + i * 32, blk] = s[lane + i * 32, cs.index]
            cute.arch.fence_view_async_shared()
            cute.arch.sync_warp()  # Ned sync_warp as only 1 thread will signal in consumer_release
            pipe.consumer_release(cs)
            cs.advance()


@cute.jit
def go(g_src, g_dst, stream):
    tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), g_src, cute.make_layout(TILE), (TILE,),
    )
    kernel(g_dst, tma_atom, tma_tensor).launch(
        grid=[1, 1, 1], block=[64, 1, 1], smem=4096, stream=stream,
    )


if __name__ == "__main__":
    src = torch.arange(TILE * N_BLKS, device="cuda", dtype=torch.float32)
    dst = torch.zeros_like(src)
    go(from_dlpack(src, assumed_align=16), from_dlpack(dst, assumed_align=16),
       cuda.CUstream(torch.cuda.current_stream().cuda_stream))
    torch.cuda.synchronize()
    assert torch.equal(src, dst), f"FAIL: max diff={torch.abs(src - dst).max().item()}"
    print("PASS")
