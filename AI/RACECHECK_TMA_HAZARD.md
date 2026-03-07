# compute-sanitizer racecheck hazard with `cp.async.bulk` (raw TMA)

## Summary

`compute-sanitizer --tool=racecheck` reports false-positive shared-memory race
hazards when `cp.async.bulk` (raw-address TMA) is used in a cross-warp
producer/consumer pipeline inside a dynamic loop. The same pattern with
`cp.async.bulk.tensor` (descriptor-based TMA) reports **zero hazards**.

The fix for the flash backward kernel is to switch the LSE/dPsum copies from
`CopyBulkG2SOp` (`cp.async.bulk`) to `CopyBulkTensorTileG2SOp`
(`cp.async.bulk.tensor`) using `cpasync.make_tiled_tma_atom`.

## Affected code

`flash_attn/cute/flash_bwd_sm100.py` — the SM100 backward attention kernel.

Only **LSE** and **dPsum** buffers are affected because they are the only
TMA-loaded buffers consumed by thread-level shared memory reads (`lds`).
Q/K/V/dO are consumed by UMMA hardware instructions, which do not generate
thread-level `lds` and therefore never trigger racecheck.

## Root cause

racecheck performs dynamic (runtime) analysis. It instruments every shared
memory access and checks for conflicting accesses lacking a recognized
happens-before relationship. It recognizes:

- `bar.sync` / `bar.arrive` + `bar.wait` (named barriers, `__syncthreads`)
- `mbarrier` operations — but **only** when paired with `cp.async.bulk.tensor`

It does **not** recognize `mbarrier` as establishing happens-before for
`cp.async.bulk` (the raw-address variant). Both instructions use the identical
`mbarrier::complete_tx::bytes` completion mechanism, and the generated PTX for
the pipeline (init, arrive_expect_tx, try_wait_parity, arrive, fence) is
byte-for-byte identical between the two cases. The only difference is the copy
instruction itself.

### Instruction comparison

| Variant | PTX instruction | racecheck result |
|---------|----------------|-----------------|
| Raw 1D | `cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes` | **hazard** |
| Raw 1D (cluster scope) | `cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes` | **hazard** |
| Descriptor 1D | `cp.async.bulk.tensor.1d.shared::cta.global.tile.mbarrier::complete_tx::bytes` | clean |
| Descriptor 2D | `cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes` | clean |

## Proof that it is a false positive

1. **Data correctness**: all variants produce bit-identical results.
2. **Single-warp test**: when one warp does both the TMA write and the thread
   read in the same dynamic loop, racecheck reports zero hazards — same
   mbarrier sync, same addresses, same instructions.
3. **Unrolled loop**: when the loop is fully unrolled (`unroll_full=True`),
   racecheck reports zero hazards — proves it can track mbarrier within a
   single static code path, just not across a dynamic branch back-edge between
   warps.
4. **Named barrier**: adding a `bar.sync` (named barrier) per iteration between
   the producer and consumer warps eliminates the hazard — proves the sync is
   correct and racecheck simply needs a primitive it recognizes.
5. **Descriptor TMA**: switching from `cp.async.bulk` to `cp.async.bulk.tensor`
   with identical pipeline code eliminates the hazard — proves the mbarrier
   protocol itself is correct.

## Minimal reproducers

All in `benchmarks/`:

| File | What it tests | Result |
|------|--------------|--------|
| `racecheck_false_positive_repro.py` | `cp.async.bulk` + mbarrier pipeline in cross-warp loop | 1 error (false positive) |
| `racecheck_1d_raw_ptx.py` | Same but with inline PTX `cp.async.bulk.shared::cta.global` | 1 error |
| `racecheck_tma2d_repro.py` | `cp.async.bulk.tensor.2d` via `make_tiled_tma_atom` | 0 hazards |
| `racecheck_tma1d_descriptor.py` | `cp.async.bulk.tensor.1d` via `make_tiled_tma_atom` | 0 hazards |

Each reproducer is a self-contained ~80-line kernel with a 2-warp
producer/consumer pipeline iterating over 4 blocks with 2-stage double
buffering. Run with:

```bash
# Verify correctness
python benchmarks/racecheck_false_positive_repro.py

# Show the hazard
compute-sanitizer --tool=racecheck python benchmarks/racecheck_false_positive_repro.py

# Show that descriptor TMA is clean
compute-sanitizer --tool=racecheck python benchmarks/racecheck_tma1d_descriptor.py
```

## Fix

Change `copy_stats` in the load function from:

```python
copy_atom_stats = cute.make_copy_atom(cpasync.CopyBulkG2SOp(), Float32)
copy_stats = partial(cute.copy, copy_atom_stats)
```

to a descriptor-based TMA using `cpasync.make_tiled_tma_atom` with
`CopyBulkTensorTileG2SOp`. This generates `cp.async.bulk.tensor.1d` instead of
`cp.async.bulk`, which racecheck can track correctly.

The change is purely in how the copy is issued — the pipeline protocol
(mbarrier init, arrive_expect_tx, try_wait_parity, consumer_release) remains
identical.

## Investigation timeline

1. Observed 2 racecheck errors on LSE (line 2136) and dPsum (line 2160) in
   `flash_bwd_sm100.py`. Q/K/V/dO clean.
2. Noticed Q/K/V/dO use UMMA consumers (no thread `lds`) while LSE/dPsum use
   thread-level `autovec_copy` from smem — explains why only LSE/dPsum trigger.
3. Built minimal 2-warp pipeline kernel reproducing the hazard.
4. Showed single-warp version is clean (same mbarrier, same addresses).
5. Showed fully-unrolled version is clean (racecheck tracks mbarrier within
   straight-line code).
6. Showed `bar.sync` per iteration fixes it (racecheck needs a sync it
   recognizes across the loop back-edge).
7. Showed `cp.async.bulk.tensor.2d` is clean — different instruction, same
   pipeline.
8. Showed `cp.async.bulk.tensor.1d` is also clean — confirms the issue is
   raw-address `cp.async.bulk` vs descriptor `cp.async.bulk.tensor`, not
   dimensionality.
9. Showed raw inline PTX `cp.async.bulk.shared::cta.global` also triggers it —
   confirms it is not a CuTe DSL abstraction issue.

## Backup

`flash_attn/cute/flash_bwd_sm100_gmem_fix.py` contains a working but slower
fix where compute warps read LSE/dPsum directly from global memory, bypassing
the TMA smem pipeline entirely.
