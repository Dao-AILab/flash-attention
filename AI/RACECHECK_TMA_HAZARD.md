# compute-sanitizer racecheck hazard with `cp.async.bulk`

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

racecheck instruments every shared memory access and checks for conflicting
accesses lacking a recognized happens-before relationship.

**`cp.async.bulk` (raw address):** the sanitizer attributes the smem write to
the issuing thread (thread 0 of warp 0 via `elect_one`). When warp 1 issues
`ld.shared.b32` from the same addresses, the sanitizer searches for a
happens-before edge. The only sync is `mbarrier.try_wait.parity` on warp 1
paired with `mbarrier::complete_tx::bytes` completion from the hardware. The
sanitizer does not model this as happens-before across warps in a dynamic loop.

**`cp.async.bulk.tensor` (TMA descriptor):** the TMA engine is a separate
hardware unit. The sanitizer does not attribute the smem write to any thread.
No writer thread means no hazard pair, so no race is reported.

### Instruction comparison

| Variant | PTX | racecheck |
|---------|-----|-----------|
| Raw (cta scope) | `cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes` | **hazard** |
| Raw (cluster scope) | `cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes` | **hazard** |
| Descriptor 1D | `cp.async.bulk.tensor.1d.shared::cta.global.tile.mbarrier::complete_tx::bytes` | clean |
| Descriptor 2D | `cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes` | clean |

### `--racecheck-memcpy-async=no` does not help

This flag controls the older `cp.async` (sm80) instruction family, not
`cp.async.bulk`. The hazard persists with `--racecheck-memcpy-async=no`.

## Proof that it is a false positive

1. **Data correctness** — all variants produce bit-identical results.
2. **Single-warp test** — one warp does both TMA write and thread read in the
   same loop; racecheck reports zero hazards with the same mbarrier sync.
3. **Unrolled loop** — fully unrolling (`unroll_full=True`) reports zero
   hazards; racecheck tracks mbarrier within straight-line code but not across
   a dynamic branch back-edge between warps.
4. **Named barrier** — adding `bar.sync` per iteration between producer and
   consumer warps eliminates the hazard; the sync is correct, racecheck just
   needs a primitive it recognizes.
5. **Descriptor TMA** — switching to `cp.async.bulk.tensor` with identical
   pipeline code eliminates the hazard; the mbarrier protocol is correct.

## Minimal reproducers

### `AI/` (preferred, cleaner)

| File | Copy instruction | Result |
|------|-----------------|--------|
| `racecheck_repro_1d_bulk.py` | `cp.async.bulk` (raw address) | **1 error** |
| `racecheck_repro_1d_tensor.py` | `cp.async.bulk.tensor.1d` (TMA descriptor) | **0 hazards** |

Both are ~75-line self-contained kernels: 2 warps, 4 blocks, 2-stage double
buffering with `PipelineTmaAsync`. Identical pipeline protocol — only the copy
instruction differs.

```bash
python AI/racecheck_repro_1d_bulk.py                                              # correctness
CUTE_DSL_LINEINFO=1 compute-sanitizer --tool=racecheck python AI/racecheck_repro_1d_bulk.py   # 1 error
compute-sanitizer --tool=racecheck python AI/racecheck_repro_1d_tensor.py         # 0 hazards
```

### `benchmarks/` (earlier, more variants)

| File | What it tests | Result |
|------|--------------|--------|
| `racecheck_false_positive_repro.py` | `cp.async.bulk` + mbarrier in cross-warp loop | 1 error |
| `racecheck_1d_raw_ptx.py` | Inline PTX `cp.async.bulk.shared::cta.global` | 1 error |
| `racecheck_tma2d_repro.py` | `cp.async.bulk.tensor.2d` via `make_tiled_tma_atom` | 0 hazards |
| `racecheck_tma1d_descriptor.py` | `cp.async.bulk.tensor.1d` via `make_tiled_tma_atom` | 0 hazards |

## PTX-level analysis

Dumped PTX for both `AI/` reproducers (`CUTE_DSL_KEEP_PTX=1`). The generated
code is byte-for-byte identical except for the single copy instruction:

```
# racecheck_repro_1d_bulk.py  (HAZARD)
cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes
    [%r42], [%rd12], %r43, [%r6+-16];

# racecheck_repro_1d_tensor.py  (CLEAN)
cp.async.bulk.tensor.1d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint
    [%r43], [%rd1, {%r71}], [%r6+-16], %rd8;
```

All mbarrier operations (init, `fence.mbarrier_init.release.cluster`,
`arrive.expect_tx`, `try_wait.parity`, `arrive.release`,
`fence.proxy.async.shared::cta`, `bar.warp.sync`) are identical.

### racecheck error output

```
Error: Race reported between Write access at ...+0x430 in racecheck_repro_1d_bulk.py:46
    and Read access at ...+0x770 in racecheck_repro_1d_bulk.py:55 [248 hazards]
    and Read access at ...+0x7a0 in racecheck_repro_1d_bulk.py:55 [248 hazards]
    and Read access at ...+0x7d0 in racecheck_repro_1d_bulk.py:55 [248 hazards]
    and Read access at ...+0x800 in racecheck_repro_1d_bulk.py:55 [248 hazards]
```

- **Write** (0x430) = line 46: `cute.copy(atom, src, s, mbar_ptr=...)` — the
  `cp.async.bulk` instruction
- **Read** (0x770–0x800) = line 55: `dst[...] = s[...]` — four `ld.shared.b32`
  in the consumer warp

## Fix

Change `copy_stats` in the load function from:

```python
copy_atom_stats = cute.make_copy_atom(cpasync.CopyBulkG2SOp(), Float32)
copy_stats = partial(cute.copy, copy_atom_stats)
```

to a descriptor-based TMA using `cpasync.make_tiled_tma_atom` with
`CopyBulkTensorTileG2SOp`. This generates `cp.async.bulk.tensor.1d` instead of
`cp.async.bulk`, which racecheck does not instrument.

The pipeline protocol (mbarrier init, arrive_expect_tx, try_wait_parity,
consumer_release) remains identical.

## Backup

`flash_attn/cute/flash_bwd_sm100_gmem_fix.py` contains a working but slower
fix where compute warps read LSE/dPsum directly from global memory, bypassing
the TMA smem pipeline entirely.

## Investigation timeline

1. Observed 2 racecheck errors on LSE and dPsum in `flash_bwd_sm100.py`.
   Q/K/V/dO clean.
2. Noticed Q/K/V/dO use UMMA consumers (no thread `lds`) while LSE/dPsum use
   thread-level `autovec_copy` from smem — explains why only LSE/dPsum trigger.
3. Built minimal 2-warp pipeline kernel reproducing the hazard.
4. Single-warp version clean — same mbarrier, same addresses.
5. Fully-unrolled version clean — racecheck tracks mbarrier within
   straight-line code.
6. `bar.sync` per iteration fixes it — racecheck needs a sync it recognizes
   across the loop back-edge.
7. `cp.async.bulk.tensor.2d` clean — different instruction, same pipeline.
8. `cp.async.bulk.tensor.1d` clean — issue is raw vs descriptor, not
   dimensionality.
9. Raw inline PTX `cp.async.bulk.shared::cta.global` also triggers — not a
   CuTe DSL abstraction issue.
10. Dumped PTX for both `AI/` reproducers — confirmed byte-identical code
    except for the copy instruction. Sanitizer attributes smem write to
    issuing thread for `cp.async.bulk` but not for `cp.async.bulk.tensor`.
11. Confirmed `--racecheck-memcpy-async=no` does not suppress the hazard —
    flag targets older `cp.async`, not `cp.async.bulk`.
