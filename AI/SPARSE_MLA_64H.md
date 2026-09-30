# Sparse MLA backward: the 64-row tile

The SM100 sparse top-k MLA (DSA) backward (`flash_bwd_mla_sm100.py`) runs a 64-row head tile
when there are at most 64 Q heads per KV head: `tile_m == 64`. Heads 1..63 pad the tile, in
both backward modes (`AI/SPARSE_MLA_RECOMPUTE_P.md`, "Head counts"). 65..128 heads run the
128-row tile, two 64-row halves across the 2-CTA pair.

Three specializations apply only to the 64-row tile. They are chosen at trace time from the
tile, so the 128-row kernels keep their TMEM / SMEM maps and binaries, and there are no
user-facing knobs:
- **dV hand-off:** a pipelined accumulator hand-off to the epilogue (main kernel);
- **fused dK_rope:** dK_rope computed in the main kernel under recompute-P;
- **`dQdQvGemmKernelH64`:** a 1-CTA dQ/dQv kernel with a whole-row gather.

They are ported from PR #2914. Code pointers in `flash_bwd_mla_sm100.py`,
`flash_bwd_mla_dq_dqv_sm100_h64.py` and `interface.py` refer to the sections below.

Measurements: GB200, 64 heads, topk 2048, bf16, causal, `gather_bwd_recompute_p=True`,
`gather_bwd_token_chunk=4096`, per-kernel medians.

## dV hand-off (`FlashAttentionSparseMLABackwardSm100`)

The dV contribution of one 128-key group (`dV_g = P^T dO + dS^T Qv`, two 256-dim hdim_v
splits) is accumulated in TMEM by the MMA warp and drained by the 4 epilogue warps
(`dVacc_store`): `tcgen05.ld` -> a 32 KiB SMEM staging tile -> 64 KiB of
`red.global.add.v4.f32` per split into the gathered rows of `dv`.

At the 64-row tile:
- **No staging guard.** `pipeline_dV_epi` guards the staging tile against the next group's
  `dO / dOt / Qvt` operand loads. That is needed only when the staging aliases an operand
  stage, as it does at 128 heads. At 64 heads the staging is appended after the operand stages
  (`dv_staging_aliases_operand = False`), so the guard's barriers, acquires and release are
  compiled out (`num_stages_dV_epi`). With the guard, the TMA warp waited out the whole drain
  and scatter of group g before issuing `dO(g+1)`: 6.8k of an 11.2k-cycle group period.
- **Three TMEM dV stages** (`num_stages_dV = 3`, 128 columns each). `mma_dV_leg1(g+1)` split s
  no longer waits for the epilogue to release group g's stage s.
  - The accumulator is one `(MMA, MMA_M, MMA_N, STAGE)` fragment. `mma_dV_leg1` / `leg2` pass
    the stage selected by the dV pipeline state's `.index` as `acc=`, and `dVacc_store` slices
    its `tcgen05.ld` partition the same way, so (group, split) maps to stage `(2g + s) mod 3`
    on both sides.
  - The pipeline states persist across work tiles: 32 commits per token, so the ring rotates.
  - TMEM at 64 heads: dV 0-383, dP^T 384-415, S^T 416-447, dK_rope 448-511 (see "Fused
    dK_rope"). SMEM is unchanged.

**Effect.** The main kernel is 16.5% faster at T = S = 16K (18.32 -> 15.29 ms) and 12.7% at
64K. In the L2-resident regime (a chunk's K extent up to about 40K keys) the group period drops
from 11.2k to 9.4k cycles, and the kernel becomes co-bound: the epilogue's drain + scatter
(7.2k cycles) and the MMA chain both sit at the period. The reductions run at about 18.7 B/clk
per SM while active, against a ~22 B/clk `red.v4` ceiling.

In the DRAM regime (K extent above about 48K keys) the period is set by the MMA warp's waits
for gathered V and the operand re-stream, and the freer overlap of reductions with inbound
traffic costs 3-6% per chunk. Locality of the `dv` target is the lever there; deeper SMEM rings
are not (a third gathered-V stage was 12% slower, a third `dO / dOt / Qvt` stage 2% slower).

**Not built:** an epilogue that scatters with 1 KiB `cp.reduce.async.bulk` rows and releases
the TMEM stage before the scatter. After the third stage the epilogue does not bound the
kernel, and the release-before-scatter half measured 0%.

**Invariants**
- `mma_inner(swap_AB_stage=True)` (the load-P `dP = V dO^T` GEMM) selects the stationary dO
  split by the gathered-V ring index. That is only correct while
  `num_stages_V == num_hdimv_splits`, which the constructor asserts.
- The dV ring index lives only in the pipeline states. Both sides must advance once per
  (group, split), including groups whose keys are all sentinels.
- Without `pipeline_dV_epi`, the TMA warp's producer-tail set differs at 64 heads. The varlen
  multi-document, batched and token-chunked tests cover both tiles.

## Fused dK_rope (`FlashAttentionSparseMLABackwardSm100`, recompute-P)

With q / k rope the backward needs `dK_rope[key, d] = sum_h dS^T[key, h] Q_rope[h, d]` (64 rope
dims), besides the latent `dK + dV` the main kernel accumulates. At the 64-row tile under
recompute-P, the main kernel computes it. Otherwise the separate `dKGemmKernel`
(`flash_bwd_mla_dk_sm100.py`) computes it from the bf16 `ds` buffer: a third read of every `ds`
chunk, a Q_rope reload, and one scalar `red.add.f32` per lane (19 ms of a 120 ms backward at
64K).

The interface passes the fp32 `dk` view to the main kernel (`mdK`) and skips the dK kernel. The
view is sliced to the same chunk-shrunk K extent as `dv`. `fuse_dk_rope = recompute_p and rope
and qhead_tile == 64` follows from the compile key.

- **GEMM.** `tiled_mma_dKr`: `cta_group::2`, M = 128 keys (64 / 64 across the pair, like the dV
  GEMMs) x N = 64 rope dims (32 / 32) x K = 16, 4 k-steps over the 64 heads.
  - A is the resident dS^T tile `sdSt`, the same operand as the dV `dS^T Qv` GEMM (the A SMEM
    layouts of the two tiled MMAs are asserted equal).
  - B is `sQr2`, a second copy of Q_rope viewed dims-first (32 dims x 64 heads, 4 KiB per CTA,
    `MN_SW64`). It is TMA-loaded once per token on `pipeline_Qr`'s barrier together with the
    K-major `sQr` of the S^T GEMM (one barrier, two boxes). Padded heads load as zeros through
    the padded heads-first view.
  - The GEMM is issued in `mma_dV_leg2` after the two dV split commits, and before
    `pipeline_dSt.consumer_release`, whose commit also covers the dK_rope MMAs.
- **Accumulator.** Two 32-column TMEM stages (`pipeline_dKr`) at columns 448-511, so TMEM is
  512 / 512. A single stage measured the same; the second is a latency margin.
- **Epilogue.** The per-CTA accumulator (64 keys x 64 dims) is exactly one of the eight 64 x 64
  dV sub-tiles the epilogue drains per group, so it drains as a fifth sub-tile after split 1:
  `tcgen05.ld` -> the dV staging tile (parity 0) -> `LDS.128` -> `red.global.add.v4.f32` into
  `dk[key, 0:64]`, with the same indices and `0 <= idx < seqlen_k` guard as dV.
  - Direct `red.v4` from registers would scatter 32 rows x 16 B per warp instruction (7.6
    B/clk per SM, vs 22 at 4 rows x 128 B).
- **SMEM** 191,488 B (`sQr2` adds 4 KiB).

**Effect.** The backward kernel sum is 13.5% faster at 16K (25.03 -> 21.65 ms) and 10.0% at
64K. The main kernel grows by the dK share of the reduction bytes (+12.5%, 2,304 instead of
2,048 B per (token, key)). In the L2-resident regime that costs +8.8% per launch, far less than
the separate kernel. In the DRAM regime the extra fp32 `dk` target adds about 3.5 GB of DRAM
traffic per 4096-token chunk.

**Numerics.** The fused product uses the same bf16 dS^T tile that is stored to `ds`, with fp32
accumulation over heads. Only the order of the fp32 atomics differs from the separate kernel
(rel-L2 3-6e-7 between them; both 4e-7 to 1.3e-6 from an fp64 einsum of the same inputs).

**Invariants**
- The dK_rope MMAs must stay between the last dV GEMM of the group and the dS^T release in
  `mma_dV_leg2`: after the release the softmax warps may overwrite `sdSt`.
- `pipeline_Qr`'s `tx_count` is the sum of both Q_rope boxes when fused (`tma_copy_bytes_Qr`
  in `__call__`). A mismatch hangs the MMA prologue.
- The dK sub-tile reuses staging parity 0 after split 1 (whose last sub-tile used parity 1).
  The barrier after its `LDS` read-back frees the tile for the next group's first sub-tile.
  Changing `num_epi_subtiles` or the parity scheme must keep that order.
- `producer_tail(pipeline_dKr)` sits with the other MMA-warp tails. The varlen multi-document
  and batched tests cover exit hazards.

## `dQdQvGemmKernelH64`: 1-CTA dQ/dQv with a whole-row gather

`_compile_sparse_mla_dq_dqv` uses `dQdQvGemmKernelH64` (`flash_bwd_mla_dq_dqv_sm100_h64.py`)
for 1..64 heads per KV head, and the generic `dQdQvGemmKernel` (128 rows, cluster (1, 2))
above that. The class name is part of the compile key.

The GEMMs are `dQv = dS @ V[idx]` and `dQ = dS @ K_rope[idx]` per token. The generic kernel pads
the M tile to 128 rows and splits the 512 latent dims across a 2-CTA cluster. Each CTA then
gathers its 256-dim half of every key in 512-B pieces, indices loaded just before use. That
producer shape limits it: 1168 gather bytes per (token, key) at 46% tensor-pipe busy.

- **Tiles.** One token per CTA, cluster (1, 1), CLC persistent scheduling. M = 64; K tile 64
  keys. Fewer heads pad the tile: the head mode takes the real, dynamic extent, so TMA
  zero-fills the dS rows past it and drops those dQ / dQv rows.
- **GEMMs.** `tcgen05.mma.cta_group::1`, M = 64: `dQv` as two N = 256 tiles and `dQ` as one
  N = 64 tile.
  - A is the dS tile (heads x 64 keys, K-major SW128, TMA, 4 stages); B is the gathered stage
    re-viewed MN-major.
  - The second dQv N tile accumulates in the idle lane half of the M = 64 layout, so TMEM is dQ
    64 + dQv 256 = 320 of 512 columns (one accumulator stage; two would need 640).
  - The plain M = 64 instruction runs at 50% of peak, which puts the MMA floor at 8.2 ms per 64K
    backward.
- **Gather.** `CpasyncGatherKVManagerH64` (shared with the kb64 forward).
  - Each 64-key stage is the whole 1152-B row of every key (latent 1024 B + rope 128 B), fetched
    once per CTA by 128 threads as 16-B `cp.async.cg` copies.
  - The indices of block n+2 load into a second register set while block n is in flight.
    Completion signals with `cp.async.mbarrier.arrive.noinc`.
  - Rows with index -1 or >= seqlen_k are zero-filled by the predicated copy, so their
    contribution is exactly zero even with non-zero dS on those slots.
- **SMEM** 205,824 B: dS 4 x 8 KiB, rope 2 x 8 KiB, latent 2 x 64 KiB, and dedicated epilogue
  tiles, so the next token's gather never waits for the epilogue. A third latent stage does not
  fit beside a 2-stage dS ring.
- **Epilogue.** TMEM -> registers (`tcgen05.ld.16x256b`) -> bf16 -> SMEM -> TMA store, one 64 x 64
  subtile at a time.
- **Registers.** No per-role `setmaxnreg`: with `min_blocks_per_mp=1` every role runs at the 168
  registers ptxas assigns (the kernel needs 49; 0 B local). Per-role budgets over 11 warps trap
  or hang once honoured (`setmaxnreg.inc` below the launch count, or waiting on registers an
  incomplete warpgroup never releases).

**Effect.** dQ / dQv takes 2.72 ms at 16K (generic kernel 4.43 ms) and 12.23 ms at 64K
(20.62 ms). At 24 / 32 heads it is 2.60 ms vs 4.16 ms at 16K. ncu at 16K: tensor pipe 75% busy,
the plain M = 64 issue time; the gather runs at 48 B/clk per SM (the 2-stage producer's ceiling
is about 54); dS traffic is 128 B per pair instead of 384. The kernel is MMA / fill co-bound;
the `.ws` M = 64 N = 256 form (80% rate) and a third latent stage are the next levers, each
worth about 10-15%.

**Numerics.** dq and dqv are bitwise identical to the generic kernel's, batched and varlen, and
at padded head counts: both accumulate the same 16-key k-blocks in the same order in fp32 and
round once to bf16.

## Tests

`tests/cute/test_flash_attn_mla.py`, the sparse backward tests (`-k "mla_sparse or mla_sink or
topk_order_invariance or precise_dpsum or preprocess_tile_tail"`):
- the fully-masked-rows, token-chunk (rectangular and varlen), learnable-sink, top-k
  order-invariance and varlen-sentinel tests run at 64 and 128 heads, covering every backward
  mode at both tiles (load-P and recompute-P, chunked and unchunked, varlen and batched);
- `test_flash_attn_mla_sparse_bwd_sentinel` and `..._sentinel_varlen` run recompute-P at every
  head count, with int32 canaries around the preallocated `dk` / `dv` that guard the fused
  scatter;
- `test_flash_attn_mla_sparse_bwd_recompute_p_padded` compares recompute-P with load-P at 24 / 1
  heads, including the H64 dQ/dQv output.
