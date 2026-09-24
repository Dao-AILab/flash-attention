# Sparse MLA at 64 Q heads: native 64-head training-path changes

Design record for the compile-time specializations of the SM100 sparse top-k MLA
(DSA) training kernels that apply when the backward head tile is 64 rows
(`nheads_per_kv == 64`, `tile_m == 64` in `flash_bwd_mla_sm100.py`). Every change is
gated on that tile at trace time; the 128-head kernels (`tile_m == 128`, two 64-row
halves across the 2-CTA pair) keep their TMEM/smem maps and their binaries. There are
no user-facing knobs: the interface dispatches on the head count as before.

Sections are added per change; this file is the target of the one-line pointers in
`flash_bwd_mla_sm100.py` / `interface.py`.

## C8: pipelining the dV accumulator hand-off (`FlashAttentionSparseMLABackwardSm100`)

### Problem

In the main backward kernel the dV contribution of one 128-key group
(`dV_g = P^T dO + dS^T Qv`, two 256-dim hdimv splits) is accumulated in TMEM by the
MMA warp and drained by the 4 epilogue warps (`dVacc_store`): `tcgen05.ld` -> a 32 KiB
smem staging tile -> 64 KiB of `red.global.add.v4.f32` per split into the gathered
rows of `dv`. Two hand-offs serialized that drain against the next group's MMAs:

1. `pipeline_dV_epi`, an Async pipeline from the TMA warp to the epilogue, guards the
   staging tile against the `dO/dOt/Qvt` operand loads of the next group. At 128 heads
   the staging aliases one of the two 16 KiB operand stages inside `sQv`, so the guard
   is needed. At 64 heads the staging is appended *after* the operand stages
   (`_get_shared_storage_cls`: `staging_elems != stage_elems`) and aliases nothing, but
   the guard still made the TMA warp wait for the whole drain + scatter of group g
   before issuing `dO(g+1)`; the MMA warp then waited for `dO(g+1)` in the dP gemm.
   Measured with `%clock64` stamps at 64 heads, T=S=16k: the TMA warp spent 6.8k of
   the 11.2k-cycle group period in that acquire.
2. The dV accumulator had one TMEM stage per hdimv split (2 x 128 columns), so
   `mma_dV_leg1(g+1)` split s could only start after the epilogue had released stage s
   of group g (after its scatter).

### Change

- **S0** (`num_stages_dV_epi`): the guard's mbarriers, pipeline object, TMA-warp
  acquires and epilogue release are compiled out when the staging does not alias an
  operand stage (`dv_staging_aliases_operand = False`, i.e. 64 heads). At 128 heads
  `num_stages_dV_epi == num_stages_dV == 2` and nothing changes.
- **S1** (`num_stages_dV = 3 if tile_m == 64 else 2`): a third 128-column TMEM dV stage.
  The accumulator is one `(MMA, MMA_M, MMA_N, STAGE)` fragment re-based at
  `tmem_offset_dV0`; `mma_dV_leg1/leg2` pass the ring stage selected by the dV
  pipeline state's `.index` as `acc=` to each gemm (`mma_inner(acc=...)`), and
  `dVacc_store` slices its `tcgen05.ld` partition by `consumer_state_dV.index`. The
  (group, split) -> stage map is therefore `(2g + s) mod 3` on both sides; the pipeline
  states persist across work tiles (32 commits per token, 32 mod 3 = 2, the ring
  rotates). TMEM at 64 heads: dV stages 0-127 / 128-255 / 256-383, dP^T 384-415, S^T
  416-447 (448 of 512; 64 columns free). 128 heads: 384 columns as before. Shared
  memory is unchanged (187,392 B at 64 heads, recompute-P + rope).
- Not built: **S2**, an epilogue that issues the scatter as 1 KiB `cp.reduce.async.bulk`
  rows from padded staging tiles and releases the TMEM stage before the scatter. The
  instrumented builds showed that after S1 the epilogue is not what bounds the kernel
  (below), and its cheap half (release before scatter) measured 0 %.

### Effect (GB200, 64 heads, W = 2048, bf16, causal, `gather_bwd_recompute_p=True`, `gather_bwd_token_chunk=4096`)

Main backward kernel (sum over token chunks, per-kernel medians):

| T = S | before | after | |
|---:|---:|---:|---|
| 16384 | 18.32 ms | 15.29 ms | -16.5 % |
| 65536 | 77.20 ms | 67.42 ms | -12.7 % |

GLM-5.2-shape train step (fwd + bwd, sequential on one GPU): 64k 148.9 -> 139.8 ms
(bwd 119.8 -> 110.5), 16k 35.6 -> 32.7 ms, 4k 8.57 -> 7.95 ms. Peak memory (13.69 GiB at
64k) and saved activations (8.02 GiB) unchanged. `out`, `lse`, `dq`, `dqv` and the bf16
`ds` intermediate are bitwise identical to the previous kernel at 64 and at 128 heads;
`dk`/`dv` differ within the run-to-run noise of the fp32 atomics; rel-L2 vs an fp64
reference is unchanged in every case measured (iid and peaked inputs).

Where the gain comes from, and where it stops (`%clock64` per-group budgets):

- L2-resident regime (K extent of a chunk <= ~40k keys, i.e. `dv` target + gather table
  fit in L2): period 11.2k -> 9.4k cycles per group. The epilogue's drain + scatter
  (7.2k) and the MMA chain with its operand waits (S 2.8k + dP 3.2k + leg1 1.75k +
  leg2 1.35k) both sit at the period: the kernel is co-bound. The reds run at ~18.7 B/clk
  per SM while active against a ~22 B/clk `red.v4` egress ceiling.
- DRAM regime (K extent > ~48k; at 64k the last 6 of 16 chunks): the period is set by
  the MMA warp's waits for gathered V and for the `dO/dOt/Qvt` re-stream under DRAM
  pressure (25 GB of DRAM traffic per 4096-token chunk, L2 hit ~47 %); the epilogue idles
  ~4.5k cycles per group. The hand-off changes buy nothing there, and the freer overlap
  of reds with the inbound traffic costs 3-6 %: one 4096-token chunk of the main kernel
  as a function of its K extent (queries at the end of the sequence, same MMA work):

  | K extent | 8k | 16k | 32k | 48k | 64k | 96k | 128k |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | before, ms | 4.66 | 4.68 | 4.78 | 4.94 | 5.13 | 5.63 | 6.04 |
  | after, ms | 3.88 | 3.89 | 3.94 | 4.41 | 5.28 | 5.98 | 6.29 |

  Consequently the 4k x 64k and 4k x 128k context-parallel tail cases are 2.6-2.8 %
  slower per train step, and 128k x 128k is only 1 % faster. Locality of the `dv`
  target (e.g. processing a chunk's top-k keys in key-range passes) is the lever for
  that regime; deeper smem rings are not (a third gathered-V stage, +32 KiB, was 12 %
  slower; a third `dO/dOt/Qvt` stage, +16 KiB, 2 % slower: the kernel pays for every
  smem byte through the L1 carve-out).

### Invariants worth knowing

- `mma_inner(swap_AB_stage=True)` (the load-P `dP = V dO^T` gemm) selects the
  stationary dO split by the gathered-V ring index; this is only correct while
  `num_stages_V == num_hdimv_splits`, which the constructor now asserts.
- The dV ring index is carried only by the pipeline states; both sides must advance
  once per (group, split), including for groups whose keys are all sentinels.
- The `pipeline_dV_epi` removal changes the TMA warp's producer-tail set at 64 heads;
  the varlen multi-document, batched and token-chunked tests cover both head tiles.

### Validation

`tests/cute/test_flash_attn.py -k "mla_sparse or mla_sink or topk_order_invariance or
precise_dpsum or preprocess_tile_tail"` (fake-tensor compile and GPU), with the
fully-masked-rows, token-chunk (rectangular and varlen), learnable-sink, top-k
order-invariance and varlen-sentinel tests parametrized over `nheads in (128, 64)` so
the 64-row tile is exercised by every backward mode (load-P and recompute-P, chunked and
unchunked, varlen and batched).

## C1: dK_rope fused into the main backward (`FlashAttentionSparseMLABackwardSm100`, recompute-P)

### Problem

With q/k rope the backward needs `dK_rope[key, d] = sum_h dS^T[key, h] Q_rope[h, d]` (64
rope dims) besides the latent `dK + dV` that the main kernel already accumulates in its
dV accumulator. A separate kernel (`flash_bwd_mla_dk_sm100.py`) computed it from the
bf16 `ds` buffer the main kernel writes: per token it re-read `ds` (128 B per
(token, key)), re-loaded `Q_rope`, ran the same 128x64x64 GEMM and scattered with one
scalar `red.add.f32` per lane (1 row x 128 B per warp instruction, ~45-57 % of the
lane-op ceiling). At 64 heads, W = 2048, T = S = 64k it was 19 ms of a 120 ms backward
(all of it head-count independent), and it added a third read of every `ds` chunk.

### Change (64 heads, `recompute_P` with q/k rope; everything else unchanged)

The interface passes the fp32 `dk` view to the main kernel (`mdK`, after `lse_log2`,
sliced to the same chunk-shrunk K extent as `dv`: `dk[:, :k_end]` on the causal
non-varlen token chunks, the full tensor with the clamped `cu_seqlens_k` for varlen) and
skips the dk kernel. `fuse_dk_rope = recompute_p and q, k given and qhead_tile == 64` is
a function of the compile key, so no key field was added. In the kernel
(`self.fuse_dk_rope = mdK is not None`, asserted to imply recompute-P + rope + 64 heads):

- **GEMM.** `tiled_mma_dKr` = cta_group::2 M = 128 keys (split 64/64 across the pair like
  the dV gemms) x N = 64 rope dims (split 32/32) x K = 16, 4 k-steps over the 64 heads,
  A MN-major, B MN-major. A is the resident dS^T tile `sdSt` (the same operand as the dV
  `dS^T Qv` gemm; the A smem layout of the two tiled MMAs is asserted equal at trace
  time, since A's layout depends on M, K and majorness only). B is a second copy of the
  stationary Q_rope viewed dims-first, `sQr2`: 32 dims x 64 heads bf16 = 4 KiB per CTA,
  `MN_SW64` atom, TMA-loaded once per token on `pipeline_Qr`'s barrier together with the
  K-major `sQr` of the S^T gemm (one barrier, two boxes, `tx_count` = both). The existing
  `sQr` cannot serve: it is K-major and heads-split. The gemm is issued in `mma_dV_leg2`
  after the two dV split commits (so the epilogue's dV hand-off is not delayed) and
  before `pipeline_dSt.consumer_release`, whose `tcgen05.commit` therefore also covers
  the dK_rope MMAs before the softmax warps may overwrite dS^T.
- **Accumulator.** Two 32-column TMEM stages (`pipeline_dKr`, UmmaAsync, producer MMA
  warp, consumer = the 128 epilogue threads of both CTAs, like `pipeline_dV`) at columns
  448-479 / 480-511: TMEM is now exactly 512/512 at 64 heads. The epilogue drains group
  g as (g, 0), (g, 1), dk(g); a one-stage accumulator (TMEM 480) was measured to cost
  nothing (main kernel 16.60 / 75.43 ms vs 16.62 / 75.66 at T=S=16k / 64k) because the
  epilogue finishes dk(g) some 3k cycles before the MMA warp, which still has S(g+1),
  dP(g+1) and leg1(g+1) to issue, reaches leg2(g+1). The second stage is kept anyway: it
  removes that latency coupling for any later change that makes the epilogue lag (a
  different drain order or owner), and the 32 columns it costs enable nothing today.
- **Epilogue.** The per-CTA accumulator is 64 keys x (32, 2) dims = exactly one of the
  eight (64 keys x 64 columns) dV sub-tiles the epilogue already drains per group, so it
  is handled as a fifth sub-tile after split 1: `tcgen05.ld 32x32b x32` -> the 32 KiB dV
  staging tile (sub-tile parity 0, whose two stages split 1's last sub-tile left free) ->
  `LDS.128` -> eight `red.global.add.v4.f32` per thread at the epilogue's
  4-rows-x-128-B-per-warp-instruction geometry into `dk[key, 0:64]`, same top-k indices,
  same guard `0 <= idx < seqlen_k` (the chunk-shrunk extent) as dV. No new staging: the
  dV staging does not alias an operand stage at 64 heads (asserted), and the epilogue
  is sequential. Direct `red.v4` from the `tcgen05.ld` registers was rejected because it
  would be the 32-rows-x-16-B geometry (7.6 B/clk per SM vs 22 at 4 x 128 B); a
  dedicated dk tile would only have paid for itself with the (unbuilt) bulk-reduce
  epilogue's padded rows.
- Shared memory 187,392 -> 191,488 B (`sQr2`; the two dKr mbarrier pairs fit in the header
  padding). 128 heads and load-P: `mdK` is None, TMEM / smem / SASS unchanged.

Numerics: the fused product uses the identical bf16 dS^T tile that is TMA-stored to
`ds` and fp32 accumulation over the heads, so only the order of the fp32 atomic
accumulation changes (fused vs standalone kernel rel-L2 3-6e-7 on the same `ds`; both
4e-7 .. 1.3e-6 from an fp64 einsum of the same bf16 inputs).

### Effect (GB200, 64 heads, W = 2048, bf16, causal, `gather_bwd_recompute_p=True`, `gather_bwd_token_chunk=4096`)

Backward kernels (per-kernel medians, `torch.profiler`, one GPU, back to back with the
previous build):

| T = S | main kernel before -> after | dk kernel before -> after | backward kernel sum | |
|---:|---:|---:|---:|---|
| 16384 | 15.29 -> 16.62 ms | 4.71 -> 0 | 25.03 -> 21.65 ms | -13.5 % |
| 65536 | 67.36 -> 75.66 ms | 19.20 -> 0 | 109.43 -> 98.54 ms | -10.0 % |

GLM-5.2-shape train step (fwd + bwd, sequential on one GPU): 64k 139.8 -> 129.4 ms
(bwd 110.5 -> 99.2), 16k 32.7 -> 29.3 ms, 4k 7.95 -> 7.12 ms, 128k 316.4 -> 300.0 ms; the
4k x 64k / 4k x 128k context-parallel tail cases 10.39 -> 10.01 / 11.72 -> 11.26 ms (now
1 % faster than before the dV hand-off change, which had cost them 2.6-2.8 %). Peak
memory (13.69 GiB at 64k) and saved activations (8.02 GiB) unchanged. `out`, `lse`, `dq`,
`dqv` and the bf16 `ds` intermediate are bitwise identical to the previous kernel at 64
and at 128 heads; `dk`/`dv` differ within the run-to-run noise of the fp32 atomics; rel-L2
vs an fp64 reference is unchanged to the printed digit in every case measured (iid and
peaked inputs, including `dk_rope`).

Where the main kernel's increment comes from (ncu, 16k chunk 2 and 64k chunk 15):

- The reduction bytes grow by exactly the dk share (+12.5 %: 2,304 instead of 2,048 B per
  (token, key), 37.7 M instead of 33.6 M `red.v4`), nothing else about the launch changes
  (same TMA re-stream and gather bytes, +4 KiB smem, same registers). In the L2-resident
  regime the epilogue's `LDS -> RED` loop is the co-bound resource (about 50 % of the
  SM->L2 write port over the launch, ~18.7 B/clk/SM while active), so the fifth sub-tile's
  16 KiB per group cost their port time almost 1:1: +8.8 % per launch (+0.3 ms per
  4096-token chunk). The MMA warp does not stall on the two-stage dK_rope hand-off (its
  acquire is not among the top stall lines).
- In the DRAM regime (K extent > ~48k keys) the launch grows by ~15 % although the write
  port is less utilised: the extra 16 MB fp32 `dk` target and its write-allocate traffic
  add ~3.5 GB of DRAM traffic per 4096-token chunk (L2 hit 47.5 -> 45.4 %) on chunks that
  were already DRAM-bound. Key-range locality of the scatter targets is the lever there.
- A one-stage dK_rope accumulator (TMEM 480) measured the same time as two stages (see
  the accumulator note above); two stages are kept as a latency margin.

### Invariants worth knowing

- The dK_rope MMAs must stay between the last dV gemm of the group and the dS^T release
  in `mma_dV_leg2`: after the release the softmax warps may overwrite `sdSt`.
- `pipeline_Qr`'s `tx_count` is the sum of both Q_rope boxes when fused (`tma_copy_bytes_Qr`
  is adjusted in `__call__`); a box or byte-count mismatch hangs the MMA prologue.
- The dk sub-tile reuses staging parity 0 after split 1 (whose last sub-tile used parity
  1); the barrier after its `LDS` read-back is what frees the tile for the next group's
  first sub-tile. Changing `num_epi_subtiles` or the parity scheme must keep that order.
- `producer_tail(pipeline_dKr)` sits with the other MMA-warp tails; the varlen
  multi-document and batched tests are the exit-hazard coverage.

### Validation

The sparse-MLA subset above, with `test_flash_attn_mla_sparse_bwd_sentinel` and
`..._sentinel_varlen` additionally parametrized over `recompute_p` (64 / 128 heads) so
the int32 canaries around preallocated `dk`/`dv` also guard the fused scatter (-1
sentinels and the per-chunk key extent); the 128-head and load-P kernels are compared
bitwise against the parent build.
