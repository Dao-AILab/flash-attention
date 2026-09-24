# Sparse MLA at 64 Q heads: native 64-head training-path changes

Design record for the compile-time specializations of the SM100 sparse top-k MLA
(DSA) training kernels that apply when the Q-head count per KV head is exactly 64:
the backward main kernel changes C8 and C1 (gated on `tile_m == 64` in
`flash_bwd_mla_sm100.py`) and the native 1-CTA forward kernel F1
(`flash_fwd_mla_sm100_h64.py`, dispatched from `interface.py`). The 128-head kernels keep
their TMEM/smem maps and their binaries. There are no user-facing knobs: the interface
dispatches on the head count as before.

Sections are added per change; this file is the target of the one-line pointers in
`flash_bwd_mla_sm100.py`, `flash_fwd_mla_sm100_h64.py` and `interface.py`.

Contents: C8 (dV accumulator hand-off pipelining), C1 (fused dK_rope), F1 (1-CTA forward),
C6-dq (1-CTA dQ/dQv kernel with a whole-row gather), F3 (Q in TMEM, `.ws` TS dual GEMM,
three latent stages).

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

---

## F1: native 1-CTA 64-head forward (`FlashAttentionMLAForwardSm100H64`)

Why sparse-MLA (top-k gathered KV, DeepSeek-style DSA) forwards with exactly 64 Q heads per KV
head run a dedicated 1-CTA kernel, what it does differently from the 2-CTA kernel, which
forwards it serves, how it is tested, and what it costs.
Code: `flash_fwd_mla_sm100_h64.py` (`FlashAttentionMLAForwardSm100H64`), `topk_gather_kv.py`
(`CpasyncGatherKVManagerH64`), `blackwell_helpers.py` (`gemm_ws_ptx_partial`), `interface.py`
(`use_mla_fwd_h64`: `qhead_per_kvhead == 64 and (gather_bwd_recompute_p or not requires_grad)`
on SM100/SM110; the head tile `sparse_mla_qhead_tile(64, min_tile=64)` in the compile key
separates its binaries from the 128-row ones).

### Why

The 2-CTA sparse-MLA forward (`FlashAttentionMLAForwardSm100`) tiles one token as 128 packed
Q-head rows across a CTA pair (`cta_group::2`, M = 128). With 64 heads (GLM-5.2 shape:
`q_latent [T,64,512] + q_rope [T,64,64]`, `kv_latent [S,512]`, top-k 2048) it pads the heads to
128 (`pack_gqa.qheads_first_tma_view`), so half of every MMA row is padding, and each CTA of the
pair gathers the full top-k latent rows for its half of the N split, so every gathered key is
fetched twice (2176 B per (token, key) instead of 1152). The 64-head kernel puts one token's 64
heads on M = 64 of `tcgen05.mma.ws` UMMAs, gathers each key once, and needs no padding.

### Design

- **Tiles.** One token per CTA (`cta_tile_m = 64` = the head count), 64 keys per block
  (`tile_n = 64`), 32 blocks for top-k 2048, walked from the last block to the first as the
  2-CTA kernel does (the index blocks' order matters for the running max: see
  `AI/SPARSE_MLA_EXACT_SOFTMAX_MAX.md`). Cluster (1,1,1), CLC persistent tile scheduler.
- **GEMMs** (`blackwell_helpers.gemm_ws_ptx_partial`, inline PTX `tcgen05.mma.ws.cta_group::1`,
  SS operands, bf16 -> fp32): `S = Q_rope K_rope^T` (M 64 x N 64 x K 64, 4 k-steps) `+ Qv V^T`
  (K 512, 32 k-steps) into one of two 32-column TMEM stages; `O_t += P V_t` for the two 256-dim
  halves (M 64 x N 256 x K 64, 4 k-steps each) into 2 x 128 TMEM columns (320 of 512 used). The B
  operand of P.V is the **same** gathered latent tile re-viewed MN-major (dims contiguous per
  key row; identical bytes and SW128 swizzle, one N-tile per 16384 elements; the per-tile view
  is re-staged with the latent stage stride of 32768 elements because the DSL layout builder
  stacks a half-stage view at 16384).
- **Accumulator layout.** `.ws` M = 64 uses the "2x2" datapath layout: a 64 x N fp32 tile is
  128 lanes x N/2 columns, row r in lane r (first N/2 columns) and in lane r + 64 (second N/2).
  This is byte-identical to the per-CTA layout of the 2-CTA kernel's M = 128 fragments
  (`((64,(N/2,2)),1,1):((65536,(1,4194304)),0,0)`), so the softmax / correction / epilogue code
  of the 2-CTA kernel carries over: softmax thread t and t + 64 own row t % 64 with 32 keys
  each (`threads_per_row = 2` replaces every `cta_group_size` in the stats indexing), the
  correction rescales 32 columns at a time, the epilogue stores thread t's 128 contiguous dims.
- **Gather** (`CpasyncGatherKVManagerH64`): 128 threads (warps 12-15) issue, per 64-key stage,
  4 rows x (8 latent + 1 rope) 16-B `cp.async.cg` copies each (36 per thread), whole rows from
  the `[S,512]` latent and `[S,64]` rope tables, into a 2-stage ring (72 KiB per stage: sV
  64 KiB + sK 8 KiB). The indices of block n-2 are loaded into a second register set while
  block n is in flight (distance-2 prefetch, the E1 finding), for both the interleaved
  ownership (row `16 m + t // 8` in lane `m` of each 8-thread group; lanes 4-7 duplicate the
  reads) and the natural ownership (row t % 64, for the bitmask). Rows with index -1 or past
  the causal limit are zero-filled (predicated cp.async) and cleared in the 2-word validity
  bitmask (warps 0-1). The 128 gather threads arrive on the KV full barrier directly
  (`cp.async.mbarrier.arrive.noinc` on a `PipelineAsyncUmma` with a 128-thread producer group);
  the 2-CTA kernel's relay warp is idle here (warp 11). A stage lands in **four parts** (the
  rope row plus latent column blocks 0-1, then blocks 2-3, 4-5, 6-7 of every row), each part
  followed by its own `cp.async.mbarrier.arrive.noinc` on a raw 128-count mbarrier (the last
  part uses the pipeline's full barrier), so the MMA warp starts the S GEMM on part 0 and
  streams the remaining k-blocks behind the fill (`gemm_ws_ptx_partial(k_range=...)`,
  `load_X(col_blocks=...)`).
- **MMA order (block pairs).** Blocks are consumed in pairs (a, b): `S(a)` into TMEM stage 0
  and `S(b)` into stage 1 (each 4 + 32 k-steps, issued part by part as the stage lands), then
  wait P(a); `O += P(a) V(a)` (8 instructions); release a's KV stage and the P buffer
  (tcgen05.commit); wait P(b); `O += P(b) V(b)`; release b's stage and P. The softmax of block
  a runs under `S(b)`, the softmax of b partly under `PV(a)`, and the refill of a's stage has
  `PV(b)` plus the next pair's `S(a')` to land. The two earlier orders both left the in-order
  MMA warp idle: "S(n-1) before PV(n)" holds both ring stages until PV(n) and exposes the
  ~1,900-cycle stage refill on every block; "PV(n) then S(n-1)" (one S stage) exposes the whole
  softmax step (~1,000-1,400 cycles) between S(n-1) and P(n-1). P is double-buffered per KV
  stage, so softmax(b) stores P(b) while PV(a) still reads P(a) (with one buffer the MMA warp
  idled ~1,400 cycles per pair between PV(a) and PV(b)); the acquire of a block's buffer is at
  the top of its softmax step, where it is free (released by the PV two blocks earlier), and
  ptxas schedules the four P stores into the exp2 sequence. The block parity that selects the
  buffer is a compile-time constant in the MMA loop (pairs) and in the softmax step.
- **Epilogue.** O is stored thread-wise with 16-B stores (`use_tma_O = False`), as the o_lo
  residual already is; no TMA staging tile aliases the latent ring and no `sO_empty` hand-off
  with the gather is needed. The drain of the O accumulator (8 chunks of 32 TMEM columns, 64
  `st.global.v4` per thread for O and o_lo) costs ~15k cycles per tile, of which ~13k are the
  stores themselves (skipping them leaves ~2k of t2r + arithmetic; skipping only the o_lo stores
  halves it); a lane-transposed variant that writes 64 contiguous bytes per row (8 rows per warp
  store instead of 32 scattered pieces) was measured slower (the shuffle/select work exceeded
  any store gain), so the store count, not the line pattern, is the cost. The next tile's first
  `PV` waits for the O TMEM release, but its Q load, `S(0)`, `S(1)` and the first softmax run
  under the drain, so ~3-4k cycles per tile (3-4 %) are exposed.
- **Shared memory** (232,448 B = the SM100 cap, 0 B spare): mbarriers + stats 3,072 (292 B of
  barriers / part barriers / CLC, `sRowMax` 64 x 4 fp32 for the double-buffered row-max
  exchange, `sRowSum` and `sScale` 64 x 2 fp32, bitmask 16 B, then the 1024-B alignment of the
  first tile); sQv 65,536; sQ 8,192; sK 2 x 8,192; sV 2 x 65,536; sP 8,192 (P buffer of the
  even blocks). The P buffer of the odd blocks is the K-rope tile of KV stage 1 (the same
  64 x 64 bf16 K-major SW128 tile), which is dead from S(b)'s commit, its last read, until the
  stage's refill, which starts only after PV(b) released the stage; the gather rewrites every
  rope row (zero-fill included) so no stale P survives. The rope tile of stage 0 is *not* used
  for P(a): it sits exactly 16 KiB below the latent tile PV(a) reads and that address relation
  measurably slows the tensor pipe's operand fetch (+300 cycles per PV; the dedicated buffer at
  its original offset does not). The shared-KV specialization (no sQ/sK, two dedicated P
  buffers) is 216,064 B. Any further buffer (e.g. a TMA-staged O tile) must alias an existing
  one. TMEM 320 of 512 columns (S 2 x 32, O 2 x 128). Registers as the 2-CTA
  kernel (load/MMA 112, softmax 192, epilogue 128, gather 80, CLC/idle 48), honoured because the
  launch passes `min_blocks_per_mp=1`.

### Which forwards it serves

The kernel stores neither `p` nor `row_max`: the load-P backward needs `row_max` per 128-key
group, which a 64-key-block running max cannot provide. `interface.py` therefore dispatches to
it only when no P is to be saved, i.e. `gather_bwd_recompute_p=True` training forwards and
inference (`requires_grad=False`) forwards with exactly 64 Q heads; the load-P training
forward at 64 heads keeps the padded 2-CTA kernel. Fewer than 64 heads keep the 2-CTA kernel
too (padding 1..63 heads to the 64-row tile is a follow-up). 128-head forwards are unchanged
(no source of the 2-CTA kernel is touched; `out`/`dq`/`dqv` are bitwise identical to the base
commit in the A/B of `agent_space/dsa-64h-design/runs/F1_fwd_h64/ab128/`).

### Test contract

`test_flash_attn_mla_sparse_bwd_recompute_p[64]` used to assert that the recompute-P forward is
bitwise the load-P forward (true at 128 heads: same kernel). At 64 heads the two run different
kernels (`.ws` M = 64 vs `cta_group::2` M = 128, one K = 576 accumulation chain vs two K = 256
halves), so the test certifies instead that the recompute-P forward agrees with the load-P
forward and with the inference forward up to bf16 output rounding (out rel-L2 < 5e-3, lse
max-abs < 1e-4; measured ~1.1e-3 and ~1e-6). `test_flash_attn_mla_absorbed` runs its 64-head
sparse cases on the recompute-P path so the kernel is exercised against `attention_ref` with
the standard tolerances plus the 10x determinism check; `precise_dpsum[64, recompute_p=True]`
and `preprocess_tile_tail[recompute_p=True]` cover the o_lo residual, varlen, odd sequence
lengths and top-k 128 (2 blocks). Bit-exact gather validation (sentinels, causal limit,
zero-fill on dirty stages): `agent_space/dsa-64h-design/runs/F1_fwd_h64/gather_validate.py`.

### Numerics (GB200, random bf16 inputs, fp64 reference; `runs/F1_fwd_h64/check_fwd_h64.json`)

| case | out rel-L2: H64 recompute / 2-CTA load-P / H64 inference / bf16 floor | lse max-abs vs fp64 | H64 vs load-P out rel-L2 |
|---|---|---|---|
| 512 tok, 256 keys, non-causal, 10% sentinels | 0.207% / 0.210% / 0.221% / 0.166% | 1.9e-6 | 0.131% |
| 512 tok, causal, 3 fully masked rows | 0.191% / 0.194% / 0.213% / 0.163% | 2.7e-6 | 0.108% |
| shared KV (no rope), causal | 0.197% / 0.202% / 0.213% / 0.166% | 2.4e-6 | 0.126% |
| varlen 2 docs, causal | 0.181% / 0.186% / 0.192% / 0.154% | 2.7e-6 | 0.103% |
| 1024 tok, 2048 keys, top-k 1024, causal | 0.204% / 0.205% / 0.219% / 0.166% | 1.8e-6 | 0.109% |

Deterministic (bitwise across launches), no NaN; fully masked rows give out = 0, lse = -inf.

### Cost

GLM-5.2 shape (64 heads, top-k 2048, bf16, causal, one document, varlen API), GB200, medians of 20
steps, same GPU back to back (`agent_space/dsa-64h-design/benchmark.csv`):

| T = S | forward (training, with o_lo) | forward (inference) | train step | peak / saved GiB |
|---|---|---|---|---|
| 16384 | 5.07 ms (2-CTA padded kernel 7.40) | 4.47 (6.93) | 33.15 (35.5) | 4.17 / 2.00 (same) |
| 65536 | 21.47 ms (28.5) | 18.83 (28.9) | 140.4 (149.3) | 13.69 / 8.02 (same) |

(The first version of this kernel, one S stage and "PV(n) then S(n-1)" per block, measured
6.18 / 24.27 ms training forward; the block-pair order with two S stages brought it to
5.32 / 22.32, the parted stage fill to 5.25 / 21.85, and the double-buffered P to the numbers
above: -3.5 % / -2.1 % training forward, +2 % / +1 % inference forward against the single-buffer
kernel measured back to back on the same GPU.)

Per block pair (CTA 0 of a T = S = 8192 run, `%clock64` stamps under a probe-only constructor
argument, instrumented build; `agent_space/dsa-64h-design/runs/F1_fwd_h64/clock_probe_t8k_v13c.json`,
`..._inf_v16.json`, `pair_analysis.py`): the pair period is the **refill chain of stage a**:
`S(a)` issue 1,160 + wait for part 0 of stage b ~320 + `S(b)` issue ~1,300 + wait P(a) ~210 +
`PV(a)` ~750 + refill of stage a ~1,770 (the gather sees the release after ~190, needs ~800-1,000
cycles to issue the 36 `cp.async` per thread and part 0 lands ~450-760 later) = ~5,500 cycles
per pair, 2,750 per block; the softmax(b) -> PV(b) chain ends earlier and the MMA waits the
last ~130-240 cycles for stage a. The tensor pipe is busy ~3,600 of those cycles (S 2 x
1,160-1,300 at the 32-cycle `.ws` M=64 N=64 K=16 rate, PV 2 x 640-1,050 for 8 x M=64 N=256).
Shared memory is the contended resource: the `.ws` SS k-step reads 4 KiB of operands per 32
cycles, and while a 72 KiB refill lands the tensor pipe's operand fetch slows (PV(b) 450 ->
1,050 cycles when it overlaps the refill of stage a, which the double-buffered P makes it do);
~612 KiB cross shared memory per pair (S 2 x 144, PV 2 x 80, fill 144, P 16 KiB) at a
practical ~110-150 B/clk. The softmax step is ~1,300 cycles per block (t2r ~160, bitmask 140,
2-thread max exchange ~300, scale + 32 exp2 + bf16 pack ~500-650, P store + fence ~80, commit
~40); it is not on the period today. Its exp2 phase is a ptxas schedule artefact, not a MUFU
limit: `mufu_bench.py` measures 32 independent `ex2.approx` at 9.3 cycles per warp instruction
(298 per step) but the kernel's `fma -> ex2, ex2 -> cvt.bf16x2` sequence at 652 (= the kernel),
because ptxas places every bf16 pack right behind its MUFU pair and recycles one temporary
register pair; scalar fmas cut it to ~500, emulating half of the exp2 on the FMA pipe to ~575.
Per tile the epilogue adds ~15k cycles (store-bound, see above), of which ~3-4k are exposed,
plus ~3k of tile start. ncu of this kernel at T = S = 16384 (`profile/F1v17_fwd_h64_T16k/`;
the parted-fill version without the P double buffer: `profile/F1v10_fwd_h64_T16k/`): 4.85 ms
(5.08), tensor pipe active 40 % (38 %), issue slots 23 %, 0.27 eligible warps per scheduler
cycle, shared-memory data pipe 66 % of peak, the gather's `cp.async` issue `lg_throttle`d
(0.43 per issue), L2 hit 89 %, gather 1152 B per (token, key), 23.07 M tensor instructions
(44 per block, the minimum). With two latent stages nothing can hide the refill of the stage
that PV(a) just released; reaching the 17 ms target at 64k needs a third latent stage, i.e. Qv
as a TMEM A operand (the `.ws` TS form: 2 KiB per k-step, 28 instead of 32 cycles) with the
freed 64 KiB of smem as that stage — the WP-F3 design — where the ~1,300-cycle softmax step
becomes co-critical and the exp2 schedule above is the first thing to fix.

Releasing a stage in two halves does **not** help within two stages (measured, rejected): a
variant that handed the dims 0-255 half of a stage back to the gather right after PV0 (a
per-stage mbarrier arrived on by a `tcgen05.commit`; rope rows and dims 256-511 after PV1; the
S GEMM consuming the latent parts in landing order and accumulating the rope GEMM when the
second half lands) shortened CTA 0's instrumented pair period 5,360 -> 5,168 cycles but made
the whole kernel slower (uninstrumented 8k kernel 2.45 -> 2.58 ms; training forward +4.5 % at
16k, +0.7 % at 64k; inference +5 % / +1 %), with identical numerics: on all 152 SMs the earlier
refill lands under PV1's operand fetch and the shared-memory contention costs more than the
~340 cycles per pair the earlier start saves. The refill has to leave the critical path (third
stage), not start earlier under the GEMMs. Evidence:
`agent_space/dsa-64h-design/runs/F1_fwd_h64/ledger.md` L20, `patches/commits/stage5/`.

## C6-dq: 1-CTA dQ/dQv kernel with a whole-row gather (`dQdQvGemmKernelH64`)

Why the sparse-MLA backward's dQ/dQv gather GEMM runs a dedicated kernel at exactly 64 Q heads
per KV head, what it does differently from the padded 2-CTA kernel, what it costs, how it is
tested. Code: `flash_bwd_mla_dq_dqv_sm100_h64.py` (`dQdQvGemmKernelH64`), `topk_gather_kv.py`
(`CpasyncGatherKVManagerH64`, shared with the F1 forward), `interface.py`
(`_compile_sparse_mla_dq_dqv` picks the class when `nheads == 64`; the compile key already
contains the head count, so the binaries are distinct). `dQdQvGemmKernel` (128 rows, cluster
(1,2)) is unchanged and still serves every other head count.

### Problem

`dQdQvGemmKernel` computes, per token, `dQv = dS @ V[idx]` and `dQ = dS @ K_rope[idx]` with a
fixed 128-row M tile (`sparse_mla_qhead_tile` with `min_tile=128`): at 64 heads half of every
MMA row is padding, and the cluster of two CTAs splits the 512 latent dims, so each CTA
gathers its 256-dim half of every key in 512-B pieces (2 stages of 128 keys) and CTA 0 also
gathers the 128-B rope rows; dS is TMA-multicast to both CTAs. The fill-ceiling study found
that producer shape (128-key stages of 256/512-B pieces, indices loaded just before use) is
what limits the kernel: 20.6 ms per 64k backward at 46 % tensor-pipe busy, 1168 gather bytes
per (token, key) plus 384 B of dS per pair.

### Change (64 heads only)

- **Tiles.** One token per CTA, cluster (1,1), CLC persistent scheduling. M = 64 = the head
  count (no padding); K tile 64 keys (32 per token for top-k 2048).
- **GEMMs.** Plain `tcgen05.mma.cta_group::1` M=64 bf16 -> fp32 (`cute.gemm`): `dQv` as two
  N=256 tiles (dims 0-255 and 256-511) and `dQ` as one N=64 tile, A = the dS tile (64 heads x
  64 keys, K-major SW128, TMA-loaded per k-tile, 4 stages), B = the gathered stage re-viewed
  MN-major (dims contiguous per key row). The second dQv N tile accumulates in the idle lane
  half of the M=64 layout (TMEM address + 16 lanes), the way the DSL itself interleaves N tiles
  at M=64, so TMEM is dQ 64 + dQv 256 = 320 of 512 columns (single accumulator stage: two
  would need 640). The plain M=64 instruction runs at 50 % of peak (128 cycles for N=256,
  32 for N=64): 288 cycles per k-block, 36.9k per token — the MMA floor is 8.2 ms per 64k
  backward, so this kernel is not MMA-bound only as long as the gather is slower.
- **Gather.** `CpasyncGatherKVManagerH64` (the F1 forward's producer): each 64-key stage is
  the whole 1152-B row of every key (latent 1024 B + rope 128 B) fetched once per CTA by 128
  threads as 16-B `cp.async.cg` copies (36 per thread per stage), the indices of block n+2
  loaded into a second register set while block n is in flight, completion signalled by
  `cp.async.mbarrier.arrive.noinc` on the KV pipeline's full barrier. Rows with index -1 or
  >= seqlen_k are zero-filled by the predicated copy every stage (verified with non-zero dS
  on those slots: their contribution is exactly zero, so there is no stale-smem 0 x NaN
  hazard and no explicit zero-fill is needed; there is no causal limit here, out-of-limit
  slots carry dS = 0). The gather writes the stage through its K-major (keys x dims) view and
  the MMAs read it through the MN-major view of each 256-dim N tile (tile 1 at +16384
  elements, stages stepped by the whole 32768-element latent stage) — the same bytes and
  the same SW128 swizzle, as in the F1 forward.
- **Shared memory** 205,824 B (shared-KV specialization 181,248 B): dS 4 x 8,192; rope
  2 x 8,192; latent 2 x 65,536; epilogue tiles dQ 8,192 + dQv 2 x 8,192 (dedicated: the
  next token's gather never waits for the epilogue, unlike the padded kernel whose epilogue
  tiles alias the operand stages); barriers 1,024. Two whole-row stages: a third 72-KiB stage
  does not fit beside a 2-stage dS ring (3 x 73,728 + 2 x 8,192 = 237,568 > 232,448).
- **Epilogue.** TMEM -> registers (`tcgen05.ld.16x256b`, the M=64 layout's 16-lane groups)
  -> bf16 -> smem tile -> TMA store, one 64 x 64 subtile at a time through the 2-stage dQv
  ring (`PipelineTmaStore`) and the dQ tile; dq/dqv bf16 in `(token, head, dim)` as before.
- **Registers.** No per-role `setmaxnreg`: with `min_blocks_per_mp=1` every role runs at the
  168 registers ptxas assigns to 352 threads (the kernel needs 49, 0 B local). The padded
  kernel's budgets (KV 224 / epilogue 128 / others 112 over 11 warps) were never honoured
  (no launch attribute) and trap (`setmaxnreg.inc 128` below the launch count) or hang (the
  gather warps' `inc 224` waits for registers the incomplete warpgroup never releases) once
  they are; evidence `agent_space/dsa-64h-design/runs/C6_dq/ledger.md` L1-L2 and
  `reports/C9_minblocks_report.md`.

### Effect (GB200, 64 heads, W = 2048, bf16, causal, `gather_bwd_recompute_p=True`, `gather_bwd_token_chunk=4096`)

`time_bwd.py`, same GPU and session as the integration branch (C8 + C1 + F1), which uses the
padded kernel for dq/dqv:

| T = S | dq_dqv padded -> C6-dq | backward kernel sum | GLM-5.2 harness train step |
|---|---|---|---|
| 16k | 4.43 -> **2.72 ms** (-38.5 %) | 21.57 -> 19.87 ms | 26.72 -> **24.96 ms** |
| 64k | 20.62 -> **12.23 ms** (-40.7 %) | 98.40 -> 90.10 ms | 119.84 -> **112.52 ms** (main 148.9; FlashMLA + cuDNN 129.3) |

Peak memory and saved activations unchanged (13.69 / 8.02 GiB at 64k). ncu at 16k chunk 2
(`profile/C6dq_dqdqv_h64_T16k/`): 693 us per launch (padded kernel 1,113), tensor pipe 75 %
busy = exactly the plain-M=64 issue time, gather 1161 B per (token, key) at 48 B/clk/SM (the
2-stage producer's ceiling is ~54), dS 128 B per pair (was 384). The kernel is MMA / fill
co-bound: the `.ws` M=64 N=256 form (80 % rate) and a third latent stage are the next levers,
each worth ~10-15 %.

### Numerics

dq and dqv are **bitwise identical** to the padded kernel's on the same inputs (batched and
varlen): both accumulate the same 16-key k-blocks in the same order in fp32 and round once to
bf16. The token-chunk, retain-graph and sentinel tests therefore keep their bitwise dq/dqv
assertions unchanged, and the harness accuracy columns equal the integration branch's.

### Validation

Sparse-MLA subset (`mla_sparse or mla_sink or topk_order_invariance or precise_dpsum or
preprocess_tile_tail`) fake and GPU; `agent_space/dsa-64h-design/runs/C6_dq/check_dq_dqv.py`
(fp32 reference, sentinels + out-of-range slots, non-zero dS on invalid slots, shared-KV,
batched + varlen; bitwise compare against the padded kernel); 128-head outputs bitwise
unchanged (out / dq / dqv sha256 identical to the integration branch, dk / dv within the fp32
atomic band); `runs/C6_dq/gates.md`, `reports/C6dq_report.md`.
## F3: Q in TMEM, `.ws` TS dual GEMM, three latent stages (`FlashAttentionMLAForwardSm100H64`)

How the 64-head forward keeps the token's Q tile in tensor memory instead of shared memory,
what that buys (a third latent stage), how the S GEMM and the softmax change, what it costs,
and where the design's time goes.
Code: `flash_fwd_mla_sm100_h64.py` (same class and constructor as F1), `blackwell_helpers.py`
(`gemm_ws_ts_ptx_partial`, `utccp_128x256b_ptx`, `tcgen05_fence_after_thread_sync`),
`topk_gather_kv.py` (`CpasyncGatherKVManagerH64.load_X(identity_rows=True)`), `interface.py`
(the H64 dispatch additionally requires a caller-provided `out` to be 32-B aligned).

### Why

With Q in shared memory (F1: sQv 64 KiB + sQ 8 KiB) only two 72 KiB latent stages fit under the
232,448 B cap, and F1 measured that with two stages nothing hides the ~1,800-cycle refill of the
stage that PV(a) just released (`ledger.md` L17, L20 of WP-F1). The 72 KiB of Q are the only
buffer that can become a third stage.

### Design

- **Q staging through the ring.** Once per tile the gather warps write the token's Q tile as a
  pseudo-block of the KV ring: the 64 head rows of Qv (1024 B each) into a latent stage with
  the stage's own K-major SW128 layout, the 64 rows of Q_rope into the rope tile
  (`load_X(identity_rows=True)`: row r of the tile <- row r of the (64, dims) Q tile, no top-k
  index, no validity predicate, the same 36 `cp.async.cg` per thread as a KV block, the same part
  and full barriers so every barrier phase advances once per ring slot). No TMA warp: warp 8 is
  idle like warp 11.
- **Q -> TMEM (`tcgen05.cp`).** The MMA warp waits for the pseudo-block, then issues 16 UTCCP
  `128x256b` copies from the **(128 rows, 256) SW128 re-view** of the latent stage (rows 64-127 of
  the view = the second 64 x 64 column tile of each pair = the upper dim half of the same 64
  rows; k-block kb -> TMEM columns Qv + 8 kb) and 2 from the (128, 32) SW64 re-view of the rope
  tile, commits, releases the stage and the rope tile (the commit tracks the copies) and waits
  on a private mbarrier before the first S GEMM. TMEM image: lane l < 64 holds Q[l, 128 t + 0..63]
  at columns 32 t .. 32 t + 31, lane 64 + l holds Q[l, 128 t + 64..127]; rope: lane l dims 0-31,
  lane 64 + l dims 32-63 (bf16 pairs per 32-bit column; measured bit-exact,
  `agent_space/dsa-64h-design/dsl-probes/probe_ws_ts_qtmem.py`).
- **Rope tiles are SW64.** A 64 x 64 bf16 K-major SW128 tile (F1) has no (128 rows, 32) re-view;
  the rope tile is now two 64 x 32 SW64 column tiles (`make_smem_layout_b(tiled_mma(64, 64),
  (64, 64, 32), bf16, 2)`: the "2 stages" are the two dim halves), whose (128, 32) SW64 view
  pairs view rows 64-127 with dims 32-63. The gather writes it through the same composed
  (row, dim) view as before (the swizzle is the tensor's). There is **one** rope tile: a block's
  rope rows are written after the rope GEMM of the previous block released the tile
  (`pipeline_K`, released by a `tcgen05.commit` after the rope GEMM), so the gather issues the
  latent parts first and the rope rows last, and the rope GEMM is the last instruction of S(n).
- **S = Q K^T is the `.ws` TS "dual GEMM"** (`gemm_ws_ts_ptx_partial`): M = 64 heads, N = 128,
  K = 16 per instruction, A from TMEM, B = the (128, 256) view of the latent stage (16 k-steps,
  issued part by part as the stage lands) then the (128, 32) view of the rope tile (2 k-steps).
  For `.ws` M = 64 the instruction reads A row r of the first N/2 accumulator columns from TMEM
  lane r and of the second N/2 from lane 64 + r; with the two dim halves of Q in the two lane
  halves and the B view pairing view rows 64-127 with the upper dim half of the same keys, one
  instruction computes two independent 64 x 64 x 16 products: lane half 0 of the 64-column S
  accumulator holds the partial sum over the lower dim halves, lane half 1 over the upper
  halves (measured: each half matches its half-dimension fp64 reference to 2.6e-7, the sum to
  2.6e-7). The idesc is the SS one (0x4200490); the TS form is the instruction text
  `[tmem_acc], [tmem_a + 8 kb], desc_b, idesc, p, 0`. The TS k-step costs 44 cycles at N = 128
  (vs 32 for F1's SS N = 64 k-step over half the keys: the same 1,408 tensor cycles per block for
  S plus the rope), and reads 4 KiB of B per k-step instead of 4 KiB of A + B.
- **Lane-half sum in the softmax warps.** Thread t (lane t) loads its lane half's partials of
  keys 0-31 and 32-63 (two `tcgen05.ld 32x32b.x32`), hands the half it does not own to its
  partner t ^ 64 through a 16 KiB exchange buffer (thread-contiguous per value) and adds the
  partner's half after a pairwise `bar.sync id, 64` of warps w and w ^ 2; thread t then owns
  keys 0-31 of row t and thread t + 64 keys 32-63, exactly F1's ownership, so the bitmask,
  running max (whose 2-thread exchange also uses the pair barrier), exp2, P store and stats
  code carry over. The sum is one fp32 add per key (the two halves are exact fp32 partial sums).
- **Pipelines.** `pipeline_KV` 3 stages (the full barrier now also covers the rope rows; the
  latent parts have their own barriers), `pipeline_K` (rope tile, 1 stage, empty side only),
  `pipeline_S` **1 stage**, `pipeline_P` **1 buffer**, O0/O1, sm_stats, bitmask as F1. MMA order per
  tile: Q -> TMEM; S(0); S(1); PV(0); for n = 2 .. N-1: S(n), PV(n-1); PV(N-1). S(n+1) only needs
  the softmax's t2r of S(n) (the S stage is released right after the loads), so the softmax step
  of block n runs under S(n+1) and PV(n), and a released stage has two block periods to refill.
- **TMEM** (every region 32-column aligned): O0 0-127, O1 128-255, S 256-319 (64 x 128 fp32 dual
  GEMM = 64 columns), Qv 320-447, Q_rope 448-463: 464 of 512. A second S stage (64 more columns)
  does not fit, which forces the single-S order above.
- **Shared memory** = 232,448 B, the cap: barriers + stats 3,072 (incl. `clc_response` at a 16-B
  alignment: the scheduler reads it with one 16-B load), sV 3 x 65,536, sK 8,192 (SW64 rope
  tile), sP 8,192, sX 16,384 (exchange). Shared-KV specialization 224,256 B (no rope tile, TMEM 448).
- **Epilogue.** As F1 (O and o_lo streamed 32 TMEM columns at a time, thread-wise stores) but
  with 256-bit `st.global.v8` stores (one request per 32-B sector), which shortens the per-tile
  drain from ~14.7k to ~10.7k cycles. The 32-B alignment promise on the O / o_lo pointers and
  strides is guaranteed for interface-allocated outputs (torch allocations, 1024-B rows); a
  caller-provided `out` that is not 32-B aligned keeps the padded 2-CTA kernel.

### Test contract

Unchanged from F1 (recompute-P forward vs load-P forward up to bf16 rounding at 64 heads, bitwise
at 128), extended to `test_flash_attn_mla_sparse_bwd_learnable_sink[64-*]`: that test was widened
to 64 heads while both forward paths still ran the padded kernel, so its bitwise comparison of the
load-P and recompute-P modes could not hold once the 64-head kernel existed (it failed on the
integration base before F3). Cross-kernel modes now agree up to bf16 rounding (out rel-L2 < 5e-3,
lse max-abs < 1e-4, dsink rel-L2 < 1e-2; measured ~1e-3 / ~2e-6 / ~3e-3), same-kernel modes stay bitwise.

### Cost (GB200, GLM-5.2 shape, GPU 1, `agent_space/dsa-64h-design/benchmark.csv`)

| T = S | forward (training, with o_lo) | forward (inference) | train step (with C8 + C1) | peak / saved GiB |
|---|---|---|---|---|
| 16384 | 4.77 ms (F1 5.05-5.12, padded 2-CTA kernel 7.40) | 4.12 (F1 4.3-4.5) | 25.93 | 4.17 / 2.00 (same) |
| 65536 | 20.42 ms (F1 21.35-21.47, padded 28.5) | 17.50 (F1 18.7) | 118.40 (bwd 99.68) | 13.69 / 8.02 (same) |

Same-GPU ABAB against the F1 tip (four legs, back to back): 64k training forward 20.35 / 20.33 vs
21.37 / 21.53 ms (-5.2 %), inference 17.52 vs 18.65 (-6.1 %); 16k 4.70 / 4.94 vs 4.96 / 5.10.

Numerics (`runs/F3_fwd_qtmem/check_fwd_h64_v3.json`): out rel-L2 vs fp64 0.181-0.207 % on the five
`check_fwd_h64` cases (F1 0.181-0.207 %, bf16 floor 0.154-0.166 %), lse max-abs 1.1-1.4e-6,
deterministic, no NaN, fully masked rows -> 0 / -inf, batched, varlen, shared-KV and top-k 1024
all as F1. Harness accuracy vs fp64 identical to F1 to the printed digit on all 11 cases (out
0.1896 % iid, 0.1660 % peaked; dq_latent 0.2259 %; lse max-abs 2.3e-6 .. 1.3e-5). Sparse-MLA test
subset 182 passed (fake + GPU), absorbed suite 1040 passed on the GPU; `out`/`dq`/`dqv` of the 128-head
recompute path and of the 64-head load-P path bitwise equal to the base commit.

ncu at T = S = 16384 (`agent_space/dsa-64h-design/profile/F3_fwd_h64_T16k/`, vs the F1 v17 run): 4.08 ms
(4.85), tensor pipe active 48.7 % (40.1 %) with 26 tensor instructions per 64-key block (18 TS +
8 SS; 44 at half the N in F1) plus 18 `tcgen05.cp` per tile, issue active 33.5 % (23 %), `mio_throttle`
0.85 stalls per issue (0.05) on the exchange stores / loads, the gather's `cp.async` issue and the
P store: the shared-memory instruction queue is saturated. Local loads 2.0 M (238 M in F1).

### Where the time goes (in-kernel `%clock64`, CTA 0, T = S = 8192, `runs/F3_fwd_qtmem/clock_probe_t8k_v2b.json`, `decode_f3.py`)

Block period ~2,400 cycles = the softmax step (S seen -> P committed ~2,250: t2r 108, exchange
stores 271, pair barrier 21, exchange loads + adds 313, mask + row-max exchange 350, stats 45,
exp2 296, P acquire 140, P store + fence 361, commit 66), while the MMA chain alone would allow
~1,350 (S issue 809 incl. part waits, PV ~700). The barriers are cheap and the exchange is
conflict-free; every **shared-memory** phase of the step is 3-10x its instruction count. That is
the shared-memory port: per 64-key block it carries the fill (72 KiB), the S B-operand reads
(72 KiB: the whole stage once through the (128, 256) view), the PV reads (V 64 KiB + P 16 KiB),
the P store (8 KiB) and the exchange (16 KiB written + 16 KiB read) = **264 KiB**, a floor of
2,060 cycles at 128 B/clk and ~2,400 at a realistic 85 %; the measured period sits on it. F1
carried 306 KiB per block (Qv read from shared memory on every S k-step, 64 KiB, and no
exchange) at 2,750 cycles per block. The remaining per-tile cost is the O drain (~10.7k cycles,
~7k exposed: the next tile's S(0), S(1) and first softmax overlap it, its PV(0) waits for the
O release), ~8 % at 64k.

Consequences for the design family "gather once, two MMAs over the same stage, lane-half
exchange": the port floor is ~2,100-2,400 cycles per 64-key block, i.e. 15-17 ms at 64k (the
public head-64 sparse-attention forward that uses the same recipe, with a TMA-store epilogue
and no o_lo, measures 15.1 ms on the same shape). Below that needs less shared-memory traffic
per key, not more overlap: candidates are
P as a TMEM A operand of a TS PV (saves the P store and reads, adds a bf16 exchange: net -8 KiB
per block) and a TMA-store epilogue staged through the exchange buffer (drain 10.7k -> ~5k).
