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
