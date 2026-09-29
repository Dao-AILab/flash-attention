# Sparse (top-k) MLA forward on the 1CTA kernel

Status (2026-09-28), opt-in via `FLASH_ATTENTION_MLA_1CTA=1`:
- **Inference forward:** MQA with up to 64 Q heads.
- **Training forward:** with the recompute-P backward (`gather_bwd_recompute_p=True`) at
  1..64 heads (padded to the 64-row tile in both passes; see "Training below 64 heads").
  The kernel produces what that backward consumes: exact-running-max LSE
  (`rescale_threshold=0`) and the O rounding residual `o_lo`, but no P / row_max.
- **Two mainloops.** 16-bit inputs with at most 64 heads run the **64-key-block (kb64)
  mainloop** (see the kb64 sections below):
  - sparse: any head count up to 64;
  - dense: 64 heads, or fewer heads on decode (`seqlen_q = 1`). It agrees with the 2CTA kernel to bf16 rounding, not
  bitwise. Everything else runs the 128-key mainloop (dense prefill with fewer than 64
  heads, more than 64 heads, fp8). On the 128-key mainloop `out`/`lse`/`o_lo` are bitwise
  identical to the 2CTA kernel's, and so are the gradients: dQ and dQv bitwise, dK and dV
  within the sparse backward's own atomic run-to-run non-determinism, which 2CTA shows too.
- **CLC.** The 1CTA sparse route always runs the persistent CLC scheduler, as the 2CTA MLA
  kernel does. Dense kb64 uses it on prefill (`seqlen_q` hint > 1); the dense 128-key
  mainloop follows `FA_CLC`. These are kernel-class policies (`use_clc`).
- **Knobs.** `FLASH_ATTENTION_MLA_1CTA=1` (the opt-in route) is the only MLA env var. The
  ablation knobs used to measure this work were removed (see "Interface cleanup").
- **Fallback:** every other sparse case (more than 64 heads, or training with load-P)
  falls back to the 2CTA kernel under the flag.
- **No split-KV** for sparse MLA on either kernel.

Plan and review log: `agent_space/SPARSE_MLA_1CTA_PORT_PLAN.md`.

## What it does

- **Tiles.** One tile is one query token. Its Q heads are padded **in-kernel** to the 64-row
  tile: Q/Qv/O use TMA on `pack_gqa.qheads_first_tma_view`, which zero-fills padded rows on
  load and drops them on store. The 2CTA kernel pads every token to 128 rows, so with <= 64
  heads its second CTA only computes zero rows.
- **Gather.** The cp.async warp group (warps 12-15) gathers K and both V dv-splits per
  n_block with `CpasyncGatherKVManager` (`cta_group_size=1`, no relay warp). V is gathered
  once, since P @ V reads it through the `sVt` view; 2CTA gathers V twice (K-major and
  transposed). Invalid slots are predicated off, and a predicated-off cp.async is emitted as
  `cp.async.cg ... 16, src_size` with `src_size = pred ? 16 : 0`, so it zero-fills (no
  stale SMEM, no `0 * NaN`).
- **Validity bitmask.** Built per n_block by the same warps, `0 <= idx < seqlen_k_limit`,
  with `seqlen_k_limit = q + 1 + s_k - s_q` under causal (bottom-right). The softmax warps
  apply it on every block. Each softmax thread owns one Layout E datapath half (64 columns
  = 2 bitmask words) and takes the bit from the logical column coordinate.
- **Varlen Q.** `cu_seqlens_q` uses the packed (flat over tokens) scheduler as in 2CTA, but
  with **uniform** batch-local indexing in all five warp roles (one `_tile_coords` helper).
  TMA O stays on, since a one-token tile cannot straddle sequences. `seqused_q` uses the
  varlen scheduler.
- **Other paths.** fp8 (with descales) and the learnable sink compose unchanged. Padded
  heads get no sink and are guarded in the per-row LSE / O stores.
- **Routing.** More than 64 heads fall back to 2CTA. A second 64-row tile per token would
  re-gather the same indices.

Correctness: the 128-key mainloop is bitwise identical to the 2CTA kernel on every
benchmarked shape and on the test matrix (`test_flash_attn_mla_1cta_sparse_*`), as well as
matching the reference. The kb64 mainloop meets the contract in "Test contract" below.

## 64-key-block mainloop (kb64): up to 64 heads, 16-bit

`flash_fwd_mla_1cta_kb64_sm100.FlashAttentionMLAForward1CtaKb64Sm100` subclasses the 1CTA
kernel. It ports PR 2914's 64-head forward mainloop into it. The 128-key mainloop kept one
key block's V in both V stages (its two dv halves), so it serialized
S -> softmax -> P -> PV -> V release -> refill -> next S. That was its binding constraint,
which the Phase B items (bitmask wait, pair barriers, index prefetch, register budgets)
could not move (`agent_space/PR2914_COMPARISON_AND_PORT_PLAN.md`).

**Design**

- **64-key blocks.** A latent stage is one block's 64 x 512 rows (64 KB, K-major SW128).
  There are three stages, plus one 64 x 64 rope tile. Each stage lands in 4 column-block
  parts, each with its own cp.async mbarrier, so S streams behind the fill. The gather is
  `CpasyncGatherKVManagerH64`: whole rows, indices loaded two blocks ahead, 2 bitmask words
  per block.
- **Q in TMEM.** The gather warps stage the token's Q tile through the KV ring once per
  tile, with identity rows (so no padded heads are possible). The MMA warp copies it with
  `tcgen05.cp.128x256b` (`utccp_128x256b_ptx`). Qv takes 128 "dual packed" columns: lanes
  0-63 hold the lower dim half of each 128-dim chunk, lanes 64-127 the upper half. Q_rope
  takes 16 columns.
- **S = Q K^T.** A `.ws` TS dual GEMM, M64 N128 (`gemm_ws_ts_ptx_partial`). The two lane
  halves of the accumulator hold the two dim halves' partial sums. The softmax warps add
  them through a 16 KB exchange buffer, with 64-thread pair barriers.
- **O += P V.** SS, M64 N256 x 2 N-tiles, over the MN-major re-view of the stage.
- **Issue order.** The MMA warp issues S(n) before PV(n-1). There is one S stage (TMEM is
  O 256 + S 64 + Qv 128 + Qr 16 = 464 of 512 columns).
- **Epilogue.** O and o_lo stream from TMEM 32 columns at a time, with 256-bit
  `st.global.v8`, and each O split is released as soon as it drains. The 256-bit stores
  need 32-B aligned rows; the interface checks this and falls back to 128-bit stores.
- **SMEM.** 3 x 64 KB + 8 KB rope + 8 KB P + 16 KB exchange + header = 232,448 B, which is
  exactly the cap.
- **Registers.** `min_blocks_per_mp=1`, so the budgets are honoured: softmax 192,
  epilogue 128, warp group 2 (load/MMA/CLC/idle) 112, gather 80, uniform per warp group.
  No spills.
- **ptxas.** It runs at the default level (`FlashAttentionMLAForward1CtaKb64Sm100.ptxas_options = ""`).
  `-O2` is 7-8% slower here.

**Test contract** (PR 2914's; `tests/cute/test_flash_attn.py::_assert_mla_fwd_close`)

- vs the 2CTA kernel: out rel-L2 < 5e-3 (measured ~2e-3, i.e. two independent bf16
  roundings), the same -inf LSE pattern, finite LSE max-abs < 1e-4 (measured ~1e-6).
- Training (`test_flash_attn_mla_1cta_sparse_train_recompute_p`):
  - out + o_lo rel-L2 < 2e-3 (measured 7.6e-4);
  - dq / dqv / dk / dv rel-L2 < 5e-3 (measured <= 1.6e-3);
  - dsink < 1e-2 (measured 3.3e-3);
  - |o_lo| <= half an ulp of out.
- Bitwise run to run (`test_flash_attn_mla_1cta_sparse_kb64`, which also covers the
  128-bit-store fallback).
- The 128-key mainloop keeps the bitwise asserts.

**Results** (GB300, 64 heads, topk 2048, causal)

The table is `agent_space/kernel_time_fwd.py` for no-grad and train, plus the train step
from `agent_space/bench_mla64_compare.py` (recompute-P).

| T | kernel | fwd no-grad (ms) | fwd train (ms) | train step (ms) |
|---|---|---|---|---|
| 16k | 128-key mainloop, no CLC, -O2 (before) | 4.63 | 5.21 | 24.96 (after Phase A) |
| 16k | 128-key mainloop + CLC | 4.18 | 4.73 | 24.26 |
| 16k | PR 2914 H64 forward (own checkout) | 3.62 | 4.01 | 24.31 |
| 16k | **kb64 + CLC** | **3.56** | **3.84** | **23.38** |
| 4k | kb64 + CLC | 0.93 | 0.99 | 5.69 |

ncu at 16k training (`agent_space/ncu_b7/`, summary: `agent_space/ncu_summary.py`):

| | kb64 | PR H64 | 128-key + CLC |
|---|---|---|---|
| duration | 3.84 ms | 3.94 ms | 4.74 ms |
| SM throughput | 70.7% | 68.9% | 57.3% |
| memory throughput | 54.9% | 54.3% | 32.9% |
| L2 hit rate | 93.1% | 93.1% | 88.4% |
| issue slots busy | 33.7% | 33.5% | 17.2% |
| local-memory spill requests (instrumented, see below) | 0 | 4.3 M | 65.9 M |
| top stall (per issue) | long scoreboard 5.8 | long scoreboard 5.9 | long scoreboard 18.0 |

kb64 and the PR's kernel have the same profile; kb64 is ahead by its spill-free register
allocation.

The 128-key row overstates its spilling. ncu counts spill requests in its SASS-instrumented
pass, which slows the kernel, so every mbarrier retry-loop iteration is counted many times.
Joining ncu's per-instruction counts with a lineinfo disassembly
(`agent_space/spill_dyn.py`, `agent_space/spill128/`) attributes the local traffic as follows:

- **About 97% of the executed local loads are one reload** (`LDL [R1+0xa4]`): the SMEM
  storage base address, stored once at kernel entry. It is reloaded inside each mbarrier
  wait's retry loop, in softmax, the CLC consumer, MMA and the gather warps. It runs only
  while the warp is already blocked.
- **The real spills are in the epilogue warps' final O store in training** (lines
  3524/3559, the bf16 O plus the `o_lo` residual at a 128-register budget). Per epilogue
  warp per tile, that is about 130 local stores and 200 loads.
- **Hardware-counted local requests are small:** 7.8 M loads and 6.9 M stores, 0.5% of
  the LSU peak.

This epilogue belonged to the bf16 sparse training forward on the 128-key mainloop, which no
route reaches any more (bf16 sparse training runs kb64).

Decode and prefill at 64 heads (`benchmarks/benchmark_sparse_mla_fwd.py --heads 64
--kernels 2cta 1cta 1cta_kb128 --ptxas shipped`; `agent_space/bench_sparse_1cta/b7_*.csv`),
each kernel at its shipped ptxas level:

| shape | 2CTA / kb64 | kb128 / kb64 |
|---|---|---|
| decode b <= 32 (cold / hot) | 0.77-0.92 | 1.17-1.57 |
| decode b = 128 | 1.12-1.32 | 1.01-1.23 |
| decode b = 512 | 1.08-1.21 | 1.05-1.23 |
| prefill s_q = 4096 | 1.64-1.76 | 1.19-1.39 |

Small-batch decode stays latency-bound and 2CTA still wins there (Follow-up 1).

## Dense kb64 with split-KV (64 heads, 16-bit)

The kb64 kernel has two load front ends, selected at compile time by `is_topk_gather`. The
MMA, softmax, epilogue and SMEM/TMEM plan are shared. Dense MLA at 64 heads routes there
unless the call is fp8 (`FlashAttentionMLAForward1CtaKb64Sm100.can_implement`). Paged KV whose page size
is not a multiple of 64 uses the cp.async gather front end (see "Dense kb64: paged KV at any
page size").

**Front end.** The TMA warp (warp 8) replaces the 4 cp.async gather warps, so dense runs
12 warps with registers split 208 / 168 / 128.
- A latent part (2 column tiles, 16 KB) is one TMA box. The builder lays out a
  `(64, 64, 128)` tiler with 12 "stages" byte-identically to the 3 latent stages, so the
  part box writes exactly the bytes the dual and MN-major views read.
- Each part lands on its own mbarrier (one `arrive_and_expect_tx` plus the box). The rope
  rows, or the last part when there is no rope, go on the stage's full barrier
  (`PipelineTmaUmma`).
- Q (the token's 64 packed rows) uses the same atoms and ring. The MMA copies Q to TMEM
  part by part, because with TMA the full barrier no longer implies the parts.
- Paged KV with `page_size % 64 == 0` takes block `n` to coordinates
  `(n % (page_size / 64), page_table[n // (page_size / 64)])`.

**Block range and masking.**
- `BlockInfo(64, 64)` gives each (tile, split) its range, as in the 128-key kernel. The
  count is a runtime value: 1 block, odd counts, and a fully masked dummy block when the
  range is empty.
- `has_kv_work` gates every role on empty splits. The epilogue then writes LSE = -inf.
- Masking is positional. The key limit is tile-uniform (one token per tile), so only the
  last block takes the select branch.
- An odd block count ends on row-max parity 0, so a pair barrier protects the final
  `sRowMax` write.

**Split-KV.**
- fp32 O / LSE partials are streamed with 256-bit stores, the sink is applied on split 0
  only, and the existing combine kernel merges the partials.
- The interface counts 64-row tiles and 64-key blocks. It caps splits at
  `num_n_blocks // 4` (at least 4 blocks per split): one block per split was 30% slower at
  b=1, s_k=8K, from per-split Q loads, 128 KB fp32 partials and combine work.
- Paged calls on the 1CTA route bound `max_seqlen_k` by the page-table row, not the pool.
  This also fixes over-splitting on the 128-key paged route.

**Scheduler and compiler.**
- CLC is on for dense kb64 prefill / extend (s_q > 1): +2-10% there, neutral on decode.
- ptxas stays at the default level; `-O2` measured neutral on decode and about 2% on prefill.

**Correctness.**
- vs the fp32 reference and vs the 128-key mainloop: the bf16-rounding contract.
- Paged runs are bitwise equal to contiguous runs.
- Bitwise run to run and with CLC on or off.
- Tests: `test_flash_attn_mla_1cta_dense_kb64` and `test_flash_attn_mla_1cta_dense_kb64_varlen_paged`,
  and 64 heads in `test_flash_attn_mla_1cta_learnable_sink`.
- The sparse front end is bitwise unchanged (`agent_space/kb64_sparse_ref.py`).

**Results** (GB300, 64 heads, bf16, cold L2)

Source: `agent_space/bench_dense_kb64.py`, `agent_space/bench_sparse_1cta/dense_kb64*.csv`.
"kb128 best" is the better of `num_splits` 1 and the heuristic. kb64 is at its shipped
defaults.

| shape | 2CTA (ms) | kb128 best | kb64 | kb128 / kb64 | 2CTA / kb64 |
|---|---|---|---|---|---|
| decode b=1, s_k 8K / 32K / 128K | 0.103 / 0.326 / 1.312 | 0.0293 / 0.0427 / 0.0682 | 0.0235 / 0.0346 / 0.0585 | 1.25 / 1.23 / 1.17 | 4.4 / 9.4 / 22 |
| decode b=8, s_k 8K / 32K / 128K | 0.096 / 0.337 / 1.272 | 0.0426 / 0.0837 / 0.2205 | 0.0343 / 0.0752 / 0.2001 | 1.24 / 1.11 / 1.10 | 2.8 / 4.5 / 6.4 |
| decode b=32, s_k 8K / 32K | 0.105 / 0.350 | 0.0859 / 0.2334 | 0.0775 / 0.2053 | 1.11 / 1.14 | 1.36 / 1.70 |
| decode b=128, s_k 8K / 32K | 0.228 / 0.823 | 0.2180 / 0.8034 | 0.1963 / 0.6958 | 1.11 / 1.15 | 1.16 / 1.18 |
| decode b=512, s_k 8K / 32K | 0.789 / 3.111 | 0.8449 / 3.2940 | 0.7324 / 2.7948 | 1.15 / 1.18 | 1.08 / 1.11 |
| paged decode b=8 / 128, s_k 32K, page 64 | - | 0.0864 / 0.8281 | 0.0766 / 0.7147 | 1.13 / 1.16 | - |
| paged decode b=8 / 128, s_k 32K, page 128 | - | 0.0840 / 0.7956 | 0.0762 / 0.7015 | 1.10 / 1.13 | - |
| causal prefill 1Kx16K / 4Kx4K / 4Kx16K | 1.130 / 0.663 / 4.295 | 1.989 / 1.114 / 6.750 | 1.442 / 0.816 / 5.229 | 1.38 / 1.37 / 1.29 | 0.78 / 0.81 / 0.82 |

ncu (`agent_space/ncu_dense/`):

| shape | kernel | time | throughput | long-scoreboard stall |
|---|---|---|---|---|
| split decode b=8, s_k=32K | kb64 | 54.7 us | DRAM 70.8% | 13.9 |
| split decode b=8, s_k=32K | kb128 | 65.3 us | DRAM 59.4% | 30.6 |
| causal prefill 4Kx4K | kb64 | 816 us | SM 83% | 7.7 |
| causal prefill 4Kx4K | kb128 | 1.12 ms | SM 63% | 19.4 |

No spills in either kernel.

Dense bf16 prefill still belongs on 2CTA (0.78-0.86x); every decode shape favours dense
kb64.

**Saturating bandwidth** (decode, cold L2)

Effective bandwidth is KV payload divided by time, with payload = `b x s_k x (64 + 512) x 2` B.
Each token's KV is streamed once per 64-head tile, and Q / O add under 0.5% at large batch.
The GPU ceiling measured with `agent_space/hbm_ceiling.py` on 8 GiB is 6.90 TB/s for a
device-to-device copy (read + write) and 6.37 TB/s for the best read-only torch reduction.
The HBM3e spec is about 8 TB/s.

| shape | KV | 2CTA | kb128 best | kb64 |
|---|---|---|---|---|
| b=8, s_k=8K / 32K / 128K | 0.07 / 0.28 / 1.12 GiB | 0.79 / 0.90 / 0.95 TB/s | 1.77 / 3.61 / 5.48 | 2.20 / 4.02 / 6.04 |
| b=32, s_k=8K / 32K / 128K | 0.28 / 1.12 / 4.50 GiB | 2.87 / 3.45 / 3.63 | 3.52 / 5.18 / 5.87 | 3.90 / 5.88 / 6.86 |
| b=128, s_k=8K / 32K | 1.12 / 4.50 GiB | 5.29 / 5.87 | 5.54 / 6.01 | 6.15 / 6.94 |
| b=512, s_k=8K / 32K | 4.50 / 18.0 GiB | 6.13 / 6.21 | 5.72 / 5.87 | 6.60 / 6.92 |
| paged b=128, s_k=32K, page 64 / 128 | 4.50 GiB | - | 5.83 / 6.07 | 6.76 / 6.89 |

- **kb64 saturates at about 6.9 TB/s** once a call streams at least ~4.5 GiB of KV. That is
  the device-to-device copy rate and about 86% of spec.
- **kb128 tops out at about 5.9-6.1 TB/s, and 2CTA at about 6.2.**
- **Below ~1 GiB the kernels are latency-bound**, not bandwidth-bound. b=1 reaches
  0.4-2.6 TB/s even with split-KV.

## Dense kb64: packed varlen scheduling

With `cu_seqlens_q` (and no `seqused_q`), dense kb64 schedules a flat grid over the `total_q` tokens, as the sparse front end does:
- `num_batch = 1`;
- CLC (`SingleTileLPTScheduler`), or `SingleTileScheduler` without CLC;
- every role recovers the batch and the batch-local token in `_tile_coords` (a binary search on `cu_seqlens_q`).

This is safe because a kb64 tile is one token (64 heads, or padded heads on decode), so it can never straddle two sequences. `seqused_q` keeps the per-batch `SingleTileVarlenScheduler`. `FLASH_ATTENTION_MLA_1CTA_PACKED_VARLEN=0` restores it for A/B runs.

Outputs are bitwise identical to the per-batch scheduler (`agent_space/kb64_varlen_ref.py`, 36 cases incl. split-KV 3 / heuristic and zero-length sequences; `test_flash_attn_mla_1cta_dense_kb64_packed_varlen_decode`).

Results (`agent_space/bench_kb64_varlen.py`, `agent_space/bench_sparse_1cta/kb64_varlen.csv`; GB300, 64 / 16 heads, cold):

| varlen shape | packed | per-batch | per-batch / packed | 128-key / packed |
|---|---|---|---|---|
| 2048 single-token seqs, s_k <= 4K | 0.76-0.80 ms | 1.00-1.05 ms | 1.31x | 1.36-1.48x |
| 512 single-token seqs, s_k <= 32K | 1.53-1.56 ms | 1.53-1.56 ms | 1.00x | 1.20-1.23x |
| 128 single-token seqs, s_k <= 32K | 0.52-0.54 ms | 0.52-0.59 ms | 1.00-1.13x | 1.36-1.37x |
| ragged causal prefill, 8 / 32 docs | 1.74 / 2.67 ms | 1.79 / 2.73 ms | 1.03x | 1.47 / 1.89x |

**Where packed wins.** Many short sequences, where the per-batch scheduler's per-tile batch lookup (a prefix scan) costs time. Bandwidth-bound shapes with few long sequences are neutral.

**CLC with packed varlen.** Neutral to 6% slower on decode (512 seqs: 1.66 vs 1.56 ms) and neutral on prefill. So the dense kb64 default is unchanged: CLC on for prefill only.

## Dense kb64: paged KV at any page size

With `page_size % 64 == 0`, a 64-key block is one TMA box inside a page. Any other page size (1, 16, 32, 48, 96, ...) puts a block across pages. Those calls now run the kb64 kernel with the sparse front end, instead of falling back to the 128-key mainloop's `PagedKVManager` gather:

- **Loader.** The 16-warp configuration: cp.async gather warps 12-15, registers 192 / 128 / 112 / 80, Q staged through the KV ring (`gather_q`).
- **Index source.** `CpasyncGatherKVManagerH64` gains a paged mode:
  - `load_index_paged` puts each row's physical page and in-page offset into the two index register sets, two blocks ahead, so the dependent page-table read stays off the issue path. The page size is a compile-time divisor.
  - `load_X` addresses the head's `(page_size, d, num_pages)` view.
  - Rows at or past `seqlen_k` are zero-filled by a row predicate. Their page-table entry is never used; the load reads entry 0 instead, so it stays in bounds.
- **Loop.** `load_cpasync_paged` walks the dense block range: runtime count (pairs plus an odd tail), split-KV, `has_work`, the dummy block. The per-block issue (`gather_block_paged`) is `gather_block` without the bitmask: latent parts on their part barriers, rope rows last.
- **Everything downstream is the dense kernel's:** MMA, positional masking, split-KV epilogue, scheduler.
- **Flag split.** `use_cpasync_kv` (the loader) is now separate from `is_topk_gather` (bitmask, fixed count). (A `FLASH_ATTENTION_MLA_1CTA_KB64_PAGED_CPASYNC=1` knob sent page sizes that are multiples of 64 through the gather too, for the A/B below; it has since been removed.)
- **Registers.** Every variant compiles to 128 registers and 0 B of local memory (`agent_space/spill_probe_kb64_paged.py`): decode with and without split, 16 heads, no rope, causal prefill, page 1. The epilogue already streams O in 32-column chunks.

**Correctness.**
- Paged runs are bitwise equal to the same kernel on contiguous KV. Page sizes 1 / 16 / 48 / 96 and forced-cp.async 64 were checked at 64 and 16 heads, with splits 1 / 3 and causal, including pages that straddle split boundaries, CLC on / off and no rope part.
- Each run also matches the reference, and the 128-key mainloop under the bf16 contract (`test_flash_attn_mla_1cta_dense_kb64_varlen_paged`, `agent_space/smoke_kb64_dense.py --paged --page-size N [--force-cpasync]`).
- The sparse, dense and varlen bitwise guards are unchanged.

**Performance** (`agent_space/bench_dense_kb64.py --pages ...`; `agent_space/bench_sparse_1cta/kb64_paged_cpasync*.csv`; GB300, bf16, s_k 32K, best of `num_splits` 1 / heuristic per kernel):

| 64 heads | page 1 | page 16 | page 32 | page 48 | page 96 |
|---|---|---|---|---|---|
| decode b 8, 128-key / kb64 (cold, hot) | 1.10, 1.16x | 1.10, 1.18x | 1.12, 1.17x | 1.12, 1.18x | 1.12, 1.18x |
| decode b 128 | 1.08, 1.09x | 1.15, 1.15x | 1.16, 1.17x | 1.16, 1.16x | 1.16, 1.17x |
| decode b 512 | 1.08, 1.08x | 1.15, 1.15x | 1.17, 1.17x | 1.17, 1.17x | 1.18, 1.18x |
| causal prefill 1K x 8K | 1.32, 1.28x | 1.32, 1.29x | 1.32, 1.29x | 1.32, 1.29x | 1.32, 1.29x |

- **16 heads (decode):** 1.08-1.16x at pages 1 / 16 / 48. Prefill with fewer than 64 heads stays on the 128-key mainloop.
- **Loader A/B at page 64** (`kb64_cpforce`): the cp.async gather is within +-1% of TMA on decode and 6% slower on causal prefill. That is below the 10% bar, so sub-64-row TMA boxes (the other design for small pages) are not worth building.

**Saturating bandwidth.** Effective payload bandwidth, `b * s_k * 576 * 2` bytes / time, against the 6.90 TB/s device-to-device copy ceiling:

| page | kb64 b 128 / 512 (cold) | 128-key b 128 / 512 (cold) |
|---|---|---|
| 1 | 6.21 / 6.14 TB/s | 5.72 / 5.67 TB/s |
| 16 | 6.67 / 6.64 | 5.82 / 5.77 |
| 32 | 6.72 / 6.67 | 5.77 / 5.72 |
| 48 | 6.64 / 6.63 | 5.73 / 5.69 |
| 96 | 6.75 / 6.74 | 5.83 / 5.73 |
| 64 (TMA) | 6.76 / 6.74 | 5.84 / 5.77 |

Pages of 16 keys or more reach within 3-4% of the ceiling, the same as the TMA path. Page 1 costs about 8%: every key is its own page-table entry, and rows are scattered 1.2 KB reads.

**ncu** (decode b 128, s_k 32K, split heuristic; `agent_space/ncu_dense/*_ps*.ncu-rep`):

| | kb64 cp.async page 16 | kb64 TMA page 64 | 128-key cp.async page 16 | kb64 cp.async page 1 |
|---|---|---|---|---|
| duration | 713 us | 700 us | 825 us | 769 us |
| DRAM throughput | 85.7% | 87.3% | 74.1% | 79.8% |
| issue slots busy | 16.4% | 13.5% | 8.9% | 14.7% |
| warp cycles per issued instruction | 16.4 | 15.3 | 32.7 | 18.3 |
| local spill requests | 0 | 0 | 0.59 M | 0 |

## kb64 with fewer than 64 heads

**Mechanism.**
- The tile stays one token and its rows pad to 64 in-kernel, as the sparse 128-key path
  does.
- Sparse gathers the Q rows with a row predicate (`CpasyncGatherKVManagerH64.load_X(...,
  num_valid_rows=H)`): predicated-off cp.async zero-fills the padded rows.
- Dense loads Q by TMA through the heads-first view with the real head extent
  (`qheads_first_tma_view` / `regroup_padded_qheads`), which zero-fills them.
- The padded rows alias the next token's heads, so the O stores are head-guarded.
  `store_lse` and the sink loader already were.
- 64 heads stay bitwise unchanged (`agent_space/kb64_{sparse,dense}_ref.py`).

**Routing** (measured, `agent_space/bench_sparse_1cta/{dense_kb64_h16,dense_kb64_h32,sparse_h1632*}.csv`):
- **Sparse, any head count up to 64, inference:** kb64 is 1.02-2.05x faster than the 128-key
  mainloop on decode and 1.20-1.58x on prefill. Small-batch sparse decode still favours
  2CTA (0.78-0.97x), as at 64 heads.
- **Dense decode (`seqlen_q = 1`):** kb64 is 1.07-1.30x faster than the 128-key mainloop at
  16 and 32 heads (split-KV, paged, and b up to 512).
- **Dense prefill, fewer than 64 heads:** stays on the 128-key mainloop. Padding wastes
  64 / H of the MMA work, while that mainloop packs 64 / H tokens per tile; kb64 measured
  0.34-0.43x at 16 heads and 0.66-0.81x at 32.
- **Training:** sparse recompute-P runs kb64 at 1..64 heads (see "Training below 64
  heads"); the other training routes run 2CTA.

**Tests.**
- `test_flash_attn_mla_1cta_dense_kb64` covers 16 / 24 / 64 heads.
- `..._dense_kb64_varlen_paged` covers 16 / 64 heads.
- `test_flash_attn_mla_1cta_dense_kb64_padded_head_canary` covers 1 / 24 / 48 heads, with and
  without a sink.
- The existing sparse 1CTA tests at 1-48 heads now take the bf16-rounding contract
  (`_mla_kb64_active(nheads <= 64)`).

## Training below 64 heads (recompute-P, padded head tiles)

Recompute-P used to require 64 or 128 heads, so sparse training below 64 heads could only
run load-P on the 2CTA forward. The recompute-P backward now pads its head tile (1..63 ->
64, 65..127 -> 128; `AI/SPARSE_MLA_RECOMPUTE_P.md`, "Head counts"):
- Q_rope and the QvB copy of Qv load through the padded heads-first TMA views;
- `lse_log2` padding is +inf, so P = 0 in the padded rows.

On the 1CTA route that training runs the kb64 forward. That forward already zero-fills the
padded Q rows and guards the O / o_lo / LSE stores.

The dQ / dQv GEMM now uses the 64-row `dQdQvGemmKernelH64` for 1..64 heads, not only 64: its
head mode takes the real, dynamic extent. The generic kernel had padded 1..63 heads to 128
rows. The output is bitwise identical to the generic kernel, and the change also speeds up
load-P.

Train step (GB300 GPU 3, b = 1, one causal document, topk 2048, token chunk 4096;
`agent_space/gate_loadp.py`, `agent_space/bench_sparse_1cta/gate_recompute_padded.log`):

| heads, T | recompute-P kb64 fwd | recompute-P 2CTA fwd | load-P 2CTA (now) | load-P 2CTA (before) |
|---|---|---|---|---|
| 24, 16K | **22.49 ms** | 24.88 | 28.68 | 30.22 |
| 32, 16K | **22.71** | 25.08 | 28.81 | 30.40 |
| 1, 16K | **21.97** | 24.31 | 28.76 | - |
| 24, 4K | **5.48** | 6.13 | 6.94 | 7.32 |
| 64, 16K (unchanged) | 23.46 | 25.89 | 29.34 | 29.27 |

- **Below 64 heads, training is 1.34x faster** than before (1.28x over today's load-P).
- **dQ / dQv at 24 / 32 heads:** 4.16 -> 2.60 ms.
- **The padded backward costs what the 64-head one does:** the main kernel takes 15.7 ms at
  24 heads vs 16.0 ms at 64.
- **At 96 / 128 heads** (tile 128, 2CTA forward) recompute-P is about 1% slower than load-P
  (37.9 vs 37.4 ms at 96, 16K). Its gain there is memory: P is never stored.

Correctness:
- recompute-P vs load-P grads rel-L2 about 3e-3 at 24 / 1 / 96 heads;
- padded rows contribute exactly zero (`test_flash_attn_mla_sparse_bwd_recompute_p_padded`);
- sentinel canaries (`..._sentinel`, `..._sentinel_varlen`) at 96 / 24 / 1;
- `precise_dpsum` at 24;
- kb64 vs 2CTA training contract at 64 / 24 / 1 heads (`..._train_recompute_p`). At one
  head, dsink is a single bf16 scalar: kb64 matches the fp32 reference exactly, while 2CTA
  is 1.4e-2 off it (bound 5e-2);
- the o_lo half-ulp bound at padded counts (`..._padded_head_canary[train]`);
- 64 / 128-head grads and the 24-head load-P dq / dqv unchanged
  (`agent_space/recompute_bwd_ref.py`: bitwise; dk / dv within atomic noise).

## fp8: S ahead of PV in the 128-key mainloop

The 128-key mainloop issues PVt(n) before S(n+1). That order is forced when one block's V
fills the V ring (bf16: 2 stages = the two dv halves of one block), because S(n+1) reads
V(n+1) and only PVt(n) frees V(n)'s slots. The MMA warp issues in order, and PVt(n) waits
for the softmax to produce P(n). So the tensor core idles through every softmax step.

fp8 has 4 V stages, i.e. two blocks resident. With `s_ahead` (`mma_pair_step`) the MMA warp
issues S(n) before PVt(n-1), and the tensor core computes S(n) while the softmax works on
block n-1. S stages keep the block parity (block n -> stage n % 2), so the softmax is
unchanged. The arithmetic is unchanged too: the output is bitwise identical to the in-order
order on dense (with and without split-KV), causal prefill (1 / 2 / 3 / odd block counts),
varlen-q, sparse, and paged (cp.async and TMA) (`agent_space/s_ahead_bitwise.py`).

S-ahead is not free. Both resident blocks are then held by the MMAs, so the loads lose their
one-block look-ahead. The interface therefore turns it on only when the tensor core is the
bottleneck: fp8 with `seqlen_q x heads >= 512`, i.e. at least 8 tiles sharing one KV stream,
which stays L2-resident. Varlen without a `max_seqlen_q` hint stays in order
(`FlashAttentionMLAForward1CtaSm100.use_s_ahead`).

Ablation (`agent_space/bench_fp8_s_ahead.py`, `agent_space/bench_sparse_1cta/fp8_s_ahead.csv`;
160 rows, all bitwise equal). Speedup over in order, min / median / max:

| fp8 regime | S-ahead forced | interface default |
|---|---|---|
| dense decode, 16 / 64 / 128 heads, b 1-512, s_k 8K / 32K, split and not (cold) | 0.84 / 0.93 / 1.17 | 0.96 / 1.00 / 1.16 (noise) |
| same, hot | 0.83 / 0.94 / 1.42 | 0.98 / 1.00 / 1.01 |
| paged decode (page 128 TMA, 64 cp.async) | 0.84-0.87 | 1.00 |
| sparse decode, 64 heads (cold / hot) | 0.92 / 1.12 median | 1.00 |
| dense causal prefill, s_q 256-4096 | 1.03 / 1.33 / 1.40 | same |
| sparse causal prefill | 1.09 | 1.09 |

**Adaptive order: tried and dropped.** At run time the MMA warp issued whichever of S(n) /
PVt(n-1) had its operand first, polled with `mbarrier.test_wait`. (`try_wait` can suspend
the thread; as a probe it made the kernel slower.) It matched in order on decode but reached
only 1.21x on prefill, and it showed an intermittent ~28 us stall on one small-batch sparse
decode shape per run. The shape moved between runs, and a nanosleep backoff did not fix it.

## Prefill: 1CTA vs 2CTA (causal, b = 1, cold)

Source: `agent_space/bench_prefill_1cta_vs_2cta.py`,
`agent_space/bench_sparse_1cta/prefill_1cta_vs_2cta.csv`. Each column is 2CTA-bf16 time
divided by the variant's time, so > 1 means faster than 2CTA. The 2CTA MLA kernel has no
fp8 path.

| kind | heads | shapes (s_q x s_k) | 1CTA bf16 | 1CTA fp8 (S-ahead) |
|---|---|---|---|---|
| dense | 16 | 1Kx16K / 4Kx4K / 4Kx16K | 0.53 / 0.58 / 0.58 | 1.06 / 1.13 / 1.26 |
| dense | 64 | same | 0.58 / 0.60 / 0.61 | 1.24 / 1.18 / 1.33 |
| dense | 128 | same | 0.57 / 0.56 / 0.64 | 1.24 / 1.06 / 1.40 |
| sparse (topk 2048) | 16 | 4Kx8K / 4Kx32K | 1.39 / 1.38 | 2.23 / 2.21 |
| sparse (topk 2048) | 64 | same | 1.66 / 1.64 (kb64) | 2.20 / 2.18 |

- **Dense bf16 prefill belongs on 2CTA.** Even the kb64 structure measured only 1.14-1.20x
  over 1CTA dense, via the arange-index proxy (`agent_space/bench_dense_vs_kb64_proxy.py`).
- **fp8 1CTA with S-ahead beats 2CTA bf16** on dense prefill. Sparse 1CTA wins at both
  precisions.

## Performance (GB300, 152 SMs, L2 129 MiB; bf16, topk 2048, h_kv 1)

These tables predate CLC on the sparse route and the kb64 mainloop (see the section
above for the current 64-head numbers).

Both kernels are compiled at the default ptxas level and at `-O2`
(`FLASH_ATTENTION_MLA_PTXAS_OPTIONS=-O2`). Ratios compare each kernel at its **best** level.
"Cold" flushes L2 before each call; "hot" replays a CUDA graph of back-to-back calls.
Data: `agent_space/bench_sparse_1cta/{decode,prefill}_p1.csv`, from
`benchmarks/benchmark_sparse_mla_fwd.py`.

### Decode (s_q = 1): 2CTA time / 1CTA time, cold (median over s_k in {8K, 32K, 128K}, has_qk in {0, 1})

| batch | h=16 | h=32 | h=48 | h=64 | 1CTA ms / 2CTA ms (h=64, has_qk, s_k=32K) |
|---|---|---|---|---|---|
| 1   | 0.61 | 0.62 | 0.61 | 0.62 | 0.059 / 0.036 |
| 8   | 0.62 | 0.62 | 0.63 | 0.63 | 0.059 / 0.037 |
| 32  | 0.67 | 0.67 | 0.67 | 0.67 | 0.063 / 0.043 |
| 128 | 1.08 | 1.08 | 1.09 | 1.09 | 0.077 / 0.084 |
| 256 | 1.07 | 1.07 | 1.07 | 1.07 | 0.137 / 0.149 |
| 512 | 0.97 | 0.97 | 0.97 | 0.97 | 0.259 / 0.254 |

Hot-cache ratios are within a few percent of these. The ratio does not depend on the head
count: padding 16 -> 64 costs the 1CTA kernel nothing measurable, and padding 16 -> 128
costs 2CTA nothing either, in wall-clock terms.

### Prefill (s_q = 4096, b = 1, causal): 1.18-1.21x in favour of 1CTA on all 16 configs

h=64, has_qk: 2CTA 1.49 ms vs 1CTA 1.23-1.25 ms (s_k 8K / 32K).

### Reading the numbers

- **Small batch is latency-bound.** With fewer tiles than SMs, per-token latency decides.
  One SM needs ~3 us per 128-slot block (1CTA *dense* over the same topk keys, without any
  gather, is within 10% of 1CTA sparse). 2CTA puts each token on two SMs; its "wasted" CTA
  still gathers half of the K/V rows, which roughly halves the latency.
- **Mid batch (~1 wave): 1CTA +7-9%.**
- **Large decode batch is DRAM-bound.** Every token reads its own ~2.25 MiB. Both kernels
  sit at ~4.7 TB/s effective payload (1CTA dense at ~5.1).
- **Prefill** shares KV across tokens, so it stays L2-resident, and 1CTA's lower per-token
  work shows as ~1.2x.
- **ptxas -O2** helps both sparse kernels by a median 15-17% (2CTA at batch 512: 25-27%).
  The default level spills: 2CTA sparse 560 B/thread local memory, 1CTA 352 B/thread;
  -O2 takes these to 32 B and 0 B. So the default level confounds any comparison. The few
  2x+ single-shape outliers at batch <= 8 look like measurement noise.

## ptxas -O2 (default for MLA kernels since 2026-09-28)

Each MLA kernel class carries its ptxas flags as a `ptxas_options` class attribute (`-O2`,
or `""` for the ptxas default on the kb64 forward); the interface adds them to the compile
options and the compile key. To rerun an ablation, change or monkeypatch the attribute.

Forward: see the numbers above (median +15-17%; local memory 560 -> 32 B on 2CTA, 352 -> 0 B
on 1CTA).

Backward, sparse MLA training step:
- Setup: b=1, T in {4K, 16K}, causal, topk 2048, heads {128, 64, 24}, load-P and recompute-P
  (the latter at 64 / 128 heads only at the time).
- Measurement: per-kernel CUDA time from the profiler. Two GPUs with opposite run orders
  (default -> -O2 on one, -O2 -> default on the other), which agree to within 1%.
- Script: `agent_space/bench_sparse_mla_bwd_ptxas.py`; data in
  `agent_space/bench_sparse_1cta/bwd_gpu*_*.csv`.

| kernel | default -> -O2 speedup (min / median / max) | local mem B/thread (default -> -O2) | default now |
|---|---|---|---|
| dQ/dQv GEMM (`dQdQvGemmKernel`) | 2.99 / 3.07 / 4.16x | 4064-4104 -> 0 | -O2 |
| main backward (`FlashAttentionSparseMLABackwardSm100`) | 1.00 / 1.01 / 1.18x (recompute-P 1.04-1.18x) | 280-1048 -> 0-64 | -O2 |
| dK GEMM (`dKGemmKernel`) | 1.03 / 1.04 / 1.04x | 0 -> 0 | -O2 |
| bwd preprocess (sparse-MLA instantiation) | 0.99 / 1.00 / 1.01x | 0 -> 0 | ptxas default |

## Interface cleanup (2026-09-29)

Before merging, the interface lost the ablation machinery used to develop and measure the
1CTA kernels (`agent_space/INTERFACE_REFACTOR_PLAN.md`):
- **Env knobs removed:** `FLASH_ATTENTION_MLA_1CTA_{KB64,Q_TMEM,PACKED_VARLEN,
  KB64_PAGED_CPASYNC,CLC,S_AHEAD}` and `FLASH_ATTENTION_MLA_PTXAS_OPTIONS`. Each resolved to
  its default: kb64 wherever it applies, Q in TMEM, packed varlen, TMA pages for whole
  64-key blocks, the CLC and S-ahead policies, and the per-kernel ptxas level.
  `FLASH_ATTENTION_MLA_1CTA=1` (the opt-in) remains until a routing heuristic replaces it.
  Measurements above that name a knob describe how they were taken.
- **Policies owned by the kernel classes:**
  - kb64: `can_implement`, `use_clc`, `TILE_MN` and `MIN_BLOCKS_PER_SPLIT`;
  - the 1CTA class: `use_clc`, `use_s_ahead`;
  - every MLA kernel: `SPARSE_HEAD_TILE` and `ptxas_options`.

  The interface picks the forward class once, and routes through `_mla_1cta_route`.
- **kb64 O stores are always 256-bit.** 32-B alignment is validated for a caller's `out`
  and declared through `to_cute_tensor(assumed_align=32)` /
  `assume_tensor_aligned(align_bits=256)`, instead of a pointer probe and a second kernel
  variant (that variant was worth 1-4% on epilogue-heavy shapes).
- **The 128-key kernel lost its two-phase (non-Q-in-TMEM) QK path:** slower on every
  shape, reachable only through the removed knob. Its stage-count and QK-order ablation
  arguments went with it.
- **Verification:** the compile-key set of the MLA suite was diffed before and after each
  step (`agent_space/snapshot_mla_keys.sh`, `agent_space/diff_keys.py`). Only the knob-only
  variants disappeared. The kb64 / 128-key / recompute-P bitwise references
  (`agent_space/kb*_ref.py`, `recompute_bwd_ref.py`) are unchanged.

## Dispatch heuristic (2026-09-29)

With `FLASH_ATTENTION_MLA_1CTA` unset, `interface._mla_1cta_route` picks the kernel. `1`
forces 1CTA wherever it is supported; `0` forces 2CTA.
- **fp8, descales, or an explicit `num_splits > 1`:** 1CTA. The 2CTA MLA kernel has none of
  these.
- **Sparse:** 1CTA whenever supported (<= 64 Q heads per KV head; training only with
  recompute-P).
- **Dense:** 1CTA on decode shapes, `seqlen_q x heads per KV head <= 64`: one 64-row tile per
  KV head, i.e. the kb64 mainloop with split-KV. 2CTA otherwise. Varlen without a host
  `max_seqlen_q` counts as prefill.

Measured (`agent_space/bench_dispatch.py`, `agent_space/bench_sparse_1cta/dispatch.csv`;
GB300, bf16, cold L2):

| case | 2CTA | 1CTA unsplit | 1CTA split (`num_splits=0`) | picks |
|---|---|---|---|---|
| dense decode h64, b 1-32, s_k 8K-32K | 0.10-0.35 ms | 0.12-0.45 | **0.02-0.20** | 1CTA |
| dense decode h64 b128 | 0.23 / 0.82 | **0.20 / 0.70** | 0.20 / 0.70 | 1CTA |
| dense 64 rows (h16 x 4, h32 x 2), b8 | 0.34 | 0.69 | **0.08** | 1CTA |
| dense 128 rows (h64 x 2, h16 x 8, h128 decode), b8 | 0.34 | 0.44-0.69 | **0.09-0.12** | 2CTA |
| dense decode h128 b128 | **0.84** | 1.41 | 1.40 | 2CTA |
| dense prefill 1K x 4K (h64 / h16) | **0.28 / 0.09** | 0.35 / 0.15 | 0.35 / 0.19 | 2CTA |
| sparse decode h64 b 1-32 | **0.036-0.042** | 0.046-0.050 | - | 1CTA |
| sparse decode h64 b128 | 0.084 | **0.074** | - | 1CTA |
| sparse prefill h64 2K | 0.79 | **0.50** | - | 1CTA |

Where it is right:
- **Dense decode**, with split-KV: 1.2-9x faster than 2CTA.
- **Dense prefill.**
- **Sparse at about one wave of tiles and above**, and sparse prefill (1.6x).

Where it is wrong:
- **Dense decode with the default `num_splits=1`.** 1CTA runs unsplit and is 20-35% slower
  than 2CTA below about one wave (b <= 32 at 64 heads). The win needs split-KV
  (`num_splits=0`, the split heuristic).
- **Small-batch 128-row dense decode**, which it leaves on 2CTA. There 1CTA with split is
  about 3-4x faster; the `<= 64` boundary is conservative.
- **Sparse decode below about one wave** (b <= 32): 2CTA is about 20% faster, at tens of
  microseconds.

## Follow-ups

1. **Routing heuristic.** Prefer 2CTA when the number of sparse tiles (b x s_q) is below
   roughly the SM count, and 1CTA from about one wave up. Measure the crossover more finely
   (96..256).
2. ~~Make -O2 the default for the MLA kernels.~~ Done (see above); the preprocess kernel is
   left at the ptxas default (no measurable change).
3. **Single-SM per-block latency** is the 1CTA ceiling at small batch. Candidates:
   - deeper V residency, where bf16 SMEM allows (it is tight: ~1 KB headroom with has_qk);
   - overlapping the per-tile Q staging (q_in_tmem handshake);
   - TMA gather4 for K/V rows, instead of 128 threads issuing per-row 16-B cp.async.
4. **Training forward:** done for recompute-P at 1..64 heads (see Status; kb64; "Training
   below 64 heads"). A kb64 load-P forward (P / row_max emission) was gated out: the
   load-P backward's separate dK GEMM makes it about 13% slower than recompute-P at 64
   heads (`agent_space/gate_loadp.py`).
5. **128-key mainloop spills: resolved, no action needed.** The 66 M figure is ncu's
   instrumented count. It is dominated by the reload of the SMEM base address inside the
   mbarrier retry loops (see the ncu table). The only real spills are in the bf16 sparse
   training epilogue (O plus `o_lo`), which default routing no longer reaches.

   Every shipped 128-key variant compiles to 0 B of local memory at -O2
   (`agent_space/spill_probe_128.py`):
   - dense bf16 prefill at 16 heads with CLC;
   - dense decode at 128 heads without CLC;
   - dense fp8;
   - sparse fp8 with CLC on and off.

   The bf16 sparse A/B build (KB64=0) has 16 B of local memory for inference with CLC,
   112 B without CLC (MMA warp), and 208-216 B for training.
6. **kb64 for < 64 heads.** It needs a predicated (or TMA) Q staging in place of the
   identity-row gather, and head guards in the O / LSE stores.
7. **Dense bf16 prefill** runs 0.53-0.64x of 2CTA on the 1CTA kernel (see "Prefill: 1CTA vs
   2CTA"); a routing heuristic should keep it on 2CTA if 1CTA ever becomes the default.

8. **Dense kb64 beyond 64 heads / fp8 / small pages / sub-64-head prefill.** These still run
   the 128-key mainloop:
   - 128 heads (two tiles per token, heads-first Q TMA view);
   - dense prefill with fewer than 64 heads (would need multi-token tiles with per-row causal
     limits);
   - fp8 (the dual TMEM packing assumes 16-bit).

   Page sizes that are not multiples of 64 now run kb64 (cp.async gather; see "Dense kb64:
   paged KV at any page size").
