# Sparse (top-k) MLA forward on the 1CTA kernel

Status (2026-09-28), opt-in via `FLASH_ATTENTION_MLA_1CTA=1`:
- **Inference forward:** MQA with up to 64 Q heads.
- **Training forward:** with the recompute-P backward (`gather_bwd_recompute_p=True`) at
  exactly 64 heads. The kernel produces what that backward consumes: exact-running-max LSE
  (`rescale_threshold=0`) and the O rounding residual `o_lo`, but no P / row_max.
- **Two mainloops.** Exactly 64 heads with 16-bit inputs run the **64-key-block (kb64)
  mainloop** (see its section below). It agrees with the 2CTA kernel to bf16 rounding, not
  bitwise. Everything else runs the 128-key mainloop (fewer than 64 heads, fp8, and
  `FLASH_ATTENTION_MLA_1CTA_KB64=0`). On the 128-key mainloop `out`/`lse`/`o_lo` are bitwise
  identical to the 2CTA kernel's, and so are the gradients: dQ and dQv bitwise, dK and dV
  within the sparse backward's own atomic run-to-run non-determinism, which 2CTA shows too.
- **CLC.** The 1CTA sparse route always runs the persistent CLC scheduler, as the 2CTA MLA
  kernel does. Dense 1CTA MLA still follows `FA_CLC`.
- **Fallback:** every other sparse case (more than 64 heads; training with load-P or
  != 64 heads) falls back to the 2CTA kernel under the flag.
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

## 64-key-block mainloop (kb64): exactly 64 heads, 16-bit

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
- **ptxas.** It runs at the default level (`_MLA_PTXAS_DEFAULTS["fwd_kb64"] = ""`).
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
| local-memory spill requests | 0 | 4.3 M | 65.9 M |
| top stall (per issue) | long scoreboard 5.8 | long scoreboard 5.9 | long scoreboard 18.0 |

kb64 and the PR's kernel have the same profile; kb64 is ahead by its spill-free register
allocation. The 128-key mainloop spills heavily under CLC at -O2 (see Follow-ups).

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
which stays L2-resident. Varlen without a `max_seqlen_q` hint stays in order.
`FLASH_ATTENTION_MLA_1CTA_S_AHEAD=0/1` forces the choice.

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

`interface._MLA_PTXAS_DEFAULTS` compiles the MLA kernels with `--ptxas-options '-O2'`. It is
part of each compile key. `FLASH_ATTENTION_MLA_PTXAS_OPTIONS` overrides every MLA kernel at
once, and `""` means the ptxas default level (use this to rerun the ablations).

Forward: see the numbers above (median +15-17%; local memory 560 -> 32 B on 2CTA, 352 -> 0 B
on 1CTA).

Backward, sparse MLA training step:
- Setup: b=1, T in {4K, 16K}, causal, topk 2048, heads {128, 64, 24}, load-P and recompute-P
  (the latter at 64 / 128 heads only).
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
4. **Training forward:** done for recompute-P at exactly 64 heads (see Status; kb64). Training
   with fewer than 64 heads would need P / row_max emission, because the recompute-P
   backward rejects padded head tiles. It is not started, and is worth it only in the
   throughput regimes above.
5. **128-key mainloop spills under CLC.** ncu shows 66 M local-memory spill requests at
   -O2 with CLC (the -O2 measurement was without CLC). It serves < 64 heads and fp8;
   re-ablate its ptxas level and budgets there.
6. **kb64 for < 64 heads.** It needs a predicated (or TMA) Q staging in place of the
   identity-row gather, and head guards in the O / LSE stores.
7. **Dense bf16 prefill** runs 0.53-0.64x of 2CTA on the 1CTA kernel (see "Prefill: 1CTA vs
   2CTA"); a routing heuristic should keep it on 2CTA if 1CTA ever becomes the default.
