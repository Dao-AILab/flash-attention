# Sparse (top-k) MLA forward on the 1CTA kernel

Status (2026-09-28), opt-in via `FLASH_ATTENTION_MLA_1CTA=1`:
- **Inference forward:** MQA with up to 64 Q heads.
- **Training forward:** with the recompute-P backward (`gather_bwd_recompute_p=True`) at
  exactly 64 heads. The kernel produces what that backward consumes: exact-running-max LSE
  (`rescale_threshold=0`) and the O rounding residual `o_lo`, but no P / row_max.
  `out`/`lse`/`o_lo` are bitwise identical to the 2CTA kernel's, so the gradients match: dQ
  and dQv bitwise, dK and dV within the sparse backward's own atomic run-to-run
  non-determinism, which 2CTA shows too.
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

Correctness: bitwise identical to the 2CTA kernel on every benchmarked shape and on the
test matrix (`test_flash_attn_mla_1cta_sparse_*`), as well as matching the reference.

## Performance (GB300, 152 SMs, L2 129 MiB; bf16, topk 2048, h_kv 1)

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
4. **Training forward:** done for recompute-P at exactly 64 heads (see Status). Training
   with fewer than 64 heads would need P / row_max emission, because the recompute-P
   backward rejects padded head tiles. It is not started, and is worth it only in the
   throughput regimes above.
