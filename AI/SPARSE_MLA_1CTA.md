# MLA forward on SM100: the 1CTA kernel

SM100 has two forward kernels for absorbed MLA (`qv` with hdim_v 512, plus an optional 64-dim
rope part `q` / `k`):

- **2CTA**, `FlashAttentionMLAForwardSm100` (`flash_fwd_mla_sm100.py`). One 128-row tile per
  2-CTA cluster.
- **1CTA**, `FlashAttentionMLAForward1CtaSm100` (`flash_fwd_mla_1cta_sm100.py`). One 64-row
  tile per CTA, using `tcgen05.mma.ws`. It has two mainloops:
  - the **128-key mainloop** (the base class);
  - the **64-key-block (kb64) mainloop**, `FlashAttentionMLAForward1CtaKb64Sm100`
    (`flash_fwd_mla_1cta_kb64_sm100.py`), a subclass.

`interface._mla_1cta_route` chooses between 1CTA and 2CTA (see "Dispatch heuristic").
`interface._mla_fwd_plan` then picks the 1CTA mainloop and its tile and split count.
`FLASH_ATTENTION_MLA_1CTA=1` / `0` forces the 1CTA kernel wherever it is supported / the 2CTA
kernel.

Related:
- `AI/SPARSE_MLA_RECOMPUTE_P.md`: the sparse backward, recompute-P and token chunking.
- `AI/SPARSE_MLA_64H.md`: the backward's 64-row-tile specializations.
- `AI/SPARSE_MLA_EXACT_SOFTMAX_MAX.md` and `AI/SPARSE_MLA_DPSUM_PRECISION.md`: the numerics
  of the training forward.

## Support

| | 2CTA | 1CTA, 128-key mainloop | 1CTA, kb64 mainloop |
|---|---|---|---|
| dense | yes | yes | 64 heads per KV head, or fewer on decode (`seqlen_q = 1`, one KV head) |
| sparse top-k gather | any head count (padded to 128) | <= 64 heads | <= 64 heads |
| more than 64 heads per KV head | yes | dense only | no |
| fp8 with descales | no | yes | no |
| split-KV | no (hdim_v 512) | dense | dense |
| paged KV | yes | any page size | any page size |
| varlen (`cu_seqlens`, `seqused`) | yes | yes | yes |
| learnable sink | yes | yes | yes |
| sparse training forward | load-P and recompute-P | recompute-P | recompute-P |

- The 1CTA training forward serves only the recompute-P backward
  (`gather_bwd_recompute_p=True`). It writes an exact-running-max LSE (`rescale_threshold=0`)
  and the O rounding residual `o_lo`, but no P / row_max.
- Neither kernel supports local attention, softcap, `score_mod`, `mask_mod` or block
  sparsity with `qv`; the interface rejects them.
- Sparse MLA has no split-KV on either kernel. `num_splits > 1` with `gather_kv_indices`
  raises `ValueError`.

## Numerics

- **128-key mainloop:** `out` / `lse` / `o_lo` are bitwise identical to the 2CTA kernel's. So
  are the sparse training gradients: dQ and dQv bitwise, dK and dV within the backward's
  atomic run-to-run noise (which 2CTA shows too).
- **kb64 mainloop:** the 64-key block order changes the running-max sequence, so it agrees with
  2CTA to bf16 rounding, not bitwise. The contract (`_assert_mla_fwd_close` in
  `tests/cute/test_flash_attn_mla.py`):
  - vs 2CTA: out rel-L2 < 5e-3 (measured about 2e-3, two independent bf16 roundings), the same
    -inf LSE pattern, finite LSE max-abs < 1e-4 (measured about 1e-6);
  - training (`test_flash_attn_mla_1cta_sparse_train_recompute_p`): out + o_lo rel-L2 < 2e-3,
    dq / dqv / dk / dv rel-L2 < 5e-3, dsink < 1e-2 (5e-2 at one head, where dsink is a single
    bf16 scalar), and |o_lo| <= half an ulp of out;
  - bitwise run to run, with CLC on or off, and paged vs contiguous KV.
- **fp8** pre-scales P by 2^8 before P @ V and removes the factor from the LSE (Note [Low
  Precision Scaling] in `flash_fwd_sm100.py`). It always runs with an exact running max: a stale
  max would let P saturate e4m3.

## Dispatch heuristic

With `FLASH_ATTENTION_MLA_1CTA` unset:
- **fp8, descales, or an explicit `num_splits > 1`** -> 1CTA. 2CTA has none of these.
- **Sparse** -> 1CTA where supported (<= 64 heads; training only with recompute-P), once 2CTA
  exceeds one wave: `2 x total_q x kv_heads > num_SMs`. Otherwise 2CTA.
- **Dense with `num_splits=0`** (the split heuristic) -> planned as 1CTA first. The split count
  comes from the 1CTA kernel's heuristic. 1CTA is kept with at least 2 splits on the kb64
  mainloop, or at least 5 on the 128-key mainloop (see "Auto split"). Otherwise the call is
  decided as unsplit.
- **Dense unsplit** -> 1CTA on decode shapes (`seqlen_q x heads_per_kv <= 64`, one 64-row tile)
  once 2CTA exceeds one wave, `2 x batch x kv_heads > num_SMs`. Otherwise 2CTA, including all
  prefill. Varlen without a host `max_seqlen_q` counts as prefill.

The thresholds were measured on GB300 (152 SMs). The one-wave rule scales with the SM count;
the split thresholds may need retuning on other parts.

### One-wave rule

A 2CTA tile holds one token's heads (sparse), or one batch element's decode rows (dense
unsplit), padded to 128 rows. So 2CTA spends 2 CTAs per token or batch element, and at 64
heads half of each tile is empty. 1CTA spends one CTA per 64-row tile.
- **Below one wave** of SMs, the empty half lands on SMs that would otherwise idle. These
  shapes are memory-bound, and 2CTA is faster: each token is served by two SMs.
- **Past one wave**, 2CTA needs a second wave, which nearly doubles its time, and 1CTA wins.

Crossover, GB300, bf16, cold L2 (2CTA ms / 1CTA ms):

| case | one wave | past one wave | further |
|---|---|---|---|
| dense unsplit decode, 64 heads, s_k 8K | b 72: 1CTA 1.13-1.23x slower | b 80: 0.211 / **0.151** | b 96-304: 1CTA 0.75-1.01x |
| dense unsplit decode, 16 heads, s_k 8K | b 64: 1CTA 1.14-1.23x slower | b 80: 0.210 / **0.149** | b 96-1024: 1CTA 0.71-0.94x |
| sparse decode, 64 heads, topk 2048 | 76 tokens: 0.052 / 0.061 | 80 tokens: 0.079 / **0.062** | 88-128 tokens: 1CTA 0.79-0.89x |
| sparse decode, 16 heads | 76 tokens: 0.052 / 0.059 | 80 tokens: 0.078 / **0.060** | 1CTA 0.79-0.87x |

The crossover is at the same batch for 64 and 16 heads, so the rule counts 2CTA CTAs, not rows.
Sparse prefill and training are far past one wave (1CTA 1.6x at 2K tokens).

### Auto split

With `num_splits=0` the 1CTA split count is `num_SMs // tiles`, capped by key blocks (kb64:
at least 4 blocks per split; an explicit `num_splits` is used as given). The 2CTA kernel never splits. Whether a split count beats 2CTA
depends on the mainloop (dense decode, one KV head, bf16, cold L2):

| rows per KV head (mainloop) | 1CTA splits | s_k 8K: 1CTA / 2CTA | s_k 32K: 1CTA / 2CTA |
|---|---|---|---|
| 128 (128-key), b 1-16 | 64-9 | 0.34-0.86x | 0.17-0.61x |
| 128 (128-key), b 24 | 6 | 1.05x | 0.82x |
| 128 (128-key), b 32 | 4 | 1.28x | 1.12x |
| 128 (128-key), b 48 / 64 | 3 / 2 | 1.50x / 1.80x | 1.37x / 1.75x |
| 64 (kb64), b 16-64 | 9-2 | 0.50-0.99x | 0.35-0.90x |
| 64 (kb64), b 76 | 2 | 1.04x | 0.94x |

- **kb64:** 2 splits already match 2CTA, which spends 2 CTAs on each half-empty tile.
- **128-key mainloop:** 2CTA covers a full 128-row tile with 2 CTAs, each about twice as fast
  as one 1CTA CTA, so 1CTA needs about 3x the CTAs: 5 or more splits.

With the variable unset and `num_splits=0`, the chosen kernel is within 1% of the faster one
on every benchmarked dense shape, cold and with CUDA graphs. The one exception is the
128-head, b 24, s_k 8K boundary case, at 1.05-1.06x. When the auto split falls back to 2CTA,
the second planning pass costs about 12 us of host time.

## 128-key mainloop

- **Tiles.** One 64-row tile per CTA. Dense packs tokens and heads into the tile (`pack_gqa`
  at any ratio). Sparse puts one token per tile and pads its heads to 64 **in-kernel**: Q / Qv /
  O use TMA on `pack_gqa.qheads_first_tma_view`, which zero-fills padded rows on load and drops
  them on store.
- **Layout E.** The `.ws` MMAs keep accumulators in the Layout E packed organization (a 64 x N
  accumulator as 128 lanes x N/2 columns), the same per-CTA packing the 2CTA kernel produces.
  So the softmax, correction and epilogue logic carry over.
- **S = Q K^T + Qv V^T.** The rope Q is staged into TMEM once per tile, and QK is one N=128
  `.ws` MMA with its A operand in TMEM. QvV runs over the two 256-dim dv splits. Without a
  rope part, QvV(dv0) zero-initializes S.
- **V is loaded once per key block.** P @ V reads the same bytes through an MN-major
  descriptor view (`sVt`). 2CTA loads V twice (K-major and transposed).
- **Load paths.**
  - Dense: TMA, including paged KV with `page_size == 128`.
  - Other page sizes: a cp.async gather warp group (warps 12-15, `PagedKVManager`).
  - Sparse: the same warp group gathers K and V rows by index (`CpasyncGatherKVManager`,
    `cta_group_size=1`). Invalid slots are predicated off, and a predicated-off cp.async
    zero-fills (`src_size = 0`), so there is no stale SMEM and no `0 * NaN`.
- **Sparse validity bitmask.** Built per key block by the gather warps: `0 <= idx <
  seqlen_k_limit`, with `seqlen_k_limit = q + 1 + s_k - s_q` under causal (bottom-right). The
  softmax warps apply it on every block. Each softmax thread owns one Layout E datapath half
  (64 columns, 2 bitmask words).
- **Varlen.** Sparse `cu_seqlens_q` uses the packed (flat over tokens) scheduler with
  batch-local indexing in every warp role (`_tile_coords`). TMA O stays on, since a one-token
  tile cannot straddle sequences. Dense Q-side varlen uses the varlen scheduler, with a
  per-row predicated O store.
- **Split-KV** (dense). fp32 O / LSE partials are stored straight from registers (no TMA O),
  and the existing combine kernel merges them. The sink applies on split 0 only.
- **Pipeline depths.** bf16 keeps one key block's V in the two V stages (its two dv halves).
  fp8 halves every byte and spends the room on a 4-stage V ring (two blocks resident): 17-29%
  on decode. Deeper K and P rings measured neutral or worse.
- **CLC.** Sparse always uses the persistent CLC scheduler: its tiles have uniform cost, and CLC
  overlaps a tile's epilogue with the next tile's gather (about 10% at 64 heads). Dense follows
  the global `FA_CLC` default (`use_clc`).
- **ptxas `-O2`** (`ptxas_options`). See "ptxas levels".

### fp8: S ahead of PV

In bf16 the MMA warp must issue PVt(n) before S(n+1). One block's V fills the V ring, S(n+1)
reads V(n+1), and only PVt(n) frees V(n)'s slots. PVt(n) waits for the softmax to produce P(n),
so the tensor core idles through every softmax step.

fp8's 4 V stages hold two blocks. With S-ahead (`mma_pair_step`) the MMA warp issues S(n)
before PVt(n-1), so the tensor core computes S(n) while the softmax works on block n-1. S
stages keep the block parity, so the softmax is unchanged, and the output is bitwise identical
to the in-order issue.

S-ahead costs the loads their one-block look-ahead, since both resident blocks are then held
by the MMAs. So it is on only when the tensor core is the bottleneck (`use_s_ahead`): fp8 with
`seqlen_q x heads >= 512`, at least 8 tiles sharing one L2-resident KV stream. Varlen without a
`max_seqlen_q` hint stays in order.

| fp8 regime | S-ahead forced (min / median / max over in-order) | as dispatched |
|---|---|---|
| dense decode, 16-128 heads, b 1-512, split and not | 0.84 / 0.93 / 1.17 | 1.00 (median) |
| paged decode | 0.84-0.87 | 1.00 |
| dense causal prefill, s_q 256-4096 | 1.03 / 1.33 / 1.40 | same |
| sparse causal prefill | 1.09 | 1.09 |

## kb64 mainloop

The 64-key-block mainloop serves 16-bit calls with at most 64 heads per KV head
(`Kb64.can_implement`): sparse at any such count, and dense at 64 heads, or with fewer on
decode. It is a port of PR #2914's 64-head sparse forward mainloop into the 1CTA kernel.

Why it is faster: the 128-key mainloop keeps one key block's V in both V stages, so it
serializes S -> softmax -> P -> PV -> V release -> refill -> next S. kb64 holds three smaller
blocks and issues S(n) before PV(n-1).

**Design**
- **Stages.** A latent stage is one block's 64 x 512 rows (64 KB, K-major SW128). There are
  three stages plus one 64 x 64 rope tile. Each stage lands in 4 column-block parts, each on
  its own mbarrier, so S streams behind the fill.
- **Q in TMEM.** Qv takes 128 "dual packed" columns: lanes 0-63 hold the lower dim half of
  each 128-dim chunk, lanes 64-127 the upper half. Q_rope takes 16 columns. The MMA warp copies
  Q in with `tcgen05.cp.128x256b` (`utccp_128x256b_ptx`).
- **S = Q K^T.** A `.ws` TS dual GEMM, M64 N128 (`gemm_ws_ts_ptx_partial`). The two lane
  halves of the accumulator hold the two dim halves' partial sums. The softmax warps add them
  through a 16 KB exchange buffer, with 64-thread pair barriers.
- **O += P V.** SS, M64 N256 x 2 N-tiles, over the MN-major view of the stage.
- **Issue order.** S(n) before PV(n-1). There is one S stage: TMEM is O 256 + S 64 + Qv 128 +
  Qr 16 = 464 of 512 columns.
- **Epilogue.** O and o_lo stream from TMEM 32 columns at a time with 256-bit `st.global.v8`,
  and each O split is released as soon as it drains. The stores need 32-B aligned rows. The
  interface allocates such buffers and rejects a caller's `out` that is not aligned.
- **SMEM.** 3 x 64 KB + 8 KB rope + 8 KB P + 16 KB exchange + header = 232,448 B, exactly the
  cap.
- **Registers.** With `min_blocks_per_mp=1` the per-warp-group budgets hold. No spills in any
  variant.
- **ptxas** at the default level; `-O2` is 7-8% slower here.

**Front ends** (compile-time; the MMA, softmax, epilogue and SMEM/TMEM plan are shared)
- **Sparse:** cp.async gather warps 12-15 with `CpasyncGatherKVManagerH64`. Whole rows,
  indices loaded two blocks ahead, 2 bitmask words per block, a fixed even block count
  (`topk / 64`). The token's Q rows are gathered through the KV ring, with a row predicate
  that zero-fills padded heads. 16 warps.
- **Dense:** the TMA warp (warp 8). A latent part (two 64 x 64 column tiles, 16 KB) is one TMA
  box on its part barrier; the rope rows, or the last part, go on the stage's full barrier.
  Q is loaded by TMA through the heads-first view, with the real head extent, which
  zero-fills padded heads. Paged KV with `page_size % 64 == 0` maps block `n` to
  `(n % (page_size / 64), page_table[n // (page_size / 64)])`. 12 warps.
- **Dense, paged, other page sizes:** the sparse gather warps, with the page table as the index
  source. `load_index_paged` puts each row's page and in-page offset into the index registers
  two blocks ahead; rows at or past `seqlen_k` are zero-filled by a row predicate. The dense
  block range, masking and epilogue are unchanged. (`use_cpasync_kv` selects the loader
  separately from `is_topk_gather`, which selects the bitmask and fixed block count.)

**Dense block range and masking**
- `BlockInfo(64, 64)` gives each (tile, split) its range: a runtime count, which may be 1, odd,
  or empty (then one fully masked dummy block). `has_kv_work` gates every role on empty splits;
  the epilogue then writes LSE = -inf.
- Masking is positional. The key limit is tile-uniform (one token per tile), so only the last
  block takes the select branch.

**Dense split-KV.** fp32 O / LSE partials go out with 256-bit stores; the sink applies on
split 0 only; the existing combine kernel merges. The split heuristic gives each split at least 4
key blocks (`MIN_BLOCKS_PER_SPLIT`; an explicit `num_splits` may leave splits empty, which the
epilogue and combine handle): one block per split was 30% slower at b 1, s_k 8K, from per-split Q
loads, 128 KB fp32 partials and combine work.

**Packed varlen (dense).** With `cu_seqlens_q` (and no `seqused_q`), dense kb64 schedules a
flat grid over the `total_q` tokens, as the sparse front end does. Every role recovers the
batch and batch-local token in `_tile_coords`, a binary search on `cu_seqlens_q`. A tile is
one token, so it never straddles two sequences. Outputs are bitwise identical to the
per-batch scheduler, which `seqused_q` still uses.

**Fewer than 64 heads.** The tile stays one token, padded to 64 rows. The padded rows alias
the next token's heads, so the O / o_lo stores are head-guarded, as `store_lse` and the sink
loader are. Dense prefill with fewer than 64 heads stays on the 128-key mainloop, which packs
64 / H tokens per tile; padding would waste 64 / H of the MMA work (kb64 measured 0.34-0.43x
at 16 heads and 0.66-0.81x at 32).

**CLC.** Sparse always. Dense on prefill / extend (`seqlen_q` hint > 1): +2-10% there,
neutral on decode (`use_clc`).

## Performance

GB300 (152 SMs, L2 129 MiB), bf16, one KV head, cold L2 (L2 flushed before each call) unless
noted. Each kernel runs at its shipped ptxas level. Forward benchmark:
`benchmarks/benchmark_sparse_mla_fwd.py`.

### Dense decode, 64 heads

| shape | 2CTA (ms) | 1CTA 128-key, best split | 1CTA kb64 | 2CTA / kb64 |
|---|---|---|---|---|
| b 1, s_k 8K / 32K / 128K | 0.103 / 0.326 / 1.312 | 0.029 / 0.043 / 0.068 | 0.024 / 0.035 / 0.059 | 4.4 / 9.4 / 22x |
| b 8, s_k 8K / 32K / 128K | 0.096 / 0.337 / 1.272 | 0.043 / 0.084 / 0.221 | 0.034 / 0.075 / 0.200 | 2.8 / 4.5 / 6.4x |
| b 32, s_k 8K / 32K | 0.105 / 0.350 | 0.086 / 0.233 | 0.078 / 0.205 | 1.36 / 1.70x |
| b 128, s_k 8K / 32K | 0.228 / 0.823 | 0.218 / 0.803 | 0.196 / 0.696 | 1.16 / 1.18x |
| b 512, s_k 8K / 32K | 0.789 / 3.111 | 0.845 / 3.294 | 0.732 / 2.795 | 1.08 / 1.11x |

**Bandwidth.** Effective bandwidth is KV payload, `b x s_k x 576 x 2` bytes, divided by time.
The device-to-device copy rate on the same GPU is 6.90 TB/s (HBM3e spec about 8).
- kb64 reaches about 6.9 TB/s once a call streams at least about 4.5 GiB of KV (b 32 at s_k
  128K, b 128 at 32K).
- The 128-key mainloop tops out at about 5.9-6.1 TB/s, and 2CTA at about 6.2.
- Below about 1 GiB the kernels are latency-bound: b 1 reaches 0.4-2.6 TB/s even with
  split-KV.

**Paged.** kb64 is 1.08-1.18x faster than the 128-key mainloop at every page size from 1 to
128 (b 8-512, s_k 32K; causal prefill 1.28-1.32x). Pages of 16 keys or more reach within 3-4%
of the copy rate, the same as TMA pages; page 1 costs about 8% (every key is its own
page-table entry).

**Fewer heads.** At 16 and 32 heads, dense decode on kb64 is 1.07-1.30x faster than on the
128-key mainloop (split-KV, paged, b up to 512).

**Packed varlen** (dense kb64, vs the per-batch scheduler): 1.31x for 2048 single-token
sequences with s_k <= 4K, neutral for a few long sequences, 1.03x for ragged causal prefill.

### Dense prefill

Dense bf16 prefill stays on 2CTA. As a ratio of 2CTA time to 1CTA time, causal, b 1, over
1K x 16K, 4K x 4K and 4K x 16K (> 1 means 1CTA is faster):

| heads | 1CTA bf16 (128-key) | 1CTA bf16 (kb64) | 1CTA fp8 (S-ahead) |
|---|---|---|---|
| 16 | 0.53-0.58 | - | 1.06-1.26 |
| 64 | 0.58-0.61 | 0.78-0.82 | 1.18-1.33 |
| 128 | 0.56-0.64 | - | 1.06-1.40 |

fp8 on 1CTA beats bf16 on 2CTA; 2CTA has no fp8 path.

### Sparse, 64 heads, topk 2048

| shape | 2CTA / kb64 | 128-key / kb64 |
|---|---|---|
| decode b <= 32 (cold / hot) | 0.77-0.92 | 1.17-1.57 |
| decode b 128 | 1.12-1.32 | 1.01-1.23 |
| decode b 512 | 1.08-1.21 | 1.05-1.23 |
| prefill s_q 4096 | 1.64-1.76 | 1.19-1.39 |

Small-batch sparse decode is latency-bound and stays on 2CTA (the one-wave rule): 2CTA puts
each token on two SMs. At 16-48 heads kb64 is 1.02-2.05x faster than the 128-key mainloop on
decode and 1.20-1.58x on prefill. Sparse fp8 prefill on 1CTA is 2.2x the 2CTA bf16 speed.

Training forward at 16K tokens, causal (ncu):

| | kb64 | 128-key mainloop |
|---|---|---|
| duration | 3.84 ms | 4.74 ms |
| SM throughput | 70.7% | 57.3% |
| memory throughput | 54.9% | 32.9% |
| issue slots busy | 33.7% | 17.2% |
| top stall (per issue) | long scoreboard 5.8 | long scoreboard 18.0 |

kb64 matches PR #2914's H64 forward (3.94 ms), and is slightly ahead on its spill-free
register allocation.

### Sparse training step

b 1, one causal document, topk 2048, token chunk 4096, forward + backward:

| heads, tokens | 1CTA kb64 fwd + recompute-P | 2CTA fwd + recompute-P | 2CTA fwd + load-P |
|---|---|---|---|
| 64, 16K | **23.46 ms** | 25.89 | 29.34 |
| 32, 16K | **22.71** | 25.08 | 28.81 |
| 24, 16K | **22.49** | 24.88 | 28.68 |
| 1, 16K | **21.97** | 24.31 | 28.76 |
| 24, 4K | **5.48** | 6.13 | 6.94 |

Below 64 heads recompute-P pads the backward tile to 64 rows (`AI/SPARSE_MLA_RECOMPUTE_P.md`,
"Head counts"); the padded backward costs what the 64-head one does. At 96 / 128 heads (tile
128, 2CTA forward) recompute-P is about 1% slower than load-P; its gain there is memory, since
P is never stored.

## ptxas levels

Each MLA kernel class carries its ptxas flags as a `ptxas_options` class attribute, which the
interface adds to the compile options. The compile key does not repeat them: the class is
already part of it.

| kernel | level | measured |
|---|---|---|
| 2CTA forward | `-O2` | sparse forward median +15-17% (batch 512: +25-27%); local memory 560 -> 32 B/thread |
| 1CTA forward, 128-key | `-O2` | sparse forward median +15-17%; local memory 352 -> 0 B/thread |
| 1CTA forward, kb64 | default | `-O2` 7-8% slower |
| dQ/dQv GEMM (`dQdQvGemmKernel`) | `-O2` | 3.07x median; local memory about 4 KB -> 0 |
| main backward (`FlashAttentionSparseMLABackwardSm100`) | `-O2` | 1.01x median (recompute-P 1.04-1.18x); local memory 280-1048 -> 0-64 B |
| dK GEMM (`dKGemmKernel`) | `-O2` | 1.04x |
| sparse-MLA bwd preprocess | default | no change |

The 128-key forward compiles to 0 B of local memory at `-O2` in every routed variant. ncu's
instrumented spill counter overstates it: about 97% of the executed local loads it counts are
one reload of the SMEM base address inside the mbarrier retry loops, which runs only while the
warp is already blocked.

## Alternatives measured and not taken

- **Two-phase QK in the 128-key mainloop** (Q in SMEM, SS QK with a half-size K slot): slower on
  every shape than the Q-in-TMEM path.
- **Adaptive MMA issue order** (fp8: issue whichever of S(n) / PVt(n-1) has its operand first,
  polled with `mbarrier.test_wait`): matched in-order on decode but reached only 1.21x on
  prefill (S-ahead: 1.33x), with an intermittent ~28 us stall on small-batch sparse decode.
- **Sub-64-row TMA boxes for small pages** in dense kb64: the cp.async gather is within 1% of TMA
  on decode and 6% slower on causal prefill, so not worth a second TMA path.
- **A kb64 load-P training forward** (P / row_max emission): the load-P backward's separate dK
  GEMM makes the step about 13% slower than recompute-P at 64 heads.
- **CLC with dense packed varlen decode:** neutral to 6% slower, so dense kb64 keeps CLC for
  prefill only.
- **One key block per split (kb64):** 30% slower than 4 at b 1, s_k 8K.

## Limitations and follow-ups

- **Small-batch sparse decode** is latency-bound on one SM per token, so it stays on 2CTA.
  Candidates for the 1CTA per-block latency:
  - deeper V residency, where SMEM allows (about 1 KB of headroom with a rope part);
  - overlapping the per-tile Q staging;
  - TMA gather4 for K / V rows instead of per-row 16-B cp.async.
- **kb64 does not cover** more than 64 heads (two tiles per token would re-gather the same
  indices), fp8 (the dual TMEM packing assumes 16-bit), or dense prefill below 64 heads (would
  need multi-token tiles with per-row causal limits). These run the 128-key mainloop.
- **Sparse split-KV** is not implemented on either kernel.
