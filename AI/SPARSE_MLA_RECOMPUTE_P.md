# Sparse MLA backward: in-kernel recompute-P and token-chunked backward

Design notes for `gather_bwd_recompute_p` and `gather_bwd_token_chunk`
(SM100 sparse top-k MLA path, `flash_fwd_mla_sm100.py` /
`flash_bwd_mla_sm100.py`).

## Problem: memory anatomy of the sparse-MLA train step

The DSA-style sparse MLA kernel computes, per query token, attention over
`topk_length` (typically 2048) gathered KV slots at 128 q-heads per kv-head.
For training, the forward saves the unnormalized probabilities
`p` (bf16 `[T, 128, topk]` = 512 KiB/token at topk 2048) plus the per-128-block
online `row_max` (fp32, 8 KiB/token), and the backward materializes a second
p-sized tensor `ds` for the dS intermediate consumed by the `dq_dqv` and `dk`
gather/scatter GEMM kernels.

At 64k tokens that is 32 GiB of saved activations held across the whole
fwd→bwd window (per layer, when activation checkpointing is off) plus a
32 GiB backward transient; a 64k train step peaks at ~82 GiB of
attention-owned memory and 128k does not fit on a 186 GiB GB200.

Two independent, composable features fix this:

1. **`gather_bwd_recompute_p`** — the forward saves only `out` + `lse`
   (saved activations drop from 520 KiB/token to ~0.5 KiB/token), and the
   backward main kernel reconstructs P in-kernel from an extra S^T GEMM.
2. **`gather_bwd_token_chunk`** — the backward runs its three kernels in
   token chunks, so `ds` (and every other backward transient) is bounded by
   the chunk size instead of the sequence length.

## Design: in-kernel recompute-P

### Math

```
S^T   = V_latent · Qv^T + K_rope · Q_rope^T        (extra UMMA per n-block group)
e     = softmax_scale·log2(e) · S − lse·log2(e)    (lse saved by fwd, natural log)
e     = −inf  where the topk slot is invalid       (idx < 0 or idx ≥ causal key limit)
P^T   = exp2(e) → bf16
```

Everything downstream is unchanged: `dS^T = P^T ⊙ (dP^T − dpsum)·scale`,
`dV += P^T@dO + …`, dS TMA store, and the `dq_dqv`/`dk` kernels are untouched.

Masking the *exponent* (not S) handles rows whose every slot is invalid:
their `lse = −inf` would otherwise produce `−inf − (−inf) = NaN`; with the
exponent forced to −inf they cleanly produce P = 0.

The forward in recompute mode simply skips the p/row_max stores — it is the
existing inference specialization, so **train-forward out/lse are
bitwise-identical** to the default path (asserted in the test) and the train
forward gets faster (no 512 KiB/token store on the softmax critical path).

The backward preprocess emits `lse_log2` (lse converted to log2 units)
alongside dpsum; `lse = −inf` rows write 0.0, which is safe because those
rows are fully masked by the bitmask.

### Numerics

P is reconstructed with a *single* rounding (`exp2(e) → bf16`), whereas the
default path stores `p = exp2(scale·S − row_max_blk)` and multiplies by a
bf16 `scale_p` in the backward — two roundings. Gradients are therefore not
bitwise-identical to the default path but measure *more* accurate against an
fp32 reference (e.g. dq rel-err 3.41e-3 vs 3.95e-3 at T=1024, W=2048).
P ∈ [0,1] exactly (lse ≥ scale·max), no overflow concerns.

Because nothing saved is consumed, re-running the backward over the same
graph (`retain_graph=True`) works in this mode (asserted in the test).
Second-order gradients (grad-of-grad) are not implemented by these autograd
Functions, same as the default path.

### Kernel restructure (SM100, 232448 B smem cap)

The main backward kernel was already at the exact SM100 smem limit, so the
recompute operands are funded by re-budgeting rather than growth:

- **S^T UMMA reuses `tiled_mma_VdO`** (M = 128 topk rows, N = 128 heads):
  A-operand is the already-gathered `sV` stages (each stage consumed by the
  S-chunk and then the dP-chunk before release), rope chunk accumulates from
  gathered `sKr` (new 8 KiB stage) × stationary `sQr` (8 KiB).
- **Role swap**: the 64 KiB stationary buffer that used to hold dO now holds
  QvB (the S-GEMM B operand); dO instead rides the existing 2-stage
  dOt/Qvt multiplex pipeline per group. Those reloads re-read
  per-token-stationary data, so they hit L2, not HBM; net HBM traffic of the
  whole design is *lower* than the default path (it no longer loads p:
  −512 KiB/token, +256 KiB/token K_rope gather).
- **Merged P/dS smem**: P and dS share one 16 KiB buffer. The P-store is
  gated on the previous group's dS drain; the dS-store is gated on a
  pre-claimed P-empty credit (P is consumed by the dV leg strictly before dS
  overwrites it).
- `sScaleP` (obsolete) becomes `sLse`. The exponent mask reuses the existing
  slot-validity machinery: the gather warps compute the per-group bitmask
  (idx valid and inside the causal key limit) and publish it through
  `sBitmask`/`pipeline_bitmask`; the softmax warps consume it to force
  invalid slots' exponents to −inf before `exp2`.
- S is issued at the top of each n-block-group so the softmax↔MMA ping-pong
  (the critical path) overlaps the dV legs of the previous group.

Both paths coexist as a compile-time specialization
(`recompute_P` constexpr); the default path's schedule is unchanged.

## Design: token-chunked backward

`_flash_attn_bwd_sparse_mla` slices the token axis into chunks of
`gather_bwd_token_chunk` and runs preprocess once plus the
(main bwd → dq_dqv → dk) triple per chunk, with `ds` allocated at chunk size
and reused (kernel launches on one stream serialize chunk i's consumers
before chunk i+1's producer). All per-token tensors are sliced per chunk;
`dk`/`dv` accumulate across chunks exactly as they accumulate across CTAs
(pre-zeroed fp32 atomics). Requires varlen or batch 1 so token slices stay
contiguous; the compiled kernels are shape-polymorphic, so chunks trigger no
recompilation.

### The causal exactness invariant

The gather bitmask validates a slot as `0 ≤ idx < seqlen_k_limit` with

```
seqlen_k_limit = m_local + 1 + seqlen_k − seqlen_q     (bottom-right aligned)
```

With chunk-local `m_local`/`seqlen_q` this limit *relaxes* for non-final
chunks. In the default (load-p) path that is harmless — entries the forward
masked carry p = 0. In recompute mode it would be a correctness bug: the
backward would recompute P ≠ 0 for entries the forward assigned −inf (and
NaN on lse = −inf rows). The chunked backward therefore shrinks the main
kernel's K extent per chunk to

```
k_end = seqlen_k − seqlen_q + tok1
```

which makes the chunk-local limit equal the forward's absolute limit for
every row. Non-varlen: pass `v[:, :k_end]` / `dv[:, :k_end]` /
`k[:, :k_end]` views (`k_end ≤ 0` means the whole chunk was fully masked in
the forward: skip all three kernels for the chunk and zero its `dq`/`dqv`
directly — `ds` is never touched). Varlen: clamp the
`cu_seqlens_k` *end* offsets per doc — only the doc containing the chunk end
actually shrinks, docs after it have no queries in the chunk, and all doc
*start* offsets that matter are unchanged (the clamped array stays
monotonic). The downstream `dq_dqv`/`dk` kernels keep full K/V and original
`cu_seqlens_k`: they apply no mask, and out-of-limit slots have dS = 0
(their gathers multiply by zero; scatters of zero are also suppressed by the
sentinel guard).

Slots beyond a shrunken K extent are excluded from the gather by the
bitmask, and the dV epilogue scatter is guarded on `0 ≤ idx < seqlen_k` (the
kernel's — possibly sliced — K extent), so such slots issue no atomic adds
at all. Their dV contribution is exactly zero anyway (their P, and hence
their P^T·dO rows, are zero), so the guard only removes wasted atomics and
keeps every write inside the sliced view's logical extent. Chunked `dv` was
measured byte-identical to unchunked in the load-p AB runs.

### What chunking does and does not change

- dq/dqv are pure GEMM consumers of identical dS tiles → **bitwise-identical**
  to the unchunked backward (asserted in the test, including causal masking
  with deliberately non-causal indices, and varlen docs split across chunks).
- dk/dv accumulate with fp32 atomics; splitting into several launches
  changes the accumulation order, giving deltas of the same magnitude as the
  unchunked kernel's own run-to-run nondeterminism (~1e-5..2e-4 rel-l2, vs
  ~4e-3 distance to an fp32 reference).

## Benchmarks

See tables below; measured on this branch vs upstream `main` (0251105) with
the same harness, GB200, B=1, W=2048, H=128, square T=S, bf16, median of 12
train steps, peak = `max_memory_allocated` delta over the step (inputs and
grad_out excluded). tilelang reference points measured 2026-08-20 on the
same machine/methodology from `tilelang` @ 0.1.13:
`examples/deepseek_v32/sparse_mla_{fwd,bwd}.py` (H=128, the head-split
backward — the apples-to-apples variant) and
`examples/dsa_sparse_finetune/` (H=64 — its backward exceeds SM100 smem at
H=128; numbers below are for HALF the head-FLOPs), both with -1 index tails
remapped to 0 (their gathers/scatters are unguarded; identical traffic).

### Train step: this PR (`recompute_p=True, token_chunk=4096`) vs upstream main

| T=S  | main (ms / GiB) | this PR (ms / GiB) | time Δ | memory |
|------|-----------------|--------------------|--------|--------|
| 4k   | 12.75 / 5.14    | 13.44 / 3.08       | +5.4%  | −40%   |
| 8k   | 25.90 / 10.28   | 27.46 / 4.15       | +6.0%  | −60%   |
| 16k  | 53.61 / 20.55   | 56.40 / 6.31       | +5.2%  | −69%   |
| 32k  | 115.58 / 41.10  | 119.68 / 10.62     | +3.5%  | −74%   |
| 64k  | 260.62 / 82.20  | 262.75 / 19.23     | +0.8%  | −77%   |
| 128k | OOM (186 GiB)   | 575.43 / 36.47     | —      | —      |

Saved activations (held fwd→bwd, per layer): 507.8 MiB/1k tokens → **0**
(only out+lse remain, which every attention layer saves anyway).
`token_chunk=2048` trades ~+1% time for another GiB (64k: 265.58 / 18.23).
Train forward alone at 8k: 4.43 → 3.77 ms (−15%), forward-phase transient
5.07 → 1.00 GiB (no p/row_max store). The relative step-time cost shrinks
with T because the recompute additions are per-group constants while the
gather/scatter GEMMs grow linearly.

The default path is untouched: same machine, this branch with no flags,
8k: 25.85 ms / 10.28 GiB; 64k: 260.27 ms / 82.20 GiB (main: 260.62 / 82.20).

For multi-layer training the comparison that matters is against activation
checkpointing: a checkpointed baseline layer pays a full extra train forward
(~25.9 + 4.4 ≈ 30.3 ms at 8k) for the same zero-saved-activation footprint;
recompute-P at 27.5 ms is faster, and only the P recompute (not the whole
attention) is redone.

### vs tilelang

tilelang deepseek_v32 (H=128, same head-FLOPs — the fair comparison):

| T=S  | tilelang dsv32 (ms / GiB) | this PR (ms / GiB) | speedup |
|------|---------------------------|--------------------|---------|
| 4k   | 64.08 / 1.08              | 13.44 / 3.08       | 4.8×    |
| 8k   | 105.50 / 2.16             | 27.46 / 4.15       | 3.8×    |
| 16k  | 188.53 / 4.32             | 56.40 / 6.31       | 3.3×    |
| 32k  | 368.55 / 8.64             | 119.68 / 10.62     | 3.1×    |
| 64k  | 730.25 / 17.27            | 262.75 / 19.23     | 2.8×    |
| 128k | 1480.45 / 34.55           | 575.43 / 36.47     | 2.6×    |

tilelang dsa_sparse_finetune (H=64 — half the head-FLOPs; its backward
exceeds SM100 smem at H=128):

| T=S  | tl finetune H=64 (ms / GiB) | this PR H=128 (ms / GiB) |
|------|-----------------------------|--------------------------|
| 8k   | 137.29 / 1.09               | 27.46 / 4.15             |
| 64k  | 1094.00 / 8.74              | 262.75 / 19.23           |
| 128k | 2213.11 / 17.48             | 575.43 / 36.47           |

Before this PR the tilelang kernels used ~5× less memory than FA4 while
being 2.6–5× slower; after it, FA4 matches their memory scaling
(285–300 MiB/1k tokens vs their 270) at the same 2.6–4.8× speed advantage.
The remaining ~2 GiB gap at 64k is the chunk-sized dS transient — removing
it entirely would require fusing the dq/dk GEMMs into the main kernel
(the tilelang single-kernel shape), a much larger rewrite.

## Validation

- New test `test_flash_attn_mla_sparse_bwd_recompute_p` (causal × shared_kv):
  bitwise fwd, fp32-reference grads, bitwise chunked-vs-unchunked dq/dqv
  with non-causal indices under causal=True, retain_graph.
- Existing sparse-MLA sentinel tests (12) pass with and without the new
  flags; the absorbed-MLA suite (96 non-varlen params) passes with
  recompute-P enabled.
- AB harnesses (adversarial non-causal indices under causal=True, varlen
  docs split across chunks, cp-fragment rectangular shapes, all-invalid
  rows, shared-kv): chunked-vs-unchunked dq/dqv bitwise in every config;
  recompute grads more accurate than the default path vs fp32 reference.
