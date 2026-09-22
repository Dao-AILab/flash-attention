# Sparse-MLA training forward: exact running softmax max

Why sparse-MLA (top-k gathered KV) forwards whose inputs require grad run the online
softmax with an exact running max (`rescale_threshold = 0`) instead of FA4's lazy rescale
(`rescale_threshold = 8` log2 units, still used for inference), what it fixes, what it
costs, and how it is tested.

Code: `flash_fwd_mla_sm100.py` (`FlashAttentionMLAForwardSm100(rescale_threshold=...)`,
forwarded to `SoftmaxSm100`), `interface.py` (`mla_fwd_rescale_threshold`: 0.0 when
`requires_grad and sparse_kv`, 8.0 otherwise; part of the compile key).

## Mechanism

The lazy rescale keeps the stale row max unless a block's max exceeds it by more than the
threshold, so a block's probabilities are `p = exp2(scale * (S - m_stale))`, up to 2^8, and
the row's dominant key gets `p = exp2(delta)` with a non-integer `delta` instead of exactly
1.0. `p` is rounded to bf16 as the P@V MMA operand while `row_sum` is accumulated from the
fp32 values, so the rounding of the dominant `p` (~2^-9 relative) is a *coherent* gain error
on the whole output row: the effective attention weights no longer sum to 1. With an exact
running max the dominant `p` is exactly 1.0 (no rounding) and the remaining roundings are
incoherent across the row.

The kernel walks the gathered index blocks from the last to the first
(`flash_fwd_mla_sm100.py`, `n_block = n_block_max - 1; ... n_block -= 1`). A top-k indexer
puts the strongest key in slot 0, which is therefore processed **last**, i.e. exactly the
case where the stale max is wrong. The error also flips with the *order* of the indices:
the same set in ascending order (self key last, processed first) does not show it.
Peaked rows (a token attending mostly to its own key) are the norm in causal DSA.

## Measurements

T = S = 4096, W = 2048, H = 128, bf16, fp64 reference from the same bf16 inputs, peaked
inputs (`q_t += 0.25 * k_t`, `qv_t += 0.25 * v_t`; median self weight 0.17, p90 0.41).
"row gain" = per-(token, head) `<out - o_ref, o_ref> / <o_ref, o_ref>`, the coherent
component; rel-L2 in absolute units (1.66e-3 = the bf16 output-rounding floor,
`bf16(o_ref)` vs `o_ref`).

Same top-k set, dominant key processed first vs last:

| forward | out rel-L2 vs fp64 | out row-gain rms vs fp64 | out diff between the two orders | row-gain of the difference / noise floor |
|---|---|---|---|---|
| lazy max, dominant first | 1.67e-3 | 1.48e-4 | – | – |
| lazy max, dominant last | 2.07e-3 | 1.23e-3 (p99 3.1e-3, p100 4.6e-3) | 2.16e-3 | 1.24e-3 / 1.95e-4 = 6.4x |
| exact max, either order | 1.67e-3 | 1.38e-4 .. 1.46e-4 | 7.3e-4 | 1.15e-4 / 1.10e-4 = 1.05x |
| FlashMLA sparse forward, either order | 1.67e-3 | 1.48e-4 .. 1.79e-4 | 7.8e-4 | 1.68e-4 / 1.14e-4 = 1.5x |

The noise floor is what independent per-element bf16 rounding of P would give
(`elem_rms * sqrt(3 / D)`); the elementwise difference between two orders is inherent to
any bf16-P online softmax (P is rounded relative to the running max at the time its block
is processed) and its grad-norm footprint is ~1e-6 in every pipeline. Only the coherent
part is fixable, and the exact max removes it.

Backward consequences (this change alone, dpsum from the bf16 output, load-P): dq rel-L2
vs fp64 3.18e-3 (dominant first) vs 3.36e-3 (dominant last) with the lazy max; with the
exact max the two orders agree to three digits. With the O-residual dpsum of
`AI/SPARSE_MLA_DPSUM_PRECISION.md` on top, dq is 2.39e-3 .. 2.40e-3 for every order.

## Cost

GB200, T = S = 16384, W = 2048, H = 128, load-P, same GPU back-to-back: training forward
+~4% (the O/row_sum rescale runs whenever the block max grows instead of only on jumps
> 2^8). Backward, memory and the inference forward are unchanged (inference keeps
threshold 8; its kernel is bitwise the same build). Training and inference forward outputs
are no longer bitwise equal.

## Test

`tests/cute/test_flash_attn.py::test_flash_attn_mla_sparse_topk_order_invariance`: same set
with the dominant key first and last (`self_including_topk_indices`, `_self_last_permutation`),
asserts the coherent component of the difference is < 2x the noise floor, each order is < 2x
the ideal bf16 row gain, and the gradient rel-L2 vs fp64 is order-independent within 5%.
Forcing `rescale_threshold = 8` fails it with a 7-10x margin (first-vs-last row gain 1.3e-3
vs floor 1.8e-4). Runs in fake and real mode.

## Alternative considered

Accumulating `row_sum` over the bf16-rounded `p` (the values the MMA consumes) makes the
normalization exact for whatever was rounded, so the output row gain also drops to the
floor, and it turned out ~3% *faster* (the fp32 exp2 tile dies at the convert, so the
softmax warps spill less: local memory 184 -> 96 B/thread). It was not adopted here because
it changes what `lse`/`row_sum` mean (log of the sum of rounded probabilities), would apply
to the inference and fp8 paths too, and does not make the backward's fp32 P agree with a
lazily-scaled dominant p. It remains a candidate follow-up on top of the exact max.
