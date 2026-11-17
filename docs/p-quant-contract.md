# P quant-dequant contract

Normative, finite-precision contract for fake quantization of attention probabilities in FA4's SM100 forward. The kernel, the format table (`flash_attn/cute/p_quant.py`) and the test oracles (`tests/cute/p_quant_oracle.py`) implement this document.

Notation: `R(.)` is FP32 round-to-nearest, ties-to-even. `satRN8` / `satRN4` are the native conversions to E4M3 / E2M1 (round-to-nearest-even with finite saturation). "Exact division" means the correctly rounded FP32 division, not a reciprocal multiply.

## 1. What is quantized

- **Ideal quantity.** The unnormalized, running-max-relative probability `p = exp2(s - m)`, where `s` is the scaled score of a key and `m` is the row's running max when the key's tile is processed. With rescale threshold 0 the max is adopted whenever a tile raises it. The max is local to one kernel call, so to one SplitKV split and one context-parallel chunk.
- **Codec input.** The FP32 value `x` the kernel computes for `p * 2**max_offset`. It is not pre-rounded to the V dtype. `x` has no clip at its ideal maximum: the bias path rounds `row_max * scale_log2` separately and then uses a fused multiply-add, so `x` can exceed `2**max_offset` by a rounding residual. The saturating conversions absorb that.
- **Consumers.** Only the P@V operand uses the codec output. Row sum, LSE, sink fold, correction rescale and empty-row handling read the unquantized FP32 buffer.

## 2. Formats

| Name | Codec | Block | `max_offset` | `rescale_threshold` | Ideal input range |
|---|---|---|---|---|---|
| `fp8_e4m3` | E4M3 | 1 | 8 | 0 | [0, 256] |
| `nvfp4` | E2M1 codes, one E4M3 scale per block | 16 | 0 | 0 | [0, 1] |
| `identity_fp8_e4m3` | none (control) | 1 | 8 | 0 | [0, 256] |
| `identity_nvfp4` | none (control) | 16 | 0 | 0 | [0, 1] |

`max_offset` is an integer. A format whose scaled probability can exceed its input range (`2**(max_offset + rescale_threshold) > input_max`) is rejected when it is constructed.

### 2.1 `fp8_e4m3`

1. Receive FP32 `x` (already in the `256 p` domain).
2. `y = decode8(satRN8(x))`, exact in FP32.
3. Convert `y` to the V dtype with the existing conversion (exact for BF16).

No multiply or divide by 256 happens in the codec. The row sum carries the same factor, so it cancels in normalization, and the correction side subtracts `max_offset` from LSE. Inputs below half the smallest E4M3 subnormal (`x <= 2**-10`) become 0 while the row sum keeps them (the "P floor").

### 2.2 `nvfp4` (unit domain)

Per 16 consecutive keys, aligned to key index mod 16 counted from the first key of the kernel call's key sequence. Masked and out-of-range keys are exactly 0.

1. `b = max_i x_i`.
2. `t = R(448 * b)`.
3. `sigma = decode8(satRN8(t))`.
4. If `sigma == 0`: all sixteen operands are +0. The kernel gets there by substituting an effective scale of 1 in steps 5 to 8: every input of such a block is at most `0x1.24924ap-19`, which rounds to code 0, and no division by zero occurs. The oracle returns the zero block directly.
5. `e = R(sigma / 2688)` (or 1 when `sigma == 0`).
6. `z_i = R(x_i / e)` for each key.
7. `c_i = decode4(satRN4(z_i))`.
8. `y_i = R(c_i * e)`.
9. Convert `y_i` to the V dtype with the existing round-to-nearest-even conversion.

Every FP32 intermediate is materialized. No reciprocal multiplication, reassociation or fused replacement is allowed inside the codec. The global amax is fixed at 1. The zero-scale condition is `R(448 b) <= 2**-10`; the largest FP32 block max that satisfies it is `0x1.24924ap-19` (about 2.18e-6).

### 2.3 Native conversion semantics

- **E4M3:** bias 7, three fraction bits, smallest subnormal `2**-9`, smallest normal `2**-6`, maximum 448. Ties go to the even significand. Finite overflow and infinities saturate to +-448. Signed zero is preserved. NaN stays NaN.
- **E2M1:** magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6. Decision boundaries 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5, with tie results 0, 1, 1, 2, 2, 4, 4. Saturates to +-6. NaN maps to +6, so qualification rejects NaN inputs instead of relying on the conversion to surface them.
- Decoding through FP16 to FP32 is exact for both.

Source for conversion semantics: the PTX `cvt` specification. They are confirmed on sm_100 by the device codec probe (`tests/cute/p_quant_codec_probe.py`).

## 3. Masks and empty rows

- Masked keys (score -inf) are exactly 0 and never raise a block max.
- A row with no valid key has zero probabilities, `O = 0`, `LSE = -inf` and zero Q/K/V gradients. `-inf` LSE there is expected.
- Padded storage is excluded from comparisons.

## 4. Surrogate gradient

The backward is the unmodified SM100 backward (`FlashAttentionBackwardPreprocess`, `FlashAttentionBackwardSm100`, `FlashAttentionBackwardPostprocess`) evaluated on the saved inputs, where `O_q` is the quantized forward output and `L` its natural-log LSE. With `alpha = softmax_scale`, `G = dO`, `g = H_q / H_kv` and `a = floor(h / g)`:

```
S_ij  = alpha * Q_ih . K_ja
P_ij  = exp(S_ij - L_ih) for a valid key in a nonempty row, else 0
D_ih  = sum_c G_ihc * O_q,ihc - dL_ih          (dL = 0 when absent)
dP_ij = G_ih . V_ja
dS_ij = P_ij * (dP_ij - D_ih)
dQ_ih = alpha * sum_j dS_ij K_ja
dK_ja = alpha * sum_{h in group a} sum_i dS_ij Q_ih
dV_ja =         sum_{h in group a} sum_i P_ij G_ih
```

- No gradient flows through scales, maxima or rounding. "Unquantized" excludes the P quant-dequant, not BF16 rounding: the backward still converts P to BF16 for dV and dS for dQ/dK.
- Masks: with effective lengths `m, n` and `c_i = i + n - m`, causal permits `j <= c_i`; a local window permits `c_i - w_l <= j <= c_i + w_r`.
- The public backward discards `dL` when `return_lse=False`.
- The deterministic flag keeps the formulas but changes accumulation order and config selection, so its results are not bit-identical to the non-deterministic backward.

Backward options that read forward state other than `O` and `LSE`, and are therefore refused together with a format: sparse or gather KV (saved P, `row_max`, `o_lo`), `gather_bwd_recompute_p`, block sparsity, score/mask modifiers and softcap (their chain rule is not part of this contract), and the learnable sink in the first version. Backward-only options are refused too: `score_mod_bwd` and `block_sparse_tensors_bwd` on the autograd entry points, since they would change the gradient even when the forward is plain dense attention.

Autotuning (`flash_attn.cute.autotune.tune_fwd_config`) is refused with a format: it validates candidate configs against the default config's output, which is not a valid comparison for a schedule-dependent result.

Enabling a format adds no backward specialization: the three backward caches (`bwd_pre`, `bwd`, `bwd_post`) must hold the same keys as a warmed `none` baseline with the same metadata, deterministic setting and dLSE presence.

## 5. Partition dependence and the visit schedule

Results depend on the order and grouping in which a row sees its keys. This is defined behaviour. The schedule's inputs are:

- `tile_n`;
- the effective query tile `q_stage * tile_m * cta_group_size` and the query-block mapping;
- the packed-GQA row mapping;
- causal and window bounds;
- traversal order (last KV tile first);
- per-sequence split counts or a fixed split length;
- context-parallel chunking.

Tests compare against an oracle given the same schedule. Drift across partitions is not gated.

## 6. Identity controls and the LSE gate

For each scaling policy an identity-codec control format exists with the same policy. For the same inputs and config, LSE of a format must be bit-identical to its control's LSE. A mismatch is a bug in the kernel or in this document, not a tolerance to widen. LSE is not claimed bit-identical to stock, because the rescale threshold changes from 8 to 0 for 16-bit V.

## 7. Error contract

### 7.1 Path bound `eps_path`

For BF16 inputs taken as exact, key `j`, head dimension `h`, `u = 2**-24`, `gamma_n(v) = n v / (1 - n v)`:

```
d_j   = sum_l q_l k_jl                 S_j = sum_l |q_l k_jl|
E_d_j = gamma_h(2u) * S_j + U_TC       (tensor-core accumulation envelope)
```

`U_TC = h * 2**-126` covers products flushed to zero. This envelope is an assumption: an FP32 accumulator does not by itself prove one-ulp-per-add accumulation. The tests on sm_100 pass under it.

The running max is uncertain too, and it may come from an earlier tile. Within one kernel call (one split or chunk) the oracle keeps the interval `[max_j (d_j - E_d_j), max_j (d_j + E_d_j)]` over every key seen so far, and `E_m` is the distance from the nominal max to the farther end of that interval. Taking `E_m` from the current tile alone understates it when an earlier, less well-conditioned key holds the max.

The kernel computes, with `a` the FP32 `scale_log2`, `C = max_offset`, `m_hat` the FP32 running max:

```
t = R(m_hat * a);  B = R(C - t);  z_hat_j = FMA32(d_hat_j, a, B)
E_z_j <= |a| (E_d_j + E_m) + |d_j - m| E_a + E_mul + E_sub + E_FMA
eps_path_j <= x*_j (2**E_z_j - 1) + sup E_exp(z)        with x*_j = 2**z_j
```

- Hardware `ex2.approx`: two ulps from the correctly rounded result, plus half an ulp against the exact value.
- Emulated pairs (`ex2_emulation_2`): the implementation-path oracle reproduces the polynomial operation for operation, so it carries no approximation term there; the mathematical path uses the exact exp2 and adds a relative bound of 9e-5 for those columns (cubic `1 + 0.695146143436431884765625 f + 0.227564394474029541015625 f**2 + 0.077119089663028717041015625 f**3`, measured extremum about 8.763e-5). `2**-126` absolute error is added to every element for results below the normal range.
- Overshoot: `x_j - 2**C <= 2**C (2**H (1 + rho) - 1) + eta` with `H = E_mul + E_sub + E_FMA` and `(rho, eta)` the exp2 path bound.

Source-verified: the operation structure and the polynomial coefficients. Assumed and exercised by the sm_100 tests: the tensor-core envelope, lowering and flush-to-zero behaviour.

### 7.2 Flip eligibility (block-aware)

Each codec input gets an outward interval `X_i = x*_i +- delta_i`, with `delta_i = 2 eps_path_i + 4 ulp32(x*_i)`. Masked inputs stay exactly 0.

- **FP8:** enumerate every E4M3 rounding cell that intersects `X_i`.
- **NVFP4:** form the block-max interval from all 16 inputs; propagate it through `R(448 B)` and enumerate every reachable scale code from the lowest to the highest, including zero (a wide interval can reach more than two); for each nonzero scale propagate division, E2M1 conversion, reconstruction and BF16 rounding; take the union of reachable operands. More than one reachable scale code marks every key of the block eligible. A code or BF16 boundary crossing marks that key eligible.

### 7.3 Allowance

With `a_i = max |y - y_i_oracle|` over the reachable operands of key `i`, `r_i` the product of the running-max rescale factors applied after its tile, `D_s` the final unquantized denominator of split or chunk `s`, and `w_s = exp(LSE_s - LSE_merged)`:

```
A_r    = sum_s w_s sum_{i in s} r_i a_i / D_s                 (probability allowance of row r)
A_r,d  = sum_s w_s sum_{i in s} r_i a_i |V_i,d| / D_s         (output allowance)
```

`A_r,d <= A_r max_i |V_i,d|`. The FP8 factor 256 cancels through `D_s`.

### 7.4 Implementation-path tolerance

The comparison between the all-FP32 mathematical oracle and the kernel allows for: the smooth input-path error on the denominator; row-sum reduction order (rescale of the old sum, then the interleaved accumulators and final tree); online numerator rescaling in TMEM and P@V accumulation; BF16 rounding of the operand; the approximate reciprocal and the final multiply in normalization; BF16 output rounding; and, for SplitKV and context parallel, partial-LSE error, merge weights and the merge's own rounding. For numerator `N` and denominator `D > E_D`: `E_quotient <= (E_N + |N / D| E_D) / (D - E_D)`.

In tests this is realized with the repository's convention: the kernel must be within `rtol * max|out_pt - out_ref| + atol` of the reference, where `out_pt` is the implementation-path oracle and `out_ref` the mathematical oracle, both for the same schedule and format.

## 8. Budgets

| Budget | Value | Applies to |
|---|---|---|
| Interval margin | `delta_i = 2 eps_path_i + 4 ulp32(x*_i)` | every codec input |
| Row check | every row is compared, with its own output allowance `A_r,d` added to the tolerance; no row is excluded | random-input tests |
| Largest probability allowance of any row | `A_r <= 2**-3` | random-input tests |
| Share of rows with `A_r > 2**-5` | at most 5% per predeclared configuration | random-input tests |
| Share of rows with `A_r > 2**-8` (BF16 unit roundoff) | at most 30% per predeclared configuration, reported | random-input tests |
| Allowance permitted | 0 | constructed witnesses, P-readout probes, masks, empty rows, LSE identity gate |

Exceeding a budget fails the test. The budgets were fixed before the first run on hardware.

Why the budgets are this loose: under the conservative tensor-core envelope of 7.1, about half of the rows of a random BF16 problem at head dimension 128 have `A_r > 1e-3`, and rows with few visible keys (the first rows of a causal or windowed problem) reach `A_r` of 0.03 to 0.06, because one flipped key is a large share of such a row. The driver is the assumed accumulation envelope `gamma_h(2u) S_j`, not the exp2 emulation. Every row stays under test; the budgets bound how many rows are checked loosely.

## 9. Support matrix

Supported: sm_100; BF16 q/k/v; `head_dim == head_dim_v` in {64, 96, 128}; dense and varlen; unequal Q/K lengths; MHA/GQA/MQA with and without pack-GQA; non-causal, causal and local window; SplitKV; with and without gradients; deterministic backward on and off.

Refused at forward entry, before the zero-length early return, with an error naming the feature: other architectures; FP16; FP8/E5M2 inputs and `TensorwiseScales`; MXFP8 `BlockScales`; MLA (`qv`); sparse or gather KV; block sparsity; `score_mod`, `mask_mod`, softcap; paged KV; `learnable_sink`; other head dimensions; an explicit SwapAB request. Automatic selection never picks SwapAB when a format is set.

2-CTA applicability (from `flash_attn/cute/config.py`, not a choice of this work): the selector admits 2-CTA kernels at head dimension up to 128 only for non-causal, non-local, unsplit calls, and rejects SplitKV with 2-CTA. So 2-CTA is qualified on the unsplit non-causal head-dimension-128 case, and every causal, windowed or SplitKV case is 1-CTA; an explicit config that asks for 2-CTA with any of those is rejected.

Architecture evidence: sm_100 is executed (`tests/cute/test_p_quant.py` passes on an NVIDIA B200, compute capability 10.0, with no kernel compiled in the execution pass). sm_103 and sm_107 compile only: they are not executed and not enabled.
