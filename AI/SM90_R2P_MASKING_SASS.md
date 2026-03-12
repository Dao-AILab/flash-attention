# SM90 FWD R2P Masking — SASS Investigation

## SASS Instruction Counts (hdim=128, seqlen=113, tile_n=128)

With tile_n=128, SM90 has 32 accumulator elements per row (1 chunk of 32).

### Non-causal (seqlen-only masking)

| Metric | Old (no R2P) | New (R2P) | Delta |
|--------|-------------|-----------|-------|
| **Total instructions** | 3104 | 3072 | **-32 (-1%)** |
| R2P | 0 | 4 | +4 |
| FSEL | 70 | 70 | 0 |
| ISETP | 55 | 22 | **-33** |
| SHF | 69 | 73 | +4 |
| LOP3 | 51 | 56 | +5 |

R2P replaces 33 ISETP (integer set-predicate) instructions with 4 R2P + a few LOP3/SHF. Net savings: 32 instructions. The 4 R2P instructions each convert one byte of a 32-bit bitmask into 7 predicates, covering all 32 elements (4 × 8 bits = 32).

### Causal

| Metric | Old (no R2P) | New (R2P) | Delta |
|--------|-------------|-----------|-------|
| **Total instructions** | 5008 | 4857 | **-151 (-3%)** |
| R2P | 0 | 24 | +24 |
| FSEL | 200 | 200 | 0 |
| ISETP | 225 | 22 | **-203** |
| SHF | 104 | 105 | +1 |
| LOP3 | 81 | 105 | +24 |

Much larger savings. The causal kernel applies masking per-row (each row has a different col_limit), so it has many more masking operations. 24 R2P instructions replace 203 ISETP instructions, saving 151 total.

### Local (sliding window, wl=64 wr=0)

| Metric | Old (no R2P) | New (R2P) | Delta |
|--------|-------------|-----------|-------|
| **Total instructions** | 7296 | 6217 | **-1079 (-15%)** |
| R2P | 0 | 32 | +32 |
| FSEL | 522 | 266 | **-256** |
| ISETP | 554 | 22 | **-532** |
| SHF | 115 | 73 | -42 |
| LOP3 | 96 | 56 | -40 |

Dramatic savings. Local masking has two bounds (left + right) per row, doubling the masking work. R2P eliminates 532 ISETP and 256 FSEL instructions, saving 1079 total (15% of kernel).

## How R2P Works in SASS

The compiler generates this pattern:

```
SHF.R.U32.HI R9, RZ, R9, R16    ; shift to create bitmask
R2P PR, R9, 0x7f                  ; byte 0 → predicates P0-P6
FSEL R15, R36, -INF, P6           ; apply P6: keep or mask to -inf
R2P PR, R9.B1, 0x7f              ; byte 1 → predicates P0-P6
FSEL R52, R52, -INF, P6           ; apply P6
R2P PR, R9.B2, 0x7f              ; byte 2
...
R2P PR, R9.B3, 0x7f              ; byte 3
```

Each `R2P` converts 7 bits of a register byte into 7 predicate registers simultaneously (1 instruction instead of 7 `ISETP`). The subsequent `FSEL` instructions use these predicates for conditional masking.

### Handling the leftover bits (32 is not divisible by 7)

The `0x7f` immediate tells R2P to map bits 0-6 of each byte to P0-P6, but bit 7 (the MSB of each byte) is not covered. For 32 elements across 4 bytes, that's 4 leftover elements (bits 7, 15, 23, 31). The compiler handles these with separate `LOP3.LUT` or `ISETP` instructions:

```
R2P PR, R12,     0x7f           ; bits 0-6   → P0-P6  (7 elements)
  14× FSEL using P0-P6           ; apply to 7 cols × 2 rows
LOP3.LUT P0, RZ, R12, 0x80, ... ; test bit 7  (1 element)
  2× FSEL using P0

R2P PR, R12.B1,  0x7f           ; bits 8-14  → P0-P6  (7 elements)
  14× FSEL using P0-P6
LOP3.LUT P1, RZ, R12, 0x8000, ..; test bit 15 (1 element)
  2× FSEL using P1

R2P PR, R12.B2,  0x7f           ; bits 16-22 → P0-P6  (7 elements)
  14× FSEL using P0-P6
LOP3.LUT P0, RZ, R12, 0x800000,..; test bit 23 (1 element)
  2× FSEL using P0

R2P PR, R12.B3,  0x7f           ; bits 24-30 → P0-P6  (7 elements)
  14× FSEL using P0-P6
ISETP.GT P0, R12, -1            ; test bit 31 (sign bit) (1 element)
  2× FSEL using P0
```

Total: 4×7 = 28 elements via R2P + 4 elements via LOP3/ISETP = 32. Each R2P replaces 7 ISETP with 1 instruction, so net savings is `(7-1) × 4 = 24` predicate-generation instructions per mask application. Additionally, ptxas can overlap R2P with FSEL since they write to separate predicate registers.

## Performance Impact

| Case | Old (ms) | New (ms) | Speedup |
|------|----------|----------|---------|
| Causal hdim=64 s=8192 | 2.463 | 2.473 | ~0% |
| Causal hdim=128 s=8192 | 1.937 | 1.944 | ~0% |
| Local hdim=64 s=8192 | 0.394 | 0.346 | **+14%** |
| Local hdim=128 s=8192 | 0.237 | 0.222 | **+7%** |
| Non-causal hdim=128 s=4096 | 1.742 | 1.728 | ~1% |

Causal sees no perf gain despite fewer instructions because masking is a tiny fraction of total work (dominated by WGMMA). Local sees significant gains because the sliding window has many partially-masked blocks where masking overhead matters more.
