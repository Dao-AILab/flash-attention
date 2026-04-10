# Analyzing SASS for HGMMA Instructions

## Dumping SASS

```bash
# Compile with cubin output
CUTE_DSL_KEEP_CUBIN=1 python -c "..."

# Find the cubin (saved in cwd with long name)
ls *.cubin

# Disassemble and extract HGMMA instructions
nvdisasm kernel.sm_90a.cubin | grep "HGMMA\."
```

## Reading HGMMA Instructions

Each `HGMMA.MxNxK.F32.BF16` instruction is one warpgroup-level MMA:
- **M**: always 64 (one warpgroup = 128 threads, fixed by hardware)
- **N**: output columns per instruction (e.g., 96, 128, 192)
- **K**: always 16 for BF16 (one K-step per instruction)

Key fields in the instruction:
```
HGMMA.64x96x16.F32.BF16 R72, gdesc[UR16], RZ, ...
                         ^^^  ^^^^^^^^^^^
                         |    operand A (see RS vs SS below)
                         destination register = accumulator
```

## RS vs SS: Reading the Source Operand

The **2nd operand** (operand A) tells you whether the MMA reads from shared memory (SS) or registers (RS):

- **`gdesc[UR..]`** — smem descriptor → **SS mode** (both A and B from shared memory)
- **`R<N>`** (plain register) — → **RS mode** (A from registers, B from shared memory)

`gdesc[UR..].tnspA.tnspB` means SS with transposed operands (used for GEMMs like dV = P.T @ dO where inputs are transposed views of smem).

### When RS is useful

RS reduces smem traffic. If the data is already in registers from a previous computation (e.g., dS from the softmax backward), RS feeds it directly to the next GEMM.

Example — dQ = dS @ K with `mma_dq_is_rs=True`:
```
HGMMA.64x192x16.F32.BF16 R24, R232, gdesc[UR4], ...    # RS: A=R232 (dS in regs), B=gdesc (K in smem)
```
vs without RS:
```
HGMMA.64x192x16.F32.BF16 R24, gdesc[UR16], gdesc[UR4], ...   # SS: both from smem
```

## Identifying GEMMs from SASS

1. **Same dest register** across consecutive HGMMA instructions = accumulating into the same output tile (iterating over K dimension)
2. **Count of instructions with same dest register** × 16 = **reduction (K) dimension**
3. **Different dest registers** in an interleaved pattern = a large MMA split into multiple 64-row parts (each part has M=64, the full M = num_parts × 64)

### Example: `dK = dS.T @ Q` with shape 192×96, K=64

The 192-row output is split into 3 parts of 64 rows each. SASS shows:
```
HGMMA.64x96x16 dst=R120   # part 0, K-step 0
HGMMA.64x96x16 dst=R72    # part 1, K-step 0
HGMMA.64x96x16 dst=R24    # part 2, K-step 0
HGMMA.64x96x16 dst=R120   # part 0, K-step 1
HGMMA.64x96x16 dst=R72    # part 1, K-step 1
HGMMA.64x96x16 dst=R24    # part 2, K-step 1
...  (4 K-steps total)
```
- 3 accumulators (R120, R72, R24) → M = 3 × 64 = 192
- 4 instructions per accumulator → K = 4 × 16 = 64

## Case Study: BWD SM90, hdim=192, hdim_v=128, tile_m=64, tile_n=112

Config: `SdP_WGs=[0], dQ_WGs=[0], dK_WGs=[1], dV_WGs=[0]`, `mma_dq_is_rs=True`
- WG0: S, dP, dV, dQ (256 regs)
- WG1: dK only (224 regs)

### SASS HGMMA breakdown

```
#1-12   64x112x16  dst=R24              src=gdesc[UR..]           SS  ×12  →  S = Q @ K.T
#13-20  64x112x16  dst=R24              src=gdesc[UR..]           SS  ×8   →  dP = dO @ V.T
#21-28  64x112x16  dst=R176/R120 alt    src=gdesc[UR..].tnsp      SS  ×4ea →  dV = P.T @ dO
#29-35  64x192x16  dst=R24              src=R232..R248            RS  ×7   →  dQ = dS @ K
#36-47  64x112x16  dst=R136/R80/R24 cyc src=gdesc[UR8].tnsp       SS  ×4ea →  dK = dS.T @ Q
```

Verification:

| GEMM | Atom | # Acc | I/acc | M | N | K | RS/SS | Check |
|------|------|-------|-------|---|---|---|-------|-------|
| S = Q @ K.T | 64×112×16 | 1 | 12 | 64 | 112 | 12×16=192=hdim | SS | ✓ |
| dP = dO @ V.T | 64×112×16 | 1 | 8 | 64 | 112 | 8×16=128=hdim_v | SS | ✓ |
| dV = P.T @ dO | 64×112×16 | 2 | 4 | 2×64=128=hdim_v | 112 | 4×16=64=tile_m | SS | ✓ |
| dQ = dS @ K | 64×192×16 | 1 | 7 | 64 | 192=hdim | 7×16=112=tile_n | **RS** | ✓ |
| dK = dS.T @ Q | 64×112×16 | 3 | 4 | 3×64=192=hdim | 112 | 4×16=64=tile_m | SS | ✓ |

Total: 47 HGMMA instructions (40 × 64×112×16 + 7 × 64×192×16).

dQ uses RS because `mma_dq_is_rs=True`: dS is computed in registers by the SdP pointwise
(P * (dP - dPsum)) and fed directly to the dQ GEMM without writing to shared memory first.
