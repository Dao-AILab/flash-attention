# CuTe DSL Dropout Implementation Notes

Issues encountered and resolved while implementing dropout for FlashAttention CuTe DSL kernels on SM121a (DGX Spark, NVIDIA GB10).

## Pre-existing bugs fixed

### 1. `use_tma_O` Arch enum comparison (`flash_fwd.py`)

**Problem**: `self.use_tma_O = self.arch >= Arch.sm_90` — SM120 > SM90 in the enum ordering, causing the kernel to use TMA output store which SM120 doesn't support. SM120 uses CpAsync (like SM80), not TMA.

**Fix**: `self.use_tma_O = self.arch >= Arch.sm_90 and self.arch < Arch.sm_120`

**Root cause**: `FlashAttentionForwardSm120` sets `arch = 80` as a class attribute, but `FlashAttentionForwardBase.__init__` overwrites `self.arch` with `BaseDSL._get_dsl().get_arch_enum()` at line 117. The class attribute is never used.

### 2. Missing `dQ_single_wg` for SM120 backward (`interface.py`)

**Problem**: SM120 backward config path didn't set `dQ_single_wg`, referenced later for postprocessing thread count. `UnboundLocalError` on any backward pass through SM120.

**Fix**: Added `dQ_single_wg = False` to SM120 config block.

## CuTe DSL compatibility issues

### 3. `cutlass.cast()` does not exist

**Problem**: Used in SM90 forward for `cutlass.cast(val, Uint32)` type conversion. Function doesn't exist in nvidia-cutlass-dsl 4.x.

**Fix**: Replaced all `cutlass.cast(x, T)` with `T(x)` constructor calls across `dropout.py`, `flash_fwd.py`, `flash_bwd.py`, `flash_bwd_sm90.py`.

### 4. LLVM inline PTX assembly incompatible with MLIR backend

**Problem**: Original Philox used `llvm.inline_asm` with PTX (`mul.wide.u32`, predicated `selp.b32`). Produces ptxas syntax errors:
```
ptxas fatal: Parsing error near '.reg': syntax error
```
The `{` brace syntax for PTX register grouping conflicts with LLVM MLIR's inline asm parser.

**Fix**: Rewrote all functions as pure DSL operations:
- `mul_wide_u32`: `Uint64(a) * Uint64(b)` + shift/truncate
- `_extract_random_byte`: Branchless mask arithmetic
- Changed decorators from `@dsl_user_op` to `@cute.jit`

**Note**: On SM90 where inline PTX works, a future optimization could revert `mul_wide_u32` to the PTX version for better performance (native `mul.wide.u32` is single-cycle vs multi-instruction Uint64 emulation).

### 5. `if`-statement mutation only works in `@cute.jit`, not `@cute.kernel`

**Root cause investigation**: The CuTe DSL AST preprocessor (`ast_preprocessor.py:visit_If`) transforms `if` statements with runtime conditions into `scf.IfOp` with proper `scf.yield` for modified variables. However, this preprocessing only runs on `@cute.jit` functions, not `@cute.kernel` functions.

All dropout mask application code is in `apply_dropout_mask()` which is `@cute.jit`, so this works correctly. Initial debugging failures occurred because test code was written in `@cute.kernel` context.

The Philox group cache uses `if (row_group != cache_rg) | (col_group != cache_cg):` to conditionally call Philox only when the group changes, matching FA2 C++ efficiency.

### 6. Scalar-to-global-memory tensor store is unsupported

**Problem**: `out[0] = Int32(val)` on a global memory tensor fails with "unsupported operation". This is a known limitation — scalar stores to global memory require TMA or gmem copy operations, not direct assignment.

**Impact**: Test harness code that writes verification values to global memory needs to use different patterns (e.g., fragment → smem → gmem pipeline).

## Architecture coverage

| Arch | Forward | Backward | Tested |
|------|---------|----------|--------|
| SM80 | ✅ | ✅ | Via SM120 inheritance |
| SM90 | ✅ | ✅ | Compilation only (no SM90 hardware available) |
| SM100/SM110 | ❌ `NotImplementedError` | ❌ `NotImplementedError` | N/A — standalone kernel, needs separate integration |
| SM120 | ✅ (inherits SM80) | ✅ (inherits SM80) | ✅ Full end-to-end on SM121a |

## Performance

The implementation uses full 10-round Philox 4x32-10 with 4×4 group batching (1 Philox call per 16 elements), matching FA2 C++ exactly.

**SM120 (tested)**: Overhead is higher than FA2 target because SM120 emulates `mul.wide.u32` via multi-instruction Uint64 operations. The Uint64 multiply is the bottleneck.

| Problem size | No dropout | With dropout | Overhead |
|---|---|---|---|
| B=1, S=2048, H=8, D=128 | 0.22 ms | 0.78 ms | +249% |
| B=4, S=512, H=16, D=64 | 0.07 ms | 0.74 ms | +1017% |
| B=8, S=256, H=32, D=64 | 0.19 ms | 0.83 ms | +325% |

**SM90 (expected)**: SM90 supports native `mul.wide.u32` PTX instruction (single-cycle). Combined with the group cache (16× fewer Philox calls), expected overhead is ~10%, matching FA2 C++. A future optimization can switch `mul_wide_u32` to inline PTX on SM90 for this improvement.

## Validated on

- NVIDIA GB10 (SM121a), DGX Spark
- CUDA 13.0, Driver 580.126.09
- nvidia-cutlass-dsl 4.4.1+, PyTorch 2.10.0+cu130

## Test results (SM121a)

All tests pass:
1. Forward without dropout (baseline) ✅
2. Forward with `dropout_p=0.0` matches baseline (bit-exact) ✅
3. Forward with `dropout_p=0.5` changes output ✅
4. Same seed → deterministic (bit-exact across runs) ✅
5. Different seeds → different outputs ✅
6. Causal + dropout deterministic ✅
7. Backward with dropout → valid non-zero gradients ✅
8. Causal backward with dropout ✅
9. Dropout scaling preserves expected value ✅
10. SM100 raises `NotImplementedError` ✅
