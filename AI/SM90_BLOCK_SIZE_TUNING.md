# SM90 Block Size Tuning Guide

How to choose tile sizes and MMA configurations for FlashAttention on Hopper (SM90).

## Tool

Use `flash_attn/cute/sm90_config_search.py` to enumerate feasible configs:

```bash
# Both fwd and bwd
python flash_attn/cute/sm90_config_search.py --headdim 128

# Forward only
python flash_attn/cute/sm90_config_search.py --mode fwd --headdim 192-128

# Backward only, custom tile choices
python flash_attn/cute/sm90_config_search.py --mode bwd --headdim 192 --tile-m 64,80 --tile-n 64,96
```

## Hardware Constraints (H100)

- **SMEM**: 228 KB total. We reserve ~3 KB for LSE, dPsum, and mbarriers, leaving **224 KB** for tensor buffers.
- **Registers**: Controlled via `setmaxnreg`. Budget per MMA warp group:
  - 2 WG: 240 regs/thread, minus 24 overhead = **216 usable**
  - 3 WG: 160 regs/thread, minus 32 overhead = **128 usable**
- **GMMA atom**: Always M=64. The effective M dimension (after swap) must be divisible by 64. N dimension must be divisible by `atom_layout_n * 8`.

## Architecture: Warp Groups

Each SM90 backward kernel has `num_wg + 1` warp groups (128 threads each):
- **WG0** (producer): TMA loads for Q, K, V, dO, LSE, dPsum
- **WG1** (producer): dQaccum store (TMA reduce-add to gmem)
- **WG2..WG(num_wg)** (MMA consumers): All GEMMs

For forward: `num_wg` MMA WGs + 1 producer WG. `tile_m = num_wg * 64` (no swap).

## Key Decisions

### 1. Number of Warp Groups (num_wg)

| num_wg | tile_m (fwd) | Threads | Reg budget | Best for |
|--------|-------------|---------|------------|----------|
| 2 | 128 | 384 | 216/thread | hdim <= 128 |
| 3 | 192 | 512 | 128/thread | hdim 129-192 |

More WGs = larger tile_m = better M-direction parallelism, but tighter register budget and higher smem usage.

### 2. swap_AB

Each MMA can optionally swap its A and B operands. This transposes the output tile, exchanging which dimension maps to M (must be divisible by 64) and which maps to N.

**When to swap:**
- If the natural M dimension isn't divisible by 64 but N is (e.g., tile_m=80 for SdP)
- To change which operand is in registers vs shared memory

**Forward**: No swap needed since tile_m = num_wg * 64 is always divisible by 64.

**Backward** (5 MMAs):
- **SdP** (S=Q@K^T, dP=dO@V^T): output (tile_m, tile_n). Swap if tile_m % 64 != 0.
- **dKV** (dK=dS^T@Q, dV=P^T@dO): output (tile_n, hdim/hdimv). Swap if tile_n % 64 != 0 but hdim % 64 == 0.
- **dQ** (dQ=dS@K): output (tile_m, hdim). Swap if tile_m % 64 != 0 but hdim % 64 == 0.

### 3. AtomLayout

The `atom_layout` distributes WGs across the M and N dimensions of an MMA output. With `num_wg` MMA WGs and `atom_layout_m = A`:
- M direction: A warp groups, each handling M/A rows
- N direction: num_wg/A warp groups, each handling N/(num_wg/A) columns

After swap, the atom layout is also swapped.

**Impact on smem traffic**: More WGs in the N direction (`wg_n` larger) means each instruction reads a smaller B slice, but more instructions total read overlapping A slices. Fewer WGs in N (`wg_n` smaller) means fewer instructions but each reads a larger B slice. Typically **smaller wg_n = less total smem traffic**.

### 4. mma_dkv_is_rs (Register-Source for dKV)

When `AtomLayoutMSdP == 1 && AtomLayoutNdKV == num_wg && SdP_swapAB && !dKV_swapAB`, the P and dS matrices can be kept in registers and fed directly as the A operand of dV and dK GEMMs. This:
- **Eliminates sP from smem** (saves tile_m * tile_n * 2 bytes)
- **Eliminates P R2S store** from smem traffic
- **Eliminates A operand reads** for dK and dV GEMMs

This is a significant optimization — always preferred when the conditions are met.

### 5. Pipeline Staging

**Forward**:
- Q: 1 stage (loaded once per n_block tile)
- K, V: 2 stages (double-buffered, pipelined with TMA)
- O: overlaps with Q in smem (reuses same buffer at epilogue)

**Backward**:
- Q: always 2 stages (double-buffered)
- dO: 2 stages if smem allows (matches Q pipeline), else 1 stage
- PdS: 1 stage
- K, V: persistent in smem (loaded once per n_block)

## Register Accounting

Accumulator registers per thread per WG = `M * N / (num_wg * 128)`, where M x N is the output tile.

**Forward peak registers**:
- With WG overlap: `regs_S + regs_P + regs_O` (S, P in bf16, O all live)
- Without overlap: `regs_S + regs_O` (S and O alternate, P reuses S regs)

Where `regs_P = regs_S / 2` (bf16 vs f32).

**Backward peak registers**:
- `max(2 * regs_SdP, regs_dQ) + regs_dK + regs_dV`
- S and dP accumulators are both live (S needed for softmax while dP computes)
- dQ reuses S+dP register space after they're consumed
- dK and dV accumulate across m_block iterations

## SMEM Accounting

Sum of tensor buffers (ignoring alignment padding, which is small):

**Forward**: `max(sQ, sO) + sK*2 + sV*2 + sP`
- sQ = tile_m * hdim * 2
- sK = tile_n * hdim * 2 * 2 stages
- sV = tile_n * hdimv * 2 * 2 stages
- sO = tile_m * hdimv * 2 (overlaps with sQ)
- sP = tile_m * tile_n * 2 (0 if RS)

**Backward**: `sQ*2 + sK + sV + sdO*dO_stage + sP + sdS + sdQaccum`
- sQ = tile_m * hdim * 2 * 2 stages
- sK = tile_n * hdim * 2
- sV = tile_n * hdimv * 2
- sdO = tile_m * hdimv * 2 * dO_stage
- sP = tile_m * tile_n * 2 (0 if mma_dkv_is_rs)
- sdS = tile_m * tile_n * 2
- sdQaccum = tile_m * hdim * 4 (f32)

## SMEM Traffic

Per-iteration smem bandwidth consumed. Each GMMA instruction reads:
- **A operand**: 64 * K_red * 2 bytes (0 if register-source)
- **B operand**: (N_eff / wg_n) * K_red * 2 bytes

Total instructions = (M_eff / 64) * wg_n. Each instruction independently reads A and B from smem.

Additional traffic: R2S stores for P, dS (bf16), dQ smem store + TMA load (f32).

**Traffic per block** (traffic / (tile_m * tile_n)) normalizes across tile sizes for comparison. Lower is better.

## Example Configs

### hdim=128 (Forward)
Best: tile_m=128, tile_n=192, RS, 2 WG. 224K smem, 9.3 tr/blk.

### hdim=128 (Backward, non-causal)
C++ FA3 config: tile_m=80, tile_n=128, SdP_swap=T, dKV_swap=F, dQ_swap=T, aSdP=1, adKV=2. mma_dkv_is_rs=True. 204K smem, 208 regs, 39.6 tr/blk.

### hdim=192 (Backward)
3 WG, tile_m=64, tile_n=96, SdP_swap=F, dKV_swap=T, adKV=1 or 3. 216K smem, 128 regs. This is the only feasible tile_n > 64 for hdim=192 due to register pressure.

### hdim=192, hdimv=128 (DeepSeek shape)
With 3 WG: need AtomLayoutNdKV=3 (since hdimv=128 not divisible by 3). tile_n=96, 212K smem.
With 2 WG: tile_n=112 feasible at 210K smem, or tile_n=64 at 168K smem.
