# Debugging GPU Kernel Hangs (Deadlocks) in CUTLASS DSL / 2CTA Kernels

## General Approach to Debugging Kernel Hangs

### Step 1: Build a minimal repro

Strip the test case down to the smallest input that triggers the hang:
- batch=1, nheads=1, smallest seqlen that hangs
- Single config, no loops, no benchmarking
- Add a timeout or run with `compute-sanitizer` so you can distinguish a hang from slow execution

### Step 2: Add printf to locate the hang

GPU `printf` (`cute.printf`) is the primary tool. The goal is binary search: narrow down which warp and which operation is blocked.

**Printf guards** — avoid print storms:
```python
# One thread per warp:
if cute.arch.thread_idx()[0] % 32 == 0:
    cute.printf("...")

# One thread per CTA (elect_one is a context manager, not a bool):
with cute.arch.elect_one():
    cute.printf("...")

# One specific thread:
if tidx == 0:
    cute.printf("...")
```

**Strategy — coarse to fine:**
1. First, print at the entry/exit of each warp's main function (load, mma, softmax, correction). This tells you which warp is stuck.
2. Then add prints before/after each pipeline wait (`consumer_wait`, `producer_acquire`). This tells you which barrier is stuck.
3. Then print the barrier index, phase, and stage to understand the pipeline state.

**What to print:**
- CTA index (`cute.arch.block_idx()[0]`) — critical for multi-CTA debugging
- Pipeline stage index and phase
- Loop iteration count
- Whether a `try_wait` succeeds or fails (use `try_wait_token` parameter)

### Step 3: Identify the deadlock chain

A hang is always a cycle. Typical chain in a pipelined kernel:

```
MMA waiting for K from load (pipeline_kv full barrier)
  -> Load finished but stuck in producer_tail (waiting for MMA to release empty barrier)
    -> MMA can't release because it's waiting for K
```

Once you see which barrier is stuck, trace backwards: who is supposed to signal it, and why haven't they?

### Step 4: Vary the problem size systematically

Test with different sequence lengths / block counts to find the pattern:

| seqlen | n_blocks | Result |
|--------|----------|--------|
| 128    | 1        | ?      |
| 256    | 2        | ?      |
| 384    | 3        | ?      |
| 512    | 4        | ?      |

If the hang correlates with the number of visits to a pipeline stage (e.g., works for n_blocks <= kv_stages but fails when stages wrap around), the problem is likely in barrier tx_count or phase tracking.

### Step 5: Check barrier byte counts (tx_count)

For TMA-based pipelines, `arrive_and_expect_tx` sets the expected transaction byte count on an mbarrier. If the expected count doesn't match the actual bytes arriving, the barrier either:
- Fires too early (expected < actual) — causes data races
- Never fires (expected > actual) — causes hangs

In **2CTA / cluster mode**, both CTAs' TMAs signal the **same** cluster-level mbarrier. If each CTA's TMA contributes N bytes, the barrier receives 2N bytes total. The tx_count must be `N * cta_group_size`, not just `N`.

**All TMA pipelines need doubling** — Q, K, and V. Even though each CTA loads a different M-tile for Q, both CTAs' TMA operations still signal the same cluster-level barrier, so the expected byte count must account for both.

### Step 6: Check phase / parity tracking

`mbarrier_try_wait_parity` uses a single parity bit (0 or 1). If your pipeline state tracks phase as a monotonically increasing counter (0, 1, 2, 3, ...), you need `phase % 2` before passing it to the barrier wait. Without this, phase=2 looks like phase=0 to the hardware, which can cause waits on already-completed barriers or misses on pending ones.

### Step 7: Beware compiler-as-bug-source

If the kernel works WITH printf but hangs WITHOUT it, the printf is acting as a **compiler barrier**. The MLIR/LLVM backend cannot optimize through an opaque function call like printf, which prevents harmful instruction reordering.

Signs this is happening:
- A single `cute.printf("\n")` in the right function fixes the hang
- PTX fences (`fence_view_async_shared`, `fence_acq_rel_cluster`, `sync_warp`, `fence_proxy`) do NOT fix it — these affect hardware memory ordering, not compiler scheduling
- The fix is location-sensitive (printf in one function fixes it, in another doesn't)

Possible workarounds:
- `@dsl_user_op` decorator on pipeline methods to make them opaque to the compiler
- `asm volatile` barriers (if available in the DSL)
- Compare generated PTX/SASS with and without printf to identify what the compiler is reordering
- File a bug against the CUTLASS DSL / MLIR pipeline

---

## 2CTA-Specific Pitfalls

### tcgen05.commit with empty commit groups

`tcgen05.commit(mbar, mask, cta_group::2)` is supposed to signal an mbarrier after all pending MMA operations complete. But if there are **no pending operations** (empty commit group), the signal only reaches the local CTA's barrier, not the remote CTA's. Fix: use explicit `mbarrier_arrive(barrier, dst_cta_rank)` to both CTAs.

### producer_tail deadlock

The default `producer_tail` (inherited from sm90 pipelines) drains the pipeline by calling `producer_acquire` in a loop. In 2CTA mode this deadlocks because the consumer (MMA warp) may have already exited without releasing all stages. Fix: make `producer_tail` a no-op for 2CTA.

### Tile scheduler must account for cluster shape

Both CTAs in a cluster must get the **same** tile coordinate. Raw `blockIdx.x` assigns consecutive values to CTAs in the same cluster. Fix: divide `blockIdx.x` by `cluster_shape_m`.

### Cross-CTA vs per-CTA pipelines

Pipelines where CTA 1's threads remotely arrive on CTA 0's barriers need cluster-sized cooperative group counts. Pipelines that are purely local to each CTA keep per-CTA counts.

### Softmax masking offset

Causal mask row positions must account for the CTA's position within the cluster. Multiply `m_block` by `cta_group_size` when computing mask coordinates.
