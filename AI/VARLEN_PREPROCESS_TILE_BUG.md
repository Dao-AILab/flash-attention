# Varlen Preprocess Tile Mismatch Bug

## Summary

`SeqlenInfo.create` in `flash_bwd_preprocess.py` defaulted `tile=128`, but the backward kernel uses `tile_m=m_block_size` (e.g. 64 for causal SM90). This caused the preprocess to zero dq_accum and write lse_log2/dpsum at wrong padded offsets for all batches after batch 0.

## How padded_offset works

For varlen, buffers like dq_accum are laid out with tile-aligned gaps between sequences:

```
padded_offset_q = ((offset_q + batch_idx * tile_m) // tile_m) * tile_m
```

The gap size depends on `tile_m`. With `tile_m=64` vs `tile_m=128`, batch 1 at `offset_q=128` gets:
- tile=64:  padded_offset = ((128 + 64) // 64) * 64  = **192**
- tile=128: padded_offset = ((128 + 128) // 128) * 128 = **256**

The preprocess was zeroing at 256, the backward was writing at 192.

## Symptoms

- Tests pass in isolation (torch.empty gets clean memory)
- Tests fail when run in sequence (CUDA memory caching reuses NaN-polluted memory)
- dq_accum valid positions contain NaN after backward kernel
- `torch.zeros` for dq_accum masks the bug (zeroes everywhere, including the "right" offsets)
- compute-sanitizer shows 0 errors (addresses are valid, just wrong offsets within the buffer)

## Fix

```python
# flash_bwd_preprocess.py line 216
# Before:
seqlen = SeqlenInfo.create(batch_idx, mO.shape[1], mCuSeqlensQ, mSeqUsedQ)
# After:
seqlen = SeqlenInfo.create(batch_idx, mO.shape[1], mCuSeqlensQ, mSeqUsedQ, tile=self.tile_m)
```

## Lesson

Any code computing `padded_offset` for varlen buffers must use the same tile size as the kernel that allocated and accesses those buffers. The `SeqlenInfo.create` default `tile=128` is a trap when `m_block_size != 128`.
