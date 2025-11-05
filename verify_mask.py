#!/usr/bin/env python3
"""Verify mask values more carefully."""

import torch

seqlen_q = 113
seqlen_k = 203
tile_m = 64
tile_n = 128
offset = seqlen_k - seqlen_q

print(f"offset={offset}")

# For m_block=1, n_block=0
m_base = 64
n_base = 0

print(f"\nm_block=1, n_block=0: rows {m_base}-{min(m_base+tile_m-1, seqlen_q-1)}, cols {n_base}-{min(n_base+tile_n-1, seqlen_k-1)}")

# Look at boundary cases
print("\nMask values at key positions:")
for q_idx in [64, 72, 80, 100, 110, 112]:
    for kv_idx in [0, 64, 127]:
        mask_val = kv_idx <= (q_idx + offset)
        in_range = (q_idx < seqlen_q and kv_idx < seqlen_k)
        print(f"  q={q_idx:3d}, kv={kv_idx:3d}: {mask_val} (in_range={in_range})")

# Full check
print("\nFull block check:")
has_unmasked = False
has_masked = False
for q_idx in range(64, min(64 + tile_m, seqlen_q)):
    for kv_idx in range(0, min(tile_n, seqlen_k)):
        mask_val = kv_idx <= (q_idx + offset)
        if mask_val:
            has_unmasked = True
        else:
            has_masked = True

print(f"has_unmasked={has_unmasked}, has_masked={has_masked}")
print(f"Result: {'FULL' if has_unmasked and not has_masked else 'PARTIAL' if has_unmasked and has_masked else 'SKIP'}")

# Check row by row
print("\nRow by row:")
for q_idx in range(64, min(64 + tile_m, seqlen_q)):
    row_has_false = False
    for kv_idx in range(0, min(tile_n, seqlen_k)):
        mask_val = kv_idx <= (q_idx + offset)
        if not mask_val:
            row_has_false = True
            break
    print(f"  row q={q_idx}: has_masked={row_has_false}")
