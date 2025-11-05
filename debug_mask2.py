#!/usr/bin/env python3
"""Debug script to check n_block=1 as well."""

import torch

# Test parameters
seqlen_q = 113
seqlen_k = 203
tile_m = 64
tile_n = 128
offset = seqlen_k - seqlen_q

print(f"Offset: {offset}")
print(f"Causal mask formula: kv_idx <= (q_idx + {offset})")

# For m_block=1, n_block=1
print("\n=== m_block=1 (rows 64-112), n_block=1 (cols 128-203) ===")
m_block = 1
n_block = 1
m_base = m_block * tile_m
n_base = n_block * tile_n

print(f"Block range: rows {m_base}-{min(m_base + tile_m - 1, seqlen_q - 1)}, cols {n_base}-{min(n_base + tile_n - 1, seqlen_k - 1)}")

# Sample corners
corners = [
    (m_base, n_base, "top-left"),
    (m_base, min(n_base + tile_n - 1, seqlen_k - 1), "top-right"),
    (min(m_base + tile_m - 1, seqlen_q - 1), n_base, "bottom-left"),
    (min(m_base + tile_m - 1, seqlen_q - 1), min(n_base + tile_n - 1, seqlen_k - 1), "bottom-right"),
]

has_unmasked = False
has_masked = False

for q_idx, kv_idx, label in corners:
    mask_val = kv_idx <= (q_idx + offset)
    print(f"  {label}: q_idx={q_idx}, kv_idx={kv_idx} -> {mask_val}")
    if mask_val:
        has_unmasked = True
    else:
        has_masked = True

print(f"  Classification: has_unmasked={has_unmasked}, has_masked={has_masked} -> {'PARTIAL' if has_unmasked and has_masked else 'FULL' if has_unmasked else 'SKIP'}")

# Full evaluation
print("\n  Full block evaluation:")
has_unmasked_full = False
has_masked_full = False

for q_idx in range(m_base, min(m_base + tile_m, seqlen_q)):
    for kv_idx in range(n_base, min(n_base + tile_n, seqlen_k)):
        mask_val = kv_idx <= (q_idx + offset)
        if mask_val:
            has_unmasked_full = True
        else:
            has_masked_full = True

print(f"  Classification: has_unmasked={has_unmasked_full}, has_masked={has_masked_full} -> {'PARTIAL' if has_unmasked_full and has_masked_full else 'FULL' if has_unmasked_full else 'SKIP'}")
