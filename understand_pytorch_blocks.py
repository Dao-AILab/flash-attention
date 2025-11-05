#!/usr/bin/env python3
"""Understand how PyTorch divides blocks."""

import torch
from torch.nn.attention.flex_attention import create_block_mask
from flash_attn.cute.mask_definitions import get_mask_pair

seqlen_q = 113
seqlen_k = 203
tile_m = 64
tile_n = 128

# Get the flex mask function (PyTorch version)
_, mask_mod_flex = get_mask_pair("causal", seqlen_q=seqlen_q, seqlen_k=seqlen_k)

print("Block calculation:")
print(f"  seqlen_q={seqlen_q}, tile_m={tile_m}")
print(f"  n_blocks_q = ceil({seqlen_q}/{tile_m}) = {(seqlen_q + tile_m - 1) // tile_m}")
print(f"  seqlen_k={seqlen_k}, tile_n={tile_n}")
print(f"  n_blocks_k = ceil({seqlen_k}/{tile_n}) = {(seqlen_k + tile_n - 1) // tile_n}")

# So we should have 2x2 blocks
# m_block=0: rows 0-63
# m_block=1: rows 64-112 (not 127 since we only have 113 rows)
# n_block=0: cols 0-127
# n_block=1: cols 128-202 (not 255 since we only have 203 cols)

print("\nBlock boundaries:")
for m in range(2):
    print(f"  m_block={m}: rows {m*tile_m} to {min((m+1)*tile_m-1, seqlen_q-1)}")
for n in range(2):
    print(f"  n_block={n}: cols {n*tile_n} to {min((n+1)*tile_n-1, seqlen_k-1)}")

# Now let's check what happens with different block patterns
print("\n\nLet's evaluate the mask on full tiles (even if extending beyond seqlen):")

for m_block in range(2):
    for n_block in range(2):
        m_start = m_block * tile_m
        m_end = (m_block + 1) * tile_m  # Note: not clamped!
        n_start = n_block * tile_n
        n_end = (n_block + 1) * tile_n  # Note: not clamped!

        has_unmasked = False
        has_masked = False

        for q_idx in range(m_start, m_end):
            for kv_idx in range(n_start, n_end):
                # Call the mask function (it handles out-of-bounds)
                mask_val = mask_mod_flex(0, 0, q_idx, kv_idx)
                if mask_val:
                    has_unmasked = True
                else:
                    has_masked = True

        classification = "FULL" if (has_unmasked and not has_masked) else "PARTIAL" if (has_unmasked and has_masked) else "SKIP"
        print(f"  block({m_block},{n_block}): rows {m_start}-{m_end-1}, cols {n_start}-{n_end-1} -> {classification}")
