#!/usr/bin/env python3
"""Debug script to understand mask evaluation for causal 113x203."""

import torch
from torch.nn.attention.flex_attention import create_block_mask
from flash_attn.cute.mask_definitions import get_mask_pair

# Test parameters
seqlen_q = 113
seqlen_k = 203
tile_m = 64
tile_n = 128
batch_size = 1
nheads = 1

# Get the causal mask function
_, mask_mod_flex = get_mask_pair("causal", seqlen_q=seqlen_q, seqlen_k=seqlen_k)

# Get reference from PyTorch
block_mask = create_block_mask(
    mask_mod_flex,
    B=batch_size,
    H=nheads,
    Q_LEN=seqlen_q,
    KV_LEN=seqlen_k,
    device="cuda",
    BLOCK_SIZE=(tile_m, tile_n),
)

_, _, mask_block_cnt_ref, mask_block_idx_ref, full_block_cnt_ref, full_block_idx_ref, *_ = (
    block_mask.as_tuple()
)

print(f"\nTest Config: seqlen_q={seqlen_q}, seqlen_k={seqlen_k}")
print(f"Tile sizes: tile_m={tile_m}, tile_n={tile_n}")
print(f"Offset: {seqlen_k - seqlen_q}")
print(f"Number of q-blocks: {(seqlen_q + tile_m - 1) // tile_m}")
print(f"Number of k-blocks: {(seqlen_k + tile_n - 1) // tile_n}")

# Print reference results
print("\nPyTorch Reference Results:")
for m in range(2):
    print(f"  m_block={m} (rows {m*tile_m}-{min((m+1)*tile_m-1, seqlen_q-1)}):")
    mask_cnt = mask_block_cnt_ref[0, 0, m].item()
    full_cnt = full_block_cnt_ref[0, 0, m].item()
    print(f"    mask_cnt={mask_cnt}, full_cnt={full_cnt}")
    if mask_cnt > 0:
        mask_blocks = mask_block_idx_ref[0, 0, m, :mask_cnt].cpu().tolist()
        print(f"    mask_blocks: {mask_blocks}")
    if full_cnt > 0:
        full_blocks = full_block_idx_ref[0, 0, m, :full_cnt].cpu().tolist()
        print(f"    full_blocks: {full_blocks}")

# Now manually evaluate the mask for m_block=1, n_block=0
print("\n\nManual Mask Evaluation for m_block=1, n_block=0:")
m_block = 1
n_block = 0
m_base = m_block * tile_m
n_base = n_block * tile_n

print(f"  m_base={m_base}, n_base={n_base}")
print(f"  Rows: {m_base} to {min(m_base + tile_m - 1, seqlen_q - 1)}")
print(f"  Cols: {n_base} to {min(n_base + tile_n - 1, seqlen_k - 1)}")

# Sample corners and center
corners = [
    (m_base, n_base, "top-left"),
    (m_base, min(n_base + tile_n - 1, seqlen_k - 1), "top-right"),
    (min(m_base + tile_m - 1, seqlen_q - 1), n_base, "bottom-left"),
    (min(m_base + tile_m - 1, seqlen_q - 1), min(n_base + tile_n - 1, seqlen_k - 1), "bottom-right"),
    (m_base + tile_m // 2, n_base + tile_n // 2, "center"),
]

offset = seqlen_k - seqlen_q
print(f"\n  Causal mask formula: kv_idx <= (q_idx + {offset})")

has_unmasked = False
has_masked = False

for q_idx, kv_idx, label in corners:
    # Check bounds
    if q_idx >= seqlen_q:
        q_idx = seqlen_q - 1
    if kv_idx >= seqlen_k:
        kv_idx = seqlen_k - 1

    mask_val = kv_idx <= (q_idx + offset)
    print(f"    {label}: q_idx={q_idx}, kv_idx={kv_idx} -> {mask_val}")
    if mask_val:
        has_unmasked = True
    else:
        has_masked = True

print(f"\n  has_unmasked={has_unmasked}, has_masked={has_masked}")
print(f"  Expected classification: ", end="")
if has_unmasked and has_masked:
    print("PARTIAL (mixed)")
elif has_unmasked and not has_masked:
    print("FULL")
elif not has_unmasked and has_masked:
    print("SKIP")
else:
    print("UNKNOWN")

# Now check all elements in the block
print("\n\nFull Block Evaluation for m_block=1, n_block=0:")
has_unmasked_full = False
has_masked_full = False

for q_idx in range(m_base, min(m_base + tile_m, seqlen_q)):
    for kv_idx in range(n_base, min(n_base + tile_n, seqlen_k)):
        mask_val = kv_idx <= (q_idx + offset)
        if mask_val:
            has_unmasked_full = True
        else:
            has_masked_full = True

print(f"  has_unmasked={has_unmasked_full}, has_masked={has_masked_full}")
print(f"  Expected classification: ", end="")
if has_unmasked_full and has_masked_full:
    print("PARTIAL (mixed)")
elif has_unmasked_full and not has_masked_full:
    print("FULL")
elif not has_unmasked_full and has_masked_full:
    print("SKIP")
else:
    print("UNKNOWN")

# Show some specific element patterns
print("\n  Sample of mask values in this block:")
print("  q_idx \\ kv_idx  0-10  20-30  50-60  100-110  120-128")
for q_idx in range(m_base, min(m_base + tile_m, seqlen_q), 10):
    print(f"    {q_idx:3d}  ", end="")
    for kv_start in [0, 20, 50, 100, 120]:
        kv_end = min(kv_start + 10, seqlen_k)
        if kv_end > n_base + tile_n:
            print("  ---   ", end="")
            continue
        vals = [kv_idx <= (q_idx + offset) for kv_idx in range(kv_start, min(kv_start + 3, kv_end))]
        print(f"  {vals}  ", end="")
    print()
