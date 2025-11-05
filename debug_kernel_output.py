#!/usr/bin/env python3
"""Debug script to see what the kernel actually returns."""

import torch
from flash_attn.cute.mask_definitions import get_mask_pair
from flash_attn.cute.compute_block_sparsity import compute_block_sparsity

# Test parameters
seqlen_q = 113
seqlen_k = 203
tile_m = 64
tile_n = 128
batch_size = 1
nheads = 1

# Get the cute mask function
cute_mask, _ = get_mask_pair("causal", seqlen_q=seqlen_q, seqlen_k=seqlen_k)

# Call compute_block_sparsity
blocksparse_tensors, torch_tensors = compute_block_sparsity(
    tile_m=tile_m,
    tile_n=tile_n,
    batch_size=batch_size,
    num_heads=nheads,
    seqlen_q=seqlen_q,
    seqlen_k=seqlen_k,
    mask_mod=cute_mask,
    aux_tensors=None,
    device="cuda",
    use_fast_sampling=False,  # Use full sampling first
)

full_block_cnt, full_block_idx, mask_block_cnt, mask_block_idx = torch_tensors

print(f"\nTest Config: seqlen_q={seqlen_q}, seqlen_k={seqlen_k}")
print(f"Tile sizes: tile_m={tile_m}, tile_n={tile_n}")
print(f"Full sampling mode")

print("\nKernel Output Results:")
for m in range(2):
    print(f"  m_block={m} (rows {m*tile_m}-{min((m+1)*tile_m-1, seqlen_q-1)}):")
    mask_cnt = mask_block_cnt[0, 0, m].item()
    full_cnt = full_block_cnt[0, 0, m].item()
    print(f"    mask_cnt={mask_cnt}, full_cnt={full_cnt}")
    if mask_cnt > 0:
        mask_blocks = mask_block_idx[0, 0, m, :mask_cnt].cpu().tolist()
        print(f"    mask_blocks: {mask_blocks}")
    if full_cnt > 0:
        full_blocks = full_block_idx[0, 0, m, :full_cnt].cpu().tolist()
        print(f"    full_blocks: {full_blocks}")
