#!/usr/bin/env python3
"""Check what PyTorch's create_block_mask actually does."""

import torch
from torch.nn.attention.flex_attention import create_block_mask
from flash_attn.cute.mask_definitions import get_mask_pair

seqlen_q = 113
seqlen_k = 203
tile_m = 64
tile_n = 128

# Get the flex mask function (PyTorch version)
_, mask_mod_flex = get_mask_pair("causal", seqlen_q=seqlen_q, seqlen_k=seqlen_k)

print("Testing the mask function directly:")
print("mask_mod_flex(b, h, q_idx, kv_idx) for various indices:")

# Test the mask directly
for q_idx in [64, 100, 112]:
    for kv_idx in [0, 64, 127, 154, 202]:
        result = mask_mod_flex(0, 0, q_idx, kv_idx)
        print(f"  q={q_idx}, kv={kv_idx}: {result}")

print("\n\nNow check what create_block_mask returns:")

block_mask = create_block_mask(
    mask_mod_flex,
    B=1,
    H=1,
    Q_LEN=seqlen_q,
    KV_LEN=seqlen_k,
    device="cuda",
    BLOCK_SIZE=(tile_m, tile_n),
)

_, _, mask_block_cnt_ref, mask_block_idx_ref, full_block_cnt_ref, full_block_idx_ref, *_ = (
    block_mask.as_tuple()
)

# Print block classification
print("\nBlock classifications (0=FULL, 1=PARTIAL, 2=SKIP):")
print("m_block \\ n_block  0  1")
n_blocks_k = (seqlen_k + tile_n - 1) // tile_n
for m_block in range(2):
    print(f"  {m_block}        ", end="")
    for n_block in range(n_blocks_k):
        # Check if this block is full, partial, or skip
        mask_cnt = mask_block_cnt_ref[0, 0, m_block].item()
        full_cnt = full_block_cnt_ref[0, 0, m_block].item()

        # Find if n_block is in full or mask list
        is_full = False
        is_partial = False
        if full_cnt > 0:
            full_blocks = full_block_idx_ref[0, 0, m_block, :full_cnt]
            if (full_blocks == n_block).any():
                is_full = True
        if mask_cnt > 0:
            mask_blocks = mask_block_idx_ref[0, 0, m_block, :mask_cnt]
            if (mask_blocks == n_block).any():
                is_partial = True

        if is_full:
            print("F ", end="")
        elif is_partial:
            print("P ", end="")
        else:
            print("S ", end="")
    print()

print("\nDetailed results:")
for m in range(2):
    print(f"  m_block={m}:")
    mask_cnt = mask_block_cnt_ref[0, 0, m].item()
    full_cnt = full_block_cnt_ref[0, 0, m].item()
    print(f"    mask_cnt={mask_cnt}, full_cnt={full_cnt}")
    if mask_cnt > 0:
        print(f"    mask_blocks: {mask_block_idx_ref[0, 0, m, :mask_cnt].cpu().tolist()}")
    if full_cnt > 0:
        print(f"    full_blocks: {full_block_idx_ref[0, 0, m, :full_cnt].cpu().tolist()}")
