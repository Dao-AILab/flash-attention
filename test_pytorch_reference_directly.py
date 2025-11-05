#!/usr/bin/env python3
"""Test if the issue is with our test configuration or PyTorch."""

import torch
from torch.nn.attention.flex_attention import create_block_mask

# Simple causal mask from PyTorch documentation
def causal_mask(b, h, q_idx, kv_idx):
    return kv_idx <= q_idx

# Test with 113x203 and 64x128 blocks
seqlen_q = 113
seqlen_k = 203
tile_m = 64
tile_n = 128

block_mask = create_block_mask(
    causal_mask,
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

print("PyTorch reference for simple causal mask (kv_idx <= q_idx):")
for m in range(2):
    mask_cnt = mask_block_cnt_ref[0, 0, m].item()
    full_cnt = full_block_cnt_ref[0, 0, m].item()
    print(f"  m_block={m}: mask_cnt={mask_cnt}, full_cnt={full_cnt}")
    if mask_cnt > 0:
        print(f"    mask_blocks: {mask_block_idx_ref[0, 0, m, :mask_cnt].cpu().tolist()}")
    if full_cnt > 0:
        print(f"    full_blocks: {full_block_idx_ref[0, 0, m, :full_cnt].cpu().tolist()}")

print("\nNow testing with offset causal mask (kv_idx <= q_idx + 90):")

# Causal with offset
def causal_offset_mask(b, h, q_idx, kv_idx):
    return kv_idx <= (q_idx + 90)

block_mask2 = create_block_mask(
    causal_offset_mask,
    B=1,
    H=1,
    Q_LEN=seqlen_q,
    KV_LEN=seqlen_k,
    device="cuda",
    BLOCK_SIZE=(tile_m, tile_n),
)

_, _, mask_block_cnt_ref2, mask_block_idx_ref2, full_block_cnt_ref2, full_block_idx_ref2, *_ = (
    block_mask2.as_tuple()
)

for m in range(2):
    mask_cnt = mask_block_cnt_ref2[0, 0, m].item()
    full_cnt = full_block_cnt_ref2[0, 0, m].item()
    print(f"  m_block={m}: mask_cnt={mask_cnt}, full_cnt={full_cnt}")
    if mask_cnt > 0:
        print(f"    mask_blocks: {mask_block_idx_ref2[0, 0, m, :mask_cnt].cpu().tolist()}")
    if full_cnt > 0:
        print(f"    full_blocks: {full_block_idx_ref2[0, 0, m, :full_cnt].cpu().tolist()}")
