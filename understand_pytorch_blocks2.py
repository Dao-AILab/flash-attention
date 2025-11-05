#!/usr/bin/env python3
"""Understand what PyTorch blocks really are."""

# Offset causal: kv_idx <= q_idx + 90
# Block layout with 113x203, 64x128 tiles

seqlen_q = 113
seqlen_k = 203
tile_m = 64
tile_n = 128
offset = 90

print("Block (1,0): m_block=1, n_block=0")
print(f"Expected rows: {1*tile_m} to {min((1+1)*tile_m-1, seqlen_q-1)} = 64 to 112")
print(f"Expected cols: {0*tile_n} to {min((0+1)*tile_n-1, seqlen_k-1)} = 0 to 127")

# But wait - maybe PyTorch is checking ALL elements in the tile, even out-of-bounds
# Let me recalculate with full tiles
print("\nFull tile interpretation (might be what PyTorch does):")
print(f"Block (1,0): rows 64 to 127, cols 0 to 127")

# For this block, when does mask become False?
#  kv_idx > q_idx + 90
# For row 64: kv_idx > 154 (out of bounds, since max kv_idx in this block is 127)
# For row 100: kv_idx > 190 (out of bounds)
# For row 112: kv_idx > 202 (out of bounds)
# For row 113: kv_idx > 203 (out of bounds)
# For row 127: kv_idx > 217 (out of bounds)

# So in range [0, 127], all are <= q_idx + 90 for q_idx in [64, 127]
# That means the block should be FULL

print("\nBut wait - let me check if PyTorch might be doing something different...")
print("What if PyTorch is checking whether the block is 'safe' to use as FULL?")
print("Maybe they check: 'is there any element in the block that could be accessed?'")
print("and only call it FULL if ALL elements including padding would be unmasked?")

print("\nLet me check the boundary: what's the smallest q_idx and kv_idx in the block?")
print(f"  min q_idx = 64, min kv_idx = 0")
print(f"  mask(64, 0) = (0 <= 64 + 90) = True")

print(f"\nWhat about the largest indices?")
print(f"  max q_idx in tile = 127, max kv_idx in tile = 127")
print(f"  mask(127, 127) = (127 <= 127 + 90) = True")

print("\nSo full tile interpretation still gives FULL")

print("\n\nLet me check what happens with block (0,1):")
print(f"Block (0,1): m_block=0, n_block=1")
print(f"Rows: 0-63, Cols: 128-255 (but seqlen_k=203, so 128-202)")

# For this block:
# row 0, col 128: 128 <= 0 + 90 = False!
print(f"  row 0, col 128: {128 <= 0 + 90} (should mask out)")
print(f"  row 63, col 128: {128 <= 63 + 90} = {128 <= 153}")

print("\nSo block (0,1) IS PARTIAL. That makes sense.")

print("\n\nWait... let me reconsider block (1,0) more carefully...")
print("Maybe PyTorch's create_block_mask doesn't look at individual in-bounds elements,")
print("but instead evaluates using FULL TILE semantics?")

print("\nLet me check: for block (1,0) with FULL TILE (rows 64-127, cols 0-127):")
print("For which (q,kv) pairs is the mask False?")
print("  kv_idx > q_idx + 90")
print("  In range q=[64,127], kv=[0,127]:")
print("    row 64: kv > 154 -> no False (all kv <= 127)")
print("    row 127: kv > 217 -> no False (all kv <= 127)")

print("\nFull tile SHOULD be FULL!")

print("\n\nOK so either:")
print("1. PyTorch's reference implementation is wrong")
print("2. Or our mask functions are different")
print("3. Or there's some other semantics I'm missing")
