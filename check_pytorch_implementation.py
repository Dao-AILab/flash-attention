#!/usr/bin/env python3
"""Check how PyTorch handles block classification."""

from flash_attn.cute.mask_definitions import get_mask_pair

seqlen_q = 113
seqlen_k = 203
tile_m = 64
tile_n = 128

# Get the flex mask function
_, mask_mod_flex = get_mask_pair("causal", seqlen_q=seqlen_q, seqlen_k=seqlen_k)

# Manually check what PyTorch should see for block (1, 0)
print("Block (1, 0) evaluation - PyTorch likely uses ceil division semantics:")
print("\nFull tile approach (what create_block_mask might use):")

m_start = 1 * tile_m
m_end = (1 + 1) * tile_m  # = 128
n_start = 0 * tile_n
n_end = (0 + 1) * tile_n  # = 128

has_unmasked = False
has_masked = False

for q_idx in range(m_start, m_end):
    for kv_idx in range(n_start, n_end):
        mask_val = mask_mod_flex(0, 0, q_idx, kv_idx)
        if mask_val:
            has_unmasked = True
        else:
            has_masked = True

print(f"Block (1,0): rows {m_start}-{m_end-1}, cols {n_start}-{n_end-1}")
print(f"  has_unmasked={has_unmasked}, has_masked={has_masked}")

# Now let's find where the first False appears
print("\nFinding first False in block (1, 0):")
for q_idx in range(m_start, min(m_end, 130)):
    for kv_idx in range(n_start, min(n_end, 200)):
        mask_val = mask_mod_flex(0, 0, q_idx, kv_idx)
        if not mask_val:
            print(f"  First False at q={q_idx}, kv={kv_idx}")
            # Found it, let's look around
            print(f"  Context:")
            for qq in range(max(q_idx-2, 0), min(q_idx+3, 200)):
                for kk in range(max(kv_idx-2, 0), min(kv_idx+3, 200)):
                    mv = mask_mod_flex(0, 0, qq, kk)
                    marker = "X" if (qq == q_idx and kk == kv_idx) else " "
                    print(f"    {marker} q={qq}, kv={kk}: {mv}")
            break
    else:
        continue
    break

print("\n\nLet me check the mask formula more carefully:")
offset = seqlen_k - seqlen_q
print(f"Offset: {offset}")
print(f"Mask formula: kv_idx <= (q_idx + {offset})")

# For which q, kv pairs is the mask False?
print("\nFinding where kv_idx > (q_idx + offset):")
false_count = 0
for q_idx in range(64, 128):
    for kv_idx in range(0, 128):
        if kv_idx > (q_idx + offset):
            if false_count < 10:
                print(f"  q={q_idx}, kv={kv_idx}: {kv_idx} > {q_idx + offset}")
            false_count += 1

print(f"Total False values in block: {false_count}")
