#!/usr/bin/env python3
"""Check data types in mask evaluation."""

from flash_attn.cute.mask_definitions import get_mask_pair

seqlen_q = 113
seqlen_k = 203
offset = 90

_, mask_mod_flex = get_mask_pair("causal", seqlen_q=seqlen_q, seqlen_k=seqlen_k)

# Test a few values
result = mask_mod_flex(0, 0, 64, 0)
print(f"mask_mod_flex(0, 0, 64, 0) = {result}")
print(f"  type: {type(result)}")
print(f"  value: {repr(result)}")

result2 = mask_mod_flex(0, 0, 64, 127)
print(f"\nmask_mod_flex(0, 0, 64, 127) = {result2}")
print(f"  type: {type(result2)}")
print(f"  value: {repr(result2)}")

# Check if there are any False values in block (1,0)
print(f"\n\nChecking block (1,0) for False values:")
false_count = 0
for q_idx in range(64, 113):
    for kv_idx in range(0, 128):
        result = mask_mod_flex(0, 0, q_idx, kv_idx)
        if not result:
            false_count += 1
            if false_count <= 5:
                print(f"  Found False at q={q_idx}, kv={kv_idx}")

print(f"Total False values: {false_count}")
