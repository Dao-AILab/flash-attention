#!/usr/bin/env python3
"""Check out-of-bounds rows."""

from flash_attn.cute.mask_definitions import get_mask_pair

seqlen_q = 113
seqlen_k = 203

# Get the flex mask function
_, mask_mod_flex = get_mask_pair("causal", seqlen_q=seqlen_q, seqlen_k=seqlen_k)

print("Mask values for out-of-bounds rows:")
for q_idx in [112, 113, 114, 120, 127]:
    for kv_idx in [0, 64, 127, 155, 202]:
        result = mask_mod_flex(0, 0, q_idx, kv_idx)
        print(f"  q={q_idx}, kv={kv_idx}: {result}")

print("\n\nLet's check row 113 fully:")
has_unmasked = False
has_masked = False
for kv_idx in range(0, 128):
    result = mask_mod_flex(0, 0, 113, kv_idx)
    if result:
        has_unmasked = True
    else:
        has_masked = True

print(f"Row 113 has_unmasked={has_unmasked}, has_masked={has_masked}")
print(f"Classification: {'FULL' if has_unmasked and not has_masked else 'PARTIAL'}")
