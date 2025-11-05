#!/usr/bin/env python3
"""Check PyTorch evaluation for in-bounds values only."""

from flash_attn.cute.mask_definitions import get_mask_pair

seqlen_q = 113
seqlen_k = 203
tile_m = 64
tile_n = 128

# Get the flex mask function
_, mask_mod_flex = get_mask_pair("causal", seqlen_q=seqlen_q, seqlen_k=seqlen_k)

print("Evaluation for block (1,0) - IN-BOUNDS ONLY:")
m_block = 1
n_block = 0
m_start = m_block * tile_m  # 64
m_end = min((m_block + 1) * tile_m, seqlen_q)  # min(128, 113) = 113
n_start = n_block * tile_n  # 0
n_end = min((n_block + 1) * tile_n, seqlen_k)  # min(128, 203) = 128

print(f"m_start={m_start}, m_end={m_end}")
print(f"n_start={n_start}, n_end={n_end}")

has_unmasked = False
has_masked = False
count_false = 0
count_true = 0

for q_idx in range(m_start, m_end):
    for kv_idx in range(n_start, n_end):
        mask_val = mask_mod_flex(0, 0, q_idx, kv_idx)
        if mask_val:
            has_unmasked = True
            count_true += 1
        else:
            has_masked = True
            count_false += 1

print(f"\nhas_unmasked={has_unmasked}, has_masked={has_masked}")
print(f"count_true={count_true}, count_false={count_false}")
print(f"Classification: {'FULL' if has_unmasked and not has_masked else 'PARTIAL' if has_unmasked and has_masked else 'SKIP'}")
