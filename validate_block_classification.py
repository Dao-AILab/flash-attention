#!/usr/bin/env python3
"""Validate the block classification logic independently."""

from flash_attn.cute.mask_definitions import get_mask_pair

seqlen_q = 113
seqlen_k = 203
tile_m = 64
tile_n = 128

# Get the causal mask function
_, mask_mod_flex = get_mask_pair("causal", seqlen_q=seqlen_q, seqlen_k=seqlen_k)

print("=" * 70)
print("VALIDATION: Block (1,0) classification")
print("=" * 70)

m_block = 1
n_block = 0
m_start = m_block * tile_m  # 64
m_end = min((m_block + 1) * tile_m, seqlen_q)  # min(128, 113) = 113
n_start = n_block * tile_n  # 0
n_end = min((n_block + 1) * tile_n, seqlen_k)  # min(128, 203) = 128

print(f"\nBlock ({m_block},{n_block}):")
print(f"  Q indices: {m_start}-{m_end-1} (inclusive, in-bounds)")
print(f"  KV indices: {n_start}-{n_end-1} (inclusive, in-bounds)")
print(f"  Total elements to check: {(m_end-m_start) * (n_end-n_start)}")

# Count unmasked vs masked
unmasked_count = 0
masked_count = 0
sample_masked = []

for q_idx in range(m_start, m_end):
    for kv_idx in range(n_start, n_end):
        mask_val = mask_mod_flex(0, 0, q_idx, kv_idx)
        if mask_val:
            unmasked_count += 1
        else:
            masked_count += 1
            if len(sample_masked) < 5:
                sample_masked.append((q_idx, kv_idx))

print(f"\nMask evaluation results:")
print(f"  Unmasked (True) count: {unmasked_count}")
print(f"  Masked (False) count: {masked_count}")

if sample_masked:
    print(f"  Sample masked positions: {sample_masked}")

print(f"\nBlock classification:")
if masked_count == 0 and unmasked_count > 0:
    print("  ✓ ALL IN-BOUNDS values are UNMASKED")
    print("  => Classification: FULL")
    print("  => Kernel result: full_cnt+=1")
elif masked_count > 0 and unmasked_count > 0:
    print("  ✓ MIXED IN-BOUNDS values")
    print("  => Classification: PARTIAL")
    print("  => Kernel result: mask_cnt+=1")
elif masked_count > 0 and unmasked_count == 0:
    print("  ✓ ALL IN-BOUNDS values are MASKED")
    print("  => Classification: SKIP")
    print("  => Kernel result: neither mask_cnt nor full_cnt")

print("\n" + "=" * 70)
print("Expected behavior according to spec:")
print("  'A block is FULL if all its IN-BOUNDS values are unmasked'")
print("=" * 70)

# Also check what PyTorch says
print("\nComparison with PyTorch's create_block_mask:")
from torch.nn.attention.flex_attention import create_block_mask

def causal_offset_mask(b, h, q_idx, kv_idx):
    return kv_idx <= (q_idx + 90)

block_mask = create_block_mask(
    causal_offset_mask,
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

mask_cnt = mask_block_cnt_ref[0, 0, m_block].item()
full_cnt = full_block_cnt_ref[0, 0, m_block].item()
mask_blocks = mask_block_idx_ref[0, 0, m_block, :mask_cnt].cpu().tolist() if mask_cnt > 0 else []
full_blocks = full_block_idx_ref[0, 0, m_block, :full_cnt].cpu().tolist() if full_cnt > 0 else []

print(f"  PyTorch says m_block={m_block}:")
print(f"    mask_cnt={mask_cnt}, mask_blocks={mask_blocks}")
print(f"    full_cnt={full_cnt}, full_blocks={full_blocks}")

if n_block in full_blocks:
    print(f"    => Block ({m_block},{n_block}) is FULL")
elif n_block in mask_blocks:
    print(f"    => Block ({m_block},{n_block}) is PARTIAL")
else:
    print(f"    => Block ({m_block},{n_block}) is SKIP")
