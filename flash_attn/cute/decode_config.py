"""Pure split/tile heuristic shared by native SwapAB routing.

Extracted unchanged from flash_fwd_decode_sm100.py at
10ec7dcb84f45c8c98da76e6da546e2a72234f76 (CUTLASS gqa_decode_opt adaptation).
No CUDA, Torch, or experimental DSL imports belong in routing configuration.
"""


def _next_pow2(x):
    if x <= 0:
        raise ValueError(f"x must be positive, got {x}")
    return 1 << (x - 1).bit_length()


# Default split count. Decode is bandwidth bound, so what matters is enough bytes in flight: each
# CTA keeps ~192 KB of K/V in its SMEM ring, and DRAM saturates with roughly `target_ctas` CTAs
# (fewer KV bytes per tile at head_dim 64 -> more CTAs). More splits than that only add per-CTA
# fixed cost and combine work. The split count is rounded down to a power of two (uneven splits
# leave some CTAs with an extra block), capped at 32, and each split keeps >= 512 tokens.
# Fitted on B200: within 2% of the best split on average, 8.5% worst case, over a 22-shape sweep.
DECODE_TARGET_CTAS = {64: 128}
DECODE_MAX_SPLITS = 32
DECODE_MIN_TOKENS_PER_SPLIT = 512


def get_decode_config(batch, num_heads, num_heads_kv, seqlen_k, head_dim=128, target_ctas=None):
    qhead_per_kvhead = num_heads // num_heads_kv
    tile_hq = min(_next_pow2(qhead_per_kvhead), 32)
    num_hq_blocks = (qhead_per_kvhead + tile_hq - 1) // tile_hq
    num_bh = batch * num_heads_kv
    if target_ctas is None:
        target_ctas = DECODE_TARGET_CTAS.get(head_dim, 64)
    num_splits = max(1, target_ctas // (num_hq_blocks * num_bh))
    num_splits = 1 << (num_splits.bit_length() - 1)
    num_splits = min(num_splits, DECODE_MAX_SPLITS, max(1, seqlen_k // DECODE_MIN_TOKENS_PER_SPLIT))
    return (num_splits, tile_hq, num_hq_blocks, num_bh)
