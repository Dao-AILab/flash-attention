"""Small CuTeDSL modifiers used by the forward configuration campaign."""

import cutlass
from cutlass import cute

from flash_attn.cute import utils


@cute.jit
def score_mod_times_two(
    score,
    b_idx,
    h_idx,
    q_idx,
    kv_idx,
    seqlen_info,
    aux_tensors,
):
    """Double scaled attention logits without reading auxiliary tensors."""
    return score * cute.full_like(score, 2)


@cute.jit
def causal_window_128_mask(
    batch: cute.TensorSSA,
    head: cute.TensorSSA,
    m_idx: cute.TensorSSA,
    n_idx: cute.TensorSSA,
    seqlen_info,
    aux_tensors,
) -> cute.TensorSSA:
    """Keep a lower-right-aligned causal window of 129 positions."""
    offset = utils.scalar_to_ssa(
        seqlen_info.seqlen_k - seqlen_info.seqlen_q, cutlass.Int32
    )
    center = m_idx + offset
    window = utils.scalar_to_ssa(128, cutlass.Int32)
    return (n_idx <= center) & (n_idx >= center - window)


SCORE_MODS = {"times_two": score_mod_times_two}
MASK_MODS = {"causal_window_128": causal_window_128_mask}
