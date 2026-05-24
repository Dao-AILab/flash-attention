from typing import Callable, Optional

import random
import math

import cutlass
import cutlass.cute as cute
import torch

from flash_attn.cute import utils
from flash_attn.cute.block_sparsity import fast_sampling


# =============================================================================
# CuTe mask_mod functions (for kernel compilation)
# All use signature: (batch, head, m_idx, n_idx, seqlen_info, aux_tensors)
# =============================================================================

# =============================================================================
# mask_mod functions that don't use global indices
# =============================================================================


@fast_sampling
@cute.jit
def cute_causal_mask(
    batch: cute.TensorSSA,
    head: cute.TensorSSA,
    m_idx: cute.TensorSSA,
    n_idx: cute.TensorSSA,
    seqlen_info,
    aux_tensors: None,
) -> cute.TensorSSA:
    offset = seqlen_info.seqlen_k - seqlen_info.seqlen_q
    offset_ssa = utils.scalar_to_ssa(offset, cutlass.Int32)
    return n_idx <= (m_idx + offset_ssa)


def get_cute_causal_mask(offset: int):
    return cute_causal_mask


def get_cute_block_causal_mask(offset: int):
    @fast_sampling
    @cute.jit
    def _cute_block_causal_mask(
        batch: cute.TensorSSA,
        head: cute.TensorSSA,
        m_idx: cute.TensorSSA,
        n_idx: cute.TensorSSA,
        seqlen_info,
        aux_tensors: None,
    ) -> cute.TensorSSA:
        offset_ssa = utils.scalar_to_ssa(offset, cutlass.Int32)
        return n_idx <= (m_idx + offset_ssa)

    return _cute_block_causal_mask


def get_cute_sliding_window_mask(window_left: int, window_right: int, offset: int):
    @fast_sampling
    @cute.jit
    def _cute_sliding_window_mask(
        batch: cute.TensorSSA,
        head: cute.TensorSSA,
        m_idx: cute.TensorSSA,
        n_idx: cute.TensorSSA,
        seqlen_info,
        aux_tensors,
    ) -> cute.TensorSSA:
        offset = seqlen_info.seqlen_k - seqlen_info.seqlen_q
        offset_ssa = utils.scalar_to_ssa(offset, cutlass.Int32)
        window_left_ssa = utils.scalar_to_ssa(window_left, cutlass.Int32)
        window_right_ssa = utils.scalar_to_ssa(window_right, cutlass.Int32)
        center = m_idx + offset_ssa
        lower = center - window_left_ssa
        upper = center + window_right_ssa
        return (n_idx >= lower) & (n_idx <= upper)

    return _cute_sliding_window_mask


@fast_sampling
@cute.jit
def cute_block_diagonal_mask(
    batch: cute.TensorSSA,
    head: cute.TensorSSA,
    m_idx: cute.TensorSSA,
    n_idx: cute.TensorSSA,
    seqlen_info,
    aux_tensors,
) -> cute.TensorSSA:
    block_size_ssa = utils.scalar_to_ssa(128, cutlass.Int32)
    return (m_idx // block_size_ssa) == (n_idx // block_size_ssa)


@cute.jit
def cute_mini_causal_mask(
    batch: cute.TensorSSA,
    head: cute.TensorSSA,
    m_idx: cute.TensorSSA,
    n_idx: cute.TensorSSA,
    seqlen_info,
    aux_tensors,
) -> cute.TensorSSA:
    tile_size_ssa = utils.scalar_to_ssa(128, cutlass.Int32)
    m_mod = m_idx % tile_size_ssa
    n_mod = n_idx % tile_size_ssa
    return m_mod >= n_mod


@fast_sampling
@cute.jit
def cute_prefix_lm_mask(
    batch: cute.TensorSSA,
    head: cute.TensorSSA,
    m_idx: cute.TensorSSA,
    n_idx: cute.TensorSSA,
    seqlen_info,
    aux_tensors,
) -> cute.TensorSSA:
    """Prefix LM mask: first 512 tokens attend bidirectionally, rest use causal masking."""
    prefix_size_ssa = utils.scalar_to_ssa(512, cutlass.Int32)
    both_in_prefix = (m_idx < prefix_size_ssa) & (n_idx < prefix_size_ssa)
    causal_part = m_idx >= n_idx
    return both_in_prefix | causal_part


@cute.jit
def cute_dilated_sliding_window_mask(
    batch: cute.TensorSSA,
    head: cute.TensorSSA,
    m_idx: cute.TensorSSA,
    n_idx: cute.TensorSSA,
    seqlen_info,
    aux_tensors,
) -> cute.TensorSSA:
    """Dilated sliding window: every other position in a 256-position window."""
    window_size_ssa = utils.scalar_to_ssa(256, cutlass.Int32)
    dilation_ssa = utils.scalar_to_ssa(2, cutlass.Int32)
    in_window = (m_idx >= n_idx) & (m_idx - n_idx < window_size_ssa)
    dilated = ((m_idx - n_idx) % dilation_ssa) == utils.scalar_to_ssa(0, cutlass.Int32)
    return in_window & dilated


@fast_sampling
@cute.jit
def cute_document_mask(
    batch: cute.TensorSSA,
    head: cute.TensorSSA,
    m_idx: cute.TensorSSA,
    n_idx: cute.TensorSSA,
    seqlen_info,
    aux_tensors: list,
) -> cute.TensorSSA:
    doc_id = aux_tensors[0]
    m_doc = utils.scalar_to_ssa(doc_id[batch[0], head[0], m_idx[0]], cutlass.Int32)
    n_doc = utils.scalar_to_ssa(doc_id[batch[0], head[0], n_idx[0]], cutlass.Int32)
    return m_doc == n_doc


@cute.jit
def cute_ima_mask(
    batch: cute.TensorSSA,
    head: cute.TensorSSA,
    m_idx: cute.TensorSSA,
    n_idx: cute.TensorSSA,
    seqlen_info,
    aux_tensors,
) -> cute.TensorSSA:
    bias = aux_tensors[0]
    threshold = utils.scalar_to_ssa(bias[n_idx[0]], cutlass.Int32)
    return n_idx >= threshold


# =============================================================================
# mask_mod functions that use global indices (for use with variable sequence length)
# Global indices computed as: m_idx_global = m_idx + seqlen_info.offset_q
#                            n_idx_global = n_idx + seqlen_info.offset_k
# =============================================================================


@fast_sampling
@cute.jit
def cute_global_packed_doc_mask(
    batch: cute.TensorSSA,
    head: cute.TensorSSA,
    m_idx: cute.TensorSSA,
    n_idx: cute.TensorSSA,
    seqlen_info,
    aux_tensors,
) -> cute.TensorSSA:
    """Document mask using globally-indexed packed 1D doc ID tensors.

    aux_tensors[0]: doc_ids_q (total_q,) int32 — packed doc IDs for Q tokens
    aux_tensors[1]: doc_ids_k (total_k,) int32 — packed doc IDs for K tokens
    Mask: doc_ids_q[m_global] == doc_ids_k[n_global]
    """
    doc_ids_q = aux_tensors[0]
    doc_ids_k = aux_tensors[1]

    offset_q = seqlen_info.offset_q
    m_global = m_idx + offset_q
    m_frag = cute.make_fragment(1, cutlass.Int32)
    m_frag.store(m_global)
    m_doc_frag = cute.make_fragment(1, cutlass.Int32)
    m_doc_frag[0] = doc_ids_q[m_frag[0]]

    offset_k = seqlen_info.offset_k
    n_global = n_idx + offset_k
    n_frag = cute.make_fragment(1, cutlass.Int32)
    n_frag.store(n_global)
    n_doc_frag = cute.make_fragment(1, cutlass.Int32)
    n_doc_frag[0] = doc_ids_k[n_frag[0]]

    m_doc = m_doc_frag.load()
    n_doc = n_doc_frag.load()
    return m_doc == n_doc


@fast_sampling
@cute.jit
def cute_global_ima_mask(
    batch: cute.TensorSSA,
    head: cute.TensorSSA,
    m_idx: cute.TensorSSA,
    n_idx: cute.TensorSSA,
    seqlen_info,
    aux_tensors,
) -> cute.TensorSSA:
    """IMA-style mask using globally-indexed threshold tensor.

    aux_tensors[0]: thresholds (total_k,) int32 — per-global-kv-position threshold
    Mask: n_idx >= thresholds[n_global]  (local n_idx >= globally-indexed threshold)
    """
    thresholds = aux_tensors[0]

    offset_k = seqlen_info.offset_k
    n_global = n_idx + offset_k
    n_frag = cute.make_fragment(1, cutlass.Int32)
    n_frag.store(n_global)
    val_frag = cute.make_fragment(1, cutlass.Int32)
    val_frag[0] = thresholds[n_frag[0]]
    threshold = val_frag.load()

    return n_idx >= threshold


@fast_sampling
@cute.jit
def cute_global_causal_window_mask(
    batch: cute.TensorSSA,
    head: cute.TensorSSA,
    m_idx: cute.TensorSSA,
    n_idx: cute.TensorSSA,
    seqlen_info,
    aux_tensors,
) -> cute.TensorSSA:
    """Causal window mask with per-token window sizes indexed globally.

    aux_tensors[0]: windows (total_q,) int32 — per-global-q-position window size
    Mask: (n_idx <= m_idx) & (m_idx - n_idx <= windows[m_global])
    """
    windows = aux_tensors[0]

    offset_q = seqlen_info.offset_q
    m_global = m_idx + offset_q
    m_frag = cute.make_fragment(1, cutlass.Int32)
    m_frag.store(m_global)
    win_frag = cute.make_fragment(1, cutlass.Int32)
    win_frag[0] = windows[m_frag[0]]
    window = win_frag.load()

    return (n_idx <= m_idx) & ((m_idx - n_idx) <= window)


# =============================================================================
# Eager reference functions (PyTorch/Flex Attention signatures)
# =============================================================================


def get_flex_causal_mask(offset: int):
    def _flex_causal_mask(b, h, q_idx, kv_idx):
        return kv_idx <= q_idx + offset

    return _flex_causal_mask


def get_flex_block_causal_mask(offset: int):
    def _flex_block_causal_mask(b, h, q_idx, kv_idx):
        return kv_idx <= q_idx + offset

    return _flex_block_causal_mask


def get_flex_sliding_window_mask(window_left: int, window_right: int, offset: int):
    def _flex_sliding_window_mask(b, h, q_idx, kv_idx):
        center = q_idx + offset
        lower = center - window_left
        upper = center + window_right
        return (kv_idx >= lower) & (kv_idx <= upper)

    return _flex_sliding_window_mask


def flex_block_diagonal_mask(b, h, q_idx, kv_idx):
    block_size = 128
    return (q_idx // block_size) == (kv_idx // block_size)


def flex_mini_causal_mask(b, h, q_idx, kv_idx):
    return (q_idx % 128) >= (kv_idx % 128)


def flex_prefix_lm_mask(b, h, q_idx, kv_idx):
    """Prefix LM mask: first 512 tokens attend bidirectionally, rest use causal masking."""
    prefix_size = 512
    both_in_prefix = (q_idx < prefix_size) & (kv_idx < prefix_size)
    causal_part = q_idx >= kv_idx
    return both_in_prefix | causal_part


def flex_dilated_sliding_window_mask(b, h, q_idx, kv_idx):
    """Dilated sliding window: every other position in a 256-position window."""
    window_size = 256
    dilation = 2
    in_window = (q_idx >= kv_idx) & (q_idx - kv_idx < window_size)
    dilated = ((q_idx - kv_idx) % dilation) == 0
    return in_window & dilated


def flex_document_mask(b, h, q_idx, kv_idx, doc_id):
    return doc_id[b, h, q_idx] == doc_id[b, h, kv_idx]


def flex_ima_mask(b, h, q_idx, kv_idx, bias):
    return kv_idx >= bias[kv_idx]


# =============================================================================
# Flex reference factories for global-index masks (per-sequence)
# Each factory(seq_idx, sq, sk) -> mask(b, h, q_idx, kv_idx)
# where q_idx/kv_idx are local (0-indexed) within the sequence.
# =============================================================================


def global_packed_doc_flex_factory(doc_ids_q, doc_ids_k, cu_seqlens_q, cu_seqlens_k):
    """Factory for per-sequence flex reference of cute_global_packed_doc_mask."""

    def factory(seq_idx, sq, sk):
        q_offset = cu_seqlens_q[seq_idx].item()
        k_offset = cu_seqlens_k[seq_idx].item()

        def mask(b, h, q_idx, kv_idx):
            q_global = q_offset + q_idx
            k_global = k_offset + kv_idx
            return doc_ids_q[q_global] == doc_ids_k[k_global]

        return mask

    return factory


def global_ima_flex_factory(thresholds, cu_seqlens_k):
    """Factory for per-sequence flex reference of cute_global_ima_mask."""

    def factory(seq_idx, sq, sk):
        k_offset = cu_seqlens_k[seq_idx].item()

        def mask(b, h, q_idx, kv_idx):
            k_global = k_offset + kv_idx
            return kv_idx >= thresholds[k_global]

        return mask

    return factory


def global_causal_window_flex_factory(windows, cu_seqlens_q):
    """Factory for per-sequence flex reference of cute_global_causal_window_mask."""

    def factory(seq_idx, sq, sk):
        q_offset = cu_seqlens_q[seq_idx].item()

        def mask(b, h, q_idx, kv_idx):
            q_global = q_offset + q_idx
            window = windows[q_global]
            return (kv_idx <= q_idx) & ((q_idx - kv_idx) <= window)

        return mask

    return factory


# =============================================================================
# Utility functions
# =============================================================================


def random_doc_id_tensor(nheads, batch, seqlen_q, device="cpu"):
    """Generate synthetic document ids shared across heads."""
    doc_ids_tensor = torch.zeros(
        batch, nheads, seqlen_q, dtype=torch.int32, device=device
    )
    for b in range(batch):
        N = seqlen_q
        max_segments = max(1, math.ceil(math.sqrt(max(N // 4, 1))))
        n = random.randint(1, max_segments)
        n = min(n, N)
        cuts = sorted(random.sample(range(1, N), n - 1))
        lengths = [b - a for a, b in zip((0, *cuts), (*cuts, N))]
        base_doc_ids = torch.repeat_interleave(
            torch.arange(len(lengths), device=device, dtype=torch.int32),
            torch.tensor(lengths, device=device, dtype=torch.int32),
        )

        for h in range(nheads):
            doc_ids_tensor[b, h, :] = base_doc_ids
    return doc_ids_tensor


def make_packed_doc_ids(seqlens_q, seqlens_k, device="cuda"):
    """Generate packed 1D doc ID tensors for Q and K for varlen global-index testing.

    For each sequence, divides tokens into sqrt(len)-ish segments.
    Returns (doc_ids_q, doc_ids_k) of shape (total_q,) and (total_k,).
    """
    total_q = sum(seqlens_q)
    total_k = sum(seqlens_k)
    doc_ids_q = torch.zeros(total_q, dtype=torch.int32, device=device)
    doc_ids_k = torch.zeros(total_k, dtype=torch.int32, device=device)

    q_off = 0
    k_off = 0
    for sq, sk in zip(seqlens_q, seqlens_k):
        # Q doc IDs
        n_docs = max(1, math.ceil(math.sqrt(max(sq // 4, 1))))
        n_docs = min(n_docs, sq)
        if n_docs > 1 and sq > 1:
            cuts = sorted(random.sample(range(1, sq), min(n_docs - 1, sq - 1)))
        else:
            cuts = []
        lengths = [b - a for a, b in zip((0, *cuts), (*cuts, sq))]
        doc_ids_q[q_off : q_off + sq] = torch.repeat_interleave(
            torch.arange(len(lengths), dtype=torch.int32, device=device),
            torch.tensor(lengths, dtype=torch.int32, device=device),
        )

        # K doc IDs (same n_docs range for potential overlap)
        if n_docs > 1 and sk > 1:
            cuts_k = sorted(random.sample(range(1, sk), min(n_docs - 1, sk - 1)))
        else:
            cuts_k = []
        lengths_k = [b - a for a, b in zip((0, *cuts_k), (*cuts_k, sk))]
        doc_ids_k[k_off : k_off + sk] = torch.repeat_interleave(
            torch.arange(len(lengths_k), dtype=torch.int32, device=device),
            torch.tensor(lengths_k, dtype=torch.int32, device=device),
        )

        q_off += sq
        k_off += sk

    return doc_ids_q, doc_ids_k


def make_global_thresholds(seqlens_k, device="cuda"):
    """Generate per-global-kv-token thresholds for cute_global_ima_mask.

    For each K token at local index i in a sequence of length sk,
    threshold = random value in [0, sk//2].
    Returns thresholds of shape (total_k,).
    """
    total_k = sum(seqlens_k)
    thresholds = torch.zeros(total_k, dtype=torch.int32, device=device)
    k_off = 0
    for sk in seqlens_k:
        for i in range(sk):
            thresholds[k_off + i] = random.randint(0, max(0, sk // 2))
        k_off += sk
    return thresholds


def make_global_windows(seqlens_q, device="cuda"):
    """Generate per-global-q-token window sizes for cute_global_causal_window_mask.

    For Q token at local index i, window = random value in [0, i] (causal).
    Returns windows of shape (total_q,).
    """
    total_q = sum(seqlens_q)
    windows = torch.zeros(total_q, dtype=torch.int32, device=device)
    q_off = 0
    for sq in seqlens_q:
        for i in range(sq):
            windows[q_off + i] = random.randint(0, i)
        q_off += sq
    return windows


# =============================================================================
# Vectorized mask_mod variants (return bit-packed Uint32)
# =============================================================================
#
# Each variant receives shape-(vec_size,) Int32 SSAs and returns a shape-
# (max(1, vec_size // 32),) Uint32 TensorSSA, where bit i of element k is the
# mask for lane (k * 32 + i). Bodies assume lane i has idx[i] = idx[0] + i, so
# they compute the packed Uint32(s) in O(1) via integer/bit arithmetic on the
# chunk-base indices instead of evaluating per lane.


@cute.jit
def cute_causal_mask_vec(
    batch: cute.TensorSSA,
    head: cute.TensorSSA,
    m_idx: cute.TensorSSA,
    n_idx: cute.TensorSSA,
    seqlen_info,
    aux_tensors: None,
) -> cute.TensorSSA:
    offset = seqlen_info.seqlen_k - seqlen_info.seqlen_q
    threshold = m_idx[0] + offset - n_idx[0] + cutlass.Int32(1)
    m = max(cutlass.Int32(32) - threshold, cutlass.Int32(0))
    result = cute.make_rmem_tensor(1, dtype=cutlass.Uint32)
    result[0] = utils.shr_u32(cutlass.Uint32(0xFFFFFFFF), cutlass.Uint32(m))
    return result.load()


def get_cute_causal_mask_vec(offset: int):
    return cute_causal_mask_vec


def get_cute_sliding_window_mask_vec(window_left: int, window_right: int, offset: int):
    @cute.jit
    def _cute_sliding_window_mask_vec(
        batch: cute.TensorSSA,
        head: cute.TensorSSA,
        m_idx: cute.TensorSSA,
        n_idx: cute.TensorSSA,
        seqlen_info,
        aux_tensors,
    ) -> cute.TensorSSA:
        runtime_offset = seqlen_info.seqlen_k - seqlen_info.seqlen_q
        center = m_idx[0] + runtime_offset
        lo = center - cutlass.Int32(window_left) - n_idx[0]
        hi_excl = center + cutlass.Int32(window_right) - n_idx[0] + cutlass.Int32(1)
        m_below = max(cutlass.Int32(32) - hi_excl, cutlass.Int32(0))
        below = utils.shr_u32(cutlass.Uint32(0xFFFFFFFF), cutlass.Uint32(m_below))
        n_above = max(lo, cutlass.Int32(0))
        above = utils.shl_u32(cutlass.Uint32(0xFFFFFFFF), cutlass.Uint32(n_above))
        result = cute.make_rmem_tensor(1, dtype=cutlass.Uint32)
        result[0] = below & above
        return result.load()

    return _cute_sliding_window_mask_vec


@cute.jit
def cute_block_diagonal_mask_vec(
    batch: cute.TensorSSA,
    head: cute.TensorSSA,
    m_idx: cute.TensorSSA,
    n_idx: cute.TensorSSA,
    seqlen_info,
    aux_tensors,
) -> cute.TensorSSA:
    block_size = cutlass.Int32(128)
    block_m = m_idx[0] // block_size
    lo = block_m * block_size - n_idx[0]
    hi = (block_m + cutlass.Int32(1)) * block_size - n_idx[0]
    m_below = max(cutlass.Int32(32) - hi, cutlass.Int32(0))
    below = utils.shr_u32(cutlass.Uint32(0xFFFFFFFF), cutlass.Uint32(m_below))
    n_above = max(lo, cutlass.Int32(0))
    above = utils.shl_u32(cutlass.Uint32(0xFFFFFFFF), cutlass.Uint32(n_above))
    result = cute.make_rmem_tensor(1, dtype=cutlass.Uint32)
    result[0] = below & above
    return result.load()


@cute.jit
def cute_prefix_lm_mask_vec(
    batch: cute.TensorSSA,
    head: cute.TensorSSA,
    m_idx: cute.TensorSSA,
    n_idx: cute.TensorSSA,
    seqlen_info,
    aux_tensors,
) -> cute.TensorSSA:
    prefix = cutlass.Int32(512)
    hi_pref = prefix - n_idx[0]
    m_below_pref = max(cutlass.Int32(32) - hi_pref, cutlass.Int32(0))
    term1_below = utils.shr_u32(cutlass.Uint32(0xFFFFFFFF), cutlass.Uint32(m_below_pref))
    row_in_prefix_mask = (
        cutlass.Uint32(0xFFFFFFFF) if m_idx[0] < prefix else cutlass.Uint32(0)
    )
    term1 = term1_below & row_in_prefix_mask
    hi_causal = m_idx[0] - n_idx[0] + cutlass.Int32(1)
    m_below_causal = max(cutlass.Int32(32) - hi_causal, cutlass.Int32(0))
    term2 = utils.shr_u32(cutlass.Uint32(0xFFFFFFFF), cutlass.Uint32(m_below_causal))
    result = cute.make_rmem_tensor(1, dtype=cutlass.Uint32)
    result[0] = term1 | term2
    return result.load()


# =============================================================================
# Packed-bitmask aux tensor mod
# =============================================================================
# aux[0] is a (batch, max_seqlen_q, ceil(max_seqlen_k / 32)) Uint32 tensor where
# bit k of packed[b, q, c] is the mask for (b, q, c*32 + k).


@cute.jit
def cute_packed_mask_aux(
    batch: cute.TensorSSA,
    head: cute.TensorSSA,
    m_idx: cute.TensorSSA,
    n_idx: cute.TensorSSA,
    seqlen_info,
    aux_tensors,
) -> cute.TensorSSA:
    packed = aux_tensors[0]
    val = packed[batch[0], m_idx[0], n_idx[0] // cutlass.Int32(32)]
    shift = cutlass.Uint32(n_idx[0] % cutlass.Int32(32))
    bit_set = cutlass.Boolean(utils.shr_u32(val, shift) & cutlass.Uint32(1))
    result = cute.make_rmem_tensor(n_idx.shape, dtype=cutlass.Boolean)
    for j in cutlass.range_constexpr(cute.size(n_idx.shape)):
        result[j] = bit_set
    return result.load()


def get_cute_packed_mask_aux_vec(vec_size: int):
    """Vec packed-mask, specialized for `vec_size`. For vec_size > 32, requires
    `aux_tensors[0].__assumed_align__ = 16` and num_words divisible by 4."""
    if vec_size <= 32:

        @cute.jit
        def _mod(
            batch: cute.TensorSSA,
            head: cute.TensorSSA,
            m_idx: cute.TensorSSA,
            n_idx: cute.TensorSSA,
            seqlen_info,
            aux_tensors,
        ) -> cute.TensorSSA:
            packed = aux_tensors[0]
            base = n_idx[0] // cutlass.Int32(32)
            val = packed[batch[0], m_idx[0], base]
            shift = cutlass.Uint32(n_idx[0] % cutlass.Int32(32))
            result = cute.make_rmem_tensor(1, dtype=cutlass.Uint32)
            result[0] = utils.shr_u32(val, shift)
            return result.load()
    else:
        num_words = vec_size // 32

        @cute.jit
        def _mod(
            batch: cute.TensorSSA,
            head: cute.TensorSSA,
            m_idx: cute.TensorSSA,
            n_idx: cute.TensorSSA,
            seqlen_info,
            aux_tensors,
        ) -> cute.TensorSSA:
            packed = aux_tensors[0]
            b_str, m_str, _ = packed.stride
            packed_aligned = cute.make_tensor(
                packed.iterator,
                cute.make_layout(
                    packed.shape,
                    stride=(
                        cute.assume(b_str, divby=4),
                        cute.assume(m_str, divby=4),
                        1,
                    ),
                ),
            )
            packed_row = packed_aligned[batch[0], m_idx[0], None]
            packed_tiled = cute.flat_divide(packed_row, (num_words,))
            base = n_idx[0] // cutlass.Int32(32)
            packed_chunk = packed_tiled[None, base // cutlass.Int32(num_words)]
            loaded = cute.make_rmem_tensor_like(packed_chunk)
            cute.autovec_copy(packed_chunk, loaded)
            result = cute.make_rmem_tensor(num_words, dtype=cutlass.Uint32)
            for k in cutlass.range_constexpr(num_words):
                result[k] = cutlass.Uint32(loaded[k])
            return result.load()

    return _mod


def make_packed_mask_aux_tensor(
    batch: int,
    seqlen_q: int,
    seqlen_k: int,
    density: float = 0.5,
    device="cuda",
    seed: int = 0,
):
    """Random Uint32 bit-packed mask. num_words is rounded up to a multiple of 4
    so each row is 16-byte aligned (LDG.E.128 requirement at vec_size=128)."""
    g = torch.Generator(device=device).manual_seed(seed)
    num_words = ((seqlen_k + 31) // 32 + 3) // 4 * 4
    bools = (
        torch.rand(batch, seqlen_q, num_words * 32, device=device, generator=g)
        < density
    )
    bools = bools.reshape(batch, seqlen_q, num_words, 32)
    powers = 1 << torch.arange(32, device=device, dtype=torch.int64)
    packed = (bools.to(torch.int64) * powers).sum(-1).to(torch.uint32)
    packed.__assumed_align__ = 16
    return packed


VEC_MASK_FACTORIES = {
    "causal": ("factory", get_cute_causal_mask_vec),
    "block_causal": ("factory", get_cute_causal_mask_vec),
    "sliding_window": ("factory_window", get_cute_sliding_window_mask_vec),
    "block_diagonal": ("static", cute_block_diagonal_mask_vec),
    "prefix_lm": ("static", cute_prefix_lm_mask_vec),
    "packed_aux": ("factory_vec_size", get_cute_packed_mask_aux_vec),
}


# Scalar mods for vec masks not in STATIC_MASKS / PARAMETERIZED_MASK_FACTORIES.
EXTRA_SCALAR_MASKS = {
    "packed_aux": cute_packed_mask_aux,
}


def get_vec_mask(
    mask_name, seqlen_q=None, seqlen_k=None, window_size=None, vec_size=None
):
    """Return a vectorized cute mask callable for `mask_name`, or None if there
    is no vec form. Caller sets `__vec_size__` on the returned callable.
    `vec_size` is required for masks whose body specializes on it (packed_aux)."""
    if mask_name not in VEC_MASK_FACTORIES:
        return None
    kind, obj = VEC_MASK_FACTORIES[mask_name]
    if kind == "static":
        return obj
    if kind == "factory_vec_size":
        if vec_size is None:
            raise ValueError(f"{mask_name} vec mask requires vec_size")
        return obj(vec_size)
    offset = (
        (seqlen_k - seqlen_q) if (seqlen_q is not None and seqlen_k is not None) else 0
    )
    if kind == "factory":
        return obj(offset)
    if kind == "factory_window":
        if window_size is None:
            raise ValueError("sliding_window vec mask requires window_size")
        return obj(window_size, window_size, offset)
    raise ValueError(f"unknown vec mask kind: {kind}")


# =============================================================================
# Mask registry and factory functions
# =============================================================================


STATIC_MASKS = {
    "block_diagonal": (cute_block_diagonal_mask, flex_block_diagonal_mask),
    "mini_causal": (cute_mini_causal_mask, flex_mini_causal_mask),
    "prefix_lm": (cute_prefix_lm_mask, flex_prefix_lm_mask),
    "dilated_sliding_window": (
        cute_dilated_sliding_window_mask,
        flex_dilated_sliding_window_mask,
    ),
    "document": (cute_document_mask, flex_document_mask),
    "ima": (cute_ima_mask, flex_ima_mask),
}

PARAMETERIZED_MASK_FACTORIES = {
    "causal": (get_cute_causal_mask, get_flex_causal_mask),
    "block_causal": (get_cute_block_causal_mask, get_flex_block_causal_mask),
    "sliding_window": (get_cute_sliding_window_mask, get_flex_sliding_window_mask),
}


def get_mask_pair(mask_name, seqlen_q=None, seqlen_k=None, window_size=None):
    """Get (cute_mask, flex_mask) pair for the given mask name.

    For static masks, seqlen info is not needed.
    For parameterized masks, seqlen_q and seqlen_k are required.
    """
    if mask_name in STATIC_MASKS:
        return STATIC_MASKS[mask_name]

    if mask_name not in PARAMETERIZED_MASK_FACTORIES:
        raise ValueError(f"Unknown mask: {mask_name}")

    if seqlen_q is None or seqlen_k is None:
        raise ValueError(
            f"Parameterized mask '{mask_name}' requires seqlen_q and seqlen_k"
        )

    cute_factory, flex_factory = PARAMETERIZED_MASK_FACTORIES[mask_name]
    offset = seqlen_k - seqlen_q

    if mask_name == "sliding_window":
        if window_size is None:
            raise ValueError("sliding_window mask requires window_size parameter")
        cute_mask = cute_factory(window_size, window_size, offset)
        flex_mask = flex_factory(window_size, window_size, offset)
    else:
        cute_mask = cute_factory(offset)
        flex_mask = flex_factory(offset)

    return cute_mask, flex_mask


if __name__ == "__main__":
    doc_ids = random_doc_id_tensor(1, 2, 128)
    print(f"{doc_ids = }")
