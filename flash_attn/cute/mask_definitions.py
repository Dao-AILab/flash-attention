from typing import Callable, Optional

import random
import math

import cutlass
import cutlass.cute as cute
import torch


MaskModCallable = Optional[
    Callable[
        [
            "cutlass.Int32",
            "cutlass.Int32",
            "cutlass.Int32",
            "cutlass.Int32",
            "cutlass.Int32",
            "cutlass.Int32",
        ],
        "cutlass.Boolean",
    ]
]


# Flex Attention mask functions (PyTorch signatures for reference implementation)


def flex_identity_mask(b, h, q_idx, kv_idx, seqlen_q=None, seqlen_k=None):
    if torch.is_tensor(q_idx):
        return torch.ones_like(q_idx, dtype=torch.bool)
    return True


def flex_identity_partial_mask(b, h, q_idx, kv_idx, seqlen_q=None, seqlen_k=None):
    if torch.is_tensor(q_idx):
        return torch.ones_like(q_idx, dtype=torch.bool)
    return True


def flex_causal_mask(b, h, q_idx, kv_idx, seqlen_q=None, seqlen_k=None):
    # Right-aligned causal masking
    if seqlen_q is not None and seqlen_k is not None:
        offset = seqlen_k - seqlen_q
        return kv_idx <= q_idx + offset
    return kv_idx <= q_idx


def flex_block_causal_mask(b, h, q_idx, kv_idx, seqlen_q=None, seqlen_k=None):
    # Right-aligned causal masking
    if seqlen_q is not None and seqlen_k is not None:
        offset = seqlen_k - seqlen_q
        return kv_idx <= q_idx + offset
    return kv_idx <= q_idx


def create_flex_sliding_window_mask(window_size=1024):
    """Factory function to create a sliding window mask with configurable window size"""

    def flex_sliding_window_mask(b, h, q_idx, kv_idx, seqlen_q=None, seqlen_k=None):
        # Sliding window: q_idx - window_size <= kv_idx <= q_idx
        if seqlen_q is not None and seqlen_k is not None:
            offset = seqlen_k - seqlen_q
            return (kv_idx <= q_idx + offset) & (kv_idx >= q_idx + offset - window_size)
        return (kv_idx <= q_idx) & (kv_idx >= q_idx - window_size)

    return flex_sliding_window_mask


# Default sliding window mask with window_size=1024 for backward compatibility
def flex_sliding_window_mask(b, h, q_idx, kv_idx, seqlen_q=None, seqlen_k=None):
    window_size = 1024
    if seqlen_q is not None and seqlen_k is not None:
        offset = seqlen_k - seqlen_q
        # Sliding window: q_pos - window_size < kv_pos <= q_pos
        # Note: using strict inequality on the left to match typical sliding window behavior
        return (kv_idx <= q_idx + offset) & (kv_idx > q_idx + offset - window_size)
    return (kv_idx <= q_idx) & (kv_idx > q_idx - window_size)


def flex_block_diagonal_mask(b, h, q_idx, kv_idx, seqlen_q=None, seqlen_k=None, block_size=64):
    return (q_idx // block_size) == (kv_idx // block_size)


def flex_mini_causal_mask(b, h, q_idx, kv_idx, seqlen_q=None, seqlen_k=None):
    return (q_idx % 128) >= (kv_idx % 128)


def flex_half_identity_mask(b, h, q_idx, kv_idx, seqlen_q=None, seqlen_k=None):
    """Even k-blocks are full blocks, odd k-blocks are masked blocks (both return True)"""
    if torch.is_tensor(kv_idx):
        return torch.ones_like(kv_idx, dtype=torch.bool)
    return True


def flex_document_mask(b, h, q_idx, kv_idx, doc_id: torch.Tensor):
    return doc_id[b, h, q_idx] == doc_id[b, h, kv_idx]


# CuTe versions for kernel compilation


@cute.jit
def cute_identity_mask(
    batch: cutlass.Int32,
    head: cutlass.Int32,
    m_idx: cutlass.Int32,
    n_idx: cutlass.Int32,
    seqlen_q: cutlass.Int32,
    seqlen_k: cutlass.Int32,
    aux_tensors: None,
) -> cutlass.Boolean:
    return cutlass.Boolean(True)


@cute.jit
def cute_identity_partial_mask(
    batch: cutlass.Int32,
    head: cutlass.Int32,
    m_idx: cutlass.Int32,
    n_idx: cutlass.Int32,
    seqlen_q: cutlass.Int32,
    seqlen_k: cutlass.Int32,
    aux_tensors: None,
) -> cutlass.Boolean:
    return cutlass.Boolean(True)


@cute.jit
def cute_causal_mask(
    batch: cutlass.Int32,
    head: cutlass.Int32,
    m_idx: cutlass.Int32,
    n_idx: cutlass.Int32,
    seqlen_q: cutlass.Int32,
    seqlen_k: cutlass.Int32,
    aux_tensors: None,
) -> cutlass.Boolean:
    # Right-aligned causal masking
    offset = seqlen_k - seqlen_q
    return cutlass.Boolean(n_idx <= m_idx + offset)


@cute.jit
def cute_block_causal_mask(
    batch: cutlass.Int32,
    head: cutlass.Int32,
    m_idx: cutlass.Int32,
    n_idx: cutlass.Int32,
    seqlen_q: cutlass.Int32,
    seqlen_k: cutlass.Int32,
    aux_tensors: None,
) -> cutlass.Boolean:
    # Right-aligned causal masking
    offset = seqlen_k - seqlen_q
    return cutlass.Boolean(n_idx <= m_idx + offset)


def create_cute_sliding_window_mask(window_size=1024):
    """Factory function to create a CuTe sliding window mask with configurable window size"""

    @cute.jit
    def cute_sliding_window_mask(
        batch: cutlass.Int32,
        head: cutlass.Int32,
        m_idx: cutlass.Int32,
        n_idx: cutlass.Int32,
        seqlen_q: cutlass.Int32,
        seqlen_k: cutlass.Int32,
        aux_tensors,
    ) -> cutlass.Boolean:
        offset = seqlen_k - seqlen_q

        return cutlass.Boolean(
            (n_idx <= m_idx + offset) and (n_idx >= m_idx + offset - window_size)
        )

    return cute_sliding_window_mask


# Default sliding window mask with window_size=1024 for backward compatibility
@cute.jit
def cute_sliding_window_mask(
    batch: cutlass.Int32,
    head: cutlass.Int32,
    m_idx: cutlass.Int32,
    n_idx: cutlass.Int32,
    seqlen_q: cutlass.Int32,
    seqlen_k: cutlass.Int32,
    aux_tensors,
) -> cutlass.Boolean:
    window_size = 1024
    # offset = seqlen_k - seqlen_q
    offset = 0
    return cutlass.Boolean((n_idx <= m_idx + offset) and (n_idx >= m_idx + offset - window_size))


@cute.jit
def cute_document_mask(
    batch: cutlass.Int32,
    head: cutlass.Int32,
    m_idx: cutlass.Int32,
    n_idx: cutlass.Int32,
    seqlen_q: cutlass.Int32,
    seqlen_k: cutlass.Int32,
    aux_tensors: list,
):
    doc_id = aux_tensors[0]
    return cutlass.Boolean(doc_id[batch, head, m_idx] == doc_id[batch, head, n_idx])


@cute.jit
def cute_block_diagonal_mask(
    batch: cutlass.Int32,
    head: cutlass.Int32,
    m_idx: cutlass.Int32,
    n_idx: cutlass.Int32,
    seqlen_q: cutlass.Int32,
    seqlen_k: cutlass.Int32,
    aux_tensors,
) -> cutlass.Boolean:
    return cutlass.Boolean((m_idx // 64) == (n_idx // 64))


@cute.jit
def cute_mini_causal_mask(
    batch: cutlass.Int32,
    head: cutlass.Int32,
    m_idx: cutlass.Int32,
    n_idx: cutlass.Int32,
    seqlen_q: cutlass.Int32,
    seqlen_k: cutlass.Int32,
    aux_tensors,
) -> cutlass.Boolean:
    """Each tile is locally causal-masked"""
    m_mod = m_idx % 128
    n_mod = n_idx % 128
    return cutlass.Boolean(m_mod >= n_mod)


@cute.jit
def cute_half_identity_mask(
    batch: cutlass.Int32,
    head: cutlass.Int32,
    m_idx: cutlass.Int32,
    n_idx: cutlass.Int32,
    seqlen_q: cutlass.Int32,
    seqlen_k: cutlass.Int32,
) -> cutlass.Boolean:
    return cutlass.Boolean(True)


def random_doc_id_tensor(nheads, batch, seqlen_q, device="cpu"):
    doc_ids_tensor = torch.zeros(batch, nheads, seqlen_q, dtype=torch.int32, device=device)
    for b in range(batch):
        for h in range(nheads):
            N = seqlen_q
            n = random.randint(1, math.ceil(math.sqrt(N // 4)))
            cuts = sorted(random.sample(range(1, N), n - 1))
            lengths = [b - a for a, b in zip((0, *cuts), (*cuts, N))]

            doc_ids = []
            for i, length in enumerate(lengths):
                doc_ids += [i for _ in range(length)]

            doc_ids_tensor[b, h, :] = torch.tensor(doc_ids, dtype=torch.int32, device=device)
    print(f"{doc_ids_tensor.shape = }")
    return doc_ids_tensor


MASK_FUNCTIONS = {
    "identity": (cute_identity_mask, flex_identity_mask),
    "identity_partial": (cute_identity_partial_mask, flex_identity_partial_mask),
    "causal": (cute_causal_mask, flex_causal_mask),
    "block_causal": (cute_block_causal_mask, flex_block_causal_mask),
    "sliding_window": (cute_sliding_window_mask, flex_sliding_window_mask),
    "block_diagonal": (cute_block_diagonal_mask, flex_block_diagonal_mask),
    "mini_causal": (cute_mini_causal_mask, flex_mini_causal_mask),
    "half_identity": (cute_half_identity_mask, flex_half_identity_mask),
    "document": (cute_document_mask, flex_document_mask),
}

if __name__ == "__main__":
    doc_ids = random_doc_id_tensor(1, 2, 128)
    print(f"{doc_ids = }")
