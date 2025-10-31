from typing import Callable, Optional

import random
import math

import cutlass
import cutlass.cute as cute
import torch

from flash_attn.cute import utils


MaskModCallable = Optional[
    Callable[
        [
            "cute.TensorSSA",
            "cute.TensorSSA",
            "cute.TensorSSA",
            "cute.TensorSSA",
            "Optional[list]",
        ],
        "cute.TensorSSA",
    ]
]


# Flex Attention mask functions (PyTorch signatures for reference implementation)
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
    block_size = 64
    return (q_idx // block_size) == (kv_idx // block_size)


def flex_mini_causal_mask(b, h, q_idx, kv_idx):
    return (q_idx % 128) >= (kv_idx % 128)


def flex_document_mask(b, h, q_idx, kv_idx, doc_id):
    return doc_id[b, h, q_idx] == doc_id[b, h, kv_idx]


# CuTe versions for kernel compilation
def get_cute_causal_mask(offset: int):
    @cute.jit
    def _cute_causal_mask(
        batch: cute.TensorSSA,
        head: cute.TensorSSA,
        m_idx: cute.TensorSSA,
        n_idx: cute.TensorSSA,
        aux_tensors: None,
    ) -> cute.TensorSSA:
        offset_ssa = utils.scalar_to_ssa(offset, cutlass.Int32)
        return n_idx <= (m_idx + offset_ssa)

    return _cute_causal_mask


def get_cute_block_causal_mask(offset: int):
    @cute.jit
    def _cute_block_causal_mask(
        batch: cute.TensorSSA,
        head: cute.TensorSSA,
        m_idx: cute.TensorSSA,
        n_idx: cute.TensorSSA,
        aux_tensors: None,
    ) -> cute.TensorSSA:
        offset_ssa = utils.scalar_to_ssa(offset, cutlass.Int32)
        return n_idx <= (m_idx + offset_ssa)

    return _cute_block_causal_mask


def get_cute_sliding_window_mask(window_left: int, window_right: int, offset: int):
    @cute.jit
    def _cute_sliding_window_mask(
        batch: cute.TensorSSA,
        head: cute.TensorSSA,
        m_idx: cute.TensorSSA,
        n_idx: cute.TensorSSA,
        aux_tensors,
    ) -> cute.TensorSSA:
        offset_ssa = utils.scalar_to_ssa(offset, cutlass.Int32)
        window_left_ssa = utils.scalar_to_ssa(window_left, cutlass.Int32)
        window_right_ssa = utils.scalar_to_ssa(window_right, cutlass.Int32)
        center = m_idx + offset_ssa
        lower = center - window_left_ssa
        upper = center + window_right_ssa
        return (n_idx >= lower) & (n_idx <= upper)

    return _cute_sliding_window_mask


@cute.jit
def cute_document_mask(
    batch: cute.TensorSSA,
    head: cute.TensorSSA,
    m_idx: cute.TensorSSA,
    n_idx: cute.TensorSSA,
    aux_tensors: list,
) -> cute.TensorSSA:
    doc_id = aux_tensors[0]
    m_doc = utils.scalar_to_ssa(doc_id[batch[0], head[0], m_idx[0]], cutlass.Int32)
    n_doc = utils.scalar_to_ssa(doc_id[batch[0], head[0], n_idx[0]], cutlass.Int32)
    return m_doc == n_doc


@cute.jit
def cute_block_diagonal_mask(
    batch: cute.TensorSSA,
    head: cute.TensorSSA,
    m_idx: cute.TensorSSA,
    n_idx: cute.TensorSSA,
    aux_tensors,
) -> cute.TensorSSA:
    block_size_ssa = utils.scalar_to_ssa(64, cutlass.Int32)
    return (m_idx // block_size_ssa) == (n_idx // block_size_ssa)


@cute.jit
def cute_mini_causal_mask(
    batch: cute.TensorSSA,
    head: cute.TensorSSA,
    m_idx: cute.TensorSSA,
    n_idx: cute.TensorSSA,
    aux_tensors,
) -> cute.TensorSSA:
    tile_size_ssa = utils.scalar_to_ssa(128, cutlass.Int32)
    m_mod = m_idx % tile_size_ssa
    n_mod = n_idx % tile_size_ssa
    return m_mod >= n_mod


def random_doc_id_tensor(nheads, batch, seqlen_q, device="cpu"):
    doc_ids_tensor = torch.zeros(batch, nheads, seqlen_q, dtype=torch.int32, device=device)
    for b in range(batch):
        for h in range(nheads):
            N = seqlen_q
            max_segments = max(1, math.ceil(math.sqrt(max(N // 4, 1))))
            n = random.randint(1, max_segments)
            n = min(n, N)
            cuts = sorted(random.sample(range(1, N), n - 1))
            lengths = [b - a for a, b in zip((0, *cuts), (*cuts, N))]

            doc_ids = []
            for i, length in enumerate(lengths):
                doc_ids += [i for _ in range(length)]

            doc_ids_tensor[b, h, :] = torch.tensor(doc_ids, dtype=torch.int32, device=device)
    return doc_ids_tensor


STATIC_MASKS = {
    "block_diagonal": (cute_block_diagonal_mask, flex_block_diagonal_mask),
    "mini_causal": (cute_mini_causal_mask, flex_mini_causal_mask),
    "document": (cute_document_mask, flex_document_mask),
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
        raise ValueError(f"Parameterized mask '{mask_name}' requires seqlen_q and seqlen_k")

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
