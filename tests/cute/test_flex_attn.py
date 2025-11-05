from dataclasses import dataclass
import operator
from typing import Callable, Optional, Tuple

import cutlass
from cutlass._mlir.dialects import math as mlir_math
import cutlass.cute as cute
import pytest
import torch
from torch.nn.attention.flex_attention import flex_attention, BlockMask

from flash_attn.cute.interface import _flash_attn_fwd


@dataclass
class CuteBlockSparsity:
    mask_block_cnt: cute.Tensor
    mask_block_idx: cute.Tensor
    full_block_cnt: cute.Tensor 
    full_block_idx: cute.Tensor
    
@dataclass 
class BlockSparsity:
    cute_block_sparsity: Optional[CuteBlockSparsity]
    flex_block_mask: Optional[BlockMask]
    
# helper methods

def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b

# score mod definitions

# cute score mods 

# flex score mods

# mask mod definitions


def create_blocked_document_mask(batch_size: int, nheads: int, seqlen_q: int, seqlen_k: int, tile_m: int, tile_n: int, doc_ids: torch.Tensor, device) -> Tuple[BlockSparsity, Callable]:
    """
    assume currently that doc_ids divide evenly across tiles, i.e. a tile is fully masked
    if and only if any one of its elements is.
    TODO: weaken this assumption
    """
    num_m_blocks = ceil_div(seqlen_q, tile_m)
    num_n_blocks = ceil_div(seqlen_k, tile_n)
    
    full_block_cnt = torch.zeros(batch_size, nheads, num_m_blocks, dtype=torch.int32, device=device)
    full_block_idx = torch.zeros(batch_size, nheads, num_m_blocks, num_n_blocks, dtype=torch.int32, device=device)
    mask_block_cnt = torch.zeros(batch_size, nheads, num_m_blocks, dtype=torch.int32, device=device)
    mask_block_idx = torch.zeros(batch_size, nheads, num_m_blocks, num_n_blocks, dtype=torch.int32, device=device)
    
    # TODO: compute block sparse tensors for document mask
    
    for batch_idx in range(batch_size):
        for head_idx in range(nheads):
            for i in range(num_m_blocks):
                global_m_idx = i * tile_m 
                n_block_cnt = 0
                for j in range(num_n_blocks):
                    global_n_idx = j * tile_n
                    if doc_ids[batch_idx, head_idx, global_m_idx] == doc_ids[batch_idx, head_idx, global_n_idx]:
                        full_block_idx[batch_idx, head_idx, i, n_block_cnt] = j 
                        n_block_cnt += 1
                full_block_cnt[batch_idx, head_idx, i] = n_block_cnt
    
    def document_mask_mod(batch, head, q_idx, k_idx, aux_tensors: list[torch.Tensor]):
        return doc_ids[batch, head, q_idx] == doc_ids[batch, head, k_idx]
    
    block_sparse_tensors = BlockSparsity(
        cute_block_sparsity=CuteBlockSparsity(
        mask_block_cnt=mask_block_cnt,
        mask_block_idx=mask_block_idx,
        full_block_cnt=full_block_cnt,
        full_block_idx=full_block_idx,
        ),
        flex_block_mask=None,
    )
    
    return (block_sparse_tensors, document_mask_mod)

def main():
    