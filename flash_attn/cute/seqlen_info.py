from typing import Optional
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import Int32, const_expr

"""
This consolidates all the info related to sequence length. This is so that we can do all
the gmem reads once at the beginning of each tile, rather than having to repeat these reads
to compute various things like n_block_min, n_block_max, etc.
"""


@dataclass(frozen=True)
class SeqlenInfo:
    offset: cutlass.Int32
    seqlen: cutlass.Int32

    @staticmethod
    def create(
        batch_idx: cutlass.Int32,
        seqlen_static: cutlass.Int32,
        cu_seqlens: Optional[cute.Tensor] = None,
        seqused: Optional[cute.Tensor] = None,
    ):
        offset = 0 if const_expr(cu_seqlens is None) else cu_seqlens[batch_idx]
        if const_expr(seqused is not None):
            seqlen = seqused[batch_idx]
        elif const_expr(cu_seqlens is not None):
            seqlen = cu_seqlens[batch_idx + 1] - cu_seqlens[batch_idx]
        else:
            seqlen = seqlen_static
        return SeqlenInfo(offset, seqlen)


@dataclass(frozen=True)
class SeqlenInfoQK:
    offset_q: cutlass.Int32
    offset_k: cutlass.Int32
    seqlen_q: cutlass.Int32
    seqlen_k: cutlass.Int32
    has_cu_seqlens_q: cutlass.Constexpr[bool]
    has_cu_seqlens_k: cutlass.Constexpr[bool]

    @staticmethod
    def create(
        batch_idx: cutlass.Int32,
        seqlen_q_static: cutlass.Int32,
        seqlen_k_static: cutlass.Int32,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
    ):
        offset_q = 0 if const_expr(mCuSeqlensQ is None) else mCuSeqlensQ[batch_idx]
        offset_k = 0 if const_expr(mCuSeqlensK is None) else mCuSeqlensK[batch_idx]
        if const_expr(mSeqUsedQ is not None):
            seqlen_q = mSeqUsedQ[batch_idx]
        else:
            seqlen_q = (
                seqlen_q_static
                if const_expr(mCuSeqlensQ is None)
                else mCuSeqlensQ[batch_idx + 1] - offset_q
            )
        if const_expr(mSeqUsedK is not None):
            seqlen_k = mSeqUsedK[batch_idx]
        else:
            seqlen_k = (
                seqlen_k_static
                if const_expr(mCuSeqlensK is None)
                else mCuSeqlensK[batch_idx + 1] - offset_k
            )
        has_cu_seqlens_q: int = mCuSeqlensQ is not None
        has_cu_seqlens_k: int = mCuSeqlensK is not None
        return SeqlenInfoQK(
            offset_q, offset_k, seqlen_q, seqlen_k, has_cu_seqlens_q, has_cu_seqlens_k
        )

    def offset_batch_Q(self, mQ: cute.Tensor, batch_idx: Int32, dim: int) -> cute.Tensor:
        """Seqlen must be the first dimension of mQ"""
        if const_expr(not self.has_cu_seqlens_q):
            idx = (None,) * dim + (batch_idx,) + (None,) * (cute.rank(mQ) - 1 - dim)
            return mQ[idx]
        else:
            offset = (
                self.offset_q if const_expr(cute.rank(mQ.shape[0]) == 1) else (0, self.offset_q)
            )
            idx = (offset,) + (0,) * (cute.rank(mQ) - 1)
            return cute.domain_offset(idx, mQ)

    def offset_batch_K(self, mK: cute.Tensor, batch_idx: Int32, dim: int) -> cute.Tensor:
        """Seqlen must be the first dimension of mK"""
        if const_expr(not self.has_cu_seqlens_k):
            idx = (None,) * dim + (batch_idx,) + (None,) * (cute.rank(mK) - 1 - dim)
            return mK[idx]
        else:
            idx = (self.offset_k,) + (0,) * (cute.rank(mK) - 1)
            return cute.domain_offset(idx, mK)
