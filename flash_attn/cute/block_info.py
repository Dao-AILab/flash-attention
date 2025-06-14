from typing import Tuple

import cutlass
import cutlass.cute as cute

from flash_attn.cute.seqlen_info import SeqlenInfo


class BlockInfo:

    def __init__(
        self,
        m_block_size: cutlass.Constexpr[int],
        n_block_size: cutlass.Constexpr[int],
        is_causal: cutlass.Constexpr[bool],
        qhead_per_kvhead_packgqa: cutlass.Constexpr[int] = 1,  # Only pass in if we're doing PackGQA
        *,
        loc=None,
        ip=None
    ):
        self.m_block_size: cutlass.Constexpr[int] = m_block_size
        self.n_block_size: cutlass.Constexpr[int] = n_block_size
        self.is_causal: cutlass.Constexpr[bool] = is_causal
        self.qhead_per_kvhead_packgqa: cutlass.Constexpr[int] = qhead_per_kvhead_packgqa
        self._loc = loc

    @cute.jit
    def get_n_block_min_max(
        self, seqlen_info: SeqlenInfo, m_block: cutlass.Int32
    ) -> Tuple[cutlass.Int32, cutlass.Int32]:
        n_block_max = cute.ceil_div(seqlen_info.seqlen_k, self.n_block_size)
        n_block_min = 0
        if cutlass.const_expr(self.is_causal):
            m_idx_max = (m_block + 1) * self.m_block_size
            if cutlass.const_expr(self.qhead_per_kvhead_packgqa > 1):
                m_idx_max = (m_idx_max - 1) // self.qhead_per_kvhead_packgqa + 1
            n_idx = m_idx_max + seqlen_info.seqlen_k - seqlen_info.seqlen_q
            n_idx_right = n_idx
            n_block_max = min(cute.ceil_div(n_idx_right, self.n_block_size), n_block_max)
        return n_block_min, n_block_max

    @cute.jit
    def get_n_block_min_causal_local_mask(
        self,
        seqlen_info: SeqlenInfo,
        m_block: cutlass.Int32,
        n_block_min: cutlass.Int32,
    ) -> cutlass.Int32:
        m_idx_min = m_block * self.m_block_size
        if cutlass.const_expr(self.qhead_per_kvhead_packgqa > 1):
            m_idx_min = m_idx_min // self.qhead_per_kvhead_packgqa
        n_idx = m_idx_min + seqlen_info.seqlen_k - seqlen_info.seqlen_q
        n_idx_right = n_idx
        return cutlass.max(n_block_min, n_idx_right // self.n_block_size)

    def __extract_mlir_values__(self):
        # We just create a dummy value. Otherwise unpack_to_irvalue in cutlass.py will complain
        return [cutlass.Int32(0).ir_value()]

    def __new_from_mlir_values__(self, values):
        return BlockInfo(self.m_block_size, self.n_block_size, self.is_causal, self.qhead_per_kvhead_packgqa, loc=self._loc)
