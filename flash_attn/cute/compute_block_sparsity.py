import math
import operator
from typing import Callable, Type, Optional, Tuple

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, const_expr, Int32, Boolean

from block_sparsity import BlockSparseTensors
import utils

class BlockSparsityKernel:
    def __init__(
        self,
        tile_mn: Tuple[int, int],
        compute_full_blocks: bool = True,
    ):
        self.tile_mn = tile_mn 
        self.compute_full_blocks = compute_full_blocks 
        
    @cute.jit
    def __call__(
        self,
        mask_mod: Callable,
        blocksparse_tensors: BlockSparseTensors,
        aux_tensors: Optional[list] = None,
    ):
        self.mask_cnt, self.mask_idx, self.full_cnt, self.full_idx = blocksparse_tensors 
        
        self.mask_mod = mask_mod
        
        if const_expr(self.compute_full_blocks):
            assert self.full_cnt is not None and self.full_idx is not None, "full block tensors must be provided when computing full blocks"
            
        # TODO: statick checks for size of blocksparse tensors
        batch_size, num_heads, m_block = list(self.mask_cnt.shape)
        grid = [m_block, num_heads, batch_size]
        num_threads = self.tile_mn[0]
        self.num_m_blocks = m_block
        self.num_n_blocks = self.mask_idx.shape[3]
        
        
        self.kernel(
            self.mask_cnt,
            self.mask_idx,
            self.full_cnt,
            self.full_idx,
            aux_tensors,
        ).launch(
            grid=grid,
            block=[num_threads, 1, 1],
        )
        
        
        
    @cute.kernel 
    def kernel(
        self,
        mask_cnt: cute.Tensor,
        mask_idx: cute.Tensor,
        full_cnt: cute.Tensor,
        full_idx: cute.Tensor,
        aux_tensors: Optional[list] = None,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        m_block, head_idx, batch_idx = cute.arch.block_idx()
        curr_mask_cnt = mask_cnt[batch_idx, head_idx, m_block]
        curr_mask_idx = mask_idx[batch_idx, head_idx, m_block, None]
        curr_full_cnt = full_cnt[batch_idx, head_idx, m_block]
        curr_full_idx = full_idx[batch_idx, head_idx, m_block, None]
        
        num_mask_blocks = Int32(0)
        num_full_blocks = Int32(0)
        
        for n_block in cutlass.range(self.num_n_blocks, unroll_full=True):
            is_masked, is_full = self.check_block_full_or_mask(
                tidx,
                batch_idx,
                head_idx,
                m_block,
                n_block,
                aux_tensors,
            )
            if is_masked:
                curr_mask_idx[num_mask_blocks] = n_block
                num_mask_blocks += 1
            elif is_full:
                curr_full_idx[num_full_blocks] = n_block
                num_full_blocks += 1
        
        curr_mask_cnt = num_mask_blocks
        curr_full_cnt = num_full_blocks
        
    @cute.jit
    def check_block_full_or_mask(
        self,
        tidx,
        batch_idx,
        head_idx,
        m_block,
        n_block,
        aux_tensors,
    ) -> Tuple[Boolean, Boolean]:
        is_masked = Boolean(False)
        is_full = Boolean(False)
        
        # TODO: compute global index
        nrow = const_expr(cute.size(tScS_mn.shape[0]))
        ncol = const_expr(cute.size(tScS_mn.shape[1]))
        thr_col_offset = tScS_mn[0, 0][1]

        for r in cutlass.range_constexpr(nrow):
            for col in cutlass.range_constexpr(ncol):
                batch_idx_ssa = utils.scalar_to_ssa(batch_idx, cutlass.Int32)
                head_idx_ssa = utils.scalar_to_ssa(head_idx, cutlass.Int32)
                q_idx_ssa = utils.scalar_to_ssa(
                    tScS_mn[r, 0][0] + m_block * self.tile_m, cutlass.Int32
                )
                kv_idx_ssa = utils.scalar_to_ssa(
                    thr_col_offset + t0ScS_mn[0, col][1] + n_block * self.tile_n,
                    cutlass.Int32,
                )
                # TODO: oob check
                oob = False
                if oob:
                    is_masked = Boolean(True)
                    is_full = Boolean(False)
                else:
                    mask_value = self.mask_mod(
                        batch_idx_ssa,
                        head_idx_ssa,
                        q_idx_ssa,
                        kv_idx_ssa,
                        aux_tensors,
                    )
                    cond = cutlass.Boolean(utils.ssa_to_scalar(mask_value))
                    is_masked = is_masked or cond
                    is_full = is_full and cond
        pass
        
        return is_masked, is_full