# Copyright (c) 2025, Tri Dao.

from dataclasses import dataclass
from typing import Union, Tuple

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync


from quack import layout_utils
import flash_attn.cute.utils as utils


def pack_gqa_layout(T, qhead_per_kvhead, nheads_kv, head_idx):
    """Reshape a tensor to fold qhead_per_kvhead into the seqlen dimension (mode 0).

    The head dimension is at mode ``head_idx``.  Modes before it (1..head_idx-1)
    are kept as-is (e.g. headdim for Q/O tensors), and modes after it are kept
    as-is (e.g. batch).

    For Q/O tensors (head_idx=2):
        (seqlen_q, headdim, nheads, batch, ...) -> ((qhead_per_kvhead, seqlen_q), headdim, nheads_kv, batch, ...)
    For LSE tensors (head_idx=1):
        (seqlen_q, nheads, batch, ...) -> ((qhead_per_kvhead, seqlen_q), nheads_kv, batch, ...)
    """
    head_stride = T.stride[head_idx]
    shape_packed = (
        (qhead_per_kvhead, T.shape[0]),
        *[T.shape[i] for i in range(1, head_idx)],
        nheads_kv,
        *[T.shape[i] for i in range(head_idx + 1, len(T.shape))],
    )
    stride_packed = (
        (head_stride, T.stride[0]),
        *[T.stride[i] for i in range(1, head_idx)],
        head_stride * qhead_per_kvhead,
        *[T.stride[i] for i in range(head_idx + 1, len(T.shape))],
    )
    return cute.make_tensor(T.iterator, cute.make_layout(shape_packed, stride=stride_packed))


def make_packgqa_tiled_tma_atom(
    op: cute.atom.CopyOp,
    gmem_tensor: cute.Tensor,
    smem_layout: Union[cute.Layout, cute.ComposedLayout],
    cta_tiler: Tuple[int, int],
    qhead_per_kvhead: int,
    head_idx: int,
):
    # This packing and unpacking of the layout is so that we keep the same TMA dimension as usual.
    # e.g. for (seqlen, d, nheads, b) layout, we still have 4D TMA after packing to
    # ((nheads, seqlen), d, b).
    # If we instead pack directly to ((qhead_per_kvhead, seqlen), d, nheads_kv, b) we'd have 5D TMA.
    # Pack headdim and seqlen dim into 1: (seqlen, d, nheads, b) -> ((nheads, seqlen), d, b)
    gmem_tensor = layout_utils.select(
        gmem_tensor, [head_idx, *range(head_idx), *range(head_idx + 1, cute.rank(gmem_tensor))]
    )
    gmem_tensor = cute.group_modes(gmem_tensor, 0, 2)
    assert cta_tiler[0] % qhead_per_kvhead == 0, (
        "CTA tile size in the seqlen dimension must be divisible by qhead_per_kvhead"
    )
    tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(
        op,
        gmem_tensor,
        smem_layout,
        ((qhead_per_kvhead, cta_tiler[0] // qhead_per_kvhead), cta_tiler[1]),  # No mcast
    )
    # Unpack from ((nheads, seqlen), d, b) -> ((qhead_per_kvhead, seqlen), d, nheads_kv, b)
    T = tma_tensor
    shape_packed = (
        (qhead_per_kvhead, T.shape[0][1]),
        *[T.shape[i] for i in range(1, head_idx)],
        T.shape[0][0] // qhead_per_kvhead,
        *[T.shape[i] for i in range(head_idx, len(T.shape))],
    )
    stride_packed = (
        *[T.stride[i] for i in range(head_idx)],
        T.stride[0][0] * qhead_per_kvhead,
        *[T.stride[i] for i in range(head_idx, len(T.shape))],
    )
    tma_tensor = cute.make_tensor(T.iterator, cute.make_layout(shape_packed, stride=stride_packed))
    return tma_atom, tma_tensor


def unpack_gqa_layout(T, qhead_per_kvhead, head_idx):
    """Reverse of pack_gqa_layout: unfold qhead_per_kvhead from the seqlen dimension (mode 0).

    The head dimension is at mode ``head_idx``.  Modes before it (1..head_idx-1)
    are kept as-is (e.g. headdim for Q/O tensors), and modes after it are kept
    as-is (e.g. batch).

    For Q/O tensors (head_idx=2):
        ((qhead_per_kvhead, seqlen_q), headdim, nheads_kv, batch, ...) -> (seqlen_q, headdim, nheads, batch, ...)
    For LSE tensors (head_idx=1):
        ((qhead_per_kvhead, seqlen_q), nheads_kv, batch, ...) -> (seqlen_q, nheads, batch, ...)
    """
    seqlen_stride = T.stride[0][1]
    head_stride = T.stride[0][0]
    shape_unpacked = (
        T.shape[0][1],
        *[T.shape[i] for i in range(1, head_idx)],
        T.shape[head_idx] * qhead_per_kvhead,
        *[T.shape[i] for i in range(head_idx + 1, len(T.shape))],
    )
    stride_unpacked = (
        seqlen_stride,
        *[T.stride[i] for i in range(1, head_idx)],
        head_stride,
        *[T.stride[i] for i in range(head_idx + 1, len(T.shape))],
    )
    return cute.make_tensor(T.iterator, cute.make_layout(shape_unpacked, stride=stride_unpacked))


@dataclass
class PackGQA:
    m_block_size: cutlass.Constexpr[int]
    head_dim_padded: cutlass.Constexpr[int]
    check_hdim_oob: cutlass.Constexpr[bool]
    qhead_per_kvhead: cutlass.Constexpr[bool]

    @cute.jit
    def compute_ptr(
        self,
        tensor: cute.Tensor,
        cRows: cute.Tensor,
        tidx: cutlass.Int32,
        block: cutlass.Int32,
        threads_per_row: cutlass.Constexpr[int],
        num_threads: cutlass.Constexpr[int],
    ):
        """Per-row gmem pointers into the packed-GQA tensor.

        ``tensor`` must keep its composite mode 0 ``(qhead_per_kvhead,
        seqlen_q)`` intact. We compute the flat element offset from
        ``stride[0][0]`` and ``stride[0][1]`` directly rather than via
        ``cute.crd2idx``: cuTeDSL 4.4-4.5 collapses the composite mode 0
        through a trailing slice (e.g. ``mO[None, 0]``), which causes
        ``crd2idx`` to reject the rank-1 composite coord at trace time.
        """
        head_stride = tensor.stride[0][0]
        seqlen_stride = tensor.stride[0][1]
        num_ptr_per_thread = cute.ceil_div(cute.size(cRows), threads_per_row)
        tPrPtr = cute.make_rmem_tensor(num_ptr_per_thread, cutlass.Int64)
        base_ptr = tensor.iterator
        for i in cutlass.range_constexpr(num_ptr_per_thread):
            row = i * num_threads + cRows[tidx % threads_per_row][0]
            idx = block * self.m_block_size + row
            m_idx = idx // self.qhead_per_kvhead
            h_idx = idx - m_idx * self.qhead_per_kvhead
            elem_offset = cutlass.Int64(h_idx) * cutlass.Int64(head_stride) + cutlass.Int64(
                m_idx
            ) * cutlass.Int64(seqlen_stride)
            tPrPtr[i] = (base_ptr + elem_offset).toint()
        return tPrPtr

    @cute.jit
    def load_Q(
        self,
        mQ: cute.Tensor,  # ((qhead_per_kvhead, seqlen_q), headdim)
        sQ: cute.Tensor,  # (m_block_size, head_dim_padded)
        gmem_tiled_copy: cute.TiledCopy,
        tidx: cutlass.Int32,
        block: cutlass.Int32,
        seqlen: cutlass.Int32,
        all_rows_valid: cutlass.Constexpr[bool] = False,
    ):
        # Note: there is no separate "zero OOB rows" path. Out-of-bounds rows
        # (m >= seqlen) are simply not copied (pred=False below). That is safe
        # because OOB Q rows never affect stored output: in the forward their O
        # rows are not written, and in the backward they produce P==0 under the
        # row mask, so they contribute nothing to dK/dV. Whatever stale value
        # sits in sQ for those rows is therefore inert.
        gmem_thr_copy = gmem_tiled_copy.get_slice(tidx)
        cQ = cute.make_identity_tensor((self.m_block_size, self.head_dim_padded))
        tQsQ = gmem_thr_copy.partition_D(sQ)
        tQcQ = gmem_thr_copy.partition_S(cQ)
        t0QcQ = gmem_thr_copy.get_slice(0).partition_S(cQ)
        tQpQ = utils.predicate_k(tQcQ, limit=mQ.shape[1])
        tQcQ_row = tQcQ[0, None, 0]
        threads_per_row = gmem_tiled_copy.layout_tv_tiled.shape[0][0]
        assert cute.arch.WARP_SIZE % threads_per_row == 0, "threads_per_row must divide WARP_SIZE"
        num_threads = gmem_tiled_copy.size
        # Pass the unsliced mQ — compute_ptr needs the composite mode 0 intact.
        tPrQPtr = self.compute_ptr(mQ, tQcQ_row, tidx, block, threads_per_row, num_threads)
        for m in cutlass.range_constexpr(cute.size(tQsQ.shape[1])):
            q_ptr_i64 = utils.shuffle_sync(
                tPrQPtr[m // threads_per_row], m % threads_per_row, width=threads_per_row
            )
            q_gmem_ptr = cute.make_ptr(
                mQ.element_type, q_ptr_i64, cute.AddressSpace.gmem, assumed_align=16
            )
            mQ_cur = cute.make_tensor(q_gmem_ptr, (self.head_dim_padded,))
            elems_per_load = cute.size(tQsQ.shape[0][0])
            mQ_cur_copy = cute.tiled_divide(mQ_cur, (elems_per_load,))
            if cutlass.const_expr(all_rows_valid):
                for k in cutlass.range_constexpr(cute.size(tQsQ.shape[2])):
                    ki = tQcQ[0, 0, k][1] // elems_per_load
                    cute.copy(
                        gmem_thr_copy,
                        mQ_cur_copy[None, ki],
                        tQsQ[None, m, k],
                        pred=tQpQ[None, m, k] if cutlass.const_expr(self.check_hdim_oob) else None,
                    )
            else:
                row_valid = (
                    t0QcQ[0, m, 0][0]
                    < seqlen * self.qhead_per_kvhead - block * self.m_block_size - tQcQ_row[0][0]
                )
                for k in cutlass.range_constexpr(cute.size(tQsQ.shape[2])):
                    ki = tQcQ[0, 0, k][1] // elems_per_load
                    coord = tQcQ[None, m, k]
                    predicate = cute.make_fragment_like(coord, cutlass.Boolean)
                    for i in cutlass.range_constexpr(cute.size(predicate)):
                        predicate[i] = (
                            cute.elem_less(coord[i][1], mQ.shape[1])
                            if cutlass.const_expr(self.check_hdim_oob)
                            else True
                        ) and row_valid
                    cute.copy(
                        gmem_thr_copy,
                        mQ_cur_copy[None, ki],
                        tQsQ[None, m, k],
                        pred=predicate,
                    )

    @cute.jit
    def load_Q_all_rows_valid(
        self,
        mQ: cute.Tensor,  # ((qhead_per_kvhead, seqlen_q), headdim)
        sQ: cute.Tensor,  # (m_block_size, head_dim_padded)
        gmem_tiled_copy: cute.TiledCopy,
        tidx: cutlass.Int32,
        block: cutlass.Int32,
        seqlen: cutlass.Int32,
    ):
        gmem_thr_copy = gmem_tiled_copy.get_slice(tidx)
        cQ = cute.make_identity_tensor((self.m_block_size, self.head_dim_padded))
        tQsQ = gmem_thr_copy.partition_D(sQ)
        tQcQ = gmem_thr_copy.partition_S(cQ)
        t0QcQ = gmem_thr_copy.get_slice(0).partition_S(cQ)
        tQpQ = utils.predicate_k(tQcQ, limit=mQ.shape[1])
        tQcQ_row = tQcQ[0, None, 0]
        threads_per_row = gmem_tiled_copy.layout_tv_tiled.shape[0][0]
        assert cute.arch.WARP_SIZE % threads_per_row == 0, "threads_per_row must divide WARP_SIZE"
        num_threads = gmem_tiled_copy.size
        tPrQPtr = self.compute_ptr(mQ, tQcQ_row, tidx, block, threads_per_row, num_threads)
        for m in cutlass.range_constexpr(cute.size(tQsQ.shape[1])):
            q_ptr_i64 = utils.shuffle_sync(
                tPrQPtr[m // threads_per_row], m % threads_per_row, width=threads_per_row
            )
            q_gmem_ptr = cute.make_ptr(
                mQ.element_type, q_ptr_i64, cute.AddressSpace.gmem, assumed_align=16
            )
            if (
                t0QcQ[0, m, 0][0]
                < seqlen * self.qhead_per_kvhead - block * self.m_block_size - tQcQ_row[0][0]
            ):
                mQ_cur = cute.make_tensor(q_gmem_ptr, (self.head_dim_padded,))
                elems_per_load = cute.size(tQsQ.shape[0][0])
                mQ_cur_copy = cute.tiled_divide(mQ_cur, (elems_per_load,))
                for k in cutlass.range_constexpr(cute.size(tQsQ.shape[2])):
                    ki = tQcQ[0, 0, k][1] // elems_per_load
                    cute.copy(
                        gmem_thr_copy,
                        mQ_cur_copy[None, ki],
                        tQsQ[None, m, k],
                        pred=tQpQ[None, m, k] if cutlass.const_expr(self.check_hdim_oob) else None,
                    )

    @cute.jit
    def store_LSE(
        self,
        mLSE: cute.Tensor,  # (qhead_per_kvhead, seqlen_q)
        tLSErLSE: cute.Tensor,  # (m_block_size, head_dim_padded)
        tiled_mma: cute.TiledMma,
        tidx: cutlass.Int32,
        block: cutlass.Int32,
        seqlen: cutlass.Int32,
        all_rows_valid: cutlass.Constexpr[bool] = False,
    ):
        thr_mma = tiled_mma.get_slice(tidx)
        caccO = cute.make_identity_tensor((self.m_block_size, self.head_dim_padded))
        taccOcO = thr_mma.partition_C(caccO)
        taccOcO_row = layout_utils.reshape_acc_to_mn(taccOcO)[None, 0]
        assert cute.size(tLSErLSE) == cute.size(taccOcO_row)
        threads_per_row = tiled_mma.tv_layout_C.shape[0][0]
        assert cute.arch.WARP_SIZE % threads_per_row == 0, "threads_per_row must divide WARP_SIZE"
        assert cute.size(tLSErLSE) <= threads_per_row
        num_threads = tiled_mma.size
        tPrLSEPtr = self.compute_ptr(mLSE, taccOcO_row, tidx, block, threads_per_row, num_threads)
        for m in cutlass.range_constexpr(cute.size(tLSErLSE)):
            lse_ptr_i64 = utils.shuffle_sync(
                tPrLSEPtr[m // threads_per_row],
                m % threads_per_row,
                width=threads_per_row,
            )
            lse_gmem_ptr = cute.make_ptr(
                mLSE.element_type, lse_ptr_i64, cute.AddressSpace.gmem, assumed_align=4
            )
            mLSE_copy = cute.make_tensor(lse_gmem_ptr, (1,))
            row = block * self.m_block_size + taccOcO_row[m][0]
            # Only the thread corresponding to column 0 writes out the lse to gmem
            if taccOcO[0][1] == 0:
                if cutlass.const_expr(all_rows_valid):
                    mLSE_copy[0] = tLSErLSE[m]
                else:
                    if row < seqlen * self.qhead_per_kvhead:
                        mLSE_copy[0] = tLSErLSE[m]

    @cute.jit
    def store_LSE_all_rows_valid(
        self,
        mLSE: cute.Tensor,  # (qhead_per_kvhead, seqlen_q)
        tLSErLSE: cute.Tensor,  # (m_block_size, head_dim_padded)
        tiled_mma: cute.TiledMma,
        tidx: cutlass.Int32,
        block: cutlass.Int32,
        seqlen: cutlass.Int32,
    ):
        thr_mma = tiled_mma.get_slice(tidx)
        caccO = cute.make_identity_tensor((self.m_block_size, self.head_dim_padded))
        taccOcO = thr_mma.partition_C(caccO)
        taccOcO_row = layout_utils.reshape_acc_to_mn(taccOcO)[None, 0]
        assert cute.size(tLSErLSE) == cute.size(taccOcO_row)
        threads_per_row = tiled_mma.tv_layout_C.shape[0][0]
        assert cute.arch.WARP_SIZE % threads_per_row == 0, "threads_per_row must divide WARP_SIZE"
        assert cute.size(tLSErLSE) <= threads_per_row
        num_threads = tiled_mma.size
        tPrLSEPtr = self.compute_ptr(mLSE, taccOcO_row, tidx, block, threads_per_row, num_threads)
        for m in cutlass.range_constexpr(cute.size(tLSErLSE)):
            lse_ptr_i64 = utils.shuffle_sync(
                tPrLSEPtr[m // threads_per_row],
                m % threads_per_row,
                width=threads_per_row,
            )
            lse_gmem_ptr = cute.make_ptr(
                mLSE.element_type, lse_ptr_i64, cute.AddressSpace.gmem, assumed_align=4
            )
            row = block * self.m_block_size + taccOcO_row[m][0]
            if taccOcO[0][1] == 0 and row < seqlen * self.qhead_per_kvhead:
                mLSE_copy = cute.make_tensor(lse_gmem_ptr, (1,))
                mLSE_copy[0] = tLSErLSE[m]

    @cute.jit
    def store_O(
        self,
        mO: cute.Tensor,  # ((qhead_per_kvhead, seqlen_q), headdim)
        tOrO: cute.Tensor,  # (m_block_size, head_dim_padded) split across threads according to gmem_tiled_copy
        gmem_tiled_copy: cute.TiledCopy,
        tidx: cutlass.Int32,
        block: cutlass.Int32,
        seqlen: cutlass.Int32,
        all_rows_valid: cutlass.Constexpr[bool] = False,
    ):
        gmem_thr_copy = gmem_tiled_copy.get_slice(tidx)
        cO = cute.make_identity_tensor((self.m_block_size, self.head_dim_padded))
        tOcO = gmem_thr_copy.partition_S(cO)
        t0OcO = gmem_thr_copy.get_slice(0).partition_S(cO)
        tOpO = utils.predicate_k(tOcO, limit=mO.shape[1])
        tOcO_row = tOcO[0, None, 0]
        threads_per_row = gmem_tiled_copy.layout_tv_tiled.shape[0][0]
        assert cute.arch.WARP_SIZE % threads_per_row == 0, "threads_per_row must divide WARP_SIZE"
        num_threads = gmem_tiled_copy.size
        # Pass the unsliced mO — compute_ptr needs the composite mode 0 intact.
        tPrOPtr = self.compute_ptr(mO, tOcO_row, tidx, block, threads_per_row, num_threads)
        for m in cutlass.range_constexpr(cute.size(tOrO.shape[1])):
            o_ptr_i64 = utils.shuffle_sync(
                tPrOPtr[m // threads_per_row], m % threads_per_row, width=threads_per_row
            )
            o_gmem_ptr = cute.make_ptr(
                mO.element_type, o_ptr_i64, cute.AddressSpace.gmem, assumed_align=16
            )
            mO_cur = cute.make_tensor(o_gmem_ptr, (self.head_dim_padded,))
            elems_per_load = cute.size(tOrO.shape[0][0])
            mO_cur_copy = cute.tiled_divide(mO_cur, (elems_per_load,))
            if cutlass.const_expr(all_rows_valid):
                for k in cutlass.range_constexpr(cute.size(tOrO.shape[2])):
                    ki = tOcO[0, 0, k][1] // elems_per_load
                    cute.copy(
                        gmem_thr_copy,
                        tOrO[None, m, k],
                        mO_cur_copy[None, ki],
                        pred=tOpO[None, m, k] if cutlass.const_expr(self.check_hdim_oob) else None,
                    )
            else:
                row_valid = (
                    t0OcO[0, m, 0][0]
                    < seqlen * self.qhead_per_kvhead - block * self.m_block_size - tOcO_row[0][0]
                )
                for k in cutlass.range_constexpr(cute.size(tOrO.shape[2])):
                    ki = tOcO[0, 0, k][1] // elems_per_load
                    coord = tOcO[None, m, k]
                    predicate = cute.make_fragment_like(coord, cutlass.Boolean)
                    for i in cutlass.range_constexpr(cute.size(predicate)):
                        predicate[i] = (
                            cute.elem_less(coord[i][1], mO.shape[1])
                            if cutlass.const_expr(self.check_hdim_oob)
                            else True
                        ) and row_valid
                    cute.copy(
                        gmem_thr_copy,
                        tOrO[None, m, k],
                        mO_cur_copy[None, ki],
                        pred=predicate,
                    )

    @cute.jit
    def store_O_all_rows_valid(
        self,
        mO: cute.Tensor,  # ((qhead_per_kvhead, seqlen_q), headdim)
        tOrO: cute.Tensor,  # (m_block_size, head_dim_padded) split across threads according to gmem_tiled_copy
        gmem_tiled_copy: cute.TiledCopy,
        tidx: cutlass.Int32,
        block: cutlass.Int32,
        seqlen: cutlass.Int32,
    ):
        gmem_thr_copy = gmem_tiled_copy.get_slice(tidx)
        cO = cute.make_identity_tensor((self.m_block_size, self.head_dim_padded))
        tOcO = gmem_thr_copy.partition_S(cO)
        t0OcO = gmem_thr_copy.get_slice(0).partition_S(cO)
        tOpO = utils.predicate_k(tOcO, limit=mO.shape[1])
        tOcO_row = tOcO[0, None, 0]
        threads_per_row = gmem_tiled_copy.layout_tv_tiled.shape[0][0]
        assert cute.arch.WARP_SIZE % threads_per_row == 0, "threads_per_row must divide WARP_SIZE"
        num_threads = gmem_tiled_copy.size
        tPrOPtr = self.compute_ptr(mO, tOcO_row, tidx, block, threads_per_row, num_threads)
        for m in cutlass.range_constexpr(cute.size(tOrO.shape[1])):
            o_ptr_i64 = utils.shuffle_sync(
                tPrOPtr[m // threads_per_row], m % threads_per_row, width=threads_per_row
            )
            o_gmem_ptr = cute.make_ptr(
                mO.element_type, o_ptr_i64, cute.AddressSpace.gmem, assumed_align=16
            )
            if (
                t0OcO[0, m, 0][0]
                < seqlen * self.qhead_per_kvhead - block * self.m_block_size - tOcO_row[0][0]
            ):
                mO_cur = cute.make_tensor(o_gmem_ptr, (self.head_dim_padded,))
                elems_per_load = cute.size(tOrO.shape[0][0])
                mO_cur_copy = cute.tiled_divide(mO_cur, (elems_per_load,))
                for k in cutlass.range_constexpr(cute.size(tOrO.shape[2])):
                    ki = tOcO[0, 0, k][1] // elems_per_load
                    cute.copy(
                        gmem_thr_copy,
                        tOrO[None, m, k],
                        mO_cur_copy[None, ki],
                        pred=tOpO[None, m, k] if cutlass.const_expr(self.check_hdim_oob) else None,
                    )

    @cute.jit
    def store_O_partial(
        self,
        mO: cute.Tensor,  # composite mode 0 (qhead_per_kvhead, seqlen_q), headdim_v
        acc_O_mn: cute.Tensor,  # reshape_acc_to_mn(acc_O): (M, N) MMA view in registers
        tiled_mma: cute.TiledMma,
        tidx: cutlass.Int32,
        block: cutlass.Int32,
        seqlen: cutlass.Int32,
        headdim_v: cutlass.Int32,
    ):
        """Direct fp32 register -> gmem scatter of the SplitKV partial output.

        Mirrors the unpacked SplitKV partial epilogue (direct MMA-layout
        register store) but scatters each packed MMA row to its physical
        (h_idx, m_idx) slot in the original-layout partial buffer via the
        composite mode-0 stride, exactly like store_O/compute_ptr do for the
        packed dtype output.  No smem roundtrip (the fp32 partial does not fit
        in the bf16-sized smem O buffer) and no dtype conversion (mO is fp32).
        """
        thr_mma = tiled_mma.get_slice(tidx)
        caccO = cute.make_identity_tensor((self.m_block_size, self.head_dim_padded))
        taccOcO = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(caccO))
        # Per-row row offset (within the tile) for this thread, and the column
        # coordinate (head_dim_v position) per MMA column element.
        taccOcO_row = taccOcO[None, 0]
        head_stride = mO.stride[0][0]
        seqlen_stride = mO.stride[0][1]
        base_ptr = mO.iterator
        for m in cutlass.range_constexpr(cute.size(acc_O_mn.shape[0])):
            packed_row = block * self.m_block_size + taccOcO_row[m][0]
            m_idx = packed_row // self.qhead_per_kvhead
            h_idx = packed_row - m_idx * self.qhead_per_kvhead
            if packed_row < seqlen * self.qhead_per_kvhead:
                elem_offset = cutlass.Int64(h_idx) * cutlass.Int64(head_stride) + cutlass.Int64(
                    m_idx
                ) * cutlass.Int64(seqlen_stride)
                o_ptr_i64 = (base_ptr + elem_offset).toint()
                o_gmem_ptr = cute.make_ptr(
                    mO.element_type, o_ptr_i64, cute.AddressSpace.gmem, assumed_align=4
                )
                mO_row = cute.make_tensor(o_gmem_ptr, (self.head_dim_padded,))
                for n in cutlass.range_constexpr(cute.size(acc_O_mn.shape[1])):
                    col = taccOcO[0, n][1]
                    if cutlass.const_expr(not self.check_hdim_oob):
                        mO_row[col] = acc_O_mn[m, n]
                    elif col < headdim_v:
                        mO_row[col] = acc_O_mn[m, n]

    @cute.jit
    def store_LSE_partial(
        self,
        mLSE: cute.Tensor,  # composite mode 0 (qhead_per_kvhead, seqlen_q)
        lse: cute.Tensor,  # (M,) per-row LSE in registers
        tiled_mma: cute.TiledMma,
        tidx: cutlass.Int32,
        block: cutlass.Int32,
        seqlen: cutlass.Int32,
    ):
        """Scatter the SplitKV partial LSE to its physical (h_idx, m_idx) slot.

        Like store_LSE but writes the fp32 partial-LSE buffer; only the thread
        owning column 0 of each MMA row writes (matches the unpacked path).
        """
        thr_mma = tiled_mma.get_slice(tidx)
        caccO = cute.make_identity_tensor((self.m_block_size, self.head_dim_padded))
        taccOcO = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(caccO))
        taccOcO_row = taccOcO[None, 0]
        head_stride = mLSE.stride[0][0]
        seqlen_stride = mLSE.stride[0][1]
        base_ptr = mLSE.iterator
        # Only the thread owning column 0 writes the per-row LSE (matches the
        # unpacked SplitKV LSE epilogue predicate taccOcO[0][1] == 0).
        if taccOcO[0][1] == 0:
            for m in cutlass.range_constexpr(cute.size(lse)):
                packed_row = block * self.m_block_size + taccOcO_row[m][0]
                m_idx = packed_row // self.qhead_per_kvhead
                h_idx = packed_row - m_idx * self.qhead_per_kvhead
                if packed_row < seqlen * self.qhead_per_kvhead:
                    elem_offset = cutlass.Int64(h_idx) * cutlass.Int64(
                        head_stride
                    ) + cutlass.Int64(m_idx) * cutlass.Int64(seqlen_stride)
                    lse_ptr_i64 = (base_ptr + elem_offset).toint()
                    lse_gmem_ptr = cute.make_ptr(
                        mLSE.element_type, lse_ptr_i64, cute.AddressSpace.gmem, assumed_align=4
                    )
                    cute.make_tensor(lse_gmem_ptr, (1,))[0] = lse[m]

    @cute.jit
    def load_scalar_per_row(
        self,
        mLSE: cute.Tensor,  # composite mode 0: (qhead_per_kvhead, seqlen_q) — rank 1
        sLSE: cute.Tensor,  # (m_block_size,)
        tidx: cutlass.Int32,
        block: cutlass.Int32,
        seqlen: cutlass.Int32,
        all_rows_valid: cutlass.Constexpr[bool] = False,
    ):
        """Load one scalar (fp32) per row from a packed-GQA tensor (LSE / dPsum).

        Uses one thread per row (m_block_size threads). The remaining threads
        do nothing. mLSE must keep its composite mode 0 intact so we can
        compute per-row gmem pointers via stride[0][0]/stride[0][1].
        """
        head_stride = mLSE.stride[0][0]
        seqlen_stride = mLSE.stride[0][1]
        base_ptr = mLSE.iterator
        if tidx < self.m_block_size:
            row = tidx
            idx = block * self.m_block_size + row
            m_idx = idx // self.qhead_per_kvhead
            h_idx = idx - m_idx * self.qhead_per_kvhead
            elem_offset = cutlass.Int64(h_idx) * cutlass.Int64(head_stride) + cutlass.Int64(
                m_idx
            ) * cutlass.Int64(seqlen_stride)
            lse_ptr_i64 = (base_ptr + elem_offset).toint()
            lse_gmem_ptr = cute.make_ptr(
                mLSE.element_type, lse_ptr_i64, cute.AddressSpace.gmem, assumed_align=4
            )
            # OOB guard on the seqlen dim (qhead dim never overshoots since qhead is constexpr)
            if cutlass.const_expr(all_rows_valid):
                sLSE[row] = cute.make_tensor(lse_gmem_ptr, (1,))[0]
            else:
                if m_idx < seqlen:
                    sLSE[row] = cute.make_tensor(lse_gmem_ptr, (1,))[0]
                else:
                    sLSE[row] = cutlass.Float32(0.0)

    @cute.jit
    def atomic_add_dQaccum(
        self,
        mdQaccum: cute.Tensor,
        acc_dQ_atomic: cute.Tensor,  # retiled fragment, same flat layout as MMA C
        tiled_mma_dq: cute.TiledMma,
        tidx: cutlass.Int32,
        block: cutlass.Int32,
        seqlen: cutlass.Int32,
        head_kv_idx: cutlass.Int32 = cutlass.Int32(0),
        dq_accum_batch_offset: cutlass.Int32 = cutlass.Int32(0),
    ):
        """Atomic-add per-MMA-element dQ values into the ORIGINAL-layout
        dq_accum, routing each element to the correct head_q slot and to
        the canonical gmem position that the postprocess kernel will read
        back as MMA (m_actual_in_unpacked, d) for that head_q.

        Under pack_gqa, the MMA computes dQ for packed rows. Each MMA
        element at (row=mma_m, col=mma_d) for thread t represents the
        gradient for the packed row mma_m, which maps to original
        (h_actual = mma_m % qh, m_actual = mma_m // qh, d=mma_d).

        The postprocess kernel (which knows nothing about pack_gqa) reads
        dq_accum[batch, head_q, gmem_position] and emits to dq[batch,
        head_q, m_pp, d_pp] where (m_pp, d_pp) is determined by its own
        partition_C of the same MMA layout — i.e., gmem_position k maps
        deterministically to (m_pp, d_pp) via:

            warp_id = k // 128   (assuming 32 threads * 4 vals = 128 per warp's flat block)
            ...

        Instead of inverting partition_C in formula, we use partition_C
        directly: for each MMA element of thread t at index i:
        1. Read (mma_m, d) from taccdQcdQ[i].
        2. Decompose to (h_actual, m_actual).
        3. Find the canonical gmem position k_target such that postprocess's
           partition_C(thread_target)[i_target] = (m_actual, d), where
           thread_target and i_target are determined by inverting the
           partition_C mapping.
        4. Atomic-add to gmem[batch, h_actual, k_target] in the original
           mdQaccum layout (sliced per batch).

        We hard-code the MMA layout pattern empirically observed:
          - warp_m = (mma_m // 16) for warps in M dim (0..3)
          - warp_n = (d // 8) % 2 for warps in N dim (0..1)
          - 8 warps total = 4 in M × 2 in N, warp_id = warp_n * 4 + warp_m
          - lane within warp: lane_row = (mma_m % 16) % 8 in [0..7], lane_col_pair_idx = (d % 8) // 2 in [0..3]
            lane = lane_row * 4 + lane_col_pair_idx
          - val_m = (mma_m % 16) // 8 in {0, 1}
          - val_n = d % 2 in {0, 1}
          - v = val_m * 2 + val_n in [0..3]
          - outer_iter = (d // 8) // 2 in [0..3]
          - i_flat = outer_iter * 4 + v
          - thread_target = warp_id * 32 + lane
          - k_target (within head_q's slot) = outer_iter * 1024 + thread_target * 4 + v

        Assumes m_block_size <= 64, head_dim_padded <= 64, num_threads=256,
        AtomLayoutMdQ=1, m16n8k16 atom, dQ_swapAB=False. For other
        configurations a separate code path would be needed.
        """
        thr_mma = tiled_mma_dq.get_slice(tidx)
        cdQ = cute.make_identity_tensor((self.m_block_size, self.head_dim_padded))
        taccdQcdQ = thr_mma.partition_C(cdQ)
        assert cute.size(taccdQcdQ) == cute.size(acc_dQ_atomic), (
            "partition_C identity must have same size as acc_dQ_atomic"
        )
        # mdQaccum has the ORIGINAL layout sliced per batch. Non-varlen:
        # rank-2 (H_q, S*D) with strides (S*D, 1). Varlen: rank-2
        # (H_q, total_q_padded*D) with strides (total_q_padded*D, 1).
        head_stride = mdQaccum.stride[0]
        seqlen_stride = mdQaccum.stride[1]
        base_ptr = mdQaccum.iterator
        n_elems = cute.size(acc_dQ_atomic)
        # Per-m_block stride in head_q's slot (between m_blocks).
        mblock_size_flat = self.m_block_size * self.head_dim_padded
        for i in cutlass.range_constexpr(n_elems):
            mn_coord = taccdQcdQ[i]
            mma_m = mn_coord[0]
            d = mn_coord[1]
            # Packed row → (h_in_kvgroup, m_actual). h_in_kvgroup is the
            # offset within the current head_kv's group of qh head_q's.
            # The absolute head_q index is head_kv_idx * qh + h_in_kvgroup.
            packed_row = block * self.m_block_size + mma_m
            m_actual = packed_row // self.qhead_per_kvhead
            h_in_kvgroup = packed_row - m_actual * self.qhead_per_kvhead
            h_actual = head_kv_idx * self.qhead_per_kvhead + h_in_kvgroup

            # Canonical (warp_m, warp_n, lane, val, outer) for postprocess's
            # interpretation of (m_actual, d) within head_q's slot, assuming
            # m_actual fits in one m_block (i.e., m_actual < m_block_size).
            # If m_actual >= m_block_size, we need m_block_in_unpacked > 0.
            m_block_in_unpacked = m_actual // self.m_block_size
            m_in_mblock = m_actual - m_block_in_unpacked * self.m_block_size

            warp_m = m_in_mblock // 16
            m_in_warp = m_in_mblock - warp_m * 16
            val_m = m_in_warp // 8
            lane_row = m_in_warp - val_m * 8

            warp_n = (d // 8) - ((d // 8) // 2) * 2  # = (d // 8) % 2
            outer_iter = (d // 8) // 2
            d_in_atom = d - (d // 8) * 8  # = d % 8
            lane_col_pair_idx = d_in_atom // 2  # 0..3
            val_n = d_in_atom - lane_col_pair_idx * 2  # = d % 2

            v = val_m * 2 + val_n
            lane = lane_row * 4 + lane_col_pair_idx
            warp_id = warp_n * 4 + warp_m
            thread_target = warp_id * 32 + lane
            k_within_mblock = outer_iter * 1024 + thread_target * 4 + v
            position_in_head_slot = m_block_in_unpacked * mblock_size_flat + k_within_mblock

            elem_offset = cutlass.Int64(h_actual) * cutlass.Int64(head_stride) + cutlass.Int64(
                dq_accum_batch_offset + position_in_head_slot
            ) * cutlass.Int64(seqlen_stride)
            dq_ptr_i64 = (base_ptr + elem_offset).toint()
            dq_gmem_ptr = cute.make_ptr(
                cutlass.Float32, dq_ptr_i64, cute.AddressSpace.gmem, assumed_align=4
            )
            if m_actual < seqlen and mma_m < self.m_block_size:
                utils.atomic_add_fp32(acc_dQ_atomic[i], dq_gmem_ptr)
