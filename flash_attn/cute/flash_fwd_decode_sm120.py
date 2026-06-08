# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# SM120 (consumer Blackwell, RTX PRO 6000) decode-specialized forward pass.
#
# Decode = seqlen_q == 1 with a large KV cache.  The general SM80-base forward
# kernel (flash_fwd.py) runs the m16n8k16 tensor-core MMA over a tile_m of
# mostly-empty query rows, making decode COMPUTE-bound on wasted MMA instead of
# MEMORY-bound on the KV stream; and with pack_gqa disabled on the SplitKV path
# it launches one CTA per *query* head, streaming the shared KV cache of a GQA
# group `qhead_per_kvhead` times.
#
# This kernel is a from-scratch GEMV-style decode path:
#   * One CTA == one (batch, kv_head, split).  All R = qhead_per_kvhead query
#     rows that share a KV head are processed together, so each K/V tile is read
#     from DRAM exactly once (no GQA redundancy).
#   * Scores Q.K^T and the P.V contraction are FMA + warp-shuffle reductions
#     (a GEMV), NOT the m16n8k16 MMA -> no tensor-core lanes wasted on empty
#     query rows.
#   * K/V tiles are streamed with cp.async (128-bit coalesced) by all threads,
#     double buffered -> the inner loop is DRAM bound.
#   * Online softmax (running max/sum per query row) within the split; the
#     existing FlashAttentionForwardCombine merges splits.  Writes fp32 partial
#     O (num_splits, b, s, h, d) and partial LSE (num_splits, b, h, s).
#
# Thread layout: tpr = head_dim / (128/dtype.width) threads cooperate on one
# K/V row's head-dim chunk for loads.  For the math, every thread owns its
# head-dim chunk (`vec` elements) for all R rows.  Each thread group of `tpr`
# lanes computes a full score via a width-`tpr` butterfly reduction.  Every
# thread iterates over ALL keys of the split (the redundant on-chip FMA is cheap
# vs. DRAM; correctness needs no cross-group reduction).

import math

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32, const_expr
from cutlass.cute.nvgpu import cpasync

from flash_attn.cute import utils


LOG2_E = Float32(math.log2(math.e))
LN2 = Float32(math.log(2.0))


class FlashAttentionDecodeSm120:
    def __init__(
        self,
        dtype,
        head_dim: int,
        head_dim_v: int,
        qhead_per_kvhead: int,
        num_splits: int,
        tile_n: int = 64,
        num_threads: int = 128,
        num_stages: int = 2,
        is_causal: bool = False,
        kv_dtype=None,
    ):
        self.dtype = dtype  # Q dtype (compute / score dtype: fp16/bf16)
        # K/V cache dtype.  Defaults to the Q dtype; may be fp8 (e4m3/e5m2) for a
        # quantized KV cache while Q stays bf16/fp16.  Only the K/V *loads* become
        # fp8 -> half the DRAM bytes streamed; the descale scalars restore range.
        self.kv_dtype = kv_dtype if kv_dtype is not None else dtype
        self.head_dim = head_dim
        self.head_dim_v = head_dim_v
        assert head_dim == head_dim_v, "decode kernel assumes head_dim == head_dim_v"
        self.qhead_per_kvhead = qhead_per_kvhead
        self.num_splits = num_splits
        self.tile_n = tile_n
        self.num_threads = num_threads
        self.num_stages = num_stages
        self.is_causal = is_causal
        self.is_fp8_kv = self.kv_dtype.width == 8
        self.R = qhead_per_kvhead
        # vec = elems per 16B cp.async load, governed by the K/V (load) dtype.
        # fp8 -> vec=16 (vs 8 for bf16): more elems per coalesced load = the BW win.
        self.vec = 128 // self.kv_dtype.width  # elems per 16B load
        self.threads_per_row = head_dim // self.vec  # tpr
        assert num_threads % self.threads_per_row == 0
        self.rows_per_iter = num_threads // self.threads_per_row
        assert tile_n % self.rows_per_iter == 0

    @staticmethod
    def can_implement(
        dtype,
        head_dim,
        head_dim_v,
        qhead_per_kvhead,
        num_threads,
        tile_n,
        num_stages=2,
        kv_dtype=None,
    ):
        if dtype not in (cutlass.Float16, cutlass.BFloat16):
            return False
        kv_dtype = kv_dtype if kv_dtype is not None else dtype
        # K/V cache may be fp8 (e4m3/e5m2) while Q/compute stays fp16/bf16.
        if kv_dtype not in (
            cutlass.Float16,
            cutlass.BFloat16,
            cutlass.Float8E4M3FN,
            cutlass.Float8E5M2,
        ):
            return False
        if head_dim != head_dim_v or head_dim not in (128, 256):
            return False
        vec = 128 // kv_dtype.width
        if head_dim % vec != 0:
            return False
        tpr = head_dim // vec
        if num_threads % tpr != 0:
            return False
        rpi = num_threads // tpr
        if tile_n % rpi != 0:
            return False
        if num_threads % 32 != 0 or qhead_per_kvhead > 32:
            return False
        R = qhead_per_kvhead
        kv_elem_bytes = kv_dtype.width // 8
        # The K/V smem region is sized to hold whichever is larger: the K/V tile
        # (NS*TN*d * kv_elem_bytes) or the fp32 cross-group reduction scratch that
        # is recast onto the same bytes after the mainloop.  For fp8 the tile
        # shrinks to 1 byte/elem so the fp32 scratch dominates; for bf16 the tile
        # bytes dominate (== legacy size), so bf16 smem is byte-identical.
        kv_tile = num_stages * tile_n * head_dim * kv_elem_bytes
        # Two-level reduction: warp-shuffle merge within each warp, then an smem
        # fan-out across only `nwarps` warps (not the rpi row-groups), so the fp32
        # reduction scratch is sized by nwarps.
        nwarps = num_threads // 32
        red_acc = nwarps * R * tpr * vec * 4
        red_ms = 2 * nwarps * R * tpr * 4
        sK_smem = max(kv_tile, red_acc)
        sV_smem = max(kv_tile, red_ms)
        if sK_smem + sV_smem > 99 * 1024:
            return False
        return True

    def _smem_bytes(self):
        kv_elem_bytes = self.kv_dtype.width // 8
        kv_tile = self.num_stages * self.tile_n * self.head_dim * kv_elem_bytes
        R, tpr, vec = self.R, self.threads_per_row, self.vec
        nwarps = self.num_threads // 32
        red_acc = nwarps * R * tpr * vec * 4
        red_ms = 2 * nwarps * R * tpr * 4
        sK_bytes = max(kv_tile, red_acc)
        sV_bytes = max(kv_tile, red_ms)
        return sK_bytes + sV_bytes

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (b, sq, hq, d)
        mK: cute.Tensor,  # (b, sk, hkv, d)
        mV: cute.Tensor,  # (b, sk, hkv, d)
        mO: cute.Tensor,  # (num_splits, b, sq, hq, d) fp32
        mLSE: cute.Tensor,  # (num_splits, b, hq, sq) fp32
        softmax_scale: Float32,
        mKDescale: cute.Tensor = None,  # (b, hkv) fp32, optional (fp8 K cache)
        mVDescale: cute.Tensor = None,  # (b, hkv) fp32, optional (fp8 V cache)
        stream=None,
    ):
        from flash_attn.cute.cute_dsl_utils import assume_tensor_aligned

        mQ, mK, mV = [assume_tensor_aligned(t) for t in (mQ, mK, mV)]
        b = mK.shape[0]
        hkv = mK.shape[2]
        grid = (self.num_splits, hkv, b)
        self.kernel(mQ, mK, mV, mO, mLSE, softmax_scale, mKDescale, mVDescale).launch(
            grid=grid,
            block=[self.num_threads, 1, 1],
            smem=self._smem_bytes(),
            stream=stream,
        )

    @cute.jit
    def load_tile(self, gKc, gVc, sKc, sVc, copy_atom, n_block, stage, lane_d, row_grp, seqlen_k):
        # gKc/gVc: (sk, tpr, vec) chunked view of K/V.
        # sKc/sVc: (NS, TN, tpr, vec) chunked smem.
        TN = const_expr(self.tile_n)
        vec = const_expr(self.vec)
        rpi = const_expr(self.rows_per_iter)
        n_waves = const_expr(TN // rpi)
        base_row = n_block * TN
        for w in cutlass.range_constexpr(n_waves):
            krow = w * rpi + row_grp
            gk = base_row + krow
            if gk < seqlen_k:
                cute.copy(copy_atom, gKc[gk, lane_d, None], sKc[stage, krow, lane_d, None])
                cute.copy(copy_atom, gVc[gk, lane_d, None], sVc[stage, krow, lane_d, None])
            else:
                for e in cutlass.range_constexpr(vec):
                    sKc[stage, krow, lane_d, e] = self.kv_dtype(0.0)
                    sVc[stage, krow, lane_d, e] = self.kv_dtype(0.0)

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        softmax_scale: Float32,
        mKDescale: cute.Tensor,
        mVDescale: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        split_idx, kv_head, batch = cute.arch.block_idx()

        d = const_expr(self.head_dim)
        R = const_expr(self.R)
        TN = const_expr(self.tile_n)
        vec = const_expr(self.vec)
        tpr = const_expr(self.threads_per_row)
        rpi = const_expr(self.rows_per_iter)
        NS = const_expr(self.num_stages)
        seqlen_k = mK.shape[1]

        # Per-(batch, kv-head) descale scalars for an fp8 K/V cache.  k_descale is
        # folded into the QK score (it scales every dot equally, so it commutes
        # through softmax with softmax_scale); v_descale rescales the P.V output.
        # Both default to 1.0 when no descale tensor is supplied (bf16 cache).
        k_descale = Float32(1.0)
        v_descale = Float32(1.0)
        if const_expr(mKDescale is not None):
            k_descale = Float32(mKDescale[batch, kv_head])
        if const_expr(mVDescale is not None):
            v_descale = Float32(mVDescale[batch, kv_head])

        n_block_total = cute.ceil_div(seqlen_k, TN)
        nblk_per_split = cute.ceil_div(n_block_total, self.num_splits)
        n_block_min = cutlass.min(split_idx * nblk_per_split, n_block_total)
        n_block_max = cutlass.min(n_block_min + nblk_per_split, n_block_total)
        n_iters = cutlass.max(n_block_max - n_block_min, Int32(0))

        # ---- shared memory: NS-buffered K and V tiles, chunked as (NS,TN,tpr,vec) ----
        # Allocate each region as a raw byte buffer sized to max(kv tile bytes,
        # fp32 reduction-scratch bytes), then view it as the K/V (kv_dtype) tile.
        # For bf16 the kv tile dominates so this is byte-identical to the legacy
        # allocate_tensor; for fp8 the 1-byte tile is padded up so the fp32 scratch
        # (recast onto the same bytes after the mainloop) still fits.
        kv_elem_bytes = const_expr(self.kv_dtype.width // 8)
        kv_tile_bytes = const_expr(NS * TN * d * kv_elem_bytes)
        nwarps = const_expr(self.num_threads // 32)
        red_acc_bytes = const_expr(nwarps * R * tpr * vec * 4)
        red_ms_bytes = const_expr(2 * nwarps * R * tpr * 4)
        sK_bytes = const_expr(max(kv_tile_bytes, red_acc_bytes))
        sV_bytes = const_expr(max(kv_tile_bytes, red_ms_bytes))
        smem = cutlass.utils.SmemAllocator()
        smem_layout = cute.make_layout((NS, TN, tpr, vec), stride=(TN * d, d, vec, 1))
        sK_ptr = smem.allocate(sK_bytes, byte_alignment=1024)
        sV_ptr = smem.allocate(sV_bytes, byte_alignment=1024)
        sKc = cute.make_tensor(cute.recast_ptr(sK_ptr, dtype=self.kv_dtype), smem_layout)
        sVc = cute.make_tensor(cute.recast_ptr(sV_ptr, dtype=self.kv_dtype), smem_layout)

        lane_d = tidx % tpr  # which 16B chunk of head dim
        row_grp = tidx // tpr  # which K/V row within a cp.async wave

        copy_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self.kv_dtype,
            num_bits_per_copy=128,
        )

        # chunked gmem views: (sk, tpr, vec).  Each thread's innermost load is
        # vec contiguous elements = one 16B chunk; the row base and lane_d*vec
        # offset are both 16B aligned.
        gK = mK[batch, None, kv_head, None]  # (sk, d)
        gV = mV[batch, None, kv_head, None]
        gK_chunk_layout = cute.make_layout((seqlen_k, tpr, vec), stride=(gK.stride[0], vec, 1))
        gKc = cute.make_tensor(gK.iterator, gK_chunk_layout)
        gVc = cute.make_tensor(gV.iterator, gK_chunk_layout)

        # ---- load Q rows (R rows, this thread's vec chunk) into registers ----
        rQ = cute.make_fragment((R, vec), Float32)
        for r in cutlass.range_constexpr(R):
            q_head = kv_head * R + r
            for e in cutlass.range_constexpr(vec):
                rQ[r, e] = Float32(mQ[batch, 0, q_head, lane_d * vec + e])

        # ---- online softmax state (per row; this thread's acc chunk) ----
        acc_o = cute.make_fragment((R, vec), Float32)
        row_max = cute.make_fragment((R,), Float32)
        row_sum = cute.make_fragment((R,), Float32)
        for r in cutlass.range_constexpr(R):
            row_max[r] = Float32(-1e30)
            row_sum[r] = Float32(0.0)
            for e in cutlass.range_constexpr(vec):
                acc_o[r, e] = Float32(0.0)

        # prologue: prefetch first tile (bounds-checked; empty split -> zero-fill)
        self.load_tile(
            gKc, gVc, sKc, sVc, copy_atom, n_block_min, Int32(0), lane_d, row_grp, seqlen_k
        )
        cute.arch.cp_async_commit_group()

        for it in cutlass.range(n_iters, unroll=1):
            n_block = n_block_min + it
            stage = it % NS
            nxt = (it + 1) % NS
            if it + 1 < n_iters:
                self.load_tile(
                    gKc,
                    gVc,
                    sKc,
                    sVc,
                    copy_atom,
                    n_block_min + it + 1,
                    nxt,
                    lane_d,
                    row_grp,
                    seqlen_k,
                )
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(1)
            else:
                cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()

            base_row = n_block * TN
            # Partition keys across the rpi row-groups: row_grp `g` handles keys
            # j = g, g+rpi, g+2*rpi, ...  (1/rpi of the tile).  A final smem
            # reduction across the rpi groups merges the per-row stats.
            for jw in cutlass.range_constexpr(TN // rpi):
                j = jw * rpi + row_grp
                gj = base_row + j
                valid = gj < seqlen_k
                for r in cutlass.range_constexpr(R):
                    p = Float32(0.0)
                    for e in cutlass.range_constexpr(vec):
                        p += rQ[r, e] * Float32(sKc[stage, j, lane_d, e])
                    # reduce partial dot across the tpr lanes (butterfly, width tpr)
                    p = utils.warp_reduce(p, lambda a, bb: a + bb, width=tpr)
                    # k_descale (==1.0 for bf16 cache) folds into the QK score: it
                    # scales every key's dot equally so it commutes through softmax.
                    s = p * softmax_scale * k_descale
                    s = s if valid else Float32(-1e30)
                    old_max = row_max[r]
                    new_max = cutlass.max(old_max, s)
                    corr = cute.math.exp2((old_max - new_max) * LOG2_E, fastmath=True)
                    corr = corr if old_max > Float32(-1e29) else Float32(0.0)
                    pexp = cute.math.exp2((s - new_max) * LOG2_E, fastmath=True)
                    pexp = pexp if valid else Float32(0.0)
                    row_max[r] = new_max
                    row_sum[r] = row_sum[r] * corr + pexp
                    for e in cutlass.range_constexpr(vec):
                        acc_o[r, e] = acc_o[r, e] * corr + pexp * Float32(sVc[stage, j, lane_d, e])
            cute.arch.barrier()

        # ---- cross-row-group reduction of (max, sum, acc) over the rpi groups ----
        # Each row_grp owns a disjoint key subset; their online-softmax states are
        # merged in two levels to keep the smem fan-out (and hence occupancy) low:
        #   1) WARP level: the `groups_per_warp` row-groups that live in the same
        #      warp (and share a lane_d) are merged with shuffle-butterfly XORs over
        #      the row-group bits of the lane index (offsets tpr, 2*tpr, ...).  After
        #      this every lane holds its warp's merged state for its lane_d.
        #   2) SMEM level: one representative lane per (warp, lane_d) writes the
        #      warp-merged state to scratch indexed by warp_id; warp 0 then merges
        #      across the `nwarps` warps.
        # This shrinks the smem fan-out from rpi -> nwarps.  For bf16 the K/V tile
        # bytes still dominate so the smem is byte-identical to the legacy size; for
        # fp8 (where the fan-out scratch dominated) it drops ~rpi/nwarps x.
        nwarps = const_expr(self.num_threads // 32)
        gpw = const_expr(32 // tpr)  # row-groups per warp sharing a lane_d
        warp_id = tidx // 32
        lane = tidx % 32

        # 1) warp-level butterfly merge of (max, sum, acc) across the gpw groups.
        for r in cutlass.range_constexpr(R):
            wm = row_max[r]
            ws = row_sum[r]
            for off in cutlass.range_constexpr(int(math.log2(gpw))):
                step = const_expr(tpr << off)
                om = cute.arch.shuffle_sync_bfly(wm, offset=step)
                os_ = cute.arch.shuffle_sync_bfly(ws, offset=step)
                nm = cutlass.max(wm, om)
                cself = cute.math.exp2((wm - nm) * LOG2_E, fastmath=True)
                cself = cself if wm > Float32(-1e29) else Float32(0.0)
                cother = cute.math.exp2((om - nm) * LOG2_E, fastmath=True)
                cother = cother if om > Float32(-1e29) else Float32(0.0)
                for e in cutlass.range_constexpr(vec):
                    oa = cute.arch.shuffle_sync_bfly(acc_o[r, e], offset=step)
                    acc_o[r, e] = acc_o[r, e] * cself + oa * cother
                ws = ws * cself + os_ * cother
                wm = nm
            row_max[r] = wm
            row_sum[r] = ws

        # 2) smem fan-out across the nwarps warps.  Scratch ALIASED onto the
        # now-finished K/V smem (recast -> fp32): sKc holds acc, sVc holds
        # [max | sum], indexed by warp_id.  can_implement guarantees these fit.
        sRedAcc = cute.make_tensor(
            cute.recast_ptr(sKc.iterator, dtype=Float32),
            cute.make_layout((nwarps, R, tpr, vec), stride=(R * tpr * vec, tpr * vec, vec, 1)),
        )
        sRedMS = cute.make_tensor(
            cute.recast_ptr(sVc.iterator, dtype=Float32),
            cute.make_layout((2, nwarps, R, tpr), stride=(nwarps * R * tpr, R * tpr, tpr, 1)),
        )
        cute.arch.barrier()
        # Lanes in local group 0 of each warp (lane < tpr) own a distinct lane_d
        # and hold the warp-merged state; they publish it keyed by warp_id.
        if lane < tpr:
            for r in cutlass.range_constexpr(R):
                sRedMS[0, warp_id, r, lane_d] = row_max[r]
                sRedMS[1, warp_id, r, lane_d] = row_sum[r]
                for e in cutlass.range_constexpr(vec):
                    sRedAcc[warp_id, r, lane_d, e] = acc_o[r, e]
        cute.arch.barrier()

        # row_grp 0 (which is lane_d == lane, warp 0) merges across all nwarps.
        if row_grp == 0:
            for r in cutlass.range_constexpr(R):
                gmax = Float32(-1e30)
                for g in cutlass.range_constexpr(nwarps):
                    gmax = cutlass.max(gmax, sRedMS[0, g, r, lane_d])
                gsum = Float32(0.0)
                for e in cutlass.range_constexpr(vec):
                    acc_o[r, e] = Float32(0.0)
                for g in cutlass.range_constexpr(nwarps):
                    gm = sRedMS[0, g, r, lane_d]
                    corr = cute.math.exp2((gm - gmax) * LOG2_E, fastmath=True)
                    corr = corr if gm > Float32(-1e29) else Float32(0.0)
                    gsum += sRedMS[1, g, r, lane_d] * corr
                    for e in cutlass.range_constexpr(vec):
                        acc_o[r, e] += sRedAcc[g, r, lane_d, e] * corr
                row_max[r] = gmax
                row_sum[r] = gsum

        # ---- write normalized partial O + natural-log LSE (row_grp 0 only) ----
        if row_grp == 0:
            for r in cutlass.range_constexpr(R):
                s = row_sum[r]
                zero_or_nan = (s == Float32(0.0)) or (s != s)
                inv = cute.arch.rcp_approx(s if not zero_or_nan else Float32(1.0))
                # v_descale (==1.0 for bf16 cache) restores the fp8-quantized V
                # range; fold it into the softmax-normalisation reciprocal.
                inv = inv * v_descale
                q_head = kv_head * R + r
                for e in cutlass.range_constexpr(vec):
                    mO[split_idx, batch, 0, q_head, lane_d * vec + e] = acc_o[r, e] * inv
                if lane_d == 0:
                    lse = (
                        (row_max[r] * LOG2_E + cute.math.log2(s, fastmath=True)) * LN2
                        if not zero_or_nan
                        else Float32(-1e30)
                    )
                    mLSE[split_idx, batch, q_head, 0] = lse
