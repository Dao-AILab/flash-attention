"""
================================================================================
 Ring / context-parallel Flash Attention on the optimized FA3 primitives
================================================================================

A self-contained, autograd-enabled ring (context-parallel) attention driver built
directly on the FA3 (Hopper / sm90) ops via their FOLDED-IN ring parameters -- there
are NO separate ring ops/wrappers; the ring path is the stock fwd/bwd with a flag:

    _fwd_op = torch.ops.flash_attn_3.fwd   # forward: fwd(..., skip_combine=True, out_accum=,
                                           #   lse_accum=) -> fp32 normalized partial + fp32 LSE, no combine
    _bwd_op = torch.ops.flash_attn_3.bwd   # backward: bwd(..., ring_phase=1/2/3, dq_accum=,
                                           #   dsoftmax_sum=, softmax_lse_log2=) -> phased backward
    from flash_attn_interface import ring_bwd_alloc  # allocate the 3 persistent fp32 buffers

Thin private `_fwd_ring` / `_bwd_ring` wrappers (below) adapt those ops. They keep every
ring intermediate in fp32 across ring hops and cast to bf16 exactly once, instead of
round-tripping through bf16 on every hop (which a naive ring built on the plain
flash_attn_func / an un-phased flash_attn_3.bwd pays). See the
`### Why these ops` section below.

--------------------------------------------------------------------------------
 The ring algorithm in one paragraph
--------------------------------------------------------------------------------
A long sequence is split into W shards (W = number of GPUs); each rank permanently
owns one shard of Q (Q is NEVER communicated) and, initially, the matching shard of
K/V. K/V then rotate around a ring one hop at a time; after W hops every rank has
seen every K/V shard. A single query's attention over the whole key set equals the
online-softmax merge of its attention over each key block, so computing block by
block and merging is numerically identical to full-sequence attention.

Forward (`_ring_forward`): each hop runs `_fwd_ring` (= fwd with skip_combine=True) on
the current K/V block to get an fp32 normalized partial + fp32 LSE (CUTLASS speed, no
bf16 round-trip), and streams it into a persistent fp32 accumulator with an online
softmax merge; after W hops the accumulator is cast to bf16 once. Under causal
masking only the diagonal block (step 0) carries the mask.

Backward (`_ring_backward`, phased via `_bwd_ring` (= bwd with ring_phase=…) + `ring_bwd_alloc`):
  * phase 1 (once/rank): compute D = rowsum(dO*O), softmax_lse_log2, and clear the
    persistent fp32 dq_accum;
  * phase 2 (per hop): run only the main kernel, atomicAdd this K/V block's dQ into
    the persistent fp32 dq_accum (never leaving fp32), and write this block's dK/dV;
    dK/dV are reduced in fp32 along a SECOND ring back to their owner rank;
  * phase 3 (once/rank): convert the accumulated fp32 dq_accum to bf16 dQ.
Compared with running a full backward per hop, this drops W-1 preprocess + W-1
convert passes per rank and keeps dQ resident in fp32.

--------------------------------------------------------------------------------
 Why the folded ring flags (vs a naive ring on the public FA3 entry points)
--------------------------------------------------------------------------------
  * A naive forward calls flash_attn_func per block: it runs a full split+combine
    and returns a bf16 block output, so the ring must cast back to fp32 before its
    own merge (one bf16 round-trip per hop) and the internal combine is wasted.
    `fwd(skip_combine=True)` emits the fp32 partial directly and skips the combine.
  * A naive backward calls the full flash_attn_3.bwd per hop: it recomputes
    preprocess and convert every hop and rounds dQ to bf16 each hop. The phased
    `bwd(ring_phase=…)` runs preprocess/convert once per rank and keeps dQ in fp32.

--------------------------------------------------------------------------------
 Supported configuration
--------------------------------------------------------------------------------
  * dtype: fp16 / bf16.
  * head_dim: 64, 96, 128, 192, 256 (sm90).
  * MHA (hq == hkv) and GQA/MQA (hq a multiple of hkv).
  * causal and non-causal. causal uses the plain causal ring: rank r computes steps
    0..r; step 0 is its own diagonal block (the only masked one). This is load-
    imbalanced by design (later ranks do more work); a zigzag shard assignment would
    balance it and is intentionally not implemented here to keep the driver simple.
  * varlen (cu_seqlens) via `ring_flash_attn_varlen_func`: each global sequence is split
    into W equal chunks along its length (every sequence length must be divisible by W)
    and rank r owns chunk r; forward = per-hop flash_attn_varlen_func + online merge
    (Option A), backward = phased bwd_ring. See `shard_varlen_along_seq`.
 NOT supported: sliding-window / local, deterministic backward, sequence lengths not
  divisible by W, and a ragged varlen CP where a sequence's chunking differs across
  ranks. sm90 (Hopper) only.

--------------------------------------------------------------------------------
 Requirements / how to run
--------------------------------------------------------------------------------
  * The in-tree hopper FA3 extension must be built (it exposes the ring primitives);
    it is resolved relative to this file (../hopper), so no absolute path is needed.
  * q/k/v layout: (batch, seqlen_local, nheads, head_dim), last dim contiguous,
    16-byte aligned, bf16/fp16.

  # run the built-in example (forward+backward + a single-GPU correctness check):
  torchrun --nproc_per_node=8 --standalone ring_flash_attn_fa3.py
  # 2 GPUs:
  CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --standalone ring_flash_attn_fa3.py
"""

import os
import sys

import torch
import torch.distributed as dist
import triton
import triton.language as tl

# Resolve the in-tree hopper build (which exposes the ring primitives) relative to
# this file: <repo>/tests/this_file.py -> <repo>/hopper. No absolute path baked in.
_HOPPER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hopper")
if _HOPPER_DIR not in sys.path:
    sys.path.insert(0, _HOPPER_DIR)

from flash_attn_interface import (  # in-tree hopper build
    ring_bwd_alloc,
    flash_attn_varlen_func,    # stock FA3 varlen forward -- used per-hop by the varlen ring (Option A)
)

# Ring forward/backward are the stock `fwd`/`bwd` ops with a flag flipped (folded in; NO
# separate ops or wrappers) -- thin call-throughs, no duplicated logic.
#   forward:  fwd(..., skip_combine=True, out_accum=, lse_accum=) forces the Split kernel to
#             write fp32 partials into caller-owned buffers and skips the combine.
#   backward: bwd(..., ring_phase=1/2/3, dq_accum=, dsoftmax_sum=, softmax_lse_log2=) is the
#             phased backward; the three fp32 buffers are caller-owned (see ring_bwd_alloc).
_fwd_op = torch.ops.flash_attn_3.fwd
_bwd_op = torch.ops.flash_attn_3.bwd


def _fwd_ring(q, k, v, out_accum, lse_accum, softmax_scale=None, causal=False,
              window_size=(-1, -1), softcap=0.0, num_splits=2,
              cu_seqlens_q=None, cu_seqlens_k=None, seqused_q=None, seqused_k=None,
              max_seqlen_q=None, max_seqlen_k=None):
    q, k, v = (x.contiguous() for x in (q, k, v))
    _fwd_op(q, k, v,
            softmax_scale=softmax_scale, is_causal=causal,
            window_size_left=window_size[0], window_size_right=window_size[1],
            softcap=softcap, num_splits=num_splits,
            cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
            seqused_q=seqused_q, seqused_k=seqused_k,
            max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
            skip_combine=True, out_accum=out_accum, lse_accum=lse_accum)


def _bwd_ring(phase, dout, q, k, v, out, softmax_lse, dq, dk, dv,
              dq_accum, dsoftmax_sum, softmax_lse_log2, softmax_scale=None, causal=False,
              cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=None, max_seqlen_k=None):
    dout, q, k, v, out = (x.contiguous() for x in (dout, q, k, v, out))
    _bwd_op(dout, q, k, v, out, softmax_lse, dq, dk, dv,
            cu_seqlens_q, cu_seqlens_k, None, None, max_seqlen_q, max_seqlen_k,
            softmax_scale, causal, -1, -1, 0.0, False, 0,
            phase, dq_accum, dsoftmax_sum, softmax_lse_log2)


# ============================================================================
# Ring communication: double-buffered. K/V (or the in-flight dK/dV accumulator) are
# packed into one [2, ...] tensor and exchanged with a single batch_isend_irecv
# ("send to the next rank, receive from the previous rank"). The sends/recvs are
# non-blocking, so a hop's communication overlaps with that hop's compute; a matched
# wait() drains the oldest outstanding exchange.
# ============================================================================
class DoubleBufRingComm:
    def __init__(self, process_group):
        self._pg = process_group
        self.rank = dist.get_rank(process_group)
        self.world_size = dist.get_world_size(process_group)
        # Ring topology: send to rank+1, receive from rank-1. Map to GLOBAL ranks so
        # this works when `process_group` is a sub-group of WORLD.
        self.send_rank = (self.rank + 1) % self.world_size
        self.recv_rank = (self.rank - 1) % self.world_size
        if process_group is not None:
            self.send_rank = dist.get_global_rank(process_group, self.send_rank)
            self.recv_rank = dist.get_global_rank(process_group, self.recv_rank)
        self._pending = []  # FIFO of issued isend/irecv handle batches

    def send_recv_packed(self, send_buf, recv_buf):
        """Send `send_buf` to the next rank while receiving into `recv_buf` from the
        previous rank. Non-blocking: returns immediately, pairs with a later wait()."""
        self._pending.append(dist.batch_isend_irecv([
            dist.P2POp(dist.isend, send_buf, self.send_rank, group=self._pg),
            dist.P2POp(dist.irecv, recv_buf, self.recv_rank, group=self._pg)]))

    def wait(self):
        """Wait on the earliest still-outstanding exchange (one-to-one with a
        preceding send_recv_packed call)."""
        if self._pending:
            for req in self._pending.pop(0):
                req.wait()


# ============================================================================
# Online softmax merge -- an inlined fused Triton kernel that merges one NORMALIZED
# partial (blk_o, blk_lse) into a persistent fp32 accumulator (acc, lse) IN PLACE:
#   blk_o/acc: (B, Hq, S, D) fp32 (each already normalized over its own key block);
#   blk_lse/lse: (B, Hq, S) fp32
# Both sides are normalized, so the merge weights sum to 1:
#   new = max(lse, blk_lse); a = exp(lse - new); b = exp(blk_lse - new)
#   acc = (acc*a + blk_o*b) / (a + b);  lse = new + log(a + b)
# is_first=True just seeds the accumulator (nothing to merge with yet). Tensors are
# indexed as a flat [B*Hq, S, D] / [B*Hq, S] contiguous layout; the grid parallelizes
# over (sequence tiles, B*Hq). Everything stays in fp32 -- no bf16 round-trip.
# ============================================================================
def _next_pow2(x):
    return 1 << (x - 1).bit_length()


@triton.jit
def _merge_kernel(blk_o_ptr, blk_lse_ptr, acc_ptr, lse_ptr, s, d,
                  is_first: tl.constexpr, BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr):
    pid_s = tl.program_id(0)        # tile along the sequence dim
    pid_bh = tl.program_id(1)       # one program per (batch*head)
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_d = tl.arange(0, BLOCK_D)
    s_mask = offs_s < s
    d_mask = offs_d < d
    o_ptrs = pid_bh * s * d + offs_s[:, None] * d + offs_d[None, :]   # element offsets for acc/blk_o
    lse_ptrs = pid_bh * s + offs_s                                    # element offsets for lse/blk_lse
    blk_o = tl.load(blk_o_ptr + o_ptrs, mask=s_mask[:, None] & d_mask[None, :], other=0.0)
    blk_lse = tl.load(blk_lse_ptr + lse_ptrs, mask=s_mask, other=-float("inf"))
    if is_first:
        tl.store(acc_ptr + o_ptrs, blk_o, mask=s_mask[:, None] & d_mask[None, :])
        tl.store(lse_ptr + lse_ptrs, blk_lse, mask=s_mask)
    else:
        acc = tl.load(acc_ptr + o_ptrs, mask=s_mask[:, None] & d_mask[None, :], other=0.0)
        old = tl.load(lse_ptr + lse_ptrs, mask=s_mask, other=-float("inf"))
        new = tl.maximum(old, blk_lse)
        a = tl.exp(old - new)
        b = tl.exp(blk_lse - new)
        denom = a + b
        tl.store(acc_ptr + o_ptrs, (acc * a[:, None] + blk_o * b[:, None]) / denom[:, None],
                 mask=s_mask[:, None] & d_mask[None, :])
        tl.store(lse_ptr + lse_ptrs, new + tl.log(denom), mask=s_mask)


def _merge(blk_o, blk_lse, acc, lse, is_first):
    """Merge a normalized partial (blk_o, blk_lse) into (acc, lse) in place. All
    tensors must be (B,Hq,S,D)/(B,Hq,S) contiguous fp32."""
    b, h, s, d = acc.shape
    BLOCK_D = _next_pow2(d)
    BLOCK_S = max(1, min(128, 8192 // BLOCK_D))
    grid = (triton.cdiv(s, BLOCK_S), b * h)
    _merge_kernel[grid](blk_o, blk_lse, acc, lse, s, d, is_first,
                        BLOCK_S=BLOCK_S, BLOCK_D=BLOCK_D, num_warps=4, num_stages=2)


# ============================================================================
# Dedicated varlen merge: fuses (a) the token-major -> head-major transpose, (b) the
# bf16/fp16 -> fp32 cast, and (c) the online-softmax merge into ONE kernel pass, so the
# per-hop block output from flash_attn_varlen_func is consumed IN ITS NATIVE bf16 layout
# with no intermediate `.permute().contiguous().float()` materialization. Layouts:
#   blk_o: (total, Hq, D) bf16/fp16, contiguous (flash_attn_varlen_func output)
#   blk_lse: (Hq, total) fp32 ; acc: (Hq, total, D) fp32 ; lse: (Hq, total) fp32
# One program per (token tile, head); the bf16 load is promoted to fp32 in registers.
# ============================================================================
@triton.jit
def _merge_varlen_kernel(blk_o_ptr, blk_lse_ptr, acc_ptr, lse_ptr, total, Hq, D,
                         is_first: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr):
    pid_t = tl.program_id(0)        # tile along the (packed) token dim
    pid_h = tl.program_id(1)        # one program per head
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_d = tl.arange(0, BLOCK_D)
    t_mask = offs_t < total
    d_mask = offs_d < D
    bo_ptrs = offs_t[:, None] * (Hq * D) + pid_h * D + offs_d[None, :]   # blk_o[t, h, d] token-major
    acc_ptrs = pid_h * (total * D) + offs_t[:, None] * D + offs_d[None, :]  # acc[h, t, d] head-major
    l_ptrs = pid_h * total + offs_t                                      # blk_lse/lse[h, t]
    blk_o = tl.load(blk_o_ptr + bo_ptrs, mask=t_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    blk_lse = tl.load(blk_lse_ptr + l_ptrs, mask=t_mask, other=-float("inf"))
    if is_first:
        tl.store(acc_ptr + acc_ptrs, blk_o, mask=t_mask[:, None] & d_mask[None, :])
        tl.store(lse_ptr + l_ptrs, blk_lse, mask=t_mask)
    else:
        acc = tl.load(acc_ptr + acc_ptrs, mask=t_mask[:, None] & d_mask[None, :], other=0.0)
        old = tl.load(lse_ptr + l_ptrs, mask=t_mask, other=-float("inf"))
        new = tl.maximum(old, blk_lse)
        a = tl.exp(old - new)
        b = tl.exp(blk_lse - new)
        denom = a + b
        tl.store(acc_ptr + acc_ptrs, (acc * a[:, None] + blk_o * b[:, None]) / denom[:, None],
                 mask=t_mask[:, None] & d_mask[None, :])
        tl.store(lse_ptr + l_ptrs, new + tl.log(denom), mask=t_mask)


def _merge_varlen(blk_o, blk_lse, acc, lse, is_first):
    """Merge one varlen block output into the fp32 accumulator IN PLACE, fusing the
    bf16->fp32 cast + layout transpose (see kernel comment). blk_o: (total, Hq, D)
    bf16/fp16 contiguous; blk_lse: (Hq, total) fp32; acc: (Hq, total, D) fp32;
    lse: (Hq, total) fp32."""
    Hq, total, D = acc.shape
    BLOCK_D = _next_pow2(D)
    BLOCK_T = max(1, min(128, 8192 // BLOCK_D))
    grid = (triton.cdiv(total, BLOCK_T), Hq)
    _merge_varlen_kernel[grid](blk_o, blk_lse, acc, lse, total, Hq, D, is_first,
                               BLOCK_T=BLOCK_T, BLOCK_D=BLOCK_D, num_warps=4, num_stages=2)


# Number of sub-splits `_fwd_ring` (fwd with skip_combine) writes for a block. A single-split
# causal block needs num_splits>=2 (FA3 convention); a non-causal block needs 1.
_NS_FULL = 1     # non-causal (off-diagonal) block
_NS_CAUSAL = 2   # causal diagonal block


# ============================================================================
# Forward: rotate K/V around the ring; each hop emits an fp32 partial via
# `_fwd_ring` (fwd with skip_combine) and merges it online into the accumulator. Returns
# out (B,S,Hq,D) in the input dtype, and lse (B,Hq,S) fp32 (already the [b,h,s]
# layout the FA3 backward wants). Runs under no_grad -- autograd is wired up by
# _RingFlashAttnFunc below.
# ============================================================================
@torch.no_grad()
def _ring_forward(process_group, q, k, v, softmax_scale, causal):
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    comm = DoubleBufRingComm(process_group)
    W, rank = comm.world_size, comm.rank
    B, S, Hq, D = q.shape
    dev = q.device

    # Double buffer for received K/V (each slot holds packed [k; v], shape
    # (2,)+k.shape), plus one "to send" buffer seeded with the local K/V.
    kv_bufs = [torch.empty((2,) + k.shape, device=dev, dtype=k.dtype) for _ in range(2)]
    kv_send = torch.empty((2,) + k.shape, device=dev, dtype=k.dtype)
    if W > 1:
        kv_send[0].copy_(k); kv_send[1].copy_(v)
    k_cur, v_cur = k, v

    # Partial output buffers (up to _NS_CAUSAL sub-splits) and the PERSISTENT fp32
    # accumulator that everything is merged into in place.
    part_o = torch.empty((_NS_CAUSAL, B, Hq, S, D), device=dev, dtype=torch.float32)
    part_lse = torch.empty((_NS_CAUSAL, B, Hq, S), device=dev, dtype=torch.float32)
    acc = torch.empty((B, Hq, S, D), device=dev, dtype=torch.float32)
    lse = torch.empty((B, Hq, S), device=dev, dtype=torch.float32)
    first = True

    for step in range(W):
        # Kick off the NEXT hop's K/V transfer before computing on the current block,
        # so communication overlaps compute.
        if step + 1 != W:
            comm.send_recv_packed(kv_send, kv_bufs[step & 1])
        # Causal ring: rank r only processes steps 0..r. step 0 is its own diagonal
        # block (the only masked block); later steps are full (unmasked) blocks.
        if (not causal) or step <= rank:
            block_causal = causal and step == 0
            ns = _NS_CAUSAL if block_causal else _NS_FULL
            # Emit `ns` normalized fp32 partials + fp32 LSE for the current K/V block.
            _fwd_ring(
                q, k_cur, v_cur, part_o[:ns], part_lse[:ns],
                softmax_scale=softmax_scale, causal=block_causal, num_splits=ns)
            for j in range(ns):
                _merge(part_o[j], part_lse[j], acc, lse, is_first=first)
                first = False
        # Wait for this hop's K/V to arrive, switch to the other buffer half, and use
        # it as the next hop's send source.
        if step + 1 != W:
            comm.wait()
            kv_send = kv_bufs[step & 1]
            k_cur, v_cur = kv_bufs[step & 1][0], kv_bufs[step & 1][1]

    out = acc.permute(0, 2, 1, 3).contiguous().to(q.dtype)  # (B,Hq,S,D)->(B,S,Hq,D), one fp32->bf16
    return out, lse


# ============================================================================
# Backward: phased FA3 backward. dQ stays in a local PERSISTENT fp32 dq_accum (never
# communicated); dK/dV are reduced in fp32 along a SECOND ring back to their owner
# rank. Order: phase 1 once -> phase 2 per hop -> phase 3 once. Runs under no_grad.
# ============================================================================
@torch.no_grad()
def _ring_backward(process_group, dout, q, k, v, out, softmax_lse, softmax_scale, causal):
    dout, q, k, v, out = [x.contiguous() for x in (dout, q, k, v, out)]
    kv_comm = DoubleBufRingComm(process_group)     # rotates K/V
    d_kv_comm = DoubleBufRingComm(process_group)   # rotates the in-flight dK/dV accumulator
    W, rank = kv_comm.world_size, kv_comm.rank
    B, S, Hq, D = q.shape
    dev = q.device

    # The three PERSISTENT fp32 buffers: dq_accum (accumulates dQ across hops),
    # dsoftmax_sum (holds D), and softmax_lse_log2. ring_bwd_alloc sizes them with the
    # block rounding the phased backward expects (hd128 uses the tuned 80-block).
    # NOTE: pass `device` as a keyword (its positional slot is head_size_v).
    dq_accum, dsoftmax, lse_log2 = ring_bwd_alloc(B, S, Hq, D, device=dev)

    dq = torch.empty_like(q)          # written by phase 3 (bf16)
    block_dk = torch.empty_like(k)    # this hop's dK contribution (bf16, h_kv heads)
    block_dv = torch.empty_like(k)

    # phase 1 (once per rank): D + softmax_lse_log2 + clear dq_accum. dO/O/LSE are
    # fixed for this rank across all ring steps, so this runs exactly once.
    _bwd_ring(
        1, dout, q, k, v, out, softmax_lse,
        dq, block_dk, block_dv, dq_accum, dsoftmax, lse_log2,
        softmax_scale=softmax_scale, causal=causal)

    # K/V double buffer + an fp32 dK/dV double buffer (packed [dk; dv], shape
    # (2,)+k.shape) that carries the running dK/dV reduction around the ring.
    kv_bufs = [torch.empty((2,) + k.shape, device=dev, dtype=k.dtype) for _ in range(2)]
    kv_send = torch.empty((2,) + k.shape, device=dev, dtype=k.dtype)
    if W > 1:
        kv_send[0].copy_(k); kv_send[1].copy_(v)
    k_cur, v_cur = k, v
    dkdv_bufs = [torch.empty((2,) + k.shape, device=dev, dtype=torch.float32) for _ in range(2)]
    dk_bufs = [dkdv_bufs[0][0], dkdv_bufs[1][0]]
    dv_bufs = [dkdv_bufs[0][1], dkdv_bufs[1][1]]
    first_iter_done = False

    for step in range(W):
        if step + 1 != W:
            kv_comm.send_recv_packed(kv_send, kv_bufs[step & 1])
        active = step <= rank or not causal
        prev_slot = (step - 1) & 1
        if active:
            # phase 2 (per hop): atomicAdd this block's dQ into the persistent
            # dq_accum, and write this block's dK/dV (h_kv-headed, GQA-reduced).
            _bwd_ring(
                2, dout, q, k_cur, v_cur, out, softmax_lse,
                dq, block_dk, block_dv, dq_accum, dsoftmax, lse_log2,
                softmax_scale=softmax_scale, causal=(causal and step == 0))
            # Fold this block's dK/dV into the rotating fp32 accumulator slot (copy the
            # first time, add afterwards). Wait for the slot arriving from the previous
            # hop before adding into it.
            if first_iter_done:
                d_kv_comm.wait()
            if not first_iter_done:
                dk_bufs[prev_slot].copy_(block_dk); dv_bufs[prev_slot].copy_(block_dv)
            else:
                dk_bufs[prev_slot].add_(block_dk); dv_bufs[prev_slot].add_(block_dv)
            first_iter_done = True
        elif step != 0:
            d_kv_comm.wait()  # inactive steps still drain the second ring to stay in sync
        if step + 1 != W:
            kv_comm.wait()
            kv_send = kv_bufs[step & 1]
            k_cur, v_cur = kv_bufs[step & 1][0], kv_bufs[step & 1][1]
        # Forward the current dK/dV accumulator slot to the next rank; it keeps
        # accumulating on other ranks and returns to this K/V's owner after W hops.
        d_kv_comm.send_recv_packed(dkdv_bufs[prev_slot], dkdv_bufs[step & 1])

    # phase 3 (once per rank): persistent fp32 dq_accum -> bf16 dQ, scaled once.
    # Launched BEFORE the final dK/dV wait: the convert only reads dq_accum and writes
    # dq (it never touches dk_bufs/dv_bufs), so it overlaps the exposed tail dK/dV
    # exchange on the comm stream instead of serializing strictly after it.
    _bwd_ring(
        3, dout, q, k, v, out, softmax_lse,
        dq, block_dk, block_dv, dq_accum, dsoftmax, lse_log2,
        softmax_scale=softmax_scale, causal=causal)

    d_kv_comm.wait()  # now block on the final dK/dV slot before reading it below
    final_slot = (W - 1) & 1  # after W hops the complete dK/dV for this rank lands here
    return dq.to(q.dtype), dk_bufs[final_slot].to(k.dtype), dv_bufs[final_slot].to(v.dtype)


# ============================================================================
# autograd.Function: the forward saves (q, k, v, out, lse) for the backward, which
# calls the phased _ring_backward. Only q/k/v receive gradients.
# ============================================================================
class _RingFlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, softmax_scale, causal, group):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** -0.5
        out, lse = _ring_forward(group, q, k, v, softmax_scale, causal)
        ctx.save_for_backward(q, k.contiguous(), v.contiguous(), out, lse.contiguous())
        ctx.softmax_scale, ctx.causal, ctx.group = softmax_scale, causal, group
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = _ring_backward(ctx.group, dout, q, k, v, out, lse, ctx.softmax_scale, ctx.causal)
        return dq, dk, dv, None, None, None  # match forward's 6 inputs (last 3 are non-tensors)


def ring_flash_attn_func(q, k, v, softmax_scale=None, causal=False, group=None):
    """Optimized FA3 ring / context-parallel attention (autograd-enabled).

    q/k/v are THIS rank's sequence shard: (B, S_local, H, D), bf16/fp16, last dim
    contiguous. Returns out (B, S_local, Hq, D) in the input dtype.

    softmax_scale: defaults to head_dim**-0.5.
    causal:        plain causal ring (rank r attends key shards 0..r; the diagonal
                   block carries the mask).
    group:         a torch.distributed process group defining the ring; defaults to
                   WORLD. World size = number of shards the sequence was split into.
    """
    return _RingFlashAttnFunc.apply(q, k, v, softmax_scale, causal, group)


# ---------------------------------------------------------------------------
# Convenience helpers for context-parallel sharding.
# ---------------------------------------------------------------------------
def shard_along_seq(x_global, rank, world_size):
    """Split (B, S_global, H, D) along the sequence dim into `world_size` contiguous
    chunks and return this rank's chunk. S_global must be divisible by world_size."""
    assert x_global.shape[1] % world_size == 0, "global sequence length must be divisible by world_size"
    return torch.chunk(x_global, world_size, dim=1)[rank].contiguous()


def ring_attention(q_local, k_local, v_local, causal=False, softmax_scale=None, group=None):
    """Alias of ring_flash_attn_func for inputs already sharded to this rank -- the
    single call you drop into a context-parallel training/inference step."""
    return ring_flash_attn_func(q_local, k_local, v_local,
                                softmax_scale=softmax_scale, causal=causal, group=group)


# ############################################################################
# Varlen (context-parallel) ring: forward = Option A (per-hop flash_attn_varlen_func
# + online fp32 merge), backward = phased bwd_ring. Each GLOBAL sequence is split
# into W equal chunks along its length (every sequence length must be divisible by
# W); rank r owns chunk r of every sequence, packed. Chunk sizes are therefore
# uniform across ranks (fixed-size ring comm), and the per-hop K/V block is itself a
# valid varlen batch with the SAME cu_seqlens as Q. The plain causal-ring logic
# carries over per sequence: at hop h rank r sees chunk (r-h) of each sequence, which
# is entirely in the past for h>0 (full block) and the diagonal for h==0 (masked).
# (the forced-Split ring forward is not used for varlen: its store-all path is unusable
# under the dynamic varlen split scheduler -- see the module docstring.)
# ############################################################################
def shard_varlen_along_seq(x_global, cu_global, rank, world_size):
    """Split each sequence (delimited by cu_global) into `world_size` EQUAL chunks
    along its length and return (this rank's chunk-r packed tensor, local cu_seqlens,
    local max_seqlen). x_global: (total_global, H, D) packed varlen. Every sequence
    length must be divisible by world_size. The local cu_seqlens is identical on every
    rank (all chunks of sequence i have length seqlen_i / world_size)."""
    seqlens = (cu_global[1:] - cu_global[:-1]).tolist()
    lens_loc, chunks = [], []
    for i, s in enumerate(seqlens):
        assert s % world_size == 0, "each sequence length must be divisible by world_size"
        c = s // world_size
        base = int(cu_global[i]) + rank * c
        chunks.append(x_global[base:base + c])
        lens_loc.append(c)
    x_loc = torch.cat(chunks, 0).contiguous()
    cu_loc = torch.tensor([0] + list(torch.tensor(lens_loc).cumsum(0)), dtype=torch.int32, device=x_global.device)
    return x_loc, cu_loc, max(lens_loc)


@torch.no_grad()
def _ring_forward_varlen(process_group, q, k, v, cu, max_s, softmax_scale, causal):
    """Option A varlen forward: each hop runs flash_attn_varlen_func on the current K/V
    chunk (out+lse) and merges online in fp32. q/k/v: (total, H, D) packed varlen; cu:
    local cu_seqlens (same per-seq lengths on every rank). Returns out (total, Hq, D) in
    q.dtype and lse (Hq, total) fp32 (the layout the FA3 backward wants)."""
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    comm = DoubleBufRingComm(process_group)
    W, rank = comm.world_size, comm.rank
    total, Hq, D = q.shape
    dev = q.device
    kv_bufs = [torch.empty((2,) + k.shape, device=dev, dtype=k.dtype) for _ in range(2)]
    kv_send = torch.empty((2,) + k.shape, device=dev, dtype=k.dtype)
    if W > 1:
        kv_send[0].copy_(k); kv_send[1].copy_(v)
    k_cur, v_cur = k, v
    acc = torch.empty((Hq, total, D), device=dev, dtype=torch.float32)      # head-major fp32 accumulator
    lse = torch.empty((Hq, total), device=dev, dtype=torch.float32)
    first = True
    for step in range(W):
        if step + 1 != W:
            comm.send_recv_packed(kv_send, kv_bufs[step & 1])
        if (not causal) or step <= rank:
            block_causal = causal and step == 0
            out_blk, lse_blk = flash_attn_varlen_func(
                q, k_cur, v_cur, cu, cu, max_s, max_s, softmax_scale=softmax_scale,
                causal=block_causal, return_attn_probs=True)  # out (total,Hq,D) bf16 ; lse (Hq,total) fp32
            # fused: consumes bf16 out_blk directly (bf16->fp32 cast + transpose + merge in one pass)
            _merge_varlen(out_blk, lse_blk, acc, lse, is_first=first); first = False
        if step + 1 != W:
            comm.wait()
            kv_send = kv_bufs[step & 1]
            k_cur, v_cur = kv_bufs[step & 1][0], kv_bufs[step & 1][1]
    out = acc.permute(1, 0, 2).contiguous().to(q.dtype)      # (Hq,total,D) -> (total, Hq, D), one fp32->bf16
    return out, lse                                          # lse (Hq, total) fp32


@torch.no_grad()
def _ring_backward_varlen(process_group, dout, q, k, v, out, softmax_lse, cu, max_s, softmax_scale, causal):
    """Phased bwd_ring varlen backward (mirrors _ring_backward, varlen-threaded). dQ
    stays in a persistent fp32 dq_accum; dK/dV are reduced in fp32 along a second ring.
    q/k/v/out: (total, H, D); dout: (total, Hq, D); softmax_lse: (Hq, total)."""
    dout, q, k, v, out = [x.contiguous() for x in (dout, q, k, v, out)]
    kv_comm = DoubleBufRingComm(process_group)
    d_kv_comm = DoubleBufRingComm(process_group)
    W, rank = kv_comm.world_size, kv_comm.rank
    total, Hq, D = q.shape
    batch = cu.numel() - 1
    dev = q.device
    dq_accum, dsoftmax, lse_log2 = ring_bwd_alloc(batch, max_s, Hq, D, device=dev, total_q=total)
    dq = torch.empty_like(q); block_dk = torch.empty_like(k); block_dv = torch.empty_like(k)

    def _bwd(phase, kk, vv, blk_causal):
        _bwd_ring(
            phase, dout, q, kk, vv, out, softmax_lse, dq, block_dk, block_dv,
            dq_accum, dsoftmax, lse_log2, softmax_scale=softmax_scale, causal=blk_causal,
            cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=max_s, max_seqlen_k=max_s)

    _bwd(1, k, v, causal)  # phase 1 (once/rank)

    kv_bufs = [torch.empty((2,) + k.shape, device=dev, dtype=k.dtype) for _ in range(2)]
    kv_send = torch.empty((2,) + k.shape, device=dev, dtype=k.dtype)
    if W > 1:
        kv_send[0].copy_(k); kv_send[1].copy_(v)
    k_cur, v_cur = k, v
    dkdv_bufs = [torch.empty((2,) + k.shape, device=dev, dtype=torch.float32) for _ in range(2)]
    dk_bufs = [dkdv_bufs[0][0], dkdv_bufs[1][0]]
    dv_bufs = [dkdv_bufs[0][1], dkdv_bufs[1][1]]
    first_iter_done = False
    for step in range(W):
        if step + 1 != W:
            kv_comm.send_recv_packed(kv_send, kv_bufs[step & 1])
        active = step <= rank or not causal
        prev_slot = (step - 1) & 1
        if active:
            _bwd(2, k_cur, v_cur, causal and step == 0)  # phase 2 (per hop): dQ into dq_accum + this chunk's dK/dV
            if first_iter_done:
                d_kv_comm.wait()
            if not first_iter_done:
                dk_bufs[prev_slot].copy_(block_dk); dv_bufs[prev_slot].copy_(block_dv)
            else:
                dk_bufs[prev_slot].add_(block_dk); dv_bufs[prev_slot].add_(block_dv)
            first_iter_done = True
        elif step != 0:
            d_kv_comm.wait()
        if step + 1 != W:
            kv_comm.wait()
            kv_send = kv_bufs[step & 1]
            k_cur, v_cur = kv_bufs[step & 1][0], kv_bufs[step & 1][1]
        d_kv_comm.send_recv_packed(dkdv_bufs[prev_slot], dkdv_bufs[step & 1])
    _bwd(3, k, v, causal)  # phase 3 (once/rank): dq_accum -> dQ; launched BEFORE the
    d_kv_comm.wait()       # final wait so the convert overlaps the exposed tail dK/dV recv
    final_slot = (W - 1) & 1
    return dq.to(q.dtype), dk_bufs[final_slot].to(k.dtype), dv_bufs[final_slot].to(v.dtype)


class _RingVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, cu, max_s, softmax_scale, causal, group):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** -0.5
        out, lse = _ring_forward_varlen(group, q, k, v, cu, max_s, softmax_scale, causal)
        ctx.save_for_backward(q, k.contiguous(), v.contiguous(), out, lse.contiguous(), cu)
        ctx.softmax_scale, ctx.causal, ctx.group, ctx.max_s = softmax_scale, causal, group, max_s
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, lse, cu = ctx.saved_tensors
        dq, dk, dv = _ring_backward_varlen(ctx.group, dout, q, k, v, out, lse, cu, ctx.max_s,
                                           ctx.softmax_scale, ctx.causal)
        return dq, dk, dv, None, None, None, None, None  # 8 forward inputs; only q/k/v get grads


def ring_flash_attn_varlen_func(q, k, v, cu_seqlens, max_seqlen, softmax_scale=None,
                                causal=False, group=None):
    """Varlen ring / context-parallel attention (autograd-enabled). Forward = Option A
    (per-hop flash_attn_varlen_func + online fp32 merge); backward = phased bwd_ring.

    q/k/v: THIS rank's per-sequence CHUNK, packed varlen -> (total_local, H, D), bf16/fp16.
    cu_seqlens / max_seqlen: the LOCAL chunk lengths (identical on every rank). Each
    global sequence is split into world_size equal chunks along its length, so every
    global sequence length must be divisible by world_size -- see shard_varlen_along_seq.
    Returns out (total_local, Hq, D) in the input dtype."""
    return _RingVarlenFunc.apply(q, k, v, cu_seqlens, max_seqlen, softmax_scale, causal, group)



# ============================================================================
# Runnable example: run the optimized ring attention (forward + backward) across the
# GPUs launched by torchrun, and sanity-check it against a single-GPU full-sequence
# FA3 flash reference on the same inputs. Every rank builds the SAME global tensors
# (seeded), shards them, runs the ring on its shard, and compares its shard of the
# result with the reference. Errors are max-reduced across ranks; rank 0 prints.
#
#   torchrun --nproc_per_node=8 --standalone ring_flash_attn_fa3.py
# ============================================================================
if __name__ == "__main__":
    dist.init_process_group("nccl")
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

        def log(*a):
            if rank == 0:
                print(*a, flush=True)

        # ---- config (override via env if you like) ----
        B = int(os.environ.get("B", 1))
        S_global = int(os.environ.get("S", 8192))          # total sequence length across all ranks
        Hq = int(os.environ.get("HQ", 16))                 # query heads
        Hkv = int(os.environ.get("HKV", 2))                # key/value heads (GQA if < Hq)
        D = int(os.environ.get("D", 128))                  # head dim
        causal = os.environ.get("CAUSAL", "1") == "1"
        dtype = torch.bfloat16
        scale = D ** -0.5
        assert S_global % world_size == 0, "S must be divisible by the number of GPUs"
        S_local = S_global // world_size

        log(f"world_size={world_size}  device={torch.cuda.get_device_name(local_rank)}")
        log(f"config: B={B} S_global={S_global} (S_local={S_local}) Hq={Hq} Hkv={Hkv} D={D} "
            f"causal={causal} {dtype}")

        # ---- build identical global Q/K/V on every rank, then shard along the seq dim ----
        torch.manual_seed(2024)
        qg = torch.randn(B, S_global, Hq, D, device=device, dtype=dtype)
        kg = torch.randn(B, S_global, Hkv, D, device=device, dtype=dtype)
        vg = torch.randn(B, S_global, Hkv, D, device=device, dtype=dtype)
        dog = torch.randn_like(qg)                          # upstream gradient for the backward

        # local shards this rank owns (require grad so we can run the ring backward)
        q = shard_along_seq(qg, rank, world_size).detach().requires_grad_(True)
        k = shard_along_seq(kg, rank, world_size).detach().requires_grad_(True)
        v = shard_along_seq(vg, rank, world_size).detach().requires_grad_(True)

        # ---- the actual usage: one call for the forward, autograd for the backward ----
        out = ring_attention(q, k, v, causal=causal, softmax_scale=scale)
        out.backward(shard_along_seq(dog, rank, world_size))
        log(f"ring output shape (this rank's shard): {tuple(out.shape)}")

        # ---- sanity check vs a single-GPU full-sequence FA3 flash on the same inputs ----
        from flash_attn_interface import flash_attn_func  # stock FA3 (whole sequence, one GPU)
        qf = qg.detach().requires_grad_(True)
        kf = kg.detach().requires_grad_(True)
        vf = vg.detach().requires_grad_(True)
        ref = flash_attn_func(qf, kf, vf, softmax_scale=scale, causal=causal)
        ref.backward(dog)

        def err(a, b):
            return (a.float() - b.float()).abs().max().item()

        # Per-tensor max abs error vs the reference (this rank's shard), max-reduced
        # across ranks. Both sides are bf16; the ring keeps out/dQ in fp32 so those
        # track flash tightly, while dK/dV differ only by bf16 rounding (a few 1e-2 on
        # a long causal sequence -- expected, not a bug).
        errs = torch.tensor([
            err(out, shard_along_seq(ref.detach(), rank, world_size)),
            err(q.grad, shard_along_seq(qf.grad, rank, world_size)),
            err(k.grad, shard_along_seq(kf.grad, rank, world_size)),
            err(v.grad, shard_along_seq(vf.grad, rank, world_size)),
        ], device=device, dtype=torch.float64)
        dist.all_reduce(errs, op=dist.ReduceOp.MAX)
        o_e, dq_e, dk_e, dv_e = errs.tolist()
        log(f"max abs error vs single-GPU flash (all ranks): "
            f"out={o_e:.2e} dq={dq_e:.2e} dk={dk_e:.2e} dv={dv_e:.2e}")
        ok = o_e < 2e-2 and max(dq_e, dk_e, dv_e) < 1.2e-1  # generous bf16 backward tolerance
        log("OK: ring matches single-GPU flash within bf16 tolerance"
            if ok else "WARNING: error larger than the bf16 tolerance -- check the build/config")

        # ---- varlen ring example: shard each sequence into W chunks, run the varlen ring,
        #      compare to a single-GPU full-sequence flash_attn_varlen_func ----
        log()
        VB = int(os.environ.get("VB", 3))                       # number of (variable-length) sequences
        unit = 8 * world_size                                   # seqlens are multiples of this so each chunk is a multiple of 8
        gcpu = torch.Generator(device="cpu").manual_seed(11)
        vseqlens = [((x // unit) + 1) * unit for x in torch.randint(unit, 12 * unit, (VB,), generator=gcpu).tolist()]
        vtotal = sum(vseqlens)
        vcu = torch.tensor([0] + list(torch.tensor(vseqlens).cumsum(0)), dtype=torch.int32, device=device)
        vmax = max(vseqlens)
        torch.manual_seed(11)
        vqg = torch.randn(vtotal, Hq, D, device=device, dtype=dtype)
        vkg = torch.randn(vtotal, Hkv, D, device=device, dtype=dtype)
        vvg = torch.randn(vtotal, Hkv, D, device=device, dtype=dtype)
        vdog = torch.randn_like(vqg)
        # reference: full-sequence varlen attention on one GPU (+ autograd)
        vqf = vqg.detach().requires_grad_(True); vkf = vkg.detach().requires_grad_(True); vvf = vvg.detach().requires_grad_(True)
        vref = flash_attn_varlen_func(vqf, vkf, vvf, vcu, vcu, vmax, vmax, softmax_scale=scale, causal=causal)
        vref.backward(vdog)
        # this rank owns chunk `rank` of every sequence
        vq, vcu_loc, vmax_loc = shard_varlen_along_seq(vqg, vcu, rank, world_size)
        vk, _, _ = shard_varlen_along_seq(vkg, vcu, rank, world_size)
        vv, _, _ = shard_varlen_along_seq(vvg, vcu, rank, world_size)
        vdo, _, _ = shard_varlen_along_seq(vdog, vcu, rank, world_size)
        vq = vq.detach().requires_grad_(True); vk = vk.detach().requires_grad_(True); vv = vv.detach().requires_grad_(True)
        vout = ring_flash_attn_varlen_func(vq, vk, vv, vcu_loc, vmax_loc, softmax_scale=scale, causal=causal)
        vout.backward(vdo)
        log(f"varlen ring: {VB} seqs, global total={vtotal}, this rank's chunk total={vout.shape[0]}; out shape {tuple(vout.shape)}")
        # compare this rank's chunk of {out, dq, dk, dv} to the reference's chunk
        vref_loc, _, _ = shard_varlen_along_seq(vref.detach(), vcu, rank, world_size)
        vdq_loc, _, _ = shard_varlen_along_seq(vqf.grad, vcu, rank, world_size)
        vdk_loc, _, _ = shard_varlen_along_seq(vkf.grad, vcu, rank, world_size)
        vdv_loc, _, _ = shard_varlen_along_seq(vvf.grad, vcu, rank, world_size)
        verrs = torch.tensor([err(vout, vref_loc), err(vq.grad, vdq_loc), err(vk.grad, vdk_loc), err(vv.grad, vdv_loc)],
                             device=device, dtype=torch.float64)
        dist.all_reduce(verrs, op=dist.ReduceOp.MAX)
        vo, vdq, vdk, vdv = verrs.tolist()
        log(f"[varlen] max abs error vs single-GPU flash_attn_varlen_func (all ranks): "
            f"out={vo:.2e} dq={vdq:.2e} dk={vdk:.2e} dv={vdv:.2e}")
        log("[varlen] OK: varlen ring matches single-GPU varlen flash within bf16 tolerance"
            if (vo < 2e-2 and max(vdq, vdk, vdv) < 1.2e-1) else "[varlen] WARNING: error larger than the bf16 tolerance")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
