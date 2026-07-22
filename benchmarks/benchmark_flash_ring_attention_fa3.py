"""
================================================================================
 Ring Flash Attention (FA3) benchmark: implementation + accuracy + before/after
================================================================================

Self-contained ring (context-parallel) attention built directly on the three
optimized FA3 ring primitives, plus an accuracy check and a before/after speed
comparison that isolates exactly what those primitives buy.

The three primitives (imported from the in-tree hopper build):
    from flash_attn_interface import (
        flash_attn_forward_ring,   # forward: one K/V block -> fp32 normalized partial + fp32 LSE
        flash_attn_backward_ring,  # backward: phased (1=preprocess / 2=step / 3=convert)
        ring_bwd_alloc,            # backward: allocate the 3 persistent fp32 buffers
    )

What the optimizations are (all additive on the FA3 side; the standard fwd/bwd
paths are untouched):

  * Forward `fwd_ring`: run the FA3 forward with Split=true so it writes an fp32
    normalized partial + fp32 LSE for a SINGLE K/V block, with NO combine and NO
    bf16 round-trip. The ring driver merges partials online in fp32 (fused Triton
    kernel) and casts to bf16 exactly once at the end. This removes the per-hop
    bf16 round-trip a naive ring pays (better accuracy, and fewer/lighter kernels).

  * Backward `bwd_ring`: split the FA3 backward into three separately launchable
    phases and keep dQ in a PERSISTENT fp32 accumulator across ring steps (the main
    kernel atomicAdds into it). preprocess (compute D / softmax_lse_log2 / clear
    dq_accum) and convert (dq_accum -> dQ) run ONCE per rank; only the main kernel
    runs per hop. This removes W-1 redundant preprocess + W-1 redundant convert per
    rank, and keeps dQ in fp32 the whole time (no per-hop bf16 round-trip for dQ).

  * hd128 specialization: in ring mode every hop (including the causal diagonal)
    uses the tuned kBlockM=80 block, so non-causal hops get the fast kernel while
    the persistent dq_accum keeps one consistent layout across mixed causal/
    non-causal steps (see run_mha_bwd_hdim128).

The before/after comparison uses two ring drivers that are IDENTICAL except for
those two primitives:
  * ring_opt   (after) : fwd_ring + phased flash_attn_backward_ring.
  * ring_stock (before): full-block flash_attn_func forward (bf16 out per hop) +
    full flash_attn_3.bwd per active hop (repeats preprocess+convert, dQ round-trips
    to bf16 each hop and is accumulated in Python fp32).
Everything else (ring communication, fp32 online merge, fp32 dK/dV ring reduction,
sequence sharding) is shared, so any difference reflects the primitives alone.

--------------------------------------------------------------------------------
 Supported configuration (of this ring driver + the primitives)
--------------------------------------------------------------------------------
  * dtype: fp16 / bf16.
  * head_dim: 64, 96, 128, 192, 256 (sm90). This script exercises 64/128/256.
  * MHA (hq == hkv) and GQA/MQA (hq a multiple of hkv).
  * causal and non-causal. causal uses the plain causal ring: rank r computes
    steps 0..r; step 0 is its own diagonal block (the only masked one).
  * softmax_scale, softcap: the primitives accept softcap (hdim<=64 + softcap is
    rejected because its kBlockM would differ across causal/non-causal ring steps;
    hd128 + softcap is consistent at kBlockM=64). This driver leaves softcap=0.

 NOT supported here:
  * window_size / local / sliding-window: this driver only implements the plain
    causal / non-causal ring (no cross-rank window remapping). Additionally, for
    hd128 the ring backward pins kBlockM=80 for plain causal but falls back to 64
    for local, while ring_bwd_alloc sizes the persistent buffers for 80 -> the
    layouts disagree, so local is not safe for hd128 rings even at the primitive
    level. Treat window_size as unsupported for the ring.
  * varlen (cu_seqlens_*): the primitives carry varlen plumbing (the backward has
    been validated single-GPU), but the forward store-all pattern used here is not
    usable under the dynamic varlen scheduler, and this driver assumes dense
    (B, S_local, H, D) inputs sharded along the sequence dim. A true varlen ring
    (sequences spanning ranks) is out of scope.
  * deterministic backward is off (dq_semaphore unused); paged-KV / appendKV / FP8
    / sm80 are out of scope.

--------------------------------------------------------------------------------
 Tensor layout / how to run
--------------------------------------------------------------------------------
  * q/k/v: (batch, seqlen_local, nheads, head_dim); last dim contiguous, 16B
    aligned; bf16/fp16.

  # 8 GPUs (recommended):
  torchrun --nproc_per_node=8 --standalone benchmark_flash_ring_attention_fa3.py
  # 2 GPUs:
  CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --standalone benchmark_flash_ring_attention_fa3.py

  Environment overrides:
    RING_HQ / RING_HKV        head counts for the compare/speed stages (default 16 / 2)
    RING_D_LIST               head dims for the speed grid   (default "64,128,256")
    RING_S_LIST               per-rank seqlens for the grid  (default "512,1024,2048,4096,8192")
    RING_REPS                 median-of-N repeats per grid cell (default 3)
    RING_SKIP_PRECISION       "1" to skip the accuracy test
    RING_SKIP_PRECISION_DIFF  "1" to skip the before/after accuracy table
    RING_SKIP_SPEED           "1" to skip the before/after speed grid

  The forward/backward ring primitives are imported from the in-tree hopper build,
  resolved relative to this file (../hopper), so no absolute path is baked in.
"""

import os
import sys

import torch
import torch.distributed as dist
import triton
import triton.language as tl

# Resolve the in-tree hopper build (which exposes the ring primitives) relative to
# this file: <repo>/benchmarks/this_file.py -> <repo>/hopper. No absolute path baked in.
_HOPPER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hopper")
if _HOPPER_DIR not in sys.path:
    sys.path.insert(0, _HOPPER_DIR)

from flash_attn_interface import (  # in-tree hopper build
    flash_attn_forward_ring,
    flash_attn_backward_ring,
    ring_bwd_alloc,
    flash_attn_func,               # full-block FA3 forward/backward (used by the "before" ring)
)

# "Before" ring backward: the standard full FA3 backward op (repeats preprocess+convert per hop).
_bwd = torch.ops.flash_attn_3.bwd


# ============================================================================
# Ring communication: double-buffered. K/V (or dK/dV) are packed into one [2, ...]
# tensor and exchanged with a single batch_isend_irecv ("send to next, recv from
# prev"). Non-blocking, so communication overlaps with compute.
# ============================================================================
class DoubleBufRingComm:
    def __init__(self, process_group):
        self._pg = process_group
        self.rank = dist.get_rank(process_group)
        self.world_size = dist.get_world_size(process_group)
        # Ring topology: send to rank+1, receive from rank-1 (mapped to global ranks
        # so sub-process-groups work).
        self.send_rank = (self.rank + 1) % self.world_size
        self.recv_rank = (self.rank - 1) % self.world_size
        if process_group is not None:
            self.send_rank = dist.get_global_rank(process_group, self.send_rank)
            self.recv_rank = dist.get_global_rank(process_group, self.recv_rank)
        self._pending = []  # FIFO of issued isend/irecv handle batches

    def send_recv_packed(self, send_buf, recv_buf):
        """Send send_buf to the next rank and receive into recv_buf from the previous
        rank (non-blocking, returns immediately)."""
        self._pending.append(dist.batch_isend_irecv([
            dist.P2POp(dist.isend, send_buf, self.send_rank, group=self._pg),
            dist.P2POp(dist.irecv, recv_buf, self.recv_rank, group=self._pg)]))

    def wait(self):
        """Wait on the earliest issued batch (paired one-to-one with send_recv_packed)."""
        if self._pending:
            for req in self._pending.pop(0):
                req.wait()


# ============================================================================
# Online softmax merge -- inlined fused Triton kernel (shared by both ring
# drivers; in-place streaming merge in fp32). It merges one NORMALIZED partial
# (blk_o, blk_lse) into the persistent accumulator (acc, lse) in place:
#   blk_o/acc: (B, Hq, S, D) fp32 (each normalized over its own keys); blk_lse/lse: (B, Hq, S) fp32
# Formula (both sides normalized, weights sum to 1):
#   new = max(lse, blk_lse); a = e^(lse-new); b = e^(blk_lse-new)
#   acc = (acc*a + blk_o*b) / (a+b);  lse = new + log(a+b)
# is_first=True just seeds the accumulator. Tensors are indexed as [B*Hq, S, D] contiguous.
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
    """Merge a normalized partial (blk_o, blk_lse) into acc/lse in place. All tensors
    must be (B,Hq,S,D)/(B,Hq,S) contiguous."""
    b, h, s, d = acc.shape
    BLOCK_D = _next_pow2(d)
    BLOCK_S = max(1, min(128, 8192 // BLOCK_D))
    grid = (triton.cdiv(s, BLOCK_S), b * h)
    _merge_kernel[grid](blk_o, blk_lse, acc, lse, s, d, is_first,
                        BLOCK_S=BLOCK_S, BLOCK_D=BLOCK_D, num_warps=4, num_stages=2)


_NS_FULL = 1     # non-causal (off-diagonal) block: a single partial is enough
_NS_CAUSAL = 2   # causal diagonal block: a single-split causal needs num_splits>=2 (FA3 convention)


# ############################################################################
# Optimized ring (ring_opt): fwd_ring + phased flash_attn_backward_ring
# ############################################################################
# ============================================================================
# Forward: rotate K/V around the ring; each hop emits an fp32 partial via
# flash_attn_forward_ring and merges it online into the accumulator. Returns
# out (B,S,Hq,D) and lse (B,Hq,S) fp32 (lse is already the [b,h,s] layout the
# FA3 backward wants).
# ============================================================================
@torch.no_grad()
def _ring_forward(process_group, q, k, v, softmax_scale, causal):
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    comm = DoubleBufRingComm(process_group)
    W, rank = comm.world_size, comm.rank
    B, S, Hq, D = q.shape
    dev = q.device

    # Double buffer for received K/V (each holds packed [k;v], shape (2,)+k.shape),
    # plus one "to send" buffer.
    kv_bufs = [torch.empty((2,) + k.shape, device=dev, dtype=k.dtype) for _ in range(2)]
    kv_send = torch.empty((2,) + k.shape, device=dev, dtype=k.dtype)
    if W > 1:
        kv_send[0].copy_(k); kv_send[1].copy_(v)  # seed the send buffer with local K/V
    k_cur, v_cur = k, v

    # Partial output buffers (up to _NS_CAUSAL sub-splits) and the PERSISTENT fp32
    # accumulator (merged into in place).
    part_o = torch.empty((_NS_CAUSAL, B, Hq, S, D), device=dev, dtype=torch.float32)
    part_lse = torch.empty((_NS_CAUSAL, B, Hq, S), device=dev, dtype=torch.float32)
    acc = torch.empty((B, Hq, S, D), device=dev, dtype=torch.float32)
    lse = torch.empty((B, Hq, S), device=dev, dtype=torch.float32)
    first = True

    for step in range(W):
        # Kick off the next hop's K/V transfer first (overlaps with this step's compute).
        if step + 1 != W:
            comm.send_recv_packed(kv_send, kv_bufs[step & 1])
        # Causal ring: rank r only processes steps 0..r (step 0 is its diagonal block,
        # the only one that carries the causal mask).
        if (not causal) or step <= rank:
            block_causal = causal and step == 0
            ns = _NS_CAUSAL if block_causal else _NS_FULL
            # Emit ns normalized fp32 partials + fp32 LSE for the current K/V block.
            flash_attn_forward_ring(
                q, k_cur, v_cur, part_o[:ns], part_lse[:ns],
                softmax_scale=softmax_scale, causal=block_causal, num_splits=ns)
            for j in range(ns):
                _merge(part_o[j], part_lse[j], acc, lse, is_first=first)  # merge into acc/lse
                first = False
        # Wait for this hop's K/V, switch to the other buffer half, and use it as the
        # next hop's send source.
        if step + 1 != W:
            comm.wait()
            kv_send = kv_bufs[step & 1]
            k_cur, v_cur = kv_bufs[step & 1][0], kv_bufs[step & 1][1]

    out = acc.permute(0, 2, 1, 3).contiguous().to(q.dtype)  # (B,Hq,S,D)->(B,S,Hq,D), one fp32->bf16
    return out, lse  # lse: (B,Hq,S) fp32


# ============================================================================
# Backward: phased FA3 backward. dQ stays in a local persistent fp32 dq_accum
# (never communicated); dK/dV are reduced in fp32 along a "second ring" back to
# their owner rank. phase1 once -> phase2 per hop -> phase3 once.
# ============================================================================
@torch.no_grad()
def _ring_backward(process_group, dout, q, k, v, out, softmax_lse, softmax_scale, causal):
    dout, q, k, v, out = [x.contiguous() for x in (dout, q, k, v, out)]
    kv_comm = DoubleBufRingComm(process_group)     # rotates K/V
    d_kv_comm = DoubleBufRingComm(process_group)   # rotates the in-flight dK/dV accumulator
    W, rank = kv_comm.world_size, kv_comm.rank
    B, S, Hq, D = q.shape
    dev = q.device

    # Three persistent fp32 buffers: dq_accum (accumulates dQ), dsoftmax_sum (D),
    # softmax_lse_log2. Shapes / kBlockM alignment are handled by ring_bwd_alloc
    # (hd128 uses the 80-block). Note: device must be passed as a keyword.
    dq_accum, dsoftmax, lse_log2 = ring_bwd_alloc(B, S, Hq, D, device=dev)

    dq = torch.empty_like(q)          # written by phase 3 (bf16)
    block_dk = torch.empty_like(k)    # this hop's dK contribution (bf16, h_kv heads)
    block_dv = torch.empty_like(k)

    # phase 1 (once per rank): D + softmax_lse_log2 + clear dq_accum. dO/O/LSE are
    # fixed for this rank across ring steps, so this runs only once.
    flash_attn_backward_ring(
        1, dout, q, k, v, out, softmax_lse,
        dq, block_dk, block_dv, dq_accum, dsoftmax, lse_log2,
        softmax_scale=softmax_scale, causal=causal)

    # K/V double buffer + fp32 dK/dV double buffer (packed [dk;dv], shape (2,)+k.shape).
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
            # phase 2 (per hop): atomicAdd this block's dQ into the persistent dq_accum,
            # and write this block's dK/dV.
            flash_attn_backward_ring(
                2, dout, q, k_cur, v_cur, out, softmax_lse,
                dq, block_dk, block_dv, dq_accum, dsoftmax, lse_log2,
                softmax_scale=softmax_scale, causal=(causal and step == 0))
            # Fold this block's dK/dV into the rotating fp32 accumulator slot
            # (first time is a copy, afterwards an add).
            if first_iter_done:
                d_kv_comm.wait()  # wait for the prev hop's dK/dV slot to arrive before adding
            if not first_iter_done:
                dk_bufs[prev_slot].copy_(block_dk); dv_bufs[prev_slot].copy_(block_dv)
            else:
                dk_bufs[prev_slot].add_(block_dk); dv_bufs[prev_slot].add_(block_dv)
            first_iter_done = True
        elif step != 0:
            d_kv_comm.wait()  # inactive steps still drain the ring to stay in sync
        if step + 1 != W:
            kv_comm.wait()
            kv_send = kv_bufs[step & 1]
            k_cur, v_cur = kv_bufs[step & 1][0], kv_bufs[step & 1][1]
        # Send the current accumulator slot onward (its dK/dV keeps accumulating on
        # other ranks and returns to the owner rank after W hops).
        d_kv_comm.send_recv_packed(dkdv_bufs[prev_slot], dkdv_bufs[step & 1])

    d_kv_comm.wait()
    # phase 3 (once per rank): persistent fp32 dq_accum -> dq (bf16), scaled by
    # softmax_scale once.
    flash_attn_backward_ring(
        3, dout, q, k, v, out, softmax_lse,
        dq, block_dk, block_dv, dq_accum, dsoftmax, lse_log2,
        softmax_scale=softmax_scale, causal=causal)

    final_slot = (W - 1) & 1  # the complete dK/dV lands in this slot after W hops
    return dq.to(q.dtype), dk_bufs[final_slot].to(k.dtype), dv_bufs[final_slot].to(v.dtype)


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
        return dq, dk, dv, None, None, None  # matches forward's 6 inputs; last 3 are non-tensors


def ring_opt(q, k, v, softmax_scale=None, causal=False, group=None):
    """[after] Self-contained FA3 ring attention (fwd_ring + phased flash_attn_backward_ring).
    q/k/v: (B, S_local, H, D) bf16/fp16; returns out (B, S_local, Hq, D); autograd-enabled."""
    return _RingFlashAttnFunc.apply(q, k, v, softmax_scale, causal, group)


# ############################################################################
# Baseline ring (ring_stock): full-block flash_attn_func forward + full
# flash_attn_3.bwd backward -- IDENTICAL to ring_opt except for those two
# primitives (same comm / merge / fp32 dK/dV ring reduction / sharding).
# ############################################################################
# ============================================================================
# Baseline forward: each hop runs a full-block flash_attn_func returning
# (bf16 out, fp32 lse), then merges online. The only difference from ring_opt:
# each block's out is already bf16 (the full forward combined+downcast internally),
# so it must be cast back to fp32 before merging -- that is the per-hop bf16 round-trip.
# ============================================================================
@torch.no_grad()
def _stock_forward(process_group, q, k, v, softmax_scale, causal):
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    comm = DoubleBufRingComm(process_group)
    W, rank = comm.world_size, comm.rank
    B, S, Hq, D = q.shape
    dev = q.device
    kv_bufs = [torch.empty((2,) + k.shape, device=dev, dtype=k.dtype) for _ in range(2)]
    kv_send = torch.empty((2,) + k.shape, device=dev, dtype=k.dtype)
    if W > 1:
        kv_send[0].copy_(k); kv_send[1].copy_(v)
    k_cur, v_cur = k, v
    acc = torch.empty((B, Hq, S, D), device=dev, dtype=torch.float32)  # fp32 accumulator (merge stays fp32)
    lse = torch.empty((B, Hq, S), device=dev, dtype=torch.float32)
    first = True
    for step in range(W):
        if step + 1 != W:
            comm.send_recv_packed(kv_send, kv_bufs[step & 1])
        if (not causal) or step <= rank:
            block_causal = causal and step == 0
            # Full-block forward: out_blk (B,S,Hq,D) bf16; lse_blk (B,Hq,S) fp32.
            out_blk, lse_blk = flash_attn_func(
                q, k_cur, v_cur, softmax_scale=softmax_scale, causal=block_causal,
                return_attn_probs=True)
            o_i = out_blk.permute(0, 2, 1, 3).contiguous().float()  # bf16->fp32 (per-hop round-trip)
            _merge(o_i, lse_blk.contiguous(), acc, lse, is_first=first)
            first = False
        if step + 1 != W:
            comm.wait()
            kv_send = kv_bufs[step & 1]
            k_cur, v_cur = kv_bufs[step & 1][0], kv_bufs[step & 1][1]
    out = acc.permute(0, 2, 1, 3).contiguous().to(q.dtype)
    return out, lse


# ============================================================================
# Baseline backward: each active hop runs a FULL flash_attn_3.bwd (repeating
# preprocess+convert), producing a bf16 block_dq that is accumulated in Python
# fp32 (the per-hop bf16 round-trip for dQ). The fp32 dK/dV ring reduction is
# identical to ring_opt.
# ============================================================================
@torch.no_grad()
def _stock_backward(process_group, dout, q, k, v, out, softmax_lse, softmax_scale, causal):
    dout, q, k, v, out = [x.contiguous() for x in (dout, q, k, v, out)]
    kv_comm = DoubleBufRingComm(process_group)
    d_kv_comm = DoubleBufRingComm(process_group)
    W, rank = kv_comm.world_size, kv_comm.rank
    dev = q.device
    dq_acc = torch.zeros(q.shape, device=dev, dtype=torch.float32)  # Python-side fp32 dQ accumulator
    block_dq = torch.empty_like(q)
    block_dk = torch.empty_like(k)
    block_dv = torch.empty_like(k)
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
            # Full backward: internal preprocess + main + convert all run; block_dq is bf16.
            _bwd(dout, q, k_cur, v_cur, out, softmax_lse,
                 block_dq, block_dk, block_dv,
                 None, None, None, None, None, None,
                 softmax_scale, (causal and step == 0), -1, -1, 0.0, False, 0)
            if not first_iter_done:
                dq_acc.copy_(block_dq)          # bf16 block_dq -> fp32 accumulator (per-hop round-trip)
            else:
                dq_acc.add_(block_dq)
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
    d_kv_comm.wait()
    final_slot = (W - 1) & 1
    return dq_acc.to(q.dtype), dk_bufs[final_slot].to(k.dtype), dv_bufs[final_slot].to(v.dtype)


class _StockRingFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, softmax_scale, causal, group):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** -0.5
        out, lse = _stock_forward(group, q, k, v, softmax_scale, causal)
        ctx.save_for_backward(q, k.contiguous(), v.contiguous(), out, lse.contiguous())
        ctx.softmax_scale, ctx.causal, ctx.group = softmax_scale, causal, group
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = _stock_backward(ctx.group, dout, q, k, v, out, lse, ctx.softmax_scale, ctx.causal)
        return dq, dk, dv, None, None, None


def ring_stock(q, k, v, softmax_scale=None, causal=False, group=None):
    """[before] Baseline FA3 ring attention (full-block flash_attn_func forward + full bwd)."""
    return _StockRingFunc.apply(q, k, v, softmax_scale, causal, group)


# ============================================================================
# Context-parallel sequence sharding + a minimal wrapper.
# ============================================================================
def shard_along_seq(x_global: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    """Split (B, S_global, H, D) along the sequence dim into world_size chunks and
    return this rank's chunk (contiguous)."""
    assert x_global.shape[1] % world_size == 0, "global sequence length must be divisible by world_size"
    return torch.chunk(x_global, world_size, dim=1)[rank].contiguous()


def ring_attention(q_local, k_local, v_local, causal=False, softmax_scale=None, group=None):
    """Run the (optimized) ring attention on (q,k,v) already sharded to this rank."""
    return ring_opt(q_local, k_local, v_local,
                    softmax_scale=softmax_scale, causal=causal, group=group)


# ============================================================================
# Reference: fp32 eager attention ("ground truth") used to gauge the absolute
# accuracy of the bf16/fp16 kernels.
# ============================================================================
def _ref_attention_fp32(q, k, v, causal, scale):
    """q (B,S,Hq,D) fp32, k/v (B,S,Hkv,D) fp32 -> naive fp32 attention, returns out
    (B,S,Hq,D) fp32. GQA: kv heads repeated to q heads; causal uses a lower-triangular
    mask (S_q==S_k, top-left aligned). Differentiable."""
    B, S, Hq, D = q.shape
    Hkv = k.shape[2]
    g = Hq // Hkv
    qf = q.transpose(1, 2)                                # B,Hq,S,D
    kf = k.transpose(1, 2).repeat_interleave(g, dim=1)    # B,Hq,S,D
    vf = v.transpose(1, 2).repeat_interleave(g, dim=1)
    scores = torch.matmul(qf, kf.transpose(-1, -2)) * scale   # B,Hq,S,S
    if causal:
        mask = torch.ones(S, S, device=q.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
    p = torch.softmax(scores, dim=-1)
    o = torch.matmul(p, vf)                                # B,Hq,S,D
    return o.transpose(1, 2).contiguous()                  # B,S,Hq,D


# ============================================================================
# Accuracy test: absolute error of the optimized ring (bf16/fp16) vs the fp32
# eager truth, compared against the error of a single-GPU full-sequence FA3 flash
# (same dtype) vs the same truth. Criterion (flash-attention's testing style):
#   ring_max <= flash_max * 2.5 + 2e-3
# i.e. the ring introduces no error beyond the reference kernel's own bf16 rounding.
# Swept over (dtype, D, head layout, causal).
# ============================================================================
def precision_test(device, rank, world_size, log):
    log("=" * 104)
    log("[accuracy] optimized ring vs single-GPU flash, each vs fp32 eager truth (max/mean abs error)")
    log("criterion: ring_max <= flash_max * 2.5 + 2e-3 (no extra error beyond the reference bf16 rounding)")
    log("=" * 104)
    log(f"{'dtype':>7} {'d':>4} {'Hq':>3} {'Hkv':>4} {'causal':>7} | "
        f"{'ring_max':>10} {'flash_max':>10} {'ring/flash':>10} | "
        f"{'ring_mean':>10} {'flash_mean':>10}  status")

    B = 1
    # A moderate global sequence length (fp32 eager is affordable and the error is
    # already representative).
    S_global = max(world_size, (2048 // world_size) * world_size)
    all_ok = True
    for dtype in (torch.bfloat16, torch.float16):
        for D in (64, 128):
            for (Hq, Hkv) in ((8, 8), (16, 2)):
                for causal in (False, True):
                    scale = D ** -0.5
                    torch.manual_seed(2024)  # every rank builds the same global tensors
                    qg = torch.randn(B, S_global, Hq, D, device=device, dtype=dtype)
                    kg = torch.randn(B, S_global, Hkv, D, device=device, dtype=dtype)
                    vg = torch.randn(B, S_global, Hkv, D, device=device, dtype=dtype)
                    dog = torch.randn_like(qg)

                    # (a) fp32 truth: eager attention fwd+bwd
                    qf = qg.float().detach().requires_grad_(True)
                    kf = kg.float().detach().requires_grad_(True)
                    vf = vg.float().detach().requires_grad_(True)
                    o_t = _ref_attention_fp32(qf, kf, vf, causal, scale)
                    o_t.backward(dog.float())
                    truth = {"out": o_t.detach(), "dq": qf.grad, "dk": kf.grad, "dv": vf.grad}

                    # (b) optimized ring (this rank's shard) fwd+bwd
                    ql = shard_along_seq(qg, rank, world_size).detach().clone().requires_grad_(True)
                    kl = shard_along_seq(kg, rank, world_size).detach().clone().requires_grad_(True)
                    vl = shard_along_seq(vg, rank, world_size).detach().clone().requires_grad_(True)
                    o_r = ring_opt(ql, kl, vl, causal=causal, softmax_scale=scale)
                    o_r.backward(shard_along_seq(dog, rank, world_size))
                    ring = {"out": o_r.detach(), "dq": ql.grad, "dk": kl.grad, "dv": vl.grad}

                    # (c) single-GPU flash baseline (same dtype, whole sequence) fwd+bwd
                    qF = qg.detach().clone().requires_grad_(True)
                    kF = kg.detach().clone().requires_grad_(True)
                    vF = vg.detach().clone().requires_grad_(True)
                    o_f = flash_attn_func(qF, kF, vF, softmax_scale=scale, causal=causal)
                    o_f.backward(dog)
                    flash = {"out": o_f.detach(), "dq": qF.grad, "dk": kF.grad, "dv": vF.grad}

                    # Error: slice ring/flash to this rank's shard and compare with the
                    # matching shard of the fp32 truth.
                    ring_max = flash_max = ring_mean = flash_mean = 0.0
                    for key in ("out", "dq", "dk", "dv"):
                        t = shard_along_seq(truth[key], rank, world_size).float()
                        r = ring[key].float()
                        f = shard_along_seq(flash[key], rank, world_size).float()
                        ring_max = max(ring_max, (r - t).abs().max().item())
                        flash_max = max(flash_max, (f - t).abs().max().item())
                        ring_mean = max(ring_mean, (r - t).abs().mean().item())
                        flash_mean = max(flash_mean, (f - t).abs().mean().item())

                    ratio = ring_max / max(flash_max, 1e-9)
                    ok_local = ring_max <= flash_max * 2.5 + 2e-3
                    ok_t = torch.tensor([int(ok_local)], device=device, dtype=torch.int32)
                    dist.all_reduce(ok_t, op=dist.ReduceOp.MIN)  # pass only if every rank passes
                    ok = bool(ok_t.item()); all_ok = all_ok and ok
                    log(f"{str(dtype).split('.')[-1]:>7} {D:>4} {Hq:>3} {Hkv:>4} {str(causal):>7} | "
                        f"{ring_max:>10.2e} {flash_max:>10.2e} {ratio:>9.2f}x | "
                        f"{ring_mean:>10.2e} {flash_mean:>10.2e}  {'PASS' if ok else 'FAIL'}")
    log()
    log(f"accuracy (S_global={S_global}): {'ALL PASSED' if all_ok else 'SOME FAILED'}")
    log()
    return all_ok


# ============================================================================
# Before/after accuracy: optimized vs baseline ring, each vs the fp32 eager truth
# (side by side). Watch out (forward primitive) and dq (backward primitive) --
# the optimized version drops the per-hop bf16 round-trip, so it is usually better.
# ============================================================================
def precision_compare(device, rank, world_size, log):
    log("=" * 118)
    log("[before/after accuracy] optimized (fwd_ring + backward_ring) vs baseline "
        "(flash_attn_func + full bwd), each vs fp32 truth (max abs error)")
    log("=" * 118)
    log(f"{'dtype':>7} {'d':>4} {'Hq':>3} {'Hkv':>4} {'causal':>7} | "
        f"{'out opt':>9} {'out old':>9} | {'dq opt':>9} {'dq old':>9} | "
        f"{'dk opt':>9} {'dk old':>9} | {'dv opt':>9} {'dv old':>9}")
    B = 1
    S_global = max(world_size, (2048 // world_size) * world_size)
    for dtype in (torch.bfloat16, torch.float16):
        for D in (64, 128):
            for (Hq, Hkv) in ((8, 8), (16, 2)):
                for causal in (False, True):
                    scale = D ** -0.5
                    torch.manual_seed(2024)
                    qg = torch.randn(B, S_global, Hq, D, device=device, dtype=dtype)
                    kg = torch.randn(B, S_global, Hkv, D, device=device, dtype=dtype)
                    vg = torch.randn(B, S_global, Hkv, D, device=device, dtype=dtype)
                    dog = torch.randn_like(qg)

                    # fp32 truth
                    qf = qg.float().detach().requires_grad_(True)
                    kf = kg.float().detach().requires_grad_(True)
                    vf = vg.float().detach().requires_grad_(True)
                    o_t = _ref_attention_fp32(qf, kf, vf, causal, scale)
                    o_t.backward(dog.float())
                    truth = {"out": o_t.detach(), "dq": qf.grad, "dk": kf.grad, "dv": vf.grad}

                    def run(ring_fn):
                        ql = shard_along_seq(qg, rank, world_size).detach().clone().requires_grad_(True)
                        kl = shard_along_seq(kg, rank, world_size).detach().clone().requires_grad_(True)
                        vl = shard_along_seq(vg, rank, world_size).detach().clone().requires_grad_(True)
                        o = ring_fn(ql, kl, vl, causal=causal, softmax_scale=scale)
                        o.backward(shard_along_seq(dog, rank, world_size))
                        return {"out": o.detach(), "dq": ql.grad, "dk": kl.grad, "dv": vl.grad}

                    opt = run(ring_opt)
                    old = run(ring_stock)

                    err = {}  # err[key] = (opt_maxerr, old_maxerr), each already all_reduce MAX
                    for key in ("out", "dq", "dk", "dv"):
                        t = shard_along_seq(truth[key], rank, world_size).float()
                        eo = (opt[key].float() - t).abs().max().item()
                        es = (old[key].float() - t).abs().max().item()
                        pair = torch.tensor([eo, es], device=device, dtype=torch.float64)
                        dist.all_reduce(pair, op=dist.ReduceOp.MAX)
                        err[key] = (pair[0].item(), pair[1].item())

                    log(f"{str(dtype).split('.')[-1]:>7} {D:>4} {Hq:>3} {Hkv:>4} {str(causal):>7} | "
                        f"{err['out'][0]:>9.2e} {err['out'][1]:>9.2e} | "
                        f"{err['dq'][0]:>9.2e} {err['dq'][1]:>9.2e} | "
                        f"{err['dk'][0]:>9.2e} {err['dk'][1]:>9.2e} | "
                        f"{err['dv'][0]:>9.2e} {err['dv'][1]:>9.2e}")
    log()


def _parse_int_list(env_name, default):
    raw = os.environ.get(env_name, default)
    return [int(x) for x in raw.replace(" ", "").split(",") if x]


# ============================================================================
# Before/after speed grid (default speed-optimization test): optimized vs baseline
# ring, forward and forward+backward, across a (head_dim x per-rank seqlen) grid.
# Each cell reports the median of RING_REPS timings; every timing is an all_reduce
# MAX across ranks (true wall-clock straggler). old/opt > 1 means the optimized
# ring is faster.
#
# Expectation: at small/medium S_local the optimized ring wins clearly (launch /
# overhead bound: no per-hop bf16 round-trip, no per-hop preprocess+convert). At
# large S_local it converges toward parity (compute bound: the main kernel
# dominates and there is little overhead left to remove), but does not invert.
# ============================================================================
def speed_grid(device, rank, world_size, log):
    Hq = int(os.environ.get("RING_HQ", 16))
    Hkv = int(os.environ.get("RING_HKV", 2))
    d_list = _parse_int_list("RING_D_LIST", "64,128,256")
    s_list = _parse_int_list("RING_S_LIST", "512,1024,2048,4096,8192")
    reps = int(os.environ.get("RING_REPS", 3))
    B = 1
    dtype = torch.bfloat16

    def bench(fn, warmup=8, iters=30):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record(); torch.cuda.synchronize()
        t = torch.tensor([s.elapsed_time(e) / iters], device=device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return t.item()

    def median(xs):
        xs = sorted(xs); n = len(xs)
        return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])

    log("=" * 104)
    log(f"[before/after speed grid] causal=True, ws={world_size}, Hq={Hq} Hkv={Hkv}, bf16 | "
        f"median of {reps} reps, all_reduce MAX")
    log("old/opt > 1 => optimized ring is faster")
    log("=" * 104)
    log(f"{'D':>4} {'S_local':>8} {'S_glob':>8} | {'fwd opt':>9} {'fwd old':>9} {'old/opt':>8} | "
        f"{'fbw opt':>9} {'fbw old':>9} {'old/opt':>8}")

    for D in d_list:
        scale = D ** -0.5
        for S_local in s_list:
            S_global = S_local * world_size
            try:
                torch.manual_seed(7)
                qg = torch.randn(B, S_global, Hq, D, device=device, dtype=dtype)
                kg = torch.randn(B, S_global, Hkv, D, device=device, dtype=dtype)
                vg = torch.randn(B, S_global, Hkv, D, device=device, dtype=dtype)
                q = shard_along_seq(qg, rank, world_size)
                k = shard_along_seq(kg, rank, world_size)
                v = shard_along_seq(vg, rank, world_size)
                dout = shard_along_seq(torch.randn_like(qg), rank, world_size)

                def mk_fwd(fn):
                    def r():
                        with torch.no_grad():
                            fn(q, k, v, causal=True, softmax_scale=scale)
                    return r

                def mk_fbw(fn):
                    def r():
                        qa = q.detach().clone().requires_grad_(True)
                        ka = k.detach().clone().requires_grad_(True)
                        va = v.detach().clone().requires_grad_(True)
                        o = fn(qa, ka, va, causal=True, softmax_scale=scale)
                        o.backward(dout)
                    return r

                fo = median([bench(mk_fwd(ring_opt)) for _ in range(reps)])
                fs = median([bench(mk_fwd(ring_stock)) for _ in range(reps)])
                bo = median([bench(mk_fbw(ring_opt)) for _ in range(reps)])
                bs = median([bench(mk_fbw(ring_stock)) for _ in range(reps)])
                log(f"{D:>4} {S_local:>8} {S_global:>8} | {fo:>9.3f} {fs:>9.3f} {fs / fo:>7.2f}x | "
                    f"{bo:>9.3f} {bs:>9.3f} {bs / bo:>7.2f}x")
            except Exception as ex:  # keep the grid going if one cell OOMs / is unsupported
                torch.cuda.synchronize()
                log(f"{D:>4} {S_local:>8} {S_global:>8} | FAILED: {type(ex).__name__}: {str(ex)[:80]}")
                torch.cuda.empty_cache()
    log()


def main():
    # init NCCL + bind the local GPU (torchrun injects RANK/WORLD_SIZE/LOCAL_RANK)
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

        log(f"world_size={world_size}  device_name={torch.cuda.get_device_name(local_rank)}")
        log(f"hopper dir: {_HOPPER_DIR}")
        log()

        # 1) accuracy of the optimized ring vs the fp32 truth (correctness gate)
        if os.environ.get("RING_SKIP_PRECISION", "0") != "1":
            precision_test(device, rank, world_size, log)

        # 2) before/after accuracy (optimized vs baseline, side by side)
        if os.environ.get("RING_SKIP_PRECISION_DIFF", "0") != "1":
            precision_compare(device, rank, world_size, log)

        # 3) before/after speed grid (the default speed-optimization test)
        if os.environ.get("RING_SKIP_SPEED", "0") != "1":
            speed_grid(device, rank, world_size, log)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
