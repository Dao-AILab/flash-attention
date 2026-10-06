"""
================================================================================
 Ring Flash Attention (FA3) benchmark: FA3 before/after (dense + varlen)
================================================================================

Measures what the optimized FA3 ring primitives buy, for BOTH dense and varlen
(context-parallel) attention, by comparing an "optimized" ring against a "stock"
ring that is IDENTICAL except for the FA3 part (the forward/backward primitive).

The optimized ring drivers are NOT re-implemented here -- they are imported from
`tests/ring_flash_attn_fa3.py` (the single source of truth):
    _RingFlashAttnFunc / ring_flash_attn_func         # dense, optimized
    _RingVarlenFunc    / ring_flash_attn_varlen_func   # varlen, optimized
plus the shared ring machinery (DoubleBufRingComm, the fp32 online-merge kernels,
sequence sharding). This file only adds the "stock" (before-optimization) rings and
the before/after comparisons.

--------------------------------------------------------------------------------
 What differs between opt and stock (the FA3 part only)
--------------------------------------------------------------------------------
Dense (ring_opt vs ring_stock):
  * forward:  opt = flash_attn_forward_ring (fp32 partial, no bf16 round-trip)
              stock = full flash_attn_func per block (bf16 out -> cast to fp32 to merge)
  * backward: opt = phased flash_attn_backward_ring (persistent fp32 dq_accum;
              preprocess/convert once per rank)
              stock = full flash_attn_3.bwd per hop (repeats preprocess+convert; dQ
              round-trips to bf16 each hop and is accumulated in Python fp32)

Varlen (ring_varlen_opt vs ring_varlen_stock):
  * forward:  IDENTICAL -- both use per-hop flash_attn_varlen_func(out+lse) + the fused
              fp32 merge (there is no varlen fwd_ring; the store-all path is unusable
              under the dynamic varlen split scheduler). So varlen has no forward
              before/after -- the only FA3 difference is the backward.
  * backward: opt = phased flash_attn_backward_ring (varlen) ; stock = full
              flash_attn_3.bwd (varlen) per hop.

Everything else (ring communication, fp32 dK/dV ring reduction, sharding) is shared,
so any difference reflects the FA3 primitives alone. Varlen CP splits each global
sequence into W equal chunks along its length (every sequence length divisible by W).

--------------------------------------------------------------------------------
 How to run
--------------------------------------------------------------------------------
  torchrun --nproc_per_node=8 --standalone benchmark_flash_ring_attention_fa3.py
  CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --standalone benchmark_flash_ring_attention_fa3.py

  Env overrides:
    RING_HQ / RING_HKV        head counts for the speed stages (default 16 / 2)
    RING_D_LIST               head dims for the dense speed grid (default "64,128,256")
    RING_S_LIST               per-rank seqlens for the dense speed grid ("512,1024,2048,4096,8192")
    RING_REPS                 median-of-N repeats per speed cell (default 3)
    RING_SKIP_PRECISION       "1" skip the dense accuracy gate
    RING_SKIP_PRECISION_DIFF  "1" skip the dense before/after accuracy table
    RING_SKIP_SPEED           "1" skip the dense before/after speed grid
    RING_SKIP_VARLEN_DIFF     "1" skip the varlen before/after accuracy table
    RING_SKIP_VARLEN_SPEED    "1" skip the varlen before/after speed table
"""

import os
import sys

import torch
import torch.distributed as dist

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HOPPER_DIR = os.path.join(_REPO_ROOT, "hopper")   # exposes the built FA3 ring primitives
_TESTS_DIR = os.path.join(_REPO_ROOT, "tests")     # exposes the optimized ring drivers
for _p in (_HOPPER_DIR, _TESTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Optimized ring drivers + shared ring machinery (single source of truth). We do NOT
# re-implement the optimized ring here; the "stock" rings below reuse the same comm /
# merge / sharding and differ only in the FA3 forward/backward calls.
from ring_flash_attn_fa3 import (  # tests/ring_flash_attn_fa3.py
    DoubleBufRingComm, _merge, _merge_varlen,
    shard_along_seq, shard_varlen_along_seq,
    _RingFlashAttnFunc, ring_flash_attn_func as ring_opt,               # dense, optimized
    _RingVarlenFunc, ring_flash_attn_varlen_func as ring_varlen_opt,    # varlen, optimized
    _ring_forward_varlen,                                               # varlen fwd (shared by the stock varlen)
)
from flash_attn_interface import flash_attn_func, flash_attn_varlen_func  # stock full-block FA3

# Stock (before-optimization) FA3 backward op -- the full backward, run per hop.
_bwd = torch.ops.flash_attn_3.bwd


# ############################################################################
# Dense stock ring (ring_stock): full flash_attn_func forward + full flash_attn_3.bwd
# per hop. IDENTICAL to the imported optimized ring_opt except for those two FA3 calls
# (same DoubleBufRingComm / _merge / fp32 dK/dV reduction / shard_along_seq).
# ############################################################################
@torch.no_grad()
def _stock_forward(process_group, q, k, v, softmax_scale, causal):
    """Each hop: full flash_attn_func -> (bf16 out, fp32 lse); cast out to fp32 (the
    per-hop bf16 round-trip) and merge online. Optimized ring_opt instead uses
    flash_attn_forward_ring, whose partial is already fp32."""
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
    acc = torch.empty((B, Hq, S, D), device=dev, dtype=torch.float32)  # fp32 merge accumulator
    lse = torch.empty((B, Hq, S), device=dev, dtype=torch.float32)
    first = True
    for step in range(W):
        if step + 1 != W:
            comm.send_recv_packed(kv_send, kv_bufs[step & 1])
        if (not causal) or step <= rank:
            block_causal = causal and step == 0
            out_blk, lse_blk = flash_attn_func(
                q, k_cur, v_cur, softmax_scale=softmax_scale, causal=block_causal,
                return_attn_probs=True)                                    # out (B,S,Hq,D) bf16
            o_i = out_blk.permute(0, 2, 1, 3).contiguous().float()         # bf16->fp32 (per-hop round-trip)
            _merge(o_i, lse_blk.contiguous(), acc, lse, is_first=first)
            first = False
        if step + 1 != W:
            comm.wait()
            kv_send = kv_bufs[step & 1]
            k_cur, v_cur = kv_bufs[step & 1][0], kv_bufs[step & 1][1]
    out = acc.permute(0, 2, 1, 3).contiguous().to(q.dtype)
    return out, lse


@torch.no_grad()
def _stock_backward(process_group, dout, q, k, v, out, softmax_lse, softmax_scale, causal):
    """Each active hop: a FULL flash_attn_3.bwd (repeats preprocess+convert), producing a
    bf16 block_dq accumulated in Python fp32 (the per-hop dQ bf16 round-trip). Optimized
    ring_opt instead uses the phased flash_attn_backward_ring with a persistent fp32
    dq_accum. The fp32 dK/dV ring reduction is identical to ring_opt."""
    dout, q, k, v, out = [x.contiguous() for x in (dout, q, k, v, out)]
    kv_comm = DoubleBufRingComm(process_group)
    d_kv_comm = DoubleBufRingComm(process_group)
    W, rank = kv_comm.world_size, kv_comm.rank
    dev = q.device
    dq_acc = torch.zeros(q.shape, device=dev, dtype=torch.float32)  # Python fp32 dQ accumulator
    block_dq = torch.empty_like(q); block_dk = torch.empty_like(k); block_dv = torch.empty_like(k)
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
            _bwd(dout, q, k_cur, v_cur, out, softmax_lse,
                 block_dq, block_dk, block_dv,
                 None, None, None, None, None, None,
                 softmax_scale, (causal and step == 0), -1, -1, 0.0, False, 0)
            if not first_iter_done:
                dq_acc.copy_(block_dq)
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
    """[before] Dense baseline ring (full flash_attn_func forward + full bwd per hop)."""
    return _StockRingFunc.apply(q, k, v, softmax_scale, causal, group)


# ############################################################################
# Varlen stock ring (ring_varlen_stock): mimics _StockRingFunc for varlen. Forward is
# IDENTICAL to the optimized _RingVarlenFunc (imported _ring_forward_varlen) -- varlen
# has no forward FA3 alternative -- so the ONLY FA3 difference is the backward: a full
# flash_attn_3.bwd (varlen) per hop vs the optimized phased bwd_ring.
# ############################################################################
@torch.no_grad()
def _ring_backward_varlen_stock(process_group, dout, q, k, v, out, softmax_lse, cu, max_s, softmax_scale, causal):
    """Varlen analog of _stock_backward: per active hop run a FULL varlen flash_attn_3.bwd
    (repeats preprocess+convert; dQ->bf16 per hop, Python fp32 accum); fp32 dK/dV ring
    reduction identical to the optimized varlen ring. q/k/v/out: (total,H,D); dout:
    (total,Hq,D); softmax_lse: (Hq,total)."""
    dout, q, k, v, out = [x.contiguous() for x in (dout, q, k, v, out)]
    kv_comm = DoubleBufRingComm(process_group)
    d_kv_comm = DoubleBufRingComm(process_group)
    W, rank = kv_comm.world_size, kv_comm.rank
    dev = q.device
    dq_acc = torch.zeros(q.shape, device=dev, dtype=torch.float32)  # Python fp32 dQ accumulator
    block_dq = torch.empty_like(q); block_dk = torch.empty_like(k); block_dv = torch.empty_like(k)
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
            # full varlen backward for the current K/V chunk
            _bwd(dout, q, k_cur, v_cur, out, softmax_lse,
                 block_dq, block_dk, block_dv,
                 cu, cu, None, None, max_s, max_s,
                 softmax_scale, (causal and step == 0), -1, -1, 0.0, False, 0)
            if not first_iter_done:
                dq_acc.copy_(block_dq)
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


class _StockRingVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, cu, max_s, softmax_scale, causal, group):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** -0.5
        out, lse = _ring_forward_varlen(group, q, k, v, cu, max_s, softmax_scale, causal)  # SAME fwd as opt
        ctx.save_for_backward(q, k.contiguous(), v.contiguous(), out, lse.contiguous(), cu)
        ctx.softmax_scale, ctx.causal, ctx.group, ctx.max_s = softmax_scale, causal, group, max_s
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, lse, cu = ctx.saved_tensors
        dq, dk, dv = _ring_backward_varlen_stock(ctx.group, dout, q, k, v, out, lse, cu, ctx.max_s,
                                                 ctx.softmax_scale, ctx.causal)
        return dq, dk, dv, None, None, None, None, None


def ring_varlen_stock(q, k, v, cu_seqlens, max_seqlen, softmax_scale=None, causal=False, group=None):
    """[before] Varlen baseline ring (Option A forward + full varlen bwd per hop)."""
    return _StockRingVarlenFunc.apply(q, k, v, cu_seqlens, max_seqlen, softmax_scale, causal, group)


# ============================================================================
# fp32 eager references ("ground truth") for accuracy.
# ============================================================================
def _ref_attention_fp32(q, k, v, causal, scale):
    """Dense fp32 eager attention. q (B,S,Hq,D)/k,v (B,S,Hkv,D) fp32 -> out (B,S,Hq,D)."""
    B, S, Hq, D = q.shape
    Hkv = k.shape[2]; g = Hq // Hkv
    qf = q.transpose(1, 2)
    kf = k.transpose(1, 2).repeat_interleave(g, dim=1)
    vf = v.transpose(1, 2).repeat_interleave(g, dim=1)
    scores = torch.matmul(qf, kf.transpose(-1, -2)) * scale
    if causal:
        mask = torch.ones(S, S, device=q.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
    p = torch.softmax(scores, dim=-1)
    return torch.matmul(p, vf).transpose(1, 2).contiguous()


def _ref_attention_fp32_varlen(q, k, v, cu, causal, scale):
    """Per-sequence fp32 eager varlen attention. q (total,Hq,D)/k,v (total,Hkv,D) fp32
    (may require grad) -> out (total,Hq,D). GQA repeats kv heads; causal is lower-tri."""
    Hq = q.shape[1]; Hkv = k.shape[1]; g = Hq // Hkv
    seqlens = (cu[1:] - cu[:-1]).tolist()
    outs = []
    for i, s in enumerate(seqlens):
        lo = int(cu[i]); hi = lo + s
        qi = q[lo:hi].transpose(0, 1)                              # (Hq, s, D)
        ki = k[lo:hi].transpose(0, 1).repeat_interleave(g, 0)      # (Hq, s, D)
        vi = v[lo:hi].transpose(0, 1).repeat_interleave(g, 0)
        sc = torch.matmul(qi, ki.transpose(-1, -2)) * scale        # (Hq, s, s)
        if causal:
            m = torch.ones(s, s, device=q.device, dtype=torch.bool).tril()
            sc = sc.masked_fill(~m, float("-inf"))
        outs.append(torch.matmul(torch.softmax(sc, dim=-1), vi).transpose(0, 1))  # (s, Hq, D)
    return torch.cat(outs, 0)                                      # (total, Hq, D)


def _make_global_varlen(nseq, W, hq, hkv, d, dtype, device, seed):
    """Packed global varlen batch whose every sequence length is a multiple of 8*W (so
    each of the W ring chunks is a multiple of 8). Returns qg,kg,vg,dog (total,H,D), cu,
    max_s, total -- identical on every rank (seeded)."""
    unit = 8 * W
    g = torch.Generator(device="cpu").manual_seed(seed)
    seqlens = [((x // unit) + 1) * unit for x in torch.randint(unit, unit * 4, (nseq,), generator=g).tolist()]
    total = sum(seqlens)
    cu = torch.tensor([0] + list(torch.tensor(seqlens).cumsum(0)), dtype=torch.int32, device=device)
    torch.manual_seed(seed)
    qg = torch.randn(total, hq, d, device=device, dtype=dtype)
    kg = torch.randn(total, hkv, d, device=device, dtype=dtype)
    vg = torch.randn(total, hkv, d, device=device, dtype=dtype)
    dog = torch.randn_like(qg)
    return qg, kg, vg, dog, cu, max(seqlens), total


def _parse_int_list(env_name, default):
    raw = os.environ.get(env_name, default)
    return [int(x) for x in raw.replace(" ", "").split(",") if x]


def _bench(fn, device, warmup=8, iters=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record(); torch.cuda.synchronize()
    t = torch.tensor([s.elapsed_time(e) / iters], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)   # true wall-clock straggler
    return t.item()


def _median(xs):
    xs = sorted(xs); n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


# ============================================================================
# 1) Dense accuracy gate: optimized ring vs single-GPU full-sequence flash, each vs the
#    fp32 eager truth. Criterion: ring_max <= flash_max*2.5 + 2e-3.
# ============================================================================
def precision_test(device, rank, world_size, log):
    log("=" * 104)
    log("[dense accuracy] optimized ring vs single-GPU flash, each vs fp32 eager truth")
    log("criterion: ring_max <= flash_max * 2.5 + 2e-3")
    log("=" * 104)
    log(f"{'dtype':>7} {'d':>4} {'Hq':>3} {'Hkv':>4} {'causal':>7} | "
        f"{'ring_max':>10} {'flash_max':>10} {'ring/flash':>10}  status")
    B = 1
    S_global = max(world_size, (2048 // world_size) * world_size)
    all_ok = True
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
                    qf = qg.float().detach().requires_grad_(True)
                    kf = kg.float().detach().requires_grad_(True)
                    vf = vg.float().detach().requires_grad_(True)
                    o_t = _ref_attention_fp32(qf, kf, vf, causal, scale)
                    o_t.backward(dog.float())
                    truth = {"out": o_t.detach(), "dq": qf.grad, "dk": kf.grad, "dv": vf.grad}
                    ql = shard_along_seq(qg, rank, world_size).detach().clone().requires_grad_(True)
                    kl = shard_along_seq(kg, rank, world_size).detach().clone().requires_grad_(True)
                    vl = shard_along_seq(vg, rank, world_size).detach().clone().requires_grad_(True)
                    o_r = ring_opt(ql, kl, vl, causal=causal, softmax_scale=scale)
                    o_r.backward(shard_along_seq(dog, rank, world_size))
                    ring = {"out": o_r.detach(), "dq": ql.grad, "dk": kl.grad, "dv": vl.grad}
                    qF = qg.detach().clone().requires_grad_(True)
                    kF = kg.detach().clone().requires_grad_(True)
                    vF = vg.detach().clone().requires_grad_(True)
                    o_f = flash_attn_func(qF, kF, vF, softmax_scale=scale, causal=causal)
                    o_f.backward(dog)
                    flash = {"out": o_f.detach(), "dq": qF.grad, "dk": kF.grad, "dv": vF.grad}
                    ring_max = flash_max = 0.0
                    for key in ("out", "dq", "dk", "dv"):
                        t = shard_along_seq(truth[key], rank, world_size).float()
                        ring_max = max(ring_max, (ring[key].float() - t).abs().max().item())
                        flash_max = max(flash_max, (shard_along_seq(flash[key], rank, world_size).float() - t).abs().max().item())
                    ok_local = ring_max <= flash_max * 2.5 + 2e-3
                    ok_t = torch.tensor([int(ok_local)], device=device, dtype=torch.int32)
                    dist.all_reduce(ok_t, op=dist.ReduceOp.MIN)
                    ok = bool(ok_t.item()); all_ok = all_ok and ok
                    log(f"{str(dtype).split('.')[-1]:>7} {D:>4} {Hq:>3} {Hkv:>4} {str(causal):>7} | "
                        f"{ring_max:>10.2e} {flash_max:>10.2e} {ring_max / max(flash_max, 1e-9):>9.2f}x  "
                        f"{'PASS' if ok else 'FAIL'}")
    log()
    log(f"dense accuracy: {'ALL PASSED' if all_ok else 'SOME FAILED'}")
    log()
    return all_ok


# ============================================================================
# 2) Dense before/after accuracy: opt vs stock, each vs the fp32 eager truth.
# ============================================================================
def precision_compare(device, rank, world_size, log):
    log("=" * 118)
    log("[dense before/after accuracy] opt (fwd_ring + bwd_ring) vs stock (flash_attn_func + full bwd), vs fp32 truth")
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

                    opt = run(ring_opt); old = run(ring_stock)
                    e = {}
                    for key in ("out", "dq", "dk", "dv"):
                        t = shard_along_seq(truth[key], rank, world_size).float()
                        pair = torch.tensor([(opt[key].float() - t).abs().max().item(),
                                             (old[key].float() - t).abs().max().item()],
                                            device=device, dtype=torch.float64)
                        dist.all_reduce(pair, op=dist.ReduceOp.MAX)
                        e[key] = (pair[0].item(), pair[1].item())
                    log(f"{str(dtype).split('.')[-1]:>7} {D:>4} {Hq:>3} {Hkv:>4} {str(causal):>7} | "
                        f"{e['out'][0]:>9.2e} {e['out'][1]:>9.2e} | {e['dq'][0]:>9.2e} {e['dq'][1]:>9.2e} | "
                        f"{e['dk'][0]:>9.2e} {e['dk'][1]:>9.2e} | {e['dv'][0]:>9.2e} {e['dv'][1]:>9.2e}")
    log()


# ============================================================================
# 3) Dense before/after speed: opt vs stock, fwd and fwd+bwd, (head_dim x per-rank S).
# ============================================================================
def speed_grid(device, rank, world_size, log):
    Hq = int(os.environ.get("RING_HQ", 16))
    Hkv = int(os.environ.get("RING_HKV", 2))
    d_list = _parse_int_list("RING_D_LIST", "64,128,256")
    s_list = _parse_int_list("RING_S_LIST", "512,1024,2048,4096,8192")
    reps = int(os.environ.get("RING_REPS", 3))
    B = 1
    dtype = torch.bfloat16
    log("=" * 120)
    log(f"[dense before/after speed] causal=True, ws={world_size}, Hq={Hq} Hkv={Hkv}, bf16 | median of {reps}, all_reduce MAX")
    log("old/opt > 1 => optimized ring is faster.  bwd = backward-only (fwd run once, untimed, then timed .backward on the retained graph)")
    log("=" * 120)
    log(f"{'D':>4} {'S_local':>8} {'S_glob':>8} | {'fwd opt':>9} {'fwd old':>9} {'o/o':>6} | "
        f"{'bwd opt':>9} {'bwd old':>9} {'o/o':>6} | {'fbw opt':>9} {'fbw old':>9} {'o/o':>6}")
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
                        fn(qa, ka, va, causal=True, softmax_scale=scale).backward(dout)
                    return r

                def mk_bwd(fn):
                    # forward ONCE (untimed), then time only repeated backward on the
                    # retained graph -> isolates the backward from the shared forward.
                    qa = q.detach().clone().requires_grad_(True)
                    ka = k.detach().clone().requires_grad_(True)
                    va = v.detach().clone().requires_grad_(True)
                    out = fn(qa, ka, va, causal=True, softmax_scale=scale)

                    def r():
                        out.backward(dout, retain_graph=True)
                    return r

                # interleave opt/stock per rep so slow clock/thermal drift cancels
                fwd_o, fwd_s, bwd_o, bwd_s, fbw_o, fbw_s = [], [], [], [], [], []
                for _ in range(reps):
                    fwd_o.append(_bench(mk_fwd(ring_opt), device))
                    fwd_s.append(_bench(mk_fwd(ring_stock), device))
                    bwd_o.append(_bench(mk_bwd(ring_opt), device))
                    bwd_s.append(_bench(mk_bwd(ring_stock), device))
                    fbw_o.append(_bench(mk_fbw(ring_opt), device))
                    fbw_s.append(_bench(mk_fbw(ring_stock), device))
                fo, fs = _median(fwd_o), _median(fwd_s)
                do, ds = _median(bwd_o), _median(bwd_s)
                bo, bs = _median(fbw_o), _median(fbw_s)
                log(f"{D:>4} {S_local:>8} {S_global:>8} | {fo:>9.3f} {fs:>9.3f} {fs / fo:>5.2f}x | "
                    f"{do:>9.3f} {ds:>9.3f} {ds / do:>5.2f}x | {bo:>9.3f} {bs:>9.3f} {bs / bo:>5.2f}x")
            except Exception as ex:
                torch.cuda.synchronize()
                log(f"{D:>4} {S_local:>8} {S_global:>8} | FAILED: {type(ex).__name__}: {str(ex)[:80]}")
                torch.cuda.empty_cache()
    log()


# ============================================================================
# 4) Varlen before/after accuracy: opt vs stock, each vs the fp32 eager varlen truth.
#    (varlen forward is identical opt==stock -> `out` columns match; the FA3 difference
#    is the backward, so watch dq.)
# ============================================================================
def varlen_precision_compare(device, rank, world_size, log):
    log("=" * 118)
    log("[varlen before/after accuracy] opt (bwd_ring) vs stock (full varlen bwd/hop), vs fp32 varlen truth")
    log("(varlen forward is shared -> out opt==old; the FA3 difference is the backward -> watch dq)")
    log("=" * 118)
    log(f"{'dtype':>7} {'d':>4} {'Hq':>3} {'Hkv':>4} {'causal':>7} | "
        f"{'out opt':>9} {'out old':>9} | {'dq opt':>9} {'dq old':>9} | "
        f"{'dk opt':>9} {'dk old':>9} | {'dv opt':>9} {'dv old':>9}")
    nseq = 3
    for dtype in (torch.bfloat16, torch.float16):
        for D in (64, 128):
            for (Hq, Hkv) in ((8, 8), (16, 2)):
                for causal in (False, True):
                    scale = D ** -0.5
                    qg, kg, vg, dog, cu, max_s, total = _make_global_varlen(nseq, world_size, Hq, Hkv, D, dtype, device, seed=2024)
                    qf = qg.float().detach().requires_grad_(True)
                    kf = kg.float().detach().requires_grad_(True)
                    vf = vg.float().detach().requires_grad_(True)
                    o_t = _ref_attention_fp32_varlen(qf, kf, vf, cu, causal, scale)
                    o_t.backward(dog.float())
                    truth = {"out": o_t.detach(), "dq": qf.grad, "dk": kf.grad, "dv": vf.grad}
                    cu_loc = shard_varlen_along_seq(qg, cu, rank, world_size)[1]
                    max_loc = shard_varlen_along_seq(qg, cu, rank, world_size)[2]
                    dol = shard_varlen_along_seq(dog, cu, rank, world_size)[0]

                    def run(ring_fn):
                        ql = shard_varlen_along_seq(qg, cu, rank, world_size)[0].detach().requires_grad_(True)
                        kl = shard_varlen_along_seq(kg, cu, rank, world_size)[0].detach().requires_grad_(True)
                        vl = shard_varlen_along_seq(vg, cu, rank, world_size)[0].detach().requires_grad_(True)
                        o = ring_fn(ql, kl, vl, cu_loc, max_loc, softmax_scale=scale, causal=causal)
                        o.backward(dol)
                        return {"out": o.detach(), "dq": ql.grad, "dk": kl.grad, "dv": vl.grad}

                    opt = run(ring_varlen_opt); old = run(ring_varlen_stock)
                    e = {}
                    for key in ("out", "dq", "dk", "dv"):
                        t = shard_varlen_along_seq(truth[key], cu, rank, world_size)[0].float()
                        pair = torch.tensor([(opt[key].float() - t).abs().max().item(),
                                             (old[key].float() - t).abs().max().item()],
                                            device=device, dtype=torch.float64)
                        dist.all_reduce(pair, op=dist.ReduceOp.MAX)
                        e[key] = (pair[0].item(), pair[1].item())
                    log(f"{str(dtype).split('.')[-1]:>7} {D:>4} {Hq:>3} {Hkv:>4} {str(causal):>7} | "
                        f"{e['out'][0]:>9.2e} {e['out'][1]:>9.2e} | {e['dq'][0]:>9.2e} {e['dq'][1]:>9.2e} | "
                        f"{e['dk'][0]:>9.2e} {e['dk'][1]:>9.2e} | {e['dv'][0]:>9.2e} {e['dv'][1]:>9.2e}")
    log()


# ============================================================================
# 5) Varlen before/after speed: opt vs stock, fwd and fwd+bwd, across per-rank chunk
#    lengths. (fwd is shared -> ~parity; the win is the backward.)
# ============================================================================
def varlen_speed_compare(device, rank, world_size, log):
    Hq = int(os.environ.get("RING_HQ", 16))
    Hkv = int(os.environ.get("RING_HKV", 2))
    reps = int(os.environ.get("RING_REPS", 3))
    dtype = torch.bfloat16
    causal = True
    D = 128
    scale = D ** -0.5
    log("=" * 120)
    log(f"[varlen before/after speed] causal=True, ws={world_size}, Hq={Hq} Hkv={Hkv} D={D}, bf16 | median of {reps}")
    log("old/opt > 1 => optimized varlen ring is faster (fwd is shared -> ~1.0; the FA3 win is the backward -> read the bwd column)")
    log("=" * 120)
    log(f"{'nseq':>5} {'seqlen':>7} {'total':>8} | {'fwd opt':>9} {'fwd old':>9} {'o/o':>6} | "
        f"{'bwd opt':>9} {'bwd old':>9} {'o/o':>6} | {'fbw opt':>9} {'fbw old':>9} {'o/o':>6}")
    for nseq, per_seq in ((4, 512), (4, 2048), (8, 4096)):
        try:
            unit = 8 * world_size
            per_seq = ((per_seq + unit - 1) // unit) * unit    # round up to a multiple of 8*W
            seqlens = [per_seq] * nseq
            total = per_seq * nseq
            cu = torch.tensor([i * per_seq for i in range(nseq + 1)], dtype=torch.int32, device=device)
            torch.manual_seed(7)
            qg = torch.randn(total, Hq, D, device=device, dtype=dtype)
            kg = torch.randn(total, Hkv, D, device=device, dtype=dtype)
            vg = torch.randn(total, Hkv, D, device=device, dtype=dtype)
            q, cu_loc, max_loc = shard_varlen_along_seq(qg, cu, rank, world_size)
            k = shard_varlen_along_seq(kg, cu, rank, world_size)[0]
            v = shard_varlen_along_seq(vg, cu, rank, world_size)[0]
            dout = shard_varlen_along_seq(torch.randn_like(qg), cu, rank, world_size)[0]

            def mk_fwd(fn):
                def r():
                    with torch.no_grad():
                        fn(q, k, v, cu_loc, max_loc, softmax_scale=scale, causal=causal)
                return r

            def mk_fbw(fn):
                def r():
                    qa = q.detach().requires_grad_(True)
                    ka = k.detach().requires_grad_(True)
                    va = v.detach().requires_grad_(True)
                    fn(qa, ka, va, cu_loc, max_loc, softmax_scale=scale, causal=causal).backward(dout)
                return r

            def mk_bwd(fn):
                # forward ONCE (untimed), then time only repeated backward on the retained
                # graph -> isolates the backward from the shared varlen forward.
                qa = q.detach().requires_grad_(True)
                ka = k.detach().requires_grad_(True)
                va = v.detach().requires_grad_(True)
                out = fn(qa, ka, va, cu_loc, max_loc, softmax_scale=scale, causal=causal)

                def r():
                    out.backward(dout, retain_graph=True)
                return r

            fwd_o, fwd_s, bwd_o, bwd_s, fbw_o, fbw_s = [], [], [], [], [], []
            for _ in range(reps):
                fwd_o.append(_bench(mk_fwd(ring_varlen_opt), device))
                fwd_s.append(_bench(mk_fwd(ring_varlen_stock), device))
                bwd_o.append(_bench(mk_bwd(ring_varlen_opt), device))
                bwd_s.append(_bench(mk_bwd(ring_varlen_stock), device))
                fbw_o.append(_bench(mk_fbw(ring_varlen_opt), device))
                fbw_s.append(_bench(mk_fbw(ring_varlen_stock), device))
            fo, fs = _median(fwd_o), _median(fwd_s)
            do, ds = _median(bwd_o), _median(bwd_s)
            bo, bs = _median(fbw_o), _median(fbw_s)
            log(f"{nseq:>5} {per_seq:>7} {total:>8} | {fo:>9.3f} {fs:>9.3f} {fs / fo:>5.2f}x | "
                f"{do:>9.3f} {ds:>9.3f} {ds / do:>5.2f}x | {bo:>9.3f} {bs:>9.3f} {bs / bo:>5.2f}x")
        except Exception as ex:
            torch.cuda.synchronize()
            log(f"{nseq:>5} {per_seq:>7} | FAILED: {type(ex).__name__}: {str(ex)[:80]}")
            torch.cuda.empty_cache()
    log()


def main():
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
        log(f"hopper: {_HOPPER_DIR}\ntests:  {_TESTS_DIR}")
        log()

        if os.environ.get("RING_SKIP_PRECISION", "0") != "1":
            precision_test(device, rank, world_size, log)
        if os.environ.get("RING_SKIP_PRECISION_DIFF", "0") != "1":
            precision_compare(device, rank, world_size, log)
        if os.environ.get("RING_SKIP_SPEED", "0") != "1":
            speed_grid(device, rank, world_size, log)
        if os.environ.get("RING_SKIP_VARLEN_DIFF", "0") != "1":
            varlen_precision_compare(device, rank, world_size, log)
        if os.environ.get("RING_SKIP_VARLEN_SPEED", "0") != "1":
            varlen_speed_compare(device, rank, world_size, log)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
