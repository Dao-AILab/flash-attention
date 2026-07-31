#!/usr/bin/env python
"""Quick demo of the 1CTA MLA absorbed forward kernel (flash_fwd_mla_1cta_sm100.py).

Each demo runs one configuration, checks it against a small torch reference, and times
it. Requires an SM100 GPU (B200/B300).

    python agent_space/demo_mla_1cta.py            # run everything
    python agent_space/demo_mla_1cta.py --list     # list the demos
    python agent_space/demo_mla_1cta.py splits     # run one

MLA "absorbed" shape: hdim 64 (the RoPE part, Q@K^T) + hdim_v 512 (the latent part,
Qv@V^T), so S = Q@K^T + Qv@V^T and O = softmax(S)@V.
"""

import argparse
import math
import os
import sys

# Opt in to the 1CTA kernel before the interface reads the env (it is read per call).
os.environ.setdefault("FLASH_ATTENTION_MLA_1CTA", "1")

import torch

from flash_attn.cute.interface import flash_attn_func, flash_attn_varlen_func

HDIM, HDIMV = 64, 512
DEV, DT = "cuda", torch.bfloat16
SCALE = 1.0 / math.sqrt(HDIM + HDIMV)


# ---------------------------------------------------------------- helpers


def ref(q, k, qv, v, causal=False):
    """Dense reference for one batch: (1, s, h, d) tensors in, (s, h, dv) out."""
    rep = qv.shape[2] // v.shape[2]
    kr, vr = k.repeat_interleave(rep, 2), v.repeat_interleave(rep, 2)
    s = torch.einsum("bqhe,bkhe->bhqk", qv.float(), vr.float())
    if q is not None:
        s = s + torch.einsum("bqhd,bkhd->bhqk", q.float(), kr.float())
    s = s * SCALE
    if causal:
        sq, sk = qv.shape[1], v.shape[1]
        col, row = torch.arange(sk, device=DEV), torch.arange(sq, device=DEV)
        s = s.masked_fill((col[None, :] > row[:, None] + (sk - sq))[None, None], -torch.inf)
    p = torch.nan_to_num(torch.softmax(s, -1), nan=0.0)
    return torch.einsum("bhqk,bkhe->bqhe", p, vr.float())[0]


def check(name, got, want, ms=None, extra=""):
    err = (got.float() - want).abs().max().item()
    tol = 2 * (want.to(DT).float() - want).abs().max().item() + 1e-3
    ok = err <= tol and not got.isnan().any()
    t = f"  {ms:7.3f} ms" if ms is not None else ""
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<42s}{t}  max_err={err:.2e}{extra}")
    return ok


def bench(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters


def rand(*shape):
    return torch.randn(*shape, device=DEV, dtype=DT)


# ---------------------------------------------------------------- demos


def demo_dense():
    """Plain decode and causal prefill through flash_attn_func."""
    for label, b, sq, sk, h, causal in [
        ("decode  b2 sq1 sk4096 h64", 2, 1, 4096, 64, False),
        ("prefill b1 sq1024 sk1024 h16 causal", 1, 1024, 1024, 16, True),
    ]:
        q, qv = rand(b, sq, h, HDIM), rand(b, sq, h, HDIMV)
        k, v = rand(b, sk, 1, HDIM), rand(b, sk, 1, HDIMV)
        out, lse = flash_attn_func(q, k, v, qv=qv, causal=causal,
                                   softmax_scale=SCALE, return_lse=True)
        ms = bench(lambda: flash_attn_func(q, k, v, qv=qv, causal=causal,
                                           softmax_scale=SCALE))
        want = ref(q[:1], k[:1], qv[:1], v[:1], causal)
        check(label, out[0], want, ms, extra=f"  lse{tuple(lse.shape)}")


def demo_varlen():
    """Ragged batch: Q and K packed with cu_seqlens (no padding)."""
    lens_q, lens_k = [200, 37, 1, 313], [300, 64, 129, 500]
    h, b = 8, 4
    q = rand(sum(lens_q), h, HDIM)
    qv = rand(sum(lens_q), h, HDIMV)
    k = rand(sum(lens_k), 1, HDIM)
    v = rand(sum(lens_k), 1, HDIMV)
    cu_q = torch.tensor([0, *torch.tensor(lens_q).cumsum(0)], device=DEV, dtype=torch.int32)
    cu_k = torch.tensor([0, *torch.tensor(lens_k).cumsum(0)], device=DEV, dtype=torch.int32)
    call = lambda: flash_attn_varlen_func(  # noqa: E731
        q, k, v, qv, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
        max_seqlen_q=max(lens_q), max_seqlen_k=max(lens_k), softmax_scale=SCALE)
    out, _ = call()
    ms = bench(call)
    ok = True
    for i in range(b):
        qs, ks = int(cu_q[i]), int(cu_k[i])
        want = ref(q[None, qs:qs + lens_q[i]], k[None, ks:ks + lens_k[i]],
                   qv[None, qs:qs + lens_q[i]], v[None, ks:ks + lens_k[i]])
        ok &= check(f"varlen seq {i} (q={lens_q[i]:3d} k={lens_k[i]:3d})",
                    out[qs:qs + lens_q[i]], want, ms if i == 0 else None)
    return ok


def demo_paged():
    """Paged KV cache with a shuffled page table (page_size 64 -> cp.async gather)."""
    page_size, b, sq, sk, h = 64, 2, 1, 1024, 8
    npages = sk // page_size
    q, qv = rand(b * sq, h, HDIM), rand(b * sq, h, HDIMV)
    # physical pages in random order; page_table maps (batch, logical page) -> physical
    perm = torch.randperm(b * npages, device=DEV).to(torch.int32)
    page_table = perm.reshape(b, npages).contiguous()
    k_paged = rand(b * npages, page_size, 1, HDIM)
    v_paged = rand(b * npages, page_size, 1, HDIMV)
    cu_q = torch.arange(0, (b + 1) * sq, sq, device=DEV, dtype=torch.int32)
    seqused_k = torch.full((b,), sk, device=DEV, dtype=torch.int32)
    call = lambda: flash_attn_varlen_func(  # noqa: E731
        q, k_paged, v_paged, qv, cu_seqlens_q=cu_q, max_seqlen_q=sq, max_seqlen_k=sk,
        seqused_k=seqused_k, page_table=page_table, softmax_scale=SCALE)
    out, _ = call()
    ms = bench(call)
    # gather the pages back into a contiguous cache for the reference
    for i in range(b):
        kc = k_paged[page_table[i].long()].reshape(1, sk, 1, HDIM)
        vc = v_paged[page_table[i].long()].reshape(1, sk, 1, HDIMV)
        want = ref(q[None, i:i + 1], kc, qv[None, i:i + 1], vc)
        check(f"paged page_size={page_size} batch {i}", out[i:i + 1], want,
              ms if i == 0 else None)


def demo_splits():
    """SplitKV: long context, small batch -- the KV range is split across more CTAs."""
    b, sq, sk, h = 1, 1, 131072, 64
    q, qv = rand(b, sq, h, HDIM), rand(b, sq, h, HDIMV)
    k, v = rand(b, sk, 1, HDIM), rand(b, sk, 1, HDIMV)
    want = ref(q, k, qv, v)
    base = None
    for ns in (1, 8, 32, 128):
        call = lambda ns=ns: flash_attn_func(q, k, v, qv=qv, softmax_scale=SCALE,
                                             num_splits=ns)
        out, _ = call()
        ms = bench(call, iters=10, warmup=3)
        base = base or ms
        kv_gb = b * sk * (HDIM + HDIMV) * 2 / 1e9
        check(f"num_splits={ns:3d}", out[0], want, ms,
              extra=f"  {kv_gb / (ms / 1e3) / 1e3:4.2f} TB/s  {base / ms:5.2f}x")


def demo_direct():
    """Drive the kernel class directly, bypassing interface.py."""
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    from flash_attn.cute.flash_fwd_mla_1cta_sm100 import FlashAttentionMLAForward1CtaSm100

    b, sq, sk, h, hk = 2, 64, 512, 8, 1
    q, qv = rand(b, sq, h, HDIM), rand(b, sq, h, HDIMV)
    k, v = rand(b, sk, hk, HDIM), rand(b, sk, hk, HDIMV)
    out = torch.empty(b, sq, h, HDIMV, device=DEV, dtype=DT)
    lse = torch.empty(b, sq, h, device=DEV, dtype=torch.float32)

    fa = FlashAttentionMLAForward1CtaSm100(
        is_causal=False, qhead_per_kvhead=h // hk, nheads_kv=hk,
        hdim=HDIM, hdimv=HDIMV, has_qk=True, pack_gqa=True, q_in_tmem=True,
    )
    cvt = lambda t: from_dlpack(t.detach(), assumed_align=16)  # noqa: E731
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    args = (cvt(q), cvt(qv), cvt(k), cvt(v), cvt(out), cvt(lse), cutlass.Float32(SCALE))
    compiled = cute.compile(fa, *args, stream=stream)   # JIT -> PTX/CUBIN
    compiled(*args, stream=stream)
    torch.cuda.synchronize()
    ms = bench(lambda: compiled(*args, stream=stream))
    check(f"direct kernel call b{b} sq{sq} sk{sk} h{h}", out[0], ref(q[:1], k[:1], qv[:1], v[:1]), ms)


DEMOS = {
    "dense": demo_dense,
    "varlen": demo_varlen,
    "paged": demo_paged,
    "splits": demo_splits,
    "direct": demo_direct,
}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("demos", nargs="*", choices=list(DEMOS) + [[]], default=[],
                    help="demos to run (default: all)")
    ap.add_argument("--list", action="store_true", help="list demos and exit")
    args = ap.parse_args()

    if args.list:
        for n, f in DEMOS.items():
            print(f"  {n:8s} {f.__doc__.splitlines()[0]}")
        sys.exit(0)

    props = torch.cuda.get_device_properties(0)
    if props.major != 10:
        sys.exit(f"needs an SM100 GPU, found sm_{props.major}{props.minor}")
    print(f"{props.name}, {props.multi_processor_count} SMs | "
          f"MLA absorbed hdim={HDIM} hdim_v={HDIMV} | bf16")
    print("(first run of each shape includes JIT compilation)\n")

    for name in args.demos or DEMOS:
        print(f"{name}:  {DEMOS[name].__doc__.splitlines()[0]}")
        DEMOS[name]()
        print()
