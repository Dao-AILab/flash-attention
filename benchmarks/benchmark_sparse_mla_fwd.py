"""Sparse (top-k gather) MLA forward: 1CTA (64-row tile) vs 2CTA (128-row tile) kernels.

MQA, hdim 64 (rope) + 512 (latent), inference (no grad). For each shape the same inputs
and index lists go through:
  2cta    FLASH_ATTENTION_MLA_1CTA=0 -- Q heads padded to 128 per token
  1cta    FLASH_ATTENTION_MLA_1CTA=1 -- padded to 64 (<= 64 heads only; the 64-key-block
          mainloop)
  dense   1CTA dense MLA over topk contiguous keys (same FLOPs, no gather): a
          speed-of-light reference for the gather

Timing:
  hot   same inputs back to back (KV can stay L2-resident across iterations)
  cold  L2 flushed before every timed call (write a 1 GiB buffer; not timed)
Effective payload bandwidth counts only the KV rows the kernel actually reads: valid slots
(0 <= idx < limit; masked slots are predicated off) x (64 + 512) elements, or x 512 when
has_qk=False (K is V). It is not DRAM bandwidth.

Example:
  python benchmarks/benchmark_sparse_mla_fwd.py --heads 16 32 48 64 --batch 1 8 32 128 \\
      --seqlen-k 8192 32768 131072 --seqlen-q 1
"""
import argparse
import itertools
import os

# large per-shape KV buffers (tens of GiB) + CUDA graph pools fragment the caching allocator
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch

from flash_attn.cute.interface import flash_attn_func


def make_indices(b, s_q, s_k, topk, causal, device, chunk=256):
    """Random distinct keys over each row's whole valid range [0, limit), -1 padded."""
    idx = torch.empty(b, s_q, topk, dtype=torch.int32, device=device)
    for q0 in range(0, s_q, chunk):
        q1 = min(s_q, q0 + chunk)
        scores = torch.rand(b, q1 - q0, s_k, device=device)
        if causal:
            t = torch.arange(q0, q1, device=device).view(1, -1, 1)
            limit = t + 1 + s_k - s_q
            scores.masked_fill_(torch.arange(s_k, device=device).view(1, 1, -1) >= limit, -1.0)
        k = min(topk, s_k)
        val, sel = scores.topk(k, dim=-1)
        sel = sel.masked_fill(val < 0, -1).to(torch.int32)
        if k < topk:
            sel = torch.cat([sel, sel.new_full((b, q1 - q0, topk - k), -1)], dim=-1)
        idx[:, q0:q1] = sel
    return idx


def time_fn(fn, iters, warmup, cold, flush_buf):
    """Mean GPU time per call, never host-bound (the Python interface costs about as much
    as a decode kernel). hot: one CUDA graph of `iters` back-to-back calls, replayed.
    cold: flush L2 (1 GiB write) then the call, timed with events around the call only."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    if not cold:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream), torch.cuda.graph(g, stream=stream):
            for _ in range(iters):
                fn()
        torch.cuda.current_stream().wait_stream(stream)
        g.replay()  # warm replay
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        g.replay()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters
    # cold: no graph needed -- the 1 GiB flush keeps the GPU busy far longer than the host
    # takes to enqueue the call, so the timed interval is the kernel alone.
    total = 0.0
    for _ in range(iters):
        flush_buf.zero_()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        total += start.elapsed_time(end)
    return total / iters


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--heads", type=int, nargs="+", default=[16, 32, 48, 64])
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 8, 32, 128])
    ap.add_argument("--seqlen-q", type=int, nargs="+", default=[1])
    ap.add_argument("--seqlen-k", type=int, nargs="+", default=[8192, 32768, 131072])
    ap.add_argument("--topk", type=int, default=2048)
    ap.add_argument("--has-qk", type=int, nargs="+", default=[1, 0])
    ap.add_argument("--causal", action="store_true", help="bottom-right causal (prefill)")
    ap.add_argument("--kernels", nargs="+", default=["2cta", "1cta", "dense"])
    ap.add_argument("--cache", nargs="+", default=["cold", "hot"])
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--csv", type=str, default=None, help="also append rows to this CSV")
    args = ap.parse_args()

    dev, dt = "cuda", torch.bfloat16
    flush_buf = torch.empty(1 << 30, dtype=torch.uint8, device=dev)
    props = torch.cuda.get_device_properties(0)
    print(f"{props.name}, {props.multi_processor_count} SMs, L2 {props.L2_cache_size / 2**20:.0f} MiB")
    header = ("kernel,cache,has_qk,heads,batch,s_q,s_k,topk,causal,ms,tokens_per_s,"
              "payload_GBps,match_2cta")
    print(header)
    csv = None
    if args.csv:
        new = not os.path.exists(args.csv)
        csv = open(args.csv, "a")
        if new:
            csv.write(header + "\n")
    for has_qk, h, b, s_q, s_k in itertools.product(
        args.has_qk, args.heads, args.batch, args.seqlen_q, args.seqlen_k
    ):
        torch.cuda.empty_cache()
        torch.manual_seed(0)
        q = torch.randn(b, s_q, h, 64, device=dev, dtype=dt)
        qv = torch.randn(b, s_q, h, 512, device=dev, dtype=dt)
        k = torch.randn(b, s_k, 1, 64, device=dev, dtype=dt)
        v = torch.randn(b, s_k, 1, 512, device=dev, dtype=dt)
        idx = make_indices(b, s_q, s_k, args.topk, args.causal, dev)
        t = torch.arange(s_q, device=dev).view(1, -1, 1)
        limit = (t + 1 + s_k - s_q) if args.causal else s_k
        valid_slots = ((idx >= 0) & (idx < limit)).sum().item()
        row_elems = 512 if not has_qk else 576
        payload_bytes = valid_slots * row_elems * 2
        if has_qk:
            kw = dict(q=q, k=k, v=v, qv=qv)
            kd, vd = k[:, : args.topk].contiguous(), v[:, : args.topk].contiguous()
            kw_dense = dict(q=q, k=kd, v=vd, qv=qv)
        else:
            kw = dict(q=qv, k=v, v=v)
            vd = v[:, : args.topk].contiguous()
            kw_dense = dict(q=qv, k=vd, v=vd)
        out_2cta = None
        for kernel, cache in itertools.product(args.kernels, args.cache):
            if kernel.startswith("1cta") and h > 64:
                continue
            os.environ["FLASH_ATTENTION_MLA_1CTA"] = "0" if kernel == "2cta" else "1"
            if kernel == "dense":
                fn = lambda: flash_attn_func(**kw_dense, causal=False)
            else:
                fn = lambda: flash_attn_func(**kw, gather_kv_indices=idx, causal=args.causal)
            out = fn()[0]
            match = ""
            if kernel == "2cta":
                out_2cta = out
            elif kernel.startswith("1cta") and out_2cta is not None:
                # "True" when bitwise; the kb64 mainloop agrees to bf16 rounding (rel-L2)
                o2 = out_2cta.float()
                match = ("True" if torch.equal(out, out_2cta)
                         else f"relL2={((out.float() - o2).norm() / o2.norm()).item():.1e}")
            ms = time_fn(fn, args.iters, args.warmup, cache == "cold", flush_buf)
            gbps = payload_bytes / (ms * 1e-3) / 1e9 if kernel != "dense" else (
                b * s_q * min(args.topk, s_k) * row_elems * 2) / (ms * 1e-3) / 1e9
            row = (f"{kernel},{cache},{int(has_qk)},{h},{b},{s_q},{s_k},{args.topk},"
                   f"{int(args.causal)},{ms:.4f},{b * s_q / (ms * 1e-3):.0f},{gbps:.1f},{match}")
            print(row, flush=True)
            if csv is not None:
                csv.write(row + "\n")
                csv.flush()
        del q, qv, k, v, idx, kw, kw_dense, fn, out, out_2cta


if __name__ == "__main__":
    main()
