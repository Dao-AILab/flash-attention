"""Stress benchmark for the FA4 (CuTe-DSL) varlen tile scheduler.

Targets the kernels in ``flash_attn/cute/`` (FA4) — specifically
``SingleTileVarlenScheduler`` in ``flash_attn/cute/tile_scheduler.py`` — and is
not applicable to the older FA2 (``csrc/``) or FA3 (``hopper/``) generations.

The scheduler maps a flat CTA index to ``(block, head, batch)`` by scanning
batches in groups of 31 (one warp) on the GPU, per CTA. Without precomputed
metadata each CTA does an O(num_batches / 31) linear scan, giving
O(num_batches^2) total scheduling work across the launched grid. The
pathological regime is many short sequences, where per-tile scheduling cost
swamps the actual attention compute.

This script sweeps num_seqs with the per-seq length fixed to a small value
(seq_len=34, mimicking the URL-embedding workload). True attention compute
scales linearly in num_seqs at fixed seq_len, so any super-linear growth in
the per-iter time comes from the scheduler.
"""

import argparse

import torch

from flash_attn.cute import flash_attn_func, flash_attn_varlen_func


def make_fixed_len_data(num_seqs, seq_len, num_heads, head_dim, device, dtype):
    total = num_seqs * seq_len
    q = torch.randn(total, num_heads, head_dim, dtype=dtype, device=device) * 0.1
    k = torch.randn(total, num_heads, head_dim, dtype=dtype, device=device) * 0.1
    v = torch.randn(total, num_heads, head_dim, dtype=dtype, device=device) * 0.1
    cu = torch.arange(0, total + 1, seq_len, dtype=torch.int32, device=device)
    return q, k, v, cu


def make_padded_view(q, k, v, num_seqs, seq_len):
    """Reshape packed (total, H, D) tensors to dense (num_seqs, seq_len, H, D) — no copy."""
    H, D = q.shape[1], q.shape[2]
    return (
        q.view(num_seqs, seq_len, H, D),
        k.view(num_seqs, seq_len, H, D),
        v.view(num_seqs, seq_len, H, D),
    )


def cuda_bench(fn, warmup=3, iters=20):
    """Time a callable with CUDA events. Returns mean ms per call."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for s, e in zip(starts, ends):
        s.record()
        fn()
        e.record()
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    return sum(times) / len(times)


def bench_one(num_seqs, seq_len, num_heads, head_dim, dtype, device, with_bwd):
    q, k, v, cu = make_fixed_len_data(num_seqs, seq_len, num_heads, head_dim, device, dtype)
    qp, kp, vp = make_padded_view(q, k, v, num_seqs, seq_len)

    def fwd_varlen():
        out = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu, cu_seqlens_k=cu,
            max_seqlen_q=seq_len, max_seqlen_k=seq_len,
        )
        if isinstance(out, tuple):
            out = out[0]
        return out

    def fwd_dense():
        out = flash_attn_func(qp, kp, vp)
        if isinstance(out, tuple):
            out = out[0]
        return out

    fwd_varlen_ms = cuda_bench(fwd_varlen)
    fwd_dense_ms = cuda_bench(fwd_dense)

    fb_varlen_ms = fb_dense_ms = float("nan")
    if with_bwd:
        def fb_varlen():
            qg = q.detach().requires_grad_(True)
            kg = k.detach().requires_grad_(True)
            vg = v.detach().requires_grad_(True)
            out = flash_attn_varlen_func(
                qg, kg, vg,
                cu_seqlens_q=cu, cu_seqlens_k=cu,
                max_seqlen_q=seq_len, max_seqlen_k=seq_len,
            )
            if isinstance(out, tuple):
                out = out[0]
            out.sum().backward()

        def fb_dense():
            qg = qp.detach().requires_grad_(True)
            kg = kp.detach().requires_grad_(True)
            vg = vp.detach().requires_grad_(True)
            out = flash_attn_func(qg, kg, vg)
            if isinstance(out, tuple):
                out = out[0]
            out.sum().backward()

        fb_varlen_ms = cuda_bench(fb_varlen, iters=10)
        fb_dense_ms = cuda_bench(fb_dense, iters=10)

    return fwd_varlen_ms, fwd_dense_ms, fb_varlen_ms, fb_dense_ms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq-len", type=int, default=34,
                   help="per-sequence length (small => scheduler dominates)")
    p.add_argument("--num-heads", type=int, default=12)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--no-bwd", action="store_true",
                   help="skip the forward+backward sweep (faster)")
    p.add_argument(
        "--num-seqs",
        type=int,
        nargs="+",
        default=[128, 256, 512, 1024, 2048, 4096, 8192],
        help="batch sizes to sweep",
    )
    args = p.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    device = torch.device(args.device)

    print(f"GPU: {torch.cuda.get_device_name(device)}  (cap {torch.cuda.get_device_capability(device)})")
    print(f"seq_len={args.seq_len}  heads={args.num_heads}  head_dim={args.head_dim}  dtype={args.dtype}")
    print()

    # Warm the JIT cache once at the smallest size to keep timing clean.
    print("(warming JIT...)")
    bench_one(args.num_seqs[0], args.seq_len, args.num_heads, args.head_dim,
              dtype, device, with_bwd=not args.no_bwd)
    print()

    cols = ["n_seq", "tot_tok",
            "fwd_vl(ms)", "fwd_dn(ms)", "vl/dn",
            "fwd_vl/seq(us)",
            "bwd_vl(ms)", "bwd_dn(ms)", "vl/dn",
            "bwd_vl/seq(us)"]
    fmt = "{:>6} {:>8} {:>10} {:>10} {:>6} {:>14} {:>10} {:>10} {:>6} {:>14}"
    print(fmt.format(*cols))
    print("-" * 110)

    for n in args.num_seqs:
        try:
            fv, fd, bv, bd = bench_one(
                n, args.seq_len, args.num_heads, args.head_dim,
                dtype, device, with_bwd=not args.no_bwd,
            )
        except torch.cuda.OutOfMemoryError:
            print(f"{n:>6} OOM — stopping sweep")
            break

        total_tok = n * args.seq_len
        per_seq_fv = fv * 1000.0 / n  # microseconds
        per_seq_bv = bv * 1000.0 / n

        print(fmt.format(
            n, total_tok,
            f"{fv:.3f}", f"{fd:.3f}", f"{fv/fd:.2f}x",
            f"{per_seq_fv:.3f}",
            f"{bv:.3f}" if bv == bv else "n/a",
            f"{bd:.3f}" if bd == bd else "n/a",
            f"{bv/bd:.2f}x" if bv == bv else "n/a",
            f"{per_seq_bv:.3f}" if bv == bv else "n/a",
        ))

    print()
    print("Reading the table:")
    print("  - 'tot_tok' is total live tokens; true attention work is O(n_seq) at fixed seq_len.")
    print("  - If the scheduler were O(n_seq), 'fwd_vl/seq(us)' would be flat across rows.")
    print("  - O(n_seq^2) scheduling shows up as 'fwd_vl/seq(us)' growing roughly linearly with n_seq.")
    print("  - 'fwd_dn' is the dense (padded) baseline at the same total token count — pure compute, no varlen scheduling.")


if __name__ == "__main__":
    main()
