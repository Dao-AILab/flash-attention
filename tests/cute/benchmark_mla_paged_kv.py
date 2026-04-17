"""
Benchmark: MLA paged KV cache on SM100 (Blackwell).

Compares non-paged (varlen) vs paged KV with different page sizes:
  - Non-paged: contiguous KV via cu_seqlens_k (baseline)
  - TMA paged: page_size=128 (matches tile_n, uses TMA path)
  - cp.async paged: page_size=16, 64 (arbitrary, uses cp.async path)

Usage:
  python tests/cute/benchmark_mla_paged_kv.py
  python tests/cute/benchmark_mla_paged_kv.py --causal
  python tests/cute/benchmark_mla_paged_kv.py --seqlen_q 1 --batch_size 64   # decode-like
  python tests/cute/benchmark_mla_paged_kv.py --seqlen_q 128 --batch_size 4  # prefill-like

Tip: Lock GPU clocks for stable results:
  sudo nvidia-smi -i 0 -pm 1
  sudo nvidia-smi -i 0 --lock-gpu-clocks 1830,1830
"""

import argparse
import time
import torch
from triton.testing import do_bench

from flash_attn.cute.interface import flash_attn_varlen_func


def make_nonpaged_tensors(batch_size, seqlen_q, seqlen_k, nheads, nheads_kv, d, dv, dtype, device):
    q = torch.randn(batch_size * seqlen_q, nheads, d, device=device, dtype=dtype)
    k = torch.randn(batch_size * seqlen_k, nheads_kv, d, device=device, dtype=dtype)
    v = torch.randn(batch_size * seqlen_k, nheads_kv, dv, device=device, dtype=dtype)
    qv = torch.randn(batch_size * seqlen_q, nheads, dv, device=device, dtype=dtype)
    cu_seqlens_q = torch.tensor(
        [i * seqlen_q for i in range(batch_size + 1)], dtype=torch.int32, device=device
    )
    cu_seqlens_k = torch.tensor(
        [i * seqlen_k for i in range(batch_size + 1)], dtype=torch.int32, device=device
    )
    return q, k, v, qv, cu_seqlens_q, cu_seqlens_k


def make_paged_tensors(k_contiguous, v_contiguous, batch_size, seqlen_k, page_size, nheads_kv, d, dv, dtype, device):
    num_pages_per_seq = (seqlen_k + page_size - 1) // page_size
    total_pages = num_pages_per_seq * batch_size
    k_paged = torch.zeros(total_pages, page_size, nheads_kv, d, device=device, dtype=dtype)
    v_paged = torch.zeros(total_pages, page_size, nheads_kv, dv, device=device, dtype=dtype)
    page_table = torch.zeros(batch_size, num_pages_per_seq, dtype=torch.int32, device=device)

    for b in range(batch_size):
        for p in range(num_pages_per_seq):
            page_idx = b * num_pages_per_seq + p
            start = p * page_size
            end = min(start + page_size, seqlen_k)
            k_offset = b * seqlen_k
            if start < seqlen_k:
                k_paged[page_idx, :end - start] = k_contiguous[k_offset + start:k_offset + end]
                v_paged[page_idx, :end - start] = v_contiguous[k_offset + start:k_offset + end]
            page_table[b, p] = page_idx

    seqused_k = torch.full((batch_size,), seqlen_k, dtype=torch.int32, device=device)
    return k_paged, v_paged, page_table, seqused_k


def compute_flops(batch_size, seqlen_q, seqlen_k, nheads, d, dv, causal):
    # QK^T: 2 * b * sq * sk * h * d, PV: 2 * b * sq * sk * h * dv
    # For MLA with qv: QK^T uses d, PV uses dv (via qv projection)
    total = 2 * batch_size * seqlen_q * seqlen_k * nheads * (d + dv)
    if causal:
        total //= 2  # approximate for causal
    return total


def compute_mem_bytes(batch_size, seqlen_q, seqlen_k, nheads, nheads_kv, d, dv, dtype_bytes):
    # Q + QV reads
    q_bytes = batch_size * seqlen_q * nheads * d * dtype_bytes
    qv_bytes = batch_size * seqlen_q * nheads * dv * dtype_bytes
    # K + V reads
    k_bytes = batch_size * seqlen_k * nheads_kv * d * dtype_bytes
    v_bytes = batch_size * seqlen_k * nheads_kv * dv * dtype_bytes
    # O write
    o_bytes = batch_size * seqlen_q * nheads * dv * dtype_bytes
    return q_bytes + qv_bytes + k_bytes + v_bytes + o_bytes


def benchmark_config(batch_size, seqlen_q, seqlen_k, causal, page_sizes, warmup=3, rep=20):
    device = "cuda"
    dtype = torch.bfloat16
    d, dv = 64, 512
    nheads = 128
    nheads_kv = 1  # MQA-128

    torch.manual_seed(0)

    q, k, v, qv, cu_seqlens_q, cu_seqlens_k = make_nonpaged_tensors(
        batch_size, seqlen_q, seqlen_k, nheads, nheads_kv, d, dv, dtype, device
    )

    flops = compute_flops(batch_size, seqlen_q, seqlen_k, nheads, d, dv, causal)
    mem_bytes = compute_mem_bytes(batch_size, seqlen_q, seqlen_k, nheads, nheads_kv, d, dv, 2)

    results = {}

    # Non-paged baseline
    fn_nonpaged = lambda: flash_attn_varlen_func(
        q, k, v, qv=qv,
        cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=seqlen_q, max_seqlen_k=seqlen_k,
        causal=causal,
    )
    # Warmup (triggers compilation)
    fn_nonpaged()
    time.sleep(0.5)
    t_ms = do_bench(fn_nonpaged, warmup=warmup, rep=rep)
    results["non-paged"] = t_ms

    # Paged variants
    for ps in page_sizes:
        if seqlen_k < ps:
            continue
        k_paged, v_paged, page_table, seqused_k = make_paged_tensors(
            k, v, batch_size, seqlen_k, ps, nheads_kv, d, dv, dtype, device
        )
        fn_paged = lambda k_p=k_paged, v_p=v_paged, pt=page_table, su=seqused_k: flash_attn_varlen_func(
            q, k_p, v_p, qv=qv,
            cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=None,
            max_seqlen_q=seqlen_q, max_seqlen_k=None,
            seqused_k=su, page_table=pt,
            causal=causal,
        )
        path = "TMA" if ps == 128 else "cp.async"
        label = f"paged-{ps} ({path})"
        # Warmup
        fn_paged()
        time.sleep(0.5)
        t_ms = do_bench(fn_paged, warmup=warmup, rep=rep)
        results[label] = t_ms

    return results, flops, mem_bytes


def main():
    parser = argparse.ArgumentParser(description="Benchmark MLA paged KV cache")
    parser.add_argument("--batch_size", type=int, nargs="+", default=[2, 8, 32])
    parser.add_argument("--seqlen_q", type=int, nargs="+", default=[1, 64, 128])
    parser.add_argument("--seqlen_k", type=int, nargs="+", default=[1024, 4096, 8192])
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--page_sizes", type=int, nargs="+", default=[16, 64, 128])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rep", type=int, default=20)
    args = parser.parse_args()

    print(f"MLA Paged KV Benchmark (d=64, dv=512, nheads=128, MQA-128, bf16)")
    print(f"Causal: {args.causal}, Page sizes: {args.page_sizes}")
    print(f"{'='*100}")

    all_results = []
    for bs in args.batch_size:
        for sq in args.seqlen_q:
            for sk in args.seqlen_k:
                print(f"\nbatch={bs}, seqlen_q={sq}, seqlen_k={sk}, causal={args.causal}")
                try:
                    results, flops, mem_bytes = benchmark_config(
                        bs, sq, sk, args.causal, args.page_sizes,
                        warmup=args.warmup, rep=args.rep,
                    )
                except torch.OutOfMemoryError:
                    print("  OOM, skipping")
                    continue

                # Print results
                baseline_ms = results.get("non-paged")
                for label, t_ms in results.items():
                    tflops = flops / (t_ms * 1e-3) / 1e12
                    gbps = mem_bytes / (t_ms * 1e-3) / 1e9
                    speedup = f"{baseline_ms / t_ms:.2f}x" if label != "non-paged" else "baseline"
                    print(f"  {label:25s}: {t_ms*1e3:8.1f} us | {tflops:6.1f} TFLOPS | {gbps:7.0f} GB/s | {speedup}")

                all_results.append((bs, sq, sk, results))

    # Summary table
    if len(all_results) > 1:
        print(f"\n{'='*100}")
        print("Summary (time in us):")
        headers = ["batch", "sq", "sk"]
        # Collect all labels
        all_labels = []
        for _, _, _, r in all_results:
            for label in r:
                if label not in all_labels:
                    all_labels.append(label)
        headers += all_labels

        rows = []
        for bs, sq, sk, results in all_results:
            row = [bs, sq, sk]
            for label in all_labels:
                t = results.get(label)
                row.append(f"{t*1e3:.1f}" if t is not None else "-")
            rows.append(row)

        # Print table
        col_widths = [max(len(str(h)), max(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
        fmt = "  ".join(f"{{:>{w}}}" for w in col_widths)
        print(fmt.format(*headers))
        print("  ".join("-" * w for w in col_widths))
        for row in rows:
            print(fmt.format(*row))


if __name__ == "__main__":
    main()
