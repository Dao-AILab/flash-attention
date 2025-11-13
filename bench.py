import torch

from flash_attn.cute.interface import flash_attn_varlen_func
from flash_attn.cute.testing import attention_ref

from triton.testing import do_bench
from cutlass.base_dsl.utils.logger import setup_log
import faulthandler
import logging
import argparse
import itertools


def benchmark(num_kv_entries, page_size, batch_size, seqlen_q, seqlen_k, nhead, head_dim, output_format, **kwargs):
    assert num_kv_entries % page_size == 0, "num_kv_entries must be divisible by page_size"
    assert seqlen_k % page_size == 0, "seqlen_k must be divisible by page_size"
    num_pages = num_kv_entries // page_size
    pages_per_seq = seqlen_k // page_size

    q = torch.randn(batch_size, seqlen_q, nhead, head_dim, device="cuda", dtype=torch.bfloat16)
    k_cache = torch.randn(num_pages, page_size, nhead, head_dim, device="cuda", dtype=torch.bfloat16)
    v_cache = torch.randn(num_pages, page_size, nhead, head_dim, device="cuda", dtype=torch.bfloat16)

    # page_table = torch.arange(0, batch_size * pages_per_seq, device="cuda", dtype=torch.int32).reshape(batch_size, pages_per_seq)
    page_table = torch.randint(0, num_pages, (batch_size, pages_per_seq), device="cuda", dtype=torch.int32)

    def fn():
        return flash_attn_varlen_func(q, k_cache, v_cache, page_table=page_table)[0]

    results = do_bench(fn, warmup=1000, rep=1000)
    o = fn()

    k = k_cache[page_table, :].flatten(1, 2)
    v = v_cache[page_table, :].flatten(1, 2)

    flops = 2 * 2 * batch_size * seqlen_q * seqlen_k * nhead * head_dim
    bytes_moved = sum(t.numel() * t.element_size() for t in (q, k, v, o))

    o_ref = attention_ref(q, k, v)[0]
    o_pytorch = attention_ref(q, k, v, upcast=True)[0]

    correctness = (o - o_ref).abs().max().item() <= 2 * (o_ref - o_pytorch).abs().max().item() + 1e-3

    if output_format == "text":
        print("Flash Attention Benchmark:")
        print(f"  num_kv_entries: {num_kv_entries}")
        print(f"  page_size: {page_size}")
        print(f"  batch_size: {batch_size}")
        print(f"  seqlen_q: {seqlen_q}")
        print(f"  seqlen_k: {seqlen_k}")
        print(f"  nhead: {nhead}")
        print(f"  head_dim: {head_dim}")
        print(f"  Arithmetic intensity: {flops / bytes_moved} FLOPS/byte")
        print(f"  Avg time: {results} ms")
        print(f"  TFLOPS: {flops / results * 1e-9} TFLOPS")
        print(f"  BW: {bytes_moved / results * 1e-9} TB/s")

        if not correctness:
            print("Output does not match reference")
            print(f"  NaN: {torch.isnan(o).sum().item()}")
            print(f"  Inf: {torch.isinf(o).sum().item()}")
            print(f"  Max diff: {(o - o_ref).abs().max().item()}")
            print(f"  Mean diff: {(o - o_ref).abs().mean().item()}")
            print(f"  First 10 elements: {o.flatten()[:10]}")
        else:
            print("Output matches reference")
    else:
        print(f"{num_kv_entries},{page_size},{batch_size},{seqlen_q},{seqlen_k},{nhead},{head_dim},{flops / bytes_moved},{results},{flops / results * 1e-9},{bytes_moved / results * 1e-9},{correctness}")


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--should_log", action="store_true")
    args.add_argument("--output_format", type=str, choices=["text", "csv"], default="text")
    args.add_argument("--num_kv_entries", type=int, nargs="+", default=[4 * 131072])
    args.add_argument("--page_size", type=int, nargs="+", default=[128])
    args.add_argument("--batch_size", type=int, nargs="+", default=[4])
    args.add_argument("--seqlen_q", type=int, nargs="+", default=[1])
    args.add_argument("--seqlen_k", type=int, nargs="+", default=[131072])
    args.add_argument("--nhead", type=int, nargs="+", default=[32])
    args.add_argument("--head_dim", type=int, nargs="+", default=[128])
    args = args.parse_args()

    faulthandler.enable()
    if args.should_log:
        setup_log("cutlass", log_to_console=True, log_level=logging.INFO)

    if args.output_format == "csv":
        print("num_kv_entries,page_size,batch_size,seqlen_q,seqlen_k,nhead,head_dim,arithmetic_intensity,avg_time,tflops,bw,correctness")

    for num_kv_entries, page_size, batch_size, seqlen_q, seqlen_k, nhead, head_dim in itertools.product(args.num_kv_entries, args.page_size, args.batch_size, args.seqlen_q, args.seqlen_k, args.nhead, args.head_dim):
        benchmark(num_kv_entries, page_size, batch_size, seqlen_q, seqlen_k, nhead, head_dim, args.output_format)

if __name__ == "__main__":
    main()