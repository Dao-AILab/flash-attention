import torch

# import importlib
# import flash_attn
# from pathlib import Path

# custom_path = Path(__file__).resolve().parent / "flash_attn"
# flash_attn.__path__.insert(0, str(custom_path))

# importlib.invalidate_caches()

from flash_attn.cute.interface import flash_attn_func, flash_attn_varlen_func
from flash_attn.cute.testing import attention_ref
# from flash_attn.flash_attn_interface import flash_attn_with_kvcache
# from flash_attn.utils.testing import attention_ref

from triton.testing import do_bench
from cutlass.base_dsl.utils.logger import setup_log
import faulthandler
import logging
import argparse
import itertools


def attn_ref_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, upcast=False):
    B = cu_seqlens_q.shape[0] - 1
    out = torch.zeros_like(q)
    for b in range(B):
        q_b = q[cu_seqlens_q[b]:cu_seqlens_q[b+1]]
        k_b = k[cu_seqlens_k[b]:cu_seqlens_k[b+1]]
        v_b = v[cu_seqlens_k[b]:cu_seqlens_k[b+1]]
        out[cu_seqlens_q[b]:cu_seqlens_q[b+1]] = attention_ref(q_b[None, :], k_b[None, :], v_b[None, :], upcast=upcast)[0]
    return out


def benchmark(benchmark_type, B, Q, K, H, D, kvheads_per_group, num_splits, output_format, **kwargs):
    if benchmark_type in ["standard", "fa2"]:
        q = torch.randn(B, Q, H, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, K, H // kvheads_per_group, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, K, H // kvheads_per_group, D, device="cuda", dtype=torch.bfloat16)

        if benchmark_type == "standard":
            def fn():
                return flash_attn_func(q, k, v, num_splits=num_splits)[0]
        else:
            def fn():
                return flash_attn_with_kvcache(q, k, v, num_splits=num_splits)[0]

        o_pytorch = attention_ref(q, k, v)[0]
        o_ref = attention_ref(q, k, v, upcast=True)[0]

    elif benchmark_type == "varlen":
        cu_seqlens_q = torch.arange(0, B + 1, device="cuda", dtype=torch.int32) * Q
        cu_seqlens_k = torch.arange(0, B + 1, device="cuda", dtype=torch.int32) * K

        num_queries = cu_seqlens_q[-1].item()
        num_keys = cu_seqlens_k[-1].item()

        q = torch.randn(num_queries, H, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(num_keys, H // kvheads_per_group, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(num_keys, H // kvheads_per_group, D, device="cuda", dtype=torch.bfloat16)

        def fn():
            return flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, num_splits=num_splits)[0]

        o_pytorch = attn_ref_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k)
        o_ref = attn_ref_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, upcast=True)

    else:
        raise ValueError(f"Invalid benchmark type: {benchmark_type}")

    results = do_bench(fn, warmup=1000, rep=1000)
    o = fn()

    flops = 2 * 2 * B * Q * K * H * D
    bytes_moved = sum(t.numel() * t.element_size() for t in (q, k, v, o))

    correctness = (o - o_ref).abs().max().item() <= 2 * (o - o_pytorch).abs().max().item() + 1e-3

    if output_format == "text":
        print("Flash Attention Benchmark:")
        print(f"  Benchmark type: {benchmark_type}")
        print(f"  B: {B}")
        print(f"  Q: {Q}")
        print(f"  K: {K}")
        print(f"  H: {H}")
        print(f"  D: {D}")
        print(f"  kvheads_per_group: {kvheads_per_group}")
        print(f"  num_splits: {num_splits}")
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
        print(f"{benchmark_type},{num_splits},{B},{Q},{K},{H},{D},{kvheads_per_group},{flops / bytes_moved},{results},{flops / results * 1e-9},{bytes_moved / results * 1e-9},{correctness}")


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--should_log", action="store_true")
    args.add_argument("--output_format", type=str, choices=["text", "csv"], default="text")
    args.add_argument("--benchmark_type", type=str, nargs="+", choices=["standard", "varlen", "fa2"], default=["standard"])
    args.add_argument("--num_splits", type=int, nargs="+", default=[4])
    args.add_argument("--B", type=int, nargs="+", default=[1])
    args.add_argument("--Q", type=int, nargs="+", default=[32])
    args.add_argument("--K", type=int, nargs="+", default=[131072])
    args.add_argument("--H", type=int, nargs="+", default=[32])
    args.add_argument("--D", type=int, nargs="+", default=[128])
    args.add_argument("--kvheads_per_group", type=int, nargs="+", default=[8])
    args = args.parse_args()

    faulthandler.enable()
    if args.should_log:
        setup_log("cutlass", log_to_console=True, log_level=logging.INFO)

    if args.output_format == "csv":
        print("benchmark_type,num_splits,B,Q,K,H,D,kvheads_per_group,arithmetic_intensity,avg_time,tflops,bw,correctness")

    for num_splits, benchmark_type, B, Q, K, H, D, kvheads_per_group in itertools.product(args.num_splits, args.benchmark_type, args.B, args.Q, args.K, args.H, args.D, args.kvheads_per_group):
        benchmark(benchmark_type, B, Q, K, H, D, kvheads_per_group, num_splits, args.output_format)

if __name__ == "__main__":
    main()