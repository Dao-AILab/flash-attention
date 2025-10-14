import torch
from flash_attn.cute.interface import flash_attn_func, flash_attn_combine, flash_attn_varlen_func
from triton.testing import do_bench
from cutlass.base_dsl.utils.logger import setup_log
import faulthandler
import logging
import argparse


def attn_ref(q, k, v):
    s = torch.einsum("bqhd,bkhd->bhqk", q, k)
    s *= q.shape[-1] ** -0.5
    s = torch.softmax(s, dim=-1)
    return torch.einsum("bhqk,bkhd->bqhd", s, v)


def attn_ref_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k):
    B = cu_seqlens_q.shape[0] - 1
    out = torch.zeros_like(q)
    for b in range(B):
        q_b = q[cu_seqlens_q[b]:cu_seqlens_q[b+1]]
        k_b = k[cu_seqlens_k[b]:cu_seqlens_k[b+1]]
        v_b = v[cu_seqlens_k[b]:cu_seqlens_k[b+1]]
        s = torch.einsum("qhd,khd->hqk", q_b, k_b)
        s *= q_b.shape[-1] ** -0.5
        s = torch.softmax(s, dim=-1)
        out[cu_seqlens_q[b]:cu_seqlens_q[b+1]] = torch.einsum("hqk,khd->qhd", s, v_b)
    return out


def benchmark_standard(num_splits: int = 4):
    B, Q, K, H, D = 1, 256, 16384, 32, 128

    q = torch.randn(B, Q, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, K, H, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, K, H, D, device="cuda", dtype=torch.bfloat16)

    def fn():
        return flash_attn_func(q, k, v, num_splits=num_splits)[0]

    results = do_bench(fn)
    flops = 2 * 2 * B * Q * K * H * D
    print("Flash Attention Benchmark:")
    print(f"  B: {B}")
    print(f"  Q: {Q}")
    print(f"  K: {K}")
    print(f"  H: {H}")
    print(f"  D: {D}")
    print(f"  num_splits: {num_splits}")
    print(f"  Avg time: {results} ms")
    print(f"  TFLOPS: {flops / results * 1e-9} TFLOPS")

    o = fn()
    o_ref = attn_ref(q, k, v)

    correctness = torch.allclose(o, o_ref, atol=1e-2, rtol=1e-2)

    if not correctness:
        print("Output does not match reference")
        print(f"  Max diff: {(o - o_ref).abs().max().item()}")
        print(f"  Mean diff: {(o - o_ref).abs().mean().item()}")
        print(f"  First 10 elements: {o.flatten()[:10]}")
    else:
        print("Output matches reference")


def benchmark_varlen(num_splits: int = 4):
    B, Q, K, H, D = 2, 256, 16384, 32, 128

    cu_seqlens_q = torch.arange(0, B + 1, device="cuda", dtype=torch.int32) * Q
    cu_seqlens_k = torch.arange(0, B + 1, device="cuda", dtype=torch.int32) * K

    num_queries = cu_seqlens_q[-1].item()
    num_keys = cu_seqlens_k[-1].item()

    q = torch.randn(num_queries, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(num_keys, H, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(num_keys, H, D, device="cuda", dtype=torch.bfloat16)

    def fn():
        return flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, num_splits=num_splits)[0]

    results = do_bench(fn)

    flops = 0
    for b in range(B):
        num_queries_b = (cu_seqlens_q[b+1] - cu_seqlens_q[b]).item()
        num_keys_b = (cu_seqlens_k[b+1] - cu_seqlens_k[b]).item()
        flops += 2 * 2 * num_queries_b * num_keys_b * H * D

    print("Flash Attention Benchmark:")
    print(f"  cu_seqlens_q: {cu_seqlens_q.tolist()}")
    print(f"  cu_seqlens_k: {cu_seqlens_k.tolist()}")
    print(f"  H: {H}")
    print(f"  D: {D}")
    print(f"  num_splits: {num_splits}")
    print(f"  Avg time: {results} ms")
    print(f"  TFLOPS: {flops / results * 1e-9} TFLOPS")

    o = fn()
    o_ref = attn_ref_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k)

    correctness = torch.allclose(o, o_ref, atol=1e-2, rtol=1e-2)

    if not correctness:
        print("Output does not match reference")
        print(f"  Max diff: {(o - o_ref).abs().max().item()}")
        print(f"  Mean diff: {(o - o_ref).abs().mean().item()}")
        print(f"  First 10 elements: {o.flatten()[:10]}")
    else:
        print("Output matches reference")

def main():
    args = argparse.ArgumentParser()
    args.add_argument("--should_log", action="store_true")
    args.add_argument("--num_splits", type=int, default=4)
    args.add_argument("--benchmark_type", type=str, choices=["standard", "varlen"], default="standard")
    args = args.parse_args()

    faulthandler.enable()
    if args.should_log:
        setup_log("cutlass", log_to_console=True, log_level=logging.INFO)

    if args.benchmark_type == "standard":
        benchmark_standard(args.num_splits)
    elif args.benchmark_type == "varlen":
        benchmark_varlen(args.num_splits)

if __name__ == "__main__":
    main()