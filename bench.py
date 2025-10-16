import torch
from flash_attn.cute.interface import flash_attn_func, flash_attn_varlen_func
from flash_attn.cute.testing import attention_ref
from triton.testing import do_bench
from cutlass.base_dsl.utils.logger import setup_log
import faulthandler
import logging
import argparse


def attn_ref_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, window_size, learnable_sink):
    B = cu_seqlens_q.shape[0] - 1
    out = torch.zeros_like(q)
    for b in range(B):
        q_b = q[cu_seqlens_q[b]:cu_seqlens_q[b+1]]
        k_b = k[cu_seqlens_k[b]:cu_seqlens_k[b+1]]
        v_b = v[cu_seqlens_k[b]:cu_seqlens_k[b+1]]
        out[cu_seqlens_q[b]:cu_seqlens_q[b+1]] = attention_ref(q_b[None, :], k_b[None, :], v_b[None, :], window_size=window_size, learnable_sink=learnable_sink)[0]
    return out


def benchmark_standard(B, Q, K, H, D, kvheads_per_group, num_splits, **kwargs):
    pack_gqa = kvheads_per_group > 1

    q = torch.randn(B, Q, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, K, H // kvheads_per_group, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, K, H // kvheads_per_group, D, device="cuda", dtype=torch.bfloat16)

    def fn():
        return flash_attn_func(q, k, v, num_splits=num_splits, pack_gqa=pack_gqa)[0]

    results = do_bench(fn)
    flops = 2 * 2 * B * Q * K * H * D
    bytes_moved = sum(t.numel() * t.element_size() for t in (q, k, v))
    print("Flash Attention Benchmark:")
    print(f"  B: {B}")
    print(f"  Q: {Q}")
    print(f"  K: {K}")
    print(f"  H: {H}")
    print(f"  D: {D}")
    print(f"  kvheads_per_group: {kvheads_per_group}")
    print(f"  pack_gqa: {pack_gqa}")
    print(f"  num_splits: {num_splits}")
    print(f"  Avg time: {results} ms")
    print(f"  TFLOPS: {flops / results * 1e-9} TFLOPS")
    print(f"  BW: {bytes_moved / results * 1e-9} TB/s")

    o = fn()
    o_ref = attention_ref(q, k, v)[0]

    correctness = torch.allclose(o, o_ref, atol=1e-2, rtol=1e-2)

    if not correctness:
        print("Output does not match reference")
        print(f"  Max diff: {(o - o_ref).abs().max().item()}")
        print(f"  Mean diff: {(o - o_ref).abs().mean().item()}")
        print(f"  First 10 elements: {o.flatten()[:10]}")
    else:
        print("Output matches reference")


def benchmark_varlen(B, Q, K, H, D, kvheads_per_group, num_splits, **kwargs):
    pack_gqa = kvheads_per_group > 1

    cu_seqlens_q = torch.arange(0, B + 1, device="cuda", dtype=torch.int32) * Q
    cu_seqlens_k = torch.arange(0, B + 1, device="cuda", dtype=torch.int32) * K

    num_queries = cu_seqlens_q[-1].item()
    num_keys = cu_seqlens_k[-1].item()

    q = torch.ones(num_queries, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.ones(num_keys, H // kvheads_per_group, D, device="cuda", dtype=torch.bfloat16)
    v = torch.ones(num_keys, H // kvheads_per_group, D, device="cuda", dtype=torch.bfloat16)

    def fn():
        return flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, num_splits=num_splits, pack_gqa=pack_gqa)[0]

    results = do_bench(fn)

    flops = 0
    for b in range(B):
        num_queries_b = (cu_seqlens_q[b+1] - cu_seqlens_q[b]).item()
        num_keys_b = (cu_seqlens_k[b+1] - cu_seqlens_k[b]).item()
        flops += 2 * 2 * num_queries_b * num_keys_b * H * D
    bytes_moved = sum(t.numel() * t.element_size() for t in (q, k, v))

    print("Flash Attention Benchmark:")
    print(f"  cu_seqlens_q: {cu_seqlens_q.tolist()}")
    print(f"  cu_seqlens_k: {cu_seqlens_k.tolist()}")
    print(f"  H: {H}")
    print(f"  D: {D}")
    print(f"  kvheads_per_group: {kvheads_per_group}")
    print(f"  pack_gqa: {pack_gqa}")
    print(f"  num_splits: {num_splits}")
    print(f"  Avg time: {results} ms")
    print(f"  TFLOPS: {flops / results * 1e-9} TFLOPS")
    print(f"  BW: {bytes_moved / results * 1e-9} TB/s")

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
    args.add_argument("--B", type=int, default=1)
    args.add_argument("--Q", type=int, default=32)
    args.add_argument("--K", type=int, default=8192)
    args.add_argument("--H", type=int, default=32)
    args.add_argument("--D", type=int, default=128)
    args.add_argument("--kvheads_per_group", type=int, default=8)
    args = args.parse_args()

    faulthandler.enable()
    if args.should_log:
        setup_log("cutlass", log_to_console=True, log_level=logging.INFO)

    if args.benchmark_type == "standard":
        benchmark_standard(**vars(args))
    elif args.benchmark_type == "varlen":
        benchmark_varlen(**vars(args))

if __name__ == "__main__":
    main()