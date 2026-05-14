# pyright: reportMissingImports=false
import argparse
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

_REPO_ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / "flash_attn" / "cute" / "interface.py").exists()
)
sys.path.insert(0, str(_REPO_ROOT))
from flash_attn.cute import interface as fa_interface  # noqa: E402
from flash_attn.cute import flash_attn_varlen_func  # noqa: E402

CORE = [
    ("typical_d64", [512, 1024, 768, 256], 20, 20, 64),
    ("typical_gqa", [512, 1024, 768], 32, 8, 128),
    ("typical_d128", [512, 768, 256], 16, 16, 128),
    ("boundary_exact_128", [128, 256, 512], 20, 20, 64),
    ("boundary_exact_256", [256, 512, 1024], 32, 8, 128),
    ("boundary_off_p1", [129, 257, 513], 20, 20, 64),
    ("boundary_off_m1", [127, 255, 511], 20, 20, 64),
    ("mixed_8_4096", [8, 4096], 20, 20, 64),
    ("mixed_4_16_2048_32", [4, 16, 2048, 32], 20, 20, 64),
    ("all_short_8", [8] * 6, 20, 20, 64),
    ("all_short_16", [16] * 8, 20, 20, 64),
    ("single_token_each", [1, 1, 1, 1], 20, 20, 64),
    ("single_token_mixed", [1, 64, 1, 256, 1], 20, 20, 64),
    ("gqa_32x8", [64, 128, 64], 32, 8, 128),
    ("gqa_16x4", [128, 512, 256], 16, 4, 64),
    ("mqa_32x1", [512, 1024, 256], 32, 1, 128),
    ("single_seq_1024", [1024], 20, 20, 64),
    ("single_seq_4096", [4096], 20, 20, 64),
    ("many_32x20", [32] * 20, 20, 20, 64),
    ("d128_long", [2048], 16, 16, 128),
]

SEQ_LENS = [1024, 4096, 16384, 65536]
SPEED_HEAD_CONFIGS = [
    ("mha_h20_d64", 20, 20, 64),
    ("gqa_h16_hkv4_d128", 16, 4, 128),
]


def random_cases(n=8, seed=0):
    rng = random.Random(seed)
    lens_pool = [1, 4, 8, 16, 32, 64, 127, 128, 129, 256, 511, 512, 1024]
    ans = []
    for i in range(n):
        lens = [rng.choice(lens_pool) for _ in range(rng.randint(1, 8))]
        hkv, qmul, d = rng.choice([1, 4, 8, 16]), rng.choice([1, 2, 4]), rng.choice([64, 128])
        ans.append((f"random_{i}", lens, hkv * qmul, hkv, d))
    return ans


@dataclass
class VarlenCase:
    name: str
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    dout: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int
    softmax_scale: float


@dataclass(frozen=True)
class DiffStats:
    out_max: float
    out_mean: float
    dq_max: float
    dq_mean: float
    dk_max: float
    dk_mean: float
    dv_max: float
    dv_mean: float


@dataclass(frozen=True)
class BenchResult:
    name: str
    nondeter_ms: float
    deter_ms: float
    diff: DiffStats


def make_case(name, lens, nheads_q, nheads_kv, headdim, dtype=torch.bfloat16) -> VarlenCase:
    seqlens = torch.tensor(lens, dtype=torch.int32)
    cu = torch.cat([torch.zeros(1, dtype=torch.int32), seqlens.cumsum(0, dtype=torch.int32)]).cuda()
    total, max_seqlen = int(cu[-1].item()), int(seqlens.max().item())
    q = torch.randn(total, nheads_q, headdim, device="cuda", dtype=dtype)
    k = torch.randn(total, nheads_kv, headdim, device="cuda", dtype=dtype)
    v = torch.randn(total, nheads_kv, headdim, device="cuda", dtype=dtype)
    return VarlenCase(name, q, k, v, torch.randn_like(q), cu, cu, max_seqlen, max_seqlen, 1.0 / math.sqrt(headdim))


def run_backward(case: VarlenCase, deterministic: bool):
    q = case.q.detach().clone().requires_grad_(True)
    k = case.k.detach().clone().requires_grad_(True)
    v = case.v.detach().clone().requires_grad_(True)
    out, _ = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=case.cu_seqlens_q,
        cu_seqlens_k=case.cu_seqlens_k,
        max_seqlen_q=case.max_seqlen_q,
        max_seqlen_k=case.max_seqlen_k,
        softmax_scale=case.softmax_scale,
        deterministic=deterministic,
    )
    out.backward(case.dout)
    return tuple(x.detach().clone() for x in (out, q.grad, k.grad, v.grad))


def bench_ms(fn, warmup=2, iters=5) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def diff_stats(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    diff = (a - b).abs()
    return diff.max().item(), diff.float().mean().item()


def grad_diff_stats(lhs, rhs) -> DiffStats:
    out_max, out_mean = diff_stats(lhs[0], rhs[0])
    dq_max, dq_mean = diff_stats(lhs[1], rhs[1])
    dk_max, dk_mean = diff_stats(lhs[2], rhs[2])
    dv_max, dv_mean = diff_stats(lhs[3], rhs[3])
    return DiffStats(out_max, out_mean, dq_max, dq_mean, dk_max, dk_mean, dv_max, dv_mean)


def check_case(case: VarlenCase):
    ref = run_backward(case, True)
    for rep in range(3):
        cur = run_backward(case, True)
        for a, b in zip(cur, ref):
            assert torch.equal(a, b), f"{case.name}: deterministic diff at rep {rep}: {(a - b).abs().max().item()}"
    det, nondet = run_backward(case, True), run_backward(case, False)
    for name, a, b in zip(("out", "dq", "dk", "dv"), det, nondet):
        assert torch.allclose(a, b, atol=5e-2, rtol=5e-2), f"{case.name} {name} diff: {(a - b).abs().max().item()}"


def speed_specs(max_seqlen: int):
    return [
        (f"{name}_s{s // 1024}k", [s], hq, hkv, d)
        for s in SEQ_LENS
        if s <= max_seqlen
        for name, hq, hkv, d in SPEED_HEAD_CONFIGS
    ]


def bench_case(case: VarlenCase, warmup: int, iters: int) -> BenchResult:
    nondeter = run_backward(case, False)
    deter = run_backward(case, True)
    diff = grad_diff_stats(deter, nondeter)
    nondeter_ms = bench_ms(lambda: run_backward(case, False), warmup=warmup, iters=iters)
    deter_ms = bench_ms(lambda: run_backward(case, True), warmup=warmup, iters=iters)
    return BenchResult(case.name, nondeter_ms, deter_ms, diff)


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Dense/varlen deterministic backward accuracy and speed test.")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--random-cases", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-seqlen", type=int, default=max(SEQ_LENS))
    parser.add_argument("--no-bench", action="store_true")
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    major, minor = torch.cuda.get_device_capability()
    print(f"Using flash_attn from: {fa_interface.__file__}")
    print(f"Device: {torch.cuda.get_device_name()} (SM{major}{minor})")
    acc_specs = CORE + random_cases(args.random_cases, args.seed)
    for spec in acc_specs:
        case = make_case(*spec)
        print(f"\n{case.name}: lens={spec[1]}, H={case.q.shape[1]}/{case.k.shape[1]}, D={case.q.shape[-1]}")
        check_case(case)
        print(f"  correctness: PASS")

    if args.no_bench:
        return

    print()
    print("DETER vs NON-DETER dense fwd+bwd e2e")
    print(
        "case                         non_deter_ms  deter_ms  slowdown  "
        "out(max/mean)      dQ(max/mean)       dK(max/mean)       dV(max/mean)"
    )
    for spec in speed_specs(args.max_seqlen):
        case = make_case(*spec)
        result = bench_case(case, warmup=args.warmup, iters=args.iters)
        slowdown = result.deter_ms / result.nondeter_ms
        d = result.diff
        print(
            f"{result.name:28s} {result.nondeter_ms:11.3f} {result.deter_ms:9.3f} "
            f"{slowdown:8.3f}x  "
            f"{d.out_max:.3e}/{d.out_mean:.3e}  "
            f"{d.dq_max:.3e}/{d.dq_mean:.3e}  "
            f"{d.dk_max:.3e}/{d.dk_mean:.3e}  "
            f"{d.dv_max:.3e}/{d.dv_mean:.3e}"
        )


if __name__ == "__main__":
    main()
