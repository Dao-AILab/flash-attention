"""Benchmark the dynamic-persistent varlen scheduler against the prior default
(`SingleTileVarlenScheduler`), CLC (if available), and — on constant-seqlen workloads — the
non-varlen `flash_attn_func` baseline.

Examples:
  python benchmarks/benchmark_varlen_sched.py --total-tokens 32k --patterns longtail
  python benchmarks/benchmark_varlen_sched.py --total-tokens 32k,64k --shapes 32x1k,16x2k \\
      --patterns constant longtail --csv > out.csv
"""

import argparse
import time
from itertools import product

import torch
from triton.testing import do_bench

from flash_attn.cute import utils as fa_utils
from flash_attn.cute.interface import (
    flash_attn_func,
    flash_attn_varlen_func,
    get_scheduler_metadata,
)


_CLC_MODES = {"clc", "clc-prep"}


def _supports_clc(device):
    return torch.cuda.get_device_capability(device)[0] == 10


def parse_int_k(s):
    """Parse an integer with optional k/K/m/M suffix, e.g. '8k' -> 8192, '1m' -> 1048576."""
    s = str(s).strip().lower()
    if s.endswith("m"):
        return int(s[:-1]) * 1024 * 1024
    if s.endswith("k"):
        return int(s[:-1]) * 1024
    return int(s)


def csv_ints(s):
    return [parse_int_k(x) for x in s.split(",")]


def parse_shape(s):
    """Parse '<batch>x<seqlen>' (seqlen accepts k suffix). Returns (batch, seqlen)."""
    b, sl = s.lower().split("x")
    return int(b), parse_int_k(sl)


def parse_shapes(s):
    return [parse_shape(x) for x in s.split(",")]


def _make_seqlens(batch, seqlen, pattern, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    if pattern == "constant":
        return [seqlen] * batch
    if pattern == "uniform":
        lo = max(1, seqlen // 2)
        return torch.randint(lo, seqlen + 1, (batch,), generator=g).tolist()
    if pattern == "wide":
        return torch.randint(1, seqlen + 1, (batch,), generator=g).tolist()
    if pattern == "longtail":
        n_long = max(1, batch // 8)
        out = torch.randint(
            max(1, seqlen // 16), max(2, seqlen // 8), (batch,), generator=g
        ).tolist()
        for i in torch.randperm(batch, generator=g)[:n_long].tolist():
            out[i] = seqlen
        return out
    if pattern == "bimodal":
        return [seqlen if i % 2 == 0 else max(1, seqlen // 8) for i in range(batch)]
    if pattern == "skew":
        return [max(1, int(seqlen * i / max(1, batch - 1))) for i in range(batch)]
    if pattern == "skew_shuffled":
        out = [max(1, int(seqlen * i / max(1, batch - 1))) for i in range(batch)]
        return [out[i] for i in torch.randperm(batch, generator=g).tolist()]
    raise ValueError(f"unknown pattern {pattern!r}")


def _causal_tiles(sq, sk, tile_m=128, tile_n=128):
    if sq <= 0 or sk <= 0:
        return 0
    nq = (sq + tile_m - 1) // tile_m
    nk = (sk + tile_n - 1) // tile_n
    if nq <= 1:
        return nk
    return nq * nk - (nq * (nq - 1)) // 2


def _apply_sort(seqlens_q, seqlens_k, sort):
    if sort == "none":
        return seqlens_q, seqlens_k
    pairs = list(zip(seqlens_q, seqlens_k))
    keyfn = {
        "asc": lambda p: _causal_tiles(*p),
        "desc": lambda p: -_causal_tiles(*p),
    }.get(sort)
    if keyfn is None:
        raise ValueError(f"unknown sort {sort!r}")
    pairs.sort(key=keyfn)
    return [p[0] for p in pairs], [p[1] for p in pairs]


def _override_random_subset(
    seqlens_q, seqlens_k, frac, seed_salt, sq_value, sk_value, seed
):
    """Pick `frac` of batches at random and overwrite their seqlens to the given values.
    `sk_value=None` leaves seqlens_k untouched (used for decode-mix)."""
    if frac <= 0:
        return seqlens_q, seqlens_k
    g = torch.Generator(device="cpu").manual_seed(seed + seed_salt)
    B = len(seqlens_q)
    n = int(round(frac * B))
    if n <= 0:
        return seqlens_q, seqlens_k
    idx = torch.randperm(B, generator=g)[:n].tolist()
    sq, sk = list(seqlens_q), list(seqlens_k)
    for i in idx:
        sq[i] = sq_value
        if sk_value is not None:
            sk[i] = sk_value
    return sq, sk


def build_ctx(
    args, batch, seqlen, pattern, sort, decode_frac, zero_frac, num_splits, seed
):
    seqlens_k = _make_seqlens(batch, seqlen, pattern, seed)
    seqlens_q = list(seqlens_k)
    seqlens_q, seqlens_k = _override_random_subset(
        seqlens_q, seqlens_k, decode_frac, 7919, sq_value=1, sk_value=None, seed=seed
    )
    seqlens_q, seqlens_k = _override_random_subset(
        seqlens_q, seqlens_k, zero_frac, 31337, sq_value=0, sk_value=0, seed=seed
    )
    seqlens_q, seqlens_k = _apply_sort(seqlens_q, seqlens_k, sort)

    dtype, device = torch.bfloat16, "cuda"
    nheads, nheads_kv, headdim = args.nheads, args.nheads_kv, args.headdim

    cu_q = torch.zeros(batch + 1, dtype=torch.int32, device=device)
    cu_q[1:] = torch.tensor(seqlens_q, dtype=torch.int32, device=device).cumsum(0)
    cu_k = torch.zeros(batch + 1, dtype=torch.int32, device=device)
    cu_k[1:] = torch.tensor(seqlens_k, dtype=torch.int32, device=device).cumsum(0)
    q_unpad = torch.randn(
        max(sum(seqlens_q), 1), nheads, headdim, device=device, dtype=dtype
    )
    k_unpad = torch.randn(
        max(sum(seqlens_k), 1), nheads_kv, headdim, device=device, dtype=dtype
    )
    v_unpad = torch.randn(
        max(sum(seqlens_k), 1), nheads_kv, headdim, device=device, dtype=dtype
    )

    return dict(
        batch=batch,
        seqlen=seqlen,
        pattern=pattern,
        decode_frac=decode_frac,
        zero_frac=zero_frac,
        nheads=nheads,
        nheads_kv=nheads_kv,
        headdim=headdim,
        seqlens_q=seqlens_q,
        seqlens_k=seqlens_k,
        q_unpad=q_unpad,
        k_unpad=k_unpad,
        v_unpad=v_unpad,
        cu_q=cu_q,
        cu_k=cu_k,
        max_seqlen_q=max(seqlens_q) if seqlens_q else 0,
        max_seqlen_k=max(seqlens_k) if seqlens_k else 0,
        causal=True,
        num_splits=num_splits,
        pack_gqa=args.pack_gqa,
    )


def _make_meta(ctx):
    tile_m = 128
    qhead_per_kvhead = ctx["nheads"] // ctx["nheads_kv"]
    arch = torch.cuda.get_device_capability()[0]
    if arch == 10 and ctx["max_seqlen_q"] * qhead_per_kvhead > tile_m:
        q_stage = 2
    else:
        q_stage = 1
    return get_scheduler_metadata(
        num_batch=ctx["batch"],
        max_seqlen_q=ctx["max_seqlen_q"],
        max_seqlen_k=ctx["max_seqlen_k"],
        nheads=ctx["nheads"],
        nheads_kv=ctx["nheads_kv"],
        headdim=ctx["headdim"],
        num_splits=ctx["num_splits"],
        tile_m=tile_m,
        tile_n=128,
        causal=ctx["causal"],
        pack_gqa=ctx["pack_gqa"],
        cu_seqlens_q=ctx["cu_q"],
        cu_seqlens_k=ctx["cu_k"],
        q_stage=q_stage,
    )


def _make_meta_no_semaphore(ctx):
    """Like _make_meta but with tile_count_semaphore nulled out, so the FA kernel
    selects SingleTileVarlenScheduler (STATIC) instead of DynamicPersistentVarlen.
    Exercises the binary-search hint path on the scheduler that lacks resumption."""
    m = _make_meta(ctx)
    return m._replace(tile_count_semaphore=None)


def setup_dense(ctx):
    """Non-varlen baseline; only meaningful when every batch has the same seqlen."""
    if ctx["pattern"] != "constant" or ctx["decode_frac"] != 0 or ctx["zero_frac"] != 0:
        return None
    batch, seqlen = ctx["batch"], ctx["seqlen"]
    nheads, nheads_kv, headdim = ctx["nheads"], ctx["nheads_kv"], ctx["headdim"]
    dtype, device = torch.bfloat16, "cuda"
    q = torch.randn(batch, seqlen, nheads, headdim, device=device, dtype=dtype)
    k = torch.randn(batch, seqlen, nheads_kv, headdim, device=device, dtype=dtype)
    v = torch.randn(batch, seqlen, nheads_kv, headdim, device=device, dtype=dtype)
    return lambda: flash_attn_func(
        q, k, v, causal=ctx["causal"], num_splits=ctx["num_splits"]
    )


def make_varlen_setup(*, clc: bool, prep: str, no_semaphore: bool = False):
    """`prep` is one of 'none', 'precompute', 'recompute'.

    `no_semaphore=True` nulls out `tile_count_semaphore` in the metadata so the
    FA kernel picks SingleTileVarlenScheduler (STATIC) instead of the auto-
    selected DynamicPersistentVarlenScheduler. Use this to exercise the binary-
    search hint path on the no-resumption scheduler that PR #2520 targets."""
    assert prep in ("none", "precompute", "recompute")
    meta_fn = _make_meta_no_semaphore if no_semaphore else _make_meta

    def setup(ctx):
        meta_precomputed = meta_fn(ctx) if prep == "precompute" else None

        def fn():
            fa_utils._fa_clc_enabled = clc
            meta = meta_fn(ctx) if prep == "recompute" else meta_precomputed
            return flash_attn_varlen_func(
                ctx["q_unpad"],
                ctx["k_unpad"],
                ctx["v_unpad"],
                cu_seqlens_q=ctx["cu_q"],
                cu_seqlens_k=ctx["cu_k"],
                max_seqlen_q=ctx["max_seqlen_q"],
                max_seqlen_k=ctx["max_seqlen_k"],
                causal=ctx["causal"],
                num_splits=ctx["num_splits"],
                scheduler_metadata=meta,
                disable_scheduler_metadata=(prep == "none"),
                pack_gqa=ctx["pack_gqa"],
            )

        return fn

    return setup


# fmt: off
MODES = [
    ("dense",        setup_dense),
    ("single-tile",  make_varlen_setup(clc=False, prep="none")),
    ("st-prep",      make_varlen_setup(clc=False, prep="precompute", no_semaphore=True)),
    ("clc",          make_varlen_setup(clc=True,  prep="none")),
    ("clc-prep",     make_varlen_setup(clc=True,  prep="precompute")),
    ("dynamic-prep", make_varlen_setup(clc=False, prep="precompute")),
    ("dynamic+prep", make_varlen_setup(clc=False, prep="recompute")),
]
# fmt: on


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark FA4 varlen scheduler modes")
    p.add_argument(
        "--total-tokens",
        type=csv_ints,
        default=[32 * 1024],
        help="Total tokens (batch*seqlen) per workload, comma-separated. e.g. 32k,64k",
    )
    p.add_argument(
        "--shapes",
        type=parse_shapes,
        default=None,
        help="Explicit (batch x seqlen) pairs, comma-separated, e.g. 32x1k,16x2k. "
        "If unset, derived from --total-tokens by sweeping a default isoline.",
    )
    p.add_argument(
        "--patterns",
        nargs="+",
        default=["constant", "longtail", "bimodal", "uniform"],
        help="Length distributions: constant, uniform, wide, longtail, bimodal, skew, skew_shuffled",
    )
    p.add_argument(
        "--sorts",
        nargs="+",
        default=["none"],
        help="Batch ordering by tile count: none, asc, desc",
    )
    p.add_argument(
        "--decode-fracs",
        nargs="+",
        type=float,
        default=[0.0],
        help="Fraction(s) of batches to force seqlen_q=1 (mixed prefill/decode)",
    )
    p.add_argument(
        "--zero-fracs",
        nargs="+",
        type=float,
        default=[0.0],
        help="Fraction(s) of batches to force seqlen=0",
    )
    p.add_argument(
        "--num-splits",
        nargs="+",
        type=int,
        default=[1],
        help="num_splits values; >1 enables SplitKV",
    )
    p.add_argument("--modes", nargs="+", default=[cli for cli, _ in MODES])
    p.add_argument("--headdim", type=int, default=128)
    p.add_argument("--nheads", type=int, default=16)
    p.add_argument("--nheads-kv", type=int, default=2)
    p.add_argument(
        "--pack-gqa",
        action="store_true",
        default=True,
        help="Force pack_gqa=True (default). --no-pack-gqa to disable.",
    )
    p.add_argument("--no-pack-gqa", dest="pack_gqa", action="store_false")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--rep", type=int, default=20)
    p.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="Sleep between modes to dodge clock throttling (seconds)",
    )
    p.add_argument("--device", type=int, default=0)
    p.add_argument(
        "--csv", action="store_true", help="Emit CSV rows instead of the pretty table"
    )
    return p.parse_args()


def _default_isoline(total_tokens):
    """(batch, seqlen) pairs where batch * seqlen == total_tokens, doubling seqlen from 256."""
    return [
        (total_tokens // s, s)
        for s in (1 << b for b in range(8, total_tokens.bit_length()))
        if total_tokens // s >= 1
    ]


def _format_row(cells, csv, widths):
    if csv:
        return ",".join(str(c) for c in cells)
    return "  ".join(f"{str(c):<{w}}" for c, w in zip(cells, widths))


def main():
    args = parse_args()
    torch.cuda.set_device(args.device)
    torch.manual_seed(0)

    if args.shapes is not None:
        shapes = args.shapes
    else:
        shapes = [s for t in args.total_tokens for s in _default_isoline(t)]

    selected_modes = [(cli, fn) for cli, fn in MODES if cli in args.modes]
    if not _supports_clc(args.device):
        dropped = [cli for cli, _ in selected_modes if cli in _CLC_MODES]
        if dropped:
            print(f"# skipping CLC modes: {', '.join(dropped)}")
        selected_modes = [
            (cli, fn) for cli, fn in selected_modes if cli not in _CLC_MODES
        ]

    print(f"# device {args.device}: {torch.cuda.get_device_name(args.device)}")
    print(
        f"# headdim={args.headdim} nheads={args.nheads} nheads_kv={args.nheads_kv} "
        f"(qhead_per_kvhead={args.nheads // args.nheads_kv})"
    )
    cols = [
        ("pattern", 14),
        ("decode", 8),
        ("zero", 6),
        ("shape", 10),
        ("splits", 8),
        ("mode", 14),
        ("mean_us", 10),
        ("tok/us", 9),
        ("tflops", 8),
        ("rel_st", 7),
        ("rel_clc", 9),
    ]
    widths = [w for _, w in cols]
    print(_format_row([h for h, _ in cols], args.csv, widths))

    for shape, pattern, sort, decode_frac, zero_frac, num_splits in product(
        shapes,
        args.patterns,
        args.sorts,
        args.decode_fracs,
        args.zero_fracs,
        args.num_splits,
    ):
        batch, seqlen = shape
        results = {}
        # Workload is identical across modes; build once to get total_q for the report.
        ref_ctx = build_ctx(
            args,
            batch,
            seqlen,
            pattern,
            sort,
            decode_frac,
            zero_frac,
            num_splits,
            seed=0,
        )
        total_q = sum(ref_ctx["seqlens_q"])
        # Causal varlen attention FLOPs per batch:
        #   per (head, query q in [0, sq)): 4 * d * effective_k where
        #   effective_k = max(0, sk - sq + q + 1).
        #   sum_q effective_k = sq*sk - sq*(sq-1)/2  (for sk >= sq; otherwise clamped).
        total_flops = 0
        for sq, sk in zip(ref_ctx["seqlens_q"], ref_ctx["seqlens_k"]):
            if sq == 0 or sk == 0:
                continue
            if ref_ctx["causal"]:
                # sum_{q=0}^{sq-1} max(0, sk - sq + q + 1)
                shift = sk - sq
                if shift >= 0:
                    eff = sq * sk - sq * (sq - 1) // 2
                else:
                    # clamp to non-negative for queries near 0
                    first_visible_q = (
                        -shift
                    )  # smallest q with sk - sq + q + 1 > 0 is q = sq - sk
                    visible = sq - first_visible_q
                    eff = visible * sk - visible * (visible - 1) // 2
                eff = max(0, eff)
            else:
                eff = sq * sk
            total_flops += 4 * args.headdim * args.nheads * eff

        for cli, setup in selected_modes:
            samples = []
            for s in range(args.seeds):
                ctx = build_ctx(
                    args,
                    batch,
                    seqlen,
                    pattern,
                    sort,
                    decode_frac,
                    zero_frac,
                    num_splits,
                    seed=s,
                )
                fn = setup(ctx)
                if fn is None:
                    samples = None
                    break
                fn()
                torch.cuda.synchronize()
                time.sleep(args.sleep)
                samples.append(do_bench(fn, warmup=args.warmup, rep=args.rep))
            results[cli] = (
                None if samples is None else sum(samples) / len(samples) * 1e3
            )

        single_tile_us = results.get("single-tile")
        clc_us = results.get("clc")
        for cli, _ in selected_modes:
            us = results.get(cli)
            if us is None:
                continue
            tok_per_us = (total_q / us) if us > 0 else 0.0
            tflops = (total_flops / (us * 1e6)) if us > 0 else 0.0
            rel_st = f"{single_tile_us / us:.3f}" if single_tile_us else "-"
            rel_cl = f"{clc_us / us:.3f}" if clc_us else "-"
            print(
                _format_row(
                    [
                        pattern,
                        f"{decode_frac:.2f}",
                        f"{zero_frac:.2f}",
                        f"{batch}x{seqlen}",
                        num_splits,
                        cli,
                        f"{us:.2f}",
                        f"{tok_per_us:.2f}",
                        f"{tflops:.2f}",
                        rel_st,
                        rel_cl,
                    ],
                    args.csv,
                    widths,
                )
            )


if __name__ == "__main__":
    main()
