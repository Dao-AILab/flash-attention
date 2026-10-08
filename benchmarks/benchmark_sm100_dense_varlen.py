"""Compare SM100 dense and packed-varlen forward for equal-length sequences.

Examples (run one fresh process per source revision):
    python benchmarks/benchmark_sm100_dense_varlen.py --output main.json
    python benchmarks/benchmark_sm100_dense_varlen.py --source-root ../old-checkout --output old.json
    python benchmarks/benchmark_sm100_dense_varlen.py --seqlen 8192 --include-1cta

CUDA Graphs retain preallocated O/LSE and inputs. Compilation, capture, output
checks and calibrated warmup are outside timing. The JSON includes raw samples,
round medians and both percentage denominators. These are forward-only timings;
output agreement between two kernels is a sanity check, not reference validation.
Persistent FA4 caching is disabled within this process for source/ablation replay.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import statistics
import subprocess
import sys
import types
from contextlib import contextmanager
from pathlib import Path


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--source-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--batch-size", type=positive_int, default=1)
    parser.add_argument("--seqlen", type=positive_int, default=86000)
    parser.add_argument("--nheads", type=positive_int, default=32)
    parser.add_argument(
        "--nheads-kv", type=positive_int, default=None, help="defaults to --nheads"
    )
    parser.add_argument("--head-dim", type=positive_int, default=128)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device index within CUDA_VISIBLE_DEVICES",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=positive_int, default=7)
    parser.add_argument("--samples", type=positive_int, default=7)
    parser.add_argument("--replays", type=positive_int, default=3)
    parser.add_argument("--warmup-ms", type=float, default=200.0)
    parser.add_argument(
        "--include-1cta",
        action="store_true",
        help="also capture a forced forward-only dense 1CTA diagnostic",
    )
    parser.add_argument(
        "--restore-old-budget",
        action="store_true",
        help="restore the pre-#2869 BF16/noncausal/D128 2CTA register table before compilation",
    )
    parser.add_argument(
        "--b25-quack-compat",
        action="store_true",
        help="process-local native CuTe alias for the Quack export used by 4.0.0b25",
    )
    parser.add_argument(
        "--do-bench",
        action="store_true",
        help="also run the issue's allocating eager do_bench(warmup=20, rep=100) method",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("benchmark_sm100_dense_varlen.json")
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        help="save compiled PTX/CUBIN before timing; directory must be empty",
    )
    args = parser.parse_args()
    args.source_root = args.source_root.resolve()
    args.nheads_kv = args.nheads if args.nheads_kv is None else args.nheads_kv
    if not (args.source_root / "flash_attn/cute/interface.py").is_file():
        parser.error("--source-root must contain flash_attn/cute/interface.py")
    if args.nheads % args.nheads_kv:
        parser.error("--nheads must be divisible by --nheads-kv")
    if args.batch_size * args.seqlen >= 2**31:
        parser.error("packed token count must fit int32 cu_seqlens")
    if args.device < 0 or not math.isfinite(args.warmup_ms) or args.warmup_ms <= 0:
        parser.error(
            "--device must be nonnegative and --warmup-ms must be finite and positive"
        )
    if args.restore_old_budget and (
        args.dtype != "bf16" or args.causal or args.head_dim != 128
    ):
        parser.error(
            "--restore-old-budget applies only to BF16 noncausal head dimension 128"
        )
    if (
        args.artifacts_dir is not None
        and args.artifacts_dir.exists()
        and (not args.artifacts_dir.is_dir() or any(args.artifacts_dir.iterdir()))
    ):
        parser.error("--artifacts-dir must be an empty directory or a new path")
    return args


def source_metadata(source_root):
    def git(*args):
        try:
            result = subprocess.run(
                ["git", "-C", str(source_root), *args],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError:
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    # A source archive inside another repository must not inherit that parent's
    # revision. Source hashes remain available for archives without their own Git.
    top_level = git("rev-parse", "--show-toplevel")
    own_checkout = top_level is not None and Path(top_level).resolve() == source_root
    status = git("status", "--porcelain") if own_checkout else None
    return {
        "source_revision": git("rev-parse", "HEAD") if own_checkout else None,
        "source_dirty": None if status is None else bool(status),
        "source_sha256": {
            name: hashlib.sha256(
                (source_root / "flash_attn/cute" / name).read_bytes()
            ).hexdigest()
            for name in ("interface.py", "flash_fwd_sm100.py", "utils.py")
        },
        "benchmark_script_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    }


@contextmanager
def forward_1cta(utils, enabled):
    original = utils._get_disable_2cta_default
    if enabled:
        utils._get_disable_2cta_default = lambda is_fwd=False: (
            True if is_fwd else original(is_fwd=is_fwd)
        )
    try:
        yield
    finally:
        utils._get_disable_2cta_default = original


def measure_graph(torch, graph, args):
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    for _ in range(3):
        graph.replay()
    start.record()
    for _ in range(3):
        graph.replay()
    end.record()
    end.synchronize()
    per_call_ms = max(start.elapsed_time(end) / 3, 0.001)
    warmup_replays = max(3, math.ceil(args.warmup_ms / per_call_ms))
    for _ in range(warmup_replays):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(args.samples):
        start.record()
        for _ in range(args.replays):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / args.replays)
    return {
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "warmup_replays": warmup_replays,
        "calibration_ms_per_call": per_call_ms,
    }


def benchmark(args):
    if any(name.startswith("flash_attn.cute") for name in sys.modules):
        raise RuntimeError("use a fresh Python process per revision/ablation")
    if os.getenv("FLASH_ATTENTION_FAKE_TENSOR") == "1":
        raise RuntimeError("run timing without FLASH_ATTENTION_FAKE_TENSOR=1")
    # Load FA4 directly without importing the checkout's optional FA2 extension.
    # This namespace and the cache override affect only this standalone process.
    inherited_cache = os.getenv("FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED")
    os.environ["FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED"] = "0"
    if args.artifacts_dir is not None:
        args.artifacts_dir.mkdir(parents=True, exist_ok=True)
        compiler_dir = args.artifacts_dir / "compiler"
        compiler_dir.mkdir()
        # Retention must be configured before importing CuTe. Keep its original
        # compiler files alongside the numbered copies, away from the checkout.
        os.environ["CUTE_DSL_KEEP"] = "ptx,cubin"
        os.environ["CUTE_DSL_DUMP_DIR"] = str(compiler_dir.resolve())
    package = types.ModuleType("flash_attn")
    package.__path__ = [str(args.source_root / "flash_attn")]
    sys.modules["flash_attn"] = package
    sys.path.insert(0, str(args.source_root))

    import torch
    from cutlass import cute

    from flash_attn.cute import flash_fwd_sm100, interface, utils

    torch.cuda.set_device(args.device)
    if torch.cuda.get_device_capability()[0] != 10:
        raise RuntimeError("this benchmark targets SM100-family GPUs")
    if args.b25_quack_compat:
        import quack.activation

        # Native primitive used by upstream #2787; no installed package is edited.
        quack.activation.sub_packed_f32x2 = cute.arch.sub_packed_f32x2
    override = None
    if args.restore_old_budget:
        key = (True, False, 128, False)
        override = {
            "key": list(key),
            "before": dict(flash_fwd_sm100._TUNING_CONFIG[key]),
        }
        flash_fwd_sm100._TUNING_CONFIG[key] = dict(
            flash_fwd_sm100._TUNING_CONFIG[key],
            num_regs_softmax=176,
            num_regs_correction=88,
        )
        override["after"] = dict(flash_fwd_sm100._TUNING_CONFIG[key])

    compilations = []
    original_compile = cute.compile
    current_mode = None

    def record_compile(*compile_args, **compile_kwargs):
        result = original_compile(*compile_args, **compile_kwargs)
        owner = compile_args[0]
        fields = (
            "use_2cta_instrs",
            "q_stage",
            "m_block_size",
            "n_block_size",
            "num_regs_softmax",
            "num_regs_correction",
            "num_regs_other",
            "scheduling_mode",
            "use_clc_scheduler",
        )
        metadata = {"mode": current_mode, "kernel": type(owner).__name__}
        for name in fields:
            value = getattr(owner, name, None)
            metadata[name] = (
                value if isinstance(value, (int, bool, str, type(None))) else str(value)
            )
        scheduler = getattr(
            owner, "tile_scheduler_cls", getattr(owner, "TileScheduler", None)
        )
        metadata["scheduler"] = getattr(scheduler, "__name__", str(scheduler))
        for suffix, attr in (("ptx", "__ptx__"), ("cubin", "__cubin__")):
            # Native artifacts contain PTX text/CUBIN bytes. Legacy __ptx__ and
            # __cubin__ attributes can instead be paths into the compiler dump.
            value = getattr(getattr(result, "artifacts", None), suffix.upper(), None)
            if value is None:
                value = getattr(result, attr, None)
            if value is None and args.artifacts_dir is not None:
                raise RuntimeError(
                    f"compiler did not expose its {suffix.upper()} artifact"
                )
            if value is not None:
                if isinstance(value, (bytes, bytearray, memoryview)):
                    data = bytes(value)
                elif isinstance(value, str) and "\n" in value:
                    data = value.encode()
                else:
                    data = Path(value).read_bytes()
                if suffix == "cubin" and not data.startswith(b"\x7fELF"):
                    raise RuntimeError("compiler artifact is not an ELF CUBIN")
                if suffix == "ptx" and not (b".version" in data and b".target" in data):
                    raise RuntimeError("compiler artifact is not PTX text")
                metadata[suffix + "_sha256"] = hashlib.sha256(data).hexdigest()
                if args.artifacts_dir is not None:
                    filename = f"{len(compilations):02d}_{current_mode}_{type(owner).__name__}.{suffix}"
                    (args.artifacts_dir / filename).write_bytes(data)
                    metadata[suffix + "_artifact"] = filename
                if suffix == "ptx":
                    metadata["ptx_header"] = data.decode().splitlines()[:11]
        compilations.append(metadata)
        return result

    modes = ["dense_default", "varlen_default"] + (
        ["dense_1cta"] if args.include_1cta else []
    )
    torch.manual_seed(args.seed)
    b, s, h, hkv, d = (
        args.batch_size,
        args.seqlen,
        args.nheads,
        args.nheads_kv,
        args.head_dim,
    )
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    q = torch.randn(b, s, h, d, device="cuda", dtype=dtype)
    k = torch.randn(b, s, hkv, d, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    cu = torch.arange(b + 1, dtype=torch.int32, device="cuda") * s
    graphs, outputs, allocating_calls = {}, {}, {}
    cute.compile = record_compile
    try:
        for mode in modes:
            current_mode = mode
            varlen = mode == "varlen_default"
            tensors = (
                (q.flatten(0, 1), k.flatten(0, 1), v.flatten(0, 1))
                if varlen
                else (q, k, v)
            )
            out = torch.empty_like(tensors[0])
            lse = torch.empty(
                (h, b * s) if varlen else (b, h, s), device="cuda", dtype=torch.float32
            )
            kwargs = {"causal": args.causal, "return_lse": True, "num_splits": 1}
            if varlen:
                kwargs.update(
                    cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=s, max_seqlen_k=s
                )

            def fwd(tensors=tensors, kwargs=kwargs, out=out, lse=lse):
                return interface._flash_attn_fwd(*tensors, out=out, lse=lse, **kwargs)

            def allocating_fwd(tensors=tensors, kwargs=kwargs):
                return interface._flash_attn_fwd(*tensors, **kwargs)

            with forward_1cta(utils, mode == "dense_1cta"):
                for _ in range(3):
                    fwd()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    fwd()
            graphs[mode], outputs[mode], allocating_calls[mode] = (
                graph,
                (out, lse),
                allocating_fwd,
            )
        if args.restore_old_budget and not any(
            item["mode"] == "dense_default"
            and item["use_2cta_instrs"]
            and item["num_regs_softmax"] == 176
            for item in compilations
        ):
            raise RuntimeError(
                "old-budget ablation did not select its intended dense 2CTA kernel"
            )

        ref_out, ref_lse = outputs["varlen_default"]
        ref_lse = ref_lse.reshape(h, b, s).permute(1, 0, 2)
        checks = {}
        for mode, (out, lse) in outputs.items():
            dense_lse = lse if mode != "varlen_default" else ref_lse
            if not bool(out.isfinite().all() and lse.isfinite().all()):
                raise RuntimeError(f"nonfinite output in {mode}")
            torch.testing.assert_close(
                out.reshape_as(ref_out),
                ref_out,
                rtol=2 * torch.finfo(dtype).eps,
                atol=2 * torch.finfo(dtype).eps,
            )
            torch.testing.assert_close(dense_lse, ref_lse, rtol=1e-5, atol=1e-5)
            checks[mode] = {
                "out_max_abs_vs_varlen": (
                    out.reshape_as(ref_out).float() - ref_out.float()
                )
                .abs()
                .max()
                .item(),
                "lse_max_abs_vs_varlen": (dense_lse - ref_lse).abs().max().item(),
            }
        rounds = []
        for index in range(args.rounds):
            order = list(reversed(modes)) if index % 2 else modes[:]
            measurements = {
                mode: measure_graph(torch, graphs[mode], args) for mode in order
            }
            rounds.append(
                {"round": index, "order": order, "measurements": measurements}
            )
        medians = {
            mode: statistics.median(
                row["measurements"][mode]["median_ms"] for row in rounds
            )
            for mode in modes
        }
        dense, varlen = medians["dense_default"], medians["varlen_default"]
        percentages = {
            "varlen_latency_reduction_percent": 100 * (1 - varlen / dense),
            "dense_slower_than_varlen_percent": 100 * (dense / varlen - 1),
        }
        formulas = {
            "varlen_latency_reduction_percent": "100 * (1 - varlen_ms / dense_ms)",
            "dense_slower_than_varlen_percent": "100 * (dense_ms / varlen_ms - 1)",
        }
        if args.include_1cta:
            percentages["forced_1cta_latency_change_percent"] = 100 * (
                medians["dense_1cta"] / dense - 1
            )
            formulas["forced_1cta_latency_change_percent"] = (
                "100 * (dense_1cta_ms / dense_ms - 1)"
            )
        eager = None
        if args.do_bench:
            from triton.testing import do_bench

            eager = {}
            for mode in modes:
                current_mode = mode
                with forward_1cta(utils, mode == "dense_1cta"):
                    eager[mode] = do_bench(allocating_calls[mode], warmup=20, rep=100)
    finally:
        cute.compile = original_compile

    versions = {}
    for name in ("nvidia-cutlass-dsl", "apache-tvm-ffi", "quack-kernels", "triton"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        **source_metadata(args.source_root),
        "configuration": {
            "batch_size": b,
            "seqlen": s,
            "nheads": h,
            "nheads_kv": hkv,
            "head_dim": d,
            "dtype": args.dtype,
            "causal": args.causal,
            "return_lse": True,
            "num_splits": 1,
            "seed": args.seed,
        },
        "environment": {
            "gpu": torch.cuda.get_device_name(),
            "capability": torch.cuda.get_device_capability(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "packages": versions,
            "persistent_cache_enabled": False,
            "inherited_cache_setting": inherited_cache,
        },
        "diagnostics": {
            "restore_old_budget": override,
            "b25_quack_compat": args.b25_quack_compat,
            "artifacts_saved": args.artifacts_dir is not None,
        },
        "method": {
            "rounds": args.rounds,
            "samples_per_round": args.samples,
            "replays_per_sample": args.replays,
            "warmup_target_ms": args.warmup_ms,
            "forward_calls_per_graph": 1,
            "allocation_in_timing": False,
            "summary_statistic": "median of per-round sample medians",
        },
        "compile_metadata": compilations,
        "checks_vs_varlen": checks,
        "rounds": rounds,
        "median_ms": medians,
        "percentages": percentages,
        "percentage_formulas": formulas,
        "eager_do_bench": None
        if eager is None
        else {"warmup_ms": 20, "rep_ms": 100, "latency_ms": eager},
    }


def main():
    args = parse_args()
    result = benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for mode, median in result["median_ms"].items():
        print(f"{mode}: {median:.6f} ms")
    for name, value in result["percentages"].items():
        print(f"{name}: {value:+.2f}% ({result['percentage_formulas'][name]})")
    print(f"Raw samples and metadata: {args.output}")


if __name__ == "__main__":
    main()
