"""Default-heuristic SM100 decode: standard FA4 vs integrated public SwapAB.

Both public-entry backends request num_splits=0. SwapAB is forced, not an auto
fallback. Optional --original-path loads a full saved standalone .py module and
compares its original kernel reduction with PDL enabled (also num_splits=0).
No production module is monkeypatched. Run this harness single-threaded: the
explicit backend selector is process-global and restored after every call.

Each backend is captured ONCE, traversing whole cycles of the same seeded input
pool (>=512 MiB KV and >=2 copies). The same graphs are measured in interleaved
rounds. Reported latency is MIN of round MEDIANS, not a global median. Bandwidth
counts useful Q/K/V/O bytes once, not physical HBM or workspace traffic.

    python benchmarks/benchmark_decode_sm100.py --shapes all --json results.jsonl
    python benchmarks/benchmark_decode_sm100.py --start 0 --end 6 --json batch.jsonl
    python benchmarks/benchmark_decode_sm100.py --original-path /saved/original.py

JSONL is append-only, exclusively locked for the entire run, and flushed/fsynced
per record. Each run has a source/runtime manifest and per-shape raw samples.
--help and importing the shape definitions require neither Torch nor CUDA.
"""

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import uuid

SHAPES = {
    # Colfax "S/P ping-pong for FA4 decode" (Fig. 9): GQA 16:1, 16:2, plus MHA 16:16
    "blog": [
        (1, hq, hkv, s, d)
        for d in (64, 128)
        for (hq, hkv) in ((16, 1), (16, 2), (16, 16))
        for s in (1024, 4096, 16384, 32768, 65536, 131072)
    ],
    # Llama-3-8B (32/8), Llama-3-70B (64/8) style decode at several batch sizes
    "models": [
        (b, hq, 8, s, 128)
        for hq in (32, 64)
        for b in (1, 8, 32, 64)
        for s in (1024, 8192, 32768)
    ],
}
SHAPES["all"] = SHAPES["blog"] + SHAPES["models"]
SELECTOR = "FLASH_ATTENTION_SM100_BACKEND"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", choices=sorted(SHAPES), default="all")
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument(
        "--original-path", type=Path, help="full saved standalone .py module"
    )
    parser.add_argument("--iters", type=int, default=128)
    parser.add_argument("--reps", type=int, default=7)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--l2-mb", type=int, default=512, help="KV pool MiB (minimum 512)"
    )
    parser.add_argument(
        "--start", type=int, default=0, help="inclusive index within --shapes"
    )
    parser.add_argument(
        "--end", type=int, default=None, help="exclusive index within --shapes"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--json", type=Path, help="append-only, exclusive-lock JSONL output"
    )
    args = parser.parse_args(argv)
    if min(args.iters, args.reps, args.rounds) < 1:
        parser.error("--iters, --reps, and --rounds must be positive")
    if args.l2_mb < 512:
        parser.error("--l2-mb must be at least 512")
    end = len(SHAPES[args.shapes]) if args.end is None else args.end
    if not 0 <= args.start < end <= len(SHAPES[args.shapes]):
        parser.error("require 0 <= start < end <= shape count")
    if args.original_path is not None:
        args.original_path = args.original_path.resolve()
        if args.original_path.suffix != ".py" or not args.original_path.is_file():
            parser.error("--original-path must be an existing full .py module")
    return args


@contextmanager
def backend_selector(mode):
    previous = os.environ.get(SELECTOR)
    os.environ[SELECTOR] = mode
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(SELECTOR, None)
        else:
            os.environ[SELECTOR] = previous


def output_tensor(result):
    # Tensor[0] would silently drop the batch axis.
    return result[0] if isinstance(result, tuple) else result


def pool_plan(shape, requested_iters, l2_mb):
    batch, hq, hkv, seqlen, dim = shape
    kv_bytes = 2 * batch * seqlen * hkv * dim * 2
    q_bytes = batch * hq * dim * 2
    copies = max(2, math.ceil(l2_mb * 2**20 / kv_bytes))
    actual_calls = math.ceil(requested_iters / copies) * copies
    return dict(
        input_copies=copies,
        requested_calls=requested_iters,
        actual_calls=actual_calls,
        final_pool_index=(actual_calls - 1) % copies,
        kv_working_set_bytes=copies * kv_bytes,
        input_pool_bytes=copies * (kv_bytes + q_bytes),
        useful_bytes_per_call=kv_bytes + 2 * q_bytes,
    )


def source_manifest(root, original_path):
    # Include every production Python/CUDA/C++ source, not just the changed files.
    suffixes = {".py", ".cu", ".cuh", ".cpp", ".cc", ".c", ".h", ".hpp"}
    paths = {
        p.resolve()
        for base in (root / "flash_attn", root / "csrc")
        for p in base.rglob("*")
        if p.is_file() and p.suffix in suffixes
    }
    paths.add(Path(__file__).resolve())
    if original_path is not None:
        paths.add(original_path)
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def command_output(command, cwd=None):
    try:
        return subprocess.check_output(
            command, cwd=cwd, text=True, stderr=subprocess.STDOUT, timeout=20
        ).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unavailable: {exc}"


def runtime_manifest(torch, root):
    versions = {}
    for package in ("nvidia-cutlass-dsl", "apache-tvm-ffi", "flash-attn-4"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    return dict(
        python=sys.version,
        pytorch=torch.__version__,
        cuda=torch.version.cuda,
        packages=versions,
        gpu=str(props),
        gpu_name=props.name,
        device=device,
        compute_capability=list(torch.cuda.get_device_capability(device)),
        driver=command_output(
            [
                "nvidia-smi",
                "--query-gpu=uuid,name,driver_version",
                "--format=csv,noheader",
            ]
        ),
        git_revision=command_output(["git", "rev-parse", "HEAD"], root),
        git_status=command_output(["git", "status", "--short"], root),
        environment={
            k: v
            for k, v in os.environ.items()
            if k.startswith(("FLASH_ATTENTION_", "CUTE_", "CUDA_"))
        },
    )


@contextmanager
def json_writer(path):
    if path is None:
        yield lambda record: None
        return
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def append(record):
            handle.write(json.dumps(record, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

        yield append


def load_original(path):
    # A unique private module identity; do not replace any flash_attn modules.
    name = "_benchmark_original_decode_" + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load original module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def reference(torch, tensors):
    # Process batches separately: never hold multi-GB FP32 K/V conversions.
    q, k, v = tensors
    batch, _, hq, dim = q.shape
    hkv = k.shape[2]
    result = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    for b in range(batch):
        qg = q[b, 0].float().reshape(hkv, hq // hkv, dim)
        scores = torch.einsum("hgd,shd->hgs", qg, k[b].float()) / math.sqrt(dim)
        probabilities = scores.softmax(-1)
        del scores
        out = torch.einsum("hgs,shd->hgd", probabilities, v[b].float())
        result[b, 0].copy_(out.reshape(hq, dim))
    return result


def check_output(torch, result, ref):
    result = output_tensor(result)
    torch.testing.assert_close(result.float(), ref, atol=2e-2, rtol=2e-2)
    return (result.float() - ref).abs().max().item()


def capture_backend(torch, call, inputs, actual_calls):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            for tensors in inputs:
                call(tensors)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    final_output = None
    with torch.cuda.graph(graph):
        for i in range(actual_calls):
            final_output = output_tensor(call(inputs[i % len(inputs)]))
    graph.replay()
    torch.cuda.synchronize()
    return graph, final_output


def measure_round(torch, graph, actual_calls, reps):
    samples = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / actual_calls)
    return samples


def benchmark_shape(torch, shape, shape_index, args, public_func, config, original):
    batch, hq, hkv, seqlen, dim = shape
    plan = pool_plan(shape, args.iters, args.l2_mb)
    generator = torch.Generator(device="cuda").manual_seed(args.seed + shape_index)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    inputs = [
        tuple(
            torch.randn(size, dtype=dtype, device="cuda", generator=generator)
            for size in (
                (batch, 1, hq, dim),
                (batch, seqlen, hkv, dim),
                (batch, seqlen, hkv, dim),
            )
        )
        for _ in range(plan["input_copies"])
    ]

    def public_call(tensors, mode):
        with backend_selector(mode):
            return public_func(*tensors, num_splits=0, causal=False)

    calls = {
        mode: (lambda tensors, mode=mode: public_call(tensors, mode))
        for mode in ("standard", "swapab")
    }
    native_splits = config(batch, hq, hkv, seqlen, dim)[0]
    details = {
        "standard": dict(
            requested_num_splits=0,
            actual_num_splits=None,
            split_note="standard heuristic not instrumented; not inferred",
        ),
        "swapab": dict(
            requested_num_splits=0,
            heuristic_num_splits=native_splits,
            actual_num_splits=native_splits,
            split_note="get_decode_config; dense tested shapes need no clamp",
        ),
    }
    if original is not None:
        calls["original_pdl"] = lambda tensors: original.flash_attn_decode_func(
            *tensors, num_splits=0, reduction_mode="kernel", use_pdl=True
        )
        original_splits = original.get_decode_config(batch, hq, hkv, seqlen, dim)[0]
        details["original_pdl"] = dict(
            requested_num_splits=0,
            heuristic_num_splits=original_splits,
            actual_num_splits=original_splits,
            reduction_mode="kernel",
            use_pdl=True,
            split_note="original kernel-mode heuristic; no atomic power-of-two clamp",
        )
    ref = reference(torch, inputs[0])
    for name, call in calls.items():
        details[name]["eager_pool0_max_abs_error"] = check_output(
            torch, call(inputs[0]), ref
        )
    del ref
    graphs = {
        name: capture_backend(torch, call, inputs, plan["actual_calls"])
        for name, call in calls.items()
    }
    # Lazy oracle for the actual last graph input; pool0's huge intermediates are gone.
    ref = reference(torch, inputs[plan["final_pool_index"]])
    for name, (graph, output) in graphs.items():
        errors = []
        for _ in range(2):
            output.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            errors.append(check_output(torch, output, ref))
        details[name]["graph_final_max_abs_errors"] = errors
    del ref
    names = list(calls)
    round_orders = []
    for round_index in range(args.rounds):
        offset = round_index % len(names)
        order = names[offset:] + names[:offset]
        round_orders.append(order)
        for name in order:
            samples = measure_round(
                torch, graphs[name][0], plan["actual_calls"], args.reps
            )
            details[name].setdefault("round_samples_us", []).append(samples)
    # Validate after timing as well; the retained tensor is the same captured output.
    ref = reference(torch, inputs[plan["final_pool_index"]])
    for name, (_, output) in graphs.items():
        details[name]["post_timing_max_abs_error"] = check_output(torch, output, ref)
        medians = [
            statistics.median(samples) for samples in details[name]["round_samples_us"]
        ]
        details[name]["round_medians_us"] = medians
        details[name]["latency_us"] = min(medians)
        details[name]["useful_bandwidth_GB_s"] = (
            plan["useful_bytes_per_call"] / min(medians) / 1000
        )
    return dict(
        shape_index=shape_index,
        b=batch,
        hq=hq,
        hkv=hkv,
        s=seqlen,
        d=dim,
        dtype=args.dtype,
        seed=args.seed + shape_index,
        **plan,
        aggregation="min_of_round_medians",
        reps=args.reps,
        rounds=args.rounds,
        round_orders=round_orders,
        backends=details,
    )


def main(argv=None):
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    frozen = source_manifest(root, args.original_path)
    run_id = uuid.uuid4().hex
    shapes = SHAPES[args.shapes][args.start : args.end]
    with json_writer(args.json) as append:
        append(
            dict(
                record_type="source_manifest",
                run_id=run_id,
                sha256=frozen,
                arguments={
                    k: str(v) if isinstance(v, Path) else v
                    for k, v in vars(args).items()
                },
                selected_shapes=shapes,
                selected_shape_count=len(shapes),
            )
        )
        completed = 0
        failure = None
        try:
            import torch
            from flash_attn.cute.decode_config import get_decode_config
            from flash_attn.cute.interface import flash_attn_func

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA required for execution (not for --help)")
            if torch.cuda.get_device_capability()[0] != 10:
                raise RuntimeError("requires an SM100-family GPU")
            runtime = runtime_manifest(torch, root)
            # Avoid recording sources from one tree while executing an installed package.
            interface_path = Path(
                sys.modules[flash_attn_func.__module__].__file__
            ).resolve()
            if interface_path != root / "flash_attn/cute/interface.py":
                raise RuntimeError(
                    f"public API loaded from unexpected source: {interface_path}"
                )
            append(dict(record_type="runtime_manifest", run_id=run_id, **runtime))
            original = load_original(args.original_path) if args.original_path else None
            print(
                f"{len(shapes)} shapes; latency us = MIN of {args.rounds} round medians "
                f"({args.reps} replays each); num_splits=0; streaming KV pool"
            )
            with torch.no_grad():
                for index, shape in enumerate(shapes, args.start):
                    try:
                        row = benchmark_shape(
                            torch,
                            shape,
                            index,
                            args,
                            flash_attn_func,
                            get_decode_config,
                            original,
                        )
                    except Exception as exc:
                        append(
                            dict(
                                record_type="shape_error",
                                run_id=run_id,
                                shape_index=index,
                                shape=shape,
                                error=repr(exc),
                            )
                        )
                        raise
                    append(dict(record_type="shape", run_id=run_id, **row))
                    completed += 1
                    timings = " ".join(
                        f"{name}={data['latency_us']:.3f}"
                        for name, data in row["backends"].items()
                    )
                    print(f"{index:02d} {shape}: {timings}", flush=True)
                    torch.cuda.empty_cache()
        except BaseException as exc:
            failure = repr(exc)
            raise
        finally:
            current = source_manifest(root, args.original_path)
            changed = sorted(
                p
                for p in frozen.keys() | current.keys()
                if frozen.get(p) != current.get(p)
            )
            append(
                dict(
                    record_type="run_end",
                    run_id=run_id,
                    completed_shapes=completed,
                    expected_shapes=len(shapes),
                    sources_unchanged=not changed,
                    changed_sources=changed,
                    error=failure,
                )
            )
            if changed:
                raise RuntimeError(f"source manifest changed during run: {changed}")


if __name__ == "__main__":
    main()
