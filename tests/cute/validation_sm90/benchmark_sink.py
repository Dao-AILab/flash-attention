"""Forward-only, steady-state CUDA-graph benchmark. Not a correctness test."""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
from functools import partial
from importlib import metadata
from pathlib import Path

import torch
from flash_attn.cute import flash_attn_func, softmax


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--return-lse", action="store_true")
    args = parser.parse_args()
    if args.rounds < 3 or args.replays < 1:
        raise ValueError("Use rounds >= 3 and replays >= 1")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("A real SM90 GPU is required")
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    source_hash = hashlib.sha256(Path(softmax.__file__).read_bytes()).hexdigest()
    for sq, sk, dim in [(1, 4096, 64), (128, 128, 64), (1024, 1024, 128), (4096, 4096, 128)]:
        for sink_value in [None, 0.0, 8.0, 100.0]:
            torch.manual_seed(0)
            q = torch.randn(1, sq, 16, dim, device="cuda", dtype=torch.bfloat16)
            k = torch.randn(1, sk, 4, dim, device="cuda", dtype=q.dtype)
            v = torch.randn_like(k)
            sink = None if sink_value is None else torch.full((16,), sink_value, device="cuda")

            call = partial(
                flash_attn_func,
                q,
                k,
                v,
                learnable_sink=sink,
                num_splits=1,
                pack_gqa=True,
                return_lse=args.return_lse,
            )

            # JIT compile outside capture; warm up on a side stream.
            with torch.no_grad():
                warm = torch.cuda.Stream()
                warm.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(warm):
                    for _ in range(10):
                        result = call()
                torch.cuda.current_stream().wait_stream(warm)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                calls_per_graph = 10
                with torch.cuda.graph(graph):
                    for _ in range(calls_per_graph):
                        result = call()
                for _ in range(5):
                    graph.replay()
                torch.cuda.synchronize()
                samples = []
                for _ in range(args.rounds):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(args.replays):
                        graph.replay()
                    end.record()
                    end.synchronize()
                    samples.append(
                        start.elapsed_time(end) * 1000 / (args.replays * calls_per_graph)
                    )
            print(
                json.dumps(
                    {
                        "label": args.label,
                        "sha": sha,
                        "softmax_sha256": source_hash,
                        "pid": os.getpid(),
                        "visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                        "gpu": torch.cuda.get_device_name(),
                        "dsl": metadata.version("nvidia-cutlass-dsl"),
                        "quack": metadata.version("quack-kernels"),
                        "sq": sq,
                        "sk": sk,
                        "dim": dim,
                        "hq": 16,
                        "hkv": 4,
                        "sink": sink_value,
                        "pack_gqa": True,
                        "num_splits": 1,
                        "return_lse": args.return_lse,
                        "measurement": "cuda_graph_forward_us_per_call",
                        "median_us": statistics.median(samples),
                        "samples_us": samples,
                    }
                ),
                flush=True,
            )
            del graph, result, call, q, k, v, sink
            torch.cuda.synchronize()


if __name__ == "__main__":
    main()
