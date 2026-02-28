#!/usr/bin/env python3
"""vLLM Serving Benchmark for V100: FLASH_ATTN vs XFORMERS

Starts a vLLM OpenAI-compatible server with each attention backend,
sends concurrent streaming requests, and measures TTFT, TPOT, throughput.

Note: TORCH_SDPA is CPU-only in vLLM v0.6.5, so only FLASH_ATTN and XFORMERS
are supported on V100 GPU.

Usage:
    python benchmark_serving.py                                   # full benchmark
    python benchmark_serving.py --dry-run                         # quick sanity check
    python benchmark_serving.py --backends FLASH_ATTN XFORMERS   # subset of backends
    python benchmark_serving.py --workloads short medium          # subset of workloads
"""

import argparse
import asyncio
import json
import os
import signal
import subprocess
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import aiohttp
from tabulate import tabulate

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "NousResearch/Llama-2-7b-hf"
DEFAULT_PORT = 8000
SERVER_STARTUP_TIMEOUT = 600  # seconds (model download can be slow)

ALL_BACKENDS = ["FLASH_ATTN", "XFORMERS"]

WORKLOADS = {
    "short":     {"input_len": 128,  "max_output_len": 128},
    "medium":    {"input_len": 512,  "max_output_len": 256},
    "long":      {"input_len": 1024, "max_output_len": 512},
    "very_long": {"input_len": 2048, "max_output_len": 256},
}

CONCURRENCY_LEVELS = [1, 2, 4, 8, 16]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RequestResult:
    success: bool
    ttft: Optional[float] = None        # Time to first token (seconds)
    total_time: Optional[float] = None   # Total request time (seconds)
    output_tokens: int = 0               # Generated tokens
    error: Optional[str] = None


@dataclass
class WorkloadResult:
    backend: str
    workload: str
    concurrency: int
    wall_time: float = 0.0              # Wall clock time for entire batch
    results: List[RequestResult] = field(default_factory=list)

    @property
    def successful(self) -> List[RequestResult]:
        return [r for r in self.results if r.success]

    @property
    def num_success(self) -> int:
        return len(self.successful)

    @property
    def num_total(self) -> int:
        return len(self.results)

    def _ttft_values(self) -> List[float]:
        return [r.ttft for r in self.successful if r.ttft is not None]

    def _tpot_values(self) -> List[float]:
        """TPOT per request = (total_time - ttft) / (output_tokens - 1)."""
        vals = []
        for r in self.successful:
            if r.ttft is not None and r.output_tokens > 1:
                vals.append((r.total_time - r.ttft) / (r.output_tokens - 1))
        return vals

    @property
    def avg_ttft(self) -> Optional[float]:
        v = self._ttft_values()
        return statistics.mean(v) if v else None

    @property
    def p99_ttft(self) -> Optional[float]:
        v = sorted(self._ttft_values())
        if not v:
            return None
        idx = min(int(len(v) * 0.99), len(v) - 1)
        return v[idx]

    @property
    def avg_tpot(self) -> Optional[float]:
        v = self._tpot_values()
        return statistics.mean(v) if v else None

    @property
    def throughput_tok_per_sec(self) -> Optional[float]:
        if self.wall_time <= 0:
            return None
        total_tokens = sum(r.output_tokens for r in self.successful)
        return total_tokens / self.wall_time if total_tokens > 0 else None

    @property
    def requests_per_sec(self) -> Optional[float]:
        if self.wall_time <= 0 or not self.successful:
            return None
        return len(self.successful) / self.wall_time


# ---------------------------------------------------------------------------
# Server management
# ---------------------------------------------------------------------------

class VLLMServer:
    def __init__(
        self,
        model: str,
        backend: str,
        port: int = DEFAULT_PORT,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.90,
    ):
        self.model = model
        self.backend = backend
        self.port = port
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.process: Optional[subprocess.Popen] = None
        self.base_url = f"http://localhost:{port}"
        self._log_file = None

    def start(self):
        env = os.environ.copy()
        env["VLLM_ATTENTION_BACKEND"] = self.backend

        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model,
            "--port", str(self.port),
            "--max-model-len", str(self.max_model_len),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--dtype", "half",
            "--trust-remote-code",
            "--disable-frontend-multiprocessing",
            "--enforce-eager",
        ]
        # FLASH_ATTN on SM70 requires page block_size=256 for KV cache
        if self.backend == "FLASH_ATTN":
            cmd.extend(["--block-size", "256"])

        print(f"  Starting vLLM server with backend={self.backend} ...")
        print(f"  Command: {' '.join(cmd)}")

        # Redirect stdout/stderr to a temp file to avoid pipe deadlock.
        # (vLLM writes a lot of output; a 64KB pipe buffer fills up quickly.)
        self._log_file = tempfile.NamedTemporaryFile(
            mode="w", prefix=f"vllm_{self.backend}_", suffix=".log", delete=False,
        )
        print(f"  Server log: {self._log_file.name}")

        self.process = subprocess.Popen(
            cmd,
            env=env,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
        )

    def _read_log_tail(self, n_chars: int = 3000) -> str:
        """Read the last n_chars from the server log file."""
        if self._log_file is None:
            return ""
        try:
            self._log_file.flush()
            with open(self._log_file.name, "r") as f:
                f.seek(0, 2)  # end
                size = f.tell()
                f.seek(max(0, size - n_chars))
                return f.read()
        except Exception:
            return ""

    async def wait_healthy(self, timeout: int = SERVER_STARTUP_TIMEOUT) -> bool:
        """Poll /health until the server is ready or timeout."""
        start = time.time()
        url = f"{self.base_url}/health"

        while time.time() - start < timeout:
            # Check if process died
            if self.process and self.process.poll() is not None:
                log_tail = self._read_log_tail()
                print(f"  Server exited with code {self.process.returncode}")
                if log_tail:
                    print(f"  Last output:\n{log_tail}")
                return False

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        url, timeout=aiohttp.ClientTimeout(total=5)
                    ) as resp:
                        if resp.status == 200:
                            elapsed = time.time() - start
                            print(f"  Server healthy after {elapsed:.1f}s")
                            return True
            except (aiohttp.ClientError, asyncio.TimeoutError, ConnectionError):
                pass

            await asyncio.sleep(3)

        print(f"  Server startup timed out after {timeout}s")
        log_tail = self._read_log_tail()
        if log_tail:
            print(f"  Last output:\n{log_tail}")
        return False

    def stop(self):
        if self.process is None:
            return

        print(f"  Stopping vLLM server (pid={self.process.pid}) ...")

        self.process.send_signal(signal.SIGTERM)
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            print("  SIGTERM timed out, sending SIGKILL ...")
            self.process.kill()
            self.process.wait(timeout=10)

        self.process = None

        # Clean up log file
        if self._log_file is not None:
            try:
                self._log_file.close()
                os.unlink(self._log_file.name)
            except Exception:
                pass
            self._log_file = None

        # Allow GPU memory to be fully released
        time.sleep(5)


# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------

def generate_prompt(input_len: int) -> str:
    """Generate a prompt that approximates `input_len` tokens.

    Heuristic: one repetition of the base sentence is ~10 tokens.
    """
    base = "The quick brown fox jumps over the lazy dog. "
    repetitions = max(1, input_len // 10)
    prompt = base * repetitions
    # Rough trim (~4 chars/token)
    target_chars = input_len * 4
    if len(prompt) > target_chars:
        prompt = prompt[:target_chars]
    return prompt


# ---------------------------------------------------------------------------
# Benchmark client
# ---------------------------------------------------------------------------

async def send_streaming_request(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
) -> RequestResult:
    """Send a single streaming /v1/completions request and measure timing."""
    url = f"{base_url}/v1/completions"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.0,
    }

    start_time = time.perf_counter()
    ttft = None
    output_tokens = 0

    try:
        async with session.post(
            url, json=payload, timeout=aiohttp.ClientTimeout(total=600),
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                return RequestResult(
                    success=False,
                    total_time=time.perf_counter() - start_time,
                    error=f"HTTP {resp.status}: {text[:300]}",
                )

            async for raw_line in resp.content:
                line = raw_line.decode("utf-8").strip()
                if not line or not line.startswith("data: "):
                    continue

                data_str = line[len("data: "):]
                if data_str == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                choices = data.get("choices", [])
                if choices and choices[0].get("text", ""):
                    if ttft is None:
                        ttft = time.perf_counter() - start_time
                    output_tokens += 1

        total_time = time.perf_counter() - start_time
        return RequestResult(
            success=True,
            ttft=ttft if ttft is not None else total_time,
            total_time=total_time,
            output_tokens=output_tokens,
        )

    except Exception as e:
        return RequestResult(
            success=False,
            total_time=time.perf_counter() - start_time,
            error=str(e)[:300],
        )


async def run_workload(
    base_url: str,
    model: str,
    workload_name: str,
    input_len: int,
    max_output_len: int,
    concurrency: int,
    num_requests: int,
) -> WorkloadResult:
    """Run `num_requests` concurrent (bounded by `concurrency`) requests."""
    prompt = generate_prompt(input_len)
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_request(session: aiohttp.ClientSession):
        async with semaphore:
            return await send_streaming_request(
                session, base_url, model, prompt, max_output_len,
            )

    wall_start = time.perf_counter()
    async with aiohttp.ClientSession() as session:
        tasks = [bounded_request(session) for _ in range(num_requests)]
        results = await asyncio.gather(*tasks)
    wall_time = time.perf_counter() - wall_start

    wr = WorkloadResult(
        backend="",  # filled by caller
        workload=workload_name,
        concurrency=concurrency,
        wall_time=wall_time,
        results=list(results),
    )
    return wr


# ---------------------------------------------------------------------------
# Benchmark orchestration
# ---------------------------------------------------------------------------

async def benchmark_backend(
    backend: str,
    model: str,
    port: int,
    workloads: Dict[str, dict],
    concurrency_levels: List[int],
    num_requests: int,
    dry_run: bool,
) -> List[WorkloadResult]:
    """Start server, benchmark all workloads, stop server."""
    print(f"\n{'=' * 60}")
    print(f"Backend: {backend}")
    print(f"{'=' * 60}")

    server = VLLMServer(
        model=model,
        backend=backend,
        port=port,
        max_model_len=4096,
        gpu_memory_utilization=0.90,
    )

    server.start()
    healthy = await server.wait_healthy()

    if not healthy:
        print(f"  SKIP: Server failed to start for backend={backend}")
        server.stop()
        return []

    all_results: List[WorkloadResult] = []

    try:
        for wl_name, wl_cfg in workloads.items():
            for conc in concurrency_levels:
                n = min(num_requests, max(conc, 2)) if dry_run else num_requests

                print(
                    f"  [{backend}] {wl_name} "
                    f"(in={wl_cfg['input_len']}, out={wl_cfg['max_output_len']}) "
                    f"| concurrency={conc} | n={n}"
                )

                wr = await run_workload(
                    base_url=server.base_url,
                    model=model,
                    workload_name=wl_name,
                    input_len=wl_cfg["input_len"],
                    max_output_len=wl_cfg["max_output_len"],
                    concurrency=conc,
                    num_requests=n,
                )
                wr.backend = backend
                all_results.append(wr)

                # Print immediate feedback
                ok = wr.num_success
                total = wr.num_total
                if ok > 0:
                    ttft_s = f"TTFT={wr.avg_ttft * 1000:.0f}ms" if wr.avg_ttft else ""
                    tpot_s = f"TPOT={wr.avg_tpot * 1000:.1f}ms" if wr.avg_tpot else ""
                    tput_s = (
                        f"Tput={wr.throughput_tok_per_sec:.1f}tok/s"
                        if wr.throughput_tok_per_sec
                        else ""
                    )
                    print(f"    -> {ok}/{total} OK | {ttft_s} {tpot_s} {tput_s}")
                else:
                    errs = [r.error for r in wr.results if r.error]
                    print(f"    -> ALL FAILED: {errs[0] if errs else 'unknown'}")

    finally:
        server.stop()

    return all_results


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------

def fmt(val: Optional[float], fmt_str: str = ".1f", scale: float = 1.0) -> str:
    if val is None:
        return "-"
    return f"{val * scale:{fmt_str}}"


def print_results_table(all_results: List[WorkloadResult]):
    rows = []
    for wr in all_results:
        rows.append({
            "Backend":      wr.backend,
            "Workload":     wr.workload,
            "Conc":         wr.concurrency,
            "OK/Total":     f"{wr.num_success}/{wr.num_total}",
            "TTFT_avg(ms)": fmt(wr.avg_ttft, ".0f", 1000),
            "TTFT_p99(ms)": fmt(wr.p99_ttft, ".0f", 1000),
            "TPOT(ms)":     fmt(wr.avg_tpot, ".1f", 1000),
            "Tput(tok/s)":  fmt(wr.throughput_tok_per_sec, ".1f"),
            "Req/s":        fmt(wr.requests_per_sec, ".2f"),
        })

    if rows:
        print(tabulate(rows, headers="keys", tablefmt="simple", stralign="right"))
    print()


def print_summary_by_backend(all_results: List[WorkloadResult]):
    """Aggregated summary: average metrics per backend (across all workloads)."""
    from collections import defaultdict

    by_backend: Dict[str, List[WorkloadResult]] = defaultdict(list)
    for wr in all_results:
        if wr.num_success > 0:
            by_backend[wr.backend].append(wr)

    rows = []
    for backend, wrs in by_backend.items():
        ttfts = [wr.avg_ttft for wr in wrs if wr.avg_ttft is not None]
        tpots = [wr.avg_tpot for wr in wrs if wr.avg_tpot is not None]
        tputs = [wr.throughput_tok_per_sec for wr in wrs if wr.throughput_tok_per_sec is not None]

        total_ok = sum(wr.num_success for wr in wrs)
        total_all = sum(wr.num_total for wr in wrs)

        rows.append({
            "Backend":       backend,
            "Scenarios":     len(wrs),
            "OK/Total":      f"{total_ok}/{total_all}",
            "TTFT_avg(ms)":  fmt(statistics.mean(ttfts) if ttfts else None, ".0f", 1000),
            "TPOT_avg(ms)":  fmt(statistics.mean(tpots) if tpots else None, ".1f", 1000),
            "Tput_avg(tok/s)": fmt(statistics.mean(tputs) if tputs else None, ".1f"),
        })

    if rows:
        print("Backend Summary (averaged across all workloads/concurrency):")
        print(tabulate(rows, headers="keys", tablefmt="simple", stralign="right"))
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def async_main(args):
    print("vLLM Serving Benchmark for V100")
    print(f"  Model:    {args.model}")
    print(f"  Backends: {args.backends}")
    print(f"  Mode:     {'dry-run' if args.dry_run else 'full benchmark'}")
    print()

    # Select workloads and concurrency
    if args.dry_run:
        workloads = {"short": WORKLOADS["short"]}
        concurrency_levels = [1, 2]
        num_requests = 4
    else:
        if args.workloads:
            workloads = {k: WORKLOADS[k] for k in args.workloads}
        else:
            workloads = WORKLOADS
        concurrency_levels = CONCURRENCY_LEVELS
        num_requests = args.num_requests

    all_results: List[WorkloadResult] = []

    for backend in args.backends:
        results = await benchmark_backend(
            backend=backend,
            model=args.model,
            port=args.port,
            workloads=workloads,
            concurrency_levels=concurrency_levels,
            num_requests=num_requests,
            dry_run=args.dry_run,
        )
        all_results.extend(results)

    # Final summary
    if all_results:
        print(f"\n{'=' * 60}")
        print("DETAILED RESULTS")
        print(f"{'=' * 60}")
        print_results_table(all_results)

        print(f"{'=' * 60}")
        print("SUMMARY")
        print(f"{'=' * 60}")
        print_summary_by_backend(all_results)
    else:
        print("\nNo results collected. All backends may have failed to start.")


def main():
    parser = argparse.ArgumentParser(
        description="vLLM V100 Serving Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"HuggingFace model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--backends", nargs="+", choices=ALL_BACKENDS, default=ALL_BACKENDS,
        help="Attention backends to benchmark (default: all)",
    )
    parser.add_argument(
        "--workloads", nargs="+", choices=list(WORKLOADS.keys()),
        help="Workloads to run (default: all)",
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT,
        help=f"vLLM server port (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--num-requests", type=int, default=32,
        help="Requests per workload/concurrency combination (default: 32)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Quick sanity check: 1 workload, low concurrency, few requests",
    )
    args = parser.parse_args()

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
