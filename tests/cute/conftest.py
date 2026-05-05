import os
import subprocess
import logging
import tempfile
import json
import time
from pathlib import Path
from getpass import getuser

_test_start_times: dict[str, float] = {}


def _get_gpu_ids():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return [g.strip() for g in visible.split(",")]

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip().splitlines()
    except (FileNotFoundError,):
        pass

    logging.warning("Failed to get gpu ids, use default '0'")
    return ["0"]


def pytest_configure(config):
    tmp = Path(tempfile.gettempdir()) / getuser() / "flash_attention_tests"
    tmp.mkdir(parents=True, exist_ok=True)

    worker_id = os.environ.get("PYTEST_XDIST_WORKER")
    logging.basicConfig(
        format=config.getini("log_file_format"),
        filename=str(tmp / f"tests_{worker_id}.log"),
        level=config.getini("log_file_level"),
    )
    if not worker_id:
        return
    worker_num = int(worker_id.replace("gw", ""))

    # cache gpu_ids, because nvidia-smi is expensive when we launch many workers doing torch initialization
    # Always elect worker_0 to get gpu_ids.
    cached_gpu_ids = tmp / "gpu_ids.json"
    if worker_num == 0:
        gpu_ids = _get_gpu_ids()
        # Atomic write: write to a temp file then rename so other workers never
        # read a partially-written or truncated file.
        tmp_file = cached_gpu_ids.with_suffix(".tmp")
        with tmp_file.open(mode="w") as f:
            json.dump(gpu_ids, f)
        tmp_file.replace(cached_gpu_ids)
    else:
        while not cached_gpu_ids.exists():
            time.sleep(1)
        # Retry on JSONDecodeError in case we race with the rename.
        for _ in range(10):
            try:
                with cached_gpu_ids.open() as f:
                    gpu_ids = json.load(f)
                break
            except (json.JSONDecodeError, OSError):
                time.sleep(0.2)
        else:
            gpu_ids = _get_gpu_ids()

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids[worker_num % len(gpu_ids)]

_SLOW_THRESHOLD_S = 30.0


def pytest_runtest_logstart(nodeid, location):
    _test_start_times[nodeid] = time.monotonic()
    logging.info(f"[START] {nodeid}")


def pytest_runtest_logreport(report):
    if report.when != "call":
        return
    start = _test_start_times.pop(report.nodeid, None)
    duration = (time.monotonic() - start) if start is not None else report.duration
    status = "PASSED" if report.passed else ("FAILED" if report.failed else "SKIPPED")
    msg = f"[TEST] {status} {duration:.1f}s {report.nodeid}"
    logging.info(msg)
    if report.failed or duration >= _SLOW_THRESHOLD_S:
        print(msg, flush=True)


def pytest_collection_finish(session):
    if not session.config.option.collectonly:
        return

    # file_name -> test_name -> counter
    test_counts: dict[str, dict[str, int]] = {}
    for item in session.items:
        funcname = item.function.__name__
        parent = test_counts.setdefault(item.parent.name, {})
        parent[funcname] = parent.setdefault(funcname, 0) + 1
    print(json.dumps(test_counts, indent=2))
