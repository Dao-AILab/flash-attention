import os
import subprocess


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
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return ["0"]


def pytest_configure(config):
    worker_id = os.environ.get("PYTEST_XDIST_WORKER")
    if not worker_id:
        return
    worker_num = int(worker_id.replace("gw", ""))
    gpu_ids = _get_gpu_ids()
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids[worker_num % len(gpu_ids)]
