import modal
import subprocess
import sys


app = modal.App(name="flash-attn")


image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .pip_install(["nvidia-cutlass-dsl==4.3.0.dev0", "torch", "einops"])
    .add_local_dir("./flash_attn/cute", remote_path="/root/flash_attn/cute", copy=True)
    .pip_install_from_pyproject("./flash_attn/cute/pyproject.toml")
)


@app.function(
    gpu="B200",
    timeout=3600,
    image=image.add_local_file("bench.py", remote_path="/root/bench.py"),
)
def run(
    num_splits: list[int] = [4],
    benchmark_type: list[str] = ["standard"],
    num_batches: list[int] = [1],
    num_queries: list[int] = [1],
    num_keys: list[int] = [131072],
    num_heads: list[int] = [1],
    head_dim: list[int] = [128],
    kvheads_per_group: list[int] = [1],
):
    command = [
        sys.executable, "bench.py",
        "--num_splits", *[str(n) for n in num_splits],
        "--benchmark_type", *benchmark_type,
        "--B", *[str(b) for b in num_batches],
        "--Q", *[str(q) for q in num_queries],
        "--K", *[str(k) for k in num_keys],
        "--H", *[str(h) for h in num_heads],
        "--D", *[str(d) for d in head_dim],
        "--kvheads_per_group", *[str(k) for k in kvheads_per_group],
        "--output_format", "csv",
    ]

    print("Command:", " ".join(command))

    subprocess.run(command)


@app.function(
    gpu="B200",
    timeout=3600,
    image=image.pip_install("pytest").add_local_dir("./tests/cute", remote_path="/root/tests/cute"),
)
def test():
    subprocess.run(["pytest", "tests/cute/test_flash_attn.py"])


@app.local_entrypoint()
def main():
    run.remote(
        num_splits=[0, 1, 2, 4, 8, 16, 32, 64, 128],
        num_queries=[1, 32],
        num_keys=[131072],
        num_heads=[32],
        kvheads_per_group=[1, 8],
        benchmark_type=["standard"],
    )