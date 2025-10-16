import modal

app = modal.App(name="flash-attn")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .pip_install(["nvidia-cutlass-dsl==4.2.1", "torch", "einops"])
    .add_local_dir("./flash_attn/cute", remote_path="/root/flash_attn/cute", copy=True)
    .pip_install_from_pyproject("flash_attn/cute/pyproject.toml")
)


@app.function(
    gpu="B200",
    timeout=3600,
    image=image.add_local_file("bench.py", remote_path="/root/bench.py"),
)
def run(
    num_splits: int = 4,
    benchmark_type: str = "standard",
    num_batches: int = 1,
    num_queries: int = 256,
    num_keys: int = 8192,
    num_heads: int = 32,
    head_dim: int = 128,
    kvheads_per_group: int = 8,
):

    import subprocess
    import sys
    subprocess.run([
        sys.executable, "bench.py",
        "--num_splits", str(num_splits),
        "--benchmark_type", benchmark_type,
        "--B", str(num_batches),
        "--Q", str(num_queries),
        "--K", str(num_keys),
        "--H", str(num_heads),
        "--D", str(head_dim),
        "--kvheads_per_group", str(kvheads_per_group),
    ])

@app.function(
    gpu="B200",
    timeout=3600,
    image=image.pip_install("pytest").add_local_dir("./tests/cute", remote_path="/root/tests/cute"),
)
def test():
    import subprocess
    subprocess.run(["pytest", "tests/cute/test_flash_attn.py", "-k", "test_flash_attn_output"])
