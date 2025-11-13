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
def run(page_size: int = 128):
    import os
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TORCH_USE_CUDA_DSA"] = "1"

    command = [
        sys.executable, "bench.py",
        "--page_size", str(page_size),
    ]

    print("Command:", " ".join(command))

    subprocess.run(command)

@app.function(
    gpu="B200",
    timeout=3600,
    image=image.pip_install(["pytest"]).add_local_dir("tests", remote_path="/root/flash_attn/tests", copy=True),
)
def test():
    subprocess.run([sys.executable, "-m", "pytest", "-k", "test_flash_attn_kvcache", "flash_attn/tests/cute/test_flash_attn.py"])