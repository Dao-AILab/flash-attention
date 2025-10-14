import modal

app = modal.App(name="flash-attn")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .pip_install(["nvidia-cutlass-dsl==4.2.1", "torch", "einops"])
    .add_local_dir("./flash_attn/cute", remote_path="/root/flash_attn/cute", copy=True)
    .pip_install_from_pyproject("flash_attn/cute/pyproject.toml")
    .add_local_file("bench.py", remote_path="/root/bench.py")
)


@app.function(
    gpu="B200",
    image=image,
)
def run():
    import subprocess
    subprocess.run(["python", "bench.py", "--benchmark_type", "varlen"])
