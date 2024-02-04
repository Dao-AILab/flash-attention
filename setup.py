# Adapted from https://github.com/NVIDIA/apex/blob/master/setup.py
import sys
import warnings
import os
import re
import ast
import shutil
import glob
import subprocess

from pathlib import Path
from packaging.version import parse, Version
from setuptools import setup, find_packages

import torch
from torch.utils.cpp_extension import (
    BuildExtension,
    CUDAExtension,
    CUDA_HOME,
    IS_HIP_EXTENSION,
)

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))


def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output(
        [cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True
    )
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_version


def check_cuda_torch_binary_vs_bare_metal(cuda_dir):
    raw_output, bare_metal_version = get_cuda_bare_metal_version(cuda_dir)
    torch_binary_version = parse(torch.version.cuda)  # type: ignore

    print("\nCompiling cuda extensions with")
    print(raw_output + "from " + cuda_dir + "/bin\n")

    if bare_metal_version != torch_binary_version:
        raise RuntimeError(
            "Cuda extensions are being compiled with a version of Cuda that does "
            "not match the version used to compile Pytorch binaries.  "
            f"Pytorch binaries were compiled with Cuda {torch.version.cuda}.\n"
            + "In some cases, a minor-version mismatch will not cause later errors:  "
            "https://github.com/NVIDIA/apex/pull/323#discussion_r287021798.  "
            "You can try commenting out this check (at your own risk)."
        )


def raise_if_cuda_home_none(global_option: str) -> None:
    if CUDA_HOME is not None:
        return
    raise RuntimeError(
        f"{global_option} was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  "
        "If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, "
        "only images whose names contain 'devel' will provide nvcc."
    )


def append_nvcc_threads(nvcc_extra_args):
    _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
    if bare_metal_version >= Version("11.2"):
        return nvcc_extra_args + ["--threads", "4"]
    return nvcc_extra_args


if not torch.cuda.is_available():
    # https://github.com/NVIDIA/apex/issues/486
    # Extension builds after https://github.com/pytorch/pytorch/pull/23408 attempt to query torch.cuda.get_device_capability(),
    # which will fail if you are compiling in an environment without visible GPUs (e.g. during an nvidia-docker build command).
    print(
        "\nWarning: Torch did not find available GPUs on this system.\n",
        "If your intention is to cross-compile, this is not an error.\n"
        "By default, Apex will cross-compile for Pascal (compute capabilities 6.0, 6.1, 6.2),\n"
        "Volta (compute capability 7.0), Turing (compute capability 7.5),\n"
        "and, if the CUDA version is >= 11.0, Ampere (compute capability 8.0).\n"
        "If you wish to cross-compile for a single specific architecture,\n"
        'export TORCH_CUDA_ARCH_LIST="compute capability" before running setup.py.\n',
    )
    if os.environ.get("TORCH_CUDA_ARCH_LIST", None) is None and CUDA_HOME is not None:
        _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
        if bare_metal_version >= Version("11.8"):
            os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5;8.0;8.6;9.0"
        elif bare_metal_version >= Version("11.1"):
            os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5;8.0;8.6"
        elif bare_metal_version == Version("11.0"):
            os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5;8.0"
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5"


print(f"\n\ntorch.__version__  = {torch.__version__}\n\n")
TORCH_MAJOR = int(torch.__version__.split(".", maxsplit=1)[0])
TORCH_MINOR = int(torch.__version__.split(".")[1])


cmdclass = {}
ext_modules = []

# Check, if ATen/CUDAGeneratorImpl.h is found, otherwise use ATen/cuda/CUDAGeneratorImpl.h
# See https://github.com/pytorch/pytorch/pull/70650
generator_flag = []
TORCH_DIR = torch.__path__[0]  # type: ignore

if os.path.exists(os.path.join(TORCH_DIR, "include", "ATen", "CUDAGeneratorImpl.h")):
    generator_flag = ["-DOLD_GENERATOR_PATH"]


def build_for_cuda():
    """build flash-attention for CUDA platform

    Raises:
        RuntimeError: None
    """
    raise_if_cuda_home_none("flash_attn")
    # Check, if CUDA11 is installed for compute capability 8.0
    cc_flag = []
    _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
    if bare_metal_version < Version("11.0"):
        raise RuntimeError("FlashAttention is only supported on CUDA 11 and above")
    # cc_flag.append("-gencode")
    # cc_flag.append("arch=compute_75,code=sm_75")
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_80,code=sm_80")
    if bare_metal_version >= Version("11.8"):
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_90,code=sm_90")

    subprocess.run(["git", "submodule", "update", "--init", "csrc/cutlass"], check=True)
    ext_modules.append(
        CUDAExtension(
            name="flash_attn_2_cuda",
            sources=[
                "csrc/flash_attn/flash_api.cpp",
                "csrc/flash_attn/src/flash_fwd_hdim32_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim32_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim64_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim64_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim96_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim96_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim128_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim128_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim160_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim160_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim192_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim192_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim224_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim224_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim256_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim256_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim32_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim32_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim64_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim64_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim96_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim96_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim128_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim128_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim160_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim160_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim192_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim192_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim224_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim224_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim256_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim256_bf16_sm80.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"] + generator_flag,
                "nvcc": append_nvcc_threads(
                    [
                        "-O3",
                        "-std=c++17",
                        "-U__CUDA_NO_HALF_OPERATORS__",
                        "-U__CUDA_NO_HALF_CONVERSIONS__",
                        "-U__CUDA_NO_HALF2_OPERATORS__",
                        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                        "--expt-relaxed-constexpr",
                        "--expt-extended-lambda",
                        "--use_fast_math",
                        "--ptxas-options=-v",
                        "-lineinfo",
                    ]
                    + generator_flag
                    + cc_flag
                ),
            },
            include_dirs=[
                Path(this_dir) / "csrc" / "flash_attn",
                Path(this_dir) / "csrc" / "flash_attn" / "src",
                Path(this_dir) / "csrc" / "cutlass" / "include",
            ],
        )
    )


def rename_cpp_to_hip(cpp_files):
    for entry in cpp_files:
        shutil.copy(entry, os.path.splitext(entry)[0] + ".hip")


# Defining a function to validate the GPU architectures and update them if necessary
def validate_and_update_archs(archs):
    # List of allowed architectures
    allowed_archs = ["native", "gfx90a", "gfx908", "gfx940", "gfx941", "gfx942"]

    # Validate if each element in archs is in allowed_archs
    assert all(
        arch in allowed_archs for arch in archs
    ), f"One of GPU archs of {archs} is invalid or not supported by Flash-Attention"


def build_for_rocm():
    """build for ROCm platform"""
    archs = os.getenv("GPU_ARCHS", "native").split(";")
    validate_and_update_archs(archs)
    cc_flag = [f"--offload-arch={arch}" for arch in archs]

    if int(os.environ.get("FLASH_ATTENTION_INTERNAL_USE_RTN", 0)):
        print("RTN IS USED")
        cc_flag.append("-DUSE_RTN_BF16_CONVERT")
    else:
        print("RTZ IS USED")

    fa_sources = ["csrc/flash_attn_rocm/flash_api.cpp"] + glob.glob(
        "csrc/flash_attn_rocm/src/*.cpp"
    )

    rename_cpp_to_hip(fa_sources)

    subprocess.run(
        [
            "git",
            "submodule",
            "update",
            "--init",
            "csrc/flash_attn_rocm/composable_kernel",
        ],
        check=True,
    )
    ext_modules.append(
        CUDAExtension(
            name="flash_attn_2_cuda",
            sources=["csrc/flash_attn_rocm/flash_api.hip"]
            + glob.glob("csrc/flash_attn_rocm/src/*.hip"),
            extra_compile_args={
                "cxx": ["-O3", "-std=c++20", "-DNDEBUG"] + generator_flag,
                "nvcc": [
                    "-O3",
                    "-std=c++20",
                    "-DNDEBUG",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                ]
                + generator_flag
                + cc_flag,
            },
            include_dirs=[
                Path(this_dir) / "csrc" / "flash_attn_rocm",
                Path(this_dir) / "csrc" / "flash_attn_rocm" / "src",
                Path(this_dir)
                / "csrc"
                / "flash_attn_rocm"
                / "composable_kernel"
                / "include",
                Path(this_dir)
                / "csrc"
                / "flash_attn_rocm"
                / "composable_kernel"
                / "library"
                / "include",
            ],
        )
    )


BUILD_TARGET = os.environ.get("BUILD_TARGET", "auto")

if BUILD_TARGET == "auto":
    if IS_HIP_EXTENSION:
        build_for_rocm()
    else:
        build_for_cuda()
else:
    if BUILD_TARGET == "cuda":
        build_for_cuda()
    elif BUILD_TARGET == "rocm":
        build_for_rocm()


def get_package_version():
    with open(Path(this_dir) / "flash_attn" / "__init__.py", "r") as f:
        version_match = re.search(r"^__version__\s*=\s*(.*)$", f.read(), re.MULTILINE)
    public_version = ast.literal_eval(version_match.group(1))
    local_version = os.environ.get("FLASH_ATTN_LOCAL_VERSION")
    if local_version:
        return f"{public_version}+{local_version}"
    else:
        return str(public_version)


setup(
    name="flash_attn",
    version=get_package_version(),
    packages=find_packages(
        exclude=(
            "build",
            "csrc",
            "include",
            "tests",
            "dist",
            "docs",
            "benchmarks",
            "flash_attn.egg-info",
        )
    ),
    author="Tri Dao",
    author_email="trid@cs.stanford.edu",
    description="Flash Attention: Fast and Memory-Efficient Exact Attention",
    url="https://github.com/Dao-AILab/flash-attention",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Unix",
    ],
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension} if ext_modules else {},
    python_requires=">=3.7",
    install_requires=[
        "torch",
        "einops",
        "packaging",
        "ninja",
    ],
)
