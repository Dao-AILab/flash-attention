# Adapted from https://github.com/NVIDIA/apex/blob/master/setup.py
import sys
import warnings
import os
from packaging.version import parse, Version

import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME
from setuptools import setup, find_packages
import subprocess

# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))


def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_version


def check_cuda_torch_binary_vs_bare_metal(cuda_dir):
    raw_output, bare_metal_version = get_cuda_bare_metal_version(cuda_dir)
    torch_binary_version = parse(torch.version.cuda)

    print("\nCompiling cuda extensions with")
    print(raw_output + "from " + cuda_dir + "/bin\n")

    if (bare_metal_version != torch_binary_version):
        raise RuntimeError(
            "Cuda extensions are being compiled with a version of Cuda that does "
            "not match the version used to compile Pytorch binaries.  "
            "Pytorch binaries were compiled with Cuda {}.\n".format(torch.version.cuda)
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


print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
TORCH_MAJOR = int(torch.__version__.split(".")[0])
TORCH_MINOR = int(torch.__version__.split(".")[1])

cmdclass = {}
ext_modules = []

# Check, if ATen/CUDAGeneratorImpl.h is found, otherwise use ATen/cuda/CUDAGeneratorImpl.h
# See https://github.com/pytorch/pytorch/pull/70650
generator_flag = []
torch_dir = torch.__path__[0]
if os.path.exists(os.path.join(torch_dir, "include", "ATen", "CUDAGeneratorImpl.h")):
    generator_flag = ["-DOLD_GENERATOR_PATH"]

raise_if_cuda_home_none("--fast_layer_norm")
# Check, if CUDA11 is installed for compute capability 8.0
cc_flag = []
_, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
if bare_metal_version < Version("11.0"):
    raise RuntimeError("dropout_layer_norm is only supported on CUDA 11 and above")
cc_flag.append("-gencode")
cc_flag.append("arch=compute_70,code=sm_70")
cc_flag.append("-gencode")
cc_flag.append("arch=compute_80,code=sm_80")
if bare_metal_version >= Version("11.8"):
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_90,code=sm_90")

ext_modules.append(
    CUDAExtension(
        name="dropout_layer_norm",
        sources=[
            "ln_api.cpp",
            "ln_fwd_256.cu",
            "ln_bwd_256.cu",
            "ln_fwd_512.cu",
            "ln_bwd_512.cu",
            "ln_fwd_768.cu",
            "ln_bwd_768.cu",
            "ln_fwd_1024.cu",
            "ln_bwd_1024.cu",
            "ln_fwd_1280.cu",
            "ln_bwd_1280.cu",
            "ln_fwd_1536.cu",
            "ln_bwd_1536.cu",
            "ln_fwd_2048.cu",
            "ln_bwd_2048.cu",
            "ln_fwd_2560.cu",
            "ln_bwd_2560.cu",
            "ln_fwd_3072.cu",
            "ln_bwd_3072.cu",
            "ln_fwd_4096.cu",
            "ln_bwd_4096.cu",
            "ln_fwd_5120.cu",
            "ln_bwd_5120.cu",
            "ln_fwd_6144.cu",
            "ln_bwd_6144.cu",
            "ln_fwd_7168.cu",
            "ln_bwd_7168.cu",
            "ln_fwd_8192.cu",
            "ln_bwd_8192.cu",
            "ln_parallel_fwd_256.cu",
            "ln_parallel_bwd_256.cu",
            "ln_parallel_fwd_512.cu",
            "ln_parallel_bwd_512.cu",
            "ln_parallel_fwd_768.cu",
            "ln_parallel_bwd_768.cu",
            "ln_parallel_fwd_1024.cu",
            "ln_parallel_bwd_1024.cu",
            "ln_parallel_fwd_1280.cu",
            "ln_parallel_bwd_1280.cu",
            "ln_parallel_fwd_1536.cu",
            "ln_parallel_bwd_1536.cu",
            "ln_parallel_fwd_2048.cu",
            "ln_parallel_bwd_2048.cu",
            "ln_parallel_fwd_2560.cu",
            "ln_parallel_bwd_2560.cu",
            "ln_parallel_fwd_3072.cu",
            "ln_parallel_bwd_3072.cu",
            "ln_parallel_fwd_4096.cu",
            "ln_parallel_bwd_4096.cu",
            "ln_parallel_fwd_5120.cu",
            "ln_parallel_bwd_5120.cu",
            "ln_parallel_fwd_6144.cu",
            "ln_parallel_bwd_6144.cu",
            "ln_parallel_fwd_7168.cu",
            "ln_parallel_bwd_7168.cu",
            "ln_parallel_fwd_8192.cu",
            "ln_parallel_bwd_8192.cu",
        ],
        extra_compile_args={
            "cxx": ["-O3"] + generator_flag,
            "nvcc": append_nvcc_threads(
                [
                    "-O3",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",
                ]
                + generator_flag
                + cc_flag
            ),
        },
        include_dirs=[this_dir],
    )
)

setup(
    name="dropout_layer_norm",
    version="0.1",
    description="Fused dropout + add + layer norm",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension} if ext_modules else {},
)
