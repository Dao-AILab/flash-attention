# BSD 3 Clause
# Copyright 2023 Advanced Micro Devices, Inc.
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Adapted from https://github.com/NVIDIA/apex/blob/master/setup.py
import glob
import os
import shutil
from pathlib import Path

from setuptools import setup, find_packages
import subprocess

import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, ROCM_HOME, CUDA_HOME


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))

# ##JCG update check from apex
# def check_if_rocm_pytorch():
#     is_rocm_pytorch = False
#     if TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 5):
#         from torch.utils.cpp_extension import ROCM_HOME
#         is_rocm_pytorch = True if ((torch.version.hip is not None) and (ROCM_HOME is not None)) else False
#     return is_rocm_pytorch

# IS_ROCM_PYTORCH = check_if_rocm_pytorch()

def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    release = output[release_idx].split(".")
    bare_metal_major = release[0]
    bare_metal_minor = release[1][0]

    return raw_output, bare_metal_major, bare_metal_minor


def check_cuda_torch_binary_vs_bare_metal(cuda_dir):
    raw_output, bare_metal_major, bare_metal_minor = get_cuda_bare_metal_version(cuda_dir)
    torch_binary_major = torch.version.cuda.split(".")[0]
    torch_binary_minor = torch.version.cuda.split(".")[1]

    print("\nCompiling cuda extensions with")
    print(raw_output + "from " + cuda_dir + "/bin\n")

    if (bare_metal_major != torch_binary_major) or (bare_metal_minor != torch_binary_minor):
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
    # _, bare_metal_major, bare_metal_minor = get_cuda_bare_metal_version(CUDA_HOME)
    # if int(bare_metal_major) >= 11 and int(bare_metal_minor) >= 2:
    #     return nvcc_extra_args + ["--threads", "4"]
    return nvcc_extra_args

def rename_cpp_cu(cpp_files):
    for entry in cpp_files:
        shutil.copy(entry, os.path.splitext(entry)[0] + '.cu')


# if not torch.cuda.is_available():
#     # https://github.com/NVIDIA/apex/issues/486
#     # Extension builds after https://github.com/pytorch/pytorch/pull/23408 attempt to query torch.cuda.get_device_capability(),
#     # which will fail if you are compiling in an environment without visible GPUs (e.g. during an nvidia-docker build command).
#     print(
#         "\nWarning: Torch did not find available GPUs on this system.\n",
#         "If your intention is to cross-compile, this is not an error.\n"
#         "By default, We cross-compile for Volta (compute capability 7.0), "
#         "Turing (compute capability 7.5),\n"
#         "and, if the CUDA version is >= 11.0, Ampere (compute capability 8.0).\n"
#         "If you wish to cross-compile for a single specific architecture,\n"
#         'export TORCH_CUDA_ARCH_LIST="compute capability" before running setup.py.\n',
#     )
#     if os.environ.get("TORCH_CUDA_ARCH_LIST", None) is None:
#         _, bare_metal_major, bare_metal_minor = get_cuda_bare_metal_version(CUDA_HOME)
#         if int(bare_metal_major) == 11:
#             os.environ["TORCH_CUDA_ARCH_LIST"] = "7.0;7.5;8.0"
#             if int(bare_metal_minor) > 0:
#                 os.environ["TORCH_CUDA_ARCH_LIST"] = "7.0;7.5;8.0;8.6"
#         else:
#             os.environ["TORCH_CUDA_ARCH_LIST"] = "7.0;7.5"

print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
TORCH_MAJOR = int(torch.__version__.split(".")[0])
TORCH_MINOR = int(torch.__version__.split(".")[1])

##JCG update check from apex
def check_if_rocm_pytorch():
    is_rocm_pytorch = False
    if TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 5):
        from torch.utils.cpp_extension import ROCM_HOME
        is_rocm_pytorch = True if ((torch.version.hip is not None) and (ROCM_HOME is not None)) else False
    return is_rocm_pytorch

IS_ROCM_PYTORCH = check_if_rocm_pytorch()

cmdclass = {}
ext_modules = []

# Check, if ATen/CUDAGeneratorImpl.h is found, otherwise use ATen/cuda/CUDAGeneratorImpl.h
# See https://github.com/pytorch/pytorch/pull/70650
generator_flag = []
torch_dir = torch.__path__[0]
if os.path.exists(os.path.join(torch_dir, "include", "ATen", "CUDAGeneratorImpl.h")):
    generator_flag = ["-DOLD_GENERATOR_PATH"]

# raise_if_cuda_home_none("flash_attn")
# # Check, if CUDA11 is installed for compute capability 8.0
cc_flag = ["-DBUILD_PYTHON_PACKAGE", f"-DFLASH_ATTENTION_INTERNAL_USE_RTZ={os.environ.get('FLASH_ATTENTION_INTERNAL_USE_RTZ', 1)}"]
# _, bare_metal_major, _ = get_cuda_bare_metal_version(CUDA_HOME)
# if int(bare_metal_major) < 11:
#     raise RuntimeError("FlashAttention is only supported on CUDA 11")
# cc_flag.append("-gencode")
# cc_flag.append("arch=compute_75,code=sm_75")
# cc_flag.append("-gencode")
# cc_flag.append("arch=compute_80,code=sm_80")


ck_sources = ["csrc/flash_attn_rocm/composable_kernel/library/src/utility/convolution_parameter.cpp", 
              "csrc/flash_attn_rocm/composable_kernel/library/src/utility/device_memory.cpp", 
              "csrc/flash_attn_rocm/composable_kernel/library/src/utility/host_tensor.cpp"]
fmha_sources = ["csrc/flash_attn_rocm/fmha_api.cpp"] + glob.glob("csrc/flash_attn_rocm/src/*.cpp")

rename_cpp_cu(ck_sources)
rename_cpp_cu(fmha_sources)

# subprocess.run(["git", "submodule", "update", "--init", "csrc/flash_attn_rocm/composable_kernel"])
ext_modules.append(
    CUDAExtension(
        name="flash_attn_cuda",
        sources=["csrc/flash_attn_rocm/fmha_api.cu"] + glob.glob("csrc/flash_attn_rocm/src/*.cu") +
                ["csrc/flash_attn_rocm/composable_kernel/library/src/utility/convolution_parameter.cu",
                 "csrc/flash_attn_rocm/composable_kernel/library/src/utility/device_memory.cu",
                 "csrc/flash_attn_rocm/composable_kernel/library/src/utility/host_tensor.cu"],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++20", "-DNDEBUG"] + generator_flag,
            "nvcc":
                [
                    "-O3",
                    "-std=c++20",
                    "--offload-arch=gfx90a",
                    "-DNDEBUG",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",

                ]
                + generator_flag
                + cc_flag
            ,
        },
        include_dirs=[
            Path(this_dir) / 'csrc' / 'flash_attn_rocm',
            Path(this_dir) / 'csrc' / 'flash_attn_rocm' / 'src',
            Path(this_dir) / 'csrc' / 'flash_attn_rocm' / 'composable_kernel' / 'include' ,
            Path(this_dir) / 'csrc' / 'flash_attn_rocm' / 'composable_kernel' / 'include' / 'ck' ,
            Path(this_dir) / 'csrc' / 'flash_attn_rocm' / 'composable_kernel' / 'include' / 'ck' / 'tensor_operation' / 'gpu' / 'device',
            Path(this_dir) / 'csrc' / 'flash_attn_rocm' / 'composable_kernel' / 'include' / 'ck' / 'tensor_operation' / 'gpu' / 'device' / 'impl',
            Path(this_dir) / 'csrc' / 'flash_attn_rocm' / 'composable_kernel' / 'include' / 'ck' / 'tensor_operation' / 'gpu' /' element',
            Path(this_dir) / 'csrc' / 'flash_attn_rocm' / 'composable_kernel' / 'include' / 'ck' / 'library' / 'utility',
            Path(this_dir) / 'csrc' / 'flash_attn_rocm' / 'composable_kernel' / 'library' / 'include' / 'ck' / 'library' / 'utility',
            Path(this_dir) / 'csrc' / 'flash_attn_rocm' / 'composable_kernel' / 'library' / 'include',
            Path(this_dir) / 'csrc' / 'flash_attn_rocm' / 'composable_kernel' / 'include' / 'ck' / 'utility' / 'library',
            Path(this_dir) / 'csrc' / 'flash_attn_rocm' / 'composable_kernel' / 'include' / 'ck' / 'library' / 'reference_tensor_operation',
            Path(this_dir) / 'csrc' / 'flash_attn_rocm' / 'composable_kernel' / 'include' / 'ck' / 'tensor_operation' / 'reference_tensor_operation',
        ],
    )
)

setup(
    name="flash_attn",
    version="0.2.0",
    packages=find_packages(
        exclude=("build", "csrc", "include", "tests", "dist", "docs", "benchmarks", "flash_attn.egg-info",)
    ),
    author="Tri Dao",
    author_email="trid@stanford.edu",
    description="Flash Attention: Fast and Memory-Efficient Exact Attention",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/HazyResearch/flash-attention",
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
    ],
)
