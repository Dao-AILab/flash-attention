# Adapted from https://github.com/NVIDIA/apex/blob/master/setup.py
import sys
import warnings
import os
import re
import ast
from pathlib import Path
from packaging.version import parse, Version
import platform

from setuptools import setup, find_packages
import subprocess

import urllib.request
import urllib.error
import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))

PACKAGE_NAME = "flash_attn_wheels"

# @pierce - TODO: Update for proper release
BASE_WHEEL_URL = "https://github.com/piercefreeman/flash-attention/releases/download/{tag_name}/{wheel_name}"

# FORCE_BUILD: Force a fresh build locally, instead of attempting to find prebuilt wheels
# SKIP_CUDA_BUILD: Intended to allow CI to use a simple `python setup.py sdist` run to copy over raw files, without any cuda compilation
FORCE_BUILD = os.getenv("FLASH_ATTENTION_FORCE_BUILD", "FALSE") == "TRUE"
SKIP_CUDA_BUILD = os.getenv("FLASH_ATTENTION_SKIP_CUDA_BUILD", "FALSE") == "TRUE"


def get_platform():
    """
    Returns the platform name as used in wheel filenames.
    """
    if sys.platform.startswith('linux'):
        return 'linux_x86_64'
    elif sys.platform == 'darwin':
        mac_version = '.'.join(platform.mac_ver()[0].split('.')[:2])
        return f'macosx_{mac_version}_x86_64'
    elif sys.platform == 'win32':
        return 'win_amd64'
    else:
        raise ValueError('Unsupported platform: {}'.format(sys.platform))


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

cmdclass = {}
ext_modules = []

if not SKIP_CUDA_BUILD:
    print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
    TORCH_MAJOR = int(torch.__version__.split(".")[0])
    TORCH_MINOR = int(torch.__version__.split(".")[1])

    # Check, if ATen/CUDAGeneratorImpl.h is found, otherwise use ATen/cuda/CUDAGeneratorImpl.h
    # See https://github.com/pytorch/pytorch/pull/70650
    generator_flag = []
    torch_dir = torch.__path__[0]
    if os.path.exists(os.path.join(torch_dir, "include", "ATen", "CUDAGeneratorImpl.h")):
        generator_flag = ["-DOLD_GENERATOR_PATH"]

    raise_if_cuda_home_none("flash_attn")
    # Check, if CUDA11 is installed for compute capability 8.0
    cc_flag = []
    _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
    if bare_metal_version < Version("11.0"):
        raise RuntimeError("FlashAttention is only supported on CUDA 11 and above")
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_75,code=sm_75")
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_80,code=sm_80")
    if bare_metal_version >= Version("11.8"):
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_90,code=sm_90")

    subprocess.run(["git", "submodule", "update", "--init", "csrc/flash_attn/cutlass"])
    ext_modules.append(
        CUDAExtension(
            name="flash_attn_cuda",
            sources=[
                "csrc/flash_attn/fmha_api.cpp",
                "csrc/flash_attn/src/fmha_fwd_hdim32.cu",
                "csrc/flash_attn/src/fmha_fwd_hdim64.cu",
                "csrc/flash_attn/src/fmha_fwd_hdim128.cu",
                "csrc/flash_attn/src/fmha_bwd_hdim32.cu",
                "csrc/flash_attn/src/fmha_bwd_hdim64.cu",
                "csrc/flash_attn/src/fmha_bwd_hdim128.cu",
                "csrc/flash_attn/src/fmha_block_fprop_fp16_kernel.sm80.cu",
                "csrc/flash_attn/src/fmha_block_dgrad_fp16_kernel_loop.sm80.cu",
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
                        "-lineinfo"
                    ]
                    + generator_flag
                    + cc_flag
                ),
            },
            include_dirs=[
                Path(this_dir) / 'csrc' / 'flash_attn',
                Path(this_dir) / 'csrc' / 'flash_attn' / 'src',
                Path(this_dir) / 'csrc' / 'flash_attn' / 'cutlass' / 'include',
            ],
        )
    )

def get_package_version():
    with open(Path(this_dir) / "flash_attn" / "__init__.py", "r") as f:
        version_match = re.search(r"^__version__\s*=\s*(.*)$", f.read(), re.MULTILINE)
    public_version = ast.literal_eval(version_match.group(1))
    local_version = os.environ.get("FLASH_ATTN_LOCAL_VERSION")
    if local_version:
        return f"{public_version}+{local_version}"
    else:
        return str(public_version)


class CachedWheelsCommand(_bdist_wheel):
     """
     The CachedWheelsCommand plugs into the default bdist wheel, which is ran by pip when it cannot
     find an existing wheel (which is currently the case for all flash attention installs). We use
     the environment parameters to detect whether there is already a pre-built version of a compatible
     wheel available and short-circuits the standard full build pipeline.

     """
     def run(self):
        if FORCE_BUILD:
            return super().run()

        raise_if_cuda_home_none("flash_attn")

        # Determine the version numbers that will be used to determine the correct wheel
        _, cuda_version_raw = get_cuda_bare_metal_version(CUDA_HOME)
        torch_version_raw = parse(torch.__version__)
        python_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
        platform_name = get_platform()
        flash_version = get_package_version()
        cuda_version = f"{cuda_version_raw.major}{cuda_version_raw.minor}"
        torch_version = f"{torch_version_raw.major}.{torch_version_raw.minor}.{torch_version_raw.micro}"

        # Determine wheel URL based on CUDA version, torch version, python version and OS
        wheel_filename = f'{PACKAGE_NAME}-{flash_version}+cu{cuda_version}torch{torch_version}-{python_version}-{python_version}-{platform_name}.whl'
        wheel_url = BASE_WHEEL_URL.format(
            tag_name=f"v{flash_version}",
            wheel_name=wheel_filename
        )
        print("Guessing wheel URL: ", wheel_url)
        
        try:
            urllib.request.urlretrieve(wheel_url, wheel_filename)

            # Make the archive
            # Lifted from the root wheel processing command
            # https://github.com/pypa/wheel/blob/cf71108ff9f6ffc36978069acb28824b44ae028e/src/wheel/bdist_wheel.py#LL381C9-L381C85
            if not os.path.exists(self.dist_dir):
                os.makedirs(self.dist_dir)

            impl_tag, abi_tag, plat_tag = self.get_tag()
            archive_basename = f"{self.wheel_dist_name}-{impl_tag}-{abi_tag}-{plat_tag}"
        
            wheel_path = os.path.join(self.dist_dir, archive_basename + ".whl")
            print("Raw wheel path", wheel_path)
            os.rename(wheel_filename, wheel_path)
        except urllib.error.HTTPError:
            print("Precompiled wheel not found. Building from source...")
            # If the wheel could not be downloaded, build from source
            super().run()


setup(
    # @pierce - TODO: Revert for official release
    name=PACKAGE_NAME,
    version=get_package_version(),
    packages=find_packages(
        exclude=("build", "csrc", "include", "tests", "dist", "docs", "benchmarks", "flash_attn.egg-info",)
    ),
    #author="Tri Dao",
    #author_email="trid@stanford.edu",
    # @pierce - TODO: Revert for official release
    author="Pierce Freeman",
    author_email="pierce@freeman.vc",
    description="Flash Attention: Fast and Memory-Efficient Exact Attention",
    long_description=long_description,
    long_description_content_type="text/markdown",
    #url="https://github.com/HazyResearch/flash-attention",
    url="https://github.com/piercefreeman/flash-attention",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Unix",
    ],
    ext_modules=ext_modules,
    cmdclass={
        'bdist_wheel': CachedWheelsCommand,
        "build_ext": BuildExtension
    } if ext_modules else {
        'bdist_wheel': CachedWheelsCommand,
    },
    python_requires=">=3.7",
    install_requires=[
        "torch",
        "einops",
        "packaging",
        "ninja",
    ],
)
