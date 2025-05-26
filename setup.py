# Copyright (c) 2023, Tri Dao.

import sys
import functools
import warnings
import os
import re
import ast
import glob
import shutil
from pathlib import Path
from packaging.version import parse, Version
import platform

from setuptools import setup, find_packages
import subprocess

import urllib.request
import urllib.error
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

import torch
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
    CUDAExtension,
    CUDA_HOME,
    ROCM_HOME,
    IS_HIP_EXTENSION,
)


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))

BUILD_TARGET = os.environ.get("BUILD_TARGET", "auto")

if BUILD_TARGET == "auto":
    if IS_HIP_EXTENSION:
        IS_ROCM = True
    else:
        IS_ROCM = False
else:
    if BUILD_TARGET == "cuda":
        IS_ROCM = False
    elif BUILD_TARGET == "rocm":
        IS_ROCM = True

PACKAGE_NAME = "flash_attn"

BASE_WHEEL_URL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/{tag_name}/{wheel_name}"
)

# FORCE_BUILD: Force a fresh build locally, instead of attempting to find prebuilt wheels
# SKIP_CUDA_BUILD: Intended to allow CI to use a simple `python setup.py sdist` run to copy over raw files, without any cuda compilation
FORCE_BUILD = os.getenv("FLASH_ATTENTION_FORCE_BUILD", "FALSE") == "TRUE"
SKIP_CUDA_BUILD = os.getenv("FLASH_ATTENTION_SKIP_CUDA_BUILD", "FALSE") == "TRUE"
# For CI, we want the option to build with C++11 ABI since the nvcr images use C++11 ABI
FORCE_CXX11_ABI = os.getenv("FLASH_ATTENTION_FORCE_CXX11_ABI", "FALSE") == "TRUE"
USE_TRITON_ROCM = os.getenv("FLASH_ATTENTION_TRITON_AMD_ENABLE", "FALSE") == "TRUE"
SKIP_CK_BUILD = os.getenv("FLASH_ATTENTION_SKIP_CK_BUILD", "TRUE") == "TRUE" if USE_TRITON_ROCM else False

@functools.lru_cache(maxsize=None)
def cuda_archs() -> str:
    return os.getenv("FLASH_ATTN_CUDA_ARCHS", "80;90;100;120").split(";")


def get_platform():
    """
    Returns the platform name as used in wheel filenames.
    """
    if sys.platform.startswith("linux"):
        return f'linux_{platform.uname().machine}'
    elif sys.platform == "darwin":
        mac_version = ".".join(platform.mac_ver()[0].split(".")[:2])
        return f"macosx_{mac_version}_x86_64"
    elif sys.platform == "win32":
        return "win_amd64"
    else:
        raise ValueError("Unsupported platform: {}".format(sys.platform))


def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_version


def get_hip_version():
    return parse(torch.version.hip.split()[-1].rstrip('-').replace('-', '+'))


def check_if_cuda_home_none(global_option: str) -> None:
    if CUDA_HOME is not None:
        return
    # warn instead of error because user could be downloading prebuilt wheels, so nvcc won't be necessary
    # in that case.
    warnings.warn(
        f"{global_option} was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  "
        "If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, "
        "only images whose names contain 'devel' will provide nvcc."
    )


def check_if_rocm_home_none(global_option: str) -> None:
    if ROCM_HOME is not None:
        return
    # warn instead of error because user could be downloading prebuilt wheels, so hipcc won't be necessary
    # in that case.
    warnings.warn(
        f"{global_option} was requested, but hipcc was not found."
    )


def append_nvcc_threads(nvcc_extra_args):
    nvcc_threads = os.getenv("NVCC_THREADS") or "4"
    return nvcc_extra_args + ["--threads", nvcc_threads]


def rename_cpp_to_cu(cpp_files):
    for entry in cpp_files:
        shutil.copy(entry, os.path.splitext(entry)[0] + ".cu")


def validate_and_update_archs(archs):
    # List of allowed architectures
    allowed_archs = ["native", "gfx90a", "gfx950", "gfx942"]

    # Validate if each element in archs is in allowed_archs
    assert all(
        arch in allowed_archs for arch in archs
    ), f"One of GPU archs of {archs} is invalid or not supported by Flash-Attention"


cmdclass = {}
ext_modules = []

# We want this even if SKIP_CUDA_BUILD because when we run python setup.py sdist we want the .hpp
# files included in the source distribution, in case the user compiles from source.
if os.path.isdir(".git"):
    if not SKIP_CK_BUILD:
        subprocess.run(["git", "submodule", "update", "--init", "csrc/composable_kernel"], check=True)
        subprocess.run(["git", "submodule", "update", "--init", "csrc/cutlass"], check=True)
else:
    if IS_ROCM:
        if not SKIP_CK_BUILD:
            assert (
                os.path.exists("csrc/composable_kernel/example/ck_tile/01_fmha/generate.py")
            ), "csrc/composable_kernel is missing, please use source distribution or git clone"
    else:
        assert (
            os.path.exists("csrc/cutlass/include/cutlass/cutlass.h")
        ), "csrc/cutlass is missing, please use source distribution or git clone"

if not SKIP_CUDA_BUILD and not IS_ROCM:
    print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
    TORCH_MAJOR = int(torch.__version__.split(".")[0])
    TORCH_MINOR = int(torch.__version__.split(".")[1])

    check_if_cuda_home_none("flash_attn")
    # Check, if CUDA11 is installed for compute capability 8.0
    cc_flag = []
    if CUDA_HOME is not None:
        _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
        if bare_metal_version < Version("11.7"):
            raise RuntimeError(
                "FlashAttention is only supported on CUDA 11.7 and above.  "
                "Note: make sure nvcc has a supported version by running nvcc -V."
            )

    if "80" in cuda_archs():
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_80,code=sm_80")
    if CUDA_HOME is not None:
        if bare_metal_version >= Version("11.8") and "90" in cuda_archs():
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_90,code=sm_90")
        if bare_metal_version >= Version("12.8") and "100" in cuda_archs():
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_100,code=sm_100")
        if bare_metal_version >= Version("12.8") and "120" in cuda_archs():
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_120,code=sm_120")

    # HACK: The compiler flag -D_GLIBCXX_USE_CXX11_ABI is set to be the same as
    # torch._C._GLIBCXX_USE_CXX11_ABI
    # https://github.com/pytorch/pytorch/blob/8472c24e3b5b60150096486616d98b7bea01500b/torch/utils/cpp_extension.py#L920
    if FORCE_CXX11_ABI:
        torch._C._GLIBCXX_USE_CXX11_ABI = True
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
                "csrc/flash_attn/src/flash_fwd_hdim192_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim192_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim256_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim256_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim32_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim32_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim64_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim64_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim96_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim96_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim128_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim128_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim192_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim192_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim256_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim256_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim32_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim32_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim64_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim64_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim96_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim96_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim128_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim128_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim192_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim192_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim256_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim256_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim32_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim32_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim64_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim64_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim96_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim96_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim128_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim128_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim192_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim192_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim256_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim256_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim32_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim32_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim64_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim64_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim96_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim96_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim128_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim128_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim192_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim192_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim256_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim256_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim32_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim32_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim64_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim64_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim96_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim96_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim128_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim128_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim192_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim192_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim256_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim256_bf16_causal_sm80.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
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
                        # "--ptxas-options=-v",
                        # "--ptxas-options=-O2",
                        # "-lineinfo",
                        # "-DFLASHATTENTION_DISABLE_BACKWARD",
                        # "-DFLASHATTENTION_DISABLE_DROPOUT",
                        # "-DFLASHATTENTION_DISABLE_ALIBI",
                        # "-DFLASHATTENTION_DISABLE_SOFTCAP",
                        # "-DFLASHATTENTION_DISABLE_UNEVEN_K",
                        # "-DFLASHATTENTION_DISABLE_LOCAL",
                    ]
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
elif not SKIP_CUDA_BUILD and IS_ROCM:
    print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
    TORCH_MAJOR = int(torch.__version__.split(".")[0])
    TORCH_MINOR = int(torch.__version__.split(".")[1])

    # Skips CK C++ extension compilation if using Triton Backend
    if not SKIP_CK_BUILD:
        ck_dir = "csrc/composable_kernel"

        #use codegen get code dispatch
        if not os.path.exists("./build"):
            os.makedirs("build")

        subprocess.run([sys.executable, f"{ck_dir}/example/ck_tile/01_fmha/generate.py", "-d", "fwd", "--output_dir", "build", "--receipt", "2"], check=True)
        subprocess.run([sys.executable, f"{ck_dir}/example/ck_tile/01_fmha/generate.py", "-d", "fwd_appendkv", "--output_dir", "build", "--receipt", "2"], check=True)
        subprocess.run([sys.executable, f"{ck_dir}/example/ck_tile/01_fmha/generate.py", "-d", "fwd_splitkv", "--output_dir", "build", "--receipt", "2"], check=True)
        subprocess.run([sys.executable, f"{ck_dir}/example/ck_tile/01_fmha/generate.py", "-d", "bwd", "--output_dir", "build", "--receipt", "2"], check=True)

        # Check, if ATen/CUDAGeneratorImpl.h is found, otherwise use ATen/cuda/CUDAGeneratorImpl.h
        # See https://github.com/pytorch/pytorch/pull/70650
        generator_flag = []
        torch_dir = torch.__path__[0]
        if os.path.exists(os.path.join(torch_dir, "include", "ATen", "CUDAGeneratorImpl.h")):
            generator_flag = ["-DOLD_GENERATOR_PATH"]

        check_if_rocm_home_none("flash_attn")
        archs = os.getenv("GPU_ARCHS", "native").split(";")
        validate_and_update_archs(archs)

        if archs != ['native']:
            cc_flag = [f"--offload-arch={arch}" for arch in archs]
        else:
            arch = torch.cuda.get_device_properties("cuda").gcnArchName.split(":")[0]
            cc_flag = [f"--offload-arch={arch}"]

        # HACK: The compiler flag -D_GLIBCXX_USE_CXX11_ABI is set to be the same as
        # torch._C._GLIBCXX_USE_CXX11_ABI
        # https://github.com/pytorch/pytorch/blob/8472c24e3b5b60150096486616d98b7bea01500b/torch/utils/cpp_extension.py#L920
        if FORCE_CXX11_ABI:
            torch._C._GLIBCXX_USE_CXX11_ABI = True

        sources = ["csrc/flash_attn_ck/flash_api.cpp",
                "csrc/flash_attn_ck/flash_common.cpp",
                "csrc/flash_attn_ck/mha_bwd.cpp",
                "csrc/flash_attn_ck/mha_fwd_kvcache.cpp",
                "csrc/flash_attn_ck/mha_fwd.cpp",
                "csrc/flash_attn_ck/mha_varlen_bwd.cpp",
                "csrc/flash_attn_ck/mha_varlen_fwd.cpp"] + glob.glob(
            f"build/fmha_*wd*.cpp"
        )

        rename_cpp_to_cu(sources)

        renamed_sources = ["csrc/flash_attn_ck/flash_api.cu",
                        "csrc/flash_attn_ck/flash_common.cu",
                        "csrc/flash_attn_ck/mha_bwd.cu",
                        "csrc/flash_attn_ck/mha_fwd_kvcache.cu",
                        "csrc/flash_attn_ck/mha_fwd.cu",
                        "csrc/flash_attn_ck/mha_varlen_bwd.cu",
                        "csrc/flash_attn_ck/mha_varlen_fwd.cu"] + glob.glob(f"build/fmha_*wd*.cu")

        cc_flag += ["-O3","-std=c++17",
                    "-DCK_TILE_FMHA_FWD_FAST_EXP2=1",
                    "-fgpu-flush-denormals-to-zero",
                    "-DCK_ENABLE_BF16",
                    "-DCK_ENABLE_BF8",
                    "-DCK_ENABLE_FP16",
                    "-DCK_ENABLE_FP32",
                    "-DCK_ENABLE_FP64",
                    "-DCK_ENABLE_FP8",
                    "-DCK_ENABLE_INT8",
                    "-DCK_USE_XDL",
                    "-DUSE_PROF_API=1",
                    # "-DFLASHATTENTION_DISABLE_BACKWARD",
                    "-D__HIP_PLATFORM_HCC__=1"]

        cc_flag += [f"-DCK_TILE_FLOAT_TO_BFLOAT16_DEFAULT={os.environ.get('CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT', 3)}"]

        # Imitate https://github.com/ROCm/composable_kernel/blob/c8b6b64240e840a7decf76dfaa13c37da5294c4a/CMakeLists.txt#L190-L214
        hip_version = get_hip_version()
        if hip_version > Version('5.5.00000'):
            cc_flag += ["-mllvm", "--lsr-drop-solution=1"]
        if hip_version > Version('5.7.23302'):
            cc_flag += ["-fno-offload-uniform-block"]
        if hip_version > Version('6.1.40090'):
            cc_flag += ["-mllvm", "-enable-post-misched=0"]
        if hip_version > Version('6.2.41132'):
            cc_flag += ["-mllvm", "-amdgpu-early-inline-all=true",
                        "-mllvm", "-amdgpu-function-calls=false"]
        if hip_version > Version('6.2.41133') and hip_version < Version('6.3.00000'):
            cc_flag += ["-mllvm", "-amdgpu-coerce-illegal-types=1"]

        extra_compile_args = {
            "cxx": ["-O3", "-std=c++17"] + generator_flag,
            "nvcc": cc_flag + generator_flag,
        }

        include_dirs = [
            Path(this_dir) / "csrc" / "composable_kernel" / "include",
            Path(this_dir) / "csrc" / "composable_kernel" / "library" / "include",
            Path(this_dir) / "csrc" / "composable_kernel" / "example" / "ck_tile" / "01_fmha",
        ]

        ext_modules.append(
            CUDAExtension(
                name="flash_attn_2_cuda",
                sources=renamed_sources,
                extra_compile_args=extra_compile_args,
                include_dirs=include_dirs,
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


def get_wheel_url():
    torch_version_raw = parse(torch.__version__)
    python_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
    platform_name = get_platform()
    flash_version = get_package_version()
    torch_version = f"{torch_version_raw.major}.{torch_version_raw.minor}"
    cxx11_abi = str(torch._C._GLIBCXX_USE_CXX11_ABI).upper()

    if IS_ROCM:
        torch_hip_version = get_hip_version()
        hip_version = f"{torch_hip_version.major}{torch_hip_version.minor}"
        wheel_filename = f"{PACKAGE_NAME}-{flash_version}+rocm{hip_version}torch{torch_version}cxx11abi{cxx11_abi}-{python_version}-{python_version}-{platform_name}.whl"
    else:
        # Determine the version numbers that will be used to determine the correct wheel
        # We're using the CUDA version used to build torch, not the one currently installed
        # _, cuda_version_raw = get_cuda_bare_metal_version(CUDA_HOME)
        torch_cuda_version = parse(torch.version.cuda)
        # For CUDA 11, we only compile for CUDA 11.8, and for CUDA 12 we only compile for CUDA 12.3
        # to save CI time. Minor versions should be compatible.
        torch_cuda_version = parse("11.8") if torch_cuda_version.major == 11 else parse("12.3")
        # cuda_version = f"{cuda_version_raw.major}{cuda_version_raw.minor}"
        cuda_version = f"{torch_cuda_version.major}"

        # Determine wheel URL based on CUDA version, torch version, python version and OS
        wheel_filename = f"{PACKAGE_NAME}-{flash_version}+cu{cuda_version}torch{torch_version}cxx11abi{cxx11_abi}-{python_version}-{python_version}-{platform_name}.whl"

    wheel_url = BASE_WHEEL_URL.format(tag_name=f"v{flash_version}", wheel_name=wheel_filename)

    return wheel_url, wheel_filename


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

        wheel_url, wheel_filename = get_wheel_url()
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
        except (urllib.error.HTTPError, urllib.error.URLError):
            print("Precompiled wheel not found. Building from source...")
            # If the wheel could not be downloaded, build from source
            super().run()


class NinjaBuildExtension(BuildExtension):
    def __init__(self, *args, **kwargs) -> None:
        # do not override env MAX_JOBS if already exists
        if not os.environ.get("MAX_JOBS"):
            import psutil

            # calculate the maximum allowed NUM_JOBS based on cores
            max_num_jobs_cores = max(1, os.cpu_count() // 2)

            # calculate the maximum allowed NUM_JOBS based on free memory
            free_memory_gb = psutil.virtual_memory().available / (1024 ** 3)  # free memory in GB
            max_num_jobs_memory = int(free_memory_gb / 9)  # each JOB peak memory cost is ~8-9GB when threads = 4

            # pick lower value of jobs based on cores vs memory metric to minimize oom and swap usage during compilation
            max_jobs = max(1, min(max_num_jobs_cores, max_num_jobs_memory))
            os.environ["MAX_JOBS"] = str(max_jobs)

        super().__init__(*args, **kwargs)


setup(
    name=PACKAGE_NAME,
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
    author_email="tri@tridao.me",
    description="Flash Attention: Fast and Memory-Efficient Exact Attention",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Dao-AILab/flash-attention",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Unix",
    ],
    ext_modules=ext_modules,
    cmdclass={"bdist_wheel": CachedWheelsCommand, "build_ext": NinjaBuildExtension}
    if ext_modules
    else {
        "bdist_wheel": CachedWheelsCommand,
    },
    python_requires=">=3.9",
    install_requires=[
        "torch",
        "einops",
    ],
    setup_requires=[
        "packaging",
        "psutil",
        "ninja",
    ],
)
