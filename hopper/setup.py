# Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.

import sys
import warnings
import os
import stat
import re
import shutil
import ast
from pathlib import Path
from packaging.version import parse, Version
import platform
import sysconfig
import tarfile
import itertools

from setuptools import setup, find_packages
import subprocess

import urllib.request
import urllib.error
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME


# with open("../README.md", "r", encoding="utf-8") as fh:
with open("../README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))

PACKAGE_NAME = "flash_attn_3"

BASE_WHEEL_URL = "https://github.com/Dao-AILab/flash-attention/releases/download/{tag_name}/{wheel_name}"

# FORCE_BUILD: Force a fresh build locally, instead of attempting to find prebuilt wheels
# SKIP_CUDA_BUILD: Intended to allow CI to use a simple `python setup.py sdist` run to copy over raw files, without any cuda compilation
FORCE_BUILD = os.getenv("FLASH_ATTENTION_FORCE_BUILD", "FALSE") == "TRUE"
SKIP_CUDA_BUILD = os.getenv("FLASH_ATTENTION_SKIP_CUDA_BUILD", "FALSE") == "TRUE"
# For CI, we want the option to build with C++11 ABI since the nvcr images use C++11 ABI
FORCE_CXX11_ABI = os.getenv("FLASH_ATTENTION_FORCE_CXX11_ABI", "FALSE") == "TRUE"

DISABLE_BACKWARD = os.getenv("FLASH_ATTENTION_DISABLE_BACKWARD", "FALSE") == "TRUE"
DISABLE_SPLIT = os.getenv("FLASH_ATTENTION_DISABLE_SPLIT", "FALSE") == "TRUE"
DISABLE_PAGEDKV = os.getenv("FLASH_ATTENTION_DISABLE_PAGEDKV", "FALSE") == "TRUE"
DISABLE_APPENDKV = os.getenv("FLASH_ATTENTION_DISABLE_APPENDKV", "FALSE") == "TRUE"
DISABLE_LOCAL = os.getenv("FLASH_ATTENTION_DISABLE_LOCAL", "FALSE") == "TRUE"
DISABLE_SOFTCAP = os.getenv("FLASH_ATTENTION_DISABLE_SOFTCAP", "FALSE") == "TRUE"
DISABLE_PACKGQA = os.getenv("FLASH_ATTENTION_DISABLE_PACKGQA", "FALSE") == "TRUE"
DISABLE_FP16 = os.getenv("FLASH_ATTENTION_DISABLE_FP16", "FALSE") == "TRUE"
DISABLE_FP8 = os.getenv("FLASH_ATTENTION_DISABLE_FP8", "FALSE") == "TRUE"
DISABLE_VARLEN = os.getenv("FLASH_ATTENTION_DISABLE_VARLEN", "FALSE") == "TRUE"
DISABLE_CLUSTER = os.getenv("FLASH_ATTENTION_DISABLE_CLUSTER", "FALSE") == "TRUE"
DISABLE_HDIM64 = os.getenv("FLASH_ATTENTION_DISABLE_HDIM64", "FALSE") == "TRUE"
DISABLE_HDIM96 = os.getenv("FLASH_ATTENTION_DISABLE_HDIM96", "FALSE") == "TRUE"
DISABLE_HDIM128 = os.getenv("FLASH_ATTENTION_DISABLE_HDIM128", "FALSE") == "TRUE"
DISABLE_HDIM192 = os.getenv("FLASH_ATTENTION_DISABLE_HDIM192", "FALSE") == "TRUE"
DISABLE_HDIM256 = os.getenv("FLASH_ATTENTION_DISABLE_HDIM256", "FALSE") == "TRUE"
DISABLE_SM8x = os.getenv("FLASH_ATTENTION_DISABLE_SM80", "FALSE") == "TRUE"

ENABLE_VCOLMAJOR = os.getenv("FLASH_ATTENTION_ENABLE_VCOLMAJOR", "FALSE") == "TRUE"

DISABLE_HDIMDIFF64 = os.getenv("FLASH_ATTENTION_DISABLE_HDIMDIFF64", "FALSE") == "TRUE"
DISABLE_HDIMDIFF192 = os.getenv("FLASH_ATTENTION_DISABLE_HDIMDIFF192", "FALSE") == "TRUE"

# HACK: we monkey patch pytorch's _write_ninja_file to pass
# "-gencode arch=compute_sm90a,code=sm_90a" to files ending in '_sm90.cu',
# and pass "-gencode arch=compute_sm80,code=sm_80" to files ending in '_sm80.cu'
from torch.utils.cpp_extension import (
    IS_HIP_EXTENSION,
    COMMON_HIP_FLAGS,
    SUBPROCESS_DECODE_ARGS,
    IS_WINDOWS,
    get_cxx_compiler,
    _join_rocm_home,
    _join_cuda_home,
    _is_cuda_file,
    _maybe_write,
)

def _write_ninja_file(path,
                      cflags,
                      post_cflags,
                      cuda_cflags,
                      cuda_post_cflags,
                      cuda_dlink_post_cflags,
                      sources,
                      objects,
                      ldflags,
                      library_target,
                      with_cuda,
                      **kwargs,  # kwargs (ignored) to absorb new flags in torch.utils.cpp_extension
                      ) -> None:
    r"""Write a ninja file that does the desired compiling and linking.

    `path`: Where to write this file
    `cflags`: list of flags to pass to $cxx. Can be None.
    `post_cflags`: list of flags to append to the $cxx invocation. Can be None.
    `cuda_cflags`: list of flags to pass to $nvcc. Can be None.
    `cuda_postflags`: list of flags to append to the $nvcc invocation. Can be None.
    `sources`: list of paths to source files
    `objects`: list of desired paths to objects, one per source.
    `ldflags`: list of flags to pass to linker. Can be None.
    `library_target`: Name of the output library. Can be None; in that case,
                      we do no linking.
    `with_cuda`: If we should be compiling with CUDA.
    """
    def sanitize_flags(flags):
        if flags is None:
            return []
        else:
            return [flag.strip() for flag in flags]

    cflags = sanitize_flags(cflags)
    post_cflags = sanitize_flags(post_cflags)
    cuda_cflags = sanitize_flags(cuda_cflags)
    cuda_post_cflags = sanitize_flags(cuda_post_cflags)
    cuda_dlink_post_cflags = sanitize_flags(cuda_dlink_post_cflags)
    ldflags = sanitize_flags(ldflags)

    # Sanity checks...
    assert len(sources) == len(objects)
    assert len(sources) > 0

    compiler = get_cxx_compiler()

    # Version 1.3 is required for the `deps` directive.
    config = ['ninja_required_version = 1.3']
    config.append(f'cxx = {compiler}')
    if with_cuda or cuda_dlink_post_cflags:
        if IS_HIP_EXTENSION:
            nvcc = _join_rocm_home('bin', 'hipcc')
        else:
            nvcc = _join_cuda_home('bin', 'nvcc')
        if "PYTORCH_NVCC" in os.environ:
            nvcc_from_env = os.getenv("PYTORCH_NVCC")    # user can set nvcc compiler with ccache using the environment variable here
        else:
            nvcc_from_env = nvcc
        config.append(f'nvcc_from_env = {nvcc_from_env}')
        config.append(f'nvcc = {nvcc}')

    if IS_HIP_EXTENSION:
        post_cflags = COMMON_HIP_FLAGS + post_cflags
    flags = [f'cflags = {" ".join(cflags)}']
    flags.append(f'post_cflags = {" ".join(post_cflags)}')
    if with_cuda:
        flags.append(f'cuda_cflags = {" ".join(cuda_cflags)}')
        flags.append(f'cuda_post_cflags = {" ".join(cuda_post_cflags)}')
        cuda_post_cflags_sm80 = [s if s != 'arch=compute_90a,code=sm_90a' else 'arch=compute_80,code=sm_80' for s in cuda_post_cflags]
        flags.append(f'cuda_post_cflags_sm80 = {" ".join(cuda_post_cflags_sm80)}')
        cuda_post_cflags_sm80_sm90 = cuda_post_cflags + ['-gencode', 'arch=compute_80,code=sm_80']
        flags.append(f'cuda_post_cflags_sm80_sm90 = {" ".join(cuda_post_cflags_sm80_sm90)}')
        cuda_post_cflags_sm100 = [s if s != 'arch=compute_90a,code=sm_90a' else 'arch=compute_100a,code=sm_100a' for s in cuda_post_cflags]
        flags.append(f'cuda_post_cflags_sm100 = {" ".join(cuda_post_cflags_sm100)}')
    flags.append(f'cuda_dlink_post_cflags = {" ".join(cuda_dlink_post_cflags)}')
    flags.append(f'ldflags = {" ".join(ldflags)}')

    # Turn into absolute paths so we can emit them into the ninja build
    # file wherever it is.
    sources = [os.path.abspath(file) for file in sources]

    # See https://ninja-build.org/build.ninja.html for reference.
    compile_rule = ['rule compile']
    if IS_WINDOWS:
        compile_rule.append(
            '  command = cl /showIncludes $cflags -c $in /Fo$out $post_cflags')
        compile_rule.append('  deps = msvc')
    else:
        compile_rule.append(
            '  command = $cxx -MMD -MF $out.d $cflags -c $in -o $out $post_cflags')
        compile_rule.append('  depfile = $out.d')
        compile_rule.append('  deps = gcc')

    if with_cuda:
        cuda_compile_rule = ['rule cuda_compile']
        nvcc_gendeps = ''
        # --generate-dependencies-with-compile is not supported by ROCm
        # Nvcc flag `--generate-dependencies-with-compile` is not supported by sccache, which may increase build time.
        if torch.version.cuda is not None and os.getenv('TORCH_EXTENSION_SKIP_NVCC_GEN_DEPENDENCIES', '0') != '1':
            cuda_compile_rule.append('  depfile = $out.d')
            cuda_compile_rule.append('  deps = gcc')
            # Note: non-system deps with nvcc are only supported
            # on Linux so use --generate-dependencies-with-compile
            # to make this work on Windows too.
            nvcc_gendeps = '--generate-dependencies-with-compile --dependency-output $out.d'
        cuda_compile_rule_sm80 = ['rule cuda_compile_sm80'] + cuda_compile_rule[1:] + [
            f'  command = $nvcc_from_env {nvcc_gendeps} $cuda_cflags -c $in -o $out $cuda_post_cflags_sm80'
        ]
        cuda_compile_rule_sm80_sm90 = ['rule cuda_compile_sm80_sm90'] + cuda_compile_rule[1:] + [
            f'  command = $nvcc_from_env {nvcc_gendeps} $cuda_cflags -c $in -o $out $cuda_post_cflags_sm80_sm90'
        ]
        cuda_compile_rule_sm100 = ['rule cuda_compile_sm100'] + cuda_compile_rule[1:] + [
            f'  command = $nvcc_from_env {nvcc_gendeps} $cuda_cflags -c $in -o $out $cuda_post_cflags_sm100'
        ]
        cuda_compile_rule.append(
            f'  command = $nvcc_from_env {nvcc_gendeps} $cuda_cflags -c $in -o $out $cuda_post_cflags')

    # Emit one build rule per source to enable incremental build.
    build = []
    for source_file, object_file in zip(sources, objects):
        is_cuda_source = _is_cuda_file(source_file) and with_cuda
        if is_cuda_source:
            if source_file.endswith('_sm90.cu'):
                rule = 'cuda_compile'
            elif source_file.endswith('_sm80.cu'):
                rule = 'cuda_compile_sm80'
            elif source_file.endswith('_sm100.cu'):
                rule = 'cuda_compile_sm100'
            else:
                rule = 'cuda_compile_sm80_sm90'
        else:
            rule = 'compile'
        if IS_WINDOWS:
            source_file = source_file.replace(':', '$:')
            object_file = object_file.replace(':', '$:')
        source_file = source_file.replace(" ", "$ ")
        object_file = object_file.replace(" ", "$ ")
        build.append(f'build {object_file}: {rule} {source_file}')

    if cuda_dlink_post_cflags:
        devlink_out = os.path.join(os.path.dirname(objects[0]), 'dlink.o')
        devlink_rule = ['rule cuda_devlink']
        devlink_rule.append('  command = $nvcc $in -o $out $cuda_dlink_post_cflags')
        devlink = [f'build {devlink_out}: cuda_devlink {" ".join(objects)}']
        objects += [devlink_out]
    else:
        devlink_rule, devlink = [], []

    if library_target is not None:
        link_rule = ['rule link']
        if IS_WINDOWS:
            cl_paths = subprocess.check_output(['where',
                                                'cl']).decode(*SUBPROCESS_DECODE_ARGS).split('\r\n')
            if len(cl_paths) >= 1:
                cl_path = os.path.dirname(cl_paths[0]).replace(':', '$:')
            else:
                raise RuntimeError("MSVC is required to load C++ extensions")
            link_rule.append(f'  command = "{cl_path}/link.exe" $in /nologo $ldflags /out:$out')
        else:
            link_rule.append('  command = $cxx $in $ldflags -o $out')

        link = [f'build {library_target}: link {" ".join(objects)}']

        default = [f'default {library_target}']
    else:
        link_rule, link, default = [], [], []

    # 'Blocks' should be separated by newlines, for visual benefit.
    blocks = [config, flags, compile_rule]
    if with_cuda:
        blocks.append(cuda_compile_rule)  # type: ignore[possibly-undefined]
        blocks.append(cuda_compile_rule_sm80)  # type: ignore[possibly-undefined]
        blocks.append(cuda_compile_rule_sm80_sm90)  # type: ignore[possibly-undefined]
        blocks.append(cuda_compile_rule_sm100)  # type: ignore[possibly-undefined]
    blocks += [devlink_rule, link_rule, build, devlink, link, default]
    content = "\n\n".join("\n".join(b) for b in blocks)
    # Ninja requires a new lines at the end of the .ninja file
    content += "\n"
    _maybe_write(path, content)


# Monkey patching
torch.utils.cpp_extension._write_ninja_file = _write_ninja_file


def get_platform():
    """
    Returns the platform name as used in wheel filenames.
    """
    if sys.platform.startswith("linux"):
        return "linux_x86_64"
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


# Taken from https://github.com/pytorch/pytorch/blob/master/tools/setup_helpers/env.py
def check_env_flag(name: str, default: str = "") -> bool:
    return os.getenv(name, default).upper() in ["ON", "1", "YES", "TRUE", "Y"]


# Copied from https://github.com/triton-lang/triton/blob/main/python/setup.py
def is_offline_build() -> bool:
    """
    Downstream projects and distributions which bootstrap their own dependencies from scratch
    and run builds in offline sandboxes
    may set `FLASH_ATTENTION_OFFLINE_BUILD` in the build environment to prevent any attempts at downloading
    pinned dependencies from the internet or at using dependencies vendored in-tree.

    Dependencies must be defined using respective search paths (cf. `syspath_var_name` in `Package`).
    Missing dependencies lead to an early abortion.
    Dependencies' compatibility is not verified.

    Note that this flag isn't tested by the CI and does not provide any guarantees.
    """
    return check_env_flag("FLASH_ATTENTION_OFFLINE_BUILD", "")


# Copied from https://github.com/triton-lang/triton/blob/main/python/setup.py
def get_flashattn_cache_path():
    user_home = os.getenv("FLASH_ATTENTION_HOME")
    if not user_home:
        user_home = os.getenv("HOME") or os.getenv("USERPROFILE") or os.getenv("HOMEPATH") or None
    if not user_home:
        raise RuntimeError("Could not find user home directory")
    return os.path.join(user_home, ".flashattn")


def open_url(url):
    user_agent = 'Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/119.0'
    headers = {
        'User-Agent': user_agent,
    }
    request = urllib.request.Request(url, None, headers)
    # Set timeout to 300 seconds to prevent the request from hanging forever.
    return urllib.request.urlopen(request, timeout=300)


def download_and_copy(name, src_func, dst_path, version, url_func):
    if is_offline_build():
        return
    flashattn_cache_path = get_flashattn_cache_path()
    base_dir = os.path.dirname(__file__)
    system = platform.system()
    arch = platform.machine()
    arch = {"arm64": "aarch64"}.get(arch, arch)
    supported = {"Linux": "linux", "Darwin": "linux"}
    url = url_func(supported[system], arch, version)
    src_path = src_func(supported[system], arch, version)
    tmp_path = os.path.join(flashattn_cache_path, "nvidia", name)  # path to cache the download
    dst_path = os.path.join(base_dir, os.pardir, "third_party", "nvidia", "backend", dst_path)  # final binary path
    src_path = os.path.join(tmp_path, src_path)
    download = not os.path.exists(src_path)
    if download:
        print(f'downloading and extracting {url} ...')
        file = tarfile.open(fileobj=open_url(url), mode="r|*")
        file.extractall(path=tmp_path)
    os.makedirs(os.path.split(dst_path)[0], exist_ok=True)
    print(f'copy {src_path} to {dst_path} ...')
    if os.path.isdir(src_path):
        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
    else:
        shutil.copy(src_path, dst_path)


def nvcc_threads_args():
    nvcc_threads = os.getenv("NVCC_THREADS") or "2"
    return ["--threads", nvcc_threads]


# NVIDIA_TOOLCHAIN_VERSION = {"nvcc": "12.3.107"}
NVIDIA_TOOLCHAIN_VERSION = {"nvcc": "12.6.85", "ptxas": "12.8.93"}

exe_extension = sysconfig.get_config_var("EXE")


cmdclass = {}
ext_modules = []

# We want this even if SKIP_CUDA_BUILD because when we run python setup.py sdist we want the .hpp
# files included in the source distribution, in case the user compiles from source.
subprocess.run(["git", "submodule", "update", "--init", "../csrc/cutlass"])

if not SKIP_CUDA_BUILD:
    print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
    TORCH_MAJOR = int(torch.__version__.split(".")[0])
    TORCH_MINOR = int(torch.__version__.split(".")[1])

    check_if_cuda_home_none(PACKAGE_NAME)
    _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
    if bare_metal_version < Version("12.3"):
        raise RuntimeError("FlashAttention-3 is only supported on CUDA 12.3 and above")
    elif bare_metal_version >= Version("13.0"):
        # CUDA 13.0+ uses system nvcc and CCCL headers are in /usr/local/cuda/include/cccl/
        cccl_include = os.path.join(CUDA_HOME, "include", "cccl")
        for env_var in ["CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH"]:
            current = os.environ.get(env_var, "")
            os.environ[env_var] = cccl_include + (":" + current if current else "")

    # ptxas 12.8 gives the best perf currently
    # We want to use the nvcc front end from 12.6 however, since if we use nvcc 12.8
    # Cutlass 3.8 will expect the new data types in cuda.h from CTK 12.8, which we don't have.
    # For CUDA 13.0+, use system nvcc instead of downloading CUDA 12.x toolchain
    if bare_metal_version >= Version("12.3") and bare_metal_version < Version("13.0") and bare_metal_version != Version("12.8"):
        download_and_copy(
            name="nvcc",
            src_func=lambda system, arch, version: f"cuda_nvcc-{system}-{arch}-{version}-archive/bin",
            dst_path="bin",
            version=NVIDIA_TOOLCHAIN_VERSION["nvcc"],
            url_func=lambda system, arch, version:
            f"https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvcc/{system}-{arch}/cuda_nvcc-{system}-{arch}-{version}-archive.tar.xz",
        )
        download_and_copy(
            name="ptxas",
            src_func=lambda system, arch, version: f"cuda_nvcc-{system}-{arch}-{version}-archive/bin/ptxas",
            dst_path="bin",
            version=NVIDIA_TOOLCHAIN_VERSION["ptxas"],
            url_func=lambda system, arch, version:
            f"https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvcc/{system}-{arch}/cuda_nvcc-{system}-{arch}-{version}-archive.tar.xz",
        )
        download_and_copy(
            name="ptxas",
            src_func=lambda system, arch, version: f"cuda_nvcc-{system}-{arch}-{version}-archive/nvvm/bin",
            dst_path="nvvm/bin",
            version=NVIDIA_TOOLCHAIN_VERSION["ptxas"],
            url_func=lambda system, arch, version:
            f"https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvcc/{system}-{arch}/cuda_nvcc-{system}-{arch}-{version}-archive.tar.xz",
        )
        base_dir = os.path.dirname(__file__)
        ctk_path_new = os.path.abspath(os.path.join(base_dir, os.pardir, "third_party", "nvidia", "backend", "bin"))
        nvcc_path_new = os.path.join(ctk_path_new, f"nvcc{exe_extension}")
        # Need to append to path otherwise nvcc can't find cicc in nvvm/bin/cicc
        # nvcc 12.8 seems to hard-code looking for cicc in ../nvvm/bin/cicc
        os.environ["PATH"] = ctk_path_new + os.pathsep + os.environ["PATH"]
        os.environ["PYTORCH_NVCC"] = nvcc_path_new
        # Make nvcc executable, sometimes after the copy it loses its permissions
        os.chmod(nvcc_path_new, os.stat(nvcc_path_new).st_mode | stat.S_IEXEC)

    cc_flag = []
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_90a,code=sm_90a")

    # HACK: The compiler flag -D_GLIBCXX_USE_CXX11_ABI is set to be the same as
    # torch._C._GLIBCXX_USE_CXX11_ABI
    # https://github.com/pytorch/pytorch/blob/8472c24e3b5b60150096486616d98b7bea01500b/torch/utils/cpp_extension.py#L920
    if FORCE_CXX11_ABI:
        torch._C._GLIBCXX_USE_CXX11_ABI = True
    repo_dir = Path(this_dir).parent
    cutlass_dir = repo_dir / "csrc" / "cutlass"

    feature_args = (
        []
        + (["-DFLASHATTENTION_DISABLE_BACKWARD"] if DISABLE_BACKWARD else [])
        + (["-DFLASHATTENTION_DISABLE_PAGEDKV"] if DISABLE_PAGEDKV else [])
        + (["-DFLASHATTENTION_DISABLE_SPLIT"] if DISABLE_SPLIT else [])
        + (["-DFLASHATTENTION_DISABLE_APPENDKV"] if DISABLE_APPENDKV else [])
        + (["-DFLASHATTENTION_DISABLE_LOCAL"] if DISABLE_LOCAL else [])
        + (["-DFLASHATTENTION_DISABLE_SOFTCAP"] if DISABLE_SOFTCAP else [])
        + (["-DFLASHATTENTION_DISABLE_PACKGQA"] if DISABLE_PACKGQA else [])
        + (["-DFLASHATTENTION_DISABLE_FP16"] if DISABLE_FP16 else [])
        + (["-DFLASHATTENTION_DISABLE_FP8"] if DISABLE_FP8 else [])
        + (["-DFLASHATTENTION_DISABLE_VARLEN"] if DISABLE_VARLEN else [])
        + (["-DFLASHATTENTION_DISABLE_CLUSTER"] if DISABLE_CLUSTER else [])
        + (["-DFLASHATTENTION_DISABLE_HDIM64"] if DISABLE_HDIM64 else [])
        + (["-DFLASHATTENTION_DISABLE_HDIM96"] if DISABLE_HDIM96 else [])
        + (["-DFLASHATTENTION_DISABLE_HDIM128"] if DISABLE_HDIM128 else [])
        + (["-DFLASHATTENTION_DISABLE_HDIM192"] if DISABLE_HDIM192 else [])
        + (["-DFLASHATTENTION_DISABLE_HDIM256"] if DISABLE_HDIM256 else [])
        + (["-DFLASHATTENTION_DISABLE_SM8x"] if DISABLE_SM8x else [])
        + (["-DFLASHATTENTION_ENABLE_VCOLMAJOR"] if ENABLE_VCOLMAJOR else [])
        + (["-DFLASHATTENTION_DISABLE_HDIMDIFF64"] if DISABLE_HDIMDIFF64 else [])
        + (["-DFLASHATTENTION_DISABLE_HDIMDIFF192"] if DISABLE_HDIMDIFF192 else [])
    )

    DTYPE_FWD_SM80 = ["bf16"] + (["fp16"] if not DISABLE_FP16 else [])
    DTYPE_FWD_SM90 = ["bf16"] + (["fp16"] if not DISABLE_FP16 else []) + (["e4m3"] if not DISABLE_FP8 else [])
    HALF_DTYPE_FWD_SM90 = ["bf16"] + (["fp16"] if not DISABLE_FP16 else [])
    DTYPE_BWD = ["bf16"] + (["fp16"] if not DISABLE_FP16 else [])
    HEAD_DIMENSIONS_BWD = (
        []
        + ([64] if not DISABLE_HDIM64 else [])
        + ([96] if not DISABLE_HDIM96 else [])
        + ([128] if not DISABLE_HDIM128 else [])
        + ([192] if not DISABLE_HDIM192 else [])
        + ([256] if not DISABLE_HDIM256 else [])
    )
    # build will now explode with this compilation grouping given all our templating
    # HEAD_DIMENSIONS_FWD = ["all", "diff"]
    HEAD_DIMENSIONS_FWD = HEAD_DIMENSIONS_BWD
    HEAD_DIMENSIONS_DIFF64_FWD = (
        []
        + (["64_256"] if not DISABLE_HDIMDIFF64 else [])
        + (["64_512"] if not DISABLE_HDIMDIFF64 else [])
    )
    HEAD_DIMENSIONS_DIFF192_FWD = (
        []
        + (["192_128"] if not DISABLE_HDIMDIFF192 else [])
    )
    HEAD_DIMENSIONS_FWD_SM80 = HEAD_DIMENSIONS_BWD
    SPLIT = [""] + (["_split"] if not DISABLE_SPLIT else [])
    PAGEDKV = [""] + (["_paged"] if not DISABLE_PAGEDKV else [])
    SOFTCAP = [""] + (["_softcap"] if not DISABLE_SOFTCAP else [])
    SOFTCAP_ALL = [""] if DISABLE_SOFTCAP else ["_softcapall"]
    PACKGQA = [""] + (["_packgqa"] if not DISABLE_PACKGQA else [])
    # We already always hard-code PackGQA=true for Sm8x
    sources_fwd_sm80 = [f"instantiations/flash_fwd_hdim{hdim}_{dtype}{paged}{split}{softcap}_sm80.cu"
                        for hdim, dtype, split, paged, softcap in itertools.product(HEAD_DIMENSIONS_FWD_SM80, DTYPE_FWD_SM80, SPLIT, PAGEDKV, SOFTCAP_ALL)]
    # We already always hard-code PackGQA=true for Sm9x if PagedKV or Split
    sources_fwd_sm90 = [f"instantiations/flash_fwd_hdim{hdim}_{dtype}{paged}{split}{softcap}{packgqa}_sm90.cu"
                        for hdim, dtype, split, paged, softcap, packgqa in itertools.product(HEAD_DIMENSIONS_FWD, DTYPE_FWD_SM90, SPLIT, PAGEDKV, SOFTCAP, PACKGQA)
                        if not (packgqa and (paged or split))]
    if not DISABLE_HDIMDIFF64:
        sources_fwd_sm90 += [f"instantiations/flash_fwd_hdim{hdim}_{dtype}{paged}{split}{softcap}{packgqa}_sm90.cu"
                             for hdim, dtype, split, paged, softcap, packgqa in itertools.product(HEAD_DIMENSIONS_DIFF64_FWD, HALF_DTYPE_FWD_SM90, SPLIT, PAGEDKV, SOFTCAP, PACKGQA)
                             if not (packgqa and (paged or split))]
    if not DISABLE_HDIMDIFF192:
        sources_fwd_sm90 += [f"instantiations/flash_fwd_hdim{hdim}_{dtype}{paged}{split}{softcap}{packgqa}_sm90.cu"
                            for hdim, dtype, split, paged, softcap, packgqa in itertools.product(HEAD_DIMENSIONS_DIFF192_FWD, DTYPE_FWD_SM90, SPLIT, PAGEDKV, SOFTCAP, PACKGQA)
                            if not (packgqa and (paged or split))]
    sources_bwd_sm80 = [f"instantiations/flash_bwd_hdim{hdim}_{dtype}{softcap}_sm80.cu"
                        for hdim, dtype, softcap in itertools.product(HEAD_DIMENSIONS_BWD, DTYPE_BWD, SOFTCAP)]
    sources_bwd_sm90 = [f"instantiations/flash_bwd_hdim{hdim}_{dtype}{softcap}_sm90.cu"
                        for hdim, dtype, softcap in itertools.product(HEAD_DIMENSIONS_BWD, DTYPE_BWD, SOFTCAP_ALL)]
    if DISABLE_BACKWARD:
        sources_bwd_sm90 = []
        sources_bwd_sm80 = []
    
    # Choose between flash_api.cpp and flash_api_stable.cpp based on torch version
    torch_version = parse(torch.__version__)
    target_version = parse("2.9.0.dev20250830")
    stable_args = []
      
    if torch_version >= target_version:
        flash_api_source = "flash_api_stable.cpp"
        stable_args = ["-DTORCH_STABLE_ONLY"]  # Checks against including unstable Tensor APIs
    else:
        flash_api_source = "flash_api.cpp"

    sources = (
        [flash_api_source]
        + (sources_fwd_sm80 if not DISABLE_SM8x else []) + sources_fwd_sm90
        + (sources_bwd_sm80 if not DISABLE_SM8x else []) + sources_bwd_sm90
    )
    if not DISABLE_SPLIT:
        sources += ["flash_fwd_combine.cu"]
    sources += ["flash_prepare_scheduler.cu"]
    nvcc_flags = [
        "-O3",
        "-std=c++17",
        "--ftemplate-backtrace-limit=0",  # To debug template code
        "--use_fast_math",
        # "--keep",
        # "--ptxas-options=--verbose,--register-usage-level=5,--warn-on-local-memory-usage",  # printing out number of registers
        "--resource-usage",  # printing out number of registers
        # f"--split-compile={os.getenv('NVCC_THREADS', '4')}",  # split-compile is faster
        "-lineinfo",  # TODO: disable this for release to reduce binary size
        "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED",  # Necessary for the WGMMA shapes that we use
        "-DCUTLASS_ENABLE_GDC_FOR_SM90",  # For PDL
        "-DCUTLASS_DEBUG_TRACE_LEVEL=0",  # Can toggle for debugging
        "-DNDEBUG",  # Important, otherwise performance is severely impacted
    ]
    if get_platform() == "win_amd64":
        nvcc_flags.extend(
            [
                "-D_USE_MATH_DEFINES",  # for M_LN2
                "-Xcompiler=/Zc:__cplusplus",  # sets __cplusplus correctly, CUTLASS_CONSTEXPR_IF_CXX17 needed for cutlass::gcd
            ]
        )
    include_dirs = [
        Path(this_dir),
        cutlass_dir / "include",
    ]

    ext_modules.append(
        CUDAExtension(
            name=f"{PACKAGE_NAME}._C",
            sources=sources,
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17", "-DPy_LIMITED_API=0x03090000"] + stable_args + feature_args,
                "nvcc": nvcc_threads_args() + nvcc_flags + cc_flag + feature_args,
            },
            include_dirs=include_dirs,
            py_limited_api=True,
        )
    )


def get_package_version():
    with open(Path(this_dir) / "__init__.py", "r") as f:
        version_match = re.search(r"^__version__\s*=\s*(.*)$", f.read(), re.MULTILINE)
    public_version = ast.literal_eval(version_match.group(1))
    local_version = os.environ.get("FLASH_ATTN_LOCAL_VERSION")
    if local_version:
        return f"{public_version}+{local_version}"
    else:
        return str(public_version)


def get_wheel_url():
    # Determine the version numbers that will be used to determine the correct wheel
    # We're using the CUDA version used to build torch, not the one currently installed
    # _, cuda_version_raw = get_cuda_bare_metal_version(CUDA_HOME)
    torch_cuda_version = parse(torch.version.cuda)
    torch_version_raw = parse(torch.__version__)
    # For CUDA 11, we only compile for CUDA 11.8, and for CUDA 12 we only compile for CUDA 12.2
    # to save CI time. Minor versions should be compatible.
    torch_cuda_version = parse("11.8") if torch_cuda_version.major == 11 else parse("12.2")
    python_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
    platform_name = get_platform()
    package_version = get_package_version()
    # cuda_version = f"{cuda_version_raw.major}{cuda_version_raw.minor}"
    cuda_version = f"{torch_cuda_version.major}{torch_cuda_version.minor}"
    torch_version = f"{torch_version_raw.major}.{torch_version_raw.minor}"
    cxx11_abi = str(torch._C._GLIBCXX_USE_CXX11_ABI).upper()

    # Determine wheel URL based on CUDA version, torch version, python version and OS
    wheel_filename = f"{PACKAGE_NAME}-{package_version}+cu{cuda_version}torch{torch_version}cxx11abi{cxx11_abi}-{python_version}-{python_version}-{platform_name}.whl"
    wheel_url = BASE_WHEEL_URL.format(tag_name=f"v{package_version}", wheel_name=wheel_filename)
    return wheel_url, wheel_filename


class CachedWheelsCommand(_bdist_wheel):
    """
    The CachedWheelsCommand plugs into the default bdist wheel, which is ran by pip when it cannot
    find an existing wheel (which is currently the case for all installs). We use
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
            shutil.move(wheel_filename, wheel_path)
        except urllib.error.HTTPError:
            print("Precompiled wheel not found. Building from source...")
            # If the wheel could not be downloaded, build from source
            super().run()

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
        )
    ),
    py_modules=["flash_attn_interface"],
    description="FlashAttention-3",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: Unix",
    ],
    ext_modules=ext_modules,
    cmdclass={"bdist_wheel": CachedWheelsCommand, "build_ext": BuildExtension}
    if ext_modules
    else {
        "bdist_wheel": CachedWheelsCommand,
    },
    python_requires=">=3.8",
    install_requires=[
        "torch",
        "einops",
        "packaging",
        "ninja",
    ],
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
)
