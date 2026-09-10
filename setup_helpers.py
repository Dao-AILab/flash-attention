"""CUDA build helpers used by ``setup.py`` and its tests."""

import os
import subprocess
from collections.abc import Iterable, MutableSequence
from pathlib import Path
from typing import Union

from packaging.version import Version


def get_cuda_supported_archs(
    cuda_dir: Union[str, os.PathLike[str]],
) -> set[str]:
    """Return the virtual GPU architectures accepted by this nvcc."""
    nvcc = Path(cuda_dir) / "bin" / "nvcc"
    output = subprocess.check_output(
        [str(nvcc), "--list-gpu-arch"], universal_newlines=True
    )
    supported_archs = {
        target.removeprefix("compute_")
        for target in output.split()
        if target.startswith("compute_")
    }
    if not supported_archs:
        raise RuntimeError(f"{nvcc} did not report any supported GPU architectures")
    return supported_archs


def add_cuda_gencodes(
    cc_flag: MutableSequence[str],
    archs: Iterable[str],
    bare_metal_version: Version,
    supported_archs: Iterable[str],
) -> MutableSequence[str]:
    """Append only the SASS and PTX targets supported by the detected nvcc.

    ``FLASH_ATTN_CUDA_ARCHS`` describes desired architectures, but an older
    toolkit cannot compile every requested target.  The PTX target must
    therefore be selected from the architectures that nvcc can actually
    emit, rather than directly from the requested set.
    """
    archs = set(archs)
    supported_archs = set(supported_archs)
    # Preserve custom numeric targets such as ``86`` that the old helper
    # emitted as PTX-only, provided the detected nvcc actually accepts them.
    ptx_archs = {arch for arch in archs if arch.isdigit() and arch in supported_archs}

    def add_sass(virtual_arch: str, real_arch: str, ptx_arch: str) -> None:
        cc_flag.extend(["-gencode", f"arch=compute_{virtual_arch},code=sm_{real_arch}"])
        ptx_archs.add(ptx_arch)

    if "80" in archs and "80" in supported_archs:
        add_sass("80", "80", "80")

    if "90" in archs and "90" in supported_archs:
        add_sass("90", "90", "90")

    if bare_metal_version >= Version("12.8"):
        if "100" in archs and "100" in supported_archs:
            add_sass(
                "100f" if bare_metal_version >= Version("12.9") else "100",
                "100",
                "100",
            )

        if "120" in archs and "120" in supported_archs:
            add_sass(
                "120f" if bare_metal_version >= Version("12.9") else "120",
                "120",
                "120",
            )

        if "110" in archs:
            if "110" in supported_archs:
                add_sass("110f", "110", "110")
            elif "101" in supported_archs:
                # CUDA 12.8/12.9 still uses the pre-release Thor name.
                add_sass("101", "101", "101")

    if ptx_archs:
        newest = max(ptx_archs, key=int)
        cc_flag.extend(["-gencode", f"arch=compute_{newest},code=compute_{newest}"])
    elif archs:
        requested = ", ".join(sorted(archs))
        raise RuntimeError(
            f"CUDA {bare_metal_version} cannot compile any requested "
            f"FLASH_ATTN_CUDA_ARCHS target ({requested})"
        )

    return cc_flag
