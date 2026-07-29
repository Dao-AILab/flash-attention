# Copyright (c) 2025, Tri Dao.

import os
from typing import Tuple
from functools import lru_cache

import torch
from torch._subclasses.fake_tensor import FakeTensor

try:
    from triton.tools.disasm import extract
except ImportError:
    extract = None

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import NumericMeta
from cutlass.cute.runtime import from_dlpack

StaticTypes = (cutlass.Constexpr, NumericMeta, int, bool, str, float, type(None))


load_cubin_module_data_og = cutlass.base_dsl.runtime.cuda.load_cubin_module_data
cute_compile_og = cute.compile


torch2cute_dtype_map = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
    torch.float8_e4m3fn: cutlass.Float8E4M3FN,
    torch.float8_e5m2: cutlass.Float8E5M2,
}


@lru_cache
def get_max_active_clusters(cluster_size):
    return cutlass.utils.HardwareInfo().get_max_active_clusters(cluster_size=cluster_size)


@lru_cache
def get_device_capacity(device: torch.device = None) -> Tuple[int, int]:
    return torch.cuda.get_device_capability(device)


@lru_cache
def _get_device_arch_and_num_sms(device_index: int) -> tuple[int, int]:
    properties = torch.cuda.get_device_properties(device_index)
    return properties.major * 10 + properties.minor, properties.multi_processor_count


def get_num_sms_for_selection(device_index: int, arch: int) -> int:
    """Return the SM count of the matching local GPU or cross-compilation target."""
    override = os.getenv("FLASH_ATTENTION_NUM_SMS")
    if override is not None:
        num_sms = int(override)
        if num_sms <= 0:
            raise ValueError("FLASH_ATTENTION_NUM_SMS must be positive")
        return num_sms
    if torch.cuda.is_available():
        device_arch, num_sms = _get_device_arch_and_num_sms(device_index)
        if device_arch == arch:
            return num_sms
    raise RuntimeError(
        "Cannot determine the target GPU's SM count; set FLASH_ATTENTION_NUM_SMS "
        "when cross-compiling without a matching local GPU"
    )


def _has_aligned_pointer(tensor: torch.Tensor, align_bytes: int) -> bool:
    address = (
        tensor.storage_offset() * tensor.element_size()
        if isinstance(tensor, FakeTensor)
        else tensor.data_ptr()
    )
    return address % align_bytes == 0


def _is_aligned_layout(tensor: torch.Tensor, align_bytes: int) -> bool:
    """Return whether a tensor satisfies the pointer and stride ABI kernels assume."""
    if tensor.stride(-1) != 1 or not _has_aligned_pointer(tensor, align_bytes):
        return False
    stride_alignment = max(1, align_bytes // tensor.element_size())
    return all(stride == 0 or stride % stride_alignment == 0 for stride in tensor.stride()[:-1])


def maybe_contiguous(tensor: torch.Tensor | None, align_bytes: int = 16):
    """Canonicalize inputs to the pointer and stride alignment kernels assume."""
    if tensor is None:
        return None
    if tensor.is_contiguous():
        return (
            tensor
            if _has_aligned_pointer(tensor, align_bytes)
            else tensor.clone(memory_format=torch.contiguous_format)
        )
    if not _has_aligned_pointer(tensor, align_bytes):
        return tensor.clone(memory_format=torch.contiguous_format)
    return tensor if _is_aligned_layout(tensor, align_bytes) else tensor.contiguous()


def validate_output_layout(tensor: torch.Tensor, name: str, align_bytes: int) -> None:
    """Validate a caller-provided output or SplitKV workspace."""
    assert 0 not in tensor.stride(), f"{name} must not have broadcast dimensions"
    if tensor.is_contiguous():
        assert _has_aligned_pointer(tensor, align_bytes), (
            f"{name} must have aligned strides and a contiguous last dimension"
        )
        return
    assert _is_aligned_layout(tensor, align_bytes), (
        f"{name} must have aligned strides and a contiguous last dimension"
    )


def assume_strides_aligned(t):
    """Assume all strides except the last are divisible by 128 bits.

    Python int strides (e.g., stride=0 from GQA expand) are kept as-is
    since they're static and don't need alignment assumptions.
    """
    divby = 128 // t.element_type.width
    strides = tuple(s if isinstance(s, int) else cute.assume(s, divby=divby) for s in t.stride[:-1])
    return (*strides, t.stride[-1])


def assume_tensor_aligned(t):
    """Rebuild a tensor with 128-bit aligned stride assumptions. Passes through None."""
    if t is None:
        return None
    return cute.make_tensor(t.iterator, cute.make_layout(t.shape, stride=assume_strides_aligned(t)))


def to_cute_tensor(t, assumed_align=16, leading_dim=-1, fully_dynamic=False, enable_tvm_ffi=True):
    """Convert torch tensor to cute tensor for TVM FFI. leading_dim=-1 defaults to t.ndim-1."""
    if t is None:
        return None
    # NOTE: torch 2.9.1 doesn't support fp8 via DLPack but 2.11.0 nightly does
    # currently export raw bytes as uint8 and tell cutlass correct type
    # can directly export as fp8 when torch supports it
    if t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        tensor = from_dlpack(
            t.view(torch.uint8).detach(),
            assumed_align=assumed_align,
            enable_tvm_ffi=enable_tvm_ffi,
        )
        tensor.element_type = (
            cutlass.Float8E4M3FN if t.dtype == torch.float8_e4m3fn else cutlass.Float8E5M2
        )
    else:
        tensor = from_dlpack(t.detach(), assumed_align=assumed_align, enable_tvm_ffi=enable_tvm_ffi)
    if fully_dynamic:
        return tensor.mark_layout_dynamic()
    if leading_dim == -1:
        leading_dim = t.ndim - 1
    return tensor.mark_layout_dynamic(leading_dim=leading_dim)


def to_cute_aux_tensor(t, enable_tvm_ffi=True):
    """Convert torch tensor to cute tensor for TVM FFI, tailored to FlexAttention aux tensors.
    This allows the user to specify alignment and leading dimension for aux tensors used in
    custom score_mod callables.
    """
    assumed_align: int = getattr(t, "__assumed_align__", None)
    leading_dim: int = getattr(t, "__leading_dim__", None)
    fully_dynamic: bool = leading_dim is None

    return to_cute_tensor(
        t,
        assumed_align=assumed_align,
        leading_dim=leading_dim,
        fully_dynamic=fully_dynamic,
        enable_tvm_ffi=enable_tvm_ffi,
    )


def _resolve_aux_leading_dim(tensor: torch.Tensor) -> int | None:
    """Pick the mode CuTe keeps as a static stride-1 leading dimension."""
    leading_dim = getattr(tensor, "__leading_dim__", None)
    if leading_dim is not None:
        if tensor.ndim == 0:
            raise ValueError("Scalar aux tensors cannot declare __leading_dim__")
        leading_dim %= tensor.ndim
        if tensor.stride(leading_dim) != 1:
            raise ValueError("Aux tensor __leading_dim__ must identify a stride-1 dimension")
        return leading_dim

    unit_stride_dims = [dim for dim, stride in enumerate(tensor.stride()) if stride == 1]
    if len(unit_stride_dims) <= 1:
        return unit_stride_dims[0] if unit_stride_dims else None
    nontrivial_dims = [dim for dim in unit_stride_dims if tensor.shape[dim] > 1]
    if len(nontrivial_dims) != 1:
        raise ValueError("Aux tensor layout has no unique stride-1 leading dimension")
    return nontrivial_dims[0]


def get_aux_tensor_metadata(aux_tensors):
    """Return the static aux-tensor ABI facts that must key the compile cache."""
    metadata = []
    for tensor in aux_tensors:
        leading_dim = _resolve_aux_leading_dim(tensor)
        static_strides = tuple(
            0 if stride == 0 else 1 if dim == leading_dim else None
            for dim, stride in enumerate(tensor.stride())
        )
        metadata.append((tensor.dtype, getattr(tensor, "__assumed_align__", None), static_strides))
    return tuple(metadata)


def get_broadcast_dims(tensor: torch.Tensor) -> Tuple[bool, ...]:
    """Return tuple of bools indicating which dims have stride=0 (broadcast).

    This is useful for compile keys since CuTe's mark_layout_dynamic() keeps
    stride=0 as static, meaning kernels compiled with different broadcast
    patterns are not interchangeable.
    """
    strides = tensor.stride()
    # Written this way for speed.
    if 0 not in strides:
        return (False,) * len(strides)
    return tuple(stride == 0 for stride in strides)


# credit: monellz (https://github.com/NVIDIA/cutlass/issues/2658#issuecomment-3630564264)
def dump_kernel_attributes(compiled_kernel):
    from cuda.bindings import driver
    from cutlass.utils import HardwareInfo
    import torch

    device_id = torch.cuda.current_device()
    hardware_info = HardwareInfo(device_id=device_id)
    cubin_data = compiled_kernel.artifacts.CUBIN
    assert cubin_data is not None, "cubin_data is None, need '--keep-cubin' option when compiling"
    cuda_library = hardware_info._checkCudaErrors(
        driver.cuLibraryLoadData(cubin_data, None, None, 0, None, None, 0)
    )
    kernels = hardware_info._checkCudaErrors(driver.cuLibraryEnumerateKernels(1, cuda_library))
    kernel = hardware_info._checkCudaErrors(driver.cuKernelGetFunction(kernels[0]))
    # more metrics: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html#group__CUDA__EXEC_1g5e92a1b0d8d1b82cb00dcfb2de15961b
    local_size_bytes = hardware_info._checkCudaErrors(
        driver.cuFuncGetAttribute(
            driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
            kernel,
        )
    )
    num_regs = hardware_info._checkCudaErrors(
        driver.cuFuncGetAttribute(
            driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS,
            kernel,
        )
    )

    print("--- Kernel Info ---")
    print(f"local_size_bytes: {local_size_bytes}")
    print(f"num_regs: {num_regs}")
    print("--- End Kernel Info ---")
