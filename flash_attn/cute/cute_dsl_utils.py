# Copyright (c) 2025, Tri Dao.

from typing import Tuple
from functools import lru_cache

import torch

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


def get_aux_tensor_metadata(aux_tensors):
    return tuple(
        (
            getattr(t, "__assumed_align__", 0),
            getattr(t, "__leading_dim__", -1),
            hasattr(t, "__leading_dim__"),
        )
        for t in aux_tensors
    )


def get_broadcast_dims(tensor: torch.Tensor) -> Tuple[bool, ...]:
    """Return tuple of bools indicating which dims have stride=0 (broadcast).

    This is useful for compile keys since CuTe's mark_layout_dynamic() keeps
    stride=0 as static, meaning kernels compiled with different broadcast
    patterns are not interchangeable.
    """
    return tuple(s == 0 for s in tensor.stride())


_DTYPE_SHORT_NAMES = {
    torch.float16: "f16",
    torch.bfloat16: "bf16",
    torch.float32: "f32",
    torch.float8_e4m3fn: "e4m3",
    torch.float8_e5m2: "e5m2",
    cutlass.Float16: "f16",
    cutlass.BFloat16: "bf16",
    cutlass.Float32: "f32",
    cutlass.Float8E4M3FN: "e4m3",
    cutlass.Float8E5M2: "e5m2",
}


def short_dtype_name(dtype):
    """Return a compact dtype label suitable for a kernel symbol."""
    return _DTYPE_SHORT_NAMES.get(dtype, str(dtype).replace("torch.", "").replace(".", "_"))


def make_kernel_name_prefix(
    prefix,
    *,
    arch=None,
    dtype=None,
    head_dim=None,
    head_dim_v=None,
    qhead_per_kvhead=None,
    tile_m=None,
    tile_n=None,
    q_stage=None,
    dout_stage=None,
    num_threads=None,
    q_subtile_factor=None,
    causal=False,
    local=False,
    varlen=False,
    paged=False,
    paged_non_tma=False,
    split_kv=False,
    pack_gqa=False,
    use_2cta=False,
    use_clc=False,
    deterministic=False,
    dq_single_wg=False,
    spt=False,
    cluster_size=None,
    mma_pv_is_rs=False,
    intra_wg_overlap=False,
    no_lse=False,
    has_score_mod=False,
    has_mask_mod=False,
    has_block_sparsity=False,
    has_learnable_sink=False,
    has_qv=False,
    has_descale=False,
    has_aux=False,
):
    """Build a readable prefix for CuTeDSL's normally mangled kernel symbol."""
    parts = [prefix]
    if arch is not None:
        parts.append(f"sm{arch}")
    if dtype is not None:
        parts.append(short_dtype_name(dtype))
    if head_dim is not None:
        if head_dim_v in (None, head_dim):
            parts.append(f"head_dim{head_dim}")
        else:
            parts.extend((f"head_dim{head_dim}", f"value_dim{head_dim_v}"))
    if qhead_per_kvhead and qhead_per_kvhead > 1:
        parts.append(f"gqa_ratio{qhead_per_kvhead}")
    if tile_m and tile_n:
        parts.append(f"tile{tile_m}x{tile_n}")
    if q_stage is not None and q_stage != 1:
        parts.append(f"q_stages{q_stage}")
    if dout_stage is not None and dout_stage != 1:
        parts.append(f"dout_stages{dout_stage}")
    if num_threads is not None:
        parts.append(f"threads{num_threads}")
    if q_subtile_factor is not None and q_subtile_factor != 1:
        parts.append(f"q_subtile{q_subtile_factor}")
    for cond, tag in (
        (causal, "causal"),
        (local, "local"),
        (varlen, "varlen"),
        (paged, "paged"),
        (paged_non_tma, "paged_non_tma"),
        (split_kv, "split_kv"),
        (pack_gqa, "pack_gqa"),
        (use_2cta, "use_2cta"),
        (use_clc, "clc_scheduler"),
        (deterministic, "deterministic"),
        (dq_single_wg, "dq_single_wg"),
        (spt, "spt_scheduler"),
        (cluster_size is not None and cluster_size > 1, f"cluster_size{cluster_size}"),
        (mma_pv_is_rs, "pv_mma_rs"),
        (intra_wg_overlap, "intra_wg_overlap"),
        (no_lse, "no_lse"),
        (has_score_mod, "score_mod"),
        (has_mask_mod, "mask_mod"),
        (has_block_sparsity, "block_sparse"),
        (has_learnable_sink, "learnable_sink"),
        (has_qv, "qv"),
        (has_descale, "descale"),
        (has_aux, "aux"),
    ):
        if cond:
            parts.append(tag)
    return "_".join(parts)


def compile_with_kernel_name_prefix(op, *args, name_prefix, options="--enable-tvm-ffi"):
    """Compile a CuTeDSL op with a temporary GPU kernel name prefix."""
    kernel = type(op).kernel
    kernel.set_name_prefix(name_prefix)
    try:
        return cute.compile(op, *args, options=options)
    finally:
        kernel.set_name_prefix(None)
        dsl = kernel.__wrapped__.__dict__.get("_dsl_object")
        if dsl is not None:
            dsl._name_prefix = None


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
