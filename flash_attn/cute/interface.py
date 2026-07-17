# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# [2025-07-04] Version in Cute-DSL, for Hopper and Blackwell. You'll need install nvidia-cutlass-dsl==4.2.0.

import os
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple, Callable

import torch

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32
from quack.compile_utils import make_fake_tensor as fake_tensor
from flash_attn.cute.cache_utils import get_jit_cache
from flash_attn.cute.testing import is_fake_mode


if os.environ.get("CUTE_DSL_PTXAS_PATH", None) is not None:
    from flash_attn.cute import cute_dsl_ptxas  # noqa: F401

    # Patch to dump ptx and then use system ptxas to compile to cubin
    cute_dsl_ptxas.patch()


from flash_attn.cute import utils
from flash_attn.cute import fa_logging
from flash_attn.cute.cute_dsl_utils import (
    get_aux_tensor_metadata,
    get_broadcast_dims,
    to_cute_aux_tensor,
    to_cute_tensor,
)
from flash_attn.cute.flash_fwd import FlashAttentionForwardSm80
from flash_attn.cute.flash_fwd_sm90 import FlashAttentionForwardSm90
from flash_attn.cute.flash_fwd_sm100 import FlashAttentionForwardSm100, DescaleTensors
from flash_attn.cute.flash_fwd_sm120 import FlashAttentionForwardSm120
from flash_attn.cute.flash_fwd_decode_sm120 import FlashAttentionDecodeSm120
from flash_attn.cute.flash_fwd_sm120_tma import FlashAttentionForwardSm120Tma
from flash_attn.cute.flash_bwd_preprocess import FlashAttentionBackwardPreprocess
from flash_attn.cute.flash_bwd import FlashAttentionBackwardSm80
from flash_attn.cute.flash_bwd_sm90 import FlashAttentionBackwardSm90
from flash_attn.cute.flash_bwd_sm100 import FlashAttentionBackwardSm100
from flash_attn.cute.flash_bwd_sm120 import FlashAttentionBackwardSm120
from flash_attn.cute.flash_bwd_postprocess import (
    FlashAttentionBackwardDkvPostprocessSm120,
    FlashAttentionBackwardPostprocess,
)
from flash_attn.cute.flash_fwd_combine import FlashAttentionForwardCombine
from flash_attn.cute.flash_fwd_mla_sm100 import FlashAttentionMLAForwardSm100
from flash_attn.cute.flash_bwd_mla_sm100 import FlashAttentionSparseMLABackwardSm100
from flash_attn.cute.flash_bwd_mla_dq_dqv_sm100 import dQdQvGemmKernel
from flash_attn.cute.flash_bwd_mla_dk_sm100 import dKGemmKernel

# SM100 head_dim=256 2CTA kernel imports
from flash_attn.cute.sm100_hd256_2cta_fmha_forward import BlackwellFusedMultiHeadAttentionForward
from flash_attn.cute.sm100_hd256_2cta_fmha_backward import BlackwellFusedMultiHeadAttentionBackward

from flash_attn.cute.utils import AuxData
from flash_attn.cute.block_sparsity import (
    BlockSparseTensorsTorch,
    block_sparse_bwd_supports_2cta,
    get_kv_subtile_factor,
    get_sparse_q_block_size,
    to_cute_block_sparse_tensors,
    normalize_block_sparse_config,
    normalize_block_sparse_config_bwd,
)


# ---------------------------------------------------------------------------
# torch.compile boundary for the FA4 (CuTe-DSL) public entry points.
#
# The CuTe-DSL kernels and the `cute.compile` machinery only accept concrete
# python scalars/dtypes. Under `torch.compile` dynamo otherwise traces *into*
# this interface and pushes fake/symbolic tensors into the DSL `const()` path,
# which fails in ways like a tensor `max_seqlen` poisoning a `const_expr` bool
# (backward) or a fake tensor corrupting dtype detection in flash_fwd
# ("Only Float16 or BFloat16 is supported", forward under dynamic=True).
#
# Marking the two public functions opaque to dynamo makes the whole FA4 call a
# single graph break: it runs eagerly (where FA4 is correct, and where the
# underlying autograd.Function registers its backward into the eager graph),
# while the surrounding model still compiles. This is a no-op in plain eager
# execution and only affects code paths that go through these FA4 entry points,
# so non-FA4 / non-sm120 kernels are behaviourally unchanged.
def _opaque_to_dynamo(fn):
    disable = getattr(getattr(torch, "compiler", None), "disable", None)
    if disable is None:  # very old torch: fall back to private API
        disable = getattr(getattr(torch, "_dynamo", None), "disable", None)
    if disable is None:
        return fn
    return disable(fn, recursive=True)

def _parse_arch_str(arch_str):
    """Parse arch string (e.g. 'sm_80', 'sm_90a', '80', '100') to int (e.g. 80, 90, 100)."""
    import re
    match = re.match(r"^(?:sm_?|SM_?)?(\d+)(\d)([af]?)$", arch_str)
    if not match:
        raise ValueError(f"Invalid arch format: {arch_str}")
    major, minor, _ = match.groups()
    return int(major) * 10 + int(minor)


def _parse_dsl_version(ver: str) -> tuple:
    """Parse a nvidia-cutlass-dsl version string (e.g. '4.5.1', '4.6.0.dev0')
    into a comparable numeric tuple, e.g. (4, 5, 1).  Trailing non-numeric
    components (rc/dev/post suffixes) are dropped; a leading numeric run is
    enough for an ordering comparison."""
    import re
    parts = []
    for tok in ver.split("."):
        m = re.match(r"^(\d+)", tok)
        if m is None:
            break
        parts.append(int(m.group(1)))
    return tuple(parts)


# nvidia-cutlass-dsl 4.5.2 introduced a DSL codegen regression that breaks the
# sm120 fp8 KV-cache decode kernel: nvgpu.cvt_fpext rejects a scalar f8E4M3FN
# operand, so the kernel fails to compile.  4.5.1 compiles and runs correctly.
# Whether a future >4.5.2 release fixes it is unknown, so the predicate guards a
# half-open interval [4.5.2, _DSL_FP8_DECODE_FIXED_VERSION) of known/assumed-broken
# versions.  When the DSL is fixed, set _DSL_FP8_DECODE_FIXED_VERSION to the first
# good release (e.g. (4, 5, 4)) -- no other code change needed.  Chosen over an
# exact "==4.5.2" check (would silently let a still-broken 4.5.3 through and emit a
# confusing compile failure) and over a compile-time try/except probe (more robust
# to version numbers but far more complex/fragile to wire into the JIT path); the
# floor-and-ceiling window is the most maintainable option that still fails loud.
_DSL_FP8_DECODE_BROKEN_FLOOR = (4, 5, 2)
_DSL_FP8_DECODE_FIXED_VERSION = None  # set to the first fixed version tuple once known


def _fp8_decode_dsl_supported(version: Optional[str] = None) -> bool:
    """Whether the installed nvidia-cutlass-dsl can compile the sm120 fp8 KV-cache
    decode kernel.  Returns False for versions in the known-broken window
    [4.5.2, _DSL_FP8_DECODE_FIXED_VERSION).  Unknown/unparseable versions are
    treated as supported (don't over-guard).  `version` is overridable for tests."""
    if version is None:
        from importlib.metadata import version as _pkg_version
        try:
            version = _pkg_version("nvidia-cutlass-dsl")
        except Exception:
            return True  # can't determine -> don't block
    v = _parse_dsl_version(version)
    if not v:
        return True
    if v < _DSL_FP8_DECODE_BROKEN_FLOOR:
        return True
    if _DSL_FP8_DECODE_FIXED_VERSION is not None and v >= _DSL_FP8_DECODE_FIXED_VERSION:
        return True
    return False


_FP8_DECODE_DSL_ERROR = (
    "sm120 fp8 (e4m3/e5m2) KV-cache decode requires nvidia-cutlass-dsl 4.5.1 "
    "(4.5.x >= 4.5.2 has a DSL codegen regression: nvgpu.cvt_fpext rejects a "
    "scalar f8E4M3FN operand, so the decode kernel fails to compile). "
    "Install nvidia-cutlass-dsl==4.5.1, or pass bf16/fp16 K/V instead of fp8."
)


@lru_cache(maxsize=None)
def _get_device_arch():
    """Cached device arch check.

    Override with FLASH_ATTENTION_ARCH (e.g. 'sm_80' or '80') to select which
    kernel path to use (SM80/SM90/SM100/SM120) independently of the compilation
    target (CUTE_DSL_ARCH).

    For CPU-only compilation (no GPU), set both:
      FLASH_ATTENTION_ARCH=sm_80  (kernel selection)
      CUTE_DSL_ARCH=sm_80         (compilation target)
    """
    arch_override = os.environ.get("FLASH_ATTENTION_ARCH", None)
    if arch_override is not None:
        return _parse_arch_str(arch_override)
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + int(minor)


def _sm120_bwd_pack_gqa_m_splits(
    *,
    arch: int,
    pack_gqa: bool,
    qhead_per_kvhead: int,
    num_head: int,
    num_head_kv: int,
    causal: bool,
    local: bool,
    seqlen_q: int,
    seqlen_k: int,
    head_dim: int,
    head_dim_v: int,
    m_block_size: int,
    n_block_size: int,
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    batch_size: int = 1,
) -> int:
    """Internal SM120 explicit-PackGQA M-split policy for backward.

    Returns the backward M-split count (used as the m-split regardless of
    pack_gqa). Normally only the explicit-PackGQA path is split; the one
    exception is the dense D256 qpkv4 S512 small-grid case below, which underfills
    the SMs and wins from a split even though it runs non-packed.
    """
    # Dense D256 qpkv4 S512 underfills the SMs: grid = ceil(S/64)*Hq*B =
    # 8*num_head*batch CTAs; num_head*batch <= 32 means <= 256 CTAs (~1.36 waves
    # on a high-SM-count sm120 part). split=2 fills to ~2.7 waves and is ~8%
    # faster than the unsplit default. This shape runs non-packed (pack_gqa=False), so
    # it must be handled before the pack-only early-return. Filled grids
    # (num_head*batch > 32, e.g. B>=4 or Hq32) regress with the split -> excluded;
    # qpkv8 regresses even when underfilled -> excluded by qpkv==4.
    if (
        arch // 10 == 12
        and not causal
        and not local
        and qhead_per_kvhead == 4
        and head_dim == 256
        and head_dim_v == 256
        and seqlen_q == seqlen_k
        and seqlen_q == 512
        and cu_seqlens_q is None
        and cu_seqlens_k is None
        and num_head * batch_size <= 32
    ):
        return 2
    # B=1 D256 backward underfills the SMs (grid = ceil(S/64)*Hq*1 CTAs, all
    # <~1.4 waves for these small-Hq shapes) so the unsplit default idles SMs.
    # These exact cells win from an M-split (+4-20% vs the unsplit dispatch,
    # robust across seeds). B=1 runs non-packed, so handle before the
    # pack-only early-return. B>=2 is excluded: it either auto-splits already or
    # the split is noise (verified). Only these validated cells are listed.
    if (
        arch // 10 == 12
        and batch_size == 1
        and not local
        and head_dim == 256
        and head_dim_v == 256
        and seqlen_q == seqlen_k
        and cu_seqlens_q is None
        and cu_seqlens_k is None
    ):
        if causal and qhead_per_kvhead == 8 and num_head == 8 and num_head_kv == 1 and seqlen_q == 512:
            return 4  # +20%
        if causal and qhead_per_kvhead == 4 and num_head == 8 and num_head_kv == 2 and seqlen_q == 512:
            return 3  # +18%
        if causal and qhead_per_kvhead == 2 and num_head == 32 and num_head_kv == 16 and seqlen_q == 2048:
            return 2  # +4%
        if not causal and qhead_per_kvhead == 4 and num_head == 16 and num_head_kv == 4 and seqlen_q == 1024:
            return 2  # +20%
        if not causal and qhead_per_kvhead == 4 and num_head == 8 and num_head_kv == 2 and seqlen_q == 2048:
            return 2  # +18%
        if not causal and qhead_per_kvhead == 6 and num_head == 24 and num_head_kv == 4 and seqlen_q == 1024:
            return 3  # +12%
    if (
        arch // 10 != 12
        or not pack_gqa
        or qhead_per_kvhead <= 1
        or cu_seqlens_q is not None
        or cu_seqlens_k is not None
    ):
        return 1

    packed_m_blocks = max(1, math.ceil(seqlen_q * qhead_per_kvhead / m_block_size))
    if causal:
        # For self-attention, the final N tile has the fewest active packed-M
        # blocks. Cap splits to keep every launched split CTA non-empty.
        if seqlen_q != seqlen_k:
            max_safe_splits = 1
        else:
            tail_k = seqlen_k % n_block_size or min(seqlen_k, n_block_size)
            max_safe_splits = max(1, math.ceil(tail_k * qhead_per_kvhead / m_block_size))
    else:
        max_safe_splits = packed_m_blocks

    sm120_qpkv4_s1024_causal = (
        causal
        and not local
        and qhead_per_kvhead == 4
        and num_head % num_head_kv == 0
        and seqlen_q == seqlen_k
        and seqlen_q == 1024
        and head_dim == 256
        and head_dim_v == 256
    )
    if sm120_qpkv4_s1024_causal:
        # The nominal causal cap avoids empty split CTAs. For this exact short
        # qpkv4 D256 Hq8/Hkv2 shape, launching extra split CTAs raises occupancy
        # toward FA2's CTA count and wins even with the empty-tail overhead.
        # Wider qpkv4 rows showed mean/outlier regressions with split16, so they
        # stay on the previous split8 policy.
        if num_head == 8 and num_head_kv == 2:
            max_safe_splits = max(max_safe_splits, 16)
            auto_splits = 16
        else:
            max_safe_splits = max(max_safe_splits, 8)
            auto_splits = 8
    elif (
        causal
        and not local
        and qhead_per_kvhead == 4
        and num_head in (8, 16)
        and num_head_kv == num_head // qhead_per_kvhead
        and seqlen_q == seqlen_k
        and seqlen_q == 2048
        and head_dim == 256
        and head_dim_v == 256
    ):
        # S2048 qpkv4 is still CTA-limited with the causal-safe split4 cap.
        # The exact B=2 Hq8/Hkv2 and Hq16/Hkv4 rows validate true split16.
        max_safe_splits = max(max_safe_splits, 16)
        auto_splits = 16
    else:
        auto_splits = min(qhead_per_kvhead, max_safe_splits, packed_m_blocks)
    return max(1, min(auto_splits, max_safe_splits, packed_m_blocks))


def _validate_head_dims(head_dim: int, head_dim_v: int, compute_capability: int, alignment: int) -> None:
    """Validate head dimension constraints based on compute capability."""
    is_deepseek_shape = head_dim == 192 and head_dim_v == 128
    is_deepseek_mla_absorbed_shape = (head_dim == 64 or head_dim == head_dim_v) and head_dim_v == 512
    is_dedicate_kernel_shape = head_dim == 256 and head_dim_v == 256
    is_standard_range = 8 <= head_dim <= 128 and 8 <= head_dim_v <= 128

    is_sm90_range = 8 <= head_dim <= 256 and 8 <= head_dim_v <= 256
    if compute_capability == 9:
        assert is_sm90_range and head_dim % alignment == 0 and head_dim_v % alignment == 0, (
            f"(head_dim, head_dim_v)=({head_dim}, {head_dim_v}) is not supported on SM90. "
            f"head_dim and head_dim_v must be between 8 and 256 and divisible by {alignment}."
        )
    elif compute_capability in [10, 11]:
        assert (is_standard_range or is_deepseek_shape or is_deepseek_mla_absorbed_shape or is_dedicate_kernel_shape) and head_dim % alignment == 0 and head_dim_v % alignment == 0, (
            f"(head_dim, head_dim_v)=({head_dim}, {head_dim_v}) is not supported on SM100/SM110. "
            f"head_dim and head_dim_v must be between 8 and 128 and divisible by {alignment}, or (192, 128) for DeepSeek, or (256, 256) for hd256."
        )
    elif compute_capability == 12:
        # Validate host-side; without this, invalid head_dims reach the kernel
        # and fault with cudaErrorMisalignedAddress.
        assert is_sm90_range and head_dim % alignment == 0 and head_dim_v % alignment == 0, (
            f"(head_dim, head_dim_v)=({head_dim}, {head_dim_v}) is not supported on SM120. "
            f"head_dim and head_dim_v must be between 8 and 256 and divisible by {alignment}."
        )


@dataclass(frozen=True)
class FwdConfig:
    m_block_size: int
    n_block_size: int
    mma_pv_is_rs: bool
    intra_wg_overlap: bool


def _tile_size_fwd_sm90(head_dim, head_dim_v, is_causal, is_local, sparse_block_size_q=None):
    """Return FwdConfig for SM90 forward.

    Tile sizes and flags based on tile_size_fwd_sm90 in hopper/tile_size.h, adjusted
    for the Python kernel's different register/smem tradeoffs (benchmarked on H100 SXM).

    When sparse_block_size_q is set, tile_m must divide it. For head_dim <= 96 the
    optimal tile_m=192 is used when compatible, otherwise we fall back to 128.
    """
    if head_dim <= 64:
        # C++: 192×192 non-causal, 192×128 causal/local.
        # Python: 192×128 RS+OL is consistently best across seqlens.
        if sparse_block_size_q is not None and sparse_block_size_q % 192 != 0:
            return FwdConfig(128, 128, True, True)
        return FwdConfig(192, 128, True, True)
    elif head_dim <= 96:
        # C++: 192×144 noRS+OL for all cases.
        # Python: RS is catastrophic with 192× tiles (~300 vs ~600 TFLOPS).
        # noRS+OL is always required. Causal: 192×128 slightly better short seqlen.
        if sparse_block_size_q is not None and sparse_block_size_q % 192 != 0:
            return FwdConfig(128, 128, False, True)
        if is_causal or is_local:
            return FwdConfig(192, 128, False, True)
        else:
            return FwdConfig(192, 144, False, True)
    elif head_dim <= 128:
        return FwdConfig(128, 128, True, True)
    elif head_dim <= 192:
        tile_n = 96 if is_local else (128 if head_dim_v <= 128 else 112)
        return FwdConfig(128, tile_n, True, True)
    else:  # hdim 256
        tile_n = 64 if is_local else 80
        return FwdConfig(128, tile_n, True, True)

@dataclass(frozen=True)
class BwdConfig:
    m_block_size: int
    n_block_size: int
    num_stages_Q: int
    num_stages_dO: int
    num_stages_PdS: int
    SdP_swapAB: bool
    dKV_swapAB: bool
    dQ_swapAB: bool
    AtomLayoutMSdP: int
    AtomLayoutNdKV: int
    AtomLayoutMdQ: int
    num_wg: int = 2  # MMA warp groups (total threads = (num_wg + 1) * 128)
    dQ_single_wg: bool = False


def _tile_size_bwd_sm90(head_dim, head_dim_v, causal, local, sparse_block_size_q=None):
    """Return BwdConfig for SM90.

    Configs based on C++ FA3 hopper/flash_bwd_launch_template.h,
    benchmarked on H100 SXM.
    """
    if head_dim <= 64:
        # C++ FA3: 128, 128, 64, ..., 2, 2, true, false, false, 2, 1, 2, 2
        return BwdConfig(
            m_block_size=128, n_block_size=128,
            num_stages_Q=2, num_stages_dO=2, num_stages_PdS=2,
            SdP_swapAB=True, dKV_swapAB=False, dQ_swapAB=False,
            AtomLayoutMSdP=1, AtomLayoutNdKV=2, AtomLayoutMdQ=2,
        )
    elif head_dim <= 96:
        # C++ FA3: 64, 128, 96, dQ_swapAB=False
        return BwdConfig(
            m_block_size=64, n_block_size=128,
            num_stages_Q=2, num_stages_dO=2, num_stages_PdS=2,
            SdP_swapAB=True, dKV_swapAB=False, dQ_swapAB=False,
            AtomLayoutMSdP=1, AtomLayoutNdKV=2, AtomLayoutMdQ=1,
            dQ_single_wg=True,
        )
    elif head_dim <= 128:
        # C++ FA3: causal/local: 64, 128; non-causal: 80, 128 with dQ_swapAB
        is_causal_or_local = causal or local
        m_block_size = 64 if is_causal_or_local else 80
        if sparse_block_size_q is not None and sparse_block_size_q % m_block_size != 0:
            m_block_size = 64
        return BwdConfig(
            m_block_size=m_block_size,
            n_block_size=128,
            num_stages_Q=2, num_stages_dO=2, num_stages_PdS=2,
            SdP_swapAB=True, dKV_swapAB=False,
            dQ_swapAB=m_block_size % 64 != 0,
            AtomLayoutMSdP=1, AtomLayoutNdKV=2, AtomLayoutMdQ=1,
        )
    elif head_dim <= 192:
        hdimv128 = head_dim_v <= 128
        if hdimv128:
            return BwdConfig(
                m_block_size=64, n_block_size=96,
                num_stages_Q=2, num_stages_dO=2, num_stages_PdS=1,
                SdP_swapAB=False, dKV_swapAB=True, dQ_swapAB=False,
                AtomLayoutMSdP=1, AtomLayoutNdKV=2, AtomLayoutMdQ=1,
                num_wg=2,
            )
        else:
            return BwdConfig(
                m_block_size=64, n_block_size=96,
                num_stages_Q=2, num_stages_dO=1, num_stages_PdS=1,
                SdP_swapAB=False, dKV_swapAB=True, dQ_swapAB=False,
                AtomLayoutMSdP=1, AtomLayoutNdKV=2, AtomLayoutMdQ=1,
                num_wg=2,
            )
    else:
        # hdim 256
        return BwdConfig(
            m_block_size=64, n_block_size=64,
            num_stages_Q=1, num_stages_dO=1, num_stages_PdS=1,
            SdP_swapAB=False, dKV_swapAB=False, dQ_swapAB=False,
            AtomLayoutMSdP=1, AtomLayoutNdKV=1, AtomLayoutMdQ=1,
        )



def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def _to_cute_int32_or_none(x: Optional[int]):
    return cutlass.Int32(x) if x is not None else None


def _validate_tensor(t, name, expected_shape, expected_dtype, expected_device):
    assert t.shape == expected_shape, f"{name} shape {t.shape} != expected {expected_shape}"
    assert t.dtype == expected_dtype, f"{name} dtype {t.dtype} != expected {expected_dtype}"
    assert t.device == expected_device, f"{name} device {t.device} != expected {expected_device}"
    if not is_fake_mode():
        assert t.is_cuda, f"{name} must be on CUDA"

torch2cute_dtype_map = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
    torch.float8_e4m3fn: cutlass.Float8E4M3FN,
    torch.float8_e5m2: cutlass.Float8E5M2,
}


def num_splits_heuristic(total_mblocks, num_SMs, num_n_blocks, max_splits):
    # If num_n_blocks is too small, use 1 split. For example, we never split for hdim = 128 and seqlen_k = 512.
    if num_n_blocks <= 4:
        return 1
    # Avoid ZeroDivisionError when batch_size or seqlen_q is 0. The empty-Q
    # early-exit in _flash_attn_fwd handles correctness for those shapes; this
    # guard just keeps the heuristic safe if called in other contexts.
    if total_mblocks == 0:
        return 1

    # NOTE: We should revisit this heuristic after persistence is supported for split KV.
    # Sometimes, it's ideal to over-schedule splits for better efficiency.
    return min(num_SMs // total_mblocks, max_splits, num_n_blocks)


def _resolve_causal_local_window(causal, window_size_left, window_size_right, mask_mod=None):
    """Resolve causal/local/window settings into canonical form.

    Returns (causal, local, window_size_left, window_size_right).
    """
    if mask_mod is not None:
        return False, False, window_size_left, window_size_right
    if causal:
        window_size_right = 0
    if window_size_left is not None and window_size_right is not None and window_size_left + window_size_right < 0:
        window_size_left = None
        window_size_right = None
    if window_size_left is not None or window_size_right is not None:
        if window_size_left is None and window_size_right == 0:
            causal, local = True, False
            window_size_right = None
        else:
            causal, local = False, True
    else:
        local = False
    return causal, local, window_size_left, window_size_right

def _flash_attn_fwd(
    q: Optional[torch.Tensor],
    k: Optional[torch.Tensor],
    v: torch.Tensor,
    qv: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    min_seqlen_k: Optional[int] = None,
    page_table: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    softcap: Optional[float] = None,
    window_size_left: Optional[int] = None,
    window_size_right: Optional[int] = None,
    learnable_sink: Optional[torch.Tensor] = None,
    tile_mn: Optional[Tuple[int, int]] = None,
    mma_pv_is_rs: Optional[bool] = None,
    intra_wg_overlap: Optional[bool] = None,
    num_threads: int = 384,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    _arch: Optional[int] = None,
    score_mod: Optional[Callable] = None,
    mask_mod: Optional[Callable] = None,
    block_sparse_tensors: Optional[BlockSparseTensorsTorch] = None,
    return_lse: bool = False,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    aux_tensors: Optional[list[torch.Tensor]] = None,
    aux_scalars: Optional[tuple] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    gather_kv_indices: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Forward pass for FlashAttention.

    Args:
        ...
        score_mod: A callable that takes the attention scores and applies a modification.
        mask_mod: A callable that takes token position information and selectively masks
        block_sparse_tensors: A tuple of tensors used for block sparsity.
        return_lse: Whether to return the log softmax of the attention scores. If set to True will always calculate
            The returned LSE supports taking gradient.
        out: Optional pre-allocated output tensor. If None, will be allocated internally.
        lse: Optional pre-allocated log-sum-exp tensor. If None, will be allocated when needed.
        aux_tensors: Some score_mods will want to read from global aux_tensors. This is how we thread them through to the inner kernel.
        aux_scalars: Runtime scalar captures used by score_mod or mask_mod.
    """
    aux_scalars = tuple(aux_scalars) if aux_scalars else None
    q, k, v, qv = [maybe_contiguous(t) for t in (q, k, v, qv)]
    assert q is not None or qv is not None
    assert v is not None
    q_descale, k_descale, v_descale = [maybe_contiguous(t) for t in (q_descale, k_descale, v_descale)]
    q_shape = q.shape if q is not None else qv.shape
    num_head, head_dim = q_shape[-2:]
    if cu_seqlens_q is None:
        batch_size, seqlen_q = q_shape[:2]
        total_q = batch_size * seqlen_q
    else:
        batch_size = cu_seqlens_q.shape[0] - 1
        seqlen_q = None
        total_q = q_shape[0]
    if page_table is not None:
        assert cu_seqlens_k is None, "page_table is not supported with cu_seqlens_k"
        assert page_table.dtype == torch.int32, "page_table must be int32"
        assert page_table.stride(-1) == 1, "page_table must be contiguous in the last dimension"
        max_num_pages_per_seq = page_table.shape[1]
        assert page_table.shape == (batch_size, max_num_pages_per_seq)
        num_pages, page_size = v.shape[:2]
        seqlen_k = num_pages * page_size
    else:
        num_pages, page_size = None, None
        seqlen_k = v.shape[-3]
    num_head_kv = v.shape[-2]
    head_dim_v = v.shape[-1]
    if cu_seqlens_k is None:
        if page_table is None:
            assert k is None or k.shape == (batch_size, seqlen_k, num_head_kv, head_dim)
            assert v.shape == (batch_size, seqlen_k, num_head_kv, head_dim_v)
        else:
            assert k is None or k.shape == (num_pages, page_size, num_head_kv, head_dim)
            assert v.shape == (num_pages, page_size, num_head_kv, head_dim_v)
    else:
        assert k is None or k.shape == (seqlen_k, num_head_kv, head_dim)
        assert v.shape == (seqlen_k, num_head_kv, head_dim_v)
        assert cu_seqlens_k.shape == (batch_size + 1,), (
            "cu_seqlens_k must have shape (batch_size + 1,)"
        )

    if cu_seqlens_q is not None:
        assert cu_seqlens_q.shape == (batch_size + 1,), (
            "cu_seqlens_q must have shape (batch_size + 1,)"
        )
    assert seqused_q is None or seqused_q.shape == (batch_size,), (
        "seqused_q must have shape (batch_size,)"
    )
    assert seqused_k is None or seqused_k.shape == (batch_size,), (
        "seqused_k must have shape (batch_size,)"
    )
    assert v.dtype in [torch.float16, torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2], (
        "inputs must be float16, bfloat16, fp8 e4m3fn, or fp8 e5m2"
    )
    # SM120 fp8 KV-cache decode: bf16/fp16 Q with an fp8 (e4m3/e5m2) K/V cache.
    # This is the only path where q.dtype may differ from k/v.dtype; every other
    # path still requires identical dtypes (default behaviour unchanged).
    #
    # Auto-enabled whenever fp8 K/V is genuinely passed (no env flag required):
    # the fp8 KV-cache decode kernel is the *only* sm_120 path that can consume an
    # fp8 K/V cache (fp8 prefill is a no-go and the standard SM120 forward asserts
    # q.dtype==k.dtype==v.dtype), so a user who quantized their cache must be able
    # to use it without an env var.  The FLASH_ATTENTION_SM120_DECODE_KERNEL flag
    # remains the manual override for the *bf16* decode kernel below; for bf16
    # inputs this expression is always False, so the default path is unchanged.
    fp8_kv_decode = (
        q.dtype in (torch.float16, torch.bfloat16)
        and k.dtype == v.dtype
        and k.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    )
    if not fp8_kv_decode:
        # Upstream pairwise dtype check across all present tensors (q, k, v, qv).
        # Skipped for fp8 KV-cache decode, where q is bf16/fp16 but k/v are fp8.
        input_tensors = {"q": q, "k": k, "v": v, "qv": qv}
        present = {name: t for name, t in input_tensors.items() if t is not None}
        names = list(present.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                assert present[a].dtype == present[b].dtype, f"{a}.dtype {present[a].dtype} != {b}.dtype {present[b].dtype}"

    q_dtype = q.dtype if q is not None else qv.dtype

    for t in [cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k]:
        if t is not None:
            assert t.dtype == torch.int32, (
                "cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k must be int32"
            )
            assert t.stride(0) == 1, (
                "cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k must be contiguous"
            )
    if learnable_sink is not None:
        assert learnable_sink.shape == (num_head,)
        assert learnable_sink.dtype == torch.bfloat16, "learnable_sink must be bfloat16"

    if not is_fake_mode():
        assert all(
            t is None or t.is_cuda
            for t in (
                q,
                k,
                v,
                qv,
                q_descale,
                k_descale,
                v_descale,
                cu_seqlens_q,
                cu_seqlens_k,
                seqused_q,
                seqused_k,
                page_table,
                learnable_sink,
            )
        ), "inputs must be on CUDA device"
    arch = _get_device_arch() if _arch is None else _arch
    assert arch // 10 in [8, 9, 10, 11, 12], "Unsupported compute capability. Supported: 8.x, 9.x, 10.x, 11.x, 12.x"
    assert num_head % num_head_kv == 0, "num_head must be divisible by num_head_kv"
    alignment = 16 // v.element_size()
    # SM120 (arch // 10 == 12) intentionally goes through _validate_head_dims:
    # our sm120 path validates head_dims host-side (see the cc==12 branch) to
    # avoid invalid dims faulting in-kernel with cudaErrorMisalignedAddress.
    # Only SM80 (arch // 10 == 8) skips it.
    if arch // 10 != 8:
        _validate_head_dims(head_dim, head_dim_v, arch // 10, alignment)
    if softmax_scale is None:
        softmax_scale = (
            1.0 / math.sqrt(head_dim) if qv is None or q is None
            else 1.0 / math.sqrt(head_dim + head_dim_v)
        )
    if softcap == 0.0:
        softcap = None
    qhead_per_kvhead = num_head // num_head_kv
    if pack_gqa is None:
        pack_gqa = qhead_per_kvhead > 1
    # pack_gqa + paged-KV on the SM80-base SM120 path produces wrong output
    # (PagedKVManager's K/V indexing doesn't consume mQ's packed composite mode).
    if page_table is not None and pack_gqa:
        pack_gqa = False
    # pack_gqa_layout makes mQ.shape[0] composite ((qhead_per_kvhead, seqlen_q));
    # cute.local_tile by (tile_m, tile_hdim) needs tile_m % qhead_per_kvhead == 0
    # at the qhead boundary. SM120's tile_m=128 covers 1/2/4/8/16-way GQA but
    # not 7-way (qwen2.5-7b 28q/4kv). Other arches choose tile_m differently.
    if arch // 10 == 12 and pack_gqa and qhead_per_kvhead > 1 and 128 % qhead_per_kvhead != 0:
        pack_gqa = False

    # Genuine fp8 attention (SM100 prefill): q, k AND v are all fp8. The sm120
    # fp8 KV-cache decode path keeps a live bf16/fp16 Q with fp8 k/v, so v.dtype
    # is fp8 there too — exclude it explicitly so it is NOT treated as the SM100
    # fp8 path (which would trip the arch==10 assert and the fp8 output/descale
    # handling below). fp8_kv_decode has its own routing further down.
    is_fp8 = v.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) and not fp8_kv_decode
    requires_grad = any(t is not None and t.requires_grad for t in [q, k, v, qv])
    if is_fp8 and requires_grad:
        raise NotImplementedError("FA4 CuTe FP8 backward is not supported yet (forward-only).")
    out_torch_dtype = torch.bfloat16 if is_fp8 else q_dtype
    device = v.device
    q_batch_seqlen_shape = (batch_size, seqlen_q) if cu_seqlens_q is None else (total_q,)

    if qv is None:
        lse_shape = (batch_size, num_head, seqlen_q) if cu_seqlens_q is None else (num_head, total_q)
    else:
        # num_head contiguous better for MQA in MLA absorbed
        lse_shape = (batch_size, seqlen_q, num_head) if cu_seqlens_q is None else (total_q, num_head)

    if out is None:
        out = torch.empty(
            *q_batch_seqlen_shape, num_head, head_dim_v, dtype=out_torch_dtype, device=device
        )
    else:
        _validate_tensor(out, "out", (*q_batch_seqlen_shape, num_head, head_dim_v), out_torch_dtype, device)

    if lse is None:
        lse = (
            torch.empty(lse_shape, dtype=torch.float32, device=device)
            if requires_grad or return_lse
            else None
        )
    elif lse is not None:
        _validate_tensor(lse, "lse", lse_shape, torch.float32, device)

    if seqlen_k == 0 or total_q == 0:
        out.zero_()
        if lse is not None:
            lse.fill_(float("-inf"))
        return out, lse, None, None

    if is_fp8 or fp8_kv_decode:
        for t, name in ((q_descale, "q_descale"), (k_descale, "k_descale"), (v_descale, "v_descale")):
            if t is not None:
                _validate_tensor(t, name, (batch_size, num_head_kv), torch.float32, device)
        if fp8_kv_decode:
            assert q_descale is None, (
                "fp8 KV-cache decode keeps a live bf16/fp16 Q; q_descale is unused"
            )
    else:
        assert q_descale is None and k_descale is None and v_descale is None, (
            "q_descale/k_descale/v_descale are only supported for FP8 inputs"
        )

    dtype = torch2cute_dtype_map[q_dtype]
    kv_dtype = torch2cute_dtype_map[k.dtype]
    if is_fp8:
        assert arch // 10 == 10, "FP8 is only supported on SM100 (compute capability 10.x) for FA4 CuTe."
    use_block_sparsity = block_sparse_tensors is not None

    causal, local, window_size_left, window_size_right = _resolve_causal_local_window(
        causal, window_size_left, window_size_right, mask_mod
    )

    requested_use_clc_scheduler = utils._get_use_clc_scheduler_default()
    requested_disable_2cta = utils._get_disable_2cta_default(is_fwd=True)

    current_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    # SM80/SM120: uses SM80 MMA, 128 threads (4 warps)
    if arch // 10 in [8, 12]:
        num_threads = 128
    sm120_seq_q = max_seqlen_q if max_seqlen_q is not None else seqlen_q
    sm120_seq_k = max_seqlen_k if max_seqlen_k is not None else seqlen_k
    sm120_qpkv5_s16384_qregs = (
        arch // 10 == 12
        and q.dtype == torch.bfloat16
        and causal
        and not local
        and head_dim == 128
        and head_dim_v == 128
        and qhead_per_kvhead == 5
        and sm120_seq_q == 16384
        and sm120_seq_k == 16384
        and not pack_gqa
        and score_mod is None
        and mask_mod is None
        and page_table is None
        and qv is None
        and cu_seqlens_q is None
        and cu_seqlens_k is None
        and seqused_q is None
        and seqused_k is None
        and not use_block_sparsity
    )
    sm120_d256_qregs128 = (
        arch // 10 == 12
        and q.dtype == torch.bfloat16
        and batch_size == 1
        and not causal
        and not local
        and head_dim == 256
        and head_dim_v == 256
        and qhead_per_kvhead in (8, 16)
        and num_head_kv == 2
        and sm120_seq_q == sm120_seq_k
        and sm120_seq_q in (16384, 32768, 65536, 131072)
        and pack_gqa
        and score_mod is None
        and mask_mod is None
        and page_table is None
        and qv is None
        and cu_seqlens_q is None
        and cu_seqlens_k is None
        and seqused_q is None
        and seqused_k is None
        and not use_block_sparsity
    )
    sm120_qpkv8_d256_causal_qregs_eligible = (
        arch // 10 == 12
        and q.dtype == torch.bfloat16
        and batch_size == 1
        and causal
        and not local
        and head_dim == 256
        and head_dim_v == 256
        and num_head == 16
        and num_head_kv == 2
        and qhead_per_kvhead == 8
        and sm120_seq_q == sm120_seq_k
        and pack_gqa
        and score_mod is None
        and mask_mod is None
        and page_table is None
        and qv is None
        and cu_seqlens_q is None
        and cu_seqlens_k is None
        and seqused_q is None
        and seqused_k is None
        and not use_block_sparsity
    )
    if sm120_qpkv8_d256_causal_qregs_eligible and sm120_seq_q in (16384, 32768, 65536, 131072):
        sm120_qpkv8_d256_causal_qregs_mode = "128x64_t256"
    else:
        sm120_qpkv8_d256_causal_qregs_mode = ""
    sm120_qpkv16_d256_causal_qregs_eligible = (
        arch // 10 == 12
        and q.dtype == torch.bfloat16
        and batch_size == 1
        and causal
        and not local
        and head_dim == 256
        and head_dim_v == 256
        and num_head == 32
        and num_head_kv == 2
        and qhead_per_kvhead == 16
        and sm120_seq_q == sm120_seq_k
        and pack_gqa
        and score_mod is None
        and mask_mod is None
        and page_table is None
        and qv is None
        and cu_seqlens_q is None
        and cu_seqlens_k is None
        and seqused_q is None
        and seqused_k is None
        and not use_block_sparsity
    )
    if sm120_qpkv16_d256_causal_qregs_eligible and sm120_seq_q in (16384, 32768, 65536, 131072):
        sm120_qpkv16_d256_causal_qregs_mode = "128x64_t256"
    else:
        sm120_qpkv16_d256_causal_qregs_mode = ""
    sm120_qpkv6_d256_qregs_eligible = (
        arch // 10 == 12
        and q.dtype == torch.bfloat16
        and batch_size in (1, 2)
        and not local
        and head_dim == 256
        and head_dim_v == 256
        and num_head == 24
        and num_head_kv == 4
        and qhead_per_kvhead == 6
        and sm120_seq_q == sm120_seq_k
        and not pack_gqa
        and score_mod is None
        and mask_mod is None
        and page_table is None
        and qv is None
        and cu_seqlens_q is None
        and cu_seqlens_k is None
        and seqused_q is None
        and seqused_k is None
        and not use_block_sparsity
    )
    if (
        sm120_qpkv6_d256_qregs_eligible
        and batch_size == 1
        and sm120_seq_q in (16384, 32768, 65536, 131072)
    ):
        sm120_qpkv6_d256_qregs_mode = "128x64_t256"
    elif (
        sm120_qpkv6_d256_qregs_eligible
        and batch_size == 2
        and (
            (not causal and sm120_seq_q in (4096, 8192))
            # causal S4096 also wins with Q-in-regs (measured on sm120)
            or (causal and sm120_seq_q in (4096, 8192))
        )
    ):
        sm120_qpkv6_d256_qregs_mode = "128x64_t256"
    else:
        sm120_qpkv6_d256_qregs_mode = ""
    # General D256 forward: the 99 KB SMEM cap forces a 64x64 tile only because
    # 128x64 won't fit Q+K+V — but staging Q through registers (max(Q,V)+K)
    # makes 128x64 fit, and it is materially faster for any reasonably long
    # sequence. Measured on sm120: at S>=4096 (square) 128x64+Qregs+256t beats
    # 64x64 by +6-14% across qpkv 4/8/16, causal and non-causal, with
    # bit-identical output (the per-key
    # reduction order is unchanged). S<=2048 is mixed (several causal shapes
    # regress) so it is gated out. Shapes already routed to a specific qregs
    # path keep theirs.
    sm120_d256_wide = (
        arch // 10 == 12
        and q.dtype == torch.bfloat16
        and head_dim == 256
        and head_dim_v == 256
        and not local
        and sm120_seq_q == sm120_seq_k
        and (
            sm120_seq_q >= 4096
            # S2048 non-causal wins +9-13% for the larger-head models
            # (qwen3.5-9b/qwen3.6-35b Hq16, qwen3.5-122b Hq32) but the small
            # Hq8 (qwen3.5-0.8b) regresses, so gate S2048 nc to num_head>=16.
            # S2048 causal only the widest head count (qpkv16, Hq32 qwen3.5-122b)
            # wins (+6.5%); Hq16 (9b/35b) regress, so gate causal to num_head>=32.
            # Validated on sm120.
            or (sm120_seq_q == 2048 and not causal and num_head >= 16)
            or (sm120_seq_q == 2048 and causal and num_head >= 32)
        )
        and not sm120_d256_qregs128
        and not sm120_qpkv8_d256_causal_qregs_mode
        and not sm120_qpkv16_d256_causal_qregs_mode
        and not sm120_qpkv6_d256_qregs_mode
        and page_table is None
        and qv is None
        and learnable_sink is None
        # varlen (cu_seqlens) is supported by the wide tile (same SM80-base
        # kernel; +7-11% on packed D256, bit-identical). seqused
        # mode stays on the 64x64 path (untested).
        and seqused_q is None
        and seqused_k is None
        and not use_block_sparsity
        and mask_mod is None
        and score_mod is None
    )
    # Local (sliding-window) D256: same Q-in-regs win as the dense wide path.
    # The narrow local-window dispatch used a 64x16/64x32 tile; 128x{32,64}
    # +Qregs+256t is faster by +3-13% (measured on sm120, SDPA-window
    # validated). tile_n scales with
    # the window: 32 for window<=512, 64 for window~1024 (gemma4-31b). Gated to
    # S>=4096 (the validated range; gemma local benches there).
    sm120_local_d256_wide = (
        arch // 10 == 12
        and q.dtype == torch.bfloat16
        and local
        and head_dim == 256
        and head_dim_v == 256
        and qhead_per_kvhead in (1, 2, 4, 8)
        and sm120_seq_q == sm120_seq_k
        and sm120_seq_q >= 4096
        and page_table is None
        and qv is None
        and learnable_sink is None
        # varlen (cu_seqlens) supported (gemma packed sliding-window training);
        # seqused mode stays on the narrow path (untested).
        and seqused_q is None
        and seqused_k is None
        and not use_block_sparsity
        and mask_mod is None
        and score_mod is None
    )
    if (
        arch // 10 == 12
        and causal
        and not local
        and head_dim == 128
        and head_dim_v == 128
        and qhead_per_kvhead == 5
        and (sm120_seq_q == 8192 or sm120_seq_q >= 32768 or sm120_qpkv5_s16384_qregs)
    ):
        num_threads = 256
    fwd_cfg = FwdConfig(128, 128, True, True)  # default
    sm120_num_stages = 1
    if tile_mn is None:
        if arch // 10 == 12:
            # SM120 forward tile lookup tuned per shape on sm120 hardware.
            # Misses fall back to the head_dim-only brackets below.
            _SM120_TILE_LOOKUP = {
                # (head_dim, qhead_per_kvhead, seqlen, causal): (tile_m, tile_n, num_stages)
                (64, 1, 512, 0): (128, 128, 1), (64, 1, 512, 1): (64, 64, 1),
                (64, 1, 1024, 0): (64, 64, 1),  (64, 1, 1024, 1): (64, 64, 1),
                (64, 1, 2048, 0): (128, 32, 1), (64, 1, 2048, 1): (64, 64, 2),
                (64, 1, 4096, 0): (64, 64, 1),  (64, 1, 4096, 1): (64, 64, 1),
                (64, 1, 8192, 0): (128, 48, 1), (64, 1, 8192, 1): (64, 64, 2),
                (64, 1, 16384, 0): (128, 48, 1),(64, 1, 16384, 1): (128, 48, 1),
                (64, 4, 512, 0): (64, 128, 1),  (64, 4, 512, 1): (64, 64, 2),
                (64, 4, 1024, 0): (128, 128, 1),(64, 4, 1024, 1): (64, 48, 1),
                (64, 4, 2048, 0): (128, 128, 1),(64, 4, 2048, 1): (64, 64, 1),
                (64, 4, 4096, 0): (64, 128, 1), (64, 4, 4096, 1): (64, 48, 1),
                (64, 4, 8192, 0): (128, 128, 1),(64, 4, 8192, 1): (64, 128, 1),
                (64, 4, 16384, 0): (128, 128, 1),(64, 4, 16384, 1): (64, 64, 2),
                # S512 D128 GQA: smaller tiles fit 2 CTA/SM (49 KB vs 64-98 KB
                # -> 8.3%->16.7% occupancy), +5-11% over the larger tile at B2 and B16
                # and beats FA2 (mirrors upstream FA2 PR #2592's small-seq hd=128 win).
                (128, 4, 512, 0): (128, 32, 1), (128, 4, 512, 1): (64, 64, 1),
                (128, 8, 512, 0): (128, 32, 1),
                (128, 4, 1024, 0): (128, 64, 1), (128, 4, 1024, 1): (64, 96, 1),  # c 64x64->64x96 (+5-6%); nc keeps 128x64
                (128, 4, 2048, 0): (128, 64, 1), (128, 4, 2048, 1): (128, 64, 2),  # nc 64x64->128x64; c 64x96->128x64+ns2: more stable than the old erratic 64x96
                (128, 4, 4096, 0): (128, 64, 1), (128, 4, 4096, 1): (128, 48, 1),  # nc 64x64->128x64; c 64x96->128x48
                (128, 4, 8192, 0): (128, 32, 1),(128, 4, 8192, 1): (128, 64, 1),
                (128, 4, 16384, 0): (128, 32, 1),(128, 4, 16384, 1): (128, 64, 1),
                (128, 5, 1024, 1): (64, 128, 1),
                (128, 5, 4096, 1): (128, 64, 1),  # 64x128->128x64 (+2.8%); S1024/S8192 keep 64x128 (those regress)
                (128, 5, 8192, 1): (128, 128, 1),
                (128, 5, 16384, 1): (64, 128, 1),
                (128, 5, 32768, 1): (128, 128, 1),
                (128, 5, 65536, 1): (128, 128, 1),
                (128, 5, 131072, 1): (128, 128, 1),
                (128, 7, 512, 0): (128, 64, 1), (128, 7, 512, 1): (64, 64, 2),
                (128, 7, 1024, 0): (64, 96, 1), (128, 7, 1024, 1): (64, 128, 1),
                (128, 7, 2048, 0): (128, 64, 1),(128, 7, 2048, 1): (64, 128, 1),
                (128, 7, 4096, 0): (128, 64, 1),(128, 7, 4096, 1): (64, 96, 1),
                (128, 7, 8192, 0): (128, 64, 1),(128, 7, 8192, 1): (64, 128, 1),
                (128, 7, 16384, 0): (128, 64, 1),(128, 7, 16384, 1): (64, 128, 1),
                (128, 8, 1024, 1): (64, 128, 1),  # 64x64->64x128 (1.13x)
                (128, 8, 4096, 1): (128, 64, 1),  # 64x64->128x64 (1.03x)
                (128, 8, 4096, 0): (128, 64, 1),  # 64x64->128x64 (1.28x)
                (128, 8, 8192, 0): (128, 32, 1),
                (128, 8, 8192, 1): (128, 64, 1),
                (128, 8, 32768, 1): (128, 32, 1),
                (128, 8, 65536, 1): (128, 32, 1),
                (128, 8, 131072, 1): (128, 32, 1),
            }
            sl = sm120_seq_k
            lookup_key = (head_dim, qhead_per_kvhead, sl, int(bool(causal)))
            # For head_dim <= 128 paged-KV uses (128, 128, ns=1), which fits
            # SMEM (48 KB at d=64, 72 KB at d=96, 96 KB at d=128 with d==dv).
            # D192/D256 paged-KV falls through to the head_dim > 128 64x64
            # non-TMA path below.
            if page_table is not None and head_dim <= 128 and head_dim_v <= 128:
                # Paged-KV D128: the old 128x128 tile is ~1.4-1.9x slower than
                # 64x64 / 128-thread on sm120 (tile_n=128 + the paged
                # cp.async load is inefficient). qpkv5 (Hq40/Hkv8) is the lone
                # exception — it prefers 128x128 — so it keeps the old tile.
                # Validated vs SDPA on reconstructed K/V (rel ~1e-3).
                if qhead_per_kvhead == 5:
                    fwd_cfg = FwdConfig(128, 128, True, True)
                else:
                    fwd_cfg = FwdConfig(64, 64, True, True)
                    num_threads = 128
                sm120_num_stages = 1
            elif sm120_qpkv5_s16384_qregs:
                # Exact qwen3-14B S16384 causal row wins by staging Q in
                # registers, which requires the 256-thread 128x128 shape.
                fwd_cfg = FwdConfig(128, 128, True, True)
                sm120_num_stages = 1
            elif sm120_local_d256_wide:
                # Gemma local D256, S>=4096: 128x{32,64}+Qregs+256t beats the
                # narrow 64x16/64x32 tile by +3-13% (see sm120_local_d256_wide).
                # tile_n scales with the window (32 for w<=512, 64 for w~1024).
                fwd_cfg = FwdConfig(
                    128, 64 if (window_size_left or 0) >= 1024 else 32, True, True
                )
                num_threads = 256
            elif (
                local
                and head_dim == 256
                and head_dim_v == 256
                and qhead_per_kvhead in (4, 8)
            ):
                # Gemma local attention only loads a narrow K window;
                # smaller N tiles reduce wasted local-window work on SM120.
                # qpkv8 (Gemma e2b) wins ~7% with N=32 vs N=16 on sm120;
                # qpkv4 (e4b) stays best at N=16.
                fwd_cfg = FwdConfig(64, 32 if qhead_per_kvhead == 8 else 16, True, True)
            elif sm120_d256_qregs128:
                # Qwen-style D256 qpkv8/qpkv16 noncausal rows fit a wider N
                # tile on SM120 only when Q is staged through registers.
                fwd_cfg = FwdConfig(128, 64, True, True)
                num_threads = 256
            elif sm120_qpkv8_d256_causal_qregs_mode:
                # Exact qwen3.6-35B-style S16384 causal row benefits from
                # staging Q in registers.
                fwd_cfg = FwdConfig(128, 64, True, True)
                num_threads = 256
            elif sm120_qpkv16_d256_causal_qregs_mode:
                # Exact qwen3.5-122B-style causal rows benefit from staging Q
                # in registers, matching the accepted qpkv8/qpkv16 D256 paths.
                fwd_cfg = FwdConfig(128, 64, True, True)
                num_threads = 256
            elif sm120_qpkv6_d256_qregs_mode:
                # Exact Qwen qpkv6 D256 long rows benefit from staging Q in
                # registers.
                fwd_cfg = FwdConfig(128, 64, True, True)
                num_threads = 256
            elif sm120_d256_wide:
                # d=256, S>=4096: 128x64 fits via Q-in-regs and beats 64x64 by
                # +6-14% (see sm120_d256_wide above). 256 threads is the A/B win.
                fwd_cfg = FwdConfig(128, 64, True, True)
                num_threads = 256
            elif head_dim > 128:
                # d=256: (128, 64) overflows the 99 KB SMEM cap; shrink to 64x64.
                fwd_cfg = FwdConfig(64, 64, True, True)
            elif (
                batch_size == 1
                and causal
                and not local
                and head_dim == 128
                and head_dim_v == 128
                and qhead_per_kvhead == 4
                and sl == 8192
                and cu_seqlens_q is None
                and cu_seqlens_k is None
                and seqused_q is None
                and seqused_k is None
            ):
                # B=1 qpkv4 S8192 causal favors a smaller M tile on sm120,
                # while the B=2 Qwen/Gemma sweep keeps the lookup path above.
                fwd_cfg = FwdConfig(64, 64, True, True)
            elif (
                batch_size > 1
                and causal
                and not local
                and head_dim == 128
                and head_dim_v == 128
                and qhead_per_kvhead == 4
                and sl == 16384
                and cu_seqlens_q is None
                and cu_seqlens_k is None
                and seqused_q is None
                and seqused_k is None
            ):
                # B>1 qpkv4 S16384 causal validates better with 128x48; B=1
                # keeps the 128x64 lookup entry.
                fwd_cfg = FwdConfig(128, 48, True, True)
            elif (
                batch_size == 1
                and not causal
                and not local
                and q.dtype == torch.bfloat16
                and head_dim == 128
                and head_dim_v == 128
                and num_head == 32
                and num_head_kv == 4
                and qhead_per_kvhead == 8
                and sm120_seq_q in (16384, 32768, 65536, 131072)
                and sm120_seq_k == sm120_seq_q
                and pack_gqa
                and cu_seqlens_q is None
                and cu_seqlens_k is None
                and seqused_q is None
                and seqused_k is None
                and page_table is None
                and qv is None
                and mask_mod is None
                and score_mod is None
                and block_sparse_tensors is None
            ):
                # qwen3-30B-style long noncausal qpkv8 favors the narrower
                # N tile already used by the S8192 lookup entry.
                fwd_cfg = FwdConfig(128, 32, True, True)
            elif (
                batch_size == 1
                and causal
                and not local
                and q.dtype == torch.bfloat16
                and head_dim == 128
                and head_dim_v == 128
                and qhead_per_kvhead == 8
                and sm120_seq_q in (32768, 65536)
                and sm120_seq_k == sm120_seq_q
                and cu_seqlens_q is None
                and cu_seqlens_k is None
                and seqused_q is None
                and seqused_k is None
                and page_table is None
                and qv is None
                and mask_mod is None
                and score_mod is None
                and block_sparse_tensors is None
            ):
                # qwen3-30B-style long causal qpkv8 is sensitive to both tile
                # width and thread count. Keep this exact to avoid disturbing
                # the noisier qpkv8 short/noncausal cells.
                if sm120_seq_q == 65536:
                    fwd_cfg = FwdConfig(128, 32, True, True)
                else:
                    fwd_cfg = FwdConfig(128, 64, True, True)
                    num_threads = 256
            elif (
                head_dim <= 128
                and head_dim_v <= 128
                and cu_seqlens_q is None
                and seqused_q is None
                and page_table is None
                and qv is None
                and sm120_seq_q <= 8
            ):
                # Decode (seqlen_q<=8): the default 128x64 tile wastes the MMA on
                # ~120 empty query rows -> compute-bound (81% SM, 19% DRAM) while
                # decode should be memory-bound. A tiny 16x64 / 1-warp tile cuts
                # the wasted MMA; with the decode SplitKV trigger this is +50-68%
                # on D128 decode (sm120). D256 decode does not benefit (kept on
                # the path below).
                fwd_cfg = FwdConfig(16, 64, True, True)
                num_threads = 32
            elif lookup_key in _SM120_TILE_LOOKUP:
                tm, tn, ns = _SM120_TILE_LOOKUP[lookup_key]
                fwd_cfg = FwdConfig(tm, tn, True, True)
                sm120_num_stages = ns
            else:
                # Conservative fallback for shapes outside the tuned lookup.
                if head_dim <= 64:
                    fwd_cfg = FwdConfig(128, 128, True, True)
                else:  # 64 < head_dim ≤ 128
                    fwd_cfg = FwdConfig(128, 64, True, True)
        elif arch // 10 == 8:
            fwd_cfg = FwdConfig(128, 64, True, True)  # SM80, should tune
        elif arch // 10 == 9:
            sparse_q = get_sparse_q_block_size(block_sparse_tensors, seqlen_q)
            fwd_cfg = _tile_size_fwd_sm90(head_dim, head_dim_v, causal, local, sparse_block_size_q=sparse_q)
    else:
        fwd_cfg = FwdConfig(tile_mn[0], tile_mn[1], fwd_cfg.mma_pv_is_rs, fwd_cfg.intra_wg_overlap)
    tile_m, tile_n = fwd_cfg.m_block_size, fwd_cfg.n_block_size
    if mma_pv_is_rs is None:
        mma_pv_is_rs = fwd_cfg.mma_pv_is_rs
    if intra_wg_overlap is None:
        intra_wg_overlap = fwd_cfg.intra_wg_overlap
    # Long qpkv5 causal D128 runs best with Q staged through registers on SM120:
    # it cuts the non-TMA shared-memory footprint from Q+K+V to max(Q,V)+K.
    sm120_q_in_regs = (
        arch // 10 == 12
        and (
            causal or sm120_d256_qregs128 or sm120_qpkv6_d256_qregs_mode
            or sm120_d256_wide or sm120_local_d256_wide
        )
        and (not local or sm120_local_d256_wide)
        and (
            (
                head_dim == 128
                and head_dim_v == 128
                and qhead_per_kvhead == 5
            )
            or sm120_d256_qregs128
            or sm120_qpkv8_d256_causal_qregs_mode
            or sm120_qpkv16_d256_causal_qregs_mode
            or sm120_qpkv6_d256_qregs_mode
            or sm120_d256_wide
            or sm120_local_d256_wide
        )
        and (
            sm120_seq_q == 8192
            or sm120_seq_q >= 32768
            or sm120_qpkv5_s16384_qregs
            or sm120_d256_qregs128
            or sm120_qpkv8_d256_causal_qregs_mode
            or sm120_qpkv16_d256_causal_qregs_mode
            or sm120_qpkv6_d256_qregs_mode
            or sm120_d256_wide
            or sm120_local_d256_wide
        )
    )
    # SM120 decode auto-split: a small-seqlen_q call (decode / speculative
    # decode) launches only ~batch*num_head_kv CTAs (1 m-block), badly
    # underfilling the 188 SMs while each streams the entire KV cache — 5-10x
    # slower than FA2. Request auto (num_splits=0) HERE, before the pack_gqa
    # disable below, so SplitKV engages with pack_gqa correctly turned off (the
    # GQA+SplitKV combo is unsupported). The num_splits heuristic further down
    # returns 1 when the grid is actually filled (e.g. large batch), so this is
    # self-protecting. Non-varlen / non-paged / non-MLA only.
    if (
        arch // 10 == 12
        and num_splits == 1
        and seqlen_q is not None
        and seqlen_q <= 8
        and cu_seqlens_q is None
        and seqused_q is None
        and page_table is None
        and qv is None
    ):
        num_splits = 0  # request the heuristic (engages SplitKV iff underfilled)

    # GQA + SplitKV + pack_gqa.
    #
    # Upstream #2629 removed the old "GQA + SplitKV + non-varlen" and qv pack_gqa
    # guards as outdated. sm120 already handles SplitKV + pack_gqa correctly: the
    # partial-O/LSE epilogue scatters the packed rows to their correct physical
    # partial slots (pack_gqa.store_O_partial / store_LSE_partial), so pack_gqa
    # stays ENABLED on sm120 for both non-varlen and varlen. With both guards
    # removed, sm120's pack_gqa is no longer disabled here either.
    if max_seqlen_q is None:
        max_seqlen_q = seqlen_q if cu_seqlens_q is None else total_q
    if max_seqlen_k is None:
        max_seqlen_k = seqlen_k
    if cu_seqlens_k is None and seqused_k is None:
        min_seqlen_k = seqlen_k 
    seqlen_q_packgqa = max_seqlen_q * qhead_per_kvhead
    if arch // 10 in [10, 11]:
        q_stage = 2 if seqlen_q_packgqa > tile_m else 1
    else:
        q_stage = 1

    m_block_size_effective = q_stage * tile_m
    if local:
        window_left_loaded = window_size_left if window_size_left is not None else max_seqlen_k
        window_right_loaded = window_size_right if window_size_right is not None else max_seqlen_k
        seqlen_k_loaded = max(
            0,
            min(max_seqlen_k, window_right_loaded + window_left_loaded + 1 + tile_m),
        )
    else:
        seqlen_k_loaded = max_seqlen_k
    num_m_blocks = (seqlen_q_packgqa + m_block_size_effective - 1) // m_block_size_effective
    total_mblocks = batch_size * num_head_kv * num_m_blocks
    num_n_blocks = (seqlen_k_loaded + tile_n - 1) // tile_n
    num_SMs = 132 if is_fake_mode() else torch.cuda.get_device_properties(device).multi_processor_count
    if arch // 10 == 12:
        assert num_splits == 1, "SM120 forward only supports num_splits=1"
    elif num_splits < 1:
        num_splits = num_splits_heuristic(total_mblocks, num_SMs, num_n_blocks, 128)

    # SM120 SplitKV (FlashDecoding-style) is implemented on the SM80-base
    # non-TMA path (FlashAttentionForwardSm120).  The TMA path
    # (FlashAttentionForwardSm120Tma) does not support it; the dispatch below
    # forces the non-TMA path when num_splits > 1.

    # SplitKV uses float32 partial output, which doubles the O buffer size
    # in shared memory, causing OOM for diff-headdim (192, 128)
    if arch // 10 in [10, 11] and head_dim != head_dim_v and num_splits > 1:
        if num_n_blocks >= 64 and head_dim_v != 512:
            tile_n = 64
            num_n_blocks = (seqlen_k_loaded + tile_n - 1) // tile_n
            num_splits = num_splits_heuristic(total_mblocks, num_SMs, num_n_blocks, 128)
        else:
            num_splits = 1

    # learnable_sink + SplitKV is correct on every SplitKV-capable arch: the sink
    # is a single virtual logit, so it must be folded into the LSE exactly once
    # across splits, and each forward does so by applying it only in split 0 —
    # SM100 via flash_fwd_sm100.py (`not is_split_kv or split_idx == 0`, which
    # also handles the empty-split row_max==-inf case), and the SM80-base / SM120
    # forward via compute_sink_val (suppressed to -inf in splits >0, with the
    # matching guard in softmax.finalize). SM90 has no SplitKV. So no single-split
    # fallback is needed. (SM120 verified in-process vs SDPA; SM100/SM80 verified
    # by the split-0 gating in their kernels — no sm100/sm90 hardware available here.)
    is_split_kv = num_splits > 1

    # fp8 KV-cache decode is the only sm_120 path that can consume an fp8 K/V
    # cache, so it must route to the decode kernel even when the split heuristic
    # returns 1 (e.g. large total_mblocks with short seqlen, where the grid is
    # already full).  The decode kernel only supports num_splits>=2 (its
    # ceil_div(seqlen_k, num_splits) mainloop tiler rejects num_splits==1), so for
    # the fp8 path we bump num_splits to 2 and allocate the fp32 partial O/LSE
    # buffers; the combine kernel handles any num_splits.  This only affects the
    # fp8 K/V path (fp8_kv_decode is always False for bf16/fp16 inputs, so the
    # default bf16 dispatch and its num_splits are byte-identical to before).
    want_fp8_decode = (
        fp8_kv_decode
        and arch // 10 == 12
        and seqlen_q is not None
        and seqlen_q == 1
        and qhead_per_kvhead > 1
        and head_dim == head_dim_v
        and head_dim in (128, 256)
        and cu_seqlens_q is None
        and cu_seqlens_k is None
        and seqused_q is None
        and seqused_k is None
        and page_table is None
        and qv is None
        and not local
        and mask_mod is None
        and score_mod is None
        and softcap is None
        and learnable_sink is None
        and block_sparse_tensors is None
        and q_descale is None
        and not is_fake_mode()
        and FlashAttentionDecodeSm120.can_implement(
            dtype, head_dim, head_dim_v, qhead_per_kvhead, 128,
            32 if head_dim == 256 else 64, kv_dtype=kv_dtype,
        )
    )
    # fp8 K/V was passed (dtype assert relaxed above) but the shape/config is not
    # a supported fp8 decode case -> there is NO fp8-capable kernel to fall through
    # to (the standard forward would run the bf16 MMA over reinterpreted fp8 bytes
    # and produce garbage).  Fail loudly instead.  We also block fake mode here:
    # want_fp8_decode excludes fake mode by design, so a fake-mode fp8-KV call
    # would otherwise fall through into the regular SM120 forward path (which is
    # instantiated with dtype=q.dtype, not an fp8-K/V decode signature) and
    # compile the wrong kernel / trip type checks for compile-only callers.
    if fp8_kv_decode and not want_fp8_decode:
        raise NotImplementedError(
            "fp8 (e4m3/e5m2) K/V is only supported for GQA decode on sm_120: "
            "seqlen_q==1, qhead_per_kvhead>1, head_dim in (128,256), bf16/fp16 Q, "
            "no varlen/paged/qv/local/mask_mod/score_mod/softcap/sink/sparsity and "
            "q_descale is None.  Got an unsupported fp8 K/V configuration."
        )
    # The sm120 fp8 KV-cache decode kernel fails to compile on a known-broken
    # nvidia-cutlass-dsl version window (see _fp8_decode_dsl_supported).  Fail loud
    # with an actionable message instead of letting it surface as a confusing DSL
    # compile error.  Do NOT silently fall back to bf16: the K/V cache is physically
    # stored as fp8, so a dtype switch would reinterpret bytes and produce garbage.
    if want_fp8_decode and not _fp8_decode_dsl_supported():
        raise NotImplementedError(_FP8_DECODE_DSL_ERROR)
    if want_fp8_decode and num_splits < 2:
        num_splits = 2
        is_split_kv = True
    if is_split_kv or want_fp8_decode:
        out_partial = torch.empty(num_splits, *q_batch_seqlen_shape, num_head, head_dim_v, dtype=torch.float32, device=device)
        lse_partial = torch.empty(num_splits, *lse_shape, dtype=torch.float32, device=device)

    # ----------------------------------------------------------------------
    # SM120 memory-bound decode kernel (gated, off by default).
    # A from-scratch GEMV decode path: one CTA per (split, kv_head, batch)
    # processes all qhead_per_kvhead query rows together (KV read once, no GQA
    # redundancy) using FMA + warp shuffles instead of the wasteful m16n8k16
    # MMA over empty query rows.  Produces the same fp32 partial O / LSE the
    # combine kernel expects, then reuses _flash_attn_fwd_combine.
    # ----------------------------------------------------------------------
    # Gate: fp8 K/V auto-routes here unconditionally (want_fp8_decode — it is the
    # only fp8-capable sm_120 path), while the bf16 decode kernel stays behind the
    # env flag AND is_split_kv exactly as before.  For bf16 inputs want_fp8_decode
    # is always False and fp8_kv_decode is always False, so when the env flag is
    # off this whole condition is False and the default path is byte-identical.
    env_decode_kernel = (
        os.environ.get("FLASH_ATTENTION_SM120_DECODE_KERNEL", "0").lower()
        in ("1", "true", "on", "yes")
    )
    if want_fp8_decode or (
        env_decode_kernel
        and arch // 10 == 12
        and is_split_kv
        and seqlen_q is not None
        and seqlen_q == 1
        and qhead_per_kvhead > 1
        and head_dim == head_dim_v
        and head_dim in (128, 256)
        and cu_seqlens_q is None
        and cu_seqlens_k is None
        and seqused_q is None
        and seqused_k is None
        and page_table is None
        and qv is None
        # seqlen_q==1 (decode): bottom-right causal == attend all keys, so a
        # causal flag is a no-op here and the kernel needs no causal masking.
        and not local
        and mask_mod is None
        and score_mod is None
        and softcap is None
        and learnable_sink is None
        and block_sparse_tensors is None
        and q_descale is None
        and not is_fake_mode()
        and FlashAttentionDecodeSm120.can_implement(
            dtype, head_dim, head_dim_v, qhead_per_kvhead, 128,
            32 if head_dim == 256 else 64, kv_dtype=kv_dtype,
        )
    ):
        decode_tile_n = 32 if head_dim == 256 else 64
        decode_key = (dtype, kv_dtype, head_dim, qhead_per_kvhead, num_splits, decode_tile_n,
                      k_descale is not None, v_descale is not None)
        if decode_key not in _flash_attn_fwd.decode_compile_cache:
            fa_decode = FlashAttentionDecodeSm120(
                dtype, head_dim, head_dim_v, qhead_per_kvhead, num_splits,
                tile_n=decode_tile_n, num_threads=128, kv_dtype=kv_dtype,
            )
            q_t = to_cute_tensor(q.detach())
            k_t = to_cute_tensor(k.detach())
            v_t = to_cute_tensor(v.detach())
            op_t = to_cute_tensor(out_partial, assumed_align=4)
            lp_t = to_cute_tensor(lse_partial, assumed_align=4)
            kd_t = to_cute_tensor(k_descale, assumed_align=4, leading_dim=1) if k_descale is not None else None
            vd_t = to_cute_tensor(v_descale, assumed_align=4, leading_dim=1) if v_descale is not None else None
            _flash_attn_fwd.decode_compile_cache[decode_key] = cute.compile(
                fa_decode, q_t, k_t, v_t, op_t, lp_t,
                Float32(softmax_scale), kd_t, vd_t, current_stream,
                options="--enable-tvm-ffi",
            )
        # torch <2.11 can't DLPack-export fp8; pass the raw bytes as uint8 and let
        # the cute kernel reinterpret (matches the prefill fp8 path).
        k_call = k.detach().view(torch.uint8) if fp8_kv_decode else k.detach()
        v_call = v.detach().view(torch.uint8) if fp8_kv_decode else v.detach()
        _flash_attn_fwd.decode_compile_cache[decode_key](
            q.detach(), k_call, v_call,
            out_partial, lse_partial, Float32(softmax_scale),
            *( (k_descale,) if k_descale is not None else () ),
            *( (v_descale,) if v_descale is not None else () ),
        )
        _flash_attn_fwd_combine(
            out_partial,
            lse_partial.transpose(-1, -2),
            out,
            lse.transpose(-1, -2) if lse is not None else None,
            None,
            None,
        )
        return out, lse

    use_2cta_instrs = (
        arch // 10 in [10, 11]
        and not requested_disable_2cta
        and not causal
        and not local
        and not is_split_kv
        and cu_seqlens_q is None
        and seqused_q is None
        and not use_block_sparsity
        and page_size in [None, 128]
        and int(math.ceil(head_dim / 16) * 16) in [128, 192]
        and int(math.ceil(head_dim_v / 16) * 16) == 128
        and seqlen_q_packgqa > 2 * tile_m
        and (tile_m % qhead_per_kvhead == 0 or not pack_gqa)
    )

    # hd=256 2CTA forward uses dedicated kernel (Blackwell family)
    use_dedicated_hd256_kernel = arch // 10 in [10, 11] and head_dim == 256 and head_dim_v == 256
    use_2cta_instrs = use_2cta_instrs or use_dedicated_hd256_kernel

    if softcap is not None:
        assert score_mod is None, "softcap and score_mod cannot be used together"
        score_mod = utils.create_softcap_scoremod(softcap)
    elif score_mod is not None:
        if arch // 10 == 8:
            raise NotImplementedError("Custom user-provided score_mod is not supported on SM8x architectures.")
        
    # hash score and mask mods for compile cache
    score_mod_hash = utils.hash_callable(score_mod) if score_mod is not None else False
    mask_mod_hash = utils.hash_callable(mask_mod) if mask_mod is not None else False

    is_varlen = (
        cu_seqlens_q is not None
        or cu_seqlens_k is not None
        or seqused_q is not None
        or seqused_k is not None
    )

    # CLC regressed for varlen MHA and dense noncausal. Imbalanced varlen shapes
    # keep more K/V blocks in flight and hurt L2; dense noncausal mostly just
    # pays work-stealing overhead.
    is_varlen_mha = is_varlen and qhead_per_kvhead == 1
    is_dense_noncausal = not is_varlen and not causal and not local
    use_clc_scheduler = requested_use_clc_scheduler and not is_varlen_mha and not is_dense_noncausal

    if use_block_sparsity:
        # NB: pack_gqa requires block sparse head dim == 1 (broadcasted)
        head_dim_idx = 0 if block_sparse_tensors.mask_block_cnt.ndim == 2 else 1
        if pack_gqa and block_sparse_tensors.mask_block_cnt.shape[head_dim_idx] != 1:
            pack_gqa = False
        if arch // 10 in [8, 12] and (cu_seqlens_q is not None or cu_seqlens_k is not None):
            raise NotImplementedError(
                "Varlen block sparsity is not supported on SM80/SM120 forward; "
                "the SM80-base block-sparse mainloop uses non-varlen block indices."
            )
        if cu_seqlens_q is not None:
            assert block_sparse_tensors.cu_total_m_blocks is not None, (
                "Varlen block sparsity requires block_sparse_tensors.cu_total_m_blocks."
            )
            if (
                block_sparse_tensors.cu_block_idx_offsets is None
                and (cu_seqlens_k is not None or seqused_k is not None)
            ):
                raise ValueError(
                    "Varlen block sparsity with cu_seqlens_k or seqused_k requires "
                    "block_sparse_tensors.cu_block_idx_offsets."
                )

    pack_gqa_all_rows_valid = (
        arch // 10 == 12
        and pack_gqa
        and cu_seqlens_q is None
        and cu_seqlens_k is None
        and seqused_q is None
        and seqused_k is None
        and (seqlen_q * qhead_per_kvhead) % tile_m == 0
    )
    sm120_pack_gqa_fast_valid_rows = (
        arch // 10 == 12
        and pack_gqa_all_rows_valid
        and not causal
        and not local
        and head_dim == 128
        and head_dim_v == 128
        and qhead_per_kvhead in (4, 8)
        and not use_block_sparsity
    )
    sm120_skip_dense_seqlen_mask = (
        arch // 10 == 12
        and not causal
        and not local
        and mask_mod is None
        and page_table is None
        and cu_seqlens_q is None
        and cu_seqlens_k is None
        and seqused_q is None
        and seqused_k is None
        and not use_block_sparsity
        and seqlen_k % tile_n == 0
    )
    # Exact qpkv5 S4096 noncausal runs faster on the SM80-base path than on
    # the SM120 TMA path; keep a narrow env override for validation/profiling.
    sm120_qpkv5_s4096_nc_exact = (
        arch // 10 == 12
        and q.dtype == torch.bfloat16
        and batch_size == 2
        and not causal
        and not local
        and head_dim == 128
        and head_dim_v == 128
        and qhead_per_kvhead == 5
        and sm120_seq_q == 4096
        and sm120_seq_k == 4096
        and not pack_gqa
        and score_mod is None
        and mask_mod is None
        and page_table is None
        and cu_seqlens_q is None
        and cu_seqlens_k is None
        and seqused_q is None
        and seqused_k is None
        and not use_block_sparsity
    )
    sm120_tma_kv_stages = 2
    sm120_qpkv5_s4096_nc_notma = sm120_qpkv5_s4096_nc_exact
    # Keep this narrow: plain bf16 qpkv6 D256 dense kernels benefit from shorter K/V copy
    # live ranges, while qpkv4 and local-window variants regressed in validation.
    sm120_qpkv6_d256_load_hooks = (
        arch // 10 == 12
        and q.dtype == torch.bfloat16
        and head_dim == 256
        and head_dim_v == 256
        and qhead_per_kvhead == 6
        and not local
        and not pack_gqa
        and score_mod is None
        and mask_mod is None
        and page_table is None
        and cu_seqlens_q is None
        and cu_seqlens_k is None
        and seqused_q is None
        and seqused_k is None
        and not use_block_sparsity
        and tile_m == 64
        and tile_n == 64
        and sm120_num_stages == 1
    )
    sm120_qpkv6_d256_hook_mode = ""
    if sm120_qpkv6_d256_load_hooks:
        # Qwen qpkv6 D256 rows prefer shortening only the V live range on the
        # reproduced long-shape wins. K-only loses, S16384 causal flipped in
        # validation, and S131072 did not hold up in the broad FA2/FA4 sweep.
        if (
            sm120_seq_q == sm120_seq_k
            and sm120_seq_q in (16384, 32768, 65536)
            and not causal
        ) or (
            sm120_seq_q == sm120_seq_k
            and sm120_seq_q in (32768, 65536)
            and causal
        ):
            sm120_qpkv6_d256_hook_mode = "v"
    if (
        sm120_qpkv6_d256_qregs_mode
        and batch_size == 2
        and (
            (not causal and sm120_seq_q == 4096)
            or (causal and sm120_seq_q == 8192)
        )
    ):
        sm120_qpkv6_d256_hook_mode = "v"
    sm120_qpkv6_d256_static_causal_default = (
        sm120_qpkv6_d256_load_hooks
        and causal
        and sm120_seq_q == sm120_seq_k
        # S16384 added. Static causal block bounds is a clean +1.6% here on
        # sm120 (controlled A/B); this
        # B=2 row uses no Q-regs (qregs is B=2 S4096/8192 only), so the
        # qregs+static wrong-output combo does not apply. Gain scales with S
        # (+1.6% S16384, +2.8% S32768). S8192 excluded (qregs is on there).
        and sm120_seq_q in (16384, 32768, 65536)
    )
    sm120_qpkv6_d256_static_causal_blocks = sm120_qpkv6_d256_static_causal_default
    sm120_qpkv5_d128_hook_eligible = (
        arch // 10 == 12
        and q.dtype == torch.bfloat16
        and head_dim == 128
        and head_dim_v == 128
        and qhead_per_kvhead == 5
        and causal
        and not local
        and not pack_gqa
        and score_mod is None
        and mask_mod is None
        and page_table is None
        and cu_seqlens_q is None
        and cu_seqlens_k is None
        and seqused_q is None
        and seqused_k is None
        and not use_block_sparsity
        and sm120_num_stages == 1
    )
    # qpkv5 causal rows are sensitive to both seqlen and batch. Keep this exact
    # to the measured per-shape winners instead of applying a broad qpkv5 rule.
    sm120_qpkv5_d128_default_hook_mode = ""
    if sm120_qpkv5_d128_hook_eligible:
        if sm120_seq_q == 8192:
            sm120_qpkv5_d128_default_hook_mode = "both"
        elif sm120_seq_q == 16384:
            sm120_qpkv5_d128_default_hook_mode = "v"
        elif sm120_seq_q in (32768, 65536):
            sm120_qpkv5_d128_default_hook_mode = "both"
        elif sm120_seq_q >= 131072:
            sm120_qpkv5_d128_default_hook_mode = "v"
    sm120_qpkv5_d128_hook_mode = sm120_qpkv5_d128_default_hook_mode
    sm120_hook_load_k = sm120_qpkv6_d256_hook_mode in {"k", "both"} or (
        sm120_qpkv5_d128_hook_eligible and sm120_qpkv5_d128_hook_mode in {"k", "both"}
    )
    sm120_hook_load_v = sm120_qpkv6_d256_hook_mode in {"v", "both"} or (
        sm120_qpkv5_d128_hook_eligible and sm120_qpkv5_d128_hook_mode in {"v", "both"}
    )
    # See get_broadcast_dims for why this is needed in compile key
    block_sparse_broadcast_pattern = None
    normalized_block_sparse_tensors = None
    q_subtile_factor = 1
    kv_subtile_factor = 1
    if block_sparse_tensors is not None:
        block_sparse_config = normalize_block_sparse_config(
            block_sparse_tensors,
            batch_size=batch_size,
            num_head=num_head,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            block_size=(tile_m, tile_n),
            q_stage=q_stage,
            allow_kv_subtile=arch // 10 in [10, 11],
        )
        normalized_block_sparse_tensors = block_sparse_config.tensors
        block_sparse_broadcast_pattern = block_sparse_config.broadcast_pattern
        q_subtile_factor = block_sparse_config.q_subtile_factor
        kv_subtile_factor = block_sparse_config.kv_subtile_factor
    if aux_tensors is not None:
        aux_tensor_metadata = get_aux_tensor_metadata(aux_tensors)
    else:
        aux_tensor_metadata = None
    aux_scalar_metadata = tuple(type(s) for s in aux_scalars) if aux_scalars is not None else None

    if qv is not None:
        assert arch // 10 in [10, 11], "only support Blackwell arch with qv"
        assert q is None or qv.shape[:-1] == q.shape[:-1]
        assert qv.shape[-1] == head_dim_v
        assert head_dim_v == 512
        assert q is None or head_dim == 64
        assert not local, "local not yet supported with qv"
        assert q_descale is None and k_descale is None and v_descale is None, (
            "q_descale/k_descale/v_descale are not yet supported with qv"
        )
        assert tile_n == 128

        assert not is_split_kv, "split kv not supported with qv"
        assert learnable_sink is None
        assert softcap is None
        assert score_mod is None
        assert mask_mod is None

        if page_table is not None:
            assert gather_kv_indices is None, "paged KV + topk sparsity not yet supported together"
        
        qv = maybe_contiguous(qv)

        gather_kv_length = 2048  # dummy value
        sparse_kv = gather_kv_indices is not None
        # always use kv bitmask by default (handles -1 sentinel)
        disable_sparse_kv_bitmask = False
        if sparse_kv:
            assert gather_kv_indices.shape[:-1] == qv.shape[:-2]
            gather_kv_length = gather_kv_indices.shape[-1]
            assert gather_kv_length % 128 == 0
            # if min_seqlen_k is None or causal:
            #     disable_sparse_kv_bitmask = False
            # else:
            #     # seqlen_k_boundary = min_seqlen_k - max_seqlen_q + 1 if causal else min_seqlen_k
            #     seqlen_k_boundary = min_seqlen_k
            #     disable_sparse_kv_bitmask = seqlen_k_boundary >= gather_kv_length
        
        if requires_grad and sparse_kv:
            if cu_seqlens_q is None:
                p = torch.empty(batch_size, seqlen_q, num_head, gather_kv_length, dtype=q_dtype, device=device)
                row_max = torch.empty(batch_size, seqlen_q, gather_kv_length//128, num_head, dtype=torch.float32, device=device)
            else:
                p = torch.empty(total_q, num_head, gather_kv_length, dtype=q_dtype, device=device)
                row_max = torch.empty(total_q, gather_kv_length//128, num_head, dtype=torch.float32, device=device)
        else:
            p = row_max = None
    else:
        assert gather_kv_indices is None, "gather_kv_indices is only supported with qv"
        gather_kv_length = None
        sparse_kv = None
        disable_sparse_kv_bitmask = None
        p = row_max = None

    compile_key = (
        dtype,
        head_dim,
        head_dim_v,
        qhead_per_kvhead,
        causal,
        score_mod_hash,
        mask_mod_hash,
        use_block_sparsity,
        block_sparse_broadcast_pattern,
        aux_tensor_metadata,
        aux_scalar_metadata,
        lse is None,
        cu_seqlens_q is None,
        cu_seqlens_k is None,
        seqused_q is None,
        seqused_k is None,
        page_table is not None,
        window_size_left is not None,
        window_size_right is not None,
        learnable_sink is not None,
        q_descale is not None,
        k_descale is not None,
        v_descale is not None,
        block_sparse_tensors is None or block_sparse_tensors.cu_total_m_blocks is None,
        block_sparse_tensors is None or block_sparse_tensors.cu_block_idx_offsets is None,
        tile_m,
        tile_n,
        q_stage,
        num_threads,
        is_split_kv,
        # SM120 SplitKV bakes num_splits into the grid shape and the
        # scheduler's FastDivmodDivisor as a compile-time constant, so
        # kernels compiled for different num_splits must not share a key.
        num_splits if (arch // 10 == 12 and is_split_kv) else None,
        pack_gqa,
        pack_gqa_all_rows_valid,
        sm120_pack_gqa_fast_valid_rows if arch // 10 == 12 else None,
        arch,
        page_size not in [None, tile_n],  # paged KV non-TMA
        # On SM120 the SM80-base paged-KV mainloop bakes
        # page_size assumptions into FastDivmodDivisor; without keying on
        # page_size, reusing the kernel across calls with different
        # page_size values produces cudaErrorIllegalAddress.
        page_size if (arch // 10 == 12 and page_size is not None) else None,
        # SM120 forward picks num_stages per shape from the tile lookup;
        # different lookup entries with the same (tile_m, tile_n) but differing
        # num_stages would otherwise share a compile_key and silently reuse the
        # first-compiled kernel.
        sm120_num_stages if arch // 10 == 12 else None,
        # The SM120 TMA forward's K/V pipeline depth (kv_stages) changes the
        # compiled kernel (SMEM layout / pipeline), so it must be in the key.
        # Constant today, but keying it now keeps any future kv_stages tuning from
        # silently reusing a binary compiled with a different depth.
        sm120_tma_kv_stages if arch // 10 == 12 else None,
        sm120_skip_dense_seqlen_mask if arch // 10 == 12 else None,
        ("notma" if sm120_qpkv5_s4096_nc_notma else "") if arch // 10 == 12 else None,
        sm120_q_in_regs if arch // 10 == 12 else None,
        sm120_hook_load_k if arch // 10 == 12 else None,
        sm120_hook_load_v if arch // 10 == 12 else None,
        sm120_qpkv6_d256_static_causal_blocks if arch // 10 == 12 else None,
        use_2cta_instrs,
        q_subtile_factor,
        kv_subtile_factor,
        mma_pv_is_rs,
        intra_wg_overlap,
        use_clc_scheduler,
        q is not None,
        qv is not None,
        p is not None,
        row_max is not None,
        gather_kv_length,
        sparse_kv,
        disable_sparse_kv_bitmask,
        fa_logging.get_fa_log_level(),
    )

    if compile_key not in _flash_attn_fwd.compile_cache:
        (
            cu_seqlens_q_tensor,
            cu_seqlens_k_tensor,
            seqused_q_tensor,
            seqused_k_tensor,
            learnable_sink_tensor,
        ) = [
            to_cute_tensor(t, assumed_align=4, leading_dim=0)
            if t is not None
            else None
            for t in (cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, learnable_sink)
        ]
        page_table_tensor = (
            to_cute_tensor(page_table, assumed_align=4, leading_dim=1)
            if page_table is not None
            else None
        )
        q_tensor, k_tensor, v_tensor, o_tensor = [
            to_cute_tensor(t) for t in (q, k, v, out if not is_split_kv else out_partial)
        ]
        if is_split_kv:
            lse_tensor = to_cute_tensor(lse_partial, assumed_align=4)
        else:
            lse_tensor = to_cute_tensor(lse, assumed_align=4)

        q_descale_tensor, k_descale_tensor, v_descale_tensor = (
            to_cute_tensor(t, assumed_align=4, leading_dim=1)
            for t in (q_descale, k_descale, v_descale)
        )
        descale_tensors_tensor = (
            DescaleTensors(
                q_descale=q_descale_tensor,
                k_descale=k_descale_tensor,
                v_descale=v_descale_tensor,
            )
            if q_descale_tensor is not None
            or k_descale_tensor is not None
            or v_descale_tensor is not None
            else None
        )

        sparse_tensors = None
        if normalized_block_sparse_tensors is not None:
            sparse_tensors = to_cute_block_sparse_tensors(normalized_block_sparse_tensors)

        cute_aux_tensors = None
        aux_tensor_metadata = None
        if aux_tensors is not None:
            cute_aux_tensors = [to_cute_aux_tensor(buf) for buf in aux_tensors]

        qv_tensor = to_cute_tensor(qv)
        gather_kv_indices_tensor = to_cute_tensor(gather_kv_indices)
        window_size_left_cute = _to_cute_int32_or_none(window_size_left)
        window_size_right_cute = _to_cute_int32_or_none(window_size_right)
        p_tensor = to_cute_tensor(p)
        row_max_tensor = to_cute_tensor(row_max)

        if arch // 10 == 8:
            assert page_table is None, "paged KV not supported on SM 8.0"
            assert not is_split_kv, "SplitKV not supported on SM 8.0"
            fa_fwd = FlashAttentionForwardSm80(
                dtype,
                head_dim,
                head_dim_v,
                qhead_per_kvhead,
                is_causal=causal,
                is_local=local,
                pack_gqa=pack_gqa,
                tile_m=tile_m,
                tile_n=tile_n,
                num_stages=1,
                num_threads=num_threads,
                Q_in_regs=False,
                score_mod=score_mod,
                mask_mod=mask_mod,
                has_aux_tensors=aux_tensors is not None,
            )
        elif arch // 10 == 9:
            assert not is_split_kv, "SplitKV not supported on SM 9.0"
            fa_fwd = FlashAttentionForwardSm90(
                dtype,
                head_dim,
                head_dim_v,
                qhead_per_kvhead,
                is_causal=causal,
                is_local=local,
                pack_gqa=pack_gqa,
                tile_m=tile_m,
                tile_n=tile_n,
                # num_stages=1,
                num_stages=2,
                num_threads=num_threads,
                Q_in_regs=False,
                intra_wg_overlap=intra_wg_overlap,
                mma_pv_is_rs=mma_pv_is_rs,
                mask_mod=mask_mod,
                score_mod=score_mod,
                has_aux_tensors=aux_tensors is not None,
                q_subtile_factor=q_subtile_factor,
                paged_kv_non_tma=page_size not in [None, tile_n],
            )
        elif arch // 10 in [10, 11]:
            if qv is not None:
                paged_kv_cpasync = page_table is not None and page_size != tile_n
                has_qk = q is not None
                fa_fwd = FlashAttentionMLAForwardSm100(
                    is_causal=causal,
                    use_cpasync_load_KV=sparse_kv or paged_kv_cpasync,
                    topk_length=gather_kv_length,
                    is_topk_gather=sparse_kv,
                    pack_gqa=pack_gqa,
                    qhead_per_kvhead=qhead_per_kvhead,
                    nheads_kv=num_head_kv,
                    has_seqused_q=seqused_q is not None,
                    has_cu_seqlens_q=cu_seqlens_q is not None,
                    disable_bitmask=disable_sparse_kv_bitmask,
                    has_qk=has_qk,
                )
            else:
                if use_dedicated_hd256_kernel:
                    # hd=256 2CTA forward: check for currently unsupported features
                    assert softcap is None, "SM100 forward with head_dim=256 does not support softcap"
                    assert not use_block_sparsity, \
                        "SM100 forward with head_dim=256 does not support block sparsity"
                    assert learnable_sink is None, \
                        "SM100 forward with head_dim=256 does not support learnable_sink"
                    assert seqused_q is None and seqused_k is None, \
                        "SM100 forward with head_dim=256 does not support seqused_q/seqused_k"
                    if page_table is not None:
                        assert max_seqlen_k % page_size == 0, (
                            f"SM100 hd256 2CTA paged KV requires max_seqlen_k divisible by "
                            f"page_size ({page_size}), got max_seqlen_k={max_seqlen_k}"
                        )
                        assert page_table.shape[1] == max_seqlen_k // page_size, (
                            f"SM100 hd256 2CTA paged KV requires page_table.shape[1] == "
                            f"max_seqlen_k // page_size ({max_seqlen_k} // {page_size} = "
                            f"{max_seqlen_k // page_size}), got {page_table.shape[1]}; "
                            f"pass page_table[:, :{max_seqlen_k // page_size}] to slice to "
                            f"the actual sequence length"
                        )
                    # pack_gqa is an auto-selected optimization; disable it for hd256 kernel
                    pack_gqa = False

                flash_fwd_obj_cls = (
                    BlackwellFusedMultiHeadAttentionForward
                    if use_dedicated_hd256_kernel
                    else FlashAttentionForwardSm100
                )

                fa_fwd = flash_fwd_obj_cls(
                    head_dim,
                    head_dim_v,
                    qhead_per_kvhead=qhead_per_kvhead,
                    is_causal=causal,
                    is_local=local,
                    is_split_kv=is_split_kv,
                    pack_gqa=pack_gqa,
                    m_block_size=tile_m,
                    n_block_size=tile_n,
                    q_stage=q_stage,
                    is_persistent=not causal
                        and not local
                        and cu_seqlens_q is None
                        and seqused_q is None
                        and not is_split_kv,
                    score_mod=score_mod,
                    mask_mod=mask_mod,
                    has_aux_tensors=aux_tensors is not None,
                    paged_kv_non_tma=page_size not in [None, tile_n],
                    is_varlen_q=cu_seqlens_q is not None or seqused_q is not None,
                    q_subtile_factor=q_subtile_factor,
                    kv_subtile_factor=kv_subtile_factor,
                    use_2cta_instrs=use_2cta_instrs,
                    use_clc_scheduler=use_clc_scheduler,
                )
        elif arch // 10 == 12:
            # SM120 (Blackwell GeForce / DGX Spark): SM80 MMA with 99 KB SMEM.
            # Paged-KV for head_dim > 128 runs through this non-TMA path on
            # SM120. The tile picker keeps those rows at 64x64, which fits the
            # 99 KB SMEM cap; PagedKVManager supports tile_n < num_threads by
            # allocating ceil(tile_n / num_threads) page-table slots.
            # The TMA kernel builds a fixed (tile_m, tile_hdim) Q TMA atom
            # from the unpacked layout, so pack_gqa=True must take the
            # SM80-base path (which calls pack_gqa_layout). is_varlen here
            # is the outer-scope value that includes seqused_q/seqused_k —
            # do not narrow it to only cu_seqlens.
            use_tma_sm120 = (
                page_table is None
                and not is_varlen
                and not use_block_sparsity
                and not pack_gqa
                and learnable_sink is None
                and not is_split_kv
                and not sm120_qpkv5_s4096_nc_notma
            )
            if use_tma_sm120 and FlashAttentionForwardSm120Tma.can_implement(
                dtype, head_dim, head_dim_v, tile_m, tile_n,
                num_mma_warps=4, kv_stages=sm120_tma_kv_stages, is_causal=causal,
            ):
                fa_fwd = FlashAttentionForwardSm120Tma(
                    dtype,
                    head_dim,
                    head_dim_v,
                    qhead_per_kvhead,
                    is_causal=causal,
                    is_local=local,
                    pack_gqa=pack_gqa,
                    tile_m=tile_m,
                    tile_n=tile_n,
                    num_mma_warps=4,
                    kv_stages=sm120_tma_kv_stages,
                    score_mod=score_mod,
                    mask_mod=mask_mod,
                    has_aux_tensors=aux_tensors is not None,
                    skip_dense_seqlen_mask=sm120_skip_dense_seqlen_mask,
                )
            else:
                # can_implement gates configs that would either overflow SMEM
                # or fault on bad head_dim divisibility. head_dim > head_dim_v
                # is supported on this (non-TMA) path; the TMA path still
                # rejects it via can_implement and falls through here.
                assert FlashAttentionForwardSm120.can_implement(
                    dtype, head_dim, head_dim_v, tile_m, tile_n,
                    num_stages=sm120_num_stages, num_threads=num_threads, is_causal=causal,
                    Q_in_regs=sm120_q_in_regs,
                ), (
                    f"FlashAttentionForwardSm120 cannot implement "
                    f"(head_dim={head_dim}, head_dim_v={head_dim_v}, "
                    f"tile_m={tile_m}, tile_n={tile_n}) on SM 12.0. "
                    f"Common causes: "
                    f"tile_m*head_dim + 2*tile_n*head_dim*num_stages > 99 KB, "
                    f"or head_dim not divisible by 8."
                )
                fa_fwd = FlashAttentionForwardSm120(
                    dtype,
                    head_dim,
                    head_dim_v,
                    qhead_per_kvhead,
                    is_causal=causal,
                    is_local=local,
                    pack_gqa=pack_gqa,
                    tile_m=tile_m,
                    tile_n=tile_n,
                    num_stages=sm120_num_stages,
                    num_threads=num_threads,
                    Q_in_regs=sm120_q_in_regs,
                    score_mod=score_mod,
                    mask_mod=mask_mod,
                    has_aux_tensors=aux_tensors is not None,
                    pack_gqa_all_rows_valid=pack_gqa_all_rows_valid,
                    pack_gqa_fast_valid_rows=sm120_pack_gqa_fast_valid_rows,
                    skip_dense_seqlen_mask=sm120_skip_dense_seqlen_mask,
                    hook_load_k=sm120_hook_load_k,
                    hook_load_v=sm120_hook_load_v,
                    static_causal_blocks=sm120_qpkv6_d256_static_causal_blocks,
                    is_split_kv=is_split_kv,
                    num_splits=num_splits,
                    # Block-sparse: when the sparse Q block size exceeds the kernel
                    # tile_m (e.g. a 256-wide BlockMask block run with a 128 tile),
                    # q_subtile_factor maps each kernel m_block to its owning sparse
                    # block (m_block // factor). Without it the kernel uses factor=1
                    # and reads past the (smaller) sparse m-block dim -> wrong output
                    # + illegal memory access. SM80/SM100 pass this; SM120 was the
                    # lone omission.
                    q_subtile_factor=q_subtile_factor,
                )
        else:
            raise ValueError(
                f"Unsupported compute capability: {arch}. Supported: 8.x, 9.x, 10.x, 11.x, 12.x"
            )
        # TODO: check @can_implement for non-SM120 paths too
        if qv is not None:
            _flash_attn_fwd.compile_cache[compile_key] = cute.compile(
                fa_fwd,
                q_tensor,
                qv_tensor,
                k_tensor,
                v_tensor,
                o_tensor,
                lse_tensor,
                softmax_scale,
                p_tensor,
                row_max_tensor,
                cu_seqlens_q_tensor,
                cu_seqlens_k_tensor,
                seqused_q_tensor,
                seqused_k_tensor,
                gather_kv_indices_tensor,
                page_table_tensor,
                window_size_left_cute,
                window_size_right_cute,
                current_stream,
                options="--enable-tvm-ffi",
            )
        else:
            compile_args = [
                fa_fwd,
                q_tensor,
                k_tensor,
                v_tensor,
                o_tensor,
                lse_tensor,
                softmax_scale,
                cu_seqlens_q_tensor,
                cu_seqlens_k_tensor,
                seqused_q_tensor,
                seqused_k_tensor,
                page_table_tensor,
                window_size_left_cute,
                window_size_right_cute,
                learnable_sink_tensor,
            ]
            if arch // 10 in [10, 11]:
                compile_args.append(descale_tensors_tensor)
            compile_args.extend([
                sparse_tensors,
                AuxData(cute_aux_tensors, aux_scalars),
            ])
            compile_args.append(current_stream)
            _flash_attn_fwd.compile_cache[compile_key] = cute.compile(
                *compile_args, options="--enable-tvm-ffi"
            )

    if not is_fake_mode():
        window_size_left_cute = _to_cute_int32_or_none(window_size_left)
        window_size_right_cute = _to_cute_int32_or_none(window_size_right)
        q_call, k_call, v_call, qv_call = [
            t.detach() if t is not None else None
            for t in (q, k, v, qv)
        ]
        if is_fp8:
            # need uint8 workaround until we pin torch >= 2.11.0 where fp8 export is supported
            q_call, k_call, v_call, qv_call = [
                t.view(torch.uint8) if t is not None else None
                for t in (q_call, k_call, v_call, qv_call)
            ]
        descale_tensors = (
            DescaleTensors(q_descale=q_descale, k_descale=k_descale, v_descale=v_descale)
            if q_descale is not None or k_descale is not None or v_descale is not None
            else None
        )
        if qv is not None:
            _flash_attn_fwd.compile_cache[compile_key](
                q_call,
                qv_call,
                k_call,
                v_call,
                out.detach(),
                lse,
                softmax_scale,
                p,
                row_max,
                cu_seqlens_q,
                cu_seqlens_k,
                seqused_q,
                seqused_k,
                gather_kv_indices,
                page_table,
                window_size_left_cute,
                window_size_right_cute,
            )
        else:
            call_args = [
                q_call,
                k_call,
                v_call,
                out.detach() if not is_split_kv else out_partial,
                lse_partial if is_split_kv else lse,
                softmax_scale,
                cu_seqlens_q,
                cu_seqlens_k,
                seqused_q,
                seqused_k,
                page_table,
                window_size_left_cute,
                window_size_right_cute,
                learnable_sink,
            ]
            if arch // 10 in [10, 11]:
                call_args.append(descale_tensors)
            call_args.extend([
                (
                    normalized_block_sparse_tensors.mask_block_cnt,
                    normalized_block_sparse_tensors.mask_block_idx,
                    normalized_block_sparse_tensors.full_block_cnt,
                    normalized_block_sparse_tensors.full_block_idx,
                    normalized_block_sparse_tensors.cu_total_m_blocks,
                    normalized_block_sparse_tensors.cu_block_idx_offsets,
                    normalized_block_sparse_tensors.dq_write_order,
                    normalized_block_sparse_tensors.dq_write_order_full,
                )
                if normalized_block_sparse_tensors is not None
                else None,
                AuxData(aux_tensors, aux_scalars),
            ])
            _flash_attn_fwd.compile_cache[compile_key](*call_args)
    if is_split_kv:
        _flash_attn_fwd_combine(
            out_partial,
            lse_partial.transpose(-1, -2),
            out,
            lse.transpose(-1, -2) if lse is not None else None,
            cu_seqlens_q,
            seqused_q,
        )
    return out, lse, p, row_max


_flash_attn_fwd.compile_cache = get_jit_cache("fwd")
_flash_attn_fwd.decode_compile_cache = {}


def make_fake_bwd_tensors(dtype, has_gqa, varlen_q, varlen_k, nheads_major=False):
    sym = cute.sym_int
    # divisibility in elements: assumed_align_bytes = divisibility * dtype.width // 8
    # For 16-byte align: fp16/bf16 → divisibility=8, float32 → divisibility=4
    div = 128 // dtype.width  # 8 for fp16/bf16
    # Shared sym_ints for dimensions that must match across tensors
    b, seqlen_q, seqlen_k, h_q, d, d_v = sym(), sym(), sym(), sym(), sym(), sym()
    topk = sym()
    h_kv = h_q if not has_gqa else sym()
    seqlen_q_rounded, seqlen_k_rounded = sym(), sym()
    seqlen_q_d_rounded, seqlen_k_d_rounded, seqlen_k_dv_rounded = sym(), sym(), sym()
    total_q, total_k, total_q_rounded, total_k_rounded = sym(), sym(), sym(), sym()
    total_q_d_rounded, total_k_d_rounded, total_k_dv_rounded = sym(), sym(), sym()
    b_seqlenq = (b, seqlen_q) if not varlen_q else (total_q,)
    b_seqlenk = (b, seqlen_k) if not varlen_k else (total_k,)
    mQ = fake_tensor(dtype, (*b_seqlenq, h_q, d), divisibility=div)
    mO = fake_tensor(dtype, (*b_seqlenq, h_q, d_v), divisibility=div)
    mdO = fake_tensor(dtype, (*b_seqlenq, h_q, d_v), divisibility=div)
    mK = fake_tensor(dtype, (*b_seqlenk, h_kv, d), divisibility=div)
    mV = fake_tensor(dtype, (*b_seqlenk, h_kv, d_v), divisibility=div)
    mdQ = fake_tensor(dtype, (*b_seqlenq, h_q, d), divisibility=div)
    mdK = fake_tensor(dtype, (*b_seqlenk, h_kv, d), divisibility=div)
    mdV = fake_tensor(dtype, (*b_seqlenk, h_kv, d_v), divisibility=div)

    sq    = seqlen_q         if not varlen_q else total_q
    sq_r  = seqlen_q_rounded if not varlen_q else total_q_rounded
    sq_dr = seqlen_q_d_rounded if not varlen_q else total_q_d_rounded

    def shape(*dims):
        batch = (b,) if not varlen_q else ()
        return (*batch, h_q, *dims) if not nheads_major else (*batch, *dims, h_q)

    mLSE     = fake_tensor(Float32, shape(sq),       divisibility=1)
    mLSElog2 = fake_tensor(Float32, shape(sq_r),     divisibility=4)
    mPdPsum  = fake_tensor(Float32, shape(sq_r),     divisibility=4)
    dQaccum  = fake_tensor(Float32, shape(sq_dr),    divisibility=4)
    mScaleP  = fake_tensor(Float32, shape(sq, topk), divisibility=4)

    if not has_gqa:
        mdKaccum, mdVaccum = None, None
    else:
        if not varlen_k:
            mdKaccum = fake_tensor(Float32, (b, h_kv, seqlen_k_rounded), divisibility=4)
            mdVaccum = fake_tensor(Float32, (b, h_kv, seqlen_k_dv_rounded), divisibility=4)
        else:
            mdKaccum = fake_tensor(Float32, (h_kv, total_k_rounded), divisibility=4)
            mdVaccum = fake_tensor(Float32, (h_kv, total_k_dv_rounded), divisibility=4)
    return mQ, mK, mV, mO, mdO, mdQ, mdK, mdV, mLSE, mLSElog2, mPdPsum, dQaccum, mdKaccum, mdVaccum, mScaleP


def _compile_bwd_preprocess(
    dtype,
    head_dim,
    head_dim_v,
    m_block_size,
    has_cuseqlens_q,
    has_seqused_q,
    has_dlse,
    has_dq_accum,
    has_scaleP,
    use_padded_offsets,
    nheads_major,
    pack_gqa,
    qhead_per_kvhead,
    nheads_kv,
):
    """Compile bwd preprocess kernel using cute fake tensors (no real GPU tensors needed)."""
    mQ, mK, mV, mO, mdO, mdQ, mdK, mdV, mLSE, mLSElog2, mPdPsum, mdQaccum, mdKaccum, mdVaccum, mScaleP = make_fake_bwd_tensors(
        dtype, has_gqa=True, varlen_q=has_cuseqlens_q, varlen_k=False, nheads_major=nheads_major,
    )
    batch = mQ.shape[0] if not has_cuseqlens_q else cute.sym_int()
    batchp1 = cute.sym_int()
    mCuSeqlensQ = fake_tensor(Int32, (batchp1,), divisibility=1) if has_cuseqlens_q else None
    mSequsedQ = fake_tensor(Int32, (batch,), divisibility=1) if has_seqused_q else None
    mdLSE = fake_tensor(Float32, mLSE.shape, divisibility=1) if has_dlse else None
    mLSElog2 = None if has_scaleP else mLSElog2
    mdQaccum = mdQaccum if has_dq_accum else None
    mRowMax = fake_tensor(Float32, mScaleP.shape, divisibility=1) if has_scaleP else None
    mScaleP = fake_tensor(Float32, mScaleP.shape, divisibility=1) if has_scaleP else None
    softmax_scale = Float32(1.0)
    fa_bwd_pre = FlashAttentionBackwardPreprocess(
        dtype, head_dim, head_dim_v, m_block_size,
        use_padded_offsets=use_padded_offsets,
        nheads_major=nheads_major,
        pack_gqa=pack_gqa,
        qhead_per_kvhead=qhead_per_kvhead,
        nheads_kv=nheads_kv,
    )
    return cute.compile(
        fa_bwd_pre, mO, mdO, mPdPsum, mLSE, mLSElog2, mdQaccum, mCuSeqlensQ, mSequsedQ, mdLSE,
        mRowMax, mScaleP, softmax_scale,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _bwd_preprocess(
    out, dout, dpsum, lse, lse_log2, dq_accum,
    cu_seqlens_q, seqused_q, dlse,
    dtype, head_dim, head_dim_v, m_block_size,
    row_max=None,
    scale_p=None,
    use_padded_offsets=True,
    nheads_major=False,
    pack_gqa=False,
    qhead_per_kvhead=1,  # only used with pack_gqa
    nheads_kv=1,         # only used with pack_gqa
    softmax_scale=1.0,   # only used with scale_p
):
    """Backward preprocess: compute (o * dout).sum(dim=-1) - dLSE, lse * log2_e, and zero out dq_accum."""
    if row_max is not None:
        assert scale_p is not None
    compile_key = (
        dtype, head_dim, head_dim_v, m_block_size,
        cu_seqlens_q is not None,
        seqused_q is not None,
        dlse is not None,
        dq_accum is not None,
        row_max is not None,
        use_padded_offsets,
        nheads_major,
        pack_gqa,
        qhead_per_kvhead,
        nheads_kv,
    )
    if compile_key not in _bwd_preprocess.compile_cache:
        _bwd_preprocess.compile_cache[compile_key] = _compile_bwd_preprocess(*compile_key)
    if not is_fake_mode():
        _bwd_preprocess.compile_cache[compile_key](
            out, dout, dpsum, lse, lse_log2, dq_accum, cu_seqlens_q, seqused_q, dlse,
            row_max, scale_p,
            softmax_scale,
        )


_bwd_preprocess.compile_cache = get_jit_cache("bwd_pre")


def _compile_bwd_postprocess(
    dtype, hdim, block_size, num_threads, atom_layout, swap_ab,
    has_cuseqlens_q, has_seqused_q,
    use_2cta_instrs, cluster_size, arch,
    pack_gqa=False, qhead_per_kvhead=1,
):
    """Compile bwd postprocess kernel using cute fake tensors."""
    mQ, mK, mV, mO, mdO, mdQ, mdK, mdV, mLSE, mLSElog2, mPdPsum, mdQaccum, mdKaccum, mdVaccum, mScaleP = make_fake_bwd_tensors(
        dtype, has_gqa=True, varlen_q=has_cuseqlens_q, varlen_k=False
    )
    batch = mQ.shape[0] if not has_cuseqlens_q else cute.sym_int()
    batchp1 = cute.sym_int()
    mCuSeqlensQ = fake_tensor(Int32, (batchp1,), divisibility=1) if has_cuseqlens_q else None
    mSeqUsedQ = fake_tensor(Int32, (batch,), divisibility=1) if has_seqused_q else None
    fa_bwd_post = FlashAttentionBackwardPostprocess(
        dtype, hdim, arch, block_size, num_threads, atom_layout, swap_ab,
        use_2cta_instrs=use_2cta_instrs,
        cluster_size=cluster_size,
        pack_gqa=pack_gqa,
        qhead_per_kvhead=qhead_per_kvhead,
    )
    return cute.compile(
        fa_bwd_post, mdQaccum, mdQ, Float32(0.0), mCuSeqlensQ, mSeqUsedQ,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _bwd_postprocess_convert(
    accum, output, scale,
    cu_seqlens, seqused,
    arch, dtype, hdim, block_size, num_threads,
    atom_layout, swap_ab,
    use_2cta_instrs=False, cluster_size=1,
    pack_gqa=False, qhead_per_kvhead=1,
):
    """Backward postprocess: convert float32 accumulator to bf16/fp16 output."""
    compile_key = (
        dtype, hdim, block_size, num_threads, atom_layout, swap_ab,
        cu_seqlens is not None, seqused is not None,
        use_2cta_instrs, cluster_size, arch,
        pack_gqa, qhead_per_kvhead,
    )
    if compile_key not in _bwd_postprocess_convert.compile_cache:
        _bwd_postprocess_convert.compile_cache[compile_key] = _compile_bwd_postprocess(*compile_key)
    if not is_fake_mode():
        _bwd_postprocess_convert.compile_cache[compile_key](
            accum, output, scale, cu_seqlens, seqused,
        )


_bwd_postprocess_convert.compile_cache = get_jit_cache("bwd_post")


def _compile_bwd_postprocess_dkv_sm120(
    dtype, hdim, block_size, num_threads, atom_layout,
):
    """Compile fused fixed-length SM120 dK+dV postprocess kernel."""
    _, _, _, _, _, _, mdK, mdV, _, _, _, _, mdKaccum, mdVaccum = make_fake_bwd_tensors(
        dtype, has_gqa=True, varlen_q=False, varlen_k=False
    )
    fa_bwd_post_dkv = FlashAttentionBackwardDkvPostprocessSm120(
        dtype, hdim, block_size, num_threads, atom_layout,
    )
    return cute.compile(
        fa_bwd_post_dkv,
        mdKaccum,
        mdVaccum,
        mdK,
        mdV,
        Float32(0.0),
        Float32(0.0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _bwd_postprocess_dkv_sm120(
    dk_accum, dv_accum, dk, dv, softmax_scale,
    dtype, hdim, block_size, num_threads, atom_layout,
):
    """Fused fixed-length SM120 dK+dV postprocess."""
    if dk.shape[-1] != dv.shape[-1]:
        raise NotImplementedError(
            "SM120 fused dK+dV postprocess requires dK and dV to have the same head_dim"
        )
    compile_key = (dtype, hdim, block_size, num_threads, atom_layout)
    if compile_key not in _bwd_postprocess_dkv_sm120.compile_cache:
        _bwd_postprocess_dkv_sm120.compile_cache[compile_key] = (
            _compile_bwd_postprocess_dkv_sm120(*compile_key)
        )
    if not is_fake_mode():
        _bwd_postprocess_dkv_sm120.compile_cache[compile_key](
            dk_accum, dv_accum, dk, dv, softmax_scale, 1.0,
        )


_bwd_postprocess_dkv_sm120.compile_cache = get_jit_cache("bwd_post_dkv_sm120")


def _sm120_use_fused_dkv_postprocess(
    *,
    arch: int,
    dtype,
    dkv_postprocess: bool,
    pack_gqa: bool,
    pack_gqa_m_splits: int,
    qhead_per_kvhead: int,
    causal: bool,
    local: bool,
    seqlen_q: int,
    seqlen_k: int,
    cu_seqlens_k,
    seqused_k,
    head_dim: int,
    head_dim_v: int,
    dKV_swapAB: bool,
) -> bool:
    """Select the fused fixed-length SM120 dK+dV postprocess kernel."""
    eligible = (
        arch // 10 == 12
        and dtype in (cutlass.BFloat16, cutlass.Float16)
        and dkv_postprocess
        and cu_seqlens_k is None
        and seqused_k is None
        and head_dim == head_dim_v
        and not dKV_swapAB
    )
    sm120_qpkv8_s1024_causal = (
        dtype == cutlass.BFloat16
        and qhead_per_kvhead == 8
        and causal
        and not local
        and seqlen_q == seqlen_k
        and seqlen_q == 1024
        and head_dim == 256
        and head_dim_v == 256
    )
    return eligible and (
        (pack_gqa and pack_gqa_m_splits > 1)
        or sm120_qpkv8_s1024_causal
    )


def _flash_attn_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    softcap: float = 0.0,
    window_size_left: Optional[int] = None,
    window_size_right: Optional[int] = None,
    m_block_size: int = 64,
    n_block_size: int = 128,
    num_threads: int = 256,
    pack_gqa: bool = False,
    num_stages_Q: int = 2,
    num_stages_dO: int = 2,
    SdP_swapAB: bool = False,
    dKV_swapAB: bool = False,
    dQ_swapAB: bool = False,
    AtomLayoutMSdP: int = 2,
    AtomLayoutNdKV: int = 2,
    AtomLayoutMdQ: int = 2,
    V_in_regs: bool = False,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    deterministic: bool = False,
    dq: Optional[torch.Tensor] = None,
    dk: Optional[torch.Tensor] = None,
    dv: Optional[torch.Tensor] = None,
    score_mod: Optional[Callable] = None,
    score_mod_bwd: Optional[Callable] = None,
    mask_mod: Optional[Callable] = None,
    aux_tensors: Optional[list[torch.Tensor]] = None,
    aux_scalars: Optional[tuple] = None,
    block_sparse_tensors: Optional[BlockSparseTensorsTorch] = None,
    dlse: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    aux_scalars = tuple(aux_scalars) if aux_scalars else None
    arch = _get_device_arch()
    assert arch // 10 in [9, 10, 11, 12], "Unsupported compute capability. Supported: 9.x, 10.x, 11.x, 12.x"
    sparse_q = None
    kv_subtile_factor = 1
    if block_sparse_tensors is not None:
        if block_sparse_tensors.block_size is not None:
            sparse_q = block_sparse_tensors.block_size[0]
        elif arch // 10 == 9:
            sparse_q = 128

    num_head, head_dim = q.shape[-2:]
    head_dim_v = v.shape[-1]

    window_size = [window_size_left, window_size_right]
    causal, local, window_size_left, window_size_right = _resolve_causal_local_window(
        causal, window_size_left, window_size_right
    )

    if arch // 10 == 12:
        # SM120: uses SM80 MMA with 99 KB SMEM, 256 threads (8 warps).
        m_block_size = 64
        n_block_size = 64
        if (
            head_dim <= 64
            and head_dim_v <= 64
            and not local
            and cu_seqlens_q is None
            and cu_seqlens_k is None
        ):
            # D<=64 backward: a 64x128 tile halves the K/V-block grid and the
            # per-CTA Q/dO gmem re-read volume (each n-CTA streams all m-blocks).
            # Smem at 64x128xD64 is 80 KB (<99 KB cap). Round-wise isolated A/B
            # on RTX PRO 6000: 24-cell new/old geomean 1.088 (21/24 wins),
            # S8192nc +22%, S16384nc +23%; closes the D64 FA2 gap from ~0.85
            # to ~0.98. Matches FA2's sm86/89 hdim64 kBlockN=128 choice.
            # Halving the n-grid underfills tiny grids (qpkv8 Hq8 B1 S4096 and
            # S512-class cells regressed 4-6%), so require >= 2 waves at n=128.
            _grid_n128 = ((k.shape[1] + 127) // 128) * q.shape[-2] * q.shape[0]
            _sm_count = torch.cuda.get_device_properties(q.device).multi_processor_count
            if _grid_n128 >= 2 * _sm_count:
                n_block_size = 128
        # D<=64 long-seq: register-resident P/dS (Mma_dKV_is_RS) dK/dV gemms.
        # With the 64x128 tile, AtomLayoutMSdP=1 + SdP_swapAB + AtomLayoutNdKV=8
        # keeps P^T/dS^T in registers as direct A-operands of the dV/dK gemms
        # (FA2's hdim64 structure), skipping the sP smem round trip. Round-wise
        # isolated A/B on RTX PRO 6000 (vs the non-RS default, incl. its nsq=2
        # rule): S8192nc +4.3%, S16384nc +4.7%, GQA qpkv4 S8192nc +4.8%,
        # causal S>=8192 +2.2-2.8% (all min-round ratios > 1.017). S4096 is
        # neutral and B4 S1024 mildly negative, so gate on seqlen >= 8192.
        # RS prefers num_stages_Q=1 (pinned below): the extra Q stage costs
        # smem/sync without hiding latency here (+1.9% over RS w/ nsq=2).
        # pack_gqa is excluded (mask.py has no swap_AB + PackGQA support);
        # auto-pack_gqa only triggers for D256 so only explicit requests hit it.
        _sm120_bwd_rs = (
            n_block_size == 128
            and q.shape[1] >= 8192
            and pack_gqa is not True
        )
        # num_stages=1 across all head_dim on consumer Blackwell. At
        # head_dim>64 the SMEM cap forces ns=1; at head_dim<=64 the SM80-base
        # default was ns=2 but the async pipeline overhead exceeds the
        # latency-hiding benefit at small tile size. Tightened paired
        # validation (n_measure=30, interleaved trials) confirms geomean
        # speedup ~1.06x on 19 d=64 cells with 0 regressions >2%.
        num_stages_Q = 1
        num_stages_dO = 1
        # D128 long-seq backward is under-pipelined at stages=1. Unlike
        # D256 (which needs the smem alias and is capped at ns=1), D128 has room
        # for a 2nd Q stage; at S>=8192 the long mainloop makes pipelining the Q
        # loads a consistent ~2% win (controlled A/B; gradients match SDPA).
        # Asymmetric (Q=2, dO=1) keeps smem under the 99KB cap (symmetric ns=2
        # overflows and fails to launch). Short seq stays ns=1 (async overhead
        # dominates the latency-hiding benefit there).
        if (
            head_dim <= 128
            and head_dim_v <= 128
            and cu_seqlens_q is None
            and q.shape[1] >= 8192
            and not _sm120_bwd_rs
        ):
            num_stages_Q = 2
        SdP_swapAB = False
        dKV_swapAB = False
        dQ_swapAB = False
        AtomLayoutMSdP = 4
        AtomLayoutNdKV = 4
        AtomLayoutMdQ = 4
        if _sm120_bwd_rs:
            # See comment above (_sm120_bwd_rs): RS register-resident dK/dV.
            SdP_swapAB = True
            AtomLayoutMSdP = 1
            AtomLayoutNdKV = 8  # num_threads // 32; required by Mma_dKV_is_RS
        if head_dim == 128 and head_dim_v == 128:
            # FA2's sm8x d128 choice (NdKV=2): the dK/dV tiled-mma covers the
            # full 64-wide head_dim slab per atom iteration instead of
            # splitting it 4 ways. 10-round isolated A/B on RTX PRO 6000:
            # positive on 7/7 cells (causal/noncausal, MHA/GQA, S1024-S16384),
            # +0.6% to +2.8%, median ~+1.1%.
            AtomLayoutNdKV = 2
        V_in_regs = False
        dQ_single_wg = False
        cluster_size = 1
        use_2cta_instrs = False
        num_threads = 256
        dQ_single_wg = True
        assert not (block_sparse_tensors is not None), "Block sparsity backward not supported on SM 12.0"
        assert score_mod is None and score_mod_bwd is None, "score_mod backward not supported on SM 12.0"
        assert mask_mod is None, "mask_mod backward not supported on SM 12.0"
        # Not an SM120-specific SMEM issue: the SM80 base kernel itself uses
        # raw atomic_add_fp32 for dQ accumulation and asserts on mdQ_semaphore
        # being None (see flash_bwd.py:~395). The semaphore-based dQ scheduler
        # for deterministic writes only exists in SM90/SM100.
        assert deterministic is False, (
            "deterministic backward not supported on SM 12.0 "
            "(SM80 base kernel lacks the dQ_semaphore code path; "
            "see flash_bwd.py:~395 'determinism not supported yet for Sm80')"
        )
        # Experimental config override for kernel-structure A/B probing.
        # Format: comma-separated key=val pairs, e.g.
        #   FLASH_ATTENTION_SM120_BWD_CFG="m=64,n=128,t=256,nsq=1,nsdo=1,msdp=1,ndkv=8,mdq=4,swapsdp=1,swapdkv=0,swapdq=0,vregs=0"
        # Unspecified keys keep the dispatch defaults above. All overridden
        # values flow into the compile_key, so cached kernels stay distinct.
        _sm120_bwd_cfg = os.environ.get("FLASH_ATTENTION_SM120_BWD_CFG")
        if _sm120_bwd_cfg:
            _cfg = dict(kv.split("=") for kv in _sm120_bwd_cfg.split(",") if kv)
            m_block_size = int(_cfg.get("m", m_block_size))
            n_block_size = int(_cfg.get("n", n_block_size))
            num_threads = int(_cfg.get("t", num_threads))
            num_stages_Q = int(_cfg.get("nsq", num_stages_Q))
            num_stages_dO = int(_cfg.get("nsdo", num_stages_dO))
            AtomLayoutMSdP = int(_cfg.get("msdp", AtomLayoutMSdP))
            AtomLayoutNdKV = int(_cfg.get("ndkv", AtomLayoutNdKV))
            AtomLayoutMdQ = int(_cfg.get("mdq", AtomLayoutMdQ))
            SdP_swapAB = bool(int(_cfg.get("swapsdp", SdP_swapAB)))
            dKV_swapAB = bool(int(_cfg.get("swapdkv", dKV_swapAB)))
            dQ_swapAB = bool(int(_cfg.get("swapdq", dQ_swapAB)))
            V_in_regs = bool(int(_cfg.get("vregs", V_in_regs)))

        # Pre-launch shared-memory guard. The SM120 backward shares the SM80
        # kernel body, whose tile footprint (sQ + sdO + sK + sV + sP + sdS +
        # sLSE/sdPsum) overflows the ~99 KB sm_120/sm_121 SMEM cap for some
        # head dims — most notably equal-dims head_dim == head_dim_v == 192,
        # which needs ~115 KB and otherwise fails at launch with an opaque
        # cudaErrorInvalidValue. head_dim == head_dim_v == 256 fits because it
        # uses the Q/dO + K/V smem-reuse path. Raise a clear error here instead.
        # (head_dim=192 with head_dim_v=128 is supported and stays well under
        # the cap.) Mirrors FlashAttentionBackwardSm80._get_shared_storage_cls.
        _hd_pad = (head_dim + 31) // 32 * 32
        _hdv_pad = (head_dim_v + 31) // 32 * 32
        _reuse_qk_dov = (
            _hd_pad == 256 and _hdv_pad == 256 and num_stages_Q == 1 and num_stages_dO == 1
        )
        if not _reuse_qk_dov:
            _smem_bwd = (
                m_block_size * _hd_pad * num_stages_Q * 2      # sQ
                + m_block_size * _hdv_pad * num_stages_dO * 2  # sdO
                + n_block_size * _hd_pad * 2                   # sK
                + n_block_size * _hdv_pad * 2                  # sV
                + 2 * (m_block_size * n_block_size * 2)        # sP + sdS
                + 2 * (((m_block_size + 63) // 64 * 64) * num_stages_Q * 4)  # sLSE + sdPsum
            )
            _SM120_SMEM_CAP = 99 * 1024  # sm_120 / sm_121a opt-in SMEM cap (101376 B)
            if _smem_bwd > _SM120_SMEM_CAP:
                raise NotImplementedError(
                    f"SM120 backward is not supported for head_dim={head_dim}, "
                    f"head_dim_v={head_dim_v}: the {m_block_size}x{n_block_size} tile needs "
                    f"~{_smem_bwd} B of shared memory, exceeding the {_SM120_SMEM_CAP} B "
                    f"sm_120/sm_121 cap. Equal-dims head_dim=head_dim_v=192 is a known "
                    f"limitation; use head_dim=192 with head_dim_v=128 (supported), or a "
                    f"head_dim<=160 / 256 configuration."
                )
    elif arch // 10 == 9:
        cfg = _tile_size_bwd_sm90(
            head_dim,
            head_dim_v,
            causal,
            local,
            sparse_block_size_q=sparse_q,
        )
        m_block_size = cfg.m_block_size
        n_block_size = cfg.n_block_size
        num_stages_Q = cfg.num_stages_Q
        num_stages_dO = cfg.num_stages_dO
        num_stages_PdS = cfg.num_stages_PdS
        SdP_swapAB = cfg.SdP_swapAB
        dKV_swapAB = cfg.dKV_swapAB
        dQ_swapAB = cfg.dQ_swapAB
        AtomLayoutMSdP = cfg.AtomLayoutMSdP
        AtomLayoutNdKV = cfg.AtomLayoutNdKV
        AtomLayoutMdQ = cfg.AtomLayoutMdQ
        num_threads = (cfg.num_wg + 1) * 128
        dQ_single_wg = cfg.dQ_single_wg
        cluster_size = 1
        use_2cta_instrs = False
        is_varlen = (
            cu_seqlens_q is not None
            or cu_seqlens_k is not None
            or seqused_q is not None
            or seqused_k is not None
        )
    else:
        m_block_size = 128
        n_block_size = 128
        dQ_swapAB = False
        dKV_swapAB = False
        AtomLayoutMdQ = 1
        AtomLayoutNdKV = 1
        requested_disable_2cta = utils._get_disable_2cta_default()
        kv_subtile_factor = get_kv_subtile_factor(block_sparse_tensors, n_block_size)
        use_2cta_instrs = (
            head_dim >= 128
            and not requested_disable_2cta
            and block_sparse_bwd_supports_2cta(block_sparse_tensors, n_block_size)
        )
        if block_sparse_tensors is not None and head_dim == 192 and not use_2cta_instrs:
            reason = (
                "2CTA was disabled by request"
                if requested_disable_2cta
                else (
                    f"sparse_block_size[1] must cover an even number of tile_n={n_block_size} "
                    f"tiles; got factor {kv_subtile_factor}"
                )
            )
            raise ValueError(
                f"SM100 block-sparse backward with head_dim=192 requires 2CTA; {reason}."
            )
        cluster_size = 2 if use_2cta_instrs else 1

    use_dedicated_hd256_kernel = arch // 10 in [10, 11] and head_dim == 256 and head_dim_v == 256
    use_2cta_instrs = use_2cta_instrs or use_dedicated_hd256_kernel

    q, k, v, out, dout, lse, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k = [
        maybe_contiguous(t)
        for t in (q, k, v, out, dout, lse, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k)
    ]
    # Under full-model torch.compile, transformers derives max_seqlen from
    # position_ids/cu_seqlens *inside the compiled forward graph*, so it reaches
    # the eager backward as a 0-d device int32 Tensor rather than a host Python
    # int. These scalars feed host-side Python control flow below (the seqlen
    # comparisons forming `sm120_skip_full_causal_mask`, the compile_cache key,
    # and the kernel's `cutlass.const_expr(...)` conditions). A Tensor there
    # turns Python `and`/`==` into per-element ops, ultimately tripping the
    # cutlass DSL `and_op` (`DSLNotImplemented: torch.Tensor is not supported`)
    # at flash_bwd.py's `skip_full_causal_mask and is_causal`. Materialize them
    # to host ints. No-op for the normal eager path (values are already ints).
    if isinstance(max_seqlen_q, torch.Tensor):
        max_seqlen_q = int(max_seqlen_q.item())
    if isinstance(max_seqlen_k, torch.Tensor):
        max_seqlen_k = int(max_seqlen_k.item())
    if cu_seqlens_q is None:
        batch_size, seqlen_q = q.shape[:2]
        total_q = batch_size * seqlen_q
    else:
        batch_size = cu_seqlens_q.shape[0] - 1
        total_q = q.shape[0]
        seqlen_q = max_seqlen_q if max_seqlen_q is not None else total_q

    if cu_seqlens_k is None:
        batch_size, seqlen_k = k.shape[:2]
        total_k = batch_size * seqlen_k
    else:
        batch_size = cu_seqlens_k.shape[0] - 1
        total_k = k.shape[0]
        seqlen_k = max_seqlen_k if max_seqlen_k is not None else total_k

    num_head_kv = k.shape[-2]

    use_block_sparsity = block_sparse_tensors is not None
    if sparse_q is not None and (sparse_q <= 0 or sparse_q % m_block_size != 0):
        raise ValueError(
            "Block sparsity requires sparse_block_size[0] to be a multiple of "
            f"tile_m={m_block_size}; got {sparse_q}."
        )
    q_subtile_factor = sparse_q // m_block_size if sparse_q is not None else 2
    seqlen_q_rounded = (seqlen_q + m_block_size - 1) // m_block_size * m_block_size
    seqlen_k_rounded = (seqlen_k + n_block_size - 1) // n_block_size * n_block_size
    num_n_blocks = seqlen_k_rounded // n_block_size
    if cluster_size == 2 and num_n_blocks % cluster_size != 0:
        seqlen_k_rounded = seqlen_k_rounded + n_block_size

    # The single-block specialization below only guards against TVM stride poisoning,
    # which is a host-side branch predicate that selects a kernel variant. When
    # max_seqlen is passed as a tensor (e.g. HF/TE varlen), seqlen_*_rounded are tensors,
    # so `seqlen_*_rounded // block == 1` would leak a tensor into the compile key. Its
    # pickle hash differs every call, forcing a recompile per step. Only specialize when
    # the seqlen is already a host scalar; tensor callers fall back to the multi-block
    # default, keeping the key stable with no device sync.
    single_q_block = (not torch.is_tensor(seqlen_q_rounded)) and (seqlen_q_rounded // m_block_size == 1)
    single_k_block = (not torch.is_tensor(seqlen_k_rounded)) and (seqlen_k_rounded // n_block_size == 1)

    if cu_seqlens_k is None:
        assert k.shape == (batch_size, seqlen_k, num_head_kv, head_dim)
        assert v.shape == (batch_size, seqlen_k, num_head_kv, head_dim_v)
    else:
        assert k.shape == (total_k, num_head_kv, head_dim)
        assert v.shape == (total_k, num_head_kv, head_dim_v)
        assert cu_seqlens_k.shape == (batch_size + 1,), (
            "cu_seqlens_k must have shape (batch_size + 1,)"
        )

    if cu_seqlens_q is not None:
        assert cu_seqlens_q.shape == (batch_size + 1,), (
            "cu_seqlens_q must have shape (batch_size + 1,)"
        )

        assert out.shape == (total_q, num_head, head_dim_v)
        assert dout.shape == (total_q, num_head, head_dim_v)
        assert lse.shape == (num_head, total_q), "lse must have shape (num_head, total_q)"
    else:
        assert out.shape == (batch_size, seqlen_q, num_head, head_dim_v)
        assert dout.shape == (batch_size, seqlen_q, num_head, head_dim_v)
        assert lse.shape == (batch_size, num_head, seqlen_q), (
            "lse must have shape (batch_size, num_head, seqlen_q)"
        )

    assert q.dtype in [torch.float16, torch.bfloat16], "inputs must be float16 or bfloat16"
    assert q.dtype == k.dtype == v.dtype == out.dtype == dout.dtype, (
        "inputs must have the same dtype"
    )
    for t in [cu_seqlens_q, cu_seqlens_k]:
        if t is not None:
            assert t.dtype == torch.int32, "cu_seqlens_q, cu_seqlens_k must be int32"
    assert lse.dtype == torch.float32, "lse must be float32"
    if dlse is not None:
        dlse = maybe_contiguous(dlse)
    if not is_fake_mode():
        assert all(
            t is None or t.is_cuda for t in (q, k, v, out, dout, lse, cu_seqlens_q, cu_seqlens_k)
        ), "inputs must be on CUDA device"
    assert num_head % num_head_kv == 0, "num_head must be divisible by num_head_kv"
    alignment = 16 // q.element_size()
    if arch // 10 != 8:
        _validate_head_dims(head_dim, head_dim_v, arch // 10, alignment)
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    qhead_per_kvhead = num_head // num_head_kv
    pack_gqa_requested = pack_gqa is True
    pack_gqa_auto = pack_gqa is None
    if pack_gqa is None:
        pack_gqa = qhead_per_kvhead > 1
    sm120_auto_pack_gqa_bwd = (
        arch // 10 == 12
        and pack_gqa_auto
        and q.dtype == torch.bfloat16
        and not local
        and head_dim == 256
        and head_dim_v == 256
        and (
            (
                not causal
                and qhead_per_kvhead == 8
            )
            or (
                not causal
                and qhead_per_kvhead == 4
                and seqlen_q == seqlen_k
                and seqlen_q == 8192
            )
            or (
                not causal
                and qhead_per_kvhead == 2
                and num_head == 32
                and num_head_kv == 16
                and seqlen_q == seqlen_k
                and seqlen_q in (4096, 8192, 16384)
            )
            or (
                not causal
                and qhead_per_kvhead == 6
                and num_head == 24
                and num_head_kv == 4
                and seqlen_q == seqlen_k
                and seqlen_q == 4096
            )
            or (
                not causal
                and qhead_per_kvhead == 16
                and num_head == 32
                and num_head_kv == 2
                and seqlen_q == seqlen_k
                and seqlen_q in (4096, 8192)
            )
            or (
                causal
                and qhead_per_kvhead == 4
                and seqlen_q == seqlen_k
                and (
                    seqlen_q == 1024
                    or (
                        seqlen_q == 2048
                        and batch_size == 2
                        and num_head in (8, 16)
                        and num_head_kv == num_head // qhead_per_kvhead
                    )
                )
            )
        )
        and cu_seqlens_q is None
        and cu_seqlens_k is None
        and seqused_q is None
        and seqused_k is None
    )
    # pack_gqa is now supported in the SM120 backward kernel
    # as an explicit opt-in.  Keep auto-selection disabled for most SM120
    # backward shapes; the packed Q/dO row-pointer path is only a measured
    # win for narrow fixed dense bf16 D256 rows. Other archs (SM80/SM90/SM100)
    # retain the original "not yet supported" override.
    if arch // 10 == 12 and pack_gqa and not (pack_gqa_requested or sm120_auto_pack_gqa_bwd):
        pack_gqa = False
    if (
        arch // 10 == 12
        and pack_gqa
        and (
            cu_seqlens_q is not None
            or cu_seqlens_k is not None
            or seqused_q is not None
            or seqused_k is not None
        )
    ):
        # The explicit SM120 packed backward path is tuned for fixed-length
        # dense GQA. Varlen/seqused keeps the correct nonpacked GQA fallback.
        pack_gqa = False
    if not (arch // 10 == 12):
        pack_gqa = False
    pack_gqa_m_splits = _sm120_bwd_pack_gqa_m_splits(
        arch=arch,
        pack_gqa=pack_gqa,
        qhead_per_kvhead=qhead_per_kvhead,
        num_head=num_head,
        num_head_kv=num_head_kv,
        causal=causal,
        local=local,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        m_block_size=m_block_size,
        n_block_size=n_block_size,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        batch_size=batch_size,
    )
    # SM120 nonpacked causal-D256 M-split policy (RTX PRO 6000, 188 SMs).
    # Splitting the nonpacked M loop adds CTAs and only helps when the backward
    # grid (~ceil(S/64) * B * Hq CTAs) underfills the SMs (<~3 waves); the split
    # counts below are the measured per-shape A/B peaks, gated to the rows that
    # win. Larger-grid rows are flat/harmful and left unsplit. This avoids the
    # rejected N32 / explicit-PackGQA changes.
    sm120_nonpack_base_ok = (
        arch // 10 == 12
        and not pack_gqa
        and q.dtype == torch.bfloat16
        and causal
        and not local
        and head_dim == 256
        and head_dim_v == 256
        and seqlen_q == seqlen_k
        and m_block_size == 64
        and n_block_size == 64
        and cu_seqlens_q is None
        and cu_seqlens_k is None
        and seqused_q is None
        and seqused_k is None
    )
    sm120_nonpack_m_split = 1
    if sm120_nonpack_base_ok:
        is_qpkv8_h8 = qhead_per_kvhead == 8 and num_head == 8 and num_head_kv == 1
        is_qpkv2_gemma31 = (
            qhead_per_kvhead == 2 and num_head == 32 and num_head_kv == 16
        )
        if seqlen_q == 1024:
            # qpkv2 Gemma31 is a clean S1024 win on sm120.
            # B=1 halves the grid, so both qpkv8 rows (Hq8/Hkv1 and Hq16/Hkv2)
            # want split4 (+6% / +9% vs the B>=2-tuned split3 / split2).
            if batch_size == 1 and qhead_per_kvhead == 8:
                sm120_nonpack_m_split = 4
            elif is_qpkv8_h8:
                sm120_nonpack_m_split = 3
            elif qhead_per_kvhead in (6, 8) or is_qpkv2_gemma31:
                sm120_nonpack_m_split = 2
        elif seqlen_q == 2048:
            # B=1 halves the grid so the small-grid qpkv4/qpkv6/qpkv8 rows
            # (num_head<=24, ~<3 waves) still underfill and gain +4-9% from
            # split4; qpkv4 at B=1 prefers nonpack split4 over packing here.
            # At B>=2 only the smallest grid (qpkv8 Hq8/Hkv1) underfills (+6%,
            # split3); larger qpkv4/6/8 rows are filled (split flat/harmful).
            if batch_size == 1 and qhead_per_kvhead in (4, 6, 8) and num_head <= 24:
                sm120_nonpack_m_split = 4
            elif is_qpkv8_h8:
                sm120_nonpack_m_split = 3
        elif seqlen_q == 4096:
            # Only the smallest grids still underfill at B=1: qpkv8 Hq8/Hkv1
            # (+10%, split6) and qpkv4 Hq8/Hkv2 (+7%, split4). qpkv8 Hq16/Hkv2,
            # qpkv6, and qpkv4 Hq16/Hkv4 are filled by S4096 (flat).
            if batch_size == 1 and is_qpkv8_h8:
                sm120_nonpack_m_split = 6
            elif (
                batch_size == 1
                and qhead_per_kvhead == 4
                and num_head == 8
                and num_head_kv == 2
            ):
                sm120_nonpack_m_split = 4
    sm120_nonpack_m_split_eligible = sm120_nonpack_base_ok and sm120_nonpack_m_split > 1
    if sm120_nonpack_m_split_eligible:
        pack_gqa_m_splits = sm120_nonpack_m_split
    # Experimental m-split override (probing only); piggybacks on the
    # FLASH_ATTENTION_SM120_BWD_CFG hook, key "msplit". pack_gqa_m_splits is
    # part of the compile_key so overridden values cache separately.
    if arch // 10 == 12:
        _sm120_bwd_cfg2 = os.environ.get("FLASH_ATTENTION_SM120_BWD_CFG")
        if _sm120_bwd_cfg2:
            _cfg2 = dict(kv.split("=") for kv in _sm120_bwd_cfg2.split(",") if kv)
            if "msplit" in _cfg2:
                pack_gqa_m_splits = int(_cfg2["msplit"])
    pack_gqa_all_rows_valid = (
        arch // 10 == 12
        and pack_gqa
        and cu_seqlens_q is None
        and cu_seqlens_k is None
        and seqused_q is None
        and seqused_k is None
        and (seqlen_q * qhead_per_kvhead) % m_block_size == 0
    )
    sm120_skip_full_causal_mask_base = (
        arch // 10 == 12
        and q.dtype == torch.bfloat16
        and causal
        and not local
        and head_dim == 256
        and head_dim_v == 256
        and qhead_per_kvhead in (2, 4, 6, 8)
        and seqlen_q == seqlen_k
        and seqlen_q % m_block_size == 0
        and seqlen_k % n_block_size == 0
        and m_block_size == 64
        and n_block_size == 64
        and softcap == 0.0
        and score_mod is None
        and score_mod_bwd is None
        and mask_mod is None
        and block_sparse_tensors is None
        and cu_seqlens_q is None
        and cu_seqlens_k is None
        and seqused_q is None
        and seqused_k is None
    )
    sm120_skip_full_causal_mask_default = sm120_skip_full_causal_mask_base and (
        (
            qhead_per_kvhead == 4
            and num_head == 8
            and num_head_kv == 2
            and batch_size in (1, 2)
            and seqlen_q == 1024
        )
        or (
            qhead_per_kvhead == 4
            and num_head == 16
            and num_head_kv == 4
            and batch_size == 2
            and seqlen_q == 1024
        )
        or (
            qhead_per_kvhead == 4
            and num_head == 32
            and num_head_kv == 8
            and batch_size <= 8
            and seqlen_q == 1024
        )
        or (
            qhead_per_kvhead == 6
            and num_head == 24
            and num_head_kv == 4
            and batch_size == 2
            and seqlen_q == 1024
        )
        or (
            qhead_per_kvhead == 2
            and num_head == 32
            and num_head_kv == 16
            and batch_size == 2
            and seqlen_q == 1024
        )
        or (
            # Packed split16 qpkv4 S2048 row: isolated on/off A/B on RTX PRO
            # 6000 showed +6.2% median (all 12 rounds positive, min +1.9%).
            qhead_per_kvhead == 4
            and num_head == 16
            and num_head_kv == 4
            and batch_size == 2
            and seqlen_q == 2048
        )
    )
    # Structural eligibility for the masked/unmasked m-loop split: any dense
    # equal-length causal row whose seqlens tile evenly (mask_fn=None also
    # skips the seqlen bounds mask, so ragged tails are excluded). The env
    # override allows forcing this beyond (=on) or below (=off) the
    # row-validated default gate for profiling and A/B validation.
    sm120_skip_full_causal_mask_struct = (
        arch // 10 == 12
        and causal
        and not local
        and seqlen_q == seqlen_k
        and seqlen_q % m_block_size == 0
        and seqlen_k % n_block_size == 0
        and softcap == 0.0
        and score_mod is None
        and score_mod_bwd is None
        and mask_mod is None
        and block_sparse_tensors is None
        and cu_seqlens_q is None
        and cu_seqlens_k is None
        and seqused_q is None
        and seqused_k is None
    )
    _maskskip_env = os.environ.get("FLASH_ATTENTION_SM120_BWD_SKIP_FULL_CAUSAL_MASK")
    if _maskskip_env is not None and _maskskip_env.lower() in ("0", "off", "false"):
        sm120_skip_full_causal_mask = False
    elif _maskskip_env is not None and _maskskip_env.lower() in ("1", "on", "true"):
        sm120_skip_full_causal_mask = sm120_skip_full_causal_mask_struct
    else:
        sm120_skip_full_causal_mask = sm120_skip_full_causal_mask_default

    if softcap != 0.0:
        assert score_mod is None and score_mod_bwd is None, (
            "softcap and score_mod/score_mod_bwd cannot be used together"
        )
        score_mod = utils.create_softcap_scoremod(softcap)
        score_mod_bwd = utils.create_softcap_scoremod_bwd(softcap)
    if score_mod is not None:
        assert score_mod_bwd is not None, "score_mod_bwd is required when score_mod is provided"
        assert cu_seqlens_q is None and cu_seqlens_k is None, (
            "varlen + score_mod not supported in bwd yet"
        )
        if arch // 10 == 8:
            raise NotImplementedError("Custom user-provided score_mod is not supported on SM8x architectures.")

    device = q.device
    out_torch_dtype = q.dtype

    if dq is None:
        dq = torch.empty_like(q)
    else:
        _validate_tensor(dq, "dq", q.shape, out_torch_dtype, device)

    if dk is None:
        dk = torch.empty_like(k)
    else:
        _validate_tensor(dk, "dk", k.shape, out_torch_dtype, device)

    if dv is None:
        dv = torch.empty_like(v)
    else:
        _validate_tensor(dv, "dv", v.shape, out_torch_dtype, device)

    head_dim_rounded = (head_dim + 32 - 1) // 32 * 32

    if cu_seqlens_q is None:
        dq_accum = (
            None
            if use_dedicated_hd256_kernel
            else torch.empty(
                batch_size,
                num_head,
                seqlen_q_rounded * head_dim_rounded,
                dtype=torch.float32,
                device=device,
            )
        )
        dpsum = torch.empty(
            batch_size, num_head, seqlen_q_rounded, dtype=torch.float32, device=device
        )
        lse_log2 = torch.empty(
            batch_size, num_head, seqlen_q_rounded, dtype=torch.float32, device=device
        )
    else:
        total_q_rounded_padded = (
            (total_q + cu_seqlens_q.shape[0] * m_block_size - 1) // m_block_size * m_block_size
        )
        dq_accum = (
            None
            if use_dedicated_hd256_kernel
            else torch.empty(
                num_head, total_q_rounded_padded * head_dim_rounded, dtype=torch.float32, device=device
            )
        )
        dpsum = torch.empty(num_head, total_q_rounded_padded, dtype=torch.float32, device=device)
        lse_log2 = torch.empty(num_head, total_q_rounded_padded, dtype=torch.float32, device=device)

    # GQA (qhead_per_kvhead > 1) needs dK/dV accum+postprocess since multiple Q heads
    # accumulate into the same dK/dV. SM90 varlen_k with qhead_per_kvhead==1 now uses
    # ragged TMA tensors for direct store, so no longer needs accum+postprocess.
    # hd=256 2CTA backward has its own internal postprocess for dK/dV.
    dKV_postprocess = qhead_per_kvhead > 1 and not use_dedicated_hd256_kernel
    if dKV_postprocess:
        head_dim_v_rounded = (head_dim_v + 32 - 1) // 32 * 32
        dkv_accum_needs_zero = not (
            arch // 10 == 12
            and pack_gqa
            and cu_seqlens_k is None
            and pack_gqa_m_splits == 1
        )
        dkv_accum_factory = torch.zeros if dkv_accum_needs_zero else torch.empty
        if cu_seqlens_k is None:
            if (
                arch // 10 == 12
                and pack_gqa
                and pack_gqa_m_splits > 1
            ):
                dk_accum_numel = batch_size * num_head_kv * seqlen_k_rounded * head_dim_rounded
                dv_accum_numel = batch_size * num_head_kv * seqlen_k_rounded * head_dim_v_rounded
                assert dk_accum_numel % 4 == 0 and dv_accum_numel % 4 == 0
                dkv_accum = torch.zeros(
                    dk_accum_numel + dv_accum_numel, dtype=torch.float32, device=device
                )
                dk_accum = dkv_accum[:dk_accum_numel].view(
                    batch_size, num_head_kv, seqlen_k_rounded * head_dim_rounded
                )
                dv_accum = dkv_accum[dk_accum_numel:].view(
                    batch_size, num_head_kv, seqlen_k_rounded * head_dim_v_rounded
                )
            else:
                dk_accum = dkv_accum_factory(
                    batch_size,
                    num_head_kv,
                    seqlen_k_rounded * head_dim_rounded,
                    dtype=torch.float32,
                    device=device,
                )
                dv_accum = dkv_accum_factory(
                    batch_size,
                    num_head_kv,
                    seqlen_k_rounded * head_dim_v_rounded,
                    dtype=torch.float32,
                    device=device,
                )
        else:
            cluster_tile_n = cluster_size * n_block_size
            total_k_rounded_padded = (
                (total_k + cu_seqlens_k.shape[0] * cluster_tile_n - 1) // cluster_tile_n * cluster_tile_n
            )
            dk_accum = torch.zeros(
                num_head_kv,
                total_k_rounded_padded * head_dim_rounded,
                dtype=torch.float32,
                device=device,
            )
            dv_accum = torch.zeros(
                num_head_kv,
                total_k_rounded_padded * head_dim_v_rounded,
                dtype=torch.float32,
                device=device,
            )

    dtype = torch2cute_dtype_map[q.dtype]
    current_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    if deterministic:
        dQ_semaphore = torch.zeros(batch_size, num_head, seqlen_q_rounded // m_block_size, cluster_size, dtype=torch.int32, device=device)
    else:
        dQ_semaphore = None

    if deterministic and qhead_per_kvhead > 1:
        dK_semaphore = torch.zeros(batch_size, num_head_kv, seqlen_k_rounded // n_block_size, 2, dtype=torch.int32, device=device)
        dV_semaphore = torch.zeros(batch_size, num_head_kv, seqlen_k_rounded // n_block_size, 2, dtype=torch.int32, device=device)
    else:
        dK_semaphore = None
        dV_semaphore = None

    # Preprocess kernel: compute (o * dout).sum(dim=-1) - dLSE, lse * log2_e, and zero out dq_accum.
    # For hd=256 dedicated path, dq_accum is None so preprocess only fills dpsum/lse_log2.
    _bwd_preprocess(
        out, dout, dpsum, lse, lse_log2, dq_accum,
        cu_seqlens_q, seqused_q, dlse,
        dtype, head_dim, head_dim_v, m_block_size,
    )
    # num_threads: SM90 derives from BwdConfig.num_wg, SM120 is set to 128 above,
    # SM100/SM110 uses default from function signature (384).
    if arch // 10 not in [9, 12]:
        num_threads = 384

    # Backward kernel: compute dk, dv, dq_accum.
    score_mod_hash = utils.hash_callable(score_mod) if score_mod else False
    score_mod_bwd_hash = utils.hash_callable(score_mod_bwd) if score_mod_bwd else False
    mask_mod_hash = utils.hash_callable(mask_mod) if mask_mod else False
    num_aux_tensors = len(aux_tensors) if aux_tensors else 0
    aux_tensor_metadata = get_aux_tensor_metadata(aux_tensors) if aux_tensors is not None else None
    aux_scalar_metadata = tuple(type(s) for s in aux_scalars) if aux_scalars is not None else None
    cute_aux_tensors = None
    if aux_tensors is not None:
        cute_aux_tensors = [to_cute_aux_tensor(buf) for buf in aux_tensors]

    block_sparse_broadcast_pattern = None
    normalized_block_sparse_tensors = None
    if use_block_sparsity:
        (
            normalized_block_sparse_tensors,
            block_sparse_broadcast_pattern,
        ) = normalize_block_sparse_config_bwd(
            block_sparse_tensors,
            batch_size=batch_size,
            num_head=num_head,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            block_size=(m_block_size, n_block_size),
            q_subtile_factor=q_subtile_factor,
            kv_subtile_factor=kv_subtile_factor,
        )
        if deterministic:
            if normalized_block_sparse_tensors.dq_write_order is None:
                raise ValueError(
                    "deterministic block-sparse backward requires dq_write_order in block_sparse_tensors"
                )
            if (
                normalized_block_sparse_tensors.full_block_cnt is not None
                and normalized_block_sparse_tensors.dq_write_order_full is None
            ):
                raise ValueError(
                    "deterministic block-sparse backward requires dq_write_order_full when full blocks are present"
                )
            if normalized_block_sparse_tensors.spt is None:
                raise ValueError(
                    "deterministic block-sparse backward requires block_sparse_tensors.spt "
                    "to match dq_write_order direction"
                )
    if (
        normalized_block_sparse_tensors is not None
        and normalized_block_sparse_tensors.spt is not None
    ):
        spt = normalized_block_sparse_tensors.spt and deterministic
    else:
        spt = (causal or local) and deterministic

    if arch // 10 in [8, 9, 12]:
        compile_key = (
            arch,
            dtype,
            head_dim,
            head_dim_v,
            qhead_per_kvhead,
            causal,
            window_size_left is not None,
            window_size_right is not None,
            m_block_size,
            n_block_size,
            num_threads,
            pack_gqa,
            pack_gqa_m_splits,
            pack_gqa_all_rows_valid,
            sm120_skip_full_causal_mask,
            num_stages_Q,
            num_stages_dO,
            SdP_swapAB,
            dKV_swapAB,
            dQ_swapAB,
            AtomLayoutMSdP,
            AtomLayoutNdKV,
            AtomLayoutMdQ,
            V_in_regs,
            dQ_single_wg,
            deterministic,
            cu_seqlens_q is None,
            cu_seqlens_k is None,
            seqused_q is None,
            seqused_k is None,
            score_mod_hash,
            score_mod_bwd_hash,
            mask_mod_hash,
            num_aux_tensors,
            aux_tensor_metadata,
            aux_scalar_metadata,
            use_block_sparsity,
            q_subtile_factor,
            block_sparse_broadcast_pattern,
            get_broadcast_dims(q),
            get_broadcast_dims(k),
            get_broadcast_dims(v),
            get_broadcast_dims(dout),
            # Prevent TVM stride poisoning when only one block is present.
            single_q_block,
            single_k_block,
        )
    else:
        compile_key = (
            arch,
            dtype,
            head_dim,
            head_dim_v,
            qhead_per_kvhead,
            causal,
            window_size_left is not None,
            window_size_right is not None,
            m_block_size,
            n_block_size,
            num_threads,
            pack_gqa,
            cluster_size,
            use_2cta_instrs,
            q_subtile_factor,
            kv_subtile_factor,
            deterministic,
            spt,
            score_mod_hash,
            score_mod_bwd_hash,
            mask_mod_hash,
            num_aux_tensors,
            aux_tensor_metadata,
            aux_scalar_metadata,
            use_block_sparsity,
            block_sparse_broadcast_pattern,
            cu_seqlens_q is None,
            cu_seqlens_k is None,
            seqused_q is None,
            seqused_k is None,
            get_broadcast_dims(q),
            get_broadcast_dims(k),
            get_broadcast_dims(v),
            get_broadcast_dims(dout),
            # Prevent TVM stride poisoning when only one block is present.
            single_q_block,
            single_k_block,
        )

    if compile_key not in _flash_attn_bwd.compile_cache:
        q_tensor, k_tensor, v_tensor, do_tensor, dq_tensor, dk_tensor, dv_tensor = [
            to_cute_tensor(t) for t in (q, k, v, dout, dq, dk, dv)
        ]
        lse_log2_tensor, dpsum_tensor = [to_cute_tensor(t) for t in (lse_log2, dpsum)]
        dq_accum_tensor = to_cute_tensor(dq_accum) if dq_accum is not None else None
        if dKV_postprocess:
            dk_accum_tensor, dv_accum_tensor = [
                to_cute_tensor(t) for t in (dk_accum, dv_accum)
            ]
        cu_seqlens_q_tensor, cu_seqlens_k_tensor, seqused_q_tensor, seqused_k_tensor = [
            to_cute_tensor(t, assumed_align=4) if t is not None else None
            for t in (cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k)
        ]
        dQ_semaphore_tensor, dK_semaphore_tensor, dV_semaphore_tensor = [
            utils.convert_from_dlpack_leading_static(t.detach(), leading_dim=3, alignment=4, stride_order=t.dim_order())
            if t is not None else None
            for t in (dQ_semaphore, dK_semaphore, dV_semaphore)
        ]
        if arch // 10 in [8, 12]:
            flash_bwd_obj_cls = FlashAttentionBackwardSm120 if arch // 10 == 12 else FlashAttentionBackwardSm80
            fa_bwd_obj = flash_bwd_obj_cls(
                dtype,
                head_dim,
                head_dim_v,
                qhead_per_kvhead,
                m_block_size,
                n_block_size,
                num_stages_Q,
                num_stages_dO,
                num_threads,
                pack_gqa,
                causal,
                SdP_swapAB,
                dKV_swapAB,
                dQ_swapAB,
                AtomLayoutMSdP,
                AtomLayoutNdKV,
                AtomLayoutMdQ,
                V_in_regs=V_in_regs,
                score_mod=score_mod,
                score_mod_bwd=score_mod_bwd,
                pack_gqa_m_splits=pack_gqa_m_splits,
                pack_gqa_all_rows_valid=pack_gqa_all_rows_valid,
                skip_full_causal_mask=sm120_skip_full_causal_mask,
                is_local=local,
            )
        elif arch // 10 == 9:
            fa_bwd_obj = FlashAttentionBackwardSm90(
                dtype,
                head_dim,
                head_dim_v,
                qhead_per_kvhead,
                causal,
                is_local=local,
                deterministic=deterministic,
                tile_m=m_block_size,
                tile_n=n_block_size,
                Q_stage=num_stages_Q,
                dO_stage=num_stages_dO,
                PdS_stage=num_stages_PdS,
                SdP_swapAB=SdP_swapAB,
                dKV_swapAB=dKV_swapAB,
                dQ_swapAB=dQ_swapAB,
                AtomLayoutMSdP=AtomLayoutMSdP,
                AtomLayoutNdKV=AtomLayoutNdKV,
                AtomLayoutMdQ=AtomLayoutMdQ,
                num_threads=num_threads,
                V_in_regs=V_in_regs,
                score_mod=score_mod,
                score_mod_bwd=score_mod_bwd,
                mask_mod=mask_mod,
                has_aux_tensors=aux_tensors is not None,
                q_subtile_factor=q_subtile_factor,
                dQ_single_wg=dQ_single_wg,
            )
        else:
            if use_dedicated_hd256_kernel:
                assert softcap == 0.0, "SM100 backward with head_dim=256 does not support softcap"
                assert block_sparse_tensors is None, \
                    "SM100 backward with head_dim=256 does not support block sparsity"
                assert dlse is None, \
                    "SM100 backward with head_dim=256 does not support dlse"
                assert seqused_q is None and seqused_k is None, \
                    "SM100 backward with head_dim=256 does not support seqused_q/seqused_k"

                dq_tile_mn = (128, 128)
                dkdv_tile_mn = (128, 64)
                fa_bwd_obj = BlackwellFusedMultiHeadAttentionBackward(
                    head_dim,
                    head_dim_v,
                    is_causal=causal,
                    is_local=local,
                    qhead_per_kvhead=qhead_per_kvhead,
                    is_persistent=False,
                    deterministic=deterministic,
                    cluster_size=cluster_size,
                    use_2cta_instrs=use_2cta_instrs,
                    score_mod=score_mod,
                    score_mod_bwd=score_mod_bwd,
                    mask_mod=mask_mod,
                    has_aux_tensors=aux_tensors is not None,
                    q_subtile_factor=q_subtile_factor,
                    tile_m_dq=dq_tile_mn[0],
                    tile_n_dq=dq_tile_mn[1],
                    tile_m_dkdv=dkdv_tile_mn[0],
                    tile_n_dkdv=dkdv_tile_mn[1],
                )
            else:
                fa_bwd_obj = FlashAttentionBackwardSm100(
                    head_dim,
                    head_dim_v,
                    is_causal=causal,
                    is_local=local,
                    qhead_per_kvhead=qhead_per_kvhead,
                    tile_m=m_block_size,
                    tile_n=n_block_size,
                    cluster_size=cluster_size,
                    use_2cta_instrs=use_2cta_instrs,
                    deterministic=deterministic,
                    spt=spt,
                    score_mod=score_mod,
                    score_mod_bwd=score_mod_bwd,
                    mask_mod=mask_mod,
                    has_aux_tensors=aux_tensors is not None,
                    q_subtile_factor=q_subtile_factor,
                    kv_subtile_factor=kv_subtile_factor,
                )

        # Block sparse tensors for backward use Q-direction indexing (transposed from forward).
        sparse_tensors_compile = None
        if normalized_block_sparse_tensors is not None:
            sparse_tensors_compile = to_cute_block_sparse_tensors(normalized_block_sparse_tensors)
        dq_accum_tensor = dq_tensor if use_dedicated_hd256_kernel else dq_accum_tensor
        window_size_left_cute = _to_cute_int32_or_none(window_size_left)
        window_size_right_cute = _to_cute_int32_or_none(window_size_right)

        # TODO: check @can_implement
        _flash_attn_bwd.compile_cache[compile_key] = cute.compile(
            fa_bwd_obj,
            q_tensor,
            k_tensor,
            v_tensor,
            do_tensor,
            lse_log2_tensor,
            dpsum_tensor,
            dq_accum_tensor,
            dk_tensor if not dKV_postprocess else dk_accum_tensor,
            dv_tensor if not dKV_postprocess else dv_accum_tensor,
            softmax_scale,
            cu_seqlens_q_tensor,
            cu_seqlens_k_tensor,
            seqused_q_tensor,
            seqused_k_tensor,
            window_size_left_cute,
            window_size_right_cute,
            dQ_semaphore_tensor,
            dK_semaphore_tensor,
            dV_semaphore_tensor,
            AuxData(cute_aux_tensors, aux_scalars),
            sparse_tensors_compile,
            current_stream,
            options="--enable-tvm-ffi",
        )
    if not is_fake_mode():
        window_size_left_cute = _to_cute_int32_or_none(window_size_left)
        window_size_right_cute = _to_cute_int32_or_none(window_size_right)
        dq_accum = dq if use_dedicated_hd256_kernel else dq_accum
        _flash_attn_bwd.compile_cache[compile_key](
            q.detach(),
            k.detach(),
            v.detach(),
            dout,
            lse_log2,
            dpsum,
            dq_accum,
            dk if not dKV_postprocess else dk_accum,
            dv if not dKV_postprocess else dv_accum,
            softmax_scale,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            window_size_left_cute,
            window_size_right_cute,
            dQ_semaphore,
            dK_semaphore,
            dV_semaphore,
            AuxData(aux_tensors, aux_scalars),
            (
                normalized_block_sparse_tensors.mask_block_cnt,
                normalized_block_sparse_tensors.mask_block_idx,
                normalized_block_sparse_tensors.full_block_cnt,
                normalized_block_sparse_tensors.full_block_idx,
                normalized_block_sparse_tensors.cu_total_m_blocks,
                normalized_block_sparse_tensors.cu_block_idx_offsets,
                normalized_block_sparse_tensors.dq_write_order,
                normalized_block_sparse_tensors.dq_write_order_full,
            )
            if normalized_block_sparse_tensors is not None
            else None,
        )
    # Postprocess: convert dq_accum from float32 to dq in bf16/fp16
    # hd=256 2CTA backward has its own internal postprocess, skip here.
    if not use_dedicated_hd256_kernel:
        if arch // 10 == 9:
            # dQ postprocess: match main kernel's MMA WG count, unless dQ_single_wg
            num_threads_post_dQ = 128 if dQ_single_wg else cfg.num_wg * 128
            num_threads_post_dKV = cfg.num_wg * 128
        elif arch // 10 == 12:
            # SM120: postprocess MUST match the main kernel's num_threads
            # because the dq_accum/dk_accum byte buffers are written by the main
            # kernel using a thread-major partition (gmem_tiled_copy_dQaccum)
            # whose stride is num_threads. The postprocess reader uses the same
            # convention; otherwise the per-thread element->address mapping
            # diverges between writer and reader.
            num_threads_post_dQ = num_threads
            num_threads_post_dKV = num_threads
        else:
            num_threads_post_dQ = 128
            num_threads_post_dKV = 128

        _bwd_postprocess_convert(
            dq_accum, dq, softmax_scale,
            cu_seqlens_q, seqused_q,
            arch, dtype, head_dim, m_block_size, num_threads_post_dQ,
            AtomLayoutMdQ, dQ_swapAB,
            use_2cta_instrs=use_2cta_instrs, cluster_size=1,
            pack_gqa=(arch // 10 == 12 and pack_gqa and cu_seqlens_q is None),
            qhead_per_kvhead=qhead_per_kvhead,
        )

        if dKV_postprocess:
            if _sm120_use_fused_dkv_postprocess(
                arch=arch,
                dtype=dtype,
                dkv_postprocess=dKV_postprocess,
                pack_gqa=pack_gqa,
                pack_gqa_m_splits=pack_gqa_m_splits,
                qhead_per_kvhead=qhead_per_kvhead,
                causal=causal,
                local=local,
                seqlen_q=seqlen_q,
                seqlen_k=seqlen_k,
                cu_seqlens_k=cu_seqlens_k,
                seqused_k=seqused_k,
                head_dim=head_dim,
                head_dim_v=head_dim_v,
                dKV_swapAB=dKV_swapAB,
            ):
                _bwd_postprocess_dkv_sm120(
                    dk_accum, dv_accum, dk, dv, softmax_scale,
                    dtype, head_dim, n_block_size, num_threads_post_dKV, AtomLayoutNdKV,
                )
            else:
                # Postprocess: convert dk_accum from float32 to dk in bf16/fp16
                _bwd_postprocess_convert(
                    dk_accum, dk, softmax_scale,
                    cu_seqlens_k, seqused_k,
                    arch, dtype, head_dim, n_block_size, num_threads_post_dKV,
                    AtomLayoutNdKV, dKV_swapAB,
                    cluster_size=cluster_size,
                )
                # Postprocess: convert dv_accum from float32 to dv in bf16/fp16
                _bwd_postprocess_convert(
                    dv_accum, dv, 1.0,
                    cu_seqlens_k, seqused_k,
                    arch, dtype, head_dim_v, n_block_size, num_threads_post_dKV,
                    AtomLayoutNdKV, dKV_swapAB,
                    cluster_size=cluster_size,
                )

    return dq, dk, dv


_flash_attn_bwd.compile_cache = get_jit_cache("bwd")


def _flash_attn_bwd_sparse_mla(
    q: Optional[torch.Tensor],
    k: Optional[torch.Tensor],
    v: torch.Tensor,
    qv: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    p: torch.Tensor,
    row_max: torch.Tensor,
    gather_kv_indices: torch.Tensor,
    learnable_sink: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    m_block_size: int = 128,
    n_block_size: int = 64,
    num_threads: int = 256,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    min_seqlen_k: Optional[int] = None,
    deterministic: bool = False,
    dq: Optional[torch.Tensor] = None,
    dk: Optional[torch.Tensor] = None,
    dv: Optional[torch.Tensor] = None,
    dqv: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    arch = _get_device_arch()
    assert arch // 10 in [10, 11], "Unsupported compute capability. Supported: 10.x, 11.x"
    assert gather_kv_indices is not None, "require gather kv indices for backward"

    q_shape = q.shape if q is not None else qv.shape
    nheads, head_dim = q_shape[-2:]
    nheads_kv, head_dim_v = v.shape[-2:]
    qhead_per_kvhead = nheads // nheads_kv
    gather_kv_length = gather_kv_indices.shape[-1]
    assert nheads_kv == 1 and qhead_per_kvhead == 128, f"sparse MLA bwd: only MQA 128 supported for now"
    assert gather_kv_length % 128 == 0, f"sparse MLA bwd: {gather_kv_length=} must be divisible by 128"
    assert deterministic is False, "sparse MLA bwd: deterministic mode not yet supported"
    assert learnable_sink is None, "sparse MLA bwd: learnable sink not yet supported"
    assert seqused_q is None and seqused_k is None, "sparse MLA bwd: seqused_q,k not yet supported"

    if softmax_scale is None:
        softmax_scale = (
            1.0 / math.sqrt(head_dim) if qv is None or q is None
            else 1.0 / math.sqrt(head_dim + head_dim_v)
        )

    q, k, v, qv, out, dout, lse, p, row_max = [
        maybe_contiguous(t)
        for t in (q, k, v, qv, out, dout, lse, p, row_max)
    ]
    gather_kv_indices, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, learnable_sink = [
        maybe_contiguous(t)
        for t in (gather_kv_indices, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, learnable_sink)
    ]
    device = v.device

    varlen_q = cu_seqlens_q is not None or seqused_q is not None
    if cu_seqlens_q is None:
        batch_size, seqlen_q = q_shape[:2]
        total_q = batch_size * seqlen_q
        p_shape = (batch_size, seqlen_q, nheads, gather_kv_length)
    else:
        batch_size = cu_seqlens_q.shape[0] - 1
        total_q = q_shape[0]
        seqlen_q = max_seqlen_q if max_seqlen_q is not None else total_q
        p_shape = (total_q, nheads, gather_kv_length)

    varlen_k = cu_seqlens_k is not None or seqused_k is not None
    if cu_seqlens_k is None:
        batch_size, seqlen_k = v.shape[:2]
        total_k = batch_size * seqlen_k
    else:
        batch_size = cu_seqlens_k.shape[0] - 1
        total_k = v.shape[0]
        seqlen_k = max_seqlen_k if max_seqlen_k is not None else total_k
    if not varlen_k:
        min_seqlen_k = seqlen_k 

    assert varlen_q == varlen_k, "sparse MLA bwd: either q and k are both varlen or not"

    # always use kv bitmask by default (handles -1 sentinel)
    disable_sparse_kv_bitmask = False
    # if min_seqlen_k is None or causal:
    #     disable_sparse_kv_bitmask = False
    # else:
    #     disable_sparse_kv_bitmask = min_seqlen_k >= gather_kv_length

    prealloc_dq = dq is not None
    prealloc_dk = dk is not None
    prealloc_dqv = dqv is not None
    prealloc_dv = dv is not None
    dq = dk = None
    if not prealloc_dq and q is not None:
        dq = torch.empty_like(q)
    if not prealloc_dk and k is not None:
        dk = torch.zeros_like(k, dtype=torch.float32)
    if not prealloc_dv:
        dv = torch.zeros_like(v, dtype=torch.float32)
    if not prealloc_dqv:
        dqv = torch.empty_like(qv)
    ds = torch.empty_like(p)

    device = v.device
    dtype = v.dtype
    if q is not None:
        _validate_tensor(dq, "dq", q.shape, dtype, device)
    if k is not None:
        _validate_tensor(dk, "dk", k.shape, torch.float32, device)
    _validate_tensor(dv, "dv", v.shape, torch.float32, device)
    _validate_tensor(dqv, "dqv", qv.shape, dtype, device)
    _validate_tensor(p, "p", p_shape, dtype, device)

    if cu_seqlens_q is None:
        dpsum = torch.empty(batch_size, seqlen_q, nheads, dtype=torch.float32, device=device)
    else:
        dpsum = torch.empty(total_q, nheads, dtype=torch.float32, device=device)
    scale_p = torch.empty_like(row_max)

    dtype = torch2cute_dtype_map[dout.dtype]
    current_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    # Preprocess kernel: compute (o * dout).sum(dim=-1), scale_p.
    _bwd_preprocess(
        out, dout, dpsum, lse, None, None,
        cu_seqlens_q, seqused_q, None,
        dtype, head_dim, head_dim_v, m_block_size,
        row_max=row_max,
        scale_p=scale_p,
        use_padded_offsets=False,
        nheads_major=True,
        pack_gqa=True,
        qhead_per_kvhead=qhead_per_kvhead,
        nheads_kv=nheads_kv,
        softmax_scale=softmax_scale,
    )

    compile_key = (
        dtype,
        head_dim,
        head_dim_v,
        qhead_per_kvhead,
        causal,
        cu_seqlens_q is None,
        cu_seqlens_k is None,
        seqused_q is None,
        seqused_k is None,
        q is not None,
        gather_kv_length,
        learnable_sink is not None,
        disable_sparse_kv_bitmask,
    )

    if compile_key not in _flash_attn_bwd_sparse_mla.compile_cache:
        (
            cu_seqlens_q_tensor,
            cu_seqlens_k_tensor,
            seqused_q_tensor,
            seqused_k_tensor,
            learnable_sink_tensor,
        ) = [
            to_cute_tensor(t, assumed_align=4, leading_dim=0)
            for t in (cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, learnable_sink)
        ]
        (
            v_tensor,
            qv_tensor,
            do_tensor,
            p_tensor,
            scale_p_tensor,
            dpsum_tensor,
            ds_tensor,
            dv_tensor,
            gather_kv_indices_tensor,
         ) = [
            to_cute_tensor(t) for t in (v, qv, dout, p, scale_p, dpsum, ds, dv, gather_kv_indices)
        ]

        fa_bwd_obj = FlashAttentionSparseMLABackwardSm100(
            is_causal=causal,
            topk_length=gather_kv_length,
            qhead_per_kvhead=qhead_per_kvhead,
            nheads_kv=nheads_kv,
            has_seqused_q=seqused_q is not None,
            disable_bitmask=disable_sparse_kv_bitmask,
        )
        fa_bwd_kernel = cute.compile(
            fa_bwd_obj,
            do_tensor,
            v_tensor,
            qv_tensor,
            p_tensor,
            dv_tensor,
            ds_tensor,
            gather_kv_indices_tensor,
            softmax_scale,
            scale_p_tensor,
            dpsum_tensor,
            cu_seqlens_q_tensor,
            cu_seqlens_k_tensor,
            seqused_q_tensor,
            seqused_k_tensor,
            current_stream,
            options="--enable-tvm-ffi",
        )
        _flash_attn_bwd_sparse_mla.compile_cache[compile_key] = fa_bwd_kernel

    if not is_fake_mode():
        _flash_attn_bwd_sparse_mla.compile_cache[compile_key](
            dout,
            v,
            qv,
            p,
            dv,
            ds,
            gather_kv_indices,
            softmax_scale,
            scale_p,
            dpsum,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
        )

    v = v.squeeze(-2)
    if k is not None:
        k = k.squeeze(-2)
    
    _sparse_mla_dq_dqv(
        ds, k, v, dq, dqv, gather_kv_indices, cu_seqlens_q, cu_seqlens_k,
    )

    if k is not None:
        dk = dk.squeeze(-2)
        _sparse_mla_dk(ds, gather_kv_indices, q, dk, cu_seqlens_q, cu_seqlens_k)
        dk = dk.unsqueeze(-2)
    
    # return dk, dv in float32: all-reduce across sequence-parallel ranks must happen
    # before downcasting to avoid rounding error during inter-rank grad accumulation
    return dq, dk, dv, dqv

_flash_attn_bwd_sparse_mla.compile_cache = get_jit_cache("bwd_dsa")


def _compile_sparse_mla_dq_dqv(
    dtype, nheads, head_dim, head_dim_v, top_k, varlen_q, varlen_k, compute_dq,
):
    sym = cute.sym_int 
    b, b_plus_1, seqlen_q, seqlen_k = sym(), sym(), sym(), sym()
    total_q, total_k = sym(), sym()
    b_seqlenq = (b, seqlen_q) if not varlen_q else (total_q,)
    b_seqlenk = (b, seqlen_k) if not varlen_k else (total_k,)
    
    div = 128 // dtype.width  # 8 for fp16/bf16
    
    mdS = fake_tensor(dtype, (*b_seqlenq, nheads, top_k), divisibility=div)
    mK = fake_tensor(dtype, (*b_seqlenk, head_dim), divisibility=div)
    mV = fake_tensor(dtype, (*b_seqlenk, head_dim_v), divisibility=div)
    mdQ = fake_tensor(dtype, (*b_seqlenq, nheads, head_dim), divisibility=div)
    mdQv = fake_tensor(dtype, (*b_seqlenq, nheads, head_dim_v), divisibility=div)
    mIdxTopK = fake_tensor(Int32, (*b_seqlenq, top_k), divisibility=div)
    
    mCuSeqlensQ = fake_tensor(Int32, (b_plus_1,), divisibility=1) if varlen_q else None 
    mCuSeqlensK = fake_tensor(Int32, (b_plus_1,), divisibility=1) if varlen_k else None 
    
    dq_dqv_gemm = dQdQvGemmKernel(
        acc_dtype=Float32,
        nheads=nheads,
        head_dim_k=head_dim,
        head_dim_v=head_dim_v,
        top_k=top_k,
    )
    
    return cute.compile(
        dq_dqv_gemm,
        mdS,
        mK if compute_dq else None,
        mV,
        mdQ if compute_dq else None,
        mdQv,
        mIdxTopK,
        mCuSeqlensQ,
        mCuSeqlensK,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _sparse_mla_dq_dqv(
    ds, k, v, dq, dqv, gather_kv_indices, cu_seqlens_q, cu_seqlens_k,
):
    """Compute dQ = dS @ K and dQv = dS @ V"""
    *_, nheads, gather_kv_length = ds.shape
    
    head_dim_v = v.shape[-1]
    head_dim = k.shape[-1] if k is not None else 0
    
    dtype = ds.dtype
    dtype_cute = torch2cute_dtype_map[dtype]
    
    varlen_q = cu_seqlens_q is not None
    varlen_k = cu_seqlens_k is not None
    
    compile_key = (
        dtype_cute, nheads, head_dim, head_dim_v, gather_kv_length, varlen_q, varlen_k, k is not None,
    )
    if compile_key not in _sparse_mla_dq_dqv.compile_cache:
        _sparse_mla_dq_dqv.compile_cache[compile_key] = _compile_sparse_mla_dq_dqv(
            *compile_key
        )
    if not is_fake_mode():
        _sparse_mla_dq_dqv.compile_cache[compile_key](
            ds, k, v, dq, dqv, gather_kv_indices, cu_seqlens_q, cu_seqlens_k
        )

_sparse_mla_dq_dqv.compile_cache = get_jit_cache("dq_dqv_gemm")


def _compile_sparse_mla_dk(
    dtype,
    dtype_acc,
    nheads: int,
    head_dim: int,
    topk: int,
    varlen: bool,
):
    kernel = dKGemmKernel(
        topk,
        nheads,
        head_dim,
        varlen,
    )
    # Check if configuration can be implemented
    kernel.check_can_implement()

    div = 128 // dtype.width

    sym = cute.sym_int
    batch_fake = sym()
    batchp1_fake = sym()
    seqlen_q_fake = sym()
    seqlen_k_fake = sym()
    total_q_fake = (batch_fake, seqlen_q_fake) if not varlen else (sym(),)
    total_k_fake = (batch_fake, seqlen_k_fake) if not varlen else (sym(),)

    mdS = fake_tensor(dtype, (*total_q_fake, nheads, topk), divisibility=div)
    mI = fake_tensor(Int32, (*total_q_fake, topk), divisibility=div)
    mQ = fake_tensor(dtype, (*total_q_fake, nheads, head_dim), divisibility=div)
    mdK = fake_tensor(dtype_acc, (*total_k_fake, head_dim), divisibility=div)
    mCuSeqlensQ = fake_tensor(Int32, (batchp1_fake,), divisibility=1) if varlen else None
    mCuSeqlensK = fake_tensor(Int32, (batchp1_fake,), divisibility=1) if varlen else None
    
    return cute.compile(
        kernel,
        mdS,
        mI,
        mQ,
        mdK,
        mCuSeqlensQ,
        mCuSeqlensK,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _sparse_mla_dk(
    dS: torch.Tensor,
    index_topk: torch.Tensor,
    q: torch.Tensor,
    dk: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
):
    """Compute dKaccum = scatter(dS'^T @ Q, I).

    Args:
      dS:          (*total_q, heads, topk), bf16
      index_topk:  (*total_q, topk), int32
      Q:           (*total_q, heads, dim), bf16
      dK:          (*total_q, dim), fp32
      cuSeqlensQ:  (batch + 1,), int32, omit for non-varlen
      cuSeqlensK:  (batch + 1,), int32, omit for non-varlen

    Accumulates in place on top of dK.

    For varlen, total_q and total_k are 1-dimensional, and the seqlen indices per batch are
    determined using the cuSeqlensQ and cuSeqlensK tensors.
    For non-varlen, total_q and total_k are (batch, seqlen_q) and (batch, seqlen_k).
    """
    dtype = dS.dtype
    dtype_cute = torch2cute_dtype_map[dtype]
    dtype_acc = dk.dtype
    dtype_acc_cute = torch2cute_dtype_map[dtype_acc]

    varlen = cu_seqlens_q is not None
    nheads, topk = dS.shape[-2], dS.shape[-1]
    head_dim = q.shape[-1] if q is not None else 0

    compile_key = (
        dtype_cute, dtype_acc_cute, nheads, head_dim, topk, varlen,
    )

    if compile_key not in _sparse_mla_dk.compile_cache:
        _sparse_mla_dk.compile_cache[compile_key] = _compile_sparse_mla_dk(*compile_key)

    if not is_fake_mode():
        _sparse_mla_dk.compile_cache[compile_key](dS, index_topk, q, dk, cu_seqlens_q, cu_seqlens_k)
    
_sparse_mla_dk.compile_cache = get_jit_cache("dk_gemm")


class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        qv: Optional[torch.Tensor] = None,
        gather_kv_indices: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        window_size: Tuple[Optional[int], Optional[int]] = (None, None),
        learnable_sink: Optional[torch.Tensor] = None,
        softcap: float = 0.0,
        num_splits: int = 1,
        pack_gqa: Optional[bool] = None,
        deterministic: bool = False,
        score_mod: Optional[Callable] = None,
        score_mod_bwd: Optional[Callable] = None,
        mask_mod: Optional[Callable] = None,
        aux_tensors: Optional[list] = None,
        aux_scalars: Optional[tuple] = None,
        block_sparse_tensors: Optional[BlockSparseTensorsTorch] = None,
        block_sparse_tensors_bwd: Optional[BlockSparseTensorsTorch] = None,
        return_lse: bool = False,
    ):
        aux_scalars = tuple(aux_scalars) if aux_scalars else None
        shared_kv = k is v
        if shared_kv and v.shape[-1] == 512:
            # specialize MLA attention formula
            # O = softmax(Q @ K.T + Qv @ V.T) @ V
            # by setting q, k to None
            qv = q if qv is None else qv
            q = k = None
        out, lse, p, row_max = _flash_attn_fwd(
            q,
            k,
            v,
            qv=qv,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            learnable_sink=learnable_sink,
            softcap=softcap,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
            score_mod=score_mod,
            mask_mod=mask_mod,
            aux_tensors=aux_tensors,
            aux_scalars=aux_scalars,
            block_sparse_tensors=block_sparse_tensors,
            return_lse=return_lse,
            gather_kv_indices=gather_kv_indices,
        )
        ctx.save_for_backward(q, k, v, qv, out, lse, p, row_max, gather_kv_indices, *(aux_tensors or ()))
        ctx.shared_kv = shared_kv
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.deterministic = deterministic
        ctx.return_lse = return_lse
        ctx.pack_gqa = pack_gqa
        ctx.score_mod = score_mod
        ctx.score_mod_bwd = score_mod_bwd
        ctx.mask_mod = mask_mod
        ctx.aux_scalars = aux_scalars
        ctx.block_sparse_tensors_bwd = block_sparse_tensors_bwd
        ctx.set_materialize_grads(False)
        return out, lse

    @staticmethod
    def backward(ctx, dout, dlse):
        q, k, v, qv, out, lse, p, row_max, gather_kv_indices, *aux = ctx.saved_tensors
        aux_tensors = aux if aux else None
        if not ctx.return_lse:
            dlse = None
        if dout is None:
            dout = torch.zeros_like(out)
        if qv is not None:
            dq, dk, dv, dqv = _flash_attn_bwd_sparse_mla(
                q,
                k,
                v,
                qv,
                out,
                dout,
                lse,
                p,
                row_max,
                gather_kv_indices,
                softmax_scale=ctx.softmax_scale,
                causal=ctx.causal,
            )
            if ctx.shared_kv:
                return dqv, dv, None, None, *((None,) * 30)
            else:
                return dq, dk, dv, dqv, *((None,) * 30)
        else:
            dq, dk, dv = _flash_attn_bwd(
                q,
                k,
                v,
                out,
                dout,
                lse,
                ctx.softmax_scale,
                ctx.causal,
                ctx.softcap,
                window_size_left=ctx.window_size[0],
                window_size_right=ctx.window_size[1],
                deterministic=ctx.deterministic,
                pack_gqa=ctx.pack_gqa,
                score_mod=ctx.score_mod,
                score_mod_bwd=ctx.score_mod_bwd,
                mask_mod=ctx.mask_mod,
                aux_tensors=aux_tensors,
                aux_scalars=ctx.aux_scalars,
                block_sparse_tensors=ctx.block_sparse_tensors_bwd,
                dlse=dlse,
            )
            return dq, dk, dv, *((None,) * 30)  # Extra Nones is fine


class FlashAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: Optional[torch.Tensor],
        k: Optional[torch.Tensor],
        v: torch.Tensor,
        qv: Optional[torch.Tensor] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        seqused_q: Optional[torch.Tensor] = None,
        seqused_k: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_k: Optional[int] = None,
        min_seqlen_k: Optional[int] = None,
        gather_kv_indices: Optional[torch.Tensor] = None,
        page_table: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        window_size: Tuple[Optional[int], Optional[int]] = (None, None),
        learnable_sink: Optional[torch.Tensor] = None,
        softcap: float = 0.0,
        num_splits: int = 1,
        pack_gqa: Optional[bool] = None,
        deterministic: bool = False,
        score_mod: Optional[Callable] = None,
        score_mod_bwd: Optional[Callable] = None,
        mask_mod: Optional[Callable] = None,
        block_sparse_tensors: Optional[list] = None,
        aux_tensors: Optional[list] = None,
        aux_scalars: Optional[tuple] = None,
        return_lse: bool = False,
    ):
        aux_scalars = tuple(aux_scalars) if aux_scalars else None
        shared_kv = k is v
        if shared_kv and v.shape[-1] == 512:
            # specialize MLA attention formula
            # O = softmax(Q @ K.T + Qv @ V.T) @ V
            # by setting q, k to None
            qv = q if qv is None else qv
            q = k = None
        out, lse, p, row_max = _flash_attn_fwd(
            q,
            k,
            v,
            qv=qv,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            min_seqlen_k=min_seqlen_k,
            page_table=page_table,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            learnable_sink=learnable_sink,
            softcap=softcap,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
            score_mod=score_mod,
            mask_mod=mask_mod,
            block_sparse_tensors=block_sparse_tensors,
            aux_tensors=aux_tensors,
            aux_scalars=aux_scalars,
            return_lse=return_lse,
            gather_kv_indices=gather_kv_indices,
        )
        ctx.save_for_backward(
            q,
            k,
            v,
            qv,
            out,
            lse,
            p,
            row_max,
            gather_kv_indices,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            *(aux_tensors or ()),
        )
        ctx.shared_kv = shared_kv
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.deterministic = deterministic
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.min_seqlen_k = min_seqlen_k
        ctx.return_lse = return_lse
        ctx.pack_gqa = pack_gqa
        ctx.score_mod = score_mod
        ctx.score_mod_bwd = score_mod_bwd
        ctx.mask_mod = mask_mod
        ctx.aux_scalars = aux_scalars
        ctx.set_materialize_grads(False)
        return out, lse

    @staticmethod
    def backward(ctx, dout, dlse):
        q, k, v, qv, out, lse, p, row_max, gather_kv_indices, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, *aux = ctx.saved_tensors
        aux_tensors = aux if aux else None
        if not ctx.return_lse:
            dlse = None
        if dout is None:
            dout = torch.zeros_like(out)
        if qv is not None:
            dq, dk, dv, dqv = _flash_attn_bwd_sparse_mla(
                q,
                k,
                v,
                qv,
                out,
                dout,
                lse,
                p,
                row_max,
                gather_kv_indices,
                softmax_scale=ctx.softmax_scale,
                causal=ctx.causal,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                seqused_q=seqused_q,
                seqused_k=seqused_k,
                max_seqlen_q=ctx.max_seqlen_q,
                max_seqlen_k=ctx.max_seqlen_k,
                min_seqlen_k=ctx.min_seqlen_k,
            )
            if ctx.shared_kv:
                return dqv, dv, None, None, *((None,) * 31)
            else:
                return dq, dk, dv, dqv, *((None,) * 31)
        else:
            dq, dk, dv = _flash_attn_bwd(
                q,
                k,
                v,
                out,
                dout,
                lse,
                ctx.softmax_scale,
                ctx.causal,
                ctx.softcap,
                window_size_left=ctx.window_size[0],
                window_size_right=ctx.window_size[1],
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                seqused_q=seqused_q,
                seqused_k=seqused_k,
                max_seqlen_q=ctx.max_seqlen_q,
                max_seqlen_k=ctx.max_seqlen_k,
                deterministic=ctx.deterministic,
                pack_gqa=ctx.pack_gqa,
                score_mod=ctx.score_mod,
                score_mod_bwd=ctx.score_mod_bwd,
                aux_tensors=aux_tensors,
                aux_scalars=ctx.aux_scalars,
                mask_mod=ctx.mask_mod,
                dlse=dlse,
            )
            return dq, dk, dv, *((None,) * 31)


@_opaque_to_dynamo
def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qv: Optional[torch.Tensor] = None,
    gather_kv_indices: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    learnable_sink: Optional[torch.Tensor] = None,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    deterministic: bool = False,
    score_mod: Optional[Callable] = None,
    score_mod_bwd: Optional[Callable] = None,
    mask_mod: Optional[Callable] = None,
    aux_tensors: Optional[list] = None,
    aux_scalars: Optional[tuple] = None,
    block_sparse_tensors: Optional[BlockSparseTensorsTorch] = None,
    block_sparse_tensors_bwd: Optional[BlockSparseTensorsTorch] = None,
    return_lse: bool = False,
):
    return FlashAttnFunc.apply(
        q,
        k,
        v,
        qv,
        gather_kv_indices,
        softmax_scale,
        causal,
        window_size,
        learnable_sink,
        softcap,
        num_splits,
        pack_gqa,
        deterministic,
        score_mod,
        score_mod_bwd,
        mask_mod,
        aux_tensors,
        aux_scalars,
        block_sparse_tensors,
        block_sparse_tensors_bwd,
        return_lse,
    )


@_opaque_to_dynamo
def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qv: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    min_seqlen_k: Optional[int] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    gather_kv_indices: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    learnable_sink: Optional[torch.Tensor] = None,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    deterministic: bool = False,
    score_mod: Optional[Callable] = None,
    score_mod_bwd: Optional[Callable] = None,
    mask_mod: Optional[Callable] = None,
    block_sparse_tensors: Optional[BlockSparseTensorsTorch] = None,
    aux_tensors: Optional[list] = None,
    aux_scalars: Optional[tuple] = None,
    return_lse: bool = False,
):
    """
    Tensor arguments:
        q:  (total_q, nheads,   hdim)   or (batch, seqlen_q, nheads,   hdim)
        k:  (total_k, nheads_k, hdim)   or (batch, seqlen_k, nheads_k, hdim)
        v:  (total_k, nheads_k, hdim_v) or (batch, seqlen_k, nheads_k, hdim_v)
        qv: (total_q, nheads,   hdim_v) or (batch, seqlen_q, nheads,   hdim_v)
        cu_seqlens_q: (batch + 1)       or seqused_q: (batch)
        cu_seqlens_k: (batch + 1)       or seqused_k: (batch)
        gather_kv_indices: (total_q, gather_kv_length) or
                           (batch, seqlen_q, gather_kv_length)
        page_table: (batch, max_num_pages_per_seq)
    
    Return:
       out: (total_q, nheads, hdim) or (batch, seqlen_q, nheads, hdim)
       lse: (nheads, total_q)       or (batch, nheads, seqlen_q) if not has_qv (standard)
            (total_q, nheads)       or (batch, seqlen_q, nheads) if has_qv

    Explanation of some optional arguments & decisions:

    qv: we write the MLA weight absorbed formula as
        O = softmax(scale * (Q @ K.T + Qv @ V.T)) @ V
        where Q = q_pe, Qv = q_nope, K = pe_cache, V = kv_cache.

    lse return shape: with Qv, MQA with nheads at least divisible by 4 is typical,
        so we arrange for nheads as the contiguous mode for better vectorization.

    gather_kv_indices: used for topk sparsity with MLA absorption kernel.
    """
    return FlashAttnVarlenFunc.apply(
        q,
        k,
        v,
        qv,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        max_seqlen_q,
        max_seqlen_k,
        min_seqlen_k,
        gather_kv_indices,
        page_table,
        softmax_scale,
        causal,
        window_size,
        learnable_sink,
        softcap,
        num_splits,
        pack_gqa,
        deterministic,
        score_mod,
        score_mod_bwd,
        mask_mod,
        block_sparse_tensors,
        aux_tensors,
        aux_scalars,
        return_lse,
    )


def _compile_fwd_combine(
    dtype, dtype_partial, head_dim, tile_m, k_block_size, log_max_splits,
    has_cu_seqlens, has_seqused, has_lse, has_varlen_batch_idx,
):
    """Compile fwd combine kernel using cute fake tensors (no real GPU tensors needed)."""
    sym = cute.sym_int
    div = 128 // dtype_partial.width  # 16-byte alignment in elements

    fa_combine = FlashAttentionForwardCombine(
        dtype=dtype,
        dtype_partial=dtype_partial,
        head_dim=head_dim,
        tile_m=tile_m,
        k_block_size=k_block_size,
        log_max_splits=log_max_splits,
    )
    if not fa_combine.can_implement(
        dtype, dtype_partial, head_dim, tile_m, k_block_size, log_max_splits,
        num_threads=256,
    ):
        raise RuntimeError(
            "FlashAttention combine kernel cannot be implemented with given parameters"
        )

    if has_cu_seqlens:
        # Varlen: (num_splits, total_q, nheads, headdim)
        num_splits, total_q, nheads = sym(), sym(), sym()
        mO_partial = fake_tensor(dtype_partial, (num_splits, total_q, nheads, head_dim), divisibility=div)
        mLSE_partial = fake_tensor(Float32, (num_splits, total_q, nheads), divisibility=1, leading_dim=1)
        mO = fake_tensor(dtype, (total_q, nheads, head_dim), divisibility=div)
        mLSE = fake_tensor(Float32, (total_q, nheads), divisibility=1, leading_dim=0) if has_lse else None
    else:
        # Batched: (num_splits, batch, seqlen, nheads, headdim)
        num_splits, batch, seqlen, nheads = sym(), sym(), sym(), sym()
        mO_partial = fake_tensor(dtype_partial, (num_splits, batch, seqlen, nheads, head_dim), divisibility=div)
        mLSE_partial = fake_tensor(Float32, (num_splits, batch, seqlen, nheads), divisibility=1, leading_dim=2)
        mO = fake_tensor(dtype, (batch, seqlen, nheads, head_dim), divisibility=div)
        mLSE = fake_tensor(Float32, (batch, seqlen, nheads), divisibility=1, leading_dim=1) if has_lse else None
        batch = mO_partial.shape[1]

    batch_for_1d = batch if not has_cu_seqlens else sym()
    batchp1 = sym()
    mCuSeqlens = fake_tensor(Int32, (batchp1,), divisibility=1) if has_cu_seqlens else None
    mSeqused = fake_tensor(Int32, (batch_for_1d,), divisibility=1) if has_seqused else None
    mNumSplitsDynamic = None  # Not parametrized in compile_key
    mVarlenBatchIdx = fake_tensor(Int32, (batch_for_1d,), divisibility=1) if has_varlen_batch_idx else None
    mSemaphore = None  # Not parametrized in compile_key

    return cute.compile(
        fa_combine,
        mO_partial, mLSE_partial, mO, mLSE,
        mCuSeqlens, mSeqused, mNumSplitsDynamic, mVarlenBatchIdx, mSemaphore,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _flash_attn_fwd_combine(
    out_partial: torch.Tensor,
    lse_partial: torch.Tensor,
    out: torch.Tensor,
    lse: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    num_splits_dynamic_ptr: Optional[torch.Tensor] = None,
    varlen_batch_idx: Optional[torch.Tensor] = None,
    semaphore_to_reset: Optional[torch.Tensor] = None,
) -> None:
    """Forward combine kernel for split attention computation.

    Combines partial outputs and log-sum-exp values from multiple splits
    of attention computation into final outputs.

    Args:
        out_partial: Partial outputs tensor (num_splits, batch, seqlen, nheads, headdim) or
                                            (num_splits, total_q, nheads, headdim) if there's cu_seqlens
        lse_partial: Partial LSE tensor (num_splits, batch, seqlen, nheads) or
                                       (num_splits, total_q, nheads) if there's cu_seqlens
        out: Output tensor (batch, seqlen, nheads, headdim) or (total_q, nheads, headdim) if there's cu_seqlens
        lse: Output LSE tensor (batch, seqlen, nheads) or (total_q, nheads) if there's cu_seqlens.
        cu_seqlens: Cumulative sequence lengths for variable length sequences
        seqused: Used sequence lengths for each batch
        num_splits_dynamic_ptr: Dynamic number of splits per batch
        semaphore_to_reset: Semaphore for synchronization
        k_block_size: Block size for head dimension

    Returns:
        None
    """
    assert out_partial.dtype in [torch.float16, torch.bfloat16, torch.float32], (
        "out_partial must be fp16, bf16, or fp32"
    )
    if not is_fake_mode():
        assert out_partial.is_cuda and lse_partial.is_cuda, "tensors must be on CUDA device"
    # Determine if this is variable length based on dimensions
    is_varlen = out_partial.dim() == 4
    # Validate optional tensors
    for t, name in [
        (cu_seqlens, "cu_seqlens"),
        (seqused, "seqused"),
        (num_splits_dynamic_ptr, "num_splits_dynamic_ptr"),
    ]:
        if t is not None:
            if not is_fake_mode():
                assert t.is_cuda, f"{name} must be on CUDA device"
            assert t.is_contiguous(), f"{name} must be contiguous"
    head_dim = out_partial.shape[-1]
    num_splits = out_partial.shape[0]
    assert num_splits <= 256
    # If hdim is 96 or 192, it's faster to round them to 128 or 256 respectively
    # so that kBlockM is smaller and we have more parallelism.
    k_block_size = 64 if head_dim <= 64 else 128
    # We want kBlockM to be as small as possible to maximize parallelism.
    # E.g., if hdim is 64, we want kBlockM to be 16 so that we can use 256 threads, each reading 4 elements (floats).
    tile_m = 8 if k_block_size % 128 == 0 else (16 if k_block_size % 64 == 0 else 32)
    log_max_splits = max(math.ceil(math.log2(num_splits)), 4)
    if tile_m == 8:
        # If kBlockM == 8 then the minimum number of splits is 32.
        # TODO: we can deal w this by using 128 threads instead
        log_max_splits = max(log_max_splits, 5)

    # Create combine kernel configuration
    dtype = torch2cute_dtype_map[out.dtype]
    dtype_partial = torch2cute_dtype_map[out_partial.dtype]
    compile_key = (
        dtype,
        dtype_partial,
        head_dim,
        tile_m,
        k_block_size,
        log_max_splits,
        cu_seqlens is not None,
        seqused is not None,
        lse is not None,
        varlen_batch_idx is not None,
    )
    if compile_key not in _flash_attn_fwd_combine.compile_cache:
        _flash_attn_fwd_combine.compile_cache[compile_key] = _compile_fwd_combine(
            *compile_key
        )
    if not is_fake_mode():
        _flash_attn_fwd_combine.compile_cache[compile_key](
            out_partial, lse_partial, out, lse,
            cu_seqlens, seqused, num_splits_dynamic_ptr, varlen_batch_idx,
            semaphore_to_reset,
        )


_flash_attn_fwd_combine.compile_cache = get_jit_cache("fwd_combine")


def flash_attn_combine(
    out_partial: torch.Tensor,
    lse_partial: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    varlen_batch_idx: Optional[torch.Tensor] = None,
    return_lse: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Flash Attention combine function for split attention computation.

    Combines partial outputs and log-sum-exp values from multiple splits
    of attention computation into final outputs. This is the main user-facing
    interface for the combine kernel.

    Args:
        out_partial: Partial outputs tensor with shape:
            - (num_splits, batch_size, seqlen, num_heads, head_size) for regular batched input
            - (num_splits, total_q, num_heads, head_size) for variable length input
        lse_partial: Partial LSE tensor with shape:
            - (num_splits, batch_size, seqlen, num_heads) for regular batched input
            - (num_splits, total_q, num_heads) for variable length input
        out: Optional output tensor. If None, will be created automatically.
        out_dtype: Optional output dtype. If None, will use fp16/bf16 based on input.
        cu_seqlens: Cumulative sequence lengths for variable length sequences
        seqused: Used sequence lengths for each batch
        varlen_batch_idx: Optional mapping from virtual batch index to real batch index
            (int32 tensor of shape (batch_size,)). Used by persistent tile schedulers
            that reorder batch processing for load balancing.
        return_lse: Whether to return the combined LSE tensor. Default is True.

    Returns:
        Tuple of (out, lse) where:
        - out: Combined output tensor with shape (batch_size, seqlen, num_heads, head_size)
              or (total_q, num_heads, head_size) for varlen
        - lse: Combined log-sum-exp tensor with shape (batch_size, seqlen, num_heads)
              or (total_q, num_heads) for varlen. None if return_lse=False

    Note:
        This function expects the input tensors to be in the format produced by
        split attention computation, where the first dimension is num_splits.
        The permuting from user format to kernel format is now done inside the kernel.
    """
    # Input validation
    assert out_partial.dim() in [4, 5], "out_partial must have 4 or 5 dimensions"
    # Determine if this is variable length based on dimensions
    is_varlen = out_partial.dim() == 4
    if is_varlen:
        # Variable length: (num_splits, total_q, num_heads, head_size)
        num_splits, total_q, num_heads, head_size = out_partial.shape
        batch_size = 1  # Treat as single batch for varlen
        seqlen = total_q
    else:
        # Regular batched: (num_splits, batch_size, seqlen, num_heads, head_size)
        num_splits, batch_size, seqlen, num_heads, head_size = out_partial.shape
    # Determine output dtype
    if out_dtype is None:
        out_dtype = out_partial.dtype
    # Create output if not provided
    device = out_partial.device
    if out is None:
        if is_varlen:
            out = torch.empty(total_q, num_heads, head_size, dtype=out_dtype, device=device)
        else:
            out = torch.empty(
                batch_size, seqlen, num_heads, head_size, dtype=out_dtype, device=device
            )
    # Create lse output only if requested
    if return_lse:
        if is_varlen:
            lse = torch.empty(num_heads, total_q, dtype=torch.float32, device=device)
        else:
            lse = torch.empty(batch_size, num_heads, seqlen, dtype=torch.float32, device=device)
        lse = lse.transpose(-1, -2)
    else:
        lse = None
    _flash_attn_fwd_combine(
        out_partial,
        lse_partial,
        out,
        lse,
        cu_seqlens,
        seqused,
        varlen_batch_idx=varlen_batch_idx,
    )
    return out, lse
