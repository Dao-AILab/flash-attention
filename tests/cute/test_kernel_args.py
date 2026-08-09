import importlib
import pkgutil

import pytest
from cutlass import Float32, Int32

import flash_attn.cute
from flash_attn.cute.flash_bwd import FlashAttentionBackwardSm80
from flash_attn.cute.flash_bwd_sm90 import FlashAttentionBackwardSm90
from flash_attn.cute.flash_bwd_sm100 import FlashAttentionBackwardSm100
from flash_attn.cute.flash_fwd import FlashAttentionForwardSm80
from flash_attn.cute.flash_fwd_mla_sm100 import FlashAttentionMLAForwardSm100
from flash_attn.cute.flash_fwd_sm90 import FlashAttentionForwardSm90
from flash_attn.cute.flash_fwd_sm100 import FlashAttentionForwardSm100
from flash_attn.cute.kernel_args import (
    BwdKernelArgs,
    FwdKernelArgs,
    normalize_kernel_args,
)
from flash_attn.cute.sm100_hd256_2cta_fmha_backward import (
    BlackwellFusedMultiHeadAttentionBackward,
)
from flash_attn.cute.sm100_hd256_2cta_fmha_forward import (
    BlackwellFusedMultiHeadAttentionForward,
)
from flash_attn.cute.utils import AuxData

# Every kernel that is handed a superset namedtuple, and the superset it is narrowed from.
KERNEL_SUPERSETS = {
    FlashAttentionForwardSm80: FwdKernelArgs,
    FlashAttentionForwardSm90: FwdKernelArgs,
    FlashAttentionForwardSm100: FwdKernelArgs,
    FlashAttentionMLAForwardSm100: FwdKernelArgs,
    BlackwellFusedMultiHeadAttentionForward: FwdKernelArgs,
    FlashAttentionBackwardSm80: BwdKernelArgs,
    FlashAttentionBackwardSm90: BwdKernelArgs,
    FlashAttentionBackwardSm100: BwdKernelArgs,
    BlackwellFusedMultiHeadAttentionBackward: BwdKernelArgs,
}

FWD_REQUIRED = dict(mQ="q", mK="k", mV="v", mO="o", softmax_scale=Float32(1.0))
BWD_REQUIRED = dict(
    mQ="q",
    mK="k",
    mV="v",
    mdO="do",
    mLSE="lse",
    mdPsum="dpsum",
    mdQaccum="dqaccum",
    mdK="dk",
    mdV="dv",
    softmax_scale=Float32(1.0),
)


def test_unsupported_arguments_are_all_reported():
    args = FwdKernelArgs(
        **FWD_REQUIRED,
        descale_tensors="descale",
        tile_count_semaphore="semaphore",
        max_seqlen_q=Int32(8),
    )
    with pytest.raises(TypeError) as excinfo:
        normalize_kernel_args(args, FlashAttentionForwardSm90.Args, "Sm90")
    message = str(excinfo.value)
    for field in ("descale_tensors", "tile_count_semaphore", "max_seqlen_q"):
        assert field in message


def test_unsupported_arguments_are_ignored_when_none():
    args = FwdKernelArgs(**FWD_REQUIRED, descale_tensors=None, max_seqlen_q=None)
    narrowed = normalize_kernel_args(args, FlashAttentionForwardSm90.Args, "Sm90")
    assert isinstance(narrowed, FlashAttentionForwardSm90.Args)
    assert narrowed.mQ == "q" and narrowed.softmax_scale == Float32(1.0)


def test_none_falls_back_to_the_kernel_default():
    args = FwdKernelArgs(**FWD_REQUIRED, aux_data=None)
    narrowed = normalize_kernel_args(args, FlashAttentionForwardSm90.Args, "Sm90")
    assert narrowed.aux_data == AuxData()


def test_none_falls_back_to_the_kernel_default_for_the_kernels_own_args():
    args = FlashAttentionForwardSm90.Args(**FWD_REQUIRED, aux_data=None)
    narrowed = normalize_kernel_args(args, FlashAttentionForwardSm90.Args, "Sm90")
    assert narrowed.aux_data == AuxData()


def test_raw_python_scalars_are_rejected():
    args = FwdKernelArgs(**{**FWD_REQUIRED, "softmax_scale": 1.0}, window_size_left=3)
    with pytest.raises(TypeError) as excinfo:
        normalize_kernel_args(args, FlashAttentionForwardSm90.Args, "Sm90")
    message = str(excinfo.value)
    assert "softmax_scale" in message and "window_size_left" in message


def test_missing_required_arguments_are_all_reported_with_the_kernel_name():
    args = FwdKernelArgs(**{**FWD_REQUIRED, "mQ": None, "mV": None})
    with pytest.raises(TypeError) as excinfo:
        normalize_kernel_args(args, FlashAttentionForwardSm90.Args, "Sm90")
    message = str(excinfo.value)
    assert "Sm90" in message and "mQ" in message and "mV" in message


def test_kernel_specific_contracts():
    hd256 = normalize_kernel_args(
        FwdKernelArgs(**FWD_REQUIRED),
        BlackwellFusedMultiHeadAttentionForward.Args,
        "hd256",
    )
    assert "learnable_sink" not in hd256._fields

    with pytest.raises(TypeError):
        normalize_kernel_args(
            FwdKernelArgs(**FWD_REQUIRED, learnable_sink="sink"),
            BlackwellFusedMultiHeadAttentionForward.Args,
            "hd256",
        )

    bwd = normalize_kernel_args(
        BwdKernelArgs(**BWD_REQUIRED), FlashAttentionBackwardSm100.Args, "Sm100"
    )
    assert bwd.mdQaccum == "dqaccum" and bwd.mCuTotalMBlocks is None


@pytest.mark.parametrize("kernel", KERNEL_SUPERSETS, ids=lambda kernel: kernel.__name__)
def test_kernel_args_are_a_subset_of_the_superset(kernel):
    superset = KERNEL_SUPERSETS[kernel]
    unknown = set(kernel.Args._fields) - set(superset._fields)
    assert not unknown, (
        f"{kernel.__name__}.Args declares {sorted(unknown)}, which {superset.__name__} "
        f"never populates, so the kernel would silently receive the default instead"
    )


@pytest.mark.parametrize(
    "superset", [FwdKernelArgs, BwdKernelArgs], ids=lambda s: s.__name__
)
def test_no_superset_field_is_unreachable(superset):
    accepted = set().union(
        *(
            set(kernel.Args._fields)
            for kernel, sup in KERNEL_SUPERSETS.items()
            if sup is superset
        )
    )
    assert not set(superset._fields) - accepted


def test_every_kernel_declaring_args_is_registered():
    declared = set()
    for module_info in pkgutil.iter_modules(flash_attn.cute.__path__):
        module = importlib.import_module(f"flash_attn.cute.{module_info.name}")
        for obj in vars(module).values():
            if not isinstance(obj, type) or obj.__module__ != module.__name__:
                continue
            args_cls = obj.__dict__.get("Args")
            if isinstance(args_cls, type) and hasattr(args_cls, "_fields"):
                declared.add(obj)
    assert declared == set(KERNEL_SUPERSETS)
