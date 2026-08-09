import pytest
from cutlass import Float32, Int32

from flash_attn.cute.flash_bwd_sm100 import FlashAttentionBackwardSm100
from flash_attn.cute.flash_fwd_sm90 import FlashAttentionForwardSm90
from flash_attn.cute.kernel_args import (
    BwdKernelArgs,
    FwdKernelArgs,
    normalize_kernel_args,
)
from flash_attn.cute.sm100_hd256_2cta_fmha_forward import (
    BlackwellFusedMultiHeadAttentionForward,
)
from flash_attn.cute.utils import AuxData

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


def test_raw_python_scalars_are_rejected():
    args = FwdKernelArgs(**{**FWD_REQUIRED, "softmax_scale": 1.0}, window_size_left=3)
    with pytest.raises(TypeError) as excinfo:
        normalize_kernel_args(args, FlashAttentionForwardSm90.Args, "Sm90")
    message = str(excinfo.value)
    assert "softmax_scale" in message and "window_size_left" in message


def test_missing_required_argument_is_rejected():
    args = FwdKernelArgs(**{**FWD_REQUIRED, "mQ": None})
    with pytest.raises(TypeError):
        normalize_kernel_args(args, FlashAttentionForwardSm90.Args, "Sm90")


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
