"""p_quant keyword: validation, dispatch and compile keys (FakeTensorMode, no kernel launch).

Calls run under FakeTensorMode with the autotuner's compile-miss probe, so a call records the
kernels it would compile without compiling or launching anything.
"""

from dataclasses import replace

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from flash_attn.cute import interface
from flash_attn.cute.config import (
    SWAPAB,
    FwdHeuristicInputs,
    is_2cta_eligible,
    select_fwd_config,
    validate_fwd_config,
)
from flash_attn.cute.interface import BlockScales, TensorwiseScales, _flash_attn_fwd
from flash_attn.cute.p_quant import (
    E4M3_MAX,
    PQuantFormat,
    register_p_quant_format,
    resolve_p_quant,
    unregister_p_quant_format,
)
from p_quant_test_utils import capture_fwd_config

FORMATS = ["fp8_e4m3", "nvfp4"]
ALL_FORMATS = FORMATS + ["identity_fp8_e4m3", "identity_nvfp4"]


@pytest.fixture(autouse=True)
def _blackwell_env(monkeypatch):
    monkeypatch.setenv("FLASH_ATTENTION_NUM_SMS", "148")
    monkeypatch.delenv("FLASH_ATTENTION_SM100_BACKEND", raising=False)


def qkv(b=2, sq=256, sk=256, h=8, hk=2, d=128, dtype=torch.bfloat16, dv=None):
    q = torch.empty(b, sq, h, d, dtype=dtype, device="cuda")
    k = torch.empty(b, sk, hk, d, dtype=dtype, device="cuda")
    v = torch.empty(b, sk, hk, dv or d, dtype=dtype, device="cuda")
    return q, k, v


def would_compile(arch=100, build=qkv, **kwargs):
    """Compile-cache keys the call would compile (forward and helper kernels)."""
    misses = set()
    token = interface._compile_misses.set(misses)
    try:
        with FakeTensorMode():
            q, k, v = build()
            extra = {name: fn() for name, fn in kwargs.pop("fake", {}).items()}
            _flash_attn_fwd(q, k, v, _arch=arch, **extra, **kwargs)
    finally:
        interface._compile_misses.reset(token)
    return {key for _, key in misses}


def forward_key(keys):
    """The forward kernel's key: the one that starts with a FwdConfig."""
    fwd = [k for k in keys if isinstance(k[0], interface.FwdConfig)]
    assert len(fwd) == 1, fwd
    return fwd[0]


# ------------------------------------------------------------------ supported


@pytest.mark.parametrize("fmt", ALL_FORMATS)
@pytest.mark.parametrize(
    "call",
    [
        dict(),
        dict(causal=True),
        dict(window_size_left=64, window_size_right=0),
        dict(num_splits=4, build=lambda: qkv(b=1, sq=128, sk=8192)),
        dict(seqlen_k_per_split=1024, build=lambda: qkv(b=1, sq=128, sk=8192)),
        dict(pack_gqa=True),
        dict(pack_gqa=False),
        dict(build=lambda: qkv(d=64)),
        dict(build=lambda: qkv(d=96)),
        dict(build=lambda: qkv(sq=100, sk=300, h=4, hk=4)),
        dict(return_lse=True),
    ],
)
def test_supported_combinations_reach_the_standard_forward_with_the_mode(fmt, call):
    call = dict(call)
    build = call.pop("build", qkv)
    key = forward_key(would_compile(build=build, p_quant=fmt, **call))
    assert key[-3] == "p_quant" and key[-2].name == fmt
    assert not key[0].use_swapab


@pytest.mark.parametrize("fmt", FORMATS)
def test_varlen_inputs_are_supported(fmt):
    def build():
        return (
            torch.empty(600, 8, 64, dtype=torch.bfloat16, device="cuda"),
            torch.empty(700, 2, 64, dtype=torch.bfloat16, device="cuda"),
            torch.empty(700, 2, 64, dtype=torch.bfloat16, device="cuda"),
        )

    fake = dict(
        cu_seqlens_q=lambda: torch.empty(4, dtype=torch.int32, device="cuda"),
        cu_seqlens_k=lambda: torch.empty(4, dtype=torch.int32, device="cuda"),
    )
    key = forward_key(
        would_compile(
            build=build,
            p_quant=fmt,
            causal=True,
            max_seqlen_q=300,
            max_seqlen_k=300,
            fake=fake,
        )
    )
    assert key[-2].name == fmt


def test_requires_grad_inputs_are_supported():
    def build():
        q, k, v = qkv()
        return q.requires_grad_(), k.requires_grad_(), v.requires_grad_()

    key = forward_key(would_compile(build=build, p_quant="nvfp4", causal=True))
    assert key[-2].name == "nvfp4"


# -------------------------------------------------------------------- refused


def _fake_sink():
    return torch.empty(8, dtype=torch.float32, device="cuda")


def _fake_page_table():
    return torch.empty(2, 4, dtype=torch.int32, device="cuda")


REFUSED = {
    "fp16": dict(build=lambda: qkv(dtype=torch.float16)),
    "fp8_v": dict(build=lambda: qkv(dtype=torch.float8_e4m3fn)),
    "head_dim_256": dict(build=lambda: qkv(d=256)),
    "head_dim_32": dict(build=lambda: qkv(d=32)),
    "unequal_head_dims": dict(build=lambda: qkv(d=64, dv=128)),
    "softcap": dict(softcap=30.0),
    "learnable_sink": dict(fake=dict(learnable_sink=_fake_sink)),
    "tensorwise_scales": dict(scales=TensorwiseScales()),
    "sm_90": dict(arch=90),
    "sm_80": dict(arch=80),
    "sm_103": dict(arch=103),
    "sm_107": dict(arch=107),
    "sm_110": dict(arch=110),
    "sm_120": dict(arch=120),
}


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("case", sorted(REFUSED))
def test_refused_combinations_raise_and_compile_nothing(fmt, case):
    call = dict(REFUSED[case])
    build = call.pop("build", qkv)
    misses = set()
    token = interface._compile_misses.set(misses)
    try:
        with pytest.raises(NotImplementedError, match="p_quant") as exc:
            with FakeTensorMode():
                q, k, v = build()
                extra = {name: fn() for name, fn in call.pop("fake", {}).items()}
                _flash_attn_fwd(
                    q, k, v, _arch=call.pop("arch", 100), p_quant=fmt, **extra, **call
                )
    finally:
        interface._compile_misses.reset(token)
    assert fmt in str(exc.value)
    assert not misses  # refused before any kernel is selected or compiled


@pytest.mark.parametrize("fmt", FORMATS)
def test_callable_modifiers_and_sparse_inputs_are_refused(fmt):
    for kwargs, feature in [
        (dict(score_mod=lambda *a: a[0]), "score_mod"),
        (dict(mask_mod=lambda *a: True), "mask_mod"),
        (dict(block_sparse_tensors=object()), "block sparsity"),
    ]:
        with pytest.raises(NotImplementedError, match=feature):
            with FakeTensorMode():
                _flash_attn_fwd(*qkv(), _arch=100, p_quant=fmt, **kwargs)


@pytest.mark.parametrize("fmt", FORMATS)
def test_mla_paged_kv_and_block_scales_are_refused(fmt):
    with FakeTensorMode():
        q, k, v = qkv()
        qv = torch.empty(2, 256, 8, 128, dtype=torch.bfloat16, device="cuda")
        with pytest.raises(NotImplementedError, match="MLA"):
            _flash_attn_fwd(q, k, v, qv=qv, _arch=100, p_quant=fmt)
        # Block scales need E4M3 Q/K; whichever check fires first, the call is refused.
        with pytest.raises((NotImplementedError, ValueError)):
            _flash_attn_fwd(
                q, k, v, _arch=100, p_quant=fmt, scales=BlockScales(None, None)
            )
        kp = torch.empty(8, 128, 2, 128, dtype=torch.bfloat16, device="cuda")
        with pytest.raises(NotImplementedError, match="paged KV"):
            _flash_attn_fwd(
                q, kp, kp, page_table=_fake_page_table(), _arch=100, p_quant=fmt
            )


@pytest.mark.parametrize("bad", ["int4", "fp8", "NVFP4", 3])
def test_unknown_format_raises_value_error_listing_formats(bad):
    with pytest.raises(ValueError, match="fp8_e4m3") as exc:
        with FakeTensorMode():
            _flash_attn_fwd(*qkv(), _arch=100, p_quant=bad)
    assert "nvfp4" in str(exc.value)


@pytest.mark.parametrize("fmt", FORMATS)
def test_validation_runs_before_the_zero_length_early_return(fmt):
    def empty_k():
        return qkv(sk=0)

    with FakeTensorMode():
        # Supported call with no keys: returns zeros / -inf without compiling.
        out, lse, *_ = _flash_attn_fwd(
            *empty_k(), _arch=100, p_quant=fmt, return_lse=True
        )
        assert out.shape == (2, 256, 8, 128)
        # Unsupported or unknown requests still raise even though nothing would run.
        with pytest.raises(ValueError):
            _flash_attn_fwd(*empty_k(), _arch=100, p_quant="int4")
        with pytest.raises(NotImplementedError, match="sm_90"):
            _flash_attn_fwd(*empty_k(), _arch=90, p_quant=fmt)
        with pytest.raises(NotImplementedError, match="softcap"):
            _flash_attn_fwd(*empty_k(), _arch=100, p_quant=fmt, softcap=20.0)


def test_compile_only_route_admits_sm10x_but_nothing_else():
    key = forward_key(
        would_compile(arch=103, p_quant="nvfp4", _p_quant_compile_only_arch=True)
    )
    assert key[-2].name == "nvfp4"
    forward_key(
        would_compile(arch=107, p_quant="fp8_e4m3", _p_quant_compile_only_arch=True)
    )
    with pytest.raises(NotImplementedError, match="sm_90"):
        would_compile(arch=90, p_quant="nvfp4", _p_quant_compile_only_arch=True)
    # The public entry points do not expose the override.
    import inspect

    for fn in (interface.flash_attn_func, interface.flash_attn_varlen_func):
        params = inspect.signature(fn).parameters
        assert "p_quant" in params and "_p_quant_compile_only_arch" not in params


# --------------------------------------------------------------------- SwapAB

_DECODE = FwdHeuristicInputs(
    device_arch=100,
    num_sms=148,
    dtype="torch.bfloat16",
    head_dim=128,
    head_dim_v=128,
    num_heads=8,
    num_heads_kv=2,
    batch_size=4,
    total_q=4,
    total_k=4 * 4096,
    max_seqlen_q=1,
    max_seqlen_k=4096,
)


def decode_qkv():
    return qkv(b=4, sq=1, sk=4096)


def test_decode_shaped_call_uses_swapab_only_without_the_mode():
    assert select_fwd_config(_DECODE).use_swapab
    assert forward_key(would_compile(build=decode_qkv))[0].use_swapab
    for fmt in ALL_FORMATS:
        assert SWAPAB.rejection(_DECODE._replace(p_quant=fmt)) == "p_quant"
        assert not select_fwd_config(_DECODE._replace(p_quant=fmt)).use_swapab
        key = forward_key(would_compile(build=decode_qkv, p_quant=fmt))
        assert not key[0].use_swapab and key[-2].name == fmt


@pytest.mark.parametrize("fmt", FORMATS)
def test_explicit_swapab_requests_are_rejected(fmt, monkeypatch):
    swapab_config = select_fwd_config(_DECODE)
    assert swapab_config.use_swapab
    with pytest.raises(ValueError, match="SwapAB unsupported: p_quant"):
        validate_fwd_config(swapab_config, _DECODE._replace(p_quant=fmt))
    with pytest.raises(NotImplementedError, match="SwapAB config"):
        with FakeTensorMode():
            _flash_attn_fwd(*decode_qkv(), _arch=100, p_quant=fmt, config=swapab_config)
    monkeypatch.setenv("FLASH_ATTENTION_SM100_BACKEND", "swapab")
    with pytest.raises(NotImplementedError, match="forced SwapAB backend"):
        with FakeTensorMode():
            _flash_attn_fwd(*decode_qkv(), _arch=100, p_quant=fmt)
    # Explicitly asking for the standard backend is fine.
    monkeypatch.setenv("FLASH_ATTENTION_SM100_BACKEND", "standard")
    assert not forward_key(would_compile(build=decode_qkv, p_quant=fmt))[0].use_swapab


def test_explicit_standard_config_is_accepted_and_keeps_its_tiles():
    base = forward_key(would_compile(causal=True))[0]
    for fmt in FORMATS:
        key = forward_key(would_compile(causal=True, p_quant=fmt))
        # The mode does not change tile sizes, scheduling or exp2 emulation choices.
        assert key[0] == base
        explicit = forward_key(would_compile(causal=True, p_quant=fmt, config=base))
        assert explicit == key


# ---------------------------------------------------------------------- 2-CTA


def resolved(build=qkv, **kwargs):
    """(FwdConfig, FwdHeuristicInputs, compile keys) of a call, resolved on the host."""
    with capture_fwd_config() as captured:
        keys = would_compile(build=build, **kwargs)
    assert len(captured) == 1, captured
    config, inputs = captured[0]
    return config, inputs, keys


def cta_qkv(sk=391, d=128):
    """Every 2-CTA condition of the selector holds at head dim 128 unless the call adds
    SplitKV, causal or a window: dense MHA with more query rows than one two-stage work tile."""
    return qkv(b=2, sq=300, sk=sk, h=2, hk=2, d=d)


def eligible_for_2cta(config, inputs):
    return is_2cta_eligible(
        inputs._replace(pack_gqa=config.pack_gqa),
        tile_m=config.tile_m,
        q_stage=config.q_stage,
        num_splits=config.num_splits,
    )


@pytest.mark.parametrize("fmt", ALL_FORMATS)
def test_unsplit_noncausal_hd128_call_resolves_to_2cta(fmt):
    config, inputs, keys = resolved(build=cta_qkv, p_quant=fmt)
    assert inputs.p_quant == fmt and inputs.head_dim == 128
    assert not inputs.causal and not inputs.local and config.num_splits == 1
    assert config.use_2cta_instrs and eligible_for_2cta(config, inputs)
    key = forward_key(keys)
    assert key[0] == config and key[-2].name == fmt
    # The mode does not change the choice, and the explicit 2-CTA config is accepted with it.
    assert resolved(build=cta_qkv)[0] == config
    validate_fwd_config(config, inputs)
    assert forward_key(would_compile(build=cta_qkv, p_quant=fmt, config=config)) == key


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("d", [64, 96])
def test_smaller_head_dims_resolve_to_1cta(fmt, d):
    config, inputs, _ = resolved(build=lambda: cta_qkv(d=d), p_quant=fmt)
    assert not config.use_2cta_instrs and not eligible_for_2cta(config, inputs)


TWO_CTA_INCOMPATIBLE = {
    "splitkv": dict(num_splits=2, build=lambda: cta_qkv(sk=2048)),
    "causal": dict(causal=True),
    "window": dict(window_size_left=64, window_size_right=0),
}


@pytest.mark.parametrize("fmt", ALL_FORMATS)
@pytest.mark.parametrize("case", sorted(TWO_CTA_INCOMPATIBLE))
def test_explicit_2cta_config_is_rejected_with_splitkv_causal_or_window(fmt, case):
    call = dict(TWO_CTA_INCOMPATIBLE[case])
    build = call.pop("build", cta_qkv)
    config, inputs, _ = resolved(build=build, p_quant=fmt, **call)
    # The call has the property the case is for, and the selector keeps it 1-CTA.
    has_property = {
        "splitkv": config.num_splits > 1,
        "causal": inputs.causal and not inputs.local,
        "window": inputs.local and not inputs.causal,
    }
    assert has_property[case] and sum(has_property.values()) == 1
    assert not config.use_2cta_instrs and not eligible_for_2cta(config, inputs)
    # Only that property stands in the way: the same shapes without it resolve to 2-CTA.
    assert resolved(build=build, p_quant=fmt)[0].use_2cta_instrs
    # An explicit 2-CTA request is refused, by the validator and at the forward entry.
    # lpt_global is a 1-CTA ordering; it is cleared so the 2-CTA rule is the one that answers.
    explicit = replace(config, use_2cta_instrs=True, lpt_global=False)
    with pytest.raises(ValueError, match="2CTA instructions are not supported"):
        validate_fwd_config(explicit, inputs)
    misses = set()
    token = interface._compile_misses.set(misses)
    try:
        with pytest.raises(ValueError, match="2CTA instructions are not supported"):
            with FakeTensorMode():
                _flash_attn_fwd(
                    *build(), _arch=100, p_quant=fmt, config=explicit, **call
                )
    finally:
        interface._compile_misses.reset(token)
    assert not misses  # refused before any kernel is compiled


# ------------------------------------------------- Transformer Engine windows


@pytest.mark.parametrize("fmt", FORMATS)
def test_causal_with_an_unbounded_left_window_is_plain_causal(fmt):
    plain = resolved(p_quant=fmt, causal=True)
    spelled = resolved(
        p_quant=fmt, causal=True, window_size_left=None, window_size_right=0
    )
    assert spelled[1].causal and not spelled[1].local
    assert spelled[:2] == plain[:2]


@pytest.mark.parametrize("fmt", FORMATS)
def test_te_causal_window_spelling_is_an_empty_local_window_in_this_tree(fmt):
    """Transformer Engine 2.20.2 passes window_size_left=-1, window_size_right=0 with
    causal=True at its causal steps. This tree reads -1 as unbounded only when both sides are
    negative, so the pair is a local window with left bound -1 and no visible key. A caller
    that forwards TE's arguments therefore has to pass None for a negative side. If this test
    starts failing, the tree reads TE's spelling itself and that translation can be removed."""
    _, inputs, _ = resolved(
        p_quant=fmt, causal=True, window_size_left=-1, window_size_right=0
    )
    assert inputs.local and not inputs.causal


# --------------------------------------------------------------- compile keys


def test_none_adds_nothing_to_the_forward_key():
    plain = would_compile(causal=True)
    assert would_compile(causal=True, p_quant=None) == plain
    assert would_compile(causal=True, p_quant="none") == plain
    assert "p_quant" not in forward_key(plain)


def test_formats_and_controls_have_pairwise_distinct_forward_keys():
    keys = {
        fmt: forward_key(would_compile(causal=True, p_quant=fmt)) for fmt in ALL_FORMATS
    }
    keys["none"] = forward_key(would_compile(causal=True))
    assert len(set(keys.values())) == len(keys)
    # Only the forward kernel is specialized: every helper kernel key is shared.
    plain = would_compile(
        causal=True, return_lse=True, num_splits=4, build=lambda: qkv(b=1, sk=8192)
    )
    for fmt in ALL_FORMATS:
        with_mode = would_compile(
            causal=True,
            return_lse=True,
            num_splits=4,
            build=lambda: qkv(b=1, sk=8192),
            p_quant=fmt,
        )
        assert len(with_mode) == len(plain)
        assert (
            len(with_mode - plain) == 1
        )  # exactly one new specialization: the forward


def test_a_registered_format_gets_its_own_key_without_dispatch_changes():
    fmt = PQuantFormat("fp8_e4m3_x128", "e4m3", 1, 7, 0.0, E4M3_MAX)
    register_p_quant_format(fmt)
    try:
        key = forward_key(would_compile(causal=True, p_quant="fp8_e4m3_x128"))
        assert key[-2] == fmt
        assert key != forward_key(would_compile(causal=True, p_quant="fp8_e4m3"))
        # Re-registering the name with another policy cannot reuse the old kernel.
        unregister_p_quant_format("fp8_e4m3_x128")
        other = register_p_quant_format(replace(fmt, max_offset=6))
        assert (
            forward_key(would_compile(causal=True, p_quant="fp8_e4m3_x128"))[-2]
            == other
        )
        assert other != fmt
    finally:
        unregister_p_quant_format("fp8_e4m3_x128")
    with pytest.raises(ValueError):
        would_compile(causal=True, p_quant="fp8_e4m3_x128")


def test_a_replaced_codec_helper_gets_another_forward_key(monkeypatch):
    """The key carries the hash of the codec helper: replacing the helper of a codec, with the
    format unchanged, cannot reuse the kernel compiled with the old helper. Putting the same
    helper back gives the first key again; identity formats have no helper."""
    from flash_attn.cute import softmax

    def key():
        return forward_key(would_compile(causal=True, p_quant="fp8_e4m3"))

    original = softmax._P_QUANT_CODECS["e4m3"]
    first = key()
    assert first[-3:-1] == ("p_quant", resolve_p_quant("fp8_e4m3"))
    assert isinstance(first[-1], str) and first[-1]

    def other_helper(p):
        return p

    monkeypatch.setitem(softmax._P_QUANT_CODECS, "e4m3", other_helper)
    replaced = key()
    assert replaced[:-1] == first[:-1] and replaced[-1] != first[-1]
    assert isinstance(replaced[-1], str)

    def other_helper_again(p):
        return p + 0

    monkeypatch.setitem(softmax._P_QUANT_CODECS, "e4m3", other_helper_again)
    assert key()[-1] not in (first[-1], replaced[-1])
    monkeypatch.setitem(softmax._P_QUANT_CODECS, "e4m3", original)
    assert key() == first
    identity = forward_key(would_compile(causal=True, p_quant="identity_fp8_e4m3"))
    assert identity[-1] is False
    # A format on another codec differs in the helper hash as well as in the format.
    assert forward_key(would_compile(causal=True, p_quant="nvfp4"))[-1] != first[-1]


def test_the_helper_hash_does_not_change_when_the_helper_is_traced(monkeypatch):
    """Tracing a ``@cute.jit`` helper replaces the function it wraps. The key must not depend on
    whether that has happened, or the same call would get one key before the first compile
    and another after it (and in a process that used the helper earlier)."""
    from flash_attn.cute import softmax, utils

    helper = softmax._P_QUANT_CODECS["nvfp4"]
    assert hasattr(helper, "__wrapped__")
    before = forward_key(would_compile(causal=True, p_quant="nvfp4"))

    def traced(p):
        return p

    monkeypatch.setattr(helper, "__wrapped__", traced)
    assert utils.hash_callable(helper, set_cute_hash=False) != before[-1]
    assert forward_key(would_compile(causal=True, p_quant="nvfp4")) == before


# ------------------------------------------- backward-only options, autotuning


@pytest.mark.parametrize("fmt", FORMATS)
def test_backward_only_options_are_refused_at_the_autograd_entry_points(fmt):
    def bwd_mod(*args):
        return args[0]

    with FakeTensorMode():
        q, k, v = qkv()
        with pytest.raises(NotImplementedError, match="score_mod_bwd"):
            interface.flash_attn_func(q, k, v, p_quant=fmt, score_mod_bwd=bwd_mod)
        with pytest.raises(NotImplementedError, match="block_sparse_tensors_bwd"):
            interface.flash_attn_func(
                q, k, v, p_quant=fmt, block_sparse_tensors_bwd=object()
            )
        qf = torch.empty(600, 8, 64, dtype=torch.bfloat16, device="cuda")
        kf = torch.empty(700, 2, 64, dtype=torch.bfloat16, device="cuda")
        cu_q = torch.empty(4, dtype=torch.int32, device="cuda")
        cu_k = torch.empty(4, dtype=torch.int32, device="cuda")
        with pytest.raises(NotImplementedError, match="score_mod_bwd"):
            interface.flash_attn_varlen_func(
                qf,
                kf,
                kf,
                cu_seqlens_q=cu_q,
                cu_seqlens_k=cu_k,
                p_quant=fmt,
                score_mod_bwd=bwd_mod,
            )
        # Also for a call with no keys, which would otherwise return early.
        with pytest.raises(NotImplementedError, match="score_mod_bwd"):
            interface.flash_attn_func(*qkv(sk=0), p_quant=fmt, score_mod_bwd=bwd_mod)
        # The low-level entry refuses the sparse-backward flag instead of ignoring it.
        with pytest.raises(NotImplementedError, match="gather_bwd_recompute_p"):
            _flash_attn_fwd(
                q, k, v, _arch=100, p_quant=fmt, gather_bwd_recompute_p=True
            )


def test_backward_only_options_stay_accepted_without_the_mode():
    # The refusal is specific to p_quant: the helper is a no-op when no format is set.
    interface._refuse_p_quant_backward_options(None, score_mod_bwd=lambda *a: a[0])
    interface._refuse_p_quant_backward_options(
        "none", block_sparse_tensors_bwd=object()
    )


@pytest.mark.parametrize("fmt", FORMATS)
def test_autotuning_is_refused_with_the_mode(fmt):
    from flash_attn.cute.autotune import tune_fwd_config

    config = forward_key(would_compile(causal=True))[0]
    with pytest.raises(NotImplementedError, match="p_quant"):
        tune_fwd_config([config], dict(q=None, k=None, v=None, p_quant=fmt))


# ------------------------------------------------------------- per-test evidence


def test_resolved_configs_and_schedules_are_written_as_evidence(tmp_path, monkeypatch):
    """With P_QUANT_EVIDENCE_DIR set (the hardware batch sets it), each forward call's
    resolved config and each oracle schedule is appended to evidence.jsonl."""
    import json

    import p_quant_test_utils as U

    monkeypatch.setenv("P_QUANT_EVIDENCE_DIR", str(tmp_path / "evidence"))
    config, inputs, _ = resolved(p_quant="nvfp4", causal=True)
    U.schedule_of(config, inputs, qhead_per_kvhead=4, causal=True)
    lines = (tmp_path / "evidence" / "evidence.jsonl").read_text().splitlines()
    records = [json.loads(line) for line in lines]
    assert [r["kind"] for r in records] == ["forward_call", "schedule"]
    call, schedule = records
    assert "test_resolved_configs_and_schedules_are_written_as_evidence" in call["test"]
    assert call["config"] == repr(config) and "FwdConfig" in call["config"]
    assert call["per_sequence_split_counts"] is None  # one static split count
    assert "causal=True" in schedule["schedule"]
    assert schedule["oracle"] == U.ORACLE_IDENTITY and len(U.ORACLE_IDENTITY) == 64
    # Without the variable nothing is written.
    monkeypatch.delenv("P_QUANT_EVIDENCE_DIR")
    resolved(p_quant="nvfp4", causal=True)
    assert len((tmp_path / "evidence" / "evidence.jsonl").read_text().splitlines()) == 2


# ------------------------------------------------------- default path stays cheap


def test_the_default_call_skips_p_quant_validation(monkeypatch):
    """Without a format the forward must not pay for the mode's validation; with any value,
    also the string "none", it must validate. (With the keyword handled on the default path a
    B200 host spent 1.4 to 2.7 us more per call, unprofiled. Neither the handling's share of
    that nor the saving from skipping it was measured.)"""
    calls = []
    real = interface._validate_p_quant

    def counting(p_quant, **kwargs):
        calls.append(p_quant)
        return real(p_quant, **kwargs)

    monkeypatch.setattr(interface, "_validate_p_quant", counting)
    plain = would_compile(causal=True)
    assert calls == []
    assert would_compile(causal=True, p_quant="none") == plain
    assert calls == ["none"]
    would_compile(causal=True, p_quant="nvfp4")
    assert calls == ["none", "nvfp4"]
    with pytest.raises(ValueError):  # an unknown name is still rejected
        would_compile(causal=True, p_quant="fp9")
    assert calls[-1] == "fp9"


def _dense_entry(**kwargs):
    q, k, v = qkv()
    return interface.flash_attn_func(q, k, v, causal=True, **kwargs)


def _varlen_entry(**kwargs):
    q = torch.empty(600, 8, 64, dtype=torch.bfloat16, device="cuda")
    k = torch.empty(700, 2, 64, dtype=torch.bfloat16, device="cuda")
    cu_q = torch.tensor([0, 100, 350, 600], dtype=torch.int32)
    cu_k = torch.tensor([0, 200, 300, 700], dtype=torch.int32)
    return interface.flash_attn_varlen_func(
        q,
        k,
        k.clone(),
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=250,
        max_seqlen_k=400,
        **kwargs,
    )


@pytest.mark.parametrize(
    "entry", [_dense_entry, _varlen_entry], ids=["dense", "varlen"]
)
def test_public_entry_points_handle_p_quant_only_when_it_is_given(monkeypatch, entry):
    """Through flash_attn_func and flash_attn_varlen_func: the autograd entry's refusal of
    backward-only options and the forward's validation each run once per call that names a
    format, and not at all on the default call."""
    seen = dict(refuse=[], validate=[])
    real_refuse = interface._refuse_p_quant_backward_options
    real_validate = interface._validate_p_quant

    def refuse(p_quant, **options):
        seen["refuse"].append(p_quant)
        return real_refuse(p_quant, **options)

    def validate(p_quant, **kwargs):
        seen["validate"].append(p_quant)
        return real_validate(p_quant, **kwargs)

    monkeypatch.setattr(interface, "_refuse_p_quant_backward_options", refuse)
    monkeypatch.setattr(interface, "_validate_p_quant", validate)
    monkeypatch.setattr(interface, "_get_device_arch", lambda: 100)

    def call(**kwargs):
        seen["refuse"].clear()
        seen["validate"].clear()
        token = interface._compile_misses.set(set())
        try:
            with FakeTensorMode():
                entry(**kwargs)
        finally:
            interface._compile_misses.reset(token)
        return dict(seen)

    assert call() == dict(refuse=[], validate=[])
    assert call(p_quant=None) == dict(refuse=[], validate=[])
    for value in ("none", "fp8_e4m3", "nvfp4"):
        assert call(p_quant=value) == dict(refuse=[value], validate=[value]), value
    # The count above is not vacuous: the wrapped helpers still do their work.
    with pytest.raises(ValueError):
        call(p_quant="fp9")
    with pytest.raises(NotImplementedError, match="score_mod_bwd"):
        call(p_quant="nvfp4", score_mod_bwd=lambda *args: args[0])
