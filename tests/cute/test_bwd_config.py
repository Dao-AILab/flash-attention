import itertools
from dataclasses import replace

import pytest

from flash_attn.cute.config import (
    BwdHeuristicInputs,
    Sm90BwdRegisters,
    Sm100BwdRegisters,
    bwd_q_subtile_factor,
    config_kv_subtile_factor,
    select_bwd_config,
    validate_bwd_config,
)

_BASE_INPUTS = BwdHeuristicInputs(
    device_arch=100,
    head_dim=128,
    head_dim_v=128,
    num_heads=16,
    num_heads_kv=16,
)


def make_inputs(**changes) -> BwdHeuristicInputs:
    return _BASE_INPUTS._replace(**changes)


# Frozen copy of the backward policy before it moved into select_bwd_config: the inline
# _flash_attn_bwd branches and _tile_size_bwd_sm90 (interface.py), the register split in FlashAttentionBackwardSm100.__init__, and the one in
# FlashAttentionBackwardSm90.__call__.
def legacy_sm90_tile(head_dim, head_dim_v, causal, local, sparse_q):
    fields = ("m", "n", "Q", "dO", "PdS", "SdP", "dKV", "dQ", "MSdP", "NdKV", "MdQ", "single")
    if head_dim <= 64:
        values = (128, 128, 2, 2, 2, True, False, False, 1, 2, 2, False)
    elif head_dim <= 96:
        values = (64, 128, 2, 2, 2, True, False, False, 1, 2, 1, True)
    elif head_dim <= 128:
        m = 64 if causal or local else 80
        if sparse_q is not None and sparse_q % m != 0:
            m = 64
        values = (m, 128, 2, 2, 2, True, False, m % 64 != 0, 1, 2, 1, False)
    elif head_dim <= 192:
        dO = 2 if head_dim_v <= 128 else 1
        values = (64, 96, 2, dO, 1, False, True, False, 1, 2, 1, False)
    elif sparse_q is not None:
        values = (64, 64, 1, 1, 1, False, False, False, 1, 1, 1, False)
    else:
        values = (64, 48, 2, 1, 1, False, True, False, 1, 1, 1, False)
    return dict(zip(fields, values))


def legacy_sm100_kernel_registers(causal, local, deterministic, use_2cta, cluster_size, head_dim):
    use_2cta = use_2cta and cluster_size == 2
    serial = use_2cta and -(-head_dim // 32) * 32 in (192, 256)
    if (not causal and not local) or deterministic:
        reduce = 136 if use_2cta else 152
        compute = 136
    else:
        reduce = 136
        compute = 136 if use_2cta else 144
    load = 104 if use_2cta else 96 - 8
    mma = 104 if use_2cta else load
    if serial:
        reduce, compute, load, mma = 128 + 8, 128 + 8, 128 - 24, 128 - 24
    return Sm100BwdRegisters(reduce, compute, load, mma, 24)


def legacy_sm90_kernel_registers(num_threads, dQ_single_wg):
    num_wg_mma = num_threads // 128 - 1
    if num_wg_mma == 2:
        return Sm90BwdRegisters(256, 224, 24) if dQ_single_wg else Sm90BwdRegisters(240, 240, 24)
    return Sm90BwdRegisters(160, 160, 32)


class LegacyKernelAssert(Exception):
    """The old interface reached FlashAttentionBackwardSm100's hdim-192/256 2CTA assert."""


def legacy_bwd_policy(inputs: BwdHeuristicInputs) -> dict:
    """Return the legacy settings (or raise its error) for the fields the old code owned."""
    arch = inputs.device_arch
    head_dim, head_dim_v = inputs.head_dim, inputs.head_dim_v
    causal, local = inputs.causal, inputs.local
    block_size = (
        (inputs.sparse_block_size_q, inputs.sparse_block_size_kv)
        if inputs.sparse_block_size_q is not None
        else None
    )
    sparse_q = None
    kv_subtile_factor = 1
    if inputs.use_block_sparsity:
        if block_size is not None:
            sparse_q = block_size[0]
        elif arch // 10 == 9:
            sparse_q = 128
    if arch // 10 == 12:
        stages = 2 if head_dim <= 64 else 1
        legacy = dict(
            m=64,
            n=64,
            Q=stages,
            dO=stages,
            SdP=False,
            dKV=False,
            dQ=False,
            MSdP=4,
            NdKV=4,
            MdQ=4,
            V_in_regs=False,
            single=False,
            num_threads=128,
            cluster_size=1,
            use_2cta=False,
            registers=None,
            spt=False,
            pack_gqa=False,
        )
    elif arch // 10 == 9:
        legacy = legacy_sm90_tile(head_dim, head_dim_v, causal, local, sparse_q)
        legacy.update(
            V_in_regs=inputs.requested_V_in_regs,
            num_threads=384,
            cluster_size=1,
            use_2cta=False,
            # The old FlashAttentionBackwardSm90.__call__ derived spt itself.
            spt=(causal or local) and inputs.deterministic,
            pack_gqa=False,
        )
        legacy["registers"] = legacy_sm90_kernel_registers(384, legacy["single"])
    else:
        n = 128
        hd256_generic = head_dim == 256 and head_dim_v == 256 and inputs.hd256_generic
        if hd256_generic:
            n = 64

        def subtile(n_block_size):
            if not inputs.use_block_sparsity or block_size is None:
                return 1
            if block_size[1] % n_block_size != 0:
                raise ValueError(
                    "Block sparsity expects sparse_block_size[1] "
                    f"to be a multiple of tile_n={n_block_size}; got {block_size[1]}."
                )
            return block_size[1] // n_block_size

        kv_subtile_factor = subtile(n)
        use_2cta = (
            head_dim >= 128
            and not inputs.disable_2cta
            and (not inputs.use_block_sparsity or subtile(n) % 2 == 0)
        )
        if (
            inputs.requested_tile_n == 64
            and use_2cta
            and head_dim <= 128
            and head_dim_v <= 128
            and all(-(-d // 32) * 32 % 64 == 0 for d in (head_dim, head_dim_v))
            and not inputs.use_block_sparsity
        ):
            n = 64
        needs_2cta = (
            "SM100 backward with head_dim=256"
            if hd256_generic
            else (
                "SM100 block-sparse backward with head_dim=192"
                if inputs.use_block_sparsity and head_dim == 192
                else None
            )
        )
        if needs_2cta is not None and not use_2cta:
            reason = (
                "2CTA was disabled by request"
                if inputs.disable_2cta
                else (
                    f"sparse_block_size[1] must cover an even number of tile_n={n} "
                    f"tiles; got factor {kv_subtile_factor}"
                )
            )
            raise ValueError(f"{needs_2cta} requires 2CTA; {reason}.")
        cluster_size = 2 if use_2cta else 1
        dedicated = head_dim == 256 and head_dim_v == 256 and not inputs.hd256_generic
        legacy = dict(
            m=128,
            n=n,
            dKV=False,
            dQ=False,
            NdKV=1,
            MdQ=1,
            cluster_size=cluster_size,
            use_2cta=use_2cta or dedicated,
            dedicated=dedicated,
            kv_subtile=kv_subtile_factor,
            # The legacy sweep is MHA, which never packs.
            pack_gqa=False,
        )
        if dedicated:
            legacy["registers"] = None
        else:
            # The old kernel derived its stages and register split itself.
            legacy.update(
                Q=1 if use_2cta else 2,
                dO=1,
                registers=legacy_sm100_kernel_registers(
                    causal, local, inputs.deterministic, use_2cta, cluster_size, head_dim
                ),
            )
        spt_given = inputs.use_block_sparsity and inputs.sparse_spt is not None
        legacy["spt"] = (
            inputs.sparse_spt and inputs.deterministic
            if spt_given
            else (causal or local) and inputs.deterministic
        )
    m = legacy["m"]
    if sparse_q is not None and (sparse_q <= 0 or sparse_q % m != 0):
        raise ValueError(
            "Block sparsity requires sparse_block_size[0] to be a multiple of "
            f"tile_m={m}; got {sparse_q}."
        )
    legacy["q_subtile"] = sparse_q // m if sparse_q is not None else 2
    generic_sm100 = arch // 10 in (10, 11) and not legacy["dedicated"]
    if generic_sm100 and -(-head_dim // 32) * 32 in (192, 256) and not legacy["use_2cta"]:
        raise LegacyKernelAssert
    return legacy


_CONFIG_FIELDS = dict(
    m="tile_m",
    n="tile_n",
    Q="num_stages_Q",
    dO="num_stages_dO",
    PdS="num_stages_PdS",
    SdP="SdP_swapAB",
    dKV="dKV_swapAB",
    dQ="dQ_swapAB",
    MSdP="AtomLayoutMSdP",
    NdKV="AtomLayoutNdKV",
    MdQ="AtomLayoutMdQ",
    single="dQ_single_wg",
    V_in_regs="V_in_regs",
    num_threads="num_threads",
    cluster_size="cluster_size",
    use_2cta="use_2cta_instrs",
    dedicated="use_dedicated_hd256_kernel",
    spt="spt",
    registers="registers",
    pack_gqa="pack_gqa",
)


def selected_settings(inputs: BwdHeuristicInputs, legacy_keys) -> dict:
    config = select_bwd_config(inputs)
    got = {k: getattr(config, f) for k, f in _CONFIG_FIELDS.items() if k in legacy_keys}
    got["q_subtile"] = bwd_q_subtile_factor(config, inputs)
    if "kv_subtile" in legacy_keys:
        got["kv_subtile"] = config_kv_subtile_factor(config, inputs)
    return got


def outcome(fn):
    try:
        return fn()
    except ValueError as error:
        return f"ValueError: {error}"


_HEAD_DIMS = [(64, 64), (96, 96), (128, 128), (192, 128), (256, 256)]
_SPARSE = [  # (use_block_sparsity, block_size, spt)
    (False, None, None),
    (True, None, None),
    (True, (128, 128), None),
    (True, (256, 256), True),
    (True, (128, 256), False),
    (True, (640, 128), None),
    (True, (64, 64), None),
    (True, (128, 192), None),
]
_ENVS = [  # (disable_2cta, hd256_generic, requested_tile_n)
    (False, False, None),
    (True, False, None),
    (False, True, None),
    (True, True, None),
    (False, False, 64),
]


@pytest.mark.parametrize("device_arch", [90, 100, 103, 110, 120])
def test_selection_matches_the_legacy_policy(device_arch):
    # Dtype does not affect this policy; V_in_regs is only forwarded on SM90.
    for (d, dv), mask, det, sparse, env in itertools.product(
        _HEAD_DIMS,
        ["none", "causal", "local"],
        [False, True],
        _SPARSE,
        _ENVS,
    ):
        use_block_sparsity, block_size, spt = sparse
        disable_2cta, hd256_generic, tile_n = env
        inputs = make_inputs(
            device_arch=device_arch,
            head_dim=d,
            head_dim_v=dv,
            causal=mask == "causal",
            local=mask == "local",
            deterministic=det,
            use_block_sparsity=use_block_sparsity,
            sparse_block_size_q=None if block_size is None else block_size[0],
            sparse_block_size_kv=None if block_size is None else block_size[1],
            sparse_spt=spt,
            requested_V_in_regs=mask == "none",
            requested_tile_n=tile_n,
            disable_2cta=disable_2cta,
            hd256_generic=hd256_generic,
        )
        try:
            expected = outcome(lambda: legacy_bwd_policy(inputs))
        except LegacyKernelAssert:
            with pytest.raises(ValueError, match="requires 2CTA"):
                select_bwd_config(inputs)
            continue
        if isinstance(expected, str):
            assert outcome(lambda: select_bwd_config(inputs)) == expected, inputs
        else:
            assert selected_settings(inputs, expected.keys()) == expected, inputs


_GQA8 = dict(num_heads=16, num_heads_kv=2, num_sms=148, max_seqlen_q=2048, max_seqlen_k=2048)


def test_pack_gqa_explicit_configs():
    inputs = make_inputs(**_GQA8, batch_size=8, total_q=8 * 2048, total_k=8 * 2048)
    config = select_bwd_config(inputs)
    assert config.pack_gqa
    validate_bwd_config(replace(config, pack_gqa=False), inputs)
    off = make_inputs(**_GQA8, requested_pack_gqa=False)
    assert not select_bwd_config(off).pack_gqa
    validate_bwd_config(replace(select_bwd_config(off), pack_gqa=True), off)
    # One long causal MQA sequence serializes its packed tiles (bwd_pack_gqa_heuristic).
    long_mqa = make_inputs(
        num_heads=16, num_heads_kv=1, num_sms=148, max_seqlen_q=8192, max_seqlen_k=8192,
        total_q=8192, total_k=8192, causal=True,
    )
    assert not select_bwd_config(long_mqa).pack_gqa


@pytest.mark.parametrize(
    "inputs, changes, error",
    [
        (make_inputs(), {"device_capacity": 9}, "targets SM9"),
        (make_inputs(), {"cluster_size": 1}, "cluster_size=2 exactly with 2CTA"),
        (make_inputs(), {"num_stages_Q": 2}, "2 Q stages"),
        (make_inputs(head_dim=64, head_dim_v=64), {"tile_n": 64}, "tile_n=64 is not supported"),
        (make_inputs(causal=True, deterministic=True), {"spt": False}, "spt=True"),
        (
            make_inputs(use_block_sparsity=True, sparse_block_size_q=128, sparse_block_size_kv=128),
            {"use_2cta_instrs": True, "cluster_size": 2, "num_stages_Q": 1},
            "2CTA instructions are not supported",
        ),
        (
            make_inputs(device_arch=90, num_heads_kv=4),
            {"num_threads": 512, "registers": Sm90BwdRegisters(160, 160, 32)},
            "need 2 MMA warp groups",
        ),
        (make_inputs(device_arch=90, causal=True, deterministic=True), {"spt": False}, "spt=True"),
        (make_inputs(device_arch=90, local=True), {"spt": True}, "spt=False"),
        (make_inputs(device_arch=90), {"num_stages_dO": 3}, "dO and PdS stages"),
        (make_inputs(device_arch=90), {"num_stages_PdS": 3}, "dO and PdS stages"),
        (make_inputs(device_arch=90), {"AtomLayoutMSdP": 3}, "SdP atom layout 3 must divide"),
        # Non-causal hdim 128 uses tile_m=80, which only dQ_swapAB puts on WGMMA's N axis.
        (make_inputs(device_arch=90), {"dQ_swapAB": False}, "SM90 dQ WGMMA .* cannot tile"),
        (
            make_inputs(),
            {"registers": Sm100BwdRegisters(160, 160, 104, 104, 24)},
            "exceeds 512",
        ),
        (make_inputs(device_arch=120), {"tile_m": 128}, "SM120 backward supports one configuration"),
        (make_inputs(), {"pack_gqa": True}, "pack_gqa needs"),
        (make_inputs(num_heads=24, num_heads_kv=8), {"pack_gqa": True}, "pack_gqa needs"),
        (make_inputs(**_GQA8, has_score_mod=True), {"pack_gqa": True}, "pack_gqa needs"),
        (make_inputs(**_GQA8, has_mask_mod=True), {"pack_gqa": True}, "pack_gqa needs"),
        (make_inputs(**_GQA8, has_learnable_sink=True), {"pack_gqa": True}, "pack_gqa needs"),
        (
            make_inputs(
                **_GQA8,
                use_block_sparsity=True,
                sparse_block_size_q=128,
                sparse_block_size_kv=256,
            ),
            {"pack_gqa": True},
            "pack_gqa needs",
        ),
        (
            make_inputs(**_GQA8, head_dim=256, head_dim_v=256),
            {"pack_gqa": True},
            "pack_gqa needs",
        ),
        (make_inputs(**_GQA8, device_arch=90), {"pack_gqa": True}, "2CTA or pack_gqa options"),
        (make_inputs(**_GQA8, device_arch=120), {"pack_gqa": True}, "2CTA or pack_gqa options"),
    ],
)
def test_invalid_configs_are_rejected(inputs, changes, error):
    config = select_bwd_config(inputs)
    with pytest.raises(ValueError, match=error):
        validate_bwd_config(replace(config, **changes), inputs)
