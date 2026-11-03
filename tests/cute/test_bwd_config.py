import itertools
from dataclasses import replace

import pytest

from flash_attn.cute.config import (
    BwdHeuristicInputs,
    Sm90BwdRegisters,
    Sm100BwdRegisters,
    bwd_pack_gqa_heuristic,
    bwd_q_subtile_factor,
    config_kv_subtile_factor,
    select_bwd_config,
    select_bwd_fold,
    validate_bwd_config,
)

_BASE_INPUTS = BwdHeuristicInputs(
    device_arch=100,
    head_dim=128,
    head_dim_v=128,
    num_heads=16,
    num_heads_kv=16,
    dtype="torch.bfloat16",
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
            pack_factor=1,
            head_loop=False,
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
            pack_factor=1,
            head_loop=False,
        )
        legacy["registers"] = legacy_sm90_kernel_registers(384, legacy["single"])
    else:
        is_hdim256 = head_dim == 256 and head_dim_v == 256
        n = 64 if is_hdim256 else 128

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
            if is_hdim256
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
        legacy = dict(
            m=128,
            n=n,
            dKV=False,
            dQ=False,
            NdKV=1,
            MdQ=1,
            cluster_size=cluster_size,
            use_2cta=use_2cta,
            kv_subtile=kv_subtile_factor,
            # The legacy sweep is MHA, which never packs.
            pack_factor=1,
            head_loop=False,
        )
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
    generic_sm100 = arch // 10 in (10, 11)
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
    spt="spt",
    registers="registers",
    pack_factor="pack_factor",
    head_loop="head_loop",
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
_ENVS = [  # (disable_2cta, requested_tile_n)
    (False, None),
    (True, None),
    (False, 64),
]


@pytest.mark.parametrize("device_arch", [90, 100, 103, 110, 120])
def test_selection_matches_the_legacy_policy(device_arch):
    # Dtype only selects fold_LSE_dPsum, which the legacy policy did not own; V_in_regs is
    # only forwarded on SM90.
    for (d, dv), mask, det, sparse, env in itertools.product(
        _HEAD_DIMS,
        ["none", "causal", "local"],
        [False, True],
        _SPARSE,
        _ENVS,
    ):
        use_block_sparsity, block_size, spt = sparse
        disable_2cta, tile_n = env
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
_HD64 = dict(head_dim=64, head_dim_v=64)
# Each problem change that keeps the SM100 LSE / dPsum fold off for a bf16 hdim-64 MHA call.
_FOLD_OFF_CHANGES = {
    "fp16": dict(_HD64, dtype="torch.float16"),
    "gqa": dict(_HD64, num_heads=16, num_heads_kv=4, num_sms=148, requested_pack_gqa=False),
    "deterministic": dict(_HD64, deterministic=True),
    "hdim128_1cta": dict(head_dim=128, head_dim_v=128, disable_2cta=True),
    "hdim_v128": dict(head_dim=64, head_dim_v=128),
    "block_sparse": dict(
        _HD64, use_block_sparsity=True, sparse_block_size_q=128, sparse_block_size_kv=128
    ),
    "score_mod_bwd": dict(_HD64, has_score_mod_bwd=True),
    "zero_softmax_scale": dict(_HD64, zero_softmax_scale=True),
    "softcap": dict(_HD64, has_softcap=True),
    "sm120": dict(_HD64, device_arch=120),
}


def test_pack_gqa_explicit_configs():
    inputs = make_inputs(**_GQA8, batch_size=8, total_q=8 * 2048, total_k=8 * 2048)
    config = select_bwd_config(inputs)
    assert config.pack_factor == 4 and not config.head_loop  # the largest pack the chooser takes
    validate_bwd_config(replace(config, pack_factor=1), inputs)
    validate_bwd_config(replace(config, pack_factor=8), inputs)
    off = make_inputs(**_GQA8, requested_pack_gqa=0)
    assert select_bwd_config(off).pack_factor == 1
    validate_bwd_config(replace(select_bwd_config(off), pack_factor=8), off)
    # A pack of 2 / 4 query heads per tile and the hdim 256 head loop over the rest by request.
    assert select_bwd_config(make_inputs(**_GQA8, requested_pack_gqa=4)).pack_factor == 4
    loop = make_inputs(
        **_GQA8, head_dim=256, head_dim_v=256, requested_pack_gqa=2,
        requested_head_loop=True,
    )
    assert (select_bwd_config(loop).pack_factor, select_bwd_config(loop).head_loop) == (2, True)
    validate_bwd_config(replace(select_bwd_config(loop), pack_factor=1, head_loop=True), loop)
    # One long causal MQA sequence serializes its packed tiles (bwd_pack_gqa_heuristic).
    long_mqa = make_inputs(
        num_heads=16, num_heads_kv=1, num_sms=148, max_seqlen_q=8192, max_seqlen_k=8192,
        total_q=8192, total_k=8192, causal=True,
    )
    assert select_bwd_config(long_mqa).pack_factor == 1


def test_fold_LSE_dPsum_is_selected_for_bf16_hdim64_mha():
    inputs = make_inputs(**_HD64)
    config = select_bwd_config(inputs)
    assert config.fold_LSE_dPsum
    # An explicit config may leave it off.
    validate_bwd_config(replace(config, fold_LSE_dPsum=False), inputs)


@pytest.mark.parametrize("changes", _FOLD_OFF_CHANGES.values(), ids=_FOLD_OFF_CHANGES.keys())
def test_fold_LSE_dPsum_is_off_outside_its_scope(changes):
    assert not select_bwd_config(make_inputs(**changes)).fold_LSE_dPsum


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
        (
            make_inputs(device_arch=90),
            {"num_stages_Q": 3, "num_stages_dO": 3, "num_stages_PdS": 3},
            "need 294912 bytes of shared memory",
        ),
        (make_inputs(device_arch=90), {"AtomLayoutMSdP": 3}, "SdP atom layout 3 must divide"),
        # Non-causal hdim 128 uses tile_m=80, which only dQ_swapAB puts on WGMMA's N axis.
        (make_inputs(device_arch=90), {"dQ_swapAB": False}, "SM90 dQ WGMMA .* cannot tile"),
        (
            make_inputs(),
            {"registers": Sm100BwdRegisters(160, 160, 104, 104, 24)},
            "exceeds 512",
        ),
        (make_inputs(device_arch=120), {"tile_m": 128}, "SM120 backward supports one configuration"),
        (make_inputs(), {"pack_factor": 2}, "query-head folds"),
        (make_inputs(**_GQA8), {"head_loop": True}, "head_loop needs head_dim 256"),
        (make_inputs(num_heads=24, num_heads_kv=8), {"pack_factor": 3}, "must divide"),
        (make_inputs(**_GQA8), {"pack_factor": 16}, "must divide"),
        (make_inputs(**_GQA8, has_score_mod=True), {"pack_factor": 8}, "query-head folds"),
        (make_inputs(**_GQA8, has_mask_mod=True), {"pack_factor": 8}, "query-head folds"),
        (make_inputs(**_GQA8, has_learnable_sink=True), {"pack_factor": 8}, "query-head folds"),
        (
            make_inputs(
                **_GQA8,
                use_block_sparsity=True,
                sparse_block_size_q=128,
                sparse_block_size_kv=256,
            ),
            {"pack_factor": 8},
            "query-head folds",
        ),
        (
            make_inputs(**_GQA8, head_dim=256, head_dim_v=256),
            {"tile_n": 128},
            "head_dim=256 needs tile_n=64",
        ),
        (make_inputs(**_GQA8, device_arch=90), {"pack_factor": 8}, "SM100 options"),
        (make_inputs(**_GQA8, device_arch=120), {"pack_factor": 8}, "SM100 options"),
        *(
            (make_inputs(**changes), {"fold_LSE_dPsum": True}, "fold_LSE_dPsum needs")
            for changes in _FOLD_OFF_CHANGES.values()
            if changes.get("device_arch") not in (90, 120)
        ),
        (make_inputs(**_HD64, device_arch=120), {"fold_LSE_dPsum": True}, "SM100 options"),
    ],
)
def test_invalid_configs_are_rejected(inputs, changes, error):
    config = select_bwd_config(inputs)
    with pytest.raises(ValueError, match=error):
        validate_bwd_config(replace(config, **changes), inputs)


def choose_fold(
    batch,
    heads,
    heads_kv,
    seqlen,
    causal,
    deterministic,
    varlen=False,
    cluster_size=2,
    m_block_size=128,
    n_block_size=128,
    head_dim=256,
    pack_gqa_override=None,
    head_loop_override=None,
):
    """select_bwd_fold on B200 (148 SMs, 74 clusters) for a shape of batch x seqlen tokens on
    the generic kernel; hdim 256 (the head loop) unless given, hdim 192 pairs with hdim_v 128."""
    inputs = BwdHeuristicInputs(
        dtype="torch.bfloat16",
        device_arch=100,
        head_dim=head_dim,
        head_dim_v=128 if head_dim == 192 else head_dim,
        num_heads=heads,
        num_heads_kv=heads_kv,
        num_sms=148,
        batch_size=batch,
        max_seqlen_q=seqlen,
        max_seqlen_k=seqlen,
        total_q=batch * seqlen,
        total_k=batch * seqlen,
        has_cu_seqlens=varlen,
        causal=causal,
        deterministic=deterministic,
        requested_pack_gqa=pack_gqa_override,
        requested_head_loop=head_loop_override,
    )
    return tuple(
        select_bwd_fold(
            inputs,
            tile_m=m_block_size,
            tile_n=n_block_size,
            cluster_size=cluster_size,
        )
    )


def fold_case(expect, batch, heads, heads_kv, seqlen, *, causal, deterministic, **kw):
    """One row of CHOOSE_FOLD_CASES: the choose_fold() inputs and the expected
    (pack_factor, head_loop), with an id like det-causal-mqa16/1-2k-b16-hd192."""
    inputs = dict(
        batch=batch,
        heads=heads,
        heads_kv=heads_kv,
        seqlen=seqlen,
        causal=causal,
        deterministic=deterministic,
        **kw,
    )
    parts = [
        "det" if deterministic else "nondet",
        "causal" if causal else "noncausal",
        f"{'mqa' if heads_kv == 1 else 'gqa'}{heads}/{heads_kv}",
        f"{seqlen // 1024}k" if seqlen >= 1024 else str(seqlen),
        f"b{batch}",
    ]
    if "head_dim" in kw:
        parts.append(f"hd{kw['head_dim']}")
    if "cluster_size" in kw:
        parts.append(f"cs{kw['cluster_size']}")
    if "m_block_size" in kw or "n_block_size" in kw:
        parts.append(f"{kw.get('m_block_size', 128)}x{kw.get('n_block_size', 128)}")
    if kw.get("varlen"):
        parts.append("varlen")
    if "pack_gqa_override" in kw:
        parts.append(f"pack{kw['pack_gqa_override']}")
    if "head_loop_override" in kw:
        parts.append(f"loop{int(kw['head_loop_override'])}")
    return pytest.param(inputs, expect, id="-".join(parts))


# Fold choice on B200 (148 SMs, 74 clusters), 32k tokens per shape: the cells of the
# chain probes (deterministic causal) and of the MQA probe (non-deterministic). Expected
# = (pack_factor, head_loop); the measurements behind each rule are in PR #3059.
# fmt: off
CHOOSE_FOLD_CASES = [
    # non-deterministic: pack 2 + loop whenever the heuristic takes the full fold; where
    # it refuses (long causal MQA: critical path / waves) a smaller fold instead of none
    fold_case((2, True), 16, 16, 1, 2048, causal=True, deterministic=False),
    fold_case((1, True), 16, 40, 8, 2048, causal=True, deterministic=False),
    fold_case((4, False), 8, 16, 1, 4096, causal=True, deterministic=False),
    # deterministic causal, hdim 256: the dQ chains pace the looped tiles, so few chains
    # take an unlooped pack 4 at 2k and pack 2 from 4k (at 1k the loop's 8 trips x 8
    # m_blocks = 64 still fit); GQA 24/4 (qh 6) keeps the loop at 2k; GQA 40/8 (odd qh)
    # has no unlooped fold, the loop stays
    fold_case((2, True), 32, 16, 1, 1024, causal=True, deterministic=True),
    fold_case((4, False), 16, 16, 1, 2048, causal=True, deterministic=True),
    fold_case((2, False), 8, 16, 1, 4096, causal=True, deterministic=True),
    fold_case((2, False), 4, 16, 1, 8192, causal=True, deterministic=True),
    fold_case((4, False), 16, 32, 2, 2048, causal=True, deterministic=True),
    fold_case((2, False), 4, 32, 2, 8192, causal=True, deterministic=True),
    fold_case((2, True), 16, 24, 4, 2048, causal=True, deterministic=True),
    fold_case((2, False), 4, 24, 4, 8192, causal=True, deterministic=True),
    fold_case((1, True), 16, 40, 8, 2048, causal=True, deterministic=True),
    # no head loop (hdim 192, 1 CTA, 64 x 96 tile), GQA 32/8: pack 4 while its chain
    # waste is small (2k), pack 2 at 4k / 8k, the heuristic's pack 1 at 16k;
    # non-deterministic keeps the largest accepted pack
    fold_case((4, False), 16, 32, 8, 2048, causal=True, deterministic=True,
              cluster_size=1, m_block_size=64,
              n_block_size=96, head_dim=192),
    fold_case((2, False), 8, 32, 8, 4096, causal=True, deterministic=True,
              cluster_size=1, m_block_size=64,
              n_block_size=96, head_dim=192),
    fold_case((2, False), 4, 32, 8, 8192, causal=True, deterministic=True,
              cluster_size=1, m_block_size=64,
              n_block_size=96, head_dim=192),
    fold_case((1, False), 2, 32, 8, 16384, causal=True, deterministic=True,
              cluster_size=1, m_block_size=64,
              n_block_size=96, head_dim=192),
    fold_case((4, False), 4, 32, 8, 8192, causal=True, deterministic=False,
              cluster_size=1, m_block_size=64,
              n_block_size=96, head_dim=192),
    # the loop stays while trips x m_blocks <= 64 (dense) / <= 96 (varlen): dense 32/2 and
    # MQA 16/1 at 1k and 64/8 at 2k (64) loop, 48/4 at 2k (96) does not; varlen 32/2 and
    # MQA 16/1 at 2k (128) do not, 48/4 at 2k and 24/4 at 4k (96) do, 64/8 at 4k (128)
    # does not
    fold_case((2, True), 32, 12, 2, 1024, causal=True, deterministic=True),
    fold_case((2, True), 16, 12, 2, 2048, causal=True, deterministic=True),
    fold_case((2, False), 8, 12, 2, 4096, causal=True, deterministic=True),
    fold_case((2, True), 32, 64, 8, 1024, causal=True, deterministic=True),
    fold_case((2, True), 16, 64, 8, 2048, causal=True, deterministic=True),
    fold_case((2, False), 8, 64, 8, 4096, causal=True, deterministic=True),
    fold_case((2, True), 32, 32, 2, 1024, causal=True, deterministic=True),
    fold_case((4, False), 16, 32, 2, 2048, causal=True, deterministic=True),
    fold_case((4, False), 16, 48, 4, 2048, causal=True, deterministic=True),
    fold_case((2, True), 32, 32, 2, 1024, causal=True, deterministic=True,
              varlen=True),
    fold_case((4, False), 16, 32, 2, 2048, causal=True, deterministic=True,
              varlen=True),
    fold_case((2, False), 8, 32, 2, 4096, causal=True, deterministic=True,
              varlen=True),
    fold_case((2, True), 32, 16, 1, 1024, causal=True, deterministic=True,
              varlen=True),
    fold_case((4, False), 16, 16, 1, 2048, causal=True, deterministic=True,
              varlen=True),
    fold_case((2, True), 16, 48, 4, 2048, causal=True, deterministic=True,
              varlen=True),
    fold_case((2, True), 8, 24, 4, 4096, causal=True, deterministic=True,
              varlen=True),
    fold_case((2, False), 8, 64, 8, 4096, causal=True, deterministic=True,
              varlen=True),
    fold_case((2, False), 16, 16, 8, 2048, causal=True, deterministic=True),
    # 2-CTA kernels drop pack 4 for pack 2 under deterministic causal once m_blocks *
    # head_dim > 4096 (hdim 192 past 2k, hdim 128 past 4k), also where the heuristic
    # refused pack 2 for its halved partial-pack gain; hdim 256 (GQA 48/4 8k) likewise;
    # the 1-CTA hdim 64 kernel keeps pack 4
    fold_case((4, False), 16, 48, 4, 2048, causal=True, deterministic=True,
              head_dim=192),
    fold_case((4, False), 16, 48, 4, 2048, causal=True, deterministic=True,
              head_dim=192, varlen=True),
    fold_case((4, False), 16, 64, 8, 2048, causal=True, deterministic=True,
              head_dim=192),
    fold_case((4, False), 16, 64, 8, 2048, causal=True, deterministic=True,
              head_dim=192, varlen=True),
    fold_case((2, False), 8, 48, 4, 4096, causal=True, deterministic=True,
              head_dim=192),
    fold_case((2, False), 8, 48, 4, 4096, causal=True, deterministic=True,
              head_dim=192, varlen=True),
    fold_case((2, False), 8, 64, 8, 4096, causal=True, deterministic=True,
              head_dim=192),
    fold_case((2, False), 8, 64, 8, 4096, causal=True, deterministic=True,
              head_dim=192, varlen=True),
    fold_case((2, False), 4, 48, 4, 8192, causal=True, deterministic=True,
              head_dim=192),
    fold_case((2, False), 4, 48, 4, 8192, causal=True, deterministic=True,
              head_dim=192, varlen=True),
    fold_case((2, False), 4, 64, 8, 8192, causal=True, deterministic=True,
              head_dim=192),
    fold_case((2, False), 4, 64, 8, 8192, causal=True, deterministic=True,
              head_dim=192, varlen=True),
    fold_case((1, False), 2, 48, 4, 16384, causal=True, deterministic=True,
              head_dim=192),
    fold_case((1, False), 2, 48, 4, 16384, causal=True, deterministic=True,
              head_dim=192, varlen=True),
    fold_case((1, False), 2, 64, 8, 16384, causal=True, deterministic=True,
              head_dim=192),
    fold_case((1, False), 2, 64, 8, 16384, causal=True, deterministic=True,
              head_dim=192, varlen=True),
    fold_case((4, False), 8, 48, 4, 4096, causal=True, deterministic=True,
              head_dim=128),
    fold_case((4, False), 8, 48, 4, 4096, causal=True, deterministic=True,
              head_dim=128, varlen=True),
    fold_case((4, False), 8, 64, 8, 4096, causal=True, deterministic=True,
              head_dim=128),
    fold_case((4, False), 8, 64, 8, 4096, causal=True, deterministic=True,
              head_dim=128, varlen=True),
    fold_case((2, False), 4, 48, 4, 8192, causal=True, deterministic=True,
              head_dim=128),
    fold_case((2, False), 4, 48, 4, 8192, causal=True, deterministic=True,
              head_dim=128, varlen=True),
    fold_case((2, False), 4, 64, 8, 8192, causal=True, deterministic=True,
              head_dim=128),
    fold_case((2, False), 4, 64, 8, 8192, causal=True, deterministic=True,
              head_dim=128, varlen=True),
    fold_case((2, False), 8, 32, 8, 4096, causal=True, deterministic=True,
              head_dim=192),
    fold_case((4, False), 8, 32, 8, 4096, causal=True, deterministic=True,
              head_dim=128),
    fold_case((2, False), 4, 48, 4, 8192, causal=True, deterministic=True),
    fold_case((4, False), 8, 64, 8, 4096, causal=True, deterministic=True,
              cluster_size=1, head_dim=64),
    fold_case((4, False), 4, 64, 8, 8192, causal=True, deterministic=True,
              cluster_size=1, head_dim=64),
    # off the deterministic path the hdim 192 kernel also steps a partial pack 4 down to
    # pack 2 from 4k on; the full pack 4 of GQA 32/8 stays, hdim 128 keeps pack 4, 2k is
    # below the rule, and non-causal is unchanged
    fold_case((2, False), 8, 48, 4, 4096, causal=True, deterministic=False,
              head_dim=192),
    fold_case((2, False), 8, 48, 4, 4096, causal=True, deterministic=False,
              head_dim=192, varlen=True),
    fold_case((2, False), 4, 48, 4, 8192, causal=True, deterministic=False,
              head_dim=192),
    fold_case((2, False), 4, 48, 4, 8192, causal=True, deterministic=False,
              head_dim=192, varlen=True),
    fold_case((2, False), 8, 64, 8, 4096, causal=True, deterministic=False,
              head_dim=192),
    fold_case((2, False), 8, 64, 8, 4096, causal=True, deterministic=False,
              head_dim=192, varlen=True),
    fold_case((2, False), 4, 64, 8, 8192, causal=True, deterministic=False,
              head_dim=192),
    fold_case((2, False), 4, 64, 8, 8192, causal=True, deterministic=False,
              head_dim=192, varlen=True),
    fold_case((2, False), 8, 32, 2, 4096, causal=True, deterministic=False,
              head_dim=192),
    fold_case((2, False), 8, 32, 2, 4096, causal=True, deterministic=False,
              head_dim=192, varlen=True),
    fold_case((2, False), 8, 16, 1, 4096, causal=True, deterministic=False,
              head_dim=192),
    fold_case((2, False), 8, 16, 1, 4096, causal=True, deterministic=False,
              head_dim=192, varlen=True),
    fold_case((4, False), 8, 32, 8, 4096, causal=True, deterministic=False,
              head_dim=192),
    fold_case((4, False), 8, 32, 8, 4096, causal=True, deterministic=False,
              head_dim=192, varlen=True),
    fold_case((4, False), 16, 48, 4, 2048, causal=True, deterministic=False,
              head_dim=192),
    fold_case((4, False), 16, 48, 4, 2048, causal=True, deterministic=False,
              head_dim=192, varlen=True),
    fold_case((4, False), 16, 16, 1, 2048, causal=True, deterministic=False,
              head_dim=192),
    fold_case((4, False), 16, 16, 1, 2048, causal=True, deterministic=False,
              head_dim=192, varlen=True),
    fold_case((4, False), 4, 48, 4, 8192, causal=True, deterministic=False,
              head_dim=128),
    fold_case((4, False), 8, 64, 8, 4096, causal=True, deterministic=False,
              head_dim=128),
    fold_case((4, False), 8, 48, 4, 4096, causal=False, deterministic=False,
              head_dim=192),
    # deterministic non-causal: no pace difference, the default fold
    fold_case((2, True), 16, 16, 1, 2048, causal=False, deterministic=True),
    # overrides: pack_gqa_override 0 off / 1 on / 2, 4 that pack; head_loop_override
    # forces or excludes the loop
    fold_case((1, True), 16, 24, 4, 2048, causal=True, deterministic=False,
              pack_gqa_override=0),
    fold_case((2, False), 16, 24, 4, 2048, causal=True, deterministic=False,
              head_loop_override=False),
    fold_case((2, True), 1, 16, 1, 8192, causal=True, deterministic=False,
              pack_gqa_override=1),
    fold_case((4, False), 2, 16, 4, 256, causal=True, deterministic=True,
              pack_gqa_override=4),
    fold_case((4, False), 16, 16, 1, 2048, causal=True, deterministic=False,
              pack_gqa_override=4),
    fold_case((2, True), 16, 16, 1, 2048, causal=True, deterministic=False,
              pack_gqa_override=2, head_loop_override=True),
    fold_case((2, False), 16, 8, 1, 2048, causal=True, deterministic=False,
              head_dim=64, pack_gqa_override=2),
    fold_case((1, False), 16, 40, 8, 2048, causal=True, deterministic=False,
              pack_gqa_override=2),
    # without the loop (not hdim 256): pack 4 / pack 2, largest accepted first; the full
    # pack 16 / 8 of MQA 16/1 loses to pack 4
    fold_case((4, False), 16, 16, 1, 2048, causal=True, deterministic=False,
              head_dim=64, cluster_size=1),
    fold_case((4, False), 16, 16, 1, 2048, causal=False, deterministic=False,
              head_dim=64, cluster_size=1),
    fold_case((2, False), 16, 24, 4, 2048, causal=True, deterministic=False,
              head_dim=64, cluster_size=1),
    fold_case((4, False), 8, 16, 1, 4096, causal=True, deterministic=False,
              head_dim=64, cluster_size=1),
    # a partial pack of four or more sub-groups keeps half the packing gain: MQA 16/1 8k
    # b4 causal stays unpacked, non-causal still packs; the full pack 4 of GQA 32/8 is
    # not partial; two sub-groups (GQA 64/8) keep the gain
    fold_case((1, False), 4, 16, 1, 8192, causal=True, deterministic=False,
              head_dim=64, cluster_size=1),
    fold_case((4, False), 4, 16, 1, 8192, causal=False, deterministic=False,
              head_dim=64, cluster_size=1),
    fold_case((4, False), 4, 32, 8, 8192, causal=True, deterministic=False,
              head_dim=64, cluster_size=1),
    fold_case((4, False), 4, 64, 8, 8192, causal=True, deterministic=False,
              head_dim=64, cluster_size=1),
    fold_case((1, False), 4, 16, 1, 8192, causal=True, deterministic=True,
              head_dim=64, cluster_size=1),
]
# fmt: on


@pytest.mark.parametrize("inputs,expected", CHOOSE_FOLD_CASES)
def test_bwd_choose_fold(inputs, expected):
    assert choose_fold(**inputs) == expected


def test_bwd_choose_fold_long_causal_mqa_partial_fold():
    """Where the heuristic refuses the full fold of a long causal MQA batch the chooser
    still takes a smaller fold instead of none."""
    assert choose_fold(4, 16, 1, 8192, causal=True, deterministic=False)[0] > 1


def test_bwd_pack_gqa_heuristic_partial_pack_keeps_half_the_gain():
    """hdim 128 MQA 16/1 at 8k causal: the full-pack model still packs 4 heads as a 4-way
    partial pack (kv_group 4), the halved gain leaves it unpacked; non-causal still packs."""
    inputs = BwdHeuristicInputs(
        dtype="torch.bfloat16",
        device_arch=100, head_dim=128, head_dim_v=128, num_heads=16, num_heads_kv=4,
        num_sms=148, batch_size=4, max_seqlen_q=8192, max_seqlen_k=8192,
        total_q=4 * 8192, total_k=4 * 8192, causal=True,
    )
    assert bwd_pack_gqa_heuristic(inputs, tile_m=128, tile_n=128, cluster_size=1)
    assert not bwd_pack_gqa_heuristic(
        inputs, tile_m=128, tile_n=128, cluster_size=1, kv_group=4
    )
    assert bwd_pack_gqa_heuristic(
        inputs._replace(causal=False), tile_m=128, tile_n=128, cluster_size=1, kv_group=4
    )


# Interface: these run on CPU with fake tensors and stop at the compile-cache lookup.
torch = pytest.importorskip("torch")


class _KeyLookedUp(Exception):
    pass


def _identity_score_mod(score, b, h, q_idx, kv_idx, aux_tensors):
    return score


@pytest.mark.parametrize(
    ("head_dim", "nheads", "score_mod", "can_pack"),
    [
        (128, 16, None, True),
        (256, 16, None, True),
        (128, 16, _identity_score_mod, False),
        # 6 query heads over 2 KV heads (ratio 3): no pack divides it and the head loop is
        # hdim 256 only, so there is no fold candidate to choose between
        (128, 6, None, False),
    ],
    ids=["pack_gqa", "hdim256", "score_mod", "odd_ratio"],
)
def test_cross_compiled_gqa_backward_needs_num_sms_only_to_pack(
    head_dim, nheads, score_mod, can_pack, monkeypatch
):
    """A GQA backward cross-compiled for SM100 without a matching GPU or FLASH_ATTENTION_NUM_SMS
    selects its config unless the pack_gqa heuristic, the only reader of the SM count, runs:
    it does not run without a fold candidate (score_mod, an odd qhead_per_kvhead off hdim
    256)."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    from flash_attn.cute import interface

    class LookupSpy:
        def get(self, key):
            raise _KeyLookedUp(key)

        __contains__ = get

    def no_matching_gpu(device_index, arch):
        raise RuntimeError("Cannot determine the target GPU's SM count")

    monkeypatch.setenv("FLASH_ATTENTION_ARCH", "sm_100")
    # Patched, not cached: the cached arch would leak into later GPU tests in this process.
    monkeypatch.setattr(interface, "_get_device_arch", lambda: 100)
    monkeypatch.setattr(interface, "get_num_sms_for_selection", no_matching_gpu)
    monkeypatch.setattr(interface, "_bwd_preprocess", lambda *args, **kwargs: None)
    monkeypatch.setattr(interface._flash_attn_bwd, "compile_cache", LookupSpy())
    with FakeTensorMode():
        q = torch.empty(2, 512, nheads, head_dim, dtype=torch.bfloat16)
        k = torch.empty(2, 512, 2 if nheads == 6 else 4, head_dim, dtype=torch.bfloat16)
        v = torch.empty_like(k)
        out = torch.empty_like(q)
        lse = torch.empty(2, nheads, 512, dtype=torch.float32)
        expected = (RuntimeError, "SM count") if can_pack else (_KeyLookedUp, None)
        with pytest.raises(expected[0], match=expected[1]):
            interface._flash_attn_bwd(
                q, k, v, out, torch.empty_like(out), lse, score_mod=score_mod,
                score_mod_bwd=score_mod,
            )
