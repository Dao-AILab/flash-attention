# Copyright (c) 2026, Tri Dao.
"""Golden compile-key snapshot for the FA4 interface refactor.

Records every compile-cache key the interface produces for a fixed matrix of
problems, without compiling anything: kernels launch under FakeTensorMode with
``cute.compile`` stubbed out, and every launcher's ``compile_cache`` is replaced
by a recording cache. Because the cache membership probe happens before kernel
construction, keys are captured even for cases that fail later.

The CuTeDSL singleton latches its target arch at first import (kernel
constructors read ``BaseDSL._get_dsl().get_arch_enum()`` and assert on it), so
simulating a non-native arch requires ``CUTE_DSL_ARCH`` to be set *before*
Python imports the DSL. The pytest entry therefore orchestrates one child
process per arch group; the child (``--child`` mode) runs that group's cases
and writes JSON records for the parent to merge.

Two modes:

  # compare against the checked-in snapshot (the default; no real compiles)
  .venv/bin/pytest tests/cute/test_compile_keys_golden.py

  # (re)capture the snapshot after an intentional key change
  GOLDEN_CAPTURE=1 .venv/bin/pytest tests/cute/test_compile_keys_golden.py

The snapshot pins, per case:
  - the exact key tuples (repr + pickle sha256) per cache, and
  - every kernel-constructor invocation with its bound arguments (the
    "decision-parity fixture": tile sizes, stage counts, scheduler flags etc.
    exactly as the current heuristics resolve them — callables recorded by
    content digest, tensors by shape). Decisions that never reach a
    constructor (e.g. the exact num_splits count) are pinned via the cache
    keys and the combine constructor instead.
or the raised error for cases that are rejected by design. A refactor must
either leave these byte-identical or regenerate the snapshot with the diff
itemized in the PR description.

Coverage deliberately not included (tracked, not silent): sparse MLA backward
(bwd_dsa/dq_dqv_gemm/dk_gemm launchers) and the standalone BlockSparsityKernel
(raw-dict cache, not reachable through the launchers). Block-sparse fwd/bwd
launches ARE covered, via hand-built BlockSparseTensorsTorch.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

SNAPSHOT_PATH = Path(__file__).parent / "data" / "golden_keys.json"
CAPTURE = os.getenv("GOLDEN_CAPTURE", "0") == "1"

# FLASH_ATTENTION_ARCH (kernel selection) -> CUTE_DSL_ARCH (DSL target, must be
# set before the child process imports the DSL).
DSL_ARCH = {
    "sm_80": "sm_80",
    "sm_90": "sm_90a",
    "sm_100": "sm_100a",
    "sm_103": "sm_103a",
    "sm_120": "sm_120a",
}

# Env vars that change key contents or selection; forced to defaults per case.
NEUTRAL_ENV = ("FA_CLC", "FA_DISABLE_2CTA", "FA_LOG_LEVEL",
               "FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED")


# ---------------------------------------------------------------------------
# Score/mask mods for the matrix. Module-level defs so hash_callable (source
# based) is stable as long as this file's text is stable.
# ---------------------------------------------------------------------------

def golden_score_mod(score, b, h, q_idx, kv_idx, aux_tensors=None, aux_scalars=None):
    return score * 2.0


def golden_score_mod_bwd(grad, score, b, h, q_idx, kv_idx, aux_tensors=None, aux_scalars=None):
    return grad * 2.0


def golden_mask_mod(b, h, q_idx, kv_idx, aux_tensors=None, aux_scalars=None):
    return q_idx >= kv_idx


class RecordingCache:
    """Dict-shaped cache that logs every membership probe and store."""

    def __init__(self):
        self.store: dict = {}
        self.keys_seen: list = []

    def __contains__(self, key):
        self._note(key)
        return key in self.store

    def __setitem__(self, key, value):
        self._note(key)
        self.store[key] = value

    def __getitem__(self, key):
        return self.store[key]

    def clear(self):
        self.store.clear()

    def _note(self, key):
        if not any(key == k for k in self.keys_seen):
            self.keys_seen.append(key)


def _stub_compile(*args, **kwargs):
    return lambda *a, **k: None


# Every kernel class interface.py instantiates; their __init__ args are the
# resolved decisions we pin.
KERNEL_CLASS_NAMES = (
    "FlashAttentionForwardSm80", "FlashAttentionForwardSm90",
    "FlashAttentionForwardSm100", "FlashAttentionForwardSm120",
    "FlashAttentionBackwardPreprocess", "FlashAttentionBackwardSm80",
    "FlashAttentionBackwardSm90", "FlashAttentionBackwardSm100",
    "FlashAttentionBackwardSm120", "FlashAttentionBackwardPostprocess",
    "FlashAttentionForwardCombine", "FlashAttentionMLAForwardSm100",
    "FlashAttentionSparseMLABackwardSm100", "dQdQvGemmKernel", "dKGemmKernel",
    "BlackwellFusedMultiHeadAttentionForward",
    "BlackwellFusedMultiHeadAttentionBackward",
)


def _decision_value(v):
    """Deterministic, JSON-safe rendering of one constructor argument.

    Recurses into containers because pass-through ``*args``/``**kwargs`` bind
    as raw tuples/dicts whose reprs would embed callable memory addresses.
    """
    import torch

    if isinstance(v, torch.Tensor):
        return f"tensor{tuple(v.shape)}"
    if callable(v) and not isinstance(v, type):
        from flash_attn.cute.utils import hash_callable

        return f"callable:{hash_callable(v)[:12]}"
    if isinstance(v, (tuple, list)):
        return [_decision_value(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _decision_value(x) for k, x in sorted(v.items())}
    return repr(v)


def _patch_ctors(fa, ctor_log):
    """Patch kernel __init__s to log bound args; returns an undo callable.

    Logging happens before the original __init__ runs, so constructions that
    raise (by-design rejections) still record their decisions. Classes with
    pass-through ``*args`` (e.g. FlashAttentionForwardSm90) log their own
    partial bind, and their base class logs the fully named one; both entries
    carry the concrete class name.
    """
    import inspect

    undos = []
    for cls_name in KERNEL_CLASS_NAMES:
        cls = getattr(fa, cls_name)
        if "__init__" not in vars(cls):  # inherits __init__; base patch covers it
            continue
        orig = vars(cls)["__init__"]
        sig = inspect.signature(orig)

        def wrapped(self, *args, __orig=orig, __sig=sig, __declared=cls_name, **kwargs):
            try:
                bound = __sig.bind(self, *args, **kwargs)
                bound.apply_defaults()
                rendered = {
                    n: _decision_value(v) for n, v in bound.arguments.items()
                    if n != "self"
                }
            except TypeError:
                rendered = {"<bind-failed>": f"args={len(args)} kwargs={sorted(kwargs)}"}
            ctor_log.append({"cls": type(self).__name__, "declared_in": __declared,
                             "args": rendered})
            return __orig(self, *args, **kwargs)

        setattr(cls, "__init__", wrapped)
        undos.append((cls, orig))
    return lambda: [setattr(c, "__init__", o) for c, o in undos]


def _launchers(fa):
    return (
        ("fwd", fa._flash_attn_fwd),
        ("bwd", fa._flash_attn_bwd),
        ("bwd_pre", fa._bwd_preprocess),
        ("bwd_post", fa._bwd_postprocess_convert),
        ("fwd_combine", fa._flash_attn_fwd_combine),
        ("bwd_dsa", fa._flash_attn_bwd_sparse_mla),
        ("dq_dqv_gemm", fa._sparse_mla_dq_dqv),
        ("dk_gemm", fa._sparse_mla_dk),
    )


@contextmanager
def _golden_ctx(arch: str):
    """Simulated arch + recording caches + stubbed cute.compile, all restored."""
    import flash_attn.cute.interface as fa

    launchers = _launchers(fa)
    saved_env = {k: os.environ.get(k) for k in NEUTRAL_ENV + ("FLASH_ATTENTION_ARCH",)}
    saved_caches = {name: fn.compile_cache for name, fn in launchers}
    saved_compile = fa.cute.compile
    recorders = {name: RecordingCache() for name, _ in launchers}
    ctor_log: list = []
    undo_ctors = lambda: None  # noqa: E731
    try:
        for k in NEUTRAL_ENV:
            os.environ.pop(k, None)
        os.environ["FLASH_ATTENTION_ARCH"] = arch
        fa._get_device_arch.cache_clear()
        for name, fn in launchers:
            fn.compile_cache = recorders[name]
        fa.cute.compile = _stub_compile
        undo_ctors = _patch_ctors(fa, ctor_log)
        yield recorders, ctor_log
    finally:
        undo_ctors()
        fa.cute.compile = saved_compile
        for name, fn in launchers:
            fn.compile_cache = saved_caches[name]
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        fa._get_device_arch.cache_clear()


# ---------------------------------------------------------------------------
# Case matrix
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Case:
    name: str
    arch: str                       # FLASH_ATTENTION_ARCH value, e.g. "sm_100"
    fn: str = "fwd"                 # "fwd" | "bwd"  (bwd runs fwd first for lse)
    b: int = 2
    sq: int = 512
    sk: int = 512
    h: int = 8
    h_kv: Optional[int] = None      # None -> h
    d: int = 128
    dv: Optional[int] = None        # None -> d
    dtype: str = "bf16"             # bf16 | fp16 | fp8e4m3
    causal: bool = False
    window: Optional[tuple] = None  # (left, right) for local attention
    softcap: Optional[float] = None
    varlen: bool = False            # cu_seqlens on q and k (packed layout)
    seqused: bool = False           # seqused_q/k with batched layout
    paged: Optional[int] = None     # page size
    splits: int = 1
    pack_gqa: Optional[bool] = None
    sink: bool = False
    qv: bool = False                # MLA absorbed path (d=64, dv=512)
    descale: bool = False           # fp8 descale tensors
    score_mod: bool = False
    mask_mod: bool = False
    aux: bool = False
    block_sparse: bool = False      # hand-built BlockSparseTensorsTorch
    deterministic: bool = False     # bwd only
    fwd_kwargs: dict = field(default_factory=dict, hash=False, compare=False)
    bwd_kwargs: dict = field(default_factory=dict, hash=False, compare=False)


CASES = [
    # ---- SM80 forward ----
    Case("fwd.sm80.bf16.dense.d64", "sm_80", d=64),
    Case("fwd.sm80.bf16.dense.causal.d128", "sm_80", causal=True),
    Case("fwd.sm80.fp16.dense.causal.gqa4.d128", "sm_80", dtype="fp16", causal=True, h_kv=2),
    Case("fwd.sm80.bf16.varlen.d64", "sm_80", d=64, varlen=True),
    # ---- SM90 forward ----
    Case("fwd.sm90.bf16.dense.d64", "sm_90", d=64),
    Case("fwd.sm90.bf16.dense.causal.d128", "sm_90", causal=True),
    Case("fwd.sm90.bf16.dense.local.d128", "sm_90", window=(128, 0)),
    Case("fwd.sm90.fp16.dense.causal.d192v128", "sm_90", dtype="fp16", causal=True, d=192, dv=128),
    Case("fwd.sm90.bf16.dense.causal.gqa8.d96", "sm_90", causal=True, h_kv=1, d=96),
    Case("fwd.sm90.bf16.varlen.causal.d128", "sm_90", causal=True, varlen=True),
    Case("fwd.sm90.bf16.batched.seqused.d64", "sm_90", d=64, seqused=True),
    Case("fwd.sm90.bf16.dense.score_mod.d128", "sm_90", score_mod=True),
    Case("fwd.sm90.bf16.dense.mask_mod.d128", "sm_90", mask_mod=True),
    Case("fwd.sm90.bf16.dense.softcap.d128", "sm_90", softcap=30.0),
    Case("fwd.sm90.bf16.dense.causal.d128.tile_override", "sm_90", causal=True,
         fwd_kwargs={"tile_mn": (128, 128)}),
    Case("fwd.sm90.bf16.dense.aux_score_mod.d64", "sm_90", d=64, score_mod=True, aux=True),
    # ---- SM100 forward ----
    Case("fwd.sm100.bf16.dense.d64", "sm_100", d=64),
    Case("fwd.sm100.bf16.dense.causal.d128", "sm_100", causal=True),
    Case("fwd.sm100.bf16.dense.causal.gqa8.d128", "sm_100", causal=True, h_kv=1),
    Case("fwd.sm100.bf16.dense.causal.gqa8.d128.nopack", "sm_100", causal=True, h_kv=1,
         pack_gqa=False),
    Case("fwd.sm100.bf16.dense.local.d64", "sm_100", d=64, window=(64, 64)),
    Case("fwd.sm100.bf16.varlen.causal.d128", "sm_100", causal=True, varlen=True),
    Case("fwd.sm100.bf16.batched.seqused.d128", "sm_100", seqused=True),
    Case("fwd.sm100.bf16.dense.d192v128", "sm_100", d=192, dv=128),
    Case("fwd.sm100.bf16.dense.causal.d256", "sm_100", causal=True, d=256),
    Case("fwd.sm100.bf16.dense.split3.d128", "sm_100", splits=3, sq=64, sk=8192),
    Case("fwd.sm100.bf16.dense.sink.d128", "sm_100", sink=True),
    Case("fwd.sm100.fp8e4m3.dense.d128", "sm_100", dtype="fp8e4m3"),
    Case("fwd.sm100.fp8e4m3.dense.descale.d128", "sm_100", dtype="fp8e4m3", descale=True),
    Case("fwd.sm100.bf16.paged.decode.d128", "sm_100", sq=1, paged=128),
    Case("fwd.sm100.bf16.dense.score_mod.d128", "sm_100", score_mod=True),
    Case("fwd.sm100.bf16.dense.mask_mod.d128", "sm_100", mask_mod=True),
    Case("fwd.sm100.bf16.dense.qv.d64v512", "sm_100", d=64, dv=512, qv=True),
    Case("fwd.sm100.bf16.dense.short.d64", "sm_100", d=64, sq=128, sk=128),
    Case("fwd.sm100.bf16.dense.noncausal.d128.sq8k", "sm_100", sq=8192, sk=8192),
    # ---- SM103 forward (arch is a key component) ----
    Case("fwd.sm103.bf16.dense.causal.d128", "sm_103", causal=True),
    Case("fwd.sm103.bf16.varlen.causal.d128", "sm_103", causal=True, varlen=True),
    # ---- block-sparse (hand-built tensors; the launcher-reachable coverage) ----
    Case("fwd.sm100.bf16.block_sparse.d128", "sm_100", block_sparse=True, mask_mod=True),
    Case("bwd.sm100.bf16.block_sparse.causal.d128", "sm_100", fn="bwd",
         causal=True, block_sparse=True),
    # ---- SM120 forward ----
    Case("fwd.sm120.bf16.dense.d64", "sm_120", d=64),
    Case("fwd.sm120.bf16.dense.causal.d128", "sm_120", causal=True),
    Case("fwd.sm120.bf16.dense.split3.d128", "sm_120", splits=3),  # raises by design
    # ---- SM90 backward ----
    Case("bwd.sm90.bf16.dense.causal.d64", "sm_90", fn="bwd", d=64, causal=True),
    Case("bwd.sm90.bf16.dense.d128", "sm_90", fn="bwd"),
    Case("bwd.sm90.bf16.dense.causal.d96", "sm_90", fn="bwd", d=96, causal=True),
    Case("bwd.sm90.bf16.varlen.causal.d128", "sm_90", fn="bwd", causal=True, varlen=True),
    Case("bwd.sm90.bf16.dense.deterministic.d64", "sm_90", fn="bwd", d=64, deterministic=True),
    Case("bwd.sm90.bf16.dense.score_mod.d64", "sm_90", fn="bwd", d=64, score_mod=True),
    # Documents today's silent discard: identical bwd key to bwd.sm90.bf16.dense.d128.
    Case("bwd.sm90.bf16.dense.d128.knob_override", "sm_90", fn="bwd",
         bwd_kwargs={"m_block_size": 128, "n_block_size": 64, "num_threads": 512,
                     "SdP_swapAB": True}),
    # ---- SM100/103 backward ----
    Case("bwd.sm100.bf16.dense.causal.d128", "sm_100", fn="bwd", causal=True),
    Case("bwd.sm100.bf16.dense.d64", "sm_100", fn="bwd", d=64),
    Case("bwd.sm100.bf16.varlen.causal.d128", "sm_100", fn="bwd", causal=True, varlen=True),
    Case("bwd.sm100.bf16.dense.deterministic.d128", "sm_100", fn="bwd", deterministic=True),
    Case("bwd.sm103.bf16.dense.causal.d128", "sm_103", fn="bwd", causal=True),
    # ---- SM120 backward ----
    Case("bwd.sm120.bf16.dense.d64", "sm_120", fn="bwd", d=64),
]


def _build_tensors(case: Case):
    """Fake q/k/v (+ trimmings) for one case. Values never matter, shapes do."""
    import torch

    dtypes = {"bf16": torch.bfloat16, "fp16": torch.float16,
              "fp8e4m3": torch.float8_e4m3fn}
    dtype = dtypes[case.dtype]
    h_kv = case.h_kv if case.h_kv is not None else case.h
    dv = case.dv if case.dv is not None else case.d
    dev = "cuda"

    def t(*shape, dt=dtype):
        return torch.empty(*shape, device=dev, dtype=dt)

    kw: dict[str, Any] = {}
    if case.varlen:
        total_q, total_k = case.b * case.sq, case.b * case.sk
        kw["q"] = t(total_q, case.h, case.d)
        kw["k"] = t(total_k, h_kv, case.d)
        kw["v"] = t(total_k, h_kv, dv)
        kw["cu_seqlens_q"] = t(case.b + 1, dt=torch.int32)
        kw["cu_seqlens_k"] = t(case.b + 1, dt=torch.int32)
        kw["max_seqlen_q"], kw["max_seqlen_k"] = case.sq, case.sk
    elif case.paged is not None:
        num_pages = case.b * (case.sk // case.paged)
        kw["q"] = t(case.b, case.sq, case.h, case.d)
        kw["k"] = t(num_pages, case.paged, h_kv, case.d)
        kw["v"] = t(num_pages, case.paged, h_kv, dv)
        kw["page_table"] = t(case.b, case.sk // case.paged, dt=torch.int32)
        kw["seqused_k"] = t(case.b, dt=torch.int32)
        kw["max_seqlen_k"] = case.sk
    else:
        kw["q"] = t(case.b, case.sq, case.h, case.d)
        kw["k"] = t(case.b, case.sk, h_kv, case.d)
        kw["v"] = t(case.b, case.sk, h_kv, dv)
        if case.seqused:
            kw["seqused_q"] = t(case.b, dt=torch.int32)
            kw["seqused_k"] = t(case.b, dt=torch.int32)

    if case.qv:
        kw["qv"] = t(*kw["q"].shape[:-1], dv)
    if case.causal:
        kw["causal"] = True
    if case.window is not None:
        kw["window_size_left"], kw["window_size_right"] = case.window
    if case.softcap is not None:
        kw["softcap"] = case.softcap
    if case.splits != 1:
        kw["num_splits"] = case.splits
    if case.pack_gqa is not None:
        kw["pack_gqa"] = case.pack_gqa
    if case.sink:
        kw["learnable_sink"] = t(case.h, dt=torch.bfloat16)
    if case.descale:
        one = torch.ones(case.b, h_kv, device=dev, dtype=torch.float32)
        kw["q_descale"], kw["k_descale"], kw["v_descale"] = one, one.clone(), one.clone()
    if case.score_mod:
        kw["score_mod"] = golden_score_mod
    if case.mask_mod:
        kw["mask_mod"] = golden_mask_mod
    if case.aux:
        kw["aux_tensors"] = [t(case.b, case.sq, dt=torch.float32)]
    if case.block_sparse:
        from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch

        # fwd wants sparse-Q blocks at q_stage*tile_m=256 granularity; bwd wants
        # Q-direction tensors at its own 128 tile.
        bs_q = 128 if case.fn == "bwd" else 256
        rows, cols = case.sq // bs_q, case.sk // 128
        kw["block_sparse_tensors"] = BlockSparseTensorsTorch(
            mask_block_cnt=t(case.b, case.h, rows, dt=torch.int32),
            mask_block_idx=t(case.b, case.h, rows, cols, dt=torch.int32),
            full_block_cnt=t(case.b, case.h, rows, dt=torch.int32),
            full_block_idx=t(case.b, case.h, rows, cols, dt=torch.int32),
            block_size=(bs_q, 128),
        )
    return kw


def _run_case(case: Case) -> dict:
    """Execute one case under the golden context; return its snapshot record."""
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode

    import flash_attn.cute.interface as fa

    record: dict[str, Any] = {"arch": case.arch, "error": None, "keys": {}, "ctors": []}
    with _golden_ctx(case.arch) as (recorders, ctor_log):
        try:
            with FakeTensorMode():
                kw = _build_tensors(case)
                if case.fn == "fwd":
                    fa._flash_attn_fwd(**kw, **case.fwd_kwargs, return_lse=True)
                else:
                    fwd_only = dict(kw)
                    fwd_only.pop("score_mod", None)  # fwd half of bwd cases runs plain
                    fwd_only.pop("block_sparse_tensors", None)  # bwd-oriented tensors
                    out, lse, *_ = fa._flash_attn_fwd(**fwd_only, return_lse=True)
                    dout = torch.empty_like(out)
                    bwd_kw = {
                        k: v for k, v in kw.items()
                        if k in ("q", "k", "v", "causal", "window_size_left",
                                 "window_size_right", "cu_seqlens_q", "cu_seqlens_k",
                                 "seqused_q", "seqused_k", "max_seqlen_q", "max_seqlen_k",
                                 "block_sparse_tensors")
                    }
                    if case.score_mod:
                        bwd_kw["score_mod"] = golden_score_mod
                        bwd_kw["score_mod_bwd"] = golden_score_mod_bwd
                    if case.deterministic:
                        bwd_kw["deterministic"] = True
                    fa._flash_attn_bwd(out=out, dout=dout, lse=lse, **bwd_kw,
                                       **case.bwd_kwargs)
        except Exception as e:  # noqa: BLE001 — by-design rejections are data here
            record["error"] = {"type": type(e).__name__, "message": str(e)[:200]}
        for name, rec in recorders.items():
            if rec.keys_seen:
                record["keys"][name] = sorted(
                    ({"repr": repr(k),
                      "sha256": hashlib.sha256(pickle.dumps(k)).hexdigest()}
                     for k in rec.keys_seen),
                    key=lambda r: r["repr"],
                )
        record["ctors"] = ctor_log
    return record


# ---------------------------------------------------------------------------
# Parent/child orchestration
# ---------------------------------------------------------------------------

def _child_main(arch: str, outfile: str) -> None:
    """Run all cases of one arch group; env (CUTE_DSL_ARCH) was set by parent."""
    results = {c.name: _run_case(c) for c in CASES if c.arch == arch}
    Path(outfile).write_text(json.dumps(results))


def _collect_all_cases() -> dict:
    """Spawn one child per arch group so the DSL initializes per target arch."""
    merged: dict[str, Any] = {}
    repo_root = Path(__file__).parents[2]
    with tempfile.TemporaryDirectory() as tmp:
        for arch in sorted({c.arch for c in CASES}):
            outfile = str(Path(tmp) / f"{arch}.json")
            env = dict(os.environ)
            for k in NEUTRAL_ENV:
                env.pop(k, None)
            env["FLASH_ATTENTION_ARCH"] = arch
            env["CUTE_DSL_ARCH"] = DSL_ARCH[arch]
            proc = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--child", arch, outfile],
                env=env, cwd=tmp, capture_output=True, text=True, timeout=600,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"golden-key child for {arch} failed:\n{proc.stderr[-3000:]}"
                )
            merged.update(json.loads(Path(outfile).read_text()))
    return merged, repo_root


def _snapshot() -> dict:
    import torch

    cases, repo_root = _collect_all_cases()
    return {
        "meta": {
            "commit": subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, cwd=repo_root,
            ).stdout.strip(),
            "device": torch.cuda.get_device_name(0),
            "device_capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
            "num_cases": len(cases),
            "dropped_coverage": [
                "sparse MLA backward (bwd_dsa/dq_dqv_gemm/dk_gemm launchers)",
                "BlockSparsityKernel (raw-dict cache, not launcher-reachable)",
            ],
        },
        "cases": cases,
    }


def test_golden_compile_keys():
    """Compare (or capture) the compile-key snapshot for the full case matrix."""
    import pytest

    got = _snapshot()
    if CAPTURE:
        SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT_PATH.write_text(json.dumps(got, indent=1, sort_keys=True) + "\n")
        pytest.skip(f"captured {len(got['cases'])} cases to {SNAPSHOT_PATH}")
    assert SNAPSHOT_PATH.exists(), (
        f"no snapshot at {SNAPSHOT_PATH}; run with GOLDEN_CAPTURE=1 to create it"
    )
    want = json.loads(SNAPSHOT_PATH.read_text())
    problems = []
    for name, want_rec in want["cases"].items():
        got_rec = got["cases"].get(name)
        if got_rec is None:
            problems.append(f"{name}: case missing from current matrix")
            continue
        if (got_rec["error"] is None) != (want_rec["error"] is None):
            problems.append(f"{name}: error status changed: "
                            f"{want_rec['error']} -> {got_rec['error']}")
            continue
        if got_rec["error"] is not None:
            if got_rec["error"]["type"] != want_rec["error"]["type"]:
                problems.append(f"{name}: error type changed: "
                                f"{want_rec['error']['type']} -> {got_rec['error']['type']}")
            continue
        for cache in sorted(set(want_rec["keys"]) | set(got_rec["keys"])):
            w = want_rec["keys"].get(cache, [])
            g = got_rec["keys"].get(cache, [])
            if [r["sha256"] for r in w] != [r["sha256"] for r in g]:
                w_reprs = {r["repr"] for r in w}
                g_reprs = {r["repr"] for r in g}
                detail = "; ".join(
                    [f"only-before: {r}" for r in sorted(w_reprs - g_reprs)]
                    + [f"only-after: {r}" for r in sorted(g_reprs - w_reprs)]
                ) or "same reprs, different pickle bytes (pickle instability!)"
                problems.append(f"{name}/{cache}: {detail}")
        if got_rec.get("ctors") != want_rec.get("ctors"):
            w_ct, g_ct = want_rec.get("ctors", []), got_rec.get("ctors", [])
            if len(w_ct) != len(g_ct):
                problems.append(
                    f"{name}/ctors: construction count {len(w_ct)} -> {len(g_ct)}"
                )
            else:
                for i, (wc, gc) in enumerate(zip(w_ct, g_ct)):
                    if wc != gc:
                        diff = {k: (wc["args"].get(k), gc["args"].get(k))
                                for k in set(wc["args"]) | set(gc["args"])
                                if wc["args"].get(k) != gc["args"].get(k)}
                        problems.append(f"{name}/ctors[{i}] {wc['cls']}: {diff}")
    for name in got["cases"]:
        if name not in want["cases"]:
            problems.append(f"{name}: new case not in snapshot (recapture to add)")
    assert not problems, (
        f"{len(problems)} golden-key mismatches:\n" + "\n".join(problems[:40])
    )


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--child":
        _child_main(sys.argv[2], sys.argv[3])
    else:
        sys.exit(f"usage: {sys.argv[0]} --child <arch> <outfile>")
