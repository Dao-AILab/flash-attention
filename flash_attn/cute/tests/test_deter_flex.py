import dis
import hashlib
import argparse
import random
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

import cutlass
import torch
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

_REPO_ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / "flash_attn" / "cute" / "interface.py").exists()
)
sys.path.insert(0, str(_REPO_ROOT))
from flash_attn.cute import interface as fa_interface  # noqa: E402
from flash_attn.cute.block_sparsity import (  # noqa: E402
    BlockSparseTensorsTorch,
    compute_dq_write_order_from_block_mask,
)
from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd  # noqa: E402


class _BoundedMaskModCache:
    def __init__(self, maxsize: int):
        self.maxsize = maxsize
        self._items: list[Tuple[Callable, Tuple[Callable, Optional[list]]]] = []

    def find(self, key: Callable) -> Optional[Tuple[Callable, Optional[list]]]:
        for i, (cached_key, value) in enumerate(self._items):
            if cached_key is key:
                self._items.append(self._items.pop(i))
                return value
        return None

    def add(self, key: Callable, value: Tuple[Callable, Optional[list]]) -> None:
        self._items.append((key, value))
        if len(self._items) > self.maxsize:
            self._items.pop(0)


WRAPPED_MASK_MOD_CACHE = _BoundedMaskModCache(maxsize=10)
BWD_TIMING_ENABLED = False
BWD_TIMING_EVENTS: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []


def enable_bwd_timing(enabled: bool, clear: bool = True):
    global BWD_TIMING_ENABLED
    BWD_TIMING_ENABLED = enabled
    if clear:
        BWD_TIMING_EVENTS.clear()


def pop_bwd_times_ms() -> list[float]:
    torch.cuda.synchronize()
    times = [start.elapsed_time(end) for start, end in BWD_TIMING_EVENTS]
    BWD_TIMING_EVENTS.clear()
    return times


def _make_cell(value):
    return (lambda: value).__closure__[0]


class _CuteIndexable:
    def __init__(self, cute_tensor):
        self._t = cute_tensor
        self._dtype = cute_tensor.element_type

    def __getitem__(self, ssa_idx):
        from torch._inductor.codegen.cutedsl._cutedsl_utils import (
            result_to_ssa,
            ssa_to_indexable,
        )

        idx = ssa_to_indexable(ssa_idx, cutlass.Int32)
        return result_to_ssa(self._t[idx], self._dtype)


def _get_tensor_idx(aux_tensors: list, tensor: torch.Tensor) -> int:
    for i, t in enumerate(aux_tensors):
        if t is tensor:
            return i
    aux_tensors.append(tensor)
    return len(aux_tensors) - 1


def _get_rebuilt_mask_mod(user_mm: Callable, aux_tensors: list, hash_items: list):
    code = user_mm.__code__
    closures = user_mm.__closure__ or ()
    hash_items.append(("fn", user_mm.__name__, code, user_mm.__defaults__))

    def get_closure_builder(closure):
        if isinstance(closure, torch.Tensor):
            idx = _get_tensor_idx(aux_tensors, closure)
            hash_items.append(("tensor", idx, str(closure.dtype), closure.dim()))
            return lambda proxies: proxies[idx]

        if callable(closure) and hasattr(closure, "__code__"):
            child = _get_rebuilt_mask_mod(closure, aux_tensors, hash_items)

            def make_child(proxies):
                def child_mask(b_idx, h_idx, q_idx, kv_idx):
                    return child(b_idx, h_idx, q_idx, kv_idx, proxies)

                return child_mask

            return make_child

        if isinstance(closure, tuple):
            hash_items.append(("tuple", len(closure)))
            builders = tuple(get_closure_builder(v) for v in closure)
            return lambda proxies: tuple(builder(proxies) for builder in builders)

        hash_items.append(("const", type(closure).__name__, repr(closure)))
        return lambda proxies: closure

    closure_builders = tuple(get_closure_builder(closure.cell_contents) for closure in closures)

    def wrapped(b_idx, h_idx, q_idx, kv_idx, proxies):
        new_closures = tuple(_make_cell(builder(proxies)) for builder in closure_builders)
        rebuilt = types.FunctionType(
            code,
            user_mm.__globals__,
            user_mm.__name__,
            user_mm.__defaults__,
            new_closures,
        )
        return rebuilt(b_idx, h_idx, q_idx, kv_idx)

    return wrapped


def _wrap_mask_mod(block_mask: BlockMask) -> Tuple[Callable, Optional[list]]:
    user_mm = block_mask.mask_mod
    cached = WRAPPED_MASK_MOD_CACHE.find(user_mm)
    if cached is not None:
        return cached

    aux_tensors: list = []
    hash_items: list = []
    mask_mod = _get_rebuilt_mask_mod(user_mm, aux_tensors, hash_items)

    def adapter(b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        proxies = [_CuteIndexable(t) for t in aux_tensors] if aux_tensors else []
        return mask_mod(b_idx, h_idx, q_idx, kv_idx, proxies)

    adapter.__cute_hash__ = "fa4_flex:" + hashlib.sha256(repr(hash_items).encode()).hexdigest()
    result = (adapter, aux_tensors if aux_tensors else None)
    WRAPPED_MASK_MOD_CACHE.add(user_mm, result)
    return result


class FlexAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_mask: BlockMask,
        softmax_scale: Optional[float],
        return_lse: bool,
        deterministic: bool,
    ):
        block_size = (int(block_mask.BLOCK_SIZE[0]), int(block_mask.BLOCK_SIZE[1]))
        mask_mod, aux_tensors = _wrap_mask_mod(block_mask)
        bst_fwd = BlockSparseTensorsTorch(
            mask_block_cnt=block_mask.kv_num_blocks,
            mask_block_idx=block_mask.kv_indices,
            full_block_cnt=block_mask.full_kv_num_blocks,
            full_block_idx=block_mask.full_kv_indices,
            block_size=block_size,
        )
        out_t, lse = _flash_attn_fwd(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            softmax_scale=softmax_scale,
            mask_mod=mask_mod,
            aux_tensors=aux_tensors,
            block_sparse_tensors=bst_fwd,
            return_lse=True,
        )
        out = out_t.transpose(1, 2)

        ctx.save_for_backward(
            q,
            k,
            v,
            out,
            lse,
            block_mask.q_num_blocks,
            block_mask.q_indices,
            block_mask.full_q_num_blocks,
            block_mask.full_q_indices,
        )
        ctx.softmax_scale = softmax_scale
        ctx.block_size = block_size
        ctx.mask_mod = mask_mod
        ctx.aux_tensors = aux_tensors
        ctx.return_lse = return_lse
        ctx.deterministic = deterministic
        ctx.dq_write_order = (
            compute_dq_write_order_from_block_mask(block_mask, spt=False) if deterministic else None
        )
        ctx.set_materialize_grads(False)
        return (out, lse) if return_lse else out

    @staticmethod
    def backward(ctx, *grads):
        if ctx.return_lse:
            dout, dlse = grads
        else:
            dout, dlse = grads[0], None
        (
            q,
            k,
            v,
            out,
            lse,
            q_num_blocks,
            q_indices,
            full_q_num_blocks,
            full_q_indices,
        ) = ctx.saved_tensors
        if dout is None:
            dout = torch.zeros_like(out)

        bst_bwd = BlockSparseTensorsTorch(
            mask_block_cnt=q_num_blocks,
            mask_block_idx=q_indices,
            full_block_cnt=full_q_num_blocks,
            full_block_idx=full_q_indices,
            block_size=ctx.block_size,
            dq_write_order=None if ctx.dq_write_order is None else ctx.dq_write_order[0],
            dq_write_order_full=None if ctx.dq_write_order is None else ctx.dq_write_order[1],
            spt=False if ctx.dq_write_order is not None else None,
        )
        if BWD_TIMING_ENABLED:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        dq_t, dk_t, dv_t = _flash_attn_bwd(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            out.transpose(1, 2),
            dout.transpose(1, 2),
            lse,
            softmax_scale=ctx.softmax_scale,
            mask_mod=ctx.mask_mod,
            aux_tensors=ctx.aux_tensors,
            block_sparse_tensors=bst_bwd,
            dlse=dlse if ctx.return_lse else None,
            deterministic=ctx.deterministic,
        )
        if BWD_TIMING_ENABLED:
            end.record()
            BWD_TIMING_EVENTS.append((start, end))
        return (
            dq_t.transpose(1, 2),
            dk_t.transpose(1, 2),
            dv_t.transpose(1, 2),
            *((None,) * 4),
        )


def flex_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask: BlockMask,
    softmax_scale: Optional[float] = None,
    return_lse: bool = False,
    deterministic: bool = False,
):
    return FlexAttnFunc.apply(q, k, v, block_mask, softmax_scale, return_lse, deterministic)


def get_block_size():
    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device)[0]
    if capability == 9:  # sm90
        return (128, 128)
    elif capability == 10:  # sm100
        return (256, 128)
    else:
        raise ValueError(
            f"Invalid capability: {capability}, flex_attention + fa4 backend only support sm90 and sm100 currently !!"
        )


WINDOW_SIZE = 1024
SEQ_LENS = [1024, 4096, 16384, 65536]
HEAD_CONFIGS = [
    ("mha_h20_d64", 20, 20, 64),
    ("gqa_h16_hkv4_d128", 16, 4, 128),
]


@dataclass(frozen=True)
class Case:
    name: str
    b: int
    hq: int
    hkv: int
    s: int
    d: int


@dataclass(frozen=True)
class DiffStats:
    dq_max: float
    dq_mean: float
    dk_max: float
    dk_mean: float
    dv_max: float
    dv_mean: float


@dataclass(frozen=True)
class BenchResult:
    case: Case
    nondeter_ms: float
    deter_ms: float
    nondeter_bwd_ms: float
    deter_bwd_ms: float
    diff: DiffStats


def make_block_mask(s):
    def mask_mod(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & (q_idx - kv_idx < WINDOW_SIZE)

    return create_block_mask(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=s,
        KV_LEN=s,
        device="cuda",
        _compile=True,
        BLOCK_SIZE=get_block_size(),
    )


def run_backward(q0, k0, v0, block_mask, dout, deterministic: bool):
    q = q0.detach().clone().requires_grad_(True)
    k = k0.detach().clone().requires_grad_(True)
    v = v0.detach().clone().requires_grad_(True)
    out = flex_attn_func(q, k, v, block_mask, deterministic=deterministic)
    dq, dk, dv = torch.autograd.grad(out, (q, k, v), dout)
    return dq.detach().clone(), dk.detach().clone(), dv.detach().clone()


def make_qkv(case: Case, seed: int):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    q = torch.randn(case.b, case.hq, case.s, case.d, device="cuda", dtype=torch.bfloat16, generator=gen)
    k = torch.randn(case.b, case.hkv, case.s, case.d, device="cuda", dtype=torch.bfloat16, generator=gen)
    v = torch.randn(case.b, case.hkv, case.s, case.d, device="cuda", dtype=torch.bfloat16, generator=gen)
    dout = torch.randn(case.b, case.hq, case.s, case.d, device="cuda", dtype=torch.bfloat16, generator=gen)
    return q, k, v, dout


def diff_stats(a, b) -> tuple[float, float]:
    diff = (a - b).abs()
    return diff.max().item(), diff.float().mean().item()


def grad_diff_stats(lhs, rhs) -> DiffStats:
    dq_max, dq_mean = diff_stats(lhs[0], rhs[0])
    dk_max, dk_mean = diff_stats(lhs[1], rhs[1])
    dv_max, dv_mean = diff_stats(lhs[2], rhs[2])
    return DiffStats(dq_max, dq_mean, dk_max, dk_mean, dv_max, dv_mean)


def check_deterministic_equal(case: Case, seed: int, repeats: int):
    q, k, v, dout = make_qkv(case, seed)
    block_mask = make_block_mask(case.s)
    ref = run_backward(q, k, v, block_mask, dout, deterministic=True)
    for rep in range(repeats):
        got = run_backward(q, k, v, block_mask, dout, deterministic=True)
        assert torch.equal(got[0], ref[0]), f"{case.name}: dQ differs at rep {rep}"
        assert torch.equal(got[1], ref[1]), f"{case.name}: dK differs at rep {rep}"
        assert torch.equal(got[2], ref[2]), f"{case.name}: dV differs at rep {rep}"


def bench_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def bench_ms_with_bwd_timing(fn, warmup: int, iters: int) -> tuple[float, float]:
    enable_bwd_timing(False, clear=True)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    enable_bwd_timing(True, clear=True)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    try:
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        e2e_ms = start.elapsed_time(end) / iters
        bwd_times = pop_bwd_times_ms()
        bwd_ms = sum(bwd_times) / len(bwd_times) if bwd_times else float("nan")
        return e2e_ms, bwd_ms
    finally:
        enable_bwd_timing(False, clear=True)


def compare_case(case: Case, seed: int, warmup: int, iters: int) -> BenchResult:
    q, k, v, dout = make_qkv(case, seed)
    block_mask = make_block_mask(case.s)
    nondeter = run_backward(q, k, v, block_mask, dout, deterministic=False)
    deter = run_backward(q, k, v, block_mask, dout, deterministic=True)
    diff = grad_diff_stats(deter, nondeter)
    nondeter_ms, nondeter_bwd_ms = bench_ms_with_bwd_timing(
        lambda: run_backward(q, k, v, block_mask, dout, deterministic=False), warmup, iters
    )
    deter_ms, deter_bwd_ms = bench_ms_with_bwd_timing(
        lambda: run_backward(q, k, v, block_mask, dout, deterministic=True), warmup, iters
    )
    return BenchResult(case, nondeter_ms, deter_ms, nondeter_bwd_ms, deter_bwd_ms, diff)


def fixed_cases(max_seqlen: int = max(SEQ_LENS)) -> list[Case]:
    return [
        Case(f"{name}_s{s // 1024}k", 1, hq, hkv, s, d)
        for s in SEQ_LENS
        if s <= max_seqlen
        for name, hq, hkv, d in HEAD_CONFIGS
    ]


def random_cases(n: int, seed: int, min_seqlen: int, max_seqlen: int) -> list[Case]:
    rng = random.Random(seed)
    cases = []
    for i in range(n):
        name, hq, hkv, d = rng.choice(HEAD_CONFIGS)
        s = rng.randint(min_seqlen, max_seqlen)
        cases.append(Case(f"rand{i}_{name}_s{s}", 1, hq, hkv, s, d))
    return cases


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Flex deterministic backward accuracy and speed test.")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--random-cases", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-seqlen", type=int, default=max(SEQ_LENS))
    parser.add_argument("--random-min-seqlen", type=int, default=4096)
    parser.add_argument("--no-bench", action="store_true")
    args = parser.parse_args(argv)

    major, minor = torch.cuda.get_device_capability()
    print(f"Using flash_attn from: {fa_interface.__file__}")
    print(f"Device: {torch.cuda.get_device_name()} (SM{major}{minor})")

    fixed = fixed_cases(args.max_seqlen)
    random_min_seqlen = min(args.random_min_seqlen, args.max_seqlen)
    acc_cases = fixed + random_cases(
        args.random_cases,
        args.seed,
        min_seqlen=random_min_seqlen,
        max_seqlen=args.max_seqlen,
    )
    for i, case in enumerate(acc_cases):
        check_deterministic_equal(case, seed=args.seed + i, repeats=args.repeats)
    print(f"DETERMINISTIC OK: {len(acc_cases)} cases x {args.repeats} repeats")

    if args.no_bench:
        return

    print()
    print("DETER vs NON-DETER fwd+bwd e2e")
    print(
        "case                         non_e2e  deter_e2e  e2e_x  non_bwd  deter_bwd  bwd_x  "
        "dQ(max/mean)       dK(max/mean)       dV(max/mean)"
    )
    for i, case in enumerate(fixed):
        result = compare_case(case, seed=args.seed + 1000 + i, warmup=args.warmup, iters=args.iters)
        e2e_slowdown = result.deter_ms / result.nondeter_ms
        bwd_slowdown = result.deter_bwd_ms / result.nondeter_bwd_ms
        d = result.diff
        print(
            f"{case.name:28s} {result.nondeter_ms:8.3f} {result.deter_ms:10.3f} "
            f"{e2e_slowdown:6.2f}x {result.nondeter_bwd_ms:8.3f} "
            f"{result.deter_bwd_ms:10.3f} {bwd_slowdown:6.2f}x  "
            f"{d.dq_max:.3e}/{d.dq_mean:.3e}  "
            f"{d.dk_max:.3e}/{d.dk_mean:.3e}  "
            f"{d.dv_max:.3e}/{d.dv_mean:.3e}"
        )


if __name__ == "__main__":
    main()
