"""Bounded native-forward integration checks; no standalone decode API."""

import math
import pytest
import torch
from flash_attn.cute.interface import (
    _flash_attn_fwd,
    _flash_attn_fwd_combine,
    flash_attn_func,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
    reason="SM100 SwapAB integration",
)


def inputs(dq=128, dv=128, ratio=3, length=1025, dtype=torch.bfloat16, strided=False):
    torch.manual_seed(42)
    q = torch.randn(2, 1, ratio * 2, dq, device="cuda", dtype=dtype)
    k = torch.randn(
        2, length, 2 * (2 if strided else 1), dq, device="cuda", dtype=dtype
    )
    v = torch.randn(
        2, length, 2 * (2 if strided else 1), dv, device="cuda", dtype=dtype
    )
    return q, k[:, :, ::2] if strided else k, v[:, :, ::2] if strided else v


def reference(q, k, v, used=None, window=None, softcap=None, mask=False, bias=None):
    ratio = q.shape[2] // k.shape[2]
    scores = torch.einsum(
        "bqhd,bkhd->bhqk", q.float(), k.repeat_interleave(ratio, 2).float()
    ) / math.sqrt(q.shape[-1])
    if softcap:
        scores = softcap * torch.tanh(scores / softcap)
    if bias is not None:
        scores = scores + bias[:, None, None, None]
    positions = torch.arange(k.shape[1], device=q.device)
    if used is not None:
        scores.masked_fill_(
            positions[None, None, None, :] >= used[:, None, None, None], -torch.inf
        )
    if window is not None:
        ends = (
            used
            if used is not None
            else torch.full((q.shape[0],), k.shape[1], device=q.device)
        )
        scores.masked_fill_(
            positions[None, None, None, :] < ends[:, None, None, None] - 1 - window,
            -torch.inf,
        )
    if mask:
        scores.masked_fill_(positions[None, None, None, :] % 3 != 0, -torch.inf)
    lse = scores.logsumexp(-1)
    probs = scores.softmax(-1).nan_to_num()
    out = torch.einsum("bhqk,bkhd->bqhd", probs, v.repeat_interleave(ratio, 2).float())
    return out, lse


def check(
    q,
    k,
    v,
    *,
    ns=3,
    used=None,
    window=None,
    softcap=None,
    mask=False,
    bias=None,
    **kwargs,
):
    import cutlass
    from cutlass import cute

    def mask_mod(b, h, q_idx, kv_idx, seqlen_info, aux_tensors):
        return kv_idx % 3 == 0

    def score_mod(score, b, h, q_idx, kv_idx, seqlen_info, aux_tensors):
        bi = cute.make_rmem_tensor(1, cutlass.Int32)
        bi.store(b)
        bias_frag = cute.make_rmem_tensor(1, cutlass.Float32)
        bias_frag[0] = aux_tensors[0][bi[0]]
        return score + bias_frag.load()

    if mask:
        kwargs["mask_mod"] = cute.jit(mask_mod)
    if bias is not None:
        kwargs.update(score_mod=cute.jit(score_mod), aux_tensors=[bias])
    o, lse, *_ = _flash_attn_fwd(
        q,
        k,
        v,
        num_splits=ns,
        seqused_k=used,
        window_size_left=window,
        softcap=softcap,
        return_lse=True,
        **kwargs,
    )
    ref, ref_lse = reference(q, k, v, used, window, softcap, mask, bias)
    torch.testing.assert_close(o.float(), ref, atol=0.004, rtol=0.025)
    torch.testing.assert_close(lse, ref_lse, atol=0.004, rtol=0.003)
    torch.cuda.synchronize()
    return o, lse


@pytest.fixture(autouse=True)
def native(monkeypatch):
    monkeypatch.setenv("FLASH_ATTENTION_SM100_BACKEND", "swapab")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "dims",
    [
        (64, 64),
        (96, 96),
        (128, 128),
        (64, 128),
        (128, 64),
        (96, 128),
        (128, 96),
        (64, 96),
        (96, 64),
    ],
)
@pytest.mark.parametrize("ns", [1, 3])
def test_dense(dtype, dims, ns):
    check(*inputs(*dims, dtype=dtype, strided=True), ns=ns)


@pytest.mark.parametrize(
    "ratio,length,ns",
    [
        (1, 1, 1),
        (1, 0, 3),
        (3, 255, 3),
        (5, 256, 1),
        (8, 257, 3),
        (16, 512, 3),
        (32, 1537, 3),
    ],
)
def test_heads_and_blocks(ratio, length, ns):
    check(*inputs(ratio=ratio, length=length), ns=ns)


@pytest.mark.parametrize(
    "kind", ["used", "empty", "causal", "local", "softcap", "mask", "aux"]
)
@pytest.mark.parametrize("ns", [1, 3])
def test_semantics(kind, ns):
    args = {}
    if kind in ("used", "empty", "local"):
        args["used"] = torch.tensor(
            [0 if kind == "empty" else 37, 789], device="cuda", dtype=torch.int32
        )
    if kind == "local":
        args["window"] = 31
    if kind == "causal":
        args["causal"] = True
    if kind == "softcap":
        args["softcap"] = 2.0
    if kind == "mask":
        args["mask"] = True
    if kind == "aux":
        args["bias"] = torch.tensor([0.2, -0.3], device="cuda")
    tensors = inputs()
    check(*tensors, ns=ns, **args)
    if kind == "aux":
        # Same compiled signature, different device pointer and values.
        args["bias"] = torch.tensor([-0.7, 0.8], device="cuda")
        check(*tensors, ns=ns, **args)


@pytest.mark.parametrize("ns", [1, 3])
def test_public_buffers_and_combine(ns, monkeypatch):
    q, k, v = inputs()
    out = torch.empty_like(q)
    lse = torch.empty((2, 6, 1), device="cuda", dtype=torch.float32)
    if ns == 1:

        def unexpected(*args, **kwargs):
            raise AssertionError("single split must bypass combine")

        monkeypatch.setattr(
            "flash_attn.cute.interface._flash_attn_fwd_combine", unexpected
        )
    o, actual_lse = check(q, k, v, ns=ns, out=out, lse=lse)
    assert o.data_ptr() == out.data_ptr() and actual_lse.data_ptr() == lse.data_ptr()
    public_o, public_l = flash_attn_func(q, k, v, num_splits=ns, return_lse=True)
    torch.testing.assert_close(public_o, o)
    torch.testing.assert_close(public_l, actual_lse)


@pytest.mark.parametrize("blocks", [3, 8])
@pytest.mark.parametrize("ns", [1, 2])
def test_paged_stats_handshake(blocks, ns):
    """Exercise the final-max / normalized-stats handoff after >2 KV blocks.

    Keep this fixture separate from used-length/graph tests: the original
    protocol stalled with no seqused_k, including identity/constant mappings.
    A bounded subprocess runner is required when testing an unfixed kernel.
    """
    torch.manual_seed(7)
    q = torch.randn(1, 1, 32, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(8, 256, 8, 128, device="cuda", dtype=q.dtype)
    v = torch.randn_like(k)
    table = torch.tensor(
        [[3, 0, 7, 2, 5, 1, 6, 4][:blocks]], device="cuda", dtype=torch.int32
    )
    for _ in range(10):
        o, lse, *_ = _flash_attn_fwd(
            q,
            k,
            v,
            page_table=table,
            num_splits=ns,
            return_lse=True,
        )
        ro, rl = reference(
            q,
            k[table.long()].reshape(1, -1, 8, 128),
            v[table.long()].reshape(1, -1, 8, 128),
        )
        torch.testing.assert_close(o.float(), ro, atol=0.004, rtol=0.025)
        torch.testing.assert_close(lse, rl, atol=0.004, rtol=0.003)
    torch.cuda.synchronize()


@pytest.mark.parametrize("page_size", [256, 512, 768])
@pytest.mark.parametrize("dims", [(128, 128), (96, 64)])
@pytest.mark.parametrize("ns", [1, 3])
def test_paged_graph(page_size, dims, ns):
    q, _, _ = inputs(*dims)
    torch.manual_seed(17)
    # Strided cache, shared physical page, and deliberately permuted mappings.
    k = torch.randn(9, page_size, 4, dims[0], device="cuda", dtype=q.dtype)[:, :, ::2]
    v = torch.randn(9, page_size, 4, dims[1], device="cuda", dtype=q.dtype)[:, :, ::2]
    table = torch.tensor([[7, 2, 5], [1, 7, 3]], device="cuda", dtype=torch.int32)
    used = torch.tensor(
        [page_size + 17, 3 * page_size - 9], device="cuda", dtype=torch.int32
    )

    def run():
        return _flash_attn_fwd(
            q, k, v, page_table=table, seqused_k=used, num_splits=ns, return_lse=True
        )[:2]

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        o, lse = run()
    for mapping, lengths in [
        ([[7, 2, 5], [1, 7, 3]], [page_size + 17, 3 * page_size - 9]),
        ([[4, 0, 8], [8, 6, 4]], [0, 2 * page_size + 1]),
    ]:
        table.copy_(torch.tensor(mapping, device="cuda", dtype=torch.int32))
        used.copy_(torch.tensor(lengths, device="cuda", dtype=torch.int32))
        graph.replay()
        kd = k[table.long()].reshape(2, 3 * page_size, 2, dims[0])
        vd = v[table.long()].reshape(2, 3 * page_size, 2, dims[1])
        ro, rl = reference(q, kd, vd, used)
        torch.testing.assert_close(o.float(), ro, atol=0.004, rtol=0.025)
        torch.testing.assert_close(lse, rl, atol=0.004, rtol=0.003)


@pytest.mark.parametrize(
    "kind", ["prefill", "training", "hd256", "page128", "page384", "override"]
)
def test_fallback(kind, monkeypatch):
    q, k, v = inputs()
    kw = {}
    if kind == "prefill":
        q = q.expand(-1, 2, -1, -1).contiguous()
    if kind == "training":
        q.requires_grad_(True)
    if kind == "hd256":
        q, k, v = inputs(256, 256)
    if kind.startswith("page"):
        size = int(kind[4:])
        k = torch.randn(8, size, 2, 128, device="cuda", dtype=q.dtype)
        v = torch.randn_like(k)
        kw["page_table"] = torch.tensor(
            [[3, 1], [6, 4]], device="cuda", dtype=torch.int32
        )
    if kind == "override":
        kw["pack_gqa"] = False
    with pytest.raises(ValueError, match="SwapAB unsupported"):
        _flash_attn_fwd(q, k, v, **kw)
    monkeypatch.setenv("FLASH_ATTENTION_SM100_BACKEND", "auto")
    auto = _flash_attn_fwd(q, k, v, **kw)[0]
    monkeypatch.setenv("FLASH_ATTENTION_SM100_BACKEND", "standard")
    standard = _flash_attn_fwd(q, k, v, **kw)[0]
    torch.testing.assert_close(auto, standard)


def test_backend_cache_separation(monkeypatch):
    q, k, v = inputs()
    for mode in ["swapab", "standard", "auto"]:
        monkeypatch.setenv("FLASH_ATTENTION_SM100_BACKEND", mode)
        check(q, k, v)
    tokens = {key[0] for key in _flash_attn_fwd.compile_cache.cache}
    assert ("sm100_swapab", 1) in tokens and ("standard", 1) in tokens


@pytest.mark.parametrize("ns", [1, 3])
def test_fake_to_real(ns):
    from torch._subclasses.fake_tensor import FakeTensorMode

    q, k, v = inputs(96, 64, ratio=5, length=777)
    with FakeTensorMode() as mode:
        _flash_attn_fwd(
            mode.from_tensor(q),
            mode.from_tensor(k),
            mode.from_tensor(v),
            num_splits=ns,
            return_lse=True,
        )
    size = len(_flash_attn_fwd.compile_cache.cache)
    check(q, k, v, ns=ns)
    assert len(_flash_attn_fwd.compile_cache.cache) == size


@pytest.mark.parametrize("page_size", [128, 256, 512])
@pytest.mark.parametrize("ns", [1, 3])
def test_normal_paging(page_size, ns, monkeypatch):
    monkeypatch.setenv("FLASH_ATTENTION_SM100_BACKEND", "standard")
    torch.manual_seed(71)
    q = torch.randn(2, 17, 6, 96, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(9, page_size, 2, 96, device="cuda", dtype=q.dtype)
    v = torch.randn(9, page_size, 2, 64, device="cuda", dtype=q.dtype)
    table = torch.tensor([[7, 1, 5], [4, 7, 2]], device="cuda", dtype=torch.int32)
    used = torch.tensor(
        [page_size + 7, 3 * page_size - 1], device="cuda", dtype=torch.int32
    )
    o, lse, *_ = _flash_attn_fwd(
        q, k, v, page_table=table, seqused_k=used, num_splits=ns, return_lse=True
    )
    ro, rl = reference(
        q,
        k[table.long()].reshape(2, -1, 2, 96),
        v[table.long()].reshape(2, -1, 2, 64),
        used,
    )
    torch.testing.assert_close(o.float(), ro, atol=0.004, rtol=0.025)
    torch.testing.assert_close(lse, rl, atol=0.004, rtol=0.003)


@pytest.mark.parametrize("ns", [1, 3])
def test_fully_masked_rows(ns):
    from cutlass import cute

    @cute.jit
    def none_kept(b, h, q_idx, kv_idx, seqlen_info, aux_tensors):
        return kv_idx < 0

    q, k, v = inputs(length=257)
    o, lse, *_ = _flash_attn_fwd(
        q, k, v, mask_mod=none_kept, num_splits=ns, return_lse=True
    )
    assert torch.count_nonzero(o) == 0
    assert torch.isneginf(lse).all()


def test_normalized_partial_abi(monkeypatch):
    q, k, v = inputs(length=513)
    used = torch.tensor([0, 257], device="cuda", dtype=torch.int32)
    observed = []

    def inspect_partials(op, lp, *args, **kwargs):
        assert op.dtype == lp.dtype == torch.float32
        assert op.shape == (8, 2, 1, 6, 128)
        assert lp.shape == (8, 2, 1, 6)
        for split in range(8):
            # Cyclic 256-token blocks; this input has at most one block per split.
            start, end = split * 256, min((split + 1) * 256, k.shape[1])
            if start >= k.shape[1]:
                assert torch.count_nonzero(op[split]) == 0
                assert torch.isneginf(lp[split]).all()
            else:
                lengths = (used - start).clamp(0, end - start)
                ro, rl = reference(q, k[:, start:end], v[:, start:end], lengths)
                torch.testing.assert_close(op[split], ro, atol=0.004, rtol=0.025)
                torch.testing.assert_close(
                    lp[split].transpose(-1, -2), rl, atol=0.004, rtol=0.003
                )
        observed.append(True)
        return _flash_attn_fwd_combine(op, lp, *args, **kwargs)

    inspect_partials.compile_cache = _flash_attn_fwd_combine.compile_cache
    monkeypatch.setattr(
        "flash_attn.cute.interface._flash_attn_fwd_combine", inspect_partials
    )
    check(q, k, v, used=used, ns=8)
    assert observed == [True]
