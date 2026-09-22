"""SM100 symmetric head-dimension 512 correctness and boundary regressions."""

from itertools import accumulate
import os
import pytest
import torch
from flash_attn.cute import flash_attn_func, flash_attn_varlen_func
from flash_attn.cute import flash_bwd_sm100_hd512 as native
from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd
from flash_attn.cute.testing import attention_ref
from functools import wraps
from flash_attn.cute import interface
from test_flash_attn import check_tensor_vs_ref

pytestmark = [
    # These regressions inspect real values, cache reuse, and output buffers.
    # Run them on the selected GPU in pass 2, not in the parallel fake-tensor pass.
    pytest.mark.skipif(
        os.getenv("FLASH_ATTENTION_FAKE_TENSOR", "0") == "1",
        reason="D512 regressions require GPU execution",
    ),
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
        reason="Requires SM100",
    ),
]


def reference(q, k, v, causal=False, window=(-1, -1), softcap=0.0, scale=None):
    q, k, v = (x.double() for x in (q, k, v))
    ratio = q.shape[-2] // k.shape[-2]
    k = k.repeat_interleave(ratio, dim=-2)
    v = v.repeat_interleave(ratio, dim=-2)
    scores = torch.einsum("bmhd,bnhd->bhmn", q, k) * (
        q.shape[-1] ** -0.5 if scale is None else scale
    )
    if softcap:
        scores = softcap * torch.tanh(scores / softcap)
    qi = torch.arange(q.shape[1], device=q.device)[:, None] + k.shape[1] - q.shape[1]
    ki = torch.arange(k.shape[1], device=q.device)[None, :]
    valid = torch.ones((q.shape[1], k.shape[1]), dtype=torch.bool, device=q.device)
    if causal:
        valid &= ki <= qi
    if window[0] >= 0:
        valid &= ki >= qi - window[0]
    if window[1] >= 0:
        valid &= ki <= qi + window[1]
    scores = scores.masked_fill(~valid, -torch.inf)
    # A fully masked row has zero output and zero gradients.
    has_keys = valid.any(-1)[None, None, :, None]
    safe_scores = torch.where(has_keys, scores, 0.0)
    probs = safe_scores.softmax(-1).masked_fill(~valid, 0.0)
    out = torch.einsum("bhmn,bnhd->bmhd", probs, v)
    lse = torch.where(
        has_keys.squeeze(-1), torch.logsumexp(safe_scores, -1), -torch.inf
    )
    return out, lse


def check(actual, expected, dtype):
    assert torch.isfinite(actual).all()
    atol, rtol = (0.01, 0.03) if dtype == torch.bfloat16 else (0.002, 0.005)
    torch.testing.assert_close(actual.double(), expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 17, 4, 2),
        (2, 127, 129, 4, 2),
        (1, 257, 193, 2, 2),
        (1, 512, 512, 4, 1),
        (1, 2048, 2048, 8, 2),
    ],
)
def test_dense(dtype, causal, shape):
    torch.manual_seed(123)
    b, sq, sk, hq, hk = shape
    q = torch.randn(b, sq, hq, 512, device="cuda", dtype=dtype, requires_grad=True)
    k, v = [
        torch.randn(b, sk, hk, 512, device="cuda", dtype=dtype, requires_grad=True)
        for _ in range(2)
    ]
    qr, kr, vr = [x.detach().double().requires_grad_() for x in (q, k, v)]
    out, lse = flash_attn_func(q, k, v, causal=causal, return_lse=True)
    ref, rlse = reference(qr, kr, vr, causal=causal)
    check(out, ref, dtype)
    finite = torch.isfinite(rlse)
    torch.testing.assert_close(
        lse.double()[finite], rlse[finite], atol=0.003, rtol=0.003
    )
    grad = torch.randn_like(out)
    out.backward(grad)
    ref.backward(grad.double())
    for a, r in zip((q, k, v), (qr, kr, vr)):
        check(a.grad, r.grad, dtype)
    torch.cuda.synchronize()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("causal", [False, True])
def test_varlen(dtype, causal):
    torch.manual_seed(234)
    lq, lk = [1, 127, 193, 0, 5], [17, 129, 65, 9, 0]
    q = torch.randn(sum(lq), 4, 512, device="cuda", dtype=dtype, requires_grad=True)
    k, v = [
        torch.randn(sum(lk), 2, 512, device="cuda", dtype=dtype, requires_grad=True)
        for _ in range(2)
    ]
    qr, kr, vr = [x.detach().double().requires_grad_() for x in (q, k, v)]
    cuq = torch.tensor([0, *accumulate(lq)], device="cuda", dtype=torch.int32)
    cuk = torch.tensor([0, *accumulate(lk)], device="cuda", dtype=torch.int32)
    out, _ = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cuq,
        cu_seqlens_k=cuk,
        max_seqlen_q=max(lq),
        max_seqlen_k=max(lk),
        causal=causal,
    )
    refs = [
        reference(a[None], b[None], c[None], causal=causal)[0][0]
        for a, b, c in zip(qr.split(lq), kr.split(lk), vr.split(lk))
    ]
    ref = torch.cat(refs)
    check(out, ref, dtype)
    grad = torch.randn_like(out)
    out.backward(grad)
    ref.backward(grad.double())
    for a, r in zip((q, k, v), (qr, kr, vr)):
        check(a.grad, r.grad, dtype)
    torch.cuda.synchronize()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "mode", ["window", "softcap", "lse_grad", "shared_kv", "strided"]
)
def test_features(dtype, mode):
    torch.manual_seed(345)
    q = torch.randn(1, 137, 4, 512, device="cuda", dtype=dtype, requires_grad=True)
    k, v = [
        torch.randn(1, 149, 2, 512, device="cuda", dtype=dtype, requires_grad=True)
        for _ in range(2)
    ]
    if mode == "shared_kv":
        v = k
    if mode == "strided":
        q = torch.randn(1, 137, 4, 1024, device="cuda", dtype=dtype)[
            ..., ::2
        ].requires_grad_()
    qr, kr = [x.detach().double().requires_grad_() for x in (q, k)]
    vr = kr if mode == "shared_kv" else v.detach().double().requires_grad_()
    window = (33, 17) if mode == "window" else (-1, -1)
    softcap = 3.0 if mode == "softcap" else 0.0
    out, lse = flash_attn_func(
        q, k, v, window_size=window, softcap=softcap, return_lse=True
    )
    ref, rlse = reference(qr, kr, vr, window=window, softcap=softcap)
    check(out, ref, dtype)
    grad = torch.randn_like(out)
    if mode == "lse_grad":
        glse = torch.randn_like(lse)
        torch.autograd.backward((out, lse), (grad, glse))
        torch.autograd.backward((ref, rlse), (grad.double(), glse.double()))
    else:
        out.backward(grad)
        ref.backward(grad.double())
    for a, r in zip((q, k, v), (qr, kr, vr)):
        check(a.grad, r.grad, dtype)
    torch.cuda.synchronize()


@pytest.mark.parametrize("dim", [64, 128, 256])
def test_existing_dimensions(dim):
    torch.manual_seed(456)
    q, k, v = [
        torch.randn(
            1, 128, 2, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        for _ in range(3)
    ]
    qr, kr, vr = [x.detach().double().requires_grad_() for x in (q, k, v)]
    out, _ = flash_attn_func(q, k, v, causal=True)
    ref, _ = reference(qr, kr, vr, causal=True)
    check(out, ref, q.dtype)
    grad = torch.randn_like(out)
    out.backward(grad)
    ref.backward(grad.double())
    for a, r in zip((q, k, v), (qr, kr, vr)):
        check(a.grad, r.grad, q.dtype)
    torch.cuda.synchronize()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_cuda_graph(dtype):
    torch.manual_seed(567)
    q, k, v = [
        torch.randn(1, 128, 2, 512, device="cuda", dtype=dtype, requires_grad=True)
        for _ in range(3)
    ]
    grad = torch.randn_like(q)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            warm, _ = flash_attn_func(q, k, v, causal=True)
            torch.autograd.grad(warm, (q, k, v), grad)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out, _ = flash_attn_func(q, k, v, causal=True)
        grads = torch.autograd.grad(out, (q, k, v), grad)
    with torch.no_grad():
        q.add_(0.1)
    graph.replay()
    qr, kr, vr = [x.detach().double().requires_grad_() for x in (q, k, v)]
    ref, _ = reference(qr, kr, vr, causal=True)
    rg = torch.autograd.grad(ref, (qr, kr, vr), grad.double())
    check(out, ref, dtype)
    for a, r in zip(grads, rg):
        check(a, r, dtype)
    torch.cuda.synchronize()


def test_unsupported_options():
    q, k, v = [
        torch.randn(
            1, 64, 2, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        for _ in range(3)
    ]
    out, _ = flash_attn_func(q, k, v, deterministic=True)
    with pytest.raises(NotImplementedError, match="deterministic"):
        out.sum().backward()


@pytest.mark.parametrize("sq, sk", [(17, 0), (0, 17)])
def test_empty_inputs(sq, sk):
    q = torch.randn(
        1, sq, 2, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    k, v = [
        torch.empty(1, sk, 2, 512, device="cuda", dtype=q.dtype, requires_grad=True)
        for _ in range(2)
    ]
    out, _ = flash_attn_func(q, k, v)
    assert torch.equal(out, torch.zeros_like(out))
    out.sum().backward()
    assert torch.equal(q.grad, torch.zeros_like(q))
    assert torch.equal(k.grad, torch.zeros_like(k))
    assert torch.equal(v.grad, torch.zeros_like(v))
    torch.cuda.synchronize()


@pytest.mark.parametrize("tile_n", [32, 64, 96])
def test_explicit_forward_tile(tile_n):
    from flash_attn.cute.interface import _flash_attn_fwd

    torch.manual_seed(678)
    q = torch.randn(1, 257, 4, 512, device="cuda", dtype=torch.bfloat16)
    k, v = [
        torch.randn(1, 193, 2, 512, device="cuda", dtype=torch.bfloat16)
        for _ in range(2)
    ]
    out, lse, _, _ = _flash_attn_fwd(
        q, k, v, causal=True, tile_mn=(128, tile_n), return_lse=True
    )
    ref, ref_lse = reference(q, k, v, causal=True)
    check(out, ref, q.dtype)
    valid = torch.isfinite(ref_lse)
    torch.testing.assert_close(
        lse.double()[valid], ref_lse[valid], atol=0.003, rtol=0.003
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("layout", ["last_stride", "row_padding", "offset"])
def test_native_output_buffers(dtype, layout):
    torch.manual_seed(901)
    q = torch.randn(1, 137, 4, 512, device="cuda", dtype=dtype)
    k, v = [torch.randn(1, 149, 2, 512, device="cuda", dtype=dtype) for _ in range(2)]
    out, lse, _, _ = _flash_attn_fwd(q, k, v, causal=True, return_lse=True)
    dout = torch.randn_like(out)

    def make_buffer(t):
        if layout == "last_stride":
            return torch.empty((*t.shape, 2), device=t.device, dtype=t.dtype)[..., 0]
        if layout == "row_padding":
            return torch.empty((*t.shape[:-1], 513), device=t.device, dtype=t.dtype)[
                ..., :512
            ]
        return torch.empty(t.numel() + 1, device=t.device, dtype=t.dtype)[1:].view(
            t.shape
        )

    buffers = tuple(make_buffer(t) for t in (q, k, v))
    actual = _flash_attn_bwd(
        q,
        k,
        v,
        out,
        dout,
        lse,
        causal=True,
        dq=buffers[0],
        dk=buffers[1],
        dv=buffers[2],
    )
    qr, kr, vr = [x.double().requires_grad_() for x in (q, k, v)]
    ref, _ = reference(qr, kr, vr, causal=True)
    expected = torch.autograd.grad(ref, (qr, kr, vr), dout.double())
    for a, buf, r in zip(actual, buffers, expected):
        assert a is buf
        check(a, r, dtype)
    torch.cuda.synchronize()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("causal", [False, True])
def test_native_dense_used_lengths(dtype, causal):
    torch.manual_seed(902)
    q = torch.randn(2, 137, 4, 512, device="cuda", dtype=dtype)
    k, v = [torch.randn(2, 149, 2, 512, device="cuda", dtype=dtype) for _ in range(2)]
    uq = torch.tensor([113, 0], device="cuda", dtype=torch.int32)
    uk = torch.tensor([117, 13], device="cuda", dtype=torch.int32)
    out, lse, _, _ = _flash_attn_fwd(
        q, k, v, seqused_q=uq, seqused_k=uk, causal=causal, return_lse=True
    )
    dout = torch.randn_like(q)
    actual = _flash_attn_bwd(
        q, k, v, out, dout, lse, seqused_q=uq, seqused_k=uk, causal=causal
    )
    qr, kr, vr = [x[0:1].double().requires_grad_() for x in (q, k, v)]
    ref, _ = reference(qr[:, :113], kr[:, :117], vr[:, :117], causal=causal)
    expected = torch.autograd.grad(ref, (qr, kr, vr), dout[0:1, :113].double())
    for a, r in zip(actual, expected):
        check(a[0:1], r, dtype)
        assert torch.count_nonzero(a[1]) == 0
    torch.cuda.synchronize()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("window", [(256, 0), (0, 0), (37, 19)])
def test_native_window_tile_skipping(dtype, window):
    torch.manual_seed(903)
    q = torch.randn(1, 769, 4, 512, device="cuda", dtype=dtype)
    k, v = [torch.randn(1, 641, 2, 512, device="cuda", dtype=dtype) for _ in range(2)]
    out, lse, _, _ = _flash_attn_fwd(
        q,
        k,
        v,
        window_size_left=window[0],
        window_size_right=window[1],
        return_lse=True,
    )
    dout = torch.randn_like(q)
    actual = _flash_attn_bwd(
        q, k, v, out, dout, lse, window_size_left=window[0], window_size_right=window[1]
    )
    qr, kr, vr = [x.double().requires_grad_() for x in (q, k, v)]
    ref, _ = reference(qr, kr, vr, window=window)
    expected = torch.autograd.grad(ref, (qr, kr, vr), dout.double())
    for a, r in zip(actual, expected):
        check(a, r, dtype)
    torch.cuda.synchronize()


def make_case(dtype, page_size, sq, hk=2):
    torch.manual_seed(512 + page_size)
    lengths = [257, 139]
    width = (max(lengths) + page_size - 1) // page_size
    pool = 2 * width + 3
    table = torch.randperm(pool, device="cuda")[: 2 * width].reshape(2, width).int()
    q = torch.randn(2, sq, 4, 512, device="cuda", dtype=dtype)
    k, v = [
        torch.randn(pool, page_size, hk, 512, device="cuda", dtype=dtype)
        for _ in range(2)
    ]
    used = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    return q, k, v, table, used, lengths


def verify(
    q,
    k,
    v,
    table,
    used,
    lengths,
    *,
    causal=False,
    packed=False,
    window=(-1, -1),
    softcap=0.0,
    splits=1,
):
    opts = dict(
        page_table=table,
        seqused_k=used,
        max_seqlen_k=max(lengths),
        causal=causal,
        window_size=window,
        softcap=softcap,
        return_lse=True,
        num_splits=splits,
    )
    if packed:
        lq = [q.shape[1], max(1, q.shape[1] - 7)]
        argq = torch.cat([q[i, :n] for i, n in enumerate(lq)])
        opts.update(
            cu_seqlens_q=torch.tensor(
                [0, lq[0], sum(lq)], device="cuda", dtype=torch.int32
            ),
            max_seqlen_q=max(lq),
        )
    else:
        lq = [q.shape[1]] * 2
        argq = q
    out, lse = flash_attn_varlen_func(argq, k, v, **opts)
    offset = 0
    for i, (nq, nk) in enumerate(zip(lq, lengths)):
        kr, vr = [x[table[i].long()].flatten(0, 1)[:nk][None] for x in (k, v)]
        ref, rlse = reference(
            q[i : i + 1, :nq], kr, vr, causal=causal, window=window, softcap=softcap
        )
        actual = out[offset : offset + nq][None] if packed else out[i : i + 1]
        alse = lse[:, offset : offset + nq][None] if packed else lse[i : i + 1]
        check(actual, ref, q.dtype)
        torch.testing.assert_close(alse.double(), rlse, atol=0.003, rtol=0.003)
        offset += nq
    torch.cuda.synchronize()
    return out


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("page_size", [16, 32, 64, 96, 128, 256])
@pytest.mark.parametrize("sq,causal", [(1, False), (65, True)])
def test_paged(dtype, page_size, sq, causal):
    verify(*make_case(dtype, page_size, sq), causal=causal)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "mode", ["packed", "window", "softcap", "mqa", "mha", "empty_kv"]
)
def test_paged_features(dtype, mode):
    args = list(make_case(dtype, 16, 65, hk={"mqa": 1, "mha": 4}.get(mode, 2)))
    if mode == "empty_kv":
        args[-1][1] = 0
        args[-2][1] = 0
    verify(
        *args,
        packed=mode == "packed",
        window=(33, 7) if mode == "window" else (-1, -1),
        softcap=8.0 if mode == "softcap" else 0.0,
    )


def test_paged_backward_rejected():
    q, k, v, table, used, _ = make_case(torch.bfloat16, 16, 1)
    q.requires_grad_()
    out, _ = flash_attn_varlen_func(q, k, v, page_table=table, seqused_k=used)
    with pytest.raises(NotImplementedError, match="forward only"):
        out.sum().backward()


def test_paged_requires_lengths():
    q, k, v, table, _, _ = make_case(torch.bfloat16, 16, 1)
    with pytest.raises(ValueError, match="requires seqused_k"):
        flash_attn_varlen_func(q, k, v, page_table=table)


@pytest.mark.parametrize("page_size", [16, 96])
def test_paged_graph_dynamic_table(page_size):
    q, k, v, table, used, lengths = make_case(torch.bfloat16, page_size, 65)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            flash_attn_varlen_func(
                q, k, v, page_table=table, seqused_k=used, causal=True
            )
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out, _ = flash_attn_varlen_func(
            q, k, v, page_table=table, seqused_k=used, causal=True
        )
    q.add_(0.25)
    table.copy_(table.flip(0).clone())
    used[1] = 77
    lengths[1] = 77
    graph.replay()
    expected = verify(q, k, v, table, used, lengths, causal=True)
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("page_size", [16, 96])
@pytest.mark.parametrize(
    "mode", ["decode", "prefill", "packed", "empty_kv", "window", "softcap"]
)
def test_paged_splitkv(dtype, page_size, mode):
    args = list(make_case(dtype, page_size, 1 if mode == "decode" else 65))
    if mode == "empty_kv":
        args[-1][1] = 0
        args[-2][1] = 0
    verify(
        *args,
        causal=True,
        packed=mode == "packed",
        splits=4,
        window=(33, 7) if mode == "window" else (-1, -1),
        softcap=8.0 if mode == "softcap" else 0.0,
    )


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("splits", [1, 4])
@pytest.mark.parametrize("mode", ["dense", "paged", "paged_tma", "descale", "packed"])
@pytest.mark.parametrize("causal", [False, True])
def test_fp8(dtype, splits, mode, causal):
    torch.manual_seed(850)
    q = torch.randn(2, 65, 4, 512, device="cuda", dtype=torch.bfloat16).to(dtype)
    k, v = [
        torch.randn(2, 257, 2, 512, device="cuda", dtype=torch.bfloat16).to(dtype)
        for _ in range(2)
    ]
    qr, kr, vr = [x.double() for x in (q, k, v)]
    kwargs = dict(causal=causal, return_lse=True, num_splits=splits)
    if mode == "descale":
        scales = [
            torch.tensor([[0.75, 1.25], [1.5, 0.5]], device="cuda") for _ in range(3)
        ]
        kwargs.update(zip(("q_descale", "k_descale", "v_descale"), scales))
        qr *= scales[0].repeat_interleave(2, -1)[:, None, :, None]
        kr *= scales[1][:, None, :, None]
        vr *= scales[2][:, None, :, None]
    elif mode in ("paged", "paged_tma"):
        page_size = 96 if mode == "paged_tma" else 16
        width = (257 + page_size - 1) // page_size
        table = (
            torch.randperm(2 * width + 3, device="cuda")[: 2 * width]
            .reshape(2, width)
            .int()
        )
        pages = []
        for x in (k, v):
            # FP8 advanced-index assignment is not implemented in PyTorch; copy byte views.
            pool = torch.zeros(
                2 * width + 3, page_size, 2, 512, device="cuda", dtype=dtype
            )
            padded = torch.zeros(
                2, width * page_size, 2, 512, device="cuda", dtype=dtype
            )
            padded.view(torch.uint8)[:, :257].copy_(x.view(torch.uint8))
            pool.view(torch.uint8)[table.long()] = padded.view(torch.uint8).reshape(
                2, width, page_size, 2, 512
            )
            pages.append(pool)
        k, v = pages
        kwargs.update(
            page_table=table,
            seqused_k=torch.tensor([257, 257], device="cuda", dtype=torch.int32),
            max_seqlen_k=257,
        )
    elif mode == "packed":
        q, k, v = q.flatten(0, 1), k.flatten(0, 1), v.flatten(0, 1)
        kwargs.update(
            cu_seqlens_q=torch.tensor([0, 65, 130], device="cuda", dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, 257, 514], device="cuda", dtype=torch.int32),
            max_seqlen_q=65,
            max_seqlen_k=257,
        )
    out, lse, *_ = _flash_attn_fwd(q, k, v, **kwargs)
    assert out.dtype == torch.bfloat16
    if mode == "packed":
        out = out.reshape(2, 65, 4, 512)
        lse = lse.reshape(4, 2, 65).permute(1, 0, 2)
    ref, rlse = reference(qr, kr, vr, causal=causal)
    # Inputs are compared after FP8 quantization. The remaining error includes
    # FP8 probability MMA operands and BF16 output rounding, matching upstream semantics.
    error = (out.double() - ref).abs()
    tolerance = 0.06 if dtype == torch.float8_e4m3fn else 0.14
    assert out.isfinite().all()
    assert error.square().mean().sqrt() <= tolerance * ref.square().mean().sqrt()
    assert error.max() <= tolerance * ref.abs().max()
    torch.testing.assert_close(lse.double(), rlse, atol=0.006, rtol=0.006)
    torch.cuda.synchronize()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("splits", [0, 2, 4, 8])
def test_splitkv_training(dtype, splits, monkeypatch):
    from flash_attn.cute import interface

    combine = interface._flash_attn_fwd_combine
    observed_splits = []

    @wraps(combine)
    def record_combine(out_partial, *args, **kwargs):
        observed_splits.append(out_partial.shape[0])
        return combine(out_partial, *args, **kwargs)

    monkeypatch.setattr(interface, "_flash_attn_fwd_combine", record_combine)
    torch.manual_seed(851)
    q = torch.randn(1, 65, 4, 512, device="cuda", dtype=dtype, requires_grad=True)
    k, v = [
        torch.randn(1, 513, 2, 512, device="cuda", dtype=dtype, requires_grad=True)
        for _ in range(2)
    ]
    qr, kr, vr = [x.detach().double().requires_grad_() for x in (q, k, v)]
    out, lse = flash_attn_func(q, k, v, causal=True, num_splits=splits, return_lse=True)
    assert len(observed_splits) == 2
    assert all(n == splits if splits > 0 else n > 1 for n in observed_splits)
    ref, rlse = reference(qr, kr, vr, causal=True)
    check(out, ref, dtype)
    torch.testing.assert_close(lse.double(), rlse, atol=0.003, rtol=0.003)
    grad = torch.randn_like(out)
    out.backward(grad)
    ref.backward(grad.double())
    for actual, expected in zip((q, k, v), (qr, kr, vr)):
        check(actual.grad, expected.grad, dtype)


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_fp8_backward_rejected(dtype):
    x = (
        torch.randn(1, 16, 2, 512, device="cuda", dtype=torch.bfloat16)
        .to(dtype)
        .requires_grad_()
    )
    with pytest.raises(NotImplementedError, match="forward-only"):
        flash_attn_func(x, x, x)


@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float16, torch.float8_e4m3fn, torch.float8_e5m2]
)
@pytest.mark.parametrize("dim", [64, 128, 256])
def test_existing_precision(dim, dtype):
    torch.manual_seed(852)
    q = torch.randn(1, 65, 4, dim, device="cuda", dtype=torch.bfloat16).to(dtype)
    k, v = [
        torch.randn(1, 129, 4, dim, device="cuda", dtype=torch.bfloat16).to(dtype)
        for _ in range(2)
    ]
    out, _ = flash_attn_func(q, k, v, causal=True)
    ref, _ = reference(q, k, v, causal=True)
    if dtype in (torch.bfloat16, torch.float16):
        check(out, ref, dtype)
    else:
        tolerance = 0.06 if dtype == torch.float8_e4m3fn else 0.14
        error = (out.double() - ref).abs()
        assert out.isfinite().all()
        assert error.square().mean().sqrt() <= tolerance * ref.square().mean().sqrt()
        assert error.max() <= tolerance * ref.abs().max()


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_fp8_graph(dtype):
    torch.manual_seed(853)
    q = torch.randn(1, 65, 4, 512, device="cuda", dtype=torch.bfloat16).to(dtype)
    k, v = [
        torch.randn(1, 257, 2, 512, device="cuda", dtype=torch.bfloat16).to(dtype)
        for _ in range(2)
    ]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            flash_attn_func(q, k, v, causal=True, num_splits=4)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out, _ = flash_attn_func(q, k, v, causal=True, num_splits=4)
    q.copy_(torch.randn(q.shape, device="cuda", dtype=torch.bfloat16).to(dtype))
    graph.replay()
    ref, _ = reference(q, k, v, causal=True)
    tolerance = 0.06 if dtype == torch.float8_e4m3fn else 0.14
    error = (out.double() - ref).abs()
    assert out.isfinite().all()
    assert error.square().mean().sqrt() <= tolerance * ref.square().mean().sqrt()
    assert error.max() <= tolerance * ref.abs().max()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("causal", [False, True])
def test_dense_inference_without_lse(dtype, causal):
    # Training tests already cover saved LSE and gradients. Exercise the absent
    # LSE descriptor, non-default scale and partial tiles in inference instead.
    torch.manual_seed(910)
    q = torch.randn(2, 137, 4, 512, device="cuda", dtype=dtype)
    k, v = [torch.randn(2, 193, 2, 512, device="cuda", dtype=dtype) for _ in range(2)]
    scale = 0.03125
    supplied = torch.empty_like(q)
    out, lse, _, _ = interface._flash_attn_fwd(
        q,
        k,
        v,
        causal=causal,
        softmax_scale=scale,
        return_lse=False,
        out=supplied,
    )
    assert lse is None
    assert out.data_ptr() == supplied.data_ptr()
    expected, _ = reference(q, k, v, causal=causal, scale=scale)
    check(out, expected, dtype)
    torch.cuda.synchronize()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("return_lse", [False, True])
@pytest.mark.parametrize("sq", [2, 3, 7, 32, 64, 65, 129, 193])
def test_empty_output_cta(sq, return_lse, causal, dtype):
    torch.manual_seed(7)
    q = torch.randn(2, sq, 4, 512, device="cuda", dtype=dtype)
    k, v = [torch.randn(2, 137, 2, 512, device="cuda", dtype=dtype) for _ in range(2)]
    expected, expected_lse = reference(q, k, v, causal=causal)
    # Repeated launches expose writes from an empty CTA racing valid output.
    for _ in range(10):
        out, lse, *_ = interface._flash_attn_fwd(
            q,
            k,
            v,
            causal=causal,
            return_lse=return_lse,
        )
        check(out, expected, dtype)
        if return_lse:
            torch.testing.assert_close(
                lse.double(), expected_lse, atol=0.003, rtol=0.003
            )
        else:
            assert lse is None


@pytest.mark.parametrize("output_name", ["dq", "dk", "dv"])
@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_large_gradient_stride(output_name, ratio, dtype):
    batch, stride = 6, 500_000_000
    free, _ = torch.cuda.mem_get_info()
    if free < 6 * 1024**3:
        pytest.skip("Requires 6 GiB of free device memory for the strided output")
    torch.manual_seed(7)
    q = torch.randn(batch, 3, 2 * ratio, 512, device="cuda", dtype=dtype)
    k, v = [torch.randn(batch, 3, 2, 512, device="cuda", dtype=dtype) for _ in range(2)]
    qr, kr, vr = [x.double().requires_grad_() for x in (q, k, v)]
    expected_out, expected_lse = reference(qr, kr, vr)
    # Isolate backward address calculation from the forward implementation.
    out = expected_out.detach().to(dtype).contiguous()
    lse = expected_lse.detach().float().contiguous()
    dout = torch.randn_like(out)
    expected = torch.autograd.grad(expected_out, (qr, kr, vr), dout.double())
    template = q if output_name == "dq" else k
    backing = torch.empty(
        (batch - 1) * stride + template[0].numel(), device="cuda", dtype=dtype
    )
    supplied = torch.as_strided(
        backing, template.shape, (stride, *template.stride()[1:])
    )
    for _ in range(10):
        actual = interface._flash_attn_bwd(
            q, k, v, out, dout, lse, **{output_name: supplied}
        )
        assert (
            actual[("dq", "dk", "dv").index(output_name)].data_ptr()
            == supplied.data_ptr()
        )
        for value, expected_value in zip(actual, expected):
            check(value, expected_value, dtype)
    torch.cuda.synchronize()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("return_lse", [False, True])
def test_fused_forward_persistent_cache(tmp_path, monkeypatch, dtype, return_lse):
    from flash_attn.cute import flash_fwd_sm100_hd512 as fused
    from flash_attn.cute.cache_utils import JITPersistentCache

    cache = JITPersistentCache(tmp_path)
    monkeypatch.setattr(fused, "_forward_cache", cache)
    torch.manual_seed(7)
    q = torch.randn(1, 3, 2, 512, device="cuda", dtype=dtype)
    k, v = [torch.randn(1, 137, 2, 512, device="cuda", dtype=dtype) for _ in range(2)]
    expected, expected_lse = reference(q, k, v)
    first, _, *_ = interface._flash_attn_fwd(q, k, v, return_lse=return_lse)
    check(first, expected, dtype)
    assert list(tmp_path.glob("*.o"))
    monkeypatch.setattr(fused, "_forward_cache", JITPersistentCache(tmp_path))

    def unexpected_compile(*args, **kwargs):
        pytest.fail("The second call must reload the compiled function from disk")

    monkeypatch.setattr(fused.cute, "compile", unexpected_compile)
    out, lse, *_ = interface._flash_attn_fwd(q, k, v, return_lse=return_lse)
    check(out, expected, dtype)
    if return_lse:
        torch.testing.assert_close(lse.double(), expected_lse, atol=0.003, rtol=0.003)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "batch,seq,hq,hk,causal,maxsk",
    [
        (2, 257, 2, 2, False, None),
        (2, 257, 2, 2, True, None),
        (2, 257, 4, 2, False, None),
        (2, 257, 4, 2, True, None),
        (1, 4103, 8, 2, False, None),
        (1, 4096, 8, 2, False, 4224),
    ],
)
def test_native_gradient_write_isolation(dtype, batch, seq, hq, hk, causal, maxsk, monkeypatch):
    """A gradient kernel must not write another kernel's output buffer."""
    from flash_attn.cute import flash_bwd_sm100_hd512 as native

    torch.manual_seed(921)
    q = torch.randn(batch, seq, hq, 512, device="cuda", dtype=dtype)
    k, v = [torch.randn(batch, seq, hk, 512, device="cuda", dtype=dtype) for _ in range(2)]
    out, lse, _, _ = _flash_attn_fwd(q, k, v, causal=causal, return_lse=True)
    dout = torch.randn_like(out)
    cache = native._native_cache
    seen = []

    class CheckedCache:
        def __contains__(self, key):
            return key in cache

        def __setitem__(self, key, value):
            cache[key] = value

        def __getitem__(self, key):
            compiled = cache[key]
            mode = key[1]
            target = ("dq", "dk", "dv").index(mode)

            def checked(*args):
                outputs = args[6:9]
                if mode == "dq":
                    for i, tensor in enumerate(outputs):
                        tensor.fill_(-17.0 - i)
                saved = {i: t.clone() for i, t in enumerate(outputs) if i != target}
                result = compiled(*args)
                for i, expected in saved.items():
                    torch.testing.assert_close(
                        outputs[i],
                        expected,
                        atol=0,
                        rtol=0,
                        msg=f"{mode} kernel modified gradient buffer {i}",
                    )
                seen.append(mode)
                return result

            return checked

    monkeypatch.setattr(native, "_native_cache", CheckedCache())
    actual = _flash_attn_bwd(
        q, k, v, out, dout, lse, causal=causal, max_seqlen_k=maxsk
    )
    assert seen == ["dq", "dk", "dv"]
    qr, kr, vr = [x.double().requires_grad_() for x in (q, k, v)]
    ref, _ = reference(qr, kr, vr, causal=causal)
    expected = torch.autograd.grad(ref, (qr, kr, vr), dout.double())
    for grad, expected_grad in zip(actual, expected):
        check(grad, expected_grad, dtype)


def check_grouped_error(name, actual, exact, eager, dtype):
    """Apply the upstream relative-error criterion to a grouped gradient."""
    assert torch.isfinite(actual).all()
    error = (actual.double() - exact).abs()
    eager_error = (eager.double() - exact).abs()
    rounding = 2 * torch.finfo(dtype).eps * exact.abs()
    check_tensor_vs_ref(
        name, actual.double(), exact, eager.double(), atol=rounding.max().item()
    )
    assert error.mean() <= 2 * eager_error.mean() + rounding.mean(), name


def fixed_threshold_failures(actual, exact, dtype):
    atol, rtol = (0.01, 0.03) if dtype == torch.bfloat16 else (0.002, 0.005)
    error = (actual.double() - exact).abs()
    return int((error > atol + rtol * exact.abs()).sum().item())


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    ("ratio", "head_group"), [(2, 2), (4, 2), (4, 4), (8, 2), (8, 4)]
)
def test_grouped_native_backward(monkeypatch, head_group, dtype, causal, ratio):
    torch.manual_seed(193)
    q = torch.randn(2, 257, ratio * 2, 512, device="cuda", dtype=dtype)
    k, v = [
        torch.randn(2, 257, 2, 512, device="cuda", dtype=dtype) for _ in range(2)
    ]
    qr, kr, vr = [x.double().requires_grad_() for x in (q, k, v)]
    exact_out, _ = reference(qr, kr, vr, causal=causal)
    out, lse, *_ = _flash_attn_fwd(q, k, v, causal=causal, return_lse=True)
    dout = torch.randn_like(out)
    exact = torch.autograd.grad(exact_out, (qr, kr, vr), dout.double())
    qe, ke, ve = [x.detach().requires_grad_() for x in (q, k, v)]
    eager_out, _ = attention_ref(qe, ke, ve, causal=causal, upcast=False)
    eager = torch.autograd.grad(eager_out, (qe, ke, ve), dout)

    monkeypatch.setattr(native, "_select_head_group", lambda *args: 1)
    baseline = _flash_attn_bwd(q, k, v, out, dout, lse, causal=causal)
    baseline_failures = [
        fixed_threshold_failures(actual, expected, dtype)
        for actual, expected in zip(baseline, exact)
    ]

    monkeypatch.setattr(native, "_select_head_group", lambda *args: head_group)
    for _ in range(3):
        actual = _flash_attn_bwd(q, k, v, out, dout, lse, causal=causal)
        for name, result, expected, eager_result, baseline_count in zip(
            ("dq", "dk", "dv"), actual, exact, eager, baseline_failures
        ):
            check_grouped_error(name, result, expected, eager_result, dtype)
            assert fixed_threshold_failures(result, expected, dtype) <= baseline_count, name
    torch.cuda.synchronize()


@pytest.mark.parametrize("head_group", [2, 4])
def test_grouped_softcap_window_write_isolation(monkeypatch, head_group):
    monkeypatch.setattr(native, "_select_head_group", lambda *args: head_group)
    torch.manual_seed(194)
    dtype = torch.bfloat16
    ratio = 8
    q = torch.randn(1, 137, ratio * 2, 512, device="cuda", dtype=dtype)
    k, v = [
        torch.randn(1, 149, 2, 512, device="cuda", dtype=dtype) for _ in range(2)
    ]
    opts = {
        "causal": True,
        "softcap": 12.0,
        "window_size_left": 32,
        "window_size_right": 48,
    }
    out, lse, *_ = _flash_attn_fwd(q, k, v, return_lse=True, **opts)
    dout = torch.randn_like(out)
    cache = native._native_cache
    seen = []

    class CheckedCache:
        def __contains__(self, key):
            return key in cache

        def __setitem__(self, key, value):
            cache[key] = value

        def __getitem__(self, key):
            compiled = cache[key]
            mode = key[1]
            target = ("dq", "dk", "dv").index(mode)

            def checked(*args):
                outputs = args[6:9]
                if mode == "dq":
                    for idx, tensor in enumerate(outputs):
                        tensor.fill_(-17.0 - idx)
                saved = {
                    idx: tensor.clone()
                    for idx, tensor in enumerate(outputs)
                    if idx != target
                }
                result = compiled(*args)
                for idx, expected in saved.items():
                    torch.testing.assert_close(outputs[idx], expected, atol=0, rtol=0)
                seen.append(mode)
                return result

            return checked

    monkeypatch.setattr(native, "_native_cache", CheckedCache())
    actual = _flash_attn_bwd(q, k, v, out, dout, lse, **opts)
    assert seen == ["dq", "dk", "dv"]

    qr, kr, vr = [x.double().requires_grad_() for x in (q, k, v)]
    exact_out, _ = reference(
        qr, kr, vr, causal=True, softcap=12.0, window=(32, 48)
    )
    exact = torch.autograd.grad(exact_out, (qr, kr, vr), dout.double())
    qe, ke, ve = [x.detach().requires_grad_() for x in (q, k, v)]
    eager_out, _ = attention_ref(
        qe,
        ke,
        ve,
        causal=True,
        softcap=12.0,
        window_size=(32, 48),
        upcast=False,
    )
    eager = torch.autograd.grad(eager_out, (qe, ke, ve), dout)
    for name, result, expected, eager_result in zip(
        ("dq", "dk", "dv"), actual, exact, eager
    ):
        check_grouped_error(name, result, expected, eager_result, dtype)
    torch.cuda.synchronize()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("packed", [False, True])
def test_dynamic_batch_token_cache_reuse(monkeypatch, dtype, causal, packed):
    """Changing batch/token extents must reuse kernels and preserve FP64 accuracy."""
    from flash_attn.cute import flash_fwd_sm100_hd512 as fused

    monkeypatch.setattr(native, "_native_cache", {})
    monkeypatch.setattr(native, "_reduce_cache", {})
    monkeypatch.setattr(fused, "_forward_cache", {})
    torch.manual_seed(482)
    for batch, seq in [(1, 129), (2, 193), (3, 257), (1, 129)]:
        lengths = [seq - i * 7 for i in range(batch)]
        shape = (sum(lengths),) if packed else (batch, seq)
        q = torch.randn(*shape, 4, 512, device="cuda", dtype=dtype, requires_grad=True)
        k, v = [
            torch.randn(*shape, 2, 512, device="cuda", dtype=dtype, requires_grad=True)
            for _ in range(2)
        ]
        qr, kr, vr = [t.detach().double().requires_grad_() for t in (q, k, v)]
        if packed:
            cu = torch.tensor([0, *accumulate(lengths)], device="cuda", dtype=torch.int32)
            out, _ = flash_attn_varlen_func(
                q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu,
                max_seqlen_q=seq, max_seqlen_k=seq, causal=causal,
            )
            ref = torch.cat([
                reference(a[None], b[None], c[None], causal=causal)[0][0]
                for a, b, c in zip(qr.split(lengths), kr.split(lengths), vr.split(lengths))
            ])
        else:
            out, _ = flash_attn_func(q, k, v, causal=causal)
            ref, _ = reference(qr, kr, vr, causal=causal)
        grad = torch.randn_like(out)
        actual = torch.autograd.grad(out, (q, k, v), grad)
        expected = torch.autograd.grad(ref, (qr, kr, vr), grad.double())
        check(out, ref, dtype)
        qp, kp, vp = [t.detach().clone().requires_grad_() for t in (q, k, v)]
        if packed:
            eager = torch.cat([
                attention_ref(a[None], b[None], c[None], causal=causal,
                              upcast=False, reorder_ops=True)[0][0]
                for a, b, c in zip(qp.split(lengths), kp.split(lengths), vp.split(lengths))
            ])
        else:
            eager = attention_ref(qp, kp, vp, causal=causal, upcast=False, reorder_ops=True)[0]
        eager_grads = torch.autograd.grad(eager, (qp, kp, vp), grad)
        for name, a, r, pt in zip(("dq", "dk", "dv"), actual, expected, eager_grads):
            assert torch.isfinite(a).all()
            check_tensor_vs_ref(name, a.double(), r, pt.double())
        assert len(native._native_cache) == 3
        assert len(native._reduce_cache) == 1
        if not packed:
            assert len(fused._forward_cache) == 1


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "sq,sk,hq,hk,window",
    [
        (sq, sk, hq, hk, window)
        for sq, sk, hq, hk in [(137, 149, 4, 2), (137, 137, 2, 2), (137, 137, 4, 2)]
        for window in [(-1, 0), (-257, 0), (-2, 3), (3, -1), (-1, None), (None, -2),
                       (-257, None), (None, -257)]
    ] + [
        # Only the leading/trailing output tiles are empty in these square cases.
        (257, 257, hq, hk, window)
        for hq, hk in [(2, 2), (4, 2)]
        for window in [(-129, None), (None, -129)]
    ],
)
def test_signed_window_boundaries(dtype, window, sq, sk, hq, hk):
    """Negative offset bounds are literal after the public window resolver."""
    torch.manual_seed(712)
    q = torch.randn(1, sq, hq, 512, device="cuda", dtype=dtype, requires_grad=True)
    k, v = [
        torch.randn(1, sk, hk, 512, device="cuda", dtype=dtype, requires_grad=True)
        for _ in range(2)
    ]
    qr, kr, vr = [t.detach().double().requires_grad_() for t in (q, k, v)]
    scores = torch.einsum("bmhd,bnhd->bhmn", qr, kr.repeat_interleave(hq // hk, dim=2)) / (512 ** 0.5)
    center = torch.arange(sq, device="cuda")[:, None] + sk - sq
    keys = torch.arange(sk, device="cuda")[None, :]
    valid = torch.ones((sq, sk), device="cuda", dtype=torch.bool)
    if window[0] is not None:
        valid &= keys >= center - window[0]
    if window[1] is not None:
        valid &= keys <= center + window[1]
    masked = scores.masked_fill(~valid, -torch.inf)
    safe = torch.where(valid.any(-1)[None, None, :, None], masked, 0.)
    probs = safe.softmax(-1).masked_fill(~valid, 0.)
    ref = torch.einsum("bhmn,bnhd->bmhd", probs, vr.repeat_interleave(hq // hk, dim=2))
    out, lse = flash_attn_func(q, k, v, window_size=window, return_lse=True)
    grad = torch.randn_like(out)
    actual = torch.autograd.grad(out, (q, k, v), grad)
    expected = torch.autograd.grad(ref, (qr, kr, vr), grad.double())
    # Entire output tiles can have no active key/query tile with signed windows.
    # Supplied nonzero buffers make missing zero writes independent of allocator reuse.
    supplied = [torch.full_like(t, 7) for t in (q, k, v)]
    written = _flash_attn_bwd(
        q, k, v, out, grad, lse,
        window_size_left=window[0], window_size_right=window[1],
        dq=supplied[0], dk=supplied[1], dv=supplied[2],
    )
    for result, buffer, autograd_result in zip(written, supplied, actual):
        assert result is buffer
        torch.testing.assert_close(result, autograd_result, atol=0, rtol=0)
    check(out, ref, dtype)
    qp, kp, vp = [t.detach().clone().requires_grad_() for t in (q, k, v)]
    eager = attention_ref(qp, kp, vp, window_size=window, upcast=False, reorder_ops=True)[0]
    eager_grads = torch.autograd.grad(eager, (qp, kp, vp), grad)
    for name, a, r, pt in zip(("dq", "dk", "dv"), actual, expected, eager_grads):
        assert torch.isfinite(a).all()
        check_tensor_vs_ref(name, a.double(), r, pt.double())


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_dynamic_cache_layout_fallback(monkeypatch, dtype):
    """Different compact orders and noncompact views cannot alias a cached layout."""
    from flash_attn.cute import flash_fwd_sm100_hd512 as fused

    monkeypatch.setattr(native, "_native_cache", {})
    monkeypatch.setattr(native, "_reduce_cache", {})
    monkeypatch.setattr(fused, "_forward_cache", {})
    torch.manual_seed(919)
    for layout, seq in [("dense", 33), ("transpose", 49), ("slice", 49), ("slice", 65)]:
        def tensor(heads):
            if layout == "transpose":
                t = torch.randn(2, heads, seq, 512, device="cuda", dtype=dtype).transpose(1, 2)
            elif layout == "slice":
                t = torch.randn(2, 2 * seq, heads, 512, device="cuda", dtype=dtype)[:, ::2]
            else:
                t = torch.randn(2, seq, heads, 512, device="cuda", dtype=dtype)
            return t.requires_grad_()

        q, k, v = tensor(4), tensor(2), tensor(2)
        qr, kr, vr = [t.detach().double().requires_grad_() for t in (q, k, v)]
        out, _ = flash_attn_func(q, k, v, causal=True)
        ref, _ = reference(qr, kr, vr, causal=True)
        g = torch.randn_like(out)
        actual = torch.autograd.grad(out, (q, k, v), g)
        exact = torch.autograd.grad(ref, (qr, kr, vr), g.double())
        qp, kp, vp = [t.detach().clone().requires_grad_() for t in (q, k, v)]
        pt = attention_ref(qp, kp, vp, causal=True, upcast=False, reorder_ops=True)[0]
        eager = torch.autograd.grad(pt, (qp, kp, vp), g)
        check(out, ref, dtype)
        for name, a, r, baseline in zip(("dq", "dk", "dv"), actual, exact, eager):
            assert torch.isfinite(a).all()
            check_tensor_vs_ref(name, a.double(), r, baseline.double())
    # The two noncompact token-strided shapes must retain separate static entries.
    assert len(native._native_cache) == 12
    assert len(fused._forward_cache) == 4


def test_dynamic_backward_persistent_cache(tmp_path, monkeypatch):
    """Reload dynamic kernels from disk, then use a different batch and length."""
    from flash_attn.cute.cache_utils import JITPersistentCache
    from flash_attn.cute import flash_fwd_sm100_hd512 as fused

    modules = [(native, "_native_cache"), (native, "_reduce_cache"), (fused, "_forward_cache")]
    for module, name in modules:
        monkeypatch.setattr(module, name, JITPersistentCache(tmp_path / name))
    torch.manual_seed(721)
    for i, (batch, seq) in enumerate([(1, 33), (2, 49)]):
        if i:
            for module, name in modules:
                monkeypatch.setattr(module, name, JITPersistentCache(tmp_path / name))
            original_compile = native.cute.compile

            def reject_recompile(kernel, *args, **kwargs):
                assert not isinstance(kernel, (native.NativeD512DqDk, native.ReduceD512Gqa,
                                               fused.FusedD512Forward)), "Dynamic kernel recompiled"
                return original_compile(kernel, *args, **kwargs)

            monkeypatch.setattr(native.cute, "compile", reject_recompile)
        q = torch.randn(batch, seq, 4, 512, device="cuda", dtype=torch.float16, requires_grad=True)
        k, v = [torch.randn(batch, seq, 2, 512, device="cuda", dtype=q.dtype, requires_grad=True)
                for _ in range(2)]
        qr, kr, vr = [t.detach().double().requires_grad_() for t in (q, k, v)]
        out, _ = flash_attn_func(q, k, v)
        ref, _ = reference(qr, kr, vr)
        grad = torch.randn_like(out)
        actual = torch.autograd.grad(out, (q, k, v), grad)
        expected = torch.autograd.grad(ref, (qr, kr, vr), grad.double())
        check(out, ref, q.dtype)
        for a, r in zip(actual, expected):
            check(a, r, q.dtype)
    assert len(list((tmp_path / "_native_cache").glob("*.o"))) == 3
    assert len(list((tmp_path / "_reduce_cache").glob("*.o"))) == 1


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("layout", ["packed_q", "packed_k", "packed_both"])
@pytest.mark.parametrize("causal", [False, True])
def test_mixed_varlen_used_lengths_lse_grad(dtype, layout, causal):
    """Independently packed Q/K, unused rows and LSE gradients compose correctly."""
    torch.manual_seed(921)
    packed_q, packed_k = layout != "packed_k", layout != "packed_q"
    q_storage = [137, 19, 73] if packed_q else [137] * 3
    k_storage = [101, 41, 11] if packed_k else [101] * 3
    used_q, used_k = [129, 0, 65], [97, 33, 0]
    q_shape = (sum(q_storage),) if packed_q else (3, q_storage[0])
    k_shape = (sum(k_storage),) if packed_k else (3, k_storage[0])
    q = torch.randn(*q_shape, 4, 512, device="cuda", dtype=dtype, requires_grad=True)
    k, v = [torch.randn(*k_shape, 2, 512, device="cuda", dtype=dtype, requires_grad=True)
            for _ in range(2)]
    cuq = torch.tensor([0, *accumulate(q_storage)], device="cuda", dtype=torch.int32) if packed_q else None
    cuk = torch.tensor([0, *accumulate(k_storage)], device="cuda", dtype=torch.int32) if packed_k else None
    uq = torch.tensor(used_q, device="cuda", dtype=torch.int32)
    uk = torch.tensor(used_k, device="cuda", dtype=torch.int32)
    out, lse = flash_attn_varlen_func(
        q, k, v, cu_seqlens_q=cuq, cu_seqlens_k=cuk, seqused_q=uq, seqused_k=uk,
        max_seqlen_q=max(q_storage), max_seqlen_k=max(k_storage),
        causal=causal, softcap=5.0, return_lse=True,
    )
    qr, kr, vr = [t.detach().double().requires_grad_() for t in (q, k, v)]
    qp, kp, vp = [t.detach().clone().requires_grad_() for t in (q, k, v)]

    def rows(t, storage, packed):
        return t.split(storage) if packed else t.unbind(0)

    outs = rows(out, q_storage, packed_q)
    lses = lse.split(q_storage, dim=-1) if packed_q else lse.unbind(0)
    ref_rows = list(zip(rows(qr, q_storage, packed_q), rows(kr, k_storage, packed_k),
                        rows(vr, k_storage, packed_k)))
    eager_rows = list(zip(rows(qp, q_storage, packed_q), rows(kp, k_storage, packed_k),
                          rows(vp, k_storage, packed_k)))
    actual_loss, ref_loss, eager_loss = 0., 0., 0.
    for i, (a, b, c) in enumerate(ref_rows):
        nq, nk = used_q[i], used_k[i]
        if nq == 0:
            continue
        ref, rlse = reference(a[None, :nq], b[None, :nk], c[None, :nk],
                              causal=causal, softcap=5.0)
        check(outs[i][:nq], ref[0], dtype)
        torch.testing.assert_close(lses[i][:, :nq].double(), rlse[0], atol=0.003, rtol=0.003)
        do = torch.randn_like(outs[i][:nq])
        dlse = torch.randn_like(lses[i][:, :nq])
        finite = torch.isfinite(rlse[0])
        actual_loss = actual_loss + (outs[i][:nq].float() * do.float()).sum()
        actual_loss = actual_loss + (lses[i][:, :nq][finite] * dlse[finite]).sum()
        ref_loss = ref_loss + (ref[0] * do.double()).sum() + (rlse[0][finite] * dlse.double()[finite]).sum()
        a, b, c = eager_rows[i]
        if nk:
            eager, _ = attention_ref(a[None, :nq], b[None, :nk], c[None, :nk],
                                      causal=causal, softcap=5.0, upcast=False, reorder_ops=True)
            # Use an independent input-precision reference for the output gradient;
            # the LSE-only term is evaluated in FP64 for both references.
            _, eager_lse = reference(a[None, :nq], b[None, :nk], c[None, :nk],
                                     causal=causal, softcap=5.0)
            eager_loss = eager_loss + (eager[0].float() * do.float()).sum()
            eager_loss = eager_loss + (eager_lse[0][finite] * dlse.double()[finite]).sum()
        else:
            eager_loss = eager_loss + (a.sum() + b.sum() + c.sum()) * 0.
    actual = torch.autograd.grad(actual_loss, (q, k, v))
    exact = torch.autograd.grad(ref_loss, (qr, kr, vr))
    eager = torch.autograd.grad(eager_loss, (qp, kp, vp))
    for name, result, expected, eager_result in zip(("dq", "dk", "dv"), actual, exact, eager):
        assert torch.isfinite(result).all()
        check_tensor_vs_ref(name, result.double(), expected, eager_result.double())
    torch.cuda.synchronize()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("use_reentrant", [False, True])
def test_checkpoint_training(dtype, use_reentrant):
    """Checkpoint recomputation preserves gradients and optimizer updates across lengths."""
    import copy
    from torch.utils.checkpoint import checkpoint

    torch.manual_seed(922)
    class AttentionBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = torch.nn.Linear(32, 4 * 512, bias=False)
            self.proj = torch.nn.Linear(2 * 512, 32, bias=False)

        def forward(self, x):
            z = self.qkv(x)
            q, k, v = z.split((1024, 512, 512), dim=-1)
            q = q.reshape(*x.shape[:-1], 2, 512)
            k, v = [t.reshape(*x.shape[:-1], 1, 512) for t in (k, v)]
            out, _ = flash_attn_func(q, k, v, causal=True)
            return self.proj(out.flatten(-2)) + x

    model = AttentionBlock().cuda()
    checked = copy.deepcopy(model)
    optimizers = [torch.optim.AdamW(m.parameters(), lr=1e-3) for m in (model, checked)]
    for batch, seq in [(1, 65), (2, 129), (1, 97)]:
        x = torch.randn(batch, seq, 32, device="cuda", requires_grad=True)
        xc = x.detach().clone().requires_grad_()
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=dtype):
            out = model(x)
            recomputed = checkpoint(checked, xc, use_reentrant=use_reentrant)
            loss = out.float().square().mean()
            checked_loss = recomputed.float().square().mean()
        loss.backward()
        checked_loss.backward()
        torch.testing.assert_close(recomputed, out, atol=0, rtol=0)
        torch.testing.assert_close(xc.grad, x.grad, atol=0, rtol=0)
        for p, pc in zip(model.parameters(), checked.parameters()):
            assert torch.isfinite(p.grad).all()
            torch.testing.assert_close(pc.grad, p.grad, atol=0, rtol=0)
        for optimizer in optimizers:
            optimizer.step()
        for p, pc in zip(model.parameters(), checked.parameters()):
            torch.testing.assert_close(pc, p, atol=0, rtol=0)
    torch.cuda.synchronize()


@pytest.mark.parametrize("batch,seq", [(16, 8192), (16, 8193), (1, 131072), (1, 131073)])
def test_large_gqa_reduction_count(batch, seq):
    """GQA reduction must address gradients at and above the signed-int32 limit."""
    import cutlass.cute as cute
    from flash_attn.cute.cute_dsl_utils import to_compact_dynamic_tensor

    # 40 GiB of input/output buffers; allow room for the CUDA context and allocator.
    torch.cuda.empty_cache()
    if torch.cuda.mem_get_info()[0] < 48 * 1024**3:
        pytest.skip("Large-index regression requires 48 GiB of free device memory")
    shape = (batch, seq, 32, 512)
    pk = torch.ones(batch, seq, 64, 512, device="cuda", dtype=torch.float32)
    pv = torch.full_like(pk, 2.)
    dk = torch.full(shape, -7., device="cuda", dtype=torch.float16)
    dv = torch.full_like(dk, -7.)
    assert dk.numel() >= 2**31
    tensors = (pk, pv, dk, dv)
    args = [to_compact_dynamic_tensor(t, 16, (0, 1)) for t in tensors]
    fn = cute.compile(
        native.ReduceD512Gqa(2), *args,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )
    fn(*tensors)
    # Sample all batches/heads at the beginning, middle and end, including indices
    # beyond 2**31; avoid allocating another full-size reference tensor.
    rows = [0, seq // 2, seq - 1]
    assert (dk[:, rows] == 2.).all()
    assert (dv[:, rows] == 4.).all()
    torch.cuda.synchronize()
