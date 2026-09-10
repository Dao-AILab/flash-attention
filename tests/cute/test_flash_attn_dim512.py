"""SM100 symmetric head-dimension 512 correctness and boundary regressions."""

from itertools import accumulate
import pytest
import torch
from flash_attn.cute import flash_attn_func, flash_attn_varlen_func
from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd
from functools import wraps
from flash_attn.cute import interface

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="Requires SM100",
)


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
