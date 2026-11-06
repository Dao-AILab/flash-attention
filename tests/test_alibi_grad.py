"""Gradient coverage for the existing FA2 ALiBi slopes argument."""

import pytest
import torch

from flash_attn import (
    flash_attn_func,
    flash_attn_kvpacked_func,
    flash_attn_qkvpacked_func,
    flash_attn_varlen_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_varlen_qkvpacked_func,
)

pytestmark = pytest.mark.skipif(
    torch.version.hip is not None, reason="ALiBi slope gradients require the CUDA backend"
)

APIS = ("separate", "kvpacked", "qkvpacked", "varlen", "varlen_kvpacked", "varlen_qkvpacked")


def _call_attention(api, q, k, v, slopes, q_lengths, k_lengths, **kwargs):
    kwargs = dict(alibi_slopes=slopes, **kwargs)
    if not api.startswith("varlen"):
        if api == "qkvpacked":
            return flash_attn_qkvpacked_func(torch.stack((q, k, v), dim=2), **kwargs)
        if api == "kvpacked":
            return flash_attn_kvpacked_func(q, torch.stack((k, v), dim=2), **kwargs)
        return flash_attn_func(q, k, v, **kwargs)

    cu_q = torch.tensor([0, *q_lengths], dtype=torch.int32, device=q.device).cumsum(
        0, dtype=torch.int32
    )
    cu_k = torch.tensor([0, *k_lengths], dtype=torch.int32, device=q.device).cumsum(
        0, dtype=torch.int32
    )
    if api == "varlen_qkvpacked":
        return flash_attn_varlen_qkvpacked_func(
            torch.stack((q, k, v), dim=1), cu_q, max(q_lengths), **kwargs
        )
    args = (cu_q, cu_k, max(q_lengths), max(k_lengths))
    if api == "varlen_kvpacked":
        return flash_attn_varlen_kvpacked_func(
            q, torch.stack((k, v), dim=1), *args, **kwargs
        )
    return flash_attn_varlen_func(q, k, v, *args, **kwargs)


def _reference_one(
    q, k, v, slopes, *, softmax_scale=None, causal=False, window_size=(-1, -1),
    softcap=0.0, dropout_p=0.0, dropout_mask=None,
):
    """FP32 oracle, including bottom-right alignment and fully masked rows."""
    sq, heads, dim = q.shape
    sk = k.shape[0]
    if sq == 0 or sk == 0:
        # Keep every input in the autograd graph, with a zero derivative.
        return q * 0 + (k.sum() + v.sum() + slopes.sum()) * 0
    k = k.repeat_interleave(heads // k.shape[1], dim=1)
    v = v.repeat_interleave(heads // v.shape[1], dim=1)
    scale = dim ** -0.5 if softmax_scale is None else softmax_scale
    scores = torch.einsum("qhd,khd->hqk", q * scale, k)
    if softcap > 0:
        scores = softcap * torch.tanh(scores / softcap)
    rows = torch.arange(sq, device=q.device)[:, None] + sk - sq
    cols = torch.arange(sk, device=q.device)[None, :]
    # ALiBi is added after softcap, and is not multiplied by softmax_scale.
    scores = scores - slopes[:, None, None] * (rows - cols).abs()
    left, right = window_size
    if causal:
        right = 0
    allowed = torch.ones((sq, sk), dtype=torch.bool, device=q.device)
    if left >= 0:
        allowed &= cols >= rows - left
    if right >= 0:
        allowed &= cols <= rows + right
    any_key = allowed.any(dim=-1, keepdim=True)
    scores = scores.masked_fill(~allowed, -torch.inf)
    # Avoid NaNs in both softmax and its derivative for fully masked rows.
    scores = torch.where(any_key, scores, torch.zeros_like(scores))
    probabilities = scores.softmax(dim=-1).masked_fill(~allowed, 0)
    if dropout_mask is not None:
        probabilities = probabilities.masked_fill(~dropout_mask, 0)
    return torch.einsum("hqk,khd->qhd", probabilities, v) / (1 - dropout_p)


def _reference(api, q, k, v, slopes, q_lengths, k_lengths, dropout_mask=None, **kwargs):
    varlen = api.startswith("varlen")
    q_parts = q.split(q_lengths) if varlen else q.unbind()
    k_parts = k.split(k_lengths) if varlen else k.unbind()
    v_parts = v.split(k_lengths) if varlen else v.unbind()
    outputs = []
    for batch, (qi, ki, vi) in enumerate(zip(q_parts, k_parts, v_parts)):
        mask = None if dropout_mask is None else dropout_mask[
            batch, :, :q_lengths[batch], :k_lengths[batch]
        ]
        outputs.append(_reference_one(
            qi, ki, vi, slopes if slopes.ndim == 1 else slopes[batch],
            dropout_mask=mask, **kwargs,
        ))
    return torch.cat(outputs) if varlen else torch.stack(outputs)


def _dropout_mask(api, s_dmask, q_lengths, k_lengths, dim, causal, window_size):
    # Reuse the established FA2 test helper for the kernel's returned S layout.
    from test_flash_attn import convert_flash_attn_S_to_softmax

    max_q, max_k = max(q_lengths), max(k_lengths)
    q_padding = k_padding = None
    if api.startswith("varlen"):
        q_padding = torch.arange(max_q, device=s_dmask.device)[None, :] < torch.tensor(
            q_lengths, device=s_dmask.device
        )[:, None]
        k_padding = torch.arange(max_k, device=s_dmask.device)[None, :] < torch.tensor(
            k_lengths, device=s_dmask.device
        )[:, None]
    return convert_flash_attn_S_to_softmax(
        s_dmask, max_q, max_k, q_padding, k_padding, dim, True,
        causal=causal, window_size=window_size,
    ) >= 0


def _assert_close(actual, expected, dtype, name):
    assert torch.isfinite(actual).all(), name
    error = (actual.float() - expected).abs()
    if error.numel() == 0:
        return
    # Reduction outputs such as dslopes can cross zero; bound both the maximum
    # and average error by the corresponding FP32 reference magnitude.
    rtol, atol = (0.01, 0.002) if dtype == torch.float16 else (0.05, 0.01)
    assert error.max() <= atol + rtol * expected.abs().max(), (
        name, "max error", error.max().item(), "reference max", expected.abs().max().item()
    )
    assert error.mean() <= atol + rtol * expected.abs().mean(), (
        name, "mean error", error.mean().item(), "reference mean", expected.abs().mean().item()
    )


def _run_case(
    api, dtype, batched_slopes, *, only_slopes=False, causal=False,
    window_size=(-1, -1), softcap=0.0, dropout_p=0.0, softmax_scale=None,
    dim=64, empty=False, deterministic=False, repeats=1,
):
    torch.manual_seed(1977)
    varlen = api.startswith("varlen")
    qkvpacked = api.endswith("qkvpacked")
    if empty:
        assert varlen
        q_lengths = (0, 7, 19)
        k_lengths = q_lengths if qkvpacked else (11, 0, 37)
    elif deterministic:
        q_lengths = (193, 71, 5) if varlen else (193,) * 3
        k_lengths = (257, 29, 17) if varlen else (257,) * 3
    else:
        q_lengths = (67, 19, 5) if varlen else (67,) * 3
        k_lengths = (131, 11, 37) if varlen else (131,) * 3
        if qkvpacked:
            k_lengths = q_lengths
    batch, heads = len(q_lengths), 4
    kv_heads = heads if qkvpacked else 2
    q_shape = (sum(q_lengths), heads, dim) if varlen else (batch, q_lengths[0], heads, dim)
    k_shape = (sum(k_lengths), kv_heads, dim) if varlen else (
        batch, k_lengths[0], kv_heads, dim
    )
    q = torch.randn(q_shape, device="cuda", dtype=dtype).requires_grad_(not only_slopes)
    k = torch.randn(k_shape, device="cuda", dtype=dtype).requires_grad_(not only_slopes)
    v = torch.randn(k_shape, device="cuda", dtype=dtype).requires_grad_(not only_slopes)
    # Non-contiguous batch stride is valid when the last dimension is contiguous.
    slopes = torch.rand(
        (batch, heads + 2) if batched_slopes else (heads,), device="cuda", dtype=torch.float32
    ) * 0.05
    if batched_slopes:
        slopes = slopes[:, :heads]
    slopes.requires_grad_()
    q_ref, k_ref, v_ref, slopes_ref = [
        x.detach().float().clone().requires_grad_(x.requires_grad) for x in (q, k, v, slopes)
    ]
    options = dict(
        softmax_scale=softmax_scale, causal=causal, window_size=window_size,
        softcap=softcap, dropout_p=dropout_p,
    )
    result = _call_attention(
        api, q, k, v, slopes, q_lengths, k_lengths,
        deterministic=deterministic, return_attn_probs=dropout_p > 0, **options,
    )
    if dropout_p > 0:
        out, _, s_dmask = result
        mask = _dropout_mask(api, s_dmask, q_lengths, k_lengths, dim, causal, window_size)
    else:
        out, mask = result, None
    assert out.requires_grad
    out_ref = _reference(
        api, q_ref, k_ref, v_ref, slopes_ref, q_lengths, k_lengths,
        dropout_mask=mask, **options,
    )
    _assert_close(out, out_ref, dtype, "out")
    dout = torch.randn_like(out)
    inputs = (slopes,) if only_slopes else (q, k, v, slopes)
    refs = (slopes_ref,) if only_slopes else (q_ref, k_ref, v_ref, slopes_ref)
    names = ("dslopes",) if only_slopes else ("dq", "dk", "dv", "dslopes")
    expected = torch.autograd.grad(out_ref, refs, dout.float())
    previous = None
    for iteration in range(repeats):
        gradients = torch.autograd.grad(out, inputs, dout, retain_graph=iteration + 1 < repeats)
        assert gradients[-1].dtype == torch.float32
        assert gradients[-1].shape == slopes.shape
        for name, actual, ref in zip(names, gradients, expected):
            _assert_close(actual, ref, dtype, name)
        if previous is not None:
            for name, actual, prior in zip(names, gradients, previous):
                assert torch.equal(actual, prior), name
        previous = gradients
    if empty and batched_slopes:
        # Neither a sequence with no Q nor one with no K contributes to slopes.
        for batch_idx, (sq, sk) in enumerate(zip(q_lengths, k_lengths)):
            if sq == 0 or sk == 0:
                assert torch.count_nonzero(gradients[-1][batch_idx]) == 0


@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batched_slopes", [False, True], ids=["shared", "batched"])
def test_alibi_slopes_grad(api, dtype, batched_slopes):
    _run_case(api, dtype, batched_slopes)


@pytest.mark.parametrize(
    "api,dtype,batched_slopes,only_slopes,causal,window_size,softcap,scale,dim",
    [
        pytest.param("separate", torch.float16, False, True, False, (-1, -1), 0.0, 0.37, 32, id="dense-only-slopes-hdim32"),
        pytest.param("kvpacked", torch.bfloat16, True, True, True, (-1, -1), 0.0, 0.2, 128, id="kvpacked-only-slopes-hdim128"),
        pytest.param("qkvpacked", torch.float16, False, True, False, (11, 7), 0.0, 0.19, 64, id="qkvpacked-local-only-slopes-hdim64"),
        pytest.param("varlen", torch.bfloat16, True, True, True, (17, -1), 0.0, 0.25, 32, id="varlen-local-only-slopes-hdim32"),
        pytest.param("varlen_kvpacked", torch.float16, False, True, False, (-1, 9), 0.0, 0.13, 128, id="varlen-kvpacked-only-slopes-hdim128"),
        pytest.param("varlen_qkvpacked", torch.bfloat16, True, True, True, (-1, -1), 0.0, 0.35, 64, id="varlen-qkvpacked-only-slopes-hdim64"),
        pytest.param("separate", torch.float16, True, False, True, (13, 0), 2.5, 0.6, 64, id="dense-local-softcap-hdim64"),
        pytest.param("varlen", torch.bfloat16, False, False, False, (17, 11), 1.5, 0.4, 64, id="varlen-local-softcap-hdim64"),
        pytest.param("kvpacked", torch.bfloat16, False, False, False, (-1, -1), 3.0, 0.25, 64, id="kvpacked-softcap-hdim64"),
        pytest.param("varlen_qkvpacked", torch.float16, True, False, True, (-1, -1), 2.0, 0.3, 64, id="varlen-qkvpacked-softcap-hdim64"),
    ],
)
def test_alibi_slopes_modes(
    api, dtype, batched_slopes, only_slopes, causal, window_size, softcap, scale, dim
):
    _run_case(
        api, dtype, batched_slopes, only_slopes=only_slopes, causal=causal,
        window_size=window_size, softcap=softcap, softmax_scale=scale, dim=dim,
    )


@pytest.mark.parametrize("api", APIS[3:])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_alibi_slopes_empty_sequences(api, dtype):
    _run_case(api, dtype, True, empty=True, causal=True)


@pytest.mark.parametrize("api", APIS)
def test_alibi_slopes_dropout(api):
    # Softcap + dropout is rejected by the existing FA2 API.
    _run_case(
        api, torch.float16, True, dropout_p=0.17, causal=True,
        window_size=(31, 0), softmax_scale=0.23,
    )


@pytest.mark.parametrize("api", ["separate", "varlen"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_alibi_slopes_deterministic(api, dtype):
    _run_case(
        api, dtype, False, deterministic=True, repeats=3,
        window_size=(73, 23), softmax_scale=0.2,
    )



def _small_inputs(api, *, only_slopes=False, trainable_slopes=True):
    varlen = api.startswith("varlen")
    qkvpacked = api.endswith("qkvpacked")
    q_lengths = (17, 9) if varlen else (17, 17)
    k_lengths = q_lengths if qkvpacked else ((23, 13) if varlen else (23, 23))
    heads, dim, batch = 4, 64, len(q_lengths)
    kv_heads = heads if qkvpacked else 2
    q_shape = (sum(q_lengths), heads, dim) if varlen else (batch, q_lengths[0], heads, dim)
    k_shape = (sum(k_lengths), kv_heads, dim) if varlen else (
        batch, k_lengths[0], kv_heads, dim
    )
    q = torch.randn(q_shape, device="cuda", dtype=torch.float16).requires_grad_(not only_slopes)
    k = torch.randn(k_shape, device="cuda", dtype=torch.float16).requires_grad_(not only_slopes)
    v = torch.randn(k_shape, device="cuda", dtype=torch.float16).requires_grad_(not only_slopes)
    slopes = (torch.rand(heads, device="cuda", dtype=torch.float32) * 0.05).requires_grad_(
        trainable_slopes
    )
    return (q, k, v, slopes), q_lengths, k_lengths


@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize("trainable_slopes", [False, True], ids=["frozen", "trainable"])
def test_alibi_saved_slopes_version(api, trainable_slopes):
    # Even a frozen bias is needed to recompute probabilities during backward.
    # Mutating it must fail instead of silently using different attention scores.
    torch.manual_seed(1977)
    inputs, q_lengths, k_lengths = _small_inputs(api, trainable_slopes=trainable_slopes)
    q, k, v, slopes = inputs
    out = _call_attention(api, q, k, v, slopes, q_lengths, k_lengths)
    with torch.no_grad():
        slopes.add_(0.01)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        torch.autograd.grad(out, q, torch.ones_like(out))


@pytest.mark.skipif(
    not hasattr(torch.library, "custom_op"), reason="FA2 compile support requires PyTorch >= 2.4"
)
@pytest.mark.parametrize("api", ["separate", "varlen"])
@pytest.mark.parametrize("gradient_mode", ["all", "slopes_only", "frozen_slopes"])
def test_alibi_slopes_compile_fullgraph(api, gradient_mode):
    # Select with -k compile_fullgraph to run the compiler checks separately.
    # Training forces AOTAutograd/Inductor to functionalize the optional mutated
    # backward buffer; a forward-only compile would not exercise that schema.
    torch.manual_seed(1977)
    inputs, q_lengths, k_lengths = _small_inputs(
        api, only_slopes=gradient_mode == "slopes_only",
        trainable_slopes=gradient_mode != "frozen_slopes",
    )
    options = dict(causal=True, softmax_scale=0.23)

    def attention(q, k, v, slopes):
        return _call_attention(api, q, k, v, slopes, q_lengths, k_lengths, **options)

    torch._dynamo.reset()
    try:
        compiled = torch.compile(attention, backend="inductor", fullgraph=True)
        for _ in range(2):
            references = tuple(
                x.detach().float().clone().requires_grad_(x.requires_grad) for x in inputs
            )
            out = compiled(*inputs)
            expected = _reference(api, *references, q_lengths, k_lengths, **options)
            _assert_close(out, expected, torch.float16, "compiled out")
            dout = torch.randn_like(out)
            requested = tuple(x for x in inputs if x.requires_grad)
            requested_refs = tuple(x for x in references if x.requires_grad)
            gradients = torch.autograd.grad(out, requested, dout)
            expected_gradients = torch.autograd.grad(expected, requested_refs, dout.float())
            names = tuple(
                name for name, value in zip(("dq", "dk", "dv", "dslopes"), inputs)
                if value.requires_grad
            )
            for name, actual, ref in zip(names, gradients, expected_gradients):
                _assert_close(actual, ref, torch.float16, "compiled " + name)
            # Reuse the compiled graph with new bias values and a new dout.
            with torch.no_grad():
                inputs[-1].add_(0.01)
    finally:
        torch._dynamo.reset()



@pytest.mark.parametrize("batched_slopes", [False, True], ids=["shared", "batched"])
def test_alibi_slopes_ragged_memory(batched_slopes):
    # One long sequence should not pad the slope-gradient workspace for every
    # short sequence in a packed batch. Compare incremental allocated memory
    # against the same attention backward with frozen slopes.
    torch.manual_seed(1977)
    batch, heads, dim = 1024, 2, 64
    lengths = [65536] + [64] * (batch - 1)
    total_k = sum(lengths)
    q = torch.randn(batch, heads, dim, device="cuda", dtype=torch.float16, requires_grad=True)
    k = torch.randn(total_k, heads, dim, device="cuda", dtype=torch.float16, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    slopes = torch.rand((batch, heads) if batched_slopes else (heads,), device="cuda") * 0.05
    cu_q = torch.arange(batch + 1, dtype=torch.int32, device="cuda")
    cu_k = torch.tensor([0, *lengths], dtype=torch.int32, device="cuda").cumsum(0, dtype=torch.int32)
    dout = torch.randn_like(q)

    def peak_memory(trainable):
        s = slopes.detach().requires_grad_(trainable)
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        out = flash_attn_varlen_func(q, k, v, cu_q, cu_k, 1, max(lengths), alibi_slopes=s)
        grads = torch.autograd.grad(out, (q, k, v, s) if trainable else (q, k, v), dout)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - before
        assert all(torch.isfinite(g).all() for g in grads)
        return peak

    peak_memory(False)
    peak_memory(True)
    frozen_peak = peak_memory(False)
    trainable_peak = peak_memory(True)
    # A batch-padded FP32 partial buffer alone would use 128 MiB here.
    assert trainable_peak - frozen_peak < 16 * 1024**2, (frozen_peak, trainable_peak)


@pytest.mark.parametrize("api", ["separate", "varlen"])
@pytest.mark.parametrize("empty_side", ["q", "k"])
@pytest.mark.parametrize("only_slopes", [False, True], ids=["all-gradients", "slopes-only"])
def test_alibi_slopes_globally_empty(api, empty_side, only_slopes):
    # A globally empty side must also work when slopes alone need gradients.
    # Packed batches with merely one empty segment exercise a different launch.
    batch, heads, dim = 2, 2, 64
    q_lengths = (0, 0) if empty_side == "q" else (7, 7)
    k_lengths = (0, 0) if empty_side == "k" else (11, 11)
    varlen = api == "varlen"
    q_shape = (sum(q_lengths), heads, dim) if varlen else (batch, q_lengths[0], heads, dim)
    k_shape = (sum(k_lengths), heads, dim) if varlen else (batch, k_lengths[0], heads, dim)
    q = torch.randn(q_shape, device="cuda", dtype=torch.float16).requires_grad_(not only_slopes)
    k = torch.randn(k_shape, device="cuda", dtype=torch.float16).requires_grad_(not only_slopes)
    v = torch.randn_like(k, requires_grad=not only_slopes)
    slopes = torch.full(
        (batch, heads) if only_slopes else (heads,), 0.01,
        device="cuda", dtype=torch.float32, requires_grad=True,
    )
    out = _call_attention(api, q, k, v, slopes, q_lengths, k_lengths)
    requested = (slopes,) if only_slopes else (q, k, v, slopes)
    gradients = torch.autograd.grad(out, requested, torch.ones_like(out))
    assert torch.count_nonzero(out) == 0
    assert gradients[-1].shape == slopes.shape
    assert gradients[-1].dtype == torch.float32
    for gradient in gradients:
        assert torch.count_nonzero(gradient) == 0
