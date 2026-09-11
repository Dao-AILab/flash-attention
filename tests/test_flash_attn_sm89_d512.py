"""D512 forward and backward on Ada, including the Gemma E4B GQA shape."""
import pytest
import torch

from flash_attn import flash_attn_func, flash_attn_varlen_func

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9),
    reason="D512 kernels require SM89",
)


def reference(q, k, v, causal, scale):
    """FP64 reference with bottom-right causal alignment and empty-row handling."""
    sq, sk = q.shape[1], k.shape[1]
    if sk == 0:
        return q * 0 + (k.sum() + v.sum()) * 0
    groups = q.shape[2] // k.shape[2]
    scores = q.transpose(1, 2) @ k.repeat_interleave(groups, 2).transpose(1, 2).transpose(-1, -2)
    scores = scores * scale
    if causal:
        mask = torch.arange(sk, device=q.device)[None, :] <= (
            torch.arange(sq, device=q.device)[:, None] + sk - sq
        )
        valid = mask.any(-1, keepdim=True)
        scores = scores.masked_fill(~mask, -torch.inf)
        scores = torch.where(valid, scores, 0)
    p = scores.softmax(-1)
    if causal:
        p = p.masked_fill(~valid, 0)
    return (p @ v.repeat_interleave(groups, 2).transpose(1, 2)).transpose(1, 2)


def check_result(out, qkv, ref, refs, dtype):
    do = torch.randn_like(out)
    actual = (out, *torch.autograd.grad(out, qkv, do, retain_graph=True))
    expected = (ref, *torch.autograd.grad(ref, refs, do.double()))
    # Bound both the total error and the worst element relative to the signal.
    # FP64 computes both the reference output and its gradients.
    eps = torch.finfo(dtype).eps
    for a, b in zip(actual, expected):
        assert torch.isfinite(a).all()
        diff = (a.double() - b).abs()
        assert diff.norm() <= 3 * eps * b.norm() + 1e-6
        assert diff.max() <= 4 * eps * b.abs().max() + 1e-6


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("sq,sk", [(1, 97), (17, 31), (97, 129), (129, 97), (256, 256)])
@pytest.mark.parametrize("heads_k", [1, 2, 8])
def test_dense(dtype, causal, sq, sk, heads_k):
    torch.manual_seed(0)
    # Slicing also covers non-contiguous batch, sequence and head strides.
    q = (torch.randn(2, sq, 16, 512, device="cuda", dtype=dtype) * .2)[:, :, ::2].requires_grad_()
    k = (torch.randn(2, sk, heads_k * 2, 512, device="cuda", dtype=dtype) * .2)[:, :, ::2].requires_grad_()
    v = torch.randn_like(k).requires_grad_()
    refs = tuple(x.detach().double().requires_grad_() for x in (q, k, v))
    out = flash_attn_func(q, k, v, causal=causal, softmax_scale=1.0)
    ref = reference(*refs, causal, 1.0)
    check_result(out, (q, k, v), ref, refs, dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("lengths", [
    [(17, 31), (97, 129), (129, 97)],
    [(0, 0), (1, 97), (0, 31), (1, 1)],
    [(17, 0), (0, 0), (33, 65)],
])
def test_varlen(dtype, causal, deterministic, lengths):
    torch.manual_seed(1)
    lq, lk = zip(*lengths)
    cuq = torch.tensor([0, *lq], device="cuda", dtype=torch.int32).cumsum(0, dtype=torch.int32)
    cuk = torch.tensor([0, *lk], device="cuda", dtype=torch.int32).cumsum(0, dtype=torch.int32)
    q = (torch.randn(sum(lq), 8, 512, device="cuda", dtype=dtype) * .2).requires_grad_()
    k = (torch.randn(sum(lk), 2, 512, device="cuda", dtype=dtype) * .2).requires_grad_()
    v = torch.randn_like(k).requires_grad_()
    refs = tuple(x.detach().double().requires_grad_() for x in (q, k, v))
    out = flash_attn_varlen_func(q, k, v, cuq, cuk, max(lq), max(lk),
                                causal=causal, softmax_scale=1.0, deterministic=deterministic)
    qr = refs[0].split(lq)
    kr, vr = (x.split(lk) for x in refs[1:])
    ref = torch.cat([reference(a[None], b[None], c[None], causal, 1.0)[0]
                     for a, b, c in zip(qr, kr, vr)])
    check_result(out, (q, k, v), ref, refs, dtype)
    if deterministic:
        do = torch.randn_like(out)
        first = torch.autograd.grad(out, (q, k, v), do, retain_graph=True)
        second = torch.autograd.grad(out, (q, k, v), do)
        for a, b in zip(first, second):
            assert torch.equal(a, b)


@pytest.mark.parametrize("kwargs, message", [
    ({"dropout_p": 0.1}, "does not support dropout"),
    ({"softcap": 30.0}, "does not support softcap"),
    ({"window_size": (32, 0)}, "does not support local attention"),
    ({"alibi_slopes": True}, "does not support ALiBi"),
])
def test_unsupported_features(kwargs, message):
    q = torch.randn(1, 64, 2, 512, device="cuda", dtype=torch.float16)
    kwargs = dict(kwargs)
    if "alibi_slopes" in kwargs:
        kwargs["alibi_slopes"] = torch.ones(2, device="cuda")
    with pytest.raises(RuntimeError, match=message):
        flash_attn_func(q, q, q, **kwargs)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("scale", [512**-0.5, 1.0])
def test_long_sequence(dtype, scale):
    torch.manual_seed(8)
    q = torch.randn(1, 1024, 8, 512, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn(1, 1024, 2, 512, device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    refs = tuple(x.detach().double().requires_grad_() for x in (q, k, v))
    out = flash_attn_func(q, k, v, softmax_scale=scale, causal=True, deterministic=True)
    ref = reference(*refs, True, scale)
    check_result(out, (q, k, v), ref, refs, dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("varlen", [False, True])
@pytest.mark.parametrize("qkvpacked", [False, True])
def test_packed(dtype, varlen, qkvpacked):
    from flash_attn import (flash_attn_kvpacked_func, flash_attn_qkvpacked_func,
                            flash_attn_varlen_kvpacked_func, flash_attn_varlen_qkvpacked_func)

    torch.manual_seed(2)
    hk = 8 if qkvpacked else 2
    q = (torch.randn(1, 50, 8, 512, device="cuda", dtype=dtype) * .2).requires_grad_()
    k = (torch.randn(1, 50, hk, 512, device="cuda", dtype=dtype) * .2).requires_grad_()
    v = torch.randn_like(k, requires_grad=True)
    refs = tuple(x.detach().double().requires_grad_() for x in (q, k, v))
    packed = torch.stack((q, k, v) if qkvpacked else (k, v), dim=2)
    if varlen:
        cu = torch.tensor([0, 17, 50], device="cuda", dtype=torch.int32)
        if qkvpacked:
            out = flash_attn_varlen_qkvpacked_func(packed[0], cu, 33, causal=True, softmax_scale=1.0)
        else:
            out = flash_attn_varlen_kvpacked_func(q[0], packed[0], cu, cu, 33, 33,
                                                 causal=True, softmax_scale=1.0)
        out = out[None]
        ref = torch.cat([reference(*(x[:, start:end] for x in refs), True, 1.0)
                         for start, end in [(0, 17), (17, 50)]], dim=1)
    else:
        if qkvpacked:
            out = flash_attn_qkvpacked_func(packed, causal=True, softmax_scale=1.0)
        else:
            out = flash_attn_kvpacked_func(q, packed, causal=True, softmax_scale=1.0)
        ref = reference(*refs, True, 1.0)
    check_result(out, (q, k, v), ref, refs, dtype)


@pytest.mark.parametrize("feature", ["num_splits", "leftpad_k", "seqused_k", "block_table"])
def test_unsupported_varlen_features(feature):
    from flash_attn.flash_attn_interface import _flash_attn_varlen_forward

    q = torch.randn(17, 8, 512, device="cuda", dtype=torch.float16)
    k = torch.randn(31, 2, 512, device="cuda", dtype=torch.float16)
    cuq = torch.tensor([0, 17], device="cuda", dtype=torch.int32)
    cuk = torch.tensor([0, 31], device="cuda", dtype=torch.int32)
    value = 2 if feature == "num_splits" else torch.zeros(1, device="cuda", dtype=torch.int32)
    if feature == "block_table":
        k = torch.randn(1, 256, 2, 512, device="cuda", dtype=torch.float16)
        value = value.reshape(1, 1)
    with pytest.raises(RuntimeError, match="Head dimension 512 does not support"):
        _flash_attn_varlen_forward(q, k, k, cuq, cuk, 17, 31, 0.0, 1.0, True, **{feature: value})


def test_intermediate_head_dim_rejected():
    q = torch.randn(1, 17, 2, 384, device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError, match="up to 256, or exactly 512"):
        flash_attn_func(q, q, q)
