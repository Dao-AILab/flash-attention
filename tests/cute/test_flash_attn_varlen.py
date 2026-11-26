import os
from typing import Optional
import pytest

import torch
import torch.nn.functional as F
from flash_attn.cute import flash_attn_varlen_func
from flash_attn.cute.interface import _flash_attn_bwd
from flash_attn.cute.testing import is_fake_mode, maybe_fake_tensor_mode

@pytest.mark.parametrize("B", [1, 7, 20])
@pytest.mark.parametrize("H", [1, 4, 6])
@pytest.mark.parametrize("D", [64, 128, 256])
@pytest.mark.parametrize("min_seq_len", [1, 32, 128])
@pytest.mark.parametrize("max_seq_len", [8, 64, 2048])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("softmax_scale", [None, 0.1])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
def test_varlen(
    B,
    H,
    D,
    min_seq_len,
    max_seq_len,
    causal,
    softmax_scale,
    dtype,
    mha_type,
):
    if min_seq_len > max_seq_len:
        pytest.skip("Skipping min_seq_len > max_seq_len")

    q, k, v, cu_seqlens_q, cu_seqlens_k, total_q, total_k = generate_varlen_args(
        batch_size=B,
        n_heads=H,
        d_head=D,
        min_len=min_seq_len,
        max_len=max_seq_len,
        mha_type=mha_type,
        dtype=dtype
    )

    ok = check_varlen_vs_torch_flash(
        q, k, v,
        cu_seqlens_q, cu_seqlens_k,
        total_q=total_q, total_k=total_k,
        softmax_scale=softmax_scale,
        causal=causal,
        mha_type=mha_type,
    )
    assert ok


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12,
    reason="requires SM12x backward",
)
@pytest.mark.parametrize("nheads_kv", [3, 1], ids=["gqa", "mqa"])
@maybe_fake_tensor_mode(os.getenv("FLASH_ATTENTION_FAKE_TENSOR", "0") == "1")
def test_varlen_unaligned_gqa_backward(nheads_kv):
    torch.manual_seed(576)
    lengths = (17, 23)
    cu_seqlens = torch.tensor([0, lengths[0], sum(lengths)], device="cuda", dtype=torch.int32)
    q, k, v = [
        torch.randn(sum(lengths), heads, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for heads in (6, nheads_kv, nheads_kv)
    ]
    out, _ = flash_attn_varlen_func(
        q, k, v, cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max(lengths), max_seqlen_k=max(lengths), causal=True,
    )
    dout = torch.randn_like(out)
    actual = (out, *torch.autograd.grad(out, (q, k, v), dout))

    if is_fake_mode():
        return

    def reference(dtype):
        inputs = [x.detach().to(dtype).requires_grad_() for x in (q, k, v)]
        outputs = []
        for qi, ki, vi in zip(*(x.split(lengths) for x in inputs)):
            qi = qi.transpose(0, 1)
            ki, vi = (x.repeat_interleave(6 // nheads_kv, dim=1).transpose(0, 1) for x in (ki, vi))
            scores = qi @ ki.transpose(-1, -2) / 8
            mask = torch.ones(scores.shape[-2:], device="cuda", dtype=torch.bool).triu(1)
            outputs.append((scores.masked_fill(mask, -torch.inf).softmax(-1) @ vi).transpose(0, 1))
        result = torch.cat(outputs)
        return (result, *torch.autograd.grad(result, inputs, dout.to(dtype)))

    low_precision, high_precision = reference(torch.bfloat16), reference(torch.float64)
    errors = {
        name: ((value.double() - ref).abs().max().item(), (low.double() - ref).abs().max().item())
        for name, value, low, ref in zip(("out", "dq", "dk", "dv"), actual, low_precision, high_precision)
    }
    # Allow for different accumulation and rounding orders, as in the existing attention tests.
    for name, (error, baseline_error) in errors.items():
        assert error <= 2 * baseline_error + 0.01, (name, errors)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] not in (10, 11),
    reason="SM100/SM110-only test",
)
@pytest.mark.parametrize("seqlens", ((128, 256), (129, 257)))
@pytest.mark.parametrize("causal", (False, True))
def test_hd256_varlen_epilogue(seqlens, causal):
    """HD256 uses aligned stores for full tiles and predicates ragged tails."""
    torch.manual_seed(20260814)
    num_heads_q, num_heads_kv, head_dim = 4, 1, 256
    total = sum(seqlens)
    cu_seqlens = torch.tensor(
        (0, seqlens[0], total),
        dtype=torch.int32,
        device="cuda",
    )
    q = torch.randn(
        total,
        num_heads_q,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
        requires_grad=True,
    )
    k = torch.randn(
        total,
        num_heads_kv,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
        requires_grad=True,
    )
    v = torch.randn_like(k, requires_grad=True)

    assert check_varlen_vs_torch_flash(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        total_q=total,
        total_k=total,
        causal=causal,
        mha_type="mqa",
    )


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] not in (10, 11),
    reason="SM100/SM110-only test",
)
@pytest.mark.parametrize("mode", ("plain", "causal", "local"))
@pytest.mark.parametrize("num_heads_q", (2, 8))
@pytest.mark.parametrize("head_dim", (64, 128, 256))
@pytest.mark.parametrize("poison", ("q", "do", "both"))
def test_varlen_backward_ignores_nan_padded_q_or_do(mode, num_heads_q, head_dim, poison):
    torch.manual_seed(1234)
    cu_seqlens_q = torch.tensor((0, 263, 526), device="cuda", dtype=torch.int32)
    cu_seqlens_k = torch.tensor((0, 256, 512), device="cuda", dtype=torch.int32)
    q0 = torch.randn(640, num_heads_q, head_dim, device="cuda", dtype=torch.bfloat16)
    k0 = torch.randn(512, 2, head_dim, device="cuda", dtype=torch.bfloat16)
    v0 = torch.randn_like(k0)
    grad_out = torch.randn_like(q0)

    def backward(padded_q, padded_do):
        q = q0.clone()
        q[526:] = padded_q
        q.requires_grad_()
        k = k0.clone().requires_grad_()
        v = v0.clone().requires_grad_()
        grad = grad_out.clone()
        grad[526:] = padded_do
        out, _ = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=263,
            max_seqlen_k=256,
            causal=mode == "causal",
            window_size=(64, 16) if mode == "local" else (None, None),
        )
        out.backward(grad)
        return q.grad[:526], k.grad, v.grad

    dq_zero, dk_zero, dv_zero = backward(0, 0)
    dq_nan, dk_nan, dv_nan = backward(
        float("nan") if poison in ("q", "both") else 0,
        float("nan") if poison in ("do", "both") else 0,
    )
    assert torch.isfinite(dq_nan).all()
    assert torch.isfinite(dk_nan).all()
    assert torch.isfinite(dv_nan).all()
    torch.testing.assert_close(dq_nan, dq_zero, rtol=0, atol=0)
    torch.testing.assert_close(dk_nan, dk_zero, rtol=1e-2, atol=2e-3)
    torch.testing.assert_close(dv_nan, dv_zero, rtol=1e-2, atol=2e-3)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] not in (10, 11),
    reason="SM100/SM110-only test",
)
@pytest.mark.parametrize("packed", (False, True))
@pytest.mark.parametrize("num_heads_q", (2, 8))
def test_backward_seqused_q_ignores_nan_padding(packed, num_heads_q):
    torch.manual_seed(1234)
    lengths = (100, 256, 37, 200, 129)
    batch, capacity, d = len(lengths), 256, 128
    q = torch.randn(batch, capacity, num_heads_q, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, capacity, 2, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    seqused_q = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    valid = torch.arange(capacity, device="cuda")[None, :] < seqused_q[:, None]
    kwargs = dict(seqused_q=seqused_q, max_seqlen_q=capacity, max_seqlen_k=capacity)
    if packed:
        cu = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * capacity
        kwargs.update(cu_seqlens_q=cu, cu_seqlens_k=cu)
        q, k, v = (t.flatten(0, 1) for t in (q, k, v))
        valid = valid.flatten()

    out, lse = flash_attn_varlen_func(q, k, v, return_lse=True, **kwargs)
    dout = torch.randn_like(out)

    def backward(q_input, dout_input):
        return _flash_attn_bwd(q_input, k, v, out, dout_input, lse, **kwargs)[:3]

    reference = backward(q, dout)
    for poison in ("q", "do", "both"):
        q_nan, dout_nan = q.clone(), dout.clone()
        if poison in ("q", "both"):
            q_nan[~valid] = float("nan")
        if poison in ("do", "both"):
            dout_nan[~valid] = float("nan")
        grads = backward(q_nan, dout_nan)
        for name, got, want in zip(("dq", "dk", "dv"), grads, reference):
            if name == "dq":
                got, want = got[valid], want[valid]
            assert torch.isfinite(got).all(), f"{poison}: {name} has NaN/Inf"
            torch.testing.assert_close(got, want, rtol=1e-2, atol=2e-3)


def _padded_cu_seqlens(lengths, capacity):
    """cu_seqlens with `capacity` slots; unused trailing slots repeat the last offset."""
    cu = torch.zeros(capacity + 1, dtype=torch.int32)
    cu[1 : len(lengths) + 1] = torch.tensor(lengths).cumsum(0)
    cu[len(lengths) + 1 :] = cu[len(lengths)]
    return cu.cuda()


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] not in (10, 11),
    reason="SM100/SM110-only test",
)
@pytest.mark.parametrize(
    "lengths,capacity,heads,num_splits",
    [
        # fewer tiles than persistent CTAs; interior empty sequence; padded tail spans several
        # 31-batch decoder groups
        ((3000, 0, 517, 4096, 9), 100, (2, 1), 1),
        # packed GQA, ~500 tiles: several dynamically claimed tiles per CTA
        ((4096, 2048, 1024, 1024), 32, (16, 4), 1),
        # split-KV with folded per-batch splits
        ((2048, 0, 1024), 40, (4, 4), 3),
    ],
)
def test_varlen_dynamic_scheduler_metadata_reuse(lengths, capacity, heads, num_splits):
    """Reused scheduler metadata (DynamicPersistentVarlenScheduler) across unsynchronized calls:
    the kernel resets the tile semaphore at exit."""
    from flash_attn.cute.interface import get_scheduler_metadata

    torch.manual_seed(0)
    hq, hkv = heads
    total = sum(lengths)
    cu = _padded_cu_seqlens(lengths, capacity)
    k = torch.randn(total, hkv, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    qs = [torch.randn(total, hq, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    kwargs = dict(cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=max(lengths),
                  max_seqlen_k=max(lengths), causal=True, num_splits=num_splits)

    def metadata():
        return get_scheduler_metadata(max(lengths), max(lengths), hq, hkv, 128,
                                      num_splits=num_splits, causal=True,
                                      cu_seqlens_q=cu, cu_seqlens_k=cu)

    refs = [flash_attn_varlen_func(q, k, v, scheduler_metadata=metadata(), **kwargs)[0]
            for q in qs]
    if num_splits == 1:  # same per-tile math as the single-tile scheduler
        single_tile = flash_attn_varlen_func(qs[0], k, v, disable_scheduler_metadata=True, **kwargs)
        assert torch.equal(refs[0], single_tile[0])
    md = metadata()
    assert md.tile_count_semaphore is not None
    outs = [flash_attn_varlen_func(q, k, v, scheduler_metadata=md, **kwargs)[0] for q in qs]
    for out, ref in zip(outs, refs):
        assert torch.equal(out, ref)
    assert md.tile_count_semaphore.item() == 0


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] not in (10, 11),
    reason="SM100/SM110-only test",
)
def test_varlen_dynamic_scheduler_many_empty_slots():
    """Request slots beyond the row count add no tiles, so the dynamic scheduler's tile bound
    must not count them: one token in 131,072 slots with 128 heads is 128 tiles."""
    from flash_attn.cute.prepare_scheduler import SchedulerMetadataTensorsTorch

    torch.manual_seed(0)
    q = torch.randn(1, 128, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 128, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    cu = _padded_cu_seqlens((1,), 1 << 17)
    semaphore = torch.zeros(1, dtype=torch.int32, device="cuda")
    md = SchedulerMetadataTensorsTorch(tile_count_semaphore=semaphore)
    kwargs = dict(cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=1, max_seqlen_k=1, causal=True)
    out = flash_attn_varlen_func(q, k, v, scheduler_metadata=md, **kwargs)[0]
    assert torch.equal(out, flash_attn_varlen_func(q, k, v, **kwargs)[0])
    assert semaphore.item() == 0


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] not in (10, 11),
    reason="SM100/SM110-only test",
)
def test_varlen_dynamic_scheduler_caller_semaphore_graph_replay():
    """The caller semaphore's initial zero fill is outside graph capture. Each replay must reset
    it for the next replay, including when there is no work."""
    from flash_attn.cute.prepare_scheduler import SchedulerMetadataTensorsTorch

    torch.manual_seed(0)
    capacity, slots = 2048, 32
    q = torch.randn(capacity, 8, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(capacity, 2, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    semaphore = torch.zeros(1, dtype=torch.int32, device="cuda")
    md = SchedulerMetadataTensorsTorch(tile_count_semaphore=semaphore)
    replays = [(1000, 0, 517, 9), (), (300, 1, 1, 1), ()]
    cu = _padded_cu_seqlens(replays[0], slots)
    kwargs = dict(max_seqlen_q=capacity, max_seqlen_k=capacity, causal=True, num_splits=1)

    def step():
        return flash_attn_varlen_func(
            q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu, scheduler_metadata=md, **kwargs
        )[0]

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        step()  # compile outside capture
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        static_out = step()
    assert semaphore.item() == 0
    for lengths in replays:
        cu.copy_(_padded_cu_seqlens(lengths, slots))
        graph.replay()
        torch.cuda.synchronize()
        assert semaphore.item() == 0
        total = sum(lengths)
        if total:
            real_cu = _padded_cu_seqlens(lengths, len(lengths))
            ref = flash_attn_varlen_func(
                q, k, v, cu_seqlens_q=real_cu, cu_seqlens_k=real_cu,
                disable_scheduler_metadata=True, **kwargs,
            )[0]
            assert torch.equal(static_out[:total], ref[:total])


def test_varlen_padded_cu_seqlens_tail():
    """Padded cu_seqlens slots (repeating the last offset) do not change forward or backward."""
    torch.manual_seed(0)
    lengths = (700, 0, 1300, 64)
    total = sum(lengths)
    q = torch.randn(total, 4, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(total, 2, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    assert check_varlen_vs_torch_flash(
        q, k, v,
        cu_seqlens_q=_padded_cu_seqlens(lengths, 100),
        cu_seqlens_k=_padded_cu_seqlens(lengths, 100),
        total_q=total, total_k=total, causal=True, mha_type="gqa",
    )


def check_varlen_vs_torch_flash(
    q, k, v,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    total_q=None,
    total_k=None,
    softmax_scale=None,
    causal=True,
    mha_type='mha',
    softcap=0.0,
    atol=3e-2,
    rtol=3e-2,
):
    assert q.requires_grad and k.requires_grad and v.requires_grad, "Set requires_grad=True on inputs"

    def clone_like(t):
        c = t.clone().detach().requires_grad_(True)
        return c

    q_fa, k_fa, v_fa = map(clone_like, (q, k, v))
    q_t,  k_t,  v_t  = map(clone_like, (q, k, v))

    if cu_seqlens_q is not None:
        cu_seqlens_q_fa = cu_seqlens_q.clone()
        cu_seqlens_q_t = cu_seqlens_q.clone()
    else:
        cu_seqlens_q_fa = None
        cu_seqlens_q_t = None

    if cu_seqlens_k is not None:
        cu_seqlens_k_fa = cu_seqlens_k.clone()
        cu_seqlens_k_t = cu_seqlens_k.clone()
    else:
        cu_seqlens_k_fa = None
        cu_seqlens_k_t = None

    out_fa, lse_fa = flash_attn_varlen_func(
        q_fa, k_fa, v_fa,
        cu_seqlens_q=cu_seqlens_q_fa,
        cu_seqlens_k=cu_seqlens_k_fa,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        softmax_scale=(1.0 / q.shape[-1]**0.5) if softmax_scale is None else softmax_scale,
        causal=causal,
        window_size=(None, None),
        learnable_sink=None,
        softcap=softcap,
        pack_gqa=None,
    )

    out_t = torch_flash_ref(
        q_t, k_t, v_t,
        cu_seqlens_q=cu_seqlens_q_t,
        cu_seqlens_k=cu_seqlens_k_t,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        total_q=total_q,
        total_k=total_k,
        softmax_scale=softmax_scale,
        causal=causal,
        mha_type=mha_type,
    )


    ok_fwd = torch.allclose(out_fa.float(), out_t.float(), atol=atol, rtol=rtol)
    if not ok_fwd:
        return False

    # Use the same upstream gradient to compare backward paths
    grad_out = torch.randn_like(out_fa)

    grad_fa = clone_like(grad_out)
    grad_t = clone_like(grad_out)

    # Cute bwd
    out_fa.backward(grad_fa, retain_graph=False)
    dq_fa, dk_fa, dv_fa = q_fa.grad, k_fa.grad, v_fa.grad

    # Ref bwd
    out_t.backward(grad_t, retain_graph=False)
    dq_t, dk_t, dv_t = q_t.grad, k_t.grad, v_t.grad

    # mean_ok_q = _stats("dQ", dq_fa, dq_t, atol=atol, rtol=rtol)
    # mean_ok_k = _stats("dK", dk_fa, dk_t, atol=atol, rtol=rtol)
    # mean_ok_v = _stats("dV", dv_fa, dv_t, atol=atol, rtol=rtol)

    # return mean_ok_q and mean_ok_k and mean_ok_v

    ok_q = torch.allclose(dq_fa.float(), dq_t.float(), atol=atol, rtol=rtol)
    ok_k = torch.allclose(dk_fa.float(), dk_t.float(), atol=atol, rtol=rtol)
    ok_v = torch.allclose(dv_fa.float(), dv_t.float(), atol=atol, rtol=rtol)
    # print(f"Close? dQ={ok_q}, dK={ok_k}, dV={ok_v}")
    return ok_q and ok_k and ok_v

def generate_varlen_args(
    batch_size=8,
    n_heads=16,
    d_head=128,
    min_len=32,
    max_len=64,
    mha_type="mha",
    dtype = torch.bfloat16,
):

    torch.manual_seed(0)
    device = "cuda"

    assert mha_type in ["mha", "mqa", "gqa"]

    lens_q = torch.randint(low=min_len, high=max_len + 1, size=(batch_size,))
    lens_k = lens_q.clone()

    cu_seqlens_q = torch.cat([torch.zeros(1, dtype=torch.int32), lens_q.cumsum(0)])
    cu_seqlens_k = torch.cat([torch.zeros(1, dtype=torch.int32), lens_k.cumsum(0)])

    total_q = cu_seqlens_q[-1]
    total_k = cu_seqlens_k[-1]

    cu_seqlens_q = cu_seqlens_q.contiguous().to(dtype=torch.int32, device=device)
    cu_seqlens_k = cu_seqlens_k.contiguous().to(dtype=torch.int32, device=device)

    if mha_type == "gqa":
        H = 3 * n_heads
        H_kv = n_heads
    elif mha_type == "mha":
        H = H_kv = n_heads
    else: # MQA
        H = n_heads
        H_kv = 1

    d_head_v = d_head

    q = torch.randn(total_q, H, d_head, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(total_k, H_kv, d_head, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(total_k, H_kv, d_head_v, device=device, dtype=dtype, requires_grad=True)

    return q, k, v, cu_seqlens_q, cu_seqlens_k, total_q, total_k

# Simple for loop over batch dim implementation
def torch_flash_ref(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor = None,
        cu_seqlens_k: torch.Tensor = None,
        total_q: int = 0,
        total_k: int = 0,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        **kwargs
    ):

    """
    q: (total_q, H, d) if cu_seqlens_q is not None, otherwise (B, L, H, d)
    k: (total_k, H_kv, d) if cu_seqlens_k is not None, otherwise (B, L, H_kv, d)
    v: (total_k, H_kv, d_v) if cu_seqlens_k is not None, otherwise (B, L, H_kv, d_v)
    cu_seqlens_q: (B+1,) int32, cumulative
    cu_seqlens_k: (B+1,) int32, cumulative

    seqused_q: (B+1,) int32
    seqused_k: (B+1,) int32
    Returns:
        out packed like q: (total_q, H, d_v)
    """

    if cu_seqlens_q is not None:
        assert cu_seqlens_q.dim() == 1
        assert total_q == q.shape[0]
        assert q.dim() == 3
        H = q.shape[1]
        B = cu_seqlens_q.shape[0] - 1
    else:
        assert q.dim() == 4
        H = q.shape[2]
        B = q.shape[0]

    if cu_seqlens_k is not None:
        assert cu_seqlens_k.dim() == 1
        assert total_k == k.shape[0] == v.shape[0]
        assert k.dim() == v.dim() == 3
        H_kv = k.shape[1]
        B_kv = cu_seqlens_k.shape[0] - 1
    else:
        assert k.dim() == v.dim() == 4
        assert k.shape[0] == v.shape[0]
        H_kv = k.shape[2]
        B_kv = k.shape[0]

    d = q.shape[-1]
    d_v = v.shape[-1]

    assert H_kv == v.shape[-2]
    assert d == k.shape[-1]
    assert B == B_kv

    assert q.device == k.device == v.device
    assert q.is_floating_point() and k.is_floating_point() and v.is_floating_point()

    device = q.device
    dtype = q.dtype

    hcseq_q = cu_seqlens_q.to(device='cpu')
    hcseq_k = cu_seqlens_k.to(device='cpu')

    outs = []
    for b in range(B):
        if hcseq_q is not None:
            q_start, q_end = int(hcseq_q[b]), int(hcseq_q[b+1])
            qb = q[q_start:q_end]
        else:
            qb = q[b]

        if hcseq_k is not None:
            k_start, k_end = int(hcseq_k[b]), int(hcseq_k[b+1])
            kb = k[k_start:k_end]
            vb = v[k_start:k_end]
        else:
            kb = k[b]
            vb = v[b]

        qb = qb.permute(1, 0, 2).unsqueeze(0)
        kb = kb.permute(1, 0, 2).unsqueeze(0)
        vb = vb.permute(1, 0, 2).unsqueeze(0)

        ob = F.scaled_dot_product_attention(
            qb, kb, vb,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=causal,
            scale=softmax_scale,
            enable_gqa=H_kv!=H
        )

        ob = ob.squeeze(0).permute(1, 0, 2).contiguous()
        outs.append(ob)

    if cu_seqlens_q is not None:
        out = torch.cat(outs, dim=0).to(device=device, dtype=dtype)
    else:
        out = torch.stack(outs, dim=0).to(device=device, dtype=dtype)
    return out

@torch.no_grad()
def _stats(name, a, b, atol, rtol):
    diff = (a - b).float()
    mean_abs = diff.abs().mean().item()
    mean_rel = (diff.abs().mean() / b.abs().clamp_min(1e-6).mean().item())
    print(f"{name}: mean_abs={mean_abs:.4e}, mean_rel={mean_rel:.4e}, sum_fa={a.sum()}, sum_ref={b.sum()}")
    return mean_abs < atol and mean_rel < rtol
