from typing import Optional
import pytest

import torch
import torch.nn.functional as F
from flash_attn.cute import flash_attn_varlen_func

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


# ---------------------------------------------------------------------------
# Scheduler scaling regression test
# ---------------------------------------------------------------------------
#
# Guards against re-introduction of the O(N^2) scheduling overhead in
# SingleTileVarlenScheduler that was fixed by precomputing per-batch tile
# counts and binary-searching them in the kernel. With the fix, per-CTA
# scheduling work is O(log N), so total scheduling cost grows linearly in N
# and the per-sequence forward time is roughly flat. Without the fix, total
# scheduling cost grows as O(N^2) and the per-sequence time grows linearly
# in N (~16x ratio between N=8192 and N=128 at seq_len=64 in measurements).
#
# The 2x ceiling is tight but tolerates the noise from L2 effects and from
# the smallest N having a slightly elevated per-seq cost (warmup, low
# occupancy). With the fix the observed ratio is ~1; without it, ~16.
#
# Marked @pytest.mark.slow so it doesn't slow the default test sweep.
@pytest.mark.slow
def test_varlen_scheduler_scaling():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    cap_major, _ = torch.cuda.get_device_capability()
    if cap_major < 8:
        pytest.skip("FA4 requires SM80+")

    seq_len = 64
    num_heads = 8
    head_dim = 128
    dtype = torch.bfloat16
    device = torch.device("cuda:0")
    num_seqs_sweep = [128, 512, 2048, 8192]

    def bench_fwd(num_seqs):
        total = num_seqs * seq_len
        cu = torch.arange(0, total + 1, seq_len, dtype=torch.int32, device=device)
        q = torch.randn(total, num_heads, head_dim, dtype=dtype, device=device) * 0.1
        k = torch.randn(total, num_heads, head_dim, dtype=dtype, device=device) * 0.1
        v = torch.randn(total, num_heads, head_dim, dtype=dtype, device=device) * 0.1

        def fwd():
            flash_attn_varlen_func(q, k, v,
                                   cu_seqlens_q=cu, cu_seqlens_k=cu,
                                   max_seqlen_q=seq_len, max_seqlen_k=seq_len)

        for _ in range(3):
            fwd()
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(10)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(10)]
        for s, e in zip(starts, ends):
            s.record(); fwd(); e.record()
        torch.cuda.synchronize()
        return sum(s.elapsed_time(e) for s, e in zip(starts, ends)) / 10

    per_seq_us = {}
    for n in num_seqs_sweep:
        ms = bench_fwd(n)
        per_seq_us[n] = ms * 1000.0 / n
        print(f"  N={n:>5d}  fwd_total={ms:7.3f} ms  per_seq={per_seq_us[n]:.3f} us")

    ratio = per_seq_us[8192] / per_seq_us[128]
    print(f"  per_seq[8192] / per_seq[128] = {ratio:.2f}x (expect ~1, fail at >2)")
    assert ratio < 2.0, (
        f"varlen scheduler appears to scale super-linearly: "
        f"per_seq_us at N=8192 is {ratio:.2f}x larger than at N=128 "
        f"(per_seq_us = {per_seq_us}). "
        f"Did the SingleTileVarlenScheduler.mTileCumsum binary-search regress?"
    )


# ---------------------------------------------------------------------------
# High-batch-count correctness with skewed seqlens
# ---------------------------------------------------------------------------
#
# The default test_varlen sweep tops out at B=20. The scheduler patch matters
# at high batch counts that *also* vary in seqlen (the pathological case is
# many short sequences). This stress test runs forward at B=1024 with a wide
# random length distribution and checks the output against PyTorch SDPA on a
# padded reference.
#
# Backward uses MHA (Hkv=Hq) to sidestep the inter-Q-head atomic-add
# non-determinism on dK/dV that the FA4 backward exhibits for GQA/MQA.
# Paper §3.2.4 ("Deterministic backward pass") describes this; it's
# pre-existing in the bwd kernel, unchanged by the scheduler patch.
@pytest.mark.slow
def test_varlen_high_batch_skewed_correctness():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    cap_major, _ = torch.cuda.get_device_capability()
    if cap_major < 8:
        pytest.skip("FA4 requires SM80+")
    # SM80 backward varlen is gated off in interface.py (asserts SM90+).
    test_backward = cap_major >= 9

    B = 1024
    H = 8
    D = 128
    min_len = 1
    max_len = 128
    dtype = torch.bfloat16
    device = torch.device("cuda:0")
    seed = 12345

    g = torch.Generator(device="cpu").manual_seed(seed)
    seqlens = torch.randint(min_len, max_len + 1, (B,), generator=g)
    total = int(seqlens.sum().item())
    cu = torch.zeros(B + 1, dtype=torch.int32, device=device)
    cu[1:] = seqlens.to(dtype=torch.int32, device=device).cumsum(0)

    torch.manual_seed(seed)
    q_pkd = (torch.randn(total, H, D, dtype=dtype, device=device) * 0.1).requires_grad_(True)
    k_pkd = (torch.randn(total, H, D, dtype=dtype, device=device) * 0.1).requires_grad_(True)
    v_pkd = (torch.randn(total, H, D, dtype=dtype, device=device) * 0.1).requires_grad_(True)

    out_fa = flash_attn_varlen_func(
        q_pkd, k_pkd, v_pkd,
        cu_seqlens_q=cu, cu_seqlens_k=cu,
        max_seqlen_q=max_len, max_seqlen_k=max_len,
    )
    if isinstance(out_fa, tuple):
        out_fa = out_fa[0]

    g_out = torch.randn_like(out_fa)
    if test_backward:
        out_fa.backward(g_out)

    # Reference: pad to (B, max_len, H, D), use SDPA with bool mask, gather valid tokens back.
    q_pad = torch.zeros(B, max_len, H, D, dtype=dtype, device=device)
    k_pad = torch.zeros(B, max_len, H, D, dtype=dtype, device=device)
    v_pad = torch.zeros(B, max_len, H, D, dtype=dtype, device=device)
    g_pad = torch.zeros(B, max_len, H, D, dtype=dtype, device=device)
    offset = 0
    for i, sl in enumerate(seqlens.tolist()):
        q_pad[i, :sl] = q_pkd.detach()[offset:offset + sl]
        k_pad[i, :sl] = k_pkd.detach()[offset:offset + sl]
        v_pad[i, :sl] = v_pkd.detach()[offset:offset + sl]
        g_pad[i, :sl] = g_out[offset:offset + sl]
        offset += sl

    q_pad = q_pad.requires_grad_(test_backward)
    k_pad = k_pad.requires_grad_(test_backward)
    v_pad = v_pad.requires_grad_(test_backward)
    sdpa_mask = torch.zeros(B, 1, max_len, max_len, dtype=torch.bool, device=device)
    for i, sl in enumerate(seqlens.tolist()):
        sdpa_mask[i, 0, :sl, :sl] = True

    out_ref = F.scaled_dot_product_attention(
        q_pad.transpose(1, 2), k_pad.transpose(1, 2), v_pad.transpose(1, 2),
        attn_mask=sdpa_mask,
    ).transpose(1, 2)
    if test_backward:
        out_ref.backward(g_pad)

    # Pack reference back to (total, H, D) layout for comparison
    out_ref_pkd = torch.zeros_like(out_fa)
    offset = 0
    for i, sl in enumerate(seqlens.tolist()):
        out_ref_pkd[offset:offset + sl] = out_ref[i, :sl]
        offset += sl
    if test_backward:
        dq_ref_pkd = torch.zeros_like(q_pkd)
        dk_ref_pkd = torch.zeros_like(k_pkd)
        dv_ref_pkd = torch.zeros_like(v_pkd)
        offset = 0
        for i, sl in enumerate(seqlens.tolist()):
            dq_ref_pkd[offset:offset + sl] = q_pad.grad[i, :sl]
            dk_ref_pkd[offset:offset + sl] = k_pad.grad[i, :sl]
            dv_ref_pkd[offset:offset + sl] = v_pad.grad[i, :sl]
            offset += sl

    # Tolerances mirror test_varlen (bf16 cross-impl): atol/rtol = 3e-2.
    def assert_close(a, b, name):
        diff = (a.float() - b.float()).abs()
        max_abs = diff.max().item()
        mean_rel = (diff.mean() / b.float().abs().clamp_min(1e-6).mean()).item()
        print(f"  {name}: max_abs={max_abs:.3e} mean_rel={mean_rel:.3e}")
        assert max_abs < 3e-2 and mean_rel < 3e-2, (
            f"{name} disagrees with SDPA: max_abs={max_abs}, mean_rel={mean_rel}"
        )

    print(f"  B={B} total_tokens={total} (mean {total/B:.1f}, "
          f"min {seqlens.min().item()}, max {seqlens.max().item()})  "
          f"test_backward={test_backward}")
    assert_close(out_fa, out_ref_pkd, "fwd")
    if test_backward:
        assert_close(q_pkd.grad, dq_ref_pkd, "dq ")
        assert_close(k_pkd.grad, dk_ref_pkd, "dk ")
        assert_close(v_pkd.grad, dv_ref_pkd, "dv ")


# Regression coverage for the binary-search path in SingleTileVarlenScheduler.
# The cumsum-on path only kicks in at num_batch >= 32; multi-m_block per batch
# (seqlen > tile_m) is needed to distinguish a wrong snap from a correct one.
# Parametrize for:
#   - B >= 63 + multi-m_block: q_stage cumsum miscomputation
#   - GQA/MQA: pack_gqa head-count remap (num_head_kv vs num_head_q)
#   - D in {64, 128}: different q_stage paths on SM100
@pytest.mark.parametrize("B", [63, 128])
@pytest.mark.parametrize("seq_len", [512, 2048])
@pytest.mark.parametrize("mha_type", ["mha", "gqa", "mqa"])
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
def test_varlen_scheduler_binary_search_correctness(B, seq_len, mha_type, D, causal):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    cap_major, _ = torch.cuda.get_device_capability()
    if cap_major < 8:
        pytest.skip("FA4 requires SM80+")

    H_q = 8
    if mha_type == "mha":
        H_kv = H_q
    elif mha_type == "gqa":
        H_kv = 2
    else:  # mqa
        H_kv = 1
    dtype = torch.bfloat16
    device = torch.device("cuda:0")
    seed = 4242

    g = torch.Generator(device="cpu").manual_seed(seed)
    seqlens = torch.full((B,), seq_len, dtype=torch.int32)
    # Vary a few of the seqlens slightly so we exercise the multi-m_block path
    # without making num_m_blocks identical for every batch.
    perturbation = torch.randint(0, 64, (B,), generator=g, dtype=torch.int32)
    seqlens = (seqlens - perturbation).clamp_min(1)
    total = int(seqlens.sum().item())
    cu = torch.zeros(B + 1, dtype=torch.int32, device=device)
    cu[1:] = seqlens.to(dtype=torch.int32, device=device).cumsum(0)

    torch.manual_seed(seed)
    q = (torch.randn(total, H_q, D, dtype=dtype, device=device) * 0.1)
    k = (torch.randn(total, H_kv, D, dtype=dtype, device=device) * 0.1)
    v = (torch.randn(total, H_kv, D, dtype=dtype, device=device) * 0.1)

    out_fa = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu, cu_seqlens_k=cu,
        max_seqlen_q=seq_len, max_seqlen_k=seq_len,
        causal=causal,
    )
    if isinstance(out_fa, tuple):
        out_fa = out_fa[0]

    # Reference: pad to (B, max_len, H, D), use SDPA, gather valid tokens back.
    max_len = seq_len
    q_pad = torch.zeros(B, max_len, H_q, D, dtype=dtype, device=device)
    k_pad = torch.zeros(B, max_len, H_kv, D, dtype=dtype, device=device)
    v_pad = torch.zeros(B, max_len, H_kv, D, dtype=dtype, device=device)
    offset = 0
    for i, sl in enumerate(seqlens.tolist()):
        q_pad[i, :sl] = q[offset:offset + sl]
        k_pad[i, :sl] = k[offset:offset + sl]
        v_pad[i, :sl] = v[offset:offset + sl]
        offset += sl

    sdpa_mask = torch.zeros(B, 1, max_len, max_len, dtype=torch.bool, device=device)
    causal_tri = torch.ones(max_len, max_len, dtype=torch.bool, device=device).tril()
    for i, sl in enumerate(seqlens.tolist()):
        if causal:
            sdpa_mask[i, 0, :sl, :sl] = causal_tri[:sl, :sl]
        else:
            sdpa_mask[i, 0, :sl, :sl] = True
    out_ref = F.scaled_dot_product_attention(
        q_pad.transpose(1, 2), k_pad.transpose(1, 2), v_pad.transpose(1, 2),
        attn_mask=sdpa_mask, enable_gqa=(H_q != H_kv),
    ).transpose(1, 2)
    out_ref_pkd = torch.zeros_like(out_fa)
    offset = 0
    for i, sl in enumerate(seqlens.tolist()):
        out_ref_pkd[offset:offset + sl] = out_ref[i, :sl]
        offset += sl

    a_f, b_f = out_fa.float(), out_ref_pkd.float()
    if torch.allclose(a_f, b_f, atol=3e-2, rtol=3e-2):
        return
    diff = (a_f - b_f).abs()
    max_abs = diff.max().item()
    # Per-batch breakdown — q_stage miscomputation makes batch 62+ diverge.
    per_batch_max = []
    offset = 0
    for i, sl in enumerate(seqlens.tolist()):
        d = (out_fa[offset:offset + sl].float()
             - out_ref_pkd[offset:offset + sl].float()).abs().max().item()
        per_batch_max.append((i, d))
        offset += sl
    bad = [(i, d) for i, d in per_batch_max if d > 3e-2]
    print(f"  bad batches (max_abs > 3e-2): count={len(bad)} first={bad[:5]} last={bad[-5:]}")
    raise AssertionError(
        f"B={B} seq_len={seq_len} mha={mha_type} D={D} causal={causal}: "
        f"fwd disagrees with SDPA, max_abs={max_abs}"
    )


# Bwd scheduler regression. Bwd parallelizes over K blocks, so the host cumsum
# is built from cu_seqlens_k with n_block_size and (on SM100, hd>=128) the bwd
# scheduler's cluster_shape_m is 2 — host cumsum must divide by 2 to match.
# High B + multi-m_block exercises the cumsum-on path.
@pytest.mark.parametrize("B", [63, 128])
@pytest.mark.parametrize("seq_len", [512, 2048])
@pytest.mark.parametrize("mha_type", ["mha", "gqa", "mqa"])
@pytest.mark.parametrize("causal", [False, True])
def test_varlen_scheduler_binary_search_correctness_bwd(B, seq_len, mha_type, causal):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    cap_major, _ = torch.cuda.get_device_capability()
    if cap_major < 9:
        pytest.skip("FA4 backward requires SM90+")

    H_q = 8
    if mha_type == "mha":
        H_kv = H_q
    elif mha_type == "gqa":
        H_kv = 2
    else:  # mqa
        H_kv = 1
    D = 128
    dtype = torch.bfloat16
    device = torch.device("cuda:0")
    seed = 9090

    g = torch.Generator(device="cpu").manual_seed(seed)
    seqlens = torch.full((B,), seq_len, dtype=torch.int32)
    perturbation = torch.randint(0, 64, (B,), generator=g, dtype=torch.int32)
    seqlens = (seqlens - perturbation).clamp_min(1)
    total = int(seqlens.sum().item())
    cu = torch.zeros(B + 1, dtype=torch.int32, device=device)
    cu[1:] = seqlens.to(dtype=torch.int32, device=device).cumsum(0)

    torch.manual_seed(seed)
    q = (torch.randn(total, H_q, D, dtype=dtype, device=device) * 0.1).requires_grad_(True)
    k = (torch.randn(total, H_kv, D, dtype=dtype, device=device) * 0.1).requires_grad_(True)
    v = (torch.randn(total, H_kv, D, dtype=dtype, device=device) * 0.1).requires_grad_(True)

    out_fa = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu, cu_seqlens_k=cu,
        max_seqlen_q=seq_len, max_seqlen_k=seq_len,
        causal=causal,
    )
    if isinstance(out_fa, tuple):
        out_fa = out_fa[0]

    g_out = torch.randn_like(out_fa)
    out_fa.backward(g_out)

    # Reference via SDPA on padded tensors.
    max_len = seq_len
    q_pad = torch.zeros(B, max_len, H_q, D, dtype=dtype, device=device)
    k_pad = torch.zeros(B, max_len, H_kv, D, dtype=dtype, device=device)
    v_pad = torch.zeros(B, max_len, H_kv, D, dtype=dtype, device=device)
    g_pad = torch.zeros(B, max_len, H_q, D, dtype=dtype, device=device)
    offset = 0
    for i, sl in enumerate(seqlens.tolist()):
        q_pad[i, :sl] = q.detach()[offset:offset + sl]
        k_pad[i, :sl] = k.detach()[offset:offset + sl]
        v_pad[i, :sl] = v.detach()[offset:offset + sl]
        g_pad[i, :sl] = g_out[offset:offset + sl]
        offset += sl
    q_pad.requires_grad_(True); k_pad.requires_grad_(True); v_pad.requires_grad_(True)

    sdpa_mask = torch.zeros(B, 1, max_len, max_len, dtype=torch.bool, device=device)
    causal_tri = torch.ones(max_len, max_len, dtype=torch.bool, device=device).tril()
    for i, sl in enumerate(seqlens.tolist()):
        if causal:
            sdpa_mask[i, 0, :sl, :sl] = causal_tri[:sl, :sl]
        else:
            sdpa_mask[i, 0, :sl, :sl] = True
    out_ref = F.scaled_dot_product_attention(
        q_pad.transpose(1, 2), k_pad.transpose(1, 2), v_pad.transpose(1, 2),
        attn_mask=sdpa_mask, enable_gqa=(H_q != H_kv),
    ).transpose(1, 2)
    out_ref.backward(g_pad)

    def gather(grad_pad):
        out = torch.zeros_like(grad_pad[:0].reshape(0, *grad_pad.shape[2:]))
        out = torch.empty(total, *grad_pad.shape[2:], dtype=grad_pad.dtype, device=grad_pad.device)
        offset = 0
        for i, sl in enumerate(seqlens.tolist()):
            out[offset:offset + sl] = grad_pad[i, :sl]
            offset += sl
        return out

    dq_ref = gather(q_pad.grad)
    dk_ref = gather(k_pad.grad)
    dv_ref = gather(v_pad.grad)

    # Mirror the existing test_varlen tolerance: torch.allclose(atol=3e-2, rtol=3e-2).
    # Strict abs-max would over-trigger on causal+GQA dv where reference values
    # near zero amplify bf16 atomic-add non-determinism (FA4 paper §3.2.4).
    def assert_close(a, b, name, atol=3e-2, rtol=3e-2):
        a_f, b_f = a.float(), b.float()
        if torch.allclose(a_f, b_f, atol=atol, rtol=rtol):
            return
        diff = (a_f - b_f).abs()
        max_abs = diff.max().item()
        # Find the worst offender to make failure messages self-explanatory.
        bad = diff > (atol + rtol * b_f.abs())
        n_bad = int(bad.sum().item())
        n_total = bad.numel()
        raise AssertionError(
            f"B={B} seq_len={seq_len} mha={mha_type} causal={causal}: "
            f"{name} disagrees with SDPA, max_abs={max_abs:.4g}, "
            f"{n_bad}/{n_total} elements outside tol"
        )

    assert_close(q.grad, dq_ref, "dq")
    assert_close(k.grad, dk_ref, "dk")
    assert_close(v.grad, dv_ref, "dv")
