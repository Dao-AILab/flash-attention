import math
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


@pytest.mark.parametrize("hq,hkv", [(8, 8), (8, 2), (16, 2), (16, 1)])
def test_sm103_hd256_bf16_causal_varlen(hq, hkv):
    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("SM103 is required")

    torch.manual_seed(20260719 + hq)
    lengths = [257, 509, 1021]
    total = sum(lengths)
    d = 256
    q = torch.randn(
        total, hq, d, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    k = torch.randn(
        total, hkv, d, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    v = torch.randn_like(k, requires_grad=True)
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()],
        device="cuda",
        dtype=torch.int32,
    )

    assert check_varlen_vs_torch_flash(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        total_q=total,
        total_k=total,
        causal=True,
    )


@pytest.mark.parametrize("hq,hkv", [(8, 8), (8, 2), (16, 2), (16, 1)])
def test_sm103_hd256_fp8_causal_varlen(hq, hkv):
    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("SM103 is required")

    torch.manual_seed(20260720 + hq)
    lengths = [257, 509]
    total = sum(lengths)
    d = 256
    q = torch.randn(total, hq, d, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    k = torch.randn(total, hkv, d, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    v = torch.randn(total, hkv, d, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()],
        device="cuda",
        dtype=torch.int32,
    )
    scale = d**-0.5

    out, _ = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max(lengths),
        max_seqlen_k=max(lengths),
        softmax_scale=scale,
        causal=True,
        pack_gqa=None,
    )
    ref = torch_flash_ref(
        q.float(),
        k.float(),
        v.float(),
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        total_q=total,
        total_k=total,
        softmax_scale=scale,
        causal=True,
    )
    assert out.dtype == torch.bfloat16
    out = out.float()
    rel_l1 = ((out - ref).abs().sum() / ref.abs().sum()).item()
    cosine = F.cosine_similarity(out.flatten(), ref.flatten(), dim=0).item()
    assert cosine > 0.99, f"cosine={cosine}, rel_l1={rel_l1}"
    assert rel_l1 < 0.08, f"rel_l1={rel_l1}, cosine={cosine}"


@pytest.mark.parametrize("page_size", [16, 32, 64, 128, 256])
@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float8_e4m3fn], ids=["bf16", "fp8"]
)
def test_sm103_hd256_causal_varlen_paged(dtype, page_size):
    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("SM103 is required")

    q_lengths = [257, 509, 1021]
    k_lengths = [333, 777, 1539]
    hq, hkv, d = 32, 2, 256
    generator = torch.Generator(device="cuda").manual_seed(20260722 + page_size)

    def random_input(shape):
        return (
            torch.randn(
                shape,
                device="cuda",
                dtype=torch.bfloat16,
                generator=generator,
            )
            .mul_(0.02)
            .to(dtype)
            .contiguous()
        )

    q_sequences = [random_input((length, hq, d)) for length in q_lengths]
    k_sequences = [random_input((length, hkv, d)) for length in k_lengths]
    v_sequences = [random_input((length, hkv, d)) for length in k_lengths]
    q = torch.cat(q_sequences)
    k = torch.cat(k_sequences)
    v = torch.cat(v_sequences)
    cu_q = torch.tensor(
        [0, *torch.tensor(q_lengths).cumsum(0).tolist()],
        device="cuda",
        dtype=torch.int32,
    )
    cu_k = torch.tensor(
        [0, *torch.tensor(k_lengths).cumsum(0).tolist()],
        device="cuda",
        dtype=torch.int32,
    )
    seqused_k = torch.tensor(k_lengths, device="cuda", dtype=torch.int32)

    page_counts = [math.ceil(length / page_size) for length in k_lengths]
    total_pages = sum(page_counts)
    page_table = torch.zeros(
        len(k_lengths), max(page_counts), device="cuda", dtype=torch.int32
    )
    physical_pages = torch.randperm(total_pages, device="cuda", generator=generator)
    k_paged = torch.zeros(total_pages, page_size, hkv, d, device="cuda", dtype=dtype)
    v_paged = torch.zeros_like(k_paged)
    page_cursor = 0
    for batch_idx, (k_sequence, v_sequence, page_count) in enumerate(
        zip(k_sequences, v_sequences, page_counts)
    ):
        page_ids = physical_pages[page_cursor : page_cursor + page_count]
        page_table[batch_idx, :page_count] = page_ids
        k_buffer = torch.zeros(
            page_count * page_size, hkv, d, device="cuda", dtype=dtype
        )
        v_buffer = torch.zeros_like(k_buffer)
        k_buffer[: k_sequence.shape[0]] = k_sequence
        v_buffer[: v_sequence.shape[0]] = v_sequence
        k_paged[page_ids] = k_buffer.view(page_count, page_size, hkv, d)
        v_paged[page_ids] = v_buffer.view(page_count, page_size, hkv, d)
        page_cursor += page_count

    kwargs = dict(
        cu_seqlens_q=cu_q,
        max_seqlen_q=max(q_lengths),
        max_seqlen_k=max(k_lengths),
        causal=True,
        softmax_scale=d**-0.5,
        num_splits=1,
        pack_gqa=True,
    )
    out_contiguous, _ = flash_attn_varlen_func(
        q, k, v, cu_seqlens_k=cu_k, **kwargs
    )
    out_paged, _ = flash_attn_varlen_func(
        q,
        k_paged,
        v_paged,
        seqused_k=seqused_k,
        page_table=page_table,
        **kwargs,
    )
    torch.testing.assert_close(out_paged, out_contiguous, rtol=0, atol=0)


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
