"""GQA / MQA decode (seqlen_q == 1, nheads > nheads_k) through every FA2 forward entry point, at batch sizes on
both sides of the num_splits_heuristic fill threshold (batch * nheads_k >= 0.8 * 2 * num_SMs). FA2 packs the
query heads into seqlen_q for this case (seqlenq_ngroups_swapped); none of tests/test_flash_attn.py reaches
that packed path with flash_attn_func, and test_flash_attn_kvcache uses batch 2 only.

Reference: grouped-query attention in fp32 without repeating K/V (out_ref) and the same in the input dtype
(out_pt); FA2 must be within `mult` x the PyTorch error, as in tests/test_flash_attn.py.
"""
import itertools
import math

import pytest
import torch
from flash_attn import flash_attn_func, flash_attn_with_kvcache
from flash_attn.layers.rotary import apply_rotary_emb


def ref_decode(q, k, v, key_padding_mask=None, softcap=0.0, upcast=True, reorder_ops=False):
    """q: (b, 1, hq, d); k, v: (b, s, hkv, d); key_padding_mask: (b, s) bool, True = attend."""
    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    b, _, hq, d = q.shape
    hkv = k.shape[2]
    g = hq // hkv
    qg = q[:, 0].reshape(b, hkv, g, d)
    if not reorder_ops:
        scores = torch.einsum("bhgd,bshd->bhgs", qg / math.sqrt(d), k)
    else:
        scores = torch.einsum("bhgd,bshd->bhgs", qg, k / math.sqrt(d))
    if softcap > 0:
        scores = torch.tanh(scores / softcap) * softcap
    if key_padding_mask is not None:
        scores = scores.masked_fill(~key_padding_mask[:, None, None, :], float("-inf"))
    lse = torch.logsumexp(scores.float(), dim=-1)
    attn = torch.softmax(scores, dim=-1).to(v.dtype)
    out = torch.einsum("bhgs,bshd->bhgd", attn, v)
    return out.reshape(b, 1, hq, d).to(dtype_og), lse.reshape(b, hq)


BATCH = [1, 48, 128, 256]
HEADS = [(64, 8), (32, 1), (16, 2)]
HDIM = [64, 96, 128, 256]
SEQLEN_K = [200, 1024, 3000]


def _cases():
    bf16, fp16 = torch.bfloat16, torch.float16
    grid = list(itertools.product(BATCH, HEADS, HDIM, SEQLEN_K))
    out = []
    for b, (hq, hkv), d, s in grid:
        for cap, causal in itertools.product([0.0, 30.0], [False, True]):
            out.append(("func", b, hq, hkv, d, s, cap, None, causal, bf16))
        out.append(("func", b, hq, hkv, d, s, 0.0, None, False, fp16))
        if b <= 128 and s <= 1024:
            for cap in (0.0, 30.0):
                out.append(("func_bwd", b, hq, hkv, d, s, cap, None, False, bf16))
        for ns in (0, 1, 3):
            out.append(("kvc", b, hq, hkv, d, s, 0.0, ns, False, bf16))
        out.append(("kvc", b, hq, hkv, d, s, 30.0, 0, False, bf16))
        out.append(("kvc", b, hq, hkv, d, s, 0.0, 0, False, fp16))
        out.append(("kvc_seqlens", b, hq, hkv, d, s, 0.0, 0, False, bf16))
        for cap in (0.0, 30.0):
            out.append(("kvc_paged", b, hq, hkv, d, s, cap, 0, False, bf16))
        out.append(("kvc_batch_idx", b, hq, hkv, d, s, 0.0, 0, False, bf16))
        for causal in (False, True):
            out.append(("kvc_newkv_rotary", b, hq, hkv, d, s, 0.0, 0, causal, bf16))
        out.append(("kvc_leftpad", b, hq, hkv, d, s, 0.0, 0, False, bf16))
    return [pytest.param(*c, id=f"{c[0]}-b{c[1]}-h{c[2]}x{c[3]}-d{c[4]}-s{c[5]}-cap{int(c[6])}-ns{c[7]}-c{int(c[8])}-"
                                f"{'bf16' if c[9] == bf16 else 'fp16'}") for c in out]


@pytest.mark.parametrize("api,batch,nheads,nheads_k,d,seqlen_k,softcap,num_splits,causal,dtype", _cases())
def test_flash_attn_decode_gqa(api, batch, nheads, nheads_k, d, seqlen_k, softcap, num_splits, causal, dtype):
    device = "cuda"
    torch.random.manual_seed(0)
    b, s = batch, seqlen_k
    q = torch.randn(b, 1, nheads, d, device=device, dtype=dtype)
    if softcap > 0:
        q = q * softcap / 4
    mult = 2 if api.startswith("func") else 3
    mask = None

    if api in ("func", "func_bwd"):
        k = torch.randn(b, s, nheads_k, d, device=device, dtype=dtype)
        v = torch.randn(b, s, nheads_k, d, device=device, dtype=dtype)
        if api == "func_bwd":
            q, k, v = (t.detach().requires_grad_() for t in (q, k, v))
        out, lse, _ = flash_attn_func(q, k, v, causal=causal, softcap=softcap, return_attn_probs=True)
        k_ref, v_ref, q_ref = k, v, q
    else:
        b_cache = b
        k_new = v_new = cache_seqlens = cache_batch_idx = block_table = leftpad = cos = sin = None
        if api == "kvc_batch_idx":
            b_cache = 2 * b
            cache_batch_idx = torch.randperm(b_cache, dtype=torch.int32, device=device)[:b]
        k_cache = torch.randn(b_cache, s, nheads_k, d, device=device, dtype=dtype)
        v_cache = torch.randn(b_cache, s, nheads_k, d, device=device, dtype=dtype)
        arange = torch.arange(s, device=device)[None, :]
        if api in ("kvc_seqlens", "kvc_paged", "kvc_batch_idx", "kvc_leftpad"):
            cache_seqlens = torch.randint(1, s + 1, (b,), dtype=torch.int32, device=device)
            mask = arange < cache_seqlens[:, None]
        if api == "kvc_leftpad":
            leftpad = (torch.rand(b, device=device) * cache_seqlens.float()).floor().to(torch.int32)
            mask = mask & (arange >= leftpad[:, None])
        q_ref = q
        if api == "kvc_newkv_rotary":
            cache_seqlens = torch.randint(0, s, (b,), dtype=torch.int32, device=device)
            k_new = torch.randn(b, 1, nheads_k, d, device=device, dtype=dtype)
            v_new = torch.randn(b, 1, nheads_k, d, device=device, dtype=dtype)
            rotary_dim = max(16, (d // 2) // 16 * 16)
            angle = torch.rand(s, rotary_dim // 2, device=device) * 2 * math.pi
            cos, sin = torch.cos(angle).to(dtype), torch.sin(angle).to(dtype)
            q_ref = apply_rotary_emb(q, cos, sin, seqlen_offsets=cache_seqlens, interleaved=False)
            k_ro = apply_rotary_emb(k_new, cos, sin, seqlen_offsets=cache_seqlens, interleaved=False)
            upd = arange == cache_seqlens[:, None]
            k_cache_ref, v_cache_ref = k_cache.clone(), v_cache.clone()
            k_cache_ref[upd] = k_ro[:, 0]
            v_cache_ref[upd] = v_new[:, 0]
            mask = arange < cache_seqlens[:, None] + 1
        if api == "kvc_paged":
            page = 256
            npg = (s + page - 1) // page
            k_paged = torch.randn(b * npg, page, nheads_k, d, device=device, dtype=dtype)
            v_paged = torch.randn(b * npg, page, nheads_k, d, device=device, dtype=dtype)
            block_table = torch.randperm(b * npg, dtype=torch.int32, device=device).view(b, npg)
            idx = block_table.long().flatten()
            k_cache = k_paged[idx].view(b, npg * page, nheads_k, d)[:, :s]
            v_cache = v_paged[idx].view(b, npg * page, nheads_k, d)[:, :s]
        out, lse = flash_attn_with_kvcache(
            q,
            k_paged if api == "kvc_paged" else k_cache,
            v_paged if api == "kvc_paged" else v_cache,
            k_new,
            v_new,
            rotary_cos=cos,
            rotary_sin=sin,
            cache_seqlens=cache_seqlens,
            cache_batch_idx=cache_batch_idx,
            cache_leftpad=leftpad,
            block_table=block_table,
            causal=causal,
            softcap=softcap,
            rotary_interleaved=False,
            num_splits=num_splits,
            return_softmax_lse=True,
        )
        if api == "kvc_newkv_rotary":
            assert torch.allclose(k_cache, k_cache_ref, rtol=1e-3, atol=1e-3)
            assert torch.equal(v_cache, v_cache_ref)
            k_ref, v_ref = k_cache_ref, v_cache_ref
        elif api == "kvc_batch_idx":
            k_ref, v_ref = k_cache[cache_batch_idx.long()], v_cache[cache_batch_idx.long()]
        else:
            k_ref, v_ref = k_cache, v_cache

    out_ref, lse_ref = ref_decode(q_ref, k_ref, v_ref, mask, softcap)
    out_pt, lse_pt = ref_decode(q_ref, k_ref, v_ref, mask, softcap, upcast=False, reorder_ops=True)
    assert out.shape == q.shape
    err, err_pt = (out - out_ref).abs().max().item(), (out_pt - out_ref).abs().max().item()
    assert err <= mult * err_pt + 1e-5, f"out err {err} vs pytorch {err_pt}"
    lse = lse.reshape(b, nheads)
    lerr, lerr_pt = (lse - lse_ref).abs().max().item(), (lse_pt - lse_ref).abs().max().item()
    assert lerr <= 2 * lerr_pt + 1e-3, f"lse err {lerr} vs pytorch {lerr_pt}"

    if api == "func_bwd":
        g = torch.randn_like(out)
        dq, dk, dv = torch.autograd.grad(out, (q, k, v), g)
        dq_ref, dk_ref, dv_ref = torch.autograd.grad(out_ref, (q, k, v), g)
        dq_pt, dk_pt, dv_pt = torch.autograd.grad(out_pt, (q, k, v), g)
        for name, x, x_ref, x_pt in (("dq", dq, dq_ref, dq_pt), ("dk", dk, dk_ref, dk_pt), ("dv", dv, dv_ref, dv_pt)):
            e, e_pt = (x - x_ref).abs().max().item(), (x_pt - x_ref).abs().max().item()
            assert e <= 3 * e_pt + 1e-5, f"{name} err {e} vs pytorch {e_pt}"
