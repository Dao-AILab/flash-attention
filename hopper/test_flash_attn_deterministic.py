import os
import math
import itertools

import pytest
import torch
import torch.nn.functional as F

from einops import rearrange, repeat
try:
    from flash_attn.layers.rotary import apply_rotary_emb
except ImportError:
    apply_rotary_emb = None

from padding import pad_input, unpad_input
from test_util import (
    attention_ref,
    generate_qkv,
    generate_random_padding_mask,
)

from flash_attn_interface import flash_attn_func, flash_attn_varlen_func


DISABLE_BACKWARD = os.getenv("FLASH_ATTENTION_DISABLE_BACKWARD", "FALSE") == "TRUE"
DISABLE_SPLIT = os.getenv("FLASH_ATTENTION_DISABLE_SPLIT", "FALSE") == "TRUE"
DISABLE_PAGEDKV = os.getenv("FLASH_ATTENTION_DISABLE_PAGEDKV", "FALSE") == "TRUE"
DISABLE_APPENDKV = os.getenv("FLASH_ATTENTION_DISABLE_APPENDKV", "FALSE") == "TRUE"
DISABLE_LOCAL = os.getenv("FLASH_ATTENTION_DISABLE_LOCAL", "FALSE") == "TRUE"
DISABLE_SOFTCAP = os.getenv("FLASH_ATTENTION_DISABLE_SOFTCAP", "FALSE") == "TRUE"
DISABLE_PACKGQA = os.getenv("FLASH_ATTENTION_DISABLE_PACKGQA", "FALSE") == "TRUE"
DISABLE_FP16 = os.getenv("FLASH_ATTENTION_DISABLE_FP16", "FALSE") == "TRUE"
DISABLE_FP8 = os.getenv("FLASH_ATTENTION_DISABLE_FP8", "FALSE") == "TRUE" or torch.cuda.get_device_capability("cuda")[0] < 9
DISABLE_HDIM64 = os.getenv("FLASH_ATTENTION_DISABLE_HDIM64", "FALSE") == "TRUE"
DISABLE_HDIM96 = os.getenv("FLASH_ATTENTION_DISABLE_HDIM96", "FALSE") == "TRUE"
DISABLE_HDIM128 = os.getenv("FLASH_ATTENTION_DISABLE_HDIM128", "FALSE") == "TRUE"
DISABLE_HDIM192 = os.getenv("FLASH_ATTENTION_DISABLE_HDIM192", "FALSE") == "TRUE"
DISABLE_HDIM256 = os.getenv("FLASH_ATTENTION_DISABLE_HDIM256", "FALSE") == "TRUE"

COMPILED_HDIMS = (
    []
    + ([64] if not DISABLE_HDIM64 else [])
    + ([96] if not DISABLE_HDIM96 else [])
    + ([128] if not DISABLE_HDIM128 else [])
    + ([192] if not DISABLE_HDIM192 else [])
    + ([256] if not DISABLE_HDIM256 else [])
)

# @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("dtype", [torch.bfloat16] + ([torch.float16] if not DISABLE_FP16 else []) + ([torch.float8_e4m3fn] if not DISABLE_FP8 else []))
# @pytest.mark.parametrize("dtype", [torch.bfloat16])
# @pytest.mark.parametrize("dtype", [torch.float8_e4m3fn])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
# @pytest.mark.parametrize("mha_type", ["mha"])
# @pytest.mark.parametrize("has_qv", [False, True])
@pytest.mark.parametrize("has_qv", [False])
@pytest.mark.parametrize("deterministic", [True])
# @pytest.mark.parametrize("deterministic", [False])
@pytest.mark.parametrize("softcap", [0.0] + ([15.0] if not DISABLE_SOFTCAP else []))
# @pytest.mark.parametrize("softcap", [0.0])
@pytest.mark.parametrize("local", [False] + ([True] if not DISABLE_LOCAL else []))
# @pytest.mark.parametrize("local", [False])
@pytest.mark.parametrize("causal", [False, True])
# @pytest.mark.parametrize("causal", [True])
# @pytest.mark.parametrize("V_colmajor", [False, True])
@pytest.mark.parametrize("V_colmajor", [False])
# @pytest.mark.parametrize("d", [32, 64, 96, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize('d', [32, 40, 64, 80, 96, 128, 160, 192, 256])
# @pytest.mark.parametrize('d', [32, 64, 96, 128, 160, 192])
# @pytest.mark.parametrize('d', [56, 80])
# @pytest.mark.parametrize("d", [64, 128, 256])
# @pytest.mark.parametrize('d', [32, 40, 64, 80, 96, 128])
# @pytest.mark.parametrize("d", [64, 96, 128, 192])
@pytest.mark.parametrize("d", COMPILED_HDIMS)
# @pytest.mark.parametrize("d", [128])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (1, 1),
        (64, 128),
        (128, 192),
        (256, 256),
        (239, 1),
        (799, 3),
        (113, 203),
        (113, 128),
        (128, 217),
        (113, 211),
        (108, 256),
        (256, 512),
        (384, 256),
        (640, 128),
        (512, 256),
        (1024, 1024),
        (1023, 1024),
        (1024, 1023),
        (4096, 4096),
        (4224, 4224),
    ],
)
# @pytest.mark.parametrize('seqlen_q,seqlen_k', [(128, 128)])
def test_flash_attn_output(
        seqlen_q, seqlen_k, d, causal, local, softcap, V_colmajor, deterministic, has_qv, mha_type, dtype
):
    if V_colmajor and (seqlen_k % 16 != 0 or dtype != torch.float8_e4m3fn):
        pytest.skip("V_colmajor requires seqlen_k to be a multiple of 16 and dtype to be float8_e4m3fn")
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    # batch_size = 40
    # nheads = 16
    batch_size = 9 if seqlen_k <= 2048 else 2
    # batch_size = 1
    nheads = 6
    # nheads = 1
    nheads_kv = nheads if mha_type == "mha" else (2 if mha_type == "gqa" else 1)
    dtype_ref = torch.bfloat16 if dtype == torch.float8_e4m3fn else dtype
    dv_vals = [128, d] if d > 128 and d <= 192 else ([d] if d <= 64 else [d])
    if dtype == torch.float8_e4m3fn:
        dv_vals = [d]
    attention_chunk_vals = [torch.randint(1, seqlen_k * 2, (1,)).item(), 0] if not DISABLE_LOCAL else [0]
    for dv, attention_chunk in itertools.product(dv_vals, attention_chunk_vals):
        q_ref = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype_ref)
        if softcap > 0.0:
            # Ensure the values of qk are at least within softcap range.
            q_ref = (q_ref * softcap / 4)
        q_ref = q_ref.to(dtype).to(dtype_ref).requires_grad_()
        k_ref = torch.randn(batch_size, seqlen_k, nheads_kv, d, device=device, dtype=dtype_ref).to(dtype).to(dtype_ref).requires_grad_()
        v_ref = torch.randn(batch_size, seqlen_k, nheads_kv, dv, device=device, dtype=dtype_ref).to(dtype).to(dtype_ref).requires_grad_()
        if has_qv:
            qv_ref = torch.randn(batch_size, seqlen_q, nheads, dv, device=device, dtype=dtype_ref).to(dtype).to(dtype_ref)
        else:
            qv_ref = None
        # Put window_size after QKV randn so that window_size changes from test to test
        window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,)).tolist()
        # window_size = (-1, -1) if not local else (16, 0)
        if dtype == torch.float8_e4m3fn:
            q_descale, k_descale, v_descale = [torch.rand(batch_size, nheads_kv, device=device, dtype=torch.float32) * 2 for _ in range(3)]
        else:
            q_descale, k_descale, v_descale = None, None, None
        q, k, v = [x.detach().to(dtype).requires_grad_() for x in (q_ref, k_ref, v_ref)]
        qv = qv_ref.detach().to(dtype).requires_grad_() if has_qv else None
        if V_colmajor:
            v = rearrange(rearrange(v.detach(), "b s h d -> b h d s").contiguous(), "b h d s -> b s h d").requires_grad_()

        pack_gqa_vals = [False, True] if not DISABLE_PACKGQA else [False]
        num_splits_vals = [1, 3] if not DISABLE_SPLIT else [1]
        for pack_gqa, num_splits in itertools.product(pack_gqa_vals, num_splits_vals):
            out, lse = flash_attn_func(
                q,
                k,
                v,
                causal=causal,
                qv=qv,
                q_descale=q_descale, k_descale=k_descale, v_descale=v_descale,
                window_size=window_size,
                attention_chunk=attention_chunk,
                softcap=softcap,
                pack_gqa=pack_gqa,
                num_splits=num_splits,
                deterministic=deterministic
            )
        if (
            not DISABLE_BACKWARD 
            and dtype != torch.float8_e4m3fn 
            and not V_colmajor 
            and not has_qv
            and not dv > 256
            and not attention_chunk != 0
            and not d >= 256
        ):
            g = torch.randn_like(out)
            dq0, dk0, dv0 = torch.autograd.grad(out, (q, k, v), g, retain_graph=True)
            for _ in range(50):
                dq, dk, dv = torch.autograd.grad(out, (q, k, v), g, retain_graph=True)
                assert torch.equal(dv, dv0)
                assert torch.equal(dk, dk0)
                assert torch.equal(dq, dq0)
                
                
# @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("dtype", [torch.bfloat16] + ([torch.float16] if not DISABLE_FP16 else []) + ([torch.float8_e4m3fn] if not DISABLE_FP8 else []))
# @pytest.mark.parametrize("dtype", [torch.bfloat16])
# @pytest.mark.parametrize("dtype", [torch.float8_e4m3fn])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
# @pytest.mark.parametrize("mha_type", ["mha"])
# @pytest.mark.parametrize("has_qv", [False, True])
@pytest.mark.parametrize("has_qv", [False])
@pytest.mark.parametrize("deterministic", [True])
# @pytest.mark.parametrize("deterministic", [False])
@pytest.mark.parametrize("softcap", [0.0] + ([15.0] if not DISABLE_SOFTCAP else []))
# @pytest.mark.parametrize("softcap", [0.0])
# @pytest.mark.parametrize("local", [False] + ([True] if not DISABLE_LOCAL else []))
@pytest.mark.parametrize("local", [False])
@pytest.mark.parametrize("causal", [False, True])
# @pytest.mark.parametrize("causal", [False])
@pytest.mark.parametrize("add_unused_qkv", [False, True])
# @pytest.mark.parametrize("add_unused_qkv", [True])
# @pytest.mark.parametrize("d", [32, 64, 96, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize('d', [32, 40, 64, 80, 96, 128, 160, 192, 256])
# @pytest.mark.parametrize('d', [32, 64, 96, 128, 160, 192])
# @pytest.mark.parametrize('d', [56, 80])
# @pytest.mark.parametrize('d', [32, 40, 64, 80, 96, 128])
# @pytest.mark.parametrize("d", [64, 96, 128])
@pytest.mark.parametrize("d", COMPILED_HDIMS)
# @pytest.mark.parametrize("d", [128])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (1, 1),
        (1, 3),
        (2, 1),
        (511, 1),
        (3, 513),
        (64, 128),
        (128, 128),
        (256, 256),
        (113, 203),
        (128, 217),
        (113, 211),
        (108, 256),
        (256, 512),
        (307, 256),
        (640, 128),
        (512, 256),
        (1024, 1024),
        (1023, 1024),
        (1024, 1023),
        (2048, 2048),
    ],
)
def test_flash_attn_varlen_output(
        seqlen_q, seqlen_k, d, add_unused_qkv, causal, local, softcap, deterministic, has_qv, mha_type, dtype
):
    device = "cuda"
    # set seed
    torch.random.manual_seed(seqlen_q + seqlen_k + d + int(causal) * 2 + int(local))
    # batch_size = 40
    # nheads = 16
    batch_size = 9 if seqlen_q <= 2048 else 2
    nheads = 6
    # batch_size = 2
    # nheads = 1
    nheads_kv = nheads if mha_type == "mha" else (2 if mha_type == "gqa" else 1)
    dtype_ref = torch.bfloat16 if dtype == torch.float8_e4m3fn else dtype
    dv_vals = [128, d] if d > 128 and d <= 192 else ([d] if d <= 64 else [d])
    if dtype == torch.float8_e4m3fn:
        dv_vals = [d]
    attention_chunk_vals = [torch.randint(1, seqlen_k * 2, (1,)).item(), 0] if seqlen_q <= seqlen_k and not DISABLE_LOCAL else [0]
    for dv, attention_chunk in itertools.product(dv_vals, attention_chunk_vals):
        q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype_ref)
        if softcap > 0.0:
            q = (q * softcap / 4)  # 应用缩放
        q = q.requires_grad_()  # 统一设置梯度

        k = torch.randn(batch_size, seqlen_k, nheads_kv, d, device=device, dtype=dtype_ref).requires_grad_()
        v = torch.randn(batch_size, seqlen_k, nheads_kv, dv, device=device, dtype=dtype_ref).requires_grad_()
        if has_qv:
            qv_ref = torch.randn(batch_size, seqlen_q, nheads, dv, device=device, dtype=dtype_ref).to(dtype).to(dtype_ref)
        else:
            qv_ref = None
        # Put window_size after QKV randn so that window_size changes from test to test
        window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))
        if dtype == torch.float8_e4m3fn:
            q_descale, k_descale, v_descale = [torch.rand(batch_size, nheads_kv, device=device, dtype=torch.float32) * 2 for _ in range(3)]
        else:
            q_descale, k_descale, v_descale = None, None, None
        # q, k, v = [x.detach().requires_grad_() for x in (q_ref, k_ref, v_ref)]
        qv = qv_ref.detach() if has_qv else None
        query_padding_mask = generate_random_padding_mask(
            seqlen_q, batch_size, device, mode="random", zero_lengths=False
        )
        key_padding_mask = generate_random_padding_mask(
            seqlen_k, batch_size, device, mode="random", zero_lengths=True
        )

        def _gen_unused_masks(padding_mask, add_unused, max_seq_len, bs, device):
            if add_unused:
                another_mask = generate_random_padding_mask(max_seq_len, bs, device)
                attn_mask = torch.logical_and(padding_mask, another_mask)
                unused_mask = torch.logical_xor(
                    torch.logical_or(padding_mask, another_mask), attn_mask
                )
            else:
                attn_mask = padding_mask
                unused_mask = None
            return attn_mask, unused_mask

        query_padding_mask, query_unused_mask = _gen_unused_masks(
            query_padding_mask, add_unused_qkv, seqlen_q, batch_size, q.device
        )
        key_padding_mask, key_unused_mask = _gen_unused_masks(
            key_padding_mask, add_unused_qkv, seqlen_k, batch_size, k.device
        )

        (
            q_unpad,
            k_unpad,
            v_unpad,
            qv_unpad,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            max_seqlen_q,
            max_seqlen_k,
            q,
            k,
            v,
            qv,
            output_pad_fn,
            dq_pad_fn,
            dk_pad_fn,
        ) = generate_qkv(q, k, v, query_padding_mask, key_padding_mask, qv=qv, kvpacked=False,
                        query_unused_mask=query_unused_mask, key_unused_mask=key_unused_mask)
        q_unpad, k_unpad, v_unpad = [x.detach().to(dtype).requires_grad_() for x in (q_unpad, k_unpad, v_unpad)]

        if query_unused_mask is not None:
            q_zero_masking = rearrange(query_unused_mask, "b s -> b s 1 1")

        out_unpad, lse = flash_attn_varlen_func(
                q_unpad,
                k_unpad,
                v_unpad,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                # seqused_q=seqused_q,
                # seqused_k=seqused_k,
                causal=causal,
                qv=qv_unpad,
                q_descale=q_descale,
                k_descale=k_descale, v_descale=v_descale,
                window_size=window_size,
                attention_chunk=attention_chunk,
                deterministic=deterministic,
                softcap=softcap,
        )
        out = output_pad_fn(out_unpad)
        if query_unused_mask is not None:
                out.masked_fill_(q_zero_masking, 0.0)


        if (
            not DISABLE_BACKWARD 
            and dtype != torch.float8_e4m3fn 
            and not has_qv
            and not dv > 256
            and not attention_chunk != 0
            and not d >= 256
        ):
            g_unpad = torch.randn_like(out_unpad)
            dq_unpad0, dk_unpad0, dv_unpad0 = torch.autograd.grad(out_unpad, (q_unpad, k_unpad, v_unpad), g_unpad, retain_graph=True)
            for _ in range(50):
                dq_unpad, dk_unpad, dv_unpad = torch.autograd.grad(out_unpad, (q_unpad, k_unpad, v_unpad), g_unpad, retain_graph=True)
                assert torch.equal(dv_unpad, dv_unpad0)
                assert torch.equal(dk_unpad, dk_unpad0)
                assert torch.equal(dq_unpad, dq_unpad0)
