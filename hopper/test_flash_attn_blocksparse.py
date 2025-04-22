import os
import math
import itertools

import pytest
import torch
import torch.nn.functional as F
import torchao

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

from flash_attn_interface import flash_attn_varlen_func


DISABLE_BACKWARD = True
DISABLE_SPLIT = True
DISABLE_PAGEDKV = True
DISABLE_APPENDKV = True
DISABLE_LOCAL = True
DISABLE_SOFTCAP = True
DISABLE_PACKGQA = True
DISABLE_FP16 = os.getenv("FLASH_ATTENTION_DISABLE_FP16", "FALSE") == "TRUE"
DISABLE_FP8 = os.getenv("FLASH_ATTENTION_DISABLE_FP8", "FALSE") == "TRUE" or torch.cuda.get_device_capability("cuda")[0] < 9
DISABLE_HDIM64 = True
DISABLE_HDIM96 = True
DISABLE_HDIM128 = os.getenv("FLASH_ATTENTION_DISABLE_HDIM128", "FALSE") == "TRUE"
DISABLE_HDIM192 = True
DISABLE_HDIM256 = True

COMPILED_HDIMS = (
    []
    + ([64] if not DISABLE_HDIM64 else [])
    + ([96] if not DISABLE_HDIM96 else [])
    + ([128] if not DISABLE_HDIM128 else [])
    + ([192] if not DISABLE_HDIM192 else [])
    + ([256] if not DISABLE_HDIM256 else [])
)


def ceil_div(x, y):
    return (x - 1) // y + 1


def generate_blocksparse_masks(batch, nheads, seqlen_q, seqlen_k, sparse_block_q, sparse_block_k, sparsity=0.75):
    blocklen_q = ceil_div(seqlen_q, sparse_block_q)
    blocklen_k = ceil_div(seqlen_k, sparse_block_k)
    masks = torch.zeros(batch, nheads, blocklen_q, blocklen_k, dtype=torch.uint8, device="cuda")
    masks.bernoulli_(p=1-sparsity)
    return masks


def multidist_randn(num_dists, dim, mean_mean=0.0, mean_std=1.0, scale_lower=0.5, scale_upper=1.5):
    means = torch.distributions.Normal(mean_mean, mean_std).sample((num_dists,))
    scales = torch.distributions.Uniform(scale_lower, scale_upper).sample((num_dists,))
    data = torch.distributions.Normal(means, scales).sample((dim,))
    return data.T.contiguous()


def pack_as_crm(masks: torch.Tensor):
    assert masks.dtype == torch.uint8
    batch, nheads, blocklen_q, blocklen_k = masks.shape
    padded = ceil_div(blocklen_k, 32) * 32
    pad_right = padded - blocklen_k
    masks = F.pad(masks, pad=[0, pad_right], mode='constant', value=0).reshape(batch, nheads, blocklen_q, padded // 8, 8)
    masks_crm = torchao.dtypes.uintx.bitpacking.pack(masks, elem_size=1)[0].squeeze(-1)
    return masks_crm.view(torch.uint32)


def test_mask_pack():
    mask = torch.tensor([[[[0,0,0,0,0,0,0,1]]]], dtype=torch.uint8, device="cuda")
    mask_crm = pack_as_crm(mask)
    assert mask_crm == torch.tensor([[[[128]]]], dtype=torch.uint32, device="cuda")

    mask = torch.tensor([[[[0,1,0,1,0,1,0,1]]]], dtype=torch.uint8, device="cuda")
    mask_crm = pack_as_crm(mask)
    assert mask_crm == torch.tensor([[[[170]]]], dtype=torch.uint32, device="cuda")

    mask = torch.tensor([[[[0,0,0,0,0,0,0,0,1]]]], dtype=torch.uint8, device="cuda")
    mask_crm = pack_as_crm(mask)
    assert mask_crm == torch.tensor([[[[256]]]], dtype=torch.uint32, device="cuda")

    mask = torch.tensor([[[[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]]]], dtype=torch.uint8, device="cuda")
    mask_crm = pack_as_crm(mask)
    assert mask_crm == torch.tensor([[[[(((170 * 256 + 170) * 256) +170) * 256 + 170]]]], dtype=torch.uint32, device="cuda")

    mask = torch.tensor([[[[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,1]]]], dtype=torch.uint8, device="cuda")
    mask_crm = pack_as_crm(mask)
    assert torch.all(mask_crm == torch.tensor([[[[(((170 * 256 + 170) * 256) +170) * 256 + 170, 1]]]], dtype=torch.uint32, device="cuda"))


@pytest.mark.parametrize("sparse_block_q,sparse_block_k", [(128,128)])
@pytest.mark.parametrize("dtype", ["float16"])
# @pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
@pytest.mark.parametrize("mha_type", ["mha"])
@pytest.mark.parametrize("has_qv", [False])
@pytest.mark.parametrize("deterministic", [False])
@pytest.mark.parametrize("softcap", [0.0])
@pytest.mark.parametrize("local", [False])
@pytest.mark.parametrize("causal", [False])
# @pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("add_unused_qkv", [False])
@pytest.mark.parametrize("d", [128])
@pytest.mark.parametrize("seqlen_q,seqlen_k", [(2048, 2048), (2048, 2047)])
def test_flash_attn_varlen_output_sparse(
    seqlen_q,
    seqlen_k,
    d,
    add_unused_qkv,
    causal,
    local,
    softcap,
    deterministic,
    has_qv,
    mha_type,
    dtype,
    sparse_block_q,
    sparse_block_k,
):
    dtype = getattr(torch, dtype)
    device = "cuda"
    # set seed
    torch.random.manual_seed(seqlen_q + seqlen_k + d + int(causal) * 2 + int(local))
    # batch_size = 40
    # nheads = 16
    batch_size = 9 if seqlen_q <= 2048 else 2
    nheads = 6
    # batch_size = 3
    # nheads = 1
    nheads_kv = nheads if mha_type == "mha" else (2 if mha_type == "gqa" else 1)
    dtype_ref = torch.bfloat16 if dtype == torch.float8_e4m3fn else dtype
    dv_vals = [128, d] if d > 128 and d <= 192 else ([256, 512, d] if d <= 64 else [d])
    if dtype == torch.float8_e4m3fn:
        dv_vals = [d]
    for dv in dv_vals:
        q_ref = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype_ref)
        if softcap > 0.0:
            # Ensure the values of qk are at least within softcap range.
            q_ref = (q_ref * softcap / 4).detach().requires_grad_()
        q_ref = q_ref.to(dtype).to(dtype_ref).requires_grad_()
        k_ref = torch.randn(batch_size, seqlen_k, nheads_kv, d, device=device, dtype=dtype_ref).to(dtype).to(dtype_ref).requires_grad_()
        v_ref = torch.randn(batch_size, seqlen_k, nheads_kv, dv, device=device, dtype=dtype_ref).to(dtype).to(dtype_ref).requires_grad_()
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
        q, k, v = [x.detach().requires_grad_() for x in (q_ref, k_ref, v_ref)]
        qv = qv_ref.detach() if has_qv else None
        query_padding_mask = generate_random_padding_mask(
            seqlen_q, batch_size, device, mode="full", zero_lengths=False
        )
        key_padding_mask = generate_random_padding_mask(
            seqlen_k, batch_size, device, mode="full", zero_lengths=False
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

        sparse_masks = generate_blocksparse_masks(batch_size, nheads, seqlen_q, seqlen_k, sparse_block_q, sparse_block_k, sparsity=0.5)
        sparse_masks[:, :, :, 0] = 1  # FIXME: we need at least one block
        attn_bias = torch.zeros_like(sparse_masks, dtype=torch.float32)
        attn_bias[sparse_masks == 0] = -float("inf")
        attn_bias = torch.repeat_interleave(attn_bias, sparse_block_k, dim=-1)
        attn_bias = torch.repeat_interleave(attn_bias, sparse_block_q, dim=-2)
        attn_bias = attn_bias[:, :, :seqlen_q, :seqlen_k]

        out_ref, attn_ref = attention_ref(
            q_ref,
            k_ref,
            v_ref,
            query_padding_mask,
            key_padding_mask,
            attn_bias=attn_bias,
            causal=causal,
            qv=qv_ref,
            q_descale=q_descale, k_descale=k_descale, v_descale=v_descale,
            window_size=window_size,
            softcap=softcap
        )
        out_pt, attn_pt = attention_ref(
            q_ref,
            k_ref,
            v_ref,
            query_padding_mask,
            key_padding_mask,
            attn_bias=attn_bias,
            causal=causal,
            qv=qv_ref,
            q_descale=q_descale, k_descale=k_descale, v_descale=v_descale,
            window_size=window_size,
            softcap=softcap,
            upcast=False,
            reorder_ops=True,
            intermediate_dtype=dtype if dtype == torch.float8_e4m3fn else None,
        )


        print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
        print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")

        if query_unused_mask is not None:
            q_zero_masking = rearrange(query_unused_mask, "b s -> b s 1 1")

        # Numerical error if we just do any arithmetic on out_ref
        fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
        rtol = 2 if softcap == 0.0 else 3

        print("q/k/v_ref", q_ref.shape, k_ref.shape, v_ref.shape)
        print("q/k/v_unpad", q_unpad.shape, k_unpad.shape, v_unpad.shape, f"max_seqlen_q={max_seqlen_q}", f"max_seqlen_k={max_seqlen_k}")
        print(cu_seqlens_q, cu_seqlens_k)
        pack_gqa_vals = [False, True] if not DISABLE_PACKGQA else [False]
        num_splits_vals = [1, 3] if not DISABLE_SPLIT else [1]

        sparse_masks_crm = pack_as_crm(sparse_masks)

        for pack_gqa, num_splits in itertools.product(pack_gqa_vals, num_splits_vals):
            # TODO:
            assert not pack_gqa
            assert num_splits == 1

            out_unpad, lse = flash_attn_varlen_func(
                q_unpad,
                k_unpad,
                v_unpad,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                seqused_q=seqused_q,
                seqused_k=seqused_k,
                causal=causal,
                qv=qv_unpad,
                q_descale=q_descale, k_descale=k_descale, v_descale=v_descale,
                sparse_masks=sparse_masks_crm,  # zero means masked out
                window_size=window_size,
                softcap=softcap,
                sparse_block_q=sparse_block_q,
                sparse_block_k=sparse_block_k,
            )
            torch.cuda.synchronize()
            out = output_pad_fn(out_unpad)
            if query_unused_mask is not None:
                out.masked_fill_(q_zero_masking, 0.0)
            # print(out)
            # print(out_ref)
            print(f"Output max diff: {(out - out_ref).abs().max().item()}")
            print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
            # if not causal:
            #     print(f"LSE max diff: {(lse - lse_ref).abs().max().item()}")
            # breakpoint()

            # Check that FlashAttention's numerical error is at most 3x the numerical error
            # of a Pytorch implementation.
            assert (out - out_ref).abs().max().item() <= rtol * (out_pt - out_ref).abs().max().item() + fwd_atol
