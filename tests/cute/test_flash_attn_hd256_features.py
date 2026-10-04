"""hd256 forward features against fp32; backward support is still limited."""

import pytest
import torch
from mask_mod_definitions import get_mask_pair
from score_mod_definitions import alibi_eager, score_mod_alibi

from flash_attn.cute.compute_block_sparsity import compute_block_sparsity
from flash_attn.cute.interface import flash_attn_func
from flash_attn.cute.testing import attention_ref

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (10, 11),
    reason="SM100/SM110-only hd256 forward features",
)

HEAD_DIM = 256
FEATURES = ["softcap", "learnable_sink", "local", "score_mod", "mask_mod", "block_sparse"]


def index_grid(seqlen_q, seqlen_k):
    q_idx = torch.arange(seqlen_q, device="cuda")[:, None]
    kv_idx = torch.arange(seqlen_k, device="cuda")[None, :]
    return q_idx, kv_idx


@pytest.mark.parametrize("causal", [False, True], ids=["noncausal", "causal"])
@pytest.mark.parametrize("nheads,nheads_kv", [(6, 6), (24, 4)], ids=["mha", "gqa24_4"])
# 128 rows exercise packed GQA; 520 rows exercise 2CTA when eligible.
@pytest.mark.parametrize("seqlen_q,seqlen_k", [(128, 640), (520, 520)])
@pytest.mark.parametrize("feature", FEATURES)
def test_flash_attn_hd256_sm100_fwd_features(
    feature, seqlen_q, seqlen_k, nheads, nheads_kv, causal
):
    if feature == "block_sparse" and 128 % (nheads // nheads_kv) != 0 and seqlen_q <= 128:
        # Pre-existing block-sparse fault with non-divisible PackGQA (cp.async Q).
        pytest.skip("block sparsity + PackGQA with a head ratio that does not divide tile_m faults")
    torch.manual_seed(0)
    dtype, batch_size = torch.bfloat16, 2
    q = torch.randn(batch_size, seqlen_q, nheads, HEAD_DIM, device="cuda", dtype=dtype)
    k = torch.randn(batch_size, seqlen_k, nheads_kv, HEAD_DIM, device="cuda", dtype=dtype)
    v = torch.randn_like(k)

    kwargs = {"causal": causal}
    ref_kwargs = {"causal": causal}
    if feature == "softcap":
        kwargs["softcap"] = ref_kwargs["softcap"] = 15.0
    elif feature == "learnable_sink":
        sink = torch.randn(nheads, device="cuda", dtype=dtype)
        kwargs["learnable_sink"] = ref_kwargs["learnable_sink"] = sink
    elif feature == "local":
        kwargs["window_size"] = ref_kwargs["window_size"] = (100, 37)
    elif feature == "score_mod":
        kwargs["score_mod"] = score_mod_alibi
        q_idx, kv_idx = index_grid(seqlen_q, seqlen_k)
        head = torch.arange(nheads, device="cuda")[:, None, None]
        ref_kwargs["attn_bias"] = alibi_eager(0.0, None, head, q_idx, kv_idx)[None].float()
    else:
        # mask_mod owns causality, including bottom-right alignment.
        mask_mod, mask_flex = get_mask_pair(
            "causal" if causal else "block_diagonal", seqlen_q=seqlen_q, seqlen_k=seqlen_k
        )
        kwargs = {"mask_mod": mask_mod}
        q_idx, kv_idx = index_grid(seqlen_q, seqlen_k)
        allowed = mask_flex(0, 0, q_idx, kv_idx)
        ref_kwargs = {
            "attn_bias": torch.zeros(allowed.shape, device="cuda").masked_fill(~allowed, -torch.inf)
        }
        if feature == "block_sparse":
            kwargs["block_sparse_tensors"] = compute_block_sparsity(
                tile_m=128, tile_n=128, batch_size=1, num_heads=1,
                seqlen_q=seqlen_q, seqlen_k=seqlen_k, mask_mod=mask_mod,
                aux_tensors=None, device="cuda",
            )

    out, _ = flash_attn_func(q, k, v, **kwargs)
    out_ref, _ = attention_ref(q, k, v, **ref_kwargs)
    out_pt, _ = attention_ref(q, k, v, **ref_kwargs, upcast=False, reorder_ops=True)
    assert torch.isfinite(out).all()
    fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
    rtol = 3 if feature == "softcap" else 2
    assert (out - out_ref).abs().max().item() <= rtol * (
        out_pt - out_ref
    ).abs().max().item() + fwd_atol
