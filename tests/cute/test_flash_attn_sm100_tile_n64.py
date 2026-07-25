import math

import pytest
import torch
from cutlass import cute
from mask_mod_definitions import cute_block_diagonal_mask
from score_mod_definitions import score_mod_times_two

import flash_attn.cute.interface as flash_attn_interface
from flash_attn.cute.flash_bwd_sm100 import FlashAttentionBackwardSm100
from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd

COMPUTE_CAPABILITY = torch.cuda.get_device_capability()[0]
pytestmark = pytest.mark.skipif(COMPUTE_CAPABILITY != 10, reason="SM100-only tests")


def reference_attention_backward(q, k, v, dout):
    """Compute dense non-causal attention gradients in FP32."""
    q_ref = q.float().detach().requires_grad_()
    k_ref = k.float().detach().requires_grad_()
    v_ref = v.float().detach().requires_grad_()
    scores = torch.einsum("bqhd,bkhd->bhqk", q_ref, k_ref) / math.sqrt(q.shape[-1])
    out = torch.einsum("bhqk,bkhd->bqhd", scores.softmax(dim=-1), v_ref)
    out.backward(dout.float())
    return q_ref.grad, k_ref.grad, v_ref.grad


def assert_gradients_close(actual, expected, *, atol):
    """Compare dQ, dK, and dV with component-labeled failures."""
    for name, grad, grad_ref in zip(("dQ", "dK", "dV"), actual, expected):
        torch.testing.assert_close(
            grad.float(), grad_ref.float(), rtol=0, atol=atol, msg=f"{name} mismatch"
        )


def make_cu_seqlens(lengths):
    """Construct device cumulative sequence lengths from Python lengths."""
    return torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )


@cute.jit
def score_mod_times_two_bwd(
    grad, score, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors
):
    return grad * 2.0


@pytest.mark.parametrize(
    "dtype,head_dim,head_dim_v,seqlen_q,seqlen_k",
    [
        pytest.param(torch.bfloat16, 128, 128, 256, 256, id="bf16-multiblock"),
        pytest.param(torch.float16, 96, 96, 129, 193, id="fp16-tail-d96"),
        pytest.param(torch.bfloat16, 64, 128, 129, 193, id="bf16-asymmetric"),
        pytest.param(torch.bfloat16, 24, 32, 65, 67, id="bf16-padded-d24-dv32"),
        pytest.param(torch.float16, 32, 48, 128, 64, id="fp16-d32-dv48"),
        pytest.param(torch.bfloat16, 64, 80, 129, 65, id="bf16-d64-dv80"),
        pytest.param(torch.float16, 64, 112, 129, 193, id="fp16-d64-dv112"),
    ],
)
def test_sm100_backward_tile_n64_matches_fp32(
    dtype, head_dim, head_dim_v, seqlen_q, seqlen_k
):
    """Check the packed 16-datapath path across multiple Q and KV tiles."""
    torch.manual_seed(0)
    q = torch.randn((1, seqlen_q, 2, head_dim), device="cuda", dtype=dtype)
    k = torch.randn((1, seqlen_k, 2, head_dim), device="cuda", dtype=dtype)
    v = torch.randn((1, seqlen_k, 2, head_dim_v), device="cuda", dtype=dtype)
    out, lse, *_ = _flash_attn_fwd(
        q=q, k=k, v=v, tile_mn=(64, 128), return_lse=True, num_splits=1
    )
    dout = torch.randn_like(out)

    grads = _flash_attn_bwd(
        q,
        k,
        v,
        out,
        dout,
        lse,
        m_block_size=128,
        n_block_size=64,
    )
    grads_ref = reference_attention_backward(q, k, v, dout)
    atol = 4e-2 if dtype == torch.bfloat16 else 2e-2
    assert_gradients_close(grads, grads_ref, atol=atol)


@pytest.mark.parametrize(
    "dtype,head_dim,head_dim_v,seqlen_q,seqlen_k",
    [
        pytest.param(torch.bfloat16, 128, 128, 129, 193, id="bf16-tail-d128"),
        pytest.param(torch.float16, 96, 96, 129, 193, id="fp16-tail-d96"),
        pytest.param(torch.bfloat16, 24, 32, 65, 67, id="bf16-padded-d24-dv32"),
        pytest.param(torch.float16, 32, 48, 128, 64, id="fp16-d32-dv48"),
    ],
)
def test_sm100_backward_tile_m64_n64_matches_fp32(
    dtype, head_dim, head_dim_v, seqlen_q, seqlen_k
):
    """Check true 64x64 backward across padded widths and query/KV tails."""
    torch.manual_seed(4)
    q = torch.randn((1, seqlen_q, 2, head_dim), device="cuda", dtype=dtype)
    k = torch.randn((1, seqlen_k, 2, head_dim), device="cuda", dtype=dtype)
    v = torch.randn((1, seqlen_k, 2, head_dim_v), device="cuda", dtype=dtype)
    out, lse, *_ = _flash_attn_fwd(
        q=q, k=k, v=v, tile_mn=(64, 64), return_lse=True, num_splits=1
    )
    dout = torch.randn_like(out)

    grads = _flash_attn_bwd(
        q,
        k,
        v,
        out,
        dout,
        lse,
        m_block_size=64,
        n_block_size=64,
    )
    grads_ref = reference_attention_backward(q, k, v, dout)
    atol = 4e-2 if dtype == torch.bfloat16 else 2e-2
    assert_gradients_close(grads, grads_ref, atol=atol)


@pytest.mark.parametrize("tile_m", [128, 64])
def test_sm100_backward_tile_n64_causal_deterministic(tile_m):
    """Check deterministic dQ scheduling with multiple Q and KV tiles."""
    torch.manual_seed(2)
    q = torch.randn((1, 129, 2, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, 193, 2, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    out, lse, *_ = _flash_attn_fwd(
        q=q,
        k=k,
        v=v,
        causal=True,
        tile_mn=(64, 64 if tile_m == 64 else 128),
        return_lse=True,
    )
    dout = torch.randn_like(out)
    grads_ref = _flash_attn_bwd(
        q,
        k,
        v,
        out,
        dout,
        lse,
        causal=True,
        deterministic=True,
        m_block_size=128,
        n_block_size=128,
    )
    grads = _flash_attn_bwd(
        q,
        k,
        v,
        out,
        dout,
        lse,
        causal=True,
        deterministic=True,
        m_block_size=tile_m,
        n_block_size=64,
    )
    grads_repeat = _flash_attn_bwd(
        q,
        k,
        v,
        out,
        dout,
        lse,
        causal=True,
        deterministic=True,
        m_block_size=tile_m,
        n_block_size=64,
    )
    for name, grad, grad_repeat in zip(("dQ", "dK", "dV"), grads, grads_repeat):
        assert torch.equal(grad, grad_repeat), f"{name} is not deterministic"
    assert_gradients_close(grads, grads_ref, atol=2e-3)


@pytest.mark.parametrize("tile_m", [128, 64])
@pytest.mark.parametrize(
    "feature", ["local", "mask_mod", "score_mod", "softcap", "dlse"]
)
def test_sm100_backward_tile_n64_features(feature, tile_m):
    """Compare supported masking, score, and LSE-gradient paths with 128x128."""
    torch.manual_seed(3)
    q = torch.randn((1, 193, 2, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    kwargs = {}
    if feature == "local":
        kwargs.update(window_size_left=63, window_size_right=31)
    elif feature == "mask_mod":
        kwargs["mask_mod"] = cute_block_diagonal_mask
    elif feature == "score_mod":
        kwargs.update(
            score_mod=score_mod_times_two,
            score_mod_bwd=score_mod_times_two_bwd,
        )
    elif feature == "softcap":
        kwargs["softcap"] = 8.0

    fwd_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key not in ("score_mod_bwd", "dlse")
    }
    out, lse, *_ = _flash_attn_fwd(
        q,
        k,
        v,
        tile_mn=(64, 64 if tile_m == 64 else 128),
        return_lse=True,
        num_splits=1,
        **fwd_kwargs,
    )
    if feature == "dlse":
        kwargs["dlse"] = torch.randn_like(lse)
    dout = torch.randn_like(out)
    grads_ref = _flash_attn_bwd(
        q,
        k,
        v,
        out,
        dout,
        lse,
        m_block_size=128,
        n_block_size=128,
        **kwargs,
    )
    grads = _flash_attn_bwd(
        q,
        k,
        v,
        out,
        dout,
        lse,
        m_block_size=tile_m,
        n_block_size=64,
        **kwargs,
    )
    assert_gradients_close(grads, grads_ref, atol=4e-3)


@pytest.mark.parametrize("tile_m", [128, 64])
def test_sm100_backward_tile_n64_varlen(tile_m):
    """Check ragged boundaries and partial 64-token KV tiles."""
    torch.manual_seed(1)
    lengths_q = [63, 129, 257]
    lengths_k = [65, 193, 256]
    q = torch.randn((sum(lengths_q), 2, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((sum(lengths_k), 2, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    kwargs = {
        "cu_seqlens_q": make_cu_seqlens(lengths_q),
        "cu_seqlens_k": make_cu_seqlens(lengths_k),
        "max_seqlen_q": max(lengths_q),
        "max_seqlen_k": max(lengths_k),
    }
    out, lse, *_ = _flash_attn_fwd(
        q=q,
        k=k,
        v=v,
        tile_mn=(64, 64 if tile_m == 64 else 128),
        return_lse=True,
        num_splits=1,
        **kwargs,
    )
    dout = torch.randn_like(out)
    grads_ref = _flash_attn_bwd(
        q,
        k,
        v,
        out,
        dout,
        lse,
        m_block_size=128,
        n_block_size=128,
        **kwargs,
    )
    grads = _flash_attn_bwd(
        q,
        k,
        v,
        out,
        dout,
        lse,
        m_block_size=tile_m,
        n_block_size=64,
        **kwargs,
    )
    assert_gradients_close(grads, grads_ref, atol=2e-3)


@pytest.mark.parametrize("head_dim", [16, 40, 48, 72, 80, 104, 112])
def test_sm100_backward_tile_n64_rejects_qk_dims_before_compile(head_dim):
    """Reject QK dimensions unsupported by dQ accumulator reduction."""
    q = torch.zeros((1, 8, 1, head_dim), device="cuda", dtype=torch.bfloat16)
    k = torch.zeros_like(q)
    v = torch.zeros((1, 8, 1, 64), device="cuda", dtype=torch.bfloat16)
    out = torch.zeros_like(v)
    dout = torch.zeros_like(v)
    lse = torch.zeros((1, 1, 8), device="cuda", dtype=torch.float32)

    with pytest.raises(NotImplementedError, match="32-column padded width"):
        _flash_attn_bwd(
            q,
            k,
            v,
            out,
            dout,
            lse,
            m_block_size=128,
            n_block_size=64,
        )


@pytest.mark.parametrize(
    "arch,head_dim,head_dim_v,match",
    [
        pytest.param(100, 32, 24, "V head dimension", id="padded-v"),
        pytest.param(103, 128, 128, "only on SM100", id="sm103"),
    ],
)
def test_sm100_backward_tile_n64_rejects_config_before_compile(
    monkeypatch, arch, head_dim, head_dim_v, match
):
    """Reject unsupported packed configurations before compilation."""
    q = torch.zeros((1, 8, 1, head_dim), device="cuda", dtype=torch.bfloat16)
    k = torch.zeros_like(q)
    v = torch.zeros((1, 8, 1, head_dim_v), device="cuda", dtype=torch.bfloat16)
    lse = torch.zeros((1, 1, 8), device="cuda", dtype=torch.float32)
    monkeypatch.setattr(flash_attn_interface, "_get_device_arch", lambda: arch)

    with pytest.raises(NotImplementedError, match=match):
        _flash_attn_bwd(
            q,
            k,
            v,
            torch.zeros_like(v),
            torch.zeros_like(v),
            lse,
            m_block_size=128,
            n_block_size=64,
        )


@pytest.mark.parametrize(
    "tile_n,head_dim,head_dim_v,expected",
    [
        pytest.param(64, 128, 128, (160, 128, 96, 96, 24), id="d128-dv128"),
        pytest.param(64, 120, 128, (160, 128, 96, 96, 24), id="d120-dv128"),
        pytest.param(64, 128, 112, (152, 136, 88, 88, 24), id="d128-dv112"),
        pytest.param(64, 96, 128, (152, 136, 88, 88, 24), id="d96-dv128"),
        pytest.param(64, 64, 64, (152, 136, 88, 88, 24), id="d64-dv64"),
        pytest.param(128, 128, 128, (152, 136, 88, 88, 24), id="default-tile"),
    ],
)
def test_sm100_backward_register_policy(tile_n, head_dim, head_dim_v, expected):
    """Keep the padded-128 packed policy on its measured selector boundary."""
    kernel = FlashAttentionBackwardSm100(
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        tile_m=128,
        tile_n=tile_n,
    )
    actual = (
        kernel.num_regs_reduce,
        kernel.num_regs_compute,
        kernel.num_regs_load,
        kernel.num_regs_mma,
        kernel.num_regs_empty,
    )
    assert actual == expected


@pytest.mark.parametrize(
    "head_dim,head_dim_v,expected_offsets",
    [
        pytest.param(24, 32, (0, 128, 160, 160, 224, 256), id="min-padded"),
        pytest.param(96, 80, (0, 128, 208, 208, 304, 400), id="asymmetric"),
        pytest.param(128, 128, (0, 128, 256, 256, 384, 512), id="max-footprint"),
    ],
)
def test_sm100_backward_tile_m64_n64_tmem_regions(
    head_dim, head_dim_v, expected_offsets
):
    """Keep live M64 accumulators disjoint within the 512-column TMEM allocation."""
    kernel = FlashAttentionBackwardSm100(
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        tile_m=64,
        tile_n=64,
    )
    actual = (
        kernel.tmem_S_offset,
        kernel.tmem_dV_offset,
        kernel.tmem_dP_offset,
        kernel.tmem_dQ_offset,
        kernel.tmem_dK_offset,
        kernel.tmem_dK_offset + kernel.tile_hdim,
    )
    assert actual == expected_offsets
    assert actual[-1] <= kernel.tmem_alloc_cols


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"qhead_per_kvhead": 2}, id="gqa"),
        pytest.param({"head_dim": 192, "head_dim_v": 128}, id="head-dim-192"),
        pytest.param({"cluster_size": 2, "use_2cta_instrs": True}, id="two-cta"),
        pytest.param({"use_2cta_instrs": True}, id="two-cta-flag"),
        pytest.param({"cluster_size": 2}, id="two-cta-cluster"),
        pytest.param({"is_persistent": True}, id="persistent-scheduler"),
        pytest.param({"head_dim": 32, "head_dim_v": 24}, id="padded-v-dimension"),
        pytest.param({"head_dim": 10, "head_dim_v": 16}, id="misaligned-head-dim"),
        pytest.param({"head_dim": 16, "head_dim_v": 32}, id="small-head-dim"),
        pytest.param({"head_dim": 40, "head_dim_v": 64}, id="qk-padded-width-48"),
    ],
)
def test_sm100_backward_tile_n64_rejects_outside_envelope(kwargs):
    """Keep unvalidated packed configurations fail-closed."""
    constructor_kwargs = {"head_dim": 128, **kwargs}
    with pytest.raises(NotImplementedError, match="tile_n=64"):
        FlashAttentionBackwardSm100(
            tile_m=128,
            tile_n=64,
            **constructor_kwargs,
        )
