import pytest
import torch
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import math as mlir_math
import operator
from torch.nn.attention.flex_attention import flex_attention
from flash_attn.cute.interface import _flash_attn_fwd


@cute.jit
def score_mod_1(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, aux_tensors):
    tmp0 = tSrS_ssa
    tSrS_ssa = tmp0
    return tSrS_ssa


@cute.jit
def score_mod_2(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, aux_tensors):
    tmp0 = q_idx
    tmp1 = kv_idx
    tmp2 = operator.ge(tmp0, tmp1)
    tmp3 = tSrS_ssa
    tmp4 = cute.where(tmp2, tmp3, cute.full_like(tmp3, float("-inf")))
    tSrS_ssa = tmp4
    return tSrS_ssa


@cute.jit
def score_mod_3(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, aux_tensors):
    tmp0 = tSrS_ssa
    tmp1 = q_idx
    tmp2 = kv_idx
    tmp3 = tmp1 - tmp2
    tmp4 = cute.TensorSSA(mlir_math.absi(tmp3), tmp3.shape, tmp3.dtype)
    tmp5 = tmp4.to(cutlass.Float32)
    tmp6 = tmp0 + tmp5
    tSrS_ssa = tmp6
    return tSrS_ssa


@cute.jit
def score_mod_4(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, aux_tensors):
    tmp0 = tSrS_ssa
    tmp1 = q_idx
    tmp2 = kv_idx
    tmp3 = tmp1 - tmp2
    tmp4 = cute.TensorSSA(mlir_math.absi(tmp3), tmp3.shape, tmp3.dtype)
    tmp5 = tmp4 * cute.full_like(tmp4, 2)
    tmp6 = tmp5.to(cutlass.Float32)
    tmp7 = tmp0 + tmp6
    tSrS_ssa = tmp7
    return tSrS_ssa


@cute.jit
def score_mod_5(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, aux_tensors):
    tmp0 = tSrS_ssa
    tmp1 = tmp0 * cute.full_like(tmp0, 2)
    tSrS_ssa = tmp1
    return tSrS_ssa


@cute.jit
def score_mod_6(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, aux_tensors):
    tmp0 = tSrS_ssa
    tmp1 = tmp0.to(cutlass.Float32)
    tmp2 = h_idx
    tmp3 = tmp2 + cute.full_like(tmp2, 1)
    tmp4 = tmp3 * cute.full_like(tmp3, -8)
    tmp5 = tmp4.to(cutlass.Float32)
    tmp6 = tmp5 * cute.full_like(tmp5, 0.125)
    tmp7 = tmp6 * cute.full_like(tmp6, 0.6931471805599453)
    tmp8 = cute.math.exp2(tmp7 * 1.4426950408889634)
    tmp9 = q_idx
    tmp10 = kv_idx
    tmp11 = tmp9 - tmp10
    tmp12 = cute.TensorSSA(mlir_math.absi(tmp11), tmp11.shape, tmp11.dtype)
    tmp13 = tmp12.to(cutlass.Float32)
    tmp14 = tmp8 * tmp13
    tmp15 = tmp1 - tmp14
    tSrS_ssa = tmp15
    return tSrS_ssa


@cute.jit
def score_mod_7(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, aux_tensors):
    tmp0 = q_idx
    tmp1 = kv_idx
    tmp2 = tmp0 - tmp1
    tmp3 = cute.TensorSSA(mlir_math.absi(tmp2), tmp2.shape, tmp2.dtype)
    tmp4 = operator.le(tmp3, cute.full_like(tmp3, 256))
    tmp5 = tSrS_ssa
    tmp6 = cute.where(tmp4, tmp5, cute.full_like(tmp5, float("-inf")))
    tSrS_ssa = tmp6
    return tSrS_ssa


@cute.jit
def score_mod_8(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, aux_tensors):
    tmp0 = q_idx
    tmp1 = kv_idx
    tmp2 = tSrS_ssa
    tmp3 = cute.where(
        operator.eq(tmp0 // 64, tmp1 // 64), tmp2, cute.full_like(tmp2, float("-inf"))
    )
    tSrS_ssa = tmp3
    return tSrS_ssa


@cute.jit
def score_mod_9(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, aux_tensors):
    tmp0 = q_idx
    tmp1 = kv_idx
    tmp2 = tmp0 - tmp1
    tmp3 = operator.ge(tmp2, cute.full_like(tmp2, 0))
    tmp4 = tSrS_ssa
    tmp5 = cute.where(tmp3, tmp4, cute.full_like(tmp4, float("-inf")))
    tSrS_ssa = tmp5
    return tSrS_ssa


@cute.jit
def score_mod_10(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, aux_tensors):
    batch_bias = aux_tensors[0]

    # Detect dtype from buffer element type
    dtype = batch_bias.element_type

    b_frag = cute.make_fragment(1, cutlass.Int32)
    b_frag.store(b_idx)
    bias_frag = cute.make_fragment(1, dtype)
    bias_frag[0] = batch_bias[b_frag[0]]
    bias_val = (bias_frag.load()).to(cutlass.Float32)

    return tSrS_ssa + bias_val


@cute.jit
def score_mod_11(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, aux_tensors):
    head_bias = aux_tensors[0]
    pos_bias = aux_tensors[1]

    # Detect dtype from buffer element type
    dtype = head_bias.element_type

    h_frag = cute.make_fragment(1, cutlass.Int32)
    h_frag.store(h_idx)
    head_val_frag = cute.make_fragment(1, dtype)
    head_val_frag[0] = head_bias[h_frag[0]]
    head_val = (head_val_frag.load()).to(cutlass.Float32)

    q_frag = cute.make_fragment(1, cutlass.Int32)
    q_frag.store(q_idx)
    pos_val_frag = cute.make_fragment(1, dtype)
    pos_val_frag[0] = pos_bias[q_frag[0]]
    pos_val = (pos_val_frag.load()).to(cutlass.Float32)

    return tSrS_ssa + head_val + pos_val


# Eager reference functions for comparison
def identity_eager(score, b, h, q_idx, kv_idx):
    return score


def causal_mask_eager(score, b, h, q_idx, kv_idx):
    return torch.where(q_idx >= kv_idx, score, float("-inf"))


def relative_bias_eager(score, b, h, q_idx, kv_idx):
    return score + torch.abs(q_idx - kv_idx)


def relative_bias_v2_eager(score, b, h, q_idx, kv_idx):
    return score + 2 * torch.abs(q_idx - kv_idx)


def times_two_eager(score, b, h, q_idx, kv_idx):
    return score * 2


def alibi_bias_eager(score, b, h, q_idx, kv_idx):
    slope = 2 ** (-8 * (h + 1) / 8)
    return score - slope * torch.abs(q_idx - kv_idx)


def sliding_window_eager(score, b, h, q_idx, kv_idx):
    return torch.where(torch.abs(q_idx - kv_idx) <= 256, score, float("-inf"))


def block_diagonal_eager(score, b, h, q_idx, kv_idx):
    q_block = q_idx // 64
    kv_block = kv_idx // 64
    return torch.where(q_block == kv_block, score, float("-inf"))


def causal_mask_v2_eager(score, b, h, q_idx, kv_idx):
    return torch.where(q_idx - kv_idx >= 0, score, float("-inf"))


def batch_bias(bias_tensor):
    """Per-batch bias (tests batch indexing)."""

    def batch_bias_mod(score, b, h, q_idx, kv_idx):
        return score + bias_tensor[b]

    return batch_bias_mod


def dual_buffer_bias(head_bias, pos_scale):
    """Dual buffer loading (tests loading from 2 separate tensors)."""

    def dual_buffer_mod(score, b, h, q_idx, kv_idx):
        head_component = head_bias[h]
        pos_component = pos_scale[q_idx]
        return score + pos_component + head_component

    return dual_buffer_mod


# Test pairs: (cute_jit_function, eager_reference_function)
TEST_PAIRS = [
    (score_mod_1, None),
    (score_mod_2, causal_mask_eager),
    (score_mod_3, relative_bias_eager),
    (score_mod_4, relative_bias_v2_eager),
    (score_mod_5, times_two_eager),
    (score_mod_6, alibi_bias_eager),
    (score_mod_7, sliding_window_eager),
    (score_mod_8, block_diagonal_eager),
    (score_mod_9, causal_mask_v2_eager),
]

# Test pairs with aux_tensors: (cute_jit_function, eager_reference_function_factory)
TEST_PAIRS_WITH_AUX_TENSORS = [
    (score_mod_10, batch_bias),
    (score_mod_11, dual_buffer_bias),
]


def create_tensors(
    batch_size=2, num_heads=4, seqlen_q=64, seqlen_kv=64, dim=128, dtype=torch.bfloat16
):
    q = torch.randn(batch_size, num_heads, seqlen_q, dim, device="cuda", dtype=dtype)
    k = torch.randn(batch_size, num_heads, seqlen_kv, dim, device="cuda", dtype=dtype)
    v = torch.randn(batch_size, num_heads, seqlen_kv, dim, device="cuda", dtype=dtype)
    return q, k, v


def run_cute_flash(
    q, k, v, cute_score_mod, aux_tensors=None, pack_gqa=False
) -> torch.Tensor:
    q_transposed, k_transposed, v_transposed = map(
        lambda x: x.transpose(1, 2), (q, k, v)
    )
    out = torch.empty_like(q_transposed)
    _flash_attn_fwd(
        q_transposed,
        k_transposed,
        v_transposed,
        return_lse=True,
        score_mod=cute_score_mod,
        out=out,
        lse=None,
        aux_tensors=aux_tensors,
        pack_gqa=pack_gqa,
    )
    return out.transpose(1, 2)


def run_flex_reference(q, k, v, eager_score_mod, dtype=None) -> torch.Tensor:
    if dtype is not None:
        q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
    return flex_attention(
        q, k, v, score_mod=eager_score_mod, enable_gqa=q.shape[1] != k.shape[1]
    )


@pytest.mark.parametrize(
    "seqlen_q,seqlen_kv",
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
@pytest.mark.parametrize("qhead_per_kvhead,num_kv_heads", [(1, 2), (4, 2)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("score_mod_pair", TEST_PAIRS)
def test_cute_vs_flex_attention(
    seqlen_q, seqlen_kv, qhead_per_kvhead, num_kv_heads, dtype, score_mod_pair
):
    torch.random.manual_seed(42)
    cute_score_mod, eager_score_mod = score_mod_pair

    num_q_heads = num_kv_heads * qhead_per_kvhead
    pack_gqa = qhead_per_kvhead > 1
    q, k, v = create_tensors(
        seqlen_q=seqlen_q, seqlen_kv=seqlen_kv, num_heads=num_q_heads, dtype=dtype
    )
    if pack_gqa:
        k = k[:, :num_kv_heads, :, :].clone()
        v = v[:, :num_kv_heads, :, :].clone()

    out_ref_fp32 = run_flex_reference(q, k, v, eager_score_mod, dtype=torch.float32)

    out_pt = run_flex_reference(q, k, v, eager_score_mod)
    out_cute = run_cute_flash(q, k, v, cute_score_mod, pack_gqa=pack_gqa)

    # Basic shape and NaN checks
    assert out_cute.shape == out_ref_fp32.shape == out_pt.shape
    assert not torch.isnan(out_cute).any()
    assert not torch.isnan(out_ref_fp32).any()
    assert not torch.isnan(out_pt).any()
    assert torch.isfinite(out_cute).all()
    assert torch.isfinite(out_ref_fp32).all()
    assert torch.isfinite(out_pt).all()

    # Numerical error if we just do any arithmetic on out_ref
    fwd_atol = 2 * (out_ref_fp32 + 0.3 - 0.3 - out_ref_fp32).abs().max().item()
    rtol = 2

    # Calculate actual errors
    pt_error = (out_pt - out_ref_fp32).abs().max().item()
    cute_error = (out_cute - out_ref_fp32).abs().max().item()

    print(f"\nNumerical comparison for {cute_score_mod.__name__}:")
    print(f"  PyTorch vs FP32 ref max error: {pt_error:.2e}")
    print(f"  CuTE vs FP32 ref max error: {cute_error:.2e}")
    print(f"  Dynamic absolute tolerance: {fwd_atol:.2e}")
    print(f"  Error ratio (CuTE/PyTorch): {cute_error / max(pt_error, 1e-10):.2f}")

    # Assert that CuTE's error is at most rtol times PyTorch's error + fwd_atol
    assert cute_error <= rtol * pt_error + fwd_atol, (
        f"CuTE error {cute_error:.2e} exceeds {rtol}x PyTorch error {pt_error:.2e} + {fwd_atol:.2e}"
    )


@pytest.mark.parametrize(
    "seqlen_q,seqlen_kv",
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
@pytest.mark.parametrize("qhead_per_kvhead,num_kv_heads", [(1, 1), (4, 2)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("score_mod_pair", TEST_PAIRS_WITH_AUX_TENSORS)
def test_cute_vs_flex_attention_with_aux_tensors(
    seqlen_q, seqlen_kv, qhead_per_kvhead, num_kv_heads, dtype, score_mod_pair
):
    torch.random.manual_seed(42)
    cute_score_mod, eager_score_mod_factory = score_mod_pair

    batch_size = 2
    num_q_heads = num_kv_heads * qhead_per_kvhead
    pack_gqa = qhead_per_kvhead > 1
    q, k, v = create_tensors(
        batch_size=batch_size,
        seqlen_q=seqlen_q,
        seqlen_kv=seqlen_kv,
        num_heads=num_q_heads,
        dtype=dtype,
    )
    if pack_gqa:
        k = k[:, :num_kv_heads, :, :].clone()
        v = v[:, :num_kv_heads, :, :].clone()

    if cute_score_mod == score_mod_10:
        buffer = torch.randn(batch_size, device="cuda", dtype=dtype) * 0.1
        aux_tensors = [buffer]
        eager_score_mod = eager_score_mod_factory(buffer)
        assert buffer.shape == (batch_size,)
    elif cute_score_mod == score_mod_11:
        head_bias = torch.randn(num_q_heads, device="cuda", dtype=dtype) * 0.2
        pos_scale = torch.arange(seqlen_q, device="cuda", dtype=dtype) * 0.01
        aux_tensors = [head_bias, pos_scale]
        eager_score_mod = eager_score_mod_factory(head_bias, pos_scale)
        assert head_bias.shape == (num_q_heads,)
        assert pos_scale.shape == (seqlen_q,)

    out_ref_fp32 = run_flex_reference(q, k, v, eager_score_mod, dtype=torch.float32)

    out_pt = run_flex_reference(q, k, v, eager_score_mod)
    out_cute = run_cute_flash(
        q, k, v, cute_score_mod, aux_tensors=aux_tensors, pack_gqa=pack_gqa
    )

    # Basic shape and NaN checks
    assert out_cute.shape == out_ref_fp32.shape == out_pt.shape
    assert not torch.isnan(out_cute).any()
    assert not torch.isnan(out_ref_fp32).any()
    assert not torch.isnan(out_pt).any()
    assert torch.isfinite(out_cute).all()
    assert torch.isfinite(out_ref_fp32).all()
    assert torch.isfinite(out_pt).all()

    # Numerical error if we just do any arithmetic on out_ref
    fwd_atol = 2 * (out_ref_fp32 + 0.3 - 0.3 - out_ref_fp32).abs().max().item()
    rtol = 2

    # Calculate actual errors
    pt_error = (out_pt - out_ref_fp32).abs().max().item()
    cute_error = (out_cute - out_ref_fp32).abs().max().item()

    print(f"\nNumerical comparison for {cute_score_mod.__name__}:")
    print(f"  PyTorch vs FP32 ref max error: {pt_error:.2e}")
    print(f"  CuTE vs FP32 ref max error: {cute_error:.2e}")
    print(f"  Dynamic absolute tolerance: {fwd_atol:.2e}")
    print(f"  Error ratio (CuTE/PyTorch): {cute_error / max(pt_error, 1e-10):.2f}")

    # Assert that CuTE's error is at most rtol times PyTorch's error + fwd_atol
    assert cute_error <= rtol * pt_error + fwd_atol, (
        f"CuTE error {cute_error:.2e} exceeds {rtol}x PyTorch error {pt_error:.2e} + {fwd_atol:.2e}"
    )


@pytest.mark.xfail(
    raises=NotImplementedError, reason="Varlen with score_mod not yet supported"
)
def test_varlen_with_score_mod():
    """Test that varlen (variable length sequences) works with score_mod.

    For varlen, tokens from different sequences should not attend to each other.
    Without proper index mapping, the causal mask will be applied to the global
    indices instead of per-sequence logical indices.
    """
    torch.random.manual_seed(42)

    seqlens = [64, 56, 128]
    total_seq = sum(seqlens)
    num_heads = 4
    dtype = torch.bfloat16

    cu_seqlens = torch.tensor(
        [0] + list(torch.tensor(seqlens).cumsum(0).tolist()),
        device="cuda",
        dtype=torch.int32,
    )
    q = torch.randn(total_seq, num_heads, 128, device="cuda", dtype=dtype)
    k = torch.randn(total_seq, num_heads, 128, device="cuda", dtype=dtype)
    v = torch.randn(total_seq, num_heads, 128, device="cuda", dtype=dtype)

    out_cute = torch.empty_like(q)

    _flash_attn_fwd(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        return_lse=True,
        score_mod=score_mod_2,
        out=out_cute,
        lse=None,
    )

    assert not torch.isnan(out_cute).any(), "Output contains NaN values"
    assert torch.isfinite(out_cute).all(), "Output contains infinite values"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
