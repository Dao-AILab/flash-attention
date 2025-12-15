import pytest
import torch
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import math as mlir_math
import operator
from torch.nn.attention.flex_attention import flex_attention
from flash_attn.cute.interface import _flash_attn_fwd
from score_mod_definitions import (
    # TensorSSA-based score mods
    score_mod_identity as score_mod_1,
    score_mod_causal as score_mod_2,
    score_mod_rel_bias as score_mod_3,
    score_mod_rel_bias_x2 as score_mod_4,
    score_mod_times_two as score_mod_5,
    score_mod_alibi as score_mod_6,
    score_mod_sliding_window as score_mod_7,
    score_mod_block_diagonal as score_mod_8,
    score_mod_causal_v2 as score_mod_9,
    score_mod_batch_bias as score_mod_10,
    score_mod_dual_buffer as score_mod_11,
)  # isort: split
from score_mod_definitions import (
    # Eager (torch) reference score mods
    identity_eager,
    causal_eager as causal_mask_eager,
    rel_bias_eager as relative_bias_eager,
    rel_bias_x2_eager as relative_bias_v2_eager,
    times_two_eager,
    alibi_eager as alibi_bias_eager,
    sliding_window_eager,
    block_diagonal_eager,
    causal_v2_eager as causal_mask_v2_eager,
    batch_bias_factory as batch_bias,
    dual_buffer_factory as dual_buffer_bias,
)

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

SEQLEN_CONFIGS = [
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


@pytest.mark.parametrize("seqlen_q,seqlen_kv", SEQLEN_CONFIGS)
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


@pytest.mark.parametrize("seqlen_q,seqlen_kv", SEQLEN_CONFIGS)
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


def _generate_block_kvcache(
    seqlen_k, page_size, batch_size, nheads_k, d, device, dtype
):
    import math
    from einops import rearrange

    num_blocks = math.ceil(seqlen_k / page_size) * batch_size * 3
    k_cache_paged = torch.randn(
        num_blocks, page_size, nheads_k, d, device=device, dtype=dtype
    )
    v_cache_paged = torch.randn(
        num_blocks, page_size, nheads_k, d, device=device, dtype=dtype
    )
    page_table = rearrange(
        torch.randperm(num_blocks, dtype=torch.int32, device=device),
        "(b nblocks) -> b nblocks",
        b=batch_size,
    )
    k_cache_bshd = rearrange(
        k_cache_paged[page_table.flatten()],
        "(b nblocks) block_size ... -> b (nblocks block_size) ...",
        b=batch_size,
    )[:, :seqlen_k]
    v_cache_bshd = rearrange(
        v_cache_paged[page_table.flatten()],
        "(b nblocks) block_size ... -> b (nblocks block_size) ...",
        b=batch_size,
    )[:, :seqlen_k]
    k_cache = k_cache_bshd.transpose(1, 2)
    v_cache = v_cache_bshd.transpose(1, 2)
    return k_cache, v_cache, page_table, k_cache_paged, v_cache_paged, num_blocks


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("page_size", [None, 1, 4, 128])
@pytest.mark.parametrize("qhead_per_kvhead,num_kv_heads", [(1, 2), (4, 2)])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_kv",
    [
        (1, 128),
        (64, 256),
        (64, 800),
        (256, 256),
        (113, 203),
    ],
)
@pytest.mark.parametrize("score_mod_pair", TEST_PAIRS)
def test_score_mod_with_paged_kvcache(
    seqlen_q,
    seqlen_kv,
    qhead_per_kvhead,
    num_kv_heads,
    page_size,
    dtype,
    score_mod_pair,
):
    if page_size is not None and seqlen_kv % page_size != 0:
        pytest.skip()

    torch.random.manual_seed(42)
    cute_score_mod, eager_score_mod = score_mod_pair

    batch_size = 2
    num_q_heads = num_kv_heads * qhead_per_kvhead
    pack_gqa = qhead_per_kvhead > 1
    dim = 128
    device = "cuda"

    q = torch.randn(batch_size, num_q_heads, seqlen_q, dim, device=device, dtype=dtype)

    if page_size is None:
        k_cache = torch.randn(
            batch_size, num_kv_heads, seqlen_kv, dim, device=device, dtype=dtype
        )
        v_cache = torch.randn(
            batch_size, num_kv_heads, seqlen_kv, dim, device=device, dtype=dtype
        )
        page_table = None
        k_cache_paged = None
        v_cache_paged = None
    else:
        (
            k_cache,
            v_cache,
            page_table,
            k_cache_paged,
            v_cache_paged,
            num_blocks,
        ) = _generate_block_kvcache(
            seqlen_kv, page_size, batch_size, num_kv_heads, dim, device, dtype
        )

    cache_seqlens = torch.randint(
        1, seqlen_kv + 1, (batch_size,), dtype=torch.int32, device=device
    )

    from einops import rearrange

    arange = rearrange(torch.arange(seqlen_kv, device=device), "s -> 1 s")
    cache_seqlens_expanded = rearrange(cache_seqlens, "b -> b 1")
    key_padding_mask = arange < cache_seqlens_expanded

    if pack_gqa:
        k_cache_rep = k_cache.repeat_interleave(qhead_per_kvhead, dim=1)
        v_cache_rep = v_cache.repeat_interleave(qhead_per_kvhead, dim=1)
    else:
        k_cache_rep = k_cache
        v_cache_rep = v_cache

    def make_masked_score_mod(base_score_mod, seqused_k_tensor):
        seqused_k_dev = seqused_k_tensor

        def masked_score_mod(score, b, h, q_idx, kv_idx):
            if base_score_mod is not None:
                score = base_score_mod(score, b, h, q_idx, kv_idx)
            seqlen_limit = torch.gather(seqused_k_dev, 0, b.long())
            valid_mask = kv_idx < seqlen_limit
            return torch.where(valid_mask, score, torch.full_like(score, float("-inf")))

        return masked_score_mod

    masked_score_mod_fp32 = make_masked_score_mod(eager_score_mod, cache_seqlens)
    masked_score_mod = make_masked_score_mod(eager_score_mod, cache_seqlens)

    out_ref_fp32 = run_flex_reference(
        q, k_cache_rep, v_cache_rep, masked_score_mod_fp32, dtype=torch.float32
    )
    out_pt = run_flex_reference(q, k_cache_rep, v_cache_rep, masked_score_mod)

    q_bshd = q.transpose(1, 2)
    out_cute = torch.empty_like(q_bshd)

    if page_size is None:
        k_bshd = k_cache.transpose(1, 2)
        v_bshd = v_cache.transpose(1, 2)
        _flash_attn_fwd(
            q_bshd,
            k_bshd,
            v_bshd,
            seqused_k=cache_seqlens,
            return_lse=True,
            score_mod=cute_score_mod,
            out=out_cute,
            lse=None,
            pack_gqa=pack_gqa,
        )
    else:
        _flash_attn_fwd(
            q_bshd,
            k_cache_paged,
            v_cache_paged,
            seqused_k=cache_seqlens,
            page_table=page_table,
            return_lse=True,
            score_mod=cute_score_mod,
            out=out_cute,
            lse=None,
            pack_gqa=pack_gqa,
        )

    out_cute = out_cute.transpose(1, 2)

    assert out_cute.shape == out_ref_fp32.shape == out_pt.shape
    assert not torch.isnan(out_cute).any()
    assert not torch.isnan(out_ref_fp32).any()
    assert not torch.isnan(out_pt).any()
    assert torch.isfinite(out_cute).all()
    assert torch.isfinite(out_ref_fp32).all()
    assert torch.isfinite(out_pt).all()

    fwd_atol = 2 * (out_ref_fp32 + 0.3 - 0.3 - out_ref_fp32).abs().max().item()
    rtol = 2

    pt_error = (out_pt - out_ref_fp32).abs().max().item()
    cute_error = (out_cute - out_ref_fp32).abs().max().item()

    print(
        f"\nNumerical comparison for {cute_score_mod.__name__} (paged={page_size is not None}):"
    )
    print(f"  PyTorch vs FP32 ref max error: {pt_error:.2e}")
    print(f"  CuTE vs FP32 ref max error: {cute_error:.2e}")
    print(f"  Dynamic absolute tolerance: {fwd_atol:.2e}")
    print(f"  Error ratio (CuTE/PyTorch): {cute_error / max(pt_error, 1e-10):.2f}")

    assert cute_error <= rtol * pt_error + fwd_atol, (
        f"CuTE error {cute_error:.2e} exceeds {rtol}x PyTorch error {pt_error:.2e} + {fwd_atol:.2e}"
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("page_size", [None, 128])
@pytest.mark.parametrize("qhead_per_kvhead,num_kv_heads", [(1, 1), (4, 2)])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_kv",
    [
        (64, 128),
        (128, 256),
        (256, 256),
    ],
)
@pytest.mark.parametrize("score_mod_pair", TEST_PAIRS_WITH_AUX_TENSORS)
def test_score_mod_with_paged_kvcache_aux_tensors(
    seqlen_q,
    seqlen_kv,
    qhead_per_kvhead,
    num_kv_heads,
    page_size,
    dtype,
    score_mod_pair,
):
    if page_size is not None and seqlen_kv % page_size != 0:
        pytest.skip()

    torch.random.manual_seed(42)
    cute_score_mod, eager_score_mod_factory = score_mod_pair

    batch_size = 2
    num_q_heads = num_kv_heads * qhead_per_kvhead
    pack_gqa = qhead_per_kvhead > 1
    dim = 128
    device = "cuda"

    q = torch.randn(batch_size, num_q_heads, seqlen_q, dim, device=device, dtype=dtype)

    if page_size is None:
        k_cache = torch.randn(
            batch_size, num_kv_heads, seqlen_kv, dim, device=device, dtype=dtype
        )
        v_cache = torch.randn(
            batch_size, num_kv_heads, seqlen_kv, dim, device=device, dtype=dtype
        )
        page_table = None
        k_cache_paged = None
        v_cache_paged = None
    else:
        (
            k_cache,
            v_cache,
            page_table,
            k_cache_paged,
            v_cache_paged,
            num_blocks,
        ) = _generate_block_kvcache(
            seqlen_kv, page_size, batch_size, num_kv_heads, dim, device, dtype
        )

    cache_seqlens = torch.randint(
        1, seqlen_kv + 1, (batch_size,), dtype=torch.int32, device=device
    )

    if cute_score_mod == score_mod_10:
        buffer = torch.randn(batch_size, device=device, dtype=dtype) * 0.1
        aux_tensors = [buffer]
        eager_score_mod = eager_score_mod_factory(buffer)
    elif cute_score_mod == score_mod_11:
        head_bias = torch.randn(num_q_heads, device=device, dtype=dtype) * 0.2
        pos_scale = torch.arange(seqlen_q, device=device, dtype=dtype) * 0.01
        aux_tensors = [head_bias, pos_scale]
        eager_score_mod = eager_score_mod_factory(head_bias, pos_scale)

    from einops import rearrange

    arange = rearrange(torch.arange(seqlen_kv, device=device), "s -> 1 s")
    cache_seqlens_expanded = rearrange(cache_seqlens, "b -> b 1")
    key_padding_mask = arange < cache_seqlens_expanded

    if pack_gqa:
        k_cache_rep = k_cache.repeat_interleave(qhead_per_kvhead, dim=1)
        v_cache_rep = v_cache.repeat_interleave(qhead_per_kvhead, dim=1)
    else:
        k_cache_rep = k_cache
        v_cache_rep = v_cache

    def make_masked_score_mod(base_score_mod, seqused_k_tensor):
        seqused_k_dev = seqused_k_tensor

        def masked_score_mod(score, b, h, q_idx, kv_idx):
            if base_score_mod is not None:
                score = base_score_mod(score, b, h, q_idx, kv_idx)
            seqlen_limit = torch.gather(seqused_k_dev, 0, b.long())
            valid_mask = kv_idx < seqlen_limit
            return torch.where(valid_mask, score, torch.full_like(score, float("-inf")))

        return masked_score_mod

    masked_score_mod_fp32 = make_masked_score_mod(eager_score_mod, cache_seqlens)
    masked_score_mod = make_masked_score_mod(eager_score_mod, cache_seqlens)

    out_ref_fp32 = run_flex_reference(
        q, k_cache_rep, v_cache_rep, masked_score_mod_fp32, dtype=torch.float32
    )
    out_pt = run_flex_reference(q, k_cache_rep, v_cache_rep, masked_score_mod)

    q_bshd = q.transpose(1, 2)
    out_cute = torch.empty_like(q_bshd)

    if page_size is None:
        k_bshd = k_cache.transpose(1, 2)
        v_bshd = v_cache.transpose(1, 2)
        _flash_attn_fwd(
            q_bshd,
            k_bshd,
            v_bshd,
            seqused_k=cache_seqlens,
            return_lse=True,
            score_mod=cute_score_mod,
            out=out_cute,
            lse=None,
            aux_tensors=aux_tensors,
            pack_gqa=pack_gqa,
        )
    else:
        _flash_attn_fwd(
            q_bshd,
            k_cache_paged,
            v_cache_paged,
            seqused_k=cache_seqlens,
            page_table=page_table,
            return_lse=True,
            score_mod=cute_score_mod,
            out=out_cute,
            lse=None,
            aux_tensors=aux_tensors,
            pack_gqa=pack_gqa,
        )

    out_cute = out_cute.transpose(1, 2)

    assert out_cute.shape == out_ref_fp32.shape == out_pt.shape
    assert not torch.isnan(out_cute).any()
    assert not torch.isnan(out_ref_fp32).any()
    assert not torch.isnan(out_pt).any()
    assert torch.isfinite(out_cute).all()
    assert torch.isfinite(out_ref_fp32).all()
    assert torch.isfinite(out_pt).all()

    fwd_atol = 2 * (out_ref_fp32 + 0.3 - 0.3 - out_ref_fp32).abs().max().item()
    rtol = 2

    pt_error = (out_pt - out_ref_fp32).abs().max().item()
    cute_error = (out_cute - out_ref_fp32).abs().max().item()

    print(
        f"\nNumerical comparison for {cute_score_mod.__name__} (paged={page_size is not None}):"
    )
    print(f"  PyTorch vs FP32 ref max error: {pt_error:.2e}")
    print(f"  CuTE vs FP32 ref max error: {cute_error:.2e}")
    print(f"  Dynamic absolute tolerance: {fwd_atol:.2e}")
    print(f"  Error ratio (CuTE/PyTorch): {cute_error / max(pt_error, 1e-10):.2f}")

    assert cute_error <= rtol * pt_error + fwd_atol, (
        f"CuTE error {cute_error:.2e} exceeds {rtol}x PyTorch error {pt_error:.2e} + {fwd_atol:.2e}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
