import json
import math

import pytest
import torch

from flash_attn.cute import flash_attn_func, flash_attn_varlen_func
from flash_attn.cute.interface import _get_fwd_config

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0),
    reason="SM80 backward coverage",
)


DTYPES = [
    pytest.param(torch.float16, id="fp16"),
    pytest.param(torch.bfloat16, id="bf16"),
]
MODES = ["mha", "gqa", "mqa"]
LAYOUTS = ["dense", "varlen"]
HEAD_DIMS = [64, 128]


@pytest.mark.parametrize(
    ("head_dim", "expected_tile_m"),
    [(128, 128), (192, 64), (256, 64)],
)
def test_sm80_forward_tile_avoids_large_head_spills(head_dim, expected_tile_m):
    config = _get_fwd_config(
        arch=80,
        head_dim=head_dim,
        head_dim_v=head_dim,
        max_seqlen_q=480,
        max_seqlen_k=480,
        num_head_kv=2,
        qhead_per_kvhead=4,
        pack_gqa=True,
        batch_size=8,
        causal=True,
        local=False,
        window_size_left=None,
        window_size_right=None,
        num_splits=1,
        device=torch.device("cuda"),
    )
    assert config.m_block_size == expected_tile_m
    assert config.n_block_size == 64


def causal_mask_mod(batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
    return kv_idx <= q_idx + seqlen_info.seqlen_k - seqlen_info.seqlen_q


def parallel_chunks_mask_mod(
    batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors
):
    prefix_len = seqlen_info.seqlen_k - seqlen_info.seqlen_q
    chunk_size = seqlen_info.seqlen_q // 4
    suffix_kv_idx = kv_idx - prefix_len
    return (kv_idx < prefix_len) | (
        (suffix_kv_idx >= 0) & ((q_idx // chunk_size) == (suffix_kv_idx // chunk_size))
    )


MASK_MODS = {
    "causal_mask_mod": causal_mask_mod,
    "parallel_chunks": parallel_chunks_mask_mod,
}


def _lengths(layout):
    if layout == "dense":
        return (128, 128), (257, 257)
    return (128, 96), (257, 193)


def _head_counts(mode):
    return 8, {"mha": 8, "gqa": 2, "mqa": 1}[mode]


def _make_inputs(dtype, mode, layout, head_dim, seed, lengths=None):
    torch.manual_seed(seed)
    q_lengths, k_lengths = _lengths(layout) if lengths is None else lengths
    num_heads, num_kv_heads = _head_counts(mode)
    if layout == "dense":
        q_shape = (len(q_lengths), q_lengths[0], num_heads, head_dim)
        kv_shape = (len(k_lengths), k_lengths[0], num_kv_heads, head_dim)
    else:
        q_shape = (sum(q_lengths), num_heads, head_dim)
        kv_shape = (sum(k_lengths), num_kv_heads, head_dim)
    q = torch.randn(q_shape, device="cuda", dtype=dtype)
    k = torch.randn(kv_shape, device="cuda", dtype=dtype)
    v = torch.randn(kv_shape, device="cuda", dtype=dtype)
    dout = torch.randn(q_shape, device="cuda", dtype=dtype)
    return (q, k, v), dout, q_lengths, k_lengths


def _visible_mask(q_len, k_len, mask_kind, device):
    if mask_kind == "none":
        return None
    q_idx = torch.arange(q_len, device=device)
    kv_idx = torch.arange(k_len, device=device)
    prefix_len = k_len - q_len
    if mask_kind in ("causal", "causal_mask_mod"):
        return kv_idx[None, :] <= q_idx[:, None] + prefix_len
    if mask_kind == "parallel_chunks":
        chunk_size = q_len // 4
        suffix_kv_idx = kv_idx - prefix_len
        return (kv_idx[None, :] < prefix_len) | (
            (suffix_kv_idx[None, :] >= 0)
            & ((q_idx[:, None] // chunk_size) == (suffix_kv_idx[None, :] // chunk_size))
        )
    raise ValueError(f"Unknown mask kind: {mask_kind}")


def _reference_forward(q, k, v, layout, q_lengths, k_lengths, mask_kind):
    num_heads = q.shape[-2]
    num_kv_heads = k.shape[-2]
    repeat_factor = num_heads // num_kv_heads
    outputs = []
    q_offset = 0
    kv_offset = 0
    for batch_idx, (q_len, k_len) in enumerate(zip(q_lengths, k_lengths)):
        if layout == "dense":
            q_seq, k_seq, v_seq = q[batch_idx], k[batch_idx], v[batch_idx]
        else:
            q_seq = q[q_offset : q_offset + q_len]
            k_seq = k[kv_offset : kv_offset + k_len]
            v_seq = v[kv_offset : kv_offset + k_len]
        q_bhsd = q_seq.transpose(0, 1)
        k_bhsd = k_seq.transpose(0, 1).repeat_interleave(repeat_factor, dim=0)
        v_bhsd = v_seq.transpose(0, 1).repeat_interleave(repeat_factor, dim=0)
        visible = _visible_mask(q_len, k_len, mask_kind, q.device)
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])
        if q.dtype == torch.float32:
            scores = torch.einsum("hqd,hkd->hqk", q_bhsd * softmax_scale, k_bhsd)
        else:
            # Match the same-dtype reference used by the other architecture tests:
            # avoid upcasting and reorder the scale multiplication relative to FP32.
            scores = torch.einsum("hqd,hkd->hqk", q_bhsd, k_bhsd * softmax_scale)
        if visible is not None:
            scores = scores.masked_fill(~visible[None], float("-inf"))
        attention = torch.softmax(scores, dim=-1).to(v_bhsd.dtype)
        out = torch.einsum("hqk,hkd->qhd", attention, v_bhsd)
        outputs.append(out)
        q_offset += q_len
        kv_offset += k_len
    return torch.stack(outputs) if layout == "dense" else torch.cat(outputs)


def _run_reference(inputs, dout, layout, q_lengths, k_lengths, mask_kind, dtype):
    q, k, v = [tensor.detach().to(dtype).requires_grad_() for tensor in inputs]
    out = _reference_forward(q, k, v, layout, q_lengths, k_lengths, mask_kind)
    grads = torch.autograd.grad(out, (q, k, v), dout.to(dtype))
    return out.detach(), tuple(grad.detach() for grad in grads)


def _run_flash(
    inputs,
    dout,
    layout,
    q_lengths,
    k_lengths,
    mask_kind,
):
    q, k, v = [tensor.detach().requires_grad_() for tensor in inputs]
    mask_mod = MASK_MODS.get(mask_kind)
    kwargs = {
        "softmax_scale": 1.0 / math.sqrt(q.shape[-1]),
        "causal": mask_kind == "causal",
        "mask_mod": mask_mod,
        "pack_gqa": None,
        "return_lse": True,
    }
    if layout == "dense":
        out, _ = flash_attn_func(q, k, v, **kwargs)
    else:
        cu_seqlens_q = torch.tensor(
            [0, *torch.tensor(q_lengths).cumsum(0).tolist()],
            device="cuda",
            dtype=torch.int32,
        )
        cu_seqlens_k = torch.tensor(
            [0, *torch.tensor(k_lengths).cumsum(0).tolist()],
            device="cuda",
            dtype=torch.int32,
        )
        out, _ = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max(q_lengths),
            max_seqlen_k=max(k_lengths),
            **kwargs,
        )
    grads = torch.autograd.grad(out, (q, k, v), dout)
    return out.detach(), tuple(grad.detach() for grad in grads)


def _max_abs(tensor):
    return tensor.float().abs().max().item()


def _assert_numeric_error(name, actual, ref_dtype, ref_fp32, dtype):
    assert torch.isfinite(actual).all(), f"{name} contains NaN or Inf"
    ref_quantized = ref_fp32.to(dtype)
    pytorch_error = _max_abs(ref_dtype - ref_quantized)
    kernel_error = _max_abs(actual - ref_quantized)
    rounding_atol = 2 * _max_abs(ref_fp32 + 0.3 - 0.3 - ref_fp32)
    atol_floor = 1e-5
    limit = 2 * pytorch_error + max(atol_floor, rounding_atol)
    assert kernel_error <= limit, (
        f"{name} kernel error {kernel_error:.3e} exceeds "
        f"2 * PyTorch error {pytorch_error:.3e} + atol "
        f"{max(atol_floor, rounding_atol):.3e}"
    )
    return {
        "kernel_error": kernel_error,
        "pytorch_error": pytorch_error,
        "limit": limit,
    }


def _assert_stable(name, first, second, dtype):
    assert torch.isfinite(second).all(), f"repeated {name} contains NaN or Inf"
    drift = _max_abs(first - second)
    scale = max(1.0, _max_abs(first), _max_abs(second))
    limit = 4 * torch.finfo(dtype).eps * scale
    assert drift <= limit, (
        f"repeated {name} drift {drift:.3e} exceeds stability limit {limit:.3e}"
    )
    return drift


def _check_case(dtype, mode, layout, head_dim, mask_kind, lengths=None):
    seed = (
        1000
        + (0 if dtype == torch.float16 else 100)
        + MODES.index(mode) * 10
        + LAYOUTS.index(layout) * 3
        + (HEAD_DIMS.index(head_dim) if head_dim in HEAD_DIMS else head_dim)
    )
    inputs, dout, q_lengths, k_lengths = _make_inputs(
        dtype, mode, layout, head_dim, seed, lengths
    )
    actual_out, actual_grads = _run_flash(
        inputs, dout, layout, q_lengths, k_lengths, mask_kind
    )
    repeated_out, repeated_grads = _run_flash(
        inputs, dout, layout, q_lengths, k_lengths, mask_kind
    )
    ref_out, ref_grads = _run_reference(
        inputs, dout, layout, q_lengths, k_lengths, mask_kind, dtype
    )
    ref_out_fp32, ref_grads_fp32 = _run_reference(
        inputs, dout, layout, q_lengths, k_lengths, mask_kind, torch.float32
    )

    tensors = {
        "out": (actual_out, repeated_out, ref_out, ref_out_fp32),
        "dq": (actual_grads[0], repeated_grads[0], ref_grads[0], ref_grads_fp32[0]),
        "dk": (actual_grads[1], repeated_grads[1], ref_grads[1], ref_grads_fp32[1]),
        "dv": (actual_grads[2], repeated_grads[2], ref_grads[2], ref_grads_fp32[2]),
    }
    errors = {}
    drift = {}
    for name, (actual, repeated, ref_dtype, ref_fp32) in tensors.items():
        errors[name] = _assert_numeric_error(name, actual, ref_dtype, ref_fp32, dtype)
        drift[name] = _assert_stable(name, actual, repeated, dtype)
    print(
        "\nSM80_MATRIX "
        + json.dumps(
            {
                "dtype": str(dtype).removeprefix("torch."),
                "mode": mode,
                "layout": layout,
                "mask": mask_kind,
                "head_dim": head_dim,
                "errors": errors,
                "repeat_max_abs_drift": drift,
            },
            sort_keys=True,
        )
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("mask_kind", ["none", "causal"])
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_sm80_backward_matrix(dtype, mode, layout, mask_kind, head_dim):
    _check_case(dtype, mode, layout, head_dim, mask_kind)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("mask_kind", list(MASK_MODS))
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_sm80_mask_mod_backward_matrix(dtype, mode, layout, mask_kind, head_dim):
    _check_case(dtype, mode, layout, head_dim, mask_kind)


@pytest.mark.parametrize("mask_kind", ["none", "causal"])
def test_sm80_r2p_warp_n_partition_tail(mask_kind):
    # D=128 partitions the 8 MMA warps along N, so each thread's accumulator
    # columns are 0, 1, 32, 33, ... . A 16-column tail distinguishes the
    # layout-derived R2P stride from the SM90 stride of 8.
    _check_case(
        torch.bfloat16,
        "mha",
        "dense",
        128,
        mask_kind,
        lengths=((64, 64), (80, 80)),
    )


@pytest.mark.parametrize("mask_kind", ["none", "causal"])
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_sm80_dense_even_tiles(head_dim, mask_kind):
    _check_case(
        torch.bfloat16,
        "gqa",
        "dense",
        head_dim,
        mask_kind,
        lengths=((256, 256), (256, 256)),
    )


@pytest.mark.parametrize("layout", LAYOUTS)
def test_sm80_hd256_forward_backward_smoke(layout):
    lengths = ((96, 96), (96, 96)) if layout == "dense" else ((96, 80), (96, 80))
    _check_case(
        torch.bfloat16,
        "gqa",
        layout,
        256,
        "causal",
        lengths=lengths,
    )


@pytest.mark.parametrize("head_dim", [72, 144, 200])
@pytest.mark.parametrize("layout", LAYOUTS)
def test_sm80_padded_head_dim_not_divisible_by_64(head_dim, layout):
    lengths = ((256, 256), (256, 256)) if layout == "dense" else ((96, 80), (96, 80))
    _check_case(
        torch.bfloat16,
        "mha",
        layout,
        head_dim,
        "none",
        lengths=lengths,
    )
