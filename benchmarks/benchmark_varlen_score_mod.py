"""Benchmark varlen attention with score_mod."""
import sys
from pathlib import Path

# Add tests/cute to path to import score_mod_definitions
sys.path.insert(0, str(Path(__file__).parent.parent / "tests" / "cute"))

import torch
from triton.testing import do_bench
from flash_attn.cute.interface import _flash_attn_fwd
import cutlass.cute as cute

from score_mod_definitions import (
    score_mod_identity,
    score_mod_causal,
    score_mod_rel_bias,
    score_mod_rel_bias_x2,
    score_mod_times_two,
    score_mod_global_kv_bias,
    score_mod_global_q_bias,
    score_mod_global_q_and_kv_bias,
)


# Score_mods for non-varlen with flattened batched aux tensors (same data as packed)
@cute.jit
def score_mod_batched_q_bias(score, b_idx, h_idx, q_idx, kv_idx, aux_tensors):
    """Non-varlen: aux_tensors[0] is flat (batch*seqlen_q,), [1] is seqlen_q scalar"""
    q_bias_flat = aux_tensors[0]
    seqlen_q_tensor = aux_tensors[1]
    dtype = q_bias_flat.element_type

    # Read seqlen_q scalar
    seqlen_frag = cute.make_fragment(1, cutlass.Int32)
    seqlen_frag[0] = seqlen_q_tensor[0]
    seqlen_q = seqlen_frag[0]

    # Compute linear index: b * seqlen + q_local
    b_frag = cute.make_fragment(1, cutlass.Int32)
    b_frag.store(b_idx)
    q_frag = cute.make_fragment(1, cutlass.Int32)
    q_frag.store(q_idx)
    linear_idx = b_frag[0] * seqlen_q + q_frag[0]

    # Read bias
    bias_frag = cute.make_fragment(1, dtype)
    bias_frag[0] = q_bias_flat[linear_idx]
    return score + (bias_frag.load()).to(cutlass.Float32)


@cute.jit
def score_mod_batched_kv_bias(score, b_idx, h_idx, q_idx, kv_idx, aux_tensors):
    """Non-varlen: aux_tensors[0] is flat (batch*seqlen_k,), [1] is seqlen_k scalar"""
    kv_bias_flat = aux_tensors[0]
    seqlen_k_tensor = aux_tensors[1]
    dtype = kv_bias_flat.element_type

    # Read seqlen_k scalar
    seqlen_frag = cute.make_fragment(1, cutlass.Int32)
    seqlen_frag[0] = seqlen_k_tensor[0]
    seqlen_k = seqlen_frag[0]

    # Compute linear index: b * seqlen + kv_local
    b_frag = cute.make_fragment(1, cutlass.Int32)
    b_frag.store(b_idx)
    kv_frag = cute.make_fragment(1, cutlass.Int32)
    kv_frag.store(kv_idx)
    linear_idx = b_frag[0] * seqlen_k + kv_frag[0]

    # Read bias
    bias_frag = cute.make_fragment(1, dtype)
    bias_frag[0] = kv_bias_flat[linear_idx]
    return score + (bias_frag.load()).to(cutlass.Float32)


@cute.jit
def score_mod_batched_q_and_kv_bias(score, b_idx, h_idx, q_idx, kv_idx, aux_tensors):
    """Non-varlen: aux[0]=flat Q, [1]=flat KV, [2]=seqlen_q, [3]=seqlen_k"""
    q_bias_flat = aux_tensors[0]
    kv_bias_flat = aux_tensors[1]
    seqlen_q_tensor = aux_tensors[2]
    seqlen_k_tensor = aux_tensors[3]
    dtype = q_bias_flat.element_type

    # Read seqlens
    seqlen_q_frag = cute.make_fragment(1, cutlass.Int32)
    seqlen_q_frag[0] = seqlen_q_tensor[0]
    seqlen_q = seqlen_q_frag[0]

    seqlen_k_frag = cute.make_fragment(1, cutlass.Int32)
    seqlen_k_frag[0] = seqlen_k_tensor[0]
    seqlen_k = seqlen_k_frag[0]

    # Compute linear indices
    b_frag = cute.make_fragment(1, cutlass.Int32)
    b_frag.store(b_idx)
    b_val = b_frag[0]

    q_frag = cute.make_fragment(1, cutlass.Int32)
    q_frag.store(q_idx)
    q_linear_idx = b_val * seqlen_q + q_frag[0]

    kv_frag = cute.make_fragment(1, cutlass.Int32)
    kv_frag.store(kv_idx)
    kv_linear_idx = b_val * seqlen_k + kv_frag[0]

    # Read biases
    q_bias_frag = cute.make_fragment(1, dtype)
    q_bias_frag[0] = q_bias_flat[q_linear_idx]

    kv_bias_frag = cute.make_fragment(1, dtype)
    kv_bias_frag[0] = kv_bias_flat[kv_linear_idx]

    return score + (q_bias_frag.load()).to(cutlass.Float32) + (kv_bias_frag.load()).to(cutlass.Float32)


def time_fn(func, *args, repeats=30, **kwargs):
    """Time a function using triton's do_bench."""
    return do_bench(lambda: func(*args, **kwargs), warmup=10, rep=repeats)


def flops(batch, nheads, seqlen_q, seqlen_k, headdim, causal=False):
    """Compute FLOPs for attention."""
    if causal:
        avg_seqlen_k = (max(0, seqlen_k - seqlen_q) + seqlen_k) / 2
    else:
        avg_seqlen_k = seqlen_k
    return batch * nheads * 2 * seqlen_q * avg_seqlen_k * headdim * 2


# Configuration
torch.manual_seed(0)
device = "cuda"
dtype = torch.bfloat16
repeats = 50
causal = False

# Test configurations: (batch_size, seqlen_q, seqlen_k, nheads, headdim)
configs = [
    # Equal length Q and K
    (4, 2048, 2048, 16, 128),
    (4, 4096, 4096, 16, 128),
    (4, 8192, 8192, 16, 128),
    # Short Q, long K (decoding/generation scenarios)
    (4, 1, 2048, 16, 128),
    (4, 1, 8192, 16, 128),
    # (4, 128, 8192, 16, 128),
    # (4, 256, 4096, 16, 128),
]

print("\n" + "=" * 110)
print(f"{'Config':<35} | {'Varlen Mode':<20} | {'Time (ms)':<12} | {'TFLOPS':<10} | {'vs Non-varlen':<12}")
print("=" * 110)

# Define score_mods to test
score_mods = [
    ("None", None),
    ("Identity", score_mod_identity),
    ("RelBias", score_mod_rel_bias),
    ("RelBias_x2", score_mod_rel_bias_x2),  # More complex index arithmetic
    ("TimesTwo", score_mod_times_two),
]

for batch_size, seqlen_q, seqlen_k, nheads, headdim in configs:
    # Create tensors
    q = torch.randn(batch_size, seqlen_q, nheads, headdim, device=device, dtype=dtype)
    k = torch.randn(batch_size, seqlen_k, nheads, headdim, device=device, dtype=dtype)
    v = torch.randn(batch_size, seqlen_k, nheads, headdim, device=device, dtype=dtype)

    # Create varlen tensors (packed format with equal-length sequences)
    total_q = batch_size * seqlen_q
    total_k = batch_size * seqlen_k
    q_varlen = q.reshape(total_q, nheads, headdim)
    k_varlen = k.reshape(total_k, nheads, headdim)
    v_varlen = v.reshape(total_k, nheads, headdim)
    cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, seqlen_q, device=device, dtype=torch.int32)
    cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_k, seqlen_k, device=device, dtype=torch.int32)

    nFLOPS = flops(batch_size, nheads, seqlen_q, seqlen_k, headdim, causal=causal)
    # Use compact format when Q and K lengths are equal
    if seqlen_q == seqlen_k:
        config_str = f"B={batch_size} S={seqlen_q} H={nheads} D={headdim}"
    else:
        config_str = f"B={batch_size} Sq={seqlen_q} Sk={seqlen_k} H={nheads} D={headdim}"

    # Test each score_mod with different varlen modes
    for score_mod_name, score_mod in score_mods:
        print(f"\n--- {config_str} | Score Mod: {score_mod_name} ---")

        # Define varlen modes for this score_mod
        varlen_modes = [
            ("Non-varlen", q, k, v, None, None),
            ("Varlen Q+K", q_varlen, k_varlen, v_varlen, cu_seqlens_q, cu_seqlens_k),
            ("Varlen Q only", q_varlen, k, v, cu_seqlens_q, None),
            ("Varlen K only", q, k_varlen, v_varlen, None, cu_seqlens_k),
        ]

        baseline_time = None
        for mode_name, q_test, k_test, v_test, cu_q, cu_k in varlen_modes:
            try:
                # Allocate output buffer
                out = torch.empty_like(q_test)

                def run_test():
                    _flash_attn_fwd(
                        q_test,
                        k_test,
                        v_test,
                        cu_seqlens_q=cu_q,
                        cu_seqlens_k=cu_k,
                        causal=causal,
                        score_mod=score_mod,
                        out=out,
                        lse=None,
                        return_lse=False,
                    )
                    return out

                time_ms = time_fn(run_test, repeats=repeats)
                tflops = nFLOPS / (time_ms * 1e-3) * 1e-12

                # Track baseline (non-varlen) time for comparison
                if mode_name == "Non-varlen":
                    baseline_time = time_ms
                    slowdown_str = "-"
                else:
                    slowdown = (time_ms / baseline_time - 1) * 100 if baseline_time else 0
                    slowdown_str = f"+{slowdown:.1f}%" if slowdown > 0 else f"{slowdown:.1f}%"

                print(f"  {mode_name:<20} | {time_ms:>10.3f} ms | {tflops:>8.1f} | {slowdown_str:>10}")
            except Exception as e:
                print(f"  {mode_name:<20} | {'ERROR':<12} | {str(e)[:50]}")

    # Test score_mods with aux_tensors: batched (non-varlen) vs packed (varlen)
    # Use the SAME bias values in both formats to isolate varlen overhead
    print(f"\n--- {config_str} | Score Mods with Aux Tensors ---")

    # Create aux tensors - same data in batched and packed formats
    # Batched: (batch, seqlen) flattened to (batch*seqlen,) - for non-varlen with manual 2D indexing
    # Packed: (batch*seqlen,) - for varlen with global indexing (same data!)
    q_bias_batched = torch.randn(batch_size, seqlen_q, device=device, dtype=dtype) * 0.1
    k_bias_batched = torch.randn(batch_size, seqlen_k, device=device, dtype=dtype) * 0.1
    q_bias_packed = q_bias_batched.reshape(total_q)  # Same data, just reshaped
    k_bias_packed = k_bias_batched.reshape(total_k)  # Same data, just reshaped

    # Seqlen scalars for non-varlen 2D indexing
    seqlen_q_scalar = torch.tensor([seqlen_q], device=device, dtype=torch.int32)
    seqlen_k_scalar = torch.tensor([seqlen_k], device=device, dtype=torch.int32)

    # Test Q bias: batched (2D manual indexing) vs packed (1D global indexing)
    print("\n  Q Bias (batched vs packed):")
    aux_test_cases = [
        ("Non-varlen (batched)", score_mod_batched_q_bias, [q_bias_packed, seqlen_q_scalar],
         q, k, v, None, None),
        ("Varlen Q only (packed)", score_mod_global_q_bias, [q_bias_packed],
         q_varlen, k, v, cu_seqlens_q, None),
        ("Varlen Q+K (packed)", score_mod_global_q_bias, [q_bias_packed],
         q_varlen, k_varlen, v_varlen, cu_seqlens_q, cu_seqlens_k),
    ]

    baseline_time = None
    for test_name, score_mod, aux_tensors, q_test, k_test, v_test, cu_q, cu_k in aux_test_cases:
        try:
            out = torch.empty_like(q_test)

            def run_test():
                _flash_attn_fwd(
                    q_test, k_test, v_test,
                    cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                    causal=causal, score_mod=score_mod, aux_tensors=aux_tensors,
                    out=out, lse=None, return_lse=False,
                )
                return out

            time_ms = time_fn(run_test, repeats=repeats)
            tflops = nFLOPS / (time_ms * 1e-3) * 1e-12

            if "Non-varlen" in test_name:
                baseline_time = time_ms
                slowdown_str = "-"
            else:
                slowdown = (time_ms / baseline_time - 1) * 100 if baseline_time else 0
                slowdown_str = f"+{slowdown:.1f}%" if slowdown > 0 else f"{slowdown:.1f}%"

            print(f"    {test_name:<30} | {time_ms:>10.3f} ms | {tflops:>8.1f} | {slowdown_str:>10}")
        except Exception as e:
            print(f"    {test_name:<30} | {'ERROR':<12} | {str(e)[:50]}")

    # Test KV bias: batched vs packed
    print("\n  KV Bias (batched vs packed):")
    aux_test_cases = [
        ("Non-varlen (batched)", score_mod_batched_kv_bias, [k_bias_packed, seqlen_k_scalar],
         q, k, v, None, None),
        ("Varlen K only (packed)", score_mod_global_kv_bias, [k_bias_packed],
         q, k_varlen, v_varlen, None, cu_seqlens_k),
        ("Varlen Q+K (packed)", score_mod_global_kv_bias, [k_bias_packed],
         q_varlen, k_varlen, v_varlen, cu_seqlens_q, cu_seqlens_k),
    ]

    baseline_time = None
    for test_name, score_mod, aux_tensors, q_test, k_test, v_test, cu_q, cu_k in aux_test_cases:
        try:
            out = torch.empty_like(q_test)

            def run_test():
                _flash_attn_fwd(
                    q_test, k_test, v_test,
                    cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                    causal=causal, score_mod=score_mod, aux_tensors=aux_tensors,
                    out=out, lse=None, return_lse=False,
                )
                return out

            time_ms = time_fn(run_test, repeats=repeats)
            tflops = nFLOPS / (time_ms * 1e-3) * 1e-12

            if "Non-varlen" in test_name:
                baseline_time = time_ms
                slowdown_str = "-"
            else:
                slowdown = (time_ms / baseline_time - 1) * 100 if baseline_time else 0
                slowdown_str = f"+{slowdown:.1f}%" if slowdown > 0 else f"{slowdown:.1f}%"

            print(f"    {test_name:<30} | {time_ms:>10.3f} ms | {tflops:>8.1f} | {slowdown_str:>10}")
        except Exception as e:
            print(f"    {test_name:<30} | {'ERROR':<12} | {str(e)[:50]}")

    # Test Q+KV bias: batched vs packed
    print("\n  Q+KV Bias (batched vs packed):")
    aux_test_cases = [
        ("Non-varlen (batched)", score_mod_batched_q_and_kv_bias,
         [q_bias_packed, k_bias_packed, seqlen_q_scalar, seqlen_k_scalar],
         q, k, v, None, None),
        ("Varlen Q+K (packed)", score_mod_global_q_and_kv_bias, [q_bias_packed, k_bias_packed],
         q_varlen, k_varlen, v_varlen, cu_seqlens_q, cu_seqlens_k),
    ]

    baseline_time = None
    for test_name, score_mod, aux_tensors, q_test, k_test, v_test, cu_q, cu_k in aux_test_cases:
        try:
            out = torch.empty_like(q_test)

            def run_test():
                _flash_attn_fwd(
                    q_test, k_test, v_test,
                    cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                    causal=causal, score_mod=score_mod, aux_tensors=aux_tensors,
                    out=out, lse=None, return_lse=False,
                )
                return out

            time_ms = time_fn(run_test, repeats=repeats)
            tflops = nFLOPS / (time_ms * 1e-3) * 1e-12

            if "Non-varlen" in test_name:
                baseline_time = time_ms
                slowdown_str = "-"
            else:
                slowdown = (time_ms / baseline_time - 1) * 100 if baseline_time else 0
                slowdown_str = f"+{slowdown:.1f}%" if slowdown > 0 else f"{slowdown:.1f}%"

            print(f"    {test_name:<30} | {time_ms:>10.3f} ms | {tflops:>8.1f} | {slowdown_str:>10}")
        except Exception as e:
            print(f"    {test_name:<30} | {'ERROR':<12} | {str(e)[:50]}")

print("\n" + "=" * 110)
