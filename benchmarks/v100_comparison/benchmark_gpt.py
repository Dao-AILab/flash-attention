"""V100 Attention Backend Benchmark: flash_attn vs PyTorch SDPA vs xFormers

Compares three attention backends on GPT-style Transformer models.

Usage:
    python benchmark_gpt.py                    # full benchmark
    python benchmark_gpt.py --dry-run          # single iteration sanity check
    python benchmark_gpt.py --backends flash_attn pytorch_sdpa  # subset
"""

import argparse
import gc
import math
import time
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from tabulate import tabulate

# --- Backend imports (graceful fallback) ---

try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

try:
    import xformers.ops as xops
    from xformers.ops import fmha
    HAS_XFORMERS = True
except ImportError:
    HAS_XFORMERS = False


# ---------------------------------------------------------------------------
# Attention backends
# ---------------------------------------------------------------------------

class AttentionBackend(Enum):
    FLASH_ATTN = "flash_attn"
    PYTORCH_SDPA = "pytorch_sdpa"
    XFORMERS = "xformers"


def attention_flash(q, k, v, causal):
    """flash_attn_func: (B, S, H, D) layout."""
    return flash_attn_func(q, k, v, causal=causal)


def attention_sdpa(q, k, v, causal):
    """PyTorch SDPA: needs (B, H, S, D) layout."""
    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)
    out = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=causal)
    return out.transpose(1, 2)


def attention_xformers(q, k, v, causal):
    """xFormers memory_efficient_attention: (B, S, H, D) layout, CUTLASS backend."""
    bias = xops.LowerTriangularMask() if causal else None
    return xops.memory_efficient_attention(
        q, k, v,
        attn_bias=bias,
        op=(fmha.cutlass.FwOp, fmha.cutlass.BwOp),
    )


BACKEND_FN = {
    AttentionBackend.FLASH_ATTN: attention_flash,
    AttentionBackend.PYTORCH_SDPA: attention_sdpa,
    AttentionBackend.XFORMERS: attention_xformers,
}


# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------

class GPTBlock(nn.Module):
    def __init__(self, d_model, n_heads, attn_fn):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.attn_fn = attn_fn
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, 4 * d_model, bias=False)
        self.ff2 = nn.Linear(4 * d_model, d_model, bias=False)

    def forward(self, x, causal):
        B, S, _ = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).reshape(B, S, 3, self.n_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        attn_out = self.attn_fn(q, k, v, causal)
        attn_out = attn_out.reshape(B, S, -1)
        x = x + self.out_proj(attn_out)
        x = x + self.ff2(F.gelu(self.ff1(self.ln2(x))))
        return x


class GPTModel(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, vocab_size, attn_fn):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(4096, d_model)  # max seqlen 4096
        self.blocks = nn.ModuleList(
            [GPTBlock(d_model, n_heads, attn_fn) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens, causal=True):
        B, S = tokens.shape
        pos = torch.arange(S, device=tokens.device).unsqueeze(0)
        x = self.embed(tokens) + self.pos_embed(pos)
        for block in self.blocks:
            x = block(x, causal)
        x = self.ln_f(x)
        return self.lm_head(x)


# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    "GPT-small": dict(d_model=768, n_heads=12, n_layers=12, vocab_size=50257),
    "GPT-medium": dict(d_model=1024, n_heads=16, n_layers=24, vocab_size=50257),
}


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def reset_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()


def pick_batch_size(model_name, seqlen):
    """Heuristic batch sizes to stay within V100 16GB."""
    if model_name == "GPT-small":
        if seqlen <= 512:
            return 16
        elif seqlen <= 1024:
            return 8
        else:
            return 4
    else:  # GPT-medium
        if seqlen <= 512:
            return 8
        elif seqlen <= 1024:
            return 4
        else:
            return 2


def measure_latency(model, tokens, targets, vocab_size, warmup=5, repeats=20):
    """Measure forward and backward latency using CUDA events."""
    # Warmup
    for _ in range(warmup):
        model.zero_grad(set_to_none=True)
        logits = model(tokens, causal=True)
        loss = F.cross_entropy(logits.view(-1, vocab_size).float(), targets.view(-1))
        loss.backward()
    torch.cuda.synchronize()

    fwd_times = []
    bwd_times = []

    for _ in range(repeats):
        model.zero_grad(set_to_none=True)

        start_fwd = torch.cuda.Event(enable_timing=True)
        end_fwd = torch.cuda.Event(enable_timing=True)
        start_bwd = torch.cuda.Event(enable_timing=True)
        end_bwd = torch.cuda.Event(enable_timing=True)

        start_fwd.record()
        logits = model(tokens, causal=True)
        loss = F.cross_entropy(logits.view(-1, vocab_size).float(), targets.view(-1))
        end_fwd.record()

        start_bwd.record()
        loss.backward()
        end_bwd.record()

        torch.cuda.synchronize()
        fwd_times.append(start_fwd.elapsed_time(end_fwd))
        bwd_times.append(start_bwd.elapsed_time(end_bwd))

    return sum(fwd_times) / len(fwd_times), sum(bwd_times) / len(bwd_times)


def measure_peak_memory(model, tokens, targets, vocab_size):
    """Measure peak GPU memory during forward + backward."""
    reset_memory()
    model.zero_grad(set_to_none=True)
    logits = model(tokens, causal=True)
    loss = F.cross_entropy(logits.view(-1, vocab_size).float(), targets.view(-1))
    loss.backward()
    torch.cuda.synchronize()
    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    return peak_mb


def measure_accuracy(model_ref, model_test, tokens, targets, vocab_size):
    """Compare logit error and gradient error between ref and test models."""
    model_ref.zero_grad(set_to_none=True)
    model_test.zero_grad(set_to_none=True)

    logits_ref = model_ref(tokens, causal=True)
    logits_test = model_test(tokens, causal=True)

    loss_ref = F.cross_entropy(logits_ref.view(-1, vocab_size).float(), targets.view(-1))
    loss_test = F.cross_entropy(logits_test.view(-1, vocab_size).float(), targets.view(-1))

    logit_err = (logits_ref.float() - logits_test.float()).abs().max().item()
    loss_err = abs(loss_ref.item() - loss_test.item())

    loss_ref.backward()
    loss_test.backward()

    max_grad_err = 0.0
    for (n1, p1), (n2, p2) in zip(
        model_ref.named_parameters(), model_test.named_parameters()
    ):
        if p1.grad is not None and p2.grad is not None:
            err = (p1.grad.float() - p2.grad.float()).abs().max().item()
            max_grad_err = max(max_grad_err, err)

    return logit_err, loss_err, max_grad_err


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def get_available_backends(requested):
    available = []
    for b in requested:
        if b == AttentionBackend.FLASH_ATTN and not HAS_FLASH_ATTN:
            print(f"WARNING: flash_attn not available, skipping")
            continue
        if b == AttentionBackend.XFORMERS and not HAS_XFORMERS:
            print(f"WARNING: xformers not available, skipping")
            continue
        available.append(b)
    return available


def run_benchmark(args):
    device = "cuda"
    dtype = torch.float16

    # Parse requested backends
    if args.backends:
        requested = [AttentionBackend(b) for b in args.backends]
    else:
        requested = list(AttentionBackend)
    backends = get_available_backends(requested)
    if not backends:
        print("ERROR: No backends available.")
        return

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Backends: {[b.value for b in backends]}")
    print(f"Mode: {'dry-run' if args.dry_run else 'full benchmark'}")
    print()

    seqlens = [512, 1024, 2048]

    for model_name, cfg in MODEL_CONFIGS.items():
        for seqlen in seqlens:
            batch_size = pick_batch_size(model_name, seqlen)
            vocab_size = cfg["vocab_size"]

            header = (
                f"=== {model_name} | seqlen={seqlen} | batch={batch_size} "
                f"| fp16 | causal ==="
            )
            print(header)

            # Create input tokens (same across backends)
            torch.manual_seed(42)
            tokens = torch.randint(0, vocab_size, (batch_size, seqlen), device=device)
            targets = torch.randint(0, vocab_size, (batch_size, seqlen), device=device)

            # Build reference model (flash_attn if available, else first backend)
            ref_backend = (
                AttentionBackend.FLASH_ATTN
                if AttentionBackend.FLASH_ATTN in backends
                else backends[0]
            )
            torch.manual_seed(123)
            ref_model = GPTModel(
                attn_fn=BACKEND_FN[ref_backend], **cfg
            ).to(device=device, dtype=dtype)

            rows = []

            for backend in backends:
                reset_memory()
                time.sleep(1)  # GPU temperature stabilization

                torch.manual_seed(123)
                model = GPTModel(
                    attn_fn=BACKEND_FN[backend], **cfg
                ).to(device=device, dtype=dtype)
                # Copy reference weights for accuracy comparison
                model.load_state_dict(ref_model.state_dict())

                row = {"Backend": backend.value}

                # --- Latency ---
                try:
                    if args.dry_run:
                        # Single forward/backward as sanity check
                        model.zero_grad(set_to_none=True)
                        logits = model(tokens, causal=True)
                        loss = F.cross_entropy(
                            logits.view(-1, vocab_size).float(), targets.view(-1)
                        )
                        loss.backward()
                        torch.cuda.synchronize()
                        row["Fwd(ms)"] = "-"
                        row["Bwd(ms)"] = "-"
                        row["Total(ms)"] = "OK"
                    else:
                        fwd_ms, bwd_ms = measure_latency(
                            model, tokens, targets, vocab_size
                        )
                        row["Fwd(ms)"] = f"{fwd_ms:.1f}"
                        row["Bwd(ms)"] = f"{bwd_ms:.1f}"
                        row["Total(ms)"] = f"{fwd_ms + bwd_ms:.1f}"
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache()
                        row["Fwd(ms)"] = "OOM"
                        row["Bwd(ms)"] = "OOM"
                        row["Total(ms)"] = "OOM"
                        row["PeakMem(MB)"] = "OOM"
                        row["Logit_Err"] = "-"
                        row["Grad_Err"] = "-"
                        rows.append(row)
                        del model
                        continue
                    raise

                # --- Memory ---
                try:
                    model.load_state_dict(ref_model.state_dict())
                    peak_mb = measure_peak_memory(model, tokens, targets, vocab_size)
                    row["PeakMem(MB)"] = f"{peak_mb:.0f}"
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache()
                        row["PeakMem(MB)"] = "OOM"
                    else:
                        raise

                # --- Accuracy ---
                try:
                    model.load_state_dict(ref_model.state_dict())
                    if backend == ref_backend:
                        row["Logit_Err"] = "(ref)"
                        row["Grad_Err"] = "(ref)"
                    else:
                        logit_err, loss_err, grad_err = measure_accuracy(
                            ref_model, model, tokens, targets, vocab_size
                        )
                        row["Logit_Err"] = f"{logit_err:.6f}"
                        row["Grad_Err"] = f"{grad_err:.6f}"
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache()
                        row["Logit_Err"] = "OOM"
                        row["Grad_Err"] = "OOM"
                    else:
                        raise

                rows.append(row)
                del model
                reset_memory()

            # Print results table
            if rows:
                columns = [
                    "Backend", "Fwd(ms)", "Bwd(ms)", "Total(ms)",
                    "PeakMem(MB)", "Logit_Err", "Grad_Err",
                ]
                print(tabulate(rows, headers="keys", tablefmt="simple", stralign="right"))
            print()

            del ref_model
            reset_memory()


def main():
    parser = argparse.ArgumentParser(description="V100 GPT Attention Backend Benchmark")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run single forward/backward per backend (sanity check)",
    )
    parser.add_argument(
        "--backends", nargs="+",
        choices=[b.value for b in AttentionBackend],
        help="Backends to benchmark (default: all available)",
    )
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
