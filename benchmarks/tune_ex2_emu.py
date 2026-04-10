#!/usr/bin/env python3
"""Sweep _TUNING_CONFIG parameters for flash_fwd_sm100.

Edits the _TUNING_CONFIG dict in flash_fwd_sm100.py, runs benchmarks, and
reports the best configuration. Restores the original file on exit.

Usage:
    CUDA_VISIBLE_DEVICES=7 python benchmarks/tune_ex2_emu_v2.py [--headdim 128] [--seqlen 8192]

Requires: the _TUNING_CONFIG dict in flash_attn/cute/flash_fwd_sm100.py.
"""
import argparse
import json
import re
import subprocess
import sys
import os
import torch

KERNEL_FILE = "flash_attn/cute/flash_fwd_sm100.py"

# ── Helpers ──────────────────────────────────────────────────────────────────

def read_file():
    with open(KERNEL_FILE) as f:
        return f.read()

def write_file(content):
    with open(KERNEL_FILE, "w") as f:
        f.write(content)

def find_tuning_block(src):
    """Return (start, end) indices of the _TUNING_CONFIG = { ... } block."""
    m = re.search(r'^_TUNING_CONFIG\s*=\s*\{', src, re.MULTILINE)
    assert m, "Could not find _TUNING_CONFIG in source"
    start = m.start()
    # Find matching closing brace
    depth = 0
    for i in range(m.end() - 1, len(src)):
        if src[i] == '{':
            depth += 1
        elif src[i] == '}':
            depth -= 1
            if depth == 0:
                return start, i + 1
    raise RuntimeError("Unmatched brace in _TUNING_CONFIG")

def parse_tuning_config(src):
    """Extract _TUNING_CONFIG as a Python dict."""
    start, end = find_tuning_block(src)
    block = src[start:end]
    ns = {}
    exec(block, ns)
    return ns["_TUNING_CONFIG"]

def serialize_tuning_config(config):
    """Serialize _TUNING_CONFIG back to source."""
    lines = ["_TUNING_CONFIG = {"]
    for key, val in config.items():
        use_2cta, is_causal, hdim, is_sm103 = key
        label_parts = []
        label_parts.append("2cta" if use_2cta else "1cta")
        label_parts.append("causal" if is_causal else "noncausal")
        label_parts.append(f"hdim={hdim}")
        if is_sm103:
            label_parts.append("sm103")
        label = ", ".join(label_parts)
        val_parts = []
        for k, v in val.items():
            val_parts.append(f'"{k}": {json.dumps(v)}')
        val_str = "{" + ", ".join(val_parts) + "}"
        lines.append(f"    {key!r}: {val_str},  # {label}")
    lines.append("}")
    return "\n".join(lines)

def patch_config(original_src, new_config):
    """Replace _TUNING_CONFIG block in source with new_config."""
    start, end = find_tuning_block(original_src)
    return original_src[:start] + serialize_tuning_config(new_config) + original_src[end:]

def detect_sm103():
    """Detect if the current GPU is SM103 (GB300)."""
    major, minor = torch.cuda.get_device_capability()
    sm = major * 10 + minor
    is_sm103 = sm >= 103 and sm <= 103  # sm_103 to sm_103f
    print(f"GPU: {torch.cuda.get_device_name()}, SM{sm}, is_sm103={is_sm103}")
    return is_sm103

def run_benchmark(causal_flag, headdim_str, seqlen, rep=20, warmup=10):
    """Run benchmark, return (ms, tflops, mfu) or (None, None, None).

    headdim_str: '128' or '192-128' (passed directly to --headdim).
    """
    result = subprocess.run(
        ["python", "benchmarks/benchmark_attn.py", "--fwd", "--backend", "fa4",
         "--headdim", headdim_str, f"--seqlen={seqlen}", "--rep", str(rep),
         "--warmup", str(warmup), "--causal", causal_flag],
        capture_output=True, text=True, timeout=600
    )
    output = result.stdout + result.stderr
    # Output lines look like: "      128  False     4   8192      1.38/1592/70.8%"
    # or:                      "  192-128  False     4   8192      1.38/1592/70.8%"
    for line in output.split("\n"):
        if ("True" in line or "False" in line):
            match = re.search(r'([\d.]+)/([\d.]+)/([\d.]+)%', line)
            if match:
                return float(match.group(1)), float(match.group(2)), float(match.group(3))
    print(f"  WARN: could not parse output:\n{output[-500:]}", file=sys.stderr)
    return None, None, None

# ── Main sweep ───────────────────────────────────────────────────────────────

def parse_headdim(s):
    """Parse headdim spec: '128' -> (128, 128), '192-128' -> (192, 128)."""
    if "-" in s:
        parts = s.split("-")
        return int(parts[0]), int(parts[1])
    hdim = int(s)
    return hdim, hdim

def parse_args():
    p = argparse.ArgumentParser(description="Tune _TUNING_CONFIG for flash_fwd_sm100")
    p.add_argument("--headdim", type=str, default="128",
                    help="Head dim spec: 128 or 192-128 (hdim-hdim_v)")
    p.add_argument("--seqlen", type=str, default="8192")
    p.add_argument("--rep", type=int, default=20)
    p.add_argument("--warmup", type=int, default=10)
    return p.parse_args()

def main():
    args = parse_args()
    original_src = read_file()
    config = parse_tuning_config(original_src)

    hdim, hdim_v = parse_headdim(args.headdim)
    # _TUNING_CONFIG keys use head_dim_padded (rounded up to multiple of 16)
    import math
    hdim_padded = int(math.ceil(hdim / 16) * 16)
    is_sm103 = detect_sm103()

    # Determine which keys to tune (matching hdim and detected arch)
    keys_to_tune = [k for k in config if k[2] == hdim_padded and k[3] == is_sm103]
    if not keys_to_tune:
        print(f"No _TUNING_CONFIG entries for hdim_padded={hdim_padded}, is_sm103={is_sm103}")
        print("Available keys:", [k for k in config])
        sys.exit(1)

    print(f"Tuning hdim={args.headdim}, hdim_padded={hdim_padded}, seqlen={args.seqlen}, is_sm103={is_sm103}")
    print(f"Keys to tune: {keys_to_tune}\n")

    # ── Phase 1: ex2_emu_freq + ex2_emu_start_frg sweep ──

    freq_values = [0, 6, 8, 10, 12, 14, 16, 20, 24, 32]
    start_frg_values = [0, 1]

    for key in keys_to_tune:
        use_2cta, is_causal, _, _ = key
        causal_flag = "true" if is_causal else "false"
        causal_label = "causal" if is_causal else "non-causal"
        cta_label = "2CTA" if use_2cta else "1CTA"

        print("=" * 70)
        print(f"Phase 1: ex2_emu sweep for {causal_label} ({cta_label})")
        print("=" * 70)
        print(f"{'freq':>5} {'start':>6} {'ms':>8} {'tflops':>10} {'mfu':>8}")
        print("-" * 45)

        best_freq = config[key]["ex2_emu_freq"]
        best_start = config[key]["ex2_emu_start_frg"]
        best_tflops = 0

        for start_frg in start_frg_values:
            for freq in freq_values:
                test_config = dict(config)
                test_config[key] = {**config[key], "ex2_emu_freq": freq, "ex2_emu_start_frg": start_frg}
                write_file(patch_config(original_src, test_config))
                try:
                    ms, tflops, mfu = run_benchmark(causal_flag, args.headdim, args.seqlen, args.rep, args.warmup)
                    if tflops is None:
                        print(f"{freq:>5} {start_frg:>6}  ERROR")
                        continue
                    marker = " ***" if tflops > best_tflops else ""
                    print(f"{freq:>5} {start_frg:>6} {ms:>8.2f} {tflops:>10.0f} {mfu:>8.1f}{marker}")
                    if tflops > best_tflops:
                        best_tflops = tflops
                        best_freq = freq
                        best_start = start_frg
                except Exception as e:
                    print(f"{freq:>5} {start_frg:>6}  ERROR: {e}")
                sys.stdout.flush()

        print(f"\n  Best: freq={best_freq}, start_frg={best_start}, {best_tflops:.0f} TFLOPS")
        config[key] = {**config[key], "ex2_emu_freq": best_freq, "ex2_emu_start_frg": best_start}

    # ── Phase 2: Register count sweep (softmax, correction; other = 512 - 2*softmax - correction) ──

    reg_combos = []
    for softmax in [176, 184, 192, 200]:
        for correction in [56, 64, 72, 80, 88, 96]:
            other = 512 - softmax * 2 - correction
            if other >= 24:  # need some minimum for other warps
                reg_combos.append((softmax, correction, other))

    for key in keys_to_tune:
        use_2cta, is_causal, _, _ = key
        causal_flag = "true" if is_causal else "false"
        causal_label = "causal" if is_causal else "non-causal"
        cta_label = "2CTA" if use_2cta else "1CTA"

        print("\n" + "=" * 70)
        print(f"Phase 2: Register sweep for {causal_label} ({cta_label})")
        print("=" * 70)
        print(f"{'softmax':>8} {'corr':>6} {'other':>6} {'ms':>8} {'tflops':>10} {'mfu':>8}")
        print("-" * 55)

        best_softmax = config[key]["num_regs_softmax"]
        best_correction = config[key]["num_regs_correction"]
        best_tflops = 0

        for softmax, correction, other in reg_combos:
            test_config = dict(config)
            test_config[key] = {**config[key],
                "num_regs_softmax": softmax,
                "num_regs_correction": correction,
            }
            write_file(patch_config(original_src, test_config))
            try:
                ms, tflops, mfu = run_benchmark(causal_flag, args.headdim, args.seqlen, args.rep, args.warmup)
                if tflops is None:
                    print(f"{softmax:>8} {correction:>6} {other:>6}  ERROR")
                    continue
                marker = " ***" if tflops > best_tflops else ""
                print(f"{softmax:>8} {correction:>6} {other:>6} {ms:>8.2f} {tflops:>10.0f} {mfu:>8.1f}{marker}")
                if tflops > best_tflops:
                    best_tflops = tflops
                    best_softmax = softmax
                    best_correction = correction
            except Exception as e:
                print(f"{softmax:>8} {correction:>6} {other:>6}  ERROR: {e}")
            sys.stdout.flush()

        best_other = 512 - best_softmax * 2 - best_correction
        print(f"\n  Best: softmax={best_softmax}, correction={best_correction}, other={best_other}, {best_tflops:.0f} TFLOPS")
        config[key] = {**config[key],
            "num_regs_softmax": best_softmax,
            "num_regs_correction": best_correction,
        }

    # ── Final summary ──

    print("\n" + "=" * 70)
    print("FINAL BEST CONFIG")
    print("=" * 70)
    for key in keys_to_tune:
        val_parts = ", ".join(f'"{k}": {json.dumps(v)}' for k, v in config[key].items())
        print(f"    {key!r}: {{{val_parts}}},")


    print(f"\nTo apply, update _TUNING_CONFIG in {KERNEL_FILE}")

    # Restore original
    write_file(original_src)
    print("Restored original file.")

if __name__ == "__main__":
    main()
