#!/usr/bin/env python3
import torch
import math
import subprocess
import json
import os
import time
import re
from datetime import datetime

try:
    from flash_attn import flash_attn_qkvpacked_func
    HAS_FLASH = True
except:
    HAS_FLASH = False

WALLER_BINARY = os.path.expanduser("~/waller-eval/waller_eval_cli_x86")

def get_gpu_info():
    return {
        "name": torch.cuda.get_device_name(0),
        "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3)
    }

def flops(batch, seqlen, headdim, nheads, causal, mode="fwd"):
    """Calculate FLOPs following FlashAttention's methodology"""
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)

def efficiency(flop, time_seconds):
    """Calculate TFLOPS/s efficiency"""
    return (flop / time_seconds / 10**12) if time_seconds > 0 else 0.0

def measure_pytorch(bs, sl, nh, hd, causal=True, warmup=5, repeats=30):
    """Measure PyTorch attention - FORWARD PASS ONLY"""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    try:
        q = torch.randn(bs, nh, sl, hd, device='cuda', dtype=torch.float16)
        k = torch.randn(bs, nh, sl, hd, device='cuda', dtype=torch.float16)
        v = torch.randn(bs, nh, sl, hd, device='cuda', dtype=torch.float16)
        
        # Warmup
        for _ in range(warmup):
            with torch.no_grad():
                s = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(hd)
                if causal:
                    mask = torch.triu(torch.ones(sl, sl, device='cuda', dtype=torch.bool), diagonal=1)
                    s.masked_fill_(mask, float('-inf'))
                attn = torch.softmax(s, dim=-1)
                out = torch.matmul(attn, v)
            torch.cuda.synchronize()
        
        # Measure
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        for _ in range(repeats):
            with torch.no_grad():
                s = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(hd)
                if causal:
                    mask = torch.triu(torch.ones(sl, sl, device='cuda', dtype=torch.bool), diagonal=1)
                    s.masked_fill_(mask, float('-inf'))
                attn = torch.softmax(s, dim=-1)
                out = torch.matmul(attn, v)
        
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        time_seconds = (end - start) / repeats
        mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
        
        # Calculate TFLOPS
        flop = flops(bs, sl, hd, nh, causal, mode="fwd")
        tflops = efficiency(flop, time_seconds)
        
        del q, k, v, s, attn, out
        torch.cuda.empty_cache()
        
        return {
            "status": "ok",
            "time_ms": round(time_seconds * 1000, 2),
            "memory_gb": round(mem_gb, 4),
            "tflops": round(tflops, 2)
        }
    except:
        torch.cuda.empty_cache()
        return {"status": "OOM", "time_ms": None, "memory_gb": None, "tflops": 0.0}

def measure_flash(bs, sl, nh, hd, causal=True, warmup=5, repeats=30):
    """Measure FlashAttention v2 - FORWARD PASS ONLY"""
    if not HAS_FLASH:
        return {"status": "N/A", "time_ms": None, "memory_gb": None, "tflops": 0.0}
    
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    try:
        qkv = torch.randn(bs, sl, 3, nh, hd, device='cuda', dtype=torch.float16)
        
        # Warmup
        for _ in range(warmup):
            with torch.no_grad():
                out = flash_attn_qkvpacked_func(qkv, dropout_p=0.0, causal=causal)
            torch.cuda.synchronize()
        
        # Measure
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        for _ in range(repeats):
            with torch.no_grad():
                out = flash_attn_qkvpacked_func(qkv, dropout_p=0.0, causal=causal)
        
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        time_seconds = (end - start) / repeats
        mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
        
        # Calculate TFLOPS
        flop = flops(bs, sl, hd, nh, causal, mode="fwd")
        tflops = efficiency(flop, time_seconds)
        
        del qkv, out
        torch.cuda.empty_cache()
        
        return {
            "status": "ok",
            "time_ms": round(time_seconds * 1000, 2),
            "memory_gb": round(mem_gb, 4),
            "tflops": round(tflops, 2)
        }
    except:
        torch.cuda.empty_cache()
        return {"status": "OOM", "time_ms": None, "memory_gb": None, "tflops": 0.0}

def measure_waller(sl, nh, hd, causal=True):
    """Measure Waller Operator (Triangle Engine) - PARSE ACTUAL TIMING FROM OUTPUT"""
    if not os.path.exists(WALLER_BINARY):
        return {"status": "missing", "time_ms": None, "memory_gb": None, "tflops": 0.0}
    
    try:
        result = subprocess.run(
            [WALLER_BINARY, str(sl), str(nh), str(hd)],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            output = result.stdout + result.stderr
            
            # Parse the ACTUAL kernel timing from binary output
            # Format: "APA V1 512k x100 iters: 14.291 ms avg (492413.6 TFLOPS), memory O(N log N)"
            time_match = re.search(r'(\d+\.?\d*)\s*ms\s+avg', output)
            
            if time_match:
                time_ms = float(time_match.group(1))
                
                # Memory: Binary only reports O(N log N), not actual usage
                # We'll report "N/A" since we can't measure it externally for a binary
                mem_gb = None
                
                # Calculate TFLOPS using our methodology (bs=1 for Waller)
                flop = flops(1, sl, hd, nh, causal, mode="fwd")
                tflops = efficiency(flop, time_ms / 1000)  # Convert ms to seconds
                
                return {
                    "status": "ok",
                    "time_ms": round(time_ms, 2),
                    "memory_gb": "O(N log N)",  # Report complexity, not actual GB
                    "tflops": round(tflops, 2)
                }
            else:
                print(f"Could not parse timing from Waller output: {output}")
                return {"status": "parse_error", "time_ms": None, "memory_gb": None, "tflops": 0.0}
        
        return {"status": "error", "time_ms": None, "memory_gb": None, "tflops": 0.0}
    except Exception as e:
        print(f"Waller error: {e}")
        return {"status": "error", "time_ms": None, "memory_gb": None, "tflops": 0.0}

def run():
    print("=" * 110)
    print("COMPREHENSIVE ATTENTION BENCHMARK - Following FlashAttention v2 Methodology (FIXED)")
    print("=" * 110)
    
    gpu = get_gpu_info()
    print(f"Hardware: {gpu['name']}")
    print(f"GPU Memory: {gpu['total_memory_gb']:.1f} GB")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    # Test configurations
    configs = [
        # FlashAttention's official range
        (32, 512, 64, True),
        (16, 1024, 64, True),
        (8, 2048, 64, True),
        (4, 4096, 64, True),
        (2, 8192, 64, True),
        (1, 16384, 64, True),
        # Extended range - WALLER TARGET
        (1, 65536, 64, True),
        (1, 131072, 64, True),
        (1, 262144, 64, True),
        (1, 524288, 64, True),
    ]
    
    results = []
    
    print("-" * 110)
    print(f"{'Config':^30} | {'PyTorch':^25} | {'Flash2':^25} | {'Waller':^25}")
    print(f"{'(bs, sl, hd, causal)':^30} | {'(ms / GB / TFLOPS)':^25} | {'(ms / GB / TFLOPS)':^25} | {'(ms / GB / TFLOPS)':^25}")
    print("-" * 110)
    
    for bs, sl, hd, causal in configs:
        config_str = f"bs={bs}, sl={sl:>6}, hd={hd}"
        print(f"{config_str:^30} | ", end="", flush=True)
        
        dim = 2048
        nh = dim // hd
        
        # PyTorch (skip for sl > 16384)
        if sl <= 16384:
            py = measure_pytorch(bs, sl, nh, hd, causal=causal)
        else:
            py = {"status": "skip", "time_ms": None, "memory_gb": None, "tflops": 0.0}
        
        if py['status'] == 'ok':
            py_str = f"{py['time_ms']:>6.1f}ms / {py['memory_gb']:>5.2f}GB / {py['tflops']:>5.0f}T"
        else:
            py_str = f"{py['status'].upper():^25}"
        print(f"{py_str:^25} | ", end="", flush=True)
        
        # FlashAttention v2
        fl = measure_flash(bs, sl, nh, hd, causal=causal)
        if fl['status'] == 'ok':
            fl_str = f"{fl['time_ms']:>6.1f}ms / {fl['memory_gb']:>5.2f}GB / {fl['tflops']:>5.0f}T"
        else:
            fl_str = f"{fl['status'].upper():^25}"
        print(f"{fl_str:^25} | ", end="", flush=True)
        
        # Waller - only for extended range
        if sl >= 65536:
            wa = measure_waller(sl, 1, hd, causal=causal)
            if wa['status'] == 'ok':
                wa_str = f"{wa['time_ms']:>6.1f}ms / O(N log N) / {wa['tflops']:>5.0f}T"
            else:
                wa_str = f"{wa['status'].upper():^25}"
        else:
            wa = {"status": "N/A", "time_ms": None, "memory_gb": None, "tflops": 0.0}
            wa_str = "N/A (< 65k range)"
        print(f"{wa_str:^25}")
        
        results.append({
            "batch_size": bs,
            "seq_len": sl,
            "head_dim": hd,
            "num_heads": nh if sl <= 16384 else 1,
            "causal": causal,
            "pytorch": py,
            "flash2": fl,
            "waller": wa
        })
    
    print("-" * 110)
    print("\nWaller Operator (Triangle Engine - O(N log N) pyramid attention)")
    print("Patent pending | e@ewaller.com | https://luxiedge.com")
    print()
    
    output_file = 'benchmark_waller_fixed_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "gpu": gpu,
            "methodology": "FlashAttention v2 official (extended to 524k, FIXED Waller timing)",
            "results": results
        }, f, indent=2)
    
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    run()
