# Install the newest triton version with
# pip install "git+https://github.com/openai/triton.git#egg=triton&subdirectory=python"
import pickle
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

from flash_attn.utils.benchmark import benchmark_all, benchmark_forward, benchmark_backward
from flash_attn.utils.benchmark import benchmark_fwd_bwd, benchmark_combined

from flash_attn import flash_attn_qkvpacked_func
from flash_attn_interface import flash_attn_func, _flash_attn_forward

def flops(batch, seqlen, headdim, nheads, causal, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)

def efficiency(flop, time):
    return (flop / time / 10**12) if not math.isnan(time) else 0.0

def time_fwd(func, *args, **kwargs):
    time.sleep(1) # Sleep to avoid residual power throttling from the previous benchmark
    time_f = benchmark_forward(func, *args, **kwargs)
    return time_f[1].mean

torch.manual_seed(0)

repeats = 20
device = 'cuda'
dtype = torch.float8_e4m3fn

# bs_seqlen_vals = [(16, 1024), (8, 2048), (4, 4224), (2, 8448), (1, 8448 * 2)]
# bs_seqlen_vals = [(16, 1024), (8, 2048), (4, 4096), (2, 8192), (1, 8192 * 2)]
bs_seqlen_vals = [(4, 4096), (2, 8192)]
# bs_seqlen_vals = [(32, 512), (16, 1024), (8, 2048)]
# bs_seqlen_vals = [(4, 4224), (2, 8448), (1, 8448 * 2)]
# bs_seqlen_vals = [(4, 4096), (2, 8192)]
causal_vals = [False, True]
# headdim_vals = [64, 128, 256]
headdim_vals = [128, 256]
dim = 2048
# dim = 256
dropout_p = 0.0

methods = ([]
        + ["Flash3 BF16"]
        + ["Flash3 FP8"]
        + (["Flash3 FP8 UpcastV"])
           )

time_f = {}
time_b = {}
time_f_b = {}
speed_f = {}
speed_b = {}
speed_f_b = {}
for causal in causal_vals:
    for headdim in headdim_vals:
        for batch_size, seqlen in bs_seqlen_vals:
            torch.cuda.empty_cache()
            config = (causal, headdim, batch_size, seqlen)
            nheads = dim // headdim
            nheads_kv = nheads
            q = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=torch.bfloat16, requires_grad=False)
            k = torch.randn(batch_size, seqlen, nheads_kv, headdim, device=device, dtype=torch.bfloat16, requires_grad=False)
            v = torch.randn(batch_size, seqlen, nheads_kv, headdim, device=device, dtype=torch.bfloat16, requires_grad=False)

            time.sleep(1)

            f = time_fwd(flash_attn_func, q, k, v, causal=causal, repeats=repeats, verbose=False)
            time_f[config, "Flash3 BF16"] = f

            time.sleep(1)

            q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
            
            f = time_fwd(flash_attn_func, q, k, v, causal=causal, repeats=repeats, verbose=False)
            time_f[config, "Flash3 FP8"] = f

            time.sleep(1)

            f = time_fwd(flash_attn_func, q, k, v, causal=causal, upcast_V=True, repeats=repeats, verbose=False)
            time_f[config, "Flash3 FP8 UpcastV"] = f

            print(f"### causal={causal}, headdim={headdim}, batch_size={batch_size}, seqlen={seqlen} ###")
            for method in methods:
                speed_f[config, method] = efficiency(
                    flops(batch_size, seqlen, headdim, nheads, causal, mode="fwd"),
                    time_f[config, method]
                )
                #print (time_f[config,method])
                print(
                    f"{method} fwd: {speed_f[config, method]:.2f} TFLOPs/s, {time_f[config, method] * 1e3} ms, "
                )


# with open('flash3_attn_time.plk', 'wb') as fp:
#     pickle.dump((time_f, time_b, time_f_b), fp, protocol=pickle.HIGHEST_PROTOCOL)
