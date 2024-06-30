import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
import torch.utils.benchmark as benchmark
import pdb
import time
from flash_attn import (
    flash_attn_func,
    flash_attn_kvpacked_func,
    flash_attn_qkvpacked_func,
    flash_attn_varlen_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_varlen_qkvpacked_func,
    flash_attn_with_kvcache,
)

# Function to measure time and memory
def benchmark_torch_function_in_microseconds(f, *args, **kwargs):
    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
    )
    return t0.blocked_autorange().mean * 1e6

def measure_time_and_memory(func, *args, **kwargs):
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    output = func(*args)
    torch.cuda.synchronize()
    end_time = time.time()
    max_memory = torch.cuda.max_memory_allocated()
    return end_time - start_time, max_memory, output


batch_size = 32
seq_len = 128
head_size = 64

# Optionally use the context manager to ensure one of the fused kernels is run
q = torch.rand(32, 8, 128, 64, dtype=torch.float16, device="cuda")
k = torch.rand(32, 8, 128, 64, dtype=torch.float16, device="cuda")
v = torch.rand(32, 8, 128, 64, dtype=torch.float16, device="cuda")

# test for pytorch version of flash-attention
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    torch_time, torch_mem, torch_output = \
        measure_time_and_memory(F.scaled_dot_product_attention, q, k, v)

print(f"pytorch implementaion of flashAttention \n {torch_output.shape}")
print(f"pytorch implementaion time is \n {torch_time}, memory use is {torch_mem}")



# test for tri version of flash-attention
tri_time, tri_mem, tri_output = \
        measure_time_and_memory(flash_attn_func, q, k, v, deterministic=True, return_attn_probs=True,)

print(f"Tri version of flashAttention: \n {tri_out.shape}")
print(f"Tri implementaion time is \n {tri_time}, memory use is {tri_mem}")


print(f"diff of output {torch_output - tri_output}")


