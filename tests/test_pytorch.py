import torch
import torch.nn.functional as F
import time
from flash_attn.flash_attn_interface import (
    flash_attn_func,
    flash_attn_kvpacked_func,
    flash_attn_qkvpacked_func,
    flash_attn_varlen_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_varlen_qkvpacked_func,
    flash_attn_with_kvcache,
)


batch_size = 32
seq_len = 128
head_size = 64

q = torch.randn(batch_size, seq_len, head_size).cuda()
k = torch.randn(batch_size, seq_len, head_size).cuda()
v = torch.randn(batch_size, seq_len, head_size).cuda()

# Optionally use the context manager to ensure one of the fused kernels is run
query = torch.rand(32, 8, 128, 64, dtype=torch.float16, device="cuda")
key = torch.rand(32, 8, 128, 64, dtype=torch.float16, device="cuda")
value = torch.rand(32, 8, 128, 64, dtype=torch.float16, device="cuda")
with torch.backends.cuda.sdp_kernel(enable_math=False):
    output, weight_output = F.scaled_dot_product_attention(query,key,value)

print(output.shape)


# Function to measure time and memory
def measure_time_and_memory(func, *args):
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    output = func(*args)
    torch.cuda.synchronize()
    end_time = time.time()
    max_memory = torch.cuda.max_memory_allocated()
    return end_time - start_time, max_memory, output