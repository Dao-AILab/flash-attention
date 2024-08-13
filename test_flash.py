
import torch
from flash_attn import flash_attn_func
batch = 4
seqlen_q = 2048
seqlen_kv = 2048
dim_qk = 64
dim_v = 128
nheads_q = 20
nheads_kv = 5
device = torch.device('cuda')
dtype = torch.float16

query = torch.randn(batch, seqlen_q, nheads_q, dim_qk, device=device, dtype=dtype)
key = torch.randn(batch, seqlen_kv, nheads_kv, dim_qk, device=device, dtype=dtype)
value = torch.randn(batch, seqlen_kv, nheads_kv, dim_v, device=device, dtype=dtype)

output = flash_attn_func(query, key, value, causal=False)
print(output[0,0,0,0])