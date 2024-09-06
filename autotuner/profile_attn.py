import torch
from flash_attn.utils.benchmark import benchmark_forward

# batch_size = 4
# seqlen = 2048
# nheads = 8
# headdim = QKHeadDim
# v_headdim = VHeadDim
# device = 'cuda'
# dtype = torch.bfloat16 if is_bf16 else torch.float16

# dropout_p = 0.0
# causal = is_causal
# repeats = 30


def profile_fwd(fn,headdim, v_headdim, batch_size=4, seqlen=2048, nheads=8, device='cuda', is_bf16=False, causal=False, dropout_p=0.0, repeats=30):
    dtype = torch.bfloat16 if is_bf16 else torch.float16
    q = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype,
                              requires_grad=True)
    k = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype,
                                requires_grad=True)
    v = torch.randn(batch_size, seqlen, nheads, v_headdim, device=device, dtype=dtype,
                                requires_grad=True)
    f = benchmark_forward(fn, q, k, v, dropout_p, causal=causal, repeats=repeats, verbose=False)
    time_f = f[1].mean
    # print(time_f)
    return time_f