import torch
from tunner import FlashFwdTunner
from arch import A100
from code_emitter import ShapeConfig,ProfileConfig

batch_size = 4
seqlen = 2048
nheads = 8
headdim = 128# 192
v_headdim = 256# 128
device = 'cuda:0'
dtype = torch.bfloat16
dropout_p = 0.0 # 0.0

q = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype,
                              requires_grad=True)
k = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype,
                                requires_grad=True)
v = torch.randn(batch_size, seqlen, nheads, v_headdim, device=device, dtype=dtype,
                                requires_grad=True)

tunner = FlashFwdTunner(A100(), [q,k,v], ShapeConfig(headdim,v_headdim), ProfileConfig(batch_size,seqlen,seqlen,nheads,nheads,nheads,device,dtype,dropout_p), "autotuner/temp128_256")  # "autotuner/temp192_128"
tunner.tune()
