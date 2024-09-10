import torch
from tunner import FlashFwdTunner
from arch import A100
from code_emitter import ShapeConfig

batch_size = 4
seqlen = 2048
nheads = 8
headdim = 192
v_headdim = 128
device = 'cuda'
dtype = torch.bfloat16
q = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype,
                              requires_grad=True)
k = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype,
                                requires_grad=True)
v = torch.randn(batch_size, seqlen, nheads, v_headdim, device=device, dtype=dtype,
                                requires_grad=True)

tunner = FlashFwdTunner(A100(), [q,k,v], ShapeConfig(headdim,v_headdim), "autotuner/temp")    
tunner.tune()
