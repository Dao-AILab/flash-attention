# Copyright (c) 2022, Tri Dao.

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from flash_attn.ops.fused_dense import FusedDenseGeluDense, ParallelFusedDenseGeluDense
except ImportError:
    FusedDenseGeluDense, ParallelFusedDenseGeluDense = None, None


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, activation=F.gelu,
                 return_residual=False, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.return_residual = return_residual
        self.fc1 = nn.Linear(in_features, hidden_features, **factory_kwargs)
        self.activation = activation
        self.fc2 = nn.Linear(hidden_features, out_features, **factory_kwargs)

    def forward(self, x):
        y = self.fc1(x)
        y = self.activation(y)
        y = self.fc2(y)
        return y if not self.return_residual else (y, x)
