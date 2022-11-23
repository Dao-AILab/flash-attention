# Copyright (c) 2022, Tri Dao.

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from flash_attn.ops.fused_dense import fused_dense_gelu_dense_function_td
    from flash_attn.ops.fused_dense import fused_dense_res_gelu_dense_function_td
except ImportError:
    fused_dense_gelu_dense_function_td = None
    fused_dense_res_gelu_dense_function_td = None


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, activation=F.gelu,
                 device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, **factory_kwargs)
        self.activation = activation
        self.fc2 = nn.Linear(hidden_features, out_features, **factory_kwargs)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class FusedDenseGeluDense(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, bias=True,
                 checkpoint_lvl=0, heuristic=0, return_residual=False, device=None, dtype=None):
        """
        checkpoint_lvl (increasing lvl means slower but more memory saving):
            0: no recomputation in the bwd
            1: recompute gelu_out in the bwd
            2: recompute gelu_in and gelu_out in the bwd
        heuristic:
            -1: don't fuse gemm + gelu (separate kernel)
            0..4: use this heuristic for the algo section in the fused gemm + gelu
            For CUDA >= 11.8, you'd want heuristic=0 for both fp16 and bf16 for best perf.
            For CUDA <= 11.7, you'd want heuristic=1 for fp16 and heuristic=-1 for bf16.
        return_residual: whether to return the input x along with the output. This is for
            performance reason: for post-norm architecture, returning the input allows us
            to fuse the backward of nn.Linear with the residual connection.
        """
        assert checkpoint_lvl in [0, 1, 2]
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        assert bias == True, "DenseGeluDense module without bias is currently not supported"
        assert (fused_dense_gelu_dense_function_td is not None
                and fused_dense_res_gelu_dense_function_td is not None), 'fused_dense_lib is not installed'
        self.checkpoint_lvl = checkpoint_lvl
        self.heuristic = heuristic
        self.return_residual = return_residual
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias, **factory_kwargs)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias, **factory_kwargs)

    def forward(self, x):
        assert x.is_cuda
        fn = (fused_dense_gelu_dense_function_td if not self.return_residual
              else fused_dense_res_gelu_dense_function_td)
        return fn(x, self.fc1.weight, self.fc1.bias, self.fc2.weight, self.fc2.bias,
                  self.checkpoint_lvl, self.heuristic)
