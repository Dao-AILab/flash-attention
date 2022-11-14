# Copyright (c) 2022, Tri Dao.

from typing import Optional
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torchvision.ops import StochasticDepth

from flash_attn.modules.mha import MHA
from flash_attn.modules.mlp import Mlp

try:
    from flash_attn.ops.layer_norm import dropout_add_layer_norm
except ImportError:
    dropout_add_layer_norm = None


class Block(nn.Module):

    def __init__(self, dim, mixer_cls=None, mlp_cls=None, norm_cls=nn.LayerNorm,
                 dropout_cls=nn.Dropout, prenorm=True, resid_dropout=0., drop_path=0.,
                 fused_dropout_add_ln=False):
        super().__init__()
        self.prenorm = prenorm
        self.fused_dropout_add_ln = fused_dropout_add_ln
        if mixer_cls is None:
            mixer_cls = partial(MHA, num_heads=dim // 64)
        if mlp_cls is None:
            mlp_cls = partial(Mlp, hidden_features=4 * dim)
        self.mixer = mixer_cls(dim)
        self.dropout1 = dropout_cls(resid_dropout)
        self.drop_path1 = StochasticDepth(drop_path, mode='row')
        self.norm1 = norm_cls(dim)
        self.mlp = mlp_cls(dim)
        if not isinstance(self.mlp, nn.Identity):
            self.dropout2 = dropout_cls(resid_dropout)
            self.drop_path2 = StochasticDepth(drop_path, mode='row')
            self.norm2 = norm_cls(dim)

        if self.fused_dropout_add_ln:
            assert dropout_add_layer_norm is not None, 'dropout_add_ln is not installed'
            assert isinstance(self.norm1, nn.LayerNorm) and isinstance(self.dropout1, nn.Dropout)

    def forward(self, hidden_states: Tensor, residual: Optional[Tensor] = None,
                mixer_kwargs=None):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: if postnorm, residual=None, If prenorm, hidden_states = LayerNorm(residual)
        """
        if self.prenorm:
            assert residual is not None
            mixer_out = self.mixer(hidden_states,
                                   **(mixer_kwargs if mixer_kwargs is not None else {}))
            if not self.fused_dropout_add_ln:
                residual = self.drop_path1(self.dropout1(mixer_out)) + residual
                hidden_states = self.norm1(residual.to(dtype=self.norm1.weight.dtype))
            else:
                if self.drop_path1.p == 0 or not self.training:
                    rowscale1 = None
                else:
                    rowscale1 = self.drop_path1(torch.ones(
                        mixer_out.shape[:-1], device=mixer_out.device, dtype=mixer_out.dtype)
                    )
                hidden_states, residual = dropout_add_layer_norm(
                    mixer_out, residual, self.norm1.weight, self.norm1.bias,
                    self.dropout1.p if self.training else 0.0, self.norm1.eps,
                    rowscale=rowscale1, prenorm=True
                )
            if not isinstance(self.mlp, nn.Identity):
                mlp_out = self.mlp(hidden_states)
                if not self.fused_dropout_add_ln:
                    residual = self.drop_path2(self.dropout2(mlp_out)) + residual
                    hidden_states = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
                else:
                    if self.drop_path2.p == 0 or not self.training:
                        rowscale2 = None
                    else:
                        rowscale2 = self.drop_path2(torch.ones(
                            mlp_out.shape[:-1], device=mlp_out.device, dtype=mlp_out.dtype)
                        )
                    hidden_states, residual = dropout_add_layer_norm(
                        mlp_out, residual, self.norm2.weight, self.norm2.bias,
                        self.dropout2.p if self.training else 0.0, self.norm2.eps,
                        rowscale=rowscale2, prenorm=True
                    )
            return hidden_states, residual
        else:
            assert residual is None
            mixer_out = self.mixer(hidden_states,
                                   **(mixer_kwargs if mixer_kwargs is not None else {}))
            if not self.fused_dropout_add_ln:
                hidden_states = self.norm1((self.drop_path1(self.dropout1(mixer_out))
                                            + hidden_states).to(dtype=self.norm1.weight.dtype))
            else:
                if self.drop_path1.p == 0 or not self.training:
                    rowscale1 = None
                else:
                    rowscale1 = self.drop_path1(torch.ones(
                        mixer_out.shape[:-1], device=mixer_out.device, dtype=mixer_out.dtype)
                    )
                hidden_states = dropout_add_layer_norm(
                    mixer_out, hidden_states, self.norm1.weight, self.norm1.bias,
                    self.dropout1.p if self.training else 0.0, self.norm1.eps,
                    rowscale=rowscale1, prenorm=False
                )
            if not isinstance(self.mlp, nn.Identity):
                mlp_out = self.mlp(hidden_states)
                if not self.fused_dropout_add_ln:
                    hidden_states = self.norm2((self.drop_path2(self.dropout2(mlp_out))
                                                + hidden_states).to(dtype=self.norm2.weight.dtype))
                else:
                    if self.drop_path2.p == 0 or not self.training:
                        rowscale2 = None
                    else:
                        rowscale2 = self.drop_path2(torch.ones(
                            mlp_out.shape[:-1], device=mlp_out.device, dtype=mlp_out.dtype)
                        )
                    hidden_states = dropout_add_layer_norm(
                        mlp_out, hidden_states, self.norm2.weight, self.norm2.bias,
                        self.dropout2.p if self.training else 0.0, self.norm2.eps,
                        rowscale=rowscale2, prenorm=False
                    )
            return hidden_states
