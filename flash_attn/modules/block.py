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

try:
    from flash_attn.ops.layer_norm import dropout_add_layer_norm_parallel_residual
except ImportError:
    dropout_add_layer_norm_parallel_residual = None

try:
    from flash_attn.ops.rms_norm import RMSNorm, dropout_add_rms_norm
except ImportError:
    RMSNorm, dropout_add_rms_norm = None, None

try:
    from flash_attn.ops.rms_norm import dropout_add_rms_norm_parallel_residual
except ImportError:
    dropout_add_rms_norm_parallel_residual = None


class Block(nn.Module):

    def __init__(self, dim, mixer_cls=None, mlp_cls=None, norm_cls=nn.LayerNorm,
                 dropout_cls=nn.Dropout, prenorm=True, resid_dropout1=0., resid_dropout2=0.,
                 drop_path1=0., drop_path2=0., fused_dropout_add_ln=False, return_residual=False,
                 residual_in_fp32=False, sequence_parallel=False, mark_shared_params=False):
        """
        For prenorm=True, this Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA -> Dropout -> Add -> LN -> MLP -> Dropout -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Dropout -> Add -> LN -> MHA -> Dropout -> Add -> LN -> MLP, returning both
        the hidden_states (output of the MLP) and the residual.
        This is for performance reasons, as we can fuse the dropout, add and LayerNorm.
        The residual needs to be provided (except for the very first block).

        For prenorm=False, this Block has the same structure as a regular postnorm Transformer
        block: MHA -> Dropout -> Add -> LN -> MLP -> Dropout -> Add -> LN.

        return_residual: whether each of the sub-layers (mixer and mlp) will return the residual.
        This is for performance reason: for post-norm architecture, returning the input allows us
        to fuse the backward of nn.Linear with the residual connection.
        """
        super().__init__()
        self.prenorm = prenorm
        self.fused_dropout_add_ln = fused_dropout_add_ln
        self.return_residual = return_residual
        self.residual_in_fp32 = residual_in_fp32
        if self.residual_in_fp32:
            assert self.prenorm, 'residual_in_fp32 is only compatible with prenorm=True'
        if mixer_cls is None:
            mixer_cls = partial(MHA, num_heads=dim // 64)
        if mlp_cls is None:
            mlp_cls = partial(Mlp, hidden_features=4 * dim)
        self.mixer = mixer_cls(dim)
        self.dropout1 = dropout_cls(resid_dropout1)
        self.drop_path1 = StochasticDepth(drop_path1, mode='row')
        self.norm1 = norm_cls(dim)
        self.mlp = mlp_cls(dim)
        if not isinstance(self.mlp, nn.Identity):
            self.dropout2 = dropout_cls(resid_dropout2)
            self.drop_path2 = StochasticDepth(drop_path2, mode='row')
            self.norm2 = norm_cls(dim)

        if self.fused_dropout_add_ln:
            assert dropout_add_layer_norm is not None, 'dropout_layer_norm is not installed'
            assert dropout_add_rms_norm is not None, 'dropout_layer_norm is not installed'
            assert (isinstance(self.norm1, (nn.LayerNorm, RMSNorm))
                    and isinstance(self.dropout1, nn.Dropout))

        # TD [2023-01-07]: TODO: During training, if sequence_parallel is False and dropout != 0.0,
        # then the input to each worker in the tensor parallel group will be different.
        # This would produce wrong outputs? Somehow we'd need to sync the RNG state across workers.
        # For now this is not an issue because we always use sequence_parallel=True during training
        # and only use sequence_parallel=False during inference.

        # Mark the norm parameters as "sequence_parallel" so that we run all-reduce on their grads.
        if sequence_parallel:
            for p in self.norm1.parameters():
                p._sequence_parallel = True
            if hasattr(self, 'norm2'):
                for p in self.norm2.parameters():
                    p._sequence_parallel = True
        # Mark the norm parameters as "shared_params" so that we sync their values at init.
        if mark_shared_params:
            for p in self.norm1.parameters():
                p._shared_params = True
            if hasattr(self, 'norm2'):
                for p in self.norm2.parameters():
                    p._shared_params = True

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def forward(self, hidden_states: Tensor, residual: Optional[Tensor] = None,
                mixer_subset=None, mixer_kwargs=None):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: if postnorm, residual=None, If prenorm, hidden_states = Attn/MLP(LN(residual))
            mixer_subset: for cross-attention only. If not None, will take a subset of x
                before applying the query projection. Useful for e.g., ViT where we only care
                about the CLS token in the last layer.
        """
        fused_add_norm_fn = (dropout_add_rms_norm if RMSNorm and isinstance(self.norm1, RMSNorm)
                             else dropout_add_layer_norm)
        if self.prenorm:
            if not self.fused_dropout_add_ln:
                dropped = self.drop_path1(self.dropout1(hidden_states))
                residual = (dropped + residual) if residual is not None else dropped
                hidden_states = self.norm1(residual.to(dtype=self.norm1.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)
            else:
                if self.drop_path1.p == 0 or not self.training:
                    rowscale1 = None
                else:
                    rowscale1 = self.drop_path1(torch.ones(
                        hidden_states.shape[:-1], device=hidden_states.device,
                        dtype=hidden_states.dtype)
                    )
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states, residual, self.norm1.weight, self.norm1.bias,
                    self.dropout1.p if self.training else 0.0, self.norm1.eps,
                    rowscale=rowscale1, prenorm=True, residual_in_fp32=self.residual_in_fp32
                )
            if mixer_kwargs is None:
                mixer_kwargs = {}
            if mixer_subset is not None:
                mixer_kwargs['mixer_subset'] = mixer_subset
            hidden_states = self.mixer(hidden_states, **mixer_kwargs)
            if mixer_subset is not None:
                residual = residual[:, mixer_subset]
            if not isinstance(self.mlp, nn.Identity):
                if not self.fused_dropout_add_ln:
                    dropped = self.drop_path2(self.dropout2(hidden_states))
                    residual = (dropped + residual) if residual is not None else dropped
                    hidden_states = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
                    if self.residual_in_fp32:
                        residual = residual.to(torch.float32)
                else:
                    if self.drop_path2.p == 0 or not self.training:
                        rowscale2 = None
                    else:
                        rowscale2 = self.drop_path2(torch.ones(
                            hidden_states.shape[:-1], device=hidden_states.device,
                            dtype=hidden_states.dtype)
                        )
                    hidden_states, residual = fused_add_norm_fn(
                        hidden_states, residual, self.norm2.weight, self.norm2.bias,
                        self.dropout2.p if self.training else 0.0, self.norm2.eps,
                        rowscale=rowscale2, prenorm=True, residual_in_fp32=self.residual_in_fp32
                    )
                hidden_states = self.mlp(hidden_states)
            return hidden_states, residual
        else:
            assert residual is None
            mixer_out = self.mixer(
                hidden_states, **(mixer_kwargs if mixer_kwargs is not None else {})
            )
            if self.return_residual:  # mixer out is actually a pair here
                mixer_out, hidden_states = mixer_out
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
                hidden_states = fused_add_norm_fn(
                    mixer_out, hidden_states, self.norm1.weight, self.norm1.bias,
                    self.dropout1.p if self.training else 0.0, self.norm1.eps,
                    rowscale=rowscale1, prenorm=False
                )
            if not isinstance(self.mlp, nn.Identity):
                mlp_out = self.mlp(hidden_states)
                if self.return_residual:  # mlp out is actually a pair here
                    mlp_out, hidden_states = mlp_out
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
                    hidden_states = fused_add_norm_fn(
                        mlp_out, hidden_states, self.norm2.weight, self.norm2.bias,
                        self.dropout2.p if self.training else 0.0, self.norm2.eps,
                        rowscale=rowscale2, prenorm=False
                    )
            return hidden_states


class ParallelBlock(nn.Module):
    """The attention (mixer) and MLP blocks are done in parallel, similar to GPT-J, GPT-NeoX,
    and PaLM.
    """

    def __init__(self, dim, mixer_cls=None, mlp_cls=None, norm_cls=nn.LayerNorm,
                 dropout_cls=nn.Dropout, resid_dropout1=0., resid_dropout2=0.,
                 tied_norm=False, fused_dropout_add_ln=False, residual_in_fp32=False,
                 sequence_parallel=False, mark_shared_params=False):
        """
        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA / MLP -> Dropout -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Dropout -> Add -> LN -> MHA / MLP, returning both
        the hidden_states (output1 of the MHA / MLP) and the residual.
        This is for performance reasons, as we can fuse the dropout, add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.tied_norm = tied_norm
        self.fused_dropout_add_ln = fused_dropout_add_ln
        self.residual_in_fp32 = residual_in_fp32
        if mixer_cls is None:
            mixer_cls = partial(MHA, num_heads=dim // 64)
        if mlp_cls is None:
            mlp_cls = partial(Mlp, hidden_features=4 * dim)
        self.mixer = mixer_cls(dim)
        self.dropout1 = dropout_cls(resid_dropout1)
        self.norm1 = norm_cls(dim)
        self.mlp = mlp_cls(dim)
        self.dropout2 = dropout_cls(resid_dropout2)
        if not self.tied_norm:
            self.norm2 = norm_cls(dim)

        if self.fused_dropout_add_ln:
            assert dropout_add_layer_norm_parallel_residual is not None, 'dropout_layer_norm is not installed'
            assert dropout_add_rms_norm_parallel_residual is not None, 'dropout_layer_norm is not installed'
            assert (isinstance(self.norm1, (nn.LayerNorm, RMSNorm))
                    and isinstance(self.dropout1, nn.Dropout))

        # TD [2023-01-07]: TODO: During training, if sequence_parallel is False and dropout != 0.0,
        # then the input to each worker in the tensor parallel group will be different.
        # This would produce wrong outputs? Somehow we'd need to sync the RNG state across workers.
        # For now this is not an issue because we always use sequence_parallel=True during training
        # and only use sequence_parallel=False during inference.

        # Mark the norm parameters as "sequence_parallel" so that we run all-reduce on their grads.
        if sequence_parallel:
            for p in self.norm1.parameters():
                p._sequence_parallel = True
            if hasattr(self, 'norm2'):
                for p in self.norm2.parameters():
                    p._sequence_parallel = True
        # Mark the norm parameters as "shared_params" so that we sync their values at init.
        if mark_shared_params:
            for p in self.norm1.parameters():
                p._shared_params = True
            if hasattr(self, 'norm2'):
                for p in self.norm2.parameters():
                    p._shared_params = True

    def forward(self, hidden_states1: Tensor, hidden_states2: Optional[Tensor] = None,
                residual: Optional[Tensor] = None, mixer_kwargs=None):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states1: the output of the previous attention (mixer) or embedding layer.
            hidden_states2: the output of the previous MLP layer (if None, will use hidden_states1).
            residual.
        """
        fused_add_norm_fn = (dropout_add_rms_norm_parallel_residual
                             if isinstance(self.norm1, RMSNorm)
                             else dropout_add_layer_norm_parallel_residual)
        if not self.fused_dropout_add_ln:
            dropped1 = self.dropout1(hidden_states1)
            # For the very 1st block, we only want 1 dropout, not two different dropouts
            if hidden_states2 is not None:
                dropped2 = self.dropout2(hidden_states2)
                residual = ((residual + dropped1 + dropped2)
                            if residual is not None else dropped1 + dropped2)
            else:
                residual = (residual + dropped1) if residual is not None else dropped1
            hidden_states1 = self.norm1(residual.to(dtype=self.norm1.weight.dtype))
            hidden_states2 = (self.norm2(residual.to(dtype=self.norm2.weight.dtype))
                              if not self.tied_norm else hidden_states1)
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            weight2, bias2 = ((self.norm2.weight, self.norm2.bias)
                              if not self.tied_norm else (None, None))
            hidden_states1, hidden_states2, residual = fused_add_norm_fn(
                hidden_states1, hidden_states2, residual, self.norm1.weight, self.norm1.bias,
                weight2, bias2, self.dropout1.p if self.training else 0.0, self.norm1.eps,
                prenorm=True, residual_in_fp32=self.residual_in_fp32
            )
            if self.tied_norm:
                hidden_states2 = hidden_states1
        if mixer_kwargs is None:
            mixer_kwargs = {}
        hidden_states1 = self.mixer(hidden_states1, **mixer_kwargs)
        hidden_states2 = self.mlp(hidden_states2)
        return hidden_states1, hidden_states2, residual
