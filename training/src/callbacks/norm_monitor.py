# Inspired by https://github.com/Lightning-AI/lightning/blob/master/src/pytorch_lightning/utilities/grads.py
# However, they compute grad at every iteration (I think), and the .item() calls incur a lot of overhead
# (6-7% slow down on GPT-2 small). Instead we only compute for iterations where we need to log, and don't
# call .item() explicitly.

from typing import Any
from collections import OrderedDict

from pytorch_lightning import Callback, Trainer
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.strategies import DeepSpeedStrategy

import torch
import torch.nn as nn

try:
    from apex.contrib.layer_norm import FastLayerNorm
except ImportError:
    FastLayerNorm = None


class NormMonitor(Callback):
    """Monitor the scales of weights and gradients.
    """

    def __init__(self, layer_norm_only: bool = False):
        super().__init__()
        self.layer_norm_only = layer_norm_only

    # Use on_before_optimizer_step instead of on_train_batch_start since there might be
    # gradient accumulation and we only care about  scale when it could change (i.e., optimizer.step).
    @rank_zero_only
    def on_before_optimizer_step(self, trainer: Trainer, pl_module, *args: Any, **kwargs: Any) -> None:
        if not trainer._logger_connector.should_update_logs:
            return
        model = pl_module.model
        named_parameters = {}
        if self.layer_norm_only:
            ln_modules = (nn.LayerNorm, nn.Embedding)
            if FastLayerNorm is not None:
                ln_modules += (FastLayerNorm,)
            for mn, m in model.named_modules():
                if isinstance(m, ln_modules):
                    for pn, p in m.named_parameters():
                        fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                        named_parameters[fpn] = p
        else:
            named_parameters = dict(model.named_parameters())

        if isinstance(trainer.strategy, DeepSpeedStrategy):
            loss_scale = trainer.model.optimizer.loss_scale
        else:
            loss_scale = 1.0

        stats = {}
        param_l1_norm, grad_l1_norm = [], []
        for param_name, param in named_parameters.items():
            param_abs = param.abs()
            param_abs_mean = param_abs.mean(dtype=torch.float32)
            stats[f'stats/{param_name}_max'] = param_abs.max()
            stats[f'stats/{param_name}_mean'] = param_abs_mean
            param_l1_norm.append(param_abs_mean * param.numel())
            if param.grad is not None:
                # If using AMP, gradient is already unscaled by the AMP loss scaler at this point
                # https://github.com/Lightning-AI/lightning/pull/9606
                # However, if using DeepSpeed, we need to scale it ourselves
                param_grad_abs = param.grad.abs()
                param_grad_abs_mean = param_grad_abs.mean(dtype=torch.float32) / loss_scale
                stats[f'stats/{param_name}_grad_max'] = param_grad_abs.max() / loss_scale
                stats[f'stats/{param_name}_grad_mean'] = param_grad_abs_mean
                grad_l1_norm.append(param_grad_abs_mean * param.grad.numel())
        stats['total_param_l1_norm'] = torch.stack(param_l1_norm).sum()
        if grad_l1_norm:
            stats['total_grad_l1_norm'] = torch.stack(grad_l1_norm).sum()
        # Sort by params name
        stats = OrderedDict(sorted(stats.items()))
        if trainer.loggers is not None:
            for logger in trainer.loggers:
                logger.log_metrics(stats, step=trainer.fit_loop.epoch_loop._batches_that_stepped)
