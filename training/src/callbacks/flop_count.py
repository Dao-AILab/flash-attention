# Adapted from https://github.com/rwightman/pytorch-image-models/blob/master/benchmark.py
from typing import Any, List, Sequence

import torch

from pytorch_lightning import Callback, Trainer, LightningModule
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.parsing import AttributeDict

from src.utils.flops import has_deepspeed_profiling, has_fvcore_profiling
from src.utils.flops import profile_deepspeed, profile_fvcore


class FlopCount(Callback):
    """Counter the number of FLOPs used by the model
    """
    def __init__(self, profilers: List[str] = ['fvcore', 'deepspeed'],
                 input_size: tuple = (3, 224, 224), input_dtype=torch.float32, device=None):
        if not isinstance(profilers, Sequence):
            profilers = [profilers]
        if any(p not in ['fvcore', 'deepspeed'] for p in profilers):
            raise NotImplementedError('Only support fvcore and deepspeed profilers')
        if 'fvcore' in profilers and not has_fvcore_profiling:
            raise ImportError('fvcore is not installed. Install it by running `pip install fvcore`')
        elif 'deepspeed' in profilers and not has_deepspeed_profiling:
            raise ImportError('deepspeed is not installed')
        super().__init__()
        self.profilers = profilers
        self.input_size = tuple(input_size)
        self.input_dtype = input_dtype
        self.device = device

    @rank_zero_only
    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if 'fvcore' in self.profilers:
            _, macs, _, acts = profile_fvcore(pl_module.to(self.device), input_size=self.input_size,
                                              input_dtype=self.input_dtype, detailed=True)
            trainer.logger.log_hyperparams({'GMACs': macs * 1e-9, 'MActs': acts * 1e-6})
        if 'deepspeed' in self.profilers:
            macs, _= profile_deepspeed(pl_module.to(self.device), input_size=self.input_size,
                                       input_dtype=self.input_dtype, detailed=True)
            if 'fvcore' not in self.profilers:  # fvcore's MACs seem more accurate
                trainer.logger.log_hyperparams({'GMACs': macs * 1e-9})
