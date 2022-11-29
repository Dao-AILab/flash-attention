from typing import Any

from pytorch_lightning import Callback, Trainer, LightningModule
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.parsing import AttributeDict


class ParamsLog(Callback):
    """Log the number of parameters of the model
    """
    def __init__(self, total_params_log: bool = True, trainable_params_log: bool = True,
                 non_trainable_params_log: bool = True):
        super().__init__()
        self._log_stats = AttributeDict(
            {
                'total_params_log': total_params_log,
                'trainable_params_log': trainable_params_log,
                'non_trainable_params_log': non_trainable_params_log,
            }
        )

    @rank_zero_only
    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        logs = {}
        if self._log_stats.total_params_log:
            logs["model/params_total"] = sum(p.numel() for p in pl_module.parameters())
        if self._log_stats.trainable_params_log:
            logs["model/params_trainable"] = sum(p.numel() for p in pl_module.parameters()
                                             if p.requires_grad)
        if self._log_stats.non_trainable_params_log:
            logs["model/params_not_trainable"] = sum(p.numel() for p in pl_module.parameters()
                                                     if not p.requires_grad)
        if trainer.logger is not None:
            trainer.logger.log_hyperparams(logs)
