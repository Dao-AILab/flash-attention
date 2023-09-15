# Adapted from https://github.com/Lightning-AI/lightning/blob/master/src/pytorch_lightning/callbacks/lr_monitor.py.
from typing import Any

from pytorch_lightning import Callback, Trainer
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.strategies import DeepSpeedStrategy


class LossScaleMonitor(Callback):
    """Monitor the loss scale for AMP (fp16).
    """

    # Use on_before_optimizer_step instead of on_train_batch_start since there might be
    # gradient accumulation and we only care about the loss scale when it could change (i.e.,
    # optimizer.step).
    @rank_zero_only
    def on_before_optimizer_step(self, trainer: Trainer, *args: Any, **kwargs: Any) -> None:
        if not trainer._logger_connector.should_update_logs:
            return
        stats = {}
        if isinstance(trainer.strategy, DeepSpeedStrategy):
            stats = {'scalar/scale': trainer.model.optimizer.loss_scale}
        if hasattr(trainer, 'precision_plugin') and hasattr(trainer.precision_plugin, 'scaler'):
            scaler = trainer.precision_plugin.scaler
            if scaler is not None:
                stats = {
                    'scaler/scale': scaler.get_scale(),
                    'scaler/growth_tracker': scaler._get_growth_tracker(),
                }
        if stats and trainer.loggers is not None:
            for logger in trainer.loggers:
                logger.log_metrics(stats, step=trainer.fit_loop.epoch_loop._batches_that_stepped)
