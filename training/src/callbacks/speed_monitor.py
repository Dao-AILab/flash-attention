# Adapted from https://pytorch-lightning.readthedocs.io/en/latest/_modules/pytorch_lightning/callbacks/gpu_stats_monitor.html#GPUStatsMonitor
# We only need the speed monitoring, not the GPU monitoring
import time
from typing import Any

from pytorch_lightning import Callback, Trainer
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.parsing import AttributeDict
from pytorch_lightning.utilities.types import STEP_OUTPUT


class SpeedMonitor(Callback):
    """Monitor the speed of each step and each epoch.
    """
    def __init__(self, intra_step_time: bool = True, inter_step_time: bool = True,
                 epoch_time: bool = True, verbose=False):
        super().__init__()
        self._log_stats = AttributeDict(
            {
                'intra_step_time': intra_step_time,
                'inter_step_time': inter_step_time,
                'epoch_time': epoch_time,
            }
        )
        self.verbose = verbose

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._snap_epoch_time = None

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._snap_intra_step_time = None
        self._snap_inter_step_time = None
        self._snap_epoch_time = time.time()

    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._snap_inter_step_time = None

    def on_test_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._snap_inter_step_time = None

    @rank_zero_only
    def on_train_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self._log_stats.intra_step_time:
            self._snap_intra_step_time = time.time()

        if not trainer._logger_connector.should_update_logs:
            return

        logs = {}
        if self._log_stats.inter_step_time and self._snap_inter_step_time:
            # First log at beginning of second step
            logs["time/inter_step (ms)"] = (time.time() - self._snap_inter_step_time) * 1000

        if trainer.logger is not None:
            trainer.logger.log_metrics(logs, step=trainer.global_step)

    @rank_zero_only
    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self._log_stats.inter_step_time:
            self._snap_inter_step_time = time.time()

        if self.verbose and self._log_stats.intra_step_time and self._snap_intra_step_time:
            pl_module.print(f"time/intra_step (ms): {(time.time() - self._snap_intra_step_time) * 1000}")

        if not trainer._logger_connector.should_update_logs:
            return

        logs = {}
        if self._log_stats.intra_step_time and self._snap_intra_step_time:
            logs["time/intra_step (ms)"] = (time.time() - self._snap_intra_step_time) * 1000

        if trainer.logger is not None:
            trainer.logger.log_metrics(logs, step=trainer.global_step)

    @rank_zero_only
    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule",) -> None:
        logs = {}
        if self._log_stats.epoch_time and self._snap_epoch_time:
            logs["time/epoch (s)"] = time.time() - self._snap_epoch_time
        if trainer.logger is not None:
            trainer.logger.log_metrics(logs, step=trainer.global_step)

