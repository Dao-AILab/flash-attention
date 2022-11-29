# Adapted from https://github.com/Lightning-AI/lightning/blob/master/src/pytorch_lightning/callbacks/fault_tolerance.py
from typing import Any
from pathlib import Path

import pytorch_lightning as pl


class ModelCheckpointMine(pl.callbacks.model_checkpoint.ModelCheckpoint):

    def __init__(self, *args, fault_tolerant=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.fault_tolerant = fault_tolerant

    def on_exception(self, trainer: "pl.Trainer", *_: Any, **__: Any) -> None:
        if self.fault_tolerant:
            # overwrite if necessary
            trainer.save_checkpoint(str(Path(self.dirpath) / '.pl_auto_save.ckpt'))

    # def teardown(self, trainer: "pl.Trainer", *_: Any, **__: Any) -> None:
    #     if self.fault_tolerant:
    #         trainer.strategy.remove_checkpoint(str(Path(self.dirpath) / '.pl_auto_save.ckpt'))


# TD [2022-07-17] I was trying to make resuming from standard checkpoint fault-tolerant.
# However, when it resumes it's off by 1 iteration. My attempt to fix it in seq.py (below) didn't work.
# So I decided to just copy _FaultToleranceCheckpoint and just save on_exception.

    # def on_save_checkpoint(self, checkpoint):
    #     # TD [2022-07-12] The "completed" counter is off by 1 so when it resumes
    #     # it's off by 1 iteration. However, the data is still off by 1 iteration, probably
    #     # because the dataloader_state_dict['counter'] is off by @batch_size, and idk how
    #     # to fix it cleanly.
    #     checkpoint['loops']['fit_loop']['epoch_loop.batch_progress']['total']['completed'] += 1
    #     checkpoint['loops']['fit_loop']['epoch_loop.batch_progress']['current']['completed'] += 1
    #     checkpoint['loops']['fit_loop']['epoch_loop.state_dict']['_batches_that_stepped'] += 1
    #     checkpoint['loops']['fit_loop']['epoch_loop.state_dict']['dataloader_state_dict'][0]['state'][0]['num_batches_fetched'] += 1
