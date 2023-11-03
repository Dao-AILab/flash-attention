from typing import Any, List
import inspect

import torch
import hydra
from pytorch_lightning import LightningModule, LightningDataModule
from torchmetrics import MetricCollection

from einops import rearrange

from omegaconf import OmegaConf

from src.utils.utils import get_logger
from src.optim.param_grouping import group_parameters_for_optimizer
from src.utils.checkpoint import load_checkpoint

logger = get_logger(__name__)


class SequenceModel(LightningModule):

    def __init__(self, cfg, model_cfg=None):
        """If model_cfg is passed, it will take precedence over cfg.model
        """
        super().__init__()
        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.model_cfg = model_cfg or self.cfg.model

        self.instantiate_datamodule()
        self.instantiate_model()
        self.warmstart()
        self.instantiate_loss()
        self.instantiate_metrics()

    def instantiate_datamodule(self):
        logger.info(f"Instantiating datamodule <{self.cfg.datamodule._target_}>")
        # Calling this self.datamodule will mess with PL since it also assigns self.datamodule
        self._datamodule: LightningDataModule = hydra.utils.instantiate(self.cfg.datamodule)
        self._datamodule.prepare_data()
        self._datamodule.setup()
        OmegaConf.clear_resolver('datamodule')
        OmegaConf.register_new_resolver('datamodule', lambda attr: getattr(self._datamodule, attr))

    def instantiate_model(self):
        # if hasattr(self._datamodule, 'num_classes'):
        #     self.model_cfg.num_classes = self._datamodule.num_classes
        # if (hasattr(self._datamodule, 'vocab_size')
        #     and self.model_cfg.get('embedding_cfg', None) is not None
        #     and self.model_cfg.embedding_cfg._target_ == "torch.nn.Embedding"):
        #     self.model_cfg.embedding_cfg.num_embeddings = self._datamodule.vocab_size
        logger.info(f"Instantiating model <{self.model_cfg._target_}>")
        recursive = getattr(self.model_cfg, '_recursive_', False)
        self.model = hydra.utils.instantiate(self.model_cfg, _recursive_=recursive)

    def instantiate_loss(self):
        loss_fn_cfg = self.cfg.train.get('loss_fn')
        if loss_fn_cfg is None:
            loss_fn_cfg = {'_target_': 'torch.nn.CrossEntropyLoss'}
        self.loss_fn = hydra.utils.instantiate(loss_fn_cfg)
        loss_fn_val_cfg = self.cfg.train.get('loss_fn_val', loss_fn_cfg)
        self.loss_fn_val = hydra.utils.instantiate(loss_fn_val_cfg)

    def instantiate_metrics(self):
        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        if 'eval' in self.cfg and 'metrics' in self.cfg.eval:
            metrics_cfg = self.cfg.eval.metrics
        else:
            metrics_cfg = {'acc': {'_target_': 'torchmetrics.Accuracy'}}
        metrics = MetricCollection({name: hydra.utils.instantiate(cfg)
                                    for name, cfg in metrics_cfg.items()})
        self.train_metrics = metrics.clone(prefix='train/')
        self.val_metrics = metrics.clone(prefix='val/')
        self.test_metrics = metrics.clone(prefix='test/')

    def warmstart(self):
        if self.cfg.train.get('warmstart', None) is not None:
            logger.info(f"Warm-starting with weights from {self.cfg.train.warmstart.path}")
            strict = self.cfg.train.warmstart.get('strict', True)
            state_dict = load_checkpoint(self.cfg.train.warmstart.path)
            if self.cfg.train.warmstart.get('post_process', None) is not None:
                state_dict = hydra.utils.instantiate(self.cfg.train.warmstart.post_process,
                                                     state_dict)
            load_return = self.model.load_state_dict(state_dict, strict=False)
            logger.info(load_return)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def step(self, batch: Any, is_train=True):
        try:
            x, y, lengths = batch
        except ValueError:
            x, y = batch
            lengths = None
        output = self.forward(x) if lengths is None else self.forward(x, lengths=lengths)
        loss = self.loss_fn(output, y) if is_train else self.loss_fn_val(output, y)
        return loss, output, y

    def shared_step(self, batch: Any, batch_idx: int, phase='train'):
        loss, output, targets = self.step(batch, is_train=(phase == 'train'))
        metrics = getattr(self, f'{phase}_metrics')
        metrics(output, targets)
        log_on_step = 'eval' in self.cfg and self.cfg.eval.get('log_on_step', False) and phase == 'train'
        self.log(f"{phase}/loss", loss, on_step=log_on_step, on_epoch=True,
                 prog_bar=False, sync_dist=True)
        # https://pytorch-lightning.readthedocs.io/en/stable/visualize/logging_advanced.html#enable-metrics-for-distributed-training
        # We need to log the Metrics object, not the metric result, since otherwise
        # pytorch-lightning will use torch.mean to reduce it.
        # This would be wrong for perplexity, for example.
        self.log_dict(metrics, on_step=log_on_step, on_epoch=True, prog_bar=True, sync_dist=True)
        return {"loss": loss, "output": output, "targets": targets}

    def training_step(self, batch: Any, batch_idx: int):
        return self.shared_step(batch, batch_idx, phase='train')

    def validation_step(self, batch: Any, batch_idx: int):
        return self.shared_step(batch, batch_idx, phase='val')

    def test_step(self, batch: Any, batch_idx: int):
        return self.shared_step(batch, batch_idx, phase='test')

    def configure_optimizers(self):
        if 'optimizer_param_grouping' in self.cfg.train:  # Set zero weight decay for some params
            parameters = group_parameters_for_optimizer(self.model, self.cfg.train.optimizer,
                                                        **self.cfg.train.optimizer_param_grouping)
        else:
            # parameters = self.model.parameters()
            parameters = self.parameters() # [21-09-08] AG: this will train task specific parameters such as Retrieval head for AAN
        optimizer = hydra.utils.instantiate(self.cfg.train.optimizer, parameters)

        # Log optimizer info
        for i, g in enumerate(optimizer.param_groups):
            ntensors = len(g['params'])
            nparams = sum(p.numel() for p in g['params'])
            hparams = {k: v for k, v in g.items() if k != 'params'}
            logger.info(f'Optimizer group {i}: {ntensors} tensors, {nparams} parameters, {hparams}')

        if 'scheduler' not in self.cfg.train:
            return optimizer
        else:
            # lr_scheduler should be called either every step (default) or every epoch
            lr_scheduler = hydra.utils.instantiate(self.cfg.train.scheduler, optimizer)
            return [optimizer], {'scheduler': lr_scheduler,
                                 'interval': self.cfg.train.get('scheduler_interval', 'step'),
                                 'monitor': self.cfg.train.get('scheduler_monitor', 'val/loss')}

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        # https://pytorch-lightning.readthedocs.io/en/latest/guides/speed.html#set-grads-to-none
        # TD [2022-04-30]: DeepSpeed optimizer uses the kwarg set_grad_to_none instead of set_to_none
        if 'set_to_none' in inspect.signature(optimizer.zero_grad).parameters:
            optimizer.zero_grad(set_to_none=True)
        else:
            optimizer.zero_grad()

    def on_save_checkpoint(self, checkpoint):
        # TD [2022-08-07] ['epoch_loop.batch_progress']['total']['completed'] is 1 iteration
        # behind, so we're using the optimizer's progress.
        checkpoint['loops']['fit_loop']['epoch_loop.batch_progress']['total']['completed'] = checkpoint['loops']['fit_loop']['epoch_loop.batch_loop.optimizer_loop.optim_progress']['optimizer']['step']['total']['completed'] * self.trainer.accumulate_grad_batches
        checkpoint['loops']['fit_loop']['epoch_loop.batch_progress']['current']['completed'] = checkpoint['loops']['fit_loop']['epoch_loop.batch_loop.optimizer_loop.optim_progress']['optimizer']['step']['current']['completed'] * self.trainer.accumulate_grad_batches
        # _batches_that_stepped tracks the number of global steps, not the number
        # of local steps, so we don't multiply with self.trainer.accumulate_grad_batches here.
        checkpoint['loops']['fit_loop']['epoch_loop.state_dict']['_batches_that_stepped'] = checkpoint['loops']['fit_loop']['epoch_loop.batch_loop.optimizer_loop.optim_progress']['optimizer']['step']['total']['completed']


class SequenceLMModel(SequenceModel):

    def step(self, batch: Any, is_train=True):
        x, y = batch
        output = self.forward(x).logits
        output = rearrange(output, '... C -> (...) C')
        y = rearrange(y, '... -> (...)')
        loss = self.loss_fn(output, y) if is_train else self.loss_fn_val(output, y)
        return loss, output, y

    def shared_step(self, batch: Any, batch_idx: int, phase='train'):
        loss, output, targets = self.step(batch, is_train=(phase == 'train'))
        # Passing the loss to the perplexity metrics to avoid recomputation
        metrics = getattr(self, f'{phase}_metrics')
        metrics(output, targets, loss=loss)
        log_on_step = 'eval' in self.cfg and self.cfg.eval.get('log_on_step', False) and phase == 'train'
        self.log(f"{phase}/loss", loss, on_step=log_on_step, on_epoch=True,
                 prog_bar=False, sync_dist=True)
        # https://pytorch-lightning.readthedocs.io/en/stable/visualize/logging_advanced.html#enable-metrics-for-distributed-training
        # We need to log the Metrics object, not the metric result, since otherwise
        # pytorch-lightning will use torch.mean to reduce it.
        # This would be wrong for perplexity, for example.
        self.log_dict(metrics, on_step=log_on_step, on_epoch=True, prog_bar=True, sync_dist=True)
        return {"loss": loss, "output": output, "targets": targets}
