# Meant to work with Apex's DistributeFusedAdam

from typing import Any, Callable, Dict, List, Optional, Union
from pathlib import Path
import types

import torch
from torch.optim.optimizer import Optimizer
from torch.optim import LBFGS

from apex.contrib.optimizers.distributed_fused_adam import DistributedFusedAdam

from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.plugins.precision import PrecisionPlugin, NativeMixedPrecisionPlugin
from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.utilities.exceptions import MisconfigurationException
try:  # pytorch_lightning <= 1.7
    from pytorch_lightning.utilities.types import _PATH
except ImportError:  # pytorch_lightning >= 1.8
    try:
        from lightning_lite.utilities.types import _PATH
    except ImportError:  # pytorch_lightning >= 1.9
        from lightning_fabric.utilities.types import _PATH


class DistAdamNativeMixedPrecisionPlugin(NativeMixedPrecisionPlugin):

    def optimizer_step(  # type: ignore[override]
        self,
        model: "pl.LightningModule",
        optimizer,
        optimizer_idx: int,
        closure: Callable[[], Any],
        **kwargs: Any,
    ) -> Any:
        if self.scaler is None:
            # skip scaler logic, as bfloat16 does not require scaler
            return NativeMixedPrecisionPlugin.optimizer_step(
                self, optimizer, model=model, optimizer_idx=optimizer_idx, closure=closure, **kwargs
            )
        if isinstance(optimizer, LBFGS):
            raise MisconfigurationException(
                f"Native AMP and the LBFGS optimizer are not compatible (optimizer {optimizer_idx})."
            )
        closure_result = closure()
        # HACK: we don't call self.scaler.unscale_ here. This is because DistributedFusedAdam
        # optimizer internally takes the scale into account.
        # If we call unscale_ here, it would be equivalent to unscaling the gradients twice.
        # Not unscaling has the side-effect that the NormMonitor callback will report the
        # gradient norm to be much larger than reality.
        # # `unscale` after the closure is executed but before the `on_before_optimizer_step` hook.
        # self.scaler.unscale_(optimizer)
        # This will call gradient clipping
        self._after_closure(model, optimizer, optimizer_idx)
        skipped_backward = closure_result is None
        # in manual optimization, the closure does not return a value
        if not model.automatic_optimization or not skipped_backward:
            # note: the scaler will skip the `optimizer.step` if nonfinite gradients are found
            step_output = self.scaler.step(optimizer, **kwargs)
            self.scaler.update()
            return step_output
        return closure_result

    def clip_grad_by_norm(self, optimizer: DistributedFusedAdam, clip_val: Union[int, float]) -> None:
        """Clip gradients by norm."""
        # DistributedFusedAdam wants list, not generator
        # Gradients have not be scaled, so we need to scale up the clip_val
        if self.scaler is not None:
            clip_val *= self.scaler.get_scale()
        return optimizer.clip_grad_norm(clip_val)


class DDPStrategyZero2(DDPStrategy):
    """To use Apex's DistributedFusedAdam, we need to shard the optimizer states when
    saving/loading checkpoints.
    """

    strategy_name = "ddp_zero2"

    def __init__(
        self,
        *args,
        precision_plugin: Optional[PrecisionPlugin] = DistAdamNativeMixedPrecisionPlugin,
        # precision_plugin: Optional[PrecisionPlugin] = None,
        **kwargs: Union[Any, Dict[str, Any]],
    ) -> None:
        super().__init__(
            *args, precision_plugin=precision_plugin, **kwargs
        )

    @property
    def precision_plugin(self) -> PrecisionPlugin:
        return self._precision_plugin if self._precision_plugin is not None else PrecisionPlugin()

    @precision_plugin.setter
    def precision_plugin(self, precision_plugin: Optional[PrecisionPlugin]) -> None:
        self._precision_plugin = precision_plugin
        # https://stackoverflow.com/questions/972/adding-a-method-to-an-existing-object-instance
        self._precision_plugin.optimizer_step = types.MethodType(
            DistAdamNativeMixedPrecisionPlugin.optimizer_step, self._precision_plugin
        )
        self._precision_plugin.clip_grad_by_norm = types.MethodType(
            DistAdamNativeMixedPrecisionPlugin.clip_grad_by_norm, self._precision_plugin
        )

    def optimizer_state(self, optimizer: Optimizer) -> Optional[dict]:
        if isinstance(optimizer, LightningOptimizer):
            optimizer = optimizer._optimizer
        if isinstance(optimizer, DistributedFusedAdam):
            return optimizer.state_dict(gather_on_root=False)
        else:
            return optimizer.state_dict()

    def save_checkpoint(
        self, checkpoint: Dict[str, Any], filepath: _PATH, storage_options: Optional[Any] = None
    ) -> None:
        """Save model/training states as a checkpoint file through state-dump and file-write.
        Args:
            checkpoint: dict containing model and trainer state
            filepath: write-target file's path
            storage_options: parameter for how to save to storage, passed to ``CheckpointIO`` plugin
        """
        filepath = Path(filepath)
        filepath.mkdir(parents=True, exist_ok=True)
        local_optimizer_states = checkpoint.pop('optimizer_states')
        if self.is_global_zero:
            self.checkpoint_io.save_checkpoint(checkpoint, filepath / 'model_states.pt',
                                               storage_options=storage_options)
        self.checkpoint_io.save_checkpoint(local_optimizer_states,
                                           filepath / f'{self.global_rank:03d}_optim_states.pt',
                                           storage_options=storage_options)

    def load_checkpoint(self, checkpoint_path: _PATH) -> Dict[str, Any]:
        torch.cuda.empty_cache()
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.is_file():
            return super().load_checkpoint(self, str(checkpoint_path))
        else:
            assert checkpoint_path.is_dir()
            global_states = self.checkpoint_io.load_checkpoint(checkpoint_path / 'model_states.pt')
            local_optimizer_states = self.checkpoint_io.load_checkpoint(
                checkpoint_path / f'{self.global_rank:03d}_optim_states.pt',
                map_location='cuda'
            )
            global_states['optimizer_states'] = local_optimizer_states
            return global_states
