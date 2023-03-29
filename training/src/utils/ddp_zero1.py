# Meant to work with Pytorch's ZeroRedundancyOptimizer

from typing import Any, Callable, Dict, List, Optional, Union
from pathlib import Path

import torch
from torch.optim.optimizer import Optimizer
from torch.distributed.optim import ZeroRedundancyOptimizer

from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.core.optimizer import LightningOptimizer
try:  # pytorch_lightning <= 1.7
    from pytorch_lightning.utilities.types import _PATH
except ImportError:  # pytorch_lightning >= 1.8
    try:
        from lightning_lite.utilities.types import _PATH
    except ImportError:  # pytorch_lightning >= 1.9
        from lightning_fabric.utilities.types import _PATH


# Copied from Pytorch's ZeroRedundancyOptimizer's state_dict method, but we only get
# the local state dict to avoid synchronization across GPUs.
# https://github.com/pytorch/pytorch/blob/0c7ca2d97ba5980a2af7dcd6b8106dc915e591cd/torch/distributed/optim/zero_redundancy_optimizer.py#L1131
def get_zero_optimizer_state_dict_local(optimizer, global_rank):
    optimizer._check_overlap_initialized()

    # Sync the exposed `param_groups` attributes to the local optimizer in
    # case they have been updated
    optimizer._sync_param_groups(optimizer.param_groups, optimizer.optim.param_groups)

    local_state_dict = optimizer.optim.state_dict()
    state_dict = super(ZeroRedundancyOptimizer, optimizer).state_dict()

    # Update the global optimizer state with local state information,
    # factoring in the translation from local to global indexing
    rank = global_rank
    # TODO: recursive copy to device
    local_param_groups = local_state_dict["param_groups"]
    global_param_groups = optimizer._partition_parameters()[rank]
    assert len(local_param_groups) == len(global_param_groups), \
        "Mismatch between number of local and global parameter groups"

    for local_param_group, global_param_group in zip(local_param_groups, global_param_groups):
        # `local_param_group` stores local indices, while
        # `global_param_group` stores the tensors directly
        local_param_indices = local_param_group["params"]
        global_params = global_param_group["params"]

        assert len(local_param_indices) == len(global_params), \
            "Mismatch between number of local and global parameters in parameter group"
        for local_param_index, global_param in zip(local_param_indices, global_params):
            # Update the global parameter state, if any
            if local_param_index in local_state_dict["state"]:
                global_param_index = optimizer._param_to_index[global_param]
                state_dict["state"][global_param_index] = local_state_dict["state"][local_param_index]

    # Sort the parameters in the state
    state_dict["state"] = dict(sorted(state_dict["state"].items()))
    return state_dict


class DDPStrategyZero1(DDPStrategy):
    """To use ZeroRedundancyOptimizer, we need to shard the optimizer states when
    saving/loading checkpoints.
    """

    strategy_name = "ddp_zero1"

    def optimizer_state(self, optimizer: Optimizer) -> Optional[dict]:
        if isinstance(optimizer, LightningOptimizer):
            optimizer = optimizer._optimizer
        if isinstance(optimizer, ZeroRedundancyOptimizer):
            return get_zero_optimizer_state_dict_local(optimizer, self.global_rank)
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
            local_optimizer_states = self.checkpoint_io.load_checkpoint(checkpoint_path / f'{self.global_rank:03d}_optim_states.pt')
            global_states['optimizer_states'] = local_optimizer_states
            return global_states
