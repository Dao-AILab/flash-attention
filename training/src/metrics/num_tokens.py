from typing import Any, Dict, Optional

import torch
from torch import Tensor

from torchmetrics import Metric


class NumTokens(Metric):
    """Keep track of how many tokens we've seen.
    """
    # TODO: how do we prevent the reset between the epochs? The reset happens on the 1st batch
    # of the next epoch.
    # Right now the hack is that we override reset(), which would mess up the forward method.
    # We then override forward to do the right thing.

    is_differentiable = False
    higher_is_better = False
    full_state_update = False
    count: Tensor

    def __init__(self, **kwargs: Dict[str, Any]):
        super().__init__(**kwargs)
        self.add_state("count", default=torch.tensor(0, dtype=torch.int64), dist_reduce_fx="sum",
                       persistent=True)  # We want the count to be saved to state-dict

    def update(self, preds: Tensor, target: Tensor, loss: Optional[Tensor] = None) -> None:  # type: ignore
        self.count += target.numel()

    def compute(self) -> Tensor:
        return self.count

    def reset(self):
        count = self.count
        super().reset()
        self.count = count

    # Adapted from https://github.com/Lightning-AI/metrics/blob/master/src/torchmetrics/metric.py
    def _forward_reduce_state_update(self, *args: Any, **kwargs: Any) -> Any:
        """forward computation using single call to `update` to calculate the metric value on the current batch and
        accumulate global state.
        This can be done when the global metric state is a sinple reduction of batch states.
        """
        self.update(*args, **kwargs)
        return self.compute()
