import torch
from torch import Tensor

from torchmetrics import Metric, Accuracy


class AccuracyMine(Accuracy):
    """Wrap torchmetrics.Accuracy to take argmax of y in case of Mixup.
    """
    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        super().update(preds, target.argmax(dim=-1) if target.is_floating_point() else target)
