# Inspired by https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/common/metrics/perplexity.py
# But we compute the perplexity correctly: exp(average(nll)), not average(exp(nll))
# Also adapted from https://github.com/Lightning-AI/metrics/blob/master/src/torchmetrics/text/perplexity.py
# But we pass in the loss to avoid recomputation

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torchmetrics import Metric

try:
    from flash_attn.losses.cross_entropy import CrossEntropyLoss
except ImportError:
    CrossEntropyLoss = torch.nn.CrossEntropyLoss

__all__ = ['Perplexity']


class Perplexity(Metric):
    r"""
    Perplexity measures how well a language model predicts a text sample. It's calculated as the average number of bits
    per word a model needs to represent the sample.
    Args:
        kwargs:
            Additional keyword arguments, see :ref:`Metric kwargs` for more info.
    Examples:
        >>> import torch
        >>> preds = torch.rand(2, 8, 5, generator=torch.manual_seed(22))
        >>> target = torch.randint(5, (2, 8), generator=torch.manual_seed(22))
        >>> target[0, 6:] = -100
        >>> metric = Perplexity(ignore_index=-100)
        >>> metric(preds, target)
        tensor(5.2545)
    """
    is_differentiable = True
    higher_is_better = False
    full_state_update = False
    total_log_probs: Tensor
    count: Tensor

    def __init__(self, **kwargs: Dict[str, Any]):
        super().__init__(**kwargs)
        self.add_state("total_log_probs", default=torch.tensor(0.0, dtype=torch.float64),
                       dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0, dtype=torch.int64), dist_reduce_fx="sum")

        self.loss_fn = CrossEntropyLoss()

    def update(self, preds: Tensor, target: Tensor, loss: Optional[Tensor] = None) -> None:  # type: ignore
        """Compute and store intermediate statistics for Perplexity.
        Args:
            preds:
                Probabilities assigned to each token in a sequence with shape [batch_size, seq_len, vocab_size].
            target:
                Ground truth values with a shape [batch_size, seq_len].
        """
        count = target.numel()
        if loss is None:
            loss = self.loss_fn(preds, target)
        self.total_log_probs += loss.double() * count
        self.count += count

    def compute(self) -> Tensor:
        """Compute the Perplexity.
        Returns:
           Perplexity
        """
        return torch.exp(self.total_log_probs / self.count)
