import torch

from timm.data import Mixup
from timm.data.mixup import mixup_target


class TimmMixup(Mixup):
    """ Wrap timm.data.Mixup that avoids the assert that batch size must be even.
    """
    def __call__(self, x, target):
        if self.mode == 'elem':
            lam = self._mix_elem(x)
        elif self.mode == 'pair':
            # We move the assert from the beginning of the function to here
            assert len(x) % 2 == 0, 'Batch size should be even when using this'
            lam = self._mix_pair(x)
        else:
            lam = self._mix_batch(x)
        target = mixup_target(target, self.num_classes, lam, self.label_smoothing, x.device)
        return x, target
