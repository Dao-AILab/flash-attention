# Inspired by https://github.com/NVIDIA/apex/blob/master/apex/transformer/tensor_parallel/cross_entropy.py
# But we make it much faster: we compute the local loss and the LSE, and by exchanging the LSE and
# the losses we can get the global loss. There's no need to do it step by step
# (compute local max, exchange, compute exp, compute local sum, exchange, etc.)
import torch
import torch.nn as nn

import xentropy_cuda_lib

from apex.transformer.parallel_state import get_tensor_model_parallel_group
from apex.transformer.parallel_state import get_tensor_model_parallel_rank
from apex.transformer.parallel_state import get_tensor_model_parallel_world_size
from apex.transformer.tensor_parallel.utils import VocabUtility

# `all_gather_into_tensor` and `reduce_scatter_tensor` are new placeholders for
# `_all_gather_base` and `_reduce_scatter_base`. They require the most recent
# version of PyTorch. The following 4 lines are for backward compatibility with
# older PyTorch.
if "all_gather_into_tensor" not in dir(torch.distributed):
    torch.distributed.all_gather_into_tensor = torch.distributed._all_gather_base
if "reduce_scatter_tensor" not in dir(torch.distributed):
    torch.distributed.reduce_scatter_tensor = torch.distributed._reduce_scatter_base


class SoftmaxCrossEntropyLossParallelFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, logits_parallel, labels, smoothing=0.0, ignored_index=-100,
                inplace_backward=False):
        """
        logits_parallel: (batch, vocab_size / world_size)
        labels: (batch,)
        """
        assert smoothing == 0.0, 'smoothing != 0.0 is not yet implemented, file an issue if you need it'
        batch, partition_vocab_size = logits_parallel.shape
        assert labels.shape == (batch,)
        rank = get_tensor_model_parallel_rank()
        world_size = get_tensor_model_parallel_world_size()

        if world_size == 1:
            losses, lse = xentropy_cuda_lib.forward(logits_parallel, labels, smoothing)
            losses.masked_fill_(labels==ignored_index, 0)
            labels_local = labels
        else:
            vocab_start_index, vocab_end_index = VocabUtility.vocab_range_from_per_partition_vocab_size(
                partition_vocab_size, get_tensor_model_parallel_rank(),
                get_tensor_model_parallel_world_size()
            )

            # Create a mask of valid vocab ids (1 means it needs to be masked).
            labels_mask = (labels < vocab_start_index) | (labels >= vocab_end_index)
            ignored_mask = labels == ignored_index
            labels_local = torch.where(ignored_mask, labels, labels - vocab_start_index)
            masked_labels = labels_local.clone()
            masked_labels[labels_mask] = ignored_index

            losses, lse_local = xentropy_cuda_lib.forward(logits_parallel, masked_labels, smoothing)
            assert lse_local.shape == (batch,)
            assert losses.shape == (batch,)
            losses.masked_fill_(masked_labels==ignored_index, 0)

            lse_allgather = torch.empty(world_size, batch, dtype=lse_local.dtype,
                                        device=lse_local.device)
            handle_lse = torch.distributed.all_gather_into_tensor(
                lse_allgather, lse_local.contiguous(),
                group=get_tensor_model_parallel_group(), async_op=True
            )
            handle_losses = torch.distributed.all_reduce(
                losses, op=torch.distributed.ReduceOp.SUM,
                group=get_tensor_model_parallel_group(), async_op=True
            )
            handle_lse.wait()
            lse = torch.logsumexp(lse_allgather, dim=0)
            # The losses are going to be lse_local - predicted_logit, we just have to subtract
            # the lse_local and add the lse (global).
            rank_per_sample = torch.div(labels, partition_vocab_size, rounding_mode='floor')
            lse_local = lse_allgather[rank_per_sample,
                                      torch.arange(batch, device=lse_allgather.device)]

            handle_losses.wait()
            losses += lse - lse_local
            losses.masked_fill_(ignored_mask, 0)

        ctx.save_for_backward(logits_parallel, lse, labels_local)
        ctx.smoothing = smoothing
        ctx.ignored_index = ignored_index
        ctx.inplace_backward = inplace_backward
        return losses

    @staticmethod
    def backward(ctx, grad_loss):
        logits_parallel, lse, labels = ctx.saved_tensors
        if not grad_loss.is_contiguous():
            grad_loss = grad_loss.contiguous()
        grad_loss.masked_fill_(labels==ctx.ignored_index, 0)
        grad_logits = xentropy_cuda_lib.backward(grad_loss, logits_parallel, lse, labels,
                                                 ctx.smoothing, ctx.inplace_backward)
        return grad_logits, None, None, None, None, None


class CrossEntropyLossParallel(nn.Module):

    def __init__(self, ignore_index=-100, reduction='mean', label_smoothing=0.0,
                 inplace_backward=False):
        super().__init__()
        if reduction not in ['mean', 'none']:
            raise NotImplementedError("Only support reduction = 'mean' or 'none'")
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.inplace_backward = inplace_backward

    def forward(self, input, target):
        assert input.is_cuda and target.is_cuda
        # SoftmaxCrossEntropyLoss implicitly casts to float
        loss = SoftmaxCrossEntropyLossParallelFn.apply(
            input, target, self.label_smoothing, self.ignore_index, self.inplace_backward
        )
        if self.reduction == 'mean':
            return loss.sum() / (target != self.ignore_index).sum()
        else:
            return loss
