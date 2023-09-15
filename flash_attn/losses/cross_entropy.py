# Inspired by https://github.com/NVIDIA/apex/blob/master/apex/transformer/tensor_parallel/cross_entropy.py
# But we make it much faster: we compute the local loss and the LSE, and by exchanging the LSE and
# the losses we can get the global loss. There's no need to do it step by step
# (compute local max, exchange, compute exp, compute local sum, exchange, etc.)
# The original xentropy interface is here: https://github.com/NVIDIA/apex/blob/master/apex/contrib/xentropy/softmax_xentropy.py
import torch
import torch.nn as nn

import xentropy_cuda_lib

# `all_gather_into_tensor` and `reduce_scatter_tensor` are new placeholders for
# `_all_gather_base` and `_reduce_scatter_base`. They require the most recent
# version of PyTorch. The following 2 lines are for backward compatibility with
# older PyTorch.
if "all_gather_into_tensor" not in dir(torch.distributed):
    torch.distributed.all_gather_into_tensor = torch.distributed._all_gather_base


class SoftmaxCrossEntropyLossFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, logits, labels, smoothing=0.0, ignored_index=-100, inplace_backward=False,
                process_group=None):
        """
        logits: (batch, vocab_size)
        labels: (batch,)
        If process_group is not None, we're doing Tensor Parallel: each process is responsible for
        one part of the vocab. The loss needs to be aggregated across processes.
        """
        batch, vocab_size = logits.shape
        assert labels.shape == (batch,)
        world_size = 1 if process_group is None else torch.distributed.get_world_size(process_group)
        ctx.total_classes = world_size * vocab_size

        if world_size == 1:
            losses, lse = xentropy_cuda_lib.forward(logits, labels, smoothing)
            losses.masked_fill_(labels==ignored_index, 0)
            labels_local = labels
        else:
            rank = torch.distributed.get_rank(process_group)
            vocab_start_index, vocab_end_index = rank * vocab_size, (rank + 1) * vocab_size

            # Create a mask of valid vocab ids (1 means it needs to be masked).
            labels_mask = (labels < vocab_start_index) | (labels >= vocab_end_index)
            ignored_mask = labels == ignored_index
            labels_local = torch.where(ignored_mask, labels, labels - vocab_start_index)

            # For tensor parallel cross entropy with smoothing, we want to pass in the total number
            # of classes so that smoothing can be applied correctly. If total_classes=-1, use the
            # last dimension of the input tensor.
            losses, lse_local = xentropy_cuda_lib.forward(logits, labels_local, smoothing,
                                                          world_size * vocab_size)
            assert lse_local.shape == (batch,)
            assert losses.shape == (batch,)
            losses.masked_fill_(ignored_mask, 0)
            # For labels == ignored_index, the loss is always 0.
            # If there's no smoothing, if labels are in the vocab of this partition, losses contains
            # lse_local - predicted logit, and 0 otherwise.
            # If there's smoothing=0.1, for labels in the vocab of this partition, losses contains
            # 0.9 * (lse_local - predicted logit) + 0.1 * (lse_local - sum logit / total_classes)
            # For labels not in the vocab of this partition, losses contains
            # 0.1 * (lse_local - sum logit / total_classes).

            lse_allgather = torch.empty(world_size, batch, dtype=lse_local.dtype,
                                        device=lse_local.device)
            torch.distributed.all_gather_into_tensor(lse_allgather, lse_local.contiguous(),
                                                     group=process_group)
            handle_losses = torch.distributed.all_reduce(
                losses, op=torch.distributed.ReduceOp.SUM, group=process_group, async_op=True
            )
            lse = torch.logsumexp(lse_allgather, dim=0)
            # If there's no smoothing, the total losses are lse_local - predicted_logit,
            # we just have to subtract the lse_local and add the lse (global).
            # If there's smoothing=0.1, the total losses are
            # 0.9 * (lse_local - predicted_logit) + 0.1 * (sum of all lse_local - sum logit / total_classes)
            # We want 0.9 * (lse - predicted_logit) + 0.1 * (lse - sum logit / total_classes).
            rank_per_sample = torch.div(labels, vocab_size, rounding_mode='floor')
            lse_local = lse_allgather[rank_per_sample,
                                      torch.arange(batch, device=lse_allgather.device)]

            handle_losses.wait()
            if smoothing == 0.0:
                losses += lse - lse_local
            else:
                losses += ((1 - smoothing) * (lse - lse_local)
                           + smoothing * (lse - lse_allgather.sum(dim=0)))
            losses.masked_fill_(ignored_mask, 0)

        ctx.save_for_backward(logits, lse, labels_local)
        ctx.smoothing = smoothing
        ctx.ignored_index = ignored_index
        ctx.inplace_backward = inplace_backward
        return losses

    @staticmethod
    def backward(ctx, grad_loss):
        logits, lse, labels = ctx.saved_tensors
        grad_loss = grad_loss.contiguous()
        grad_loss.masked_fill_(labels==ctx.ignored_index, 0)
        grad_logits = xentropy_cuda_lib.backward(grad_loss, logits, lse, labels,
                                                 ctx.smoothing, ctx.inplace_backward,
                                                 ctx.total_classes)
        return grad_logits, None, None, None, None, None, None


class CrossEntropyLoss(nn.Module):

    def __init__(self, ignore_index=-100, reduction='mean', label_smoothing=0.0,
                 inplace_backward=False, process_group=None):
        super().__init__()
        if reduction not in ['mean', 'none']:
            raise NotImplementedError("Only support reduction = 'mean' or 'none'")
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.inplace_backward = inplace_backward
        self.process_group = process_group

    def forward(self, input, target):
        assert input.is_cuda and target.is_cuda
        # SoftmaxCrossEntropyLoss implicitly casts to float
        loss = SoftmaxCrossEntropyLossFn.apply(
            input, target, self.label_smoothing, self.ignore_index, self.inplace_backward,
            self.process_group
        )
        if self.reduction == 'mean':
            return loss.sum() / (target != self.ignore_index).sum()
        else:
            return loss
