from typing import Optional

import torch
from torch import Tensor
from torch.distributed import ProcessGroup

# `all_gather_into_tensor` and `reduce_scatter_tensor` are new placeholders for
# `_all_gather_base` and `_reduce_scatter_base`. They require the most recent
# version of PyTorch. The following 4 lines are for backward compatibility with
# older PyTorch.
if "all_gather_into_tensor" not in dir(torch.distributed):
    torch.distributed.all_gather_into_tensor = torch.distributed._all_gather_base
if "reduce_scatter_tensor" not in dir(torch.distributed):
    torch.distributed.reduce_scatter_tensor = torch.distributed._reduce_scatter_base


# Raw operation, oes does support autograd, but does support async
def all_gather_raw(input_: Tensor, process_group: ProcessGroup, async_op: bool = False):
    world_size = torch.distributed.get_world_size(process_group)
    output = torch.empty(world_size * input_.shape[0], *input_.shape[1:],
                         dtype=input_.dtype, device=input_.device)
    handle = torch.distributed.all_gather_into_tensor(output, input_.contiguous(),
                                                      group=process_group, async_op=async_op)
    return output, handle


# Raw operation, oes does support autograd, but does support async
def reduce_scatter_raw(input_: Tensor, process_group: ProcessGroup, async_op: bool = False):
    world_size = torch.distributed.get_world_size(process_group)
    assert input_.shape[0] % world_size == 0
    output = torch.empty(input_.shape[0] // world_size, *input_.shape[1:],
                         dtype=input_.dtype, device=input_.device)
    handle = torch.distributed.reduce_scatter_tensor(output, input_.contiguous(),
                                                     group=process_group,
                                                     async_op=async_op)
    return output, handle


class AllGatherFunc(torch.autograd.Function):
    """Gather the input from sequence parallel region and concatenate."""

    @staticmethod
    def forward(ctx, input_: Tensor, process_group: ProcessGroup) -> Tensor:
        ctx.process_group = process_group
        output, _ = all_gather_raw(input_, process_group)
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        grad_input, _ = reduce_scatter_raw(grad_output, ctx.process_group)
        return grad_input, None


# Supports autograd, but does not support async
all_gather = AllGatherFunc.apply


class ReduceScatterFunc(torch.autograd.Function):
    """Reduce scatter the input from the sequence parallel region and concatenate."""

    @staticmethod
    def forward(ctx, input_: Tensor, process_group: ProcessGroup) -> Tensor:
        ctx.process_group = process_group
        output, _ = reduce_scatter_raw(input_, process_group)
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        grad_input, _ = all_gather_raw(grad_output, ctx.process_group)
        return grad_input, None


# Supports autograd, but does not support async
reduce_scatter = ReduceScatterFunc.apply
