# Adapted from https://pytorch.org/docs/stable/_modules/torch/distributed/algorithms/ddp_comm_hooks/default_hooks.html
# We divide by world_size first before converting to fp16, so it's safer.
from typing import Any, Callable

import torch
import torch.distributed as dist


def fp16_compress_hook(
    process_group: dist.ProcessGroup, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    """
    This DDP communication hook implements a simple gradient compression
    approach that casts ``GradBucket`` tensor to half-precision floating-point format (``torch.float16``)
    and then divides it by the process group size.
    It allreduces those ``float16`` gradient tensors. Once compressed gradient
    tensors are allreduced, the chained callback ``decompress`` casts it back to the input data type (such as ``float32``).

    Example::
        >>> ddp_model.register_comm_hook(process_group, fp16_compress_hook)
    """
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    world_size = group_to_use.size()

    # Divide first before converting to fp16
    # Use out argument to fuse the division and the conversion.
    compressed_tensor = torch.div(bucket.buffer(), world_size,
                                  out=torch.empty_like(bucket.buffer(), dtype=torch.float16))

    fut = dist.all_reduce(
        compressed_tensor, group=group_to_use, async_op=True
    ).get_future()

    def decompress(fut):
        decompressed_tensor = bucket.buffer()
        # Decompress in place to reduce the peak memory.
        # See: https://github.com/pytorch/pytorch/issues/45968
        decompressed_tensor.copy_(fut.value()[0])
        return decompressed_tensor

    # TODO: maybe have a backoff strategy: check if the buffer has inf / NaN, in that case
    # resend with fp32?
    return fut.then(decompress)
