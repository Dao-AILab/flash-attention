# Copyright (c) 2026, Tri Dao.
"""Detect device-to-host synchronization inside a warmed kernel call."""

from torch.utils._python_dispatch import TorchDispatchMode


class RejectHostScalarExtraction(TorchDispatchMode):
    """Fail on aten._local_scalar_dense, the op behind .item()/int()/bool() on a tensor.

    Wrap a call whose kernels are already compiled: any host scalar extraction is a
    device-to-host synchronization, which stalls the stream and makes the call
    impossible to capture in a CUDA graph.
    """

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func.name() == "aten::_local_scalar_dense":
            raise AssertionError(
                "host scalar extraction (aten::_local_scalar_dense) during a warmed "
                "call; this is a device-to-host synchronization"
            )
        return func(*args, **(kwargs or {}))
