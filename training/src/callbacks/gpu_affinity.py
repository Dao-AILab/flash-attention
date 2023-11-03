import torch

from pytorch_lightning import Callback, Trainer, LightningModule

import logging

log = logging.getLogger(__name__)  # We want a logger for each process, not just the rank 0


def l2_promote():
    import ctypes
    _libcudart = ctypes.CDLL('libcudart.so')
    # Set device limit on the current device
    # cudaLimitMaxL2FetchGranularity = 0x05
    pValue = ctypes.cast((ctypes.c_int*1)(), ctypes.POINTER(ctypes.c_int))
    _libcudart.cudaDeviceSetLimit(ctypes.c_int(0x05), ctypes.c_int(128))
    _libcudart.cudaDeviceGetLimit(pValue, ctypes.c_int(0x05))
    assert pValue.contents.value == 128


def set_affinity(trainer):
    try:
        from src.utils.gpu_affinity import set_affinity
        nproc_per_node = torch.cuda.device_count()
        affinity = set_affinity(trainer.local_rank, nproc_per_node, 'socket_unique_continuous')
        log.info(f'{trainer.local_rank}: thread affinity: {affinity}')
        # TD [2022-05-07] Somehow calling this causes GPU 0 to allocate extra ~800MB of memory per
        # number of GPUs (e.g., 6.4GB of extra memory in a 8-GPU setup). H/t Dan.
        # l2_promote()
    except:
        pass


class GpuAffinity(Callback):
    """Set GPU affinity and increase the L2 fetch granularity.
    Adapted from https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/Transformer-XL
    """

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage=None) -> None:
        set_affinity(trainer)
