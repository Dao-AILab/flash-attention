from .arch_base import Arch
class RTX4090(Arch):
    def __init__(self):
        self.reg_cap = 65536 # 32768
        self.smem_cap = 100*1024 # 164*1024
        self.compute_max_core = 128
        self.warp_size = 32
        self.sm_partition = 4
        self.transaction_size = [32, 128]   # in bytes
        self.max_smem_usage = 100 * 1024
        self.bandwidth = [1008, 0] # TODO: 1
        self.platform = "CUDA"
        self.compute_capability = "89"
        self.cutlass_mma = [16, 8, 16]
        self.register_per_thread = 255