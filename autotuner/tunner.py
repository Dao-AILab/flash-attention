
import ctypes
import os
import torch

from base_tunner import BaseTunner
from configs.fwd_config import FlashFwdConfig

class FlashFwdTunner(BaseTunner):
    def __init__(self, arch, torch_array: list, shape_config, profile_config, tempdir: str):
        super().__init__(arch, torch_array, "flash_fwd", shape_config, profile_config, tempdir)
    
    def validate_register_fuse(self, config):
        Br = config.Br
        Bc = config.Bc
        Kd = config.Kd
        D = config.D
        Nthreads = config.Nwarps * 32
        mmam, mman, mmak = self.arch.cutlass_mma
        belem_per_thread = mman*mmak/self.arch.warp_size
    
        # check tile size
        if Br % (mmam*Nthreads/self.arch.warp_size) != 0:
            return False
        # check shared memory
        smem_size_q = config.Br * config.Kd * 2
        smem_size_k = config.Bc * config.Kd * 2
        smem_size_qk = smem_size_q + smem_size_k
        smem_size_v = config.Bc * config.D * 2
        smem_out = config.Br * config.D * 2
        if config.SharedQKSmem:
            smem_size = max(smem_size_q, smem_size_k+smem_size_v)
        else:
            smem_size = smem_size_qk + smem_size_v
        smem_size = max(smem_size, smem_out)
        if smem_size > self.arch.smem_cap:
            return False
        # check register
        reg_used_accum = (Br * D * 4 + Br*Bc*4)/(Nthreads * 4)
        reg_used_matmul2 = (Br * D * 4 + Br*Bc*2)/(Nthreads * 4) + (D/(mman*1) * belem_per_thread*2) / 4
        reg_used_matmul1 = (Br * D * 4 + Br * Bc * 4)/(Nthreads * 4) + (Bc/(mman*1) * belem_per_thread*2) / 4
        reg_used_qinregs = (Br * Kd * 2)/(Nthreads * 4)
        if config.isQinRegs:
            reg_used = reg_used_accum + reg_used_qinregs
        else:
            reg_used = reg_used_accum # max(reg_used_accum, reg_used_matmul2, reg_used_matmul1)
        if reg_used > min(self.arch.register_per_thread, self.arch.reg_cap/Nthreads):
            return False
        return True
    
    def generate_configs(self,Br:int,Bc:int,dim_qk:int,dim_v:int):
        configs = []
        # TODO: more general
        for Nthreads in [128, 256]:
            config1 = FlashFwdConfig(dim_qk,dim_v,Br,Bc,Nthreads//32,False,False)
            config2 = FlashFwdConfig(dim_qk,dim_v,Br,Bc,Nthreads//32,True,False)
            config3 = FlashFwdConfig(dim_qk,dim_v,Br,Bc,Nthreads//32,True,True)
            configs.append(config1)
            configs.append(config2)
            configs.append(config3)
        return configs
