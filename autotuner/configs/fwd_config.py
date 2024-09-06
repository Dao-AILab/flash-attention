import os
from .base_config import BaseConfig

class FlashFwdConfig(BaseConfig):
    def __init__(self, Kd, D, Br, Bc, Nwarps=8, isQinRegs:bool = False, SharedQKSmem:bool = False) -> None:
        super().__init__(Kd, D, Br, Bc, Nwarps)
        
        self.isQinRegs = isQinRegs or SharedQKSmem
        self.SharedQKSmem = SharedQKSmem

        self.operation = "flash_fwd"
        self.template_dir = os.path.join(os.path.dirname(__file__), "../../../csrc/kernels/attention")

    def __repr__(self) -> str:
        return "Config(Kd={}, D={}, Br={}, Bc={}, Nwarps={}, isQinRegs={}, SharedQKSmem={}".format(self.Kd, self.D, self.Br, self.Bc, self.Nwarps, self.isQinRegs, self.SharedQKSmem)

    def __str__(self) -> str:
        return f"{self.Kd}_{self.D}_{self.Br}_{self.Bc}_{self.Nwarps}_{self.isQinRegs}_{self.SharedQKSmem}"