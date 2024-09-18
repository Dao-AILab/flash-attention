class BaseConfig:
    def __init__(self, Kd, D, Br, Bc, Nwarps=8) -> None:
        self.Br = Br
        self.Bc = Bc
        self.Kd = Kd
        self.D = D
        self.Nwarps = Nwarps

        self.operation = None
        self.template_dir = None

    def __repr__(self) -> str:
        return "Config(Kd={}, D={}, Br={}, Bc={}, Nwarps={})".format(self.Kd, self.D, self.Br, self.Bc, self.Nwarps)

    def __str__(self) -> str:
        return f"{self.Kd}_{self.D}_{self.Br}_{self.Bc}_{self.Nwarps}"
        
    @classmethod
    def from_dict(cls, dd:dict):
        cc = cls.__new__(cls) # cls: 子类
        cc.__dict__.update(dd)
        return cc
    
    @property
    def output_dir(self):
        return str(self)
    
if __name__ == "__main__":
    cc = BaseConfig(1,2,3,4)
    print(cc)
    print(repr(cc))
    print(cc.__dict__)
    dd = cc.__dict__
    cc2 = BaseConfig.from_dict(dd)
    print(cc2)
    print(repr(cc2))
    print(cc2.__dict__)