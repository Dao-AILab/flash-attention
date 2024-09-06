from configs.base_config import BaseConfig
from pathlib import Path
import os
import tempfile

class ShapeConfig:
    def __init__(self, Kd, D, is_bf16: bool=False, is_causal: bool=False) -> None:
        self.Kd = Kd
        self.D = D
        self.is_bf16 = is_bf16
        self.is_causal = is_causal


class CodeEmitter:
    def __init__(self, template_dir, output_dir) -> None:
        self.template_dir = template_dir
        self.output_dir = output_dir

        self.profile_api_file_list = [
            "flash_fwd.cu",
            "flash_profile_api.cpp",
        ]
        self.kernel_file_list = [
            "flash_fwd.h",
        ]

    def generate_code(self, shape_config:ShapeConfig, configs:list[BaseConfig]):
        template_dir = self.template_dir
        output_dir = self.output_dir

        if not Path(output_dir).exists():
            os.mkdir(output_dir)

        # generate api code
        for file_name in self.profile_api_file_list:
            with open(Path(template_dir) / Path(file_name)) as f:
                code_template = f.read()
            code_template = self.emit_code_profile_api(code_template, shape_config)

            with open(Path(output_dir) / Path(file_name), "w") as f:
                f.write(code_template)

        # generate kernel code
        # TODO: parallelize
        for config in configs:
            kernel_code_dir = Path(output_dir) / Path(str(config))
            if not kernel_code_dir.exists():
                os.mkdir(kernel_code_dir)
            for file_name in self.kernel_file_list:
                with open(Path(template_dir) / Path(file_name)) as f:
                    code_template = f.read()
                code_template = self.emit_code_kernel(code_template, config)

                with open(kernel_code_dir / Path(file_name), "w") as f:
                    f.write(code_template)


    def emit_code_kernel(self, code_template:str, config:BaseConfig):
        kv = config.__dict__
        for k,v in kv.items():
            code_template = code_template.replace(f"/*{{{k}}}*/",str(v))
        return code_template
    
    def emit_code_profile_api(self, code_template:str, shape_config: ShapeConfig):
        kv = shape_config.__dict__
        for k,v in kv.items():
            code_template = code_template.replace(f"/*{{{k}}}*/",str(v))
        return code_template

    
if __name__ == "__main__":
    from configs.fwd_config import FlashFwdConfig
    config = FlashFwdConfig(1,2,3,4)
    ce = CodeEmitter("autotuner/template/", "autotuner/template/output/")
    ce.generate_code(ShapeConfig(64,128), [config])
