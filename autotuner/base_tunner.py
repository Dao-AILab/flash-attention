import ctypes
import os
from concurrent.futures import ThreadPoolExecutor
# import multiprocessing
# from functools import partial
import tempfile
import subprocess
import importlib.util

import ctypes
import torch
from configs import BaseConfig, supported_configs

import pprint
import json

import time

from code_emitter import CodeEmitter, ShapeConfig, ProfileConfig
from profile_attn import profile_fwd




        
class CompileResult:
    def __init__(self, config: BaseConfig, lib_name: str) -> None:
        self.config = config
        self.lib_name = lib_name

def _create_code_for_profiling(config):
    profile_code_path = os.path.join(config.template_dir , config.operation, "profile_code.py")
    
    spec = importlib.util.spec_from_file_location("ProfileCode", profile_code_path)
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    # from template.flash_kernels.retnet.regfuse.profile_code import profile_code
    # return profile_code.format(Br=config.Br, Bc=config.Bc, Kd=config.Kd, D=config.D, unrollLastIter=int(config.unrollLastIter), BlockKSmem=config.BlockKSmem, num_stages_qk=config.num_stages_qk, num_stages_mask=config.num_stages_mask, BlockKSmem2=config.BlockKSmem2, num_stages_v=config.num_stages_v, Nthreads=config.Nthreads)
    # from template.flash_kernels.retnet.smemfuse.profile_code import profile_code
    # return profile_code.format(Br=config.Br, Bc=config.Bc, Kd=config.Kd, D=config.D, unrollLastIter=int(config.unrollLastIter), BlockKSmem=config.BlockKSmem, num_stages_qk=config.num_stages_qk, num_stages_mask=config.num_stages_mask, BlockKSmem2=config.BlockKSmem2, num_stages_v=config.num_stages_v, Nthreads=config.Nthreads, warps_mma1_n=config.warps_mma1_n, warps_mma_n=config.warps_mma_n)
    return foo.profile_code.format_map(config.__dict__)

# def _compile(config, arch, temp_dir:str, timeout: float = None):
#     ## compile

#     profiling_code = _create_code_for_profiling(config)
#     src = tempfile.NamedTemporaryFile(mode="w",suffix=".cu", delete=True, dir=temp_dir)
#     lib_name = src.name.replace(".cu", ".so")
#     compute_version = arch.compute_capability
#     cutlass_dir = os.path.join(os.path.dirname(__file__), "../../third_party/cutlass/include")
#     csrc_dir = os.path.join(os.path.dirname(__file__), "../../csrc")    
#     if config.fuse_type == "register":
#         template_dir = os.path.join(config.template_dir , "regfuse/")
#     elif config.fuse_type == "shared":
#         template_dir = os.path.join(config.template_dir , "smemfuse/")
#     else: # bwd
#         template_dir = config.template_dir
#     command = ["nvcc","-std=c++17","-O3","--use_fast_math","--expt-relaxed-constexpr","--disable-warnings", "--compiler-options", "'-fPIC'", "--shared", src.name, "-lcuda",
#             f"-gencode=arch=compute_{compute_version},code=sm_{compute_version}",
#             f"-I{cutlass_dir}",f"-I{template_dir}",f"-I{csrc_dir}", "-o", lib_name]
#     src.write(profiling_code)
#     src.flush()
#     try:
#         ret = subprocess.run(command, timeout=timeout)
#     except subprocess.TimeoutExpired:
#         return None
#     if ret.returncode != 0:
#         return None
#     return CompileResult(config,lib_name)

class BaseTunner:
    def __init__(self, arch, torch_array: list, op_name, shape_config: ShapeConfig, profile_config: ProfileConfig, tempdir):
        self.arch = arch
        self.torch_array = torch_array
        self.Br_list = [32, 64, 96, 128, 160, 192, 224, 256] # [32, 64, 128, 256]
        self.Bc_list = [32, 64, 96, 128, 160, 192, 224, 256] # [32, 64, 128, 256]

        self.template_dir = "autotuner/template"
        self.op_name = op_name
        # TODO: workaround for dropout_p
        self.cache_path = os.path.join(os.path.dirname(__file__), "./cache/", str(profile_config.dropout_p!=0))
        self.problem_key = {
            "dim_qk": torch_array[0].shape[-1],
            "dim_v": torch_array[2].shape[-1]
        }
        assert torch_array[0].shape[-1] == shape_config.Kd
        assert torch_array[2].shape[-1] == shape_config.D
        self.shape_config = shape_config
        self.profile_config = profile_config
        self.tempdir = tempdir

    def compile(self, configs:list, timeout: float = None):
        temp_dir = self.tempdir
        code_emitter = CodeEmitter(self.template_dir, temp_dir)
        code_emitter.generate_code(self.shape_config, configs)

    
    def profile(self, config:BaseConfig, repeat=30, load_only=False) -> float:
        spec = importlib.util.spec_from_file_location("flash_attn_func", self.tempdir+"/"+config.output_dir+"/flash_attn_profile_interface.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        flash_attn_func = mod.flash_attn_func
        if load_only:
            return None
        latency = profile_fwd(flash_attn_func, self.shape_config.Kd, self.shape_config.D, batch_size=self.profile_config.batch_size, seqlen=self.profile_config.seqlen_q, nheads=self.profile_config.nheads, dropout_p=self.profile_config.dropout_p,is_bf16=self.shape_config.is_bf16, causal=self.shape_config.is_causal, device=self.profile_config.device, repeats=repeat)
        if latency < 0:
            latency = 1e8
        # remove lib
        # subprocess.run(["rm", libname], check=True)
        return latency
    
    def get_tuned_configs(self):
        dim_qk = self.problem_key["dim_qk"]
        dim_v = self.problem_key["dim_v"]
        configs = []
        for Br in self.Br_list:
            for Bc in self.Bc_list:
                cur_configs = self.generate_configs(Br,Bc,dim_qk,dim_v)
                for cur_config in cur_configs:
                    if self.op_name  == "flash_fwd" and self.validate_register_fuse(cur_config):
                        configs.append(cur_config)
                    else: # BWD
                        if self.validate_kernel(cur_config):
                            configs.append(cur_config)
        return configs

    def tune(self, log_path="./logs/"):
        st = time.time()

        dim_qk = self.problem_key["dim_qk"]
        dim_v = self.problem_key["dim_v"]

        best_config = self.check_cache()
        if best_config is not None:
            # print("Best config found in cache: ")
            # pprint.pprint(best_config)
            return best_config

        configs = self.get_tuned_configs()

        # print configs
        print("Configs to be tuned: ")
        for config in configs:
            # print(config)
            pprint.pprint(config)


        # cresults = self.compile(configs,src_dir.name,timeout=1200)
        # cresults = self.compile_parallel(configs,src_dir.name,timeout=120)
        self.compile(configs,timeout=120)

        # warm up (parallel compile module)
        # module name must be different in api.py
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            latencys = executor.map(self.profile, configs, [1 for _ in range(len(configs))], [True for _ in range(len(configs))])
        # with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        #     latencys = executor.map(_profile,[self.tempdir for _ in range(len(configs))],[self.shape_config for _ in range(len(configs))], configs, ["cuda:0" for _ in range(len(configs))], [1 for _ in range(len(configs))])
        # multiprocessing.set_start_method('spawn', force=True)
        # pool = multiprocessing.Pool(os.cpu_count())
        # outs = pool.map(partial(self.profile, repeat=1), configs)

        profile_dict = {}
        latency = 1e8
        best_config = None
        for config in configs:
            lib_latency = self.profile(config)
            if lib_latency == 1e8:
                # print(cresult.config)
                pprint.pprint(config)
                print("profile runtime error")
            if lib_latency < latency:
                latency = lib_latency
                best_config = config
            profile_dict[config] = lib_latency

        end = time.time()

        print("##########################################################")
        print("Operation type: ", best_config.operation)
        print("Best config: ")# , best_config)
        pprint.pprint(best_config)
        print("Latency: ", latency)

        file_name = "profile_result_{}_{}_{}_p{}_{}_{}_{}_c{}.txt".format(best_config.operation,dim_qk, dim_v, self.profile_config.batch_size, self.profile_config.seqlen_q, self.profile_config.nheads, self.profile_config.dropout_p,self.shape_config.is_causal)
        os.makedirs(log_path,exist_ok=True)
        with open(os.path.join(log_path,file_name),"a") as f:
            for config in profile_dict:
                f.write(repr(config)+"\n")
                f.write(str(profile_dict[config])+"\n")
            f.write("\n")
            f.write("best config: \n")
            f.write(repr(best_config)+"\n")
            f.write(str(latency)+"\n")
            f.write("\nsearch time: "+str(end-st)+"s" + "\n\n")

        cache_path = self.cache_path
        os.makedirs(cache_path,exist_ok=True)
        with open(os.path.join(cache_path,"best_config_{}_{}_{}.json".format(self.op_name,dim_qk, dim_v)),"w") as f:
            json.dump(best_config.__dict__,f)

        return best_config
    
    def check_cache(self):
        cache_path = self.cache_path
        op_name = self.op_name
        dim_qk = self.problem_key["dim_qk"]
        dim_v = self.problem_key["dim_v"]
        if os.path.exists(os.path.join(cache_path, "best_config_{}_{}_{}.json".format(op_name,dim_qk, dim_v))):
            with open(os.path.join(cache_path,"best_config_{}_{}_{}.json".format(op_name,dim_qk, dim_v)),"r") as f:
                best_config_dict = json.load(f)
            best_config = supported_configs[best_config_dict["operation"]].from_dict(best_config_dict)
            return best_config
        
        return None
            
        
    def validate_shared_fuse(self, config):
        return False
    def validate_register_fuse(self, config):
        return False
    def validate_kernel(self, config):
        return False
    def generate_configs(self,Br:int,Bc:int,dim_qk:int,dim_v:int):
        configs = []
        return configs
    
if __name__=="__main__":
    import torch
    from configs.fwd_config import FlashFwdConfig
    batch_size = 4
    seqlen = 2048
    nheads = 8
    headdim = 192
    v_headdim = 128
    device = 'cuda'
    dtype = torch.bfloat16
    q = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype,
                              requires_grad=True)
    k = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype,
                                requires_grad=True)
    v = torch.randn(batch_size, seqlen, nheads, v_headdim, device=device, dtype=dtype,
                                requires_grad=True)
    base_tunner = BaseTunner(arch=None, torch_array=[q,k,v], op_name="flash_fwd", shape_config=ShapeConfig(headdim,v_headdim), profle_config=ProfileConfig(batch_size,seqlen,seqlen,nheads,nheads,nheads,device,dtype,0), tempdir="autotuner/temp")

    config = FlashFwdConfig(headdim,v_headdim,64,64)
    base_tunner.compile([config])
    base_tunner.profile(config)