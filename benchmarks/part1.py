#!/usr/bin/env python3
import torch, math, subprocess, json, os, re
from datetime import datetime
try:
    from flash_attn import flash_attn_qkvpacked_func
    HAS_FLASH = True
except:
    HAS_FLASH = False
WALLER_BINARY = os.path.expanduser("~/waller-eval-repo/waller_eval_cli_x86")
def get_gpu_info():
    return {"name": torch.cuda.get_device_name(0), "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3)}
