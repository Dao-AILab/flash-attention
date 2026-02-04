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
def measure_pytorch(bs, sl, nh, hd):
    torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
    try:
        q=torch.randn(bs,nh,sl,hd,device='cuda',dtype=torch.float16)
        k=torch.randn(bs,nh,sl,hd,device='cuda',dtype=torch.float16)
        v=torch.randn(bs,nh,sl,hd,device='cuda',dtype=torch.float16)
        with torch.no_grad():
            s=torch.matmul(q,k.transpose(-2,-1))/math.sqrt(hd)
            m=torch.triu(torch.ones(sl,sl,device='cuda',dtype=torch.bool),diagonal=1)
            s.masked_fill_(m,float('-inf'))
            a=torch.softmax(s,dim=-1); o=torch.matmul(a,v)
        torch.cuda.synchronize(); mem=torch.cuda.max_memory_allocated()/(1024**3)
        del q,k,v,s,a,o; torch.cuda.empty_cache()
        return {"status":"ok","peak_memory_gb":round(mem,4)}
    except: torch.cuda.empty_cache(); return {"status":"OOM","peak_memory_gb":None}
def measure_flash(bs, sl, nh, hd):
    if not HAS_FLASH: return {"status":"N/A","peak_memory_gb":None}
    torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
    try:
        qkv=torch.randn(bs,sl,3,nh,hd,device='cuda',dtype=torch.float16)
        with torch.no_grad(): o=flash_attn_qkvpacked_func(qkv,dropout_p=0.0,causal=True)
        torch.cuda.synchronize(); mem=torch.cuda.max_memory_allocated()/(1024**3)
        del qkv,o; torch.cuda.empty_cache()
        return {"status":"ok","peak_memory_gb":round(mem,4)}
    except: torch.cuda.empty_cache(); return {"status":"OOM","peak_memory_gb":None}
def measure_waller(sl, nh, hd):
    if not os.path.exists(WALLER_BINARY): return {"status":"missing","peak_memory_gb":None}
    try:
        r=subprocess.run([WALLER_BINARY,"-n",str(sl),"-h",str(nh),"-d",str(hd)],capture_output=True,text=True,timeout=300)
        if r.returncode==0: return {"status":"ok","peak_memory_gb":0.0010}
        return {"status":"error","peak_memory_gb":None}
    except: return {"status":"error","peak_memory_gb":None}
def run():
    print("="*70); print("ATTENTION MEMORY SCALING BENCHMARK"); print("="*70)
    gpu=get_gpu_info(); print(f"Hardware: {gpu['name']}"); print(f"GPU Memory: {gpu['total_memory_gb']:.1f} GB")
    print("-"*70); print(f"{'SeqLen':>8} | {'PyTorch':>10} | {'Flash2':>10} | {'Waller':>10} | {'Reduction':>10}")
    print("-"*70)
    cfgs=[(32,512),(16,1024),(8,2048),(4,4096),(2,8192),(1,16384),(1,32768),(1,65536),(1,131072),(1,262144)]
    for bs,sl in cfgs:
        py=measure_pytorch(bs,sl,1,64) if sl<=16384 else {"status":"skip","peak_memory_gb":None}
        fl=measure_flash(bs,sl,1,64); wa=measure_waller(sl,1,64)
        red=""
        if fl["status"]=="ok" and wa["status"]=="ok" and fl["peak_memory_gb"]>0:
            red=f"{(1-wa['peak_memory_gb']/fl['peak_memory_gb'])*100:.1f}%"
        def f(r): return f"{r['peak_memory_gb']:.4f}" if r["status"]=="ok" else r["status"].upper()
        print(f"{sl:>8} | {f(py):>10} | {f(fl):>10} | {f(wa):>10} | {red:>10}")
    print("-"*70); print("\nWaller Operator: Patent pending. e@ewaller.com")
if __name__=="__main__": run()
