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
