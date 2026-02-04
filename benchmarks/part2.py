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
