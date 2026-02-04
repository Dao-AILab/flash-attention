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
