# vLLM Serving Benchmark Results — V100 (SM70)

## Setup
- **GPU**: NVIDIA V100 32GB
- **Model**: NousResearch/Llama-2-7b-hf (FP16)
- **Framework**: vLLM v0.6.5
- **PyTorch**: 2.6.0a0+df5bbc0 (NGC 24.11)
- **CUDA**: 12.6
- **Settings**: `--max-model-len 4096`, `--gpu-memory-utilization 0.9`, `--enforce-eager`
- **Flash Attention**: SM70-compatible build with `--block-size 256` (required for paged KV cache)
- **xFormers**: v0.0.29.post2 (CUTLASS backend)
- **Requests per scenario**: 32

## Summary

| Metric | FLASH_ATTN (SM70) | XFORMERS | XFORMERS Speedup |
|--------|-------------------|----------|-------------------|
| Avg TTFT | 1200ms | 862ms | 1.39x |
| Avg TPOT | 57.1ms | 32.9ms | 1.74x |
| Avg Throughput | 91.3 tok/s | 155.7 tok/s | 1.71x |

## Detailed Results

### FLASH_ATTN (SM70)

| Workload | Conc | OK | TTFT avg(ms) | TTFT p99(ms) | TPOT(ms) | Tput(tok/s) | Req/s |
|----------|------|----|--------------|--------------|----------|-------------|-------|
| short (128/128) | 1 | 32/32 | 62 | 73 | 27.1 | 36.5 | 0.29 |
| short (128/128) | 2 | 32/32 | 90 | 123 | 29.1 | 67.5 | 0.53 |
| short (128/128) | 4 | 32/32 | 232 | 1223 | 33.7 | 113.5 | 0.89 |
| short (128/128) | 8 | 32/32 | 209 | 250 | 41.4 | 187.1 | 1.46 |
| short (128/128) | 16 | 32/32 | 1042 | 1847 | 56.5 | 249.3 | 1.95 |
| medium (512/256) | 1 | 32/32 | 114 | 116 | 30.0 | 33.0 | 0.13 |
| medium (512/256) | 2 | 32/32 | 171 | 229 | 32.7 | 60.1 | 0.23 |
| medium (512/256) | 4 | 32/32 | 336 | 410 | 40.7 | 95.5 | 0.37 |
| medium (512/256) | 8 | 32/32 | 697 | 784 | 56.4 | 135.8 | 0.53 |
| medium (512/256) | 16 | 32/32 | 1299 | 1994 | 82.7 | 182.9 | 0.71 |
| long (1024/512) | 1 | 32/32 | 220 | 224 | 35.3 | 28.0 | 0.05 |
| long (1024/512) | 2 | 32/32 | 329 | 441 | 38.0 | 51.8 | 0.10 |
| long (1024/512) | 4 | 32/32 | 678 | 836 | 50.1 | 77.9 | 0.15 |
| long (1024/512) | 8 | 32/32 | 1098 | 1670 | 75.2 | 103.7 | 0.20 |
| long (1024/512) | 16 | 32/32 | 2170 | 3721 | 114.7 | 134.7 | 0.26 |
| very_long (2048/256) | 1 | 32/32 | 459 | 462 | 42.9 | 22.4 | 0.09 |
| very_long (2048/256) | 2 | 32/32 | 689 | 924 | 46.3 | 41.0 | 0.16 |
| very_long (2048/256) | 4 | 32/32 | 1147 | 1840 | 66.5 | 56.5 | 0.22 |
| very_long (2048/256) | 8 | 32/32 | 2062 | 3674 | 107.6 | 69.4 | 0.27 |
| very_long (2048/256) | 16 | 32/32 | 10885 | 41150 | 134.1 | 78.9 | 0.31 |

### XFORMERS (CUTLASS)

| Workload | Conc | OK | TTFT avg(ms) | TTFT p99(ms) | TPOT(ms) | Tput(tok/s) | Req/s |
|----------|------|----|--------------|--------------|----------|-------------|-------|
| short (128/128) | 1 | 32/32 | 60 | 73 | 25.3 | 39.2 | 0.31 |
| short (128/128) | 2 | 32/32 | 90 | 120 | 28.0 | 70.2 | 0.55 |
| short (128/128) | 4 | 32/32 | 131 | 163 | 28.6 | 136.0 | 1.06 |
| short (128/128) | 8 | 32/32 | 218 | 249 | 29.5 | 257.9 | 2.01 |
| short (128/128) | 16 | 32/32 | 401 | 455 | 31.8 | 460.4 | 3.60 |
| medium (512/256) | 1 | 32/32 | 110 | 112 | 28.2 | 35.0 | 0.14 |
| medium (512/256) | 2 | 32/32 | 165 | 222 | 29.0 | 67.7 | 0.26 |
| medium (512/256) | 4 | 32/32 | 329 | 402 | 29.3 | 131.1 | 0.51 |
| medium (512/256) | 8 | 32/32 | 1182 | 3039 | 31.6 | 221.8 | 0.87 |
| medium (512/256) | 16 | 32/32 | 1072 | 1540 | 37.8 | 382.5 | 1.49 |
| long (1024/512) | 1 | 32/32 | 209 | 211 | 27.6 | 35.8 | 0.07 |
| long (1024/512) | 2 | 32/32 | 314 | 421 | 28.5 | 68.8 | 0.13 |
| long (1024/512) | 4 | 32/32 | 647 | 798 | 29.5 | 130.1 | 0.25 |
| long (1024/512) | 8 | 32/32 | 1131 | 1989 | 34.7 | 217.0 | 0.42 |
| long (1024/512) | 16 | 32/32 | 1859 | 3130 | 45.0 | 329.3 | 0.64 |
| very_long (2048/256) | 1 | 32/32 | 419 | 421 | 27.6 | 34.3 | 0.13 |
| very_long (2048/256) | 2 | 32/32 | 629 | 839 | 29.3 | 63.2 | 0.25 |
| very_long (2048/256) | 4 | 32/32 | 1048 | 1671 | 33.9 | 105.6 | 0.41 |
| very_long (2048/256) | 8 | 32/32 | 1882 | 3342 | 44.6 | 154.6 | 0.60 |
| very_long (2048/256) | 16 | 32/32 | 5342 | 19279 | 57.6 | 173.8 | 0.68 |

## Analysis

### Why XFORMERS is faster in vLLM serving on V100

1. **KV Cache Block Size**: Flash Attention SM70 requires `block_size=256` for paged KV cache
   (the page table dimension must be divisible by the attention block size). XFORMERS uses
   `block_size=16` (default). The 16x difference means FLASH_ATTN wastes significantly more
   memory on partially-filled KV cache blocks, reducing the effective batch size for continuous
   batching.

2. **Memory Efficiency**: With `block_size=256`, each KV cache block allocates memory for 256
   tokens even if only 1 token is stored. This severely limits the number of concurrent
   sequences that can fit in GPU memory, which explains the disproportionate throughput gap
   at high concurrency.

3. **Raw Kernel Speed**: The earlier microbenchmark (`benchmarks/v100_comparison/`) showed
   flash_attn and xFormers have comparable raw attention kernel speed. The serving-level gap
   is primarily a system-level issue from KV cache inefficiency, not kernel performance.

### Key Takeaway

While flash-attn SM70 produces correct results and works end-to-end in vLLM serving, the
`block_size=256` constraint makes it less practical for production serving compared to
xFormers on V100. The primary value of flash-attn SM70 support is for training workloads
(where paged KV cache is not used) and for applications that don't require high-concurrency
serving.
