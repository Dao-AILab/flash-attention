## FlashAttention - Alpha release (0.1).

To compile (requiring NVCC and an A100 GPU):
```
cd csrc/flash_attn
python setup.py install
```

Interface: `flash_attention.py`

To run the benchmark against PyTorch standard attention: 
```
PYTHONPATH=$PWD python benchmarks/benchmark_flash_attention.py
```

FlashAttention currently supports:
1. A100 GPUs.
2. fp16.
3. Head dimensions 16, 32, 64.

Our roadmap to broaden the support:
1. Refactor to use Cutlass.
2. Support SM86 GPUs (e.g. RTX 3080, 3090), support SM75 GPUs (e.g. T4).
3. Support bf16.
4. Support head dimension 128.
5. Make package pip-installable.
6. Support SM70 GPUs (V100).
7. Fused rotary embedding.
8. Attention linear bias (e.g. ALiBi).


### Acknowledgments
Our implementation uses Apex's
[FMHA](https://github.com/NVIDIA/apex/tree/master/apex/contrib/csrc/fmha) code
as a starting point.

We thank [Young-Jun Ko](https://yjk21.github.io/) for the in-depth explanation of his FMHA implementation
and for his thoughtful answers to our questions about CUDA.
