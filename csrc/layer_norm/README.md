This CUDA extension implements fused dropout + residual + LayerNorm, building on
Apex's [FastLayerNorm](https://github.com/NVIDIA/apex/tree/master/apex/contrib/layer_norm).
Major changes:
- Add dropout and residual.
- Make it work for both pre-norm and post-norm architecture.
- Support more hidden dimensions (all dimensions divisible by 8, up to 8192).
- Implement RMSNorm as an option.
- Support layer norm with parallel residual (e.g., GPT-J, GPT-NeoX, PaLM).

If you want to use it for dimensions larger than 8k, please file an issue.

This extension has only been tested on A100s.

```sh
cd csrc/layer_norm && pip install .
```

As of 2024-01-05, this extension is no longer used in the FlashAttention repo.
We've instead switched to a Triton-based
[implementation](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/ops/triton/layer_norm.py).
