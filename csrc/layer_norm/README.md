This CUDA extension implements fused dropout + residual + LayerNorm, building on
Apex's [FastLayerNorm](https://github.com/NVIDIA/apex/tree/master/apex/contrib/layer_norm).
We add dropout and residual, and make it work for both pre-norm and post-norm architecture.
We also make it work for more hidden dimensions (all dimensions divisible by 8, up to 6144).

If you want to use it for dimensions larger than 6k, please file an issue.

This extension has only been tested on A100s.

```sh
cd csrc/layer_norm && pip install .
```
