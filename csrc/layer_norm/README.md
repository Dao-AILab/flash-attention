This CUDA extension implements fused dropout + residual + LayerNorm, based on
Apex's [FastLayerNorm](https://github.com/NVIDIA/apex/tree/master/apex/contrib/layer_norm).
We add dropout and residual, and make it work for both pre-norm and post-norm architecture.

This only supports a limited set of dimensions, see `csrc/layer_norm/ln_fwd_cuda_kernel.cu`.

It has only been tested on A100s.

```sh
cd csrc/layer_norm && pip install .
```
