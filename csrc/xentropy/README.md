This CUDA extension implements optimized cross-entropy loss, adapted from Apex's
[Xentropy](https://github.com/NVIDIA/apex/tree/master/apex/contrib/xentropy).
We make it work for bfloat16 and support in-place backward to save memory.

It has only been tested on A100s.

```sh
cd csrc/xentropy && pip install .
```

As of 2023-09-15, this extension is no longer used in the FlashAttention repo.
We've instead switched to a Triton-based
[implementation](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/ops/triton/cross_entropy.py). 
See the CrossEntropyLoss [module](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/losses/cross_entropy.py) for more details.
