This CUDA extension implements optimized cross-entropy loss, adapted from Apex's
[Xentropy](https://github.com/NVIDIA/apex/tree/master/apex/contrib/xentropy).
We make it work for bfloat16 and support in-place backward to save memory.

It has only been tested on A100s.

```sh
cd csrc/xentropy && pip install .
```
