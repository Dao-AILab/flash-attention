# FlashAttention for vLLM

This is a fork of https://github.com/Dao-AILab/flash-attention customized for vLLM.

We have the following customizations:

- Build: Cmake, torch library (this package is bundled into vLLM).
- Size: reduced templating and removal of (training) kernels
- Features: Small page size support (FA2), DCP support (FA3)
- Performance: Some decode specific optimizations for sizes we care about; as well as mixed batch performance optimizations. (Upstream is understandably hesitant on specializing for inference as they also need to support training; we on the other hand compile out the backward pass kernels and do not test that our optimizations do not break them.)
