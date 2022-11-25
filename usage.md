# FlashAttention adoption

We've been very happy to see FlashAttention being adopted by many organizations
and research labs to speed up their training / inference (within 6 months after
FlashAttention's release, at the time of writing).
This page contains a partial list of places where FlashAttention is being used.
If you'd like to add links to your organization / product / codebase, please open a
PR or email us. We'd very much like to hear from you!

## Integrated into machine learning frameworks

- Pytorch: [integrated](https://github.com/pytorch/pytorch/pull/81434) into core Pytorch in nn.Transformer.

- Huggingface's [transformers](https://github.com/huggingface/transformers) library.
  [On-going](https://github.com/huggingface/transformers/pull/18439), blogpost
  coming soon.

- Microsoft's [DeepSpeed](https://github.com/microsoft/DeepSpeed):
  FlashAttention is [integrated](https://github.com/microsoft/DeepSpeed/blob/ec13da6ba7cabc44bb4745a64a208b8580792954/deepspeed/ops/transformer/inference/triton_ops.py) into DeepSpeed's inference engine.

- MosaicML [Composer](https://github.com/mosaicml/composer)
  [library](https://www.mosaicml.com/blog/gpt-3-quality-for-500k). Composer is a
  library for efficient neural network training.

## MLPerf benchmarks

[MLPerf](https://mlcommons.org/en/) is a competitive machine learning performance benchmark. FlashAttention
yields the fastest BERT training on cloud instances in MLPerf training 2.0 (June
2022) and MLPerf training 2.1 (November 2022).

- MLPerf 2.0: IEEE Spectrum [article](https://spectrum.ieee.org/mlperf-rankings-2022) about our submission to the MLPerf 2.0 benchmark using FlashAttention.

- MLPerf 2.1 -
  collaboration
  between [Azure and Hazy Research](https://techcommunity.microsoft.com/t5/azure-high-performance-computing/azure-collaborates-with-hazy-research-and-nvidia-to-achieve/ba-p/3667511): for the first time, we can train MLPerf BERT
  in under 2 minutes on 16 nodes.

- MLPerf 2.1 -
  [Nvidia](https://developer.nvidia.com/blog/leading-mlperf-training-2-1-with-full-stack-optimizations-for-ai/):
  Nvidia uses techniques from FlashAttention to make their (already extremely optimized) BERT
  implementation go even faster.

- MLPerf 2.1 - [MosaicML](https://www.mosaicml.com/blog/mlperf-nlp-nov2022): FlashAttention
  helps train BERT 2.7x faster in the open division.

## Language model training & inference

- Meta's
  [AITemplate](https://ai.facebook.com/blog/gpu-inference-engine-nvidia-amd-open-source/)
  uses FlashAttention as part of their approach to speed up Transformer
  inference (up to 5.3x on BERT).

- [Kernl](https://github.com/ELS-RD/kernl) is a library for fast Transformer
  inference. They use FlashAttention as part of their
  [approach](https://twitter.com/pommedeterre33/status/1585284221014245377) to
  speed up Transformers by up to 12x.

## Diffusion model training and inference

- Huggingface's [diffusers](https://github.com/huggingface/diffusers) library
  for diffusion models. FlashAttention is integrated into [diffusers
  v0.7.0](https://github.com/huggingface/diffusers/releases/tag/v0.7.0).
  Up to 2x faster inference and lower memory usage.

- Colossal-AI's
  [implementation](https://github.com/hpcaitech/ColossalAI/tree/main/examples/images/diffusion)
  of Stable Diffusion: with FlashAttention as one of its components, it speeds up
  pretraining by up to 6.5x, and reduces the hardware cost of fine-tuning by 7x.

- Meta's
  [AITemplate](https://ai.facebook.com/blog/gpu-inference-engine-nvidia-amd-open-source/)
  with FlashAttention one of the components, is currently the [fastest](https://twitter.com/bing_xu_/status/1590447334055632897) Stable
  Diffusion inference engine that we know of.

- Stable Diffusion inference from
  [Labml.ai](https://twitter.com/labmlai/status/1573634095732490240): 50% speedup.

- Our own Stable Diffusion [fork](https://twitter.com/realDanFu/status/1580641495991754752) uses FlashAttention to get 3-4x speedup compared
  to the original version.

## Other models

- [Uni-Fold](https://github.com/dptech-corp/Uni-Fold): Uni-Fold is an
  open-source platform for developing protein models beyond AlphaFold. With
  FlashAttention, Uni-Fold is 2.6x
  [faster](https://twitter.com/guolin_ke/status/1580532071901995008) than AlphaFold.

- [OpenFold](https://github.com/aqlaboratory/openfold): a trainable,
  memory-efficient, and GPU-friendly PyTorch reproduction of AlphaFold 2. With
  FlashAttention as one of its
  [components](https://twitter.com/gahdritz/status/1595420944880779266), it is
  up to 3x faster than AlphaFold2 to run inference on short sequences, and can
  predict 2x longer structures.

## Different implementations

- [Triton](https://github.com/openai/triton): an [implementation](https://github.com/openai/triton/blob/master/python/tutorials/06-fused-attention.py) of
  FlashAttention in Triton by Phil Tillet from OpenAI. Triton is a Python-based
  language and compiler for parallel programming.

- [xformers](https://github.com/facebookresearch/xformers): The xformers team
  has implemented [memory-efficient
  attention](https://twitter.com/fvsmassa/status/1580229170629849089) in a
  similar spirit to FlashAttention.
  xformers dynamically dispatches to whichever implementation is available / faster.

- [Jax](https://github.com/google/jax): an [implementation](https://github.com/lucidrains/flash-attention-jax)
  in Jax by [lucidrains](https://github.com/lucidrains/).
