Examples of how FlashAttention can be integrated into a model (e.g., GPT, ViT)
and trained end-to-end.
We also added optimized implementations of other layers (e.g., MLP, LayerNorm,
cross-entropy loss, rotary embedding).

Goals:
- Performance: we optimize for model speed and memory, especially on 1-node
  (e.g., with 8 A100s).
- Flexibility: we provide optimized building blocks (MLP, attention, LayerNorm),
  and the model code illustrates how these components can be put together.
  The training code also aims to be model- & task-agnostic.

Non-goals (and other resources):
- Support as many models as possible: Huggingface's
  [transformers](https://github.com/huggingface/transformers) and
  [timm](https://github.com/rwightman/pytorch-image-models/) are great for this.
- Large-scale distributed training: our codebase has been used for multi-GPU and multi-node
  training for models up to 2.7B parameters. However, if you're looking for large-scale distributed
  training techniques (e.g., pipeline parallelism, tensor parallelism),
  check out [Megatron-LM](https://github.com/NVIDIA/Megatron-LM/) and
  [DeepSpeed](https://github.com/microsoft/deepspeed).
- Inference: we currently focus on training (this might change in the future).
  If you want fast inference, take a look at
  [FasterTransformer](https://github.com/NVIDIA/FasterTransformer).
- Production: this codebase was written during several research projects to validate ideas
  on speeding up ML models.

## Model Components

The GPT model is implemented
[here](https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/models/gpt.py).

We provide the following optimized components:

- FlashAttention: fast and memory-efficient exact attention. This makes
attention much faster and saves a lot of activation memory. As a result we don't need
to use any activation checkpointing.
```sh
pip install flash-attn
```

- Fused matmul + bias (forward and backward), and fused matmul + bias + gelu
(forward and backward), adapted from Apex's
[FusedDense](https://github.com/NVIDIA/apex/tree/master/apex/fused_dense). We
make it work for bfloat16. For best performance, you should use CUDA >= 11.8. CuBLAS versions before
this doesn't have the best matmul + bias + gelu performance for bfloat16.
```sh
cd ../csrc/fused_dense_lib && pip install .
```
- Optimized cross-entropy loss, adapted from Apex's
[Xentropy](https://github.com/NVIDIA/apex/tree/master/apex/contrib/xentropy). We make it work for bfloat16 and support in-place backward to save memory.
```sh
cd ../csrc/xentropy && pip install .
```
- Fused rotary embedding:
```sh
cd ../csrc/rotary && pip install .
```
- Fused dropout + residual + LayerNorm, adapted from Apex's
[FastLayerNorm](https://github.com/NVIDIA/apex/tree/master/apex/contrib/layer_norm). We add dropout and residual, and make it work for both pre-norm and post-norm architecture.
This only supports a limited set of dimensions, see `csrc/layer_norm/ln_fwd_cuda_kernel.cu`.
```sh
cd ../csrc/layer_norm && pip install .
```

## Training

Feel free to use the model in your training setup. We also provide here training
scripts to train GPT2 on Openwebtext and GPT3 on The Pile as examples.

We use [Hydra](https://hydra.cc/) for configuration,
[Pytorch-Lightning](https://github.com/Lightning-AI/lightning) for training, and
[Wandb](https://wandb.ai/) for logging.

We use the template from `https://github.com/ashleve/lightning-hydra-template`.
Please read the instructions there to understand the repo structure.

### Dataset preparation

Running the training command would automatically download the datasets
(Openwebtext, Pile), tokenize with the GPT2 tokenizer, concatenate all the
tokens, then save this cache to disk. Alternatively, you can also prepare the
datasets as a separate steps.

The cached datasets are saved to `${DATA_DIR}/openwebtext` and
`${DATA_DIR}/the_pile`. If `${DATA_DIR}` is not set, they will be saved to
`./data/{openwebtext,the_pile}`. 

- Openwebtext:
```sh
export PYTHONPATH=$PWD:$PYTHONPATH
pytest -q -s tests/datamodules/test_language_modeling_hf.py -k "openwebtext"
```
This takes around 1h on a 64-core CPU. The processed dataset has size 17GB.

- The Pile:
```sh
export PYTHONPATH=$PWD:$PYTHONPATH
pytest -q -s tests/datamodules/test_language_modeling_hf.py -k "pile"
```
This takes around 20h on a 96-core CPU. The processed dataset has size 699GB.

### GPT2 training on Openwebtext
To train GPT2 on Openwebtext with 8 GPUs:
```sh
python run.py experiment=owt/gpt2s-flash trainer.devices=8
python run.py experiment=owt/gpt2m-flash trainer.devices=8
python run.py experiment=owt/gpt2l-flash trainer.devices=8
python run.py experiment=owt/gpt2xl-flash trainer.devices=8
```
The default parameters are set for 8 x A100 80GB.

To train with bf16 instead of fp16, add `trainer.precision=bf16`.
To adjust device batch size to fit GPU memory (the global batch size stays the
same, and gradient accumulation is calculated automatically), set `datamodule.batch_size=blah`.

### GPT3 training on The Pile
To train GPT3 on The Pile with 8 GPUs:
```sh
python run.py experiment=pile/gpt3s-flash trainer.devices=8
python run.py experiment=pile/gpt3m-flash trainer.devices=8
python run.py experiment=pile/gpt3l-flash trainer.devices=8
python run.py experiment=pile/gpt3xl-flash trainer.devices=8
```
The default parameters are set for 8 x A100 80GB.

## Requirements

Python 3.8+, Pytorch 1.12+, torchvision, einops, timm, hydra-core,
hydra-colorlog, python-dotenv, rich, pytorch-lightning, triton, flash-attn.
We recommend CUDA 11.8 (e.g., using the Nvidia's Pytorch Docker image from https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)

We provide a Dockerfile that lists all the required packages.
