# Copyright (c) 2023, Tri Dao.

import logging
import math
import re
from collections import OrderedDict, namedtuple
from collections.abc import Sequence
from functools import partial

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F
from transformers import GPT2Config

from flash_attn.models.gpt_neox import remap_state_dict_hf_gpt_neox
from flash_attn.models.gptj import remap_state_dict_hf_gptj
from flash_attn.models.opt import remap_state_dict_hf_opt
from flash_attn.modules.block import Block, ParallelBlock
from flash_attn.modules.embedding import GPT2Embeddings, ParallelGPT2Embeddings
from flash_attn.modules.mha import MHA, ParallelMHA
from flash_attn.modules.mlp import FusedMLP, Mlp, ParallelFusedMLP
from flash_attn.utils.distributed import all_gather_raw, sync_shared_params
from flash_attn.utils.generation import GenerationMixin
from flash_attn.utils.pretrained import state_dict_from_pretrained

try:
    from flash_attn.functional.fused_dense import ColumnParallelLinear
except ImportError:
    ColumnParallelLinear = None

try:
    from flash_attn.functional.layer_norm import dropout_add_layer_norm
except ImportError:
    dropout_add_layer_norm = None

try:
    from flash_attn.functional.layer_norm import dropout_add_layer_norm_parallel_residual
except ImportError:
    dropout_add_layer_norm_parallel_residual = None

try:
    from flash_attn.functional.triton.mlp import FusedDenseSqreluDense, sqrelu_fwd
except ImportError:
    FusedDenseSqreluDense = None
    sqrelu_fwd = None


logger = logging.getLogger(__name__)


def create_mixer_cls(config, layer_idx=None, process_group=None, device=None, dtype=None):
    factory_kwargs = {"device": device, "dtype": dtype}
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    softmax_scale = 1.0 if not config.scale_attn_weights else head_dim ** (-0.5)
    if config.scale_attn_by_inverse_layer_idx:
        assert layer_idx is not None
        softmax_scale /= float(layer_idx + 1)
    dwconv = getattr(config, "attn_dwconv", False)
    if dwconv:
        assert process_group is None, "TensorParallel MHA does not support dwconv yet"
    qkv_proj_bias = getattr(config, "qkv_proj_bias", True)
    out_proj_bias = getattr(config, "out_proj_bias", True)
    rotary_emb_dim = int(getattr(config, "rotary_emb_fraction", 0.0) * head_dim)
    rotary_emb_scale_base = getattr(config, "rotary_emb_scale_base", None)
    rotary_emb_interleaved = getattr(config, "rotary_emb_interleaved", False)
    use_flash_attn = getattr(config, "use_flash_attn", False)
    fused_bias_fc = getattr(config, "fused_bias_fc", False)
    if not fused_bias_fc:
        assert process_group is None, "TensorParallel MHA requires fused_bias_fc"
    mha_cls = MHA if process_group is None else ParallelMHA
    serial_kwargs = {"fused_bias_fc": fused_bias_fc, "dwconv": dwconv} if process_group is None else {}
    parallel_kwargs = (
        {"process_group": process_group, "sequence_parallel": getattr(config, "sequence_parallel", True)}
        if process_group is not None
        else {}
    )
    mixer_cls = partial(
        mha_cls,
        num_heads=config.num_attention_heads,
        qkv_proj_bias=qkv_proj_bias,
        out_proj_bias=out_proj_bias,
        dropout=config.attn_pdrop,
        softmax_scale=softmax_scale,
        causal=True,
        layer_idx=layer_idx,
        rotary_emb_dim=rotary_emb_dim,
        rotary_emb_scale_base=rotary_emb_scale_base,
        rotary_emb_interleaved=rotary_emb_interleaved,
        use_flash_attn=use_flash_attn,
        **serial_kwargs,
        **parallel_kwargs,
        **factory_kwargs,
    )
    return mixer_cls


def create_mlp_cls(config, layer_idx=None, process_group=None, device=None, dtype=None):
    factory_kwargs = {"device": device, "dtype": dtype}
    inner_dim = config.n_inner if config.n_inner is not None else 4 * config.hidden_size
    fused_mlp = getattr(config, "fused_mlp", False)
    if fused_mlp:
        assert config.activation_function in ["gelu_new", "gelu_fast", "gelu_approx", "relu", "sqrelu"]
    fused_dense_sqrelu_dense = getattr(config, "fused_dense_sqrelu_dense", False)
    if fused_dense_sqrelu_dense:
        assert config.activation_function == "sqrelu", (
            "fused_dense_sqrelu_dense only " "supports approximate activation_function sqrelu"
        )
    assert not (fused_dense_sqrelu_dense and fused_mlp)
    if process_group is not None:
        assert fused_mlp, "Tensor Parallel is only implemented for FusedMLP"
    if not fused_mlp and not fused_dense_sqrelu_dense:
        assert config.activation_function in ["gelu_new", "gelu_fast", "gelu_approx", "relu", "sqrelu"]
        if config.activation_function == "relu":
            activation = partial(F.relu, inplace=True)
        elif config.activation_function == "sqrelu":
            assert sqrelu_fwd is not None, "sqrelu_fwd is not implemented"
            activation = sqrelu_fwd
        else:
            approximate = "tanh" if config.activation_function in ["gelu_new", "gelu_fast", "gelu_approx"] else "none"
            activation = partial(F.gelu, approximate=approximate)
        mlp_cls = partial(Mlp, hidden_features=inner_dim, activation=activation, **factory_kwargs)
    else:
        mlp_checkpoint_lvl = getattr(config, "mlp_checkpoint_lvl", 0)
        # mlp_checkpoint_lvl could be a list, which contains the checkpoint_lvl for each layer
        if isinstance(mlp_checkpoint_lvl, Sequence):
            assert layer_idx is not None
            mlp_checkpoint_lvl = mlp_checkpoint_lvl[layer_idx]
        if fused_mlp:
            if FusedMLP is None:
                raise ImportError("fused_dense is not installed")
            activation = (
                "gelu_approx"
                if config.activation_function in ["gelu_new", "gelu_fast", "gelu_approx"]
                else config.activation_function
            )
            mlp_cls = FusedMLP if process_group is None else ParallelFusedMLP
            parallel_kwargs = (
                {"process_group": process_group, "sequence_parallel": getattr(config, "sequence_parallel", True)}
                if process_group is not None
                else {}
            )
            mlp_cls = partial(
                mlp_cls,
                hidden_features=inner_dim,
                activation=activation,
                checkpoint_lvl=mlp_checkpoint_lvl,
                **parallel_kwargs,
                **factory_kwargs,
            )
        elif fused_dense_sqrelu_dense:
            assert FusedDenseSqreluDense is not None
            mlp_cls = partial(
                FusedDenseSqreluDense, hidden_features=inner_dim, checkpoint_lvl=mlp_checkpoint_lvl, **factory_kwargs
            )
        else:
            raise RuntimeError("MLP type not supported")
    return mlp_cls


def create_block(config, layer_idx=None, process_group=None, device=None, dtype=None):
    factory_kwargs = {"device": device, "dtype": dtype}
    sequence_parallel = getattr(config, "sequence_parallel", True)
    mixer_cls = create_mixer_cls(config, layer_idx, process_group=process_group, **factory_kwargs)
    mlp_cls = create_mlp_cls(config, layer_idx, process_group=process_group, **factory_kwargs)
    norm_cls = partial(nn.LayerNorm, eps=config.layer_norm_epsilon, **factory_kwargs)
    # TD [2022-07-30]: Force residual in fp32, seems to make fp16 training more stable
    residual_in_fp32 = getattr(config, "residual_in_fp32", False)
    resid_dropout1 = config.resid_pdrop if layer_idx is None or layer_idx > 0 else config.embd_pdrop
    prenorm = getattr(config, "prenorm", True)
    parallel_block = getattr(config, "parallel_block", False)
    if not parallel_block:
        block = Block(
            config.hidden_size,
            mixer_cls,
            mlp_cls,
            norm_cls=norm_cls,
            prenorm=prenorm,
            resid_dropout1=resid_dropout1,
            resid_dropout2=config.resid_pdrop,
            fused_dropout_add_ln=getattr(config, "fused_dropout_add_ln", False),
            residual_in_fp32=residual_in_fp32,
            sequence_parallel=sequence_parallel and process_group is not None,
            mark_shared_params=process_group is not None,
        )
    else:
        assert prenorm
        block = ParallelBlock(
            config.hidden_size,
            mixer_cls,
            mlp_cls,
            norm_cls=norm_cls,
            resid_dropout1=resid_dropout1,
            resid_dropout2=config.resid_pdrop,
            tied_norm=getattr(config, "parallel_block_tied_norm", False),
            fused_dropout_add_ln=getattr(config, "fused_dropout_add_ln", False),
            residual_in_fp32=residual_in_fp32,
            sequence_parallel=sequence_parallel and process_group is not None,
            mark_shared_params=process_group is not None,
        )
    block.layer_idx = layer_idx
    return block


class GPTPreTrainedModel(nn.Module):
    """An abstract class to handle weights initialization and
    a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, config, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, GPT2Config):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `GPT2Config`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                )
            )
        self.config = config

    @classmethod
    def from_pretrained(
        cls, model_name, config, *args, strict=True, device=None, dtype=None, world_size=1, rank=0, **kwargs
    ):
        """
        Instantiate a GPTPreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.
        """
        # Instantiate model.
        model = cls(config, *args, device=device, dtype=dtype, **kwargs)
        # Load state_dict in cpu because we already initialized the model in GPU, and we don't
        # want extra stuff taking up more GPU memory
        state_dict = state_dict_from_pretrained(model_name, device="cpu", dtype=dtype)
        if model_name.startswith("gpt2"):
            state_dict = remap_state_dict_hf_gpt2(state_dict, config)
        elif model_name.startswith("facebook/opt"):
            state_dict = remap_state_dict_hf_opt(state_dict, config)
        elif model_name.startswith("EleutherAI/gpt-j-"):
            state_dict = remap_state_dict_hf_gptj(state_dict, config)
            strict = False  # We have rotary_emb.inf_freq buffers not in the GPT-J checkpoint
        elif model_name.startswith("EleutherAI/gpt-neox-"):
            state_dict = remap_state_dict_hf_gpt_neox(state_dict, config)
        else:
            raise NotImplementedError(f"Model {model_name} not supported")
        if world_size > 1:
            state_dict = shard_state_dict_tp(state_dict, config, world_size, rank)
        load_return = model.load_state_dict(state_dict, strict=strict)
        logger.info(load_return)
        return model


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(module, n_layer, initializer_range=0.02, rescale_prenorm_residual=True):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=initializer_range)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                nn.init.normal_(p, mean=0.0, std=initializer_range / math.sqrt(2 * n_layer))


class GPTModel(GPTPreTrainedModel):
    def __init__(self, config: GPT2Config, process_group=None, device=None, dtype=None):
        super().__init__(config)
        factory_kwargs = {"device": device, "dtype": dtype}
        self.process_group = process_group
        self.sequence_parallel = getattr(config, "sequence_parallel", True)
        assert config.activation_function in ["gelu", "gelu_new", "gelu_fast", "gelu_approx", "relu", "sqrelu"]
        pad_vocab_size_multiple = getattr(config, "pad_vocab_size_multiple", 1)
        vocab_size = math.ceil(config.vocab_size / pad_vocab_size_multiple) * pad_vocab_size_multiple
        # TD [2022-07-30]: Force residual in fp32, seems to make fp16 training more stable
        self.residual_in_fp32 = getattr(config, "residual_in_fp32", False)
        # These 2 options are for OPT-350m
        self.prenorm = getattr(config, "prenorm", True)
        word_embed_proj_dim = getattr(config, "word_embed_proj_dim", None)
        # For GPT-J, GPT-NeoX
        self.parallel_block = getattr(config, "parallel_block", False)

        if process_group is None:
            self.embeddings = GPT2Embeddings(
                config.hidden_size,
                vocab_size,
                config.max_position_embeddings,
                word_embed_proj_dim=word_embed_proj_dim,
                **factory_kwargs,
            )
        else:
            self.embeddings = ParallelGPT2Embeddings(
                config.hidden_size,
                vocab_size,
                config.max_position_embeddings,
                process_group=process_group,
                sequence_parallel=self.sequence_parallel,
                **factory_kwargs,
            )

        # We change the order of dropout, residual and layer norm:
        # Instead of LN -> Attn / MLP -> Dropout -> Add, we do:
        # Dropout -> Add -> LN -> Attn / MLP, returning both the residual branch (output of Add) and
        # the main branch (output of MLP). The model definition is unchanged, but the mapping of the
        # nn.Dropout probabilities are changed.
        # This is for performance reason: we can fuse dropout + add + layer_norm.
        self.layers = nn.ModuleList(
            [
                create_block(config, layer_idx=i, process_group=process_group, **factory_kwargs)
                for i in range(config.num_hidden_layers)
            ]
        )

        self.fused_dropout_add_ln = getattr(config, "fused_dropout_add_ln", False)
        if self.fused_dropout_add_ln:
            if (not self.parallel_block and dropout_add_layer_norm is None) or (
                self.parallel_block and dropout_add_layer_norm_parallel_residual is None
            ):
                raise ImportError("dropout_layer_norm is not installed")
        if self.prenorm:
            self.drop_f = nn.Dropout(config.resid_pdrop)
            self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon, **factory_kwargs)
        if process_group is not None:
            for p in self.ln_f.parameters():
                # Mark the norm parameters as "shared_params" so that we sync their values at init.
                p._shared_params = True
                # Mark the norm params as "sequence_parallel" so we run all-reduce on their grads.
                if self.sequence_parallel:
                    p._sequence_parallel = True

        self.apply(partial(_init_weights, n_layer=config.num_hidden_layers, initializer_range=config.initializer_range))
        self.tie_weights()

    def tie_weights(self):
        if self.process_group is not None:
            sync_shared_params(self, self.process_group)

    def forward(self, input_ids, position_ids=None, inference_params=None):
        # If using Tensor Parallel with sequence parallel, we combine the batch and the seqlen
        # dimensions so that we can split on it easily, in case of small batch size.
        # Only the attention layers need to know the seqlen.
        embedding_kwargs = (
            {"combine_batch_seqlen_dim": True} if self.process_group is not None and self.sequence_parallel else {}
        )
        hidden_states = self.embeddings(input_ids, position_ids=position_ids, **embedding_kwargs)
        if self.parallel_block:
            hidden_states2 = None
        residual = None
        mixer_kwargs = (
            {"seqlen": input_ids.shape[1]} if self.process_group is not None and self.sequence_parallel else {}
        )
        if inference_params is not None:
            mixer_kwargs["inference_params"] = inference_params
        for layer in self.layers:
            if self.prenorm:
                if not self.parallel_block:
                    hidden_states, residual = layer(hidden_states, residual, mixer_kwargs=mixer_kwargs)
                else:
                    hidden_states, hidden_states2, residual = layer(
                        hidden_states, hidden_states2, residual, mixer_kwargs=mixer_kwargs
                    )
            else:
                hidden_states = layer(hidden_states, mixer_kwargs=mixer_kwargs)
        if self.prenorm:
            if not self.fused_dropout_add_ln:
                dropped = self.drop_f(hidden_states)
                if not self.parallel_block:
                    residual = (dropped + residual) if residual is not None else dropped
                else:
                    dropped2 = self.drop_f(hidden_states2)
                    residual = (residual + dropped + dropped2) if residual is not None else dropped + dropped2
                hidden_states = self.ln_f(residual.to(dtype=self.ln_f.weight.dtype))
            else:
                # Set prenorm=False here since we don't need the residual
                if not self.parallel_block:
                    hidden_states = dropout_add_layer_norm(
                        hidden_states,
                        residual,
                        self.ln_f.weight,
                        self.ln_f.bias,
                        self.drop_f.p if self.training else 0.0,
                        self.ln_f.eps,
                        prenorm=False,
                        residual_in_fp32=self.residual_in_fp32,
                    )
                else:
                    hidden_states, _ = dropout_add_layer_norm_parallel_residual(
                        hidden_states,
                        hidden_states2,
                        residual,
                        self.ln_f.weight,
                        self.ln_f.bias,
                        None,
                        None,
                        self.drop_f.p if self.training else 0.0,
                        self.ln_f.eps,
                        prenorm=False,
                        residual_in_fp32=self.residual_in_fp32,
                    )
        return hidden_states


class GPTLMHeadModel(GPTPreTrainedModel, GenerationMixin):
    def __init__(self, config: GPT2Config, process_group=None, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(config)
        self.process_group = process_group
        self.transformer = GPTModel(config, process_group=process_group, **factory_kwargs)
        self.tie_word_embeddings = getattr(config, "tie_word_embeddings", True)
        lm_head_bias = getattr(config, "lm_head_bias", False)
        pad_vocab_size_multiple = getattr(config, "pad_vocab_size_multiple", 1)
        vocab_size = math.ceil(config.vocab_size / pad_vocab_size_multiple) * pad_vocab_size_multiple
        # This option is for OPT-350m
        word_embed_proj_dim = getattr(config, "word_embed_proj_dim", None)
        embed_dim = config.n_embd if word_embed_proj_dim is None else word_embed_proj_dim
        if word_embed_proj_dim is not None:
            self.project_out = nn.Linear(config.n_embd, embed_dim, bias=False, **factory_kwargs)
        else:
            self.project_out = None
        if process_group is None:
            self.lm_head = nn.Linear(embed_dim, vocab_size, bias=lm_head_bias, **factory_kwargs)
        else:
            if ColumnParallelLinear is None:
                raise ImportError("fused_dense_lib is not installed")
            self.lm_head = ColumnParallelLinear(
                embed_dim,
                vocab_size,
                process_group,
                bias=lm_head_bias,
                sequence_parallel=getattr(config, "sequence_parallel", True),
                **factory_kwargs,
            )
        # Initialize weights and apply final processing
        self.apply(partial(_init_weights, n_layer=config.num_hidden_layers, initializer_range=config.initializer_range))
        self.tie_weights()

    def tie_weights(self):
        if self.tie_word_embeddings:
            self.lm_head.weight = self.transformer.embeddings.word_embeddings.weight
        if self.process_group is not None:
            sync_shared_params(self, self.process_group)

    def forward(self, input_ids, position_ids=None, inference_params=None):
        """
        inference_params: for generation. Adapted from Megatron-LM (and Apex)
        https://github.com/NVIDIA/apex/blob/3ff1a10f72ec07067c4e44759442329804ac5162/apex/transformer/testing/standalone_transformer_lm.py#L470
        """
        hidden_states = self.transformer(input_ids, position_ids=position_ids, inference_params=inference_params)
        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)
        lm_logits = self.lm_head(hidden_states)
        # During inference, we want the full logit for sampling
        if isinstance(self.lm_head, ColumnParallelLinear) and inference_params is not None:
            lm_logits, _ = all_gather_raw(lm_logits, self.lm_head.process_group)
            lm_logits = rearrange(lm_logits, "(n b) s d -> b s (n d)", b=hidden_states.shape[0])
        CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
        return CausalLMOutput(logits=lm_logits)

    def load_state_dict(self, state_dict, strict=True):
        # Remapping from our checkpoints that used a different ordering of layers in the block
        # Previous: Attn / MLP -> Dropout -> Add -> LN
        # Current: Dropout -> Add -> LN -> Attn / MLP
        if "transformer.ln_0.weight" in state_dict:
            n_layers = len(self.transformer.layers)
            ln_weight = state_dict.pop(f"transformer.layers.{n_layers - 1}.norm2.weight")
            ln_bias = state_dict.pop(f"transformer.layers.{n_layers - 1}.norm2.bias")
            state_dict["transformer.ln_f.weight"] = ln_weight
            state_dict["transformer.ln_f.bias"] = ln_bias
            for l in reversed(range(n_layers)):
                ln_weight = state_dict.pop(f"transformer.layers.{l}.norm1.weight")
                ln_bias = state_dict.pop(f"transformer.layers.{l}.norm1.bias")
                state_dict[f"transformer.layers.{l}.norm2.weight"] = ln_weight
                state_dict[f"transformer.layers.{l}.norm2.bias"] = ln_bias
                if l > 0:
                    ln_weight = state_dict.pop(f"transformer.layers.{l - 1}.norm2.weight")
                    ln_bias = state_dict.pop(f"transformer.layers.{l - 1}.norm2.bias")
                    state_dict[f"transformer.layers.{l}.norm1.weight"] = ln_weight
                    state_dict[f"transformer.layers.{l}.norm1.bias"] = ln_bias
            ln_weight = state_dict.pop("transformer.ln_0.weight")
            ln_bias = state_dict.pop("transformer.ln_0.bias")
            state_dict[f"transformer.layers.0.norm1.weight"] = ln_weight
            state_dict[f"transformer.layers.0.norm1.bias"] = ln_bias
        return super().load_state_dict(state_dict, strict=strict)


def shard_state_dict_tp(state_dict, config, world_size, rank):
    """Convert the state_dict of a standard GPT model to the state_dict of a GPT model
    with tensor parallel.
    """
    pad_vocab_size_multiple = getattr(config, "pad_vocab_size_multiple", 1)
    vocab_size = math.ceil(config.vocab_size / pad_vocab_size_multiple) * pad_vocab_size_multiple
    assert vocab_size % world_size == 0
    assert config.hidden_size % world_size == 0
    inner_dim = config.n_inner if config.n_inner is not None else 4 * config.hidden_size
    assert inner_dim % world_size == 0

    def shard_first_dim(state_dict, key):
        x = state_dict[key]
        dim = x.shape[0] // world_size
        state_dict[key] = x[rank * dim : (rank + 1) * dim]

    def shard_last_dim(state_dict, key):
        x = state_dict[key]
        dim = x.shape[-1] // world_size
        state_dict[key] = x[..., rank * dim : (rank + 1) * dim]

    def shard_qkv_headdim(state_dict, key):
        x = rearrange(state_dict[key], "(three d) ... -> three d ...", three=3)
        dim = x.shape[1] // world_size
        state_dict[key] = rearrange(x[:, rank * dim : (rank + 1) * dim], "three d ... -> (three d) ...")

    shard_first_dim(state_dict, "transformer.embeddings.word_embeddings.weight")
    if "lm_head.weight" in state_dict:
        shard_first_dim(state_dict, "lm_head.weight")
    if "transformer.embeddings.position_embeddings.weight" in state_dict:
        shard_last_dim(state_dict, "transformer.embeddings.position_embeddings.weight")
    for i in range(config.num_hidden_layers):
        shard_qkv_headdim(state_dict, f"transformer.layers.{i}.mixer.Wqkv.weight")
        shard_qkv_headdim(state_dict, f"transformer.layers.{i}.mixer.Wqkv.bias")
        shard_last_dim(state_dict, f"transformer.layers.{i}.mixer.out_proj.weight")
        if rank != 0:
            state_dict.pop(f"transformer.layers.{i}.mixer.out_proj.bias")
        shard_first_dim(state_dict, f"transformer.layers.{i}.mlp.fc1.weight")
        shard_first_dim(state_dict, f"transformer.layers.{i}.mlp.fc1.bias")
        shard_last_dim(state_dict, f"transformer.layers.{i}.mlp.fc2.weight")
        if rank != 0:
            state_dict.pop(f"transformer.layers.{i}.mlp.fc2.bias")
    return state_dict


def combine_state_dicts_tp(state_dicts, config):
    """Convert the state_dict of a standard GPT model to the state_dict of a GPT model
    with tensor parallel.
    """
    world_size = len(state_dicts)
    keys = state_dicts[0].keys()
    pad_vocab_size_multiple = getattr(config, "pad_vocab_size_multiple", 1)
    vocab_size = math.ceil(config.vocab_size / pad_vocab_size_multiple) * pad_vocab_size_multiple
    assert vocab_size % world_size == 0
    assert config.hidden_size % world_size == 0
    inner_dim = config.n_inner if config.n_inner is not None else 4 * config.hidden_size
    assert inner_dim % world_size == 0

    # The word embeddings from Megatron are weird, for each shard only the first
    # vocab_size // world_size coordinates are nonzero.
    def combine_word_embeddings(state_dicts, state_dict, key):
        assert all(s[key].shape[0] == vocab_size for s in state_dicts)
        state_dict[key] = torch.cat([s[key][: vocab_size // world_size] for s in state_dicts], dim=0)

    def combine_dim(state_dicts, state_dict, key, dim=-1):
        state_dict[key] = torch.cat([s[key] for s in state_dicts], dim=dim)

    def combine_qkv_headdim(state_dicts, state_dict, key):
        xs = [rearrange(s[key], "(three d) ... -> three d ...", three=3) for s in state_dicts]
        state_dict[key] = rearrange(torch.cat(xs, dim=1), "three d ... -> (three d) ...")

    state_dict = state_dicts[0].copy()  # don't modify state_dict[0] inplace
    combine_word_embeddings(state_dicts, state_dict, "transformer.embeddings.word_embeddings.weight")
    if "lm_head.weight" in state_dict:
        combine_word_embeddings(state_dicts, state_dict, "lm_head.weight")
    if "transformer.embeddings.position_embeddings.weight" in state_dict:
        combine_dim(state_dicts, state_dict, "transformer.embeddings.position_embeddings.weight", -1)
    for i in range(config.num_hidden_layers):
        combine_qkv_headdim(state_dicts, state_dict, f"transformer.layers.{i}.mixer.Wqkv.weight")
        combine_qkv_headdim(state_dicts, state_dict, f"transformer.layers.{i}.mixer.Wqkv.bias")
        combine_dim(state_dicts, state_dict, f"transformer.layers.{i}.mixer.out_proj.weight", -1)
        combine_dim(state_dicts, state_dict, f"transformer.layers.{i}.mlp.fc1.weight", 0)
        combine_dim(state_dicts, state_dict, f"transformer.layers.{i}.mlp.fc1.bias", 0)
        combine_dim(state_dicts, state_dict, f"transformer.layers.{i}.mlp.fc2.weight", -1)
    return state_dict


def remap_state_dict_hf_gpt2(state_dict, config):
    # Word embedding and position embedding
    def key_mapping_pos_emb(key):
        return re.sub(r"^wpe.", "transformer.embeddings.position_embeddings.", key)

    state_dict = OrderedDict((key_mapping_pos_emb(k), v) for k, v in state_dict.items())
    word_embeddings = state_dict.pop("wte.weight")
    # It's possible that vocab_size is padded to be a multiple of 8, for example.
    pad_vocab_size_multiple = getattr(config, "pad_vocab_size_multiple", 1)
    vocab_size = math.ceil(config.vocab_size / pad_vocab_size_multiple) * pad_vocab_size_multiple
    state_dict["transformer.embeddings.word_embeddings.weight"] = F.pad(
        word_embeddings, (0, 0, 0, vocab_size - word_embeddings.shape[0])
    )
    state_dict["lm_head.weight"] = state_dict["transformer.embeddings.word_embeddings.weight"]

    # LayerNorm
    def key_mapping_ln(key):
        key = re.sub(r"^ln_f.(weight|bias)", r"transformer.ln_f.\1", key)
        key = re.sub(r"^h.(\d+).ln_(1|2).(weight|bias)", r"transformer.layers.\1.norm\2.\3", key)
        return key

    state_dict = OrderedDict((key_mapping_ln(k), v) for k, v in state_dict.items())

    # MLP
    for d in range(config.num_hidden_layers):
        W1 = state_dict.pop(f"h.{d}.mlp.c_fc.weight")
        state_dict[f"transformer.layers.{d}.mlp.fc1.weight"] = W1.t()
        W2 = state_dict.pop(f"h.{d}.mlp.c_proj.weight")
        state_dict[f"transformer.layers.{d}.mlp.fc2.weight"] = W2.t()

    def key_mapping_mlp(key):
        key = re.sub(r"^h.(\d+).mlp.c_fc.bias", r"transformer.layers.\1.mlp.fc1.bias", key)
        key = re.sub(r"^h.(\d+).mlp.c_proj.bias", r"transformer.layers.\1.mlp.fc2.bias", key)
        return key

    state_dict = OrderedDict((key_mapping_mlp(k), v) for k, v in state_dict.items())

    # Attention
    for d in range(config.num_hidden_layers):
        state_dict.pop(f"h.{d}.attn.bias")  # We don't store this bias
        Wqkv = state_dict.pop(f"h.{d}.attn.c_attn.weight")
        state_dict[f"transformer.layers.{d}.mixer.Wqkv.weight"] = Wqkv.t()
        Wout = state_dict.pop(f"h.{d}.attn.c_proj.weight")
        state_dict[f"transformer.layers.{d}.mixer.out_proj.weight"] = Wout.t()

    def key_mapping_attn(key):
        key = re.sub(r"^h.(\d+).attn.c_attn.bias", r"transformer.layers.\1.mixer.Wqkv.bias", key)
        key = re.sub(r"^h.(\d+).attn.c_proj.bias", r"transformer.layers.\1.mixer.out_proj.bias", key)
        return key

    state_dict = OrderedDict((key_mapping_attn(k), v) for k, v in state_dict.items())

    return state_dict


def remap_state_dict_megatron(state_dict, config):
    def key_mapping_transformer(key):
        key = re.sub(r"^language_model.encoder.", "transformer.", key)
        key = re.sub(r"^language_model.", "transformer.", key)
        return key

    state_dict = OrderedDict((key_mapping_transformer(k), v) for k, v in state_dict.items())

    # Word embedding and position embedding
    def key_mapping_pos_emb(key):
        return re.sub(r"^wpe.", "transformer.embeddings.position_embeddings.", key)

    state_dict = OrderedDict((key_mapping_pos_emb(k), v) for k, v in state_dict.items())
    word_embeddings = state_dict.pop("transformer.embedding.word_embeddings.weight")
    # It's possible that vocab_size is padded to be a multiple of 8, for example.
    pad_vocab_size_multiple = getattr(config, "pad_vocab_size_multiple", 1)
    vocab_size = math.ceil(config.vocab_size / pad_vocab_size_multiple) * pad_vocab_size_multiple
    state_dict["transformer.embeddings.word_embeddings.weight"] = F.pad(
        word_embeddings, (0, 0, 0, vocab_size - word_embeddings.shape[0])
    )
    state_dict["lm_head.weight"] = state_dict["transformer.embeddings.word_embeddings.weight"]

    # LayerNorm
    def key_mapping_ln(key):
        key = re.sub(r"^transformer.final_layernorm.(weight|bias)", r"transformer.ln_f.\1", key)
        key = re.sub(r"^transformer.layers.(\d+).input_layernorm.(weight|bias)", r"transformer.layers.\1.norm1.\2", key)
        key = re.sub(
            r"^transformer.layers.(\d+).post_attention_layernorm.(weight|bias)", r"transformer.layers.\1.norm2.\2", key
        )
        return key

    state_dict = OrderedDict((key_mapping_ln(k), v) for k, v in state_dict.items())

    # MLP
    def key_mapping_mlp(key):
        key = re.sub(
            r"^transformer.layers.(\d+).mlp.dense_h_to_4h.(weight|bias)", r"transformer.layers.\1.mlp.fc1.\2", key
        )
        key = re.sub(
            r"^transformer.layers.(\d+).mlp.dense_4h_to_h.(weight|bias)", r"transformer.layers.\1.mlp.fc2.\2", key
        )
        return key

    state_dict = OrderedDict((key_mapping_mlp(k), v) for k, v in state_dict.items())

    # Attention
    def key_mapping_attn(key):
        key = re.sub(
            r"^transformer.layers.(\d+).self_attention.rotary_emb.inv_freq",
            r"transformer.layers.\1.mixer.rotary_emb.inv_freq",
            key,
        )
        key = re.sub(
            r"^transformer.layers.(\d+).self_attention.query_key_value.(weight|bias)",
            r"transformer.layers.\1.mixer.Wqkv.\2",
            key,
        )
        key = re.sub(
            r"^transformer.layers.(\d+).self_attention.dense.(weight|bias)",
            r"transformer.layers.\1.mixer.out_proj.\2",
            key,
        )
        return key

    state_dict = OrderedDict((key_mapping_attn(k), v) for k, v in state_dict.items())
    # Megatron stores Wqkv as ((nheads 3 headdim), hidden_dim)
    # while we store Wqkv as ((3 nheads headdim), hidden_dim)
    headdim = config.hidden_size // config.num_attention_heads
    for d in range(config.num_hidden_layers):
        Wqkv = state_dict.pop(f"transformer.layers.{d}.mixer.Wqkv.weight")
        state_dict[f"transformer.layers.{d}.mixer.Wqkv.weight"] = rearrange(
            Wqkv, "(nheads three headdim) ... -> (three nheads headdim) ...", three=3, headdim=headdim
        )
        bqkv = state_dict.pop(f"transformer.layers.{d}.mixer.Wqkv.bias")
        state_dict[f"transformer.layers.{d}.mixer.Wqkv.bias"] = rearrange(
            bqkv, "(nheads three headdim) -> (three nheads headdim)", three=3, headdim=headdim
        )

    return state_dict
