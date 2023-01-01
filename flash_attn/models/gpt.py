# Copyright (c) 2022, Tri Dao.

import logging
import math
import re
from functools import partial

from collections import namedtuple, OrderedDict
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import GPT2Config

from einops import rearrange

from flash_attn.modules.mha import MHA, ParallelMHA
from flash_attn.modules.mlp import Mlp, FusedDenseGeluDense, ParallelFusedDenseGeluDense
from flash_attn.modules.block import Block
from flash_attn.modules.embedding import GPT2Embeddings, ParallelGPT2Embeddings
from flash_attn.utils.distributed import sync_sequence_parallel_params
from flash_attn.utils.pretrained import state_dict_from_pretrained
from flash_attn.utils.generation import GenerationMixin

try:
    from flash_attn.ops.fused_dense import ColumnParallelLinear
except ImportError:
    ColumnParallelLinear = None

try:
    from flash_attn.ops.layer_norm import dropout_add_layer_norm
except ImportError:
    dropout_add_layer_norm = None

try:
    from flash_attn.ops.triton.mlp import FusedDenseSqreluDense
except ImportError:
    FusedDenseSqreluDense = None


logger = logging.getLogger(__name__)


def create_mixer_cls(config, layer_idx=None, process_group=None, device=None, dtype=None):
    factory_kwargs = {'device': device, 'dtype': dtype}
    head_dim = getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)
    softmax_scale = 1.0 if not config.scale_attn_weights else head_dim ** (-0.5)
    if config.scale_attn_by_inverse_layer_idx:
        assert layer_idx is not None
        softmax_scale /= float(layer_idx + 1)
    dwconv = getattr(config, 'attn_dwconv', False)
    if dwconv:
        assert process_group is None, 'TensorParallel MHA does not support dwconv yet'
    rotary_emb_dim = int(getattr(config, 'rotary_emb_fraction', 0.0) * head_dim)
    rotary_emb_scale_base = getattr(config, 'rotary_emb_scale_base', 0)
    use_flash_attn = getattr(config, 'use_flash_attn', False)
    fused_bias_fc = getattr(config, 'fused_bias_fc', False)
    if not fused_bias_fc:
        assert process_group is None, 'TensorParallel MHA requires fused_bias_fc'
    mha_cls = MHA if process_group is None else ParallelMHA
    serial_kwargs = ({'fused_bias_fc': fused_bias_fc, 'dwconv': dwconv}
                     if process_group is None else {})
    parallel_kwargs = {'process_group': process_group} if process_group is not None else {}
    mixer_cls = partial(mha_cls, num_heads=config.num_attention_heads, dropout=config.attn_pdrop,
                        softmax_scale=softmax_scale, causal=True, layer_idx=layer_idx,
                        rotary_emb_dim=rotary_emb_dim, rotary_emb_scale_base=rotary_emb_scale_base,
                        use_flash_attn=use_flash_attn,
                        **serial_kwargs, **parallel_kwargs, **factory_kwargs)
    return mixer_cls


def create_mlp_cls(config, layer_idx=None, process_group=None, device=None, dtype=None):
    factory_kwargs = {'device': device, 'dtype': dtype}
    inner_dim = config.n_inner if config.n_inner is not None else 4 * config.hidden_size
    fused_dense_gelu_dense = getattr(config, 'fused_dense_gelu_dense', False)
    if fused_dense_gelu_dense:
        assert config.activation_function in ['gelu_new', 'gelu_fast'], ('fused_dense_gelu_dense only '
                                                                'supports approximate gelu')
    fused_dense_sqrelu_dense = getattr(config, 'fused_dense_sqrelu_dense', False)
    if fused_dense_sqrelu_dense:
        assert config.activation_function == 'sqrelu', ('fused_dense_sqrelu_dense only '
                                               'supports approximate activation_function sqrelu')
    assert not (fused_dense_sqrelu_dense and fused_dense_gelu_dense)
    if process_group is not None:
        assert fused_dense_gelu_dense, 'Tensor Parallel is only implemented for FusedDenseGeluDense'
    if not fused_dense_gelu_dense and not fused_dense_sqrelu_dense:
        approximate = 'tanh' if config.activation_function in ['gelu_new', 'gelu_fast'] else 'none'
        mlp_cls = partial(Mlp, hidden_features=inner_dim,
                          activation=partial(F.gelu, approximate=approximate), **factory_kwargs)
    else:
        mlp_checkpoint_lvl = getattr(config, 'mlp_checkpoint_lvl', 0)
        # mlp_checkpoint_lvl could be a list, which contains the checkpoint_lvl for each layer
        if isinstance(mlp_checkpoint_lvl, Sequence):
            assert layer_idx is not None
            mlp_checkpoint_lvl = mlp_checkpoint_lvl[layer_idx]
        if fused_dense_gelu_dense:
            if FusedDenseGeluDense is None:
                raise ImportError('fused_dense is not installed')
            mlp_cls = FusedDenseGeluDense if process_group is None else ParallelFusedDenseGeluDense
            parallel_kwargs = {'process_group': process_group} if process_group is not None else {}
            mlp_cls = partial(mlp_cls, hidden_features=inner_dim, checkpoint_lvl=mlp_checkpoint_lvl,
                              **parallel_kwargs, **factory_kwargs)
        elif fused_dense_sqrelu_dense:
            assert FusedDenseSqreluDense is not None
            mlp_cls = partial(FusedDenseSqreluDense, hidden_features=inner_dim,
                              checkpoint_lvl=mlp_checkpoint_lvl, **factory_kwargs)
        else:
            raise RuntimeError('MLP type not supported')
    return mlp_cls


def create_block(config, layer_idx=None, process_group=None, device=None, dtype=None):
    factory_kwargs = {'device': device, 'dtype': dtype}
    mixer_cls = create_mixer_cls(config, layer_idx, process_group=process_group, **factory_kwargs)
    mlp_cls = create_mlp_cls(config, layer_idx, process_group=process_group, **factory_kwargs)
    norm_cls = partial(nn.LayerNorm, eps=config.layer_norm_epsilon, **factory_kwargs)
    block = Block(config.hidden_size, mixer_cls, mlp_cls, norm_cls=norm_cls,
                  prenorm=True, resid_dropout=config.resid_pdrop,
                  fused_dropout_add_ln=getattr(config, 'fused_dropout_add_ln', False),
                  sequence_parallel=process_group is not None)
    block.layer_idx = layer_idx
    return block


class GPTPreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
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
                ))
        self.config = config

    @classmethod
    def from_pretrained(cls, model_name, config, *inputs, **kwargs):
        """
        Instantiate a GPTPreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.
        """
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        load_return = model.load_state_dict(
            remap_state_dict_gpt2(state_dict_from_pretrained(model_name), config))
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
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.process_group = process_group
        assert config.activation_function in ['gelu', 'gelu_new', 'gelu_fast', 'sqrelu']
        self.pad_vocab_size_multiple = getattr(config, 'pad_vocab_size_multiple', 1)
        if config.vocab_size % self.pad_vocab_size_multiple != 0:
            config.vocab_size += (self.pad_vocab_size_multiple
                                  - (config.vocab_size % self.pad_vocab_size_multiple))

        if process_group is None:
            self.embeddings = GPT2Embeddings(config.hidden_size, config.vocab_size,
                                             config.max_position_embeddings, **factory_kwargs)
        else:
            self.embeddings = ParallelGPT2Embeddings(
                config.hidden_size, config.vocab_size, config.max_position_embeddings,
                process_group=process_group, **factory_kwargs
            )
        self.emb_drop = nn.Dropout(config.embd_pdrop)

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Dropout -> Add, we do:
        # Attn / MLP -> Dropout -> Add -> LN, returning both the residual branch (output of Add) and
        # the main branch (output of LN). The model definition is unchanged, but the mapping of the
        # nn.LayerNorm weights are changed.
        # This is for performance reason: we can fuse dropout + add + layer_norm.
        self.fused_dropout_add_ln = getattr(config, 'fused_dropout_add_ln', False)
        if self.fused_dropout_add_ln and dropout_add_layer_norm is None:
            raise ImportError('dropout_add_layer_norm is not installed')
        # self.ln_0 is the first layer norm in the model, while self.ln_f (in the pretrained weight)
        # is the final layer norm.
        self.ln_0 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon,
                                 **factory_kwargs)
        # Mark the norm parameters as "sequence_parallel" so that we run all-reduce on their grads.
        if process_group is not None:
            for p in self.ln_0.parameters():
                p._sequence_parallel = True

        self.layers = nn.ModuleList([create_block(config, layer_idx=i, process_group=process_group,
                                                  **factory_kwargs)
                                     for i in range(config.num_hidden_layers)])

        self.apply(partial(_init_weights, n_layer=config.num_hidden_layers,
                           initializer_range=config.initializer_range))
        self.tie_weights()

    def tie_weights(self):
        if self.process_group is not None:
            sync_sequence_parallel_params(self, self.process_group)

    def forward(self, input_ids, position_ids=None, inference_params=None):
        # If using Tensor Parallel with sequence parallel, we combine the batch and the seqlen
        # dimensions so that we can split on it easily, in case of small batch size.
        # Only the attention layers need to know the seqlen.
        embedding_kwargs = ({'combine_batch_seqlen_dim': True}
                            if self.process_group is not None else {})
        hidden_states = self.embeddings(input_ids, position_ids=position_ids, **embedding_kwargs)
        # TD [2022-07-30]: Force residual in fp32, seems to make fp16 training more stable
        if not self.fused_dropout_add_ln:
            residual = self.emb_drop(hidden_states).float()
            hidden_states = self.ln_0(residual.to(dtype=self.ln_0.weight.dtype))
        else:
            hidden_states, residual = dropout_add_layer_norm(
                hidden_states, None, self.ln_0.weight, self.ln_0.bias,
                self.emb_drop.p if self.training else 0.0, self.ln_0.eps, prenorm=True,
                residual_in_fp32=True
            )
        mixer_kwargs = ({'seqlen': input_ids.shape[1]} if self.process_group is not None else {})
        if inference_params is not None:
            mixer_kwargs['inference_params'] = inference_params
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual, mixer_kwargs=mixer_kwargs)
        return hidden_states


class GPTLMHeadModel(GPTPreTrainedModel, GenerationMixin):

    def __init__(self, config: GPT2Config, process_group=None, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(config)
        self.process_group = process_group
        self.transformer = GPTModel(config, process_group=process_group, **factory_kwargs)
        if process_group is None:
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False, **factory_kwargs)
        else:
            if ColumnParallelLinear is None:
                raise ImportError('fused_dense_lib is not installed')
            self.lm_head = ColumnParallelLinear(config.n_embd, config.vocab_size, process_group,
                                                bias=False, **factory_kwargs)
        # Initialize weights and apply final processing
        self.apply(partial(_init_weights, n_layer=config.num_hidden_layers,
                           initializer_range=config.initializer_range))
        self.tie_weights()

    def tie_weights(self):
        self.lm_head.weight = self.transformer.embeddings.word_embeddings.weight
        if self.process_group is not None:
            sync_sequence_parallel_params(self, self.process_group)

    def forward(self, input_ids, position_ids=None, inference_params=None):
        """
            inference_params: for generation. Adapted from Megatron-LM (and Apex)
            https://github.com/NVIDIA/apex/blob/3ff1a10f72ec07067c4e44759442329804ac5162/apex/transformer/testing/standalone_transformer_lm.py#L470
        """
        hidden_states = self.transformer(input_ids, position_ids=position_ids,
                                         inference_params=inference_params)
        lm_logits = self.lm_head(hidden_states)
        CausalLMOutput = namedtuple('CausalLMOutput', ['logits'])
        return CausalLMOutput(logits=lm_logits)


def remap_state_dict_gpt2(state_dict, config):
    # Word embedding and position embedding
    def key_mapping_pos_emb(key):
        return re.sub(r'^wpe.', 'transformer.embeddings.position_embeddings.', key)
    state_dict = OrderedDict((key_mapping_pos_emb(k), v) for k, v in state_dict.items())
    word_embeddings = state_dict.pop('wte.weight')
    # It's possible that vocab_size is padded to be a multiple of 8, for example.
    state_dict['transformer.embeddings.word_embeddings.weight'] = F.pad(
        word_embeddings, (0, 0, 0, config.vocab_size - word_embeddings.shape[0])
    )
    state_dict['lm_head.weight'] = state_dict['transformer.embeddings.word_embeddings.weight']

    # LayerNorm
    ln_weight, ln_bias = state_dict.pop('ln_f.weight'), state_dict.pop('ln_f.bias')
    state_dict[f'transformer.layers.{config.num_hidden_layers - 1}.norm2.weight'] = ln_weight
    state_dict[f'transformer.layers.{config.num_hidden_layers - 1}.norm2.bias'] = ln_bias
    ln_weight, ln_bias = state_dict.pop('h.0.ln_1.weight'), state_dict.pop('h.0.ln_1.bias')
    state_dict['transformer.ln_0.weight'] = ln_weight
    state_dict['transformer.ln_0.bias'] = ln_bias
    for d in range(config.num_hidden_layers):
        ln_weight = state_dict.pop(f'h.{d}.ln_2.weight')
        ln_bias = state_dict.pop(f'h.{d}.ln_2.bias')
        state_dict[f'transformer.layers.{d}.norm1.weight'] = ln_weight
        state_dict[f'transformer.layers.{d}.norm1.bias'] = ln_bias
        if d > 0:
            ln_weight = state_dict.pop(f'h.{d}.ln_1.weight')
            ln_bias = state_dict.pop(f'h.{d}.ln_1.bias')
            state_dict[f'transformer.layers.{d - 1}.norm2.weight'] = ln_weight
            state_dict[f'transformer.layers.{d - 1}.norm2.bias'] = ln_bias

    # MLP
    for d in range(config.num_hidden_layers):
        W1 = state_dict.pop(f'h.{d}.mlp.c_fc.weight')
        state_dict[f'transformer.layers.{d}.mlp.fc1.weight'] = W1.t()
        W2 = state_dict.pop(f'h.{d}.mlp.c_proj.weight')
        state_dict[f'transformer.layers.{d}.mlp.fc2.weight'] = W2.t()
    def key_mapping_mlp(key):
        key = re.sub(r'^h.(\d+).mlp.c_fc.bias', r'transformer.layers.\1.mlp.fc1.bias', key)
        key = re.sub(r'^h.(\d+).mlp.c_proj.bias', r'transformer.layers.\1.mlp.fc2.bias', key)
        return key
    state_dict = OrderedDict((key_mapping_mlp(k), v) for k, v in state_dict.items())

    # Attention
    for d in range(config.num_hidden_layers):
        state_dict.pop(f'h.{d}.attn.bias')  # We don't store this bias
        Wqkv = state_dict.pop(f'h.{d}.attn.c_attn.weight')
        state_dict[f'transformer.layers.{d}.mixer.Wqkv.weight'] = Wqkv.t()
        Wout = state_dict.pop(f'h.{d}.attn.c_proj.weight')
        state_dict[f'transformer.layers.{d}.mixer.out_proj.weight'] = Wout.t()
    def key_mapping_attn(key):
        key = re.sub(r'^h.(\d+).attn.c_attn.bias', r'transformer.layers.\1.mixer.Wqkv.bias', key)
        key = re.sub(r'^h.(\d+).attn.c_proj.bias', r'transformer.layers.\1.mixer.out_proj.bias', key)
        return key
    state_dict = OrderedDict((key_mapping_attn(k), v) for k, v in state_dict.items())

    return state_dict


def shard_state_dict_tp(state_dict, config, world_size, rank):
    """Convert the state_dict of a standard GPT model to the state_dict of a GPT model
    with tensor parallel.
    """
    vocab_size = config.vocab_size
    if config.vocab_size % config.pad_vocab_size_multiple != 0:
        vocab_size += (config.pad_vocab_size_multiple
                       - (config.vocab_size % config.pad_vocab_size_multiple))
    assert vocab_size % world_size == 0
    assert config.hidden_size % world_size == 0
    inner_dim = config.n_inner if config.n_inner is not None else 4 * config.hidden_size
    assert inner_dim % world_size == 0

    def shard_first_dim(state_dict, key):
        x = state_dict[key]
        dim = x.shape[0] // world_size
        state_dict[key] = x[rank * dim:(rank + 1) * dim]

    def shard_last_dim(state_dict, key):
        x = state_dict[key]
        dim = x.shape[-1] // world_size
        state_dict[key] = x[..., rank * dim:(rank + 1) * dim]

    def shard_qkv_headdim(state_dict, key):
        x = rearrange(state_dict[key], '(three d) ... -> three d ...', three=3)
        dim = x.shape[1] // world_size
        state_dict[key] = rearrange(x[:, rank * dim:(rank + 1) * dim],
                                    'three d ... -> (three d) ...')

    shard_first_dim(state_dict, 'transformer.embeddings.word_embeddings.weight')
    if 'lm_head.weight' in state_dict:
        shard_first_dim(state_dict, 'lm_head.weight')
    if 'transformer.embeddings.position_embeddings.weight' in state_dict:
        shard_last_dim(state_dict, 'transformer.embeddings.position_embeddings.weight')
    for i in range(config.num_hidden_layers):
        shard_qkv_headdim(state_dict, f'transformer.layers.{i}.mixer.Wqkv.weight')
        shard_qkv_headdim(state_dict, f'transformer.layers.{i}.mixer.Wqkv.bias')
        shard_last_dim(state_dict, f'transformer.layers.{i}.mixer.out_proj.weight')
        if rank != 0:
            state_dict.pop(f'transformer.layers.{i}.mixer.out_proj.bias')
        shard_first_dim(state_dict, f'transformer.layers.{i}.mlp.fc1.weight')
        shard_first_dim(state_dict, f'transformer.layers.{i}.mlp.fc1.bias')
        shard_last_dim(state_dict, f'transformer.layers.{i}.mlp.fc2.weight')
        if rank != 0:
            state_dict.pop(f'transformer.layers.{i}.mlp.fc2.bias')
    return state_dict
