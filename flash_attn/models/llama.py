# Copyright (c) 2023, Tri Dao.

import math
import json
import re
from pathlib import Path

from collections import OrderedDict

import torch
import torch.nn.functional as F

from transformers import GPT2Config, LlamaConfig


def remap_state_dict_meta_llama(state_dict, config):
    def key_mapping_layers(key):
        return f'transformer.{key}' if not key.startswith('output.') else key
    state_dict = OrderedDict((key_mapping_layers(k), v) for k, v in state_dict.items())
    # Word embedding
    def key_mapping_emb(key):
        return re.sub(r'^transformer.tok_embeddings.', 'transformer.embeddings.word_embeddings.', key)
    state_dict = OrderedDict((key_mapping_emb(k), v) for k, v in state_dict.items())
    word_embeddings = state_dict.pop('transformer.embeddings.word_embeddings.weight')
    # It's possible that vocab_size is padded to be a multiple of 8, for example.
    pad_vocab_size_multiple = getattr(config, 'pad_vocab_size_multiple', 1)
    vocab_size = (math.ceil(word_embeddings.shape[0] / pad_vocab_size_multiple)
                  * pad_vocab_size_multiple)
    state_dict['transformer.embeddings.word_embeddings.weight'] = F.pad(
        word_embeddings, (0, 0, 0, vocab_size - word_embeddings.shape[0])
    )
    if getattr(config, 'tie_word_embeddings'):
        state_dict['lm_head.weight'] = state_dict['transformer.embeddings.word_embeddings.weight']
    else:
        output_embeddings = state_dict.pop('output.weight')
        # Need to recompute vocab_size since LLaMa shards the word embeddings and output embeddings
        # differently.
        vocab_size = (math.ceil(output_embeddings.shape[0] / pad_vocab_size_multiple)
                    * pad_vocab_size_multiple)
        # It's possible that vocab_size is padded to be a multiple of 8, for example.
        state_dict['lm_head.weight'] = F.pad(
            output_embeddings, (0, 0, 0, vocab_size - output_embeddings.shape[0])
        )

    # LayerNorm
    def key_mapping_ln(key):
        key = re.sub(r'^transformer.norm.', r'transformer.ln_f.', key)
        key = re.sub(r'^transformer.layers.(\d+).attention_norm.', r'transformer.layers.\1.norm1.', key)
        key = re.sub(r'^transformer.layers.(\d+).ffn_norm.', r'transformer.layers.\1.norm2.', key)
        return key
    state_dict = OrderedDict((key_mapping_ln(k), v) for k, v in state_dict.items())

    # MLP
    for l in range(config.n_layer):
        w1 = state_dict.pop(f'transformer.layers.{l}.feed_forward.w1.weight')
        w3 = state_dict.pop(f'transformer.layers.{l}.feed_forward.w3.weight')
        # Our ordering is different
        state_dict[f'transformer.layers.{l}.mlp.fc1.weight'] = torch.cat([w3, w1], dim=0)
    def key_mapping_mlp(key):
        return re.sub(r'^transformer.layers.(\d+).feed_forward.w2.',
                      r'transformer.layers.\1.mlp.fc2.', key)
    state_dict = OrderedDict((key_mapping_mlp(k), v) for k, v in state_dict.items())

    # Attention
    for l in range(config.n_layer):
        Wq = state_dict.pop(f'transformer.layers.{l}.attention.wq.weight')
        Wk = state_dict.pop(f'transformer.layers.{l}.attention.wk.weight')
        Wv = state_dict.pop(f'transformer.layers.{l}.attention.wv.weight')
        state_dict[f'transformer.layers.{l}.mixer.Wqkv.weight'] = torch.cat([Wq, Wk, Wv], dim=0)
        # We don't store these
        state_dict.pop(f'transformer.layers.{l}.attention.inner_attention.rope.freqs', None)
    def key_mapping_attn(key):
        return re.sub(r'^transformer.layers.(\d+).attention.wo.',
                      r'transformer.layers.\1.mixer.out_proj.', key)
    state_dict = OrderedDict((key_mapping_attn(k), v) for k, v in state_dict.items())

    return state_dict


def config_from_checkpoint(checkpoint_path: str, model_name: str) -> LlamaConfig:
    """Load a LlamaConfig from a checkpoint path."""
    with open(Path(checkpoint_path) / model_name / 'params.json') as f:
        params = json.load(f)
    config = LlamaConfig(hidden_size=params['dim'], intermediate_size=None,
                         num_attention_heads=params['n_heads'],
                         num_hidden_layers=params['n_layers'],
                         rms_norm_eps=params['norm_eps'])
    return config


def state_dicts_from_checkpoint(checkpoint_path: str, model_name: str) -> dict:
    # Need to sort, otherwise we mess up the ordering and the weights are wrong
    return [torch.load(path, map_location='cpu')
            for path in sorted((Path(checkpoint_path) / model_name).glob('consolidated.*.pth'))]


def llama_config_to_gpt2_config(llama_config: LlamaConfig) -> GPT2Config:
    return GPT2Config(
        vocab_size=llama_config.vocab_size,
        n_positions=0,  # No absolute position embedding
        n_embd=llama_config.hidden_size,
        n_layer=llama_config.num_hidden_layers,
        n_head=llama_config.num_attention_heads,
        n_inner=llama_config.intermediate_size,
        activation_function='swiglu',  # Hardcode since HF calls it 'silu'
        # Llama doesn't have dropout, idk if it's because they only release the inference code
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        layer_norm_epsilon=llama_config.rms_norm_eps,
        initializer_range=llama_config.initializer_range,
        bos_token_id=llama_config.bos_token_id,
        eos_token_id=llama_config.eos_token_id,
        # These are new arguments not in the original GPT2Config
        pad_token_id=llama_config.pad_token_id,  # Idk if this does anything
        rms_norm=True,
        rotary_emb_fraction=1.0,
        rotary_emb_interleaved=True,
        tie_word_embeddings=False,
        qkv_proj_bias=False,
        out_proj_bias=False,
        mlp_fc1_bias=False,
        mlp_fc2_bias=False,
    )
