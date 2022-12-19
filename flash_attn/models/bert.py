# Copyright (c) 2022, Tri Dao.
# This BERT implementation is based on our MLPerf 2.0 and MLPerf 2.1 BERT implementation.
# https://github.com/mlcommons/training_results_v2.0/blob/main/HazyResearch/benchmarks/bert/implementations/pytorch/modeling.py
# https://github.com/mlcommons/training_results_v2.1/blob/main/Azure-HazyResearch/benchmarks/bert/implementations/ND96amsr_A100_v4/modeling.py

# Inspired by https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py

import re
import logging
from functools import partial

from collections.abc import Sequence
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertConfig

from einops import rearrange

from flash_attn.modules.mha import MHA
from flash_attn.modules.mlp import Mlp, FusedDenseGeluDense
from flash_attn.modules.block import Block
from flash_attn.modules.embedding import BertEmbeddings
from flash_attn.bert_padding import unpad_input, pad_input, index_first_axis

try:
    from flash_attn.ops.fused_dense import FusedDenseTD
except ImportError:
    FusedDenseTD = None

try:
    from flash_attn.ops.layer_norm import dropout_add_layer_norm, layer_norm
except ImportError:
    dropout_add_layer_norm, layer_norm = None, None

try:
    from flash_attn.losses.cross_entropy_apex import CrossEntropyLossApex
except ImportError:
    CrossEntropyLossApex = None


logger = logging.getLogger(__name__)


def create_mixer_cls(config):
    use_flash_attn = getattr(config, 'use_flash_attn', False)
    fused_bias_fc = getattr(config, 'fused_bias_fc', False)
    mixer_cls = partial(MHA, num_heads=config.num_attention_heads,
                        dropout=config.attention_probs_dropout_prob, causal=False,
                        fused_bias_fc=fused_bias_fc, use_flash_attn=use_flash_attn)
    return mixer_cls


def create_mlp_cls(config, layer_idx=None):
    inner_dim = config.intermediate_size
    fused_dense_gelu_dense = getattr(config, 'fused_dense_gelu_dense', False)
    if not fused_dense_gelu_dense:
        mlp_cls = partial(Mlp, hidden_features=inner_dim,
                          activation=partial(F.gelu, approximate='tanh'))
    else:
        mlp_checkpoint_lvl = getattr(config, 'mlp_checkpoint_lvl', 0)
        # mlp_checkpoint_lvl could be a list, which contains the checkpoint_lvl for each layer
        if isinstance(mlp_checkpoint_lvl, Sequence):
            assert layer_idx is not None
            mlp_checkpoint_lvl = mlp_checkpoint_lvl[layer_idx]
        mlp_cls = partial(FusedDenseGeluDense, hidden_features=inner_dim,
                          checkpoint_lvl=mlp_checkpoint_lvl)
    return mlp_cls


def create_block(config, layer_idx=None):
    mixer_cls = create_mixer_cls(config)
    mlp_cls = create_mlp_cls(config, layer_idx)
    norm_cls = partial(nn.LayerNorm, eps=config.layer_norm_eps)
    block = Block(config.hidden_size, mixer_cls, mlp_cls, norm_cls=norm_cls,
                  prenorm=False, resid_dropout=config.hidden_dropout_prob,
                  fused_dropout_add_ln=getattr(config, 'fused_dropout_add_ln', False))
    return block


# https://github.com/huggingface/transformers/blob/7032e0203262ebb2ebf55da8d2e01f873973e835/src/transformers/models/bert/modeling_bert.py#L748
def _init_weights(module, initializer_range=0.02):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=initializer_range)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)
        if module.padding_idx is not None:
            nn.init.zeros_(module.weight[module.padding_idx])


class BertEncoder(nn.Module):

    def __init__(self, config: BertConfig):
        super().__init__()
        self.use_flash_attn = getattr(config, 'use_flash_attn', False)
        self.layers = nn.ModuleList([create_block(config, layer_idx=i)
                                     for i in range(config.num_hidden_layers)])

    def forward(self, hidden_states, key_padding_mask=None):
        if key_padding_mask is None or not self.use_flash_attn:
            mixer_kwargs = ({'key_padding_mask': key_padding_mask}
                            if key_padding_mask is not None else None)
            for layer in self.layers:
                hidden_states = layer(hidden_states, mixer_kwargs=mixer_kwargs)
        else:
            batch, seqlen = hidden_states.shape[:2]
            hidden_states, indices, cu_seqlens, max_seqlen_in_batch = unpad_input(
                hidden_states, key_padding_mask
            )
            mixer_kwargs = {'cu_seqlens': cu_seqlens, 'max_seqlen': max_seqlen_in_batch}
            for layer in self.layers:
                hidden_states = layer(hidden_states, mixer_kwargs=mixer_kwargs)
            hidden_states = pad_input(hidden_states, indices, batch, seqlen)
        return hidden_states


class BertPooler(nn.Module):

    def __init__(self, config):
        super().__init__()
        fused_bias_fc = getattr(config, 'fused_bias_fc', False)
        if fused_bias_fc and FusedDenseTD is None:
            raise ImportError('fused_dense is not installed')
        linear_cls = nn.Linear if not fused_bias_fc else FusedDenseTD
        self.dense = linear_cls(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states, pool=True):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0] if pool else hidden_states
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):

    def __init__(self, config):
        super().__init__()
        fused_bias_fc = getattr(config, 'fused_bias_fc', False)
        if fused_bias_fc and FusedDenseTD is None:
            raise ImportError('fused_dense is not installed')
        self.fused_dropout_add_ln = getattr(config, 'fused_dropout_add_ln', False)
        if self.fused_dropout_add_ln and layer_norm is None:
            raise ImportError('dropout_add_layer_norm is not installed')
        linear_cls = nn.Linear if not fused_bias_fc else FusedDenseTD
        self.dense = linear_cls(config.hidden_size, config.hidden_size)
        self.transform_act_fn = nn.GELU(approximate='tanh')
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        if not self.fused_dropout_add_ln:
            hidden_states = self.layer_norm(hidden_states)
        else:
            hidden_states = layer_norm(hidden_states, self.layer_norm.weight, self.layer_norm.bias,
                                       self.layer_norm.eps)
        return hidden_states


class BertLMPredictionHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        fused_bias_fc = getattr(config, 'fused_bias_fc', False)
        if fused_bias_fc and FusedDenseTD is None:
            raise ImportError('fused_dense is not installed')
        linear_cls = nn.Linear if not fused_bias_fc else FusedDenseTD

        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = linear_cls(config.hidden_size, config.vocab_size, bias=True)

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class BertPreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, config, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, BertConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    @classmethod
    def from_pretrained(cls, model_name, config, *inputs, **kwargs):
        """
        Instantiate a BertPreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.

        Params:
            pretrained_model_name_or_path: either:
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a BertForPretraining instance
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `model.chkpt` a TensorFlow checkpoint
            *inputs, **kwargs: additional input for the specific Bert class
                (ex: num_labels for BertForSequenceClassification)
        """
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        load_return = model.load_state_dict(remap_state_dict(state_dict_from_pretrained(model_name),
                                                             config), strict=False)
        logger.info(load_return)
        return model


class BertModel(BertPreTrainedModel):

    def __init__(self, config: BertConfig, add_pooling_layer=True):
        super().__init__(config)
        self.pad_vocab_size_multiple = getattr(config, 'pad_vocab_size_multiple', 1)
        if config.vocab_size % self.pad_vocab_size_multiple != 0:
            config.vocab_size += (self.pad_vocab_size_multiple
                                  - (config.vocab_size % self.pad_vocab_size_multiple))
        self.fused_dropout_add_ln = getattr(config, 'fused_dropout_add_ln', False)
        if self.fused_dropout_add_ln and dropout_add_layer_norm is None:
            raise ImportError('dropout_add_layer_norm is not installed')
        assert config.position_embedding_type == 'absolute'
        assert config.hidden_act in ['gelu', 'gelu_new', 'gelu_fast']

        self.embeddings = BertEmbeddings(config.hidden_size, config.vocab_size,
                                         config.max_position_embeddings, config.type_vocab_size,
                                         padding_idx=config.pad_token_id)
        self.emb_drop = nn.Dropout(config.hidden_dropout_prob)
        self.emb_ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.apply(partial(_init_weights, initializer_range=config.initializer_range))

    def forward(self, input_ids, position_ids=None, token_type_ids=None, attention_mask=None,
                masked_tokens_mask=None):
        hidden_states = self.embeddings(input_ids, position_ids=position_ids,
                                        token_type_ids=token_type_ids)
        # TD [2022-12:18]: Don't need to force residual in fp32
        if not self.fused_dropout_add_ln:
            hidden_states = self.emb_drop(hidden_states)
            hidden_states = self.emb_ln(hidden_states)
        else:
            hidden_states = dropout_add_layer_norm(
                hidden_states, None, self.emb_ln.weight, self.emb_ln.bias,
                self.emb_drop.p if self.training else 0.0, self.emb_ln.eps, prenorm=False,
            )
        sequence_output = self.encoder(hidden_states, key_padding_mask=attention_mask)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        return sequence_output, pooled_output


class BertForPreTraining(BertPreTrainedModel):

    def __init__(self, config: BertConfig):
        super().__init__(config)
        # If dense_seq_output, we only need to pass the hidden states for the masked out tokens
        # (around 15%) to the classifier heads.
        self.dense_seq_output = getattr(config, 'dense_seq_output', False)
        # If last_layer_subset, we only need the compute the last layer for a subset of tokens
        # (e.g., the tokens we need to compute the masked LM loss and the next-sentence prediction).
        self.last_layer_subset = getattr(config, 'last_layer_subset', False)
        assert not self.last_layer_subset, 'last_layer_subset is not implemented yet'
        use_xentropy = getattr(config, 'use_xentropy', False)
        if use_xentropy and CrossEntropyLossApex is None:
            raise ImportError('xentropy_cuda is not installed')
        loss_cls = nn.CrossEntropyLoss if not use_xentropy else CrossEntropyLossApex

        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config)
        self.mlm_loss = loss_cls(ignore_index=0)
        self.nsp_loss = loss_cls(ignore_index=-1)

        # Initialize weights and apply final processing
        self.apply(partial(_init_weights, initializer_range=config.initializer_range))
        self.tie_weights()

    def tie_weights(self):
        self.cls.predictions.decoder.weight = self.bert.embeddings.word_embeddings.weight

    def forward(self, input_ids, position_ids=None, token_type_ids=None, attention_mask=None,
                labels=None, next_sentence_label=None):
        """
        Outputs:
            if `labels` and `next_sentence_label` are not `None`:
                Outputs the total_loss which is the sum of the masked language modeling loss and the next
                sentence classification loss.
            if `labels` or `next_sentence_label` is `None`:
                Outputs a tuple comprising
                - the masked language modeling logits of shape [batch_size, sequence_length, vocab_size], and
                - the next sentence classification logits of shape [batch_size, 2].

        """
        masked_tokens_mask = labels > 0 if (self.last_layer_subset and labels is not None) else None
        sequence_output, pooled_output = self.bert(
            input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
            attention_mask=attention_mask.bool(), masked_tokens_mask=masked_tokens_mask
        )
        if self.dense_seq_output and labels is not None:
            masked_token_idx = torch.nonzero(labels.flatten() > 0, as_tuple=False).flatten()
            if not self.last_layer_subset:
                sequence_output = index_first_axis(rearrange(sequence_output, 'b s d -> (b s) d'),
                                                   masked_token_idx)
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        if labels is not None and next_sentence_label is not None:
            if masked_token_idx is not None:  # prediction_scores are already flattened
                masked_lm_loss = self.mlm_loss(prediction_scores,
                                               labels.flatten()[masked_token_idx])
            else:
                masked_lm_loss = self.mlm_loss(rearrange(prediction_scores, '... v -> (...) v'),
                                               rearrange(labels, '... -> (...)'))
            next_sentence_loss = self.nsp_loss(rearrange(seq_relationship_score, '... t -> (...) t'),
                                               rearrange(next_sentence_label, '... -> (...)'))
            total_loss = (masked_lm_loss + next_sentence_loss).float()

            # Masked Language Model Accuracy
            masked_lm_labels_flat = labels.view(-1)
            mlm_labels = masked_lm_labels_flat[masked_lm_labels_flat != 0]
            if not self.dense_seq_output:
                prediction_scores_flat = rearrange(prediction_scores, '... v -> (...) v')
                mlm_predictions_scores = prediction_scores_flat[masked_lm_labels_flat != 0]
                mlm_predictions = mlm_predictions_scores.argmax(dim=-1)
            else:
                mlm_predictions = prediction_scores.argmax(dim=-1)
            mlm_acc = (mlm_predictions == mlm_labels).sum(dtype=torch.float) / mlm_labels.numel()

            return total_loss, prediction_scores, seq_relationship_score, mlm_acc, mlm_labels.numel()
        else:
            return prediction_scores, seq_relationship_score


def state_dict_from_pretrained(model_name):
    from transformers.utils import WEIGHTS_NAME
    from transformers.utils.hub import cached_file
    return torch.load(cached_file(model_name, WEIGHTS_NAME))


def remap_state_dict(state_dict, config):
    # LayerNorm
    def key_mapping_ln_gamma_beta(key):
        key = re.sub(r'LayerNorm.gamma$', 'LayerNorm.weight', key)
        key = re.sub(r'LayerNorm.beta$', 'LayerNorm.bias', key)
        return key
    state_dict = OrderedDict((key_mapping_ln_gamma_beta(k), v) for k, v in state_dict.items())

    # Layers
    def key_mapping_layers(key):
        return re.sub(r'^bert.encoder.layer.', 'bert.encoder.layers.', key)
    state_dict = OrderedDict((key_mapping_layers(k), v) for k, v in state_dict.items())

    # LayerNorm
    def key_mapping_ln(key):
        key = re.sub(r'^bert.embeddings.LayerNorm.', 'bert.emb_ln.', key)
        key = re.sub(r'^bert.encoder.layers.(\d+).attention.output.LayerNorm.(weight|bias)',
                     r'bert.encoder.layers.\1.norm1.\2', key)
        key = re.sub(r'^bert.encoder.layers.(\d+).output.LayerNorm.(weight|bias)',
                     r'bert.encoder.layers.\1.norm2.\2', key)
        key = re.sub(r'^cls.predictions.transform.LayerNorm.(weight|bias)',
                     r'cls.predictions.transform.layer_norm.\1', key)
        return key
    state_dict = OrderedDict((key_mapping_ln(k), v) for k, v in state_dict.items())

    # MLP
    def key_mapping_mlp(key):
        key = re.sub(r'^bert.encoder.layers.(\d+).intermediate.dense.(weight|bias)',
                     r'bert.encoder.layers.\1.mlp.fc1.\2', key)
        key = re.sub(r'^bert.encoder.layers.(\d+).output.dense.(weight|bias)',
                     r'bert.encoder.layers.\1.mlp.fc2.\2', key)
        return key
    state_dict = OrderedDict((key_mapping_mlp(k), v) for k, v in state_dict.items())

    # Attention
    for d in range(config.num_hidden_layers):
        Wq = state_dict.pop(f'bert.encoder.layers.{d}.attention.self.query.weight')
        Wk = state_dict.pop(f'bert.encoder.layers.{d}.attention.self.key.weight')
        Wv = state_dict.pop(f'bert.encoder.layers.{d}.attention.self.value.weight')
        bq = state_dict.pop(f'bert.encoder.layers.{d}.attention.self.query.bias')
        bk = state_dict.pop(f'bert.encoder.layers.{d}.attention.self.key.bias')
        bv = state_dict.pop(f'bert.encoder.layers.{d}.attention.self.value.bias')
        state_dict[f'bert.encoder.layers.{d}.mixer.Wqkv.weight'] = torch.cat(
            [Wq, Wk, Wv], dim=0
        )
        state_dict[f'bert.encoder.layers.{d}.mixer.Wqkv.bias'] = torch.cat(
            [bq, bk, bv], dim=0
        )
    def key_mapping_attn(key):
        return re.sub(r'^bert.encoder.layers.(\d+).attention.output.dense.(weight|bias)',
                      r'bert.encoder.layers.\1.mixer.out_proj.\2', key)
    state_dict = OrderedDict((key_mapping_attn(k), v) for k, v in state_dict.items())

    def key_mapping_decoder_bias(key):
        return re.sub(r'^cls.predictions.bias', 'cls.predictions.decoder.bias', key)
    state_dict = OrderedDict((key_mapping_decoder_bias(k), v) for k, v in state_dict.items())

    return state_dict
