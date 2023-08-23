import re
from collections import OrderedDict

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange
from flash_attn.models.bert import BertForPreTraining, BertModel, remap_state_dict
from flash_attn.utils.pretrained import state_dict_from_pretrained
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertForPreTraining as BertForPreTrainingHF
from transformers.models.bert.modeling_bert import BertModel as BertModelHF


@pytest.mark.parametrize("model_name", ["bert-base-uncased", "bert-large-uncased"])
# @pytest.mark.parametrize('model_name', ["bert-base-uncased"])
def test_bert_state_dict(model_name):
    config = BertConfig.from_pretrained(model_name)
    pretrained_state_dict = remap_state_dict(state_dict_from_pretrained(model_name), config)
    model = BertForPreTraining(config)
    state_dict = model.state_dict()
    assert state_dict.keys() == pretrained_state_dict.keys()
    for k in state_dict.keys():
        assert state_dict[k].shape == pretrained_state_dict[k].shape


def get_hf_models(model_name, config, dtype):
    pretrained_state_dict = state_dict_from_pretrained(model_name)

    def key_mapping_ln_gamma_beta(key):
        key = re.sub(r"LayerNorm.gamma$", "LayerNorm.weight", key)
        key = re.sub(r"LayerNorm.beta$", "LayerNorm.bias", key)
        return key

    pretrained_state_dict = OrderedDict(
        (key_mapping_ln_gamma_beta(k), v) for k, v in pretrained_state_dict.items()
    )
    model_hf = BertForPreTrainingHF(config)
    # Missing key(s) in state_dict: "bert.embeddings.position_ids", "cls.predictions.decoder.bias"
    # position_ids is a buffer, and predictions.decoder.bias is tied to predictions.bias.
    model_hf.load_state_dict(pretrained_state_dict, strict=False)
    model_hf.cuda().to(dtype=dtype)
    return model_hf


@pytest.mark.parametrize('model_name', ["bert-base-uncased"])
def test_bert_non_optimized(model_name):
    """Check that our implementation of BERT (without any optimizations enabled) matches the
    HF implementation: the output of our forward pass in fp16 should be around the same as the HF
    forward pass in fp16, when compared to the HF forward pass in fp32.
    """
    dtype = torch.float16
    config = BertConfig.from_pretrained(model_name)

    model = BertForPreTraining.from_pretrained(model_name, config)
    model = model.cuda().to(dtype=dtype)

    model_ref = get_hf_models(model_name, config, torch.float32)
    model_hf = get_hf_models(model_name, config, dtype)

    model.eval()
    model_ref.eval()
    model_hf.eval()

    torch.manual_seed(0)
    batch_size = 4
    max_seqlen = 512
    seqlens = torch.randint(max_seqlen // 2, max_seqlen + 1, (batch_size,), device="cuda")
    attention_mask = torch.arange(max_seqlen, device="cuda")[None, :] < seqlens[:, None]
    input_ids = torch.randint(
        0, config.vocab_size, (batch_size, max_seqlen), dtype=torch.long, device="cuda"
    )
    out = model.bert(input_ids, attention_mask=attention_mask)
    sequence_output, pooled_output = out.last_hidden_state, out.pooler_output
    out_hf = model_hf.bert(input_ids, attention_mask=attention_mask)
    sequence_output_hf, pooled_output_hf = out_hf.last_hidden_state, out_hf.pooler_output
    out_ref = model_ref.bert(input_ids, attention_mask=attention_mask)
    sequence_output_ref, pooled_output_ref = out_ref.last_hidden_state, out_ref.pooler_output

    print(f"Output max diff: {(sequence_output - sequence_output_ref).abs().max().item()}")
    print(f"Output mean diff: {(sequence_output - sequence_output_ref).abs().mean().item()}")
    print(f"HF fp16 max diff: {(sequence_output_hf - sequence_output_ref).abs().max().item()}")
    print(f"HF fp16 mean diff: {(sequence_output_hf - sequence_output_ref).abs().mean().item()}")
    assert (sequence_output - sequence_output_ref).abs().max().item() < 3 * (
        sequence_output_hf - sequence_output_ref
    ).abs().max().item()
    assert (pooled_output - pooled_output_ref).abs().max().item() < 3 * (
        pooled_output_hf - pooled_output_ref
    ).abs().max().item()


@pytest.mark.parametrize("model_name", ["bert-base-uncased", "bert-large-uncased"])
# @pytest.mark.parametrize('model_name', ["bert-base-uncased"])
def test_bert_optimized(model_name):
    """Check that our implementation of BERT (with all optimizations enabled) matches the
    HF implementation: the output of our forward pass in fp16 should be around the same as the HF
    forward pass in fp16, when compared to the HF forward pass in fp32.
    """
    dtype = torch.float16
    config = BertConfig.from_pretrained(model_name)
    # Our implementation of fused_mlp assumes the activation is
    # nn.GELU(approximate='tanh'). Huggingface calls it "gelu_new" or "gelu_fast".
    # If you just want "gelu", disable fused_mlp.
    config.hidden_act = "gelu_new"
    config.use_flash_attn = True
    config.fused_bias_fc = True
    config.fused_mlp = True
    config.fused_dropout_add_ln = True

    model = BertForPreTraining.from_pretrained(model_name, config)
    model = model.cuda().to(dtype=dtype)

    model_ref = get_hf_models(model_name, config, torch.float32)
    model_hf = get_hf_models(model_name, config, dtype)

    model.eval()
    model_ref.eval()
    model_hf.eval()

    torch.manual_seed(0)
    batch_size = 4
    max_seqlen = 512
    seqlens = torch.randint(max_seqlen // 2, max_seqlen + 1, (batch_size,), device="cuda")
    attention_mask = torch.arange(max_seqlen, device="cuda")[None, :] < seqlens[:, None]
    input_ids = torch.randint(
        0, config.vocab_size, (batch_size, max_seqlen), dtype=torch.long, device="cuda"
    )
    out = model.bert(input_ids, attention_mask=attention_mask)
    sequence_output, pooled_output = out.last_hidden_state, out.pooler_output
    out_hf = model_hf.bert(input_ids, attention_mask=attention_mask)
    sequence_output_hf, pooled_output_hf = out_hf.last_hidden_state, out_hf.pooler_output
    # Need to zero out the padded tokens in the sequence before comparison.
    sequence_output_hf[~attention_mask, :] = 0.0
    out_ref = model_ref.bert(input_ids, attention_mask=attention_mask)
    sequence_output_ref, pooled_output_ref = out_ref.last_hidden_state, out_ref.pooler_output
    sequence_output_ref[~attention_mask, :] = 0.0

    print(
        f"BertModel output max diff: {(sequence_output - sequence_output_ref).abs().max().item()}"
    )
    print(
        f"BertModel output mean diff: {(sequence_output - sequence_output_ref).abs().mean().item()}"
    )
    print(
        f"HF fp16 BertModel max diff: {(sequence_output_hf - sequence_output_ref).abs().max().item()}"
    )
    print(
        f"HF fp16 BertModel mean diff: {(sequence_output_hf - sequence_output_ref).abs().mean().item()}"
    )
    assert (sequence_output - sequence_output_ref).abs().max().item() < 4 * (
        sequence_output_hf - sequence_output_ref
    ).abs().max().item()
    assert (pooled_output - pooled_output_ref).abs().max().item() < 4 * (
        pooled_output_hf - pooled_output_ref
    ).abs().max().item()

    out = model(input_ids, attention_mask=attention_mask)
    prediction_scores, seq_relationship_scores = out.prediction_logits, out.seq_relationship_logits
    # Need to zero out the padded tokens in the sequence before comparison.
    prediction_scores = prediction_scores.clone()
    prediction_scores[~attention_mask, :] = 0.0
    out_hf = model_hf(input_ids, attention_mask=attention_mask)
    prediction_scores_hf, seq_relationship_scores_hf = (
        out_hf.prediction_logits,
        out_hf.seq_relationship_logits,
    )
    prediction_scores_hf[~attention_mask, :] = 0.0
    out_ref = model_ref(input_ids, attention_mask=attention_mask)
    prediction_scores_ref, seq_relationship_scores_ref = (
        out_ref.prediction_logits,
        out_ref.seq_relationship_logits,
    )
    prediction_scores_ref[~attention_mask, :] = 0.0

    print(
        f"prediction_scores max diff: {(prediction_scores - prediction_scores_ref).abs().max().item()}"
    )
    print(
        f"prediction_scores mean diff: {(prediction_scores - prediction_scores_ref).abs().mean().item()}"
    )
    print(
        f"HF fp16 prediction_scoresff: {(prediction_scores_hf - prediction_scores_ref).abs().max().item()}"
    )
    print(
        f"HF fp16 prediction_scoresiff: {(prediction_scores_hf - prediction_scores_ref).abs().mean().item()}"
    )
    assert (prediction_scores - prediction_scores_ref).abs().max().item() < 2 * (
        prediction_scores_hf - prediction_scores_ref
    ).abs().max().item()
    assert (seq_relationship_scores - seq_relationship_scores_ref).abs().max().item() < 2 * (
        seq_relationship_scores_hf - seq_relationship_scores_ref
    ).abs().max().item()


@pytest.mark.parametrize("last_layer_subset", [False, True])
# @pytest.mark.parametrize('last_layer_subset', [True])
@pytest.mark.parametrize("has_key_padding_mask", [True, False])
# @pytest.mark.parametrize('has_key_padding_mask', [True])
@pytest.mark.parametrize("model_name", ["bert-base-uncased", "bert-large-uncased"])
# @pytest.mark.parametrize('model_name', ["bert-base-uncased"])
def test_bert_dense_seq_output(model_name, has_key_padding_mask, last_layer_subset):
    """Check that our implementation of BERT (with all optimizations enabled) matches the
    HF implementation: the output of our forward pass in fp16 should be around the same as the HF
    forward pass in fp16, when compared to the HF forward pass in fp32.
    """
    dtype = torch.float16
    config = BertConfig.from_pretrained(model_name)
    # Our implementation of fused_mlp assumes the activation is
    # nn.GELU(approximate='tanh'). Huggingface calls it "gelu_new" or "gelu_fast".
    # If you just want "gelu", disable fused_mlp.
    config.hidden_act = "gelu_new"
    config.use_flash_attn = True
    config.fused_bias_fc = True
    config.fused_mlp = True
    config.fused_dropout_add_ln = True
    config.dense_seq_output = True
    config.last_layer_subset = last_layer_subset
    config.use_xentropy = True

    model = BertForPreTraining.from_pretrained(model_name, config)
    model = model.cuda().to(dtype=dtype)

    model_ref = get_hf_models(model_name, config, torch.float32)
    model_hf = get_hf_models(model_name, config, dtype)

    model.eval()
    model_ref.eval()
    model_hf.eval()

    torch.manual_seed(0)
    batch_size = 4
    max_seqlen = 512
    seqlens = torch.randint(max_seqlen // 2, max_seqlen + 1, (batch_size,), device="cuda")
    if has_key_padding_mask:
        attention_mask = torch.arange(max_seqlen, device="cuda")[None, :] < seqlens[:, None]
    else:
        attention_mask = None
    input_ids = torch.randint(
        0, config.vocab_size, (batch_size, max_seqlen), dtype=torch.long, device="cuda"
    )
    labels = torch.randint(
        0, config.vocab_size, (batch_size, max_seqlen), dtype=torch.long, device="cuda"
    )
    if attention_mask is not None:
        labels[~attention_mask] = 0
    labels[(torch.rand(batch_size, max_seqlen, device="cuda") > 0.15)] = 0
    masked_tokens_mask = labels.flatten() > 0
    next_sequence_label = torch.randint(0, 2, (batch_size,), device="cuda")

    out = model(
        input_ids,
        attention_mask=attention_mask,
        labels=labels,
        next_sentence_label=next_sequence_label,
    )
    prediction_scores, seq_relationship_scores = out.prediction_logits, out.seq_relationship_logits
    out_hf = model_hf(
        input_ids,
        attention_mask=attention_mask,
        labels=labels,
        next_sentence_label=next_sequence_label,
    )
    prediction_scores_hf, seq_relationship_scores_hf = (
        out_hf.prediction_logits,
        out_hf.seq_relationship_logits,
    )
    prediction_scores_hf = rearrange(prediction_scores_hf, "b s d -> (b s) d")[masked_tokens_mask]
    out_ref = model_ref(
        input_ids,
        attention_mask=attention_mask,
        labels=labels,
        next_sentence_label=next_sequence_label,
    )
    prediction_scores_ref, seq_relationship_scores_ref = (
        out_ref.prediction_logits,
        out_ref.seq_relationship_logits,
    )
    prediction_scores_ref = rearrange(prediction_scores_ref, "b s d -> (b s) d")[masked_tokens_mask]

    print(
        f"prediction_scores max diff: {(prediction_scores - prediction_scores_ref).abs().max().item()}"
    )
    print(
        f"prediction_scores mean diff: {(prediction_scores - prediction_scores_ref).abs().mean().item()}"
    )
    print(
        f"HF fp16 prediction_scoresff: {(prediction_scores_hf - prediction_scores_ref).abs().max().item()}"
    )
    print(
        f"HF fp16 prediction_scoresiff: {(prediction_scores_hf - prediction_scores_ref).abs().mean().item()}"
    )
    assert (prediction_scores - prediction_scores_ref).abs().max().item() < 2 * (
        prediction_scores_hf - prediction_scores_ref
    ).abs().max().item()
    assert (seq_relationship_scores - seq_relationship_scores_ref).abs().max().item() < 2 * (
        seq_relationship_scores_hf - seq_relationship_scores_ref
    ).abs().max().item()
    # The loss calculation from HF is wrong: it doesn't ignore the labels that are 0.
    # assert (out.loss - out_ref.loss).abs().max().item() < 2 * (out_hf.loss - out_ref.loss).abs().max().item()
