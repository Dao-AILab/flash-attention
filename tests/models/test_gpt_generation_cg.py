import os
import re
import time

import torch
import pytest

from einops import rearrange

from transformers import GPT2Config

from flash_attn.models.gpt import GPTLMHeadModel
from flash_attn.utils.generation import update_graph_cache


def get_logits(model, input_ids, max_length, teacher_outputs=None, **kwargs):
    out = model.generate(input_ids=input_ids, max_length=max_length, fused_ft_kernel=True,
                         teacher_outputs=teacher_outputs, return_dict_in_generate=True,
                         output_scores=True, timing=True, **kwargs)
    return torch.stack(out.scores, dim=1)


@pytest.mark.parametrize('seqlen,maxlen', [(10, 20), (30, 150), (3000, 3400), (14000, 15000)])
# @pytest.mark.parametrize('seqlen,maxlen', [(10, 20)])
@pytest.mark.parametrize('rotary', [None, "interleaved", "block"])
# @pytest.mark.parametrize('rotary', [None])
@pytest.mark.parametrize('model_name', ["gpt2"])
def test_greedy_decode_gpt2_cg(model_name, rotary, seqlen, maxlen):
    """Check that decoding with CUDA graph is the same as decoding without CUDA graph.
    """
    dtype = torch.float16
    device = 'cuda'
    rtol, atol = 3e-3, 3e-1
    config = GPT2Config.from_pretrained(model_name)
    config.n_positions = 16 * 1024
    assert seqlen <= maxlen <= config.n_positions
    if rotary is not None:
        config.n_positions = 0
        config.rotary_emb_dim = 32
        config.rotary_emb_interleaved = rotary == "interleaved"
    config.residual_in_fp32 = True
    config.use_flash_attn = True
    config.fused_bias_fc = True
    config.fused_mlp = True
    config.fused_dropout_add_ln = True

    model = GPTLMHeadModel(config, device=device, dtype=dtype)
    model.eval()

    torch.manual_seed(0)
    batch_size = 1
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seqlen), dtype=torch.long,
                              device=device)
    teacher_outputs = torch.randint(0, config.vocab_size, (batch_size, maxlen), dtype=torch.long,
                                    device=device)

    logits = get_logits(model, input_ids, maxlen, teacher_outputs=teacher_outputs)
    logits_cg = get_logits(model, input_ids, maxlen, teacher_outputs=teacher_outputs, cg=True)
    assert torch.equal(logits, logits_cg)

    # Try increasing batch size and seqlen, then decrease them to see if it's still correct
    batch_size = 3
    maxlen += 30
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seqlen), dtype=torch.long,
                              device=device)
    teacher_outputs = torch.randint(0, config.vocab_size, (batch_size, maxlen), dtype=torch.long,
                                    device=device)
    logits = get_logits(model, input_ids, maxlen, teacher_outputs=teacher_outputs)
    logits_cg = get_logits(model, input_ids, maxlen, teacher_outputs=teacher_outputs, cg=True)
    assert torch.equal(logits, logits_cg)

    batch_size = 2
    maxlen -= 35
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seqlen), dtype=torch.long,
                              device=device)
    teacher_outputs = torch.randint(0, config.vocab_size, (batch_size, maxlen), dtype=torch.long,
                                    device=device)
    logits = get_logits(model, input_ids, maxlen, teacher_outputs=teacher_outputs)
    logits_cg = get_logits(model, input_ids, maxlen, teacher_outputs=teacher_outputs, cg=True)
    assert torch.equal(logits, logits_cg)
