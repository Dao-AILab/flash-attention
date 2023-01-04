# Copyright (c) 2022, Tri Dao.
# Adapted from https://github.com/NVIDIA/Megatron-LM/blob/0bb597b42c53355a567aba2a1357cc34b9d99ddd/megatron/text_generation/forward_step.py#L31
from typing import Optional

from dataclasses import dataclass, field
import torch
from torch import Tensor

from einops import rearrange

from transformers.generation import GreedySearchDecoderOnlyOutput


@dataclass
class InferenceParams:
    """Inference parameters that are passed to the main model in order
    to efficienly calculate and store the context during inference."""
    max_sequence_len: int
    max_batch_size: int
    sequence_len_offset: int = 0
    batch_size_offset: int = 0
    key_value_memory_dict: dict = field(default_factory=dict)
    fused_ft_kernel: bool = False
    lengths_per_sample: Optional[Tensor] = None


def greedy_decode(input_ids, model, max_length, fused_ft_kernel=True):
    """Greedy decoding. This is a very simple implementation.
    We assume that all sequences in the same batch have the same length.
    Arguments:
        input_ids: (batch, seq_len)
        max_length: int
    Returns: GreedySearchDecoderOnlyOutput, with the following fields:
        sequences: (batch, max_length)
        scores: tuples of (batch, vocab_size)
    """
    batch_size, seqlen_og = input_ids.shape
    inference_params = InferenceParams(max_sequence_len=max_length, max_batch_size=batch_size,
                                       fused_ft_kernel=fused_ft_kernel)
    scores = []
    with torch.inference_mode():
        logits = model(input_ids, inference_params=inference_params).logits[:, -1]
        scores.append(logits)
        next_token = logits.argmax(dim=-1)
        sequences = [next_token]
        inference_params.sequence_len_offset = seqlen_og
        while True:
            position_ids = torch.full((batch_size, 1), inference_params.sequence_len_offset,
                                      dtype=torch.long, device=input_ids.device)
            logits = model(rearrange(next_token, 'b -> b 1'), position_ids=position_ids,
                           inference_params=inference_params).logits[:, -1]
            scores.append(logits)
            next_token = logits.argmax(dim=-1)
            sequences.append(next_token)
            inference_params.sequence_len_offset += 1
            if inference_params.sequence_len_offset >= max_length - 1:
                break
    return GreedySearchDecoderOnlyOutput(
        sequences=torch.cat([input_ids, torch.stack(sequences, dim=1)], dim=1),
        scores=tuple(scores)
    )


class GenerationMixin:

    def generate(self, input_ids, max_length, return_dict_in_generate=False, output_scores=False,
                 **kwargs):
        output = greedy_decode(input_ids, self, max_length, **kwargs)
        if not output_scores:
            output.scores = None
        return output if return_dict_in_generate else output.sequences
