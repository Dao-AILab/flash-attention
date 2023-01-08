# Copyright (c) 2022, Tri Dao.
# Adapted from https://github.com/NVIDIA/Megatron-LM/blob/0bb597b42c53355a567aba2a1357cc34b9d99ddd/megatron/text_generation/forward_step.py#L31
from typing import Optional
import time

from dataclasses import dataclass, field
from collections import namedtuple

import torch
from torch import Tensor
from torch.profiler import profile, record_function, ProfilerActivity

from einops import rearrange

from transformers.generation import GreedySearchDecoderOnlyOutput, SampleDecoderOnlyOutput


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


# https://github.com/NVIDIA/Megatron-LM/blob/0bb597b42c53355a567aba2a1357cc34b9d99ddd/megatron/text_generation/sampling.py
# https://github.com/huggingface/transformers/blob/a44985b41cfa2de48a5e1de7f1f93b7483da25d1/src/transformers/generation/logits_process.py#L170
def modify_logits_for_top_p_filtering(logits, top_p):
    """Set the logits for none top-p values to -inf."""
    if top_p <= 0.0:
        return
    # First sort and calculate cumulative sum of probabilities.
    sorted_logits, sorted_indices = torch.sort(logits, descending=False)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
     # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
    sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
    # scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(indices_to_remove, float('-inf'))


def sample(logits, top_k=1, top_p=0.0, temperature=1.0):
    """Sample from top-k logits.
    Arguments:
        logits: Tensor of shape (batch_size, vocab_size)
    """
    if top_k == 1:  # Short-circuit for greedy decoding
        return logits.argmax(dim=-1)
    else:
        if top_p > 0.0:
            assert top_p <= 1.0, 'top-p should be in (0, 1].'
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))  # Safety check
            logits_top, indices = torch.topk(logits, top_k, dim=-1)
            logits_top /= temperature
            modify_logits_for_top_p_filtering(logits_top, top_p)
            return indices[
                torch.arange(indices.shape[0], device=indices.device),
                torch.multinomial(torch.softmax(logits_top, dim=-1), num_samples=1).squeeze(dim=-1)
            ]
        else:
            logits_top = logits / temperature
            modify_logits_for_top_p_filtering(logits_top, top_p)
            return torch.multinomial(torch.softmax(logits_top, dim=-1), num_samples=1).squeeze(dim=-1)


def decode(input_ids, model, max_length, top_k=1, top_p=0.0, temperature=1.0,
           fused_ft_kernel=False, cg=False, timing=False):
    """Decoding, either greedy or with top-k or top-p sampling.
    If top-k = 0, don't limit the number of candidates (pure sampling).
    Top-k and top-p can be used together. If top_k > 0 and top_p > 0, then top-k is applied first,
    then top-p.
    We assume that all sequences in the same batch have the same length.

    Arguments:
        input_ids: (batch, seq_len)
        max_length: int
    Returns: GreedySearchDecoderOnlyOutput or SampleDecoderOnlyOutput, with the following fields:
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
        next_token = sample(logits, top_k=top_k, top_p=top_p, temperature=temperature)
        sequences = [next_token]
        inference_params.sequence_len_offset = seqlen_og
        if cg:
            assert fused_ft_kernel
            run, cg_cache = capture_cg(model, inference_params, batch_size, seqlen_og, max_length)
        if timing:
            start = time.time()
        while True:
            position_ids = torch.full((batch_size, 1), inference_params.sequence_len_offset,
                                    dtype=torch.long, device=input_ids.device)
            if not cg:
                logits = model(rearrange(next_token, 'b -> b 1'), position_ids=position_ids,
                               inference_params=inference_params).logits[:, -1]
            else:
                logits = run(rearrange(next_token, 'b -> b 1'), position_ids,
                             inference_params.sequence_len_offset)
            scores.append(logits)
            next_token = sample(logits, top_k=top_k, temperature=temperature)
            sequences.append(next_token)
            inference_params.sequence_len_offset += 1
            if inference_params.sequence_len_offset >= max_length - 1:
                break
        if timing:
            print(f'Decoding time: {time.time() - start}')
    output_cls = GreedySearchDecoderOnlyOutput if top_k == 1 else SampleDecoderOnlyOutput
    return output_cls(
        sequences=torch.cat([input_ids, torch.stack(sequences, dim=1)], dim=1),
        scores=tuple(scores)
    )


class GenerationMixin:

    def generate(self, input_ids, max_length, top_k=1, top_p=0.0, temperature=1.0,
                 return_dict_in_generate=False, output_scores=False, **kwargs):
        output = decode(input_ids, self, max_length, top_k=top_k, top_p=top_p,
                        temperature=temperature, **kwargs)
        if not output_scores:
            output.scores = None
        return output if return_dict_in_generate else output.sequences


CgKey = namedtuple('CgKey', ['batch_size', 'seqlen_type', 'max_length'])
CgVal = namedtuple('CgVal', ['graph', 'input_ids', 'position_ids', 'lengths', 'logits'])


def seqlen_to_seqlen_type(seqlen: int) -> int:
    """Convert sequence length to a seqlen_type.
    This is used to determine which cuda graph to use.
    Arguments:
        seqlen: int
    """
    return 0 if seqlen < 32 else (1 if seqlen < 2048 else 2)


def seqlen_type_to_seqlen(seqlen_type: int) -> int:
    assert seqlen_type in [0, 1, 2]
    return 1 if seqlen_type == 0 else (32 if seqlen_type == 1 else 2048)


def capture_cg(model, inference_params, batch_size, seqlen_og, max_length, copy_output=False):
    """Build a cache of cuda graphs for decoding.
    Arguments:
        model: a GPTLMHeadModel
        batch_size: int
        seqlen_og: int. Length of the prompt.
        max_length: int
    TODO: how do we deal with the k_cache and v_cache memory? I think the CUDA graph also
    has to own the k_cache and v_cache?
    Here we assume that the model already has inference_params from the prompt processing.
    """
    assert max_length > seqlen_og
    cg_cache: dict[CgKey, CgVal] = {}
    device = next(iter(model.parameters())).device
    sequence_length_offset_og = inference_params.sequence_len_offset
    input_ids = torch.full((batch_size, 1), 0, dtype=torch.long, device=device)
    position_ids = torch.full((batch_size, 1), 0, dtype=torch.long, device=device)
    inference_params.lengths_per_sample = torch.full((batch_size,), seqlen_og, dtype=torch.int32,
                                                        device=device)

    memory_pool = None
    for s_type in range(seqlen_to_seqlen_type(seqlen_og), seqlen_to_seqlen_type(max_length) + 1):
        seqlen = max(seqlen_og, seqlen_type_to_seqlen(s_type))
        input_ids = torch.full((batch_size, 1), 0, dtype=torch.long, device=device)
        position_ids = torch.full((batch_size, 1), 0, dtype=torch.long, device=device)
        inference_params.lengths_per_sample[:] = seqlen
        inference_params.sequence_len_offset = seqlen
        g = torch.cuda.CUDAGraph()
        # Warmup before capture
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                logits = model(input_ids, position_ids=position_ids,
                               inference_params=inference_params).logits[:, -1]
        torch.cuda.current_stream().wait_stream(s)
        # Captures the graph
        # To allow capture, automatically sets a side stream as the current stream in the context
        with torch.cuda.graph(g, pool=memory_pool):
            logits = model(input_ids, position_ids=position_ids,
                            inference_params=inference_params).logits[:, -1]
        if memory_pool is None:
            memory_pool = g.pool()
        cg_cache[CgKey(batch_size, s_type, max_length)] = CgVal(
            g, input_ids, position_ids, inference_params.lengths_per_sample, logits
        )

    def run(new_input_ids, new_position_ids, seqlen):
        cg_val = cg_cache[CgKey(batch_size, seqlen_to_seqlen_type(seqlen), max_length)]
        inference_params.lengths_per_sample = cg_val.lengths
        inference_params.lengths_per_sample[:] = seqlen
        cg_val.input_ids.copy_(new_input_ids)
        cg_val.position_ids.copy_(new_position_ids)
        cg_val.graph.replay()
        output = cg_val.logits
        return output.clone() if copy_output else output

    inference_params.sequence_len_offset = sequence_length_offset_og

    return run, cg_cache
