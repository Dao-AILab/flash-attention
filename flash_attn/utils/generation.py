# Copyright (c) 2023, Tri Dao.
# Adapted from https://github.com/NVIDIA/Megatron-LM/blob/0bb597b42c53355a567aba2a1357cc34b9d99ddd/megatron/text_generation/forward_step.py#L31
from typing import Optional, Union, Sequence, Callable
import gc
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
           eos_token_id=None, vocab_size=None, tensor_parallel=1, fused_ft_kernel=False,
           cg=False, timing=False):
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
    if cg:
        assert fused_ft_kernel
        if not hasattr(model, '_decoding_cache'):
            model._decoding_cache = None
        model._decoding_cache = update_graph_cache(
            model, model._decoding_cache, batch_size, seqlen_og, max_length,
            tensor_parallel=tensor_parallel
        )
        inference_params = model._decoding_cache.inference_params
        inference_params.max_sequence_len = max_length
        inference_params.max_batch_size = batch_size
        inference_params.sequence_len_offset = 0
    else:
        inference_params = InferenceParams(max_sequence_len=max_length, max_batch_size=batch_size,
                                           fused_ft_kernel=fused_ft_kernel)
    scores = []
    with torch.inference_mode():
        logits = model(input_ids, inference_params=inference_params).logits[:, -1]
        if timing:
            torch.cuda.synchronize()
            start = time.time()
        if vocab_size is not None:
            logits = logits[..., :vocab_size]
        scores.append(logits)
        next_token = sample(logits, top_k=top_k, top_p=top_p, temperature=temperature)
        sequences = [next_token]
        inference_params.sequence_len_offset = seqlen_og
        while True:
            position_ids = torch.full((batch_size, 1), inference_params.sequence_len_offset,
                                    dtype=torch.long, device=input_ids.device)
            if not cg:
                logits = model(rearrange(next_token, 'b -> b 1'), position_ids=position_ids,
                               inference_params=inference_params).logits[:, -1]
            else:
                logits = model._decoding_cache.run(rearrange(next_token, 'b -> b 1'), position_ids,
                                                   inference_params.sequence_len_offset)
            if vocab_size is not None:
                logits = logits[..., :vocab_size]
            scores.append(logits)
            next_token = sample(logits, top_k=top_k, temperature=temperature)
            sequences.append(next_token)
            inference_params.sequence_len_offset += 1
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
            if inference_params.sequence_len_offset >= max_length - 1:
                break
        if timing:
            torch.cuda.synchronize()
            print(f'Decoding time: {(time.time() - start) * 1000:.0f}ms')
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


def allocate_kv_cache(max_batch_size, max_seqlen, nheads, headdim, layers: Union[int, Sequence],
                      device, dtype=torch.float16):
    assert dtype in [torch.float16, torch.bfloat16, torch.float32]
    packsize = 4 if dtype == torch.float32 else 8
    assert headdim % packsize == 0
    k_cache_shape = (max_batch_size, nheads, headdim // packsize, max_seqlen, packsize)
    v_cache_shape = (max_batch_size, nheads, max_seqlen, headdim)
    if isinstance(layers, int):
        layers = range(layers)
    return {i: (torch.empty(k_cache_shape, device=device, dtype=dtype),
                torch.empty(v_cache_shape, device=device, dtype=dtype))
            for i in layers}


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


@dataclass
class DecodingCGCache:
    max_batch_size: int = 0
    max_seqlen: int = 0
    device = None
    dtype = None
    callables: dict = field(default_factory=dict)
    mempool = None
    inference_params: Optional[InferenceParams] = None
    run: Optional[Callable] = None


@torch.inference_mode()
def update_graph_cache(model, cache, batch_size, seqlen_og, max_seqlen, tensor_parallel=1,
                       dtype=None):
    if cache is None:
        cache = DecodingCGCache()
    param_example = next(iter(model.parameters()))
    device = param_example.device
    if dtype is None:
        dtype = param_example.dtype
    if ((device, dtype) != (cache.device, cache.dtype) or batch_size > cache.max_batch_size
        or max_seqlen > cache.max_seqlen):  # Invalidate the cache
        cache.callables = {}
        cache.mempool = None
        cache.inference_params = None
        gc.collect()
        cache.device, cache.dtype = device, dtype
        cache.max_batch_size, cache.max_seqlen = batch_size, max_seqlen
        headdim = getattr(model.config, 'head_dim',
                          model.config.hidden_size // model.config.num_attention_heads)
        kv_cache = allocate_kv_cache(
            batch_size, max_seqlen, model.config.num_attention_heads // tensor_parallel, headdim,
            model.config.num_hidden_layers, device, dtype
        )
        lengths_per_sample = torch.full((batch_size,), seqlen_og, dtype=torch.int32, device=device)
        cache.inference_params = InferenceParams(
            max_sequence_len=max_seqlen, max_batch_size=batch_size,
            sequence_len_offset=seqlen_og, key_value_memory_dict=kv_cache, fused_ft_kernel=True,
            lengths_per_sample=lengths_per_sample
        )
        cache.mempool = torch.cuda.graphs.graph_pool_handle()
    for s_type in range(seqlen_to_seqlen_type(seqlen_og), seqlen_to_seqlen_type(max_seqlen) + 1):
        if s_type not in cache.callables:
            seqlen = min(max(seqlen_og, seqlen_type_to_seqlen(s_type)), max_seqlen)
            cache.callables[s_type] = capture_graph(
                model, cache.inference_params, batch_size, seqlen_og, seqlen, mempool=cache.mempool
            )

    def dispatch(input_ids, position_ids, seqlen):
        return cache.callables[seqlen_to_seqlen_type(seqlen)](input_ids, position_ids, seqlen)

    cache.run = dispatch
    cache.inference_params.sequence_length_offset = 0  # Reset so it's not confusing
    return cache


def capture_graph(model, inference_params, batch_size, seqlen_og, max_seqlen, mempool=None):
    assert max_seqlen >= seqlen_og
    device = next(iter(model.parameters())).device
    input_ids = torch.full((batch_size, 1), 0, dtype=torch.long, device=device)
    position_ids = torch.full((batch_size, 1), 0, dtype=torch.long, device=device)
    inference_params.lengths_per_sample[:] = seqlen_og

    # Warmup before capture
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            logits = model(input_ids, position_ids=position_ids,
                           inference_params=inference_params).logits[:, -1]
        s.synchronize()
    torch.cuda.current_stream().wait_stream(s)
    # Captures the graph
    # To allow capture, automatically sets a side stream as the current stream in the context
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=mempool):
        logits = model(input_ids, position_ids=position_ids,
                        inference_params=inference_params).logits[:, -1]

    def run(new_input_ids, new_position_ids, seqlen):
        inference_params.lengths_per_sample[:] = seqlen
        input_ids.copy_(new_input_ids)
        position_ids.copy_(new_position_ids)
        graph.replay()
        return logits

    return run
