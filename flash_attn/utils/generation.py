# Copyright (c) 2023, Tri Dao.
# Adapted from https://github.com/NVIDIA/Megatron-LM/blob/0bb597b42c53355a567aba2a1357cc34b9d99ddd/megatron/text_generation/forward_step.py#L31
import gc
import time
from collections import namedtuple
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Optional, Sequence, Union

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor
from torch.profiler import ProfilerActivity, profile, record_function
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
# https://github.com/huggingface/transformers/blob/a44985b41cfa2de48a5e1de7f1f93b7483da25d1/src/transformers/generation/logits_process.py#L231
def modify_logits_for_top_k_filtering(logits, top_k):
    """Set the logits for none top-k values to -inf. Done in-place."""
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits.masked_fill_(indices_to_remove, float("-Inf"))


# https://github.com/NVIDIA/Megatron-LM/blob/0bb597b42c53355a567aba2a1357cc34b9d99ddd/megatron/text_generation/sampling.py
# https://github.com/huggingface/transformers/blob/a44985b41cfa2de48a5e1de7f1f93b7483da25d1/src/transformers/generation/logits_process.py#L170
def modify_logits_for_top_p_filtering(logits, top_p):
    """Set the logits for none top-p values to -inf. Done in-place."""
    if top_p <= 0.0 or top_p >= 1.0:
        return
    # First sort and calculate cumulative sum of probabilities.
    sorted_logits, sorted_indices = torch.sort(logits, descending=False)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
    sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
    # scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove
    )
    logits.masked_fill_(indices_to_remove, float("-inf"))


def sample(logits, top_k=1, top_p=0.0, temperature=1.0):
    """Sample from top-k logits.
    Arguments:
        logits: Tensor of shape (batch_size, vocab_size)
    """
    if top_k == 1:  # Short-circuit for greedy decoding
        return logits.argmax(dim=-1)
    else:
        if top_p > 0.0:
            assert top_p <= 1.0, "top-p should be in (0, 1]."
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))  # Safety check
            logits_top, indices = torch.topk(logits, top_k, dim=-1)
            if temperature != 1.0:
                logits_top /= temperature
            modify_logits_for_top_p_filtering(logits_top, top_p)
            return indices[
                torch.arange(indices.shape[0], device=indices.device),
                torch.multinomial(torch.softmax(logits_top, dim=-1), num_samples=1).squeeze(dim=-1),
            ]
        else:
            # Clone so that when we modify for top_p we don't change the original logits
            logits_top = logits / temperature if temperature != 1.0 else logits.clone()
            modify_logits_for_top_p_filtering(logits_top, top_p)
            return torch.multinomial(torch.softmax(logits_top, dim=-1), num_samples=1).squeeze(
                dim=-1
            )


def decode(
    input_ids,
    model,
    max_length,
    top_k=1,
    top_p=0.0,
    temperature=1.0,
    eos_token_id=None,
    teacher_outputs=None,
    vocab_size=None,
    tensor_parallel=1,
    fused_ft_kernel=False,
    cg=False,
    timing=False,
):
    """Decoding, either greedy or with top-k or top-p sampling.
    If top-k = 0, don't limit the number of candidates (pure sampling).
    Top-k and top-p can be used together. If top_k > 0 and top_p > 0, then top-k is applied first,
    then top-p.
    We assume that all sequences in the same batch have the same length.

    Arguments:
        input_ids: (batch, seq_len)
        max_length: int
        teacher_outputs (optional): (batch, seq_len). If provided, instead of sampling from the
            logits, the next token is taken from the teacher_outputs. Useful for testing.
    Returns: GreedySearchDecoderOnlyOutput or SampleDecoderOnlyOutput, with the following fields:
        sequences: (batch, max_length)
        scores: tuples of (batch, vocab_size)
    """
    batch_size, seqlen_og = input_ids.shape
    teacher_output_len = teacher_outputs.shape[1] if teacher_outputs is not None else 0
    if cg:
        assert fused_ft_kernel
        if not hasattr(model, "_decoding_cache"):
            model._decoding_cache = None
        model._decoding_cache = update_graph_cache(
            model,
            model._decoding_cache,
            batch_size,
            seqlen_og,
            max_length,
            tensor_parallel=tensor_parallel,
        )
        inference_params = model._decoding_cache.inference_params
        inference_params.max_sequence_len = max_length
        inference_params.max_batch_size = batch_size
        inference_params.sequence_len_offset = 0
    else:
        inference_params = InferenceParams(
            max_sequence_len=max_length, max_batch_size=batch_size, fused_ft_kernel=fused_ft_kernel
        )

    def logits_forward_fn(input_ids, position_ids, inference_params):
        if not cg:
            return model(
                input_ids,
                position_ids=position_ids,
                inference_params=inference_params,
                num_last_tokens=1,
            ).logits.squeeze(dim=1)
        else:
            return model._decoding_cache.run(
                input_ids, position_ids, inference_params.sequence_len_offset
            ).clone()

    logits_postprocess_fn = (
        lambda logits: logits[..., :vocab_size] if vocab_size is not None else logits
    )

    scores = []
    with torch.inference_mode():
        if timing:
            if tensor_parallel > 1:
                torch.distributed.barrier()
            torch.cuda.synchronize()
            start = time.time()
        logits = model(
            input_ids, inference_params=inference_params, num_last_tokens=1
        ).logits.squeeze(dim=1)
        logits = logits_postprocess_fn(logits)
        scores.append(logits if not cg else logits.clone())
        if teacher_outputs is None or teacher_output_len <= seqlen_og:
            next_token = sample(logits, top_k=top_k, top_p=top_p, temperature=temperature)
        else:
            next_token = teacher_outputs[:, seqlen_og]
        sequences = [next_token]
        inference_params.sequence_len_offset = seqlen_og
        while True:
            position_ids = torch.full(
                (batch_size, 1),
                inference_params.sequence_len_offset,
                dtype=torch.long,
                device=input_ids.device,
            )
            logits = logits_postprocess_fn(
                logits_forward_fn(rearrange(next_token, "b -> b 1"), position_ids, inference_params)
            )
            scores.append(logits)
            if (
                teacher_outputs is None
                or teacher_output_len <= inference_params.sequence_len_offset + 1
            ):
                next_token = sample(logits, top_k=top_k, top_p=top_p, temperature=temperature)
            else:
                next_token = teacher_outputs[:, inference_params.sequence_len_offset + 1]
            sequences.append(next_token)
            inference_params.sequence_len_offset += 1
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
            if inference_params.sequence_len_offset >= max_length - 1:
                break
        if timing:
            if tensor_parallel > 1:
                torch.distributed.barrier()
            torch.cuda.synchronize()
            print(f"Prompt processing + decoding time: {(time.time() - start) * 1000:.0f}ms")
    output_cls = GreedySearchDecoderOnlyOutput if top_k == 1 else SampleDecoderOnlyOutput
    return output_cls(
        sequences=torch.cat([input_ids, torch.stack(sequences, dim=1)], dim=1), scores=tuple(scores)
    )


def sample_speculative(logits, logits_draft, tokens_draft, top_k=1, top_p=0.0, temperature=1.0):
    """Algorithm 1 from [1]
    [1] Fast Inference from Transformers via Speculative Decoding
    Yaniv Leviathan, Matan Kalman, Yossi Matias
    https://arxiv.org/abs/2211.17192

    Arguments:
        logits: Tensor of shape (batch_size, seqlen + 1, vocab_size)
        logits_draft: Tensor of shape (batch_size, seqlen, vocab_size)
        tokens_draft: Tensor of shape (batch_size, seqlen)
    Return:
        tokens: Tensor of shape (batch_size, seqlen + 1)
        num_generated_tokens: Tensor of shape (batch_size), with value in [1, seqlen + 1].
            For each sequence in the batch, the number of valid tokens that were sampled by
            speculative sampling.
    """
    batch, seqlen_p_1, vocab_size = logits.shape
    seqlen = seqlen_p_1 - 1
    assert logits_draft.shape == (batch, seqlen, vocab_size)
    assert tokens_draft.shape == (batch, seqlen)
    assert tokens_draft.dtype in [torch.int64, torch.int32]
    # TODO: if top_k = 1 we can simplify things and only work with indices
    if top_p > 0.0:
        assert top_p <= 1.0, "top-p should be in (0, 1]."
    # Clone so that when we modify for top_p we don't change the original logits
    logits = logits / temperature if temperature != 1.0 else logits.clone()
    logits_draft = logits_draft / temperature if temperature != 1.0 else logits_draft.clone()
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))  # Safety check
        modify_logits_for_top_k_filtering(logits, top_k)
        modify_logits_for_top_k_filtering(logits_draft, top_k)
    modify_logits_for_top_p_filtering(logits, top_p)
    modify_logits_for_top_p_filtering(logits_draft, top_p)
    probs = torch.softmax(logits, dim=-1)
    probs_draft = torch.softmax(logits_draft, dim=-1)
    gather = lambda probs, tokens: rearrange(
        probs.gather(dim=-1, index=rearrange(tokens, "... -> ... 1")), "... 1 -> ..."
    )
    # (batch, seqlen)
    accepted = torch.rand(batch, seqlen, device=probs.device) * gather(
        probs_draft, tokens_draft
    ) <= gather(probs[:, :-1], tokens_draft)
    accepted_all = accepted.all(dim=-1)
    # (batch,)
    first_rejected_idx = torch.where(accepted_all, seqlen, accepted.int().argmin(dim=-1))
    probs_diff = torch.clamp(probs[:, :-1] - probs_draft, min=0.0)
    # torch.multinomial can deal with unnormalized probabilities
    # probs_diff /= probs_diff.sum(dim=-1, keepdim=True)
    resample_probs = torch.cat([probs_diff, probs[:, -1:]], dim=1)
    resample_probs = rearrange(
        resample_probs.gather(dim=1, index=repeat(first_rejected_idx, "b -> b 1 d", d=vocab_size)),
        "b 1 d -> b d",
    )
    resample = torch.multinomial(resample_probs, num_samples=1).squeeze(dim=-1)  # (batch,)
    tokens = F.pad(tokens_draft, (0, 1))
    tokens[:, first_rejected_idx] = resample
    return tokens, first_rejected_idx + 1


def decode_speculative(
    input_ids,
    model,
    model_draft,
    max_length,
    speculative_lookahead=3,
    top_k=1,
    top_p=0.0,
    temperature=1.0,
    eos_token_id=None,
    vocab_size=None,
    tensor_parallel=1,
    fused_ft_kernel=False,
    cg=False,
    timing=False,
    debug=False,
):
    """
    TD: WIP, for my own understanding, lightly tested. Only support batch_size == 1 for now.

    Speculative decoding, either greedy or with top-k or top-p sampling.
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
    assert batch_size == 1, "Speculative decoding implementation only supports batch_size=1"
    assert eos_token_id is None, "Speculative decoding implementation doesn't support eos_token_id"
    if cg:
        assert fused_ft_kernel
        if not hasattr(model_draft, "_decoding_cache"):
            model_draft._decoding_cache = None
        model_draft._decoding_cache = update_graph_cache(
            model_draft,
            model_draft._decoding_cache,
            batch_size,
            seqlen_og,
            max_length,
            tensor_parallel=tensor_parallel,
        )
        inference_params_draft = model_draft._decoding_cache.inference_params
        inference_params_draft.max_sequence_len = max_length
        inference_params_draft.max_batch_size = batch_size
        inference_params_draft.sequence_len_offset = 0
        # fused_ft_kernel doesn't support passing in multiple tokens at once
        inference_params = InferenceParams(
            max_sequence_len=max_length, max_batch_size=batch_size, fused_ft_kernel=False
        )
    else:
        inference_params_draft = InferenceParams(
            max_sequence_len=max_length, max_batch_size=batch_size, fused_ft_kernel=fused_ft_kernel
        )
        inference_params = InferenceParams(
            max_sequence_len=max_length, max_batch_size=batch_size, fused_ft_kernel=False
        )

    def logits_forward_fn(model, input_ids, position_ids, inference_params, cg=False):
        if not cg:
            return model(
                input_ids,
                position_ids=position_ids,
                inference_params=inference_params,
                num_last_tokens=1,
            ).logits.squeeze(dim=1)
        else:
            return model._decoding_cache.run(
                input_ids, position_ids, inference_params.sequence_len_offset
            ).clone()

    logits_postprocess_fn = (
        lambda logits: logits[..., :vocab_size] if vocab_size is not None else logits
    )

    def sample_tokens(
        input_ids, model, inference_params, sample_fn, num_tokens=1, cg=False, decoding=True,
        last_token_logits=False
    ):
        """Sample `num_tokens` tokens from the model, given the previous logits.
        Also return the logits of the sampled tokens.
        Arguments:
            input_ids: (batch, seqlen)
            decoding: whether we're in the decoding phase or the prefilling phase. Prefill doesn't
                need special position_ids.
            last_token_logits: whether to return the logits of the last token. Normally we don't need this.
                However, for speculative sampling, if the main model accepts all the draft tokens, plus it
                samples one new token, then by right at the next iteration the draft model need to evaluate
                the logits of the last draft token and the logits of the newly sampled token.
                This makes implementation more complicated. So here we just evaluate the logits of the last
                token in the draft model to simplify the implementation.
        Return:
            tokens: (batch, num_tokens)
            scores: (batch, num_tokens), which contains @previous_logits and the logits of the next
                (num_tokens - 1) tokens. The logits of the last token isn't computed unless last_token_logits=True.
                In which case we have scores of shape (batch, num_tokens + 1)
        """
        batch_size, seqlen = input_ids.shape
        assert num_tokens >= 1
        sequences = []
        if decoding:
            assert seqlen == 1
            position_ids = torch.full(
                (batch_size, 1),
                inference_params.sequence_len_offset,
                dtype=torch.long,
                device=input_ids.device,
            )
        else:
            position_ids = None
        logits = logits_postprocess_fn(
            logits_forward_fn(model, input_ids, position_ids, inference_params, cg=decoding and cg)
        )
        inference_params.sequence_len_offset += input_ids.shape[1]
        scores = [logits]
        next_token = sample_fn(logits)
        sequences.append(next_token)
        for i in range(num_tokens):
            if i < num_tokens - 1 or last_token_logits:
                position_ids = torch.full(
                    (batch_size, 1),
                    inference_params_draft.sequence_len_offset,
                    dtype=torch.long,
                    device=input_ids.device,
                )
                logits = logits_postprocess_fn(
                    logits_forward_fn(
                        model, rearrange(next_token, "b -> b 1"), position_ids, inference_params, cg=cg
                    )
                )
                inference_params.sequence_len_offset += 1
                scores.append(logits)
            if i < num_tokens - 1:
                next_token = sample_fn(logits)
                sequences.append(next_token)
        return torch.stack(sequences, dim=1), torch.stack(scores, dim=1)

    sampling_kwargs = dict(top_k=top_k, top_p=top_p, temperature=temperature)
    sample_fn = partial(sample, **sampling_kwargs)
    sample_tokens_main = partial(
        sample_tokens, model=model, sample_fn=sample_fn, inference_params=inference_params, cg=False
    )  # main model doesn't use CUDA graph
    sample_tokens_draft = partial(
        sample_tokens,
        model=model_draft,
        sample_fn=sample_fn,
        last_token_logits=True,
        inference_params=inference_params_draft,
        cg=cg
    )

    if debug:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    sequences = [input_ids]
    scores = []
    with torch.inference_mode():
        if timing:
            if tensor_parallel > 1:
                torch.distributed.barrier()
            torch.cuda.synchronize()
            start = time.time()

        if seqlen_og >= max_length - 1:
            # Don't do speculative sampling, just sample 1 token from the model
            tokens, scores_new = sample_tokens_main(input_ids, num_tokens=1, decoding=False)
            sequences.append(tokens)
            scores.append(scores_new)
        else:
            # Sample from draft model, which produces @n_spec_tokens, and @model
            # will then use to produce between 1 and 1 + @n_spec_tokens tokens.
            # We want seqlen_og + 1 + @n_spec_tokens to be <= @max_length.
            n_spec_tokens = min(speculative_lookahead, max_length - seqlen_og - 1)
            tokens_draft, scores_draft = sample_tokens_draft(
                input_ids,
                num_tokens=n_spec_tokens,
                decoding=False,
            )
            if debug:
                scores_draft_ref = model_draft(
                    torch.cat([input_ids, tokens_draft], dim=1), num_last_tokens=n_spec_tokens + 1
                ).logits
                print((scores_draft[:, :-1] - scores_draft_ref[:, :-1]).abs().max())

            # Evaluate the draft tokens with the model
            logits = model(
                torch.cat([input_ids, tokens_draft], dim=1),
                inference_params=inference_params,
                num_last_tokens=n_spec_tokens + 1,
            ).logits
            logits = logits_postprocess_fn(logits)
            tokens, num_generated_tokens = sample_speculative(
                logits, scores_draft[:, :-1], tokens_draft, **sampling_kwargs
            )
            if debug:
                print(tokens)
                print(num_generated_tokens)
                # breakpoint()
            # TODO: we're using the fact that batch_size == 1
            # TODO: check eos_token_id
            sequences.append(tokens[:1, : num_generated_tokens[0]])
            scores.append(logits[:1, : num_generated_tokens[0]])
            # Note that @model has not evaluated the last sampled token yet, so we'll need to pass
            # that in the next time we call @model.
            inference_params.sequence_len_offset = seqlen_og + num_generated_tokens[0].item() - 1
            inference_params_draft.sequence_len_offset = inference_params.sequence_len_offset
            if debug:
                cur_ids = torch.cat([input_ids, sequences[-1]], dim=1)
                scores_ref = model(
                    cur_ids, num_last_tokens=num_generated_tokens[0].item() + 1
                ).logits
                print((scores[-1] - scores_ref[:, :-1]).abs().max())

        while True:
            # sequence_len_offset is total length generated - 1
            if inference_params.sequence_len_offset >= max_length - 1:
                break
            if inference_params.sequence_len_offset >= max_length - 2:
                # Don't do speculative sampling, just sample 1 token from the model
                tokens, scores_new = sample_tokens_main(sequences[-1][:, -1:], num_tokens=1)
                sequences.append(tokens)
                scores.append(scores_new)
                break
            # Sample from draft model
            n_spec_tokens = min(
                speculative_lookahead, max_length - inference_params_draft.sequence_len_offset - 2
            )
            tokens_draft, scores_draft = sample_tokens_draft(
                sequences[-1][:, -1:], num_tokens=n_spec_tokens
            )
            if debug:
                scores_draft_ref = model_draft(
                    torch.cat([cur_ids, tokens_draft], dim=1), num_last_tokens=n_spec_tokens + 1
                ).logits
                print((scores_draft[:, :-1] - scores_draft_ref[:, :-1]).abs().max())
            # Evaluate the draft tokens with the model
            position_ids = repeat(
                torch.arange(
                    inference_params.sequence_len_offset,
                    # 1 extra token from last time that hasn't been passed through model
                    inference_params.sequence_len_offset + n_spec_tokens + 1,
                    dtype=torch.long,
                    device=input_ids.device,
                ),
                "s -> b s",
                b=batch_size,
            )
            logits = model(
                torch.cat([sequences[-1][:, -1:], tokens_draft], dim=1),
                position_ids=position_ids,
                inference_params=inference_params,
            ).logits  # (batch, n_spec_tokens, vocab_size)
            logits = logits_postprocess_fn(logits)
            inference_params.sequence_len_offset += 1
            if debug:
                logits_ref = model(
                    torch.cat([cur_ids, tokens_draft], dim=1), num_last_tokens=n_spec_tokens + 1
                ).logits
                print((logits - logits_ref).abs().max())
            tokens, num_generated_tokens = sample_speculative(
                logits, scores_draft[:, :-1], tokens_draft, **sampling_kwargs
            )
            if debug:
                print(tokens)
                print(num_generated_tokens)
            sequences.append(tokens[:1, : num_generated_tokens[0]])
            scores.append(logits[:1, : num_generated_tokens[0]])
            inference_params.sequence_len_offset += num_generated_tokens[0].item() - 1
            inference_params_draft.sequence_len_offset = inference_params.sequence_len_offset
            # breakpoint()
            if debug:
                cur_ids = torch.cat([cur_ids, sequences[-1]], dim=1)
                scores_ref = model(
                    cur_ids, num_last_tokens=num_generated_tokens[0].item() + 1
                ).logits
                print((scores[-1] - scores_ref[:, :-1]).abs().max())

        if timing:
            if tensor_parallel > 1:
                torch.distributed.barrier()
            torch.cuda.synchronize()
            print(f"Prompt processing + decoding time: {(time.time() - start) * 1000:.0f}ms")
    sequences = torch.cat(sequences, dim=1)
    scores = torch.cat(scores, dim=1)
    if debug:
        scores_ref = model(sequences).logits
        print((scores - scores_ref[:, seqlen_og - 1 : -1]).abs().max())
    output_cls = GreedySearchDecoderOnlyOutput if top_k == 1 else SampleDecoderOnlyOutput
    return output_cls(sequences=sequences, scores=scores)


class GenerationMixin:
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        raise NotImplementedError

    def generate(
        self,
        input_ids,
        max_length,
        top_k=1,
        top_p=0.0,
        temperature=1.0,
        return_dict_in_generate=False,
        output_scores=False,
        **kwargs,
    ):
        output = decode(
            input_ids, self, max_length, top_k=top_k, top_p=top_p, temperature=temperature, **kwargs
        )
        if not output_scores:
            output.scores = None
        return output if return_dict_in_generate else output.sequences


def allocate_inference_cache(
    max_batch_size,
    max_seqlen,
    nheads,
    headdim,
    layers: Union[int, Sequence],
    device,
    dtype=torch.float16,
):
    assert dtype in [torch.float16, torch.bfloat16, torch.float32]
    packsize = 4 if dtype == torch.float32 else 8
    assert headdim % packsize == 0
    k_cache_shape = (max_batch_size, nheads, headdim // packsize, max_seqlen, packsize)
    v_cache_shape = (max_batch_size, nheads, max_seqlen, headdim)
    if isinstance(layers, int):
        layers = range(layers)
    return {
        i: (
            torch.empty(k_cache_shape, device=device, dtype=dtype),
            torch.empty(v_cache_shape, device=device, dtype=dtype),
        )
        for i in layers
    }


def seqlen_to_seqlen_type(seqlen: int) -> int:
    """Convert sequence length to a seqlen_type.
    This is used to determine which cuda graph to use.
    Arguments:
        seqlen: int
    """
    return 0 if seqlen < 32 else (1 if seqlen < 2048 else 2)


def seqlen_type_to_max_seqlen(seqlen_type: int) -> int:
    assert seqlen_type in [0, 1, 2]
    return 32 if seqlen_type == 0 else (2048 if seqlen_type == 1 else 2**32)


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
def update_graph_cache(
    model, cache, batch_size, seqlen_og, max_seqlen, tensor_parallel=1, dtype=None, n_warmups=2
):
    if cache is None:
        cache = DecodingCGCache()
    param_example = next(iter(model.parameters()))
    device = param_example.device
    if dtype is None:
        dtype = param_example.dtype
    if (
        (device, dtype) != (cache.device, cache.dtype)
        or batch_size > cache.max_batch_size
        or max_seqlen > cache.max_seqlen
    ):  # Invalidate the cache
        cache.callables = {}
        cache.mempool = None
        cache.inference_params = None
        gc.collect()
        cache.device, cache.dtype = device, dtype
        cache.max_batch_size, cache.max_seqlen = batch_size, max_seqlen
        if hasattr(model, "allocate_inference_cache"):
            inf_cache = model.allocate_inference_cache(batch_size, max_seqlen, dtype)
        else:
            headdim = getattr(
                model.config,
                "head_dim",
                model.config.hidden_size // model.config.num_attention_heads,
            )
            inf_cache = allocate_inference_cache(
                batch_size,
                max_seqlen,
                model.config.num_attention_heads // tensor_parallel,
                headdim,
                model.config.num_hidden_layers,
                device,
                dtype,
            )
        lengths_per_sample = torch.full((batch_size,), seqlen_og, dtype=torch.int32, device=device)
        cache.inference_params = InferenceParams(
            max_sequence_len=max_seqlen,
            max_batch_size=batch_size,
            sequence_len_offset=seqlen_og,
            key_value_memory_dict=inf_cache,
            fused_ft_kernel=True,
            lengths_per_sample=lengths_per_sample,
        )
        cache.mempool = torch.cuda.graphs.graph_pool_handle()
    for s_type in range(seqlen_to_seqlen_type(seqlen_og), seqlen_to_seqlen_type(max_seqlen) + 1):
        if (batch_size, s_type) not in cache.callables:
            max_seqlen_ = min(max(seqlen_og, seqlen_type_to_max_seqlen(s_type)), max_seqlen)
            cache.callables[batch_size, s_type] = capture_graph(
                model,
                cache.inference_params,
                batch_size,
                max_seqlen_,
                mempool=cache.mempool,
                n_warmups=n_warmups,
            )

    def dispatch(input_ids, position_ids, seqlen):
        batch_size = input_ids.shape[0]
        return cache.callables[batch_size, seqlen_to_seqlen_type(seqlen)](
            input_ids, position_ids, seqlen
        )

    cache.run = dispatch
    cache.inference_params.sequence_len_offset = 0  # Reset so it's not confusing
    return cache


def capture_graph(model, inference_params, batch_size, max_seqlen, mempool=None, n_warmups=2):
    device = next(iter(model.parameters())).device
    input_ids = torch.full((batch_size, 1), 0, dtype=torch.long, device=device)
    position_ids = torch.full((batch_size, 1), 0, dtype=torch.long, device=device)
    sequence_len_offset_og = inference_params.sequence_len_offset
    # TD [2023-04-14]: important for correctness of the FT's attention kernel, as seqlen_cpu is
    # used to determine the size of smem. Hence seqlen_cpu must be >= lengths_per_sample.
    inference_params.sequence_len_offset = max_seqlen - 1
    inference_params.lengths_per_sample[:] = max_seqlen - 1

    # Warmup before capture
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(n_warmups):
            logits = model(
                input_ids,
                position_ids=position_ids,
                inference_params=inference_params,
                num_last_tokens=1,
            ).logits
        s.synchronize()
        # This might be needed for correctness if we run with NCCL_GRAPH_MIXING_SUPPORT=0,
        # which requires that graph launch and non-captured launch to not overlap (I think,
        # that's how I interpret the documentation). I'm not sure if this is required.
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
    torch.cuda.current_stream().wait_stream(s)
    # Captures the graph
    # To allow capture, automatically sets a side stream as the current stream in the context
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=mempool):
        logits = model(
            input_ids,
            position_ids=position_ids,
            inference_params=inference_params,
            num_last_tokens=1,
        ).logits.squeeze(dim=1)

    def run(new_input_ids, new_position_ids, seqlen):
        inference_params.lengths_per_sample[:] = seqlen
        input_ids.copy_(new_input_ids)
        position_ids.copy_(new_position_ids)
        graph.replay()
        return logits.clone()

    inference_params.sequence_len_offset = sequence_len_offset_og
    return run
