import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).
    The hidden states go from (batch, num_key_value_heads, seqlen, head_dim)
    to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    print(num_key_value_heads, n_rep)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# from https://github.com/huggingface/transformers/blob/369c99d0cea403b77bd0aef818527106453fd9fc/src/transformers/models/gpt_oss/modeling_gpt_oss.py#L227
def eager_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sink: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    num_key_value_groups: int = 8,
    **kwargs,
):
    key_states = repeat_kv(key, num_key_value_groups)
    value_states = repeat_kv(value, num_key_value_groups)
    print("==== sink: ", sink.shape, sink)
    print("==== query shape: ", query.shape)
    print("==== key_states shape: ", key_states.shape)
    print("==== value_states shape: ", value_states.shape)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    print("==== attn_weights shape: ", attn_weights.shape)
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        print("==== causal_mask shape: ", causal_mask.shape)
        attn_weights = attn_weights + causal_mask

    sinks = sink.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], -1)
    print("==== sinks shape: ", sinks.shape)
    combined_logits = torch.cat([attn_weights, sinks], dim=-1)
    print("==== combined_logits shape: ", combined_logits.shape)
    # This was not in the original implementation and slightly affect results;
    # it prevents overflow in BF16/FP16 when training with bsz>1 we clamp max values.
    combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
    
    probs = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
    print("==== probs shape: ", probs.shape)
    scores = probs[..., :-1]  # we drop the sink here
    print("==== scores shape: ", scores.shape)
    attn_weights = nn.functional.dropout(scores, p=dropout, training=True)
    attn_output = torch.matmul(attn_weights, value_states)
    print("==== attn_output shape: ", attn_output.shape)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights
