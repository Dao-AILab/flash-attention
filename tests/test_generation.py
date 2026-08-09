import torch

from flash_attn.utils.generation import (
    modify_logits_for_top_p_filtering,
    sample_speculative,
)


def test_top_p_filtering_supports_multiple_leading_dimensions():
    logits = torch.tensor(
        [
            [[4.0, 3.0, 2.0, 1.0], [1.0, 3.0, 2.0, 4.0]],
            [[2.0, 1.0, 4.0, 3.0], [3.0, 4.0, 1.0, 2.0]],
        ]
    )
    expected = logits.flatten(0, -2).clone()
    modify_logits_for_top_p_filtering(expected, top_p=0.75)

    actual = logits.clone()
    modify_logits_for_top_p_filtering(actual, top_p=0.75)

    assert torch.isneginf(expected).any()
    assert torch.equal(actual.flatten(0, -2), expected)


def test_speculative_sampling_keeps_batch_rows_independent():
    def peaked_logits(token_ids, vocab_size=4):
        logits = torch.full((*token_ids.shape, vocab_size), -torch.inf)
        return logits.scatter_(-1, token_ids.unsqueeze(-1), 0.0)

    tokens_draft = torch.tensor([[0, 0], [2, 2]])
    logits_draft = peaked_logits(tokens_draft)
    logits = peaked_logits(torch.tensor([[1, 1, 1], [2, 2, 3]]))

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        tokens, num_generated_tokens = sample_speculative(
            logits,
            logits_draft,
            tokens_draft,
            top_k=1,
        )

    assert torch.equal(num_generated_tokens, torch.tensor([1, 3]))
    assert torch.equal(tokens, torch.tensor([[1, 0, 0], [2, 2, 3]]))
