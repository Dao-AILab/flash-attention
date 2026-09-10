from types import SimpleNamespace

import pytest
import torch

from flash_attn.utils.generation import decode


class FakeGenerationModel:
    def __init__(self, sampled_tokens=None):
        self.sampled_tokens = iter(sampled_tokens) if sampled_tokens is not None else None

    def __call__(self, input_ids, **kwargs):
        batch_size = input_ids.shape[0]
        logits = torch.zeros(batch_size, 1, 16, device=input_ids.device)
        if self.sampled_tokens is not None:
            token_ids = torch.as_tensor(next(self.sampled_tokens), device=input_ids.device)
            logits.fill_(-torch.inf)
            logits[torch.arange(batch_size, device=input_ids.device), 0, token_ids] = 0
        return SimpleNamespace(logits=logits)


def test_decode_tracks_finished_batch_rows():
    input_ids = torch.tensor([[1, 2], [1, 2], [1, 2]])
    teacher_outputs = torch.tensor(
        [
            [1, 2, 9, 4, 5, 6, 7],
            [1, 2, 3, 4, 9, 6, 7],
            [1, 2, 3, 9, 5, 6, 7],
        ]
    )

    output = decode(
        input_ids,
        FakeGenerationModel(),
        max_length=7,
        eos_token_id=9,
        teacher_outputs=teacher_outputs,
        pad_token_id=0,
    )

    expected = torch.tensor(
        [
            [1, 2, 9, 0, 0],
            [1, 2, 3, 4, 9],
            [1, 2, 3, 9, 0],
        ]
    )
    assert torch.equal(output.sequences, expected)
    assert len(output.scores) == 3


def test_decode_defaults_padding_to_eos_token():
    input_ids = torch.tensor([[1, 2], [1, 2]])
    teacher_outputs = torch.tensor(
        [
            [1, 2, 9, 4, 5, 6, 7],
            [1, 2, 3, 4, 9, 6, 7],
        ]
    )

    output = decode(
        input_ids,
        FakeGenerationModel(),
        max_length=7,
        eos_token_id=9,
        teacher_outputs=teacher_outputs,
    )

    expected = torch.tensor(
        [
            [1, 2, 9, 9, 9],
            [1, 2, 3, 4, 9],
        ]
    )
    assert torch.equal(output.sequences, expected)
    assert len(output.scores) == 3


def test_decode_accepts_multiple_eos_token_ids():
    input_ids = torch.tensor([[1, 2], [1, 2]])
    teacher_outputs = torch.tensor(
        [
            [1, 2, 8, 4, 5],
            [1, 2, 3, 9, 5],
        ]
    )

    output = decode(
        input_ids,
        FakeGenerationModel(),
        max_length=5,
        eos_token_id=[8, 9],
        teacher_outputs=teacher_outputs,
        pad_token_id=0,
    )

    expected = torch.tensor(
        [
            [1, 2, 8, 0],
            [1, 2, 3, 9],
        ]
    )
    assert torch.equal(output.sequences, expected)
    assert len(output.scores) == 2


def test_decode_tracks_finished_batch_rows_when_sampling():
    input_ids = torch.tensor([[1, 2], [1, 2], [1, 2]])
    model = FakeGenerationModel(sampled_tokens=[[9, 3, 3], [4, 3, 9], [5, 9, 5]])

    output = decode(
        input_ids,
        model,
        max_length=7,
        eos_token_id=9,
        pad_token_id=0,
    )

    expected = torch.tensor(
        [
            [1, 2, 9, 0, 0],
            [1, 2, 3, 3, 9],
            [1, 2, 3, 9, 0],
        ]
    )
    assert torch.equal(output.sequences, expected)
    assert len(output.scores) == 3


@pytest.mark.parametrize("eos_token_id", [None, 9])
def test_decode_runs_to_max_length_when_no_eos_is_generated(eos_token_id):
    input_ids = torch.tensor([[1, 2], [1, 2]])
    teacher_outputs = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 4, 5, 6]])

    output = decode(
        input_ids,
        FakeGenerationModel(),
        max_length=5,
        eos_token_id=eos_token_id,
        teacher_outputs=teacher_outputs,
    )

    assert torch.equal(output.sequences, teacher_outputs)
    assert len(output.scores) == 3


def test_decode_stops_when_batch_rows_finish_together():
    input_ids = torch.tensor([[1, 2], [1, 2]])
    teacher_outputs = torch.tensor([[1, 2, 9, 3], [1, 2, 9, 4]])

    output = decode(
        input_ids,
        FakeGenerationModel(),
        max_length=4,
        eos_token_id=9,
        teacher_outputs=teacher_outputs,
        pad_token_id=0,
    )

    assert torch.equal(output.sequences, teacher_outputs[:, :3])
    assert len(output.scores) == 1


def test_decode_rejects_empty_eos_token_ids():
    with pytest.raises(ValueError, match="at least one token ID"):
        decode(
            torch.tensor([[1, 2]]),
            FakeGenerationModel(),
            max_length=3,
            eos_token_id=[],
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph requires a CUDA device")
def test_decode_finished_batch_rows_match_with_cuda_graph():
    class TransitionModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            logits = torch.full((16, 16), -torch.inf, device="cuda")
            for token_id, next_token_id in {
                0: 1,
                2: 9,
                3: 5,
                4: 6,
                5: 9,
                6: 7,
                7: 9,
                9: 1,
            }.items():
                logits[token_id, next_token_id] = 0
            self.logits = torch.nn.Parameter(logits, requires_grad=False)

        def allocate_inference_cache(self, *args):
            return {}

        def forward(self, input_ids, **kwargs):
            return SimpleNamespace(logits=self.logits[input_ids[:, -1:]])

    input_ids = torch.tensor([[1, 2], [1, 3], [1, 4]], device="cuda")
    outputs = [
        decode(
            input_ids,
            TransitionModel(),
            max_length=8,
            eos_token_id=9,
            pad_token_id=0,
            cg=cg,
        )
        for cg in (False, True)
    ]

    expected = torch.tensor(
        [
            [1, 2, 9, 0, 0],
            [1, 3, 5, 9, 0],
            [1, 4, 6, 7, 9],
        ],
        device="cuda",
    )
    assert all(torch.equal(output.sequences, expected) for output in outputs)
    assert all(len(output.scores) == 3 for output in outputs)
