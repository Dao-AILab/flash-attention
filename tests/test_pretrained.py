import torch
from safetensors.torch import save_file as safe_save_file
from transformers.utils import (
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
)

from flash_attn.utils import pretrained


def mock_remote_files(monkeypatch, files):
    calls = []

    def mock_cached_file(model_name, filename, **kwargs):
        assert model_name == "org/model"
        assert kwargs == {"_raise_exceptions_for_missing_entries": False}
        calls.append(filename)
        return files.get(filename)

    monkeypatch.setattr(pretrained, "cached_file", mock_cached_file)
    return calls


def test_remote_safetensors_fallback_and_dtype(tmp_path, monkeypatch):
    weights_path = tmp_path / SAFE_WEIGHTS_NAME
    weights = {"weight": torch.arange(6, dtype=torch.float32).reshape(2, 3)}
    safe_save_file(weights, weights_path)
    calls = mock_remote_files(monkeypatch, {SAFE_WEIGHTS_NAME: str(weights_path)})

    state_dict = pretrained.state_dict_from_pretrained("org/model", dtype=torch.bfloat16)

    assert calls == [WEIGHTS_NAME, WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME]
    assert state_dict.keys() == weights.keys()
    assert state_dict["weight"].dtype == torch.bfloat16
    torch.testing.assert_close(state_dict["weight"].float(), weights["weight"])


def test_remote_sharded_safetensors_fallback(tmp_path, monkeypatch):
    index_path = tmp_path / SAFE_WEIGHTS_INDEX_NAME
    index_path.write_text("{}")
    shard_paths = [
        tmp_path / "model-00001-of-00002.safetensors",
        tmp_path / "model-00002-of-00002.safetensors",
    ]
    weights = {
        "weight": torch.arange(4, dtype=torch.float32),
        "bias": torch.tensor([1.5, -2.0], dtype=torch.float32),
    }
    safe_save_file({"weight": weights["weight"]}, shard_paths[0])
    safe_save_file({"bias": weights["bias"]}, shard_paths[1])
    calls = mock_remote_files(monkeypatch, {SAFE_WEIGHTS_INDEX_NAME: str(index_path)})
    shard_calls = []

    def mock_get_checkpoint_shard_files(model_name, resolved_index_path):
        shard_calls.append((model_name, resolved_index_path))
        return [str(path) for path in shard_paths], {}

    monkeypatch.setattr(pretrained, "get_checkpoint_shard_files", mock_get_checkpoint_shard_files)

    state_dict = pretrained.state_dict_from_pretrained("org/model")

    assert calls == [
        WEIGHTS_NAME,
        WEIGHTS_INDEX_NAME,
        SAFE_WEIGHTS_NAME,
        SAFE_WEIGHTS_INDEX_NAME,
    ]
    assert shard_calls == [("org/model", str(index_path))]
    assert state_dict.keys() == weights.keys()
    for name, expected in weights.items():
        torch.testing.assert_close(state_dict[name], expected)


def test_remote_pytorch_weights_keep_priority(tmp_path, monkeypatch):
    weights_path = tmp_path / WEIGHTS_NAME
    safe_weights_path = tmp_path / SAFE_WEIGHTS_NAME
    weights = {"format": torch.tensor([1])}
    torch.save(weights, weights_path)
    safe_save_file({"format": torch.tensor([2])}, safe_weights_path)
    calls = mock_remote_files(
        monkeypatch,
        {
            WEIGHTS_NAME: str(weights_path),
            SAFE_WEIGHTS_NAME: str(safe_weights_path),
        },
    )

    state_dict = pretrained.state_dict_from_pretrained("org/model")

    assert calls == [WEIGHTS_NAME]
    torch.testing.assert_close(state_dict["format"], weights["format"])
