import logging
from types import SimpleNamespace

import torch

import flash_attn.cute.cache_utils as cache_utils
from flash_attn.cute import fa_logging


def test_persistent_cache_hit_logs_at_host_level_only(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="flash_attn")
    original_level = fa_logging.get_fa_log_level()
    key = ("test-key",)
    cache = cache_utils.JITPersistentCache(tmp_path)
    obj_path = tmp_path / f"{cache._key_to_hash(key)}.o"
    obj_path.write_bytes(b"cache-hit")
    monkeypatch.setattr(
        cache_utils.cute.runtime,
        "load_module",
        lambda *_args, **_kwargs: SimpleNamespace(func=object()),
    )
    try:
        monkeypatch.setattr(fa_logging, "_fa_log_level", 0)
        assert cache_utils.JITPersistentCache(tmp_path)._try_load_from_storage(key)
        assert "Loading compiled function from disk" not in caplog.text

        caplog.clear()
        monkeypatch.setattr(fa_logging, "_fa_log_level", 1)
        assert cache_utils.JITPersistentCache(tmp_path)._try_load_from_storage(key)
        assert "Loading compiled function from disk" in caplog.text
    finally:
        monkeypatch.setattr(fa_logging, "_fa_log_level", original_level)


def test_scalar_tensor_compile_keys_are_normalized():
    cache = cache_utils.JITCache()
    fn = object()

    key0 = ("kernel", torch.tensor(128, dtype=torch.int32), (torch.tensor(False),))
    key1 = ("kernel", torch.tensor(128, dtype=torch.int64), (torch.tensor(False),))

    cache[key0] = fn

    assert key1 in cache
    assert cache[key1] is fn
    assert list(cache.cache.keys()) == [("kernel", 128, (False,))]


def test_persistent_cache_hash_normalizes_scalar_tensors(tmp_path):
    cache = cache_utils.JITPersistentCache(tmp_path)

    key0 = ("kernel", torch.tensor(128, dtype=torch.int32), (torch.tensor(False),))
    key1 = ("kernel", torch.tensor(128, dtype=torch.int64), (torch.tensor(False),))

    assert cache._key_to_hash(key0) == cache._key_to_hash(key1)
