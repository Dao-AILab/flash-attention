from pathlib import Path
import logging
from types import SimpleNamespace

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


class DummyCompiledFunction:
    def __init__(self):
        self.exports = 0

    def export_to_c(self, object_file_path, function_name):
        self.exports += 1
        Path(object_file_path).write_bytes(function_name.encode())


def test_in_memory_get_or_compile_runs_factory_once():
    cache = cache_utils.JITCache()
    compiled = object()
    calls = {"n": 0}

    def compile_fn():
        calls["n"] += 1
        return compiled

    assert cache.get_or_compile(("key",), compile_fn) is compiled
    assert cache.get_or_compile(("key",), compile_fn) is compiled
    assert calls["n"] == 1


def test_persistent_get_or_compile_exports_then_loads(tmp_path, monkeypatch):
    key = ("persistent-key",)
    compiled = DummyCompiledFunction()
    calls = {"n": 0}
    cache = cache_utils.JITPersistentCache(tmp_path)

    def compile_fn():
        calls["n"] += 1
        return compiled

    assert cache.get_or_compile(key, compile_fn) is compiled
    assert calls["n"] == 1
    assert compiled.exports == 1

    loaded = object()
    monkeypatch.setattr(
        cache_utils.cute.runtime,
        "load_module",
        lambda *_args, **_kwargs: SimpleNamespace(func=loaded),
    )

    def fail_compile():
        raise AssertionError("compile factory should not run on persistent cache hit")

    assert cache_utils.JITPersistentCache(tmp_path).get_or_compile(key, fail_compile) is loaded
