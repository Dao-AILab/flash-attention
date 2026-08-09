import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

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


def test_failed_export_does_not_publish_partial_object(tmp_path):
    key = ("failed-export",)
    cache = cache_utils.JITPersistentCache(tmp_path)
    obj_path = tmp_path / f"{cache._key_to_hash(key)}.o"
    reference_path = tmp_path / "reference-mode"
    reference_path.touch()
    expected_mode = reference_path.stat().st_mode & 0o777
    reference_path.unlink()

    class FailingExporter:
        def export_to_c(self, object_file_path, function_name):
            assert function_name == cache.EXPORT_FUNCTION_PREFIX
            path = Path(object_file_path)
            assert path != obj_path
            assert path.name.endswith(".o")
            path.write_bytes(b"partial-object")
            raise RuntimeError("export failed")

    with pytest.raises(RuntimeError, match="export failed"):
        cache._try_export_to_storage(key, FailingExporter())

    assert not obj_path.exists()
    assert not list(tmp_path.glob("*.tmp.o"))

    exported_paths = []

    class SuccessfulExporter:
        def export_to_c(self, object_file_path, function_name):
            assert function_name == cache.EXPORT_FUNCTION_PREFIX
            exported_paths.append(Path(object_file_path))
            Path(object_file_path).write_bytes(b"valid-object")

    cache._try_export_to_storage(key, SuccessfulExporter())

    assert len(exported_paths) == 1
    assert exported_paths[0] != obj_path
    assert obj_path.read_bytes() == b"valid-object"
    assert obj_path.stat().st_mode & 0o777 == expected_mode
    assert not list(tmp_path.glob("*.tmp.o"))


def test_invalid_object_is_removed_and_can_be_reexported(tmp_path, monkeypatch):
    key = ("invalid-object",)
    cache = cache_utils.JITPersistentCache(tmp_path)
    obj_path = tmp_path / f"{cache._key_to_hash(key)}.o"
    obj_path.write_bytes(b"invalid-object")
    load_calls = []

    def fail_load(path, **_kwargs):
        load_calls.append(path)
        raise RuntimeError("invalid ELF")

    monkeypatch.setattr(cache_utils.cute.runtime, "load_module", fail_load)

    assert not cache._try_load_from_storage(key)
    assert len(load_calls) == 2
    assert key not in cache.cache
    assert not obj_path.exists()

    class SuccessfulExporter:
        def export_to_c(self, object_file_path, function_name):
            assert function_name == cache.EXPORT_FUNCTION_PREFIX
            Path(object_file_path).write_bytes(b"recompiled-object")

    cache._try_export_to_storage(key, SuccessfulExporter())
    assert obj_path.read_bytes() == b"recompiled-object"


def test_load_failure_is_retried_before_removing_object(tmp_path, monkeypatch):
    key = ("revalidated-object",)
    cache = cache_utils.JITPersistentCache(tmp_path)
    obj_path = tmp_path / f"{cache._key_to_hash(key)}.o"
    obj_path.write_bytes(b"valid-object")
    loaded_fn = object()
    load_calls = 0

    def load_after_retry(*_args, **_kwargs):
        nonlocal load_calls
        load_calls += 1
        if load_calls == 1:
            raise RuntimeError("transient load failure")
        return SimpleNamespace(func=loaded_fn)

    monkeypatch.setattr(cache_utils.cute.runtime, "load_module", load_after_retry)

    assert cache._try_load_from_storage(key)
    assert load_calls == 2
    assert obj_path.exists()
    assert cache[key] is loaded_fn
