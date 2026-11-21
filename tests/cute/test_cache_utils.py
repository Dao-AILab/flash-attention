import errno
import logging
import os
from types import SimpleNamespace

import pytest
import quack
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


def test_persistent_cache_unloadable_entry_is_a_miss(tmp_path, monkeypatch):
    """A bad entry (e.g. a zero-byte object left by an export that hit a full disk) must read
    as a miss, so the caller recompiles, instead of failing every later run."""
    key = ("bad-entry",)
    cache = cache_utils.JITPersistentCache(tmp_path)
    (tmp_path / f"{cache._key_to_hash(key)}.o").write_bytes(b"")

    def failing_load(*_args, **_kwargs):
        raise RuntimeError("Failed to lookup function '__tvm_ffi_func'")

    monkeypatch.setattr(cache_utils.cute.runtime, "load_module", failing_load)
    assert not cache._try_load_from_storage(key)


def test_persistent_cache_export_is_atomic(tmp_path):
    """A failed export leaves no entry behind; a successful one lands at the final path."""
    cache = cache_utils.JITPersistentCache(tmp_path)

    class Failing:
        def export_to_c(self, object_file_path, function_name):
            open(object_file_path, "wb").close()  # created, then the write fails
            raise OSError(28, "No space left on device")

    class Good:
        def export_to_c(self, object_file_path, function_name):
            with open(object_file_path, "wb") as f:
                f.write(b"object")

    cache._try_export_to_storage(("k1",), Failing())
    assert not list(tmp_path.glob("*.o")), "a failed export left a file behind"
    cache._try_export_to_storage(("k2",), Good())
    assert (tmp_path / f"{cache._key_to_hash(('k2',))}.o").read_bytes() == b"object"
    assert not list(tmp_path.glob(".*.tmp.o"))


def test_persistent_cache_replaces_nonempty_unloadable_entry(tmp_path, monkeypatch):
    """A truncated but nonempty entry fails to load; the recompiled function's export must
    replace it (size alone does not make an entry valid), so the next process loads it
    instead of recompiling forever."""
    key = ("truncated-entry",)
    cache = cache_utils.JITPersistentCache(tmp_path)
    obj_path = tmp_path / f"{cache._key_to_hash(key)}.o"
    obj_path.write_bytes(b"trunc")

    def load_module(path, **_kwargs):
        if open(path, "rb").read() != b"object":
            raise RuntimeError("Failed to lookup function '__tvm_ffi_func'")
        return SimpleNamespace(func=object())

    monkeypatch.setattr(cache_utils.cute.runtime, "load_module", load_module)

    class Good:
        def export_to_c(self, object_file_path, function_name):
            with open(object_file_path, "wb") as f:
                f.write(b"object")

    assert not cache._try_load_from_storage(key)
    cache[key] = Good()  # recompiled: stored in memory and exported
    assert obj_path.read_bytes() == b"object"
    assert not list(tmp_path.glob(".*.tmp.o"))
    # a fresh process loads the replaced entry
    assert cache_utils.JITPersistentCache(tmp_path)._try_load_from_storage(key)

    # a nonempty entry this process loaded (or never tried) is still left alone
    other = ("other",)
    other_path = tmp_path / f"{cache._key_to_hash(other)}.o"
    other_path.write_bytes(b"from another process")
    cache[other] = Good()
    assert other_path.read_bytes() == b"from another process"


def test_get_jit_cache_falls_back_to_memory_when_dir_cannot_be_created(tmp_path, monkeypatch):
    """A cache dir that cannot be created (e.g. on a read-only mount) must not make
    get_jit_cache, which runs at import, raise; kernels are then cached in memory only."""

    def read_only_mkdir(self, *_args, **_kwargs):
        raise OSError(errno.EROFS, os.strerror(errno.EROFS), str(self))

    monkeypatch.setattr(cache_utils, "CUTE_DSL_CACHE_ENABLED", True)
    monkeypatch.setattr(cache_utils, "CUTE_DSL_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(cache_utils.Path, "mkdir", read_only_mkdir)
    assert type(cache_utils.get_jit_cache("fwd")) is cache_utils.JITCache


def test_read_only_persistent_cache_keeps_misses_in_memory(tmp_path, monkeypatch):
    """On a read-only cache, entries already on disk still load, and a miss is kept in memory
    instead of raising when its lock file cannot be created."""

    class Good:
        def export_to_c(self, object_file_path, function_name):
            with open(object_file_path, "wb") as f:
                f.write(b"object")

    real_open = os.open

    def read_only_open(path, flags, *args, **kwargs):
        if flags & os.O_WRONLY or (flags & os.O_CREAT and not os.path.exists(path)):
            raise OSError(errno.EROFS, os.strerror(errno.EROFS), path)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(
        cache_utils.cute.runtime,
        "load_module",
        lambda *_args, **_kwargs: SimpleNamespace(func=object()),
    )
    cache_utils.JITPersistentCache(tmp_path)[("hit",)] = Good()
    monkeypatch.setattr(os, "open", read_only_open)

    cache = cache_utils.JITPersistentCache(tmp_path)
    assert ("hit",) in cache
    assert ("miss",) not in cache
    fn = Good()
    cache[("miss",)] = fn
    assert cache[("miss",)] is fn
    assert not (tmp_path / f"{cache._key_to_hash(('miss',))}.o").exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="retry helper requires CUDA")
def test_retry_on_oom_preserves_persistent_cache(tmp_path, monkeypatch):
    from test_flash_attn import retry_on_oom
    from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd

    fwd_cache = cache_utils.JITPersistentCache(tmp_path / "fwd")
    bwd_cache = cache_utils.JITPersistentCache(tmp_path / "bwd")
    disk_entries = []
    for cache, key in ((fwd_cache, ("fwd",)), (bwd_cache, ("bwd",))):
        cache.cache[key] = object()
        entry = cache.cache_path / f"{cache._key_to_hash(key)}.o"
        entry.write_bytes(b"cached kernel")
        disk_entries.append(entry)

    monkeypatch.setattr(_flash_attn_fwd, "compile_cache", fwd_cache)
    monkeypatch.setattr(_flash_attn_bwd, "compile_cache", bwd_cache)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    calls = 0

    @retry_on_oom
    def run():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise torch.OutOfMemoryError("CUDA out of memory.")
        assert not fwd_cache.cache
        assert not bwd_cache.cache
        return "retried"

    assert run() == "retried"
    assert calls == 2
    assert [entry.read_bytes() for entry in disk_entries] == [b"cached kernel"] * 2


def test_source_fingerprint_tracks_quack_version(monkeypatch):
    """FA4 kernels compile Quack device code in (e.g. the SM90 backward's bulk copies), so a
    Quack upgrade must not load kernels compiled against the previous version."""
    cache_utils._compute_source_fingerprint.cache_clear()
    try:
        before = cache_utils._compute_source_fingerprint()
        monkeypatch.setattr(quack, "__version__", "0.0.0+test")
        cache_utils._compute_source_fingerprint.cache_clear()
        assert cache_utils._compute_source_fingerprint() != before
    finally:
        cache_utils._compute_source_fingerprint.cache_clear()
