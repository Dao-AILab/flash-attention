"""Persistent namespaces track the installed Quack distribution version."""

from importlib import metadata

import pytest

from flash_attn.cute import cache_utils


@pytest.fixture
def clean_fingerprint():
    cache_utils._compute_source_fingerprint.cache_clear()
    try:
        yield
    finally:
        cache_utils._compute_source_fingerprint.cache_clear()


def _set_quack_stamp(monkeypatch, state):
    real_version = metadata.version

    def version(name):
        if name == "quack-kernels":
            if state[0] is None:
                raise metadata.PackageNotFoundError(name)
            return state[0]
        return real_version(name)

    # Patch the standard-library module, not a new attribute missing on baseline.
    monkeypatch.setattr(metadata, "version", version)


def _fresh_fingerprint():
    cache_utils._compute_source_fingerprint.cache_clear()
    return cache_utils._compute_source_fingerprint()


def test_fingerprint_tracks_quack_version(monkeypatch, clean_fingerprint):
    stamp = ["test-quack-A"]
    _set_quack_stamp(monkeypatch, stamp)
    a = _fresh_fingerprint()
    assert _fresh_fingerprint() == a
    stamp[0] = "test-quack-B"
    assert _fresh_fingerprint() != a


def test_persistent_namespace_tracks_quack_version(tmp_path, monkeypatch, clean_fingerprint):
    stamp = ["test-quack-A"]
    _set_quack_stamp(monkeypatch, stamp)
    monkeypatch.setattr(cache_utils, "CUTE_DSL_CACHE_ENABLED", True)
    monkeypatch.setattr(cache_utils, "CUTE_DSL_CACHE_DIR", str(tmp_path))
    first = cache_utils.get_jit_cache("quack-test")
    cache_utils._compute_source_fingerprint.cache_clear()
    again = cache_utils.get_jit_cache("quack-test")
    assert first.cache_path == again.cache_path
    stamp[0] = "test-quack-B"
    cache_utils._compute_source_fingerprint.cache_clear()
    changed = cache_utils.get_jit_cache("quack-test")
    assert changed.cache_path != first.cache_path
    assert first.cache_path.is_dir() and changed.cache_path.is_dir()


def test_missing_quack_metadata_fails_closed(monkeypatch, clean_fingerprint):
    _set_quack_stamp(monkeypatch, [None])
    with pytest.raises(metadata.PackageNotFoundError, match="quack-kernels"):
        _fresh_fingerprint()


@pytest.mark.parametrize("module", [cache_utils.cutlass, cache_utils.tvm_ffi])
def test_existing_version_stamps_still_invalidate(module, monkeypatch, clean_fingerprint):
    _set_quack_stamp(monkeypatch, ["test-quack-A"])
    before = _fresh_fingerprint()
    monkeypatch.setattr(module, "__version__", "test-different-runtime")
    assert _fresh_fingerprint() != before


def test_disabled_persistent_cache_does_not_require_metadata(monkeypatch, clean_fingerprint):
    _set_quack_stamp(monkeypatch, [None])
    monkeypatch.setattr(cache_utils, "CUTE_DSL_CACHE_ENABLED", False)
    cache = cache_utils.get_jit_cache("quack-test")
    assert isinstance(cache, cache_utils.JITCache)
    assert not isinstance(cache, cache_utils.JITPersistentCache)
