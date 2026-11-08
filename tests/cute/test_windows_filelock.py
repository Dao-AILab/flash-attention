"""Windows file locking preserves shared/exclusive cache coordination."""

import os
import sys

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows LockFileEx")


@pytest.fixture
def locks(tmp_path):
    from flash_attn.cute import _filelock

    path = tmp_path / "cache.lock"
    first = os.open(path, os.O_CREAT | os.O_RDWR)
    second = os.open(path, os.O_RDWR)
    try:
        yield _filelock, first, second
    finally:
        os.close(second)
        os.close(first)


def test_exclusive_contention_and_unlock(locks):
    lock, first, second = locks
    lock.flock(first, lock.LOCK_EX | lock.LOCK_NB)
    with pytest.raises(BlockingIOError):
        lock.flock(second, lock.LOCK_EX | lock.LOCK_NB)
    lock.flock(first, lock.LOCK_UN)
    lock.flock(second, lock.LOCK_EX | lock.LOCK_NB)
    lock.flock(second, lock.LOCK_UN)


def test_shared_readers_exclude_writer(locks):
    lock, first, second = locks
    lock.flock(first, lock.LOCK_SH | lock.LOCK_NB)
    lock.flock(second, lock.LOCK_SH | lock.LOCK_NB)
    lock.flock(second, lock.LOCK_UN)
    with pytest.raises(BlockingIOError):
        lock.flock(second, lock.LOCK_EX | lock.LOCK_NB)
    lock.flock(first, lock.LOCK_UN)


def test_blocking_mode_acquires_free_file(locks):
    lock, first, _ = locks
    lock.flock(first, lock.LOCK_EX)
    lock.flock(first, lock.LOCK_UN)


@pytest.mark.parametrize("operation", [0, 3, 16])
def test_invalid_operation(locks, operation):
    lock, first, _ = locks
    with pytest.raises(ValueError):
        lock.flock(first, operation)
