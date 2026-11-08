"""Shared/exclusive file locks backed by Windows LockFileEx."""

import ctypes
import errno
import msvcrt
from ctypes import wintypes

LOCK_SH = 1
LOCK_EX = 2
LOCK_NB = 4
LOCK_UN = 8


class _OVERLAPPED(ctypes.Structure):
    _fields_ = [
        ("Internal", ctypes.c_size_t),
        ("InternalHigh", ctypes.c_size_t),
        ("Offset", wintypes.DWORD),
        ("OffsetHigh", wintypes.DWORD),
        ("hEvent", wintypes.HANDLE),
    ]


_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_kernel32.LockFileEx.argtypes = [
    wintypes.HANDLE,
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.DWORD,
    ctypes.POINTER(_OVERLAPPED),
]
_kernel32.LockFileEx.restype = wintypes.BOOL
_kernel32.UnlockFileEx.argtypes = [
    wintypes.HANDLE,
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.DWORD,
    ctypes.POINTER(_OVERLAPPED),
]
_kernel32.UnlockFileEx.restype = wintypes.BOOL


def flock(fd: int, operation: int) -> None:
    """Lock byte zero, including on empty files; locks die with the handle."""
    handle = msvcrt.get_osfhandle(fd)
    overlapped = _OVERLAPPED()
    if operation == LOCK_UN:
        ok = _kernel32.UnlockFileEx(handle, 0, 1, 0, ctypes.byref(overlapped))
    else:
        if operation & ~(LOCK_SH | LOCK_EX | LOCK_NB) or bool(
            operation & LOCK_SH
        ) == bool(operation & LOCK_EX):
            raise ValueError("Specify exactly one of LOCK_SH or LOCK_EX")
        flags = (1 if operation & LOCK_NB else 0) | (2 if operation & LOCK_EX else 0)
        ok = _kernel32.LockFileEx(handle, flags, 0, 1, 0, ctypes.byref(overlapped))
    if not ok:
        error = ctypes.get_last_error()
        if error == 33:  # ERROR_LOCK_VIOLATION: another handle owns the byte range.
            raise BlockingIOError(errno.EAGAIN, "File lock is held by another handle")
        raise ctypes.WinError(error)
