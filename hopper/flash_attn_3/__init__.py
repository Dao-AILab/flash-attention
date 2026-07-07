from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("flash_attn_3")
except PackageNotFoundError:  # not installed (e.g. running from a source tree)
    __version__ = "3.0.0"
