import os
import sys
import pytest


def _markexpr_includes_fast() -> bool:
    if "-m" not in sys.argv:
        return False
    expr_index = sys.argv.index("-m") + 1
    if expr_index >= len(sys.argv):
        return False
    expr = sys.argv[expr_index]
    return "fast" in expr


if _markexpr_includes_fast():
    os.environ["FLASH_ATTN_FAST_TESTS"] = "1"

from test_params import FAST_TEST_MODE


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests as slow-only (skip with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "fast: marks tests included in fast runs"
    )


def pytest_collection_modifyitems(config, items):
    if not _markexpr_includes_fast():
        return
    for item in items:
        if item.get_closest_marker("slow") is None:
            item.add_marker(pytest.mark.fast)


@pytest.fixture(scope="session")
def fast_test_mode():
    return FAST_TEST_MODE
