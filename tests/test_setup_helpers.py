import importlib.util
from pathlib import Path

import pytest
from packaging.version import Version


_spec = importlib.util.spec_from_file_location(
    "setup_helpers", Path(__file__).parents[1] / "setup_helpers.py"
)
assert _spec is not None and _spec.loader is not None
_setup_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_setup_helpers)
add_cuda_gencodes = _setup_helpers.add_cuda_gencodes
get_cuda_supported_archs = _setup_helpers.get_cuda_supported_archs


DEFAULT_ARCHS = frozenset({"80", "90", "100", "110", "120"})
SUPPORTED_ARCHS = {
    "11.7": {"80", "86", "87"},
    "11.8": {"80", "86", "87", "89", "90"},
    "12.7": {"80", "86", "87", "89", "90"},
    "12.8": {"80", "86", "87", "89", "90", "100", "101", "120"},
    "12.9": {"80", "86", "87", "89", "90", "100", "101", "120"},
    "13.0": {"80", "86", "87", "89", "90", "100", "110", "120"},
}


def gencode_specs(version, archs=DEFAULT_ARCHS, supported_archs=None):
    supported_archs = (
        SUPPORTED_ARCHS[version] if supported_archs is None else supported_archs
    )
    flags = add_cuda_gencodes([], archs, Version(version), supported_archs)
    assert flags[::2] == ["-gencode"] * (len(flags) // 2)
    return flags[1::2]


@pytest.mark.parametrize(
    "version,expected",
    [
        (
            "11.7",
            [
                "arch=compute_80,code=sm_80",
                "arch=compute_80,code=compute_80",
            ],
        ),
        (
            "11.8",
            [
                "arch=compute_80,code=sm_80",
                "arch=compute_90,code=sm_90",
                "arch=compute_90,code=compute_90",
            ],
        ),
        (
            "12.7",
            [
                "arch=compute_80,code=sm_80",
                "arch=compute_90,code=sm_90",
                "arch=compute_90,code=compute_90",
            ],
        ),
        (
            "12.8",
            [
                "arch=compute_80,code=sm_80",
                "arch=compute_90,code=sm_90",
                "arch=compute_100,code=sm_100",
                "arch=compute_120,code=sm_120",
                "arch=compute_101,code=sm_101",
                "arch=compute_120,code=compute_120",
            ],
        ),
        (
            "12.9",
            [
                "arch=compute_80,code=sm_80",
                "arch=compute_90,code=sm_90",
                "arch=compute_100f,code=sm_100",
                "arch=compute_120f,code=sm_120",
                "arch=compute_101,code=sm_101",
                "arch=compute_120,code=compute_120",
            ],
        ),
        (
            "13.0",
            [
                "arch=compute_80,code=sm_80",
                "arch=compute_90,code=sm_90",
                "arch=compute_100f,code=sm_100",
                "arch=compute_120f,code=sm_120",
                "arch=compute_110f,code=sm_110",
                "arch=compute_120,code=compute_120",
            ],
        ),
    ],
)
def test_default_gencodes_follow_nvcc_capabilities(version, expected):
    assert gencode_specs(version) == expected


@pytest.mark.parametrize(
    "version,expected",
    [
        (
            "12.8",
            [
                "arch=compute_101,code=sm_101",
                "arch=compute_101,code=compute_101",
            ],
        ),
        (
            "13.0",
            [
                "arch=compute_110f,code=sm_110",
                "arch=compute_110,code=compute_110",
            ],
        ),
    ],
)
def test_thor_ptx_uses_the_toolkit_arch_name(version, expected):
    assert gencode_specs(version, {"110"}) == expected


def test_custom_supported_architecture_remains_available_as_ptx():
    assert gencode_specs("12.7", {"86"}) == ["arch=compute_86,code=compute_86"]


@pytest.mark.parametrize("version,arch", [("11.7", "90"), ("12.7", "120")])
def test_unsupported_only_request_fails_clearly(version, arch):
    with pytest.raises(RuntimeError, match="cannot compile any requested"):
        gencode_specs(version, {arch})


def test_appends_to_existing_flags_in_place():
    flags = ["--use_fast_math"]

    result = add_cuda_gencodes(flags, {"80"}, Version("12.1"), {"80", "90"})

    assert result is flags
    assert flags == [
        "--use_fast_math",
        "-gencode",
        "arch=compute_80,code=sm_80",
        "-gencode",
        "arch=compute_80,code=compute_80",
    ]


def test_empty_request_leaves_flags_unchanged():
    flags = ["--use_fast_math"]

    assert add_cuda_gencodes(flags, set(), Version("12.1"), {"80", "90"}) == flags


def test_queries_nvcc_for_supported_architectures(monkeypatch):
    invocation = {}

    def fake_check_output(command, universal_newlines):
        invocation["command"] = command
        invocation["universal_newlines"] = universal_newlines
        return "compute_80\ncompute_90\ncompute_120\n"

    monkeypatch.setattr(_setup_helpers.subprocess, "check_output", fake_check_output)

    assert get_cuda_supported_archs("/opt/cuda") == {"80", "90", "120"}
    assert invocation == {
        "command": ["/opt/cuda/bin/nvcc", "--list-gpu-arch"],
        "universal_newlines": True,
    }
