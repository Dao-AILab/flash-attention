"""CPU contracts for shared paging components; no CUDA context or GPU launch.

The scalar tests execute the actual helper body with only DSL scalar/decorator
bindings replaced. They do not claim to validate device lowering or GPU copies.
"""

import ast
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[2]
PAGED_KV = ROOT / "flash_attn/cute/paged_kv.py"
FORWARD = ROOT / "flash_attn/cute/flash_fwd_sm100.py"


def _tree(path):
    return ast.parse(path.read_text())


def _cpu_function(name, **bindings):
    """Execute source, not a separately maintained address implementation."""
    functions = [
        n
        for n in _tree(PAGED_KV).body
        if isinstance(n, ast.FunctionDef) and n.name == name
    ]
    assert len(functions) == 1, f"missing shared component: {name}"
    node = functions[0]
    node.decorator_list = []
    for arg in node.args.args:
        arg.annotation = None
    node.returns = None
    namespace = {"Int32": int, "const_expr": bool, **bindings}
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(PAGED_KV), "exec"),
        namespace,
    )
    return namespace[name]


class TrackedRow:
    def __init__(self, pages):
        self.pages = pages
        self.reads = []

    def __getitem__(self, index):
        self.reads.append(index)
        return self.pages[index]


@pytest.mark.parametrize("page_size", [16, 128, 256, 512, 768])
@pytest.mark.parametrize("leftpad", [0, 7, 257])
def test_resolve_paged_token_boundaries_leftpad_and_guard(page_size, leftpad):
    resolve = _cpu_function("resolve_paged_token")
    pages = [17, 2, 31, 5, 23, 11, 41, 3] * 4
    row = TrackedRow(pages)
    for token in [0, page_size - 1, page_size, page_size + 1, 2 * page_size - 1]:
        logical_page, offset = divmod(token + leftpad, page_size)
        assert resolve(row, token, page_size, leftpad=leftpad) == (
            pages[logical_page],
            offset,
        )
        assert row.reads[-1] == logical_page
    row.reads.clear()
    # No page-table read, even when the invalid logical token is far out of range.
    token = 1000 * page_size + 3
    assert resolve(row, token, page_size, valid=False, leftpad=leftpad) == (
        0,
        (token + leftpad) % page_size,
    )
    assert row.reads == []


def _function(path, name):
    matches = [
        n
        for n in ast.walk(_tree(path))
        if isinstance(n, ast.FunctionDef) and n.name == name
    ]
    assert len(matches) == 1
    return matches[0]


def _calls(node, name):
    return [
        n
        for n in ast.walk(node)
        if isinstance(n, ast.Call)
        and (
            isinstance(n.func, ast.Name)
            and n.func.id == name
            or isinstance(n.func, ast.Attribute)
            and n.func.attr == name
        )
    ]


def test_existing_loaders_share_address_and_transfer_components():
    manager = _function(PAGED_KV, "load_page_table")
    assert len(_calls(manager, "resolve_paged_token")) == 1
    assert not _calls(manager, "divmod")
    kernel = _function(FORWARD, "load")
    # Initial K/V, next K, delayed V and final delayed V all share lookup.
    assert len(_calls(kernel, "resolve_paged_token")) == 4
    for node in ast.walk(kernel):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            if node.value.id == "mPageTable":
                assert isinstance(node.slice, ast.Tuple)
                assert isinstance(node.slice.elts[1], ast.Constant)
                assert node.slice.elts[1].value is None  # a row view, never a lookup
    loader = _function(FORWARD, "load_KV")
    assert len(_calls(loader, "copy_kv_tma")) == 1
    assert not _calls(loader, "copy")
    # Stage acquisition and cp.async completion remain in the normal wrapper.
    assert len(_calls(loader, "producer_acquire")) == 1
    assert len(_calls(loader, "arrive_cp_async_mbarrier")) == 1


@pytest.mark.parametrize("with_atom", [False, True])
def test_copy_kv_tma_preserves_backend_and_barrier_contract(monkeypatch, with_atom):
    import sys

    calls = []
    experimental = ModuleType("cutlass.cute.experimental")
    experimental.tma_load = lambda *args, **kwargs: calls.append(("ext", args, kwargs))
    cute_module = ModuleType("cutlass.cute")
    cute_module.experimental = experimental
    cute_module.copy = lambda *args, **kwargs: calls.append(("atom", args, kwargs))
    cutlass_module = ModuleType("cutlass")
    cutlass_module.cute = cute_module
    for module in (cutlass_module, cute_module, experimental):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    transfer = _cpu_function("copy_kv_tma", cute=cute_module)
    gmem, smem, barrier, atom = (object() for _ in range(4))
    if with_atom:
        transfer(gmem, smem, barrier, tma_atom=atom)
        assert calls == [("atom", (atom, gmem, smem), {"tma_bar_ptr": barrier})]
    else:
        transfer(gmem, smem, barrier)
        # No update_expect_tx=False: cute_ext must arm its own transaction bytes.
        assert calls == [("ext", (gmem, smem, barrier), {})]
    node = _function(PAGED_KV, "copy_kv_tma")
    for name in ("producer_acquire", "producer_commit", "consumer_release", "advance"):
        assert not _calls(node, name)
    # Importing normal forward must not require the experimental DSL.
    for node in _tree(PAGED_KV).body:
        if isinstance(node, ast.ImportFrom):
            assert "experimental" not in (node.module or "")
            assert all(alias.name != "experimental" for alias in node.names)
