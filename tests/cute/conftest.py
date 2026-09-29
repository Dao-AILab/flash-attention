import os
import subprocess
import logging
import tempfile
import json
import time
from pathlib import Path
from getpass import getuser

import pytest


def _get_gpu_ids():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return [g.strip() for g in visible.split(",")]

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip().splitlines()
    except (FileNotFoundError,):
        pass

    logging.warning("Failed to get gpu ids, use default '0'")
    return ["0"]


def pytest_configure(config):
    # MLA routing: the tests pick the kernel explicitly. Unset, the interface would apply its
    # 1CTA / 2CTA dispatch heuristic; the suite runs 2CTA by default and 1CTA wherever
    # supported with FLASH_ATTENTION_MLA_1CTA=1 (test_flash_attn_mla_dispatch_heuristic
    # covers the heuristic itself, with the variable unset).
    os.environ.setdefault("FLASH_ATTENTION_MLA_1CTA", "0")
    tmp = Path(tempfile.gettempdir()) / getuser() / "flash_attention_tests"
    tmp.mkdir(parents=True, exist_ok=True)

    worker_id = os.environ.get("PYTEST_XDIST_WORKER")
    logging.basicConfig(
        format=config.getini("log_file_format"),
        filename=str(tmp / f"tests_{worker_id}.log"),
        level=config.getini("log_file_level"),
    )
    if worker_id:
        worker_num = int(worker_id.replace("gw", ""))
        gpu_ids = _shared_gpu_ids(tmp, worker_num)
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids[worker_num % len(gpu_ids)]
    # after the GPU pick: it imports the interface (which may initialize CUDA)
    _install_compile_instrumentation(config)


def _shared_gpu_ids(tmp: Path, worker_num: int) -> list:
    """The GPU list every xdist worker round-robins over.

    With CUDA_VISIBLE_DEVICES set (inherited from the controller) each worker reads it
    directly. Otherwise worker 0 queries nvidia-smi once (expensive with many workers) and
    publishes the list in a file keyed by this run's PYTEST_XDIST_TESTRUNUID, written
    atomically (temp file + os.replace); the others wait for it. A per-run name means a stale
    file from an earlier run is never read; the atomic write means a reader never sees a
    partial file.
    """
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return _get_gpu_ids()
    run_id = os.environ.get("PYTEST_XDIST_TESTRUNUID", "norunid")
    cached_gpu_ids = tmp / f"gpu_ids_{run_id}.json"
    if worker_num == 0:
        gpu_ids = _get_gpu_ids()
        fd, tmp_name = tempfile.mkstemp(dir=tmp, prefix=".gpu_ids_", suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump(gpu_ids, f)
        os.replace(tmp_name, cached_gpu_ids)
        return gpu_ids
    deadline = time.monotonic() + 600
    while True:
        try:
            with cached_gpu_ids.open() as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            if time.monotonic() > deadline:
                raise
            time.sleep(0.5)


# ---------------------------------------------------------------------------------------------
# Compile instrumentation (two-pass testing, see CLAUDE.md "Fast two-pass testing").
#
#   FLASH_ATTENTION_TEST_COUNT_COMPILES=1  count cute.compile calls (= JIT cache misses: a hit,
#                                          in memory or on disk, never reaches cute.compile)
#                                          per test; the terminal summary lists tests that
#                                          compiled.
#   FLASH_ATTENTION_TEST_EXPECT_CACHED=1   the same, and a test that compiles fails (use on the
#                                          execution pass after a FLASH_ATTENTION_FAKE_TENSOR=1
#                                          compile pass).
#   FLASH_ATTENTION_TEST_RECORD_KEYS=path  append every compile-cache lookup (cache name, key)
#                                          to path.<worker>: the set of kernels a selection
#                                          compiles, for diffing across refactors / test tiers.
#   FLASH_ATTENTION_TEST_RECORD_KEYS_ONLY=1  with RECORD_KEYS and FLASH_ATTENTION_FAKE_TENSOR=1:
#                                          every lookup reports a hit, so nothing compiles
#                                          (a key snapshot in minutes; compile errors are
#                                          left to the real two-pass run).
# ---------------------------------------------------------------------------------------------
_COMPILES = {"n": 0}
_EXPECT_CACHED = os.environ.get("FLASH_ATTENTION_TEST_EXPECT_CACHED", "0") == "1"
_COUNT_COMPILES = _EXPECT_CACHED or os.environ.get("FLASH_ATTENTION_TEST_COUNT_COMPILES", "0") == "1"


def _install_compile_instrumentation(config):
    record_path = os.environ.get("FLASH_ATTENTION_TEST_RECORD_KEYS")
    if not (_COUNT_COMPILES or record_path):
        return
    import cutlass.cute as cute
    import flash_attn.cute.interface as fa_interface
    from flash_attn.cute import cache_utils

    if _COUNT_COMPILES and not getattr(cute.compile, "_fa_counted", False):
        original = cute.compile

        def counted_compile(*args, **kwargs):
            _COMPILES["n"] += 1
            return original(*args, **kwargs)

        counted_compile._fa_counted = True
        cute.compile = counted_compile

    if record_path and not getattr(cache_utils.JITCache.__contains__, "_fa_recorded", False):
        names = {
            id(fn.compile_cache): name
            for name, fn in vars(fa_interface).items()
            if callable(fn) and hasattr(fn, "compile_cache")
        }
        out = open(f"{record_path}.{os.environ.get('PYTEST_XDIST_WORKER', 'main')}", "a")
        original_contains = cache_utils.JITCache.__contains__

        keys_only = (
            os.environ.get("FLASH_ATTENTION_TEST_RECORD_KEYS_ONLY", "0") == "1"
            and os.environ.get("FLASH_ATTENTION_FAKE_TENSOR", "0") == "1"
        )

        def recording_contains(self, key):
            out.write(f"{names.get(id(self), type(self).__name__)}\t{key!r}\n")
            out.flush()
            return True if keys_only else original_contains(self, key)

        recording_contains._fa_recorded = True
        cache_utils.JITCache.__contains__ = recording_contains
        if keys_only:
            original_getitem = cache_utils.JITCache.__getitem__

            def noop_getitem(self, key):
                try:
                    return original_getitem(self, key)
                except KeyError:
                    return lambda *args, **kwargs: None

            cache_utils.JITCache.__getitem__ = noop_getitem


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    before = _COMPILES["n"]
    item._fa_compiles_before = before
    yield


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    if not _COUNT_COMPILES or call.when != "call":
        return
    report = outcome.get_result()
    compiled = _COMPILES["n"] - getattr(item, "_fa_compiles_before", _COMPILES["n"])
    report.user_properties.append(("compiles", compiled))
    if compiled and _EXPECT_CACHED and report.passed:
        report.outcome = "failed"
        report.longrepr = (
            f"{compiled} kernel(s) compiled during the execution pass "
            "(FLASH_ATTENTION_TEST_EXPECT_CACHED=1): the FLASH_ATTENTION_FAKE_TENSOR=1 compile "
            "pass did not produce the same compile key (fake / real key drift, a test that skips "
            "or returns before this kernel under fake mode, or a cache that is not persisted)"
        )


def pytest_terminal_summary(terminalreporter):
    if not _COUNT_COMPILES:
        return
    compiled = []
    for reports in terminalreporter.stats.values():
        for rep in reports:
            n = dict(getattr(rep, "user_properties", ())).get("compiles", 0)
            if getattr(rep, "when", None) == "call" and n:
                compiled.append((rep.nodeid, n))
    total = sum(n for _, n in compiled)
    terminalreporter.write_sep("-", f"kernel compiles: {total} in {len(compiled)} test(s)")
    for nodeid, n in sorted(compiled)[:200]:
        terminalreporter.write_line(f"{n:4d}  {nodeid}")

def _disable_torch_native_triton_bmm():
    """Work around an int32 overflow in torch's Triton override for aten::bmm.

    torch 2.13 routes bmm to a Triton "outer product" kernel when the contraction
    dim is 1 (torch/_native/ops/bmm_outer_product/). That kernel addresses the
    output as `pid_b * stride_ob + ...` in int32, so once B * M * N exceeds 2**31 the
    address wraps negative and the launch takes an illegal memory access, which
    poisons the CUDA context and cascades into every later test in the process.

    Reference (not kernel) code in these tests hits it: with seqlen_q == 1 the
    backward of P @ V is a K=1 bmm, e.g. B=batch*nheads=1536, M=seqlen_k=8192,
    N=head_dim_v=512 -> B*M*N = 6.4e9. Deregistering just this override falls back
    to eager (cuBLAS) bmm, which handles 64-bit offsets correctly.
    """
    try:
        from torch._native import registry
    except ImportError:
        return  # torch too old to have the override at all
    try:
        registry.deregister_op_overrides(disable_op_symbols="bmm")
    except Exception as exc:  # never let a workaround break collection
        logging.warning("could not disable torch._native bmm override: %s", exc)


def pytest_sessionstart(session):
    _disable_torch_native_triton_bmm()


def pytest_collection_finish(session):
    if not session.config.option.collectonly:
        return

    # file_name -> test_name -> counter
    test_counts: dict[str, dict[str, int]] = {}
    for item in session.items:
        funcname = item.function.__name__
        parent = test_counts.setdefault(item.parent.name, {})
        parent[funcname] = parent.setdefault(funcname, 0) + 1
    print(json.dumps(test_counts, indent=2))
