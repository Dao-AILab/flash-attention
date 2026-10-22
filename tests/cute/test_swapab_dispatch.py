"""CPU-only dispatch contracts; importing these tests never imports CUDA/CuTe."""

import ast
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
DISPATCH_PATH = ROOT / "flash_attn/cute/swapab_dispatch.py"


def load_dispatch():
    assert DISPATCH_PATH.exists(), "native SwapAB dispatch module is missing"
    spec = importlib.util.spec_from_file_location(
        "_swapab_dispatch_cpu_test", DISPATCH_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def forward_ast():
    tree = ast.parse((ROOT / "flash_attn/cute/interface.py").read_text())
    return next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_flash_attn_fwd"
    )


@dataclass
class TensorMetadata:
    shape: tuple
    dtype: str = "torch.bfloat16"
    requires_grad: bool = False
    strides: tuple = ()

    def stride(self, dim=None):
        if self.strides:
            strides = self.strides
        else:
            strides, running = [], 1
            for extent in reversed(self.shape):
                strides.insert(0, running)
                running *= extent
            strides = tuple(strides)
        return strides if dim is None else strides[dim]

    def element_size(self):
        return 2


def request_defaults(**overrides):
    fn = forward_ast()
    args = fn.args.args
    result = {arg.arg: None for arg in args}
    result.update(
        {
            a.arg: ast.literal_eval(v)
            for a, v in zip(args[-len(fn.args.defaults) :], fn.args.defaults)
        }
    )
    result.update(
        q=TensorMetadata((2, 1, 6, 96)),
        k=TensorMetadata((2, 1025, 2, 96)),
        v=TensorMetadata((2, 1025, 2, 128)),
    )
    result.update(overrides)
    return result


def decide(dispatch, request=None, **facts):
    request = request_defaults() if request is None else request
    resolved = dict(
        q=request["q"],
        k=request["k"],
        v=request["v"],
        page_table=request["page_table"],
        arch=100,
        requires_grad=False,
    )
    resolved.update(facts)
    return dispatch.select_swapab_backend(request, **resolved)


class DispatchTests(unittest.TestCase):
    def test_auto_selects_eligible_unequal_dimensions(self):
        decision = decide(load_dispatch())
        self.assertTrue(decision.use_swapab)
        self.assertEqual(decision.cache_token, ("sm100_swapab", 1))
        self.assertEqual(decision.logical_kv_capacity, 1025)

    def test_entire_forward_signature_is_classified(self):
        dispatch = load_dispatch()
        self.assertTrue(hasattr(dispatch, "FORWARD_OPTION_POLICY"))
        self.assertEqual(
            set(dispatch.FORWARD_OPTION_POLICY),
            {a.arg for a in forward_ast().args.args},
        )

    def test_unknown_future_option_falls_back_even_when_none(self):
        decision = decide(load_dispatch(), request_defaults(future_option=None))
        self.assertFalse(decision.use_swapab)
        self.assertIn("future_option", decision.reason)

    def test_rejected_original_options_and_forced_error(self):
        dispatch = load_dispatch()
        rejected = {
            "qv": object(),
            "cu_seqlens_q": object(),
            "cu_seqlens_k": object(),
            "seqused_q": object(),
            "learnable_sink": object(),
            "tile_mn": (32, 256),
            "mma_pv_is_rs": False,
            "intra_wg_overlap": False,
            "num_threads": 128,
            "pack_gqa": False,
            "block_sparse_tensors": object(),
            "aux_scalars": (),
            "q_descale": object(),
            "k_descale": object(),
            "v_descale": object(),
            "gather_kv_indices": object(),
            "scheduler_metadata": object(),
            "seqlen_k_per_split": 512,
            "disable_scheduler_metadata": True,
            "gather_bwd_recompute_p": True,
            "future_option": None,
        }
        for name, value in rejected.items():
            with self.subTest(name=name):
                request = request_defaults(**{name: value})
                decision = decide(dispatch, request)
                self.assertFalse(decision.use_swapab)
                self.assertIn(name, decision.reason)
                with self.assertRaisesRegex(ValueError, name):
                    decide(dispatch, request, mode="swapab")

    def test_architecture_training_dtype_and_shape_guards(self):
        dispatch = load_dispatch()
        for arch in (80, 90, 101, 110, 120):
            with self.subTest(arch=arch):
                self.assertFalse(decide(dispatch, arch=arch).use_swapab)
        self.assertFalse(decide(dispatch, requires_grad=True).use_swapab)
        for name, shape in (
            ("q", (2, 2, 6, 96)),
            ("q", (2, 1, 66, 96)),
            ("q", (2, 1, 6, 192)),
            ("v", (2, 1025, 2, 256)),
            ("k", (2050, 2, 96)),
        ):
            with self.subTest(name=name, shape=shape):
                self.assertFalse(
                    decide(
                        dispatch, request_defaults(**{name: TensorMetadata(shape)})
                    ).use_swapab
                )
        self.assertFalse(
            decide(
                dispatch,
                request_defaults(
                    v=TensorMetadata((2, 1025, 2, 128), "torch.float8_e4m3fn")
                ),
            ).use_swapab
        )
        self.assertFalse(
            decide(
                dispatch,
                request_defaults(
                    aux_tensors=[TensorMetadata((2,), requires_grad=True)]
                ),
            ).use_swapab
        )

    def test_supported_semantics_and_canonical_tensors(self):
        dispatch = load_dispatch()
        for dq in (64, 96, 128):
            for dv in (64, 96, 128):
                for ratio in (1, 3, 8, 32):
                    for dtype in ("torch.float16", "torch.bfloat16"):
                        with self.subTest(dq=dq, dv=dv, ratio=ratio, dtype=dtype):
                            req = request_defaults(
                                q=TensorMetadata((2, 1, ratio * 2, dq), dtype),
                                k=TensorMetadata((2, 1025, 2, dq), dtype),
                                v=TensorMetadata((2, 1025, 2, dv), dtype),
                                seqused_k=object(),
                                causal=True,
                                window_size_left=31,
                                window_size_right=0,
                                softcap=2.0,
                                mask_mod=object(),
                                aux_tensors=[TensorMetadata((2,))],
                                return_lse=True,
                                out=object(),
                                lse=object(),
                                _arch=100,
                                pack_gqa=True,
                            )
                            self.assertTrue(decide(dispatch, req).use_swapab)
        req = request_defaults(
            q=TensorMetadata((2, 1, 6, 96), strides=(1152, 1152, 192, 2))
        )
        self.assertFalse(decide(dispatch, req).use_swapab)
        self.assertTrue(
            decide(dispatch, req, q=TensorMetadata((2, 1, 6, 96))).use_swapab
        )
        req = request_defaults(
            k=TensorMetadata((2, 1025, 2, 96), strides=(393600, 384, 192, 1))
        )
        self.assertTrue(decide(dispatch, req).use_swapab)

    def test_paging_uses_logical_capacity_not_physical_pool(self):
        dispatch = load_dispatch()
        for page_size, supported in (
            (128, False),
            (255, False),
            (256, True),
            (384, False),
            (512, True),
            (768, True),
        ):
            with self.subTest(page_size=page_size):
                req = request_defaults(
                    k=TensorMetadata((19, page_size, 2, 96)),
                    v=TensorMetadata((19, page_size, 2, 128)),
                    page_table=TensorMetadata((2, 3), "torch.int32"),
                )
                decision = decide(dispatch, req)
                self.assertEqual(decision.use_swapab, supported)
                if supported:
                    self.assertEqual(decision.logical_kv_capacity, 3 * page_size)
                else:
                    with self.assertRaisesRegex(ValueError, "page"):
                        decide(dispatch, req, mode="swapab")

    def test_nonpositive_or_nonfinite_scale_uses_original_backend(self):
        dispatch = load_dispatch()
        for scale in (0.0, -0.1, float("inf"), float("-inf"), float("nan")):
            with self.subTest(scale=scale):
                req = request_defaults(softmax_scale=scale)
                self.assertFalse(decide(dispatch, req).use_swapab)
                with self.assertRaisesRegex(ValueError, "softmax_scale"):
                    decide(dispatch, req, mode="swapab")
        for scale in (None, 0.1, 1.0):
            self.assertTrue(
                decide(dispatch, request_defaults(softmax_scale=scale)).use_swapab
            )

    def test_modes_and_scheduler_overrides(self):
        dispatch = load_dispatch()
        for mode in ("", "AUTO", "typo"):
            with (
                self.subTest(mode=mode),
                self.assertRaisesRegex(ValueError, "FLASH_ATTENTION_SM100_BACKEND"),
            ):
                decide(dispatch, mode=mode)
        self.assertTrue(decide(dispatch, mode="swapab").use_swapab)
        self.assertFalse(decide(dispatch, use_clc_scheduler=True).use_swapab)
        self.assertFalse(decide(dispatch, disable_s_ping_pong=True).use_swapab)
        self.assertTrue(decide(dispatch, disable_2cta=True).use_swapab)

    def test_decisions_are_immutable_reentrant_and_independent(self):
        from concurrent.futures import ThreadPoolExecutor
        from dataclasses import FrozenInstanceError

        dispatch = load_dispatch()
        req = request_defaults()
        with ThreadPoolExecutor(max_workers=4) as pool:
            decisions = list(
                pool.map(
                    lambda mode: decide(dispatch, req, mode=mode),
                    ["auto", "standard", "swapab", "auto"] * 8,
                )
            )
        self.assertEqual(
            [d.use_swapab for d in decisions], [True, False, True, True] * 8
        )
        with self.assertRaises(FrozenInstanceError):
            decisions[0].reason = "mutated"
        self.assertIsNone(req["pack_gqa"])

    def test_standard_override_does_not_select_swapab(self):
        dispatch = load_dispatch()
        decision = dispatch.select_swapab_backend(
            {},
            q=None,
            k=None,
            v=None,
            page_table=None,
            arch=100,
            requires_grad=False,
            mode="standard",
        )
        self.assertFalse(decision.use_swapab)
        self.assertEqual(decision.cache_token, ("standard", 1))
        self.assertIn("standard", decision.reason)


class SourceContracts(unittest.TestCase):
    def test_adapter_forward_abi(self):
        def methods(filename, cls):
            tree = ast.parse((ROOT / "flash_attn/cute" / filename).read_text())
            return {
                n.name: n
                for c in tree.body
                if isinstance(c, ast.ClassDef) and c.name == cls
                for n in c.body
                if isinstance(n, ast.FunctionDef)
            }

        normal = methods("flash_fwd_sm100.py", "FlashAttentionForwardSm100")["__call__"]
        native = methods(
            "flash_fwd_decode_sm100.py", "FlashAttentionForwardSm100SwapAB"
        )["__call__"]
        self.assertEqual(
            [a.arg for a in normal.args.args], [a.arg for a in native.args.args]
        )
        self.assertIsNone(native.args.kwarg)

    def test_private_core_and_shared_paging(self):
        source = (ROOT / "flash_attn/cute/flash_fwd_decode_sm100.py").read_text()
        self.assertNotIn("def flash_attn_decode_func", source)
        self.assertNotIn("launch_pdl", source)
        self.assertNotIn("atomic_add", source)
        self.assertIn("cute.experimental.kernel", source)
        self.assertEqual(source.count("resolve_paged_token("), 2)
        self.assertEqual(source.count("copy_kv_tma("), 2)


if __name__ == "__main__":
    unittest.main()
