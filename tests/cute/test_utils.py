"""Unit tests for flash_attn.cute.utils module."""

import functools

from flash_attn.cute import utils as cute_utils
from flash_attn.cute.utils import hash_callable


class TestHashCallable:
    """Tests for hash_callable function."""

    def test_returns_cute_hash_when_set_on_function(self):
        """hash_callable should return __cute_hash__ immediately when set on function."""

        def my_func():
            pass

        my_func.__cute_hash__ = "precomputed-hash-123"

        result = hash_callable(my_func)
        assert result == "precomputed-hash-123"

    def test_returns_cute_hash_from_wrapped_function(self):
        """hash_callable should check __wrapped__ for __cute_hash__."""

        def inner_func():
            pass

        inner_func.__cute_hash__ = "inner-hash-456"

        # Simulate a decorator that sets __wrapped__
        @functools.wraps(inner_func)
        def wrapper_func():
            return inner_func()

        result = hash_callable(wrapper_func)
        assert result == "inner-hash-456"

    def test_prefers_wrapper_cute_hash_over_wrapped(self):
        """When both wrapper and wrapped have __cute_hash__, prefer wrapper."""

        def inner_func():
            pass

        inner_func.__cute_hash__ = "inner-hash"

        @functools.wraps(inner_func)
        def wrapper_func():
            return inner_func()

        wrapper_func.__cute_hash__ = "wrapper-hash"

        result = hash_callable(wrapper_func)
        assert result == "wrapper-hash"

    def test_fallback_to_source_hashing(self):
        """hash_callable should fall back to source hashing when no __cute_hash__."""

        def my_func():
            return 42

        result = hash_callable(my_func)
        # Should return a hex string (SHA256 hash)
        assert isinstance(result, str)
        assert len(result) == 64  # SHA256 produces 64 hex chars

    def test_same_function_produces_same_hash(self):
        """Same function should produce consistent hash."""

        def my_func():
            return 42

        hash1 = hash_callable(my_func)
        hash2 = hash_callable(my_func)
        assert hash1 == hash2

    def test_different_functions_produce_different_hashes(self):
        """Different functions should produce different hashes."""

        def func_a():
            return 1

        def func_b():
            return 2

        hash_a = hash_callable(func_a)
        hash_b = hash_callable(func_b)
        assert hash_a != hash_b

    def test_fast_path_skips_expensive_hashing(self):
        """When __cute_hash__ is set, expensive operations should be skipped."""

        def my_func():
            pass

        my_func.__cute_hash__ = "fast-hash"

        # Mock at module level since we loaded it directly
        original_getsource = cute_utils.inspect.getsource
        call_tracker = {"getsource": 0, "sha256": 0}

        def tracking_getsource(*args, **kwargs):
            call_tracker["getsource"] += 1
            return original_getsource(*args, **kwargs)

        original_sha256 = cute_utils.hashlib.sha256

        def tracking_sha256(*args, **kwargs):
            call_tracker["sha256"] += 1
            return original_sha256(*args, **kwargs)

        cute_utils.inspect.getsource = tracking_getsource
        cute_utils.hashlib.sha256 = tracking_sha256
        try:
            result = hash_callable(my_func)
        finally:
            cute_utils.inspect.getsource = original_getsource
            cute_utils.hashlib.sha256 = original_sha256

        # Neither inspect.getsource nor hashlib.sha256 should be called
        assert call_tracker["getsource"] == 0, "getsource should not be called"
        assert call_tracker["sha256"] == 0, "sha256 should not be called"
        assert result == "fast-hash"

    def test_fast_path_on_wrapped_skips_expensive_hashing(self):
        """When __cute_hash__ is on __wrapped__, expensive operations should be skipped."""

        def inner_func():
            pass

        inner_func.__cute_hash__ = "wrapped-fast-hash"

        @functools.wraps(inner_func)
        def wrapper_func():
            return inner_func()

        # Mock at module level
        original_getsource = cute_utils.inspect.getsource
        call_tracker = {"getsource": 0, "sha256": 0}

        def tracking_getsource(*args, **kwargs):
            call_tracker["getsource"] += 1
            return original_getsource(*args, **kwargs)

        original_sha256 = cute_utils.hashlib.sha256

        def tracking_sha256(*args, **kwargs):
            call_tracker["sha256"] += 1
            return original_sha256(*args, **kwargs)

        cute_utils.inspect.getsource = tracking_getsource
        cute_utils.hashlib.sha256 = tracking_sha256
        try:
            result = hash_callable(wrapper_func)
        finally:
            cute_utils.inspect.getsource = original_getsource
            cute_utils.hashlib.sha256 = original_sha256

        assert call_tracker["getsource"] == 0, "getsource should not be called"
        assert call_tracker["sha256"] == 0, "sha256 should not be called"
        assert result == "wrapped-fast-hash"

    def test_closure_values_affect_hash(self):
        """Functions with different closure values should have different hashes."""
        value1 = 10
        value2 = 20

        def make_func(val):
            def inner():
                return val

            return inner

        func1 = make_func(value1)
        func2 = make_func(value2)

        hash1 = hash_callable(func1)
        hash2 = hash_callable(func2)
        assert hash1 != hash2


class TestHashCallableIntegration:
    """Integration tests for hash_callable with flash attention."""

    def test_repeated_calls_use_cached_hash(self):
        """Repeated calls with same score_mod should use cached/fast hash path."""

        def score_mod(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, aux_tensors):
            return tSrS_ssa

        # Set __cute_hash__ to simulate Inductor-generated code
        score_mod.__cute_hash__ = "inductor-generated-hash"

        original_getsource = cute_utils.inspect.getsource
        call_count = [0]  # Use list for mutable counter in nested function

        def counting_getsource(*args, **kwargs):
            call_count[0] += 1
            return original_getsource(*args, **kwargs)

        cute_utils.inspect.getsource = counting_getsource
        try:
            # Call hash_callable multiple times
            hash1 = hash_callable(score_mod)
            hash2 = hash_callable(score_mod)
            hash3 = hash_callable(score_mod)
        finally:
            cute_utils.inspect.getsource = original_getsource

        # getsource should never be called because __cute_hash__ is set
        assert call_count[0] == 0, f"getsource was called {call_count[0]} times"
        assert hash1 == hash2 == hash3 == "inductor-generated-hash"

