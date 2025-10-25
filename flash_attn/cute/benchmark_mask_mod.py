"""
FlashAttention benchmarking script with Flex Attention-style
mask mod support and varlen sequences.
"""

from dataclasses import dataclass
import math
from typing import Any, Dict, Optional, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import numpy as np
import torch

from flash_fwd import FlashAttentionForwardSm90
from mask_definitions import (
    MASK_FUNCTIONS,
    random_doc_id_tensor,
    create_cute_sliding_window_mask,
    create_flex_sliding_window_mask,
)
from flash_attn.cute.block_sparsity import (
    compute_block_sparsity,
    BlockSparseTensorsTorch,
    to_cute_block_sparse_tensors,
)


@dataclass
class BenchmarkConfig:
    """Benchmark configuration"""

    # Model parameters
    headdim: int
    headdim_v: int
    nheads: int
    nheads_kv: int
    dtype: torch.dtype

    # Sequence parameters
    batch_size: int = 2
    seqlen_q: int = 8192
    seqlen_k: int = 8192

    # Varlen parameters
    use_varlen: bool = False
    min_seqlen_q: Optional[int] = None  # If None, use seqlen_q // 2
    max_seqlen_q: Optional[int] = None  # If None, use seqlen_q
    min_seqlen_k: Optional[int] = None  # If None, use seqlen_k // 2
    max_seqlen_k: Optional[int] = None  # If None, use seqlen_k

    # Mask parameters
    use_mask_mod: bool = True
    mask_mod_name: str = "causal"
    has_aux_tensors: bool = mask_mod_name == "document"

    # Sliding window parameter (used when mask_mod_name == "sliding_window")
    window_size: int = 128

    # Attention parameters
    causal: bool = False
    is_local: bool = False
    window_left: Optional[int] = 128  # For base Flash Attention local
    window_right: Optional[int] = 0  # For base Flash Attention local
    softcap: Optional[float] = None
    use_learnable_sink: bool = False

    # Kernel configuration
    tile_m: int = 128
    tile_n: int = 128
    num_stages: int = 2
    num_threads: int = 384
    intra_wg_overlap: bool = True
    mma_pv_is_rs: bool = True

    # Benchmark parameters
    warmup_iters: int = 5
    benchmark_iters: int = 20
    verbose: bool = False
    seed: int = 42


class FlashAttentionBenchmark:
    def __init__(self, config: BenchmarkConfig):
        self.config = config

        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        # Verify SM90 compute capability
        compute_capability = torch.cuda.get_device_capability()
        assert compute_capability >= (9, 0), (
            f"Requires SM90+, got SM{compute_capability[0]}{compute_capability[1]}"
        )
        # causal overrides use_mask_mod
        if config.causal:
            config.use_mask_mod = False

        if config.use_mask_mod:
            if config.mask_mod_name == "sliding_window":
                # Use factory function for custom window size
                self.mask_mod_cute = create_cute_sliding_window_mask(config.window_size)
                self.mask_mod_flex = create_flex_sliding_window_mask(config.window_size)
            else:
                self.mask_mod_cute, self.mask_mod_flex = MASK_FUNCTIONS[config.mask_mod_name]
        else:
            self.mask_mod_cute = None
            self.mask_mod_flex = None

        self._validate_config()

    def _validate_config(self):
        config = self.config

        assert config.headdim <= 256, "headdim must be <= 256"
        assert config.headdim_v <= 256, "headdim_v must be <= 256"
        assert config.nheads % config.nheads_kv == 0, "nheads must be divisible by nheads_kv"

        alignment = 16 // config.dtype.itemsize
        assert config.headdim % alignment == 0, f"headdim must be divisible by {alignment}"
        assert config.headdim_v % alignment == 0, f"headdim_v must be divisible by {alignment}"

        # Validate is_local configuration
        if config.is_local:
            assert config.window_left is not None or config.window_right is not None, (
                "When is_local=True, at least one of window_left or window_right must be set"
            )
            assert not config.use_mask_mod, (
                "Cannot use both is_local and use_mask_mod simultaneously"
            )
            assert not config.causal, "Cannot use both is_local and causal simultaneously"

        # Validate mask_mod configuration
        if config.use_mask_mod and config.mask_mod_name == "sliding_window":
            assert config.window_size > 0, (
                "window_size must be positive when using sliding_window mask"
            )

    def _generate_varlen_seqlens(self, min_len: int, max_len: int) -> Tuple[torch.Tensor, int]:
        """Generate random sequence lengths and compute cumulative lengths."""
        seqlens = torch.randint(
            min_len, max_len + 1, (self.config.batch_size,), dtype=torch.int32, device="cuda"
        )
        cu_seqlens = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device="cuda"),
                torch.cumsum(seqlens, dtype=torch.int32, dim=0),
            ]
        )

        total_tokens = cu_seqlens[-1].item()
        return cu_seqlens, total_tokens

    def _create_tensors(self) -> Dict[str, torch.Tensor]:
        config = self.config
        device = "cuda"

        if config.use_varlen:
            # Set defaults for varlen range
            min_q = config.min_seqlen_q if config.min_seqlen_q is not None else config.seqlen_q // 2
            max_q = config.max_seqlen_q if config.max_seqlen_q is not None else config.seqlen_q
            min_k = config.min_seqlen_k if config.min_seqlen_k is not None else config.seqlen_k // 2
            max_k = config.max_seqlen_k if config.max_seqlen_k is not None else config.seqlen_k

            # Generate cu_seqlens
            cu_seqlens_q, total_q = self._generate_varlen_seqlens(min_q, max_q)
            cu_seqlens_k, total_k = self._generate_varlen_seqlens(min_k, max_k)

            # Varlen shape: (total_tokens, nheads, headdim)
            q = torch.randn(
                total_q, config.nheads, config.headdim, dtype=config.dtype, device=device
            )
            k = torch.randn(
                total_k, config.nheads_kv, config.headdim, dtype=config.dtype, device=device
            )
            v = torch.randn(
                total_k, config.nheads_kv, config.headdim_v, dtype=config.dtype, device=device
            )
            out = torch.empty(
                total_q, config.nheads, config.headdim_v, dtype=config.dtype, device=device
            )
            lse = torch.empty(config.nheads, total_q, dtype=torch.float32, device=device)

            tensors = {
                "q": q.contiguous(),
                "k": k.contiguous(),
                "v": v.contiguous(),
                "out": out.contiguous(),
                "lse": lse.contiguous(),
                "cu_seqlens_q": cu_seqlens_q.contiguous(),
                "cu_seqlens_k": cu_seqlens_k.contiguous(),
            }

            if config.verbose:
                print(f"Varlen: total_q={total_q}, total_k={total_k}")
                print(f"Q seqlens: {cu_seqlens_q[1:] - cu_seqlens_q[:-1]}")
                print(f"K seqlens: {cu_seqlens_k[1:] - cu_seqlens_k[:-1]}")
        else:
            # Standard shape: (batch, seqlen, nheads, headdim)
            q = torch.randn(
                config.batch_size,
                config.seqlen_q,
                config.nheads,
                config.headdim,
                dtype=config.dtype,
                device=device,
            )
            k = torch.randn(
                config.batch_size,
                config.seqlen_k,
                config.nheads_kv,
                config.headdim,
                dtype=config.dtype,
                device=device,
            )
            v = torch.randn(
                config.batch_size,
                config.seqlen_k,
                config.nheads_kv,
                config.headdim_v,
                dtype=config.dtype,
                device=device,
            )
            out = torch.empty(
                config.batch_size,
                config.seqlen_q,
                config.nheads,
                config.headdim_v,
                dtype=config.dtype,
                device=device,
            )
            lse = torch.empty(
                config.batch_size,
                config.nheads,
                config.seqlen_q,
                dtype=torch.float32,
                device=device,
            )

            tensors = {
                "q": q.contiguous(),
                "k": k.contiguous(),
                "v": v.contiguous(),
                "out": out.contiguous(),
                "lse": lse.contiguous(),
            }

        if config.use_learnable_sink:
            learnable_sink = torch.rand(config.nheads, dtype=torch.bfloat16, device=device)

            tensors["learnable_sink"] = learnable_sink.contiguous()

        # Compute block sparsity when using mask_mod
        if config.use_mask_mod:
            if config.mask_mod_name == "document":
                doc_id = random_doc_id_tensor(
                    config.batch_size, config.nheads, config.seqlen_q, device=device
                )
                tensors["aux_tensors"] = [doc_id.contiguous()]
            full_cnt, full_idx, mask_cnt, mask_idx = compute_block_sparsity(
                config=self.config,
                mask_mod_flex=self.mask_mod_flex,
                device=device,
                cu_seqlens_q=tensors.get("cu_seqlens_q"),
                cu_seqlens_k=tensors.get("cu_seqlens_k"),
                aux_tensors=tensors.get("aux_tensors"),
            )

            if all(t is not None for t in [full_cnt, full_idx, mask_cnt, mask_idx]):
                tensors["block_sparse_tensors"] = BlockSparseTensorsTorch(
                    mask_block_cnt=mask_cnt.contiguous(),
                    mask_block_idx=mask_idx.contiguous(),
                    full_block_cnt=full_cnt.contiguous(),
                    full_block_idx=full_idx.contiguous(),
                )

                if config.verbose:
                    total_full = full_cnt.sum().item()
                    total_partial = mask_cnt.sum().item()

                    if config.use_varlen:
                        # Compute max possible blocks across all sequences
                        max_blocks = 0
                        for i in range(config.batch_size):
                            seq_len_q = (
                                tensors["cu_seqlens_q"][i + 1] - tensors["cu_seqlens_q"][i]
                            ).item()
                            seq_len_k = (
                                tensors["cu_seqlens_k"][i + 1] - tensors["cu_seqlens_k"][i]
                            ).item()
                            n_blocks_q = (seq_len_q + config.tile_m - 1) // config.tile_m
                            n_blocks_k = (seq_len_k + config.tile_n - 1) // config.tile_n
                            max_blocks += n_blocks_q * n_blocks_k * config.nheads
                    else:
                        n_blocks_k = (config.seqlen_k + config.tile_n - 1) // config.tile_n
                        n_blocks_q = (config.seqlen_q + config.tile_m - 1) // config.tile_m
                        max_blocks = n_blocks_k * n_blocks_q * config.nheads * config.batch_size

                    skipped = max_blocks - total_full - total_partial
                    print(
                        f"Block stats: Full={total_full}, Partial={total_partial}, "
                        f"Skipped={skipped}/{max_blocks}"
                    )

        return tensors

    def _compile_kernel(self, tensors: Dict[str, torch.Tensor]) -> Tuple[Any, tuple]:
        config = self.config

        dtype_map = {
            torch.float16: cutlass.Float16,
            torch.bfloat16: cutlass.BFloat16,
            torch.float32: cutlass.Float32,
        }
        cute_dtype = dtype_map[config.dtype]

        qhead_per_kvhead = config.nheads // config.nheads_kv
        kernel = FlashAttentionForwardSm90(
            cute_dtype,
            config.headdim,
            config.headdim_v,
            qhead_per_kvhead,
            is_causal=config.causal,
            is_local=config.is_local,
            pack_gqa=False,
            tile_m=config.tile_m,
            tile_n=config.tile_n,
            num_stages=config.num_stages,
            num_threads=config.num_threads,
            intra_wg_overlap=config.intra_wg_overlap,
            mma_pv_is_rs=config.mma_pv_is_rs,
            mask_mod=self.mask_mod_cute,
            Q_in_regs=False,
            has_aux_tensors=config.has_aux_tensors,
        )

        softmax_scale = 1.0 / math.sqrt(config.headdim)
        current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

        # Convert tensors to cute
        q_cute = from_dlpack(tensors["q"].detach(), assumed_align=16).mark_layout_dynamic(
            leading_dim=tensors["q"].ndim - 1
        )
        k_cute = from_dlpack(tensors["k"].detach(), assumed_align=16).mark_layout_dynamic(
            leading_dim=tensors["k"].ndim - 1
        )
        v_cute = from_dlpack(tensors["v"].detach(), assumed_align=16).mark_layout_dynamic(
            leading_dim=tensors["v"].ndim - 1
        )
        out_cute = from_dlpack(tensors["out"].detach(), assumed_align=16).mark_layout_dynamic(
            leading_dim=tensors["out"].ndim - 1
        )
        lse_cute = from_dlpack(tensors["lse"].detach(), assumed_align=4).mark_layout_dynamic(
            leading_dim=tensors["lse"].ndim - 1
        )

        # Varlen tensors
        cu_seqlens_q_cute = (
            from_dlpack(tensors["cu_seqlens_q"].detach(), assumed_align=4).mark_layout_dynamic(
                leading_dim=0
            )
            if "cu_seqlens_q" in tensors
            else None
        )
        cu_seqlens_k_cute = (
            from_dlpack(tensors["cu_seqlens_k"].detach(), assumed_align=4).mark_layout_dynamic(
                leading_dim=0
            )
            if "cu_seqlens_k" in tensors
            else None
        )
        learnable_sink_cute = (
            from_dlpack(tensors["learnable_sink"].detach(), assumed_align=4).mark_layout_dynamic(
                leading_dim=0
            )
            if "learnable_sink" in tensors
            else None
        )

        blocksparse_tensors_cute = (
            to_cute_block_sparse_tensors(tensors["block_sparse_tensors"])
            if "block_sparse_tensors" in tensors
            else None
        )

        if "aux_tensors" in tensors:
            aux_tensors_cute = []
            for i in range(len(tensors["aux_tensors"])):
                buf = from_dlpack(tensors["aux_tensors"][i].detach(), assumed_align=4)
                aux_tensors_cute.append(buf.mark_layout_dynamic(leading_dim=2))

        else:
            aux_tensors_cute = None

        # Window parameters for is_local
        window_left_cute = (
            cutlass.Int32(config.window_left) if config.window_left is not None else None
        )
        window_right_cute = (
            cutlass.Int32(config.window_right) if config.window_right is not None else None
        )

        compiled = cute.compile(
            kernel,
            q_cute,
            k_cute,
            v_cute,
            out_cute,
            lse_cute,
            softmax_scale,
            current_stream,
            cu_seqlens_q_cute,
            cu_seqlens_k_cute,
            None,  # seqused_q
            None,  # seqused_k
            None,  # page_table
            window_left_cute,
            window_right_cute,
            learnable_sink_cute,
            blocksparse_tensors_cute,
            aux_tensors_cute,
            # None,
        )

        args = (
            q_cute,
            k_cute,
            v_cute,
            out_cute,
            lse_cute,
            softmax_scale,
            current_stream,
            cu_seqlens_q_cute,
            cu_seqlens_k_cute,
            None,
            None,
            None,
            window_left_cute,
            window_right_cute,
            learnable_sink_cute,
            blocksparse_tensors_cute,
            aux_tensors_cute,
            # None,
        )

        return compiled, args

    def _calculate_flops(self, tensors: Dict[str, torch.Tensor]) -> float:
        config = self.config

        # Estimate sparsity for known mask patterns
        if config.is_local:
            # Local attention with window_left and window_right
            window_left = config.window_left if config.window_left is not None else 0
            window_right = config.window_right if config.window_right is not None else 0
            total_window = window_left + window_right + 1  # +1 for current position
            sparsity_ratio = min(1.0, total_window / config.seqlen_k)
        elif config.use_mask_mod:
            if config.mask_mod_name in ["identity", "identity_partial"]:
                sparsity_ratio = 1.0
            elif config.mask_mod_name in ["causal", "block_causal"]:
                sparsity_ratio = 0.5
            elif config.mask_mod_name == "sliding_window":
                # Use configured window size
                sparsity_ratio = min(1.0, config.window_size / config.seqlen_k)
            elif config.mask_mod_name == "block_diagonal":
                block_size = 64
                num_blocks = (config.seqlen_k + block_size - 1) // block_size
                sparsity_ratio = 1.0 / num_blocks if num_blocks > 1 else 1.0
            elif config.mask_mod_name == "document":
                vals = tensors["aux_tensors"][0]
                val_mask = torch.ones_like(vals, dtype=torch.bool)
                val_mask[..., 1:] = vals[..., 1:] != vals[..., :-1]
                total = torch.where(val_mask, vals.square(), 0).sum()
                sparsity_ratio = total / (config.seqlen_q * config.seqlen_k)
            else:
                sparsity_ratio = 1.0
        elif config.causal:
            sparsity_ratio = 0.5
        else:
            sparsity_ratio = 1.0

        if config.use_varlen:
            # Compute FLOPs per sequence and sum
            total_flops = 0
            cu_q = tensors["cu_seqlens_q"]
            cu_k = tensors["cu_seqlens_k"]
            for i in range(config.batch_size):
                seq_len_q = (cu_q[i + 1] - cu_q[i]).item()
                seq_len_k = (cu_k[i + 1] - cu_k[i]).item()

                # Adjust sparsity for local attention in varlen case
                if config.is_local:
                    window_left = config.window_left if config.window_left is not None else 0
                    window_right = config.window_right if config.window_right is not None else 0
                    total_window = window_left + window_right + 1
                    seq_sparsity = min(1.0, total_window / seq_len_k)
                elif config.use_mask_mod and config.mask_mod_name == "sliding_window":
                    seq_sparsity = min(1.0, config.window_size / seq_len_k)
                else:
                    seq_sparsity = sparsity_ratio

                num_cells = int(seq_len_q * seq_len_k * seq_sparsity)

                if config.headdim == config.headdim_v:
                    flops_this_seq = 4 * config.nheads * num_cells * config.headdim
                else:
                    flops_this_seq = (
                        2 * config.nheads * num_cells * config.headdim
                        + 2 * config.nheads * num_cells * config.headdim_v
                    )
                total_flops += flops_this_seq
            return total_flops
        else:
            num_cells = int(config.seqlen_q * config.seqlen_k * sparsity_ratio)
            if config.headdim == config.headdim_v:
                flops_per_batch = 4 * config.nheads * num_cells * config.headdim
            else:
                flops_per_batch = (
                    2 * config.nheads * num_cells * config.headdim
                    + 2 * config.nheads * num_cells * config.headdim_v
                )
            return flops_per_batch * config.batch_size

    def benchmark(self) -> Dict[str, Any]:
        config = self.config

        tensors = self._create_tensors()
        compiled_kernel, args = self._compile_kernel(tensors)

        # Warmup
        for _ in range(config.warmup_iters):
            compiled_kernel(*args)
        torch.cuda.synchronize()

        # Benchmark
        times = []
        for _ in range(config.benchmark_iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            compiled_kernel(*args)
            end.record()
            torch.cuda.synchronize()

            times.append(start.elapsed_time(end))

        times_tensor = torch.tensor(times)
        mean_time = times_tensor.mean().item()
        std_time = times_tensor.std().item() if len(times) > 1 else 0.0

        total_flops = self._calculate_flops(tensors)
        tflops = total_flops / (mean_time * 1e-3) / 1e12

        # Bandwidth calculation
        bytes_per_element = config.dtype.itemsize
        if config.use_varlen:
            total_q = tensors["q"].shape[0]
            total_k = tensors["k"].shape[0]
            memory_accessed = (
                total_q * config.nheads * config.headdim * bytes_per_element
                + total_k * config.nheads_kv * config.headdim * bytes_per_element
                + total_k * config.nheads_kv * config.headdim_v * bytes_per_element
                + total_q * config.nheads * config.headdim_v * bytes_per_element
            )
        else:
            memory_accessed = (
                config.batch_size
                * config.seqlen_q
                * config.nheads
                * config.headdim
                * bytes_per_element
                + config.batch_size
                * config.seqlen_k
                * config.nheads_kv
                * config.headdim
                * bytes_per_element
                + config.batch_size
                * config.seqlen_k
                * config.nheads_kv
                * config.headdim_v
                * bytes_per_element
                + config.batch_size
                * config.seqlen_q
                * config.nheads
                * config.headdim_v
                * bytes_per_element
            )
        bandwidth_gbps = memory_accessed / (mean_time * 1e-3) / 1e9

        results = {
            "mean_time_ms": mean_time,
            "std_time_ms": std_time,
            "tflops": tflops,
            "bandwidth_gbps": bandwidth_gbps,
        }

        if config.verbose:
            self._print_results(results)

        return results

    def _print_results(self, results: Dict[str, Any]):
        config = self.config

        # Basic configuration
        if config.use_varlen:
            print(
                f"Shape: B={config.batch_size} (varlen), HD={config.headdim}, "
                f"NH={config.nheads}, NKV={config.nheads_kv}"
            )
        else:
            print(
                f"Shape: B={config.batch_size}, Q={config.seqlen_q}, K={config.seqlen_k}, "
                f"HD={config.headdim}, NH={config.nheads}, NKV={config.nheads_kv}"
            )

        # Attention pattern
        attn_info = []
        if config.causal:
            attn_info.append("causal")
        if config.is_local:
            window_info = f"local(L={config.window_left},R={config.window_right})"
            attn_info.append(window_info)
        if config.use_mask_mod:
            if config.mask_mod_name == "sliding_window":
                attn_info.append(f"mask_mod={config.mask_mod_name}(w={config.window_size})")
            else:
                attn_info.append(f"mask_mod={config.mask_mod_name}")
        if config.use_varlen:
            attn_info.append("varlen")
        if attn_info:
            print(f"Attention: {', '.join(attn_info)}")

        # Performance metrics
        print(f"Time: {results['mean_time_ms']:.3f} ± {results['std_time_ms']:.3f} ms")
        print(f"Throughput: {results['tflops']:.2f} TFLOPS")
        print(f"Bandwidth: {results['bandwidth_gbps']:.1f} GB/s")


if __name__ == "__main__":
    B = 2
    config = BenchmarkConfig(
        headdim=128,
        headdim_v=128,
        nheads=16,
        nheads_kv=16,
        dtype=torch.bfloat16,
        batch_size=B,
        # batch_size=1,
        seqlen_q=16384 // B,
        # seqlen_q=128,
        seqlen_k=16384 // B,
        # seqlen_k=192,
        use_varlen=False,
        use_mask_mod=True,
        mask_mod_name="causal",
        window_size=128,  # Configurable window size for mask_mod
        use_learnable_sink=False,
        causal=False,
        is_local=False,
        verbose=True,
    )

    # Example 2: Base Flash Attention Local
    # config = BenchmarkConfig(
    #     headdim=64,
    #     headdim_v=64,
    #     nheads=64,
    #     nheads_kv=8,
    #     dtype=torch.bfloat16,
    #     batch_size=2,
    #     seqlen_q=8192,
    #     seqlen_k=8192,
    #     use_varlen=False,
    #     use_mask_mod=False,
    #     causal=False,
    #     is_local=True,
    #     window_left=128,   # Left window size for base local attention
    #     window_right=0,    # Right window size for base local attention
    #     verbose=True,
    # )

    benchmark = FlashAttentionBenchmark(config)
    results = benchmark.benchmark()
