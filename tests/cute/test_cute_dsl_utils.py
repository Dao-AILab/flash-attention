import torch
import cutlass

from flash_attn.cute.cute_dsl_utils import make_kernel_name_prefix


def test_make_kernel_name_prefix_dense_forward():
    assert make_kernel_name_prefix(
        "flash_fwd",
        arch=100,
        dtype=cutlass.BFloat16,
        head_dim=128,
        head_dim_v=128,
        tile_m=128,
        tile_n=128,
        q_stage=2,
        num_threads=384,
        causal=True,
        use_2cta=True,
    ) == (
        "flash_fwd_sm100_bf16_head_dim128_tile128x128_q_stages2_threads384_"
        "causal_use_2cta"
    )


def test_make_kernel_name_prefix_backward():
    assert make_kernel_name_prefix(
        "flash_bwd",
        arch=100,
        dtype=cutlass.BFloat16,
        head_dim=128,
        qhead_per_kvhead=8,
        tile_m=128,
        tile_n=128,
        q_stage=2,
        dout_stage=2,
        num_threads=384,
        q_subtile_factor=2,
        causal=True,
        varlen=True,
        pack_gqa=True,
        use_2cta=True,
        deterministic=True,
        dq_single_wg=True,
        spt=True,
        cluster_size=2,
        has_score_mod=True,
        has_mask_mod=True,
        has_block_sparsity=True,
        has_aux=True,
    ) == (
        "flash_bwd_sm100_bf16_head_dim128_gqa_ratio8_tile128x128_q_stages2_"
        "dout_stages2_threads384_q_subtile2_causal_varlen_pack_gqa_use_2cta_"
        "deterministic_dq_single_wg_spt_scheduler_cluster_size2_score_mod_mask_mod_"
        "block_sparse_aux"
    )


def test_make_kernel_name_prefix_feature_tags():
    assert make_kernel_name_prefix(
        "flash_fwd",
        arch=100,
        dtype=torch.float16,
        head_dim=192,
        head_dim_v=128,
        qhead_per_kvhead=8,
        tile_m=192,
        tile_n=128,
        q_subtile_factor=2,
        varlen=True,
        paged=True,
        paged_non_tma=True,
        split_kv=True,
        pack_gqa=True,
        use_clc=True,
        has_score_mod=True,
        has_block_sparsity=True,
        has_aux=True,
    ) == (
        "flash_fwd_sm100_f16_head_dim192_value_dim128_gqa_ratio8_tile192x128_"
        "q_subtile2_varlen_paged_paged_non_tma_split_kv_pack_gqa_clc_scheduler_"
        "score_mod_block_sparse_aux"
    )
