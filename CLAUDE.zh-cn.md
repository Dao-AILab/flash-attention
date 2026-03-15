# CLAUDE.md 中文版

本文件为 FlashAttention 仓库的代码阅读与开发指南（中文版）。

## 项目概览

FlashAttention-4（FA4）：基于 CuTeDSL（NVIDIA CUTLASS DSL）实现的高效注意力核，支持 Hopper（SM90）与 Blackwell（SM100/110）GPU。包名：flash-attn-4。

仓库还包含旧版本（FA2 在 csrc/，FA3 在 hopper/），但主力开发在 flash_attn/cute/。

## 构建与安装

```
pip install flash-attn-4
# 或开发模式：
pip install -e "flash_attn/cute[dev]"
```
依赖：nvidia-cutlass-dsl、torch、einops、apache-tvm-ffi、quack-kernels。

## 测试

```
pytest tests/cute/test_flash_attn.py
pytest tests/cute/test_flash_attn_varlen.py
pytest tests/cute/test_mask_mod.py
pytest tests/cute/test_score_mod.py
pytest tests/cute/test_block_sparsity.py
```

### 快速两阶段测试
第一阶段并行编译所有 kernel（无需 GPU），第二阶段用缓存 kernel 跑测试。

## 代码架构

### 公共 API
- flash_attn_func(q, k, v, ...)
- flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, ...)

### 前向/反向 Kernel
- flash_fwd.py / flash_fwd_sm100.py / flash_fwd_combine.py
- flash_bwd.py / flash_bwd_sm90.py / flash_bwd_sm100.py / flash_bwd_preprocess.py / flash_bwd_postprocess.py

### 核心抽象
- softmax.py、mask.py、block_info.py、seqlen_info.py、pipeline.py、tile_scheduler.py、copy_utils.py、named_barrier.py

### 架构相关
- hopper_helpers.py、blackwell_helpers.py、mma_sm100_desc.py

### 其他组件
- pack_gqa.py、paged_kv.py、fast_math.py、utils.py、cache_utils.py、cute_dsl_utils.py

### 编译与缓存
- kernel JIT 编译，支持内存与磁盘缓存

## 调试 GPU Kernel
- 详见 AI/DEBUG_2CTA.md
- 关键工具：cute.printf、compute-sanitizer、CUTE_DSL_KEEP_PTX=1

如需详细模块说明或英文原文，可随时查阅原 CLAUDE.md。