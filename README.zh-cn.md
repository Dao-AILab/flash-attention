# FlashAttention 项目简介

本仓库为 FlashAttention 及 FlashAttention-2 的官方实现。

**FlashAttention: 快速且高效的精确注意力机制**
作者：Tri Dao 等
论文：https://arxiv.org/abs/2205.14135

**FlashAttention-2: 更快的注意力机制与更优的并行与分工**
作者：Tri Dao
论文：https://tridao.me/publications/flash2/flash2.pdf

## 用法
FlashAttention 被广泛采用，详细用法见 usage.md。

## FlashAttention-3 Beta
针对 Hopper GPU 优化。安装与测试方法：
```
cd hopper
python setup.py install
export PYTHONPATH=$PWD
pytest -q -s test_flash_attn.py
```

## FlashAttention-4 (CuTeDSL)
针对 Hopper/Blackwell GPU，使用 CuTeDSL 编写。
安装：
```
pip install flash-attn-4
```
用法：
```
from flash_attn.cute import flash_attn_func
out = flash_attn_func(q, k, v, causal=True)
```

## 安装与依赖
- 需 CUDA 或 ROCm 工具包
- 需 PyTorch 2.2+
- 需 packaging、psutil、ninja 等 Python 包
- 推荐 Linux

安装：
```
pip install flash-attn --no-build-isolation
```
或源码编译：
```
python setup.py install
```

## CUDA 支持
- 支持 Ampere、Ada、Hopper GPU
- 支持 fp16/bf16
- 支持 head_dim ≤ 256

## ROCm 支持
- 支持 MI200/250/300/355 GPU
- 支持 fp16/bf16
- Triton 后端支持更多特性

## 主要函数
- flash_attn_qkvpacked_func
- flash_attn_func

## 测试
```
pytest -q -s tests/test_flash_attn.py
```

## 参考文献
如使用本代码，请引用原论文。
