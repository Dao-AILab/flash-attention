# Flash Attention NPU实现：任务分块与张量偏移计算解析

## 1. 引言

本文档详细解析Flash Attention在Ascend NPU上实现的核心代码段`mha_fwd_kvcache.cpp`第336-370行，该代码段负责将注意力计算任务分解为多个子任务，并精确计算每个子任务对应的张量数据在全局内存中的偏移量，为后续的矩阵乘法和Softmax计算做准备。

## 2. 代码功能概述

这段代码实现了以下核心功能：
- 计算当前批次内的任务索引
- 确定任务对应的查询序列分块和查询头分块
- 计算KV头索引和查询头起始位置
- 计算LSE张量的令牌偏移
- 计算各张量（Q、K、V、O、LSE）在全局内存中的精确偏移
- 处理边界分块的实际大小
- 计算实际行数并向上对齐到硬件块大小

## 3. 详细代码解析

### 3.1 当前批次内任务索引计算

```cpp
// 计算当前批次内的任务索引
uint32_t taskIdxCurBatch = taskIdx - preTotalTaskNum;
```

- **变量说明**：
  - `taskIdx`：全局任务索引（跨批次）
  - `preTotalTaskNum`：之前所有批次的总任务数
  - `taskIdxCurBatch`：当前批次内的相对任务索引

### 3.2 分块索引计算

```cpp
// 计算当前任务对应的查询序列分块索引和查询头分块索引
uint32_t qSBlockIdx = taskIdxCurBatch / curQNBlockNum;
uint32_t qNBlockIdx = taskIdxCurBatch - qSBlockIdx * curQNBlockNum;
uint32_t qNBlockIdxCurGroup = qNBlockIdx % qNBlockNumPerGroup;
```

- **变量说明**：
  - `qSBlockIdx`：查询序列分块索引（沿序列维度S的分块）
  - `qNBlockIdx`：查询头分块索引（沿头维度N的分块）
  - `qNBlockIdxCurGroup`：当前组内的查询头分块索引

### 3.3 KV头与查询头起始索引计算

```cpp
// 计算KV头索引和查询头起始索引
uint32_t kvNIdx = qNBlockIdx / qNBlockNumPerGroup;
uint32_t qNStartIdx = kvNIdx * groupSize + qNBlockIdxCurGroup * curQNBlockTile;
```

- **变量说明**：
  - `kvNIdx`：KV头索引（用于GQA分组注意力机制）
  - `qNStartIdx`：当前分块在查询头维度的起始索引

### 3.4 LSE张量令牌偏移计算

```cpp
// 计算LSE张量的令牌偏移
uint32_t lseTokenOffset = qSBlockIdx * curQSBlockTile * qHeads;
```

- **变量说明**：
  - `lseTokenOffset`：LSE（对数求和指数）张量在当前序列分块的令牌偏移量

### 3.5 全局内存偏移计算

```cpp
// 计算各张量在全局内存(GM)中的偏移量
uint64_t gmOffsetQ = qBOffset +
    static_cast<uint64_t>(qSBlockIdx * curQSBlockTile) * strideQ +
    static_cast<uint64_t>(qNStartIdx * embed);  // Q张量GM偏移
uint64_t gmOffsetK = kBOffset + static_cast<uint64_t>(kvNIdx * embed);  // K张量GM偏移
uint64_t gmOffsetV = vBOffset + static_cast<uint64_t>(kvNIdx * embedV);  // V张量GM偏移
uint64_t gmOffsetO = oBOffset +
    static_cast<uint64_t>(qSBlockIdx * curQSBlockTile) * strideO +
    static_cast<uint64_t>(qNStartIdx * embedV);  // O张量GM偏移
uint64_t gmOffsetLse = lseBOffset +
    static_cast<uint64_t>(lseTokenOffset + qNStartIdx);  // LSE张量GM偏移
```

- **变量说明**：
  - `gmOffsetQ`：Q张量在全局内存中的起始偏移
  - `gmOffsetK`：K张量在全局内存中的起始偏移
  - `gmOffsetV`：V张量在全局内存中的起始偏移
  - `gmOffsetO`：输出O张量在全局内存中的起始偏移
  - `gmOffsetLse`：LSE张量在全局内存中的起始偏移

### 3.6 实际分块大小计算

```cpp
// 计算实际的分块大小（最后一个分块可能小于标准分块大小）
uint32_t qSBlockSize = (qSBlockIdx == (curQSBlockNum - 1U)) ?
    (qSeqlen - qSBlockIdx * curQSBlockTile) : curQSBlockTile;  // 查询序列分块大小
uint32_t qNBlockSize = (qNBlockIdxCurGroup == (qNBlockNumPerGroup - 1U)) ?
    (groupSize - qNBlockIdxCurGroup * curQNBlockTile) : curQNBlockTile;  // 查询头分块大小
```

- **变量说明**：
  - `qSBlockSize`：当前查询序列分块的实际大小（处理最后一个不完整分块）
  - `qNBlockSize`：当前查询头分块的实际大小（处理最后一个不完整分块）

### 3.7 行数计算与对齐

```cpp
// 计算行数和向上对齐的行数
uint32_t rowNum = qSBlockSize * qNBlockSize;  // 实际行数
uint32_t rowNumRound = RoundUp(rowNum, FaiKenel::BLOCK_SIZE);  // 向上对齐到块大小的行数
```

- **变量说明**：
  - `rowNum`：当前任务需要处理的实际行数
  - `rowNumRound`：向上对齐到硬件块大小的行数（用于高效硬件计算）

## 4. 示意图解

```
┌───────────────────────────────────────────────────────────────────────────┐
│                             全局任务索引                                  │
│                                                                           │
│  taskIdx=0          taskIdx=1          taskIdx=2          taskIdx=3       │
│     │                  │                  │                  │            │
│     ▼                  ▼                  ▼                  ▼            │
│  ┌───────┐          ┌───────┐          ┌───────┐          ┌───────┐      │
│  │任务0   │          │任务1   │          │任务2   │          │任务3   │      │
│  └───────┘          └───────┘          └───────┘          └───────┘      │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                             批次内任务索引                                │
│                                                                           │
│ taskIdxCurBatch=0  taskIdxCurBatch=1  taskIdxCurBatch=2  taskIdxCurBatch=3│
│     │                  │                  │                  │            │
│     ▼                  ▼                  ▼                  ▼            │
│ ┌───────────────────────────────────────────────────────────────────────┐ │
│ │                          查询头分块 (qNBlockIdx)                        │ │
│ │                                                                       │ │
│ │       qNBlockIdx=0       qNBlockIdx=1       qNBlockIdx=2              │ │
│ │          │                  │                  │                       │ │
│ │          ▼                  ▼                  ▼                       │ │
│ │     ┌─────────┐        ┌─────────┐        ┌─────────┐                 │ │
│ │     │头分块0   │        │头分块1   │        │头分块2   │                 │ │
│ │     └─────────┘        └─────────┘        └─────────┘                 │ │
│ └───────────────────────────────────────────────────────────────────────┘ │
│                              ▲                                            │
│                              │                                            │
│ ┌───────────────────────────────────────────────────────────────────────┐ │
│ │                          查询序列分块 (qSBlockIdx)                      │ │
│ │                                                                       │ │
│ │            qSBlockIdx=0               qSBlockIdx=1                    │ │
│ │                 │                           │                          │ │
│ │                 ▼                           ▼                          │ │
│ │            ┌─────────┐                 ┌─────────┐                    │ │
│ │            │序列分块0 │                 │序列分块1 │                    │ │
│ │            └─────────┘                 └─────────┘                    │ │
│ └───────────────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                            张量全局内存偏移                                │
│                                                                           │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   │
│  │   Q张量      │   │   K张量      │   │   V张量      │   │   O张量      │   │
│  ├─────────────┤   ├─────────────┤   ├─────────────┤   ├─────────────┤   │
│  │             │   │             │   │             │   │             │   │
│  │             │   │             │   │             │   │             │   │
│  │             │   │             │   │             │   │             │   │
│  │    ▲        │   │    ▲        │   │    ▲        │   │    ▲        │   │
│  │    │        │   │    │        │   │    │        │   │    │        │   │
│  │ gmOffsetQ   │   │ gmOffsetK   │   │ gmOffsetV   │   │ gmOffsetO   │   │
│  └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘   │
│                                                                           │
│  ┌─────────────┐                                                         │
│  │   LSE张量     │                                                         │
│  ├─────────────┤                                                         │
│  │             │                                                         │
│  │             │                                                         │
│  │             │                                                         │
│  │    ▲        │                                                         │
│  │    │        │                                                         │
│  │ gmOffsetLse  │                                                         │
│  └─────────────┘                                                         │
└───────────────────────────────────────────────────────────────────────────┘
```

## 5. 关键概念说明

### 5.1 任务分解策略

- **二维任务网格**：将注意力计算分解为查询序列分块（S维度）和查询头分块（N维度）的二维网格
- **任务索引映射**：全局任务索引 → 批次内任务索引 → (序列分块索引, 头分块索引)
- **分组注意力支持**：通过`kvNIdx`和`qNStartIdx`支持分组查询注意力（GQA）机制

### 5.2 内存布局与偏移计算

- **多维偏移叠加**：批次偏移 + 序列分块偏移 + 头分块偏移
- **不同张量差异化处理**：
  - Q/O张量：需要序列和头两个维度的偏移
  - K/V张量：仅需要头维度的偏移（序列维度通过KV缓存机制处理）
  - LSE张量：特殊的偏移计算，与查询头数量相关

### 5.3 分块大小处理

- **标准分块**：大部分情况下使用固定的分块大小（`curQSBlockTile`, `curQNBlockTile`）
- **边界分块**：最后一个分块可能小于标准大小，需要特殊计算
- **硬件对齐**：将实际行数向上对齐到硬件块大小（`FaiKenel::BLOCK_SIZE`），提高计算效率

## 6. 设计意图与技术亮点

1. **高效并行化**：通过二维任务网格实现细粒度并行，充分利用NPU多核架构

2. **内存访问优化**：
   - 精确计算内存偏移，避免不必要的数据复制
   - 分块处理提高缓存命中率
   - 硬件对齐提升内存访问效率

3. **灵活性支持**：
   - 支持可变序列长度
   - 支持分组查询注意力（GQA）
   - 支持不同的输入布局（如TND）

4. **数值稳定性**：通过LSE张量的专门处理，维护Softmax计算的数值稳定性

## 7. 后续处理

完成这些计算后，代码将继续执行：
- KV序列的掩码处理
- QK^T矩阵乘法
- 在线Softmax计算
- PV矩阵乘法
- 结果输出

这些步骤共同构成了完整的Flash Attention前向计算流程。

## 8. 总结

这段代码是Flash Attention在Ascend NPU上高效实现的核心，通过精心设计的任务分解和内存管理策略，实现了注意力机制的高性能计算。它展示了如何在专用硬件上通过细粒度优化来充分发挥硬件能力，同时保持算法的正确性和灵活性。