# Flash Attention NPU 中 KV Cache 实现分析

## 1. 概述

在 `mha_fwd_kvcache.cpp` 文件中，KV Cache（键值缓存）是通过两种模式实现的：**分页缓存模式**和**非分页缓存模式**。通过模板参数 `PAGED_CACHE_FLAG` 控制使用哪种模式。

## 2. 核心数据结构

### 2.1 块表 (Block Table)

块表是 KV Cache 实现的核心数据结构，用于管理缓存块的位置：

```cpp
AscendC::GlobalTensor<int32_t> gBlockTable;  // 块表（用于分页KV缓存）
gBlockTable.SetGlobalBuffer((__gm__ int32_t *)(params.blockTables));
```

块表存储了 KV 缓存块在全局内存中的位置信息，是实现高效缓存访问的关键。

### 2.2 分页块大小

定义了 KV 缓存中每个块的大小：

```cpp
uint32_t pagedBlockSize = fATilingData->blockSize;   // 分页块大小
```

## 3. KV Cache 模式区分

### 3.1 批次偏移处理

在批次切换时，两种模式有不同的偏移处理方式：

```cpp
if constexpr (!PAGED_CACHE_FLAG) {
    // 非分页缓存模式：更新K和V张量的批次偏移
    kBOffset += static_cast<uint64_t>(kvSeqlen * strideK);
    vBOffset += static_cast<uint64_t>(kvSeqlen * strideV);
} else {
    // 分页缓存模式：更新块表的批次偏移
    blockBOffset += static_cast<uint64_t>(maxNumBlocksPerBatch);
}
```

- **非分页模式**：直接更新 K 和 V 张量的全局内存偏移
- **分页模式**：更新块表的偏移，通过块表间接访问 KV 缓存

### 3.2 QK 矩阵乘法中的 KV Cache 使用

在 QK 矩阵乘法中，根据缓存模式选择不同的块表访问方式：

```cpp
if constexpr (PAGED_CACHE_FLAG) {
    // 分页缓存模式下的QK矩阵乘法
    blockMmadQK(
        gQ[gmOffsetQ],      // Q矩阵全局内存地址
        gK[gmOffsetK],      // K矩阵全局内存地址
        gS[gmOffsetS],      // S矩阵全局内存地址
        gBlockTable[blockBOffset],  // 块表（带批次偏移）
        layoutQTemp,        // Q矩阵布局
        layoutKTemp,        // K矩阵布局
        layOutS,            // S矩阵布局
        actualBlockShapeQK, // 实际块形状
        kvSIdx,             // 当前KV栈索引
        kvSLoopNumTotal,    // 总KV栈数量
        pagedBlockSize,     // 分页块大小
        strideK);           // K矩阵步长
} else {
    // 非分页缓存模式下的QK矩阵乘法
    blockMmadQK(
        gQ[gmOffsetQ],
        gK[gmOffsetK],
        gS[gmOffsetS],
        gBlockTable,        // 完整块表（无批次偏移）
        layoutQTemp,
        layoutKTemp,
        layOutS,
        actualBlockShapeQK,
        kvSIdx,
        kvSLoopNumTotal,
        pagedBlockSize,
        strideK);
}
```

### 3.3 PV 矩阵乘法中的 KV Cache 使用

在 PV 矩阵乘法中，同样区分了两种缓存模式：

```cpp
if constexpr (PAGED_CACHE_FLAG) {
    blockMmadPV(
        gP[gmOffsetP],  // P矩阵（注意力权重）
        gV[gmOffsetV],  // V矩阵（值张量）
        gOTmp[gmOffsetOTmp],  // 临时输出张量OTmp
        gBlockTable[blockBOffset],  // 块表（带批次偏移）
        layoutPTemp,  // P矩阵布局
        layoutVTemp,  // V矩阵布局
        layoutOTmp,  // OTmp矩阵布局
        actualBlockShapePV,
        nowkvSIdx,
        kvSLoopNumTotal,
        pagedBlockSize,
        noSkipKvS,
        strideV,
        blockStackNum,
        softmaxReady);
} else {
    // 非分页缓存模式下的PV矩阵乘法
    blockMmadPV(
        gP[gmOffsetP],  // P矩阵（注意力权重）
        gV[gmOffsetV],  // V矩阵（值张量）
        gOTmp[gmOffsetOTmp],  // 临时输出张量OTmp
        gBlockTable,  // 完整块表（无批次偏移）
        layoutPTemp,  // P矩阵布局
        layoutVTemp,  // V矩阵布局
        layoutOTmp,  // OTmp矩阵布局
        actualBlockShapePV,  // PV矩阵乘法的实际块形状
        nowkvSIdx,  // 当前KV序列索引
        kvSLoopNumTotal,  // KV序列总循环次数
        pagedBlockSize,  // 分页块大小（非分页模式下可能未使用）
        noSkipKvS,  // 不跳过的KV序列长度
        strideV,  // V矩阵的步长
        blockStackNum,  // 块栈数量
        softmaxReady);  // Softmax完成标志
}
```

## 4. KV Cache 工作机制

### 4.1 分页缓存模式

在分页缓存模式下：
1. KV 张量被分割成固定大小的块（`pagedBlockSize`）
2. 块表（`gBlockTable`）记录每个块在全局内存中的位置
3. 矩阵乘法操作通过块表间接访问 KV 缓存
4. 批次切换时，仅更新块表的偏移量

### 4.2 非分页缓存模式

在非分页缓存模式下：
1. KV 张量以连续内存形式存储
2. 矩阵乘法直接通过偏移量访问 KV 缓存
3. 批次切换时，需要更新 K 和 V 张量的全局偏移量

## 5. 优势与应用场景

### 5.1 分页缓存模式优势

- **内存利用率高**：可以更灵活地管理碎片化内存
- **支持动态扩展**：能够有效处理变长序列
- **减少内存拷贝**：通过块表直接访问所需数据

### 5.2 非分页缓存模式优势

- **访问速度快**：连续内存访问具有更好的缓存局部性
- **实现简单**：无需额外的块表管理逻辑

## 6. 代码优化建议

1. **块大小调优**：根据实际硬件和模型参数调整 `pagedBlockSize`，以达到最佳性能
2. **预取优化**：在分页模式下，可以考虑添加预取机制，减少内存访问延迟
3. **缓存淘汰策略**：如果支持长序列处理，可以实现 LRU 等缓存淘汰策略

## 7. 总结

Flash Attention NPU 实现中的 KV Cache 通过分页和非分页两种模式，灵活适应不同的硬件环境和应用场景。块表是实现高效缓存访问的核心机制，通过模板参数可以在编译时选择合适的缓存模式。

KV Cache 的使用显著减少了重复计算，提高了模型推理效率，特别是在长序列处理和实时应用中具有重要意义。