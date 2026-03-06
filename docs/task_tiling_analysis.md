# FlashAttention-NPU任务分块策略分析

## 1. 核心分块代码分析

### 1.1 任务分块主逻辑

```cpp
for (int32_t batchIdx = 0; batchIdx < batch_size; batchIdx++) {
    uint64_t qSeqlen = seqlen_q;  // 当前批次的查询序列长度
    uint64_t kvSeqlen = *(seqlens_k_cpu + batchIdx);  // 当前批次的KV序列长度
    
    // 获取当前批次的QN块分块大小（N维度：头维度）
    uint64_t curQNBlockTile = GetQNBlockTile(qSeqlen, groupSize);
    // 计算每组的QN块数量（向上取整）
    uint64_t qNBlockNumPerGroup = (groupSize + curQNBlockTile - 1) / curQNBlockTile;
    // 计算当前批次的QN块总数（组数 × KV头数）
    uint64_t curQNBlockNum = qNBlockNumPerGroup * num_heads_k;
    
    // 获取当前批次的QS块分块大小（S维度：序列长度维度）
    uint64_t curQSBlockTile = GetQSBlockTile(kvSeqlen);
    // 计算当前批次的QS块数量（向上取整）
    uint64_t curQSBlockNum = (qSeqlen + curQSBlockTile - 1) / curQSBlockTile;
    
    // 计算当前批次的任务数量（QN块数 × QS块数）
    uint64_t curTaskNum = curQNBlockNum * curQSBlockNum;
    // 设置第一批次的任务数量（用于内核调度）
    if (batchIdx == 0) {
        tiling_cpu_ptr->set_firstBatchTaskNum(curTaskNum);
    }
    // 累加到总任务数量
    totalTaskNum += curTaskNum;
}
```

### 1.2 分块维度说明

FlashAttention采用二维分块策略，从两个维度对注意力计算进行分块：

| 维度 | 名称 | 描述 | 分块函数 |
|------|------|------|----------|
| N维度 | 头维度 | 注意力头的分组维度 | GetQNBlockTile |
| S维度 | 序列长度维度 | 输入序列的时间维度 | GetQSBlockTile |

## 2. 分块大小计算函数详解

### 2.1 QN块分块大小计算（头维度）

```cpp
uint32_t GetQNBlockTile(uint32_t qSeqlen, uint32_t groupSize) {
    uint32_t qRowNumCeil = Q_TILE_CEIL;  // Q序列的行分块大小上限，默认为128
    
    // 计算QN块的分块大小，确保能被N_SPLIT_HELPER(2)整除
    uint32_t qNBlockTile = (qSeqlen != 0) ?
        (qRowNumCeil / qSeqlen) / N_SPLIT_HELPER * N_SPLIT_HELPER : Q_TILE_CEIL;
    
    // 确保分块大小不超过groupSize，且至少为1
    qNBlockTile = std::min(qNBlockTile, groupSize);
    qNBlockTile = std::max(qNBlockTile, static_cast<uint32_t>(1));
    
    return qNBlockTile;
}
```

**设计特点**：
- 序列长度自适应：序列越长，分块越小
- 硬件优化：确保分块大小为2的倍数（N_SPLIT_HELPER=2）
- GQA支持：分块大小不超过groupSize
- 边界安全：确保分块大小至少为1

### 2.2 QS块分块大小计算（序列长度维度）

```cpp
uint32_t GetQSBlockTile(int64_t kvSeqlen) {
    // 目前简单返回Q_TILE_CEIL，可根据KV序列长度进行扩展优化
    uint32_t qSBlockTile = Q_TILE_CEIL;
    return qSBlockTile;
}
```

**设计特点**：
- 简单高效：当前实现直接返回Q_TILE_CEIL(128)
- 可扩展性：预留了根据KV序列长度动态调整的接口
- 硬件匹配：128的分块大小匹配NPU矢量处理单元宽度

## 3. 任务数量计算逻辑

### 3.1 头维度分块数量

```cpp
// 计算每组的QN块数量（向上取整）
uint64_t qNBlockNumPerGroup = (groupSize + curQNBlockTile - 1) / curQNBlockTile;
// 计算当前批次的QN块总数（组数 × KV头数）
uint64_t curQNBlockNum = qNBlockNumPerGroup * num_heads_k;
```

- **向上取整**：确保所有头都能被处理，避免遗漏
- **组数计算**：`num_heads_k`表示KV头的数量，每个KV头对应一组查询头

### 3.2 序列长度维度分块数量

```cpp
// 计算当前批次的QS块数量（向上取整）
uint64_t curQSBlockNum = (qSeqlen + curQSBlockTile - 1) / curQSBlockTile;
```

- **向上取整**：确保所有序列位置都能被处理
- **分块粒度**：由`curQSBlockTile`决定，当前固定为128

### 3.3 总任务数量

```cpp
// 计算当前批次的任务数量（QN块数 × QS块数）
uint64_t curTaskNum = curQNBlockNum * curQSBlockNum;
```

- **二维乘积**：每个QN块和QS块的组合形成一个独立任务
- **批次累加**：所有批次的任务数量总和为总任务数

## 4. 任务分块策略的设计原因

### 4.1 硬件资源适配

**NPU架构特点**：
- 多AI Core并行处理
- 有限的片上内存（L1/L0缓存）
- 矢量处理单元（VPU）宽度优化

**分块适配**：
- 128的基础分块大小匹配NPU VPU宽度
- 二维分块充分利用多核并行能力
- 分块大小限制确保数据能放入片上内存

### 4.2 内存访问优化

**FlashAttention核心思想**：
- 减少全局内存访问次数
- 利用片上内存进行分块计算
- 重叠计算与内存访问（流水线优化）

**分块的内存优势**：
- 提高缓存命中率：小分块数据更容易被缓存
- 减少内存带宽压力：每次只处理部分数据
- 隐藏内存延迟：通过双缓冲技术重叠计算与访存

### 4.3 并行计算最大化

**二维并行策略**：
- **N维度并行**：不同注意力头分组可以并行处理
- **S维度并行**：不同序列位置可以并行处理

**任务调度优化**：
```cpp
// 设置第一批次的任务数量（用于内核调度）
if (batchIdx == 0) {
    tiling_cpu_ptr->set_firstBatchTaskNum(curTaskNum);
}
```

- 第一批次任务数量用于内核调度决策
- 平衡工作负载分布
- 优化任务启动顺序

### 4.4 GQA（分组查询注意力）支持

**GQA特点**：
- 多个查询头共享同一个键值头
- 减少内存使用和计算量
- 适合大语言模型推理

**分块策略适配（GQA核心限制）**：
```cpp
// 确保分块大小不超过groupSize
qNBlockTile = std::min(qNBlockTile, groupSize);
```

这行代码是分组查询注意力（Grouped Query Attention, GQA）实现的核心限制，具有以下关键作用：

1. **GQA分组机制保障**
   - `groupSize = num_heads / num_heads_k`表示每组查询头的数量
   - 在GQA中，每组内的查询头共享相同的KV头
   - 分块大小超过groupSize会导致单个分块跨越多个分组

2. **避免跨分组访问的性能损失**
   - 跨分组访问需要额外的内存寻址和数据移动
   - 不同分组的KV缓存可能位于不同的内存位置
   - 保持分块在单个分组内可最大化缓存命中率

3. **优化内存访问模式**
   - 同一分组内的查询头具有连续的内存布局
   - 按分组分块可实现高效的向量加载操作
   - 减少内存访问的随机性，提升内存带宽利用率

4. **与硬件并行单元的匹配**
   - NPU的向量处理单元（VPU）一次可处理固定数量的头
   - 分块大小与groupSize匹配可最大化VPU利用率
   - 避免硬件资源的碎片化使用

5. **数值计算一致性保障**
   - 同一分组内的查询头共享KV，计算逻辑一致
   - 保持分块在分组内可简化数值计算流程
   - 提高计算精度和结果一致性

例如：当`num_heads=32`，`num_heads_k=8`时，`groupSize=4`，此时qNBlockTile将被限制为不超过4，确保每个分块只包含同一分组内的查询头。

### 4.5 长序列处理能力

**分块的长序列优势**：
- 避免长序列导致的内存溢出
- 支持任意长度的输入序列
- 保持计算复杂度为O(N²)但降低内存占用

**序列长度自适应**：
- 短序列：使用较大分块，提高计算效率
- 长序列：自动调整为较小分块，确保稳定性

## 5. 分块策略的技术优势

### 5.1 性能提升

- **计算利用率**：提高AI Core的计算资源利用率
- **内存带宽**：减少全局内存访问，提高有效带宽
- **流水线效率**：通过分块实现计算与访存的流水线重叠

### 5.2 内存效率

- **内存占用**：降低峰值内存使用，支持更大模型
- **缓存友好**：优化内存访问模式，提高缓存命中率
- **内存复用**：分块数据可以在片上内存中复用

### 5.3 灵活性与可扩展性

- **硬件适配**：容易适配不同NPU架构
- **序列长度**：支持从短序列到超长序列的各种场景
- **模型规模**：支持不同大小的注意力头和嵌入维度

### 5.4 稳定性与可靠性

- **边界处理**：完善的边界条件处理
- **错误避免**：防止除零错误和无效分块
- **异常输入**：处理零长度等特殊情况

## 6. 分块策略在实际计算中的应用

### 6.1 QK矩阵乘法

从`qk_matmul.hpp`中可以看到分块策略的实际应用：

```cpp
// N维度L1循环：处理序列长度方向的分块
for (uint32_t nL1Idx = 0; nL1Idx < nL1Loop; ++nL1Idx) {
    // 获取当前块的形状
    getBlockShape(actualShape, nL1Idx, nL1Loop, stackSeqTile);
    // ... 处理当前分块
    
    // M维度L0循环：处理token数量方向的分块
    uint32_t mL0Loop = CeilDiv(mActual, L0TileShape::M);
    // K维度L0循环：处理嵌入维度方向的分块
    uint32_t kL0Loop = CeilDiv(kActual, L0TileShape::K);
    
    // ... 进行矩阵乘法计算
}
```

### 6.2 结果更新

从`rescale_o.hpp`中可以看到分块结果的更新逻辑：

```cpp
// 计算子块分割参数
uint32_t qNSplitSubBlock = qNBlockSize / subBlockNum;      ///< 查询头的子块分割大小
uint32_t qNThisSubBlock = (qNBlockSize == 1U) ? 0          ///< 当前子块的查询头数
                          : (subBlockIdx == 1U) ? (qNBlockSize - qNSplitSubBlock)
                                               : qNSplitSubBlock;
uint32_t qSThisSubBlock = (qNBlockSize == 1U) ? inRowActualThisSubBlock : qSBlockSize;  ///< 当前子块的查询序列长度
```

## 7. 计算示例

### 7.1 短序列示例（qSeqlen=64）

```
参数：
- qSeqlen = 64
- kvSeqlen = 1024
- groupSize = 8
- num_heads_k = 4
- Q_TILE_CEIL = 128

计算过程：
1. curQNBlockTile = GetQNBlockTile(64, 8) = (128/64)/2*2 = 2 → min(2, 8) = 2
2. qNBlockNumPerGroup = (8 + 2 - 1)/2 = 4.5 → 5（向上取整）
3. curQNBlockNum = 5 × 4 = 20
4. curQSBlockTile = GetQSBlockTile(1024) = 128
5. curQSBlockNum = (64 + 128 - 1)/128 = 0.49 → 1（向上取整）
6. curTaskNum = 20 × 1 = 20
```

### 7.2 长序列示例（qSeqlen=1024）

```
参数：
- qSeqlen = 1024
- kvSeqlen = 4096
- groupSize = 8
- num_heads_k = 4
- Q_TILE_CEIL = 128

计算过程：
1. curQNBlockTile = GetQNBlockTile(1024, 8) = (128/1024)/2*2 = 0 → min(0, 8) = 0 → max(0, 1) = 1
2. qNBlockNumPerGroup = (8 + 1 - 1)/1 = 8
3. curQNBlockNum = 8 × 4 = 32
4. curQSBlockTile = GetQSBlockTile(4096) = 128
5. curQSBlockNum = (1024 + 128 - 1)/128 = 8.99 → 9（向上取整）
6. curTaskNum = 32 × 9 = 288
```

## 8. 优化建议

### 8.1 QS块分块大小优化

当前QS块大小固定为128，可以考虑根据KV序列长度动态调整：

```cpp
uint32_t GetQSBlockTile(int64_t kvSeqlen) {
    // 根据KV序列长度动态调整分块大小
    if (kvSeqlen < 1024) {
        return Q_TILE_CEIL;  // 短KV序列使用较大分块
    } else if (kvSeqlen < 8192) {
        return Q_TILE_CEIL / 2;  // 中等KV序列使用中等分块
    } else {
        return Q_TILE_CEIL / 4;  // 长KV序列使用较小分块
    }
}
```

### 8.2 动态调整N_SPLIT_HELPER

根据硬件平台和序列长度动态调整N_SPLIT_HELPER：

```cpp
uint32_t GetOptimalNSplitHelper(uint32_t qSeqlen, uint32_t hardwareType) {
    if (hardwareType == HW_TYPE_Ascend910B) {
        return (qSeqlen < 64) ? 4 : 2;
    } else {
        return 2;
    }
}
```

### 8.3 考虑嵌入维度

当前实现没有考虑嵌入维度大小，可以进一步优化：

```cpp
uint32_t GetQNBlockTile(uint32_t qSeqlen, uint32_t groupSize, uint32_t headSize) {
    uint32_t baseTile = Q_TILE_CEIL;
    // 根据头维度大小调整基础分块
    if (headSize > 128) {
        baseTile /= 2;
    }
    // 剩余计算逻辑...
}
```

## 9. 总结

FlashAttention-NPU的任务分块策略是其高性能的关键之一。通过二维分块（头维度和序列长度维度），它实现了：

1. **硬件高效利用**：充分发挥NPU的多核并行能力和矢量处理单元
2. **内存优化**：减少全局内存访问，提高缓存命中率，降低内存占用
3. **灵活性**：支持不同序列长度、注意力头数量和分组大小
4. **可扩展性**：容易适配不同NPU架构和模型规模

这种分块策略使得FlashAttention能够在保持O(N²)计算复杂度的同时，显著降低内存占用和提高计算效率，特别适合大语言模型的训练和推理场景。