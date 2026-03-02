/*
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef FLASH_ATTENTION_REGULAR_H
#define FLASH_ATTENTION_REGULAR_H

/**
 * @file tilingdata.h
 * @brief FlashAttention推理时的分块数据结构定义
 *
 * 该文件定义了FlashAttention在推理过程中使用的核心分块数据结构，
 * 包含了注意力计算所需的各种参数配置，是连接高层API和底层内核实现的关键数据结构。
 * 所有与分块大小、序列长度、头数、嵌入维度等相关的参数都集中存储在此结构中，
 * 并通过访问器方法提供统一的访问接口。
 */

/**
 * @struct FAInferTilingData
 * @brief FlashAttention推理分块数据结构
 *
 * 存储FlashAttention推理过程中所需的所有分块和计算参数，
 * 包括输入输出大小、分块配置、掩码类型等。这些参数由高层API根据输入数据的特性
 * 预先计算好，然后传递给底层内核进行高效的注意力计算。
 */
struct FAInferTilingData {
    uint32_t numHeads;            ///< 注意力头的总数 (Q矩阵的头数)
    uint32_t embeddingSize;       ///< 键值对的嵌入维度大小 (Q和K矩阵的头维度)
    uint32_t embeddingSizeV;      ///< 值的嵌入维度大小 (V矩阵的头维度，通常与embeddingSize相同)
    uint32_t numBlocks;           ///< KV缓存的总分块数量
    uint32_t blockSize;           ///< KV缓存中每个分块的大小 (序列长度维度)
    uint32_t maxQSeqlen;          ///< 查询序列的最大长度 (整个批次中的最大Q序列长度)
    uint32_t maxKvSeqlen;         ///< 键值序列的最大长度 (整个批次中的最大KV序列长度)
    uint32_t kvHeads;             ///< 键值头的数量 (用于分组查询注意力GQA，通常小于等于numHeads)
    uint32_t batch;               ///< 批次大小
    uint32_t maxNumBlocksPerBatch; ///< 每批次的最大分块数量 (用于分页缓存管理)
    uint32_t firstBatchTaskNum;   ///< 第一批次的任务数量 (用于多批次处理的任务分配)
    uint32_t totalTaskNum;        ///< 总任务数量 (所有批次的任务总数)
    uint32_t maskType;            ///< 掩码类型 (0: 无掩码, 1: 因果掩码, 2: 填充掩码等)
    uint64_t mm1OutSize;          ///< 第一个矩阵乘法（Q*K^T）的输出大小 (字节)
    uint64_t smOnlineOutSize;     ///< 在线Softmax的输出大小 (字节)
    uint64_t mm2OutSize;          ///< 第二个矩阵乘法（Attn*V）的输出大小 (字节)
    uint64_t UpdateSize;          ///< 更新缓冲区大小 (字节，用于中间结果累加)
    uint64_t workSpaceSize;       ///< 工作空间大小 (字节，用于内核执行过程中的临时存储)
    float scaleValue;             ///< 缩放因子 (用于QK乘积的缩放，通常为1/sqrt(embeddingSize))
    uint64_t padding1;            ///< 填充字段1 (用于内存对齐)
    uint64_t padding2;            ///< 填充字段2 (用于内存对齐)
    uint32_t padding3;            ///< 填充字段3 (用于内存对齐)

    /** @brief 获取注意力头的总数 */
    uint32_t get_numHeads() const { return numHeads; }
    
    /** @brief 获取键值对的嵌入维度大小 */
    uint32_t get_embeddingSize() const { return embeddingSize; }
    
    /** @brief 获取值的嵌入维度大小 */
    uint32_t get_embeddingSizeV() const { return embeddingSizeV; }
    
    /** @brief 获取分块的总数 */
    uint32_t get_numBlocks() const { return numBlocks; }
    
    /** @brief 获取每个分块的大小 */
    uint32_t get_blockSize() const { return blockSize; }
    
    /** @brief 获取查询序列的最大长度 */
    uint32_t get_maxQSeqlen() const { return maxQSeqlen; }
    
    /** @brief 获取键值序列的最大长度 */
    uint32_t get_maxKvSeqlen() const { return maxKvSeqlen; }
    
    /** @brief 获取键值头的数量 (用于分组查询注意力GQA) */
    uint32_t get_kvHeads() const { return kvHeads; }
    
    /** @brief 获取批次大小 */
    uint32_t get_batch() const { return batch; }
    
    /** @brief 获取每批次的最大分块数量 */
    uint32_t get_maxNumBlocksPerBatch() const { return maxNumBlocksPerBatch; }
    
    /** @brief 获取第一批次的任务数量 */
    uint32_t get_firstBatchTaskNum() const { return firstBatchTaskNum; }
    
    /** @brief 获取总任务数量 */
    uint32_t get_totalTaskNum() const { return totalTaskNum; }
    
    /** @brief 获取掩码类型 */
    uint32_t get_maskType() const { return maskType; }
    
    /** @brief 获取第一个矩阵乘法(Q*K^T)的输出大小 */
    uint64_t get_mm1OutSize() const { return mm1OutSize; }
    
    /** @brief 获取在线Softmax的输出大小 */
    uint64_t get_smOnlineOutSize() const { return smOnlineOutSize; }
    
    /** @brief 获取第二个矩阵乘法(Attn*V)的输出大小 */
    uint64_t get_mm2OutSize() const { return mm2OutSize; }
    
    /** @brief 获取更新缓冲区大小 */
    uint64_t get_UpdateSize() const { return UpdateSize; }
    
    /** @brief 获取工作空间大小 */
    uint64_t get_workSpaceSize() const { return workSpaceSize; }
    
    /** @brief 获取缩放因子 */
    float get_scaleValue() const { return scaleValue; }
    
    /** @brief 获取填充字段1 */
    uint64_t get_padding1() const { return padding1; }
    
    /** @brief 获取填充字段2 */
    uint64_t get_padding2() const { return padding2; }
    /** @brief 获取填充字段3 */
    uint32_t get_padding3() const { return padding3; }

    /** @brief 设置注意力头的总数 */
    void set_numHeads(uint32_t value) { numHeads = value; }
    /** @brief 设置键值对的嵌入维度大小 */
    void set_embeddingSize(uint32_t value) { embeddingSize = value; }
    /** @brief 设置值的嵌入维度大小 */
    void set_embeddingSizeV(uint32_t value) { embeddingSizeV = value; }
    /** @brief 设置分块的总数 */
    void set_numBlocks(uint32_t value) { numBlocks = value; }
    /** @brief 设置每个分块的大小 */
    void set_blockSize(uint32_t value) { blockSize = value; }
    /** @brief 设置查询序列的最大长度 */
    void set_maxQSeqlen(uint32_t value) { maxQSeqlen = value; }
    /** @brief 设置键值序列的最大长度 */
    void set_maxKvSeqlen(uint32_t value) { maxKvSeqlen = value; }
    /** @brief 设置键值头的数量 */
    void set_kvHeads(uint32_t value) { kvHeads = value; }
    /** @brief 设置批次大小 */
    void set_batch(uint32_t value) { batch = value; }
    /** @brief 设置每批次的最大分块数量 */
    void set_maxNumBlocksPerBatch(uint32_t value) { maxNumBlocksPerBatch = value; }
    /** @brief 设置第一批次的任务数量 */
    void set_firstBatchTaskNum(uint32_t value) { firstBatchTaskNum = value; }
    /** @brief 设置总任务数量 */
    void set_totalTaskNum(uint32_t value) { totalTaskNum = value; }
    /** @brief 设置掩码类型 */
    void set_maskType(uint32_t value) { maskType = value; }
    /** @brief 设置第一个矩阵乘法的输出大小 */
    void set_mm1OutSize(uint64_t value) { mm1OutSize = value; }
    /** @brief 设置在线Softmax的输出大小 */
    void set_smOnlineOutSize(uint64_t value) { smOnlineOutSize = value; }
    /** @brief 设置第二个矩阵乘法的输出大小 */
    void set_mm2OutSize(uint64_t value) { mm2OutSize = value; }
    /** @brief 设置更新大小 */
    void set_UpdateSize(uint64_t value) { UpdateSize = value; }
    /** @brief 设置工作空间大小 */
    void set_workSpaceSize(uint64_t value) { workSpaceSize = value; }
    /** @brief 设置缩放因子 */
    void set_scaleValue(float value) { scaleValue = value; }
    /** @brief 设置填充字段1 */
    void set_padding1(uint64_t value) { padding1 = value; }
    /** @brief 设置填充字段2 */
    void set_padding2(uint64_t value) { padding2 = value; }
    /** @brief 设置填充字段3 */
    void set_padding3(uint32_t value) { padding3 = value; }
};

#endif /* FLASH_ATTENTION_REGULAR_H */