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

#ifndef CATLASS_GEMM_BLOCK_MMAD_QK_HPP_T
#define CATLASS_GEMM_BLOCK_MMAD_QK_HPP_T

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "fa_block.h"

namespace Catlass::Gemm::Block {

/**
 * @brief Q-K矩阵乘法的块级矩阵乘法实现
 * 
 * 为Atlas A2架构实现的Q-K矩阵乘法块级操作，用于FlashAttention计算中的Q*K^T步骤。
 * 
 * @tparam PAGED_CACHE_FLAG_ 分页缓存标志
 * @tparam ENABLE_UNIT_FLAG_ 启用单元标志
 * @tparam L1TileShape_ L1缓存的分块形状
 * @tparam L0TileShape_ L0缓存的分块形状
 * @tparam AType_ A矩阵类型（查询矩阵Q）
 * @tparam BType_ B矩阵类型（键矩阵K）
 * @tparam CType_ C矩阵类型（输出矩阵）
 * @tparam BiasType_ 偏置类型
 * @tparam TileCopy_ 分块复制策略
 * @tparam TileMmad_ 分块矩阵乘法策略
 */
template <
    bool PAGED_CACHE_FLAG_,      ///< 分页缓存标志
    bool ENABLE_UNIT_FLAG_,      ///< 启用单元标志
    class L1TileShape_,          ///< L1缓存的分块形状
    class L0TileShape_,          ///< L0缓存的分块形状
    class AType_,                ///< A矩阵类型（查询矩阵Q）
    class BType_,                ///< B矩阵类型（键矩阵K）
    class CType_,                ///< C矩阵类型（输出矩阵）
    class BiasType_,             ///< 偏置类型
    class TileCopy_,             ///< 分块复制策略
    class TileMmad_>             ///< 分块矩阵乘法策略
struct BlockMmad<
    MmadAtlasA2FAIQKT<PAGED_CACHE_FLAG_, ENABLE_UNIT_FLAG_>,  ///< Atlas A2 Q-K矩阵乘法调度策略
    L1TileShape_,
    L0TileShape_,
    AType_,
    BType_,
    CType_,
    BiasType_,
    TileCopy_,
    TileMmad_> {
public:
    // 类型别名定义
    using DispatchPolicy = MmadAtlasA2FAIQKT<PAGED_CACHE_FLAG_, ENABLE_UNIT_FLAG_>;  ///< 调度策略类型
    using ArchTag = typename DispatchPolicy::ArchTag;                                ///< 架构标签类型
    using L1TileShape = L1TileShape_;                                                ///< L1缓存分块形状
    using L0TileShape = L0TileShape_;                                                ///< L0缓存分块形状
    using ElementA = typename AType_::Element;                                       ///< A矩阵元素类型（Q）
    using LayoutA = typename AType_::Layout;                                         ///< A矩阵布局类型（Q）
    using ElementB = typename BType_::Element;                                       ///< B矩阵元素类型（K）
    using LayoutB = typename BType_::Layout;                                         ///< B矩阵布局类型（K）
    using ElementC = typename CType_::Element;                                       ///< C矩阵元素类型（输出）
    using LayoutC = typename CType_::Layout;                                         ///< C矩阵布局类型（输出）
    using TileMmad = TileMmad_;                                                      ///< 分块矩阵乘法类型
    using CopyGmToL1A = typename TileCopy_::CopyGmToL1A;                             ///< GM到L1的A矩阵复制
    using CopyGmToL1B = typename TileCopy_::CopyGmToL1B;                             ///< GM到L1的B矩阵复制
    using CopyL1ToL0A = typename TileCopy_::CopyL1ToL0A;                             ///< L1到L0的A矩阵复制
    using CopyL1ToL0B = typename TileCopy_::CopyL1ToL0B;                             ///< L1到L0的B矩阵复制
    using CopyL0CToGm = typename TileCopy_::CopyL0CToGm;                             ///< L0到GM的C矩阵复制
    using ElementAccumulator = typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;  ///< 累加器元素类型
    using LayoutAInL1 = typename CopyL1ToL0A::LayoutSrc;                             ///< A矩阵在L1中的布局
    using LayoutBInL1 = typename CopyL1ToL0B::LayoutSrc;                             ///< B矩阵在L1中的布局
    using LayoutAInL0 = typename CopyL1ToL0A::LayoutDst;                             ///< A矩阵在L0中的布局
    using LayoutBInL0 = typename CopyL1ToL0B::LayoutDst;                             ///< B矩阵在L0中的布局
    using LayoutCInL0 = layout::zN;                                                  ///< C矩阵在L0中的布局

    using L1AAlignHelper = Gemm::helper::L1AlignHelper<ElementA, LayoutA>;  ///< A矩阵L1对齐辅助类
    using L1BAlignHelper = Gemm::helper::L1AlignHelper<ElementB, LayoutB>;  ///< B矩阵L1对齐辅助类

    // 常量定义
    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;  ///< 流水线阶段数
    static constexpr uint32_t L1A_SIZE = L1TileShape::M * L1TileShape::K * sizeof(ElementA);  ///< L1中A矩阵的大小
    static constexpr uint32_t L1B_SIZE = L1TileShape::N * L1TileShape::K * sizeof(ElementB);  ///< L1中B矩阵的大小
    static constexpr uint32_t L0A_SIZE = ArchTag::L0A_SIZE;  ///< L0中A矩阵的大小
    static constexpr uint32_t L0B_SIZE = ArchTag::L0B_SIZE;  ///< L0中B矩阵的大小
    static constexpr uint32_t L0C_SIZE = ArchTag::L0C_SIZE;  ///< L0中C矩阵的大小
    static constexpr uint32_t L0A_PINGPONG_BUF_SIZE = L0A_SIZE / STAGES;  ///< L0 A矩阵乒乓缓冲区大小
    static constexpr uint32_t L0B_PINGPONG_BUF_SIZE = L0B_SIZE / STAGES;  ///< L0 B矩阵乒乓缓冲区大小
    static constexpr uint32_t L0C_PINGPONG_BUF_SIZE = L0C_SIZE / STAGES;  ///< L0 C矩阵乒乓缓冲区大小
    static constexpr uint32_t BLOCK_SIZE = 16;  ///< 块大小
    static constexpr uint32_t EMBED_SPLIT_SIZE = 128;  ///< 嵌入维度分割大小
    static constexpr uint32_t UNIT_BLOCK_STACK_NUM = 4;  ///< 单元块堆叠数量
    static constexpr uint32_t KV_BASE_BLOCK = 512;  ///< KV基础块大小
    static constexpr uint32_t KV_SPLIT_SIZE = 128;  ///< KV分割大小
    static constexpr uint32_t COORD_DIM0 = 0;  ///< 坐标维度0（M维度）
    static constexpr uint32_t COORD_DIM1 = 1;  ///< 坐标维度1（N维度）
    static constexpr uint32_t COORD_DIM2 = 2;  ///< 坐标维度2（K维度）

    // 静态断言：确保LayoutC是行优先布局
    static_assert(std::is_same_v<LayoutC, layout::RowMajor>, "LayoutC only support RowMajor yet!");

    /**
     * @brief 构造函数
     * 
     * 初始化BlockMmad类的实例，分配L1和L0缓存空间，设置动态维度参数。
     * 
     * @param resource 架构资源，用于获取L1和L0缓存缓冲区
     * @param nDyn N维度的动态大小，用于计算缓存分配
     * @param kDyn K维度的动态大小，用于计算缓存分配
     * @param KVStackLen KV堆叠长度，默认512
     * @param l1BufAddrStart L1缓冲区起始地址，默认0
     */
    __aicore__ inline
    BlockMmad(Arch::Resource<ArchTag> &resource, uint32_t nDyn, uint32_t kDyn, uint32_t KVStackLen = 512, uint32_t l1BufAddrStart = 0)
    {
        maxKVStackLen = KVStackLen;  ///< 设置KV堆叠的最大长度
        
        // 分配L1内存空间：
        // 1. 首先分配L1中的A矩阵(Q)缓冲区
        l1ATensor = resource.l1Buf.template GetBufferByByte<ElementA>(l1BufAddrStart);
        
        // 2. 为每个流水线阶段分配L1中的B矩阵(K)缓冲区和L0中的A/B/C矩阵缓冲区
        for (uint32_t i = 0; i < STAGES; i++) {
            // L1中B矩阵的偏移量计算：A矩阵大小 + 当前阶段的B矩阵偏移
            l1BTensor[i] = resource.l1Buf.template GetBufferByByte<ElementB>(l1BufAddrStart +
                L1TileShape::M * kDyn * sizeof(ElementA) + nDyn * kDyn * sizeof(ElementB) * i);
            
            // L0中A/B/C矩阵的乒乓缓冲区分配，每个阶段使用独立的缓冲区
            l0ATensor[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_PINGPONG_BUF_SIZE * i);
            l0BTensor[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_PINGPONG_BUF_SIZE * i);
            l0CTensor[i] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(L0C_PINGPONG_BUF_SIZE * i);
        }
        
        // 保存动态维度参数
        l1NDynamic = nDyn;
        l1KDynamic = kDyn;
    }

    /**
     * @brief 析构函数
     */
    __aicore__ inline
    ~BlockMmad() {}

    /**
     * @brief 从全局内存加载查询矩阵Q到L1缓存
     * 
     * 该函数将查询矩阵Q从全局内存加载到L1缓存中，支持分组查询头的处理。
     * 
     * @param gA 全局内存中的A矩阵（Q）
     * @param layoutA A矩阵的布局
     * @param rowNum 行数（token数量）
     * @param singleGroupHeads 单组头数
     * @param qHeads 查询头数
     */
    __aicore__ inline
    void loadQGM(
        AscendC::GlobalTensor<ElementA> gA,
        LayoutA layoutA,
        uint32_t rowNum, uint32_t &singleGroupHeads, uint32_t &qHeads)
    {
        uint32_t embed = layoutA.shape(1);  ///< 获取嵌入维度大小
        
        // 将行数向上对齐到L1对齐边界
        uint32_t rowNumRound = RoundUp(rowNum, L1AAlignHelper::M_ALIGNED);
        
        // 计算每组的token数量
        uint32_t tokenNumPerGroup = rowNum / singleGroupHeads;
        
        // 获取单组的布局和L1中的布局
        auto layoutSingleANd = layoutA.GetTileLayout(MakeCoord(singleGroupHeads, embed));
        LayoutAInL1 layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(rowNum, embed);
        
        // 执行从全局内存到L1的复制操作
        copyGmToL1A(
            l1ATensor, gA,              ///< 目标L1张量和源全局张量
            layoutAInL1, layoutSingleANd,///< 目标和源布局
            tokenNumPerGroup, qHeads * embed, tokenNumPerGroup, BLOCK_SIZE, rowNumRound);///< 复制参数
        
        // 设置并等待硬事件，确保复制完成
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID3);
    }

    /**
     * @brief 设置块参数
     * 
     * 计算并设置块的起始位置、结束位置和总块数。
     * 
     * @param stackSeqTile 堆叠序列的tile数
     * @param blockStart 块起始位置（输出参数）
     * @param blockEnd 块结束位置（输出参数）
     * @param curBlockTotalNum 当前块总数（输出参数）
     * @param blockSize 块大小
     */
    __aicore__ inline
    void setBlockParam(uint32_t stackSeqTile, uint32_t &blockStart, uint32_t &blockEnd, uint32_t &curBlockTotalNum, uint32_t blockSize){
        if(stackSeqTile >= blockStart && blockSize != 0) {
            // 计算当前块的结束位置
            blockEnd = ((stackSeqTile - blockStart) % blockSize == 0) ? blockSize : (stackSeqTile - blockStart) % blockSize;
            // 计算总块数（向上取整）
            curBlockTotalNum = (((stackSeqTile - blockStart) + blockSize - 1) / blockSize) + 1;
        } else {
            // 特殊情况处理：只有一个块
            curBlockTotalNum = 1;
            blockStart = stackSeqTile;
            blockEnd = stackSeqTile + blockStartOffset;
        }
    }
    
    __aicore__ inline
    void getBlockShape(GemmCoord &actualShape, uint32_t nL1Idx, uint32_t nL1Loop, uint32_t stackSeqTile)
    {
        uint32_t nSplitSize = l1NDynamic;
        if (nL1Idx == nL1Loop - 1U) {
            nSplitSize = stackSeqTile - nL1Idx * l1NDynamic;
        }
        actualShape[COORD_DIM1] = nSplitSize;
    }

    /**
     * @brief 获取块形状（用于分页缓存模式）
     * 
     * 计算当前处理块的实际形状，考虑块起始偏移、剩余动态大小和已处理长度。
     * 
     * @param actualShape 实际形状（输出参数）
     * @param blockStartOffset 块起始偏移
     * @param l1NResDynamic L1中N维度的剩余动态大小
     * @param kvL1Len 已处理的KV长度
     * @param nowLen 当前处理长度（输出参数）
     * @param blockSize 块大小
     */
    __aicore__ inline
    void getBlockShape(GemmCoord &actualShape, uint32_t& blockStartOffset, uint32_t& l1NResDynamic, uint32_t& kvL1Len, uint32_t& nowLen, uint32_t& blockSize)
    {
        // 计算当前处理长度：取块剩余部分和L1剩余部分中的较小值
        nowLen = (blockSize - blockStartOffset < l1NResDynamic - kvL1Len) ?
                blockSize - blockStartOffset :
                l1NResDynamic - kvL1Len;
        // 设置实际形状的N维度
        actualShape[COORD_DIM1] = nowLen;
    }

    __aicore__ inline
    void getKVOffset(uint32_t &kOffset, uint32_t nIdx, uint32_t nowNIdx, uint32_t strideKV)
    {
        kOffset = nIdx * maxKVStackLen * strideKV + nowNIdx * l1NDynamic * strideKV;
    }

    /**
     * @brief 获取KV偏移（用于分页缓存模式）
     * 
     * 根据块表获取KV在全局内存中的偏移量。
     * 
     * @param gBlockTable 块表，存储块的位置信息
     * @param kOffset K矩阵在全局内存中的偏移（输出参数）
     * @param nowNIdx 当前N维度索引
     * @param startOffset 起始偏移
     * @param strideKV KV的步长
     * @param blockSize 块大小
     */
    __aicore__ inline
    void getKVOffset(AscendC::GlobalTensor<int32_t> &gBlockTable, uint32_t &kOffset, uint32_t nowNIdx, 
        uint32_t startOffset, uint32_t strideKV, uint32_t blockSize)
    {
        // 获取当前块在块表中的ID
        uint32_t blockTableId = gBlockTable.GetValue(nowNIdx);
        // 计算K矩阵在全局内存中的偏移量
        kOffset = blockTableId * blockSize * strideKV + startOffset * strideKV;
    }

    __aicore__ inline
    void resetBlockStart(){
        blockStartOffset = 0;
    }

    /**
     * @brief 更新块偏移
     * 
     * 更新块的起始偏移和当前块索引，用于分页缓存模式的块迭代。
     * 
     * @param nowLen 当前处理的长度
     * @param curBlockIdx 当前块索引（输出参数）
     * @param blockSize 块大小
     */
    __aicore__ inline
    void updateBlockOffset(uint32_t nowLen, uint32_t &curBlockIdx, uint32_t blockSize){
        if(blockStartOffset + nowLen == blockSize){
            // 当前块处理完成，重置偏移并进入下一个块
            blockStartOffset = 0;
            curBlockIdx++;
        } else{
            // 更新块起始偏移，继续处理当前块的剩余部分
            blockStartOffset += nowLen;
        }
    }

    /**
     * @brief 核心操作符：执行Q-K矩阵乘法
     * 
     * 实现Q-K矩阵乘法的核心逻辑，支持普通模式和分页缓存模式，采用三级循环结构（N/M/K维度）
     * 和双缓冲机制以提高性能。
     * 
     * @param gA 全局内存中的A矩阵（Q）
     * @param gB 全局内存中的B矩阵（K）
     * @param gC 全局内存中的C矩阵（输出）
     * @param gBlockTable 块表（用于分页缓存模式）
     * @param layoutA A矩阵的布局
     * @param layoutB B矩阵的布局
     * @param layoutC C矩阵的布局
     * @param actualOriShape 实际原始形状
     * @param nIdx N维度索引
     * @param nLoop N维度循环次数
     * @param blockSize 块大小
     * @param strideKV KV的步长
     */
    __aicore__ inline
    void operator()(AscendC::GlobalTensor<ElementA> gA,
                    AscendC::GlobalTensor<ElementB> gB,
                    AscendC::GlobalTensor<ElementC> gC,
                    AscendC::GlobalTensor<int32_t> gBlockTable,
                    LayoutA layoutA, LayoutB layoutB, LayoutC layoutC, GemmCoord actualOriShape,
                    uint32_t nIdx, uint32_t nLoop, uint32_t blockSize, uint32_t strideKV)
    {
        // 解析输入参数
        uint32_t rowNum = actualOriShape[COORD_DIM0];       ///< M维度大小（token数量）
        uint32_t stackSeqTile = actualOriShape[COORD_DIM1];  ///< N维度大小（序列长度）
        uint32_t embed = actualOriShape[COORD_DIM2];         ///< K维度大小（嵌入维度）

        GemmCoord actualShape{rowNum, 0, embed};  ///< 实际处理形状
        uint32_t gBOffset = 0;                    ///< B矩阵在全局内存中的偏移

        // 创建L1中A矩阵的布局
        LayoutAInL1 layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(rowNum, embed);

        // 计算循环参数
        uint32_t tileNNumPerBaseBlock = blockSize / l1NDynamic;  ///< 每个基础块的N维度tile数
        uint32_t nL1Loop = CeilDiv(stackSeqTile, l1NDynamic);    ///< N维度L1循环次数（向上取整）
        uint32_t curBlockIdx = 0;                                ///< 当前块索引
        uint32_t blockStart = 0;                                 ///< 块起始位置
        uint32_t blockEnd = 0;                                   ///< 块结束位置
        uint32_t curBlockTotalNum = 0;                           ///< 当前块总数
        
        // 分页缓存模式初始化
        if constexpr (PAGED_CACHE_FLAG_){
            blockStart = blockSize - blockStartOffset;
            setBlockParam(stackSeqTile, blockStart, blockEnd, curBlockTotalNum, blockSize);
        }
        
        // N维度L1循环：处理序列长度方向的分块
        for (uint32_t nL1Idx = 0; nL1Idx < nL1Loop; ++nL1Idx) {
            uint32_t mActual = actualShape.m();  ///< M维度实际大小
            uint32_t kActual = actualShape.k();  ///< K维度实际大小
            uint32_t nActual = actualShape.n();  ///< N维度实际大小
            LayoutBInL1 layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(kActual, nActual);
            
            // 分页缓存模式处理逻辑
            if constexpr (PAGED_CACHE_FLAG_){
                // 计算当前L1块的N维度剩余动态大小
                uint32_t l1NResDynamic = (nL1Idx < (nL1Loop-1)) ? l1NDynamic : (stackSeqTile - nL1Idx * l1NDynamic);
                layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(embed, l1NResDynamic);
                
                uint32_t kvL1Len = 0;  ///< 已处理的KV长度
                
                // 等待L1 KV乒乓缓冲区就绪
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1KvPingPongFlag);
                
                // 循环加载KV数据到L1缓存，直到填满当前L1块
                while(kvL1Len < l1NResDynamic){
                    uint32_t nowLen = 0;  ///< 当前处理的长度
                    // 计算当前块的大小（最后一块可能较小）
                    uint32_t curBlockSize = (curBlockIdx < (curBlockTotalNum-1)) ? blockSize : blockEnd;
                    // 计算当前N维度索引
                    uint32_t nowNIdx = nIdx * maxKVStackLen / blockSize + curBlockIdx;
                    
                    // 获取当前块的形状
                    getBlockShape(actualShape, blockStartOffset, l1NResDynamic, kvL1Len, nowLen, curBlockSize);
                    // 获取KV在全局内存中的偏移
                    getKVOffset(gBlockTable, gBOffset, nowNIdx, blockStartOffset, strideKV, blockSize);
                    
                    // 获取B矩阵的tile布局和L1中的目标位置
                    auto layoutBTile = layoutB.GetTileLayout(MakeCoord(embed, nowLen));
                    MatrixCoord l1BTileCoord{0, kvL1Len};
                    auto l1BTile = l1BTensor[l1KvPingPongFlag][layoutBInL1.GetOffset(l1BTileCoord)];
                    
                    // 从全局内存复制B矩阵到L1缓存
                    copyGmToL1B(l1BTile, gB[gBOffset], layoutBInL1, layoutBTile);
                    
                    // 更新已处理长度和块偏移
                    kvL1Len += nowLen;
                    updateBlockOffset(nowLen, curBlockIdx, blockSize);
                }
                
                // 标记L1 KV乒乓缓冲区数据已准备好
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1KvPingPongFlag);
                
                // 更新实际形状参数
                mActual = actualShape.m();
                kActual = actualShape.k();
                nActual = l1NResDynamic;
            } else {  // 普通模式处理逻辑
                // 获取当前块的形状
                getBlockShape(actualShape, nL1Idx, nL1Loop, stackSeqTile);
                // 获取KV在全局内存中的偏移
                getKVOffset(gBOffset, nIdx, nL1Idx, strideKV);
                
                // 更新实际形状参数
                mActual = actualShape.m();
                kActual = actualShape.k();
                nActual = actualShape.n();
                layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(kActual, nActual);

                // 获取B矩阵的tile布局
                auto layoutBTile = layoutB.GetTileLayout(MakeCoord(kActual, nActual));
                
                // 等待L1 KV乒乓缓冲区就绪，然后从全局内存复制B矩阵到L1缓存
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1KvPingPongFlag);
                copyGmToL1B(l1BTensor[l1KvPingPongFlag], gB[gBOffset], layoutBInL1, layoutBTile);
                // 标记L1 KV乒乓缓冲区数据已准备好
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1KvPingPongFlag);
            }
            // M维度L0循环：处理token数量方向的分块
            uint32_t mL0Loop = CeilDiv(mActual, L0TileShape::M);
            // K维度L0循环：处理嵌入维度方向的分块
            uint32_t kL0Loop = CeilDiv(kActual, L0TileShape::K);
            
            // M维度L0循环
            for (uint32_t mL0Idx = 0; mL0Idx < mL0Loop; mL0Idx++) {
                // 计算当前M维度分块的实际大小（最后一块可能较小）
                uint32_t mL0Actual = (mL0Idx < mL0Loop - 1U) ? L0TileShape::M : (mActual - mL0Idx * L0TileShape::M);
                
                // 等待L0 C乒乓缓冲区就绪
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CPingPongFlag);
                
                // K维度L0循环
                for (uint32_t kL0Idx = 0; kL0Idx < kL0Loop; kL0Idx++) {
                    // 计算当前K维度分块的实际大小（最后一块可能较小）
                    uint32_t kL0Actual = (kL0Idx < kL0Loop - 1U) ? L0TileShape::K : (kActual - kL0Idx * L0TileShape::K);

                    // 从L1加载A矩阵到L0缓存
                    LayoutAInL0 layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(mL0Actual, kL0Actual);
                    MatrixCoord l1ATileCoord{mL0Idx * L0TileShape::M, kL0Idx * L0TileShape::K};
                    auto l1ATile = l1ATensor[layoutAInL1.GetOffset(l1ATileCoord)];
                    
                    // 等待L0 AB乒乓缓冲区就绪，然后复制数据
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag);
                    copyL1ToL0A(l0ATensor[l0ABPingPongFlag], l1ATile, layoutAInL0, layoutAInL1);

                    // 从L1加载B矩阵到L0缓存
                    LayoutBInL0 layoutBInL0 = LayoutBInL0::template MakeLayout<ElementB>(kL0Actual, nActual);
                    MatrixCoord l1BTileCoord{kL0Idx * L0TileShape::K, 0};
                    auto l1BTile = l1BTensor[l1KvPingPongFlag][layoutBInL1.GetOffset(l1BTileCoord)];
                    
                    // 第一次迭代时等待L1 KV数据就绪
                    if ((mL0Idx == 0U) && (kL0Idx == 0U)) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1KvPingPongFlag);
                    }
                    
                    // 等待L0 B乒乓缓冲区就绪，然后复制数据
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag + 2U);
                    copyL1ToL0B(l0BTensor[l0ABPingPongFlag], l1BTile, layoutBInL0, layoutBInL1);
                    
                    // 最后一次迭代时标记L1 KV数据已使用
                    if ((mL0Idx == mL0Loop - 1U) && (kL0Idx == kL0Loop - 1U)) {
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1KvPingPongFlag);
                    }

                    // 同步事件，确保数据复制完成
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
                    
                    // 执行矩阵乘法：只有第一次K维度迭代时初始化累加器
                    bool initMmad = (kL0Idx == 0U);
                    // M维度向上对齐到块大小
                    uint32_t mL0Align = (mL0Actual + BLOCK_SIZE - 1U) / BLOCK_SIZE * BLOCK_SIZE;
                    tileMmad(l0CTensor[l0CPingPongFlag],
                        l0ATensor[l0ABPingPongFlag],
                        l0BTensor[l0ABPingPongFlag],
                        mL0Align,
                        nActual,
                        kL0Actual,
                        initMmad);
                    
                    // 标记L0 AB乒乓缓冲区已使用
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag);
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag + 2U);
                    // 切换L0 AB乒乓缓冲区标志
                    l0ABPingPongFlag = 1U - l0ABPingPongFlag;
                }
                // 同步事件，确保矩阵乘法完成
                AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
                
                // 将结果从L0缓存复制到全局内存
                MatrixCoord gmCTileCoord{mL0Idx * L0TileShape::M, nL1Idx * l1NDynamic};
                LayoutC layoutCTile = layoutC.GetTileLayout(MakeCoord(mL0Actual, nActual));
                auto layoutInL0C = LayoutCInL0::MakeLayoutInL0C(MakeCoord(mL0Actual, nActual));
                copyL0CToGm(gC[layoutC.GetOffset(gmCTileCoord)], l0CTensor[l0CPingPongFlag], layoutCTile, layoutInL0C);
                
                // 标记L0 C乒乓缓冲区已使用
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CPingPongFlag);
                // 切换L0 C乒乓缓冲区标志
                l0CPingPongFlag = 1U - l0CPingPongFlag;
            }
            // 切换L1 KV乒乓缓冲区标志
            l1KvPingPongFlag = 1U - l1KvPingPongFlag;
        }
    }
protected:
    /// Data members
    AscendC::LocalTensor<ElementA> l1ATensor;         ///< L1缓存中的A矩阵(Q)张量
    AscendC::LocalTensor<ElementB> l1BTensor[STAGES];  ///< L1缓存中的B矩阵(K)张量数组（支持双缓冲）
    AscendC::LocalTensor<ElementA> l0ATensor[STAGES];  ///< L0缓存中的A矩阵(Q)张量数组（支持双缓冲）
    AscendC::LocalTensor<ElementB> l0BTensor[STAGES];  ///< L0缓存中的B矩阵(K)张量数组（支持双缓冲）
    AscendC::LocalTensor<ElementAccumulator> l0CTensor[STAGES];  ///< L0缓存中的C矩阵(输出)张量数组（支持双缓冲）

    TileMmad tileMmad;            ///< 分块矩阵乘法操作对象
    CopyGmToL1A copyGmToL1A;      ///< GM到L1的A矩阵复制操作对象
    CopyGmToL1B copyGmToL1B;      ///< GM到L1的B矩阵复制操作对象
    CopyL1ToL0A copyL1ToL0A;      ///< L1到L0的A矩阵复制操作对象
    CopyL1ToL0B copyL1ToL0B;      ///< L1到L0的B矩阵复制操作对象
    CopyL0CToGm copyL0CToGm;      ///< L0到GM的C矩阵复制操作对象

    uint32_t l1KvPingPongFlag = 0;  ///< L1 KV双缓冲标志（0或1）
    uint32_t l0CPingPongFlag = 0;   ///< L0 C双缓冲标志（0或1）
    uint32_t l0ABPingPongFlag = 0;  ///< L0 AB双缓冲标志（0或1）

    uint32_t l1MDynamic = 0;  ///< L1中M维度的动态大小
    uint32_t l1NDynamic = 0;  ///< L1中N维度的动态大小
    uint32_t l1KDynamic = 0;  ///< L1中K维度的动态大小

    uint32_t blockStartOffset = 0;  ///< 块起始偏移（用于分页缓存模式）
    uint32_t maxKVStackLen = 0;     ///< KV堆叠的最大长度
};

}

#endif