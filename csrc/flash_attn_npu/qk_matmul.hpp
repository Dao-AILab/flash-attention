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

/**
 * @file qk_matmul.hpp
 * @brief Atlas A2平台Flash Attention推理中Q-K矩阵乘法的块级实现
 *
 * 本文件实现了Flash Attention推理计算中第一个矩阵乘法步骤：Q * K^T（查询与键的乘法），
 * 生成注意力分数矩阵S。这是Flash Attention算法的核心计算之一。
 *
 * == 主要实现的算法 ==
 * 实现了分块矩阵乘法（Tiled GEMM），将大规模的Q*K^T计算分解为适配昇腾NPU多级缓存
 * 层次结构的小块计算。核心算法流程：
 *   1. 将Q矩阵一次性加载到L1缓存（loadQGM）
 *   2. 将K矩阵按KV序列分块加载到L1缓存（支持普通模式和分页缓存模式）
 *   3. 在L0缓存中执行分块矩阵乘法（tileMmad）
 *   4. 将结果C（注意力分数S）从L0缓存写回全局内存
 *
 * == 分页缓存（Paged Cache）支持 ==
 * 当PAGED_CACHE_FLAG_为true时，K矩阵的加载使用分页缓存机制：
 * - KV缓存按固定大小的块（block）组织，通过块表（block table）索引
 * - 支持非连续的KV缓存布局，适配vLLM等推理框架的PagedAttention机制
 * - 块内偏移（blockStartOffset）跟踪当前处理位置
 *
 * == 多级缓存分块策略 ==
 * - L1分块: 将K矩阵的N维度（序列长度方向）分为多个L1块
 * - L0分块: 将M维度（token方向）和K维度（嵌入维度方向）进一步细分
 * - 乒乓缓冲: L1/L0缓存均使用双缓冲技术，实现计算与数据搬移的流水线重叠
 *
 * == 依赖关系 ==
 * - catlass/catlass.hpp: Catlass库核心头文件
 * - catlass/arch/resource.hpp: 硬件资源管理
 * - catlass/gemm/dispatch_policy.hpp: GEMM调度策略
 * - catlass/gemm/helper.hpp: GEMM辅助工具（对齐、累加器类型选择）
 * - catlass/gemm/tile/tile_copy.hpp: 分块数据拷贝操作
 * - catlass/gemm/tile/tile_mmad.hpp: 分块矩阵乘法操作
 * - fa_block.h: Flash Attention分块参数定义
 *
 * == 使用场景 ==
 * 本文件用于Flash Attention推理场景中的QK矩阵乘法步骤。
 * 典型调用路径: mha_fwd_kvcache.cpp -> FAInferKernel -> BlockMmadQK -> 本文件
 * 与pv_matmul.hpp（PV矩阵乘法）配合，构成Flash Attention的两个GEMM步骤。
 */

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "fa_block.h"

/**
 * @namespace Catlass::Gemm::Block
 * @brief Catlass库中GEMM（通用矩阵乘法）的块级实现命名空间
 *
 * 该命名空间包含Flash Attention计算流水线中两个矩阵乘法步骤的块级实现：
 * - QK矩阵乘法（qk_matmul.hpp）: 计算 Q * K^T 得到注意力分数S
 * - PV矩阵乘法（pv_matmul.hpp）: 计算 P * V 得到注意力输出O
 *
 * "块级"（Block）意味着这些实现将大规模矩阵乘法分解为适配硬件缓存层次的小块，
 * 通过精心设计的多级循环、乒乓缓冲和事件同步实现高效的流水线执行。
 */
namespace Catlass::Gemm::Block {

/**
 * @brief Q-K矩阵乘法的块级矩阵乘法实现（BlockMmadQK）
 *
 * 为Atlas A2架构实现的Q-K矩阵乘法块级操作，用于Flash Attention推理计算中的Q*K^T步骤。
 * 计算查询矩阵Q与键矩阵K的转置的乘积，生成注意力分数矩阵S。
 *
 * == 设计思路 ==
 * 在Flash Attention推理中，Q*K^T的计算特点：
 * - Q矩阵较小（新产生的token数量 × 嵌入维度），可以一次性加载到L1缓存
 * - K矩阵较大（KV缓存序列长度 × 嵌入维度），需要按序列长度方向分块加载
 * - 输出S矩阵（token数量 × 序列长度），需要逐块写回全局内存供Softmax使用
 *
 * == 多级缓存分块策略 ==
 * - L1级: 将K矩阵的N维度（序列长度方向）分为多个L1块
 *   - 每个L1块包含kL1Size个token的K向量
 * - L0级: 将M维度（token方向）和K维度（嵌入维度方向）进一步细分
 *   - 每个L0块包含mL0Size个token × kL0Size个嵌入维度
 *
 * == 乒乓缓冲策略 ==
 * - L1 KV乒乓: 使用两个L1缓冲区交替加载K矩阵数据
 *   - 当一个缓冲区在进行L0计算时，另一个缓冲区在从GM加载新数据
 *   - 通过硬件事件（HardEvent）同步两个缓冲区的使用
 * - L0 AB乒乓: 使用两套L0缓冲区交替加载A和B矩阵数据
 *
 * == 分页缓存支持 ==
 * 当PAGED_CACHE_FLAG_为true时，K矩阵的加载使用分页缓存机制：
 * - KV缓存按固定大小的块（block）组织，通过blockTable索引
 * - blockTable[blockIdx]指向物理块号，支持非连续内存布局
 * - blockStartOffset跟踪当前处理位置在块内的偏移
 *
 * @tparam PAGED_CACHE_FLAG_ 是否启用分页缓存模式
 *         - true: 使用blockTable索引KV缓存，支持PagedAttention
 *         - false: 使用连续内存布局，直接偏移访问KV缓存
 * @tparam ENABLE_UNIT_FLAG_ 是否启用单元测试模式
 *         - true: 启用单元测试标志，用于调试和验证
 *         - false: 正常运行模式
 * @tparam L1TileShape_ L1缓存的分块形状，定义L1级分块的M/K/N维度大小
 * @tparam L0TileShape_ L0缓存的分块形状，定义L0级分块的M/K/N维度大小
 * @tparam AType_ A矩阵（Q矩阵）的类型和布局，包含元素类型和矩阵布局信息
 * @tparam BType_ B矩阵（K矩阵）的类型和布局，包含元素类型和矩阵布局信息
 * @tparam CType_ C矩阵（S矩阵）的类型和布局，包含元素类型和矩阵布局信息
 * @tparam BiasType_ 偏置类型，Flash Attention中通常不使用偏置
 * @tparam TileCopy_ 分块数据拷贝策略，定义GM<->L1<->L0的数据搬移方式
 * @tparam TileMmad_ 分块矩阵乘法策略，定义L0级矩阵乘法的执行方式
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
     * @brief 从全局内存加载Q矩阵到L1缓存
     *
     * 将查询矩阵Q从全局内存（GM）一次性加载到L1缓存中。Q矩阵在Flash Attention推理中
     * 通常较小（新产生的token数量 × 嵌入维度），可以完整放入L1缓存。
     *
     * == 数据搬移量 ==
     * 搬移数据量 = rowNumRound * qHeads * embed * sizeof(ElementA)
     * 其中 rowNumRound 是向上对齐到L1AAlignHelper::M_ALIGNED的行数
     *
     * == 加载策略 ==
     * Q矩阵按分组注意力头（Grouped Query Attention）的方式组织：
     * - 每组包含 singleGroupHeads 个注意力头
     * - 每组有 tokenNumPerGroup = rowNum / singleGroupHeads 个token
     * - 数据按 [tokenNumPerGroup, qHeads * embed] 的布局复制到L1
     *
     * @param gA               源全局内存张量，存储查询矩阵Q
     * @param layoutA          Q矩阵的布局信息，包含形状和步长
     * @param rowNum           行数（token数量），即Q矩阵的M维度
     * @param singleGroupHeads 单组注意力头数，用于GQA分组
     * @param qHeads           查询头总数，即Q矩阵的注意力头数量
     *
     * @note 数据搬移后通过SetFlag/WaitFlag确保DMA传输完成
     * @note 算法复杂度: O(rowNum * qHeads * embed) 数据搬移
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
     * 实现Flash Attention推理中Q*K^T矩阵乘法的核心逻辑。采用三级循环结构
     * （N维度L1循环 → M维度L0循环 → K维度L0循环）和双缓冲机制实现高效的流水线执行。
     *
     * == 算法流程 ==
     * 1. 加载Q矩阵到L1缓存（loadQGM，在外部调用）
     * 2. N维度L1循环：按序列长度方向分块加载K矩阵
     *    a) 计算当前L1块的实际形状（处理最后一个不完整的块）
     *    b) 从GM加载K矩阵的一个L1块到L1缓存（乒乓缓冲）
     *    c) M维度L0循环：按token方向分块加载Q矩阵
     *       i) 从L1加载Q矩阵的一个L0块到L0A缓存
     *       ii) K维度L0循环：按嵌入维度方向分块
     *           - 从L1加载K矩阵的一个L0块到L0B缓存
     *           - 执行L0级矩阵乘法（tileMmad）
     *       iii) 将L0C结果写回全局内存
     *    d) 更新K矩阵的全局内存偏移
     *
     * == 乒乓缓冲同步机制 ==
     * - l1KvPingPongFlag: L1 KV乒乓缓冲区同步标志
     *   - WaitFlag: 等待缓冲区空闲（可以写入新数据）
     *   - SetFlag: 标记缓冲区已占用（数据已写入，可以开始计算）
     * - l0AbPingPongFlag: L0 AB乒乓缓冲区同步标志
     *   - 用于L0A和L0B缓冲区的读写同步
     *
     * == 分页缓存模式 ==
     * 当PAGED_CACHE_FLAG_为true时，K矩阵的加载使用分页缓存机制：
     * - 通过blockTable索引物理块号，支持非连续内存布局
     * - 块内偏移（blockStartOffset）跟踪当前处理位置
     * - 当一个块处理完成后，自动切换到下一个物理块
     *
     * @param gA             全局内存中的A矩阵（Q矩阵）张量
     * @param gB             全局内存中的B矩阵（K矩阵）张量
     * @param gC             全局内存中的C矩阵（S矩阵/输出）张量
     * @param gBlockTable    块表张量（仅分页缓存模式使用），映射逻辑块到物理块
     * @param layoutA        A矩阵的布局信息
     * @param layoutB        B矩阵的布局信息
     * @param layoutC        C矩阵的布局信息（必须为RowMajor）
     * @param actualOriShape 实际原始形状（M/N/K维度大小）
     * @param nIdx           当前N维度索引（在多核并行中的位置）
     * @param nLoop          N维度总循环次数（多核并行时的总块数）
     * @param blockSize      KV缓存的块大小（仅分页缓存模式使用）
     * @param strideKV       KV缓存的步长（相邻token之间的地址偏移）
     *
     * @note 算法复杂度: O(M * N * K) 矩阵乘法，通过分块和乒乓缓冲优化实际执行效率
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
        // 性能优化: L1级循环是外层循环，每次迭代加载K矩阵的一个L1块
        // 通过乒乓缓冲，当前L1块的计算与下一个L1块的加载可以并行
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
            // 性能优化: M维度循环是中间层循环，每次迭代加载Q矩阵的一个L0块
            uint32_t mL0Loop = CeilDiv(mActual, L0TileShape::M);
            // K维度L0循环：处理嵌入维度方向的分块
            // 性能优化: K维度循环是最内层循环，累加多个K分块的结果
            uint32_t kL0Loop = CeilDiv(kActual, L0TileShape::K);
            
            // M维度L0循环
            for (uint32_t mL0Idx = 0; mL0Idx < mL0Loop; mL0Idx++) {
                // 计算当前M维度分块的实际大小（最后一块可能较小）
                uint32_t mL0Actual = (mL0Idx < mL0Loop - 1U) ? L0TileShape::M : (mActual - mL0Idx * L0TileShape::M);
                
                // 等待L0 C乒乓缓冲区就绪（确保上一次L0C写回GM完成）
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CPingPongFlag);
                
                // K维度L0循环
                for (uint32_t kL0Idx = 0; kL0Idx < kL0Loop; kL0Idx++) {
                    // 计算当前K维度分块的实际大小（最后一块可能较小）
                    uint32_t kL0Actual = (kL0Idx < kL0Loop - 1U) ? L0TileShape::K : (kActual - kL0Idx * L0TileShape::K);

                    // 从L1加载A矩阵(Q)到L0缓存
                    LayoutAInL0 layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(mL0Actual, kL0Actual);
                    MatrixCoord l1ATileCoord{mL0Idx * L0TileShape::M, kL0Idx * L0TileShape::K};
                    auto l1ATile = l1ATensor[layoutAInL1.GetOffset(l1ATileCoord)];
                    
                    // 等待L0 AB乒乓缓冲区就绪，然后复制Q数据到L0A
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag);
                    copyL1ToL0A(l0ATensor[l0ABPingPongFlag], l1ATile, layoutAInL0, layoutAInL1);

                    // 从L1加载B矩阵(K)到L0缓存
                    LayoutBInL0 layoutBInL0 = LayoutBInL0::template MakeLayout<ElementB>(kL0Actual, nActual);
                    MatrixCoord l1BTileCoord{kL0Idx * L0TileShape::K, 0};
                    auto l1BTile = l1BTensor[l1KvPingPongFlag][layoutBInL1.GetOffset(l1BTileCoord)];
                    
                    // 第一次迭代时等待L1 KV数据就绪
                    // 性能优化: 只在第一次迭代等待，后续迭代L1数据已在后台加载
                    if ((mL0Idx == 0U) && (kL0Idx == 0U)) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1KvPingPongFlag);
                    }
                    
                    // 等待L0 B乒乓缓冲区就绪，然后复制K数据到L0B
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag + 2U);
                    copyL1ToL0B(l0BTensor[l0ABPingPongFlag], l1BTile, layoutBInL0, layoutBInL1);
                    
                    // 最后一次迭代时标记L1 KV数据已使用，可以开始加载下一个L1块
                    // 性能优化: 提前释放L1缓冲区，允许后台加载下一个L1块
                    if ((mL0Idx == mL0Loop - 1U) && (kL0Idx == kL0Loop - 1U)) {
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1KvPingPongFlag);
                    }

                    // 同步事件，确保L1->L0数据复制完成
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
                    
                    // 执行矩阵乘法：只有第一次K维度迭代时初始化累加器（清零L0C）
                    // 后续K维度迭代累加到L0C中
                    bool initMmad = (kL0Idx == 0U);
                    // M维度向上对齐到BLOCK_SIZE(16)，满足硬件MAD指令的对齐要求
                    uint32_t mL0Align = (mL0Actual + BLOCK_SIZE - 1U) / BLOCK_SIZE * BLOCK_SIZE;
                    tileMmad(l0CTensor[l0CPingPongFlag],
                        l0ATensor[l0ABPingPongFlag],
                        l0BTensor[l0ABPingPongFlag],
                        mL0Align,
                        nActual,
                        kL0Actual,
                        initMmad);
                    
                    // 标记L0 AB乒乓缓冲区已使用，可以开始加载下一个L0块
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag);
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag + 2U);
                    // 切换L0 AB乒乓缓冲区标志（0<->1交替）
                    l0ABPingPongFlag = 1U - l0ABPingPongFlag;
                }
                // 同步事件，确保矩阵乘法完成（L0C数据就绪）
                AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
                
                // 将结果从L0缓存复制到全局内存
                MatrixCoord gmCTileCoord{mL0Idx * L0TileShape::M, nL1Idx * l1NDynamic};
                LayoutC layoutCTile = layoutC.GetTileLayout(MakeCoord(mL0Actual, nActual));
                auto layoutInL0C = LayoutCInL0::MakeLayoutInL0C(MakeCoord(mL0Actual, nActual));
                copyL0CToGm(gC[layoutC.GetOffset(gmCTileCoord)], l0CTensor[l0CPingPongFlag], layoutCTile, layoutInL0C);
                
                // 标记L0 C乒乓缓冲区已使用，可以开始写回下一个L0C
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CPingPongFlag);
                // 切换L0 C乒乓缓冲区标志（0<->1交替）
                l0CPingPongFlag = 1U - l0CPingPongFlag;
            }
            // 切换L1 KV乒乓缓冲区标志（0<->1交替）
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