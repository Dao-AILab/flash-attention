// Copyright (c) 2023, flash-attention
// All rights reserved.
//
// 该文件实现了FlashAttention在Atlas A2架构上的推理内核
// 主要包含FAInferKernel类，用于执行带KV缓存的多头注意力计算
// 支持分页KV缓存、因果掩码和不同的输入布局

#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/debug.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "online_softmax_low_prec.hpp"  // 低精度在线Softmax实现
#include "online_softmax.hpp"            // 在线Softmax实现
#include "rescale_o_low_prec.hpp"        // 低精度输出重缩放实现
#include "rescale_o.hpp"                  // 输出重缩放实现
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "pv_matmul.hpp"                  // P-V矩阵乘法实现
#include "qk_matmul.hpp"                  // Q-K矩阵乘法实现
#include "catlass/gemm/dispatch_policy.hpp"
#include "fa_block.h"                     // FlashAttention块定义
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"

#include "kernel_common.hpp"              // 内核通用定义
#include "kernel_operator.h"
#include "tilingdata.h"                   // 分块数据结构

using namespace Catlass;
using namespace KernelCommon;

namespace SplitFuse {
    // FlashAttention推理内核类
    // 模板参数说明：
    // - BlockMmadQK: Q-K矩阵乘法的块实现
    // - BlockMmadPV: P-V矩阵乘法的块实现
    // - EpilogueOnlineSoftmax: 在线Softmax的Epilogue实现
    // - EpilogueRescaleO: 输出重缩放的Epilogue实现
    // - PAGED_CACHE_FLAG: 是否使用分页KV缓存
    // - MASK_TYPE: 掩码类型（无掩码/因果掩码/特殊掩码）
    // - INPUT_LAYOUT: 输入张量的布局
    template <
        class BlockMmadQK,                  // Q-K矩阵乘法块模板
        class BlockMmadPV,                  // P-V矩阵乘法块模板
        class EpilogueOnlineSoftmax,        // 在线Softmax Epilogue
        class EpilogueRescaleO,             // 输出重缩放 Epilogue
        bool PAGED_CACHE_FLAG,              // 分页缓存标志
        FaiKenel::MaskType MASK_TYPE = FaiKenel::MaskType::NO_MASK,  // 掩码类型
        FaiKenel::inputLayout INPUT_LAYOUT = FaiKenel::inputLayout::BSND>  // 输入布局
    class FAInferKernel {
    public:
        // 架构标签：指定目标硬件架构（如Atlas A2）
        using ArchTag = typename BlockMmadQK::ArchTag;
        // L1缓存分块形状：定义L1缓存中的数据分块大小
        using L1TileShape = typename BlockMmadQK::L1TileShape;
        
        // Q-K矩阵乘法相关类型
        using ElementQ = typename BlockMmadQK::ElementA;  // 查询张量Q的元素类型
        using LayoutQ = typename BlockMmadQK::LayoutA;    // 查询张量Q的布局
        using ElementK = typename BlockMmadQK::ElementB;  // 键张量K的元素类型
        using LayoutK = typename BlockMmadQK::LayoutB;    // 键张量K的布局
        using ElementS = typename BlockMmadQK::ElementC;  // QK^T结果S的元素类型
        using LayoutS = typename BlockMmadQK::LayoutC;    // QK^T结果S的布局

        // P-V矩阵乘法相关类型
        using ElementP = typename BlockMmadPV::ElementA;  // 注意力权重P的元素类型
        using LayoutP = typename BlockMmadPV::LayoutA;    // 注意力权重P的布局
        using ElementV = typename BlockMmadPV::ElementB;  // 值张量V的元素类型
        using LayoutV = typename BlockMmadPV::LayoutB;    // 值张量V的布局

        // 掩码相关类型
        using ElementMask = typename EpilogueOnlineSoftmax::ElementMask;  // 掩码元素类型
        using LayoutMask = typename EpilogueOnlineSoftmax::LayoutMask;    // 掩码布局

        // 输出相关类型
        using ElementO = typename EpilogueRescaleO::ElementOutput;   // 最终输出O的元素类型
        using LayoutO = typename EpilogueRescaleO::LayoutOutput;     // 最终输出O的布局

        // 临时输出相关类型
        using ElementOTmp = typename EpilogueRescaleO::ElementInput;  // 临时输出的元素类型
        using LayoutOTmp = typename EpilogueRescaleO::LayoutInput;    // 临时输出的布局

        // 对数和相关类型
        using ElementLse = typename EpilogueRescaleO::ElementLse;  // 对数和的元素类型
        using LayoutLse = typename EpilogueRescaleO::LayoutLse;    // 对数和的布局

        // 更新相关类型
        using ElementUpdate = typename EpilogueRescaleO::ElementUpdate;  // 更新的元素类型
        using LayoutUpdate = typename EpilogueRescaleO::LayoutUpdate;    // 更新的布局

        // LSE模式：指定对数和的计算方式
        static constexpr Epilogue::LseModeT LSE_MODE = EpilogueRescaleO::LSE_MODE;

        __aicore__ inline
        FAInferKernel() {}

        // 内核执行运算符：FlashAttention推理的核心计算逻辑
        // @param params 内核参数结构体，包含所有输入输出张量地址和配置信息
        __aicore__ inline
        void operator()(FAIKernelParams const &params)
        {
            // 获取分块数据结构指针
            __gm__ FAInferTilingData *fATilingData = reinterpret_cast<__gm__ FAInferTilingData *>(params.tiling);
            
            // 工作区各部分大小
            uint64_t mm1OutSize = fATilingData->mm1OutSize;      // QK矩阵乘法输出大小
            uint64_t smOnlineOutSize = fATilingData->smOnlineOutSize;  // 在线Softmax输出大小
            uint64_t mm2OutSize = fATilingData->mm2OutSize;      // PV矩阵乘法输出大小
            
            // 模型参数
            uint32_t batch = fATilingData->batch;                // 批次大小
            uint32_t qHeads = fATilingData->numHeads;            // 查询头数量
            uint32_t kvHeads = fATilingData->kvHeads;            // KV头数量
            uint32_t embed = fATilingData->embeddingSize;        // 查询嵌入维度
            uint32_t embedV = fATilingData->embeddingSizeV;      // 值嵌入维度
            uint32_t pagedBlockSize = fATilingData->blockSize;   // 分页块大小
            uint32_t maxNumBlocksPerBatch = fATilingData->maxNumBlocksPerBatch;  // 每批次最大块数
            uint32_t firstBatchTaskNum = fATilingData->firstBatchTaskNum;  // 第一批次任务数量
            uint32_t totalTaskNum = fATilingData->totalTaskNum;  // 总任务数量
            uint32_t blockSize = fATilingData->blockSize;        // 块大小
            uint32_t maskType = fATilingData->maskType;          // 掩码类型
            float scaleValue = fATilingData->scaleValue;         // Softmax缩放因子
            
            // 初始化全局张量
            AscendC::GlobalTensor<ElementQ> gQ;  // 查询张量Q
            gQ.SetGlobalBuffer((__gm__ ElementQ *)params.q);
            AscendC::GlobalTensor<ElementK> gK;  // 键张量K
            gK.SetGlobalBuffer((__gm__ ElementK *)params.k);
            AscendC::GlobalTensor<ElementK> gV;  // 值张量V
            gV.SetGlobalBuffer((__gm__ ElementK *)params.v);
            AscendC::GlobalTensor<ElementMask> gMask;  // 掩码张量
            gMask.SetGlobalBuffer((__gm__ ElementMask *)params.mask);
            AscendC::GlobalTensor<int32_t> gBlockTable;  // 块表（用于分页KV缓存）
            gBlockTable.SetGlobalBuffer((__gm__ int32_t *)(params.blockTables));
            AscendC::GlobalTensor<int64_t> gActualQseqlen;  // 实际查询序列长度
            gActualQseqlen.SetGlobalBuffer((__gm__ int64_t *)params.actualQseqlen);
            AscendC::GlobalTensor<int64_t> gActualKvseqlen;  // 实际KV序列长度
            gActualKvseqlen.SetGlobalBuffer((__gm__ int64_t *)params.actualKvseqlen);
            AscendC::GlobalTensor<ElementO> gO;  // 输出张量O
            gO.SetGlobalBuffer((__gm__ ElementO *)params.o);
            AscendC::GlobalTensor<ElementLse> gLse;  // 对数和张量（用于数值稳定性）
            gLse.SetGlobalBuffer((__gm__ ElementLse *)params.lse);
            
            // 初始化工作区张量
            AscendC::GlobalTensor<ElementS> gS;  // QK^T结果S
            gS.SetGlobalBuffer((__gm__ ElementS *)(params.workSpace));
            AscendC::GlobalTensor<ElementP> gP;  // 注意力权重P（Softmax结果）
            gP.SetGlobalBuffer((__gm__ ElementP *)(params.workSpace + mm1OutSize));
            AscendC::GlobalTensor<ElementOTmp> gOTmp;  // 临时输出张量
            gOTmp.SetGlobalBuffer((__gm__ ElementOTmp *)(params.workSpace + mm1OutSize + smOnlineOutSize));
            AscendC::GlobalTensor<ElementOTmp> gOUpdate;  // 更新输出张量
            gOUpdate.SetGlobalBuffer((__gm__ ElementOTmp *)(params.workSpace +
                mm1OutSize + smOnlineOutSize + mm2OutSize));

            // 获取核心索引和核心数量
            uint32_t coreIdx = AscendC::GetBlockIdx();  // 当前核心索引
            uint32_t coreNum = AscendC::GetBlockNum();  // 总核心数量
            
            // 设置硬件事件标志（仅在C220架构上有效）
#ifdef __DAV_C220_CUBE__
            // 设置MTE1事件标志，用于指令同步
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID4);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID5);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID6);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID7);
            
            // 设置固定M维度标志
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
            
            // 设置MTE1和MTE2协同工作的事件标志
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID6);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
            
            // 动态计算K维度的分块大小
            uint32_t kDynNum = RoundUp(embed, NUM_128);  // K维度向上对齐到128
            kDynNum = kDynNum < NUM_256 ? NUM_256 : kDynNum;  // 确保至少为256
            
            // 计算L1缓存中可用于QK^T计算的最大空间
            uint32_t maxQKPL1Size = L1_MAX_SIZE - embedV * MAX_KV_STACK_LEN * sizeof(ElementV);
            uint32_t maxQL1Size = Q_TILE_CEIL * kDynNum * sizeof(ElementQ);  // Q矩阵L1缓存大小
            // 计算N维度的最大动态分块大小（考虑双缓冲）
            uint32_t maxNDynNum = ((maxQKPL1Size - maxQL1Size) / kDynNum / sizeof(ElementV) / DOUBLE_BUFFER) / NUM_32 * NUM_32;

            // 确定实际使用的N维度动态分块大小
            uint32_t nDynNum = maxNDynNum < L1_MAX_N_NUM ? maxNDynNum : L1_MAX_N_NUM;
            // 确保分块大小是L1_MAX_N_NUM的约数
            nDynNum = L1_MAX_N_NUM % nDynNum != 0 ? RoundDown((nDynNum - 1), NUM_32) : nDynNum;

            // 计算QK矩阵乘法的L1缓存占用
            uint32_t L1_QK_SIZE = BlockMmadQK::L1TileShape::M * kDynNum * sizeof(ElementQ);
            // 初始化QK矩阵乘法块
            BlockMmadQK blockMmadQK(resource, nDynNum, kDynNum, MAX_KV_STACK_LEN);
            // 计算PV矩阵乘法的K维度分块大小
            uint32_t kPVDynNum = nDynNum * kDynNum / BlockMmadPV::L1TileShape::M;
            // 初始化PV矩阵乘法块
            BlockMmadPV blockMmadPV(resource, nDynNum, kPVDynNum, MAX_KV_STACK_LEN, L1_QK_SIZE);
#endif
#ifdef __DAV_C220_VEC__
            // 设置向量指令相关的硬件事件标志
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);

            // 初始化在线Softmax模块
            EpilogueOnlineSoftmax epilogueOnlineSoftmax(resource, scaleValue);
            // 初始化输出重缩放模块
            EpilogueRescaleO epilogueRescaleO(resource);

            // 计算核心索引（考虑子块）
            coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
#endif
            // 计算各张量的步长
            uint64_t strideQ = static_cast<uint64_t>(qHeads * embed);  // Q张量步长
            uint64_t strideO = static_cast<uint64_t>(qHeads * embedV);  // O张量步长
            uint64_t strideK = static_cast<uint64_t>(kvHeads * embed);  // K张量步长
            uint64_t strideV = static_cast<uint64_t>(kvHeads * embedV);  // V张量步长
            
            // 嵌入维度向上对齐到块大小
            uint32_t embedRound = RoundUp(embed, FaiKenel::BLOCK_SIZE);
            uint32_t embedRoundV = RoundUp(embedV, FaiKenel::BLOCK_SIZE);
            
            // 计算头分组大小（多头注意力中每组的头数）
            uint32_t groupSize = qHeads / kvHeads;

            // 初始化各张量的批次偏移量
            uint64_t qBOffset = 0;      // Q张量批次偏移
            uint64_t kBOffset = 0;      // K张量批次偏移
            uint64_t vBOffset = 0;      // V张量批次偏移
            uint64_t oBOffset = 0;      // O张量批次偏移
            uint64_t lseBOffset = 0;    // LSE张量批次偏移
            uint64_t blockBOffset = 0;  // 块表批次偏移

            // 初始化任务计数器和批次索引
            uint32_t preTotalTaskNum = 0;  // 之前批次的总任务数
            uint32_t curBatch = 0;         // 当前批次索引
            uint32_t totalQTokens = static_cast<uint32_t>(gActualQseqlen.GetValue(batch - 1));  // 总查询令牌数
            uint32_t qSeqlen = fATilingData->maxQSeqlen;  // 查询序列长度
            uint32_t kvSeqlen = static_cast<uint32_t>(gActualKvseqlen.GetValue(curBatch));  // KV序列长度
            
            // TND输入布局处理
            if constexpr(INPUT_LAYOUT == FaiKenel::inputLayout::TND) {
                // 计算前一批次的查询序列长度总和
                uint32_t prevQSeqlenSum = (curBatch == 0) ? 0 : fATilingData->maxQSeqlen;
                qSeqlen = fATilingData->maxQSeqlen;
                
                // 非分页缓存模式下，计算当前批次的KV序列长度
                if constexpr (!PAGED_CACHE_FLAG) {
                    uint32_t prevKvSeqlenSum = (curBatch == 0) ?
                        0 : static_cast<uint32_t>(gActualKvseqlen.GetValue(curBatch - 1));
                    kvSeqlen = kvSeqlen - prevKvSeqlenSum;
                }
            }
            
            // 计算查询头分块大小和每组的分块数量
            uint32_t curQNBlockTile = GetQNBlockTile(qSeqlen, groupSize);  // 查询头分块大小
            uint32_t qNBlockNumPerGroup = CeilDiv(groupSize, curQNBlockTile);  // 每组的查询头分块数量
            uint32_t curQNBlockNum = qNBlockNumPerGroup * kvHeads;  // 总的查询头分块数量
            uint32_t curQSBlockTile = GetQSBlockTile(kvSeqlen);  // 查询序列分块大小
            uint32_t curQSBlockNum = CeilDiv(qSeqlen, curQSBlockTile);  // 查询序列分块数量
            uint32_t curTotalTaskNum = firstBatchTaskNum;  // 当前总任务数
            
            // 遍历每个任务（按核心分配任务，每个核心处理间隔coreNum的任务）
            for (uint32_t taskIdx = coreIdx; taskIdx < totalTaskNum; taskIdx += uint32_t(coreNum)) {
                // 处理批次切换：如果当前任务索引超过当前批次的总任务数，则切换到下一批次
                while (taskIdx >= curTotalTaskNum) {
                    ++curBatch;  // 移动到下一批次
                    preTotalTaskNum = curTotalTaskNum;  // 记录之前批次的总任务数
                    
                    // 更新各张量的批次偏移量
                    qBOffset += qSeqlen * strideQ;  // Q张量批次偏移
                    if constexpr (!PAGED_CACHE_FLAG) {
                        // 非分页缓存模式：更新K和V张量的批次偏移
                        kBOffset += static_cast<uint64_t>(kvSeqlen * strideK);
                        vBOffset += static_cast<uint64_t>(kvSeqlen * strideV);
                    } else {
                        // 分页缓存模式：更新块表的批次偏移
                        blockBOffset += static_cast<uint64_t>(maxNumBlocksPerBatch);
                    }
                    oBOffset += static_cast<uint64_t>(qSeqlen * strideO);  // O张量批次偏移
                    lseBOffset += static_cast<uint64_t>(qSeqlen * qHeads);  // LSE张量批次偏移

                    // 获取当前批次的序列长度
                    qSeqlen = fATilingData->maxQSeqlen;
                    kvSeqlen = static_cast<uint32_t>(gActualKvseqlen.GetValue(curBatch));
                    
                    // TND输入布局处理
                    if constexpr(INPUT_LAYOUT == FaiKenel::inputLayout::TND) {
                        uint32_t prevQSeqlenSum = (curBatch == 0) ? 0 : fATilingData->maxQSeqlen;
                        qSeqlen = fATilingData->maxQSeqlen;
                        
                        // 非分页缓存模式下，计算当前批次的KV序列长度
                        if constexpr (!PAGED_CACHE_FLAG) {
                            uint32_t prevKvSeqlenSum = (curBatch == 0) ?
                                0 : static_cast<uint32_t>(gActualKvseqlen.GetValue(curBatch - 1));
                            kvSeqlen = kvSeqlen - prevKvSeqlenSum;
                        }
                    }
                    
                    // 重新计算当前批次的分块参数
                    curQNBlockTile = GetQNBlockTile(qSeqlen, groupSize);
                    qNBlockNumPerGroup = CeilDiv(groupSize, curQNBlockTile);
                    curQNBlockNum = qNBlockNumPerGroup * kvHeads;
                    curQSBlockTile = GetQSBlockTile(kvSeqlen);
                    curQSBlockNum = CeilDiv(qSeqlen, curQSBlockTile);
                    
                    // 更新当前总任务数
                    curTotalTaskNum += curQNBlockNum * curQSBlockNum;
                }
                
                // 计算当前批次内的任务索引
                uint32_t taskIdxCurBatch = taskIdx - preTotalTaskNum;
                
                // 计算当前任务对应的查询序列分块索引和查询头分块索引
                uint32_t qSBlockIdx = taskIdxCurBatch / curQNBlockNum;
                uint32_t qNBlockIdx = taskIdxCurBatch - qSBlockIdx * curQNBlockNum;
                uint32_t qNBlockIdxCurGroup = qNBlockIdx % qNBlockNumPerGroup;

                // 计算KV头索引和查询头起始索引
                uint32_t kvNIdx = qNBlockIdx / qNBlockNumPerGroup;
                uint32_t qNStartIdx = kvNIdx * groupSize + qNBlockIdxCurGroup * curQNBlockTile;
                
                // 计算LSE张量的令牌偏移
                uint32_t lseTokenOffset = qSBlockIdx * curQSBlockTile * qHeads;

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

                // 计算实际的分块大小（最后一个分块可能小于标准分块大小）
                uint32_t qSBlockSize = (qSBlockIdx == (curQSBlockNum - 1U)) ?
                    (qSeqlen - qSBlockIdx * curQSBlockTile) : curQSBlockTile;  // 查询序列分块大小
                uint32_t qNBlockSize = (qNBlockIdxCurGroup == (qNBlockNumPerGroup - 1U)) ?
                    (groupSize - qNBlockIdxCurGroup * curQNBlockTile) : curQNBlockTile;  // 查询头分块大小
                
                // 计算行数和向上对齐的行数
                uint32_t rowNum = qSBlockSize * qNBlockSize;  // 实际行数
                uint32_t rowNumRound = RoundUp(rowNum, FaiKenel::BLOCK_SIZE);  // 向上对齐到块大小的行数

                // 计算不跳过的KV序列长度（掩码相关处理）
                uint32_t noSkipKvS = kvSeqlen;
                if (maskType != 0U) {  // 存在掩码时的特殊处理
                    uint32_t diffS = kvSeqlen - qSeqlen;
                    noSkipKvS = (qSBlockIdx + 1U) * curQSBlockTile + diffS;
                    noSkipKvS = AscendC::Std::min((uint32_t)kvSeqlen, noSkipKvS);  // 确保不超过实际KV序列长度
                }
                
                // 计算KV序列的总循环次数
                uint32_t kvSLoopNumTotal = CeilDiv(noSkipKvS, MAX_KV_STACK_LEN);
 	 
                // 分页缓存相关计算
                uint32_t blockStackNum = (MAX_KV_STACK_LEN - 1 + pagedBlockSize) / pagedBlockSize;  // 每栈的块数量
                uint32_t stackSeqTile = MAX_KV_STACK_LEN;  // 栈序列分块大小
                uint32_t stackSeqTilePad = MAX_KV_STACK_LEN;  // 填充后的栈序列分块大小
                uint32_t preKVNum = PRE_LAUNCH;  // 预启动数量
                int32_t stackSeqCount = 0;  // 栈序列计数器

#ifdef __DAV_C220_CUBE__
                // CUBE架构下的布局初始化
                LayoutQ layoutQTemp(rowNum, embed);  // Q矩阵临时布局
                LayoutK layoutKTemp(strideK, stackSeqTile);  // K矩阵临时布局
                LayoutV layoutVTemp(stackSeqTile, strideV);  // V矩阵临时布局
                
                // 重置矩阵乘法块的起始位置
                blockMmadQK.resetBlockStart();
                blockMmadPV.resetBlockStart();
                
                // 从全局内存加载Q矩阵到L1缓存
                blockMmadQK.loadQGM(gQ[gmOffsetQ], layoutQTemp, rowNum, qNBlockSize, qHeads);
#endif
                
                // 遍历处理KV序列的每个栈（包括预启动部分）
                for (uint32_t kvSIdx = 0; kvSIdx < kvSLoopNumTotal + preKVNum; kvSIdx ++) {
                    if (kvSIdx < kvSLoopNumTotal) {  // 处理实际的KV序列栈
                        // 计算当前栈的实际序列长度（最后一个栈可能较小）
                        if (kvSIdx + 1 > kvSLoopNumTotal - 1U) {
                            stackSeqTile = noSkipKvS - kvSIdx * MAX_KV_STACK_LEN;
                        } else {
                            stackSeqTile = MAX_KV_STACK_LEN;
                        }
                        
                        // 计算当前栈在工作区中的偏移
                        uint32_t curStackTileMod = stackSeqCount % (PRE_LAUNCH + 1U);
                        uint64_t gmOffsetS = static_cast<uint64_t>(coreIdx * WORKSPACE_BLOCK_SIZE_DB * (PRE_LAUNCH + 1U) +
                            curStackTileMod * WORKSPACE_BLOCK_SIZE_DB);  // QK^T结果S的偏移
                        
                        // 设置实际的QK矩阵乘法块形状
                        GemmCoord actualBlockShapeQK{rowNum, stackSeqTile, embed};
                        LayoutS layOutS(rowNum, stackSeqTile, stackSeqTilePad);  // S矩阵布局
                        
#ifdef __DAV_C220_CUBE__
                        // CUBE架构下执行QK矩阵乘法
                        if constexpr (PAGED_CACHE_FLAG) {
                            // 分页缓存模式下的QK矩阵乘法
                            blockMmadQK(
                                gQ[gmOffsetQ],      // Q矩阵全局内存地址
                                gK[gmOffsetK],      // K矩阵全局内存地址
                                gS[gmOffsetS],      // S矩阵全局内存地址
                                gBlockTable[blockBOffset],  // 块表
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
                                gBlockTable,        // 完整块表
                                layoutQTemp,
                                layoutKTemp,
                                layOutS,
                                actualBlockShapeQK,
                                kvSIdx,
                                kvSLoopNumTotal,
                                pagedBlockSize,
                                strideK);
                        }
                        // 设置QK矩阵乘法完成标志
                        Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(qkReady);
#endif
#ifdef __DAV_C220_VEC__
                        // VEC架构下的处理
                        LayoutP layOutP(rowNum, stackSeqTile, stackSeqTilePad);  // P矩阵布局（注意力权重）
                        LayoutMask layOutMask(COMP_TRIU_MASK_DIM_LEN, COMP_TRIU_MASK_DIM_LEN);  // 掩码布局（上三角掩码维度）
                        uint64_t gmOffsetP = gmOffsetS;  // P矩阵偏移（与S矩阵共享同一内存空间）
                        
                        // 计算因果掩码相关参数
                        uint32_t triUp = noSkipKvS - qSBlockSize;  // 上三角掩码的上边界
                        uint32_t triDown = noSkipKvS;  // 上三角掩码的下边界
                        uint32_t kvSStartIdx = kvSIdx * MAX_KV_STACK_LEN;  // 当前KV序列栈的起始索引
                        uint32_t kvSEndIdx = kvSStartIdx + stackSeqTile;  // 当前KV序列栈的结束索引
                        bool doTriUMask = triUp < kvSEndIdx - 1;  // 判断是否需要应用上三角掩码
                        
                        // 因果掩码处理分支
                        if constexpr (MASK_TYPE == FaiKenel::MaskType::MASK_CAUSAL) {
                            // 需要应用因果掩码的情况
                            if (doTriUMask) {
                                epilogueOnlineSoftmax(
                                    gP[gmOffsetP],  // P矩阵（注意力权重输出）
                                    gS[gmOffsetS],  // S矩阵（QK乘积结果）
                                    gMask,  // 掩码数据
                                    layOutP,  // P矩阵布局
                                    layOutS,  // S矩阵布局
                                    layOutMask,  // 掩码布局
                                    actualBlockShapeQK,  // 实际的QK矩阵乘法块形状
                                    (stackSeqCount == 0),  // 是否为第一个序列栈
                                    qSBlockSize,  // 查询序列的块大小
                                    qNBlockSize,  // 查询头的块大小
                                    curStackTileMod,  // 当前栈块的模块索引
                                    qkReady,  // QK矩阵乘法完成标志
                                    triUp,  // 上三角掩码上边界
                                    triDown,  // 上三角掩码下边界
                                    kvSStartIdx,  // KV序列起始索引
                                    kvSEndIdx);  // KV序列结束索引
                            } else {
                                // 不需要应用因果掩码的情况
                                uint32_t noMaskStackSeqNum = triUp / MAX_KV_STACK_LEN;  // 无掩码的序列栈数量
                                Arch::CrossCoreWaitFlag(qkReady);  // 等待QK矩阵乘法完成
                                epilogueOnlineSoftmax(
                                    gP[gmOffsetP],  // P矩阵（注意力权重输出）
                                    gS[gmOffsetS],  // S矩阵（QK乘积结果）
                                    layOutP,  // P矩阵布局
                                    layOutS,  // S矩阵布局
                                    actualBlockShapeQK,  // 实际的QK矩阵乘法块形状
                                    (stackSeqCount == 0),  // 是否为第一个序列栈
                                    (stackSeqCount == noMaskStackSeqNum - 1),  // 是否为最后一个无掩码序列栈
                                    qSBlockSize,  // 查询序列的块大小
                                    qNBlockSize,  // 查询头的块大小
                                    curStackTileMod);  // 当前栈块的模块索引
                            }
                        } else {
                            // 非因果掩码类型处理
                            Arch::CrossCoreWaitFlag(qkReady);  // 等待QK矩阵乘法完成
                            epilogueOnlineSoftmax(
                                gP[gmOffsetP],  // P矩阵（注意力权重输出）
                                gS[gmOffsetS],  // S矩阵（QK乘积结果）
                                layOutP,  // P矩阵布局
                                layOutS,  // S矩阵布局
                                actualBlockShapeQK,  // 实际的QK矩阵乘法块形状
                                (stackSeqCount == 0),  // 是否为第一个序列栈
                                0,  // 非因果掩码标志
                                qSBlockSize,  // 查询序列的块大小
                                qNBlockSize,  // 查询头的块大小
                                curStackTileMod);  // 当前栈块的模块索引
                        }
                        Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(softmaxReady);  // 设置Softmax完成标志，用于后续PV矩阵乘法同步
#endif
                    }
                    // PV矩阵乘法处理（仅当处理当前KV序列而非缓存KV序列时执行）
                    if (kvSIdx >= preKVNum) {
                        uint32_t nowkvSIdx = kvSIdx - preKVNum;  // 当前KV序列在总序列中的索引（排除预加载缓存）
                        
                        // 计算当前KV序列栈的实际大小（最后一个栈可能小于最大栈大小）
                        if (nowkvSIdx + 1 > kvSLoopNumTotal - 1U) {
                            stackSeqTile = noSkipKvS - nowkvSIdx * MAX_KV_STACK_LEN;  // 最后一个栈的实际大小
                        } 
                        else {
                            stackSeqTile = MAX_KV_STACK_LEN;  // 非最后一个栈使用最大栈大小
                        }
                        
                        // 计算当前栈块的模块索引，用于工作区内存管理
                        uint32_t curStackTileMod = (stackSeqCount - PRE_LAUNCH) % (PRE_LAUNCH + 1U);
                        
                        // 计算临时输出张量OTmp的全局内存偏移
                        uint64_t gmOffsetOTmp = 
                            static_cast<uint64_t>(coreIdx * WORKSPACE_BLOCK_SIZE_DB * (PRE_LAUNCH + 1U) + 
                            curStackTileMod * WORKSPACE_BLOCK_SIZE_DB);
                        
                        // PV矩阵乘法的实际块形状
                        GemmCoord actualBlockShapePV{rowNum, embedV, stackSeqTile};
                        
                        // 临时输出张量OTmp的布局
                        LayoutOTmp layoutOTmp(rowNum, embedV, embedRoundV);
                        
#ifdef __DAV_C220_CUBE__
                        // CUBE架构下的PV矩阵乘法处理
                        LayoutP layoutPTemp(rowNum, stackSeqTile, stackSeqTilePad);  // 临时P矩阵（注意力权重）布局
                        
                        // 计算P矩阵的全局内存偏移
                        uint64_t gmOffsetP = coreIdx * WORKSPACE_BLOCK_SIZE_DB * (PRE_LAUNCH + 1) + 
                            curStackTileMod * WORKSPACE_BLOCK_SIZE_DB;
                        
                        // 分页缓存模式下的PV矩阵乘法
                        if constexpr (PAGED_CACHE_FLAG) {
                            blockMmadPV(
                                gP[gmOffsetP],  // P矩阵（注意力权重）
                                gV[gmOffsetV],  // V矩阵（值张量）
                                gOTmp[gmOffsetOTmp],  // 临时输出张量OTmp
                                gBlockTable[blockBOffset],  // 块表（分页缓存用）
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
                                gBlockTable,  // 块表（非分页模式下可能未使用）
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
                        Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(pvReady);  // 设置PV矩阵乘法完成标志，用于后续输出处理同步
#endif
#ifdef __DAV_C220_VEC__
                        // VEC架构下的输出缩放和处理
                        LayoutO layoutO(qSeqlen, embed * qHeads);  // 输出张量O的布局
                        LayoutUpdate layoutUpdate(rowNum, embed, embedRound);  // 更新张量的布局
                        LayoutLse layoutLse(totalQTokens, qHeads);  // 对数和张量的布局
                        uint64_t gmOffsetUpdate = (uint64_t)(coreIdx * WORKSPACE_BLOCK_SIZE_DB);  // 更新张量的全局内存偏移

                        Arch::CrossCoreWaitFlag(pvReady);  // 等待PV矩阵乘法完成
                        epilogueRescaleO(
                            gO[gmOffsetO],  // 最终输出张量O
                            gOTmp[gmOffsetOTmp],  // 临时输出张量OTmp（PV矩阵乘法结果）
                            gOUpdate[gmOffsetUpdate],  // 更新张量（用于累加结果）
                            gLse[gmOffsetLse],  // 对数和张量（用于数值稳定性）
                            layoutO,  // 输出张量O的布局
                            layoutOTmp,  // 临时输出张量OTmp的布局
                            layoutUpdate,  // 更新张量的布局
                            layoutLse,  // 对数和张量的布局
                            actualBlockShapePV,  // PV矩阵乘法的实际块形状
                            qSBlockSize,  // 查询序列的块大小
                            qNBlockSize,  // 查询头的块大小
                            (stackSeqCount - PRE_LAUNCH == 0),  // 是否为第一个输出块
                            nowkvSIdx + 1 >= kvSLoopNumTotal,  // 是否为最后一个KV序列块
                            curStackTileMod);  // 当前栈块的模块索引
#endif
                    }
                    stackSeqCount++;
                }
            }
#ifdef __DAV_C220_CUBE__
            // CUBE架构下的硬件事件同步等待
            // 等待所有MTE1事件完成
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID4);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID5);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID6);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID7);

            // 等待所有FIX_M事件完成
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);

            // 等待所有MTE1_MTE2事件完成
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID6);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
#endif
#ifdef __DAV_C220_VEC__
            // VEC架构下的硬件事件同步等待
            // 等待所有MTE3_V事件完成
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
            
            // 等待所有MTE3_MTE2事件完成
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
            
            // 等待所有V_MTE2事件完成
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
#endif
            AscendC::PipeBarrier<PIPE_ALL>();  // 管道屏障，确保所有计算管道完成当前任务
        }

    private:
        Arch::Resource<ArchTag> resource;  // 硬件资源管理对象
        Arch::CrossCoreFlag qkReady{QK_READY_ID};  // QK矩阵乘法完成标志（跨核心同步）
        Arch::CrossCoreFlag softmaxReady{SOFTMAX_READY_ID};  // Softmax完成标志（跨核心同步）
        Arch::CrossCoreFlag pvReady{PV_READY_ID};  // PV矩阵乘法完成标志（跨核心同步）
    };
}

// SplitFuse命名空间，包含FlashAttention推理的核心函数
namespace SplitFuse {
    // FlashAttention推理内核函数模板
    // 模板参数：
    // - InputDtypeQ: 查询张量Q的数据类型，默认为half
    // - InputDtypeKv: 键值张量K/V的数据类型，默认为half
    // - IntermCalcPrec: 中间计算精度，默认为float
    // - PagedCacheFlag: 是否使用分页缓存，默认为false
    // - maskCategory: 掩码类型，默认为无掩码
    // - inLayout: 输入张量布局，默认为TND（批次-头-序列-维度）
    // - lseMode: 对数和计算模式，默认为NONE
    template <
        typename InputDtypeQ = half,
        typename InputDtypeKv = half,
        typename IntermCalcPrec = float,
        bool PagedCacheFlag = false,
        FaiKenel::MaskType maskCategory = FaiKenel::MaskType::NO_MASK,
        FaiKenel::inputLayout inLayout = FaiKenel::inputLayout::TND,
        Epilogue::LseModeT lseMode = Epilogue::LseModeT::NONE>
    __global__ __aicore__ void FAInfer(
        uint64_t fftsAddr,  // 快速傅里叶变换同步地址
        GM_ADDR q,  // 查询张量Q的全局内存地址
        GM_ADDR k,  // 键张量K的全局内存地址
        GM_ADDR v,  // 值张量V的全局内存地址
        GM_ADDR mask,  // 掩码张量的全局内存地址
        GM_ADDR blockTables,  // 块表的全局内存地址（分页缓存用）
        GM_ADDR o,  // 输出张量O的全局内存地址
        GM_ADDR lse,  // 对数和张量的全局内存地址
        GM_ADDR actualQseqlen,  // 实际查询序列长度的全局内存地址
        GM_ADDR actualKvseqlen,  // 实际键值序列长度的全局内存地址
        GM_ADDR workspace,  // 工作区内存地址
        GM_ADDR tiling)  // 分块数据的全局内存地址
    {
        AscendC::SetSyncBaseAddr(fftsAddr);  // 设置同步基地址

        // 定义硬件架构和各种张量的元素类型与布局
        using ArchTag = Arch::AtlasA2;  // 硬件架构标签（Atlas A2）
        using ElementQ = InputDtypeQ;  // Q张量元素类型
        using LayoutQ = layout::RowMajor;  // Q张量布局（行优先）
        using ElementK = InputDtypeKv;  // K张量元素类型
        using LayoutK = layout::ColumnMajor;  // K张量布局（列优先）
        using ElementV = InputDtypeKv;  // V张量元素类型
        using LayoutV = layout::RowMajor;  // V张量布局（行优先）
        using ElementS = IntermCalcPrec;  // S矩阵元素类型（QK乘积结果）
        using LayoutS = layout::RowMajor;  // S矩阵布局（行优先）
        using ElementP = InputDtypeQ;  // P矩阵元素类型（注意力权重）
        using LayoutP = layout::RowMajor;  // P矩阵布局（行优先）
        using ElementO = InputDtypeQ;  // 输出张量O元素类型
        using LayoutO = layout::RowMajor;  // 输出张量O布局（行优先）
        using ElementLse = float;  // 对数和张量元素类型
        using LayoutLse = layout::RowMajor;  // 对数和张量布局（行优先）
        using ElementMask = int8_t;  // 掩码张量元素类型
        using LayoutMask = layout::RowMajor;  // 掩码张量布局（行优先）
        using ElementOTmp = IntermCalcPrec;  // 临时输出张量OTmp元素类型
        using LayoutOTmp = layout::RowMajor;  // 临时输出张量OTmp布局（行优先）
        using ElementUpdate = IntermCalcPrec;  // 更新张量元素类型
        using LayoutUpdate = layout::RowMajor;  // 更新张量布局（行优先）

        // QK矩阵乘法相关类型定义
        using L1TileShapeQK = GemmShape<Q_TILE_CEIL, 128, 128>;  // L1缓存分块形状
        using L0TileShapeQK = GemmShape<128, 128, 128>;  // L0缓存分块形状
        using DispatchPolicyQK = Gemm::MmadAtlasA2FAIQKT<PagedCacheFlag, false>;  // 调度策略
        using QType = Gemm::GemmType<ElementQ, LayoutQ>;  // Q张量类型
        using KType = Gemm::GemmType<ElementK, LayoutK>;  // K张量类型
        using SType = Gemm::GemmType<ElementS, LayoutS>;  // S矩阵类型
        using BlockMmadQK = Gemm::Block::BlockMmad<DispatchPolicyQK, L1TileShapeQK, L0TileShapeQK,
                                                   QType, KType, SType>;  // QK矩阵乘法块

        // 在线Softmax相关类型定义
        using DispatchPolicyOnlineSoftmax = Epilogue::EpilogueAtlasA2OnlineSoftmaxT<lseMode, IntermCalcPrec>;  // 调度策略
        using PType = Gemm::GemmType<ElementP, LayoutP>;  // P矩阵类型
        using maskType = Gemm::GemmType<ElementMask, LayoutMask>;  // 掩码类型
        using EpilogueOnlineSoftmax = 
            Epilogue::Block::BlockEpilogue<DispatchPolicyOnlineSoftmax, PType, SType, maskType>;  // 在线Softmax块

        // PV矩阵乘法相关类型定义
        using L1TileShapePV = GemmShape<128, 128, 256>;  // L1缓存分块形状
        using L0TileShapePV = GemmShape<128, 128, 128>;  // L0缓存分块形状
        using DispatchPolicyPV = Gemm::MmadAtlasA2FAIPVT<PagedCacheFlag, false>;  // 调度策略
        using VType = Gemm::GemmType<ElementV, LayoutV>;  // V张量类型
        using OTmpType = Gemm::GemmType<ElementOTmp, LayoutOTmp>;  // 临时输出张量类型
        using BlockMmadPV = Gemm::Block::BlockMmad<DispatchPolicyPV, L1TileShapePV, L0TileShapePV,
                                                   PType, VType, OTmpType>;  // PV矩阵乘法块

        // 输出缩放相关类型定义
        using DispatchPolicyRescaleO = Epilogue::EpilogueAtlasA2RescaleOT<lseMode, IntermCalcPrec>;  // 调度策略
        using OType = Gemm::GemmType<ElementO, LayoutO>;  // 输出张量类型
        using OUpdateType = Gemm::GemmType<ElementUpdate, LayoutUpdate>;  // 更新张量类型
        using LseType = Gemm::GemmType<ElementLse, LayoutLse>;  // 对数和张量类型
        using EpilogueRescaleO = 
            Epilogue::Block::BlockEpilogue<DispatchPolicyRescaleO, OType, OTmpType, OUpdateType, LseType>;  // 输出缩放块

        // 定义FlashAttention推理内核类型
        using FAInferKernel = FAInferKernel<BlockMmadQK, BlockMmadPV, EpilogueOnlineSoftmax, EpilogueRescaleO,
                                            PagedCacheFlag, maskCategory, inLayout>;
        
        // 创建内核参数对象，包含所有输入输出张量和配置信息
        FAIKernelParams params{q, k, v, mask, blockTables, actualQseqlen, actualKvseqlen, o, lse, workspace, tiling};
        
        // 创建FlashAttention推理内核实例
        FAInferKernel flashAttnInfer;
        
        // 调用内核执行FlashAttention推理计算
        flashAttnInfer(params);
    }
}  // 结束SplitFuse命名空间