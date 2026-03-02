/*
 * Copyright 2023 Huawei Technologies Co., Ltd.
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

#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_RESCALE_O_NO_SPLIT_ROW_HPP_T
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_RESCALE_O_NO_SPLIT_ROW_HPP_T

// Catlass库核心头文件
#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"

// 自定义块结构头文件
#include "fa_block.h"

namespace Catlass::Epilogue::Block {
/**
 * @brief FlashAttention输出重缩放的块级Epilogue实现
 *
 * 专门为Atlas A2平台优化的输出重缩放Epilogue实现，用于FlashAttention计算的最后阶段。
 * 该实现不分割行，使用float精度进行计算，支持LSE（对数求和指数）的多种计算模式。
 *
 * @tparam OutputType_ 输出类型和布局
 * @tparam InputType_ 输入类型和布局
 * @tparam UpdateType_ 更新类型和布局
 * @tparam LseType_ LSE（对数求和指数）类型和布局
 * @tparam LSE_MODE_ LSE计算模式
 */
template <
    class OutputType_,     ///< 输出类型和布局
    class InputType_,      ///< 输入类型和布局
    class UpdateType_,     ///< 更新类型和布局
    class LseType_,        ///< LSE（对数求和指数）类型和布局
    LseModeT LSE_MODE_>    ///< LSE计算模式
class BlockEpilogue<
    EpilogueAtlasA2RescaleOT<LSE_MODE_, float>,
    OutputType_,
    InputType_,
    UpdateType_,
    LseType_>
{
public:
    // 类型别名
    using DispatchPolicy = EpilogueAtlasA2RescaleOT<LSE_MODE_, float>;  ///< Atlas A2输出重缩放Epilogue调度策略
    using ArchTag = typename DispatchPolicy::ArchTag;                   ///< 架构标签类型

    // 元素类型定义
    using ElementOutput = typename OutputType_::Element;                ///< 输出元素类型
    using ElementInput = typename InputType_::Element;                  ///< 输入元素类型
    using ElementUpdate = typename UpdateType_::Element;                ///< 更新元素类型
    using ElementLse = typename LseType_::Element;                      ///< LSE元素类型

    // 布局类型定义
    using LayoutOutput = typename OutputType_::Layout;                  ///< 输出布局类型
    using LayoutInput = typename InputType_::Layout;                    ///< 输入布局类型
    using LayoutUpdate = typename UpdateType_::Layout;                  ///< 更新布局类型
    using LayoutLse = typename LseType_::Layout;                        ///< LSE布局类型

    // 静态常量定义
    static constexpr LseModeT LSE_MODE = DispatchPolicy::LSE_MODE;      ///< LSE计算模式

    // 元素数量和块大小常量（针对Atlas A2架构优化）
    static constexpr uint32_t HALF_ELENUM_PER_BLK = 16;                 ///< 每个块的半精度元素数量
    static constexpr uint32_t BLOCK_SIZE = 16;                          ///< 基本块大小（适配硬件计算单元）
    static constexpr uint32_t HALF_ELENUM_PER_VECCALC = 128;            ///< 每个向量计算的半精度元素数量
    static constexpr uint32_t FLOAT_ELENUM_PER_VECCALC = 64;            ///< 每个向量计算的浮点数元素数量
    static constexpr uint32_t HALF_ELENUM_PER_LINE = 256;               ///< 每行的半精度元素数量
    static constexpr uint32_t FLOAT_ELENUM_PER_LINE = 128;              ///< 每行的浮点数元素数量
    static constexpr uint32_t MULTIPLIER = 2;                           ///< 乘法因子（用于类型转换）
    static constexpr uint32_t FLOAT_BLOCK_SIZE = 8;                     ///< 浮点数块大小
    static constexpr uint32_t FLOAT_VECTOR_SIZE = 64;                   ///< 浮点数向量大小
    static constexpr uint32_t UB_UINT8_VECTOR_SIZE = 1024;              ///< UB中uint8向量大小（字节）
    static constexpr uint32_t UB_UINT8_BLOCK_SIZE = 16384;              ///< UB中uint8块大小（字节）
    static constexpr uint32_t HALF_DM_UB_SIZE = 64;                     ///< 半精度DM UB大小
    static constexpr uint32_t HALF_LL_UB_SIZE = 256;                    ///< 半精度LL UB大小
    static constexpr uint32_t VECTOR_SIZE = 128;                        ///< 通用向量大小
    static constexpr uint32_t NUM4 = 4;                                 ///< 数字4常量（用于循环控制）
    static constexpr uint32_t MAX_UB_O_ELEM_NUM = 8192;                 ///< UB中O元素的最大数量（限制内存使用）
    static constexpr uint32_t MAX_ROW_NUM_SUB_CORE = 256;               ///< 子核最大行数（硬件限制）
    static constexpr uint32_t SIZE_OF_16BIT = 2;                        ///< 16位数据大小（字节）

    /**
     * @brief 构造函数：初始化UB缓冲区
     *
     * 为各种中间张量分配统一缓冲区(UB)空间，这是Atlas A2架构上的高速内存区域，
     * 用于存储计算过程中的中间结果，减少全局内存访问。
     *
     * @param resource 架构资源对象，用于获取UB缓冲区
     */
    __aicore__ inline
    BlockEpilogue(Arch::Resource<ArchTag> &resource)
    {
        // UB空间偏移常量定义（基于uint8大小计算，确保内存对齐）
        constexpr uint32_t LO_UB_TENSOR_OFFSET = 6 * UB_UINT8_BLOCK_SIZE;                          ///< LO张量在UB中的起始偏移
        constexpr uint32_t GO_UB_TENSOR_OFFSET = 8 * UB_UINT8_BLOCK_SIZE;                          ///< GO张量在UB中的起始偏移
        constexpr uint32_t TV_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE;                         ///< TV张量在UB中的起始偏移
        constexpr uint32_t HM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 9 * UB_UINT8_VECTOR_SIZE;  ///< HM张量在UB中的起始偏移
        constexpr uint32_t GM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 10 * UB_UINT8_VECTOR_SIZE; ///< GM张量在UB中的起始偏移
        constexpr uint32_t LSE_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 11 * UB_UINT8_VECTOR_SIZE; ///< LSE张量在UB中的起始偏移
        constexpr uint32_t DM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 12 * UB_UINT8_VECTOR_SIZE;   ///< DM张量在UB中的起始偏移
        constexpr uint32_t GL_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 13 * UB_UINT8_VECTOR_SIZE;   ///< GL张量在UB中的起始偏移

        // 从资源中获取各种UB缓冲区，将不同类型的指针指向UB中的特定偏移位置
        loUbTensor = resource.ubBuf.template GetBufferByByte<float>(LO_UB_TENSOR_OFFSET);           ///< 本地输出UB张量
        dmUbTensor = resource.ubBuf.template GetBufferByByte<float>(DM_UB_TENSOR_OFFSET);           ///< 临时矩阵DM UB张量
        glUbTensor = resource.ubBuf.template GetBufferByByte<float>(GL_UB_TENSOR_OFFSET);           ///< 全局LSE UB张量
        tvUbTensor = resource.ubBuf.template GetBufferByByte<float>(TV_UB_TENSOR_OFFSET);           ///< 临时向量UB张量
        goUbTensor16 = resource.ubBuf.template GetBufferByByte<ElementOutput>(GO_UB_TENSOR_OFFSET); ///< 全局输出16位UB张量
        goUbTensor32 = resource.ubBuf.template GetBufferByByte<float>(GO_UB_TENSOR_OFFSET);         ///< 全局输出32位UB张量
        hmUbTensor = resource.ubBuf.template GetBufferByByte<float>(HM_UB_TENSOR_OFFSET);           ///< 本地矩阵M UB张量
        gmUbTensor = resource.ubBuf.template GetBufferByByte<float>(GM_UB_TENSOR_OFFSET);           ///< 全局矩阵M UB张量
        lse32_ubuf_tensor = resource.ubBuf.template GetBufferByByte<float>(LSE_UB_TENSOR_OFFSET);   ///< LSE 32位UB张量
    }

    /**
     * @brief 析构函数：释放资源（无实际操作）
     *
     * UB缓冲区由架构资源对象管理，不需要在析构函数中显式释放。
     */
    __aicore__ inline
    ~BlockEpilogue() {}

    /**
     * @brief 设置向量掩码，用于处理非完整向量的情况
     *
     * 在处理序列末尾的非完整向量时，需要设置掩码来只计算有效元素，
     * 忽略填充元素。该方法根据有效元素数量设置合适的向量掩码。
     *
     * @param len 有效元素数量
     */
    __aicore__ inline
    void SetMask(int32_t len)
    {
        uint64_t mask = 0;
        uint64_t one = 1;
        uint64_t temp = static_cast<uint64_t>(len) % static_cast<uint64_t>(FLOAT_VECTOR_SIZE);
        
        // 创建掩码，标记有效元素
        for (uint64_t i = 0; i < temp; i++) {
            mask |= one << i;
        }

        // 根据长度设置不同的向量掩码
        if (len == VECTOR_SIZE) {
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);  ///< 完整向量，全1掩码
        } else if (len >= FLOAT_VECTOR_SIZE) {
            AscendC::SetVectorMask<int8_t>(mask, (uint64_t)-1);           ///< 大于等于浮点数向量大小
        } else {
            AscendC::SetVectorMask<int8_t>(0x0, mask);                    ///< 小于浮点数向量大小
        }
    }

    /**
     * @brief 将输出矩阵从UB复制到全局内存
     *
     * 该方法负责将处理完成的输出张量从统一缓冲区(UB)复制到全局内存，
     * 支持处理前导token、完整头和尾随token的不同情况，确保内存布局正确。
     *
     * @param gOutput 全局内存输出张量
     * @param proTokenIdx 前导token的索引
     * @param proTokenNum 前导token的数量
     * @param epiTokenNum 尾随token的数量
     * @param integralHeadNum 完整头的数量
     * @param qSThisSubBlock 当前子块的查询序列长度
     * @param embed 嵌入维度
     * @param oHiddenSize 输出隐藏层大小
     */
    __aicore__ inline
    void CopyOToGm(AscendC::GlobalTensor<ElementOutput> gOutput, uint32_t proTokenIdx, uint32_t proTokenNum, 
        uint32_t epiTokenNum, uint32_t integralHeadNum, uint32_t qSThisSubBlock, uint32_t embed, uint32_t oHiddenSize)
    {
        uint32_t innerOGmOffset = 0;   ///< GM内偏移
        uint32_t innerGOUbOffset = 0;  ///< UB内偏移
        
        // 复制前导标记（如果有）
        if (proTokenNum != 0U) {
            AscendC::DataCopyPad(
                gOutput[innerOGmOffset + proTokenIdx * oHiddenSize],
                goUbTensor16[innerGOUbOffset],
                AscendC::DataCopyExtParams(
                    proTokenNum, embed * SIZE_OF_16BIT, 0, (oHiddenSize - embed) * SIZE_OF_16BIT, 0));
            innerOGmOffset += embed;
            innerGOUbOffset += proTokenNum * embed;
        }
        
        // 复制整数头部数据
        for (uint32_t qN_idx = 0; qN_idx < integralHeadNum; qN_idx++) {
            AscendC::DataCopyPad(
                gOutput[innerOGmOffset],
                goUbTensor16[innerGOUbOffset],
                AscendC::DataCopyExtParams(
                    qSThisSubBlock, embed * SIZE_OF_16BIT, 0, (oHiddenSize - embed) * SIZE_OF_16BIT, 0));
            innerOGmOffset += embed;
            innerGOUbOffset += qSThisSubBlock * embed;
        }
        
        // 复制尾随标记（如果有）
        if (epiTokenNum != 0U) {
            AscendC::DataCopyPad(
                gOutput[innerOGmOffset],
                goUbTensor16[innerGOUbOffset],
                AscendC::DataCopyExtParams(
                    epiTokenNum, embed * SIZE_OF_16BIT, 0, (oHiddenSize - embed) * SIZE_OF_16BIT, 0));
        }
    }

    /**
     * @brief 子核计算核心函数，执行输出重缩放操作
     *
     * 这是Epilogue的核心计算函数，负责执行输出重缩放、LSE处理和结果转换等操作。
     * 具体实现会根据LSE模式和数据类型进行不同的处理流程。
     *
     * @param gOutput 全局输出张量
     * @param gInput 全局输入张量（通常是PV矩阵乘法的输出）
     * @param gUpdate 全局更新张量（用于累加计算）
     * @param gLse 全局LSE张量（对数求和指数）
     * @param layoutOutput 输出布局
     * @param layoutInput 输入布局
     * @param layoutUpdate 更新布局
     * @param layoutLse LSE布局
     * @param qNThisSubBlock 当前子块的查询头数
     * @param qSThisSubBlock 当前子块的查询序列长度
     */
    __aicore__ inline
    void SubCoreCompute(
        AscendC::GlobalTensor<ElementOutput> gOutput,      ///< 全局输出张量
        AscendC::GlobalTensor<ElementInput> gInput,        ///< 全局输入张量
        AscendC::GlobalTensor<ElementUpdate> gUpdate,      ///< 全局更新张量
        AscendC::GlobalTensor<ElementLse> gLse,            ///< 全局LSE张量
        const LayoutOutput &layoutOutput,                  ///< 输出布局
        const LayoutInput &layoutInput,                    ///< 输入布局
        const LayoutUpdate &layoutUpdate,                  ///< 更新布局
        const LayoutLse &layoutLse,                        ///< LSE布局
        uint32_t qNThisSubBlock,                           ///< 当前子块的查询头数
        uint32_t qSThisSubBlock,                           ///< 当前子块的查询序列长度
        uint32_t totalRowNum,                              // 总行数
        uint32_t isFirstStackTile,                         // 是否为第一个堆叠瓦片
        uint32_t isLastStackTile,                          // 是否为最后一个堆叠瓦片
        uint32_t curStackTileMod,                          // 当前堆叠瓦片模块
        uint32_t needRowLoop,                              // 是否需要行循环
        uint32_t isLastRowLoop,                            // 是否为最后一个行循环
        uint32_t rowOffsetLoop,                            // 行循环偏移
        uint32_t proTokenIdx,                              // 前导标记索引
        uint32_t proTokenNum,                              // 前导标记数量
        uint32_t epiTokenNum,                              // 尾随标记数量
        uint32_t integralHeadNum)                          // 整数头部数量
    {
        // 子核计算的核心函数，执行输出重缩放操作
        
        // 计算各种尺寸参数
        uint32_t curRowNum = layoutInput.shape(0);           // 当前行数
        uint32_t embed = layoutInput.shape(1);               // 嵌入维度
        uint32_t embedRound = layoutInput.stride(0);         // 嵌入步长（对齐后）
        uint32_t curRowNumRound = RoundUp(curRowNum, FLOAT_BLOCK_SIZE);  // 当前行数（对齐后）
        uint32_t qSBlockSize = layoutOutput.shape(0);        // 查询序列块大小
        uint32_t oHiddenSize = layoutOutput.shape(1);        // 输出隐藏层大小
        uint32_t qHeads = layoutLse.shape(1);                // 查询头数
        uint32_t dmUbOffsetCurStackTile = curStackTileMod * MAX_ROW_NUM_SUB_CORE + rowOffsetLoop;  // 当前堆叠瓦片的DM UB偏移

        // 处理非第一个堆叠瓦片的情况
        if (!isFirstStackTile) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);  // 等待事件
            // 从全局内存复制输入数据到本地UB
            AscendC::DataCopy(
                loUbTensor, gInput, AscendC::DataCopyParams(1, curRowNum * embedRound / FLOAT_BLOCK_SIZE, 0, 0));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);   // 设置事件
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);   // 等待事件6
        if (!isFirstStackTile) {
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            AscendC::Brcb(tvUbTensor.ReinterpretCast<uint32_t>(),
                dmUbTensor[dmUbOffsetCurStackTile].ReinterpretCast<uint32_t>(),
                curRowNumRound / FLOAT_BLOCK_SIZE,
                AscendC::BrcbRepeatParams(1, 8));
            AscendC::PipeBarrier<PIPE_V>();
            if (needRowLoop) {
                AscendC::DataCopy(
                    goUbTensor32, gUpdate,
                    AscendC::DataCopyParams(1, curRowNum * embedRound / FLOAT_BLOCK_SIZE, 0, 0));
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
            }
            // *** go = go * dm_block
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            for (uint32_t vmul_idx = 0; vmul_idx < embed / FLOAT_VECTOR_SIZE; ++vmul_idx) {
                AscendC::Mul<float, false>(
                    goUbTensor32[vmul_idx * FLOAT_VECTOR_SIZE],
                    goUbTensor32[vmul_idx * FLOAT_VECTOR_SIZE],
                    tvUbTensor,
                    (uint64_t)0,
                    curRowNum,
                    AscendC::BinaryRepeatParams(
                        1, 1, 0, embedRound / FLOAT_BLOCK_SIZE, embedRound / FLOAT_BLOCK_SIZE, 1));
            }
            if (embed % FLOAT_VECTOR_SIZE > 0) {
                SetMask(embed % FLOAT_VECTOR_SIZE);
                AscendC::Mul<float, false>(
                    goUbTensor32[embed / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                    goUbTensor32[embed / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                    tvUbTensor,
                    (uint64_t)0,
                    curRowNum,
                    AscendC::BinaryRepeatParams(
                        1, 1, 0, embedRound / FLOAT_BLOCK_SIZE, embedRound / FLOAT_BLOCK_SIZE, 1));
                AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            // *** go = lo + go
            AscendC::Add<float, false>(
                goUbTensor32,
                goUbTensor32,
                loUbTensor,
                (uint64_t)0,
                (curRowNum * embedRound + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
        } else {
            // *** go = lo
            AscendC::DataCopy(
                goUbTensor32, gInput, AscendC::DataCopyParams(1, curRowNum * embedRound / FLOAT_BLOCK_SIZE, 0, 0));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        }

        if (isLastStackTile) {
            // *** gl_block = expand_to_block(gl)
            AscendC::Brcb(
                tvUbTensor.ReinterpretCast<uint32_t>(),
                glUbTensor.ReinterpretCast<uint32_t>()[rowOffsetLoop],
                curRowNumRound / FLOAT_BLOCK_SIZE,
                AscendC::BrcbRepeatParams(1, 8));
            AscendC::PipeBarrier<PIPE_V>();
            // *** go = go / gl_block
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            for (uint32_t vdiv_idx = 0; vdiv_idx < embed / FLOAT_VECTOR_SIZE; ++vdiv_idx) {
                AscendC::Div<float, false>(
                    goUbTensor32[vdiv_idx * FLOAT_VECTOR_SIZE],
                    goUbTensor32[vdiv_idx * FLOAT_VECTOR_SIZE],
                    tvUbTensor,
                    (uint64_t)0,
                    curRowNum,
                    AscendC::BinaryRepeatParams(
                        1, 1, 0, embedRound / FLOAT_BLOCK_SIZE, embedRound / FLOAT_BLOCK_SIZE, 1));
            }
            if (embed % FLOAT_VECTOR_SIZE > 0) {
                SetMask(embed % FLOAT_VECTOR_SIZE);
                AscendC::Div<float, false>(
                    goUbTensor32[embed / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                    goUbTensor32[embed / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                    tvUbTensor,
                    (uint64_t)0,
                    curRowNum,
                    AscendC::BinaryRepeatParams(
                        1, 1, 0, embedRound / FLOAT_BLOCK_SIZE, embedRound / FLOAT_BLOCK_SIZE, 1));
                AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            }
            AscendC::PipeBarrier<PIPE_V>();

            // *** go = castfp32to16(go)
            if (std::is_same<ElementOutput, bfloat16_t>::value) {
                AscendC::Cast<ElementOutput, float, false>(
                    goUbTensor16, goUbTensor32,
                    AscendC::RoundMode::CAST_RINT, (uint64_t)0,
                    (curRowNum * embedRound + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                    AscendC::UnaryRepeatParams(1, 1, 4, 8));
            } else {
                AscendC::Cast<ElementOutput, float, false>(
                    goUbTensor16, goUbTensor32,
                    AscendC::RoundMode::CAST_NONE, (uint64_t)0,
                    (curRowNum * embedRound + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                    AscendC::UnaryRepeatParams(1, 1, 4, 8));
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

            // ***move O to GM
            CopyOToGm(
                gOutput, proTokenIdx, proTokenNum, epiTokenNum, integralHeadNum, qSThisSubBlock, embed, oHiddenSize);
            if constexpr (LSE_MODE_ == LseModeT::OUT_ONLY) {
                if (isLastRowLoop) {
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Ln<float, false>(
                        lse32_ubuf_tensor,
                        glUbTensor,
                        (uint64_t)0, CeilDiv(totalRowNum, FLOAT_VECTOR_SIZE),
                        AscendC::UnaryRepeatParams(1, 1, 8, 8));

                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add<float, false>(
                        lse32_ubuf_tensor,
                        lse32_ubuf_tensor,
                        gmUbTensor,
                        (uint64_t)0, CeilDiv(totalRowNum, FLOAT_VECTOR_SIZE),
                        AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
                    AscendC::PipeBarrier<PIPE_V>();

                    // *** lse_block = expand_to_block(lse)
                    AscendC::Brcb(
                        tvUbTensor.ReinterpretCast<uint32_t>(),
                        lse32_ubuf_tensor.ReinterpretCast<uint32_t>(),
                        CeilDiv(totalRowNum, FLOAT_BLOCK_SIZE),
                        AscendC::BrcbRepeatParams(1, 8));
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID4);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID4);
                    
                    if (qNThisSubBlock == 0U) {
                        AscendC::DataCopyPad(
                            gLse, tvUbTensor,
                            AscendC::DataCopyExtParams(
                                totalRowNum, sizeof(float), 0, (qHeads - 1) * sizeof(float), 0));
                    } else {
                        for (uint32_t qNIdx = 0; qNIdx < qNThisSubBlock; qNIdx++) {
                            AscendC::DataCopyPad(
                                gLse[qNIdx],
                                tvUbTensor[qNIdx * qSBlockSize * FLOAT_BLOCK_SIZE],
                                AscendC::DataCopyExtParams(
                                    qSBlockSize, sizeof(float), 0, (qHeads - 1) * sizeof(float), 0));
                        }
                    }
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
                }
            }
        } else if (needRowLoop) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID5);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID5);
            AscendC::DataCopy(
                gUpdate, goUbTensor32, AscendC::DataCopyParams(1, curRowNum * embedRound / FLOAT_BLOCK_SIZE, 0, 0));
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
    }

    /**
     * @brief 执行入口函数，协调整个输出重缩放Epilogue过程
     *
     * 该函数负责协调整个输出重缩放Epilogue的执行流程，包括分块处理、
     * 子块调度和核心计算调用。它将输入数据分块并分发给SubCoreCompute函数处理。
     *
     * @param gOutput 全局输出张量
     * @param gInput 全局输入张量（通常是PV矩阵乘法的输出）
     * @param gUpdate 全局更新张量（用于累加计算）
     * @param gLse 全局LSE张量（对数求和指数）
     * @param layoutOutput 输出布局
     * @param layoutInput 输入布局
     * @param layoutUpdate 更新布局
     * @param layoutLse LSE布局
     * @param actualBlockShape 实际块形状
     * @param qSBlockSize 查询序列块大小
     * @param qNBlockSize 查询头块大小
     * @param isFirstStackTile 是否为第一个堆叠瓦片
     * @param isLastStackTile 是否为最后一个堆叠瓦片
     * @param curStackTileMod 当前堆叠瓦片模块
     */
    __aicore__ inline
    void operator()(
        AscendC::GlobalTensor<ElementOutput> gOutput,      ///< 全局输出张量
        AscendC::GlobalTensor<ElementInput> gInput,        ///< 全局输入张量
        AscendC::GlobalTensor<ElementUpdate> gUpdate,      ///< 全局更新张量
        AscendC::GlobalTensor<ElementLse> gLse,            ///< 全局LSE张量
        const LayoutOutput &layoutOutput,                  ///< 输出布局
        const LayoutInput &layoutInput,                    ///< 输入布局
        const LayoutUpdate &layoutUpdate,                  ///< 更新布局
        const LayoutLse &layoutLse,                        ///< LSE布局
        GemmCoord actualBlockShape,                        ///< 实际块形状
        uint32_t qSBlockSize,                              ///< 查询序列块大小
        uint32_t qNBlockSize,                              ///< 查询头块大小
        uint32_t isFirstStackTile,                         ///< 是否为第一个堆叠瓦片
        uint32_t isLastStackTile,                          ///< 是否为最后一个堆叠瓦片
        uint32_t curStackTileMod)                          ///< 当前堆叠瓦片模块
    {
        // 获取实际块尺寸参数
        uint32_t rowNum = actualBlockShape.m();                     ///< 行数
        uint32_t embed = actualBlockShape.n();                     ///< 嵌入维度
        uint32_t maxRowNumPerLoop = MAX_UB_O_ELEM_NUM / embed;     ///< 每轮循环的最大行数
        uint32_t rowNumTile = RoundDown(maxRowNumPerLoop, FLOAT_BLOCK_SIZE);  ///< 对齐后的每轮循环行数

        // 获取当前子块信息
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();          ///< 当前子块索引
        uint32_t subBlockNum = AscendC::GetSubBlockNum();          ///< 子块总数

        // 计算子块分割参数
        uint32_t qNSplitSubBlock = qNBlockSize / subBlockNum;      ///< 查询头的子块分割大小
        uint32_t qNThisSubBlock = (qNBlockSize == 1U) ? 0          ///< 当前子块的查询头数
                                  : (subBlockIdx == 1U) ? (qNBlockSize - qNSplitSubBlock)
                                                       : qNSplitSubBlock;
        uint32_t inRowSplitSubBlock =                               ///< 输入行的子块分割大小
            (qNBlockSize == 1U) ? (qSBlockSize / subBlockNum) : (qSBlockSize * qNSplitSubBlock);
        uint32_t inRowActualThisSubBlock = (subBlockIdx == 1U) ? (rowNum - inRowSplitSubBlock) : inRowSplitSubBlock;  ///< 当前子块的实际输入行数
        uint32_t inRowOffsetThisSubBlock = subBlockIdx * inRowSplitSubBlock;  ///< 当前子块的输入行偏移
        uint32_t outRowOffsetThisSubBlock = (qNBlockSize == 1U) ? inRowOffsetThisSubBlock : 0;  ///< 当前子块的输出行偏移
        uint32_t outColOffsetThisSubBlock = (qNBlockSize == 1U) ? 0 : subBlockIdx * qNSplitSubBlock * embed;  ///< 当前子块的输出列偏移
        uint32_t qSThisSubBlock = (qNBlockSize == 1U) ? inRowActualThisSubBlock : qSBlockSize;  ///< 当前子块的查询序列长度
        int64_t outOffsetSubBlock =  ///< 当前子块的输出偏移
            layoutOutput.GetOffset(MatrixCoord(outRowOffsetThisSubBlock, outColOffsetThisSubBlock));

        // 计算LSE相关偏移
        uint32_t outLseRowOffsetThisSubBlock = (qNBlockSize == 1U) ? inRowOffsetThisSubBlock : 0;  ///< 当前子块的LSE行偏移
        uint32_t outLseColOffsetThisSubBlock = (qNBlockSize == 1U) ? 0 : subBlockIdx * qNSplitSubBlock;  ///< 当前子块的LSE列偏移
        int64_t offsetLse =  ///< 当前子块的LSE偏移
            layoutLse.GetOffset(MatrixCoord(outLseRowOffsetThisSubBlock, outLseColOffsetThisSubBlock));
        auto gLseThisSubBlock = gLse[offsetLse];  ///< 当前子块的LSE张量
        auto layoutOutLseThisSubBlock = layoutLse;  ///< 当前子块的LSE布局

        // 如果当前子块有实际数据需要处理
        if (inRowActualThisSubBlock > 0U) {
            uint32_t rowLoop = CeilDiv(inRowActualThisSubBlock, rowNumTile);  ///< 行循环次数
            uint32_t needRowLoop = (rowLoop > 1U) ? 1 : 0;  ///< 是否需要行循环

            // 每轮循环的行由多个头组成，包含几个完整头、一个前导头和一个尾随头
            uint32_t proTokenIdx = 0;      ///< 前导部分起始标记的索引
            uint32_t proTokenNum = 0;      ///< 前导部分的标记数量
            uint32_t epiTokenNum = 0;      ///< 尾随部分的标记数量
            uint32_t integralHeadNum = 0;  ///< 每轮循环内的完整头数量

            // 行循环处理
            for (uint32_t rowLoopIdx = 0; rowLoopIdx < rowLoop; rowLoopIdx++) {
                uint32_t rowOffsetLoop = rowLoopIdx * rowNumTile;  ///< 行循环偏移
                uint32_t rowOffsetCurLoop = inRowOffsetThisSubBlock + rowOffsetLoop;  ///< 当前循环的行偏移
                uint32_t rowActualCurLoop =  ///< 当前循环的实际行数
                    (rowLoopIdx == (rowLoop - 1U)) ? inRowActualThisSubBlock - rowLoopIdx * rowNumTile : rowNumTile;

                // 计算当前循环的输出偏移和张量
                int64_t offsetOutput = 
                    static_cast<int64_t>(rowLoopIdx * rowNumTile / qSThisSubBlock * embed) + outOffsetSubBlock;
                auto gOutputCurLoop = gOutput[offsetOutput];  ///< 当前循环的输出张量
                auto layoutOutputCurLoop = layoutOutput;  ///< 当前循环的输出布局

                // 计算当前循环的输入偏移和张量
                int64_t offsetInput = layoutInput.GetOffset(MatrixCoord(rowOffsetCurLoop, 0));
                auto gInputCurLoop = gInput[offsetInput];  ///< 当前循环的输入张量
                auto layoutInputCurLoop = layoutInput.GetTileLayout(MatrixCoord(rowActualCurLoop, embed));  ///< 当前循环的输入布局

                // 计算当前循环的更新偏移和张量
                int64_t offsetUpdate = layoutUpdate.GetOffset(MatrixCoord(rowOffsetCurLoop, 0));
                auto gUpdateCurLoop = gUpdate[offsetUpdate];  ///< 当前循环的更新张量
                auto layoutUpdateCurLoop = layoutUpdate.GetTileLayout(MatrixCoord(rowActualCurLoop, embed));  ///< 当前循环的更新布局

                // 计算当前循环的前导、完整和尾随标记数量
                proTokenIdx = rowOffsetLoop % qSThisSubBlock;
                proTokenNum = AscendC::Std::min(rowActualCurLoop, (qSThisSubBlock - proTokenIdx)) % qSThisSubBlock;
                integralHeadNum = (rowActualCurLoop - proTokenNum) / qSThisSubBlock;
                epiTokenNum = rowActualCurLoop - proTokenNum - integralHeadNum * qSThisSubBlock;

                // 调用子核计算函数处理当前循环的数据
                SubCoreCompute(
                    gOutputCurLoop,       // 当前循环的输出张量
                    gInputCurLoop,        // 当前循环的输入张量
                    gUpdateCurLoop,       // 当前循环的更新张量
                    gLseThisSubBlock,     // 当前子块的LSE张量
                    layoutOutputCurLoop,  // 当前循环的输出布局
                    layoutInputCurLoop,   // 当前循环的输入布局
                    layoutUpdateCurLoop,  // 当前循环的更新布局
                    layoutOutLseThisSubBlock,  // 当前子块的LSE布局
                    qNThisSubBlock,       // 当前子块的查询头数
                    qSThisSubBlock,       // 当前子块的查询序列长度
                    inRowActualThisSubBlock,  // 当前子块的实际输入行数
                    isFirstStackTile,     // 是否为第一个堆叠瓦片
                    isLastStackTile,      // 是否为最后一个堆叠瓦片
                    curStackTileMod,      // 当前堆叠瓦片模块
                    needRowLoop,          // 是否需要行循环
                    (rowLoopIdx == rowLoop - 1U),  // 是否为最后一个行循环
                    rowOffsetLoop,        // 行循环偏移
                    proTokenIdx,          // 前导标记索引
                    proTokenNum,          // 前导标记数量
                    epiTokenNum,          // 尾随标记数量
                    integralHeadNum);     // 完整头数量
            }
        }
    }

private:
    // UB缓冲区成员变量
    AscendC::LocalTensor<float> loUbTensor;             ///< 本地输出张量UB缓冲区
    AscendC::LocalTensor<float> dmUbTensor;             ///< 数据矩阵张量UB缓冲区
    AscendC::LocalTensor<float> hmUbTensor;             ///< 本地矩阵M UB缓冲区
    AscendC::LocalTensor<float> glUbTensor;             ///< 全局LSE张量UB缓冲区
    AscendC::LocalTensor<float> tvUbTensor;             ///< 临时向量UB张量
    AscendC::LocalTensor<ElementOutput> goUbTensor16;   ///< 全局输出16位UB张量
    AscendC::LocalTensor<float> goUbTensor32;           ///< 全局输出32位UB张量
    AscendC::LocalTensor<float> gmUbTensor;             ///< 全局矩阵M UB缓冲区
    AscendC::LocalTensor<float> lse32_ubuf_tensor;      ///< LSE 32位UB张量
};

}

#endif
