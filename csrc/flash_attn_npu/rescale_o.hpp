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

/**
 * @file rescale_o.hpp
 * @brief Atlas A2平台的单精度(float)输出重缩放块级Epilogue实现
 *
 * 本文件实现了Flash Attention推理中针对Atlas A2平台优化的单精度输出重缩放块级Epilogue，
 * 是Flash Attention计算流水线的最后阶段，负责将PV矩阵乘法的输出转换为最终的注意力输出。
 *
 * == 主要实现的算法 ==
 * 实现了输出重缩放（Rescale O）算法，核心操作包括：
 *   1. 累加修正: 对于非第一个stack tile，将历史输出O乘以修正因子dm后加上当前块输出
 *      - O = O * dm + O_current （dm = exp(old_max - new_max)，来自OnlineSoftmax）
 *   2. 归一化: 在最后一个stack tile，将累加输出除以全局行求和gl
 *      - O = O / gl （gl是OnlineSoftmax维护的全局指数求和）
 *   3. 精度转换: 将float精度的O转换为输出精度（half/bfloat16）
 *   4. LSE计算: 可选地计算并输出对数求和指数（Log-Sum-Exp）
 *      - LSE = ln(gl) + gm （gm是OnlineSoftmax维护的全局最大值）
 *
 * == 数据流 ==
 *   PV输出(O_tmp) -> GM -> UB(lo) \
 *                                   -> UB(go) -> 降精度 -> GM(O)
 *   历史O(Update) -> GM -> UB(go) /
 *   修正因子(dm)  -> UB(tv) -> 广播 -> 乘法
 *   全局求和(gl)  -> UB(tv) -> 广播 -> 除法
 *
 * == 与rescale_o_low_prec.hpp的区别 ==
 * - 本文件使用float精度进行中间计算，数值稳定性更好
 * - low_prec版本使用half精度，性能更高但精度略低
 * - 本文件需要额外的float->half降精度转换步骤
 *
 * == 依赖关系 ==
 * - catlass/catlass.hpp: Catlass库核心头文件
 * - catlass/arch/resource.hpp: 硬件资源管理
 * - catlass/epilogue/dispatch_policy.hpp: Epilogue调度策略
 * - catlass/epilogue/tile/tile_copy.hpp: 分块数据拷贝操作
 * - catlass/gemm_coord.hpp: 矩阵坐标系统
 * - catlass/matrix_coord.hpp: 矩阵坐标辅助工具
 * - fa_block.h: Flash Attention分块参数定义
 *
 * == 使用场景 ==
 * 本文件用于Flash Attention推理场景的输出后处理阶段。
 * 典型调用路径: mha_fwd_kvcache.cpp -> FAInferKernel -> EpilogueRescaleO -> 本文件
 */

// Catlass库核心头文件
#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"

// 自定义块结构头文件
#include "fa_block.h"

/**
 * @namespace Catlass::Epilogue::Block
 * @brief Catlass库中Epilogue（收尾操作）的块级实现命名空间
 *
 * 该命名空间包含Flash Attention计算流水线中矩阵乘法后的各种后处理操作的块级实现，
 * 包括在线Softmax计算和输出重缩放。
 */
namespace Catlass::Epilogue::Block {
/**
 * @brief 单精度输出重缩放的块级Epilogue实现（RescaleO）
 *
 * 专门为Atlas A2平台优化的单精度输出重缩放Epilogue实现，用于Flash Attention推理的
 * 最后阶段。负责将PV矩阵乘法的输出O_tmp转换为最终的注意力输出O。
 *
 * == 设计思路 ==
 * 在Flash Attention的在线Softmax算法中，输出O的计算分为两步：
 * 1. 累加修正: 对于每个stack tile，O = O * dm + O_current
 *    - dm = exp(old_max - new_max) 是修正因子，来自OnlineSoftmax
 *    - O_current 是当前stack tile的PV乘法输出
 *    - O 是历史累加输出（第一个stack tile时O=0）
 * 2. 归一化: 在最后一个stack tile，O = O / gl
 *    - gl 是OnlineSoftmax维护的全局指数求和
 *
 * == 数据流 ==
 *   PV输出(O_tmp) -> GM -> UB(lo) \
 *                                   -> UB(go) -> 降精度 -> GM(O)
 *   历史O(Update) -> GM -> UB(go) /
 *   修正因子(dm)  -> UB(tv) -> Brcb广播 -> 乘法
 *   全局求和(gl)  -> UB(tv) -> Brcb广播 -> 除法
 *
 * == UB内存布局 ==
 * - loUbTensor: 当前stack tile的PV输出（float类型，从GM加载）
 * - goUbTensor: 全局输出（float类型，累加结果）
 * - tvUbTensor: 临时向量（float类型，用于广播dm/gl）
 * - dmUbTensor: 修正因子（float类型，来自OnlineSoftmax）
 * - glUbTensor: 全局行求和（float类型，来自OnlineSoftmax）
 *
 * == 与rescale_o_low_prec.hpp的区别 ==
 * - 本文件使用float精度进行中间计算，数值稳定性更好
 * - low_prec版本使用half精度，性能更高但精度略低
 * - 本文件需要额外的float->half降精度转换步骤
 *
 * @tparam OutputType_ 输出类型和布局，包含元素类型（通常为half/bfloat16）和矩阵布局
 * @tparam InputType_ 输入类型和布局，包含元素类型（float）和矩阵布局
 * @tparam UpdateType_ 更新类型和布局，包含元素类型（float）和矩阵布局
 *         - Update矩阵是历史累加输出O，从全局内存加载
 * @tparam LseType_ LSE类型和布局，包含元素类型（float）和矩阵布局
 * @tparam LSE_MODE_ LSE计算模式:
 *         - LseModeT::NONE: 不计算LSE
 *         - LseModeT::OUT_ONLY: 计算并输出LSE值（LSE = ln(gl) + gm）
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
    /**
     * @brief 单精度输出重缩放子核计算核心函数
     *
     * 执行输出重缩放的核心计算逻辑，处理一个子块的数据。根据stack tile的位置
     * （第一个/中间/最后一个）执行不同的计算路径。
     *
     * == 计算流程 ==
     * 路径1 - 非第一个stack tile (!isFirstStackTile):
     *   a) 从GM加载当前PV输出O_tmp到loUbTensor（float精度）
     *   b) 广播修正因子dm: Brcb将dm（每行1个值）广播为每行64个值存入tvUbTensor
     *   c) 从GM加载历史累加输出O到goUbTensor32（float精度）
     *   d) O = O * dm: 逐向量将历史输出乘以修正因子
     *   e) O = O + O_tmp: 加上当前PV输出
     *   合并: O = dm * O_old + O_tmp
     *
     * 路径2 - 第一个stack tile (isFirstStackTile):
     *   O = O_tmp: 直接将PV输出复制到goUbTensor32
     *
     * 路径3 - 最后一个stack tile (isLastStackTile):
     *   a) 广播全局行求和gl: Brcb将gl广播为每行64个值
     *   b) O = O / gl: 逐向量除法完成归一化
     *   c) 降精度: Cast将float的O转换为half/bfloat16
     *   d) 写回GM: CopyOToGm将归一化后的O写回全局内存
     *   e) 可选LSE计算: LSE = ln(gl) + gm
     *
     * == 性能优化技巧 ==
     * - Brcb广播: 将1D的dm/gl广播为2D矩阵，避免逐行复制
     * - 逐向量Mul/Div: srcBStride=0，每行复用同一个广播值，减少数据搬移
     * - 事件驱动的流水线: 通过HardEvent实现GM<->UB数据传输与向量计算的并行
     *   - EVENT_ID0: MTE2_V同步，确保数据加载完成后再计算
     *   - EVENT_ID3: V_MTE2同步，确保计算完成后再加载新数据
     *   - EVENT_ID6: MTE3_MTE2同步，确保上次写回GM完成后再加载新数据
     * - 非最后一个stack tile时，将goUbTensor32写回GM的gUpdate供下一个stack tile使用
     * - 最后一个stack tile才执行降精度和写回最终输出，减少中间转换开销
     *
     * @param gOutput           全局输出张量
     * @param gInput            全局输入张量（PV输出O_tmp）
     * @param gUpdate           全局更新张量（历史累加输出O）
     * @param gLse              全局LSE张量
     * @param layoutOutput      输出布局
     * @param layoutInput       输入布局
     * @param layoutUpdate      更新布局
     * @param layoutLse         LSE布局
     * @param qNThisSubBlock    当前子块的查询头数
     * @param qSThisSubBlock    当前子块的查询序列长度
     * @param totalRowNum       总行数
     * @param isFirstStackTile  是否为第一个stack tile
     * @param isLastStackTile   是否为最后一个stack tile
     * @param curStackTileMod   当前stack tile的模值
     * @param needRowLoop       是否需要行循环
     * @param isLastRowLoop     是否为最后一个行循环
     * @param rowOffsetLoop     行循环偏移
     * @param proTokenIdx       前导标记索引
     * @param proTokenNum       前导标记数量
     * @param epiTokenNum       尾随标记数量
     * @param integralHeadNum   整数头部数量
     */
    __aicore__ inline
    void SubCoreCompute(
        AscendC::GlobalTensor<ElementOutput> gOutput,
        AscendC::GlobalTensor<ElementInput> gInput,
        AscendC::GlobalTensor<ElementUpdate> gUpdate,
        AscendC::GlobalTensor<ElementLse> gLse,
        const LayoutOutput &layoutOutput,
        const LayoutInput &layoutInput,
        const LayoutUpdate &layoutUpdate,
        const LayoutLse &layoutLse,
        uint32_t qNThisSubBlock,
        uint32_t qSThisSubBlock,
        uint32_t totalRowNum,
        uint32_t isFirstStackTile,
        uint32_t isLastStackTile,
        uint32_t curStackTileMod,
        uint32_t needRowLoop,
        uint32_t isLastRowLoop,
        uint32_t rowOffsetLoop,
        uint32_t proTokenIdx,
        uint32_t proTokenNum,
        uint32_t epiTokenNum,
        uint32_t integralHeadNum)
    {
        // 计算各种尺寸参数
        uint32_t curRowNum = layoutInput.shape(0);
        uint32_t embed = layoutInput.shape(1);
        uint32_t embedRound = layoutInput.stride(0);         // 嵌入步长（对齐后，可能大于embed）
        uint32_t curRowNumRound = RoundUp(curRowNum, FLOAT_BLOCK_SIZE);
        uint32_t qSBlockSize = layoutOutput.shape(0);
        uint32_t oHiddenSize = layoutOutput.shape(1);
        uint32_t qHeads = layoutLse.shape(1);
        // dm在UB中的偏移：根据stack tile索引和行偏移计算
        uint32_t dmUbOffsetCurStackTile = curStackTileMod * MAX_ROW_NUM_SUB_CORE + rowOffsetLoop;

        // ========== 路径1: 非第一个stack tile - 累加修正 ==========
        if (!isFirstStackTile) {
            // 步骤1a: 从GM加载当前PV输出O_tmp到loUbTensor
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
            AscendC::DataCopy(
                loUbTensor, gInput, AscendC::DataCopyParams(1, curRowNum * embedRound / FLOAT_BLOCK_SIZE, 0, 0));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        }
        // 等待上次GM写回完成（确保gUpdate/gInput的数据已写入）
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
        if (!isFirstStackTile) {
            // 步骤1b: 广播修正因子dm到tvUbTensor
            // Brcb将每行的dm值复制到8个float的位置，供后续逐向量Mul使用
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            AscendC::Brcb(tvUbTensor.ReinterpretCast<uint32_t>(),
                dmUbTensor[dmUbOffsetCurStackTile].ReinterpretCast<uint32_t>(),
                curRowNumRound / FLOAT_BLOCK_SIZE,
                AscendC::BrcbRepeatParams(1, 8));
            AscendC::PipeBarrier<PIPE_V>();
            // 步骤1c: 从GM加载历史累加输出O到goUbTensor32
            if (needRowLoop) {
                AscendC::DataCopy(
                    goUbTensor32, gUpdate,
                    AscendC::DataCopyParams(1, curRowNum * embedRound / FLOAT_BLOCK_SIZE, 0, 0));
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
            }
            // 步骤1d: O = O * dm（逐向量乘法）
            // 按FLOAT_VECTOR_SIZE(64)为粒度循环处理每一列块
            // srcBStride=1块表示tvUbTensor每行步进8个float（一个广播的dm值）
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
            // 处理尾部非对齐元素
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
            // 步骤1e: O = O + O_tmp（加上当前PV输出）
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
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
            // ========== 路径2: 第一个stack tile - 直接复制 ==========
            // O = O_tmp: 直接将PV输出加载到goUbTensor32
            AscendC::DataCopy(
                goUbTensor32, gInput, AscendC::DataCopyParams(1, curRowNum * embedRound / FLOAT_BLOCK_SIZE, 0, 0));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        }

        // ========== 路径3: 最后一个stack tile - 归一化和输出 ==========
        if (isLastStackTile) {
            // 步骤3a: 广播全局行求和gl到tvUbTensor
            AscendC::Brcb(
                tvUbTensor.ReinterpretCast<uint32_t>(),
                glUbTensor.ReinterpretCast<uint32_t>()[rowOffsetLoop],
                curRowNumRound / FLOAT_BLOCK_SIZE,
                AscendC::BrcbRepeatParams(1, 8));
            AscendC::PipeBarrier<PIPE_V>();
            // 步骤3b: O = O / gl（逐向量除法，完成Softmax归一化）
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
            // 处理尾部非对齐元素
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

            // 步骤3c: 降精度 float -> half/bfloat16
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

            // 步骤3d: 将归一化后的O写回全局内存
            CopyOToGm(
                gOutput, proTokenIdx, proTokenNum, epiTokenNum, integralHeadNum, qSThisSubBlock, embed, oHiddenSize);
            // 步骤3e: 可选LSE计算: LSE = ln(gl) + gm
            if constexpr (LSE_MODE_ == LseModeT::OUT_ONLY) {
                if (isLastRowLoop) {
                    AscendC::PipeBarrier<PIPE_V>();
                    // lse = ln(gl): 计算全局行求和的自然对数
                    AscendC::Ln<float, false>(
                        lse32_ubuf_tensor,
                        glUbTensor,
                        (uint64_t)0, CeilDiv(totalRowNum, FLOAT_VECTOR_SIZE),
                        AscendC::UnaryRepeatParams(1, 1, 8, 8));

                    AscendC::PipeBarrier<PIPE_V>();
                    // lse = lse + gm: 加上全局行最大值
                    AscendC::Add<float, false>(
                        lse32_ubuf_tensor,
                        lse32_ubuf_tensor,
                        gmUbTensor,
                        (uint64_t)0, CeilDiv(totalRowNum, FLOAT_VECTOR_SIZE),
                        AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
                    AscendC::PipeBarrier<PIPE_V>();

                    // 广播LSE到tvUbTensor，然后写回GM
                    AscendC::Brcb(
                        tvUbTensor.ReinterpretCast<uint32_t>(),
                        lse32_ubuf_tensor.ReinterpretCast<uint32_t>(),
                        CeilDiv(totalRowNum, FLOAT_BLOCK_SIZE),
                        AscendC::BrcbRepeatParams(1, 8));
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID4);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID4);
                    
                    // 将LSE写回全局内存，按GQA分组布局
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
            // 非最后一个stack tile: 将累加结果O写回GM供下一个stack tile使用
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID5);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID5);
            AscendC::DataCopy(
                gUpdate, goUbTensor32, AscendC::DataCopyParams(1, curRowNum * embedRound / FLOAT_BLOCK_SIZE, 0, 0));
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
    }

    /**
     * @brief 输出重缩放核心操作符
     *
     * 执行输出重缩放算法的核心计算流程，将PV矩阵乘法的输出O_tmp转换为最终注意力输出O。
     * 该函数是RescaleO模块的入口点，由FAInferKernel在每个stack tile上调用。
     *
     * == 算法流程 ==
     * 对于每个stack tile，执行以下步骤：
     * 1. 分块处理: 将行数按UB容量分为多个tile，每个tile处理一部分行
     * 2. 子块并行: 将每个tile的行数分为两个子块，利用双子核并行处理
     * 3. 核心计算（SubCoreCompute）:
     *    a) 从GM加载当前PV输出O_tmp到UB（loUbTensor）
     *    b) 如果不是第一个stack tile:
     *       - 从GM加载历史累加输出O到UB（goUbTensor）
     *       - 广播修正因子dm，执行 O = O * dm + O_tmp
     *    c) 如果是第一个stack tile:
     *       - 直接将O_tmp复制到goUbTensor
     *    d) 如果是最后一个stack tile:
     *       - 广播全局行求和gl，执行 O = O / gl（归一化）
     *       - 可选计算LSE = ln(gl) + gm
     *    e) 将goUbTensor降精度后写回GM
     *
     * == 子块并行策略 ==
     * 当subBlockNum > 1时，将行数分为两个子块并行处理：
     * - 子块0处理前半部分行
     * - 子块1处理后半部分行
     *
     * @param gOutput           全局输出张量（最终注意力输出O，half/bfloat16精度）
     * @param gInput            全局输入张量（PV矩阵乘法的输出O_tmp，float精度）
     * @param gUpdate           全局更新张量（历史累加输出O，float精度）
     * @param gLse              全局LSE张量（对数求和指数，float精度）
     * @param layoutOutput      输出布局信息
     * @param layoutInput       输入布局信息
     * @param layoutUpdate      更新布局信息
     * @param layoutLse         LSE布局信息
     * @param actualBlockShape  实际块形状（M/N维度大小）
     * @param qSBlockSize       Q的S维度块大小（分组注意力头的组数）
     * @param qNBlockSize       Q的N维度块大小（每组内的token数）
     * @param isFirstStackTile  是否是第一个stack tile（1=是，0=否）
     * @param isLastStackTile   是否是最后一个stack tile（1=是，0=否）
     * @param curStackTileMod   当前stack tile的模值
     *
     * @note 第一个stack tile不需要计算修正因子dm（因为没有历史结果需要调整）
     * @note 最后一个stack tile需要执行归一化 O = O / gl
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
