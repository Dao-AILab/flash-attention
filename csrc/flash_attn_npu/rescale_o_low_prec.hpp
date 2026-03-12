
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

#ifndef EPILOGUE_BLOCK_BLOCK_EPILOGUE_RESCALE_LOW_PREC_O_HPP_T
#define EPILOGUE_BLOCK_BLOCK_EPILOGUE_RESCALE_LOW_PREC_O_HPP_T

/**
 * @file rescale_o_low_prec.hpp
 * @brief Atlas A2平台的低精度输出重缩放块级Epilogue实现
 *
 * 该文件实现了Flash Attention中针对Atlas A2平台优化的低精度输出重缩放块级Epilogue，
 * 主要负责对注意力计算的最终结果进行后处理，包括：
 * 1. 对PV矩阵乘法(Attn*V)的输出进行缩放
 * 2. 应用LSE(对数求和指数)进行数值稳定性处理
 * 3. 转换到目标输出精度并写入全局内存
 *
 * 该实现使用half精度进行中间计算，以提高性能并减少内存带宽需求，同时保持足够的计算精度。
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

namespace Catlass::Epilogue::Block {
    /**
     * @brief 低精度输出重缩放的块级Epilogue实现
     *
     * 专门为Atlas A2平台优化，使用half精度进行中间计算，以提高Flash Attention的性能和效率。
     * 该类实现了Flash Attention的最后阶段，负责将注意力权重与值矩阵的乘积转换为最终输出。
     */

template <
    class OutputType_,     ///< 输出类型和布局
    class InputType_,      ///< 输入类型和布局（通常是PV矩阵乘法的输出）
    class UpdateType_,     ///< 更新类型和布局（用于累加计算）
    class LseType_,        ///< LSE（对数求和指数）类型和布局
    LseModeT LSE_MODE_>    ///< LSE计算模式
class BlockEpilogue<
    EpilogueAtlasA2RescaleOT<LSE_MODE_, half>,
    OutputType_,
    InputType_,
    UpdateType_,
    LseType_>
{
public:
    // 类型别名
    using DispatchPolicy = EpilogueAtlasA2RescaleOT<LSE_MODE_, half>;  ///< 调度策略类型，定义了Atlas A2上的Epilogue行为
    using ArchTag = typename DispatchPolicy::ArchTag;                   ///< 架构标签类型，标识目标硬件架构

    // 元素类型
    using ElementOutput = typename OutputType_::Element;                ///< 输出元素类型（最终输出的数据类型）
    using ElementInput = typename InputType_::Element;                  ///< 输入元素类型（PV矩阵乘法的输出类型）
    using ElementUpdate = typename UpdateType_::Element;                ///< 更新元素类型（用于累加计算的数据类型）
    using ElementLse = typename LseType_::Element;                      ///< LSE元素类型（对数求和指数的数据类型）

    // 布局类型
    using LayoutOutput = typename OutputType_::Layout;                  ///< 输出布局类型（如NHWC、NCHW等）
    using LayoutInput = typename InputType_::Layout;                    ///< 输入布局类型
    using LayoutUpdate = typename UpdateType_::Layout;                  ///< 更新布局类型
    using LayoutLse = typename LseType_::Layout;                        ///< LSE布局类型

    // 静态常量
    static constexpr LseModeT LSE_MODE = DispatchPolicy::LSE_MODE;      ///< LSE计算模式（0: 无LSE, 1: 有LSE）

    // 元素数量和块大小常量（针对Atlas A2架构优化）
    static constexpr uint32_t HALF_ELENUM_PER_BLK = 16;                 ///< 每个块的半精度元素数量
    static constexpr uint32_t BLOCK_SIZE = 16;                          ///< 基本块大小（适配硬件计算单元）
    static constexpr uint32_t HALF_ELENUM_PER_VECCALC = 128;            ///< 每个向量计算的半精度元素数量
    static constexpr uint32_t FLOAT_ELENUM_PER_VECCALC = 64;            ///< 每个向量计算的浮点数元素数量
    static constexpr uint32_t HALF_ELENUM_PER_LINE = 256;               ///< 每行的半精度元素数量
    static constexpr uint32_t FLOAT_ELENUM_PER_LINE = 128;              ///< 每行的浮点数元素数量
    static constexpr uint32_t MULTIPLIER = 2;                           ///< 乘法因子（用于类型转换）
    static constexpr uint32_t FLOAT_BLOCK_SIZE = 8;                     ///< 浮点数块大小
    static constexpr uint32_t HALF_BLOCK_SIZE = 16;                     ///< 半精度块大小
    static constexpr uint32_t FLOAT_VECTOR_SIZE = 64;                   ///< 浮点数向量大小
    static constexpr uint32_t HALF_VECTOR_SIZE = 128;                   ///< 半精度向量大小
    static constexpr uint32_t UB_UINT8_VECTOR_SIZE = 1024;              ///< UB中uint8向量大小（字节）
    static constexpr uint32_t UB_UINT8_BLOCK_SIZE = 16384;              ///< UB中uint8块大小（字节）
    static constexpr uint32_t HALF_DM_UB_SIZE = 64;                     ///< 半精度DM UB大小
    static constexpr uint32_t HALF_LL_UB_SIZE = 256;                    ///< 半精度LL UB大小
    static constexpr uint32_t VECTOR_SIZE = 128;                        ///< 通用向量大小
    static constexpr uint32_t NUM4 = 4;                                 ///< 数字4常量（用于循环控制）
    static constexpr uint32_t MAX_UB_O_ELEM_NUM = 8192;                 ///< UB中O元素的最大数量（限制内存使用）
    static constexpr uint32_t MAX_ROW_NUM_SUB_CORE = 256;               ///< 子核最大行数（硬件限制）
    static constexpr uint32_t SIZE_OF_16BIT = 2;                        ///< 16位数据大小（字节）

    // UB缓冲区成员变量
    half *loUbTensor;              ///< 本地输出张量UB缓冲区
    half *dmUbTensor;              ///< 数据矩阵张量UB缓冲区
    half *glUbTensor;              ///< 全局LSE张量UB缓冲区
    half *tvUbTensor;              ///< TV张量UB缓冲区（半精度）
    float *tvUbTensor32;           ///< TV张量UB缓冲区（单精度）
    ElementOutput *goUbTensor;     ///< 全局输出张量UB缓冲区
    half *hmUbTensor;              ///< 本地矩阵张量UB缓冲区
    half *gmUbTensor;              ///< 全局矩阵张量UB缓冲区
    half *lse16_ubuf_tensor;       ///< 16位LSE张量UB缓冲区
    float *lse32_ubuf_tensor;      ///< 32位LSE张量UB缓冲区

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
        // 偏移量设置遵循以下策略：
        // 1. 大张量（LO/GO/TV）使用UB_UINT8_BLOCK_SIZE(16KB)的整数倍对齐，提高访问效率
        // 2. 小张量（HM/GM/LSE/GL/DM）使用UB_UINT8_VECTOR_SIZE(1KB)精细分配
        // 3. 部分张量共享起始地址（如GM和LSE32，GL和LSE16），因为它们在不同计算阶段使用，不会同时占用内存
        constexpr uint32_t LO_UB_TENSOR_OFFSET = 6 * UB_UINT8_BLOCK_SIZE;                          ///< LO张量在UB中的起始偏移（96KB）
        constexpr uint32_t GO_UB_TENSOR_OFFSET = 8 * UB_UINT8_BLOCK_SIZE;                          ///< GO张量在UB中的起始偏移（128KB）
        constexpr uint32_t TV_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE;                         ///< TV张量在UB中的起始偏移（160KB）
        constexpr uint32_t HM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 9 * UB_UINT8_VECTOR_SIZE;  ///< HM张量在UB中的起始偏移（169KB）
        constexpr uint32_t GM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 10 * UB_UINT8_VECTOR_SIZE; ///< GM张量在UB中的起始偏移（170KB）
        constexpr uint32_t LSE32_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 10 * UB_UINT8_VECTOR_SIZE; ///< 32位LSE张量在UB中的起始偏移（170KB，与GM共享）
        constexpr uint32_t GL_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 12 * UB_UINT8_VECTOR_SIZE;   ///< GL张量在UB中的起始偏移（172KB）
        constexpr uint32_t LSE16_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 12 * UB_UINT8_VECTOR_SIZE; ///< 16位LSE张量在UB中的起始偏移（172KB，与GL共享）
        constexpr uint32_t DM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 13 * UB_UINT8_VECTOR_SIZE;   ///< DM张量在UB中的起始偏移（173KB）

        // 初始化各种UB张量缓冲区，将不同类型的指针指向UB中的特定偏移位置
        loUbTensor = resource.ubBuf.template GetBufferByByte<half>(LO_UB_TENSOR_OFFSET);          ///< 分配LO张量UB缓冲区
        dmUbTensor = resource.ubBuf.template GetBufferByByte<half>(DM_UB_TENSOR_OFFSET);          ///< 分配DM张量UB缓冲区
        glUbTensor = resource.ubBuf.template GetBufferByByte<half>(GL_UB_TENSOR_OFFSET);          ///< 分配GL张量UB缓冲区
        tvUbTensor = resource.ubBuf.template GetBufferByByte<half>(TV_UB_TENSOR_OFFSET);          ///< 分配TV张量UB缓冲区（半精度）
        tvUbTensor32 = resource.ubBuf.template GetBufferByByte<float>(TV_UB_TENSOR_OFFSET);       ///< 分配TV张量UB缓冲区（单精度）
        goUbTensor = resource.ubBuf.template GetBufferByByte<ElementOutput>(GO_UB_TENSOR_OFFSET); ///< 分配GO张量UB缓冲区
        hmUbTensor = resource.ubBuf.template GetBufferByByte<half>(HM_UB_TENSOR_OFFSET);          ///< 分配HM张量UB缓冲区
        gmUbTensor = resource.ubBuf.template GetBufferByByte<half>(GM_UB_TENSOR_OFFSET);          ///< 分配GM张量UB缓冲区
        lse16_ubuf_tensor = resource.ubBuf.template GetBufferByByte<half>(LSE16_UB_TENSOR_OFFSET); ///< 分配16位LSE张量UB缓冲区
        lse32_ubuf_tensor = resource.ubBuf.template GetBufferByByte<float>(LSE32_UB_TENSOR_OFFSET); ///< 分配32位LSE张量UB缓冲区
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
        const int32_t MAX_MASK_LEN = 128;  ///< 最大掩码长度
        const int32_t HALF_MASK_LEN = 64;   ///< 半掩码长度（64位）
        
        // 如果有效长度大于等于最大掩码长度，设置全1掩码（所有元素都有效）
        if (len >= MAX_MASK_LEN) {
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            return;
        }
        
        // 计算高32位和低32位的有效长度
        int32_t highMask = len - HALF_MASK_LEN > 0 ? len - HALF_MASK_LEN : 0;
        int32_t lowMask = len - HALF_MASK_LEN >= 0 ? HALF_MASK_LEN : len;
        
        // 根据有效长度设置不同的掩码
        if (len < HALF_MASK_LEN) {
            // 有效长度小于64，只设置低32位掩码
            AscendC::SetVectorMask<int8_t>(0x0, ((uint64_t)1 << lowMask) - 1);
        } else {
            // 有效长度大于等于64，设置高32位和低32位掩码
            AscendC::SetVectorMask<int8_t>(((uint64_t)1 << highMask) - 1, 0xffffffffffffffff);
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
        uint32_t innerOGmOffset = 0;   ///< 全局内存输出偏移
        uint32_t innerGOUbOffset = 0;  ///< UB输出偏移
        
        // 复制前导token的输出数据（如果有）
        if (proTokenNum != 0U) {
            AscendC::DataCopyPad(
                gOutput[innerOGmOffset + proTokenIdx * oHiddenSize],  ///< 目标位置：全局内存
                goUbTensor[innerGOUbOffset],                           ///< 源位置：UB
                AscendC::DataCopyExtParams(
                    proTokenNum, embed * SIZE_OF_16BIT, 0, (oHiddenSize - embed) * SIZE_OF_16BIT, 0));  ///< 复制参数
            innerOGmOffset += embed;              ///< 更新全局内存偏移
            innerGOUbOffset += proTokenNum * embed;  ///< 更新UB偏移
        }
        
        // 复制完整头的输出数据
        for (uint32_t qN_idx = 0; qN_idx < integralHeadNum; qN_idx++) {
            AscendC::DataCopyPad(
                gOutput[innerOGmOffset],                               ///< 目标位置：全局内存
                goUbTensor[innerGOUbOffset],                           ///< 源位置：UB
                AscendC::DataCopyExtParams(
                    qSThisSubBlock, embed * SIZE_OF_16BIT, 0, (oHiddenSize - embed) * SIZE_OF_16BIT, 0));  ///< 复制参数
            innerOGmOffset += embed;              ///< 更新全局内存偏移
            innerGOUbOffset += qSThisSubBlock * embed;  ///< 更新UB偏移
        }
        
        // 复制尾随token的输出数据（如果有）
        if (epiTokenNum != 0U) {
            AscendC::DataCopyPad(
                gOutput[innerOGmOffset],                               ///< 目标位置：全局内存
                goUbTensor[innerGOUbOffset],                           ///< 源位置：UB
                AscendC::DataCopyExtParams(
                    epiTokenNum, embed * SIZE_OF_16BIT, 0, (oHiddenSize - embed) * SIZE_OF_16BIT, 0));  ///< 复制参数
        }
    }

    /**
     * @brief 子核计算核心函数，执行输出重缩放操作
     *
     * 这是Epilogue的核心计算函数，负责执行输出重缩放、LSE处理和结果转换等操作。
     * 具体实现会根据LSE模式和数据类型进行不同的处理流程，包括：
     * 1. 数据从全局内存加载到UB缓冲区
     * 2. 执行缩放、加法、除法等重缩放操作
     * 3. 处理LSE（对数求和指数）
     * 4. 将结果从UB缓冲区复制到全局内存
     *
     * @param gOutput 全局内存输出张量
     * @param gInput 全局内存输入张量（PV矩阵乘法的输出）
     * @param gUpdate 全局内存更新张量（用于累加计算）
     * @param gLse 全局内存LSE张量
     * @param layoutOutput 输出布局
     * @param layoutInput 输入布局
     * @param layoutUpdate 更新布局
     * @param layoutLse LSE布局
     * @param qNThisSubBlock 当前子块的查询头数量
     * @param qSThisSubBlock 当前子块的查询序列长度
     * @param totalRowNum 总行数
     * @param isFirstStackTile 是否为第一个堆叠tile
     * @param isLastStackTile 是否为最后一个堆叠tile
     * @param curStackTileMod 当前堆叠tile的模
     * @param needRowLoop 是否需要行循环
     * @param isLastRowLoop 是否为最后一个行循环
     * @param rowOffsetLoop 行循环偏移
     * @param proTokenIdx 前导token的索引
     * @param proTokenNum 前导token的数量
     * @param epiTokenNum 尾随token的数量
     * @param integralHeadNum 完整头的数量
     */
    __aicore__ inline
    void SubCoreCompute(
        AscendC::GlobalTensor<ElementOutput> gOutput,      ///< 全局内存输出张量
        AscendC::GlobalTensor<ElementInput> gInput,        ///< 全局内存输入张量
        AscendC::GlobalTensor<ElementUpdate> gUpdate,      ///< 全局内存更新张量
        AscendC::GlobalTensor<ElementLse> gLse,            ///< 全局内存LSE张量
        const LayoutOutput &layoutOutput,                  ///< 输出布局
        const LayoutInput &layoutInput,                    ///< 输入布局
        const LayoutUpdate &layoutUpdate,                  ///< 更新布局
        const LayoutLse &layoutLse,                        ///< LSE布局
        uint32_t qNThisSubBlock,                           ///< 当前子块的查询头数量
        uint32_t qSThisSubBlock,                           ///< 当前子块的查询序列长度
        uint32_t totalRowNum,                              ///< 总行数
        uint32_t isFirstStackTile,                         ///< 是否为第一个堆叠tile
        uint32_t isLastStackTile,                          ///< 是否为最后一个堆叠tile
        uint32_t curStackTileMod,                          ///< 当前堆叠tile的模
        uint32_t needRowLoop,                              ///< 是否需要行循环
        uint32_t isLastRowLoop,                            ///< 是否为最后一个行循环
        uint32_t rowOffsetLoop,                            ///< 行循环偏移
        uint32_t proTokenIdx,                              ///< 前导token的索引
        uint32_t proTokenNum,                              ///< 前导token的数量
        uint32_t epiTokenNum,                              ///< 尾随token的数量
        uint32_t integralHeadNum)                          ///< 完整头的数量
    {
        // 获取当前计算块的参数
        uint32_t curRowNum = layoutInput.shape(0);         ///< 当前行数
        uint32_t embed = layoutInput.shape(1);             ///< 嵌入维度
        uint32_t embedRound = layoutInput.stride(0);       ///< 嵌入维度对齐后的步长
        uint32_t curRowNumRound = RoundUp(curRowNum, HALF_BLOCK_SIZE);  ///< 当前行数对齐到半精度块大小
        uint32_t qSBlockSize = layoutOutput.shape(0);      ///< 查询序列块大小
        uint32_t oHiddenSize = layoutOutput.shape(1);      ///< 输出隐藏层大小
        uint32_t qHeads = layoutLse.shape(1);              ///< 查询头数量
        uint32_t dmUbOffsetCurStackTile = curStackTileMod * MAX_ROW_NUM_SUB_CORE + rowOffsetLoop;  ///< DM UB当前堆叠tile偏移

        // 非第一个堆叠tile的处理逻辑
        if (!isFirstStackTile) {
            // 等待前一个tile的计算完成
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
            
            // 将输入数据从全局内存复制到LO UB张量
            AscendC::DataCopy(
                loUbTensor, gInput, AscendC::DataCopyParams(1, curRowNum * embedRound / HALF_BLOCK_SIZE, 0, 0));
            
            // 设置事件标记，通知输入数据已准备好
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        }
        
        // 等待MTE3和MTE2管道的完成
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
        
        // 非第一个堆叠tile的计算逻辑
        if (!isFirstStackTile) {
            // 设置全1掩码（所有元素都有效）
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            
            // 将DM UB张量广播到TV UB张量（用于后续乘法操作）
            AscendC::Brcb(
                tvUbTensor.ReinterpretCast<uint16_t>(),
                dmUbTensor[dmUbOffsetCurStackTile].ReinterpretCast<uint16_t>(),
                curRowNumRound / FLOAT_BLOCK_SIZE,
                AscendC::BrcbRepeatParams(1, 8));
            
            // 等待V管道操作完成
            AscendC::PipeBarrier<PIPE_V>();
            
            // 如果需要行循环，将更新张量从全局内存复制到GO UB张量
            if (needRowLoop) {
                AscendC::DataCopy(
                    goUbTensor, gUpdate, 
                    AscendC::DataCopyParams(1, curRowNum * embedRound / HALF_BLOCK_SIZE, 0, 0));
                
                // 设置事件标记，通知更新数据已准备好
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
                
                // 等待更新数据复制完成
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
            }
            
            // 执行乘法操作：go = go * dm_block（将GO张量与DM张量相乘）
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            for (uint32_t vmul_idx = 0; vmul_idx < embed / HALF_VECTOR_SIZE; ++vmul_idx) {
                AscendC::Mul<half, false>(
                    goUbTensor[vmul_idx * HALF_VECTOR_SIZE],   // 目标位置
                    goUbTensor[vmul_idx * HALF_VECTOR_SIZE],   // 操作数A
                    tvUbTensor,                                // 操作数B
                    (uint64_t)0,                               // 掩码
                    curRowNum,                                 // 行数
                    AscendC::BinaryRepeatParams(               // 重复参数
                        1, 1, 0, embedRound / HALF_BLOCK_SIZE, embedRound / HALF_BLOCK_SIZE, 1));
            }
            
            // 处理嵌入维度不是半精度向量大小整数倍的情况
            if (embed % HALF_VECTOR_SIZE > 0) {
                SetMask(embed % HALF_VECTOR_SIZE);  // 设置掩码只处理有效元素
                AscendC::Mul<half, false>(
                    goUbTensor[embed / HALF_VECTOR_SIZE * HALF_VECTOR_SIZE],
                    goUbTensor[embed / HALF_VECTOR_SIZE * HALF_VECTOR_SIZE],
                    tvUbTensor,
                    (uint64_t)0,
                    curRowNum,
                    AscendC::BinaryRepeatParams(
                        1, 1, 0, embedRound / HALF_BLOCK_SIZE, embedRound / HALF_BLOCK_SIZE, 1));
                AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);  // 恢复全1掩码
            }
            
            // 等待V管道操作完成
            AscendC::PipeBarrier<PIPE_V>();
            
            // 等待LO张量数据准备好
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            
            // 执行加法操作：go = lo + go（将LO张量与GO张量相加）
            AscendC::Add<half, false>(
                goUbTensor,                                // 目标位置
                goUbTensor,                                // 操作数A
                loUbTensor,                                // 操作数B
                (uint64_t)0,                               // 掩码
                (curRowNum * embedRound + HALF_VECTOR_SIZE - 1) / HALF_VECTOR_SIZE,  // 向量数
                AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));  // 重复参数
            
            // 等待V管道操作完成
            AscendC::PipeBarrier<PIPE_V>();
            
            // 设置事件标记，通知当前tile的计算完成
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
        } else {
            // 第一个堆叠tile的处理逻辑：直接将输入复制到GO UB张量（go = lo）
            AscendC::DataCopy(
                goUbTensor, gInput, AscendC::DataCopyParams(1, curRowNum * embedRound / HALF_BLOCK_SIZE, 0, 0));
            
            // 设置事件标记，通知输入数据已准备好
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            
            // 等待输入数据复制完成
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        }

        // 最后一个堆叠tile的处理逻辑
        if (isLastStackTile) {
            // 将GL UB张量广播到TV UB张量（用于后续除法操作）
            AscendC::Brcb(
                tvUbTensor.ReinterpretCast<uint16_t>(),
                glUbTensor.ReinterpretCast<uint16_t>()[rowOffsetLoop],
                curRowNumRound / FLOAT_BLOCK_SIZE,
                AscendC::BrcbRepeatParams(1, 8));
            
            // 等待V管道操作完成
            AscendC::PipeBarrier<PIPE_V>();
            
            // 执行除法操作：go = go / gl_block（将GO张量除以GL张量）
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            for (uint32_t vdiv_idx = 0; vdiv_idx < embed / HALF_VECTOR_SIZE; ++vdiv_idx) {
                AscendC::Div<half, false>(
                    goUbTensor[vdiv_idx * HALF_VECTOR_SIZE],   // 目标位置
                    goUbTensor[vdiv_idx * HALF_VECTOR_SIZE],   // 操作数A（被除数）
                    tvUbTensor,                                // 操作数B（除数）
                    (uint64_t)0,                               // 掩码
                    curRowNum,                                 // 行数
                    AscendC::BinaryRepeatParams(               // 重复参数
                        1, 1, 0, embedRound / HALF_BLOCK_SIZE, embedRound / HALF_BLOCK_SIZE, 1));
            }
            
            // 处理嵌入维度不是半精度向量大小整数倍的情况
            if (embed % HALF_VECTOR_SIZE > 0) {
                SetMask(embed % HALF_VECTOR_SIZE);  // 设置掩码只处理有效元素
                AscendC::Div<half, false>(
                    goUbTensor[embed / HALF_VECTOR_SIZE * HALF_VECTOR_SIZE],
                    goUbTensor[embed / HALF_VECTOR_SIZE * HALF_VECTOR_SIZE],
                    tvUbTensor,
                    (uint64_t)0,
                    curRowNum,
                    AscendC::BinaryRepeatParams(
                        1, 1, 0, embedRound / HALF_BLOCK_SIZE, embedRound / HALF_BLOCK_SIZE, 1));
                AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);  // 恢复全1掩码
            }
            
            // 等待V管道操作完成
            AscendC::PipeBarrier<PIPE_V>();
            
            // 设置事件标记，通知除法操作完成
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            
            // 等待除法操作完成
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

            // 将处理完成的输出数据从UB复制到全局内存
            CopyOToGm(
                gOutput, proTokenIdx, proTokenNum, epiTokenNum, integralHeadNum, qSThisSubBlock, embed, oHiddenSize);
            // 处理LSE（对数求和指数），当LSE模式为OUT_ONLY且是最后一个行循环时
            if constexpr (LSE_MODE_ == LseModeT::OUT_ONLY) {
                if (isLastRowLoop) {
                    // 等待V管道操作完成
                    AscendC::PipeBarrier<PIPE_V>();
                    
                    // 计算GL张量的自然对数：lse16 = ln(gl)
                    AscendC::Ln<half, false>(
                        lse16_ubuf_tensor,                     // 目标位置
                        glUbTensor,                            // 输入张量
                        (uint64_t)0,                            // 掩码
                        CeilDiv(totalRowNum, HALF_VECTOR_SIZE), // 向量数
                        AscendC::UnaryRepeatParams(1, 1, 8, 8));// 重复参数
                    
                    // 等待V管道操作完成
                    AscendC::PipeBarrier<PIPE_V>();
                    
                    // 执行加法操作：lse16 = lse16 + gm（将LSE16张量与GM张量相加）
                    AscendC::Add<half, false>(
                        lse16_ubuf_tensor,                     // 目标位置
                        lse16_ubuf_tensor,                     // 操作数A
                        gmUbTensor,                            // 操作数B
                        (uint64_t)0,                            // 掩码
                        CeilDiv(totalRowNum, HALF_VECTOR_SIZE), // 向量数
                        AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));  // 重复参数
                    
                    // 等待V管道操作完成
                    AscendC::PipeBarrier<PIPE_V>();
                    
                    // 将16位LSE张量转换为32位LSE张量：lse32 = (float)lse16
                    AscendC::Cast<float, half, false>(
                        lse32_ubuf_tensor,                     // 目标位置
                        lse16_ubuf_tensor,                     // 输入张量
                        AscendC::RoundMode::CAST_NONE,         // 舍入模式
                        (uint64_t)0,                            // 掩码
                        CeilDiv(totalRowNum, FLOAT_VECTOR_SIZE), // 向量数
                        AscendC::UnaryRepeatParams(1, 1, 8, 4));// 重复参数
                    
                    // 等待V管道操作完成
                    AscendC::PipeBarrier<PIPE_V>();

                    // 将32位LSE张量广播到TV UB张量（用于后续复制到全局内存）
                    AscendC::Brcb(
                        tvUbTensor32.ReinterpretCast<uint32_t>(),
                        lse32_ubuf_tensor.ReinterpretCast<uint32_t>(),
                        CeilDiv(totalRowNum, FLOAT_BLOCK_SIZE),
                        AscendC::BrcbRepeatParams(1, 8));
                    
                    // 等待V管道操作完成
                    AscendC::PipeBarrier<PIPE_V>();
                    
                    // 设置事件标记，通知LSE处理完成
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID4);
                    
                    // 等待LSE处理完成
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID4);
                    
                    // 将处理完成的LSE数据从UB复制到全局内存
                    if (qNThisSubBlock == 0U) {
                        // 单个查询头的情况
                        AscendC::DataCopyPad(
                            gLse, tvUbTensor32,
                            AscendC::DataCopyExtParams(
                                totalRowNum, sizeof(float), 0, (qHeads - 1) * sizeof(float), 0));
                    } else {
                        // 多个查询头的情况
                        for (uint32_t qNIdx = 0; qNIdx < qNThisSubBlock; qNIdx++) {
                            AscendC::DataCopyPad(
                                gLse[qNIdx],
                                tvUbTensor32[qNIdx * qSBlockSize * FLOAT_BLOCK_SIZE],
                                AscendC::DataCopyExtParams(
                                    qSBlockSize, sizeof(float), 0, (qHeads - 1) * sizeof(float), 0));
                        }
                    }
                    
                    // 设置事件标记，通知LSE数据复制完成
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
                }
            }
        } else if (needRowLoop) {
            // 需要行循环的处理逻辑（不是最后一个堆叠tile）
            
            // 设置事件标记，通知当前循环完成
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID5);
            
            // 等待当前循环完成
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID5);
            
            // 将处理完成的GO张量复制回全局内存的更新张量
            AscendC::DataCopy(
                gUpdate, goUbTensor, AscendC::DataCopyParams(1, curRowNum * embedRound / HALF_BLOCK_SIZE, 0, 0));
        }
        
        // 设置事件标记，通知MTE3和MTE2管道完成
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
        uint32_t rowNumTile = RoundDown(maxRowNumPerLoop, HALF_BLOCK_SIZE);  ///< 对齐后的每轮循环行数

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
    AscendC::LocalTensor<half> loUbTensor;             ///< 本地输出张量UB缓冲区，用于存储从全局内存加载的输入数据
    AscendC::LocalTensor<half> dmUbTensor;             ///< 数据矩阵张量UB缓冲区(Data Matrix Tensor)，存储需要广播的缩放因子数据
    AscendC::LocalTensor<half> hmUbTensor;             ///< 本地矩阵M UB缓冲区
    AscendC::LocalTensor<half> glUbTensor;             ///< 全局LSE张量UB缓冲区，存储全局对数求和指数数据，用于数值稳定性处理
    AscendC::LocalTensor<half> tvUbTensor;             ///< 临时向量UB张量(Temporary Vector Tensor)，半精度版本，用于存储广播后的临时数据，供乘法/除法操作使用
    AscendC::LocalTensor<float> tvUbTensor32;          ///< 临时向量UB张量(Temporary Vector Tensor)，单精度版本，用于LSE数据处理
    AscendC::LocalTensor<ElementOutput> goUbTensor;    ///< 全局输出张量UB缓冲区，用于存储最终输出结果
    AscendC::LocalTensor<half> gmUbTensor;             ///< 全局矩阵M UB缓冲区
    AscendC::LocalTensor<half> lse16_ubuf_tensor;      ///< 16位LSE张量UB缓冲区，用于LSE中间计算
    AscendC::LocalTensor<float> lse32_ubuf_tensor;     ///< 32位LSE张量UB缓冲区，用于LSE最终计算结果
};

}

#endif