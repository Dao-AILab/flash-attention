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

#ifndef EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_LOW_PREC_HPP_T
#define EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_LOW_PREC_HPP_T

// Catlass库核心头文件
#include "catlass/catlass.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"

// 自定义块结构头文件
#include "fa_block.h"

namespace Catlass::Epilogue::Block {
    /**
     * @file online_softmax_low_prec.hpp
     * @brief Atlas A2平台的低精度在线Softmax块级Epilogue实现
     * 
     * 该文件定义了针对Atlas A2平台优化的低精度在线Softmax块级Epilogue实现，
     * 主要用于Transformer模型中的注意力机制计算。实现了高效的半精度(half)Softmax计算，
     * 包括数据重缩放、掩码处理、行最大值计算、行求和、指数计算等核心功能，
     * 充分利用了Atlas A2架构的UB内存和向量计算能力。
     */

    /**
     * @brief 低精度在线Softmax的块级Epilogue实现类模板
     * 
     * 该类负责在矩阵乘法完成后执行在线Softmax操作，包括数据重缩放、掩码处理、
     * Softmax计算和结果输出。专门为Atlas A2平台优化，使用半精度(half)数据类型，
     * 支持多种LSE（对数求和指数）计算模式。
     * 
     * @tparam OutputType_ 输出类型和布局，包含元素类型和矩阵布局信息
     * @tparam InputType_ 输入类型和布局，包含元素类型和矩阵布局信息
     * @tparam MaskType_ 掩码类型和布局，包含元素类型和矩阵布局信息
     * @tparam LSE_MODE_ LSE（对数求和指数）计算模式，控制是否计算和输出LSE值
     */

template <
    class OutputType_,     // 输出类型和布局
    class InputType_,      // 输入类型和布局
    class MaskType_,       // 掩码类型和布局
    LseModeT LSE_MODE_>    // LSE（对数求和指数）计算模式
class BlockEpilogue<
    EpilogueAtlasA2OnlineSoftmaxT<LSE_MODE_, half>,
    OutputType_,
    InputType_,
    MaskType_>
{
public:
    // 类型别名
    using DispatchPolicy = EpilogueAtlasA2OnlineSoftmaxT<LSE_MODE_, half>;  ///< 调度策略类型，定义了Atlas A2平台的在线Softmax执行策略
    using ArchTag = typename DispatchPolicy::ArchTag;                       ///< 架构标签类型，标识Atlas A2架构
    using ElementOutput = typename OutputType_::Element;                    ///< 输出元素类型
    using ElementInput = typename InputType_::Element;                      ///< 输入元素类型
    using ElementMask = typename MaskType_::Element;                        ///< 掩码元素类型

    using LayoutOutput = typename OutputType_::Layout;                      ///< 输出布局类型
    using LayoutInput = typename InputType_::Layout;                        ///< 输入布局类型
    using LayoutMask = typename MaskType_::Layout;                          ///< 掩码布局类型

    // 静态常量
    static constexpr LseModeT LSE_MODE = DispatchPolicy::LSE_MODE;          ///< LSE（对数求和指数）计算模式

    // 内存块和向量大小常量
    static constexpr uint32_t BLOCK_SIZE_IN_BYTE = 32;                      ///< 内存块大小（字节），Atlas A2架构的基本内存访问单元
    static constexpr uint32_t REPEAT_SIZE_IN_BYTE = 256;                    ///< 重复操作的内存大小（字节）
    static constexpr uint32_t FLOAT_BLOCK_SIZE = 8;                         ///< 浮点数块大小，用于浮点数运算的基本单元
    static constexpr uint32_t FLOAT_VECTOR_SIZE = 64;                       ///< 浮点数向量大小，单条指令可处理的浮点数数量
    static constexpr uint32_t HALF_VECTOR_SIZE = 128;                       ///< 半精度向量大小，单条指令可处理的半精度浮点数数量
    static constexpr uint32_t BLOCK_SIZE = 16;                              ///< 基本块大小，用于数据划分
    static constexpr uint32_t UB_UINT8_VECTOR_SIZE = 1024;                  ///< UB（Unified Buffer）中uint8类型的向量大小
    static constexpr uint32_t UB_UINT8_BLOCK_SIZE = 16384;                  ///< UB中uint8类型的块大小
    static constexpr uint32_t VECTOR_SIZE = 128;                            ///< 通用向量大小
    static constexpr uint32_t MAX_UB_S_ELEM_NUM = 16384;                    ///< UB中可存储的S矩阵元素的最大数量

    // 归约和行操作相关常量
    static constexpr uint32_t REDUCE_UB_SIZE = 1024;                        ///< 归约操作使用的UB大小
    static constexpr uint32_t ROW_OPS_SPEC_MASK_32 = 32;                   ///< 行操作特定掩码（32位）
    static constexpr uint32_t ROW_OPS_SPEC_MASK_8 = 8;                     ///< 行操作特定掩码（8位）
    static constexpr uint32_t ROW_OPS_SPEC_MASK_4 = 4;                     ///< 行操作特定掩码（4位）
    static constexpr uint32_t ROW_OPS_SPEC_MASK_2 = 2;                     ///< 行操作特定掩码（2位）
    static constexpr uint32_t MAX_ROW_NUM_SUB_CORE = 256;                  ///< 单个子核可处理的最大行数
    static constexpr int64_t UB_FLOAT_LINE_SIZE = 64;                      ///< UB中浮点数的行大小

    // 分割列索引
    static constexpr uint32_t SPLIT_COL_IDX_2 = 2;                         ///< 用于分块处理的列索引分割点2
    static constexpr uint32_t SPLIT_COL_IDX_3 = 3;                         ///< 用于分块处理的列索引分割点3

    /**
     * @brief 构造函数：初始化低精度在线Softmax块级Epilogue
     * 
     * @param resource Atlas架构资源引用，包含UB内存等硬件资源
     * @param scaleValue_ 缩放值，用于对输入数据进行重缩放
     */
    __aicore__ inline
    BlockEpilogue(Arch::Resource<ArchTag> &resource, float scaleValue_)
    {
        // UB空间偏移常量定义
        constexpr uint32_t LS_UB_TENSOR_OFFSET = 0;                            ///< LSE（对数求和指数）结果UB张量偏移
        constexpr uint32_t COMPUTE_UB_TENSOR_OFFSET = 2 * UB_UINT8_BLOCK_SIZE;  ///< 计算用UB张量偏移
        constexpr uint32_t LP_UB_TENSOR_OFFSET = 4 * UB_UINT8_BLOCK_SIZE;       ///< 输出准备用UB张量偏移
        constexpr uint32_t MASK16_UB_TENSOR_OFFSET = 0;                         ///< 16位掩码UB张量偏移

        constexpr uint32_t TV_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE;      ///< TV张量UB偏移
        constexpr uint32_t LM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 8 * UB_UINT8_VECTOR_SIZE;  ///< LM张量UB偏移

        constexpr uint32_t HM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 9 * UB_UINT8_VECTOR_SIZE;  ///< HM张量UB偏移
        constexpr uint32_t GM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 10 * UB_UINT8_VECTOR_SIZE; ///< GM张量UB偏移
        constexpr uint32_t LL_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 11 * UB_UINT8_VECTOR_SIZE; ///< LL张量UB偏移
        constexpr uint32_t GL_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 12 * UB_UINT8_VECTOR_SIZE; ///< GL张量UB偏移
        constexpr uint32_t DM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 13 * UB_UINT8_VECTOR_SIZE; ///< DM张量UB偏移

        constexpr uint32_t MASK_UB_TENSOR_OFFSET = 11 * UB_UINT8_BLOCK_SIZE;    ///< 掩码UB张量偏移

        // 初始化缩放值
        scaleValue = static_cast<half>(scaleValue_);
        
        // 分配UB内存空间
        lsUbTensor = resource.ubBuf.template GetBufferByByte<half>(LS_UB_TENSOR_OFFSET);           ///< LSE结果UB张量
        computeUbTensor = resource.ubBuf.template GetBufferByByte<half>(COMPUTE_UB_TENSOR_OFFSET);  ///< 计算用UB张量
        lpUbTensor = resource.ubBuf.template GetBufferByByte<ElementOutput>(LP_UB_TENSOR_OFFSET);  ///< 输出准备用UB张量
        maskUbTensor = resource.ubBuf.template GetBufferByByte<ElementMask>(MASK_UB_TENSOR_OFFSET);  ///< 掩码UB张量
        maskUbTensor16 = resource.ubBuf.template GetBufferByByte<half>(MASK16_UB_TENSOR_OFFSET);  ///< 16位掩码UB张量
        lmUbTensor = resource.ubBuf.template GetBufferByByte<half>(LM_UB_TENSOR_OFFSET);  ///< LM张量UB
        hmUbTensor = resource.ubBuf.template GetBufferByByte<half>(HM_UB_TENSOR_OFFSET);  ///< HM张量UB
        gmUbTensor = resource.ubBuf.template GetBufferByByte<half>(GM_UB_TENSOR_OFFSET);  ///< GM张量UB
        dmUbTensor = resource.ubBuf.template GetBufferByByte<half>(DM_UB_TENSOR_OFFSET);  ///< DM张量UB
        llUbTensor = resource.ubBuf.template GetBufferByByte<half>(LL_UB_TENSOR_OFFSET);  ///< LL张量UB
        tvUbTensor = resource.ubBuf.template GetBufferByByte<half>(TV_UB_TENSOR_OFFSET);  ///< TV张量UB
        glUbTensor = resource.ubBuf.template GetBufferByByte<half>(GL_UB_TENSOR_OFFSET);  ///< GL张量UB
    }

    /**
     * @brief 析构函数：清理资源
     */
    __aicore__ inline
    ~BlockEpilogue() {}

    /**
     * @brief 设置向量掩码
     * 
     * 根据向量长度设置适当的向量掩码，用于标识向量中哪些元素是有效的
     * 
     * @param len 向量长度
     */
    __aicore__ inline
    void SetVecMask(int32_t len)
    {
        const int32_t MAX_MASK_LEN = 128;  ///< 最大掩码长度
        const int32_t HALF_MASK_LEN = 64;  ///< 半掩码长度
        
        // 根据长度设置不同的掩码
        if (len >= MAX_MASK_LEN) {
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);  ///< 全1掩码
            return;
        }
        
        int32_t highMask = len - HALF_MASK_LEN > 0 ? len - HALF_MASK_LEN : 0;
        int32_t lowMask = len - HALF_MASK_LEN >= 0 ? HALF_MASK_LEN : len;
        
        if (len < HALF_MASK_LEN) {
            AscendC::SetVectorMask<int8_t>(0x0, ((uint64_t)1 << lowMask) - 1);  ///< 只设置低半部分掩码
        } else {
            AscendC::SetVectorMask<int8_t>(((uint64_t)1 << highMask) - 1, 0xffffffffffffffff);  ///< 设置高低两部分掩码
        }
    }

    /**
     * @brief 设置块归约掩码
     * 
     * 根据长度设置块归约操作的掩码，用于标识归约操作中哪些元素需要参与计算
     * 
     * @param len 归约长度
     */
    __aicore__ inline
    void SetBlockReduceMask(int32_t len)
    {
        const int32_t MAX_LEN = 16;
        if (len > MAX_LEN) {
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);  ///< 全1掩码
            return;
        }
        uint64_t subMask = (static_cast<uint64_t>(1) << len) - 1;  ///< 生成基本掩码
        uint64_t maskValue = (subMask << 48) + (subMask << 32) + (subMask << 16) + subMask;  ///< 扩展掩码到所有字节
        AscendC::SetVectorMask<int8_t>(maskValue, maskValue);  ///< 设置掩码
    }

    /**
     * @brief 计算SPEC TILE（512元素）的行求和
     * 
     * 对512元素大小的特定分块进行高效的行求和操作，使用向量加法和归约指令
     * 
     * @param srcUb 源UB张量，存储输入数据
     * @param rowsumUb 行求和结果UB张量
     * @param tvUbTensor 临时变量UB张量
     * @param numRowsRound 行数（已对齐）
     * @param numElems 元素数量
     * @param numElemsAligned 元素数量（已对齐）
     */
    __aicore__ inline
    void RowsumSPECTILE512(const AscendC::LocalTensor<half> &srcUb, const AscendC::LocalTensor<half> &rowsumUb,
        const AscendC::LocalTensor<half> &tvUbTensor, uint32_t numRowsRound, uint32_t numElems,
        uint32_t numElemsAligned)
    {
        // 第一阶段向量加法：将第2个向量添加到第1个向量
        AscendC::Add<half, false>(
            srcUb,
            srcUb,
            srcUb[HALF_VECTOR_SIZE],
            (uint64_t)0,
            numRowsRound,
            AscendC::BinaryRepeatParams(
                1, 1, 1,
                numElemsAligned / BLOCK_SIZE,
                numElemsAligned / BLOCK_SIZE,
                numElemsAligned / BLOCK_SIZE));
        
        // 第一阶段向量加法：将第4个向量添加到第3个向量
        AscendC::Add<half, false>(
            srcUb[HALF_VECTOR_SIZE * SPLIT_COL_IDX_2],
            srcUb[HALF_VECTOR_SIZE * SPLIT_COL_IDX_2],
            srcUb[HALF_VECTOR_SIZE * SPLIT_COL_IDX_3],
            (uint64_t)0,
            numRowsRound,
            AscendC::BinaryRepeatParams(
                1, 1, 1,
                numElemsAligned / BLOCK_SIZE,
                numElemsAligned / BLOCK_SIZE,
                numElemsAligned / BLOCK_SIZE));
        
        AscendC::PipeBarrier<PIPE_V>();  // 向量管道同步
        
        // 第二阶段向量加法：将第3个向量添加到第1个向量
        AscendC::Add<half, false>(
            srcUb,
            srcUb,
            srcUb[HALF_VECTOR_SIZE * SPLIT_COL_IDX_2],
            (uint64_t)0,
            numRowsRound,
            AscendC::BinaryRepeatParams(
                1, 1, 1,
                numElemsAligned / BLOCK_SIZE,
                numElemsAligned / BLOCK_SIZE,
                numElemsAligned / BLOCK_SIZE));
        
        AscendC::PipeBarrier<PIPE_V>();  // 向量管道同步
        
        // 行归约求和
        AscendC::WholeReduceSum<half, false>(
            rowsumUb, srcUb, (int32_t)0, numRowsRound, 1, 1,
            numElemsAligned / BLOCK_SIZE);
        
        AscendC::PipeBarrier<PIPE_V>();  // 向量管道同步
    }

    /**
     * @brief 计算TAIL TILE（剩余元素）的行求和
     * 
     * 对尾部分块（非完整512元素）进行行求和操作，处理边界情况
     * 
     * @param srcUb 源UB张量，存储输入数据
     * @param rowsumUb 行求和结果UB张量
     * @param tvUbTensor 临时变量UB张量
     * @param numRowsRound 行数（已对齐）
     * @param numElems 元素数量
     * @param numElemsAligned 元素数量（已对齐）
     */
    __aicore__ inline
    void RowsumTAILTILE(const AscendC::LocalTensor<half> &srcUb, const AscendC::LocalTensor<half> &rowsumUb,
        const AscendC::LocalTensor<half> &tvUbTensor, uint32_t numRowsRound, uint32_t numElems,
        uint32_t numElemsAligned)
    {
        if (numElems <= HALF_VECTOR_SIZE) {
            // 元素数量小于等于一个向量大小，直接归约
            SetVecMask(numElems);  // 设置向量掩码
            AscendC::WholeReduceSum<half, false>(
                rowsumUb, srcUb, (int32_t)0, numRowsRound, 1, 1,
                numElemsAligned / BLOCK_SIZE);
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);  // 恢复全1掩码
        } else {
            // 元素数量大于一个向量大小，分阶段归约
            for (uint32_t vmaxIdx = 1; vmaxIdx < numElems / HALF_VECTOR_SIZE; vmaxIdx++) {
                AscendC::Add<half, false>(
                    srcUb,
                    srcUb,
                    srcUb[vmaxIdx * HALF_VECTOR_SIZE],
                    (uint64_t)0,
                    numRowsRound,
                    AscendC::BinaryRepeatParams(
                        1, 1, 1,
                        numElemsAligned / BLOCK_SIZE,
                        numElemsAligned / BLOCK_SIZE,
                        numElemsAligned / BLOCK_SIZE));
                AscendC::PipeBarrier<PIPE_V>();  // 向量管道同步
            }
            
            // 处理剩余元素
            if (numElems % HALF_VECTOR_SIZE > 0) {
                SetVecMask(numElems % HALF_VECTOR_SIZE);  // 设置向量掩码
                AscendC::Add<half, false>(
                    srcUb,
                    srcUb,
                    srcUb[numElems / HALF_VECTOR_SIZE * HALF_VECTOR_SIZE],
                    (uint64_t)0,
                    numRowsRound,
                    AscendC::BinaryRepeatParams(
                        1, 1, 1,
                        numElemsAligned / BLOCK_SIZE,
                        numElemsAligned / BLOCK_SIZE,
                        numElemsAligned / BLOCK_SIZE));
                AscendC::PipeBarrier<PIPE_V>();  // 向量管道同步
                AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);  // 恢复全1掩码
            }
            
            // 最终行归约求和
            AscendC::WholeReduceSum<half, false>(
                rowsumUb, srcUb, (int32_t)0, numRowsRound, 1, 1,
                numElemsAligned / BLOCK_SIZE);
        }
        AscendC::PipeBarrier<PIPE_V>();  // 向量管道同步
    }

    /**
     * @brief 计算尾部分块的行最大值
     * 
     * 对于元素数量小于等于512的尾部分块，计算每行的最大值。支持向量级掩码处理以
     * 处理非对齐的元素数量。
     * 
     * @param srcUb 输入数据的UB张量
     * @param rowmaxUb 输出行最大值的UB张量
     * @param tvUbTensor 临时变量UB张量（未使用）
     * @param numRowsRound 行数（已对齐到向量大小）
     * @param numElems 实际元素数量
     * @param numElemsAligned 对齐后的元素数量
     */
    __aicore__ inline
    void RowmaxTAILTILE(const AscendC::LocalTensor<half> &srcUb, const AscendC::LocalTensor<half> &rowmaxUb,
        const AscendC::LocalTensor<half> &tvUbTensor, uint32_t numRowsRound, uint32_t numElems,
        uint32_t numElemsAligned)
    {
        if (numElems <= HALF_VECTOR_SIZE) {
            // 元素数量小于等于向量大小，直接进行归约
            SetVecMask(numElems);  // 设置向量掩码，处理非对齐元素
            AscendC::WholeReduceMax<half, false>(
                rowmaxUb, srcUb, (int32_t)0, numRowsRound, 1, 1,
                numElemsAligned / BLOCK_SIZE, AscendC::ReduceOrder::ORDER_ONLY_VALUE);
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);  // 恢复全1掩码
        } else {
            // 元素数量大于向量大小，分步计算最大值
            // 首先复制第一组向量到临时存储
            AscendC::DataCopy(
                lsUbTensor,
                srcUb,
                AscendC::DataCopyParams(
                    numRowsRound,
                    HALF_VECTOR_SIZE / BLOCK_SIZE,
                    (numElemsAligned - HALF_VECTOR_SIZE) / BLOCK_SIZE,
                    (numElemsAligned - HALF_VECTOR_SIZE) / BLOCK_SIZE));
            AscendC::PipeBarrier<PIPE_V>();  // 向量管道同步
            
            // 循环与剩余向量组计算最大值
            for (uint32_t vmaxIdx = 1; vmaxIdx < numElems / HALF_VECTOR_SIZE; vmaxIdx++) {
                AscendC::Max<half, false>(
                    lsUbTensor,
                    lsUbTensor,  // 输入和输出都是临时存储，保存当前最大值
                    srcUb[vmaxIdx * HALF_VECTOR_SIZE],  // 当前处理的向量组
                    (uint64_t)0,
                    numRowsRound,
                    AscendC::BinaryRepeatParams(
                        1, 1, 1,
                        numElemsAligned / BLOCK_SIZE,
                        numElemsAligned / BLOCK_SIZE,
                        numElemsAligned / BLOCK_SIZE));
                AscendC::PipeBarrier<PIPE_V>();  // 向量管道同步
            }
            
            // 处理剩余的非对齐元素
            if (numElems % HALF_VECTOR_SIZE > 0) {
                SetVecMask(numElems % HALF_VECTOR_SIZE);  // 设置向量掩码
                AscendC::Max<half, false>(
                    lsUbTensor,
                    lsUbTensor,
                    srcUb[numElems / HALF_VECTOR_SIZE * HALF_VECTOR_SIZE],  // 剩余元素起始位置
                    (uint64_t)0,
                    numRowsRound,
                    AscendC::BinaryRepeatParams(
                        1, 1, 1,
                        numElemsAligned / BLOCK_SIZE,
                        numElemsAligned / BLOCK_SIZE,
                        numElemsAligned / BLOCK_SIZE));
                AscendC::PipeBarrier<PIPE_V>();  // 向量管道同步
                AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);  // 恢复全1掩码
            }
            
            // 对临时存储中的中间结果进行最终归约，得到每行最大值
            AscendC::WholeReduceMax<half, false>(
                rowmaxUb, lsUbTensor, (int32_t)0, numRowsRound, 1, 1,
                numElemsAligned / BLOCK_SIZE, AscendC::ReduceOrder::ORDER_ONLY_VALUE);
        }
        AscendC::PipeBarrier<PIPE_V>();  // 向量管道同步
    }

    /**
     * @brief 将全局内存中的输入数据复制到UB
     * 
     * 从全局内存加载输入数据到UB（Unified Buffer）中，为后续计算做准备。
     * 支持处理对齐和填充的列数。
     * 
     * @param gInput 全局内存中的输入张量
     * @param sUbOffset UB中的起始偏移量（未使用）
     * @param rowNumCurLoop 当前循环处理的行数
     * @param columnNumRound 对齐后的列数
     * @param columnNumPad 填充后的列数
     */
    __aicore__ inline
    void CopySGmToUb(AscendC::GlobalTensor<half> gInput, uint32_t sUbOffset, uint32_t rowNumCurLoop,
        uint32_t columnNumRound, uint32_t columnNumPad)
    {
        AscendC::DataCopy(
            lsUbTensor,  // 目标UB张量
            gInput,      // 源全局内存张量
            AscendC::DataCopyParams(rowNumCurLoop,
                columnNumRound / BLOCK_SIZE,  // 每次复制的块数
                (columnNumPad - columnNumRound) / BLOCK_SIZE,  // 填充块数
                0));  // 源偏移块数
    }

    /**
     * @brief 将掩码数据从全局内存复制到UB
     * 
     * 从全局内存加载掩码数据到UB中，支持处理前导token、完整注意力头token和尾token。
     * 使用DataCopyPad进行复制，自动处理填充。
     * 
     * @param gMask 全局内存中的掩码张量
     * @param columnNum 实际列数
     * @param columnNumRound 对齐后的列数
     * @param maskStride 掩码的步长
     * @param tokenNumPerHead 每个注意力头的token数量
     * @param proTokenIdx 前导token的索引
     * @param proTokenNum 前导token的数量
     * @param integralHeadNum 完整注意力头的数量
     * @param epiTokenNum 尾token的数量
     */
    __aicore__ inline
    void CopyMaskGmToUb(AscendC::GlobalTensor<ElementMask> gMask, uint32_t columnNum, uint32_t columnNumRound,
        uint32_t maskStride, uint32_t tokenNumPerHead, uint32_t proTokenIdx, uint32_t proTokenNum,
        uint32_t integralHeadNum, uint32_t epiTokenNum)
    {
        uint32_t innerUbRowOffset = 0;
        
        // 复制前导token的掩码
        if (proTokenNum != 0U) {
            AscendC::DataCopyPad(
                maskUbTensor[innerUbRowOffset],  // 目标UB位置
                gMask[proTokenIdx * maskStride],  // 源全局内存位置
                AscendC::DataCopyExtParams(
                    proTokenNum, columnNum * sizeof(ElementMask),  // 复制数量和源步长
                    (maskStride - columnNum) * sizeof(ElementMask), 0, 0),  // 源间隙和偏移
                AscendC::DataCopyPadExtParams<ElementMask>(false, 0, 0, 0));  // 填充参数
            innerUbRowOffset += proTokenNum * columnNumRound;  // 更新UB偏移
        }
        
        // 复制完整注意力头的掩码
        for (uint32_t headIdx = 0; headIdx < integralHeadNum; headIdx++) {
            AscendC::DataCopyPad(
                maskUbTensor[innerUbRowOffset],  // 目标UB位置
                gMask,  // 源全局内存位置
                AscendC::DataCopyExtParams(
                    tokenNumPerHead, columnNum * sizeof(ElementMask),  // 复制数量和源步长
                    (maskStride - columnNum) * sizeof(ElementMask), 0, 0),  // 源间隙和偏移
                AscendC::DataCopyPadExtParams<ElementMask>(false, 0, 0, 0));  // 填充参数
            innerUbRowOffset += tokenNumPerHead * columnNumRound;  // 更新UB偏移
        }
        
        // 复制尾token的掩码
        if (epiTokenNum != 0) {
            AscendC::DataCopyPad(
                maskUbTensor[innerUbRowOffset],  // 目标UB位置
                gMask,  // 源全局内存位置
                AscendC::DataCopyExtParams(
                    epiTokenNum, columnNum * sizeof(ElementMask),  // 复制数量和源步长
                    (maskStride - columnNum) * sizeof(ElementMask), 0, 0),  // 源间隙和偏移
                AscendC::DataCopyPadExtParams<ElementMask>(false, 0, 0, 0));  // 填充参数
        }
    }

    /**
     * @brief 对输入数据进行缩放
     * 
     * 将输入数据与缩放因子相乘，用于Softmax计算前的输入预处理。
     * 
     * @param sUbOffset UB中的起始偏移量（未使用）
     * @param rowNumCurLoop 当前循环处理的行数
     * @param columnNumRound 对齐后的列数
     */
    __aicore__ inline
    void ScaleS(uint32_t sUbOffset, uint32_t rowNumCurLoop, uint32_t columnNumRound)
    {
        // 计算每个行向量的元素数量（对齐到向量大小）
        uint32_t numVecs = (rowNumCurLoop * columnNumRound + HALF_VECTOR_SIZE - 1) / HALF_VECTOR_SIZE;
        
        // 执行向量乘操作：ls = scaleValue * ls
        AscendC::Muls<half, false>(
            computeUbTensor,  // 输出UB张量
            lsUbTensor,       // 输入UB张量
            scaleValue,       // 缩放因子
            (uint64_t)0,      // 输出偏移
            numVecs,          // 向量数量
            AscendC::UnaryRepeatParams(1, 1, 8, 8));  // 重复参数
        AscendC::PipeBarrier<PIPE_V>();  // 向量管道同步
    }

    /**
     * @brief 将掩码类型转换到目标类型
     * 
     * 模板函数，将掩码数据从源类型转换到目标类型，用于后续的掩码应用操作。
     * 例如，将int8掩码转换为half类型以便与输入数据进行运算。
     * 
     * @tparam ElementMaskDst 目标掩码类型
     * @tparam ElementMaskSrc 源掩码类型
     * @param maskUbTensorDst 目标掩码UB张量
     * @param maskUbTensorSrc 源掩码UB张量
     * @param rowNumCurLoop 当前循环处理的行数
     * @param columnNumRound 对齐后的列数
     */
    template<typename ElementMaskDst, typename ElementMaskSrc>
    __aicore__ inline 
    void UpCastMask(
        const AscendC::LocalTensor<ElementMaskDst> &maskUbTensorDst,
        const AscendC::LocalTensor<ElementMaskSrc> &maskUbTensorSrc,
        uint32_t rowNumCurLoop,
        uint32_t columnNumRound)
    {
        // 计算需要处理的块数
        uint32_t numBlocks = CeilDiv(rowNumCurLoop * columnNumRound, 
            (uint32_t)(REPEAT_SIZE_IN_BYTE / sizeof(ElementMaskDst)));
        
        // 执行类型转换操作
        AscendC::Cast<ElementMaskDst, ElementMaskSrc, false>(
            maskUbTensorDst,        // 目标张量
            maskUbTensorSrc,        // 源张量
            AscendC::RoundMode::CAST_NONE,  // 无舍入模式
            (uint64_t)0,            // 输出偏移
            numBlocks,              // 块数量
            AscendC::UnaryRepeatParams(1, 1, 8, 4));  // 重复参数
        AscendC::PipeBarrier<PIPE_V>();  // 向量管道同步
    }

    /**
     * @brief 应用掩码到输入数据
     * 
     * 将掩码应用到输入数据上，用于实现注意力掩码功能。首先将掩码乘以一个大的负数
     * （-6e4），然后与输入数据相加，实现对被掩码位置的抑制。
     * 
     * @param sUbOffset UB中的起始偏移量（未使用）
     * @param rowNumCurLoop 当前循环处理的行数
     * @param columnNumRound 对齐后的输入列数
     * @param maskColumnRound 对齐后的掩码列数
     * @param addMaskUbOffset 掩码UB的添加偏移量（未使用）
     */
    __aicore__ inline
    void ApplyMask(uint32_t sUbOffset, uint32_t rowNumCurLoop, uint32_t columnNumRound, uint32_t maskColumnRound,
        uint32_t addMaskUbOffset)
    {
        // 计算向量数量
        uint32_t numVecs = (rowNumCurLoop * maskColumnRound + HALF_VECTOR_SIZE - 1) / HALF_VECTOR_SIZE;
        
        // 将掩码乘以一个大的负数：mask = mask * (-6e4)
        // 被掩码的位置（mask=1）会变成-6e4，未掩码的位置（mask=0）保持0
        AscendC::Muls<half, false>(
            maskUbTensor16,         // 输出掩码张量
            maskUbTensor16,         // 输入掩码张量
            (half)-6e4,             // 乘数（-65504）
            (uint64_t)0,            // 输出偏移
            numVecs,                // 向量数量
            AscendC::UnaryRepeatParams(1, 1, 8, 8));  // 重复参数
        AscendC::PipeBarrier<PIPE_V>();  // 向量管道同步
        
        // 根据掩码列数和输入列数是否相等，选择不同的添加方式
        if (maskColumnRound == columnNumRound) {
            // 掩码列数和输入列数相等，直接向量加
            AscendC::Add<half, false>(
                computeUbTensor,     // 输出张量（输入数据+掩码）
                computeUbTensor,     // 输入数据张量
                maskUbTensor16,      // 掩码张量
                (uint64_t)0,         // 输出偏移
                numVecs,             // 向量数量
                AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));  // 重复参数
        } else {
            uint32_t loop = maskColumnRound / HALF_VECTOR_SIZE;
            for (uint32_t i = 0; i < loop; i++) {
                AscendC::Add<half, false>(
                    computeUbTensor[addMaskUbOffset + i * HALF_VECTOR_SIZE],
                    computeUbTensor[addMaskUbOffset + i * HALF_VECTOR_SIZE],
                    maskUbTensor16[i * HALF_VECTOR_SIZE],
                    (uint64_t)0,
                    rowNumCurLoop,
                    AscendC::BinaryRepeatParams(1,
                        1,
                        1,
                        columnNumRound / BLOCK_SIZE,
                        columnNumRound / BLOCK_SIZE,
                        maskColumnRound / BLOCK_SIZE));
            }
            if (maskColumnRound % HALF_VECTOR_SIZE > 0) {
                SetVecMask(maskColumnRound % HALF_VECTOR_SIZE);
                AscendC::Add<half, false>(
                    computeUbTensor[addMaskUbOffset + loop * HALF_VECTOR_SIZE],
                    computeUbTensor[addMaskUbOffset + loop * HALF_VECTOR_SIZE],
                    maskUbTensor16[loop * HALF_VECTOR_SIZE],
                    (uint64_t)0,
                    rowNumCurLoop,
                    AscendC::BinaryRepeatParams(1,
                        1,
                        1,
                        columnNumRound / BLOCK_SIZE,
                        columnNumRound / BLOCK_SIZE,
                        maskColumnRound / BLOCK_SIZE));
                AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    /**
     * @brief 计算本地行最大值
     * 
     * 调用RowmaxTAILTILE函数计算当前块中每行的最大值，并将结果存储在本地最大值UB张量中。
     * 
     * @param sUbOffset UB中的起始偏移量（未使用）
     * @param rowNumCurLoopRound 当前循环处理的行数（已对齐到块大小）
     * @param columnNum 实际列数
     * @param columnNumRound 对齐后的列数
     * @param rowOffset 行偏移量
     */
    __aicore__ inline
    void CalcLocalRowMax(uint32_t sUbOffset, uint32_t rowNumCurLoopRound, uint32_t columnNum, uint32_t columnNumRound,
        uint32_t rowOffset)
    {
        RowmaxTAILTILE(
            computeUbTensor,        // 输入数据UB张量
            lmUbTensor[rowOffset],  // 输出本地最大值UB张量
            tvUbTensor,             // 临时变量UB张量
            rowNumCurLoopRound,     // 行数（已对齐）
            columnNum,              // 实际列数
            columnNumRound);        // 对齐后的列数
    }

    /**
     * @brief 更新全局行最大值
     * 
     * 根据是否为第一个堆叠块，执行不同的操作：
     * - 如果是第一个堆叠块，直接将本地最大值复制到全局最大值
     * - 否则，计算本地最大值和全局最大值的最大值，并更新全局最大值
     * 同时计算指数项，用于后续的行求和更新
     * 
     * @param rowNumCurLoop 当前循环处理的行数
     * @param rowNumCurLoopRound 当前循环处理的行数（已对齐到块大小）
     * @param columnNum 实际列数（未使用）
     * @param columnNumRound 对齐后的列数（未使用）
     * @param dmUbOffsetCurCycle 当前周期的dm UB偏移量
     * @param rowOffset 行偏移量
     * @param isFirstStackTile 是否为第一个堆叠块
     */
    __aicore__ inline
    void UpdateGlobalRowMax(uint32_t rowNumCurLoop, uint32_t rowNumCurLoopRound, uint32_t columnNum,
        uint32_t columnNumRound, uint32_t dmUbOffsetCurCycle, uint32_t rowOffset, uint32_t isFirstStackTile)
    {
        if (isFirstStackTile) {
            // 第一个堆叠块，直接将本地最大值复制到历史最大值
            AscendC::DataCopy(
                hmUbTensor[rowOffset],  // 目标历史最大值UB张量
                lmUbTensor[rowOffset],  // 源本地最大值UB张量
                AscendC::DataCopyParams(1, rowNumCurLoopRound / BLOCK_SIZE, 0, 0));
            AscendC::PipeBarrier<PIPE_V>();  // 向量管道同步
        } else {
            // 不是第一个堆叠块，计算新的最大值
            SetVecMask(rowNumCurLoop);  // 设置向量掩码
            
            // 计算本地最大值和全局最大值的最大值：hm = vmax(lm, gm)
            AscendC::Max<half, false>(
                hmUbTensor[rowOffset],  // 输出历史最大值
                lmUbTensor[rowOffset],  // 输入本地最大值
                gmUbTensor[rowOffset],  // 输入全局最大值
                (uint64_t)0,
                1,
                AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));

            AscendC::PipeBarrier<PIPE_V>();  // 向量管道同步
            
            // 计算差值：dm = gm - hm
            AscendC::Sub<half, false>(
                dmUbTensor[dmUbOffsetCurCycle],  // 输出差值UB张量
                gmUbTensor[rowOffset],           // 输入全局最大值
                hmUbTensor[rowOffset],           // 输入历史最大值
                (uint64_t)0,
                1,
                AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();  // 向量管道同步
            
            // 计算指数：dm = exp(dm)
            AscendC::Exp<half, false>(dmUbTensor[dmUbOffsetCurCycle],
                dmUbTensor[dmUbOffsetCurCycle],
                (uint64_t)0,
                1,
                AscendC::UnaryRepeatParams(1, 1, 8, 8));
        }
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);  // 恢复全1掩码
        AscendC::PipeBarrier<PIPE_V>();  // 向量管道同步
        
        // 更新全局最大值：gm = hm
        AscendC::DataCopy(gmUbTensor[rowOffset],
            hmUbTensor[rowOffset],
            AscendC::DataCopyParams(1, rowNumCurLoopRound / BLOCK_SIZE, 0, 0));
        AscendC::PipeBarrier<PIPE_V>();  // 向量管道同步
    }

    /**
     * @brief 计算指数值
     * 
     * 执行Softmax计算中的指数部分：
     * 1. 将每行最大值广播到与输入数据相同的形状
     * 2. 从输入数据中减去每行最大值，避免数值溢出
     * 3. 对结果计算指数
     * 
     * @param sUbOffset UB中的起始偏移量（未使用）
     * @param rowNumCurLoop 当前循环处理的行数
     * @param rowNumCurLoopRound 当前循环处理的行数（已对齐到块大小）
     * @param columnNum 实际列数
     * @param columnNumRound 对齐后的列数
     * @param rowOffset 行偏移量
     */
    __aicore__ inline
    void CalcExp(uint32_t sUbOffset, uint32_t rowNumCurLoop, uint32_t rowNumCurLoopRound, uint32_t columnNum,
        uint32_t columnNumRound, uint32_t rowOffset)
    {
        // 将每行最大值广播到与输入数据相同的形状：hm_block = expand_to_block(hm)
        AscendC::Brcb(
            tvUbTensor.template ReinterpretCast<uint16_t>(),  // 输出广播结果
            hmUbTensor[rowOffset].template ReinterpretCast<uint16_t>(),  // 输入最大值
            rowNumCurLoopRound / FLOAT_BLOCK_SIZE,
            AscendC::BrcbRepeatParams(1, 8));
        AscendC::PipeBarrier<PIPE_V>();  // 向量管道同步
        
        // 从输入数据中减去每行最大值：ls = ls - hm_block
        // 处理对齐的向量组
        for (uint32_t subIdx = 0; subIdx < columnNum / HALF_VECTOR_SIZE; ++subIdx) {
            AscendC::Sub<half, false>(
                computeUbTensor[subIdx * HALF_VECTOR_SIZE],  // 输出张量
                computeUbTensor[subIdx * HALF_VECTOR_SIZE],  // 输入张量
                tvUbTensor,  // 广播后的最大值张量
                (uint64_t)0,
                rowNumCurLoop,
                AscendC::BinaryRepeatParams(
                    1, 1, 0, columnNumRound / BLOCK_SIZE, columnNumRound / BLOCK_SIZE, 1));
        }
        
        // 处理剩余的非对齐元素
        if (columnNum % HALF_VECTOR_SIZE > 0) {
            SetVecMask(columnNum % HALF_VECTOR_SIZE);  // 设置向量掩码
            AscendC::Sub<half, false>(
                computeUbTensor[columnNum / HALF_VECTOR_SIZE * HALF_VECTOR_SIZE],  // 剩余元素起始位置
                computeUbTensor[columnNum / HALF_VECTOR_SIZE * HALF_VECTOR_SIZE],
                tvUbTensor,
                (uint64_t)0,
                rowNumCurLoop,
                AscendC::BinaryRepeatParams(
                    1, 1, 0, columnNumRound / BLOCK_SIZE, columnNumRound / BLOCK_SIZE, 1));
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);  // 恢复全1掩码
        }
        AscendC::PipeBarrier<PIPE_V>();  // 向量管道同步
        
        // 计算指数：ls = exp(ls)
        AscendC::Exp<half, false>(
            computeUbTensor,  // 输出张量
            computeUbTensor,  // 输入张量
            (uint64_t)0,
            (rowNumCurLoop * columnNumRound + HALF_VECTOR_SIZE - 1) / HALF_VECTOR_SIZE,  // 向量数量
            AscendC::UnaryRepeatParams(1, 1, 8, 8));
        AscendC::PipeBarrier<PIPE_V>();  // 向量管道同步
    }

    /**
     * @brief 计算本地行求和
     * 
     * 根据列数选择不同的行求和函数：
     * - 如果列数等于512，使用RowsumSPECTILE512函数
     * - 否则，使用RowsumTAILTILE函数
     * 计算当前块中每行的和，并将结果存储在本地求和UB张量中。
     * 
     * @param sUbOffset UB中的起始偏移量（未使用）
     * @param rowNumCurLoopRound 当前循环处理的行数（已对齐到块大小）
     * @param columnNum 实际列数
     * @param columnNumRound 对齐后的列数
     * @param rowOffset 行偏移量
     */
    __aicore__ inline
    void CalcLocalRowSum(uint32_t sUbOffset, uint32_t rowNumCurLoopRound, uint32_t columnNum, uint32_t columnNumRound,
        uint32_t rowOffset)
    {
        // 计算每行的和：ll = rowsum(ls)
        if (columnNum == 512U) {
            // 列数等于512，使用专用的512元素分块求和函数
            RowsumSPECTILE512(
                computeUbTensor,        // 输入数据UB张量
                llUbTensor[rowOffset],  // 输出本地求和UB张量
                tvUbTensor,             // 临时变量UB张量
                rowNumCurLoopRound,     // 行数（已对齐）
                columnNum,              // 实际列数
                columnNumRound);        // 对齐后的列数
        } else {
            // 列数不等于512，使用尾部分块求和函数
            RowsumTAILTILE(
                computeUbTensor,        // 输入数据UB张量
                llUbTensor[rowOffset],  // 输出本地求和UB张量
                tvUbTensor,             // 临时变量UB张量
                rowNumCurLoopRound,     // 行数（已对齐）
                columnNum,              // 实际列数
                columnNumRound);        // 对齐后的列数
        }
    }

    /**
     * @brief 更新全局行求和
     * 
     * 根据是否为第一个堆叠块，执行不同的操作：
     * - 如果是第一个堆叠块，直接将本地求和复制到全局求和
     * - 否则，先将全局求和乘以指数项，再加上本地求和，更新全局求和
     * 这是为了在计算多个堆叠块时，正确累积行求和结果。
     * 
     * @param sUbOffset UB中的起始偏移量（未使用）
     * @param rowNumCurLoop 当前循环处理的行数
     * @param rowNumCurLoopRound 当前循环处理的行数（已对齐到块大小）
     * @param dmUbOffsetCurCycle 当前周期的dm UB偏移量
     * @param rowOffset 行偏移量
     * @param isFirstStackTile 是否为第一个堆叠块
     */
    __aicore__ inline
    void UpdateGlobalRowSum(uint32_t sUbOffset, uint32_t rowNumCurLoop, uint32_t rowNumCurLoopRound,
        uint32_t dmUbOffsetCurCycle, uint32_t rowOffset, uint32_t isFirstStackTile)
    {
        if (isFirstStackTile) {
            // 第一个堆叠块，直接将本地求和复制到全局求和：gl = ll
            AscendC::DataCopy(
                glUbTensor[rowOffset],  // 目标全局求和UB张量
                llUbTensor[rowOffset],  // 源本地求和UB张量
                AscendC::DataCopyParams(1, rowNumCurLoopRound / BLOCK_SIZE, 0, 0));
            AscendC::PipeBarrier<PIPE_V>();  // 向量管道同步
        } else {
            // 不是第一个堆叠块，累积求和结果
            SetVecMask(rowNumCurLoop);  // 设置向量掩码
            
            // 先将全局求和乘以指数项：gl = dm * gl
            AscendC::Mul<half, false>(
                glUbTensor[rowOffset],  // 输出全局求和
                dmUbTensor[dmUbOffsetCurCycle],  // 输入指数项
                glUbTensor[rowOffset],  // 输入全局求和
                (uint64_t)0,
                1,
                AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();  // 向量管道同步
            
            // 再加上本地求和：gl = ll + gl
            AscendC::Add<half, false>(
                glUbTensor[rowOffset],  // 输出全局求和
                glUbTensor[rowOffset],  // 输入全局求和（已乘以指数项）
                llUbTensor[rowOffset],  // 输入本地求和
                (uint64_t)0,
                1,
                AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();  // 向量管道同步
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);  // 恢复全1掩码
        }
    }

    /**
     * @brief 将计算结果移动到输出UB张量
     * 
     * 将计算得到的指数值从computeUbTensor移动到lpUbTensor，为后续输出到全局内存做准备。
     * 
     * @param sUbOffset UB中的起始偏移量（未使用）
     * @param rowNumCurLoop 当前循环处理的行数
     * @param columnNumRound 对齐后的列数
     */
    __aicore__ inline
    void MoveP(uint32_t sUbOffset, uint32_t rowNumCurLoop, uint32_t columnNumRound)
    {
        AscendC::DataCopyParams repeatParams;
        repeatParams.blockCount = 1;          // 块数量
        repeatParams.srcStride = 0;           // 源步长
        repeatParams.blockLen = CeilDiv(rowNumCurLoop * columnNumRound, BLOCK_SIZE);  // 块长度
        
        // 执行数据复制：lpUbTensor = computeUbTensor
        AscendC::DataCopy<half>(lpUbTensor, computeUbTensor, repeatParams);
        AscendC::PipeBarrier<PIPE_V>();  // 向量管道同步
    }

    /**
     * @brief 将UB中的输出数据复制到全局内存
     * 
     * 将计算得到的结果从UB张量复制到全局内存中，完成数据输出。
     * 
     * @param gOutput 全局内存中的输出张量
     * @param sUbOffset UB中的起始偏移量（未使用）
     * @param rowNumCurLoop 当前循环处理的行数
     * @param columnNumRound 对齐后的列数
     * @param columnNumPad 填充后的列数
     */
    __aicore__ inline
    void CopyPUbToGm(AscendC::GlobalTensor<ElementOutput> gOutput, uint32_t sUbOffset, uint32_t rowNumCurLoop,
        uint32_t columnNumRound, uint32_t columnNumPad)
    {
        AscendC::DataCopy(
            gOutput,  // 目标全局内存张量
            lpUbTensor,  // 源UB张量
            AscendC::DataCopyParams(
                rowNumCurLoop,  // 行数
                columnNumRound / BLOCK_SIZE,  // 每次复制的块数
                0,  // 填充块数
                (columnNumPad - columnNumRound) / BLOCK_SIZE));  // 源偏移块数
    }

    /**
     * @brief 子核心计算函数
     * 
     * 执行子核心级别的Softmax计算流程，包括：
     * 1. 计算本地行最大值
     * 2. 更新全局行最大值
     * 3. 计算指数值
     * 4. 移动计算结果
     * 5. 计算本地行求和
     * 6. 将结果复制到全局内存
     * 7. 更新全局行求和
     * 
     * @param gOutput 全局内存中的输出张量
     * @param layoutOutput 输出数据的布局
     * @param rowOffset 行偏移量
     * @param isFirstStackTile 是否为第一个堆叠块
     * @param isFirstRowLoop 是否为第一个行循环
     * @param columnNumRound 对齐后的列数
     * @param pingpongFlag 乒乓标志，用于双缓冲机制
     * @param curStackTileMod 当前堆叠块的模
     */
    __aicore__ inline
    void SubCoreCompute(
        AscendC::GlobalTensor<ElementOutput> gOutput, const LayoutOutput &layoutOutput,
        uint32_t rowOffset, uint32_t isFirstStackTile, uint32_t isFirstRowLoop,
        uint32_t columnNumRound, uint32_t pingpongFlag,
        uint32_t curStackTileMod)
    {
        // 获取当前循环的参数
        uint32_t rowNumCurLoop = layoutOutput.shape(0);
        uint32_t rowNumCurLoopRound = RoundUp(rowNumCurLoop, BLOCK_SIZE);
        uint32_t columnNum = layoutOutput.shape(1);
        uint32_t columnNumPad = layoutOutput.stride(0);
        uint32_t sUbOffset = pingpongFlag * MAX_UB_S_ELEM_NUM;
        uint32_t dmUbOffsetCurCycle = curStackTileMod * MAX_ROW_NUM_SUB_CORE + rowOffset;

        // LSE模式下的特殊处理
        if constexpr (LSE_MODE_ == LseModeT::OUT_ONLY) {
            // 在LSE仅输出模式下，tv在最后一个堆叠块中用于传输lse
            if (isFirstStackTile && isFirstRowLoop) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
            }
        }
        
        // 计算本地行最大值
        CalcLocalRowMax(sUbOffset, rowNumCurLoopRound, columnNum, columnNumRound, rowOffset);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);  // 设置事件标志
        
        // 更新全局行最大值
        UpdateGlobalRowMax(
            rowNumCurLoop,
            rowNumCurLoopRound,
            columnNum,
            columnNumRound,
            dmUbOffsetCurCycle,
            rowOffset,
            isFirstStackTile);
        
        // 计算指数值
        CalcExp(sUbOffset, rowNumCurLoop, rowNumCurLoopRound, columnNum, columnNumRound, rowOffset);

        // 等待事件并移动计算结果
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        MoveP(sUbOffset, rowNumCurLoop, columnNumRound);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);  // 设置事件标志

        // 计算本地行求和
        CalcLocalRowSum(sUbOffset, rowNumCurLoopRound, columnNum, columnNumRound, rowOffset);

        // 等待事件并将结果复制到全局内存
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        CopyPUbToGm(gOutput, sUbOffset, rowNumCurLoop, columnNumRound, columnNumPad);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);  // 设置事件标志
        
        // 更新全局行求和
        UpdateGlobalRowSum(
            sUbOffset, rowNumCurLoop, rowNumCurLoopRound, dmUbOffsetCurCycle, rowOffset, isFirstStackTile);
    }

    /**
     * @brief 无掩码版本的Softmax计算入口
     * 
     * 执行无掩码的在线Softmax计算，处理输入数据并生成输出结果。
     * 使用分块和循环处理大型张量，通过乒乓缓冲机制提高性能。
     * 
     * @param gOutput 全局内存中的输出张量
     * @param gInput 全局内存中的输入张量
     * @param layoutOutput 输出数据的布局
     * @param layoutInput 输入数据的布局
     * @param actualBlockShape 实际块形状
     * @param isFirstStackTile 是否为第一个堆叠块
     * @param isLastNoMaskStackTile 是否为最后一个无掩码堆叠块（未使用）
     * @param qSBlockSize 查询序列块大小
     * @param qNBlockSize 查询头数量块大小
     * @param curStackTileMod 当前堆叠块的模
     */
    __aicore__ inline
    void operator()(AscendC::GlobalTensor<ElementOutput> gOutput, AscendC::GlobalTensor<half> gInput,
        const LayoutOutput &layoutOutput, const LayoutInput &layoutInput, GemmCoord actualBlockShape,
        uint32_t isFirstStackTile, uint32_t isLastNoMaskStackTile,
        uint32_t qSBlockSize, uint32_t qNBlockSize, uint32_t curStackTileMod)
    {
        uint32_t rowNum = actualBlockShape.m();
        uint32_t columnNum = actualBlockShape.n();
        uint32_t columnNumRound = RoundUp(columnNum, BLOCK_SIZE);
        uint32_t columnNumPad = layoutInput.stride(0);

        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();

        uint32_t qNSplitSubBlock = qNBlockSize / subBlockNum;
        uint32_t qNThisSubBlock = (qNBlockSize == 1U) ?
            0 : (subBlockIdx == 1U) ? (qNBlockSize - qNSplitSubBlock) : qNSplitSubBlock;
        uint32_t rowSplitSubBlock = (qNBlockSize == 1U) ? (qSBlockSize / 2U) : (qSBlockSize * qNSplitSubBlock);
        uint32_t rowActualThisSubBlock = (subBlockIdx == 1U) ? (rowNum - rowSplitSubBlock) : rowSplitSubBlock;
        uint32_t rowOffsetThisSubBlock = subBlockIdx * rowSplitSubBlock;
        uint32_t maxRowNumPerLoop = MAX_UB_S_ELEM_NUM / columnNumRound;
        uint32_t rowNumTile = RoundDown(maxRowNumPerLoop, BLOCK_SIZE);
        rowNumTile = AscendC::Std::min(rowNumTile, HALF_VECTOR_SIZE);
        uint32_t rowLoopNum = CeilDiv(rowActualThisSubBlock, rowNumTile);

        for (uint32_t rowLoopIdx = 0; rowLoopIdx < rowLoopNum; rowLoopIdx++) {
            uint32_t pingpongFlag = rowLoopIdx % 2U;
            uint32_t rowOffsetCurLoop = rowLoopIdx * rowNumTile;
            uint32_t rowOffsetIoGm = rowOffsetCurLoop + rowOffsetThisSubBlock;
            uint32_t rowNumCurLoop =
                (rowLoopIdx == rowLoopNum - 1U) ? (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;

            int64_t offsetInput = layoutInput.GetOffset(MatrixCoord(rowOffsetIoGm, 0));
            auto gInputCurLoop = gInput[offsetInput];

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            CopySGmToUb(
                gInputCurLoop, (pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound, columnNumPad);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            ScaleS((pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound);

            int64_t offsetOutput = layoutOutput.GetOffset(MatrixCoord(rowOffsetIoGm, 0));
            auto gOutputCurLoop = gOutput[offsetOutput];
            auto layoutOutputCurLoop = layoutOutput.GetTileLayout(MatrixCoord(rowNumCurLoop, columnNum));
            SubCoreCompute(
                gOutputCurLoop,
                layoutOutputCurLoop,
                rowOffsetCurLoop,
                isFirstStackTile,
                (rowLoopIdx == 0U),
                columnNumRound,
                pingpongFlag,
                curStackTileMod);
        }
    }

    /**
     * @brief 有掩码版本的Softmax计算入口
     * 
     * 执行有掩码的在线Softmax计算，处理输入数据和注意力掩码，并生成输出结果。
     * 支持上三角和下三角掩码，用于实现因果注意力等功能。
     * 
     * @param gOutput 全局内存中的输出张量
     * @param gInput 全局内存中的输入张量
     * @param gMask 全局内存中的掩码张量
     * @param layoutOutput 输出数据的布局
     * @param layoutInput 输入数据的布局
     * @param layoutMask 掩码数据的布局
     * @param actualBlockShape 实际块形状
     * @param isFirstStackTile 是否为第一个堆叠块
     * @param qSBlockSize 查询序列块大小
     * @param qNBlockSize 查询头数量块大小
     * @param curStackTileMod 当前堆叠块的模
     * @param qkReady 跨核心同步标志
     * @param triUp 上三角掩码的起始位置
     * @param triDown 下三角掩码的结束位置
     * @param kvSStartIdx KV序列的起始索引
     * @param kvSEndIdx KV序列的结束索引
     */
    __aicore__ inline
    void operator()(AscendC::GlobalTensor<ElementOutput> gOutput, AscendC::GlobalTensor<half> gInput,
        AscendC::GlobalTensor<ElementMask> gMask, const LayoutOutput &layoutOutput, const LayoutInput &layoutInput,
        const LayoutInput &layoutMask, GemmCoord actualBlockShape, uint32_t isFirstStackTile, uint32_t qSBlockSize,
        uint32_t qNBlockSize, uint32_t curStackTileMod, Arch::CrossCoreFlag qkReady, uint32_t triUp, uint32_t triDown,
        uint32_t kvSStartIdx, uint32_t kvSEndIdx)
    {
        uint32_t rowNum = actualBlockShape.m();
        uint32_t columnNum = actualBlockShape.n();
        uint32_t columnNumRound = RoundUp(columnNum, BLOCK_SIZE);
        uint32_t columnNumPad = layoutInput.stride(0);
        uint32_t maskStride = layoutMask.stride(0);
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();

        uint32_t qNSplitSubBlock = qNBlockSize / subBlockNum;
        uint32_t qNThisSubBlock = (qNBlockSize == 1U) ?
            0 : (subBlockIdx == 1U) ? (qNBlockSize - qNSplitSubBlock) : qNSplitSubBlock;
        uint32_t rowSplitSubBlock = (qNBlockSize == 1U) ? (qSBlockSize / 2U) : (qSBlockSize * qNSplitSubBlock);
        uint32_t rowActualThisSubBlock = (subBlockIdx == 1U) ? (rowNum - rowSplitSubBlock) : rowSplitSubBlock;
        uint32_t rowOffsetThisSubBlock = subBlockIdx * rowSplitSubBlock;

        uint32_t tokenNumPerHeadThisSubBlock = AscendC::Std::min(qSBlockSize, rowActualThisSubBlock);

        uint32_t maskOffsetThisSubBlock = (qNBlockSize == 1U) ? rowOffsetThisSubBlock : 0;

        uint32_t gmOffsetMaskRow;
        uint32_t gmOffsetMaskColumn;
        uint32_t maskColumn;
        uint32_t addMaskUbOffset;
        if (triUp >= kvSStartIdx) {
            uint32_t triUpRoundDown = RoundDown(triUp, BLOCK_SIZE);
            gmOffsetMaskRow = triUp - triUpRoundDown;
            gmOffsetMaskColumn = 0U;
            maskColumn = kvSEndIdx - triUpRoundDown;
            addMaskUbOffset = triUpRoundDown - kvSStartIdx;
        } else {
            gmOffsetMaskRow = 0U;
            gmOffsetMaskColumn = kvSStartIdx - triUp;
            maskColumn = columnNum;
            addMaskUbOffset = 0U;
        }
        uint32_t maskColumnRound = RoundUp(maskColumn, BLOCK_SIZE);

        int64_t offsetMask =
            layoutMask.GetOffset(MatrixCoord(gmOffsetMaskRow + maskOffsetThisSubBlock, gmOffsetMaskColumn));
        auto gMaskThisSubBlock = gMask[offsetMask];
        auto layoutMaskThisSubBlock = layoutMask;

        uint32_t maxRowNumPerLoop = MAX_UB_S_ELEM_NUM / columnNumRound;
        uint32_t rowNumTile = RoundDown(maxRowNumPerLoop, BLOCK_SIZE);
        rowNumTile = AscendC::Std::min(rowNumTile, HALF_VECTOR_SIZE);
        uint32_t rowLoopNum = CeilDiv(rowActualThisSubBlock, rowNumTile);

        if (rowActualThisSubBlock == 0U) {
            Arch::CrossCoreWaitFlag(qkReady);
            return;
        }
        Arch::CrossCoreWaitFlag(qkReady);
        for (uint32_t rowLoopIdx = 0; rowLoopIdx < rowLoopNum; rowLoopIdx++) {
            uint32_t pingpongFlag = rowLoopIdx % 2U;
            uint32_t rowOffsetCurLoop = rowLoopIdx * rowNumTile;
            uint32_t rowOffsetIoGm = rowOffsetCurLoop + rowOffsetThisSubBlock;
            uint32_t rowNumCurLoop =
                (rowLoopIdx == rowLoopNum - 1U) ? (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;

            uint32_t proTokenIdx = rowOffsetCurLoop % tokenNumPerHeadThisSubBlock;
            uint32_t proTokenNum = AscendC::Std::min(rowNumCurLoop, (tokenNumPerHeadThisSubBlock - proTokenIdx)) %
                tokenNumPerHeadThisSubBlock;
            uint32_t integralHeadNum = (rowNumCurLoop - proTokenNum) / tokenNumPerHeadThisSubBlock;
            uint32_t epiTokenNum = rowNumCurLoop - proTokenNum - integralHeadNum * tokenNumPerHeadThisSubBlock;

            int64_t offsetInput = layoutInput.GetOffset(MatrixCoord(rowOffsetIoGm, 0));
            auto gInputCurLoop = gInput[offsetInput];
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            CopySGmToUb(
                gInputCurLoop, (pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound, columnNumPad);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            ScaleS((pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound);
            
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
            CopyMaskGmToUb(
                gMaskThisSubBlock,
                maskColumn,
                maskColumnRound,
                maskStride,
                tokenNumPerHeadThisSubBlock,
                proTokenIdx,
                proTokenNum,
                integralHeadNum,
                epiTokenNum);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
            UpCastMask<half, ElementMask>(maskUbTensor16, maskUbTensor, rowNumCurLoop, columnNumRound);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
            ApplyMask(
                (pingpongFlag * MAX_UB_S_ELEM_NUM),
                rowNumCurLoop,
                columnNumRound,
                maskColumnRound,
                addMaskUbOffset);

            // online softmax vectorized compute
            int64_t offsetOutput = layoutOutput.GetOffset(MatrixCoord(rowOffsetIoGm, 0));
            auto gOutputCurLoop = gOutput[offsetOutput];
            auto layoutOutputCurLoop = layoutOutput.GetTileLayout(MatrixCoord(rowNumCurLoop, columnNum));
            SubCoreCompute(
                gOutputCurLoop,
                layoutOutputCurLoop,
                rowOffsetCurLoop,
                isFirstStackTile,
                (rowLoopIdx == 0),
                columnNumRound,
                pingpongFlag,
                curStackTileMod);
        }
    }

private:
    half scaleValue;                     ///< 缩放因子
    AscendC::LocalTensor<half> lsUbTensor;           ///< 输入数据UB张量
    AscendC::LocalTensor<half> computeUbTensor;      ///< 计算中间结果UB张量
    AscendC::LocalTensor<ElementOutput> lpUbTensor;  ///< 输出数据UB张量
    AscendC::LocalTensor<ElementMask> maskUbTensor;  ///< 掩码数据UB张量（原始类型）
    AscendC::LocalTensor<half> maskUbTensor16;       ///< 掩码数据UB张量（half类型）
    AscendC::LocalTensor<half> lmUbTensor;           ///< 本地最大值UB张量
    AscendC::LocalTensor<half> hmUbTensor;           ///< 历史最大值UB张量
    AscendC::LocalTensor<half> gmUbTensor;           ///< 全局最大值UB张量
    AscendC::LocalTensor<half> dmUbTensor;           ///< 差值UB张量
    AscendC::LocalTensor<half> llUbTensor;           ///< 本地求和UB张量
    AscendC::LocalTensor<half> tvUbTensor;           ///< 临时变量UB张量
    AscendC::LocalTensor<half> glUbTensor;           ///< 全局求和UB张量
};

}

#endif