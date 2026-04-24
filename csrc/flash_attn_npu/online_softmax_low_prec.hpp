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

/**
 * @file online_softmax_low_prec.hpp
 * @brief Atlas A2平台的半精度(half)在线Softmax块级Epilogue实现
 *
 * 本文件实现了Flash Attention算法中针对Atlas A2平台优化的半精度在线Softmax块级Epilogue，
 * 是Flash Attention推理计算流水线中QK矩阵乘法之后的核心后处理模块。
 *
 * == 主要实现的算法 ==
 * 与online_softmax.hpp（单精度版本）实现相同的在线Softmax算法，但中间计算使用half精度：
 *   1. 分块处理KV序列，每次处理一个stack tile
 *   2. 维护运行时的行最大值（running max）和行求和（running sum）
 *   3. 每处理一个新的stack tile时，通过指数修正因子更新历史累加结果
 *   4. 最终通过全局行求和归一化得到Softmax输出
 *
 * == 与online_softmax.hpp（单精度版本）的区别 ==
 * - 中间计算使用half精度而非float，减少内存占用和计算开销
 * - S矩阵从L0C（float）加载到UB后立即降精度为half
 * - P矩阵直接以half精度计算和输出，无需额外的降精度步骤
 * - UB空间占用减半，可以处理更大的分块
 * - 归约策略不同：使用WholeReduceMax/WholeReduceSum而非BlockReduceMax/BlockReduceSum
 *   （half类型的BlockReduce硬件支持有限，WholeReduce更高效）
 *
 * == 性能优化考量 ==
 * 半精度版本的主要优势：
 * - UB空间占用减半，可以处理更大的分块
 * - 向量操作吞吐量翻倍（相同带宽下可处理2倍元素）
 * - 减少GM与UB之间的数据传输量
 * 但需要注意：
 * - 半精度的数值范围较小（最大65504），Softmax前需要充分缩放
 * - 累加精度较低，长序列可能导致误差累积
 * - exp函数在half精度下的精度损失
 *
 * == 依赖关系 ==
 * - catlass/catlass.hpp: Catlass库核心头文件
 * - catlass/arch/cross_core_sync.hpp: 跨核心同步原语
 * - catlass/arch/resource.hpp: 硬件资源管理
 * - catlass/epilogue/dispatch_policy.hpp: Epilogue调度策略
 * - catlass/epilogue/tile/tile_copy.hpp: 分块数据拷贝操作
 * - catlass/gemm_coord.hpp: 矩阵坐标系统
 * - catlass/matrix_coord.hpp: 矩阵坐标辅助工具
 * - fa_block.h: Flash Attention分块参数定义
 *
 * == 使用场景 ==
 * 本文件用于Flash Attention推理场景，当中间计算精度为half（半精度）时使用。
 * 典型调用路径: mha_fwd_kvcache.cpp -> FAInferKernel -> EpilogueOnlineSoftmax -> 本文件
 */

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

/**
 * @namespace Catlass::Epilogue::Block
 * @brief Catlass库中Epilogue（收尾操作）的块级实现命名空间
 *
 * 该命名空间包含Flash Attention计算流水线中矩阵乘法后的各种后处理操作的块级实现，
 * 包括在线Softmax计算和输出重缩放。
 */
namespace Catlass::Epilogue::Block {
    /**
     * @brief 半精度在线Softmax的块级Epilogue实现类模板
     *
     * 该类是Flash Attention推理中半精度在线Softmax计算的核心实现，负责在QK矩阵乘法完成后
     * 执行Softmax归一化操作。采用"在线"（online）计算策略，即分块处理KV序列，
     * 逐步维护行最大值和行求和，避免一次性加载完整的注意力矩阵。
     *
     * == 与online_softmax.hpp（单精度版本）的设计差异 ==
     * 1. 数据类型: 所有中间计算使用half而非float
     * 2. 归约策略: 使用WholeReduceMax/WholeReduceSum而非BlockReduceMax/BlockReduceSum
     *    - half类型的BlockReduce硬件支持有限，WholeReduce更高效
     *    - WholeReduceMax直接将整行归约为一个最大值，无需多级归约
     * 3. UB内存布局: 由于half占用空间减半，可以存储更多元素
     *    - lsUbTensor可存储MAX_UB_S_ELEM_NUM=16384个half元素（vs float版本的8192）
     *    - lpUbTensor与lsUbTensor共享空间，通过乒乓缓冲交替使用
     *
     * == UB内存布局 ==
     * - lsUbTensor [0, 2*16KB): 输入S矩阵数据（half类型，支持乒乓缓冲）
     * - computeUbTensor [2*16KB, 4*16KB): 计算中间结果（half类型）
     * - lpUbTensor [4*16KB, 6*16KB): 输出P矩阵数据（half类型，支持乒乓缓冲）
     * - maskUbTensor16 [0, ...): 掩码数据（half类型）
     * - maskUbTensor [11*16KB, ...): 掩码数据（原始类型）
     * - tvUbTensor [10*16KB, 10*16KB+8*1KB): 临时向量，用于广播和归约中间结果
     * - lmUbTensor [10*16KB+8*1KB, ...): 本地行最大值（当前stack tile）
     * - hmUbTensor [10*16KB+9*1KB, ...): 历史行最大值（max(lm, gm)）
     * - gmUbTensor [10*16KB+10*1KB, ...): 全局行最大值（跨stack tile的运行最大值）
     * - llUbTensor [10*16KB+11*1KB, ...): 本地行求和（当前stack tile）
     * - glUbTensor [10*16KB+12*1KB, ...): 全局行求和（跨stack tile的运行求和）
     * - dmUbTensor [10*16KB+13*1KB, ...): 修正因子 exp(old_max - new_max)
     *
     * @tparam OutputType_ 输出类型和布局，包含元素类型（通常为half/bfloat16）和矩阵布局信息
     * @tparam InputType_ 输入类型和布局，包含元素类型（float）和矩阵布局信息
     * @tparam MaskType_ 掩码类型和布局，包含元素类型（通常为int8/uint8）和矩阵布局信息
     * @tparam LSE_MODE_ LSE（Log-Sum-Exp）计算模式:
     *                   - LseModeT::NONE: 不计算LSE
     *                   - LseModeT::OUT_ONLY: 计算并输出LSE值
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
     * @brief 计算尾部分块（TAIL TILE）的行求和（half精度版本）
     *
     * 对列数不等于512或256的尾部分块进行行求和计算。采用WholeReduceSum + Add的
     * 混合策略，与online_softmax.hpp中的float版本采用不同的归约策略。
     *
     * == 与online_softmax.hpp中行求和的区别 ==
     * - float版本使用BlockReduceSum进行多级归约
     * - half版本使用WholeReduceSum + Add的混合策略
     *   - half类型的BlockReduceSum硬件支持有限，WholeReduceSum更高效
     *
     * == 处理流程 ==
     * 1. 如果numElems <= HALF_VECTOR_SIZE(128):
     *    - 直接使用WholeReduceSum归约
     * 2. 如果numElems > HALF_VECTOR_SIZE:
     *    a) 循环将相邻向量组相加: srcUb[0] += srcUb[vmaxIdx*128]
     *    b) 处理尾部非对齐元素: srcUb[0] += srcUb[tail]（使用掩码）
     *    c) 对合并后的结果进行WholeReduceSum归约
     *
     * @param srcUb       源UB张量，存储输入的注意力概率矩阵P（half类型）
     * @param rowsumUb    行求和结果UB张量（half类型）
     * @param tvUbTensor  临时变量UB张量（本函数未使用，保留接口一致性）
     * @param numRowsRound 行数（已对齐到BLOCK_SIZE=16的倍数）
     * @param numElems    实际元素数量（每行的列数，不一定是128的倍数）
     * @param numElemsAligned 对齐后的元素数量（向上对齐到BLOCK_SIZE=16的倍数）
     *
     * @note 算法复杂度: O(numRowsRound * numElemsAligned / HALF_VECTOR_SIZE)
     * @note Add操作会原地修改srcUb的数据，将多个向量组合并到第一个向量组
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
     * @brief 计算尾部分块（TAIL TILE）的行最大值（half精度版本）
     *
     * 对列数不等于512或256的尾部分块进行行最大值计算。采用WholeReduceMax + Max的
     * 混合策略，与online_softmax.hpp中的float版本采用不同的归约策略。
     *
     * == 与online_softmax.hpp中RowmaxTAILTILE的区别 ==
     * 1. float版本使用BlockReduceMax进行多级归约，half版本使用WholeReduceMax + Max
     *    - half类型的BlockReduceMax硬件支持有限，WholeReduceMax更高效
     *    - WholeReduceMax直接将整行归约为一个最大值，但要求行长度不超过向量大小
     * 2. half版本需要先将第一个向量组复制到临时存储，再逐一与剩余向量组取Max
     *    - 因为WholeReduceMax会原地修改输入，需要保留原始数据
     * 3. half版本使用DataCopy复制初始数据，float版本使用BlockReduceMax归约初始数据
     *
     * == 处理流程 ==
     * 1. 如果numElems <= HALF_VECTOR_SIZE(128):
     *    - 直接使用WholeReduceMax归约，设置向量掩码处理非对齐元素
     * 2. 如果numElems > HALF_VECTOR_SIZE:
     *    a) 将第一个HALF_VECTOR_SIZE(128)个元素复制到lsUbTensor
     *    b) 循环与剩余的完整向量组取Max: lsUbTensor = max(lsUbTensor, srcUb[vmaxIdx*128])
     *    c) 处理尾部非对齐元素: lsUbTensor = max(lsUbTensor, srcUb[tail])
     *    d) 对lsUbTensor进行WholeReduceMax归约得到每行最大值
     *
     * @param srcUb       源UB张量，存储输入的注意力分数矩阵S（half类型）
     * @param rowmaxUb    行最大值结果UB张量（half类型）
     * @param tvUbTensor  临时变量UB张量（本函数未使用，保留接口一致性）
     * @param numRowsRound 行数（已对齐到BLOCK_SIZE=16的倍数）
     * @param numElems    实际元素数量（每行的列数，不一定是128的倍数）
     * @param numElemsAligned 对齐后的元素数量（向上对齐到BLOCK_SIZE=16的倍数）
     *
     * @note 算法复杂度: O(numRowsRound * numElemsAligned / HALF_VECTOR_SIZE)
     * @note WholeReduceMax会消耗输入数据，因此需要先复制到临时存储再归约
     */
    __aicore__ inline
    void RowmaxTAILTILE(const AscendC::LocalTensor<half> &srcUb, const AscendC::LocalTensor<half> &rowmaxUb,
        const AscendC::LocalTensor<half> &tvUbTensor, uint32_t numRowsRound, uint32_t numElems,
        uint32_t numElemsAligned)
    {
        // 情况1: 元素数量不超过一个向量大小(128个half元素)
        // 直接使用WholeReduceMax归约，无需分步处理
        if (numElems <= HALF_VECTOR_SIZE) {
            // 设置向量掩码，只对有效的numElems个元素进行归约
            // 当numElems不是BLOCK_SIZE(16)的倍数时，掩码确保不处理无效元素
            SetVecMask(numElems);
            // WholeReduceMax: 将每行的numElems个元素归约为1个最大值
            // ORDER_ONLY_VALUE表示只输出值，不输出索引
            AscendC::WholeReduceMax<half, false>(
                rowmaxUb, srcUb, (int32_t)0, numRowsRound, 1, 1,
                numElemsAligned / BLOCK_SIZE, AscendC::ReduceOrder::ORDER_ONLY_VALUE);
            // 恢复全1掩码，确保后续操作不受影响
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        } else {
            // 情况2: 元素数量超过一个向量大小(128个half元素)
            // 需要分步处理：先逐向量组取Max，再最终归约

            // 步骤1: 将第一个HALF_VECTOR_SIZE(128)个元素复制到临时存储lsUbTensor
            // DataCopyParams参数:
            //   - numRowsRound: 复制的行数
            //   - HALF_VECTOR_SIZE / BLOCK_SIZE: 每行复制的块数(128/16=8块)
            //   - (numElemsAligned - HALF_VECTOR_SIZE) / BLOCK_SIZE: 源行间距(跳过已复制的部分)
            //   - (numElemsAligned - HALF_VECTOR_SIZE) / BLOCK_SIZE: 目标行间距(同上)
            AscendC::DataCopy(
                lsUbTensor,
                srcUb,
                AscendC::DataCopyParams(
                    numRowsRound,
                    HALF_VECTOR_SIZE / BLOCK_SIZE,
                    (numElemsAligned - HALF_VECTOR_SIZE) / BLOCK_SIZE,
                    (numElemsAligned - HALF_VECTOR_SIZE) / BLOCK_SIZE));
            AscendC::PipeBarrier<PIPE_V>();

            // 步骤2: 循环与剩余的完整向量组取Max
            // lsUbTensor = max(lsUbTensor, srcUb[vmaxIdx * HALF_VECTOR_SIZE])
            // 每次迭代将当前最大值与下一个向量组比较，更新最大值
            for (uint32_t vmaxIdx = 1; vmaxIdx < numElems / HALF_VECTOR_SIZE; vmaxIdx++) {
                AscendC::Max<half, false>(
                    lsUbTensor,
                    lsUbTensor,  // 输入和输出都是临时存储，原地更新最大值
                    srcUb[vmaxIdx * HALF_VECTOR_SIZE],  // 当前处理的向量组起始位置
                    (uint64_t)0,
                    numRowsRound,
                    // BinaryRepeatParams: (repeatM, repeatN, repeatK, srcStrideA, srcStrideB, dstStride)
                    // stride = numElemsAligned / BLOCK_SIZE，即每行的块数
                    AscendC::BinaryRepeatParams(
                        1, 1, 1,
                        numElemsAligned / BLOCK_SIZE,
                        numElemsAligned / BLOCK_SIZE,
                        numElemsAligned / BLOCK_SIZE));
                AscendC::PipeBarrier<PIPE_V>();
            }

            // 步骤3: 处理尾部非对齐元素（numElems不是HALF_VECTOR_SIZE的整数倍）
            if (numElems % HALF_VECTOR_SIZE > 0) {
                // 设置向量掩码，只对有效的尾部元素取Max
                SetVecMask(numElems % HALF_VECTOR_SIZE);
                AscendC::Max<half, false>(
                    lsUbTensor,
                    lsUbTensor,
                    srcUb[numElems / HALF_VECTOR_SIZE * HALF_VECTOR_SIZE],  // 尾部元素起始位置
                    (uint64_t)0,
                    numRowsRound,
                    AscendC::BinaryRepeatParams(
                        1, 1, 1,
                        numElemsAligned / BLOCK_SIZE,
                        numElemsAligned / BLOCK_SIZE,
                        numElemsAligned / BLOCK_SIZE));
                AscendC::PipeBarrier<PIPE_V>();
                // 恢复全1掩码
                AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            }

            // 步骤4: 对临时存储中的中间结果进行最终归约
            // lsUbTensor中每行有HALF_VECTOR_SIZE(128)个元素，归约为1个最大值
            AscendC::WholeReduceMax<half, false>(
                rowmaxUb, lsUbTensor, (int32_t)0, numRowsRound, 1, 1,
                numElemsAligned / BLOCK_SIZE, AscendC::ReduceOrder::ORDER_ONLY_VALUE);
        }
        AscendC::PipeBarrier<PIPE_V>();
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
    /**
     * @brief 更新全局行最大值并计算修正因子（half精度版本）
     *
     * 与online_softmax.hpp中的UpdateGlobalRowMax实现相同的算法逻辑，
     * 但使用half精度进行中间计算。关键区别：
     * - DataCopy步长使用BLOCK_SIZE(16)而非FLOAT_BLOCK_SIZE(8)，
     *   因为half类型1个block=16个元素=32字节
     * - Max/Sub/Exp操作使用half类型指令
     *
     * == 性能优化技巧 ==
     * - half类型步长为BLOCK_SIZE(16)，float类型步长为FLOAT_BLOCK_SIZE(8)
     *   因为half占2字节，相同32字节块大小可容纳16个half元素
     * - BinaryRepeatParams(1,1,1,8,8,8)中步长8块=8*16=128个half元素=HALF_VECTOR_SIZE
     *   这与float版本中步长8块=8*8=64个float元素=FLOAT_VECTOR_SIZE对应
     *
     * @param rowNumCurLoop      当前行数
     * @param rowNumCurLoopRound 当前行数（已对齐到BLOCK_SIZE=16）
     * @param columnNum          实际列数
     * @param columnNumRound     对齐后的列数
     * @param dmUbOffsetCurCycle 修正因子dm在UB中的偏移
     * @param rowOffset          行偏移
     * @param isFirstStackTile   是否是第一个stack tile
     */
    __aicore__ inline
    void UpdateGlobalRowMax(uint32_t rowNumCurLoop, uint32_t rowNumCurLoopRound, uint32_t columnNum,
        uint32_t columnNumRound, uint32_t dmUbOffsetCurCycle, uint32_t rowOffset, uint32_t isFirstStackTile)
    {
        if (isFirstStackTile) {
            // 第一个stack tile: hm = lm，无需计算dm
            AscendC::DataCopy(
                hmUbTensor[rowOffset],
                lmUbTensor[rowOffset],
                AscendC::DataCopyParams(1, rowNumCurLoopRound / BLOCK_SIZE, 0, 0));
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            SetVecMask(rowNumCurLoop);
            // hm = max(lm, gm): 取本地最大值和历史全局最大值的较大者
            AscendC::Max<half, false>(
                hmUbTensor[rowOffset],
                lmUbTensor[rowOffset],
                gmUbTensor[rowOffset],
                (uint64_t)0,
                1,
                AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
            // dm = gm - hm: 计算修正因子的指数部分（结果 <= 0）
            AscendC::Sub<half, false>(
                dmUbTensor[dmUbOffsetCurCycle],
                gmUbTensor[rowOffset],
                hmUbTensor[rowOffset],
                (uint64_t)0,
                1,
                AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
            // dm = exp(dm): 计算修正因子
            AscendC::Exp<half, false>(dmUbTensor[dmUbOffsetCurCycle],
                dmUbTensor[dmUbOffsetCurCycle],
                (uint64_t)0,
                1,
                AscendC::UnaryRepeatParams(1, 1, 8, 8));
        }
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        AscendC::PipeBarrier<PIPE_V>();
        // gm = hm: 更新全局最大值
        AscendC::DataCopy(gmUbTensor[rowOffset],
            hmUbTensor[rowOffset],
            AscendC::DataCopyParams(1, rowNumCurLoopRound / BLOCK_SIZE, 0, 0));
        AscendC::PipeBarrier<PIPE_V>();
    }

    /**
     * @brief 计算指数值 P = exp(S - hm)（half精度版本）
     *
     * 与online_softmax.hpp中的CalcExp实现相同的算法逻辑，但使用half精度。
     *
     * == 与float版本CalcExp的区别 ==
     * 1. 操作对象为computeUbTensor而非lsUbTensor
     *    - float版本直接在lsUbTensor上原地操作（S -> S-hm -> exp(S-hm)）
     *    - half版本使用独立的computeUbTensor，因为S需要保留供后续使用
     * 2. 向量粒度为HALF_VECTOR_SIZE(128)而非FLOAT_VECTOR_SIZE(64)
     *    - half类型向量宽度是float的2倍（256字节/2字节=128个元素）
     * 3. Brcb使用uint16_t重解释而非uint32_t
     *    - half占2字节，需要用uint16_t进行重解释
     *
     * == 性能优化技巧 ==
     * - Brcb广播: 将每行1个hm值复制到8个half的位置（1个block=16个half=32字节）
     * - 逐向量Sub: 每次处理HALF_VECTOR_SIZE(128)个half元素，比float版本多1倍吞吐
     * - srcBStride=0: 广播值的步长为0，每行复用同一个tvUbTensor
     * - Exp批量操作: 一次处理整个矩阵，利用half向量的2倍吞吐量
     *
     * @param sUbOffset         S矩阵在UB中的偏移（本函数未使用，保留接口一致性）
     * @param rowNumCurLoop     当前行数
     * @param rowNumCurLoopRound 对齐后的行数
     * @param columnNum         实际列数
     * @param columnNumRound    对齐后的列数
     * @param rowOffset         行偏移（在hmUbTensor中的起始位置）
     */
    __aicore__ inline
    void CalcExp(uint32_t sUbOffset, uint32_t rowNumCurLoop, uint32_t rowNumCurLoopRound, uint32_t columnNum,
        uint32_t columnNumRound, uint32_t rowOffset)
    {
        // 步骤1: 广播行最大值hm到tvUbTensor
        // Brcb将每个half值复制到16个half的位置（1个block=16个half=32字节）
        AscendC::Brcb(
            tvUbTensor.template ReinterpretCast<uint16_t>(),
            hmUbTensor[rowOffset].template ReinterpretCast<uint16_t>(),
            rowNumCurLoopRound / FLOAT_BLOCK_SIZE,
            AscendC::BrcbRepeatParams(1, 8));
        AscendC::PipeBarrier<PIPE_V>();
        // 步骤2: S = S - hm_broadcast（逐向量减法）
        // 按HALF_VECTOR_SIZE(128)为粒度循环处理每一列块
        for (uint32_t subIdx = 0; subIdx < columnNum / HALF_VECTOR_SIZE; ++subIdx) {
            AscendC::Sub<half, false>(
                computeUbTensor[subIdx * HALF_VECTOR_SIZE],
                computeUbTensor[subIdx * HALF_VECTOR_SIZE],
                tvUbTensor,
                (uint64_t)0,
                rowNumCurLoop,
                // BinaryRepeatParams: srcBStride=1块=16个half，表示tvUbTensor每行步进16个half
                AscendC::BinaryRepeatParams(
                    1, 1, 0, columnNumRound / BLOCK_SIZE, columnNumRound / BLOCK_SIZE, 1));
        }
        // 处理尾部非对齐元素
        if (columnNum % HALF_VECTOR_SIZE > 0) {
            SetVecMask(columnNum % HALF_VECTOR_SIZE);
            AscendC::Sub<half, false>(
                computeUbTensor[columnNum / HALF_VECTOR_SIZE * HALF_VECTOR_SIZE],
                computeUbTensor[columnNum / HALF_VECTOR_SIZE * HALF_VECTOR_SIZE],
                tvUbTensor,
                (uint64_t)0,
                rowNumCurLoop,
                AscendC::BinaryRepeatParams(
                    1, 1, 0, columnNumRound / BLOCK_SIZE, columnNumRound / BLOCK_SIZE, 1));
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }
        AscendC::PipeBarrier<PIPE_V>();
        // 步骤3: P = exp(S)，对整个矩阵执行指数运算
        AscendC::Exp<half, false>(
            computeUbTensor,
            computeUbTensor,
            (uint64_t)0,
            (rowNumCurLoop * columnNumRound + HALF_VECTOR_SIZE - 1) / HALF_VECTOR_SIZE,
            AscendC::UnaryRepeatParams(1, 1, 8, 8));
        AscendC::PipeBarrier<PIPE_V>();
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
     * @brief 半精度在线Softmax核心操作符（无掩码版本）
     *
     * 执行半精度在线Softmax算法的核心计算流程，处理一个stack tile的注意力分数。
     * 该函数是OnlineSoftmax模块的入口点，由FAInferKernel在每个stack tile上调用。
     *
     * == 与online_softmax.hpp中operator()的区别 ==
     * 1. 输入数据类型为half而非float
     * 2. 中间计算使用half精度
     * 3. 归约策略使用WholeReduceMax/WholeReduceSum而非BlockReduceMax/BlockReduceSum
     *
     * == 算法流程 ==
     * 对于每个stack tile，执行以下步骤：
     * 1. 从全局内存加载注意力分数S到UB（CopyInputGmToUb）
     * 2. 计算本地行最大值lm（CalcLocalRowMax → RowmaxTAILTILE）
     * 3. 更新全局行最大值gm，计算修正因子dm（UpdateGlobalRowMax）
     * 4. 计算指数值P = exp(S - gm)（CalcExp）
     * 5. 计算本地行求和ll（CalcLocalRowSum）
     * 6. 将P矩阵写回全局内存（CopyPUbToGm）
     * 7. 更新全局行求和gl（UpdateGlobalRowSum）
     *
     * == 子块并行 ==
     * 当subBlockNum > 1时，将行数分为两个子块并行处理：
     * - 子块0处理前半部分行
     * - 子块1处理后半部分行
     *
     * @param gOutput              输出全局内存张量（P矩阵，Softmax概率值）
     * @param gInput               输入全局内存张量（S矩阵，注意力分数，half类型）
     * @param layoutOutput         输出布局信息
     * @param layoutInput          输入布局信息
     * @param actualBlockShape     实际块形状（M/N/K维度）
     * @param isFirstStackTile     是否是第一个stack tile（1=是，0=否）
     * @param isLastNoMaskStackTile 是否是最后一个无掩码stack tile（1=是，0=否）
     * @param qSBlockSize          Q的S维度块大小（分组注意力头的组数）
     * @param qNBlockSize          Q的N维度块大小（每组内的token数）
     * @param curStackTileMod      当前stack tile的模值
     *
     * @note 第一个stack tile不需要计算修正因子dm
     * @note 最后一个stack tile后需要执行归一化（在RescaleO中完成）
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