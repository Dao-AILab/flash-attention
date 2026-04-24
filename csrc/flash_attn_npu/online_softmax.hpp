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

#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_NO_MASK_HPP_T
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_NO_MASK_HPP_T

/**
 * @file online_softmax.hpp
 * @brief Atlas A2平台的单精度(float)在线Softmax块级Epilogue实现
 *
 * 本文件实现了Flash Attention算法中针对Atlas A2平台优化的单精度在线Softmax块级Epilogue，
 * 是Flash Attention推理计算流水线中QK矩阵乘法之后的核心后处理模块。
 *
 * == 主要实现的算法 ==
 * 实现了在线Softmax算法（Online Softmax / Flash Attention Softmax），该算法的核心思想是：
 *   1. 分块处理KV序列，每次处理一个stack tile
 *   2. 维护运行时的行最大值（running max）和行求和（running sum）
 *   3. 每处理一个新的stack tile时，通过指数修正因子（dm = exp(old_max - new_max)）更新历史累加结果
 *   4. 最终通过全局行求和归一化得到Softmax输出
 *
 * 算法流程（以第i个stack tile为例）：
 *   a) 计算本地行最大值: lm = rowmax(S_i)
 *   b) 更新全局行最大值: hm = max(lm, gm), dm = exp(gm - hm)
 *   c) 计算指数值: P_i = exp(S_i - hm)
 *   d) 计算本地行求和: ll = rowsum(P_i)
 *   e) 更新全局行求和: gl = dm * gl + ll
 *   f) 输出P_i到全局内存（供PV矩阵乘法使用）
 *
 * == 依赖关系 ==
 * - catlass/catlass.hpp: Catlass库核心头文件，提供基础类型和宏定义
 * - catlass/arch/cross_core_sync.hpp: 跨核心同步原语，用于多核间通信
 * - catlass/arch/resource.hpp: 硬件资源管理，提供UB/L1/L0缓冲区分配
 * - catlass/epilogue/dispatch_policy.hpp: Epilogue调度策略定义
 * - catlass/epilogue/tile/tile_copy.hpp: 分块数据拷贝操作
 * - catlass/gemm_coord.hpp: 矩阵坐标系统
 * - catlass/matrix_coord.hpp: 矩阵坐标辅助工具
 * - fa_block.h: Flash Attention分块参数定义
 *
 * == 使用场景 ==
 * 本文件用于Flash Attention推理场景，当中间计算精度为float（单精度）时使用。
 * 与online_softmax_low_prec.hpp（half精度版本）配合，根据精度需求选择不同的模板特化。
 * 典型调用路径: mha_fwd_kvcache.cpp -> FAInferKernel -> EpilogueOnlineSoftmax -> 本文件
 *
 * == 性能优化要点 ==
 * - 针对不同列数（512/256/其他）采用不同的归约策略（SPECTILE vs TAILTILE）
 * - 使用BlockReduceMax/BlockReduceSum进行硬件加速的多级归约
 * - 通过乒乓缓冲（ping-pong buffer）实现计算与数据搬移的流水线重叠
 * - 使用向量掩码（Vector Mask）处理非对齐的尾部元素
 * - 利用Brcb指令进行行最大值的广播，避免逐元素操作
 */

#include "catlass/catlass.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "fa_block.h"

/**
 * @namespace Catlass::Epilogue::Block
 * @brief Catlass库中Epilogue（收尾操作）的块级实现命名空间
 *
 * 该命名空间包含Flash Attention计算流水线中矩阵乘法后的各种后处理操作的块级实现，
 * 包括：
 * - 在线Softmax计算（online_softmax.hpp / online_softmax_low_prec.hpp）
 * - 输出重缩放（rescale_o.hpp / rescale_o_low_prec.hpp）
 *
 * "块级"（Block）意味着这些实现以分块（tile）为单位处理数据，
 * 适配昇腾NPU的多级缓存层次结构（GM -> L1 -> L0 -> UB），
 * 通过精心设计的数据流和同步策略最大化硬件利用率。
 */
namespace Catlass::Epilogue::Block {

/**
 * @brief 单精度在线Softmax的块级Epilogue实现类模板
 *
 * 该类是Flash Attention推理中在线Softmax计算的核心实现，负责在QK矩阵乘法完成后
 * 执行Softmax归一化操作。采用"在线"（online）计算策略，即分块处理KV序列，
 * 逐步维护行最大值和行求和，避免一次性加载完整的注意力矩阵。
 *
 * == 设计思路 ==
 * 在线Softmax算法的核心挑战是在不知道完整行数据的情况下正确计算Softmax。
 * 解决方案是维护两个运行时状态：
 * - gm（全局最大值）: 当前已处理的所有stack tile中每行的最大值
 * - gl（全局求和）: 当前已处理的所有stack tile中每行的指数求和
 *
 * 每处理一个新的stack tile时：
 * 1. 计算新块的本地最大值lm
 * 2. 用lm和gm的较大值更新全局最大值hm
 * 3. 计算修正因子dm = exp(gm - hm)，用于调整历史累加结果
 * 4. 计算新块的指数值P = exp(S - hm)
 * 5. 计算新块的本地求和ll
 * 6. 更新全局求和gl = dm * gl + ll
 *
 * == UB内存布局 ==
 * UB（Unified Buffer）空间分配策略：
 * - lsUbTensor [0, 4*16KB): 输入S矩阵数据（支持乒乓缓冲）
 * - lpUbTensor [4*16KB, 8*16KB): 输出P矩阵数据（降精度后）
 * - maskUbTensor [4*16KB, ...): 掩码数据（原始类型）
 * - maskUbTensor16 [11*16KB, ...): 掩码数据（half类型）
 * - maskUbTensor32 [4*16KB, ...): 掩码数据（float类型）
 * - tvUbTensor [10*16KB, 10*16KB+8*1KB): 临时向量，用于广播和归约中间结果
 * - lmUbTensor [10*16KB+9*1KB, ...): 本地行最大值（当前stack tile）
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
    class OutputType_,          ///< 输出类型模板参数
    class InputType_,           ///< 输入类型模板参数
    class MaskType_,            ///< 掩码类型模板参数
    LseModeT LSE_MODE_>         ///< 对数和计算模式
class BlockEpilogue<
    EpilogueAtlasA2OnlineSoftmaxT<LSE_MODE_, float>,  ///< Atlas A2在线Softmax调度策略
    OutputType_,
    InputType_,
    MaskType_>
{
public:
    using DispatchPolicy = EpilogueAtlasA2OnlineSoftmaxT<LSE_MODE_, float>;  ///< 调度策略类型
    using ArchTag = typename DispatchPolicy::ArchTag;                        ///< 架构标签类型
    using ElementOutput = typename OutputType_::Element;                     ///< 输出元素类型
    using ElementInput = typename InputType_::Element;                       ///< 输入元素类型
    using ElementMask = typename MaskType_::Element;                         ///< 掩码元素类型

    using LayoutOutput = typename OutputType_::Layout;  ///< 输出布局类型
    using LayoutInput = typename InputType_::Layout;    ///< 输入布局类型
    using LayoutMask = typename MaskType_::Layout;      ///< 掩码布局类型

    static constexpr LseModeT LSE_MODE = DispatchPolicy::LSE_MODE;  ///< 对数和计算模式

    // UB (Unified Buffer) 相关常量定义
    // 以下常量定义了Atlas A2架构上向量计算单元的硬件参数，
    // 这些值由硬件规格决定，直接影响数据对齐和向量操作的粒度。
    static constexpr uint32_t BLOCK_SIZE_IN_BYTE = 32;        ///< 内存块大小（字节），Atlas A2 DMA传输的最小对齐单元
    static constexpr uint32_t REPEAT_SIZE_IN_BYTE = 256;      ///< 重复操作大小（字节），向量计算指令单次重复处理的数据量
    static constexpr uint32_t FLOAT_BLOCK_SIZE = 8;           ///< 浮点数块大小，float类型DMA传输的基本单元（32字节/4字节=8个元素）
    static constexpr uint32_t FLOAT_VECTOR_SIZE = 64;         ///< 浮点数向量大小，向量计算单元单次可处理的float元素数量（256字节/4字节）
    static constexpr uint32_t HALF_VECTOR_SIZE = 128;         ///< 半精度向量大小，向量计算单元单次可处理的half元素数量（256字节/2字节）
    static constexpr uint32_t BLOCK_SIZE = 16;                ///< 基本块大小，矩阵计算中数据对齐的基本粒度（16个元素）
    static constexpr uint32_t UB_UINT8_VECTOR_SIZE = 1024;    ///< UB中uint8向量大小（字节），即1KB
    static constexpr uint32_t UB_UINT8_BLOCK_SIZE = 16384;    ///< UB中uint8块大小（字节），即16KB
    static constexpr uint32_t VECTOR_SIZE = 128;              ///< 通用向量大小，half类型的向量元素数量
    static constexpr uint32_t MAX_UB_S_ELEM_NUM = 8192;       ///< UB中S矩阵元素的最大数量，限制单次处理的float数据量（8192*4B=32KB）

    // 归约相关常量定义
    // 这些常量用于控制BlockReduceMax/BlockReduceSum操作的掩码和参数
    static constexpr uint32_t REDUCE_UB_SIZE = 1024;          ///< 归约操作UB临时空间大小（元素数），用于存储BlockReduce的中间结果
    static constexpr uint32_t ROW_OPS_SPEC_MASK_32 = 32;      ///< 行操作特定掩码(32)，用于SPECTILE256第二级归约的向量掩码
    static constexpr uint32_t ROW_OPS_SPEC_MASK_4 = 4;        ///< 行操作特定掩码(4)，用于SPECTILE256第三级归约的BlockReduce掩码
    static constexpr uint32_t MAX_ROW_NUM_SUB_CORE = 256;     ///< 子核心可处理的最大行数，限制dmUbTensor的存储容量
    static constexpr int64_t UB_FLOAT_LINE_SIZE = 64;         ///< UB浮点行大小，单行float数据的最大元素数

    /**
     * @brief 构造函数
     * 
     * @param resource 架构资源
     * @param scaleValue_ 缩放值
     */
    __aicore__ inline
    BlockEpilogue(Arch::Resource<ArchTag> &resource, float scaleValue_)
    {
        // 分配UB空间的偏移量定义
        constexpr uint32_t LS_UB_TENSOR_OFFSET = 0;
        constexpr uint32_t LP_UB_TENSOR_OFFSET = 4 * UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t MASK_UB_TENSOR_OFFSET = 4 * UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t MASK32_UB_TENSOR_OFFSET = 4 * UB_UINT8_BLOCK_SIZE;

        constexpr uint32_t TV_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t LM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 8 * UB_UINT8_VECTOR_SIZE;

        constexpr uint32_t HM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 9 * UB_UINT8_VECTOR_SIZE;
        constexpr uint32_t GM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 10 * UB_UINT8_VECTOR_SIZE;
        constexpr uint32_t LL_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 11 * UB_UINT8_VECTOR_SIZE;
        constexpr uint32_t GL_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 12 * UB_UINT8_VECTOR_SIZE;
        constexpr uint32_t DM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 13 * UB_UINT8_VECTOR_SIZE;

        constexpr uint32_t MASK16_UB_TENSOR_OFFSET = 11 * UB_UINT8_BLOCK_SIZE;

        // 初始化成员变量和UB缓冲区
        scaleValue = scaleValue_;
        lsUbTensor = resource.ubBuf.template GetBufferByByte<float>(LS_UB_TENSOR_OFFSET);
        lpUbTensor = resource.ubBuf.template GetBufferByByte<ElementOutput>(LP_UB_TENSOR_OFFSET);
        maskUbTensor = resource.ubBuf.template GetBufferByByte<ElementMask>(MASK_UB_TENSOR_OFFSET);
        maskUbTensor16 = resource.ubBuf.template GetBufferByByte<half>(MASK16_UB_TENSOR_OFFSET);
        maskUbTensor32 = resource.ubBuf.template GetBufferByByte<float>(MASK32_UB_TENSOR_OFFSET);
        lmUbTensor = resource.ubBuf.template GetBufferByByte<float>(LM_UB_TENSOR_OFFSET);
        hmUbTensor = resource.ubBuf.template GetBufferByByte<float>(HM_UB_TENSOR_OFFSET);
        gmUbTensor = resource.ubBuf.template GetBufferByByte<float>(GM_UB_TENSOR_OFFSET);
        dmUbTensor = resource.ubBuf.template GetBufferByByte<float>(DM_UB_TENSOR_OFFSET);
        llUbTensor = resource.ubBuf.template GetBufferByByte<float>(LL_UB_TENSOR_OFFSET);
        tvUbTensor = resource.ubBuf.template GetBufferByByte<float>(TV_UB_TENSOR_OFFSET);
        glUbTensor = resource.ubBuf.template GetBufferByByte<float>(GL_UB_TENSOR_OFFSET);
    }

    /**
     * @brief 析构函数
     */
    __aicore__ inline
    ~BlockEpilogue() {}

    /**
     * @brief 获取两个值中的较小值
     * 
     * @tparam T 数据类型
     * @param a 第一个值
     * @param b 第二个值
     * @return T 较小的值
     */
    template <typename T>
    __aicore__ inline T Min(T a, T b)
    {
        return (a > b) ? b : a;
    }

    /**
     * @brief 设置向量掩码
     * 
     * @param len 长度
     */
    __aicore__ inline
    void SetVecMask(int32_t len)
    {
        uint64_t mask = 0;
        uint64_t one = 1;
        uint64_t temp = len % FLOAT_VECTOR_SIZE;
        for (int64_t i = 0; i < temp; i++) {
            mask |= one << i;
        }

        // 根据长度设置不同的向量掩码
        if (len == VECTOR_SIZE || len == 0) {
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        } else if (len >= FLOAT_VECTOR_SIZE) {
            AscendC::SetVectorMask<int8_t>(mask, (uint64_t)-1);
        } else {
            AscendC::SetVectorMask<int8_t>(0x0, mask);
        }
    }

    __aicore__ inline
    void SetBlockReduceMask(int32_t len)
    {
        if (len > 8 || len < 1) {
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            return;
        }
        uint64_t subMask = ((uint64_t)1 << len) - 1;
        uint64_t maskValue = (subMask << 48) + (subMask << 32) + (subMask << 16) + subMask + (subMask << 56) +
                             (subMask << 40) + (subMask << 24) + (subMask << 8);
        AscendC::SetVectorMask<int8_t>(maskValue, maskValue);
    }

    __aicore__ inline
    void RowsumSPECTILE512(const AscendC::LocalTensor<float> &srcUb, const AscendC::LocalTensor<float> &rowsumUb,
        const AscendC::LocalTensor<float> &tvUbTensor, uint32_t numRowsRound, uint32_t numElems,
        uint32_t numElemsAligned)
    {
        AscendC::BlockReduceSum<float, false>(
            tvUbTensor,
            srcUb,
            numRowsRound * numElemsAligned / FLOAT_VECTOR_SIZE,
            0, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::BlockReduceSum<float, false>(
            tvUbTensor[REDUCE_UB_SIZE],
            tvUbTensor,
            numRowsRound * numElemsAligned / FLOAT_BLOCK_SIZE / FLOAT_VECTOR_SIZE,
            0, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::BlockReduceSum<float, false>(
            rowsumUb,
            tvUbTensor[REDUCE_UB_SIZE],
            numRowsRound * numElemsAligned / FLOAT_VECTOR_SIZE / FLOAT_VECTOR_SIZE,
            0, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline
    void RowsumSPECTILE256(const AscendC::LocalTensor<float> &srcUb, const AscendC::LocalTensor<float> &rowsumUb,
        const AscendC::LocalTensor<float> &tvUbTensor, uint32_t numRowsRound, uint32_t numElems,
        uint32_t numElemsAligned)
    {
        AscendC::BlockReduceSum<float, false>(
            tvUbTensor,
            srcUb,
            numRowsRound * numElemsAligned / FLOAT_VECTOR_SIZE,
            0, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        SetVecMask(ROW_OPS_SPEC_MASK_32);
        AscendC::BlockReduceSum<float, false>(
            tvUbTensor[REDUCE_UB_SIZE],
            tvUbTensor,
            numRowsRound,
            0, 1, 1, 4);
        AscendC::PipeBarrier<PIPE_V>();
        SetBlockReduceMask(ROW_OPS_SPEC_MASK_4);
        AscendC::BlockReduceSum<float, false>(
            rowsumUb,
            tvUbTensor[REDUCE_UB_SIZE],
            CeilDiv(numRowsRound * FLOAT_BLOCK_SIZE, FLOAT_VECTOR_SIZE),
            0, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    }

    __aicore__ inline
    void RowsumTAILTILE(const AscendC::LocalTensor<float> &srcUb, const AscendC::LocalTensor<float> &rowsumUb,
        const AscendC::LocalTensor<float> &tvUbTensor, uint32_t numRowsRound, uint32_t numElems,
        uint32_t numElemsAligned)
    {
        if (numElems >= FLOAT_VECTOR_SIZE) {
            AscendC::BlockReduceSum<float, false>(
                tvUbTensor,
                srcUb,
                numRowsRound,
                0, 1, 1, numElemsAligned / FLOAT_BLOCK_SIZE);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::BlockReduceSum<float, false>(
                rowsumUb,
                tvUbTensor,
                CeilDiv(numRowsRound * FLOAT_BLOCK_SIZE, FLOAT_VECTOR_SIZE),
                0, 1, 1, 8);
            AscendC::PipeBarrier<PIPE_V>();
            for (uint64_t rowSumIdx = 1; rowSumIdx < (uint64_t)numElems / FLOAT_VECTOR_SIZE; ++rowSumIdx) {
                AscendC::BlockReduceSum<float, false>(
                    tvUbTensor,
                    srcUb[rowSumIdx * FLOAT_VECTOR_SIZE],
                    numRowsRound,
                    0, 1, 1, numElemsAligned / FLOAT_BLOCK_SIZE);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::BlockReduceSum<float, false>(
                    tvUbTensor[REDUCE_UB_SIZE],
                    tvUbTensor,
                    CeilDiv(numRowsRound * FLOAT_BLOCK_SIZE, FLOAT_VECTOR_SIZE),
                    0, 1, 1, 8);
                AscendC::PipeBarrier<PIPE_V>();
                SetVecMask(numRowsRound);
                AscendC::Add<float, false>(
                    rowsumUb,
                    rowsumUb,
                    tvUbTensor[REDUCE_UB_SIZE],
                    (uint64_t)0,
                    1,
                    AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            }
        }
        if (numElems % FLOAT_VECTOR_SIZE > 0) {
            SetVecMask(numElems % FLOAT_VECTOR_SIZE);
            AscendC::BlockReduceSum<float, false>(
                tvUbTensor,
                srcUb[numElems / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                numRowsRound,
                0, 1, 1, numElemsAligned / FLOAT_BLOCK_SIZE);
            AscendC::PipeBarrier<PIPE_V>();
            SetBlockReduceMask(CeilDiv(numElems % FLOAT_VECTOR_SIZE, FLOAT_BLOCK_SIZE));
            if (numElems < FLOAT_VECTOR_SIZE) {
                AscendC::BlockReduceSum<float, false>(
                    rowsumUb,
                    tvUbTensor,
                    CeilDiv(numRowsRound * FLOAT_BLOCK_SIZE, FLOAT_VECTOR_SIZE),
                    0, 1, 1, 8);
                AscendC::PipeBarrier<PIPE_V>();
            } else {
                AscendC::BlockReduceSum<float, false>(
                    tvUbTensor[REDUCE_UB_SIZE],
                    tvUbTensor,
                    CeilDiv(numRowsRound * FLOAT_BLOCK_SIZE, FLOAT_VECTOR_SIZE),
                    0, 1, 1, 8);
                AscendC::PipeBarrier<PIPE_V>();
                SetVecMask(numRowsRound);
                AscendC::Add<float, false>(
                    rowsumUb,
                    rowsumUb,
                    tvUbTensor[REDUCE_UB_SIZE],
                    (uint64_t)0,
                    1,
                    AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }
    }

    /**
     * @brief 计算512元素SPEC TILE的行最大值（三级BlockReduce归约）
     *
     * 对列数为512的特定分块进行高效的行最大值计算。采用三级BlockReduceMax归约策略，
     * 充分利用Atlas A2的硬件归约指令，将512个元素逐步归约到每行一个最大值。
     *
     * == 归约策略 ==
     * 三级归约过程（以单行为例，512个float元素）：
     *   第一级: 512元素 / FLOAT_VECTOR_SIZE(64) = 8个BlockReduceMax操作
     *           每个操作将64个元素归约为8个中间结果（BlockReduce粒度为FLOAT_BLOCK_SIZE=8）
     *           结果: 8 * 8 = 64个中间值存入tvUbTensor
     *   第二级: 64个中间值 / FLOAT_BLOCK_SIZE(8) = 8个BlockReduceMax操作
     *           每个操作将8个元素归约为1个结果
     *           结果: 8个中间值存入tvUbTensor[REDUCE_UB_SIZE]
     *   第三级: 8个中间值归约为1个最终结果
     *           结果: 每行1个最大值存入rowmaxUb
     *
     * == 为什么采用三级归约而非WholeReduceMax ==
     * BlockReduceMax是Atlas A2上的硬件加速指令，相比WholeReduceMax：
     * - BlockReduceMax可以利用Cube单元进行并行归约，吞吐量更高
     * - 对于512这种恰好是2的幂次的大列数，三级归约的效率优于WholeReduceMax
     * - 但对于较小的列数（如256），需要使用掩码来处理非标准对齐
     *
     * @param srcUb       源UB张量，存储输入的注意力分数矩阵S（形状: [numRowsRound, 512]）
     * @param rowmaxUb    行最大值结果UB张量（形状: [numRowsRound]）
     * @param tvUbTensor  临时变量UB张量，用于存储归约中间结果（需至少2*REDUCE_UB_SIZE空间）
     * @param numRowsRound 行数（已对齐到FLOAT_BLOCK_SIZE=8的倍数）
     * @param numElems    实际元素数量（每行的列数，本函数应为512）
     * @param numElemsAligned 对齐后的元素数量（通常等于numElems或向上对齐到BLOCK_SIZE=16的倍数）
     *
     * @note 算法复杂度: O(numRowsRound * numElemsAligned)，但实际硬件执行为三级流水线归约
     * @note 每级归约后必须调用PipeBarrier<PIPE_V>()确保向量计算完成后再进行下一级
     */
    __aicore__ inline
    void RowmaxSPECTILE512(const AscendC::LocalTensor<float> &srcUb, const AscendC::LocalTensor<float> &rowmaxUb,
        const AscendC::LocalTensor<float> &tvUbTensor, uint32_t numRowsRound, uint32_t numElems,
        uint32_t numElemsAligned)
    {
        // 第一级归约: 将每FLOAT_VECTOR_SIZE(64)个元素归约为FLOAT_BLOCK_SIZE(8)个中间结果
        // 归约次数 = numRowsRound * numElemsAligned / FLOAT_VECTOR_SIZE
        // 例如: numRowsRound=16, numElemsAligned=512 -> 16*512/64 = 128次归约
        AscendC::BlockReduceMax<float, false>(
            tvUbTensor,
            srcUb,
            numRowsRound * numElemsAligned / FLOAT_VECTOR_SIZE,
            0, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();

        // 第二级归约: 将第一级的中间结果进一步归约
        // 归约次数 = numRowsRound * numElemsAligned / (FLOAT_BLOCK_SIZE * FLOAT_VECTOR_SIZE)
        // 例如: 16*512/(8*64) = 16次归约
        AscendC::BlockReduceMax<float, false>(
            tvUbTensor[REDUCE_UB_SIZE],
            tvUbTensor,
            numRowsRound * numElemsAligned / FLOAT_BLOCK_SIZE / FLOAT_VECTOR_SIZE,
            0, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();

        // 第三级归约: 将第二级的中间结果归约为每行一个最终最大值
        // 归约次数 = numRowsRound * numElemsAligned / (FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE)
        // 例如: 16*512/(64*64) = 2次归约
        AscendC::BlockReduceMax<float, false>(
            rowmaxUb,
            tvUbTensor[REDUCE_UB_SIZE],
            numRowsRound * numElemsAligned / FLOAT_VECTOR_SIZE / FLOAT_VECTOR_SIZE,
            0, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
    }

    /**
     * @brief 计算256元素SPEC TILE的行最大值（三级BlockReduce归约 + 掩码处理）
     *
     * 对列数为256的特定分块进行行最大值计算。与RowmaxSPECTILE512类似采用三级归约，
     * 但由于256不是512的整数倍关系，第二级和第三级归约需要使用向量掩码来处理
     * 非标准对齐的中间结果。
     *
     * == 与RowmaxSPECTILE512的区别 ==
     * 1. 第二级归约使用SetVecMask(ROW_OPS_SPEC_MASK_32=32)限制有效向量长度
     *    - 因为256/64=4个向量组，归约后中间结果只有4*8=32个值，不足一个完整向量(64)
     *    - 需要掩码确保只对有效的32个元素进行归约
     * 2. 第三级归约使用SetBlockReduceMask(ROW_OPS_SPEC_MASK_4=4)限制BlockReduce粒度
     *    - 32个中间值归约后只有4个值，需要掩码确保只对4个block进行归约
     * 3. 最后恢复全1掩码AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1)
     *
     * @param srcUb       源UB张量，存储输入的注意力分数矩阵S（形状: [numRowsRound, 256]）
     * @param rowmaxUb    行最大值结果UB张量（形状: [numRowsRound]）
     * @param tvUbTensor  临时变量UB张量，用于存储归约中间结果
     * @param numRowsRound 行数（已对齐到FLOAT_BLOCK_SIZE=8的倍数）
     * @param numElems    实际元素数量（每行的列数，本函数应为256）
     * @param numElemsAligned 对齐后的元素数量
     *
     * @note 算法复杂度: O(numRowsRound * numElemsAligned)
     * @note 掩码操作必须成对使用：设置掩码 -> 执行操作 -> 恢复全1掩码
     */
    __aicore__ inline
    void RowmaxSPECTILE256(const AscendC::LocalTensor<float> &srcUb, const AscendC::LocalTensor<float> &rowmaxUb,
        const AscendC::LocalTensor<float> &tvUbTensor, uint32_t numRowsRound, uint32_t numElems,
        uint32_t numElemsAligned)
    {
        // 第一级归约: 与SPECTILE512相同，将每64个元素归约为8个中间结果
        AscendC::BlockReduceMax<float, false>(
            tvUbTensor,
            srcUb,
            numRowsRound * numElemsAligned / FLOAT_VECTOR_SIZE,
            0, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();

        // 第二级归约: 使用向量掩码处理非标准对齐
        // SetVecMask(32)限制向量操作只处理前32个元素
        SetVecMask(ROW_OPS_SPEC_MASK_32);
        AscendC::BlockReduceMax<float, false>(
            tvUbTensor[REDUCE_UB_SIZE],
            tvUbTensor,
            numRowsRound,
            0, 1, 1, 4);  // repeatStride=4，适配256列的中间结果数量
        AscendC::PipeBarrier<PIPE_V>();

        // 第三级归约: 使用BlockReduce掩码处理少量中间结果
        // SetBlockReduceMask(4)限制BlockReduce只对4个block进行归约
        SetBlockReduceMask(ROW_OPS_SPEC_MASK_4);
        AscendC::BlockReduceMax<float, false>(
            rowmaxUb,
            tvUbTensor[REDUCE_UB_SIZE],
            CeilDiv(numRowsRound * FLOAT_BLOCK_SIZE, FLOAT_VECTOR_SIZE),
            0, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();

        // 恢复全1掩码，确保后续操作不受影响
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    }

    /**
     * @brief 计算尾部分块（TAIL TILE）的行最大值
     *
     * 对列数不等于512或256的尾部分块进行行最大值计算。采用BlockReduceMax + Max的
     * 混合策略，先对每个FLOAT_VECTOR_SIZE(64)的向量组进行BlockReduce归约，
     * 再将各组结果用逐元素Max操作合并。
     *
     * == 与SPECTILE系列的区别 ==
     * SPECTILE系列（512/256）使用纯三级BlockReduce归约，因为列数恰好是2的幂次，
     * 可以完美适配硬件归约指令的粒度。而TAILTILE需要处理任意列数，因此：
     * 1. 先对每个64元素的向量组进行BlockReduceMax归约到8个中间值
     * 2. 再将8个中间值BlockReduceMax归约到1个值
     * 3. 对多个向量组的归约结果，使用逐元素Max操作合并
     * 4. 对尾部非对齐元素，使用向量掩码处理后单独归约并合并
     *
     * == 处理流程 ==
     * 1. 如果numElems >= FLOAT_VECTOR_SIZE:
     *    a) 对第一个向量组进行BlockReduceMax归约，得到初始rowmaxUb
     *    b) 对剩余的完整向量组逐一归约，并与rowmaxUb取Max
     * 2. 如果numElems % FLOAT_VECTOR_SIZE > 0（存在非对齐尾部）:
     *    a) 设置向量掩码只处理有效元素
     *    b) 对尾部元素进行BlockReduceMax归约
     *    c) 如果之前已有结果，与rowmaxUb取Max；否则直接作为结果
     * 3. 恢复全1掩码
     *
     * @param srcUb       源UB张量，存储输入的注意力分数矩阵S
     * @param rowmaxUb    行最大值结果UB张量
     * @param tvUbTensor  临时变量UB张量，用于存储归约中间结果
     * @param numRowsRound 行数（已对齐到FLOAT_BLOCK_SIZE=8的倍数）
     * @param numElems    实际元素数量（每行的列数，不一定是64的倍数）
     * @param numElemsAligned 对齐后的元素数量（向上对齐到FLOAT_BLOCK_SIZE=8的倍数）
     *
     * @note 算法复杂度: O(numRowsRound * numElemsAligned / FLOAT_VECTOR_SIZE)
     * @note 当numElems < FLOAT_VECTOR_SIZE时，直接使用BlockReduceMax归约，无需逐元素Max合并
     */
    __aicore__ inline
    void RowmaxTAILTILE(const AscendC::LocalTensor<float> &srcUb, const AscendC::LocalTensor<float> &rowmaxUb,
        const AscendC::LocalTensor<float> &tvUbTensor, uint32_t numRowsRound, uint32_t numElems,
        uint32_t numElemsAligned)
    {
        // 阶段1: 处理完整的FLOAT_VECTOR_SIZE(64)向量组
        if (numElems >= FLOAT_VECTOR_SIZE) {
            // 对第一个向量组进行两级BlockReduceMax归约
            // 第一级: 将每FLOAT_BLOCK_SIZE(8)个元素归约为1个中间结果
            AscendC::BlockReduceMax<float, false>(
                tvUbTensor,
                srcUb,
                numRowsRound,
                0, 1, 1, numElemsAligned / FLOAT_BLOCK_SIZE);
            AscendC::PipeBarrier<PIPE_V>();

            // 第二级: 将中间结果归约为每行1个最大值
            AscendC::BlockReduceMax<float, false>(
                rowmaxUb,
                tvUbTensor,
                CeilDiv(numRowsRound * FLOAT_BLOCK_SIZE, FLOAT_VECTOR_SIZE),
                0, 1, 1, 8);
            AscendC::PipeBarrier<PIPE_V>();

            // 对剩余的完整向量组逐一归约，并与已有结果取Max
            for (uint64_t rowmax_idx = 1; rowmax_idx < (uint64_t)numElems / FLOAT_VECTOR_SIZE; ++rowmax_idx) {
                // 对当前向量组进行两级BlockReduceMax归约
                AscendC::BlockReduceMax<float, false>(
                    tvUbTensor,
                    srcUb[rowmax_idx * FLOAT_VECTOR_SIZE],
                    numRowsRound,
                    0, 1, 1, numElemsAligned / FLOAT_BLOCK_SIZE);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::BlockReduceMax<float, false>(
                    tvUbTensor[REDUCE_UB_SIZE],
                    tvUbTensor,
                    CeilDiv(numRowsRound * FLOAT_BLOCK_SIZE, FLOAT_VECTOR_SIZE),
                    0, 1, 1, 8);
                AscendC::PipeBarrier<PIPE_V>();

                // 与已有最大值取Max: rowmaxUb = max(rowmaxUb, tvUbTensor[REDUCE_UB_SIZE])
                SetVecMask(numRowsRound);
                AscendC::Max<float, false>(rowmaxUb,
                    rowmaxUb,
                    tvUbTensor[REDUCE_UB_SIZE],
                    (uint64_t)0,
                    1,
                    AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            }
        }

        // 阶段2: 处理尾部非对齐元素（numElems不是FLOAT_VECTOR_SIZE的整数倍）
        if (numElems % FLOAT_VECTOR_SIZE > 0) {
            // 设置向量掩码，只处理有效的尾部元素
            SetVecMask(numElems % FLOAT_VECTOR_SIZE);

            // 对尾部元素进行第一级BlockReduceMax归约
            AscendC::BlockReduceMax<float, false>(
                tvUbTensor,
                srcUb[numElems / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                numRowsRound,
                0, 1, 1, numElemsAligned / FLOAT_BLOCK_SIZE);
            AscendC::PipeBarrier<PIPE_V>();

            // 设置BlockReduce掩码，适配尾部元素数量
            SetBlockReduceMask(CeilDiv(numElems % FLOAT_VECTOR_SIZE, FLOAT_BLOCK_SIZE));

            if (numElems < FLOAT_VECTOR_SIZE) {
                // 整行元素不足一个向量大小，直接归约得到最终结果
                AscendC::BlockReduceMax<float, false>(rowmaxUb,
                    tvUbTensor,
                    CeilDiv(numRowsRound * FLOAT_BLOCK_SIZE, FLOAT_VECTOR_SIZE),
                    0, 1, 1, 8);
                AscendC::PipeBarrier<PIPE_V>();
            } else {
                // 已有完整向量组的归约结果，需要与尾部归约结果取Max
                AscendC::BlockReduceMax<float, false>(tvUbTensor[REDUCE_UB_SIZE],
                    tvUbTensor,
                    CeilDiv(numRowsRound * FLOAT_BLOCK_SIZE, FLOAT_VECTOR_SIZE),
                    0, 1, 1, 8);
                AscendC::PipeBarrier<PIPE_V>();

                // rowmaxUb = max(rowmaxUb, tvUbTensor[REDUCE_UB_SIZE])
                SetVecMask(numRowsRound);
                AscendC::Max<float, false>(rowmaxUb,
                    rowmaxUb,
                    tvUbTensor[REDUCE_UB_SIZE],
                    (uint64_t)0,
                    1,
                    AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
                AscendC::PipeBarrier<PIPE_V>();
            }
            // 恢复全1掩码
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }
    }

    /**
     * @brief 从全局内存加载S矩阵到UB
     *
     * 将QK矩阵乘法的输出S（注意力分数）从全局内存（GM）加载到UB的lsUbTensor中。
     * 这是Softmax计算的第一步，后续所有操作都基于UB中的数据进行。
     *
     * == 性能优化技巧 ==
     * - DataCopyParams参数说明:
     *   - rowNumCurLoop: 复制的行数
     *   - columnNumRound / FLOAT_BLOCK_SIZE: 每行复制的block数（1block=8个float=32字节）
     *   - (columnNumPad - columnNumRound) / FLOAT_BLOCK_SIZE: 源行间距（跳过padding部分）
     *   - 0: 目标行间距（UB中数据紧凑排列，无padding）
     * - 只复制有效列数columnNumRound，跳过GM中的padding列，减少数据传输量
     *
     * @param gInput         全局内存中的S矩阵张量
     * @param sUbOffset      S矩阵在UB中的偏移（乒乓缓冲索引，0或MAX_UB_S_ELEM_NUM）
     * @param rowNumCurLoop  当前行数
     * @param columnNumRound 对齐后的列数（向上对齐到FLOAT_BLOCK_SIZE=8的倍数）
     * @param columnNumPad   填充后的列数（包含GM中的padding，通常等于stride）
     */
    __aicore__ inline
    void CopySGmToUb(
        AscendC::GlobalTensor<ElementInput> gInput,
        uint32_t sUbOffset,
        uint32_t rowNumCurLoop,
        uint32_t columnNumRound,
        uint32_t columnNumPad)
    {
        AscendC::DataCopy(
            lsUbTensor[sUbOffset],
            gInput,
            AscendC::DataCopyParams(
                rowNumCurLoop, columnNumRound / FLOAT_BLOCK_SIZE,
                (columnNumPad - columnNumRound) / FLOAT_BLOCK_SIZE, 0));
    }

    __aicore__ inline
    void CopyMaskGmToUb(
        AscendC::GlobalTensor<ElementMask> gMask,
        uint32_t columnNum, uint32_t columnNumRound,
        uint32_t maskStride, uint32_t tokenNumPerHead,
        uint32_t proTokenIdx, uint32_t proTokenNum,
        uint32_t integralHeadNum, uint32_t epiTokenNum)
    {
        uint32_t innerUbRowOffset = 0;
        if (proTokenNum != 0) {
            AscendC::DataCopyPad(
                maskUbTensor[innerUbRowOffset], gMask[proTokenIdx * maskStride],
                AscendC::DataCopyExtParams(
                    proTokenNum, columnNum * sizeof(ElementMask),
                    (maskStride - columnNum) * sizeof(ElementMask), 0, 0),
                AscendC::DataCopyPadExtParams<ElementMask>(false, 0, 0, 0));
            innerUbRowOffset += proTokenNum * columnNumRound;
        }
        for (uint32_t headIdx = 0; headIdx < integralHeadNum; headIdx++) {
            AscendC::DataCopyPad(
                maskUbTensor[innerUbRowOffset], gMask,
                AscendC::DataCopyExtParams(
                    tokenNumPerHead, columnNum * sizeof(ElementMask),
                    (maskStride - columnNum) * sizeof(ElementMask), 0, 0),
                AscendC::DataCopyPadExtParams<ElementMask>(false, 0, 0, 0));
            innerUbRowOffset += tokenNumPerHead * columnNumRound;
        }
        if (epiTokenNum != 0) {
            AscendC::DataCopyPad(
                maskUbTensor[innerUbRowOffset], gMask,
                AscendC::DataCopyExtParams(
                    epiTokenNum, columnNum * sizeof(ElementMask),
                    (maskStride - columnNum) * sizeof(ElementMask), 0, 0),
                AscendC::DataCopyPadExtParams<ElementMask>(false, 0, 0, 0));
        }
    }

    __aicore__ inline
    void ScaleS(uint32_t sUbOffset, uint32_t rowNumCurLoop, uint32_t columnNumRound)
    {
        AscendC::Muls<float, false>(
            lsUbTensor[sUbOffset],
            lsUbTensor[sUbOffset],
            scaleValue,
            (uint64_t)0,
            CeilDiv(rowNumCurLoop * columnNumRound, FLOAT_VECTOR_SIZE),
            AscendC::UnaryRepeatParams(1, 1, 8, 8));

        AscendC::PipeBarrier<PIPE_V>();
    }

    template<typename ElementMaskDst, typename ElementMaskSrc>
    __aicore__ inline 
    void UpCastMask(
        const AscendC::LocalTensor<ElementMaskDst> &maskUbTensorDst,
        const AscendC::LocalTensor<ElementMaskSrc> &maskUbTensorSrc,
        uint32_t rowNumCurLoop,
        uint32_t columnNumRound)
    {
        AscendC::Cast<ElementMaskDst, ElementMaskSrc, false>(
            maskUbTensorDst, maskUbTensorSrc, AscendC::RoundMode::CAST_NONE, (uint64_t)0,
            CeilDiv(rowNumCurLoop * columnNumRound, (uint32_t)(REPEAT_SIZE_IN_BYTE / sizeof(ElementMaskDst))),
            AscendC::UnaryRepeatParams(1, 1, 8, 4));
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline
    void ApplyMask(uint32_t sUbOffset, uint32_t rowNumCurLoop, uint32_t columnNumRound, uint32_t maskColumnRound,
        uint32_t addMaskUbOffset)
    {
        AscendC::Muls<float, false>(
            maskUbTensor32,
            maskUbTensor32,
            (float)-3e38,
            (uint64_t)0,
            CeilDiv(rowNumCurLoop * maskColumnRound, FLOAT_VECTOR_SIZE),
            AscendC::UnaryRepeatParams(1, 1, 8, 8));
        AscendC::PipeBarrier<PIPE_V>();
        if (maskColumnRound == columnNumRound) {
            AscendC::Add<float, false>(
                lsUbTensor[sUbOffset],
                lsUbTensor[sUbOffset],
                maskUbTensor32,
                (uint64_t)0,
                CeilDiv(rowNumCurLoop * maskColumnRound, FLOAT_VECTOR_SIZE),
                AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
        } else {
            uint32_t loop = maskColumnRound / FLOAT_VECTOR_SIZE;
            for (uint32_t i = 0; i < loop; i++) {
                AscendC::Add<float, false>(lsUbTensor[sUbOffset][addMaskUbOffset + i * FLOAT_VECTOR_SIZE],
                    lsUbTensor[sUbOffset][addMaskUbOffset + i * FLOAT_VECTOR_SIZE],
                    maskUbTensor32[i * FLOAT_VECTOR_SIZE],
                    (uint64_t)0,
                    rowNumCurLoop,
                    AscendC::BinaryRepeatParams(
                        1, 1, 1,
                        columnNumRound / FLOAT_BLOCK_SIZE,
                        columnNumRound / FLOAT_BLOCK_SIZE,
                        maskColumnRound / FLOAT_BLOCK_SIZE));
            }
            if (maskColumnRound % FLOAT_VECTOR_SIZE > 0) {
                SetVecMask(maskColumnRound % FLOAT_VECTOR_SIZE);
                AscendC::Add<float, false>(lsUbTensor[sUbOffset][addMaskUbOffset + loop * FLOAT_VECTOR_SIZE],
                    lsUbTensor[sUbOffset][addMaskUbOffset + loop * FLOAT_VECTOR_SIZE],
                    maskUbTensor32[loop * FLOAT_VECTOR_SIZE],
                    (uint64_t)0,
                    rowNumCurLoop,
                    AscendC::BinaryRepeatParams(
                        1, 1, 1,
                        columnNumRound / FLOAT_BLOCK_SIZE,
                        columnNumRound / FLOAT_BLOCK_SIZE,
                        maskColumnRound / FLOAT_BLOCK_SIZE));
                AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    /**
     * @brief 计算本地行最大值（Local Row Max）
     *
     * 根据列数选择最优的行最大值计算策略：
     * - columnNum == 512: 使用RowmaxSPECTILE512（三级BlockReduce归约）
     * - columnNum == 256: 使用RowmaxSPECTILE256（三级BlockReduce归约 + 掩码）
     * - 其他: 使用RowmaxTAILTILE（BlockReduce + Max混合策略）
     *
     * == 策略选择原因 ==
     * 512和256是Atlas A2硬件BlockReduce指令的高效工作点，可以直接使用三级归约。
     * 其他列数无法完美适配BlockReduce的粒度，需要使用更灵活的混合策略。
     *
     * @param sUbOffset         S矩阵在UB中的偏移（乒乓缓冲索引）
     * @param rowNumCurLoopRound 当前行数（已对齐到FLOAT_BLOCK_SIZE）
     * @param columnNum         实际列数（当前stack tile的序列长度）
     * @param columnNumRound    对齐后的列数
     * @param rowOffset         行偏移（在lmUbTensor中的起始位置）
     */
    __aicore__ inline
    void CalcLocalRowMax(uint32_t sUbOffset, uint32_t rowNumCurLoopRound, uint32_t columnNum, uint32_t columnNumRound,
        uint32_t rowOffset)
    {
        if (columnNum == 512) {
            RowmaxSPECTILE512(
                lsUbTensor[sUbOffset],
                lmUbTensor[rowOffset],
                tvUbTensor,
                rowNumCurLoopRound,
                columnNum,
                columnNumRound);
        } else if (columnNum == 256) {
            RowmaxSPECTILE256(
                lsUbTensor[sUbOffset],
                lmUbTensor[rowOffset],
                tvUbTensor,
                rowNumCurLoopRound,
                columnNum,
                columnNumRound);
        } else {
            RowmaxTAILTILE(
                lsUbTensor[sUbOffset],
                lmUbTensor[rowOffset],
                tvUbTensor,
                rowNumCurLoopRound,
                columnNum,
                columnNumRound);
        }
    }

    /**
     * @brief 更新全局行最大值并计算修正因子
     *
     * 在线Softmax算法的核心步骤之一。当处理新的stack tile时，需要将本地行最大值lm
     * 与全局行最大值gm比较，计算修正因子dm用于调整历史累加结果。
     *
     * == 计算流程 ==
     * 情况1 - 第一个stack tile (isFirstStackTile=1):
     *   hm = lm （直接复制，因为没有历史最大值）
     *   dm = 不需要计算（没有历史结果需要修正）
     *
     * 情况2 - 非第一个stack tile (isFirstStackTile=0):
     *   hm = max(lm, gm)  ← 取新旧最大值的较大者
     *   dm = exp(gm - hm) ← 修正因子，用于缩放历史累加结果
     *   注意: 当gm > hm时不可能（因为hm=max(lm,gm)），所以dm <= 1.0
     *   当lm > gm时，dm = exp(gm - lm) < 1.0，历史结果需要缩小
     *   当lm <= gm时，dm = exp(0) = 1.0，历史结果不需要调整
     *
     * 最后: gm = hm （更新全局最大值供下一个stack tile使用）
     *
     * == 性能优化技巧 ==
     * - 使用SetVecMask(rowNumCurLoop)只处理有效行，避免越界
     * - Max/Sub/Exp都是向量操作，一次处理8个float元素（FLOAT_BLOCK_SIZE=8）
     * - BinaryRepeatParams(1,1,1,8,8,8)表示：repeat=1, 源A步长8块, 源B步长8块, 目标步长8块
     *   这里步长8块=8*8=64个元素=FLOAT_VECTOR_SIZE，正好是一行的行最大值间距
     * - DataCopy用于hm/gm之间的批量复制，比逐元素赋值更高效
     *
     * @param rowNumCurLoop      当前行数（可能非对齐）
     * @param rowNumCurLoopRound 当前行数（已对齐到FLOAT_BLOCK_SIZE=8）
     * @param columnNum          实际列数
     * @param columnNumRound     对齐后的列数
     * @param dmUbOffsetCurCycle 修正因子dm在UB中的偏移（乒乓缓冲索引）
     * @param rowOffset          行偏移（在lm/hm/gm张量中的起始位置）
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
                AscendC::DataCopyParams(1, rowNumCurLoopRound / FLOAT_BLOCK_SIZE, 0, 0));
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            SetVecMask(rowNumCurLoop);
            // hm = max(lm, gm): 取本地最大值和历史全局最大值的较大者
            AscendC::Max<float, false>(
                hmUbTensor[rowOffset],
                lmUbTensor[rowOffset],
                gmUbTensor[rowOffset],
                (uint64_t)0,
                1,
                AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
            // dm = gm - hm: 计算修正因子的指数部分（结果 <= 0）
            AscendC::Sub<float, false>(
                dmUbTensor[dmUbOffsetCurCycle],
                gmUbTensor[rowOffset],
                hmUbTensor[rowOffset],
                (uint64_t)0,
                1,
                AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
            // dm = exp(dm): 计算修正因子，用于缩放历史累加结果O和gl
            AscendC::Exp<float, false>(
                dmUbTensor[dmUbOffsetCurCycle],
                dmUbTensor[dmUbOffsetCurCycle],
                (uint64_t)0,
                1,
                AscendC::UnaryRepeatParams(1, 1, 8, 8));
        }
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        AscendC::PipeBarrier<PIPE_V>();
        // gm = hm: 更新全局最大值，供下一个stack tile使用
        AscendC::DataCopy(
            gmUbTensor[rowOffset],
            hmUbTensor[rowOffset],
            AscendC::DataCopyParams(1, rowNumCurLoopRound / FLOAT_BLOCK_SIZE, 0, 0));
        AscendC::PipeBarrier<PIPE_V>();
    }

    /**
     * @brief 计算指数值 P = exp(S - hm)
     *
     * 在线Softmax算法的核心步骤：计算注意力概率矩阵P。
     * 将S矩阵减去行最大值hm后取指数，得到Softmax的分子部分。
     *
     * == 计算流程 ==
     * 1. 广播行最大值: 使用Brcb指令将hm（每行1个值）广播为每行FLOAT_VECTOR_SIZE(64)个值
     *    - Brcb是Atlas A2的高效广播指令，将1个值复制到整个向量宽度
     *    - 结果存入tvUbTensor，每行有64个相同的hm值
     * 2. 逐向量减法: S = S - hm_broadcast
     *    - 按FLOAT_VECTOR_SIZE(64)为粒度循环处理
     *    - BinaryRepeatParams(1,1,0,...)中srcBStride=0表示每行复用同一个hm广播值
     *    - 尾部非对齐元素使用SetVecMask处理
     * 3. 指数运算: P = exp(S)
     *    - 对整个S矩阵执行Exp操作
     *    - 结果直接覆盖lsUbTensor（S矩阵不再需要）
     *
     * == 性能优化技巧 ==
     * - Brcb广播: 比逐行DataCopy+重复更高效，1条指令完成1行到64个元素的广播
     * - 逐向量Sub: 每次处理64个元素，充分利用向量计算单元
     * - srcBStride=0: 广播值的步长为0，每行复用同一个tvUbTensor，减少数据搬移
     * - Exp批量操作: 一次处理整个矩阵，比逐行Exp更高效
     * - 尾部掩码: 避免为非对齐尾部分配完整向量的额外计算
     *
     * @param sUbOffset         S矩阵在UB中的偏移（乒乓缓冲索引）
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
        // Brcb将每个float值复制到8个float的位置（1个block=8个float）
        // 例如: hm=[3.0, 5.0] -> tv=[3.0*8, 5.0*8] = [3,3,3,3,3,3,3,3, 5,5,5,5,5,5,5,5]
        AscendC::Brcb(
            tvUbTensor.template ReinterpretCast<uint32_t>(),
            hmUbTensor[rowOffset].template ReinterpretCast<uint32_t>(),
            rowNumCurLoopRound / FLOAT_BLOCK_SIZE,
            AscendC::BrcbRepeatParams(1, 8));
        AscendC::PipeBarrier<PIPE_V>();
        // 步骤2: S = S - hm_broadcast（逐向量减法）
        // 按FLOAT_VECTOR_SIZE(64)为粒度循环处理每一列块
        for (uint32_t subIdx = 0; subIdx < columnNum / FLOAT_VECTOR_SIZE; ++subIdx) {
            AscendC::Sub<float, false>(
                lsUbTensor[sUbOffset][subIdx * FLOAT_VECTOR_SIZE],
                lsUbTensor[sUbOffset][subIdx * FLOAT_VECTOR_SIZE],
                tvUbTensor,
                (uint64_t)0,
                rowNumCurLoop,
                // BinaryRepeatParams: (repeatM, repeatN, repeatK, srcAStride, srcBStride, dstStride)
                // srcBStride=1块=8个float，表示tvUbTensor每行步进8个float（一个广播值）
                // srcAStride/dstStride=columnNumRound/FLOAT_BLOCK_SIZE块，表示S矩阵每行的步长
                AscendC::BinaryRepeatParams(
                    1, 1, 0, columnNumRound / FLOAT_BLOCK_SIZE, columnNumRound / FLOAT_BLOCK_SIZE, 1));
        }
        // 处理尾部非对齐元素（columnNum不是FLOAT_VECTOR_SIZE的整数倍）
        if (columnNum % FLOAT_VECTOR_SIZE > 0) {
            SetVecMask(columnNum % FLOAT_VECTOR_SIZE);
            AscendC::Sub<float, false>(
                lsUbTensor[sUbOffset][columnNum / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                lsUbTensor[sUbOffset][columnNum / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                tvUbTensor,
                (uint64_t)0,
                rowNumCurLoop,
                AscendC::BinaryRepeatParams(
                    1, 1, 0, columnNumRound / FLOAT_BLOCK_SIZE, columnNumRound / FLOAT_BLOCK_SIZE, 1));
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }
        AscendC::PipeBarrier<PIPE_V>();
        // 步骤3: P = exp(S)，对整个矩阵执行指数运算
        // CeilDiv(rowNumCurLoop * columnNumRound, FLOAT_VECTOR_SIZE)计算需要处理的向量数
        AscendC::Exp<float, false>(
            lsUbTensor[sUbOffset],
            lsUbTensor[sUbOffset],
            (uint64_t)0,
            CeilDiv(rowNumCurLoop * columnNumRound, FLOAT_VECTOR_SIZE),
            AscendC::UnaryRepeatParams(1, 1, 8, 8));
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline
    void CalcLocalRowSum(uint32_t sUbOffset, uint32_t rowNumCurLoopRound, uint32_t columnNum, uint32_t columnNumRound,
        uint32_t rowOffset)
    {
        // *** ll = rowsum(ls32)
        if (columnNum == 512) {
            RowsumSPECTILE512(
                lsUbTensor[sUbOffset],
                llUbTensor[rowOffset],
                tvUbTensor,
                rowNumCurLoopRound,
                columnNum,
                columnNumRound);
        } else if (columnNum == 256) {
            RowsumSPECTILE256(
                lsUbTensor[sUbOffset],
                llUbTensor[rowOffset],
                tvUbTensor,
                rowNumCurLoopRound,
                columnNum,
                columnNumRound);
        } else {
            RowsumTAILTILE(
                lsUbTensor[sUbOffset],
                llUbTensor[rowOffset],
                tvUbTensor,
                rowNumCurLoopRound,
                columnNum,
                columnNumRound);
        }
    }

    /**
     * @brief 更新全局行求和
     *
     * 在线Softmax算法的核心步骤之一。将当前stack tile的本地行求和ll
     * 累加到全局行求和gl中，同时使用修正因子dm调整历史累加结果。
     *
     * == 计算流程 ==
     * 情况1 - 第一个stack tile (isFirstStackTile=1):
     *   gl = ll （直接复制，因为没有历史求和）
     *
     * 情况2 - 非第一个stack tile (isFirstStackTile=0):
     *   gl = dm * gl  ← 用修正因子缩放历史求和（因为最大值变了，指数值也需要调整）
     *   gl = ll + gl  ← 加上当前stack tile的本地求和
     *   合并: gl = dm * gl_old + ll
     *
     * == 数学推导 ==
     * 设第i个stack tile处理后，全局求和应为:
     *   gl_new = sum(exp(S_1 - hm)) + sum(exp(S_2 - hm)) + ... + sum(exp(S_i - hm))
     * 其中hm是当前的全局最大值。
     * 对于之前计算的gl_old，其基于旧的最大值gm:
     *   gl_old = sum(exp(S_1 - gm)) + ... + sum(exp(S_{i-1} - gm))
     * 需要调整: gl_old * exp(gm - hm) = sum(exp(S_1 - hm)) + ... + sum(exp(S_{i-1} - hm))
     * 即: dm * gl_old = 修正后的历史求和
     * 所以: gl_new = dm * gl_old + ll
     *
     * == 性能优化技巧 ==
     * - Mul+Add两步操作比先广播dm再乘加更高效（dm和gl都是1D向量，长度相同）
     * - BinaryRepeatParams(1,1,1,8,8,8)步长8块=64个元素，适配行最大值的存储间距
     * - SetVecMask只处理有效行，避免越界访问
     *
     * @param sUbOffset          S矩阵在UB中的偏移（乒乓缓冲索引）
     * @param rowNumCurLoop      当前行数
     * @param rowNumCurLoopRound 对齐后的行数
     * @param dmUbOffsetCurCycle 修正因子dm在UB中的偏移
     * @param rowOffset          行偏移
     * @param isFirstStackTile   是否是第一个stack tile
     */
    __aicore__ inline
    void UpdateGlobalRowSum(uint32_t sUbOffset, uint32_t rowNumCurLoop, uint32_t rowNumCurLoopRound,
        uint32_t dmUbOffsetCurCycle, uint32_t rowOffset, uint32_t isFirstStackTile)
    {
        if (isFirstStackTile) {
            // 第一个stack tile: gl = ll，直接复制
            AscendC::DataCopy(
                glUbTensor[rowOffset],
                llUbTensor[rowOffset],
                AscendC::DataCopyParams(1, rowNumCurLoopRound / FLOAT_BLOCK_SIZE, 0, 0));
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            SetVecMask(rowNumCurLoop);
            // gl = dm * gl: 用修正因子缩放历史求和
            AscendC::Mul<float, false>(
                glUbTensor[rowOffset],
                dmUbTensor[dmUbOffsetCurCycle],
                glUbTensor[rowOffset],
                (uint64_t)0,
                1,
                AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
            // gl = ll + gl: 加上当前stack tile的本地求和
            AscendC::Add<float, false>(
                glUbTensor[rowOffset],
                glUbTensor[rowOffset],
                llUbTensor[rowOffset],
                (uint64_t)0,
                1,
                AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }
    }

    /**
     * @brief 将float精度的P矩阵降精度为half/bfloat16
     *
     * 在线Softmax计算完成后，P矩阵以float精度存储在lsUbTensor中。
     * 为了节省全局内存带宽和存储空间，需要将其降精度为ElementOutput类型（half/bfloat16）。
     *
     * == 性能优化技巧 ==
     * - Cast指令可以批量转换整个矩阵，比逐元素转换更高效
     * - bfloat16使用CAST_RINT舍入模式（四舍五入到偶数），half使用CAST_NONE
     * - UnaryRepeatParams(1,1,4,8)中srcStride=4块表示float源每行步进4*8=32个元素
     *   dstStride=8块表示half目标每行步进8*8=64个元素（half元素数是float的2倍）
     *
     * @param sUbOffset      P矩阵在UB中的偏移（乒乓缓冲索引）
     * @param rowNumCurLoop  当前行数
     * @param columnNumRound 对齐后的列数
     */
    __aicore__ inline
    void DownCastP(uint32_t sUbOffset, uint32_t rowNumCurLoop, uint32_t columnNumRound)
    {
        // *** lp = castfp32to16(ls)
        if (std::is_same<ElementOutput, bfloat16_t>::value) {
            AscendC::Cast<ElementOutput, float, false>(
                lpUbTensor[sUbOffset],
                lsUbTensor[sUbOffset],
                AscendC::RoundMode::CAST_RINT,
                (uint64_t)0,
                CeilDiv(rowNumCurLoop * columnNumRound, FLOAT_VECTOR_SIZE),
                AscendC::UnaryRepeatParams(1, 1, 4, 8));
        } else {
            AscendC::Cast<ElementOutput, float, false>(
                lpUbTensor[sUbOffset],
                lsUbTensor[sUbOffset],
                AscendC::RoundMode::CAST_NONE,
                (uint64_t)0,
                CeilDiv(rowNumCurLoop * columnNumRound, FLOAT_VECTOR_SIZE),
                AscendC::UnaryRepeatParams(1, 1, 4, 8));
        }
    }

    __aicore__ inline
    void CopyPUbToGm(AscendC::GlobalTensor<ElementOutput> gOutput, uint32_t sUbOffset, uint32_t rowNumCurLoop,
        uint32_t columnNumRound, uint32_t columnNumPad)
    {
        AscendC::DataCopy(
            gOutput,
            lpUbTensor[sUbOffset],
            AscendC::DataCopyParams(
                rowNumCurLoop, columnNumRound / BLOCK_SIZE, 0, (columnNumPad - columnNumRound) / BLOCK_SIZE));
    }

    /**
     * @brief 在线Softmax子核计算核心函数
     *
     * 执行一个子块的在线Softmax计算，按照严格的顺序执行以下步骤：
     *
     * == 计算流程（严格顺序，每步依赖前一步的结果） ==
     * 1. CalcLocalRowMax: 计算本地行最大值 lm = rowmax(S)
     *    - 输入: S矩阵（注意力分数）
     *    - 输出: lmUbTensor（每行1个最大值）
     *
     * 2. UpdateGlobalRowMax: 更新全局行最大值 hm = max(lm, gm)，计算修正因子 dm = exp(gm - hm)
     *    - 输入: lmUbTensor, gmUbTensor
     *    - 输出: hmUbTensor, dmUbTensor, gmUbTensor（更新后）
     *
     * 3. CalcExp: 计算指数值 P = exp(S - hm)
     *    - 输入: lsUbTensor（S矩阵）, hmUbTensor
     *    - 输出: lsUbTensor（原地覆盖为P矩阵）
     *
     * 4. DownCastP: 将float精度的P降精度为half/bfloat16
     *    - 输入: lsUbTensor（float类型的P矩阵）
     *    - 输出: lpUbTensor（half/bfloat16类型的P矩阵）
     *
     * 5. CalcLocalRowSum: 计算本地行求和 ll = rowsum(P)
     *    - 输入: lsUbTensor（P矩阵）
     *    - 输出: llUbTensor（每行1个求和值）
     *
     * 6. CopyPUbToGm: 将P矩阵写回全局内存（供PV矩阵乘法使用）
     *    - 输入: lpUbTensor
     *    - 输出: gOutput（全局内存）
     *
     * 7. UpdateGlobalRowSum: 更新全局行求和 gl = dm * gl + ll
     *    - 输入: dmUbTensor, glUbTensor, llUbTensor
     *    - 输出: glUbTensor（更新后）
     *
     * == 性能优化技巧 ==
     * - 事件驱动的流水线: 通过HardEvent实现P矩阵写回GM与后续计算（rowsum/gl更新）的并行
     *   - V_MTE3(pingpongFlag): P矩阵降精度完成，可以开始写回GM
     *   - V_MTE2(pingpongFlag): rowsum计算完成，可以开始加载下一个S矩阵
     *   - MTE3_V(pingpongFlag): P矩阵写回GM完成，可以开始下一次Softmax计算
     * - 乒乓缓冲: lsUbTensor和lpUbTensor各支持双缓冲，交替处理两个子块
     * - 依赖关系: CalcExp必须在UpdateGlobalRowMax之后（需要hm）
     *            CalcLocalRowSum必须在CalcExp之后（需要P矩阵）
     *            CopyPUbToGm必须在DownCastP之后（需要降精度后的P）
     *
     * @tparam doTriUMask 是否执行三角上掩码处理
     * @param gOutput              输出全局内存张量
     * @param layoutOutput         输出布局信息
     * @param rowOffset            行偏移
     * @param isFirstStackTile     是否是第一个stack tile
     * @param isLastNoMaskStackTile 是否是最后一个无掩码stack tile
     * @param isFirstRowLoop       是否是第一个行循环
     * @param isLastRowLoop        是否是最后一个行循环
     * @param columnNumRound       对齐后的列数
     * @param pingpongFlag         乒乓缓冲标志（0或1）
     * @param curStackTileMod      当前stack tile的模值
     */
    template <bool doTriUMask>
    __aicore__ inline
    void SubCoreCompute(
        AscendC::GlobalTensor<ElementOutput> gOutput, const LayoutOutput &layoutOutput,
        uint32_t rowOffset, uint32_t isFirstStackTile, uint32_t isLastNoMaskStackTile,
        uint32_t isFirstRowLoop, uint32_t isLastRowLoop,
        uint32_t columnNumRound, uint32_t pingpongFlag,
        uint32_t curStackTileMod)
    {
        uint32_t rowNumCurLoop = layoutOutput.shape(0);
        uint32_t rowNumCurLoopRound = RoundUp(rowNumCurLoop, FLOAT_BLOCK_SIZE);
        uint32_t columnNum = layoutOutput.shape(1);
        uint32_t columnNumPad = layoutOutput.stride(0);
        // S矩阵在UB中的偏移：乒乓缓冲，0或MAX_UB_S_ELEM_NUM
        uint32_t sUbOffset = pingpongFlag * MAX_UB_S_ELEM_NUM;
        // 修正因子dm在UB中的偏移：根据stack tile索引和行偏移计算
        uint32_t dmUbOffsetCurCycle = curStackTileMod * MAX_ROW_NUM_SUB_CORE + rowOffset;

        if constexpr (LSE_MODE_ == LseModeT::OUT_ONLY) {
            // LSE模式下，tvUbTensor在上一个stack tile的最后被用于传输LSE到GM
            // 需要等待LSE传输完成后才能使用tvUbTensor
            if (isFirstStackTile && isFirstRowLoop) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
            }
        }

        // 步骤1: 计算本地行最大值 lm = rowmax(S)
        CalcLocalRowMax(sUbOffset, rowNumCurLoopRound, columnNum, columnNumRound, rowOffset);

        // 步骤2: 更新全局行最大值 hm = max(lm, gm)，计算修正因子 dm = exp(gm - hm)
        UpdateGlobalRowMax(
            rowNumCurLoop, rowNumCurLoopRound,
            columnNum, columnNumRound,
            dmUbOffsetCurCycle,
            rowOffset,
            isFirstStackTile);

        // 步骤3: 计算指数值 P = exp(S - hm)
        CalcExp(sUbOffset, rowNumCurLoop, rowNumCurLoopRound, columnNum, columnNumRound, rowOffset);

        // 等待上一次P矩阵写回GM完成（非三角掩码模式）
        if constexpr (!doTriUMask) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(pingpongFlag);
        }

        // 步骤4: 将float精度的P降精度为half/bfloat16
        DownCastP(sUbOffset, rowNumCurLoop, columnNumRound);
        // 标记P矩阵降精度完成，可以开始写回GM
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(pingpongFlag);

        // 步骤5: 计算本地行求和 ll = rowsum(P)
        CalcLocalRowSum(sUbOffset, rowNumCurLoopRound, columnNum, columnNumRound, rowOffset);
        // 标记rowsum计算完成，可以开始加载下一个S矩阵
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(pingpongFlag);

        // 步骤6: 将P矩阵写回全局内存（供PV矩阵乘法使用）
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(pingpongFlag);
        CopyPUbToGm(gOutput, sUbOffset, rowNumCurLoop, columnNumRound, columnNumPad);
        if constexpr (!doTriUMask) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(pingpongFlag);
            if (isLastNoMaskStackTile && isLastRowLoop) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            }
        } else {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        }

        // 步骤7: 更新全局行求和 gl = dm * gl + ll
        UpdateGlobalRowSum(
            sUbOffset, rowNumCurLoop, rowNumCurLoopRound, dmUbOffsetCurCycle, rowOffset, isFirstStackTile);
    }

    /**
     * @brief 在线Softmax核心操作符
     *
     * 执行在线Softmax算法的核心计算流程，处理一个stack tile的注意力分数。
     * 该函数是OnlineSoftmax模块的入口点，由FAInferKernel在每个stack tile上调用。
     *
     * == 算法流程 ==
     * 对于每个stack tile，执行以下步骤：
     * 1. 从全局内存加载注意力分数S到UB（CopyInputGmToUb）
     * 2. 应用掩码（CopyMaskGmToUb + AddMask）
     * 3. 计算本地行最大值lm（CalcLocalRowMax）
     * 4. 更新全局行最大值gm，计算修正因子dm（UpdateGlobalRowMax）
     * 5. 计算指数值P = exp(S - gm)（CalcExp）
     * 6. 计算本地行求和ll（CalcLocalRowSum）
     * 7. 将P矩阵写回全局内存（CopyOutputUbToGm）
     * 8. 更新全局行求和gl（UpdateGlobalRowSum）
     *
     * == 子块并行 ==
     * 当subBlockNum > 1时，将行数分为两个子块并行处理：
     * - 子块0处理前半部分行
     * - 子块1处理后半部分行
     * 这充分利用了Atlas A2的双子核并行能力。
     *
     * @param gOutput              输出全局内存张量（P矩阵，Softmax概率值）
     * @param gInput               输入全局内存张量（S矩阵，注意力分数）
     * @param layoutOutput         输出布局信息
     * @param layoutInput          输入布局信息
     * @param actualBlockShape     实际块形状（M/N/K维度）
     * @param isFirstStackTile     是否是第一个stack tile（1=是，0=否）
     * @param isLastNoMaskStackTile 是否是最后一个无掩码stack tile（1=是，0=否）
     * @param qSBlockSize          Q的S维度块大小（分组注意力头的组数）
     * @param qNBlockSize          Q的N维度块大小（每组内的token数）
     * @param curStackTileMod      当前stack tile的模值（用于掩码处理）
     *
     * @note 第一个stack tile不需要计算修正因子dm（因为没有历史结果需要调整）
     * @note 最后一个stack tile后需要执行归一化（在RescaleO中完成）
     */
    __aicore__ inline
    void operator()(AscendC::GlobalTensor<ElementOutput> gOutput, AscendC::GlobalTensor<ElementInput> gInput,
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
        uint32_t qNThisSubBlock = (qNBlockSize == 1) ?
            0 : (subBlockIdx == 1) ? (qNBlockSize - qNSplitSubBlock) : qNSplitSubBlock;
        uint32_t rowSplitSubBlock = (qNBlockSize == 1) ?
            (qSBlockSize / 2) : (qSBlockSize * qNSplitSubBlock);
        uint32_t rowActualThisSubBlock = (subBlockIdx == 1) ? (rowNum - rowSplitSubBlock) : rowSplitSubBlock;
        uint32_t rowOffsetThisSubBlock = subBlockIdx * rowSplitSubBlock;
        uint32_t maxRowNumPerLoop = MAX_UB_S_ELEM_NUM / columnNumRound;
        uint32_t rowNumTile = RoundDown(maxRowNumPerLoop, FLOAT_BLOCK_SIZE);
        rowNumTile = AscendC::Std::min(rowNumTile, FLOAT_VECTOR_SIZE);
        uint32_t rowLoopNum = CeilDiv(rowActualThisSubBlock, rowNumTile);
        uint32_t preLoad = 1;

        for (uint32_t rowLoopIdx = 0; rowLoopIdx < rowLoopNum + preLoad; rowLoopIdx++) {
            if (rowLoopIdx < rowLoopNum) {
                uint32_t pingpongFlag = rowLoopIdx % 2;
                uint32_t rowOffsetCurLoop = rowLoopIdx * rowNumTile;
                uint32_t rowOffsetIoGm = rowOffsetCurLoop + rowOffsetThisSubBlock;
                uint32_t rowNumCurLoop = (rowLoopIdx == rowLoopNum - 1) ?
                    (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;

                int64_t offsetInput = layoutInput.GetOffset(MatrixCoord(rowOffsetIoGm, 0));
                auto gInputCurLoop = gInput[offsetInput];

                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(pingpongFlag);
                CopySGmToUb(
                    gInputCurLoop, (pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound, columnNumPad);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(pingpongFlag);
            }
            if (rowLoopIdx >= preLoad) {
                uint32_t delayedRowLoopIdx = rowLoopIdx - preLoad;
                uint32_t pingpongFlag = delayedRowLoopIdx % 2;
                uint32_t rowOffsetCurLoop = delayedRowLoopIdx * rowNumTile;
                uint32_t rowOffsetIoGm = rowOffsetCurLoop + rowOffsetThisSubBlock;
                uint32_t rowNumCurLoop =
                    (delayedRowLoopIdx == rowLoopNum - 1) ? (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;

                int64_t offsetOutput = layoutOutput.GetOffset(MatrixCoord(rowOffsetIoGm, 0));
                auto gOutputCurLoop = gOutput[offsetOutput];
                auto layoutOutputCurLoop = layoutOutput.GetTileLayout(MatrixCoord(rowNumCurLoop, columnNum));
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(pingpongFlag);
                ScaleS((pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound);
                SubCoreCompute<false>(
                    gOutputCurLoop,
                    layoutOutputCurLoop,
                    rowOffsetCurLoop,
                    isFirstStackTile,
                    isLastNoMaskStackTile,
                    (delayedRowLoopIdx == 0),
                    (delayedRowLoopIdx == rowLoopNum - 1),
                    columnNumRound,
                    pingpongFlag,
                    curStackTileMod);
            }
        }
    }

    __aicore__ inline
    void operator()(AscendC::GlobalTensor<ElementOutput> gOutput, AscendC::GlobalTensor<ElementInput> gInput,
        AscendC::GlobalTensor<ElementMask> gMask, const LayoutOutput &layoutOutput, const LayoutInput &layoutInput,
        const LayoutInput &layoutMask, GemmCoord actualBlockShape, uint32_t isFirstStackTile, uint32_t qSBlockSize,
        uint32_t qNBlockSize, uint32_t curStackTileMod, Arch::CrossCoreFlag qkReady, uint32_t triUp, uint32_t triDown,
        uint32_t kvSStartIdx, uint32_t kvSEndIdx)
    {
        uint32_t rowNum = actualBlockShape.m();
        uint32_t columnNum = actualBlockShape.n();
        uint32_t columnNumRound = RoundUp(columnNum, BLOCK_SIZE_IN_BYTE);
        uint32_t columnNumPad = layoutInput.stride(0);
        uint32_t maskStride = layoutMask.stride(0);
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();

        uint32_t qNSplitSubBlock = qNBlockSize / subBlockNum;
        uint32_t qNThisSubBlock = (qNBlockSize == 1) ?
            0 : (subBlockIdx == 1) ? (qNBlockSize - qNSplitSubBlock) : qNSplitSubBlock;
        uint32_t rowSplitSubBlock = (qNBlockSize == 1) ?
            (qSBlockSize / 2) : (qSBlockSize * qNSplitSubBlock);
        uint32_t rowActualThisSubBlock = (subBlockIdx == 1) ?
            (rowNum - rowSplitSubBlock) : rowSplitSubBlock;
        uint32_t rowOffsetThisSubBlock = subBlockIdx * rowSplitSubBlock;

        uint32_t tokenNumPerHeadThisSubBlock = Min(qSBlockSize, rowActualThisSubBlock);
        uint32_t maskOffsetThisSubBlock = (qNBlockSize == 1) ?
            rowOffsetThisSubBlock : 0;

        // calc mask shift in gm
        uint32_t gmOffsetMaskRow;
        uint32_t gmOffsetMaskColumn;
        uint32_t maskColumn;
        uint32_t addMaskUbOffset;
        if (triUp >= kvSStartIdx) {
            uint32_t triUpRoundDown = RoundDown(triUp, BLOCK_SIZE_IN_BYTE);
            gmOffsetMaskRow = triUp - triUpRoundDown;
            gmOffsetMaskColumn = 0;
            maskColumn = kvSEndIdx - triUpRoundDown;
            addMaskUbOffset = triUpRoundDown - kvSStartIdx;
        } else {
            gmOffsetMaskRow = 0;
            gmOffsetMaskColumn = kvSStartIdx - triUp;
            maskColumn = columnNum;
            addMaskUbOffset = 0;
        }
        uint32_t maskColumnRound = RoundUp(maskColumn, BLOCK_SIZE_IN_BYTE);

        int64_t offsetMask =
            layoutMask.GetOffset(MatrixCoord(gmOffsetMaskRow + maskOffsetThisSubBlock, gmOffsetMaskColumn));
        auto gMaskThisSubBlock = gMask[offsetMask];
        auto layoutMaskThisSubBlock = layoutMask;

        uint32_t maxRowNumPerLoop = MAX_UB_S_ELEM_NUM / columnNumRound;
        uint32_t rowNumTile = RoundDown(maxRowNumPerLoop, FLOAT_BLOCK_SIZE);
        rowNumTile = AscendC::Std::min(rowNumTile, FLOAT_VECTOR_SIZE);
        uint32_t rowLoopNum = CeilDiv(rowActualThisSubBlock, rowNumTile);
        uint32_t preLoad = 1;

        if (rowActualThisSubBlock == 0) {
            Arch::CrossCoreWaitFlag(qkReady);
            return;
        }

        for (uint32_t rowLoopIdx = 0; rowLoopIdx < rowLoopNum + preLoad; rowLoopIdx++) {
            if (rowLoopIdx < rowLoopNum) {
                uint32_t pingpongFlag = rowLoopIdx % 2;
                uint32_t rowOffsetCurLoop = rowLoopIdx * rowNumTile;
                uint32_t rowOffsetIoGm = rowOffsetCurLoop + rowOffsetThisSubBlock;
                uint32_t rowNumCurLoop = (rowLoopIdx == rowLoopNum - 1) ?
                    (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;
                // loop 0 mask load before cross core sync
                if (rowLoopIdx == 0) {
                    // the token idx of the start token of the prologue part
                    uint32_t proTokenIdx = rowOffsetCurLoop % tokenNumPerHeadThisSubBlock;
                    // the token num of the prologue part
                    uint32_t proTokenNum =
                        Min(rowNumCurLoop, (tokenNumPerHeadThisSubBlock - proTokenIdx)) % tokenNumPerHeadThisSubBlock;
                    // the token num of the epilogue part
                    uint32_t integralHeadNum = (rowNumCurLoop - proTokenNum) / tokenNumPerHeadThisSubBlock;
                    // the number of integral heads within a cycle
                    uint32_t epiTokenNum = rowNumCurLoop - proTokenNum - integralHeadNum * tokenNumPerHeadThisSubBlock;
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                    CopyMaskGmToUb(
                        gMaskThisSubBlock,
                        maskColumn, maskColumnRound, maskStride,
                        tokenNumPerHeadThisSubBlock,
                        proTokenIdx, proTokenNum, integralHeadNum, epiTokenNum);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
                    Arch::CrossCoreWaitFlag(qkReady);
                }
                int64_t offsetInput = layoutInput.GetOffset(MatrixCoord(rowOffsetIoGm, 0));
                auto gInputCurLoop = gInput[offsetInput];
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(pingpongFlag);
                CopySGmToUb(
                    gInputCurLoop, (pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound, columnNumPad);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(pingpongFlag);
            }
            if (rowLoopIdx >= preLoad) {
                uint32_t delayedRowLoopIdx = rowLoopIdx - preLoad;
                uint32_t pingpongFlag = delayedRowLoopIdx % 2;
                uint32_t rowOffsetCurLoop = delayedRowLoopIdx * rowNumTile;
                uint32_t rowNumCurLoop = (delayedRowLoopIdx == rowLoopNum - 1) ?
                    (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;

                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
                UpCastMask<half, ElementMask>(maskUbTensor16, maskUbTensor, rowNumCurLoop, columnNumRound);
                UpCastMask<float, half>(maskUbTensor32, maskUbTensor16, rowNumCurLoop, columnNumRound);
                
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(pingpongFlag);
                ScaleS((pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound);
                ApplyMask(
                    (pingpongFlag * MAX_UB_S_ELEM_NUM),
                    rowNumCurLoop, columnNumRound,
                    maskColumnRound, addMaskUbOffset);
                // next loop mask load
                if (rowLoopIdx < rowLoopNum) {
                    uint32_t rowOffsetCurLoop = rowLoopIdx * rowNumTile;
                    uint32_t rowNumCurLoop =
                        (rowLoopIdx == rowLoopNum - 1) ? (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;
                    // the token idx of the start token of the prologue part
                    uint32_t proTokenIdx = rowOffsetCurLoop % tokenNumPerHeadThisSubBlock;
                    // the token num of the prologue part
                    uint32_t proTokenNum =
                        Min(rowNumCurLoop, (tokenNumPerHeadThisSubBlock - proTokenIdx)) % tokenNumPerHeadThisSubBlock;
                    // the number of integral heads within a cycle
                    uint32_t integralHeadNum = (rowNumCurLoop - proTokenNum) / tokenNumPerHeadThisSubBlock;
                    // the token num of the epilogue part
                    uint32_t epiTokenNum = rowNumCurLoop - proTokenNum - integralHeadNum * tokenNumPerHeadThisSubBlock;
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                    CopyMaskGmToUb(
                        gMaskThisSubBlock,
                        maskColumn, maskColumnRound, maskStride,
                        tokenNumPerHeadThisSubBlock,
                        proTokenIdx, proTokenNum, integralHeadNum, epiTokenNum);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
                }
                // online softmax vectorized compute
                uint32_t rowOffsetIoGm = rowOffsetCurLoop + rowOffsetThisSubBlock;
                int64_t offsetOutput = layoutOutput.GetOffset(MatrixCoord(rowOffsetIoGm, 0));
                auto gOutputCurLoop = gOutput[offsetOutput];
                auto layoutOutputCurLoop = layoutOutput.GetTileLayout(MatrixCoord(rowNumCurLoop, columnNum));
                SubCoreCompute<true>(
                    gOutputCurLoop,
                    layoutOutputCurLoop,
                    rowOffsetCurLoop,
                    isFirstStackTile,
                    0,
                    (delayedRowLoopIdx == 0),
                    (delayedRowLoopIdx == rowLoopNum - 1),
                    columnNumRound,
                    pingpongFlag,
                    curStackTileMod);
            }
        }
    }

private:
    // ==================== 运行时状态变量 ====================
    float scaleValue;                     ///< Softmax缩放因子，通常为1/sqrt(d_k)，用于在Softmax前缩放注意力分数

    // ==================== UB张量（Unified Buffer） ====================
    // 以下张量存储在UB高速缓存中，用于在线Softmax计算的中间结果
    AscendC::LocalTensor<float> lsUbTensor;           ///< 输入S矩阵UB张量，存储从全局内存加载的注意力分数（支持乒乓缓冲，偏移0和MAX_UB_S_ELEM_NUM）
    AscendC::LocalTensor<ElementOutput> lpUbTensor;   ///< 输出P矩阵UB张量，存储降精度后的Softmax概率值（half/bfloat16）
    AscendC::LocalTensor<ElementMask> maskUbTensor;   ///< 掩码UB张量（原始类型，通常为int8/uint8）
    AscendC::LocalTensor<half> maskUbTensor16;        ///< 掩码UB张量（half类型），用于中间类型转换
    AscendC::LocalTensor<float> maskUbTensor32;       ///< 掩码UB张量（float类型），用于与S矩阵相加

    // ==================== 在线Softmax状态张量 ====================
    // 以下张量维护在线Softmax算法的运行时状态，每行一个值
    AscendC::LocalTensor<float> lmUbTensor;           ///< 本地行最大值（Local Max），当前stack tile中每行的最大值
    AscendC::LocalTensor<float> hmUbTensor;           ///< 历史行最大值（Historical Max），max(lm, gm)的结果
    AscendC::LocalTensor<float> gmUbTensor;           ///< 全局行最大值（Global Max），跨stack tile的运行最大值
    AscendC::LocalTensor<float> dmUbTensor;           ///< 修正因子（Delta Max），exp(old_max - new_max)，用于调整历史累加结果
    AscendC::LocalTensor<float> llUbTensor;           ///< 本地行求和（Local Log-sum），当前stack tile中每行的指数求和
    AscendC::LocalTensor<float> tvUbTensor;           ///< 临时向量（Temporary Vector），用于广播和归约中间结果
    AscendC::LocalTensor<float> glUbTensor;           ///< 全局行求和（Global Log-sum），跨stack tile的运行指数求和
};

}

#endif