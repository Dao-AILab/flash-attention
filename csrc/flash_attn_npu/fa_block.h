// Copyright (c) 2023, flash-attention
// All rights reserved.
//
// 该文件定义了FlashAttention在Atlas A2架构上的基本块结构和模板类
// 主要包含Epilogue和Gemm命名空间下的核心组件定义

#ifndef FAI_BLOCK_HPP
#define FAI_BLOCK_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/gemm/dispatch_policy.hpp"

using namespace Catlass;

namespace Catlass::Epilogue {
    // LSE模式枚举：用于控制对数和的计算方式
    enum class LseModeT {
        NONE = 0,   // 不计算对数和
        OUT_ONLY = 1  // 仅输出对数和
    };

    // Atlas A2架构上的在线Softmax Epilogue模板类
    // 用于在矩阵乘法之后执行在线Softmax操作
    template <LseModeT LSE_MODE_, typename SM_DTYPE_>
    struct EpilogueAtlasA2OnlineSoftmaxT {
        using ArchTag = Arch::AtlasA2;  // 架构标签
        using IntermPrec = SM_DTYPE_;   // 中间计算精度
        static constexpr LseModeT LSE_MODE = LSE_MODE_;  // LSE模式
    };

    // Atlas A2架构上的输出重缩放Epilogue模板类
    // 用于对输出结果进行重缩放操作
    template <LseModeT LSE_MODE_, typename SM_DTYPE_>
    struct EpilogueAtlasA2RescaleOT {
        using ArchTag = Arch::AtlasA2;  // 架构标签
        using IntermPrec = SM_DTYPE_;   // 中间计算精度
        static constexpr LseModeT LSE_MODE = LSE_MODE_;  // LSE模式
    };
}

namespace Catlass::Gemm {
    // Atlas A2架构上的P-V矩阵乘法模板类
    // 用于执行P(Softmax(QK^T))与V的矩阵乘法
    template <bool PAGED_CACHE_FLAG_ = false, bool ENABLE_UNIT_FLAG_ = false>
    struct MmadAtlasA2FAIPVT : public Gemm::MmadAtlasA2{
        static constexpr uint32_t STAGES = 2;  // 流水线级数
        static constexpr bool PAGED_CACHE_FLAG = PAGED_CACHE_FLAG_;  // 是否使用分页缓存
        static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;  // 是否启用单元标志
    };

    // Atlas A2架构上的Q-K矩阵乘法模板类
    // 用于执行Q与K^T的矩阵乘法
    template <bool PAGED_CACHE_FLAG_ = false, bool ENABLE_UNIT_FLAG_ = false>
    struct MmadAtlasA2FAIQKT : public Gemm::MmadAtlasA2{
        static constexpr uint32_t STAGES = 2;  // 流水线级数
        static constexpr bool PAGED_CACHE_FLAG = PAGED_CACHE_FLAG_;  // 是否使用分页缓存
        static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;  // 是否启用单元标志
    };
}

#endif // FAI_BLOCK_HPP