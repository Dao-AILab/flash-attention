// Copyright (c) 2023, flash-attention
// All rights reserved.
//
// 该文件定义了FlashAttention内核的通用常量、枚举和结构体
// 包含了内核实现所需的基本配置和辅助函数

#ifndef KERNEL_COMMON
#define KERNEL_COMMON


namespace KernelCommon {
    // 内核状态就绪标识：用于同步不同阶段的计算
    constexpr uint32_t QK_READY_ID = 1;      // QK矩阵乘法就绪标识（Q*K^T计算完成）
    constexpr uint32_t SOFTMAX_READY_ID = 2; // Softmax操作就绪标识（注意力权重计算完成）
    constexpr uint32_t PV_READY_ID = 3;      // PV矩阵乘法就绪标识（注意力权重与V相乘完成）

    // 预启动参数：用于优化流水线执行
    constexpr uint32_t PRE_LAUNCH = 2;       // 预启动任务数量（提前启动后续任务以隐藏延迟）
    constexpr uint32_t N_SPLIT_HELPER = 2;   // 分割辅助参数（用于N维度的分块分割）

    // 内存和缓存配置：定义了内核使用的内存和缓存参数
    constexpr uint32_t MAX_KV_STACK_LEN = 512;    // KV栈的最大长度（KV序列的最大分块大小）
    constexpr uint32_t Q_TILE_CEIL = 128;         // Q矩阵的分块大小（N维度的基础分块单位）
    constexpr uint32_t WORKSPACE_BLOCK_SIZE_DB = Q_TILE_CEIL * MAX_KV_STACK_LEN;  // 双缓冲工作区大小（用于中间结果存储）
    constexpr uint32_t L1_MAX_SIZE = 524288;      // L1缓存的最大大小（字节）
    constexpr uint32_t L1_MAX_N_NUM = 128;        // L1缓存中N维度的最大数量（头维度的分块限制）
    constexpr uint32_t DOUBLE_BUFFER = 2;         // 双缓冲标志（启用双缓冲技术）
    constexpr uint32_t COMP_TRIU_MASK_DIM_LEN = 2048;  // 上三角掩码的维度长度（用于因果掩码）

    // 常用数值常量：便于代码中统一使用
    constexpr uint32_t NUM_32 = 32;
    constexpr uint32_t NUM_128 = 128;
    constexpr uint32_t NUM_256 = 256;

    // 向上对齐函数：将数值a向上对齐到b的倍数
    // @param a 要对齐的数值
    // @param b 对齐的步长
    // @return 对齐后的数值
    template <typename T>
    __aicore__ inline
    T AlignUp(T a, T b)
    {
        return (b == 0) ? 0 : (a + b - 1) / b * b;
    }

    // 最大值函数：返回两个数中的较大值
    // @param a 第一个比较值
    // @param b 第二个比较值
    // @return 较大的数值
    template <typename T>
    __aicore__ inline
    T Max(T a, T b)
    {
        return (a > b) ? a : b;
    }

    namespace FaiKenel {
        constexpr uint32_t BLOCK_SIZE = 16;  // 块大小（内核执行的基本块单元）

        // 流水线类型枚举：定义不同的流水线执行模式
        enum class cvPipeLineType : uint32_t {
            FAI_COMMON_NORMAL = 0,        // 普通流水线（标准执行流程）
            FAI_COMMON_CHUNK_MASK = 1,    // 分块掩码流水线（处理分块掩码的特殊情况）
        };

        // 掩码类型枚举：定义不同的注意力掩码类型
        enum class MaskType : uint32_t {
            NO_MASK = 0,       // 无掩码（所有位置可见）
            MASK_CAUSAL = 1,   // 因果掩码（上三角掩码，用于自回归模型）
            MASK_SPEC = 2      // 特殊掩码（自定义掩码）
        };

        // 输入布局枚举：定义输入张量的维度布局
        enum class inputLayout : uint32_t {
            BSND = 0,  // [batch_size, seq_len, num_heads, head_size] 布局（标准批量布局）
            TND = 1    // [num_heads, seq_len, head_size] 布局（无批次维度的布局）
        };
    };

    // FlashAttention内核参数结构体：封装了内核执行所需的所有参数
    struct FAIKernelParams {
        GM_ADDR q;               // 查询张量Q的地址
        GM_ADDR k;               // 键张量K的地址
        GM_ADDR v;               // 值张量V的地址
        GM_ADDR mask;            // 掩码张量的地址
        GM_ADDR blockTables;     // 块表的地址（用于分页KV缓存）
        GM_ADDR actualQseqlen;   // 实际Q序列长度的地址
        GM_ADDR actualKvseqlen;  // 实际KV序列长度的地址
        GM_ADDR o;               // 输出张量O的地址
        GM_ADDR lse;             // 对数和的地址（用于数值稳定性）
        GM_ADDR workSpace;       // 工作区的地址（用于中间结果存储）
        GM_ADDR tiling;          // 分块数据的地址（存储分块配置信息）

        // 默认构造函数
        __aicore__ inline FAIKernelParams() {}

        // 带参数的构造函数
        // @param q_ 查询张量Q的地址
        // @param k_ 键张量K的地址
        // @param v_ 值张量V的地址
        // @param mask_ 掩码张量的地址
        // @param blockTables_ 块表的地址
        // @param actualQseqlen_ 实际Q序列长度的地址
        // @param actualKvseqlen_ 实际KV序列长度的地址
        // @param o_ 输出张量O的地址
        // @param lse_ 对数和的地址
        // @param workSpace_ 工作区的地址
        // @param tiling_ 分块数据的地址
        __aicore__ inline FAIKernelParams(GM_ADDR q_, GM_ADDR k_, GM_ADDR v_, GM_ADDR mask_, GM_ADDR blockTables_,
                GM_ADDR actualQseqlen_, GM_ADDR actualKvseqlen_, GM_ADDR o_, GM_ADDR lse_, GM_ADDR workSpace_, GM_ADDR tiling_)
            : q(q_), k(k_), v(v_), mask(mask_), blockTables(blockTables_), actualQseqlen(actualQseqlen_),
                actualKvseqlen(actualKvseqlen_), o(o_), lse(lse_), workSpace(workSpace_), tiling(tiling_) {}
    };

    // 获取Q序列的N维度分块大小（头维度）
    // @param qSeqlen 查询序列长度
    // @param groupSize 每组的头数（用于GQA）
    // @return N维度的分块大小
    __aicore__ inline uint32_t GetQNBlockTile(uint32_t qSeqlen, uint32_t groupSize)
    {
        // 计算QN块的分块大小，确保能被N_SPLIT_HELPER整除
        uint32_t qNBlockTile = (qSeqlen != 0) ?
            (Q_TILE_CEIL / qSeqlen) / N_SPLIT_HELPER * N_SPLIT_HELPER : Q_TILE_CEIL;
        // 确保分块大小不超过groupSize，且至少为1
        qNBlockTile = qNBlockTile < groupSize ? qNBlockTile : groupSize;
        qNBlockTile = qNBlockTile < 1 ? 1 : qNBlockTile;
        return qNBlockTile;
    }

    // 获取Q序列的S维度分块大小（序列长度维度）
    // @param kvSeqlen KV序列长度
    // @return S维度的分块大小
    __aicore__ inline uint32_t GetQSBlockTile(uint32_t kvSeqlen)
    {
        // 目前简单返回Q_TILE_CEIL，可根据需要扩展
        uint32_t qSBlockTile = Q_TILE_CEIL;
        return qSBlockTile;
    }
}

#endif