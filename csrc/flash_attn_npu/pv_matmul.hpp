#ifndef CATLASS_GEMM_BLOCK_MMAD_PV_HPP_T
#define CATLASS_GEMM_BLOCK_MMAD_PV_HPP_T

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "fa_block.h"

namespace Catlass::Gemm::Block {

/**
 * @brief Flash Attention推理中P-V矩阵乘法的块级实现
 *
 * 该类实现了Flash Attention中P(注意力权重)与V(值)矩阵的块级乘法，
 * 是Flash Attention计算流程的第二个矩阵乘法步骤(Attn*V)。
 * 专门针对Atlas A2平台进行了优化，采用了多级缓存分块和双缓冲技术
 * 以提高内存访问效率和计算性能。
 *
 * @tparam PAGED_CACHE_FLAG_ 是否启用页缓存机制
 * @tparam ENABLE_UNIT_FLAG_ 是否启用单元级优化
 * @tparam L1TileShape_ L1缓存级别的分块形状
 * @tparam L0TileShape_ L0缓存级别的分块形状
 * @tparam AType_ 矩阵A的数据类型和布局
 * @tparam BType_ 矩阵B的数据类型和布局
 * @tparam CType_ 矩阵C(输出)的数据类型和布局
 * @tparam BiasType_ 偏置数据类型
 * @tparam TileCopy_ 分块拷贝策略类型
 * @tparam TileMmad_ 分块矩阵乘法操作类型
 */
template <
    bool PAGED_CACHE_FLAG_,
    bool ENABLE_UNIT_FLAG_,
    class L1TileShape_,
    class L0TileShape_,
    class AType_,
    class BType_,
    class CType_,
    class BiasType_,
    class TileCopy_,
    class TileMmad_>
struct BlockMmad<
    MmadAtlasA2FAIPVT<PAGED_CACHE_FLAG_, ENABLE_UNIT_FLAG_>,
    L1TileShape_,
    L0TileShape_,
    AType_,
    BType_,
    CType_,
    BiasType_,
    TileCopy_,
    TileMmad_> {
public:
    // Type Aliases
    using DispatchPolicy = MmadAtlasA2FAIPVT<PAGED_CACHE_FLAG_, ENABLE_UNIT_FLAG_>;  ///< 调度策略类型
    using ArchTag = typename DispatchPolicy::ArchTag;                                 ///< 架构标签类型
    using L1TileShape = L1TileShape_;                                                 ///< L1缓存分块形状
    using L0TileShape = L0TileShape_;                                                 ///< L0缓存分块形状
    using ElementA = typename AType_::Element;                                        ///< 矩阵A的元素类型
    using LayoutA = typename AType_::Layout;                                          ///< 矩阵A的布局类型
    using ElementB = typename BType_::Element;                                        ///< 矩阵B的元素类型
    using LayoutB = typename BType_::Layout;                                          ///< 矩阵B的布局类型
    using ElementC = typename CType_::Element;                                        ///< 矩阵C的元素类型
    using LayoutC = typename CType_::Layout;                                          ///< 矩阵C的布局类型
    using TileMmad = TileMmad_;                                                       ///< 分块矩阵乘法器类型
    using CopyGmToL1A = typename TileCopy_::CopyGmToL1A;                              ///< 全局内存到L1的矩阵A拷贝器
    using CopyGmToL1B = typename TileCopy_::CopyGmToL1B;                              ///< 全局内存到L1的矩阵B拷贝器
    using CopyL1ToL0A = typename TileCopy_::CopyL1ToL0A;                              ///< L1到L0的矩阵A拷贝器
    using CopyL1ToL0B = typename TileCopy_::CopyL1ToL0B;                              ///< L1到L0的矩阵B拷贝器
    using CopyL0CToGm = typename TileCopy_::CopyL0CToGm;                              ///< L0到全局内存的矩阵C拷贝器
    using ElementAccumulator = 
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;  ///< 累加器元素类型
    using LayoutAInL1 = typename CopyL1ToL0A::LayoutSrc;                              ///< L1中矩阵A的布局
    using LayoutBInL1 = typename CopyL1ToL0B::LayoutSrc;                              ///< L1中矩阵B的布局
    using LayoutAInL0 = typename CopyL1ToL0A::LayoutDst;                              ///< L0中矩阵A的布局
    using LayoutBInL0 = typename CopyL1ToL0B::LayoutDst;                              ///< L0中矩阵B的布局
    using LayoutCInL0 = layout::zN;                                                   ///< L0中矩阵C的布局

    using L1AAlignHelper = Gemm::helper::L1AlignHelper<ElementA, LayoutA>;  ///< 矩阵A的L1对齐辅助类
    using L1BAlignHelper = Gemm::helper::L1AlignHelper<ElementB, LayoutB>;  ///< 矩阵B的L1对齐辅助类

    // 静态常量定义
    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;                     ///< 流水线阶段数
    static constexpr uint32_t L1A_SIZE = L1TileShape::M * L1TileShape::K * sizeof(ElementA);  ///< L1中矩阵A的大小
    static constexpr uint32_t L1B_SIZE = L1TileShape::N * L1TileShape::K * sizeof(ElementB);  ///< L1中矩阵B的大小
    static constexpr uint32_t L0A_SIZE = ArchTag::L0A_SIZE;                       ///< L0中矩阵A的大小
    static constexpr uint32_t L0B_SIZE = ArchTag::L0B_SIZE;                       ///< L0中矩阵B的大小
    static constexpr uint32_t L0C_SIZE = ArchTag::L0C_SIZE;                       ///< L0中矩阵C的大小
    static constexpr uint32_t L0A_PINGPONG_BUF_SIZE = L0A_SIZE / STAGES;          ///< L0中矩阵A的乒乓缓冲区大小
    static constexpr uint32_t L0B_PINGPONG_BUF_SIZE = L0B_SIZE / STAGES;          ///< L0中矩阵B的乒乓缓冲区大小
    static constexpr uint32_t L0C_PINGPONG_BUF_SIZE = L0C_SIZE / STAGES;          ///< L0中矩阵C的乒乓缓冲区大小
    static constexpr uint32_t BLOCK_SIZE = 16;                                    ///< 基本块大小
    static constexpr uint32_t EMBED_SPLIT_SIZE = 128;                             ///< 嵌入维度的分割大小
    static constexpr uint32_t UNIT_BLOCK_STACK_NUM = 4;                           ///< 单元块堆叠数量
    static constexpr uint32_t KV_BASE_BLOCK = 512;                                ///< KV缓存的基础块大小
    static constexpr uint32_t KV_SPLIT_SIZE = 128;                                ///< KV缓存的分割大小
    static constexpr uint32_t LOAB_BLOCK = 1;                                     ///< LOAB块大小
    static constexpr uint32_t COORD_DIM0 = 0;                                     ///< 坐标维度0 (行)
    static constexpr uint32_t COORD_DIM1 = 1;                                     ///< 坐标维度1 (列)
    static constexpr uint32_t COORD_DIM2 = 2;                                     ///< 坐标维度2 (深度)

    // 静态断言：确保矩阵C的布局是行优先
    static_assert(std::is_same_v<LayoutC, layout::RowMajor>, "LayoutC only support RowMajor yet!");

    /**
     * @brief 构造函数：初始化块矩阵乘法器
     *
     * 为矩阵乘法操作分配和初始化L1和L0缓存空间，设置动态维度参数。
     *
     * @param resource 架构资源引用，用于获取缓存空间
     * @param nDyn 动态N维度大小
     * @param kDyn 动态K维度大小
     * @param KVStackLen KV堆栈长度，默认512
     * @param l1BufAddrStart L1缓冲区起始地址，默认0
     */
    __aicore__ inline
    BlockMmad(Arch::Resource<ArchTag> &resource,uint32_t nDyn, uint32_t kDyn, uint32_t KVStackLen = 512, uint32_t l1BufAddrStart = 0)
    {
        maxKVStackLen = KVStackLen;
        // 分配L1内存空间
        l1BTensor = resource.l1Buf.template GetBufferByByte<ElementB>(l1BufAddrStart +
            L1TileShape::M * kDyn * sizeof(ElementA) * STAGES);
        for (uint32_t i = 0; i < STAGES; i++) {
            l1ATensor[i] = resource.l1Buf.template GetBufferByByte<ElementA>(l1BufAddrStart +
                L1TileShape::M * kDyn * sizeof(ElementA) * i);
            l0ATensor[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_PINGPONG_BUF_SIZE * i);
            l0BTensor[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_PINGPONG_BUF_SIZE * i);
            l0CTensor[i] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(L0C_PINGPONG_BUF_SIZE * i);
        }
        l1NDynamic = nDyn;
        l1KDynamic = kDyn;
    }

    /**
     * @brief 析构函数：清理资源
     */
    __aicore__ inline
    ~BlockMmad() {}

    /**
     * @brief 重置块起始偏移量
     *
     * 在处理新的数据块时，重置块的起始偏移量为0。
     */
    __aicore__ inline
    void resetBlockStart(){
        blockStartOffset = 0;
    }

    /**
     * @brief 设置实际块形状
     *
     * 根据当前长度设置实际的块形状。
     *
     * @param actualShape 输出参数，实际块形状
     * @param nowLen 当前处理的长度
     */
    __aicore__ inline
    void getBlockShape(GemmCoord &actualShape, uint32_t& nowLen)
    {        
        actualShape[COORD_DIM2] = nowLen;
    }

    /**
     * @brief 获取KV缓存偏移量（非页缓存版本）
     *
     * 计算KV缓存的偏移量，用于非页缓存模式。
     *
     * @param kOffset 输出参数，K维度偏移量
     * @param nIdx N维度索引
     * @param strideKV KV缓存的步长
     */
    __aicore__ inline
    void getKVOffset(uint32_t &kOffset, uint32_t nIdx, uint32_t &strideKV)
    {
        kOffset = nIdx * maxKVStackLen * strideKV;
    }

    /**
     * @brief 获取KV缓存偏移量（页缓存版本）
     *
     * 根据块表计算KV缓存的偏移量，用于页缓存模式。
     *
     * @param gBlockTable 全局块表张量
     * @param kOffset 输出参数，K维度偏移量
     * @param blockStartOffset 块起始偏移量
     * @param nowNIdx 当前N维度索引
     * @param strideKV KV缓存的步长
     * @param blockSize 块大小
     */
    __aicore__ inline
    void getKVOffset(AscendC::GlobalTensor<int32_t> &gBlockTable, uint32_t &kOffset, uint32_t blockStartOffset, 
        uint32_t nowNIdx, uint32_t &strideKV, uint32_t &blockSize)
    {
        uint32_t blockTableId = gBlockTable.GetValue(nowNIdx);
        kOffset = blockTableId * blockSize * strideKV + blockStartOffset * strideKV;
    }

    /**
     * @brief 设置块参数
     *
     * 根据当前的堆叠序列分块计算块的起始位置、结束位置和总块数。
     *
     * @param stackSeqTile 堆叠序列分块
     * @param blockStart 输出参数，块起始位置
     * @param blockEnd 输出参数，块结束位置
     * @param curBlockTotalNum 输出参数，当前块总数
     * @param blockSize 块大小
     */
    __aicore__ inline
    void setBlockParam(uint32_t stackSeqTile, uint32_t &blockStart, uint32_t &blockEnd, uint32_t &curBlockTotalNum, uint32_t blockSize){
        if(stackSeqTile >= blockStart && blockSize != 0) {
            blockEnd = ((stackSeqTile - blockStart) % blockSize == 0) ? blockSize : (stackSeqTile - blockStart) % blockSize;
            curBlockTotalNum = (((stackSeqTile - blockStart) + blockSize - 1) / blockSize) + 1;
        } else {
            blockStart = stackSeqTile;
            blockEnd = stackSeqTile + blockStartOffset;
            curBlockTotalNum = 1;
        }
    }

    /**
     * @brief 更新块偏移量
     *
     * 根据当前处理的长度更新块的起始偏移量和当前块索引。
     *
     * @param nowLen 当前处理的长度
     * @param curBlockIdx 输出参数，当前块索引
     * @param blockSize 块大小
     */
    __aicore__ inline
    void updateBlockOffset(uint32_t nowLen, uint32_t &curBlockIdx, uint32_t blockSize){
        if (blockStartOffset + nowLen == blockSize) {
            blockStartOffset = 0;
        } else {
            blockStartOffset += nowLen;
        }
        curBlockIdx++;
    }

    /**
     * @brief 核心操作符：执行块矩阵乘法
     *
     * 这是类的核心方法，实现了完整的P-V矩阵乘法逻辑。
     * 支持页缓存和非页缓存两种模式，采用多级缓存分块和双缓冲技术优化性能。
     *
     * @param gA 全局内存中的矩阵A张量（注意力权重P）
     * @param gB 全局内存中的矩阵B张量（值矩阵V）
     * @param gC 全局内存中的矩阵C张量（输出结果）
     * @param gBlockTable 全局块表张量，用于页缓存模式
     * @param layoutA 矩阵A的布局
     * @param layoutB 矩阵B的布局
     * @param layoutC 矩阵C的布局
     * @param actualOriShape 实际原始形状
     * @param nIdx N维度索引
     * @param nLoop N维度循环次数
     * @param blockSize 块大小
     * @param kvSeqlen KV序列长度
     * @param strideKV KV缓存步长
     * @param blockStackNum 块堆叠数量
     * @param softmaxFlag 跨核软max标志
     */
    __aicore__ inline
    void operator()(
        AscendC::GlobalTensor<ElementA> gA,
        AscendC::GlobalTensor<ElementB> gB,
        AscendC::GlobalTensor<ElementC> gC,
        AscendC::GlobalTensor<int32_t> gBlockTable,
        LayoutA layoutA, LayoutB layoutB, LayoutC layoutC, GemmCoord actualOriShape,
        uint32_t &nIdx, uint32_t &nLoop, uint32_t &blockSize, uint32_t kvSeqlen, uint32_t strideKV,
        uint32_t blockStackNum, Arch::CrossCoreFlag softmaxFlag)
    {
        // 解析输入形状参数
        uint32_t rowNum = actualOriShape[COORD_DIM0];   // 行数（M维度）
        uint32_t embed = actualOriShape[COORD_DIM1];    // 嵌入维度（N维度）
        uint32_t stackSeqTile = actualOriShape[COORD_DIM2];  // 堆叠序列分块（K维度）
        GemmCoord actualShape{rowNum, embed, 0};        // 实际形状变量
        uint32_t gBOffset = 0;                          // 矩阵B在全局内存中的偏移量

        // 创建L1中矩阵B的布局
        LayoutBInL1 layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(stackSeqTile, embed);
        
        // 等待事件4完成
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
        
        // 根据是否启用页缓存模式，将矩阵B从全局内存拷贝到L1缓存
        if constexpr (PAGED_CACHE_FLAG_) {
            // 页缓存模式：分块处理矩阵B
            uint32_t curBlockIdx =  0;                  // 当前块索引
            uint32_t blockStart = blockSize - blockStartOffset;  // 块起始位置
            uint32_t blockEnd = 0;                      // 块结束位置
            uint32_t curBlockTotalNum = 0;              // 当前块总数
            
            // 设置块参数
            setBlockParam(stackSeqTile, blockStart, blockEnd, curBlockTotalNum, blockSize);
            
            // 遍历所有块
            while(curBlockIdx < curBlockTotalNum) {
                // 计算当前块的长度
                uint32_t nowLen = (curBlockIdx < (curBlockTotalNum-1)) ? (blockSize - blockStartOffset) : (blockEnd - blockStartOffset);
                // 计算当前N维度索引
                uint32_t nowNIdx = nIdx * maxKVStackLen / blockSize + curBlockIdx;
                // 设置实际形状
                getBlockShape(actualShape, nowLen);
                // 获取KV缓存偏移量
                getKVOffset(gBlockTable, gBOffset, blockStartOffset, nowNIdx, strideKV, blockSize);
                // 创建矩阵B的分块布局
                auto layoutBTile = layoutB.GetTileLayout(MakeCoord(actualShape.k(), actualShape.n()));
                // 计算L1中矩阵B的分块坐标
                uint32_t curBlockSize = (curBlockIdx > 0) ? ((curBlockIdx - 1) * blockSize + blockStart) : 0;
                MatrixCoord l1BTileCoord{curBlockSize, 0};
                auto l1BTile = l1BTensor[layoutBInL1.GetOffset(l1BTileCoord)];
                // 将矩阵B从全局内存拷贝到L1缓存
                copyGmToL1B(l1BTile, gB[gBOffset], layoutBInL1, layoutBTile);
                // 更新块偏移量和当前块索引
                updateBlockOffset(nowLen, curBlockIdx, blockSize);
            }
        } else {
            // 非页缓存模式：直接拷贝整个矩阵B
            getBlockShape(actualShape, stackSeqTile);
            getKVOffset(gBOffset, nIdx, strideKV);
            auto layoutBTile = layoutB.GetTileLayout(MakeCoord(actualShape.k(), actualShape.n()));
            copyGmToL1B(l1BTensor, gB[gBOffset], layoutBInL1, layoutBTile);
        }
        
        // 设置事件0完成标志
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
        // 等待事件0完成
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
        // 等待跨核同步信号（来自softmax阶段）
        Arch::CrossCoreWaitFlag(softmaxFlag);

        // 计算三级循环的次数
        uint32_t mL1Loop = CeilDiv(rowNum, L1TileShape::M);     // M维度L1分块循环次数
        uint32_t kL1Loop = CeilDiv(stackSeqTile, l1KDynamic);   // K维度L1分块循环次数
        uint32_t nL1Loop = CeilDiv(embed, L0TileShape::N);      // N维度L1分块循环次数

        // 外层循环：遍历嵌入维度（N维度）的L1分块
        for (uint32_t nL1Idx = 0; nL1Idx < nL1Loop; nL1Idx++) {
            // 计算当前N维度L1分块的实际大小
            uint32_t nL1Actual = (nL1Idx < nL1Loop - 1U) ? L0TileShape::N : (embed - nL1Idx * L0TileShape::N);
            
            // 中层循环：遍历行（M维度）的L1分块
            for (uint32_t mL1Idx = 0; mL1Idx < mL1Loop; mL1Idx++) {
                // 计算当前M维度L1分块的实际大小
                uint32_t mL1Actual = (mL1Idx < mL1Loop - 1U) ? L1TileShape::M : (rowNum - mL1Idx * L1TileShape::M);
                
                // 等待L0C乒乓缓冲区可用
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CPingPongFlag);
                
                // 内层循环：遍历序列长度（K维度）的L1分块
                for (uint32_t kL1Idx = 0; kL1Idx < kL1Loop; kL1Idx++) {
                    // 计算当前K维度L1分块的实际大小
                    uint32_t kL1Actual = (kL1Idx < kL1Loop - 1U) ? l1KDynamic : (stackSeqTile - kL1Idx * l1KDynamic);
                    
                    // 等待L1乒乓缓冲区可用
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1PPingPongFlag);
                    
                    // 将矩阵A从全局内存拷贝到L1缓存
                    MatrixCoord gmATileCoord{mL1Idx * L1TileShape::M, kL1Idx * l1KDynamic};  // 全局内存中矩阵A的分块坐标
                    auto gmTileA = gA[layoutA.GetOffset(gmATileCoord)];  // 获取全局内存中矩阵A的分块
                    auto layoutTileA = layoutA.GetTileLayout(MakeCoord(mL1Actual, kL1Actual));  // 创建矩阵A的分块布局
                    LayoutAInL1 layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(mL1Actual, kL1Actual);  // 创建L1中矩阵A的布局
                    copyGmToL1A(l1ATensor[l1PPingPongFlag], gmTileA, layoutAInL1, layoutTileA);  // 执行拷贝
                    
                    // 设置L1乒乓缓冲区完成标志
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1PPingPongFlag);

                    // 计算L0分块的循环次数
                    uint32_t kL0Loop = CeilDiv(kL1Actual, L0TileShape::K);
                    
                    // 遍历K维度的L0分块
                    for (uint32_t kL0Idx = 0; kL0Idx < kL0Loop; kL0Idx++) {
                        // 计算当前K维度L0分块的实际大小
                        uint32_t kL0Actual = (kL0Idx < kL0Loop - 1U) ? L0TileShape::K : (kL1Actual - kL0Idx * L0TileShape::K);
                        
                        // 将矩阵A从L1缓存拷贝到L0缓存
                        LayoutAInL0 layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(mL1Actual, kL0Actual);  // 创建L0中矩阵A的布局
                        MatrixCoord l1ATileCoord{0, kL0Idx * L0TileShape::K};  // L1中矩阵A的分块坐标
                        auto l1ATile = l1ATensor[l1PPingPongFlag][layoutAInL1.GetOffset(l1ATileCoord)];  // 获取L1中矩阵A的分块
                        
                        // 等待L0AB乒乓缓冲区可用
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag);
                        
                        // 如果是第一个L0分块，等待L1拷贝完成
                        if (kL0Idx == 0U) {
                            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1PPingPongFlag);
                        }
                        
                        // 执行L1到L0的拷贝
                        copyL1ToL0A(l0ATensor[l0ABPingPongFlag], l1ATile, layoutAInL0, layoutAInL1);
                        
                        // 如果是最后一个L0分块，设置L1乒乓缓冲区标志
                        if (kL0Idx == kL0Loop - 1U) {
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1PPingPongFlag);
                        }

                        // 将矩阵B从L1缓存拷贝到L0缓存
                        LayoutBInL0 layoutBInL0 = LayoutBInL0::template MakeLayout<ElementB>(kL0Actual, nL1Actual);  // 创建L0中矩阵B的布局
                        MatrixCoord l1BTileCoord{kL1Idx * l1KDynamic + kL0Idx * L0TileShape::K, L0TileShape::N * nL1Idx};  // L1中矩阵B的分块坐标
                        auto l1BTile = l1BTensor[layoutBInL1.GetOffset(l1BTileCoord)];  // 获取L1中矩阵B的分块
                        
                        // 等待L0B乒乓缓冲区可用
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag + 2U);
                        
                        // 执行L1到L0的拷贝
                        copyL1ToL0B(l0BTensor[l0ABPingPongFlag], l1BTile, layoutBInL0, layoutBInL1);

                        // 设置事件0完成标志
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
                        // 等待事件0完成
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
                        
                        // 判断是否需要初始化矩阵乘法（仅在第一个K维度分块时初始化）
                        bool initMmad = (kL1Idx == 0U) && (kL0Idx == 0U);
                        // 计算M维度L0对齐大小
                        uint32_t mL0Align = (mL1Actual + BLOCK_SIZE - 1U) / BLOCK_SIZE * BLOCK_SIZE;
                        
                        // 执行分块矩阵乘法
                        tileMmad(l0CTensor[l0CPingPongFlag],
                            l0ATensor[l0ABPingPongFlag],
                            l0BTensor[l0ABPingPongFlag],
                            mL0Align,
                            nL1Actual,
                            kL0Actual,
                            initMmad);
                        
                        // 设置L0AB乒乓缓冲区完成标志
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag);
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag + 2U);
                        
                        // 切换L0AB乒乓缓冲区
                        l0ABPingPongFlag = 1U - l0ABPingPongFlag;
                    }
                    
                    // 切换L1乒乓缓冲区
                    l1PPingPongFlag = 1U - l1PPingPongFlag;
                }
                // 设置事件0完成标志
                AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
                // 等待事件0完成
                AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
                
                // 将结果从L0缓存写回全局内存
                MatrixCoord gmCTileCoord{mL1Idx * L0TileShape::M, L0TileShape::N * nL1Idx};  // 全局内存中矩阵C的分块坐标
                LayoutC layoutCTile = layoutC.GetTileLayout(MakeCoord(mL1Actual, nL1Actual));  // 创建矩阵C的分块布局
                auto layoutInL0C = LayoutCInL0::MakeLayoutInL0C(MakeCoord(mL1Actual, nL1Actual));  // 创建L0中矩阵C的布局
                copyL0CToGm(gC[layoutC.GetOffset(gmCTileCoord)], l0CTensor[l0CPingPongFlag], layoutCTile, layoutInL0C);  // 执行拷贝
                
                // 设置L0C乒乓缓冲区完成标志
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CPingPongFlag);
                
                // 切换L0C乒乓缓冲区
                l0CPingPongFlag = 1U - l0CPingPongFlag;
            }
        }
        
        // 设置事件4完成标志
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
    }
 
protected:
    // L1缓存张量
    AscendC::LocalTensor<ElementA> l1ATensor[STAGES];      ///< L1缓存中的矩阵A张量（支持双缓冲）
    AscendC::LocalTensor<ElementB> l1BTensor;              ///< L1缓存中的矩阵B张量
    
    // L0缓存张量
    AscendC::LocalTensor<ElementA> l0ATensor[STAGES];      ///< L0缓存中的矩阵A张量（支持双缓冲）
    AscendC::LocalTensor<ElementB> l0BTensor[STAGES];      ///< L0缓存中的矩阵B张量（支持双缓冲）
    AscendC::LocalTensor<ElementAccumulator> l0CTensor[STAGES];  ///< L0缓存中的矩阵C张量（支持双缓冲）

    // 操作对象
    TileMmad tileMmad;                  ///< 分块矩阵乘法操作对象
    CopyGmToL1A copyGmToL1A;            ///< 全局内存到L1的矩阵A拷贝器
    CopyGmToL1B copyGmToL1B;            ///< 全局内存到L1的矩阵B拷贝器
    CopyL1ToL0A copyL1ToL0A;            ///< L1到L0的矩阵A拷贝器
    CopyL1ToL0B copyL1ToL0B;            ///< L1到L0的矩阵B拷贝器
    CopyL0CToGm copyL0CToGm;            ///< L0到全局内存的矩阵C拷贝器

    // 乒乓缓冲区标志
    uint32_t l1PPingPongFlag = 0;       ///< L1缓存乒乓缓冲区标志（0或1）
    uint32_t l0CPingPongFlag = 0;       ///< L0缓存矩阵C乒乓缓冲区标志（0或1）
    uint32_t l0ABPingPongFlag = 0;      ///< L0缓存矩阵A和B乒乓缓冲区标志（0或1）

    // 动态维度大小
    uint32_t l1MDynamic = 0;            ///< L1缓存中M维度的动态大小
    uint32_t l1NDynamic = 0;            ///< L1缓存中N维度的动态大小
    uint32_t l1KDynamic = 0;            ///< L1缓存中K维度的动态大小

    // 块偏移和大小参数
    uint32_t blockStartOffset = 0;      ///< 块起始偏移量
    uint32_t maxKVStackLen = 0;         ///< KV堆栈的最大长度
};

}

#endif