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
#include "fai_block.h"

namespace Catlass::Gemm::Block {

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
    using DispatchPolicy = MmadAtlasA2FAIPVT<PAGED_CACHE_FLAG_, ENABLE_UNIT_FLAG_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using ElementA = typename AType_::Element;
    using LayoutA = typename AType_::Layout;
    using ElementB = typename BType_::Element;
    using LayoutB = typename BType_::Layout;
    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;
    using TileMmad = TileMmad_;
    using CopyGmToL1A = typename TileCopy_::CopyGmToL1A;
    using CopyGmToL1B = typename TileCopy_::CopyGmToL1B;
    using CopyL1ToL0A = typename TileCopy_::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy_::CopyL1ToL0B;
    using CopyL0CToGm = typename TileCopy_::CopyL0CToGm;
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;
    using LayoutAInL1 = typename CopyL1ToL0A::LayoutSrc;
    using LayoutBInL1 = typename CopyL1ToL0B::LayoutSrc;
    using LayoutAInL0 = typename CopyL1ToL0A::LayoutDst;
    using LayoutBInL0 = typename CopyL1ToL0B::LayoutDst;
    using LayoutCInL0 = layout::zN;

    using L1AAlignHelper = Gemm::helper::L1AlignHelper<ElementA, LayoutA>;
    using L1BAlignHelper = Gemm::helper::L1AlignHelper<ElementB, LayoutB>;

    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;
    static constexpr uint32_t L1A_SIZE = L1TileShape::M * L1TileShape::K * sizeof(ElementA);
    static constexpr uint32_t L1B_SIZE = L1TileShape::N * L1TileShape::K * sizeof(ElementB);
    static constexpr uint32_t L0A_SIZE = ArchTag::L0A_SIZE;
    static constexpr uint32_t L0B_SIZE = ArchTag::L0B_SIZE;
    static constexpr uint32_t L0C_SIZE = ArchTag::L0C_SIZE;
    static constexpr uint32_t L0A_PINGPONG_BUF_SIZE = L0A_SIZE / STAGES;
    static constexpr uint32_t L0B_PINGPONG_BUF_SIZE = L0B_SIZE / STAGES;
    static constexpr uint32_t L0C_PINGPONG_BUF_SIZE = L0C_SIZE / STAGES;
    static constexpr uint32_t BLOCK_SIZE = 16;
    static constexpr uint32_t EMBED_SPLIT_SIZE = 128;
    static constexpr uint32_t UNIT_BLOCK_STACK_NUM = 4;
    static constexpr uint32_t KV_BASE_BLOCK = 512;
    static constexpr uint32_t KV_SPLIT_SIZE = 128;
    static constexpr uint32_t LOAB_BLOCK = 1;
    static constexpr uint32_t COORD_DIM0 = 0;
    static constexpr uint32_t COORD_DIM1 = 1;
    static constexpr uint32_t COORD_DIM2 = 2;

    static_assert(std::is_same_v<LayoutC, layout::RowMajor>, "LayoutC only support RowMajor yet!");

    __aicore__ inline
    BlockMmad(Arch::Resource<ArchTag> &resource,uint32_t nDyn, uint32_t kDyn, uint32_t KVStackLen = 512, uint32_t l1BufAddrStart = 0)
    {
        maxKVStackLen = KVStackLen;
        // Allocate L1 memory space
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

    __aicore__ inline
    ~BlockMmad() {}

    __aicore__ inline
    void resetBlockStart(){
        blockStartOffset = 0;
    }

    __aicore__ inline
    void getBlockShape(GemmCoord &actualShape, uint32_t& nowLen)
    {        
        actualShape[COORD_DIM2] = nowLen;
    }

    __aicore__ inline
    void getKVOffset(uint32_t &kOffset, uint32_t nIdx, uint32_t &strideKV)
    {
        kOffset = nIdx * maxKVStackLen * strideKV;
    }

    __aicore__ inline
    void getKVOffset(AscendC::GlobalTensor<int32_t> &gBlockTable, uint32_t &kOffset, uint32_t blockStartOffset, 
        uint32_t nowNIdx, uint32_t &strideKV, uint32_t &blockSize)
    {
        uint32_t blockTableId = gBlockTable.GetValue(nowNIdx);
        kOffset = blockTableId * blockSize * strideKV + blockStartOffset * strideKV;
    }

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

    __aicore__ inline
    void updateBlockOffset(uint32_t nowLen, uint32_t &curBlockIdx, uint32_t blockSize){
        if (blockStartOffset + nowLen == blockSize) {
            blockStartOffset = 0;
        } else {
            blockStartOffset += nowLen;
        }
        curBlockIdx++;
    }

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
        uint32_t rowNum = actualOriShape[COORD_DIM0];
        uint32_t embed = actualOriShape[COORD_DIM1];
        uint32_t stackSeqTile = actualOriShape[COORD_DIM2];
        GemmCoord actualShape{rowNum, embed, 0};
        uint32_t gBOffset = 0;

        LayoutBInL1 layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(stackSeqTile, embed);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
        if constexpr (PAGED_CACHE_FLAG_) {
            uint32_t curBlockIdx =  0;
            uint32_t blockStart = blockSize - blockStartOffset;
            uint32_t blockEnd = 0;
            uint32_t curBlockTotalNum = 0;
            setBlockParam(stackSeqTile, blockStart, blockEnd, curBlockTotalNum, blockSize);
            while(curBlockIdx < curBlockTotalNum) {
                uint32_t nowLen = (curBlockIdx < (curBlockTotalNum-1)) ? (blockSize - blockStartOffset) : (blockEnd - blockStartOffset);
                uint32_t nowNIdx = nIdx * maxKVStackLen / blockSize + curBlockIdx;
                getBlockShape(actualShape, nowLen);
                getKVOffset(gBlockTable, gBOffset, blockStartOffset, nowNIdx, strideKV, blockSize);
                auto layoutBTile = layoutB.GetTileLayout(MakeCoord(actualShape.k(), actualShape.n()));
                uint32_t curBlockSize = (curBlockIdx > 0) ? ((curBlockIdx - 1) * blockSize + blockStart) : 0;
                MatrixCoord l1BTileCoord{curBlockSize, 0};
                auto l1BTile = l1BTensor[layoutBInL1.GetOffset(l1BTileCoord)];
                copyGmToL1B(l1BTile, gB[gBOffset], layoutBInL1, layoutBTile);
                updateBlockOffset(nowLen, curBlockIdx, blockSize);
            }
        } else {
            getBlockShape(actualShape, stackSeqTile);
            getKVOffset(gBOffset, nIdx, strideKV);
            auto layoutBTile = layoutB.GetTileLayout(MakeCoord(actualShape.k(), actualShape.n()));
            copyGmToL1B(l1BTensor, gB[gBOffset], layoutBInL1, layoutBTile);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
        Arch::CrossCoreWaitFlag(softmaxFlag);

        uint32_t mL1Loop = CeilDiv(rowNum, L1TileShape::M);
        uint32_t kL1Loop = CeilDiv(stackSeqTile, l1KDynamic);
        uint32_t nL1Loop = CeilDiv(embed, L0TileShape::N);

        for (uint32_t nL1Idx = 0; nL1Idx < nL1Loop; nL1Idx++) {
            uint32_t nL1Actual = (nL1Idx < nL1Loop - 1U) ? L0TileShape::N : (embed - nL1Idx * L0TileShape::N);
            for (uint32_t mL1Idx = 0; mL1Idx < mL1Loop; mL1Idx++) {
                uint32_t mL1Actual = (mL1Idx < mL1Loop - 1U) ? L1TileShape::M : (rowNum - mL1Idx * L1TileShape::M);
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CPingPongFlag);
                for (uint32_t kL1Idx = 0; kL1Idx < kL1Loop; kL1Idx++) {
                    uint32_t kL1Actual = (kL1Idx < kL1Loop - 1U) ? l1KDynamic : (stackSeqTile - kL1Idx * l1KDynamic);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1PPingPongFlag);
                    MatrixCoord gmATileCoord{mL1Idx * L1TileShape::M, kL1Idx * l1KDynamic};
                    auto gmTileA = gA[layoutA.GetOffset(gmATileCoord)];
                    auto layoutTileA = layoutA.GetTileLayout(MakeCoord(mL1Actual, kL1Actual));
                    LayoutAInL1 layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(mL1Actual, kL1Actual);
                    copyGmToL1A(l1ATensor[l1PPingPongFlag], gmTileA, layoutAInL1, layoutTileA);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1PPingPongFlag);

                    uint32_t kL0Loop = CeilDiv(kL1Actual, L0TileShape::K);
                    for (uint32_t kL0Idx = 0; kL0Idx < kL0Loop; kL0Idx++) {
                        uint32_t kL0Actual =
                            (kL0Idx < kL0Loop - 1U) ? L0TileShape::K : (kL1Actual - kL0Idx * L0TileShape::K);
                        LayoutAInL0 layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(mL1Actual, kL0Actual);
                        MatrixCoord l1ATileCoord{0, kL0Idx * L0TileShape::K};
                        auto l1ATile = l1ATensor[l1PPingPongFlag][layoutAInL1.GetOffset(l1ATileCoord)];

                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag);
                        if (kL0Idx == 0U) {
                            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1PPingPongFlag);
                        }
                        copyL1ToL0A(l0ATensor[l0ABPingPongFlag], l1ATile, layoutAInL0, layoutAInL1);
                        if (kL0Idx == kL0Loop - 1U) {
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1PPingPongFlag);
                        }

                        LayoutBInL0 layoutBInL0 = LayoutBInL0::template MakeLayout<ElementB>(kL0Actual, nL1Actual);
                        MatrixCoord l1BTileCoord{kL1Idx * l1KDynamic + kL0Idx * L0TileShape::K, L0TileShape::N * nL1Idx};
                        auto l1BTile = l1BTensor[layoutBInL1.GetOffset(l1BTileCoord)];

                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag + 2U);
                        copyL1ToL0B(l0BTensor[l0ABPingPongFlag], l1BTile, layoutBInL0, layoutBInL1);

                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
                        bool initMmad = (kL1Idx == 0U) && (kL0Idx == 0U);
                        uint32_t mL0Align = (mL1Actual + BLOCK_SIZE - 1U) / BLOCK_SIZE * BLOCK_SIZE;
                        tileMmad(l0CTensor[l0CPingPongFlag],
                            l0ATensor[l0ABPingPongFlag],
                            l0BTensor[l0ABPingPongFlag],
                            mL0Align,
                            nL1Actual,
                            kL0Actual,
                            initMmad);
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag);
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag + 2U);
                        l0ABPingPongFlag = 1U - l0ABPingPongFlag;
                    }
                    l1PPingPongFlag = 1U - l1PPingPongFlag;
                }
                AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
                MatrixCoord gmCTileCoord{mL1Idx * L0TileShape::M, L0TileShape::N * nL1Idx};
                LayoutC layoutCTile = layoutC.GetTileLayout(MakeCoord(mL1Actual, nL1Actual));
                auto layoutInL0C = LayoutCInL0::MakeLayoutInL0C(MakeCoord(mL1Actual, nL1Actual));
                copyL0CToGm(gC[layoutC.GetOffset(gmCTileCoord)], l0CTensor[l0CPingPongFlag], layoutCTile, layoutInL0C);
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CPingPongFlag);
                l0CPingPongFlag = 1U - l0CPingPongFlag;
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
    }
 
protected:
    AscendC::LocalTensor<ElementA> l1ATensor[STAGES];
    AscendC::LocalTensor<ElementB> l1BTensor;
    AscendC::LocalTensor<ElementA> l0ATensor[STAGES];
    AscendC::LocalTensor<ElementB> l0BTensor[STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensor[STAGES];

    TileMmad tileMmad;
    CopyGmToL1A copyGmToL1A;
    CopyGmToL1B copyGmToL1B;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
    CopyL0CToGm copyL0CToGm;

    uint32_t l1PPingPongFlag = 0;
    uint32_t l0CPingPongFlag = 0;
    uint32_t l0ABPingPongFlag = 0;

    uint32_t l1MDynamic = 0;
    uint32_t l1NDynamic = 0;
    uint32_t l1KDynamic = 0;

    uint32_t blockStartOffset = 0;
    uint32_t maxKVStackLen = 0;
};

}

#endif