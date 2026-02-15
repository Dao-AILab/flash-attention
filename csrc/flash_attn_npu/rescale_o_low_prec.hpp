
#ifndef EPILOGUE_BLOCK_BLOCK_EPILOGUE_RESCALE_LOW_PREC_O_HPP_T
#define EPILOGUE_BLOCK_BLOCK_EPILOGUE_RESCALE_LOW_PREC_O_HPP_T

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "fai_block.h"

namespace Catlass::Epilogue::Block {

template <
    class OutputType_,
    class InputType_,
    class UpdateType_,
    class LseType_,
    LseModeT LSE_MODE_>
class BlockEpilogue<
    EpilogueAtlasA2RescaleOT<LSE_MODE_, half>,
    OutputType_,
    InputType_,
    UpdateType_,
    LseType_>
{
public:
    using DispatchPolicy = EpilogueAtlasA2RescaleOT<LSE_MODE_, half>;
    using ArchTag = typename DispatchPolicy::ArchTag;

    using ElementOutput = typename OutputType_::Element;
    using ElementInput = typename InputType_::Element;
    using ElementUpdate = typename UpdateType_::Element;
    using ElementLse = typename LseType_::Element;

    using LayoutOutput = typename OutputType_::Layout;
    using LayoutInput = typename InputType_::Layout;
    using LayoutUpdate = typename UpdateType_::Layout;
    using LayoutLse = typename LseType_::Layout;

    static constexpr LseModeT LSE_MODE = DispatchPolicy::LSE_MODE;

    static constexpr uint32_t HALF_ELENUM_PER_BLK = 16;
    static constexpr uint32_t BLOCK_SIZE = 16;
    static constexpr uint32_t HALF_ELENUM_PER_VECCALC = 128;
    static constexpr uint32_t FLOAT_ELENUM_PER_VECCALC = 64;
    static constexpr uint32_t HALF_ELENUM_PER_LINE = 256;
    static constexpr uint32_t FLOAT_ELENUM_PER_LINE = 128;
    static constexpr uint32_t MULTIPLIER = 2;
    static constexpr uint32_t FLOAT_BLOCK_SIZE = 8;
    static constexpr uint32_t HALF_BLOCK_SIZE = 16;
    static constexpr uint32_t FLOAT_VECTOR_SIZE = 64;
    static constexpr uint32_t HALF_VECTOR_SIZE = 128;
    static constexpr uint32_t UB_UINT8_VECTOR_SIZE = 1024;
    static constexpr uint32_t UB_UINT8_BLOCK_SIZE = 16384;
    static constexpr uint32_t HALF_DM_UB_SIZE = 64;
    static constexpr uint32_t HALF_LL_UB_SIZE = 256;
    static constexpr uint32_t VECTOR_SIZE = 128;
    static constexpr uint32_t NUM4 = 4;
    static constexpr uint32_t MAX_UB_O_ELEM_NUM = 8192;
    static constexpr uint32_t MAX_ROW_NUM_SUB_CORE = 256;
    static constexpr uint32_t SIZE_OF_16BIT = 2;

    __aicore__ inline
    BlockEpilogue(Arch::Resource<ArchTag> &resource)
    {
        constexpr uint32_t LO_UB_TENSOR_OFFSET = 6 * UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t GO_UB_TENSOR_OFFSET = 8 * UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t TV_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE;

        constexpr uint32_t HM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 9 * UB_UINT8_VECTOR_SIZE;
        constexpr uint32_t GM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 10 * UB_UINT8_VECTOR_SIZE;
        constexpr uint32_t LSE32_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 10 * UB_UINT8_VECTOR_SIZE;
        constexpr uint32_t GL_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 12 * UB_UINT8_VECTOR_SIZE;
        constexpr uint32_t LSE16_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 12 * UB_UINT8_VECTOR_SIZE;
        constexpr uint32_t DM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 13 * UB_UINT8_VECTOR_SIZE;

        loUbTensor = resource.ubBuf.template GetBufferByByte<half>(LO_UB_TENSOR_OFFSET);
        dmUbTensor = resource.ubBuf.template GetBufferByByte<half>(DM_UB_TENSOR_OFFSET);
        glUbTensor = resource.ubBuf.template GetBufferByByte<half>(GL_UB_TENSOR_OFFSET);
        tvUbTensor = resource.ubBuf.template GetBufferByByte<half>(TV_UB_TENSOR_OFFSET);
        tvUbTensor32 = resource.ubBuf.template GetBufferByByte<float>(TV_UB_TENSOR_OFFSET);
        goUbTensor = resource.ubBuf.template GetBufferByByte<ElementOutput>(GO_UB_TENSOR_OFFSET);
        hmUbTensor = resource.ubBuf.template GetBufferByByte<half>(HM_UB_TENSOR_OFFSET);
        gmUbTensor = resource.ubBuf.template GetBufferByByte<half>(GM_UB_TENSOR_OFFSET);
        lse16_ubuf_tensor = resource.ubBuf.template GetBufferByByte<half>(LSE16_UB_TENSOR_OFFSET);
        lse32_ubuf_tensor = resource.ubBuf.template GetBufferByByte<float>(LSE32_UB_TENSOR_OFFSET);
    }

    __aicore__ inline
    ~BlockEpilogue() {}

    __aicore__ inline
    void SetMask(int32_t len)
    {
        const int32_t MAX_MASK_LEN = 128;
        const int32_t HALF_MASK_LEN = 64;
        if (len >= MAX_MASK_LEN) {
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            return;
        }
        int32_t highMask = len - HALF_MASK_LEN > 0 ? len - HALF_MASK_LEN : 0;
        int32_t lowMask = len - HALF_MASK_LEN >= 0 ? HALF_MASK_LEN : len;
        if (len < HALF_MASK_LEN) {
            AscendC::SetVectorMask<int8_t>(0x0, ((uint64_t)1 << lowMask) - 1);
        } else {
            AscendC::SetVectorMask<int8_t>(((uint64_t)1 << highMask) - 1, 0xffffffffffffffff);
        }
    }

    __aicore__ inline
    void CopyOToGm(AscendC::GlobalTensor<ElementOutput> gOutput, uint32_t proTokenIdx, uint32_t proTokenNum,
        uint32_t epiTokenNum, uint32_t integralHeadNum, uint32_t qSThisSubBlock, uint32_t embed, uint32_t oHiddenSize)
    {
        uint32_t innerOGmOffset = 0;
        uint32_t innerGOUbOffset = 0;
        if (proTokenNum != 0U) {
            AscendC::DataCopyPad(
                gOutput[innerOGmOffset + proTokenIdx * oHiddenSize],
                goUbTensor[innerGOUbOffset],
                AscendC::DataCopyExtParams(
                    proTokenNum, embed * SIZE_OF_16BIT, 0, (oHiddenSize - embed) * SIZE_OF_16BIT, 0));
            innerOGmOffset += embed;
            innerGOUbOffset += proTokenNum * embed;
        }
        for (uint32_t qN_idx = 0; qN_idx < integralHeadNum; qN_idx++) {
            AscendC::DataCopyPad(
                gOutput[innerOGmOffset],
                goUbTensor[innerGOUbOffset],
                AscendC::DataCopyExtParams(
                    qSThisSubBlock, embed * SIZE_OF_16BIT, 0, (oHiddenSize - embed) * SIZE_OF_16BIT, 0));
            innerOGmOffset += embed;
            innerGOUbOffset += qSThisSubBlock * embed;
        }
        if (epiTokenNum != 0U) {
            AscendC::DataCopyPad(
                gOutput[innerOGmOffset],
                goUbTensor[innerGOUbOffset],
                AscendC::DataCopyExtParams(
                    epiTokenNum, embed * SIZE_OF_16BIT, 0, (oHiddenSize - embed) * SIZE_OF_16BIT, 0));
        }
    }

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
        uint32_t qNThisSubBlock, uint32_t qSThisSubBlock, uint32_t totalRowNum,
        uint32_t isFirstStackTile, uint32_t isLastStackTile, uint32_t curStackTileMod,
        uint32_t needRowLoop, uint32_t isLastRowLoop, uint32_t rowOffsetLoop,
        uint32_t proTokenIdx, uint32_t proTokenNum, uint32_t epiTokenNum, uint32_t integralHeadNum)
    {
        uint32_t curRowNum = layoutInput.shape(0);
        uint32_t embed = layoutInput.shape(1);
        uint32_t embedRound = layoutInput.stride(0);
        uint32_t curRowNumRound = RoundUp(curRowNum, HALF_BLOCK_SIZE);
        uint32_t qSBlockSize = layoutOutput.shape(0);
        uint32_t oHiddenSize = layoutOutput.shape(1);
        uint32_t qHeads = layoutLse.shape(1);
        uint32_t dmUbOffsetCurStackTile = curStackTileMod * MAX_ROW_NUM_SUB_CORE + rowOffsetLoop;

        if (!isFirstStackTile) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
            AscendC::DataCopy(
                loUbTensor, gInput, AscendC::DataCopyParams(1, curRowNum * embedRound / HALF_BLOCK_SIZE, 0, 0));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
        if (!isFirstStackTile) {
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            AscendC::Brcb(
                tvUbTensor.ReinterpretCast<uint16_t>(),
                dmUbTensor[dmUbOffsetCurStackTile].ReinterpretCast<uint16_t>(),
                curRowNumRound / FLOAT_BLOCK_SIZE,
                AscendC::BrcbRepeatParams(1, 8));
            AscendC::PipeBarrier<PIPE_V>();
            if (needRowLoop) {
                AscendC::DataCopy(
                    goUbTensor, gUpdate,
                    AscendC::DataCopyParams(1, curRowNum * embedRound / HALF_BLOCK_SIZE, 0, 0));
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
            }
            // *** go = go * dm_block
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            for (uint32_t vmul_idx = 0; vmul_idx < embed / HALF_VECTOR_SIZE; ++vmul_idx) {
                AscendC::Mul<half, false>(
                    goUbTensor[vmul_idx * HALF_VECTOR_SIZE],
                    goUbTensor[vmul_idx * HALF_VECTOR_SIZE],
                    tvUbTensor,
                    (uint64_t)0,
                    curRowNum,
                    AscendC::BinaryRepeatParams(
                        1, 1, 0, embedRound / HALF_BLOCK_SIZE, embedRound / HALF_BLOCK_SIZE, 1));
            }
            if (embed % HALF_VECTOR_SIZE > 0) {
                SetMask(embed % HALF_VECTOR_SIZE);
                AscendC::Mul<half, false>(
                    goUbTensor[embed / HALF_VECTOR_SIZE * HALF_VECTOR_SIZE],
                    goUbTensor[embed / HALF_VECTOR_SIZE * HALF_VECTOR_SIZE],
                    tvUbTensor,
                    (uint64_t)0,
                    curRowNum,
                    AscendC::BinaryRepeatParams(
                        1, 1, 0, embedRound / HALF_BLOCK_SIZE, embedRound / HALF_BLOCK_SIZE, 1));
                AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            // *** go = lo + go
            AscendC::Add<half, false>(
                goUbTensor,
                goUbTensor,
                loUbTensor,
                (uint64_t)0,
                (curRowNum * embedRound + HALF_VECTOR_SIZE - 1) / HALF_VECTOR_SIZE,
                AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
        } else {
            // *** go = lo
            AscendC::DataCopy(
                goUbTensor, gInput, AscendC::DataCopyParams(1, curRowNum * embedRound / HALF_BLOCK_SIZE, 0, 0));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        }

        if (isLastStackTile) {
            // *** gl_block = expand_to_block(gl)
            AscendC::Brcb(
                tvUbTensor.ReinterpretCast<uint16_t>(),
                glUbTensor.ReinterpretCast<uint16_t>()[rowOffsetLoop],
                curRowNumRound / FLOAT_BLOCK_SIZE,
                AscendC::BrcbRepeatParams(1, 8));
            AscendC::PipeBarrier<PIPE_V>();
            // *** go = go / gl_block
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            for (uint32_t vdiv_idx = 0; vdiv_idx < embed / HALF_VECTOR_SIZE; ++vdiv_idx) {
                AscendC::Div<half, false>(
                    goUbTensor[vdiv_idx * HALF_VECTOR_SIZE],
                    goUbTensor[vdiv_idx * HALF_VECTOR_SIZE],
                    tvUbTensor,
                    (uint64_t)0,
                    curRowNum,
                    AscendC::BinaryRepeatParams(
                        1, 1, 0, embedRound / HALF_BLOCK_SIZE, embedRound / HALF_BLOCK_SIZE, 1));
            }
            if (embed % HALF_VECTOR_SIZE > 0) {
                SetMask(embed % HALF_VECTOR_SIZE);
                AscendC::Div<half, false>(
                    goUbTensor[embed / HALF_VECTOR_SIZE * HALF_VECTOR_SIZE],
                    goUbTensor[embed / HALF_VECTOR_SIZE * HALF_VECTOR_SIZE],
                    tvUbTensor,
                    (uint64_t)0,
                    curRowNum,
                    AscendC::BinaryRepeatParams(
                        1, 1, 0, embedRound / HALF_BLOCK_SIZE, embedRound / HALF_BLOCK_SIZE, 1));
                AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

            // ***move O to GM
            CopyOToGm(
                gOutput, proTokenIdx, proTokenNum, epiTokenNum, integralHeadNum, qSThisSubBlock, embed, oHiddenSize);
            if constexpr (LSE_MODE_ == LseModeT::OUT_ONLY) {
                if (isLastRowLoop) {
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Ln<half, false>(
                        lse16_ubuf_tensor,
                        glUbTensor,
                        (uint64_t)0, CeilDiv(totalRowNum, HALF_VECTOR_SIZE),
                        AscendC::UnaryRepeatParams(1, 1, 8, 8));
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add<half, false>(
                        lse16_ubuf_tensor,
                        lse16_ubuf_tensor,
                        gmUbTensor,
                        (uint64_t)0, CeilDiv(totalRowNum, HALF_VECTOR_SIZE),
                        AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Cast<float, half, false>(
                        lse32_ubuf_tensor,
                        lse16_ubuf_tensor,
                        AscendC::RoundMode::CAST_NONE,
                        (uint64_t)0, CeilDiv(totalRowNum, FLOAT_VECTOR_SIZE),
                        AscendC::UnaryRepeatParams(1, 1, 8, 4));
                    AscendC::PipeBarrier<PIPE_V>();

                    // *** lse_block = expand_to_block(lse)
                    AscendC::Brcb(
                        tvUbTensor32.ReinterpretCast<uint32_t>(),
                        lse32_ubuf_tensor.ReinterpretCast<uint32_t>(),
                        CeilDiv(totalRowNum, FLOAT_BLOCK_SIZE),
                        AscendC::BrcbRepeatParams(1, 8));
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID4);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID4);
                    
                    if (qNThisSubBlock == 0U) {
                        AscendC::DataCopyPad(
                            gLse, tvUbTensor32,
                            AscendC::DataCopyExtParams(
                                totalRowNum, sizeof(float), 0, (qHeads - 1) * sizeof(float), 0));
                    } else {
                        for (uint32_t qNIdx = 0; qNIdx < qNThisSubBlock; qNIdx++) {
                            AscendC::DataCopyPad(
                                gLse[qNIdx],
                                tvUbTensor32[qNIdx * qSBlockSize * FLOAT_BLOCK_SIZE],
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
                gUpdate, goUbTensor, AscendC::DataCopyParams(1, curRowNum * embedRound / HALF_BLOCK_SIZE, 0, 0));
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
    }

    __aicore__ inline
    void operator()(
        AscendC::GlobalTensor<ElementOutput> gOutput,
        AscendC::GlobalTensor<ElementInput> gInput,
        AscendC::GlobalTensor<ElementUpdate> gUpdate,
        AscendC::GlobalTensor<ElementLse> gLse,
        const LayoutOutput &layoutOutput,
        const LayoutInput &layoutInput,
        const LayoutUpdate &layoutUpdate,
        const LayoutLse &layoutLse,
        GemmCoord actualBlockShape,
        uint32_t qSBlockSize, uint32_t qNBlockSize,
        uint32_t isFirstStackTile, uint32_t isLastStackTile, uint32_t curStackTileMod)
    {
        uint32_t rowNum = actualBlockShape.m();
        uint32_t embed = actualBlockShape.n();
        uint32_t maxRowNumPerLoop = MAX_UB_O_ELEM_NUM / embed;
        uint32_t rowNumTile = RoundDown(maxRowNumPerLoop, HALF_BLOCK_SIZE);

        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();

        uint32_t qNSplitSubBlock = qNBlockSize / subBlockNum;
        uint32_t qNThisSubBlock = (qNBlockSize == 1U) ? 0
                                  : (subBlockIdx == 1U) ? (qNBlockSize - qNSplitSubBlock)
                                                       : qNSplitSubBlock;
        uint32_t inRowSplitSubBlock =
            (qNBlockSize == 1U) ? (qSBlockSize / subBlockNum) : (qSBlockSize * qNSplitSubBlock);
        uint32_t inRowActualThisSubBlock = (subBlockIdx == 1U) ? (rowNum - inRowSplitSubBlock) : inRowSplitSubBlock;
        uint32_t inRowOffsetThisSubBlock = subBlockIdx * inRowSplitSubBlock;
        uint32_t outRowOffsetThisSubBlock = (qNBlockSize == 1U) ? inRowOffsetThisSubBlock : 0;
        uint32_t outColOffsetThisSubBlock = (qNBlockSize == 1U) ? 0 : subBlockIdx * qNSplitSubBlock * embed;
        uint32_t qSThisSubBlock = (qNBlockSize == 1U) ? inRowActualThisSubBlock : qSBlockSize;
        int64_t outOffsetSubBlock =
            layoutOutput.GetOffset(MatrixCoord(outRowOffsetThisSubBlock, outColOffsetThisSubBlock));

        uint32_t outLseRowOffsetThisSubBlock = (qNBlockSize == 1U) ?
            inRowOffsetThisSubBlock : 0;
        uint32_t outLseColOffsetThisSubBlock = (qNBlockSize == 1U) ?
            0 : subBlockIdx * qNSplitSubBlock;
        int64_t offsetLse =
            layoutLse.GetOffset(MatrixCoord(outLseRowOffsetThisSubBlock, outLseColOffsetThisSubBlock));
        auto gLseThisSubBlock = gLse[offsetLse];
        auto layoutOutLseThisSubBlock = layoutLse;

        if (inRowActualThisSubBlock > 0U) {
            uint32_t rowLoop = CeilDiv(inRowActualThisSubBlock, rowNumTile);
            uint32_t needRowLoop = (rowLoop > 1U) ? 1 : 0;

            // The rows of each cycle consist of multiple heads with several tokens.
            // There are several integral heads, one prologue head, one epilogue head.
            uint32_t proTokenIdx = 0;      // the token idx of the start token of the prologue part
            uint32_t proTokenIdxPre = 0;   // the token idx of the start token of the pre prologue part
            uint32_t proTokenNum = 0;      // the token num of the prologue part
            uint32_t epiTokenNum = 0;      // the token num of the epilogue part
            uint32_t integralHeadNum = 0;  // the number of integral heads within a cycle
            uint32_t qSRemian = qSThisSubBlock;
            for (uint32_t rowLoopIdx = 0; rowLoopIdx < rowLoop; rowLoopIdx++) {
                uint32_t rowOffsetLoop = rowLoopIdx * rowNumTile;
                uint32_t rowOffsetCurLoop = inRowOffsetThisSubBlock + rowOffsetLoop;
                uint32_t rowActualCurLoop =
                    (rowLoopIdx == (rowLoop - 1U)) ? inRowActualThisSubBlock - rowLoopIdx * rowNumTile : rowNumTile;

                int64_t offsetOutput =
                    static_cast<int64_t>(rowLoopIdx * rowNumTile / qSThisSubBlock * embed) + outOffsetSubBlock;
                auto gOutputCurLoop = gOutput[offsetOutput];
                auto layoutOutputCurLoop = layoutOutput;
                int64_t offsetInput = layoutInput.GetOffset(MatrixCoord(rowOffsetCurLoop, 0));
                auto gInputCurLoop = gInput[offsetInput];
                auto layoutInputCurLoop = layoutInput.GetTileLayout(MatrixCoord(rowActualCurLoop, embed));

                int64_t offsetUpdate = layoutUpdate.GetOffset(MatrixCoord(rowOffsetCurLoop, 0));
                auto gUpdateCurLoop = gUpdate[offsetUpdate];
                auto layoutUpdateCurLoop = layoutUpdate.GetTileLayout(MatrixCoord(rowActualCurLoop, embed));

                proTokenIdx = rowOffsetLoop % qSThisSubBlock;
                proTokenNum = AscendC::Std::min(rowActualCurLoop, (qSThisSubBlock - proTokenIdx)) % qSThisSubBlock;
                integralHeadNum = (rowActualCurLoop - proTokenNum) / qSThisSubBlock;
                epiTokenNum = rowActualCurLoop - proTokenNum - integralHeadNum * qSThisSubBlock;

                SubCoreCompute(
                    gOutputCurLoop,
                    gInputCurLoop,
                    gUpdateCurLoop,
                    gLseThisSubBlock,
                    layoutOutputCurLoop,
                    layoutInputCurLoop,
                    layoutUpdateCurLoop,
                    layoutOutLseThisSubBlock,
                    qNThisSubBlock,
                    qSThisSubBlock,
                    inRowActualThisSubBlock,
                    isFirstStackTile,
                    isLastStackTile,
                    curStackTileMod,
                    needRowLoop,
                    (rowLoopIdx == rowLoop - 1U),
                    rowOffsetLoop,
                    proTokenIdx,
                    proTokenNum,
                    epiTokenNum,
                    integralHeadNum);
            }
        }
    }

private:
    AscendC::LocalTensor<half> loUbTensor;
    AscendC::LocalTensor<half> dmUbTensor;
    AscendC::LocalTensor<half> hmUbTensor;
    AscendC::LocalTensor<half> glUbTensor;
    AscendC::LocalTensor<half> tvUbTensor;
    AscendC::LocalTensor<float> tvUbTensor32;
    AscendC::LocalTensor<ElementOutput> goUbTensor;
    AscendC::LocalTensor<half> gmUbTensor;
    AscendC::LocalTensor<half> lse16_ubuf_tensor;
    AscendC::LocalTensor<float> lse32_ubuf_tensor;
};

}

#endif