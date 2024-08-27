/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include "cute/tensor.hpp"

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>

#include "utils.h"

namespace flash {

using namespace cute;

template <class TileShape_MK_, class Element, class ElementAccum, class ArchTag_, bool Clear_dQaccum, bool Varlen>
class FlashAttnBwdPreprocess {

public:

    // Type Aliases
    using TileShape_MK = TileShape_MK_;
    using ArchTag = ArchTag_;

    static_assert(std::is_same_v<Element, cutlass::half_t> && ArchTag::kMinComputeCapability >= 75 ||
                  std::is_same_v<Element, cutlass::bfloat16_t> && ArchTag::kMinComputeCapability >= 80 ||
                  std::is_same_v<Element, cutlass::float_e4m3_t> && ArchTag::kMinComputeCapability >= 89);

    static constexpr uint32_t MaxThreadsPerBlock = 256;
    static constexpr uint32_t MinBlocksPerMultiprocessor = 2;
    static constexpr int SharedStorageSize = 0;

    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(get<1>(TileShape_MK{}) % kGmemElemsPerLoad == 0, "Headdim must be a multiple of kGmemElemsPerLoad");
    static constexpr int kHeadDim = get<1>(TileShape_MK{});
    // We want kBlockKGmem to be a power of 2 so that when we do the summing,
    // it's just between threads in the same warp
    static constexpr int kBlockKGmem = kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);
    static constexpr int kGmemThreadsPerRow = kBlockKGmem / kGmemElemsPerLoad;
    static_assert(MaxThreadsPerBlock % kGmemThreadsPerRow == 0, "MaxThreadsPerBlock must be a multiple of kGmemThreadsPerRow");
    using GmemLayoutAtom = Layout<Shape <Int<MaxThreadsPerBlock / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                  Stride<Int<kGmemThreadsPerRow>, _1>>;
    using GmemTiledCopy = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));  // Val layout, 8 or 16 vals per load

    static constexpr int kGmemElemsPerLoadAccum = sizeof(cute::uint128_t) / sizeof(ElementAccum);
    static_assert(get<1>(TileShape_MK{}) % kGmemElemsPerLoadAccum == 0, "Headdim must be a multiple of kGmemElemsPerLoadAccum");
    static constexpr int kGmemThreadsPerRowAccum = kBlockKGmem / kGmemElemsPerLoadAccum;
    static_assert(MaxThreadsPerBlock % kGmemThreadsPerRowAccum == 0, "MaxThreadsPerBlock must be a multiple of kGmemThreadsPerRowAccum");
    using GmemLayoutAtomAccum = Layout<Shape <Int<MaxThreadsPerBlock / kGmemThreadsPerRowAccum>, Int<kGmemThreadsPerRowAccum>>,
                                       Stride<Int<kGmemThreadsPerRowAccum>, _1>>;
    using GmemTiledCopyAccum = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{},
                        GmemLayoutAtomAccum{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoadAccum>>>{}));  // Val layout, 4 vals per store

    using ShapeO = cute::Shape<int32_t, int32_t, int32_t, int32_t>;  // (seqlen_q, d, head, batch)
    using StrideO = cute::Stride<int64_t, _1, int64_t, int64_t>;
    using ShapedPsum = cute::Shape<int32_t, int32_t, int32_t>;  // (seqlen_q, head, batch)
    using StridedPsum = cute::Stride<_1, int64_t, int64_t>;

    // Device side arguments
    struct Arguments {
        Element const* ptr_O;
        ShapeO const shape_O;
        StrideO const stride_O;
        Element const* ptr_dO;
        StrideO const stride_dO;
        float* ptr_dPsum;
        ShapedPsum const shape_dPsum;
        StridedPsum const stride_dPsum;
        float const* ptr_LSE;
        StridedPsum const stride_LSE;
        float *ptr_LSE_log2;
        StridedPsum const stride_LSE_log2;
        ElementAccum* ptr_dQaccum;
        ShapeO const shape_dQaccum;
        StrideO const stride_dQaccum;
        int num_batch;  // We need this to know the size of dq_semaphore in case of varlen
        int* dq_semaphore;
        int const* cu_seqlens = nullptr;
    };

    // Kernel entry point API
    struct Params {
        Element const* ptr_O;
        ShapeO const shape_O;
        StrideO const stride_O;
        Element const* ptr_dO;
        StrideO const stride_dO;
        float* ptr_dPsum;
        ShapedPsum const shape_dPsum;
        StridedPsum const stride_dPsum;
        float const* ptr_LSE;
        StridedPsum const stride_LSE;
        float* ptr_LSE_log2;
        StridedPsum const stride_LSE_log2;
        ElementAccum* ptr_dQaccum;
        ShapeO const shape_dQaccum;
        StrideO const stride_dQaccum;
        int num_batch;
        int* dq_semaphore;
        int const* cu_seqlens = nullptr;
    };

    // Convert to underlying arguments. In this case, a simple copy for the aliased type.
    static
    Params
    to_underlying_arguments(Arguments const& args) {
        return {
            args.ptr_O,
            args.shape_O,
            args.stride_O,
            args.ptr_dO,
            args.stride_dO,
            args.ptr_dPsum,
            args.shape_dPsum,
            args.stride_dPsum,
            args.ptr_LSE,
            args.stride_LSE,
            args.ptr_LSE_log2,
            args.stride_LSE_log2,
            args.ptr_dQaccum,
            args.shape_dQaccum,
            args.stride_dQaccum,
            args.num_batch,
            args.dq_semaphore,
            args.cu_seqlens
        };
    }

    CUTLASS_DEVICE
    void
    operator()(Params const& params, [[maybe_unused]] char* smem_buf) {

        static constexpr int kBlockM = get<0>(TileShape_MK{});

        int const thread_idx = threadIdx.x;
        int const m_block = blockIdx.x;
        int const bidh = blockIdx.y;
        int const bidb = blockIdx.z;

        bool const is_varlen = Varlen && params.cu_seqlens != nullptr;
        int const offset_o = !is_varlen ? 0 : params.cu_seqlens[bidb];
        int const seqlen_o = !is_varlen ? get<0>(params.shape_O) : params.cu_seqlens[bidb + 1] - offset_o;
        if (is_varlen && m_block * kBlockM >= seqlen_o) { return; }

        Tensor mO = make_tensor(make_gmem_ptr(params.ptr_O), params.shape_O, params.stride_O)(_, _, bidh, !is_varlen ? bidb : 0);
        Tensor gO = local_tile(cute::domain_offset(make_coord(offset_o, _0{}), mO), TileShape_MK{}, make_coord(m_block, _0{}));  // (M, K)
        Tensor mdO = make_tensor(make_gmem_ptr(params.ptr_dO), params.shape_O, params.stride_dO)(_, _, bidh, !is_varlen ? bidb : 0);
        Tensor gdO = local_tile(cute::domain_offset(make_coord(offset_o, _0{}), mdO), TileShape_MK{}, make_coord(m_block, _0{}));  // (M, K)

        auto shape_LSE = select<0, 2, 3>(params.shape_O);
        Tensor mLSE = make_tensor(make_gmem_ptr(params.ptr_LSE), shape_LSE, params.stride_LSE)(_, bidh, !is_varlen ? bidb : 0);
        Tensor gLSE = local_tile(cute::domain_offset(make_coord(offset_o), mLSE), Shape<Int<kBlockM>>{}, make_coord(m_block));
        static_assert(kBlockM <= MaxThreadsPerBlock);
        float lse = thread_idx < seqlen_o - m_block * kBlockM && thread_idx < kBlockM ? gLSE(thread_idx) : INFINITY;

        GmemTiledCopy gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(thread_idx);

        Tensor tOgO = gmem_thr_copy_O.partition_S(gO);
        Tensor tOgdO = gmem_thr_copy_O.partition_S(gdO);
        // Construct identity layout for gO
        Tensor cO = cute::make_identity_tensor(TileShape_MK{});  // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
        #pragma unroll
        for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(_0{}, _0{}, k)) < get<1>(params.shape_O); }

        // (8, kBlockM / 32, kHeadDim / 64) or (8, kBlockM / 16, kHeadDim / 128)
        Tensor tOrO = make_fragment_like(tOgO);
        Tensor tOrdO = make_fragment_like(tOgdO);
        flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/true>(
            gmem_tiled_copy_O, tOgO, tOrO, tOcO, tOpO, seqlen_o - m_block * kBlockM
        );
        flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/true>(
            gmem_tiled_copy_O, tOgdO, tOrdO, tOcO, tOpO, seqlen_o - m_block * kBlockM
        );

        // Reshape from e.g. (8, kBlockM / 32, kHeadDim / 64) to (kBlockM / 32, (8, kHeadDim / 64))
        Layout l = make_layout(get<1>(tOrO.layout()), make_layout(get<0>(tOrO.layout()), get<2>(tOrO.layout())));
        Tensor o_fp32 = flash::convert_type<float>(make_tensor(tOrO.data(), l));
        Tensor do_fp32 = flash::convert_type<float>(make_tensor(tOrdO.data(), l));
        // Sum across the last dimension
        Tensor dP_sum = make_tensor<float>(make_shape(size<0>(o_fp32)));
        #pragma unroll
        for (int mi = 0; mi < size<0>(o_fp32); ++mi) {
            float dP_sum_cur = do_fp32(mi, 0) * o_fp32(mi, 0);
            #pragma unroll
            for (int ni = 1; ni < size<1>(o_fp32); ni++) {
                dP_sum_cur += do_fp32(mi, ni) * o_fp32(mi, ni);
            }
            flash::SumOp<float> sum_op;
            dP_sum(mi) = flash::Allreduce<kGmemThreadsPerRow>::run(dP_sum_cur, sum_op);
        }

        // If varlen, the layout for dPSum, LSE_log2, and dQaccum is that we pad each sequence in the batch
        // by an extra 128, so that the write for each sequence doesn't touch the next sequence.
        // Sequence i starts at params.cu_seqlens[i] + i * 128 and ends at params.cu_seqlens[i + 1] + i * 128
        int const offset_padded = !is_varlen ? 0 : (params.cu_seqlens[bidb] + bidb * 128) / 128 * 128;
        Tensor mdPsum = make_tensor(make_gmem_ptr(params.ptr_dPsum), params.shape_dPsum, params.stride_dPsum)(_, bidh, !is_varlen ? bidb : 0);
        Tensor gdPsum = local_tile(cute::domain_offset(make_coord(offset_padded), mdPsum), Shape<Int<kBlockM>>{}, make_coord(m_block));
        if (thread_idx % kGmemThreadsPerRow == 0) {
            #pragma unroll
            for (int mi = 0; mi < size(dP_sum); ++mi) {
                int row = thread_idx / kGmemThreadsPerRow + mi * MaxThreadsPerBlock / kGmemThreadsPerRow;
                gdPsum(row) = row < seqlen_o - m_block * kBlockM ? dP_sum(mi) : 0;
            }
        }

        int const seqlen_rounded = cute::round_up(seqlen_o, kBlockM);
        Tensor mLSElog2 = make_tensor(make_gmem_ptr(params.ptr_LSE_log2), params.shape_dPsum, params.stride_LSE_log2)(_, bidh, !is_varlen ? bidb : 0);
        Tensor gLSElog2 = local_tile(cute::domain_offset(make_coord(offset_padded), mLSElog2), Shape<Int<kBlockM>>{}, make_coord(m_block));
        if (thread_idx < seqlen_rounded - m_block * kBlockM && thread_idx < kBlockM) {
            gLSElog2(thread_idx) = lse == -INFINITY ? 0.f : lse * float(M_LOG2E);
        }

        if constexpr (Clear_dQaccum) {
            Tensor mdQaccum = make_tensor(make_gmem_ptr(params.ptr_dQaccum), params.shape_dQaccum, params.stride_dQaccum)(_, _, bidh, !is_varlen ? bidb : 0);
            Tensor gdQaccum = local_tile(cute::domain_offset(make_coord(offset_padded, _0{}), mdQaccum), TileShape_MK{}, make_coord(m_block, _0{}));
            GmemTiledCopyAccum gmem_tiled_copy_dQaccum;
            auto gmem_thr_copy_dQaccum = gmem_tiled_copy_dQaccum.get_thread_slice(thread_idx);
            Tensor tdQgdQaccum = gmem_thr_copy_dQaccum.partition_D(gdQaccum);
            Tensor zero = make_fragment_like(tdQgdQaccum);
            clear(zero);
            // cute::copy(zero, tdQgdQaccum);  // Somehow this doesn't vectorize the write
            #pragma unroll
            for (int m = 0; m < size<1>(zero); ++m) {
                cute::copy(zero(_, m, _), tdQgdQaccum(_, m, _));
            }
        }

        if (params.dq_semaphore != nullptr && thread_idx == 0) {
            int const num_batch = params.num_batch;
            int const num_head = get<2>(params.shape_dQaccum);
            params.dq_semaphore[bidh + bidb * num_head + m_block * num_head * num_batch] = 0;
        }

    }

};

} // namespace flash
