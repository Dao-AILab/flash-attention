/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include "cute/algorithm/copy.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"

using namespace cute;

template <int kStages, class Gemm1Type, class Gemm2Type, class OutputType, class SmemLayoutQ,
          class SmemLayoutK, class SmemLayoutV, class SmemLayoutO>
struct SharedStorageQKVO {
    cute::array_aligned<Gemm1Type, cute::cosize_v<SmemLayoutQ>> smem_q;
    cute::array_aligned<Gemm1Type, cute::cosize_v<SmemLayoutK>> smem_k;
    union {
        cute::array_aligned<Gemm2Type, cute::cosize_v<SmemLayoutV>> smem_v;
        cute::array_aligned<OutputType, cute::cosize_v<SmemLayoutO>> smem_o;
    };
    struct {
        cutlass::arch::ClusterTransactionBarrier barrier_Q;
        cutlass::arch::ClusterBarrier barrier_O;
        typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_k;
        typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_v;
        int tile_count_semaphore;
    };
};

template <int kStages, class Gemm1Type, class Gemm2Type, class OutputType, class SmemLayoutQ,
          class SmemLayoutK, class SmemLayoutV, class SmemLayoutO>
struct SharedStorageQKVOVt {
  struct {
    cute::array_aligned<Gemm1Type, cute::cosize_v<SmemLayoutQ>> smem_q;
    cute::array_aligned<Gemm1Type, cute::cosize_v<SmemLayoutK>> smem_k;
    cute::array_aligned<Gemm2Type, cute::cosize_v<SmemLayoutV>> smem_v;  
    union {
        cute::array_aligned<Gemm2Type, cute::cosize_v<SmemLayoutV>> smem_v_out;
        cute::array_aligned<OutputType, cute::cosize_v<SmemLayoutO>> smem_o;
    };
  };
  struct {    
    cutlass::arch::ClusterTransactionBarrier barrier_Q;
    cutlass::arch::ClusterBarrier barrier_O;
    typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_k;
    typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_v;
    typename cutlass::PipelineAsync<kStages>::SharedStorage pipeline_vt;
    int tile_count_semaphore;
    float softmax_scale_qk_log2;
    float descale_v;
  };
};

// If Share_Q_K_smem is true, that forces Is_Q_in_regs to be true
template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, int kStages_, bool Is_Q_in_regs_=false,
         int kClusterM_ = 1, typename elem_type=cutlass::half_t>
struct Flash_fwd_kernel_traits {
    using Element = elem_type;
    using ElementAccum = float;
    using OutputType = elem_type;
    using index_t = int64_t;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * cutlass::NumThreadsPerWarp;
    static constexpr int NumProducerThreads = cutlass::NumThreadsPerWarp;

    static constexpr bool Is_Q_in_regs = Is_Q_in_regs_;
    static_assert(kNWarps_ == 4 || kNWarps_ == 8 || kNWarps_ == 12 || kNWarps_ == 16);
    static constexpr bool Is_WS = kNWarps_ >= 12;
    static_assert(!(Is_WS && Is_Q_in_regs), "Warp-specialization does not support Q in registers");

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static_assert(kHeadDim % 32 == 0);
    using TileShape_MNK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;

    static constexpr int kClusterM = kClusterM_;
    using ClusterShape_MNK = Shape<Int<kClusterM>, _1, _1>;

    static constexpr int kStages = kStages_;

    using AtomLayoutMNK = Layout<Shape<Int<kBlockM / 64>, _1, _1>>;
    using TiledMma0 = decltype(cute::make_tiled_mma(
        std::conditional_t<
            Is_Q_in_regs,
            decltype(cute::GMMA::rs_op_selector<Element, Element, ElementAccum, TileShape_MNK>()),
            decltype(cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShape_MNK>())
        >{},
        AtomLayoutMNK{}));
    using TiledMma1 = decltype(cute::make_tiled_mma(
        cute::GMMA::rs_op_selector<Element, Element, ElementAccum, decltype(select<0, 2, 1>(TileShape_MNK{})),
                                   GMMA::Major::K, GMMA::Major::MN>(),
        AtomLayoutMNK{}));

    using SmemLayoutAtomQ = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQ{}, select<0, 2>(TileShape_MNK{})));

    using SmemLayoutAtomK = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutK =
        decltype(tile_to_shape(SmemLayoutAtomK{},
                 make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{})));

    using SmemLayoutAtomV = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutV =
        decltype(tile_to_shape(SmemLayoutAtomV{},
                 make_shape(get<1>(TileShape_MNK{}), get<2>(TileShape_MNK{}), Int<kStages>{})));

    // Note this is the transpose in terms of the view, not in terms of memory.
    using SmemLayoutVt =
        decltype(composition(SmemLayoutV{},
                    make_ordered_layout(
                        make_shape(get<2>(TileShape_MNK{}), get<1>(TileShape_MNK{}), Int<kStages>{}),
                        Step<_2, _1, _3>{})));

    using SmemLayoutAtomO = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, OutputType,
        decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutO = decltype(tile_to_shape(SmemLayoutAtomO{}, select<0, 2>(TileShape_MNK{})));

    using SmemCopyAtomQ = Copy_Atom<cute::SM75_U32x4_LDSM_N, Element>;

    using SharedStorage = SharedStorageQKVO<kStages, Element, Element, Element, SmemLayoutQ,
                                            SmemLayoutK, SmemLayoutV, SmemLayoutO>;

    using MainloopPipeline = typename cutlass::PipelineTmaAsync<kStages>;
    using MainloopPipelineNoTMA = typename cutlass::PipelineAsync<kStages>;
    using PipelineState = typename cutlass::PipelineState<kStages>;
    // using BarrierType = typename MainloopPipeline::ProducerBarrierType;

};

// Traits struct for fp8 kernel with in-kernel transpose
template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, int kStages_, bool Is_Q_in_regs_=false,
         int kClusterM_ = 1, typename elem_type=cutlass::float_e4m3_t>
struct Flash_fwd_kernel_traits_fp8 {
    using Element = elem_type;
    static_assert(cutlass::sizeof_bits_v<Element> == 8);
    using ElementAccum = float;
    using OutputType = cutlass::half_t;
    using index_t = int64_t;      

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * cutlass::NumThreadsPerWarp;
    static constexpr int NumProducerThreads = cutlass::NumThreadsPerWarpGroup;

    static constexpr bool Is_Q_in_regs = Is_Q_in_regs_;
    static_assert(kNWarps_ == 12 || kNWarps_ == 16);
    static constexpr bool Is_WS = true;    
    static_assert(!Is_Q_in_regs, "Warp-specialization does not support Q in registers");    

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static_assert(kHeadDim % 32 == 0);
    using TileShape_MNK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;

    static constexpr int kClusterM = kClusterM_;
    using ClusterShape_MNK = Shape<Int<kClusterM>, _1, _1>;

    static constexpr int kStages = kStages_;
    static_assert(kStages > 1);

    using AtomLayoutMNK = Layout<Shape<Int<kBlockM / 64>, _1, _1>>;    
    using TiledMma0 = decltype(cute::make_tiled_mma(
        cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShape_MNK>(),
        AtomLayoutMNK{}));
    
    using TiledMma1 = decltype(cute::make_tiled_mma(
        cute::GMMA::rs_op_selector<Element, Element, ElementAccum, decltype(select<0, 2, 1>(TileShape_MNK{}))>(),
        AtomLayoutMNK{}));

    using SmemLayoutAtomQ = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQ{}, select<0, 2>(TileShape_MNK{})));

    using SmemLayoutAtomK = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutK =
        decltype(tile_to_shape(SmemLayoutAtomK{},
                 make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{})));

    using TransposeShapeAtomV = Shape<_64, _64>;    
    using SmemLayoutAtomV = decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<Element>{}, TransposeShapeAtomV{}));
    using SmemLayoutV =
        decltype(tile_to_shape(SmemLayoutAtomV{},
                 make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{})));
    
    // for fp8 in-kernel transpose -- src layout
    using SmemLayoutDivideV = decltype(tiled_divide(SmemLayoutV{}, TransposeShapeAtomV{}));
    using SmemShapeLDSM = Shape<Shape<_8, _8>, Shape<_16, _4>>;
    using FactoringShapeV = decltype(make_shape(SmemShapeLDSM{},
        shape<1>(SmemLayoutDivideV{}), shape<2>(SmemLayoutDivideV{}), shape<3>(SmemLayoutDivideV{})));
    using SmemLayoutTransposeV = decltype(composition(SmemLayoutDivideV{}, make_layout(FactoringShapeV{})));

    // For fp8, this is the memory transpose.
    using SmemLayoutAtomVt = decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<Element>{}, TransposeShapeAtomV{}));
    using SmemLayoutVt =
        decltype(tile_to_shape(SmemLayoutAtomVt{},
                 make_shape(shape<2>(TileShape_MNK{}), shape<1>(TileShape_MNK{}), Int<kStages>{})));

    // for fp8 in-kernel transpose -- dst layout
    using SmemLayoutVtTrans =
        decltype(composition(SmemLayoutVt{},
                             make_ordered_layout(product_each(shape(SmemLayoutV{})), Step<_2, _1, _3>{})));
    using SmemLayoutDivideVt = decltype(tiled_divide(SmemLayoutVtTrans{}, TransposeShapeAtomV{}));
#ifndef NO_FP8_COLUMN_PERMUTE
    using SmemShapeSTSM = Shape<Shape<_16, _4>, Shape<_8, _8>>;
#else
    using SmemShapeSTSM = Shape<Shape<_16, _4>, Shape<_16, _4>>;
#endif
    using FactoringShapeVt = decltype(make_shape(SmemShapeSTSM{},
        shape<1>(SmemLayoutDivideVt{}), shape<2>(SmemLayoutDivideVt{}), shape<3>(SmemLayoutDivideVt{})));
    using SmemLayoutTransposeVt = decltype(composition(SmemLayoutDivideVt{}, make_layout(FactoringShapeVt{})));

    using SmemLayoutAtomO = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, OutputType,
        decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutO = decltype(tile_to_shape(SmemLayoutAtomO{}, select<0, 2>(TileShape_MNK{})));

    // used for rmem -> smem O copy in fp8 kernel to undo column permutation
    using ThreadLayoutrO = Layout<Shape<_8, Int<kBlockM/16>, _4, _1>,
                                 Stride<_4, _32, _1, _0>>;
    using ValueLayoutrO = Layout<Shape<_1, _2, Shape<_2, _2>, Int<kHeadDim/16>>,
                                Stride<_0, _2, Stride<_4, _1>, _8>>;
    using TiledCopyrO = decltype(make_tiled_copy(Copy_Atom<UniversalCopy<uint16_t>, OutputType>{},
                      ThreadLayoutrO{}, ValueLayoutrO{}));

    using TiledCopyShaperO = Shape<_8, Int<kBlockM/8>, _16, Int<kHeadDim/16>>;
    using SmemLayoutrO = decltype(composition(SmemLayoutO{}, Layout<TiledCopyShaperO>{}));

    using SmemCopyAtomQ = Copy_Atom<cute::SM75_U32x4_LDSM_N, Element>;

    using SharedStorage = SharedStorageQKVOVt<kStages, Element, Element, OutputType, SmemLayoutQ,
                          SmemLayoutK, SmemLayoutV, SmemLayoutO>;

    using MainloopPipeline = typename cutlass::PipelineTmaAsync<kStages>;
    using MainloopPipelineNoTMA = typename cutlass::PipelineAsync<kStages>;
    using PipelineState = typename cutlass::PipelineState<kStages>;
    // using BarrierType = typename MainloopPipeline::ProducerBarrierType;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Has_P_smem, int kStages, class Element, class OutputType, class SmemLayoutQ, class SmemLayoutdO,
          class SmemLayoutK, class SmemLayoutV, class SmemLayoutP, class SmemLayoutdS,
          class SmemLayoutdK, class SmemLayoutdV>
struct SharedStorageQKVdOdKV;

template <int kStages, class Element, class OutputType, class SmemLayoutQ, class SmemLayoutdO,
          class SmemLayoutK, class SmemLayoutV, class SmemLayoutP, class SmemLayoutdS,
          class SmemLayoutdK, class SmemLayoutdV>
struct SharedStorageQKVdOdKV<true, kStages, Element, OutputType, SmemLayoutQ, SmemLayoutdO,
        SmemLayoutK, SmemLayoutV, SmemLayoutP, SmemLayoutdS, SmemLayoutdK, SmemLayoutdV> {
    struct {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> smem_q;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutdO>> smem_do;
        union {
            struct {
                cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>> smem_k;
                cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>> smem_v;
            };
            struct {
                cute::array_aligned<Element, cute::cosize_v<SmemLayoutdK>> smem_dk;
                cute::array_aligned<Element, cute::cosize_v<SmemLayoutdV>> smem_dv;
            };
        };
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutP>> smem_p;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutdS>> smem_ds;
    };
    struct {
        cute::uint64_t tma_load_mbar[8]; // 8 TMA barrier pre-allcoated for usage.
        cutlass::arch::ClusterTransactionBarrier barrier_K;
        cutlass::arch::ClusterTransactionBarrier barrier_V;
        typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_q;
        typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_do;
    };
};

template <int kStages, class Element, class OutputType, class SmemLayoutQ, class SmemLayoutdO,
          class SmemLayoutK, class SmemLayoutV, class SmemLayoutP, class SmemLayoutdS,
          class SmemLayoutdK, class SmemLayoutdV>
struct SharedStorageQKVdOdKV<false, kStages, Element, OutputType, SmemLayoutQ, SmemLayoutdO,
        SmemLayoutK, SmemLayoutV, SmemLayoutP, SmemLayoutdS, SmemLayoutdK, SmemLayoutdV> {
    struct {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> smem_q;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutdO>> smem_do;
        union {
            struct {
                cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>> smem_k;
                cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>> smem_v;
            };
            struct {
                cute::array_aligned<Element, cute::cosize_v<SmemLayoutdK>> smem_dk;
                cute::array_aligned<Element, cute::cosize_v<SmemLayoutdV>> smem_dv;
            };
        };
        union {  // Put smem_p in a union just so we can still refer to it in the struct, even if it's not used.
            cute::array_aligned<Element, cute::cosize_v<SmemLayoutP>> smem_p;
            cute::array_aligned<Element, cute::cosize_v<SmemLayoutdS>> smem_ds;
        };
    };
    struct {
        cute::uint64_t tma_load_mbar[8]; // 8 TMA barrier pre-allcoated for usage.
        cutlass::arch::ClusterTransactionBarrier barrier_K;
        cutlass::arch::ClusterTransactionBarrier barrier_V;
        typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_q;
        typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_do;
    };
};

template <bool Has_P_smem, int kStages, class Element, class OutputType, class SmemLayoutQ, class SmemLayoutdO,
          class SmemLayoutK, class SmemLayoutV, class SmemLayoutP, class SmemLayoutdS, class SmemLayoutdQacc,
          class SmemLayoutdK, class SmemLayoutdV>
struct SharedStorageQKVdOdKVWS;

template <int kStages, class Element, class OutputType, class SmemLayoutQ, class SmemLayoutdO,
          class SmemLayoutK, class SmemLayoutV, class SmemLayoutP, class SmemLayoutdS, class SmemLayoutdQacc,
          class SmemLayoutdK, class SmemLayoutdV>
struct SharedStorageQKVdOdKVWS<true, kStages, Element, OutputType, SmemLayoutQ, SmemLayoutdO,
        SmemLayoutK, SmemLayoutV, SmemLayoutP, SmemLayoutdS, SmemLayoutdQacc, SmemLayoutdK, SmemLayoutdV> {
    struct {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> smem_q;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutdO>> smem_do;
        union {
            struct {
                cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>> smem_k;
                cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>> smem_v;
            };
            struct {
                cute::array_aligned<Element, cute::cosize_v<SmemLayoutdK>> smem_dk;
                cute::array_aligned<Element, cute::cosize_v<SmemLayoutdV>> smem_dv;
            };
        };
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutP>> smem_p;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutdS>> smem_ds;
        cute::array_aligned<float, cute::cosize_v<SmemLayoutdQacc>> smem_dqacc;
        cute::array_aligned<float, 128> smem_lse;
        cute::array_aligned<float, 128> smem_dpsum;
    };
    struct {
        cute::uint64_t tma_load_mbar[8]; // 8 TMA barrier pre-allcoated for usage.
        cutlass::arch::ClusterTransactionBarrier barrier_K;
        cutlass::arch::ClusterTransactionBarrier barrier_V;
        typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_q;
        typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_do;
    };
};

template <int kStages, class Element, class OutputType, class SmemLayoutQ, class SmemLayoutdO,
          class SmemLayoutK, class SmemLayoutV, class SmemLayoutP, class SmemLayoutdS, class SmemLayoutdQacc,
          class SmemLayoutdK, class SmemLayoutdV>
struct SharedStorageQKVdOdKVWS<false, kStages, Element, OutputType, SmemLayoutQ, SmemLayoutdO,
        SmemLayoutK, SmemLayoutV, SmemLayoutP, SmemLayoutdS, SmemLayoutdQacc, SmemLayoutdK, SmemLayoutdV> {
    struct {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> smem_q;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutdO>> smem_do;
        union {
            struct {
                cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>> smem_k;
                cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>> smem_v;
            };
            struct {
                cute::array_aligned<Element, cute::cosize_v<SmemLayoutdK>> smem_dk;
                cute::array_aligned<Element, cute::cosize_v<SmemLayoutdV>> smem_dv;
            };
        };
        union {  // Put smem_p in a union just so we can still refer to it in the struct, even if it's not used.
            cute::array_aligned<Element, cute::cosize_v<SmemLayoutP>> smem_p;
            cute::array_aligned<Element, cute::cosize_v<SmemLayoutdS>> smem_ds;
        };
        cute::array_aligned<float, cute::cosize_v<SmemLayoutdQacc>> smem_dqacc;
        cute::array_aligned<float, 128> smem_lse;
        cute::array_aligned<float, 128> smem_dpsum;
    };
    struct {
        cute::uint64_t tma_load_mbar[8]; // 8 TMA barrier pre-allcoated for usage.
        cutlass::arch::ClusterTransactionBarrier barrier_K;
        cutlass::arch::ClusterTransactionBarrier barrier_V;
        typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_q;
        typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_do;
    };
};

template <bool Has_P_smem, int kStages, class Element, class OutputType, class SmemLayoutQ, class SmemLayoutdO,
          class SmemLayoutK, class SmemLayoutV, class SmemLayoutP, class SmemLayoutdS,
          class SmemLayoutdQ>
struct SharedStorageQKVdOdKVSeqqPar;

template <int kStages, class Element, class OutputType, class SmemLayoutQ, class SmemLayoutdO,
          class SmemLayoutK, class SmemLayoutV, class SmemLayoutP, class SmemLayoutdS,
          class SmemLayoutdQ>
struct SharedStorageQKVdOdKVSeqqPar<true, kStages, Element, OutputType, SmemLayoutQ, SmemLayoutdO,
        SmemLayoutK, SmemLayoutV, SmemLayoutP, SmemLayoutdS, SmemLayoutdQ> {
    struct {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>> smem_k;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>> smem_v;
        union {
            struct {
                cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> smem_q;
                cute::array_aligned<Element, cute::cosize_v<SmemLayoutdO>> smem_do;
            };
            struct {
                cute::array_aligned<Element, cute::cosize_v<SmemLayoutdQ>> smem_dq;
            };
        };
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutP>> smem_p;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutdS>> smem_ds;
    };
    struct {
        cute::uint64_t tma_load_mbar[8]; // 8 TMA barrier pre-allcoated for usage.
        cutlass::arch::ClusterTransactionBarrier barrier_Q;
        cutlass::arch::ClusterTransactionBarrier barrier_dO;
        typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_k;
        typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_v;
    };
};

template <int kStages, class Element, class OutputType, class SmemLayoutQ, class SmemLayoutdO,
          class SmemLayoutK, class SmemLayoutV, class SmemLayoutP, class SmemLayoutdS,
          class SmemLayoutdQ>
struct SharedStorageQKVdOdKVSeqqPar<false, kStages, Element, OutputType, SmemLayoutQ, SmemLayoutdO,
        SmemLayoutK, SmemLayoutV, SmemLayoutP, SmemLayoutdS, SmemLayoutdQ> {
    struct {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>> smem_k;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>> smem_v;
        union {
            struct {
                cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> smem_q;
                cute::array_aligned<Element, cute::cosize_v<SmemLayoutdO>> smem_do;
            };
            struct {
                cute::array_aligned<Element, cute::cosize_v<SmemLayoutdQ>> smem_dq;
            };
        };
        union {  // Put smem_p in a union just so we can still refer to it in the struct, even if it's not used.
            cute::array_aligned<Element, cute::cosize_v<SmemLayoutP>> smem_p;
            cute::array_aligned<Element, cute::cosize_v<SmemLayoutdS>> smem_ds;
        };
    };
    struct {
        cute::uint64_t tma_load_mbar[8]; // 8 TMA barrier pre-allcoated for usage.
        cutlass::arch::ClusterTransactionBarrier barrier_Q;
        cutlass::arch::ClusterTransactionBarrier barrier_dO;
        typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_k;
        typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_v;
    };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_,
         bool SdP_swapAB_, bool dKV_swapAB_, bool dQ_swapAB_,
         int AtomLayoutMSdP=1, int AtomLayoutNdKV=2, int AtomLayoutMdQ=1,
         int kClusterN_ = 1, typename elem_type=cutlass::half_t>
struct Flash_bwd_kernel_traits {
    using Element = elem_type;
    using ElementAccum = float;
    using index_t = int64_t;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * cutlass::NumThreadsPerWarp;
    static constexpr int kNThreadsNonWS = 8 * cutlass::NumThreadsPerWarp;
    // static constexpr int kNThreadsdQ = cutlass::NumThreadsPerWarpGroup;
    static constexpr int kNThreadsdQ = 2 * cutlass::NumThreadsPerWarpGroup;

    static_assert(kNWarps_ == 8 || kNWarps_ == 12);

    static constexpr bool Is_WS = kNWarps_ >= 12;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static_assert(kHeadDim % 32 == 0);
    using TileShape_MNK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;

    static constexpr int kClusterN = kClusterN_;
    using ClusterShape_MNK = Shape<_1, Int<kClusterN>, _1>;

    static constexpr int kStages = 2;

    static constexpr bool SdP_swapAB = SdP_swapAB_;
    static constexpr bool dKV_swapAB = dKV_swapAB_;
    static constexpr bool dQ_swapAB = dQ_swapAB_;
    static_assert(!(SdP_swapAB && dKV_swapAB));  // If SdP_swapAB, then we don't swap for dKV

    static constexpr bool Mma_dQ_is_RS = AtomLayoutMSdP == 2 && AtomLayoutMdQ == 2 && !SdP_swapAB && !dQ_swapAB;  // If dQ_swapAB we can't use RS

    using TileShapeAtomSdP = std::conditional_t<
        !SdP_swapAB,
        Shape<Int<kBlockM>, Int<kBlockN / (2 / AtomLayoutMSdP)>, Int<kHeadDim>>,
        Shape<Int<kBlockN / (2 / AtomLayoutMSdP)>, Int<kBlockM>, Int<kHeadDim>>
    >;
    using AtomLayoutSdP = std::conditional_t<
        !SdP_swapAB,
        Layout<Shape<Int<AtomLayoutMSdP>, Int<2 / AtomLayoutMSdP>, _1>>,
        Layout<Shape<Int<2 / AtomLayoutMSdP>, Int<AtomLayoutMSdP>, _1>>
    >;
    using TiledMmaSdP = decltype(cute::make_tiled_mma(
        cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShapeAtomSdP>(),
        AtomLayoutSdP{}));

    using TileShapeAtomdKV = std::conditional_t<
        !dKV_swapAB,
        Shape<Int<kBlockN>, Int<kHeadDim / (2 / AtomLayoutNdKV)>, Int<kBlockM>>,
        Shape<Int<kHeadDim / (2 / AtomLayoutNdKV)>, Int<kBlockN>, Int<kBlockM>>
    >;
    using AtomLayoutdKV = std::conditional_t<
        !dKV_swapAB,
        Layout<Shape<Int<AtomLayoutNdKV>, Int<2 / AtomLayoutNdKV>, _1>>,
        Layout<Shape<Int<2 / AtomLayoutNdKV>, Int<AtomLayoutNdKV>, _1>>
    >;
    using TiledMmadKV = decltype(cute::make_tiled_mma(
        std::conditional_t<
            !SdP_swapAB,
            decltype(cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShapeAtomdKV, GMMA::Major::MN, GMMA::Major::MN>()),
            decltype(cute::GMMA::rs_op_selector<Element, Element, ElementAccum, TileShapeAtomdKV, GMMA::Major::K, GMMA::Major::MN>())
        >{},
        AtomLayoutdKV{}));

    using TileShapeAtomdQ = std::conditional_t<
        !dQ_swapAB,
        Shape<Int<kBlockM>, Int<kHeadDim / (2 / AtomLayoutMdQ)>, Int<kBlockN>>,
        Shape<Int<kHeadDim / (2 / AtomLayoutMdQ)>, Int<kBlockM>, Int<kBlockN>>
        // Shape<Int<kBlockM>, Int<kHeadDim >, Int<kBlockN>>,
        // Shape<Int<kHeadDim>, Int<kBlockM>, Int<kBlockN>>
    >;
    using AtomLayoutdQ = std::conditional_t<
        !dQ_swapAB,
        Layout<Shape<Int<AtomLayoutMdQ>, Int<2 / AtomLayoutMdQ>, _1>>,
        Layout<Shape<Int<2 / AtomLayoutMdQ>, Int<AtomLayoutMdQ>, _1>>
        // Layout<Shape<Int<1>, Int<1>, _1>>,
        // Layout<Shape<Int<1>, Int<1>, _1>>
    >;
    static constexpr GMMA::Major MmadQMajorA = !dQ_swapAB ? GMMA::Major::K : GMMA::Major::MN;
    static constexpr GMMA::Major MmadQMajorB = !dQ_swapAB ? GMMA::Major::MN : GMMA::Major::K;
    using TiledMmadQ = decltype(cute::make_tiled_mma(
        std::conditional_t<
            !dQ_swapAB,
            std::conditional_t<
                Mma_dQ_is_RS,
                decltype(cute::GMMA::rs_op_selector<Element, Element, ElementAccum, TileShapeAtomdQ, GMMA::Major::K, GMMA::Major::MN>()),
                decltype(cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShapeAtomdQ, GMMA::Major::K, GMMA::Major::MN>())
            >,
            decltype(cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShapeAtomdQ, GMMA::Major::MN, GMMA::Major::K>())
        >{},
        AtomLayoutdQ{}));

    using GmemTiledCopyQdO = decltype(cutlass::gemm::collective::detail::sm90_cluster_shape_to_tma_atom(shape<1>(ClusterShape_MNK{})));
    using GmemTiledCopyKV = cute::SM90_TMA_LOAD;
    using GmemTiledCopydKV = cute::SM90_TMA_STORE;

#if defined(__CUDA_ARCH__) &&  __CUDA_ARCH__ >= 800
    static constexpr bool Has_cp_async = true;
#else
    static constexpr bool Has_cp_async = false;
#endif
    // For the dot_do_o preprocessing kernel
    using Gmem_copy_struct = std::conditional_t<
        Has_cp_async,
        SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
        DefaultCopy
    >;
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");
    // Using kBlockKSmem instead of kHeadDim here to avoid bank conflicts, but doesn't seem
    // to affect speed in practice.
    static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;
    static_assert(kNThreadsNonWS % kGmemThreadsPerRow == 0, "kNThreadsNonWS must be a multiple of kGmemThreadsPerRow");
    using GmemLayoutAtom = Layout<Shape <Int<kNThreadsNonWS / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                  Stride<Int<kGmemThreadsPerRow>, _1>>;
    using GmemLayoutAtomdQ = Layout<Shape <Int<kNThreadsdQ / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                  Stride<Int<kGmemThreadsPerRow>, _1>>;
    using GmemTiledCopydO = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, elem_type>{},
                        GmemLayoutAtom{},
                        Layout<Shape < _1, _8>>{}));  // Val layout, 8 vals per store
    using GmemTiledCopydQ = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, elem_type>{},
                        GmemLayoutAtomdQ{},
                        Layout<Shape < _1, _8>>{}));  // Val layout, 8 vals per store
    using GmemLayoutAtomdQaccum = std::conditional_t<
        kBlockKSmem == 32,
        Layout<Shape <Int<kNThreadsdQ / 8>, _8>,  // Thread layout, 8 threads per row
               Stride< _8, _1>>,
        Layout<Shape <Int<kNThreadsdQ / 16>, _16>,  // Thread layout, 16 threads per row
               Stride< _16, _1>>
    >;
    using GmemTiledCopydQaccum = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{},
                        GmemLayoutAtomdQaccum{},
                        Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per store

    using SmemLayoutAtomQ = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutQ =
        decltype(tile_to_shape(SmemLayoutAtomQ{},
                 make_shape(shape<0>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{})));
    using SmemLayoutdO = SmemLayoutQ;

    using SmemLayoutAtomK = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutK = decltype(tile_to_shape(SmemLayoutAtomK{}, select<1, 2>(TileShape_MNK{})));

    using SmemLayoutAtomV = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutV = decltype(tile_to_shape(SmemLayoutAtomV{}, select<1, 2>(TileShape_MNK{})));

    using SmemLayoutAtomP = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<1>(TileShape_MNK{}))>());
    using SmemLayoutP = decltype(tile_to_shape(SmemLayoutAtomP{}, select<0, 1>(TileShape_MNK{})));
    using SmemLayoutAtomdS = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<1>(TileShape_MNK{}))>());
    using SmemLayoutdS = decltype(tile_to_shape(SmemLayoutAtomdS{}, select<0, 1>(TileShape_MNK{})));

    // using SmemLayoutAtomdQacc = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, ElementAccum,
    //     decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    // using SmemLayoutdQacc = decltype(tile_to_shape(SmemLayoutAtomdQacc{}, select<0, 2>(TileShape_MNK{})));

    // Note this is the transpose in terms of the view, not in terms of memory.
    using SmemLayoutQt =
        decltype(cute::composition(SmemLayoutQ{},
                                   make_layout(make_shape(get<2>(TileShape_MNK{}), get<0>(TileShape_MNK{}), Int<kStages>{}),
                                               make_stride(Int<kBlockM>{}, _1{}, Int<kBlockM * kHeadDim>{}))));
    using SmemLayoutdOt =
        decltype(cute::composition(SmemLayoutdO{},
                                   make_layout(make_shape(get<2>(TileShape_MNK{}), get<0>(TileShape_MNK{}), Int<kStages>{}),
                                               make_stride(Int<kBlockM>{}, _1{}, Int<kBlockM * kHeadDim>{}))));
    using SmemLayoutKt =
        decltype(cute::composition(SmemLayoutK{},
                                   make_layout(make_shape(get<2>(TileShape_MNK{}), get<1>(TileShape_MNK{})),
                                               make_stride(Int<kBlockN>{}, _1{}))));
    using SmemLayoutPt =
        decltype(cute::composition(SmemLayoutP{},
                                   make_layout(make_shape(get<1>(TileShape_MNK{}), get<0>(TileShape_MNK{})),
                                               make_stride(Int<kBlockM>{}, _1{}))));
    using SmemLayoutdSt =
        decltype(cute::composition(SmemLayoutdS{},
                                   make_layout(make_shape(get<1>(TileShape_MNK{}), get<0>(TileShape_MNK{})),
                                               make_stride(Int<kBlockM>{}, _1{}))));

    // using SmemLayoutdQacct =
    //     decltype(cute::composition(SmemLayoutdQacc{},
    //                                make_layout(make_shape(get<2>(TileShape_MNK{}), get<0>(TileShape_MNK{})),
    //                                            make_stride(Int<kBlockM>{}, _1{}))));

    using SmemLayoutdK = SmemLayoutK;
    using SmemLayoutdV = SmemLayoutV;
    using SmemLayoutdKt = SmemLayoutKt;
    using SmemLayoutdVt = SmemLayoutKt;

    static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;
    using SmemLayoutAtomdQ = decltype(
        // composition(Swizzle<kSwizzle, 3, 3>{},
        composition(Swizzle<3, 3, 3>{},
                    Layout<Shape<Int<kNThreadsdQ / 32>, Int<32>>,
                           Stride<Int<32>, _1>>{}));
    using SmemLayoutdQ = decltype(tile_to_shape(
        SmemLayoutAtomdQ{},
        make_shape(Int<kBlockM>{}, Int<kHeadDim>{})));
    using SmemLayoutdQt =
        decltype(cute::composition(SmemLayoutdQ{},
                                   make_layout(make_shape(get<2>(TileShape_MNK{}), get<0>(TileShape_MNK{})),
                                               make_stride(Int<kBlockM>{}, _1{}))));
    static constexpr int kSmemdQSize = size(SmemLayoutdQ{}) * sizeof(Element);

    using SmemLayoutAtomdQaccTMA = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, ElementAccum,
        decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<1>(TileShape_MNK{}))>());
    using SmemLayoutdQaccTMA = decltype(tile_to_shape(SmemLayoutAtomdQaccTMA{}, select<0, 2>(TileShape_MNK{})));
    using SmemLayoutdQacc = SmemLayoutdQ;
    using SmemLayoutdQacct = SmemLayoutdQt;
    using SmemLayoutdQacc2 = decltype(tile_to_shape(
        SmemLayoutAtomdQ{},
        make_shape(Int<kBlockM>{}, Int<kHeadDim>{}, _2{})));
    // using SmemLayoutdQacc = decltype(tile_to_shape(SmemLayoutAtomdQacc{}, select<0, 2>(TileShape_MNK{})));
    // using SmemLayoutdQacct =
    //     decltype(cute::composition(SmemLayoutdQacc{},
    //                                make_layout(make_shape(get<2>(TileShape_MNK{}), get<0>(TileShape_MNK{})),
    //                                            make_stride(Int<kBlockM>{}, _1{}))));
    using RmemTiledCopydQacc = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{},
                        GmemLayoutAtomdQaccum{},
                        Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per store

    // using SmemCopyAtomQ = Copy_Atom<cute::SM75_U32x4_LDSM_N, Element>;
    using SmemCopyAtomPdS = Copy_Atom<
        std::conditional_t<!SdP_swapAB, cute::SM90_U32x4_STSM_N, cute::SM90_U16x8_STSM_T>,
        Element>;
    using SmemCopyAtomdKV = Copy_Atom<
        std::conditional_t<!dKV_swapAB, cute::SM90_U32x4_STSM_N, cute::SM90_U16x8_STSM_T>,
        Element>;
    using SmemCopyAtomdQ = Copy_Atom<
        std::conditional_t<!dQ_swapAB, cute::SM90_U32x4_STSM_N, cute::SM90_U16x8_STSM_T>,
        Element>;

    using SharedStorage = std::conditional_t<
        !Is_WS,
        SharedStorageQKVdOdKV<!SdP_swapAB, kStages, Element, Element, SmemLayoutQ, SmemLayoutdO,
                              SmemLayoutK, SmemLayoutV, SmemLayoutP, SmemLayoutdS, SmemLayoutdK, SmemLayoutdV>,
        SharedStorageQKVdOdKVWS<!SdP_swapAB, kStages, Element, Element, SmemLayoutQ, SmemLayoutdO,
                              SmemLayoutK, SmemLayoutV, SmemLayoutP, SmemLayoutdS, SmemLayoutdQacc, SmemLayoutdK, SmemLayoutdV>
                              // SmemLayoutK, SmemLayoutV, SmemLayoutdS, SmemLayoutdQacc2, SmemLayoutdK, SmemLayoutdV>
    >;

    // using MainloopPipeline = typename cutlass::PipelineTmaAsync<kStages * 2>;
    // using PipelineState = typename cutlass::PipelineState<kStages * 2>;
    using MainloopPipeline = typename cutlass::PipelineTmaAsync<kStages>;

};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_,
         bool SdP_swapAB_, bool dKV_swapAB_, bool dQ_swapAB_,
         int AtomLayoutMSdP=1, int AtomLayoutNdKV=2, int AtomLayoutMdQ=1,
         int kClusterN_ = 1, typename elem_type=cutlass::half_t>
struct Flash_bwd_seqqpar_kernel_traits {
    using Element = elem_type;
    using ElementAccum = float;
    using index_t = int64_t;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * cutlass::NumThreadsPerWarp;

    static_assert(kNWarps_ == 8);

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static_assert(kHeadDim % 32 == 0);
    using TileShape_MNK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;

    static constexpr int kClusterN = kClusterN_;
    using ClusterShape_MNK = Shape<_1, Int<kClusterN>, _1>;

    static constexpr int kStages = 2;

    static constexpr bool SdP_swapAB = SdP_swapAB_;
    static constexpr bool dKV_swapAB = dKV_swapAB_;
    static constexpr bool dQ_swapAB = dQ_swapAB_;
    static_assert(!(SdP_swapAB && dKV_swapAB));  // If SdP_swapAB, then we don't swap for dKV

    static constexpr bool Mma_dQ_is_RS = AtomLayoutMSdP == 2 && AtomLayoutMdQ == 2 && !SdP_swapAB && !dQ_swapAB;  // If dQ_swapAB we can't use RS

    using TileShapeAtomSdP = std::conditional_t<
        !SdP_swapAB,
        Shape<Int<kBlockM>, Int<kBlockN / (2 / AtomLayoutMSdP)>, Int<kHeadDim>>,
        Shape<Int<kBlockN / (2 / AtomLayoutMSdP)>, Int<kBlockM>, Int<kHeadDim>>
    >;
    using AtomLayoutSdP = std::conditional_t<
        !SdP_swapAB,
        Layout<Shape<Int<AtomLayoutMSdP>, Int<2 / AtomLayoutMSdP>, _1>>,
        Layout<Shape<Int<2 / AtomLayoutMSdP>, Int<AtomLayoutMSdP>, _1>>
    >;
    using TiledMmaSdP = decltype(cute::make_tiled_mma(
        cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShapeAtomSdP>(),
        AtomLayoutSdP{}));

    using TileShapeAtomdKV = std::conditional_t<
        !dKV_swapAB,
        Shape<Int<kBlockN>, Int<kHeadDim / (2 / AtomLayoutNdKV)>, Int<kBlockM>>,
        Shape<Int<kHeadDim / (2 / AtomLayoutNdKV)>, Int<kBlockN>, Int<kBlockM>>
    >;
    using AtomLayoutdKV = std::conditional_t<
        !dKV_swapAB,
        Layout<Shape<Int<AtomLayoutNdKV>, Int<2 / AtomLayoutNdKV>, _1>>,
        Layout<Shape<Int<2 / AtomLayoutNdKV>, Int<AtomLayoutNdKV>, _1>>
    >;
    using TiledMmadKV = decltype(cute::make_tiled_mma(
        std::conditional_t<
            !SdP_swapAB,
            decltype(cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShapeAtomdKV, GMMA::Major::MN, GMMA::Major::MN>()),
            decltype(cute::GMMA::rs_op_selector<Element, Element, ElementAccum, TileShapeAtomdKV, GMMA::Major::K, GMMA::Major::MN>())
        >{},
        AtomLayoutdKV{}));

    using TileShapeAtomdQ = std::conditional_t<
        !dQ_swapAB,
        Shape<Int<kBlockM>, Int<kHeadDim / (2 / AtomLayoutMdQ)>, Int<kBlockN>>,
        Shape<Int<kHeadDim / (2 / AtomLayoutMdQ)>, Int<kBlockM>, Int<kBlockN>>
    >;
    using AtomLayoutdQ = std::conditional_t<
        !dQ_swapAB,
        Layout<Shape<Int<AtomLayoutMdQ>, Int<2 / AtomLayoutMdQ>, _1>>,
        Layout<Shape<Int<2 / AtomLayoutMdQ>, Int<AtomLayoutMdQ>, _1>>
    >;
    static constexpr GMMA::Major MmadQMajorA = !dQ_swapAB ? GMMA::Major::K : GMMA::Major::MN;
    static constexpr GMMA::Major MmadQMajorB = !dQ_swapAB ? GMMA::Major::MN : GMMA::Major::K;
    using TiledMmadQ = decltype(cute::make_tiled_mma(
        std::conditional_t<
            !dQ_swapAB,
            std::conditional_t<
                Mma_dQ_is_RS,
                decltype(cute::GMMA::rs_op_selector<Element, Element, ElementAccum, TileShapeAtomdQ, GMMA::Major::K, GMMA::Major::MN>()),
                decltype(cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShapeAtomdQ, GMMA::Major::K, GMMA::Major::MN>())
            >,
            decltype(cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShapeAtomdQ, GMMA::Major::MN, GMMA::Major::K>())
        >{},
        AtomLayoutdQ{}));

    using GmemTiledCopyQdO = decltype(cutlass::gemm::collective::detail::sm90_cluster_shape_to_tma_atom(shape<1>(ClusterShape_MNK{})));
    using GmemTiledCopyKV = cute::SM90_TMA_LOAD;
    using GmemTiledCopydKV = cute::SM90_TMA_STORE;

#if defined(__CUDA_ARCH__) &&  __CUDA_ARCH__ >= 800
    static constexpr bool Has_cp_async = true;
#else
    static constexpr bool Has_cp_async = false;
#endif
    // For the dot_do_o preprocessing kernel
    using Gmem_copy_struct = std::conditional_t<
        Has_cp_async,
        SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
        DefaultCopy
    >;
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");
    // Using kBlockKSmem instead of kHeadDim here to avoid bank conflicts, but doesn't seem
    // to affect speed in practice.
    static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;
    static_assert(kNThreads % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");
    using GmemLayoutAtom = Layout<Shape <Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                  Stride<Int<kGmemThreadsPerRow>, _1>>;
    using GmemTiledCopydO = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, elem_type>{},
                        GmemLayoutAtom{},
                        Layout<Shape < _1, _8>>{}));  // Val layout, 8 vals per store
    using GmemTiledCopydQ = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, elem_type>{},
                        GmemLayoutAtom{},
                        Layout<Shape < _1, _8>>{}));  // Val layout, 8 vals per store
    using GmemLayoutAtomdQaccum = std::conditional_t<
        kBlockKSmem == 32,
        Layout<Shape <_32, _8>,  // Thread layout, 8 threads per row
               Stride< _8, _1>>,
        Layout<Shape <_16, _16>,  // Thread layout, 16 threads per row
               Stride< _16, _1>>
    >;
    using GmemTiledCopydQaccum = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{},
                        GmemLayoutAtomdQaccum{},
                        Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per store

    using SmemLayoutAtomQ = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQ{}, select<0, 2>(TileShape_MNK{})));
    using SmemLayoutdO = SmemLayoutQ;

    using SmemLayoutAtomK = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutK = decltype(tile_to_shape(SmemLayoutAtomK{},
                 make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{})));

    using SmemLayoutAtomV = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutV = decltype(tile_to_shape(SmemLayoutAtomV{},
                 make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{})));

    using SmemLayoutAtomP = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<1>(TileShape_MNK{}))>());
    using SmemLayoutP = decltype(tile_to_shape(SmemLayoutAtomP{}, select<0, 1>(TileShape_MNK{})));
    using SmemLayoutAtomdS = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<1>(TileShape_MNK{}))>());
    using SmemLayoutdS = decltype(tile_to_shape(SmemLayoutAtomdS{}, select<0, 1>(TileShape_MNK{})));

    // Note this is the transpose in terms of the view, not in terms of memory.
    using SmemLayoutQt =
        decltype(cute::composition(SmemLayoutQ{},
                                   make_layout(make_shape(get<2>(TileShape_MNK{}), get<0>(TileShape_MNK{})),
                                               make_stride(Int<kBlockM>{}, _1{}))));
    using SmemLayoutdOt =
        decltype(cute::composition(SmemLayoutdO{},
                                   make_layout(make_shape(get<2>(TileShape_MNK{}), get<0>(TileShape_MNK{})),
                                               make_stride(Int<kBlockM>{}, _1{}))));
    using SmemLayoutKt =
        decltype(cute::composition(SmemLayoutK{},
                                   make_layout(make_shape(get<2>(TileShape_MNK{}), get<1>(TileShape_MNK{}), Int<kStages>{}),
                                               make_stride(Int<kBlockN>{}, _1{}, Int<kBlockN * kHeadDim>{}))));
    using SmemLayoutPt =
        decltype(cute::composition(SmemLayoutP{},
                                   make_layout(make_shape(get<1>(TileShape_MNK{}), get<0>(TileShape_MNK{})),
                                               make_stride(Int<kBlockM>{}, _1{}))));
    using SmemLayoutdSt =
        decltype(cute::composition(SmemLayoutdS{},
                                   make_layout(make_shape(get<1>(TileShape_MNK{}), get<0>(TileShape_MNK{})),
                                               make_stride(Int<kBlockM>{}, _1{}))));

    using SmemLayoutdK = decltype(tile_to_shape(SmemLayoutAtomK{}, select<1, 2>(TileShape_MNK{})));
    using SmemLayoutdV = SmemLayoutdK;
    using SmemLayoutdKt = SmemLayoutKt;
    using SmemLayoutdVt = SmemLayoutKt;
    using SmemLayoutdQTMA = decltype(tile_to_shape(SmemLayoutAtomK{}, select<0, 2>(TileShape_MNK{})));

    static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;
    using SmemLayoutAtomdQ = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<_8, Int<kBlockKSmem>>,
                           Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutdQ = decltype(tile_to_shape(
        SmemLayoutAtomdQ{},
        make_shape(Int<kBlockM>{}, Int<kHeadDim>{})));
    using SmemLayoutdQt =
        decltype(cute::composition(SmemLayoutdQ{},
                                   make_layout(make_shape(get<2>(TileShape_MNK{}), get<0>(TileShape_MNK{})),
                                               make_stride(Int<kBlockM>{}, _1{}))));
    static constexpr int kSmemdQSize = size(SmemLayoutdQ{}) * sizeof(Element);

    using SmemLayoutAtomdKV = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<_8, Int<kBlockKSmem>>,
                           Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutdKV = decltype(tile_to_shape(
        SmemLayoutAtomdKV{},
        make_shape(Int<kBlockN>{}, Int<kHeadDim>{})));
    using SmemLayoutdKVt =
        decltype(cute::composition(SmemLayoutdKV{},
                                   make_layout(make_shape(get<2>(TileShape_MNK{}), get<1>(TileShape_MNK{})),
                                               make_stride(Int<kBlockN>{}, _1{}))));
    static constexpr int kSmemdKVSize = size(SmemLayoutdKV{}) * sizeof(Element) * 2;

    // using SmemCopyAtomQ = Copy_Atom<cute::SM75_U32x4_LDSM_N, Element>;
    using SmemCopyAtomPdS = Copy_Atom<
        std::conditional_t<!SdP_swapAB, cute::SM90_U32x4_STSM_N, cute::SM90_U16x8_STSM_T>,
        Element>;
    using SmemCopyAtomdKV = Copy_Atom<
        std::conditional_t<!dKV_swapAB, cute::SM90_U32x4_STSM_N, cute::SM90_U16x8_STSM_T>,
        Element>;
    using SmemCopyAtomdQ = Copy_Atom<
        std::conditional_t<!dQ_swapAB, cute::SM90_U32x4_STSM_N, cute::SM90_U16x8_STSM_T>,
        Element>;

    using SharedStorage = SharedStorageQKVdOdKVSeqqPar<!SdP_swapAB, kStages, Element, Element, SmemLayoutQ, SmemLayoutdO,
        SmemLayoutK, SmemLayoutV, SmemLayoutP, SmemLayoutdS, SmemLayoutdQTMA>;

    // using MainloopPipeline = typename cutlass::PipelineTmaAsync<kStages * 2>;
    // using PipelineState = typename cutlass::PipelineState<kStages * 2>;
    using MainloopPipeline = typename cutlass::PipelineTmaAsync<kStages>;

};

////////////////////////////////////////////////////////////////////////////////////////////////////
