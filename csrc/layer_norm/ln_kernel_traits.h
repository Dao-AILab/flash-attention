#pragma once

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace layer_norm {
template<
    uint32_t HIDDEN_SIZE_,
    typename weight_t_,
    typename input_t_,
    typename residual_t_,
    typename output_t_,
    typename compute_t_,
    typename index_t_,
    uint32_t THREADS_PER_CTA_
>
struct Kernel_traits_base {

    using weight_t = weight_t_;
    using input_t = input_t_;
    using residual_t = residual_t_;
    using output_t = output_t_;
    using compute_t = compute_t_;
    using index_t = index_t_;

    enum { HIDDEN_SIZE = HIDDEN_SIZE_ };
    enum { THREADS_PER_CTA = THREADS_PER_CTA_ };
    enum { THREADS_PER_WARP = 32 };

};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    uint32_t HIDDEN_SIZE_,
    typename weight_t_,
    typename input_t_,
    typename residual_t_,
    typename output_t_,
    typename compute_t_,
    typename index_t_,
    bool Has_colscale,
    uint32_t THREADS_PER_CTA_,
    uint32_t BYTES_PER_LDG_,
    typename Base = Kernel_traits_base<HIDDEN_SIZE_,
                                        weight_t_,
                                        input_t_,
                                        residual_t_,
                                        output_t_,
                                        compute_t_,
                                        index_t_,
                                        THREADS_PER_CTA_>
>
struct Kernel_traits_finalize : public Base {
    enum { ROWS_PER_CTA = Base::THREADS_PER_CTA / Base::THREADS_PER_WARP };
    static_assert((int) ROWS_PER_CTA <= (int) Base::THREADS_PER_WARP);
    // Bytes per global load from the input. 
    enum { BYTES_PER_LDG = BYTES_PER_LDG_ };
    // Number of elements fetched by a global load.
    enum { ELTS_PER_LDG = BYTES_PER_LDG / sizeof(compute_t_) };
    // Bytes per global store of the weights.
    enum { BYTES_PER_STG = ELTS_PER_LDG * sizeof(weight_t_) };
    static_assert(sizeof(BYTES_PER_LDG) == 4, "Conflict-free smem transpose only implemented for 4B compute type!");
    static_assert(Base::THREADS_PER_CTA == ROWS_PER_CTA * Base::THREADS_PER_WARP, "We assume one warp per row!");
    // The total number of BYTES_PER_LDG-wide words in a hidden vector.
    enum { COLS = HIDDEN_SIZE_ * sizeof(compute_t_) / BYTES_PER_LDG };
    static_assert(COLS * BYTES_PER_LDG == HIDDEN_SIZE_ * sizeof(compute_t_));

    // Shared memory size to transpose the CTA result.
    enum { SMEM_BYTES_TRANSPOSE = Base::THREADS_PER_CTA * BYTES_PER_LDG };
    // Shared memory size to coalsece the CTA result.
    enum { SMEM_BYTES_OUTPUT = Base::THREADS_PER_WARP * BYTES_PER_LDG };
    // Shared memory requirement per CTA. 
    static constexpr int NUM_FACTORS = Has_colscale ? 3 : 2;
    enum { SMEM_BYTES_PER_CTA = NUM_FACTORS * SMEM_BYTES_TRANSPOSE + NUM_FACTORS * SMEM_BYTES_OUTPUT };

    // The type of the reducer.
    using Reducer = layer_norm::Reducer<compute_t_, 1, 1, 1>;

    // Condition for the whole CTA to participate in syncthreads.
    static_assert(COLS % Base::THREADS_PER_WARP == 0);
    enum { CTAS = COLS / Base::THREADS_PER_WARP };
}; 

////////////////////////////////////////////////////////////////////////////////////////////////////


template<
    typename weight_t_,
    typename input_t_,
    typename residual_t_,
    typename output_t_,
    typename compute_t_,
    typename index_t_,
    uint32_t HIDDEN_SIZE_, 
    uint32_t CTAS_PER_ROW_, 
    uint32_t WARPS_M_, 
    uint32_t WARPS_N_, 
    uint32_t BYTES_PER_LDG_ = 16,
    typename Base = Kernel_traits_base<
        HIDDEN_SIZE_,
        weight_t_, 
        input_t_,
        residual_t_,
        output_t_, 
        compute_t_, 
        index_t_, 
        WARPS_M_*WARPS_N_*THREADS_PER_WARP
        >
>
struct Kernel_traits : public Base {

    using input_t = typename Base::input_t;
    using residual_t = typename Base::residual_t;
    using weight_t = typename Base::weight_t;
    using compute_t = typename Base::compute_t;
    using output_t = typename Base::output_t;
    using index_t = typename Base::index_t;
    // using mask_t = unsigned char;
    using mask_t = bool;

    enum { CTAS_PER_ROW = CTAS_PER_ROW_ };
    enum { WARPS_M = WARPS_M_ };
    enum { WARPS_N = WARPS_N_ };
    enum { COLS = HIDDEN_SIZE_ };
    enum { HIDDEN_SIZE = HIDDEN_SIZE_ };
    enum { BYTES_PER_LDG = BYTES_PER_LDG_ };
    enum { NUM_ELTS = BYTES_PER_LDG / sizeof(input_t) };

    enum { THREADS_PER_ROW = WARPS_N * THREADS_PER_WARP };
    enum { THREADS_PER_CTA = WARPS_M * THREADS_PER_ROW };
    enum { ROWS_PER_CTA = WARPS_M };

    enum { BYTES_PER_ROW = COLS * sizeof(input_t) };
    enum { BYTES_PER_ROW_PER_CTA = THREADS_PER_ROW * BYTES_PER_LDG };
    // Multi-row per CTA not supported for multi-CTA => no smem for WGRAD needed
    enum { SMEM_BYTES_WGRAD = CTAS_PER_ROW > 1 ? 0 : ROWS_PER_CTA * COLS * sizeof(compute_t) };
    static_assert(WARPS_M == 1 || CTAS_PER_ROW == 1);

    using reduce_t = typename layer_norm::TypeToVec2<compute_t>::Type;
    using Reducer = layer_norm::Reducer<reduce_t, CTAS_PER_ROW, WARPS_M, WARPS_N>; 

    enum { SMEM_BYTES_DGRAD = Reducer::SMEM_BYTES };
    enum { SMEM_BYTES = SMEM_BYTES_DGRAD  + SMEM_BYTES_WGRAD };

    using Ivec = layer_norm::Vec<input_t, NUM_ELTS>;
    using Rvec = layer_norm::Vec<residual_t, NUM_ELTS>;
    using Ovec = layer_norm::Vec<output_t, NUM_ELTS>;
    using Wvec = layer_norm::Vec<weight_t, NUM_ELTS>;
    using Cvec = layer_norm::Vec<compute_t, NUM_ELTS>;
    using Mvec = layer_norm::Vec<mask_t, NUM_ELTS>;
    enum { ELTS_PER_LDG = BYTES_PER_LDG / sizeof(input_t) };

    // Assume that each thread can handle the same number of elements in the output and weights as in the input.
    static_assert(sizeof(input_t) == sizeof(output_t));
    static_assert(sizeof(input_t) <= sizeof(residual_t));
    // The number of columns fetched per load from input: one per thread.
    enum { VEC_COLS_PER_LDG =  CTAS_PER_ROW * THREADS_PER_ROW };
    // The total number of vectorized loads/stores per hidden vector.
    enum { VEC_COLS = COLS / ELTS_PER_LDG };
    // The number of loads per thread for the input.
    enum { LDGS = VEC_COLS / VEC_COLS_PER_LDG };
    static_assert(LDGS * VEC_COLS_PER_LDG  == VEC_COLS);
    //static_assert(LDGS * BYTES_PER_ROW_PER_CTA * CTAS_PER_ROW == BYTES_PER_ROW, "");

    using Stats = layer_norm::Stats<compute_t, CTAS_PER_ROW, WARPS_M, WARPS_N>;
    enum { SMEM_BYTES_FWD = Stats::SMEM_BYTES };

};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace layer_norm
