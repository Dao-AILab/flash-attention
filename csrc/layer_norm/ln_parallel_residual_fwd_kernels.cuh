#pragma once

#ifdef OLD_GENERATOR_PATH
#include <ATen/CUDAGeneratorImpl.h>
#else
#include <ATen/cuda/CUDAGeneratorImpl.h>
#endif

#include <ATen/cuda/detail/UnpackRaw.cuh>  // For at::cuda::philox::unpack
#include <curand_kernel.h>

#include "ln.h"
#include "ln_utils.cuh"
#include "ln_kernel_traits.h"
#include "static_switch.h"

namespace layer_norm {

template<typename Ktraits, bool Is_dropout, bool Tied_norm, bool Is_even_cols>
__global__ __launch_bounds__(Ktraits::THREADS_PER_CTA) 
void ln_parallel_residual_fwd_kernel(FwdParams params) {

    enum { ROWS_PER_CTA = Ktraits::ROWS_PER_CTA };
    enum { WARPS_N = Ktraits::WARPS_N };
    enum { WARPS_M = Ktraits::WARPS_M };
    enum { THREADS_PER_ROW = Ktraits::THREADS_PER_ROW };
    enum { VEC_COLS_PER_LDG = Ktraits::VEC_COLS_PER_LDG };
    enum { BYTES_PER_ROW = Ktraits::BYTES_PER_ROW };
    enum { LDGS = Ktraits::LDGS };
    enum { NUM_ELTS = Ktraits::NUM_ELTS };
    enum { CTAS_PER_ROW = Ktraits::CTAS_PER_ROW };

    using input_t = typename Ktraits::input_t;
    using residual_t = typename Ktraits::residual_t;
    using output_t = typename Ktraits::output_t;
    using index_t = typename Ktraits::index_t;
    using compute_t = typename Ktraits::compute_t;
    using mask_t = typename Ktraits::mask_t;
    using Ivec = typename Ktraits::Ivec;
    using Rvec = typename Ktraits::Rvec;
    using Ovec = typename Ktraits::Ovec;
    using Wvec = typename Ktraits::Wvec;
    using Cvec = typename Ktraits::Cvec;
    using Mvec = typename Ktraits::Mvec;

    using Stats = typename Ktraits::Stats;
    using stats_t = typename Stats::stats_t;

    const bool has_residual = params.residual != nullptr;
    const bool has_x1 = params.x1 != nullptr;
    const bool save_x = has_residual || has_x1 || Is_dropout || !(std::is_same<input_t, residual_t>::value);

    extern __shared__ char smem_[];

    const index_t tidx = threadIdx.x;
    const index_t bidn = blockIdx.x % CTAS_PER_ROW;
    const index_t bidm = blockIdx.x / CTAS_PER_ROW;
    const index_t lane = tidx % THREADS_PER_WARP;
    const index_t warp = tidx / THREADS_PER_WARP;
    const index_t warp_m = warp / WARPS_N;
    const index_t warp_n = warp % WARPS_N;

    const index_t r = bidm * ROWS_PER_CTA + warp_m;
    const index_t c = bidn * THREADS_PER_ROW + warp_n * THREADS_PER_WARP + lane;

    Stats stats(params, bidm, bidn, warp_m, warp_n, lane, smem_);

    compute_t *mu_ptr = static_cast<compute_t *>(params.mu);
    compute_t *rs_ptr = static_cast<compute_t *>(params.rs);

    // https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cuda/Dropout.cu
    curandStatePhilox4_32_10_t state;
    if (Is_dropout) {
        auto seeds = at::cuda::philox::unpack(params.philox_args);
        const index_t tidx_global = blockIdx.x * blockDim.x + threadIdx.x;
        curand_init(std::get<0>(seeds), tidx_global, std::get<1>(seeds), &state);
    }

    const index_t num_valid_ldgs = ((params.cols / Ktraits::ELTS_PER_LDG) - 1 - c + VEC_COLS_PER_LDG) / VEC_COLS_PER_LDG;

    Wvec gamma0[LDGS];
    Wvec beta0[LDGS];
    Wvec gamma1[LDGS];
    Wvec beta1[LDGS];
    index_t idx = c;
    #pragma unroll
    for( int it = 0; it < LDGS; it++ ) {
        if (Is_even_cols || (it < num_valid_ldgs)) {
            gamma0[it].load_from(params.gamma, idx);
            if (params.beta != nullptr) {
                beta0[it].load_from(params.beta, idx);
            } else {
                beta0[it].zero_();
            }
            if (!Tied_norm) {
                gamma1[it].load_from(params.gamma1, idx);
                if (params.beta1 != nullptr) {
                    beta1[it].load_from(params.beta1, idx);
                } else {
                    beta1[it].zero_();
                }
            }
            idx += VEC_COLS_PER_LDG;
        }
    }

    for( int row = r; row < params.rows; row += params.ctas_per_col * ROWS_PER_CTA ) {
        index_t idx = row * params.cols / Ktraits::ELTS_PER_LDG + c;
        compute_t xf[LDGS * NUM_ELTS];
        #pragma unroll
        for( int it = 0; it < LDGS; it++ ) {
            if (Is_even_cols || (it < num_valid_ldgs)) {
                Ivec x0;
                Ivec x1;
                Rvec residual;
                Rvec x;
                Mvec dmask0;
                Mvec dmask1;
                x0.load_from(params.x0, idx);
                if (has_x1) { x1.load_from(params.x1, idx); }
                if (has_residual) { residual.load_from(params.residual, idx); }
                #pragma unroll
                for( int jt = 0; jt < NUM_ELTS; jt++ ) {
                    // TD [2022-04-22]: We're memory bound, not compute bound, so we don't need to use
                    // the more efficient curand_uniform4.
                    compute_t x_ij;
                    mask_t keep0 = !Is_dropout ? true : curand_uniform(&state) <= params.dropout_keep_p;
                    if (Is_dropout) { dmask0.data.elt[jt] = keep0; }
                    compute_t x0_ij = compute_t(x0.data.elt[jt]);
                    x0_ij = keep0 ? (Is_dropout ? x0_ij * params.dropout_scale : x0_ij) : 0.0f;
                    if (has_x1) {
                        mask_t keep1 = !Is_dropout ? true : curand_uniform(&state) <= params.dropout_keep_p;
                        if (Is_dropout) { dmask1.data.elt[jt] = keep1; }
                        compute_t x1_ij = compute_t(x1.data.elt[jt]);
                        x1_ij = keep1 ? (Is_dropout ? x1_ij * params.dropout_scale : x1_ij) : 0.0f;
                        x_ij = has_residual ? x0_ij + x1_ij + compute_t(residual.data.elt[jt]) : x0_ij + x1_ij;
                    } else {
                        x_ij = has_residual ? x0_ij + compute_t(residual.data.elt[jt]) : x0_ij;
                    }
                    if (save_x) { x.data.elt[jt] = x_ij; }
                    xf[it * NUM_ELTS + jt] = x_ij;
                }
                if (save_x) { x.store_to(params.x, idx); }
                if (Is_dropout) {
                    dmask0.store_to(params.dmask, idx);
                    if (has_x1) { dmask1.store_to(params.dmask1, idx); }
                }
                idx += VEC_COLS_PER_LDG;
            }
        }

        static_assert(CTAS_PER_ROW == 1, "Don't support multiple CTAs per row for now");
        const index_t num_vecs = params.cols / Ktraits::ELTS_PER_LDG;
        const index_t num_full_ldgs = num_vecs / Ktraits::VEC_COLS_PER_LDG;
        const index_t remaining_vecs = num_vecs % Ktraits::VEC_COLS_PER_LDG;
        auto valid_elts_in_warp_fn = [num_full_ldgs, remaining_vecs] (int warp_n) -> int {
            // Need to convert to int, otherwise the subtraction will wrap around.
            const index_t valid_partial_vecs_in_warp =
                std::min(std::max(int(remaining_vecs) - int(warp_n * THREADS_PER_WARP), int(0)),
                        int(THREADS_PER_WARP));
            return (num_full_ldgs * THREADS_PER_WARP + valid_partial_vecs_in_warp) * NUM_ELTS;
        };
        stats_t s = stats.template compute<Is_even_cols>(
            xf, params.inverse_cols, valid_elts_in_warp_fn, num_valid_ldgs * NUM_ELTS
        );

        compute_t mu = layer_norm::Get<0>::of<stats_t, compute_t>(s);
        compute_t m2 = layer_norm::Get<1>::of<stats_t, compute_t>(s);

        if( bidn == 0 && warp_n == 0 && lane == 0 ) {
            mu_ptr[row] = mu;
        }

        compute_t rs = rsqrtf(m2 * params.inverse_cols + params.epsilon + (!params.is_rms_norm ? 0.f : mu * mu));

        if( bidn == 0 && warp_n == 0 && lane == 0 ) {
            rs_ptr[row] = rs;
        }

        idx = row * params.cols / Ktraits::ELTS_PER_LDG + c;
        #pragma unroll
        for( int it = 0; it < LDGS; it++ ) {
            if (Is_even_cols || (it < num_valid_ldgs)) {
                Ovec z0;
                Ovec z1;
                #pragma unroll
                for( int jt = 0; jt < NUM_ELTS; jt++ ) {
                    compute_t y_ij = compute_t(rs * (xf[it * NUM_ELTS + jt] - (!params.is_rms_norm ? mu : 0.f)));
                    compute_t g0_ij = gamma0[it].data.elt[jt];
                    compute_t b0_ij = beta0[it].data.elt[jt];
                    z0.data.elt[jt] = output_t(g0_ij * y_ij + b0_ij);
                    if (!Tied_norm) {
                        compute_t g1_ij = gamma1[it].data.elt[jt];
                        compute_t b1_ij = beta1[it].data.elt[jt];
                        z1.data.elt[jt] = output_t(g1_ij * y_ij + b1_ij);
                    }
                }
                z0.store_to(params.z, idx);
                if (!Tied_norm) { z1.store_to(params.z1, idx); }
                idx += VEC_COLS_PER_LDG;
            }
        }

    }
}

}  // namespace layer_norm

using namespace layer_norm;

template<
    typename weight_t,
    typename input_t,
    typename residual_t,
    typename output_t,
    typename compute_t,
    typename index_t,
    int HIDDEN_SIZE,
    int CTAS_PER_ROW,
    int WARPS_M,
    int WARPS_N,
    int BYTES_PER_LDG
>
void launch_parallel_residual_(LaunchParams<FwdParams> &launch_params, const bool configure_params){

    using Kernel_traits = Kernel_traits<weight_t,
                                        input_t,
                                        residual_t,
                                        output_t,
                                        compute_t,
                                        index_t,
                                        HIDDEN_SIZE,
                                        CTAS_PER_ROW,
                                        WARPS_M,
                                        WARPS_N,
                                        BYTES_PER_LDG
                                        >;
    bool is_even_cols = launch_params.params.cols == HIDDEN_SIZE;
    bool tied_norm = launch_params.params.gamma1 == nullptr;
    BOOL_SWITCH(launch_params.params.dropout_keep_p < 1.f, IsDropoutConst, [&] {
        BOOL_SWITCH(tied_norm, TiedNormConst, [&] {
            BOOL_SWITCH(is_even_cols, IsEvenColsConst, [&] {
                auto kernel = &ln_parallel_residual_fwd_kernel<Kernel_traits, IsDropoutConst, TiedNormConst, IsEvenColsConst>;
                if( configure_params ) {
                    int ctas_per_sm;
                    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                        &ctas_per_sm, kernel, Kernel_traits::THREADS_PER_CTA, Kernel_traits::SMEM_BYTES_FWD));
                    launch_params.params.ctas_per_col = launch_params.props->multiProcessorCount * ctas_per_sm / Kernel_traits::CTAS_PER_ROW;
                    const size_t rows_per_loop = launch_params.params.ctas_per_col * Kernel_traits::ROWS_PER_CTA;
                    launch_params.elts_per_thread = (launch_params.params.rows + rows_per_loop - 1) / rows_per_loop * Kernel_traits::LDGS * Kernel_traits::NUM_ELTS;
                    launch_params.barrier_size = 0;
                    launch_params.workspace_bytes = 0;
                    if(Kernel_traits::CTAS_PER_ROW > 1) {
                        launch_params.barrier_size = 2 * launch_params.params.ctas_per_col;
                        launch_params.workspace_bytes = launch_params.params.ctas_per_col
                                                      * Kernel_traits::WARPS_M
                                                      * Kernel_traits::CTAS_PER_ROW
                                                      * sizeof(typename Kernel_traits::Stats::stats_t)
                                                      * 2;
                    }
                    return;
                }

                if( Kernel_traits::SMEM_BYTES_FWD >= 48 * 1024 ) {
                    CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Kernel_traits::SMEM_BYTES_FWD));
                }
                auto stream = launch_params.stream;
                auto ctas_per_col = launch_params.params.ctas_per_col;

                if( Kernel_traits::CTAS_PER_ROW == 1 ) {
                    kernel<<<ctas_per_col, Kernel_traits::THREADS_PER_CTA, Kernel_traits::SMEM_BYTES_FWD, stream>>>(launch_params.params);
                } else {
                    dim3 grid(Kernel_traits::CTAS_PER_ROW * ctas_per_col);
                    dim3 block(Kernel_traits::THREADS_PER_CTA);
                    void *params_ = (void *)&launch_params.params;
                    cudaLaunchCooperativeKernel((void *)kernel, grid, block, (void **)&params_, Kernel_traits::SMEM_BYTES_FWD, stream);
                }
            });
        });
    });
}
