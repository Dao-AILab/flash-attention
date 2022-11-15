#pragma once

#ifdef OLD_GENERATOR_PATH
#include <ATen/CUDAGeneratorImpl.h>
#else
#include <ATen/cuda/CUDAGeneratorImpl.h>
#endif

#include <ATen/cuda/detail/UnpackRaw.cuh>  // For at::cuda::philox::unpack
#include <curand_kernel.h>

#include "ln.h"

namespace layer_norm {

template<typename Ktraits, bool Is_dropout, bool Has_residual, bool Has_rowscale>
__global__ __launch_bounds__(Ktraits::THREADS_PER_CTA) 
void ln_fwd_kernel(FwdParams params) {

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

    constexpr bool save_x = Has_residual || Is_dropout || !(std::is_same<input_t, residual_t>::value);

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

    const input_t *rowscale = static_cast<input_t *>(params.rowscale);

    // https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cuda/Dropout.cu
    curandStatePhilox4_32_10_t state;
    if (Is_dropout) {
        auto seeds = at::cuda::philox::unpack(params.philox_args);
        const index_t tidx_global = blockIdx.x * blockDim.x + threadIdx.x;
        curand_init(std::get<0>(seeds), tidx_global, std::get<1>(seeds), &state);
    }

    Wvec gamma[LDGS];
    Wvec beta[LDGS];
    index_t idx = c;
    #pragma unroll
    for( int it = 0; it < LDGS; it++ ) {
        gamma[it].load_from(params.gamma, idx);
        beta[it].load_from(params.beta, idx);
        idx += VEC_COLS_PER_LDG;
    }

    constexpr compute_t rn = 1.f / compute_t(Ktraits::COLS);

    for( int row = r; row < params.rows; row += params.ctas_per_col * ROWS_PER_CTA ) {
        const compute_t rowscale_val = Has_rowscale ? compute_t(rowscale[row]) : 1.0f;
        index_t idx = row * Ktraits::VEC_COLS + c;
        compute_t xf[LDGS * NUM_ELTS];
        #pragma unroll
        for( int it = 0; it < LDGS; it++ ) {
            Ivec x0;
            Rvec x1;
            Rvec x;
            Mvec dmask;
            x0.load_from(params.x0, idx);
            if (Has_residual) { x1.load_from(params.x1, idx); }
            #pragma unroll
            for( int jt = 0; jt < NUM_ELTS; jt++ ) {
                // TD [2022-04-22]: We're memory bound, not compute bound, so we don't need to use
                // the more efficient curand_uniform4.
                mask_t keep = true;
                if (Is_dropout) {
                    float rand = curand_uniform(&state);
                    keep = mask_t(rand <= params.dropout_keep_p);
                }
                compute_t x0_ij = Has_rowscale ? compute_t(x0.data.elt[jt]) * rowscale_val : compute_t(x0.data.elt[jt]);
                compute_t x_ij;
                if (Has_residual) {
                    compute_t x1_ij = compute_t(x1.data.elt[jt]);
                    x_ij = keep ? (Is_dropout ? x0_ij * params.dropout_scale : x0_ij) + x1_ij : x1_ij;
                } else  {
                    x_ij = keep ? (Is_dropout ? x0_ij * params.dropout_scale : x0_ij) : 0.f;
                }
                if (save_x) { x.data.elt[jt] = x_ij; }
                xf[it * NUM_ELTS + jt] = x_ij;
                if (Is_dropout) { dmask.data.elt[jt] = keep; }
            }
            if (save_x) { x.store_to(params.x, idx); }
            if (Is_dropout) { dmask.store_to(params.dmask, idx); }
            idx += VEC_COLS_PER_LDG;
        }

        stats_t s = stats.compute(xf, rn);

        compute_t mu = layer_norm::Get<0>::of<stats_t, compute_t>(s);
        compute_t m2 = layer_norm::Get<1>::of<stats_t, compute_t>(s);

        if( bidn == 0 && warp_n == 0 && lane == 0 ) {
            mu_ptr[row] = mu;
        }

        compute_t rs = rsqrtf(rn * m2 + params.epsilon);

        if( bidn == 0 && warp_n == 0 && lane == 0 ) {
            rs_ptr[row] = rs;
        }

        idx = row * Ktraits::VEC_COLS + c;
        #pragma unroll
        for( int it = 0; it < LDGS; it++ ) {
            Ovec z;
            #pragma unroll
            for( int jt = 0; jt < NUM_ELTS; jt++ ) {
                output_t y_ij = output_t(rs * (xf[it * NUM_ELTS + jt] - mu));
                output_t g_ij = gamma[it].data.elt[jt];
                output_t b_ij = beta[it].data.elt[jt];
                z.data.elt[jt] = (g_ij * y_ij + b_ij);
            }
            z.store_to(params.z, idx);
            idx += VEC_COLS_PER_LDG;
        }

    }
}

}  // namespace layer_norm
