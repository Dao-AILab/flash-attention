#pragma once

#include "ln.h"
#include "ln_utils.cuh"
#include "ln_kernel_traits.h"
#include "static_switch.h"
#include "ln_bwd_kernels.cuh"

namespace layer_norm {

template<typename Ktraits, bool Is_dropout, bool Tied_norm, bool Is_even_cols>
__global__ __launch_bounds__(Ktraits::THREADS_PER_CTA) 
void ln_parallel_residual_bwd_kernel(layer_norm::BwdParams params) {

    enum { ROWS_PER_CTA = Ktraits::ROWS_PER_CTA };
    enum { WARPS_M = Ktraits::WARPS_M };
    enum { WARPS_N = Ktraits::WARPS_N };
    enum { THREADS_PER_ROW = Ktraits::THREADS_PER_ROW };
    enum { COLS = Ktraits::COLS };
    enum { BYTES_PER_ROW = Ktraits::BYTES_PER_ROW };
    enum { LDGS = Ktraits::LDGS };
    enum { NUM_ELTS = Ktraits::ELTS_PER_LDG };
    enum { THREADS_PER_WARP = Ktraits::THREADS_PER_WARP };
    enum { CTAS_PER_ROW = Ktraits::CTAS_PER_ROW };

    using input_t = typename Ktraits::input_t;
    using compute_t = typename Ktraits::compute_t;
    using index_t = typename Ktraits::index_t;
    using mask_t = typename Ktraits::mask_t;
    using Ivec = typename Ktraits::Ivec;
    using Rvec = typename Ktraits::Rvec;
    using Ovec = typename Ktraits::Ovec;
    using Wvec = typename Ktraits::Wvec;
    using Cvec = typename Ktraits::Cvec;
    using Mvec = typename Ktraits::Mvec;
    using Reducer = typename Ktraits::Reducer;
    using reduce_t = typename Reducer::Type;

    extern __shared__ char smem_[];

    const bool has_residual = params.dresidual != nullptr;
    const bool has_x1 = params.dx1 != nullptr;
    const bool prenorm = params.dx != nullptr;

    const index_t tidx = threadIdx.x;
    const index_t bidn = blockIdx.x % CTAS_PER_ROW;
    const index_t bidm = blockIdx.x / CTAS_PER_ROW;
    const index_t lane = tidx % THREADS_PER_WARP;
    const index_t warp = tidx / THREADS_PER_WARP;
    const index_t warp_m = warp / Ktraits::WARPS_N;
    const index_t warp_n = warp % Ktraits::WARPS_N;
    const index_t tid_r = warp_n * THREADS_PER_WARP + lane;

    const index_t r = bidm * Ktraits::ROWS_PER_CTA + warp_m;
    const index_t c = bidn * THREADS_PER_ROW + warp_n * THREADS_PER_WARP + lane;

    static_assert(COLS == THREADS_PER_ROW * LDGS * NUM_ELTS * CTAS_PER_ROW);

    Cvec dz0y_sum[LDGS];
    Cvec dz0_sum[LDGS];
    Cvec dz1y_sum[LDGS];
    Cvec dz1_sum[LDGS];

    memset(dz0y_sum, 0, sizeof(dz0y_sum));
    memset(dz0_sum, 0, sizeof(dz0_sum));
    if (!Tied_norm) {
        memset(dz1y_sum, 0, sizeof(dz1y_sum));
        memset(dz1_sum, 0, sizeof(dz1_sum));
    }

    compute_t * smem_wgrad = reinterpret_cast<compute_t*>(smem_);
    char *smem_dgrad = smem_ + Ktraits::SMEM_BYTES_WGRAD;

    Reducer reducer(params, bidm, bidn, warp_m, warp_n, lane, smem_dgrad);

    Sum<reduce_t> sum;

    const index_t num_valid_ldgs =
        ((params.cols / Ktraits::ELTS_PER_LDG) - 1 - c + Ktraits::VEC_COLS_PER_LDG) / Ktraits::VEC_COLS_PER_LDG;

    Wvec gamma0[LDGS];
    Wvec gamma1[LDGS];
    index_t idx = c;
    #pragma unroll
    for( int it = 0; it < LDGS; it++ ) {
        if (Is_even_cols || (it < num_valid_ldgs)) {
            gamma0[it].load_from(params.gamma, idx);
            if (!Tied_norm) { gamma1[it].load_from(params.gamma1, idx); }
            idx += Ktraits::VEC_COLS_PER_LDG;
        }
    }
    // TODO if ROWS_PER_CTA does not divide rows, we might get divergence in the
    // last blocks with syncthreads!
    // grid stride over rows
    #pragma unroll 1
    for( int row = r; row < params.rows; row += params.ctas_per_col * ROWS_PER_CTA ) {
        const compute_t mu_r = static_cast<const compute_t *>(params.mu)[row];
        const compute_t rs_r = static_cast<const compute_t *>(params.rs)[row];
        Mvec dmask0[LDGS], dmask1[LDGS];
        Rvec dx[LDGS];
        compute_t dy[LDGS * NUM_ELTS];
        compute_t y[LDGS * NUM_ELTS];
        compute_t mdy_local = 0.f;
        compute_t mdyy_local = 0.f;
        index_t idx = row * params.cols / Ktraits::ELTS_PER_LDG + c;
        #pragma unroll
        for( int it = 0; it < LDGS; it++ ) {
            if (Is_even_cols || (it < num_valid_ldgs)) {
                Rvec x;
                Ovec dz0, dz1;
                dz0.load_from(params.dz, idx);
                if (!Tied_norm) { dz1.load_from(params.dz1, idx); }
                if (prenorm) { dx[it].load_from(params.dx, idx); }
                x.load_from(params.x, idx);
                if (Is_dropout) {
                    dmask0[it].load_from(params.dmask, idx);
                    if (has_x1) { dmask1[it].load_from(params.dmask1, idx); }
                }
                idx += Ktraits::VEC_COLS_PER_LDG;
                #pragma unroll
                for( int jt = 0; jt < NUM_ELTS; jt++ ) {
                    compute_t x_tmp = x.data.elt[jt];
                    compute_t y_tmp = rs_r * (x_tmp - (!params.is_rms_norm ? mu_r : 0.f));
                    compute_t dy_tmp = compute_t(gamma0[it].data.elt[jt]) * compute_t(dz0.data.elt[jt]);
                    if (!Tied_norm) {
                        dy_tmp += compute_t(gamma1[it].data.elt[jt]) * compute_t(dz1.data.elt[jt]);
                    }
                    compute_t dz0_tmp = dz0.data.elt[jt];
                    compute_t dz1_tmp;
                    if (!Tied_norm) { dz1_tmp = dz1.data.elt[jt]; }

                    mdy_local += dy_tmp;
                    mdyy_local += dy_tmp * y_tmp;

                    dy[it * NUM_ELTS + jt] = dy_tmp;
                    y[it * NUM_ELTS + jt] = y_tmp;

                    dz0y_sum[it].data.elt[jt] += dz0_tmp * y_tmp;
                    dz0_sum[it].data.elt[jt] += dz0_tmp;
                    if (!Tied_norm) {
                        dz1y_sum[it].data.elt[jt] += dz1_tmp * y_tmp;
                        dz1_sum[it].data.elt[jt] += dz1_tmp;
                    }
                }
            }
        }

        reduce_t result = reducer.allreduce({mdy_local, mdyy_local}, sum);
        mdy_local = layer_norm::Get<0>::of<reduce_t, compute_t>(result) * params.inverse_cols;
        mdyy_local = layer_norm::Get<1>::of<reduce_t, compute_t>(result) * params.inverse_cols;

        idx = row * params.cols / Ktraits::ELTS_PER_LDG + c;
        #pragma unroll
        for( int it = 0; it < LDGS; it++ ) {
            if (Is_even_cols || (it < num_valid_ldgs)) {
                Ivec dx0, dx1;
                Rvec dresidual;
                #pragma unroll
                for( int jt = 0; jt < NUM_ELTS; jt++ ) {
                    compute_t dx_tmp_res;
                    compute_t dy_tmp = dy[it * NUM_ELTS + jt];
                    compute_t y_tmp = y[it * NUM_ELTS + jt];
                    compute_t dx_tmp = rs_r * (dy_tmp - (mdyy_local * y_tmp + (!params.is_rms_norm ? mdy_local : 0.f)));
                    dx_tmp_res = prenorm ? dx_tmp + compute_t(dx[it].data.elt[jt]) : dx_tmp;
                    if (has_residual) { dresidual.data.elt[jt] = dx_tmp_res; }
                    if (Is_dropout) {
                        dx0.data.elt[jt] = dmask0[it].data.elt[jt] ? dx_tmp_res * params.dropout_scale : 0.f;
                        if (has_x1) { dx1.data.elt[jt] = dmask1[it].data.elt[jt] ? dx_tmp_res * params.dropout_scale : 0.f; }
                    } else {
                        dx0.data.elt[jt] = dx_tmp_res;
                        if (has_x1) { dx1.data.elt[jt] = dx_tmp_res; }
                    }
                }
                if (has_residual) { dresidual.store_to(params.dresidual, idx); }
                dx0.store_to(params.dx0, idx);
                if (has_x1) { dx1.store_to(params.dx1, idx); }
                idx += Ktraits::VEC_COLS_PER_LDG;
            }
        }

    }  // end: grid stride loop

    if( WARPS_M == 1 ) {
        idx = r * params.cols / Ktraits::ELTS_PER_LDG + c;
        #pragma unroll
        for( int it = 0; it < LDGS; it++ ) {
            if (Is_even_cols || (it < num_valid_ldgs)) {
                dz0_sum[it].store_to(params.dbeta_part, idx);
                dz0y_sum[it].store_to(params.dgamma_part, idx);
                if (!Tied_norm) {
                    dz1_sum[it].store_to(params.dbeta1_part, idx);
                    dz1y_sum[it].store_to(params.dgamma1_part, idx);
                }
                idx += Ktraits::VEC_COLS_PER_LDG;
            }
        }
    } else {
        static_assert(WARPS_M == 1 || Ktraits::CTAS_PER_ROW == 1, "Multiple rows per CTA not supported for Multi-CTA.");
        // Finalize reduction of part dgamma and dbeta for this CTA
        // by reducing over the rows held across the WARPS_M warps

        // Assumption: blockSize divides hidden size.
        enum { NUM_RES = COLS / Ktraits::THREADS_PER_CTA };
        static_assert(NUM_RES * Ktraits::THREADS_PER_CTA == COLS, "");

        idx = warp_m * Ktraits::VEC_COLS + tid_r;
        #pragma unroll
        for( int it = 0; it < LDGS; it++ ) {
            dz0_sum[it].store_to(smem_wgrad, idx);
            idx += THREADS_PER_ROW;
        }
        __syncthreads();
        compute_t cta_dz0_sum[NUM_RES];
        memset(cta_dz0_sum, 0, sizeof(compute_t) * NUM_RES);
        for( int it = 0; it < ROWS_PER_CTA; it++ ) {
            for( int jt = 0; jt < NUM_RES; jt++ ) {
                cta_dz0_sum[jt] += smem_wgrad[it * COLS + tidx + jt * Ktraits::THREADS_PER_CTA];
            }
        }
        __syncthreads();

        idx = warp_m * Ktraits::VEC_COLS + tid_r;
        #pragma unroll
        for( int it = 0; it < LDGS; it++ ) {
            dz0y_sum[it].store_to(smem_wgrad, idx);
            idx += THREADS_PER_ROW;
        }
        __syncthreads();
        compute_t cta_dz0y_sum[NUM_RES];
        memset(cta_dz0y_sum, 0, sizeof(compute_t) * NUM_RES);
        for( int it = 0; it < ROWS_PER_CTA; it++ ) {
            for( int jt = 0; jt < NUM_RES; jt++ ) {
                cta_dz0y_sum[jt] += smem_wgrad[it * COLS + tidx + jt * Ktraits::THREADS_PER_CTA];
            }
        }

        compute_t cta_dz1_sum[NUM_RES], cta_dz1y_sum[NUM_RES];
        if (!Tied_norm) {
            __syncthreads();
            idx = warp_m * Ktraits::VEC_COLS + tid_r;
            #pragma unroll
            for( int it = 0; it < LDGS; it++ ) {
                dz1_sum[it].store_to(smem_wgrad, idx);
                idx += THREADS_PER_ROW;
            }
            __syncthreads();
            memset(cta_dz1_sum, 0, sizeof(compute_t) * NUM_RES);
            for( int it = 0; it < ROWS_PER_CTA; it++ ) {
                for( int jt = 0; jt < NUM_RES; jt++ ) {
                    cta_dz1_sum[jt] += smem_wgrad[it * COLS + tidx + jt * Ktraits::THREADS_PER_CTA];
                }
            }
            __syncthreads();
            idx = warp_m * Ktraits::VEC_COLS + tid_r;
            #pragma unroll
            for( int it = 0; it < LDGS; it++ ) {
                dz1y_sum[it].store_to(smem_wgrad, idx);
                idx += THREADS_PER_ROW;
            }
            __syncthreads();
            memset(cta_dz1y_sum, 0, sizeof(compute_t) * NUM_RES);
            for( int it = 0; it < ROWS_PER_CTA; it++ ) {
                for( int jt = 0; jt < NUM_RES; jt++ ) {
                    cta_dz1y_sum[jt] += smem_wgrad[it * COLS + tidx + jt * Ktraits::THREADS_PER_CTA];
                }
            }
        }

        const index_t num_valid_writes
            = (params.cols - 1 - tidx + Ktraits::THREADS_PER_CTA) / Ktraits::THREADS_PER_CTA;
        compute_t *dgamma0_part = static_cast<compute_t *>(params.dgamma_part) + bidm * params.cols + tidx;
        compute_t *dbeta0_part = static_cast<compute_t *>(params.dbeta_part) + bidm * params.cols + tidx;
        compute_t *dgamma1_part = !Tied_norm ? static_cast<compute_t *>(params.dgamma1_part) + bidm * params.cols + tidx : nullptr;
        compute_t *dbeta1_part = !Tied_norm ? static_cast<compute_t *>(params.dbeta1_part) + bidm * params.cols + tidx : nullptr;
        for( int jt = 0; jt < NUM_RES; jt++ ) {
            if (Is_even_cols || (jt < num_valid_writes)) {
                *dgamma0_part = cta_dz0y_sum[jt];
                dgamma0_part += Ktraits::THREADS_PER_CTA;
                *dbeta0_part = cta_dz0_sum[jt];
                dbeta0_part += Ktraits::THREADS_PER_CTA;
                if (!Tied_norm) {
                    *dgamma1_part = cta_dz1y_sum[jt];
                    dgamma1_part += Ktraits::THREADS_PER_CTA;
                    *dbeta1_part = cta_dz1_sum[jt];
                    dbeta1_part += Ktraits::THREADS_PER_CTA;
                }
            }
        }

    }
}

template<typename Kernel_traits, bool Is_even_cols>
__global__ __launch_bounds__(Kernel_traits::THREADS_PER_CTA)
void ln_parallel_residual_bwd_finalize_kernel(BwdParams params)
{

    using compute_t = typename Kernel_traits::compute_t;
    using weight_t = typename Kernel_traits::weight_t;
    using index_t = typename Kernel_traits::index_t;
    using Reducer = typename Kernel_traits::Reducer;
    using reduce_t = typename Reducer::Type;

    Sum<reduce_t> sum;
    enum { NUM_ELT = Kernel_traits::ELTS_PER_LDG };
    enum { THREADS_PER_WARP = Kernel_traits::THREADS_PER_WARP };

    // Multiplying by 2 since we have both gamma0 and gamma1
    __shared__ char smem_[2 * Kernel_traits::SMEM_BYTES_PER_CTA];

    constexpr uint32_t bidm = 0;

    const uint32_t bidn = blockIdx.x;
    const uint32_t tidx = threadIdx.x;
    const uint32_t warp = tidx / THREADS_PER_WARP;
    const uint32_t lane = tidx % THREADS_PER_WARP;

    Reducer reducer(params, bidm, bidn, 0, 0, lane, smem_);

    const uint32_t c = bidn * THREADS_PER_WARP + lane;
    const uint32_t c_out = bidn * THREADS_PER_WARP / 2 + lane;
    constexpr uint32_t COL_STRIDE = Kernel_traits::CTAS * THREADS_PER_WARP;
    for( uint32_t col = c, col_out = c_out; col < Kernel_traits::COLS; col += COL_STRIDE, col_out += COL_STRIDE / 2 ) {
        // Each thread sums over NUM_ELT columns.
        Vec<compute_t, NUM_ELT> dbeta0_local, dgamma0_local, dbeta1_local, dgamma1_local;
        memset(&dgamma0_local, 0, sizeof(dgamma0_local));
        memset(&dbeta0_local, 0, sizeof(dbeta0_local));
        memset(&dgamma1_local, 0, sizeof(dgamma1_local));
        memset(&dbeta1_local, 0, sizeof(dbeta1_local));
        if (Is_even_cols || col < params.cols) {
            for( uint32_t row = warp; row < params.ctas_per_col; row += Kernel_traits::ROWS_PER_CTA ) {
                index_t idx = row * params.cols + col;

                Vec<compute_t, NUM_ELT> dbeta0_part, dgamma0_part, dbeta1_part, dgamma1_part;
                dbeta0_part.load_from(params.dbeta_part, idx);
                dgamma0_part.load_from(params.dgamma_part, idx);
                dbeta1_part.load_from(params.dbeta1_part, idx);
                dgamma1_part.load_from(params.dgamma1_part, idx);
                #pragma unroll
                for( int it = 0; it < NUM_ELT; it++ ) {
                    dgamma0_local.data.elt[it] += dgamma0_part.data.elt[it];
                    dbeta0_local.data.elt[it] += dbeta0_part.data.elt[it];
                    dgamma1_local.data.elt[it] += dgamma1_part.data.elt[it];
                    dbeta1_local.data.elt[it] += dbeta1_part.data.elt[it];
                }
            }
        }
        void * smem_gamma0 = smem_;
        void * smem_beta0 = &smem_[Kernel_traits::SMEM_BYTES_TRANSPOSE];
        void * smem_gamma1 = &smem_[2 * Kernel_traits::SMEM_BYTES_TRANSPOSE];
        void * smem_beta1 = &smem_[3 * Kernel_traits::SMEM_BYTES_TRANSPOSE];

        const int write_row = warp;
        const int write_col = lane ^ write_row;
        const int write_idx = write_row * THREADS_PER_WARP + write_col;

        dgamma0_local.store_to(smem_gamma0, write_idx);
        dbeta0_local.store_to(smem_beta0, write_idx);
        dgamma1_local.store_to(smem_gamma1, write_idx);
        dbeta1_local.store_to(smem_beta1, write_idx);

        __syncthreads();

        // It would be probably safe to reuse the first row of smem_beta0 and smem_gamma0
        void * smem_gamma0_out = &smem_[4 * Kernel_traits::SMEM_BYTES_TRANSPOSE];
        void * smem_beta0_out = &smem_[4 * Kernel_traits::SMEM_BYTES_TRANSPOSE + Kernel_traits::SMEM_BYTES_OUTPUT];
        void * smem_gamma1_out = &smem_[4 * Kernel_traits::SMEM_BYTES_TRANSPOSE + 2 * Kernel_traits::SMEM_BYTES_OUTPUT];
        void * smem_beta1_out = &smem_[4 * Kernel_traits::SMEM_BYTES_TRANSPOSE + 3 * Kernel_traits::SMEM_BYTES_OUTPUT];

        // More than one iter iff ROWS_PER_CTA < 32.
        for( int w = warp; w < THREADS_PER_WARP; w += Kernel_traits::ROWS_PER_CTA ) {
            const int read_row = lane;
            const int read_col = w ^ read_row;
            const int read_idx = read_row * THREADS_PER_WARP + read_col;

            memset(&dbeta0_local, 0, sizeof(dbeta0_local));
            memset(&dgamma0_local, 0, sizeof(dgamma0_local));
            memset(&dbeta1_local, 0, sizeof(dbeta1_local));
            memset(&dgamma1_local, 0, sizeof(dgamma1_local));

            // Load beta and gamma transposed
            if(read_row < Kernel_traits::ROWS_PER_CTA){
                dbeta0_local.load_from(smem_beta0, read_idx);
                dgamma0_local.load_from(smem_gamma0, read_idx);
                dbeta1_local.load_from(smem_beta1, read_idx);
                dgamma1_local.load_from(smem_gamma1, read_idx);
            }

            // Call reducer on the loaded value(s) and convert.
            #pragma unroll
            for( int it = 0; it < NUM_ELT; it++ ) {
                compute_t b0_i = dbeta0_local.data.elt[it];
                compute_t g0_i = dgamma0_local.data.elt[it];
                compute_t b1_i = dbeta1_local.data.elt[it];
                compute_t g1_i = dgamma1_local.data.elt[it];
                b0_i = reducer.allreduce(b0_i, sum);
                g0_i = reducer.allreduce(g0_i, sum);
                b1_i = reducer.allreduce(b1_i, sum);
                g1_i = reducer.allreduce(g1_i, sum);

                dgamma0_local.data.elt[it] = g0_i;
                dbeta0_local.data.elt[it] = b0_i;
                dgamma1_local.data.elt[it] = g1_i;
                dbeta1_local.data.elt[it] = b1_i;
            }

            // Leader stores the result at the current column.
            if(lane == 0){
                dgamma0_local.store_to(smem_gamma0_out, w);
                dbeta0_local.store_to(smem_beta0_out, w);
                dgamma1_local.store_to(smem_gamma1_out, w);
                dbeta1_local.store_to(smem_beta1_out, w);
            }

        }

        // All writes done.
        __syncthreads();

        // Pack and store: 2-wide stores with half the threads.
        if (Is_even_cols || col_out * 2 < params.cols) {
            if( warp == Kernel_traits::ROWS_PER_CTA - 1 && lane < THREADS_PER_WARP / 2 ) {

                using src_t = typename TypeToVec2<compute_t>::Type;
                using dst_t = typename TypeToVec2<weight_t>::Type;
                Vec<src_t, NUM_ELT> dbeta0_vec2, dgamma0_vec2, dbeta1_vec2, dgamma1_vec2;
                Vec<dst_t, NUM_ELT> dbeta0_out2, dgamma0_out2, dbeta1_out2, dgamma1_out2;

                dgamma0_vec2.load_from(smem_gamma0_out, lane);
                dbeta0_vec2.load_from(smem_beta0_out, lane);
                dgamma1_vec2.load_from(smem_gamma1_out, lane);
                dbeta1_vec2.load_from(smem_beta1_out, lane);
                #pragma unroll
                for( int it = 0; it < NUM_ELT; it++ ) {
                    dgamma0_out2.data.elt[it] = Converter<src_t,dst_t>::convert(dgamma0_vec2.data.elt[it]);
                    dbeta0_out2.data.elt[it] = Converter<src_t,dst_t>::convert(dbeta0_vec2.data.elt[it]);
                    dgamma1_out2.data.elt[it] = Converter<src_t,dst_t>::convert(dgamma1_vec2.data.elt[it]);
                    dbeta1_out2.data.elt[it] = Converter<src_t,dst_t>::convert(dbeta1_vec2.data.elt[it]);
                }
                dgamma0_out2.store_to(params.dgamma, col_out);
                dbeta0_out2.store_to(params.dbeta, col_out);
                dgamma1_out2.store_to(params.dgamma1, col_out);
                dbeta1_out2.store_to(params.dbeta1, col_out);
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
    int BYTES_PER_LDG_MAIN,
    int BYTES_PER_LDG_FINAL
>
void launch_parallel_residual_(LaunchParams<BwdParams> &launch_params, const bool configure_params){

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
                                        BYTES_PER_LDG_MAIN
                                        >;
    bool is_dropout = launch_params.params.dropout_keep_p < 1.f;
    bool tied_norm = launch_params.params.gamma1 == nullptr;
    bool is_even_cols = launch_params.params.cols == HIDDEN_SIZE;
    BOOL_SWITCH(is_dropout, IsDropoutConst, [&] {
        BOOL_SWITCH(tied_norm, TiedNormConst, [&] {
            BOOL_SWITCH(is_even_cols, IsEvenColsConst, [&] {
                auto kernel = &ln_parallel_residual_bwd_kernel<Kernel_traits, IsDropoutConst, TiedNormConst, IsEvenColsConst>;
                if( configure_params ) {
                    int ctas_per_sm;
                    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                        &ctas_per_sm, kernel, Kernel_traits::THREADS_PER_CTA, Kernel_traits::SMEM_BYTES));
                    launch_params.params.ctas_per_col = launch_params.props->multiProcessorCount * ctas_per_sm / Kernel_traits::CTAS_PER_ROW;
                    launch_params.barrier_size = 0;
                    launch_params.workspace_bytes = 0;
                    if(Kernel_traits::CTAS_PER_ROW > 1) {
                        launch_params.barrier_size = 2 * launch_params.params.ctas_per_col;
                        launch_params.workspace_bytes = launch_params.params.ctas_per_col
                                                      * Kernel_traits::WARPS_M
                                                      * Kernel_traits::CTAS_PER_ROW
                                                      * sizeof(typename Kernel_traits::reduce_t)
                                                      * 2;
                    }
                    return;
                }

                if( Kernel_traits::SMEM_BYTES >= 48 * 1024 ) {
                    CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Kernel_traits::SMEM_BYTES));
                }
                auto stream = launch_params.stream;
                auto ctas_per_col = launch_params.params.ctas_per_col;

                if( Kernel_traits::CTAS_PER_ROW == 1 ) {
                    kernel<<<ctas_per_col, Kernel_traits::THREADS_PER_CTA, Kernel_traits::SMEM_BYTES, stream>>>(launch_params.params);
                } else {
                    dim3 grid(Kernel_traits::CTAS_PER_ROW * ctas_per_col);
                    dim3 block(Kernel_traits::THREADS_PER_CTA);
                    void *params_ = (void *)&launch_params.params;
                    cudaLaunchCooperativeKernel((void *)kernel, grid, block, (void **)&params_, Kernel_traits::SMEM_BYTES, stream);
                }

                using Kernel_traits_f = layer_norm::Kernel_traits_finalize<HIDDEN_SIZE,
                                                                          weight_t,
                                                                          input_t,
                                                                          residual_t,
                                                                          output_t,
                                                                          compute_t,
                                                                          index_t,
                                                                          /*HasColscaleConst=*/false,
                                                                          32 * 32,  // THREADS_PER_CTA
                                                                          BYTES_PER_LDG_FINAL>;

                auto kernel_f = !TiedNormConst
                    ? &layer_norm::ln_parallel_residual_bwd_finalize_kernel<Kernel_traits_f, IsEvenColsConst>
                    : &layer_norm::ln_bwd_finalize_kernel<Kernel_traits_f, /*HasColscaleConst=*/false, IsEvenColsConst>;
                kernel_f<<<Kernel_traits_f::CTAS, Kernel_traits_f::THREADS_PER_CTA, 0, stream>>>(launch_params.params);

            });
        });
    });
}
