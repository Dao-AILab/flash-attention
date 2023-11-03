#pragma once

#include "ln.h"
#include "ln_utils.cuh"
#include "ln_kernel_traits.h"
#include "static_switch.h"

namespace layer_norm {

template<typename Ktraits, bool Is_dropout, bool Has_colscale, bool Has_subset, bool Is_even_cols>
__global__ __launch_bounds__(Ktraits::THREADS_PER_CTA) 
void ln_bwd_kernel(layer_norm::BwdParams params) {

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

    const input_t *rowscale = static_cast<input_t *>(params.rowscale);
    const index_t *x0_subset = static_cast<index_t *>(params.x0_subset);
    const index_t *z_subset = static_cast<index_t *>(params.z_subset);

    Cvec dzy_sum[LDGS];
    Cvec dz_sum[LDGS];
    Cvec dcolscale_sum[LDGS];

    memset(dzy_sum, 0, sizeof(dzy_sum));
    memset(dz_sum, 0, sizeof(dz_sum));
    if (Has_colscale) { memset(dcolscale_sum, 0, sizeof(dcolscale_sum)); }

    compute_t * smem_wgrad = reinterpret_cast<compute_t*>(smem_);
    char *smem_dgrad = smem_ + Ktraits::SMEM_BYTES_WGRAD;

    Reducer reducer(params, bidm, bidn, warp_m, warp_n, lane, smem_dgrad);

    Sum<reduce_t> sum;

    const index_t num_valid_ldgs =
        ((params.cols / Ktraits::ELTS_PER_LDG) - 1 - c + Ktraits::VEC_COLS_PER_LDG) / Ktraits::VEC_COLS_PER_LDG;

    Wvec gamma[LDGS];
    Wvec colscale[LDGS];
    index_t idx = c;
    #pragma unroll
    for( int it = 0; it < LDGS; it++ ) {
        if (Is_even_cols || (it < num_valid_ldgs)) {
            gamma[it].load_from(params.gamma, idx);
            if (Has_colscale) { colscale[it].load_from(params.colscale, idx); }
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
        const compute_t rowscale_val = !Has_subset ? (params.rowscale == nullptr ? 1.0f : compute_t(rowscale[row])) : params.rowscale_const;
        const int row_z = !Has_subset ? row + 1 : z_subset[row];
        const int row_x0 = !Has_subset ? row + 1 : x0_subset[row];
        const bool load_dz = !Has_subset || row_z > 0;
        const bool save_dx0 = !Has_subset || row_x0 > 0;
        Mvec dmask[LDGS];
        Rvec dx[LDGS];
        compute_t dy[LDGS * NUM_ELTS];
        compute_t y[LDGS * NUM_ELTS];
        compute_t mdy_local = 0.f;
        compute_t mdyy_local = 0.f;
        // If dz is not loaded, then dy should be 0 and we don't care about the value of y.
        if (load_dz) {
            index_t idx_x = row * params.cols / Ktraits::ELTS_PER_LDG + c;
            index_t idx_z = !Has_subset ? idx_x : (load_dz ? (row_z - 1) * params.cols / Ktraits::ELTS_PER_LDG + c : 0);
            index_t idx_x0 = !Has_subset ? idx_x : (save_dx0 ? (row_x0 - 1) * params.cols / Ktraits::ELTS_PER_LDG + c : 0);
            #pragma unroll
            for( int it = 0; it < LDGS; it++ ) {
                if (Is_even_cols || (it < num_valid_ldgs)) {
                    Rvec x;
                    Ovec dz;
                    dz.load_from(params.dz, !Has_subset ? idx_x : idx_z);
                    if (prenorm) { dx[it].load_from(params.dx, idx_x); }
                    x.load_from(params.x, idx_x);
                    if (Is_dropout) { dmask[it].load_from(params.dmask, !Has_subset ? idx_x : idx_x0); }
                    idx_x += Ktraits::VEC_COLS_PER_LDG;
                    idx_z += Ktraits::VEC_COLS_PER_LDG;
                    idx_x0 += Ktraits::VEC_COLS_PER_LDG;
                    #pragma unroll
                    for( int jt = 0; jt < NUM_ELTS; jt++ ) {
                        compute_t x_tmp = x.data.elt[jt];
                        compute_t y_tmp = rs_r * (x_tmp - (!params.is_rms_norm ? mu_r : 0.f));
                        compute_t dy_tmp = compute_t(gamma[it].data.elt[jt]) * compute_t(dz.data.elt[jt]);
                        compute_t dz_tmp = dz.data.elt[jt];

                        mdy_local += dy_tmp;
                        mdyy_local += dy_tmp * y_tmp;

                        dy[it * NUM_ELTS + jt] = dy_tmp;
                        y[it * NUM_ELTS + jt] = y_tmp;

                        dzy_sum[it].data.elt[jt] += dz_tmp * y_tmp;
                        dz_sum[it].data.elt[jt] += dz_tmp;
                    }
                }
            }
        } else {
            index_t idx_x = row * params.cols / Ktraits::ELTS_PER_LDG + c;
            index_t idx_x0 = !Has_subset ? idx_x : (save_dx0 ? (row_x0 - 1) * params.cols / Ktraits::ELTS_PER_LDG + c : 0);
            #pragma unroll
            for( int it = 0; it < LDGS; it++ ) {
                if (Is_even_cols || (it < num_valid_ldgs)) {
                    if (prenorm) { dx[it].load_from(params.dx, idx_x); }
                    if (Is_dropout) { dmask[it].load_from(params.dmask, !Has_subset ? idx_x : idx_x0); }
                    idx_x += Ktraits::VEC_COLS_PER_LDG;
                    idx_x0 += Ktraits::VEC_COLS_PER_LDG;
                }
            }
        }

        reduce_t result = reducer.allreduce({mdy_local, mdyy_local}, sum);
        mdy_local = layer_norm::Get<0>::of<reduce_t, compute_t>(result) * params.inverse_cols;
        mdyy_local = layer_norm::Get<1>::of<reduce_t, compute_t>(result) * params.inverse_cols;

        index_t idx_x = row * params.cols / Ktraits::ELTS_PER_LDG + c;
        index_t idx_x0 = !Has_subset ? idx_x : (save_dx0 ? (row_x0 - 1) * params.cols / Ktraits::ELTS_PER_LDG + c : 0);
        #pragma unroll
        for( int it = 0; it < LDGS; it++ ) {
            if (Is_even_cols || (it < num_valid_ldgs)) {
                Ivec dx0;
                Rvec dresidual;
                Ivec x0;
                if (Has_colscale && save_dx0) { x0.load_from(params.x0, !Has_subset ? idx_x : idx_x0); }
                #pragma unroll
                for( int jt = 0; jt < NUM_ELTS; jt++ ) {
                    compute_t dx_tmp_res;
                    if (load_dz) {
                        compute_t dy_tmp = dy[it * NUM_ELTS + jt];
                        compute_t y_tmp = y[it * NUM_ELTS + jt];
                        compute_t dx_tmp = rs_r * (dy_tmp - (mdyy_local * y_tmp + (!params.is_rms_norm ? mdy_local : 0.f)));
                        dx_tmp_res = prenorm ? dx_tmp + compute_t(dx[it].data.elt[jt]) : dx_tmp;
                    } else {
                        dx_tmp_res = prenorm ? compute_t(dx[it].data.elt[jt]) : 0.f;
                    }
                    if (has_residual) { dresidual.data.elt[jt] = dx_tmp_res; }
                    if (save_dx0) {
                        compute_t dx0_tmp_res = dx_tmp_res * rowscale_val;
                        if (Is_dropout) {
                            dx0_tmp_res *= params.dropout_scale;
                            if (Has_colscale) {
                                dcolscale_sum[it].data.elt[jt] += dmask[it].data.elt[jt] ? dx0_tmp_res * compute_t(x0.data.elt[jt]) : 0.f;
                                dx0.data.elt[jt] = dmask[it].data.elt[jt] ? dx0_tmp_res * compute_t(colscale[it].data.elt[jt]) : 0.f;
                            } else {
                                dx0.data.elt[jt] = dmask[it].data.elt[jt] ? dx0_tmp_res : 0.f;
                            }
                        } else {
                            if (Has_colscale) {
                                dcolscale_sum[it].data.elt[jt] += dx0_tmp_res * compute_t(x0.data.elt[jt]);
                                dx0.data.elt[jt] = dx0_tmp_res * compute_t(colscale[it].data.elt[jt]);
                            } else {
                                dx0.data.elt[jt] = dx0_tmp_res;
                            }
                        }
                    }
                }
                if (has_residual) { dresidual.store_to(params.dresidual, idx_x); }
                if (save_dx0) { dx0.store_to(params.dx0, !Has_subset ? idx_x : idx_x0); }
                idx_x += Ktraits::VEC_COLS_PER_LDG;
                idx_x0 += Ktraits::VEC_COLS_PER_LDG;
            }
        }

    }  // end: grid stride loop

    if( WARPS_M == 1 ) {
        idx = r * params.cols / Ktraits::ELTS_PER_LDG + c;
        #pragma unroll
        for( int it = 0; it < LDGS; it++ ) {
            if (Is_even_cols || (it < num_valid_ldgs)) {
                dz_sum[it].store_to(params.dbeta_part, idx);
                dzy_sum[it].store_to(params.dgamma_part, idx);
                if (Has_colscale) { dcolscale_sum[it].store_to(params.dcolscale_part, idx); }
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
            dz_sum[it].store_to(smem_wgrad, idx);
            idx += THREADS_PER_ROW;
        }
        __syncthreads();
        compute_t cta_dz_sum[NUM_RES];
        memset(cta_dz_sum, 0, sizeof(compute_t) * NUM_RES);
        for( int it = 0; it < ROWS_PER_CTA; it++ ) {
            for( int jt = 0; jt < NUM_RES; jt++ ) {
                cta_dz_sum[jt] += smem_wgrad[it * COLS + tidx + jt * Ktraits::THREADS_PER_CTA];
            }
        }
        __syncthreads();

        idx = warp_m * Ktraits::VEC_COLS + tid_r;
        #pragma unroll
        for( int it = 0; it < LDGS; it++ ) {
            dzy_sum[it].store_to(smem_wgrad, idx);
            idx += THREADS_PER_ROW;
        }
        __syncthreads();
        compute_t cta_dzy_sum[NUM_RES];
        memset(cta_dzy_sum, 0, sizeof(compute_t) * NUM_RES);
        for( int it = 0; it < ROWS_PER_CTA; it++ ) {
            for( int jt = 0; jt < NUM_RES; jt++ ) {
                cta_dzy_sum[jt] += smem_wgrad[it * COLS + tidx + jt * Ktraits::THREADS_PER_CTA];
            }
        }

        compute_t cta_dcolscale_sum[NUM_RES];
        if (Has_colscale) {
            __syncthreads();
            idx = warp_m * Ktraits::VEC_COLS + tid_r;
            #pragma unroll
            for( int it = 0; it < LDGS; it++ ) {
                dcolscale_sum[it].store_to(smem_wgrad, idx);
                idx += THREADS_PER_ROW;
            }
            __syncthreads();
            memset(cta_dcolscale_sum, 0, sizeof(compute_t) * NUM_RES);
            for( int it = 0; it < ROWS_PER_CTA; it++ ) {
                for( int jt = 0; jt < NUM_RES; jt++ ) {
                    cta_dcolscale_sum[jt] += smem_wgrad[it * COLS + tidx + jt * Ktraits::THREADS_PER_CTA];
                }
            }
        }

        const index_t num_valid_writes
            = (params.cols - 1 - tidx + Ktraits::THREADS_PER_CTA) / Ktraits::THREADS_PER_CTA;
        compute_t *dgamma_part = static_cast<compute_t *>(params.dgamma_part) + bidm * params.cols + tidx;
        compute_t *dbeta_part = static_cast<compute_t *>(params.dbeta_part) + bidm * params.cols + tidx;
        compute_t *dcolscale_part = Has_colscale ? static_cast<compute_t *>(params.dcolscale_part) + bidm * params.cols + tidx : nullptr;
        for( int jt = 0; jt < NUM_RES; jt++ ) {
            if (Is_even_cols || (jt < num_valid_writes)) {
                *dgamma_part = cta_dzy_sum[jt];
                dgamma_part += Ktraits::THREADS_PER_CTA;
                *dbeta_part = cta_dz_sum[jt];
                dbeta_part += Ktraits::THREADS_PER_CTA;
                if (Has_colscale) {
                    *dcolscale_part = cta_dcolscale_sum[jt];
                    dcolscale_part += Ktraits::THREADS_PER_CTA;
                }
            }
        }

    }
}

template<typename Kernel_traits, bool Has_colscale, bool Is_even_cols>
__global__ __launch_bounds__(Kernel_traits::THREADS_PER_CTA)
void ln_bwd_finalize_kernel(BwdParams params)
{

    using compute_t = typename Kernel_traits::compute_t;
    using weight_t = typename Kernel_traits::weight_t;
    using index_t = typename Kernel_traits::index_t;
    using Reducer = typename Kernel_traits::Reducer;
    using reduce_t = typename Reducer::Type;

    Sum<reduce_t> sum;
    enum { NUM_ELT = Kernel_traits::ELTS_PER_LDG };
    enum { THREADS_PER_WARP = Kernel_traits::THREADS_PER_WARP };

    __shared__ char smem_[Kernel_traits::SMEM_BYTES_PER_CTA];

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
        Vec<compute_t, NUM_ELT> dbeta_local, dgamma_local, dcolscale_local;
        memset(&dgamma_local, 0, sizeof(dgamma_local));
        memset(&dbeta_local, 0, sizeof(dbeta_local));
        if (Has_colscale) { memset(&dcolscale_local, 0, sizeof(dcolscale_local)); }
        if (Is_even_cols || col < params.cols) {
            for( uint32_t row = warp; row < params.ctas_per_col; row += Kernel_traits::ROWS_PER_CTA ) {
                index_t idx = row * params.cols + col;

                Vec<compute_t, NUM_ELT> dbeta_part, dgamma_part, dcolscale_part;
                dbeta_part.load_from(params.dbeta_part, idx);
                dgamma_part.load_from(params.dgamma_part, idx);
                if (Has_colscale) { dcolscale_part.load_from(params.dcolscale_part, idx); }
                #pragma unroll
                for( int it = 0; it < NUM_ELT; it++ ) {
                    dgamma_local.data.elt[it] += dgamma_part.data.elt[it];
                    dbeta_local.data.elt[it] += dbeta_part.data.elt[it];
                    if (Has_colscale) { dcolscale_local.data.elt[it] += dcolscale_part.data.elt[it]; }
                }
            }
        }
        void * smem_gamma = smem_;
        void * smem_beta = &smem_[Kernel_traits::SMEM_BYTES_TRANSPOSE];
        void * smem_colscale = &smem_[2 * Kernel_traits::SMEM_BYTES_TRANSPOSE];

        const int write_row = warp;
        const int write_col = lane ^ write_row;
        const int write_idx = write_row * THREADS_PER_WARP + write_col;

        dgamma_local.store_to(smem_gamma, write_idx);
        dbeta_local.store_to(smem_beta, write_idx);
        if (Has_colscale) { dcolscale_local.store_to(smem_colscale, write_idx); }

        __syncthreads();

        // It would be probably safe to reuse the first row of smem_beta and smem_gamma
        void * smem_gamma_out = &smem_[Kernel_traits::NUM_FACTORS * Kernel_traits::SMEM_BYTES_TRANSPOSE];
        void * smem_beta_out = &smem_[Kernel_traits::NUM_FACTORS * Kernel_traits::SMEM_BYTES_TRANSPOSE + Kernel_traits::SMEM_BYTES_OUTPUT];
        void * smem_colscale_out = &smem_[Kernel_traits::NUM_FACTORS * Kernel_traits::SMEM_BYTES_TRANSPOSE + 2 * Kernel_traits::SMEM_BYTES_OUTPUT];


        // More than one iter iff ROWS_PER_CTA < 32.
        for( int w = warp; w < THREADS_PER_WARP; w += Kernel_traits::ROWS_PER_CTA ) {
            const int read_row = lane;
            const int read_col = w ^ read_row;
            const int read_idx = read_row * THREADS_PER_WARP + read_col;

            memset(&dbeta_local, 0, sizeof(dbeta_local));
            memset(&dgamma_local, 0, sizeof(dgamma_local));
            if (Has_colscale) { memset(&dcolscale_local, 0, sizeof(dcolscale_local)); }

            // Load beta and gamma transposed 
            if(read_row < Kernel_traits::ROWS_PER_CTA){
                dbeta_local.load_from(smem_beta, read_idx);
                dgamma_local.load_from(smem_gamma, read_idx);
                if (Has_colscale) { dcolscale_local.load_from(smem_colscale, read_idx); }
            }

            // Call reducer on the loaded value(s) and convert.
            #pragma unroll
            for( int it = 0; it < NUM_ELT; it++ ) {
                compute_t b_i = dbeta_local.data.elt[it];
                compute_t g_i = dgamma_local.data.elt[it];
                b_i = reducer.allreduce(b_i, sum);
                g_i = reducer.allreduce(g_i, sum);

                dgamma_local.data.elt[it] = g_i;
                dbeta_local.data.elt[it] = b_i;
                if (Has_colscale) {
                    compute_t cs_i = dcolscale_local.data.elt[it];
                    cs_i = reducer.allreduce(cs_i, sum);
                    dcolscale_local.data.elt[it] = cs_i;
                }
            }

            // Leader stores the result at the current column.
            if(lane == 0){
                dgamma_local.store_to(smem_gamma_out, w);
                dbeta_local.store_to(smem_beta_out, w);
                if (Has_colscale) { dcolscale_local.store_to(smem_colscale_out, w); }
            }

        }

        // All writes done.
        __syncthreads();

        // Pack and store: 2-wide stores with half the threads.
        if (Is_even_cols || col_out * 2 < params.cols) {
            if( warp == Kernel_traits::ROWS_PER_CTA - 1 && lane < THREADS_PER_WARP / 2 ) {

                using src_t = typename TypeToVec2<compute_t>::Type;
                using dst_t = typename TypeToVec2<weight_t>::Type;
                Vec<src_t, NUM_ELT> dbeta_vec2, dgamma_vec2, dcolscale_vec2;
                Vec<dst_t, NUM_ELT> dbeta_out2, dgamma_out2, dcolscale_out2;

                dgamma_vec2.load_from(smem_gamma_out, lane);
                dbeta_vec2.load_from(smem_beta_out, lane);
                if (Has_colscale) { dcolscale_vec2.load_from(smem_colscale_out, lane); }
                #pragma unroll
                for( int it = 0; it < NUM_ELT; it++ ) {
                    dgamma_out2.data.elt[it] = Converter<src_t,dst_t>::convert(dgamma_vec2.data.elt[it]);
                    dbeta_out2.data.elt[it] = Converter<src_t,dst_t>::convert(dbeta_vec2.data.elt[it]);
                    if (Has_colscale) { dcolscale_out2.data.elt[it] = Converter<src_t,dst_t>::convert(dcolscale_vec2.data.elt[it]); }
                }
                dgamma_out2.store_to(params.dgamma, col_out);
                dbeta_out2.store_to(params.dbeta, col_out);
                if (Has_colscale) { dcolscale_out2.store_to(params.dcolscale, col_out); }
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
void launch_(LaunchParams<BwdParams> &launch_params, const bool configure_params){

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
    bool has_colscale = launch_params.params.colscale != nullptr;
    bool has_subset = launch_params.params.x0_subset != nullptr;
    bool is_even_cols = launch_params.params.cols == HIDDEN_SIZE;
    BOOL_SWITCH(is_dropout, IsDropoutConst, [&] {
        BOOL_SWITCH(has_colscale, HasColscaleConst, [&] {
            BOOL_SWITCH(has_subset, HasSubsetConst, [&] {
                BOOL_SWITCH(is_even_cols, IsEvenColsConst, [&] {
                    auto kernel = &ln_bwd_kernel<Kernel_traits, IsDropoutConst, HasColscaleConst, HasSubsetConst, IsEvenColsConst>;
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
                                                                              HasColscaleConst,
                                                                              32 * 32,  // THREADS_PER_CTA
                                                                              BYTES_PER_LDG_FINAL>;

                    auto kernel_f = &layer_norm::ln_bwd_finalize_kernel<Kernel_traits_f, HasColscaleConst, IsEvenColsConst>;
                    kernel_f<<<Kernel_traits_f::CTAS, Kernel_traits_f::THREADS_PER_CTA, 0, stream>>>(launch_params.params);
                });
            });
        });
    });
}
