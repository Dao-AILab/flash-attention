#pragma once

namespace layer_norm {

template<typename Ktraits, bool Prenorm, bool Is_dropout, bool Has_residual, bool Has_rowscale>
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

    Cvec dzy_sum[LDGS];
    Cvec dz_sum[LDGS];

    memset(dzy_sum, 0, sizeof(dzy_sum));
    memset(dz_sum, 0, sizeof(dz_sum));

    compute_t * smem_wgrad = reinterpret_cast<compute_t*>(smem_);
    char *smem_dgrad = smem_ + Ktraits::SMEM_BYTES_WGRAD;

    Reducer reducer(params, bidm, bidn, warp_m, warp_n, lane, smem_dgrad);

    Sum<reduce_t> sum;

    constexpr float rn = 1.f / float(COLS);
    Wvec gamma[LDGS];
    index_t idx = c;
    #pragma unroll
    for( int it = 0; it < LDGS; it++ ) {
        gamma[it].load_from(params.gamma, idx);
        idx += Ktraits::VEC_COLS_PER_LDG;
    }
    // TODO if ROWS_PER_CTA does not divide rows, we might get divergence in the
    // last blocks with syncthreads!
    // grid stride over rows
    #pragma unroll 1
    for( int row = r; row < params.rows; row += params.ctas_per_col * ROWS_PER_CTA ) {
        const compute_t mu_r = static_cast<const compute_t *>(params.mu)[row];
        const compute_t rs_r = static_cast<const compute_t *>(params.rs)[row];
        const compute_t rowscale_val = Has_rowscale ? compute_t(static_cast<const input_t *>(params.rowscale)[row]) : 1.0f;
        Mvec dmask[LDGS];
        Rvec dx[LDGS];
        compute_t dy[LDGS * NUM_ELTS];
        compute_t y[LDGS * NUM_ELTS];
        compute_t mdy_local = 0.f;
        compute_t mdyy_local = 0.f;
        index_t idx = row * Ktraits::VEC_COLS + c;
        #pragma unroll
        for( int it = 0; it < LDGS; it++ ) {
            Rvec x;
            Ovec dz;
            dz.load_from(params.dz, idx);
            if (Prenorm) { dx[it].load_from(params.dx, idx); }
            x.load_from(params.x, idx);
            if (Is_dropout) { dmask[it].load_from(params.dmask, idx); }
            idx += Ktraits::VEC_COLS_PER_LDG;
            #pragma unroll
            for( int jt = 0; jt < NUM_ELTS; jt++ ) {
                compute_t x_tmp = x.data.elt[jt];
                compute_t y_tmp = rs_r * (x_tmp - mu_r);
                compute_t dy_tmp = compute_t(gamma[it].data.elt[jt]);
                dy_tmp *= compute_t(dz.data.elt[jt]);
                compute_t dz_tmp = dz.data.elt[jt];

                mdy_local += dy_tmp;
                mdyy_local += dy_tmp * y_tmp;

                dy[it * NUM_ELTS + jt] = dy_tmp;
                y[it * NUM_ELTS + jt] = y_tmp;

                dzy_sum[it].data.elt[jt] += dz_tmp * y_tmp;
                dz_sum[it].data.elt[jt] += dz_tmp;
            }
        }

        reduce_t result = reducer.allreduce({mdy_local, mdyy_local}, sum);
        mdy_local = layer_norm::Get<0>::of<reduce_t, compute_t>(result) * rn;
        mdyy_local = layer_norm::Get<1>::of<reduce_t, compute_t>(result) * rn;

        idx = row * Ktraits::VEC_COLS + c;
        #pragma unroll
        for( int it = 0; it < LDGS; it++ ) {
            Ivec dx0;
            Rvec dx1;
            #pragma unroll
            for( int jt = 0; jt < NUM_ELTS; jt++ ) {
                compute_t dy_tmp = dy[it * NUM_ELTS + jt];
                compute_t y_tmp = y[it * NUM_ELTS + jt];
                compute_t dx_tmp = rs_r * (dy_tmp - (mdyy_local * y_tmp + mdy_local));
                compute_t dx_tmp_res = Prenorm ? dx_tmp + compute_t(dx[it].data.elt[jt]) : dx_tmp;
                if (Has_residual) { dx1.data.elt[jt] = dx_tmp_res; }
                compute_t dx0_tmp_res = Has_rowscale ? dx_tmp_res * rowscale_val : dx_tmp_res;
                if (Is_dropout) {
                    dx0.data.elt[jt] = dmask[it].data.elt[jt] ? dx0_tmp_res * params.dropout_scale : 0.f;
                } else {
                    dx0.data.elt[jt] = dx0_tmp_res;
                }
            }
            if (Has_residual) { dx1.store_to(params.dx1, idx); }
            dx0.store_to(params.dx0, idx);
            idx += Ktraits::VEC_COLS_PER_LDG;
        }

    }  // end: grid stride loop

    if( WARPS_M == 1 ) {
        idx = r * Ktraits::VEC_COLS + c;
        #pragma unroll
        for( int it = 0; it < LDGS; it++ ) {
            dz_sum[it].store_to(params.dbeta_part, idx);
            dzy_sum[it].store_to(params.dgamma_part, idx);
            idx += Ktraits::VEC_COLS_PER_LDG;
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

        compute_t *dgamma_part = static_cast<compute_t *>(params.dgamma_part) + bidm * COLS + tidx;
        for( int jt = 0; jt < NUM_RES; jt++ ) {
            *dgamma_part = cta_dzy_sum[jt];
            dgamma_part += Ktraits::THREADS_PER_CTA;
        }

        compute_t *dbeta_part = static_cast<compute_t *>(params.dbeta_part) + bidm * COLS + tidx;
        for( int jt = 0; jt < NUM_RES; jt++ ) {
            *dbeta_part = cta_dz_sum[jt];
            dbeta_part += Ktraits::THREADS_PER_CTA;
        }
    }
}

template<typename Kernel_traits>
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
        Vec<compute_t, NUM_ELT> dbeta_local, dgamma_local;
        memset(&dgamma_local, 0, sizeof(dgamma_local));
        memset(&dbeta_local, 0, sizeof(dbeta_local));
        for( uint32_t row = warp; row < params.ctas_per_col; row += Kernel_traits::ROWS_PER_CTA ) {
            index_t idx = row * Kernel_traits::COLS + col;

            Vec<compute_t, NUM_ELT> dbeta_part, dgamma_part;
            dbeta_part.load_from(params.dbeta_part, idx);
            dgamma_part.load_from(params.dgamma_part, idx);
            #pragma unroll
            for( int it = 0; it < NUM_ELT; it++ ) {
                dgamma_local.data.elt[it] += dgamma_part.data.elt[it];
                dbeta_local.data.elt[it] += dbeta_part.data.elt[it];
            }
        }

        void * smem_gamma = smem_;
        void * smem_beta = &smem_[Kernel_traits::SMEM_BYTES_TRANSPOSE];

        const int write_row = warp;
        const int write_col = lane ^ write_row;
        const int write_idx = write_row * THREADS_PER_WARP + write_col;

        dgamma_local.store_to(smem_gamma, write_idx);
        dbeta_local.store_to(smem_beta, write_idx);

        __syncthreads();

        // It would be probably safe to reuse the first row of smem_beta and smem_gamma
        void * smem_gamma_out = &smem_[2 * Kernel_traits::SMEM_BYTES_TRANSPOSE];
        void * smem_beta_out = &smem_[2 * Kernel_traits::SMEM_BYTES_TRANSPOSE + Kernel_traits::SMEM_BYTES_OUTPUT];


        // More than one iter iff ROWS_PER_CTA < 32.
        for( int w = warp; w < THREADS_PER_WARP; w += Kernel_traits::ROWS_PER_CTA ) {
            const int read_row = lane;
            const int read_col = w ^ read_row;
            const int read_idx = read_row * THREADS_PER_WARP + read_col;

            memset(&dbeta_local, 0, sizeof(dbeta_local));
            memset(&dgamma_local, 0, sizeof(dgamma_local));

            // Load beta and gamma transposed 
            if(read_row < Kernel_traits::ROWS_PER_CTA){
                dbeta_local.load_from(smem_beta, read_idx);
                dgamma_local.load_from(smem_gamma, read_idx);
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
            }

            // Leader stores the result at the current column.
            if(lane == 0){
                dgamma_local.store_to(smem_gamma_out, w);
                dbeta_local.store_to(smem_beta_out, w);
            }

        }

        // All writes done.
        __syncthreads();

        // Pack and store: 2-wide stores with half the threads.
        if( warp == Kernel_traits::ROWS_PER_CTA - 1 && lane < THREADS_PER_WARP / 2 ) {

            using src_t = typename TypeToVec2<compute_t>::Type;
            using dst_t = typename TypeToVec2<weight_t>::Type;
            Vec<src_t, NUM_ELT> dbeta_vec2, dgamma_vec2;
            Vec<dst_t, NUM_ELT> dbeta_out2, dgamma_out2;

            dgamma_vec2.load_from(smem_gamma_out, lane);
            dbeta_vec2.load_from(smem_beta_out, lane);
            #pragma unroll
            for( int it = 0; it < NUM_ELT; it++ ) {
                dgamma_out2.data.elt[it] = Converter<src_t,dst_t>::convert(dgamma_vec2.data.elt[it]);
                dbeta_out2.data.elt[it] = Converter<src_t,dst_t>::convert(dbeta_vec2.data.elt[it]);
            }
            dgamma_out2.store_to(params.dgamma, col_out);
            dbeta_out2.store_to(params.dbeta, col_out);

        }
    }
}
}  // namespace layer_norm
