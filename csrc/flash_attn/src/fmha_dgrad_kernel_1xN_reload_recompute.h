/* Copyright (c) 2022, Tri Dao.
 */

#pragma once

#include "fmha_fprop_kernel_1xN.h"
#include "fmha_kernel.h"
#include <fmha/kernel_traits.h>
#include <fmha/gemm.h>

namespace fmha {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int MMAS_M>
inline __device__ void dot_fragments(float (&sum)[MMAS_M * 2],
                                     const fmha::Fragment_a<fmha::Row> (&x)[MMAS_M],
                                     const fmha::Fragment_a<fmha::Row> (&y)[MMAS_M]) {
    #pragma unroll
    for (int mi = 0; mi < MMAS_M; ++mi) {
        sum[mi * 2 + 0] += hfma2_to_float(x[mi].template elt_as<__half2>(0),
                                          y[mi].template elt_as<__half2>(0));
        sum[mi * 2 + 0] += hfma2_to_float(x[mi].template elt_as<__half2>(2),
                                          y[mi].template elt_as<__half2>(2));
        sum[mi * 2 + 1] += hfma2_to_float(x[mi].template elt_as<__half2>(1),
                                          y[mi].template elt_as<__half2>(1));
        sum[mi * 2 + 1] += hfma2_to_float(x[mi].template elt_as<__half2>(3),
                                          y[mi].template elt_as<__half2>(3));
        // hfma2_to_float(sum[mi * 2 + 0], x[mi].template elt_as<__half2>(0), y[mi].template elt_as<__half2>(0));
        // hfma2_to_float(sum[mi * 2 + 0], x[mi].template elt_as<__half2>(2), y[mi].template elt_as<__half2>(2));
        // hfma2_to_float(sum[mi * 2 + 1], x[mi].template elt_as<__half2>(1), y[mi].template elt_as<__half2>(1));
        // hfma2_to_float(sum[mi * 2 + 1], x[mi].template elt_as<__half2>(3), y[mi].template elt_as<__half2>(3));
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, typename Params>
inline __device__ void compute_dp_dq_1xN(const Params &params) {

    // The description of the CTA tile for the 1st batched GEMM.
    using Cta_tile_p = typename Kernel_traits::Cta_tile_p;
    // The description of the CTA tile for the 2nd batched GEMM.
    using Cta_tile_dq = typename Kernel_traits::Cta_tile_o;

    // The MMA tile for the 1st GEMM.
    using Mma_tile_p = fmha::Hmma_tile<Cta_tile_p>;
    // The MMA tile for the 2nd GEMM.
    using Mma_tile_dq = fmha::Hmma_tile<Cta_tile_dq>;

    // The global memory tile to load Q.
    using Gmem_tile_q = typename Kernel_traits::Gmem_tile_q;

    // The global memory tile to load K.
    using Gmem_tile_k = typename Kernel_traits::Gmem_tile_k;
    // The shared memory tile to swizzle K^T. Treat K^T as V
    using Smem_tile_kt = typename Kernel_traits::Smem_tile_v;

    // Treating V as K. We need to use Kernel_traits::Smem_tile_k otherwise loading will be wrong
    // The global memory tile to load V.
    using Gmem_tile_v = typename Kernel_traits::Gmem_tile_k;
    // The shared memory tile to swizzle V.
    using Smem_tile_v = typename Kernel_traits::Smem_tile_k;

    // The global memory tile to load dO.
    using Gmem_tile_do = typename Kernel_traits::Gmem_tile_do;
    // The shared memory tile to load dO.
    // Treating dO as Q.
    using Smem_tile_do = typename Kernel_traits::Smem_tile_q;

    // The global memory tile to load O.Loading O here is similar to loading dO.
    using Gmem_tile_o = Gmem_tile_do;
    // The shared memory tile to load O.
    using Smem_tile_o = Smem_tile_do;

    // The global memory tile to store dQ.
    // using Gmem_tile_dq = typename Kernel_traits::Gmem_tile_dq;
    using Gmem_tile_dq = fmha::Gmem_tile_dq<Cta_tile_dq>;
    // The shared memory tile to swizzle dQ.
    using Smem_tile_dq = typename Kernel_traits::Smem_tile_o;

    using Gmem_tile_s = typename Kernel_traits::Gmem_tile_s;

    using Gmem_softmax_sum = typename Kernel_traits::Gmem_softmax_sum;

    // using Gemm1 = Gemm_Q_K<Kernel_traits, Kernel_traits::K_IN_REGS>;
    using Gemm1 = Gemm_Q_K<Kernel_traits, /*K-in_regs=*/false>;

    using Softmax = fmha::Softmax<Cta_tile_p, Kernel_traits>;

    // Shared memory.
    extern __shared__ char smem_[];

    // The block index for the batch.
    const int bidb = blockIdx.y;
    // The block index for the head.
    const int bidh = blockIdx.x;
    // The thread index.
    const int tidx = threadIdx.x;

    const BlockInfoPadded<Kernel_traits::THREADS> binfo(params, bidb, bidh, tidx);
    if( binfo.stop_early() ) return;

    Gemm1 gemm_q_k(&smem_[Smem_tile_do::BYTES_PER_TILE + Smem_tile_o::BYTES_PER_TILE], tidx);
    // Allocate the global memory tile loader for Q.
    Gmem_tile_q gmem_q(params, 0, binfo, tidx);
    // Allocate the global memory tile loader for dQ.
    Gmem_tile_dq gmem_dq(params, 0, binfo, tidx);
    // Allocate the global memory tile loader for S.
    Gmem_tile_s gmem_s(params, binfo, tidx);

    fmha::Mask<Cta_tile_p> mask(binfo, tidx);

    // Allocate the global memory tile loader for K.
    Gmem_tile_k gmem_k(params, 1, binfo, tidx);
    // Allocate the global memory tile loader for V.
    Gmem_tile_v gmem_v(params, 2, binfo, tidx);
    // The base pointer of smem_v;
    char *smem_v_ = &smem_[Smem_tile_do::BYTES_PER_TILE + Smem_tile_o::BYTES_PER_TILE + Gemm1::SMEM_OFFSET_V];

    // Allocate the shared memory tile loader for V. We use the same as K so be careful!!!
    Smem_tile_v smem_v(smem_v_, tidx);
    // Allocate the shared memory tile loader for K^T. We use the same as K so be careful!!!
    Smem_tile_kt smem_kt(&smem_[Smem_tile_do::BYTES_PER_TILE + Smem_tile_o::BYTES_PER_TILE + Gemm1::Smem_tile_q::BYTES_PER_TILE], tidx);

    // Allocate the global memory tile loader for dO.
    Gmem_tile_do gmem_do(params.do_ptr, params, binfo, tidx);
    // Allocate the shared memory tile loader for dO.
    Smem_tile_do smem_do(&smem_[0], tidx);

    // Allocate the global memory tile loader for O.
    Gmem_tile_o gmem_o(params.o_ptr, params, binfo, tidx);
    // Allocate the shared memory tile loader for O.
    Smem_tile_o smem_o(&smem_[Smem_tile_do::BYTES_PER_TILE], tidx);

    // Allocate the shared memory tile loader for O. We use the same as K so be careful!!!
    Smem_tile_dq smem_dq(&smem_[Smem_tile_do::BYTES_PER_TILE + Smem_tile_o::BYTES_PER_TILE + Gemm1::SMEM_OFFSET_O], tidx);

    // Trigger the loads for K.
    gmem_k.load();
    // Trigger the loads for Q.
    gmem_q.load();
    // Trigger the loads for V.
    gmem_v.load();
    // Trigger the loads for dO.
    gmem_do.load();
    // Trigger the loads for O.
    gmem_o.load();

    const uint32_t scale_bmm1 = reinterpret_cast<const uint32_t&>(params.scale_bmm1);
    #pragma unroll
    for(int it=0; it < Gmem_tile_k::LDGS; it++){
        gmem_k.fetch_[it] = fmha::hmul8(scale_bmm1, gmem_k.fetch_[it]);
    }

    // Commit the data for Q, dO, and V to shared memory.
    gmem_q.commit(gemm_q_k.smem_q);
    gmem_do.commit(smem_do);
    gmem_o.commit(smem_o);
    gmem_v.commit(smem_v);

    // Commit the data for K to shared memory.
    if( !Kernel_traits::SHARE_SMEM_FOR_K_AND_V ) {
        gmem_k.commit(gemm_q_k.smem_k);
    }

    __syncthreads();

    // Load the fragments for Q.
    gemm_q_k.load_q();

    // Load the fragments for dO.
    typename Smem_tile_do::Fragment frag_do[2][Mma_tile_p::MMAS_M];
    smem_do.load(frag_do[0], 0);

    // Load the fragments for O.
    typename Smem_tile_o::Fragment frag_o[2][Mma_tile_p::MMAS_M];
    smem_o.load(frag_o[0], 0);

    // Load the fragments for V. We keep the data in registers during the entire kernel.
    typename Smem_tile_v::Fragment frag_v[Mma_tile_p::MMAS_K][Mma_tile_p::MMAS_N];
    #pragma unroll
    for( int ki = 0; ki < Mma_tile_p::MMAS_K; ++ki ) {
        smem_v.load(frag_v[ki], ki);
    }

    // Commit the data for V to shared memory if it has not been done already.
    if( Kernel_traits::SHARE_SMEM_FOR_K_AND_V ) {
        // Make sure we are done loading the fragments for K.
        __syncthreads();

        // Commit the data to shared memory for V.
        gmem_k.commit(gemm_q_k.smem_k);

        // Make sure the data is in shared memory.
        __syncthreads();
    }

    // Load the fragments for K.
    gemm_q_k.load_k();
    // Load the fragments for K^T.
    typename Smem_tile_kt::Fragment frag_kt[2][Mma_tile_dq::MMAS_N];
    smem_kt.load(frag_kt[0], 0);
    // typename Smem_tile_kt::Fragment frag_kt[Mma_tile_dq::MMAS_K][Mma_tile_dq::MMAS_N];
    // #pragma unroll
    // for( int ki = 0; ki < Mma_tile_dq::MMAS_K; ++ki ) {
    //     smem_kt.load(frag_kt[ki], ki);
    // }

    // Create the object to do the softmax.
    // We won't be using the shared memory for this softmax at all
    // Softmax softmax(params, &smem_[Smem_tile_do::BYTES_PER_TILE + Gemm1::SMEM_OFFSET_O + Smem_tile_dq::BYTES_PER_TILE], bidb, tidx);
    Softmax softmax(params, smem_, tidx);
    // Softmax softmax_dp(params, &smem_[Smem_tile_do::BYTES_PER_TILE + Gemm1::SMEM_OFFSET_O + Smem_tile_dq::BYTES_PER_TILE], bidb, tidx);
    Gmem_softmax_sum gmem_softmax_sum(params.softmax_lse_ptr, params, tidx);
    Gmem_softmax_sum gmem_softmax_d(params.dsoftmax_sum, params, tidx);

    constexpr int STEPS = Cta_tile_p::N / Cta_tile_p::M;
    // Load over the entire sequence length.
    for( int l = 0; l < STEPS; l++ ) {
        const int loop = l * Cta_tile_p::M;
        if( loop >= binfo.actual_seqlen )
            break;

        float p_lse[Mma_tile_p::MMAS_M * 2];
        gmem_softmax_sum.load(reinterpret_cast<uint32_t(&)[Mma_tile_p::MMAS_M * 2]>(p_lse));
        gmem_softmax_sum.move();

        // Declare the accumulators for the 1st gemm.
        fmha::Fragment_accumulator acc_p[Mma_tile_p::MMAS_M][Mma_tile_p::MMAS_N];
        fmha::Clear_accumulator<typename fmha::Accumulator_type, Cta_tile_p::WARPS_K>::apply(acc_p);

        // Do this part of P^T = (Q * K^T)^T.
        gemm_q_k(acc_p);

        // Trigger the load for the next Q values.
        if( l < STEPS - 1) {
            gemm_q_k.smem_q.move_to_next_write_buffer();
            gmem_q.move();
            gmem_q.load();
        }

        // Load the mask for that iteration.
        mask.load(l);

        // Convert from the accumulator type to FP32 for Softmax.
        softmax.unpack_noscale(acc_p);

        // Apply the mask.
        softmax.apply_mask(mask);

        // Scale by log-sum-exp of the softmax
        softmax.template apply_exp</*max_in_base2=*/true>(p_lse);

        // softmax.unpack_noscale_half_and_apply_mask(acc_p, mask);

        using Frag_p = fmha::Fragment_a<fmha::Row>;
        Frag_p frag_p[Mma_tile_dq::MMAS_K][Mma_tile_dq::MMAS_M];
        static_assert(Mma_tile_dq::MMAS_M == Mma_tile_p::MMAS_M);
        static_assert(Mma_tile_dq::MMAS_K == Mma_tile_p::MMAS_N);
        softmax.pack(frag_p);

        // if( Kernel_traits::SHARE_SMEM_FOR_K_AND_V && l == 0 ) {
        //     // if we share K and V, it could be that V was not fully read yet but we write into smem for reduction
        //     __syncthreads();
        // }

        float dp_sum_new[Mma_tile_p::MMAS_M * 2] = {0};
        dot_fragments(dp_sum_new, frag_do[0], frag_o[0]);

        fmha::Fragment_accumulator acc_dp[Mma_tile_p::MMAS_M][Mma_tile_p::MMAS_N];
        fmha::Clear_accumulator<fmha::Accumulator_type, Cta_tile_p::WARPS_K>::apply(acc_dp);
        // Do this part of dP^T = (dO * V^T)^T.
        #pragma unroll
        for( int ki = 1; ki < Mma_tile_p::MMAS_K; ++ki ) {
            // Trigger the load from shared memory for the next series of dO values.
            smem_do.load(frag_do[ki & 1], ki);
            smem_o.load(frag_o[ki & 1], ki);
            dot_fragments(dp_sum_new, frag_do[ki & 1], frag_o[ki & 1]);
            // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0) && (l == 0))  {
            //     printf("dp_sum_new=%.6f, %.6f\n", dp_sum_new[0], dp_sum_new[1]);
            // }
            // smem_v.load(frag_v[ki & 1], ki);
            // Do the math for the values already in registers.
            // fmha::gemm(acc_dp, frag_do[(ki - 1) & 1], frag_v[(ki - 1) & 1]);
            // if ((threadIdx.x == 1) && (blockIdx.x == 0) && (blockIdx.y == 0) && (l == 0))  {
            //     float2 tmp = __half22float2(reinterpret_cast<__half2 &>(frag_do[(ki - 1) & 1]));
            //     printf("frag_do=%.6f, %.6f\n", tmp.x, tmp.y);
            //     tmp = __half22float2(reinterpret_cast<__half2 &>(frag_v[ki - 1]));
            //     printf("frag_v=%.6f, %.6f\n", tmp.x, tmp.y);
            // }
            fmha::gemm(acc_dp, frag_do[(ki - 1) & 1], frag_v[ki - 1]);
        }

        // Do the final stage of math.
        {
            int ki = Mma_tile_p::MMAS_K;
            // fmha::gemm(acc_dp, frag_do[(ki - 1) & 1], frag_v[(ki - 1) & 1]);
            fmha::gemm(acc_dp, frag_do[(ki - 1) & 1], frag_v[(ki - 1)]);
        }

        // if ((threadIdx.x == 1) && (blockIdx.x == 0) && (blockIdx.y == 0) && (l == 0))  {
        //     printf("acc_dp=%.6f, %.6f\n", acc_dp[0][0].elt(0), acc_dp[0][0].elt(1));
        // }

        // Trigger the load for the next dO values.
        if( l < STEPS - 1) {
            smem_do.move_to_next_write_buffer();
            gmem_do.move();
            gmem_do.load();
            smem_o.move_to_next_write_buffer();
            gmem_o.move();
            gmem_o.load();
        }

        // softmax_dp.unpack_noscale(acc_dp);
        // // TD [2022-04-01]: Don't need to apply mask since the corresponding value in softmax
        // // will be zero.

        // #pragma unroll
        // for( int mi = 0; mi < Mma_tile_p::MMAS_M * 2; mi++ ) {
        //     #pragma unroll
        //     for( int ni = 0; ni < Mma_tile_p::MMAS_N * 4; ni++ ) {
        //         softmax_dp.elt_[mi][ni] *= softmax.elt_[mi][ni];
        //     }
        // }

        // float dp_sum[Mma_tile_p::MMAS_M * 2];
        // softmax_dp.reduce_sum(dp_sum);

        // gmem_softmax_d.store(reinterpret_cast<uint32_t(&)[Mma_tile_p::MMAS_M * 2]>(dp_sum));
        // gmem_softmax_d.move();

        // #pragma unroll
        // for( int mi = 0; mi < Mma_tile_p::MMAS_M * 2; mi++ ) {
        //     #pragma unroll
        //     for( int ni = 0; ni < Mma_tile_p::MMAS_N * 4; ni++ ) {
        //         softmax_dp.elt_[mi][ni] -= dp_sum[mi] * softmax.elt_[mi][ni];
        //     }
        // }

        fmha::SumOp<float> sum_op;
        fmha::quad_allreduce(dp_sum_new, dp_sum_new, sum_op);

        // softmax_dp.unpack_noscale(acc_dp);
        softmax.unpack_noscale(acc_dp);
        // #pragma unroll
        // for( int mi = 0; mi < Mma_tile_p::MMAS_M * 2; mi++ ) {
        //     #pragma unroll
        //     for( int ni = 0; ni < Mma_tile_p::MMAS_N * 4; ni++ ) {
        //         // softmax_dp.elt_[mi][ni] -= dp_sum_new[mi];
        //         softmax.elt_[mi][ni] -= dp_sum_new[mi];
        //     }
        // }
        softmax.subtract_dp_sum(dp_sum_new);

        Frag_p frag_dp[Mma_tile_dq::MMAS_K][Mma_tile_dq::MMAS_M];
        // softmax_dp.pack(frag_dp);
        softmax.pack(frag_dp);

        gmem_softmax_d.store(reinterpret_cast<uint32_t(&)[Mma_tile_p::MMAS_M * 2]>(dp_sum_new));
        gmem_softmax_d.move();

        #pragma unroll
        for( int mi = 0; mi < Mma_tile_p::MMAS_M; mi++ ) {
            #pragma unroll
            for( int ni = 0; ni < Mma_tile_p::MMAS_N; ni++ ) {
                frag_p[mi][ni].hmul(frag_dp[mi][ni]);
            }
        }

        // softmax_dp.pack(frag_p);
        // gmem_s.store(frag_p, mask);
        // gmem_s.move();

        // __syncthreads();
        // Commit the values for Q and dO into shared memory.
        if(l < STEPS - 1) {
            gmem_q.commit(gemm_q_k.smem_q);
            gmem_do.commit(smem_do);
            gmem_o.commit(smem_o);
        }

        // Declare the accumulators for the 2nd gemm.
        fmha::Fragment_accumulator acc_dq[Mma_tile_dq::MMAS_M][Mma_tile_dq::MMAS_N];
        fmha::Clear_accumulator<typename fmha::Accumulator_type, Cta_tile_dq::WARPS_K>::apply(acc_dq);

        // Do this part of O = P^T * V^T.
        #pragma unroll
        for( int ki = 1; ki < Mma_tile_dq::MMAS_K; ++ki ) {
            // Trigger the load from shared memory for the next series of Q values.
            smem_kt.load(frag_kt[ki & 1], ki);
            // Do the math for the values already in registers.
            fmha::gemm(acc_dq, frag_p[ki - 1], frag_kt[(ki - 1) & 1]);
            // fmha::gemm(acc_dq, frag_p[ki - 1], frag_kt[(ki - 1)]);
        }
        // Do the final stage of math.
        {
            int ki = Mma_tile_dq::MMAS_K;
            fmha::gemm(acc_dq, frag_p[ki - 1], frag_kt[(ki - 1) & 1]);
            // fmha::gemm(acc_dq, frag_p[ki - 1], frag_kt[(ki - 1)]);
        }

        // Loop over MMAS_M.
        #pragma unroll
        for( int ii = 0; ii < Gmem_tile_dq::LOOPS; ++ii ) {

            // Swizzle the elements and do the final reduction.
            smem_dq.store(acc_dq, ii);

            // Make sure the data is in shared memory.
            __syncthreads();

            // Load from shared memory.
            uint4 out[Gmem_tile_dq::STGS_PER_LOOP];
            smem_dq.load(out);

            // Make sure the data was read from shared memory.
            if( ii < Gmem_tile_dq::LOOPS - 1 ) {
                __syncthreads();
            }

            // Output the values.
            gmem_dq.store(out, ii);
        }

        // Move to the next part of the output.
        gmem_dq.move();

        gemm_q_k.reload_k();
        smem_kt.load(frag_kt[0], 0);

        // // Make sure the data is in shared memory.
        // __syncthreads();

        // Commit the values for Q and dO into shared memory.
        if(l < STEPS - 1) {
            gemm_q_k.smem_q.move_to_next_read_buffer();
            gemm_q_k.reload_q();
            smem_do.move_to_next_read_buffer();
            smem_do.load(frag_do[0], 0);
            smem_o.move_to_next_read_buffer();
            smem_o.load(frag_o[0], 0);
        }

    }  // Outer loop over the sequence length.
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, typename Params>
inline __device__ void compute_dv_dk_1xN(const Params &params) {

    // The description of the CTA tile for the 1st batched GEMM.
    using Cta_tile_p = typename Kernel_traits::Cta_tile_p;
    // The description of the CTA tile for the 2nd batched GEMM.
    using Cta_tile_dkv = typename Kernel_traits::Cta_tile_o;

    // The MMA tile for the 1st GEMM.
    using Mma_tile_p = fmha::Hmma_tile<Cta_tile_p>;
    // The MMA tile for the 2nd GEMM.
    using Mma_tile_dk = fmha::Hmma_tile<Cta_tile_dkv>;

    // The global memory tile to load Q. Treating Q as K.
    using Gmem_tile_q = typename Kernel_traits::Gmem_tile_k;

    // The global memory tile to load K. Treating K as Q.
    using Gmem_tile_k = typename Kernel_traits::Gmem_tile_q;

    // The shared memory tile to swizzle Q^T. Treat Q^T as V
    using Smem_tile_qt = typename Kernel_traits::Smem_tile_v;

    // Treating V as dO.
    // The global memory tile to load V.
    using Gmem_tile_v = typename Kernel_traits::Gmem_tile_q;
    // The shared memory tile to swizzle V.
    using Smem_tile_v = typename Kernel_traits::Smem_tile_q;

    // Treating dO as V in dQ kernel, which is the same as K in the forward kernel.
    // The global memory tile to load dO.
    using Gmem_tile_do = typename Kernel_traits::Gmem_tile_dot;
    // The shared memory tile to load dO.
    using Smem_tile_do = typename Kernel_traits::Smem_tile_k;
    // The shared memory tile to load dO^T.
    using Smem_tile_dot = typename Kernel_traits::Smem_tile_v;

    // The global memory tile to store dK and dV.
    // using Gmem_tile_dkv = typename Kernel_traits::Gmem_tile_dkv;
    using Gmem_tile_dkv = fmha::Gmem_tile_dq<Cta_tile_dkv>;
    // The shared memory tile to swizzle dK and dV.
    using Smem_tile_dkv = typename Kernel_traits::Smem_tile_o;

    using Gmem_tile_s = typename Kernel_traits::Gmem_tile_s;

    using Gmem_softmax_sum = typename Kernel_traits::Gmem_softmax_sum;

    // using Gemm1 = Gemm_Q_K<Kernel_traits, Kernel_traits::K_IN_REGS>;
    using Gemm1 = Gemm_Q_K<Kernel_traits, /*K-in_regs=*/false>;

    using Softmax = fmha::Softmax<Cta_tile_p, Kernel_traits>;

    // Shared memory.
    extern __shared__ char smem_[];

    // The block index for the batch.
    const int bidb = blockIdx.y;
    // The block index for the head.
    const int bidh = blockIdx.x;
    // The thread index.
    const int tidx = threadIdx.x;

    const BlockInfoPadded<Kernel_traits::THREADS> binfo(params, bidb, bidh, tidx);
    if( binfo.stop_early() ) return;

    Gemm1 gemm_q_k(&smem_[Smem_tile_v::BYTES_PER_TILE], tidx);
    // Allocate the global memory tile loader for Q.
    Gmem_tile_q gmem_q(params, 0, binfo, tidx);
    // Allocate the global memory tile loader for dK.
    Gmem_tile_dkv gmem_dk(params, 1, binfo, tidx);
    // Allocate the global memory tile loader for dV.
    Gmem_tile_dkv gmem_dv(params, 2, binfo, tidx);
    // Allocate the global memory tile loader for S.
    Gmem_tile_s gmem_s(params, binfo, tidx);

    fmha::Mask<Cta_tile_p> mask(binfo, tidx);

    // Allocate the global memory tile loader for K.
    Gmem_tile_k gmem_k(params, 1, binfo, tidx);
    // Allocate the global memory tile loader for V.
    Gmem_tile_v gmem_v(params, 2, binfo, tidx);
    // Allocate the shared memory tile loader for dO.
    Smem_tile_v smem_v(&smem_[0], tidx);

    // Allocate the shared memory tile loader for Q^T. We use the same as Q so be careful!!!
    Smem_tile_qt smem_qt(&smem_[Smem_tile_v::BYTES_PER_TILE + Gemm1::Smem_tile_q::BYTES_PER_TILE], tidx);

    // Allocate the global memory tile loader for dO.
    Gmem_tile_do gmem_do(params.do_ptr, params, binfo, tidx);
    // The base pointer of smem_do;
    char *smem_do_ = &smem_[Smem_tile_v::BYTES_PER_TILE + Gemm1::SMEM_OFFSET_V];
    // Allocate the shared memory tile loader for V. We use the same as K so be careful!!!
    Smem_tile_do smem_do(smem_do_, tidx);
    Smem_tile_dot smem_dot(smem_do_, tidx);

    // Allocate the shared memory tile loader for dK and dV. We use the same as K so be careful!!!
    Smem_tile_dkv smem_dkv(&smem_[Smem_tile_v::BYTES_PER_TILE + Gemm1::SMEM_OFFSET_O], tidx);

    // Trigger the loads for Q.
    gmem_q.load();
    // Trigger the loads for K.
    gmem_k.load();
    // Trigger the loads for dO.
    gmem_do.load();
    // Trigger the loads for V.
    gmem_v.load();

    const uint32_t scale_bmm1 = reinterpret_cast<const uint32_t&>(params.scale_bmm1);
    #pragma unroll
    for(int it=0; it < Gmem_tile_q::LDGS; it++){
        gmem_q.fetch_[it] = fmha::hmul8(scale_bmm1, gmem_q.fetch_[it]);
    }

    // Commit the data for K, dO, and V to shared memory.
    gmem_k.commit(gemm_q_k.smem_q);
    gmem_v.commit(smem_v);
    gmem_do.commit(smem_do);

    // Commit the data for Q to shared memory.
    if( !Kernel_traits::SHARE_SMEM_FOR_K_AND_V ) {
        gmem_q.commit(gemm_q_k.smem_k);
    }

    __syncthreads();

    // Load the fragments for K.
    gemm_q_k.load_q();

    // Load the fragments for V.
    typename Smem_tile_v::Fragment frag_v[2][Mma_tile_p::MMAS_M];
    smem_v.load(frag_v[0], 0);

    // Load the fragments for dO. We keep the data in registers during the entire kernel.
    typename Smem_tile_do::Fragment frag_do[Mma_tile_p::MMAS_K][Mma_tile_p::MMAS_N];
    #pragma unroll
    for( int ki = 0; ki < Mma_tile_dk::MMAS_K; ++ki ) {
        smem_do.load(frag_do[ki], ki);
    }

    using Smem_tile_mma_t = fmha::Smem_tile_transpose<Cta_tile_p>;
    // Smem_tile_mma_t smem_mmat(&smem_[Smem_tile_v::BYTES_PER_TILE + Gemm1::SMEM_OFFSET_O], tidx);
    Smem_tile_mma_t smem_mmat(&smem_[Smem_tile_v::BYTES_PER_TILE + Gemm1::SMEM_OFFSET_O + Smem_tile_dkv::BYTES_PER_TILE], tidx);

    // Commit the data for V to shared memory if it has not been done already.
    if( Kernel_traits::SHARE_SMEM_FOR_K_AND_V ) {
        // Make sure we are done loading the fragments for K.
        __syncthreads();

        // Commit the data to shared memory for V.
        gmem_q.commit(gemm_q_k.smem_k);

        // Make sure the data is in shared memory.
        __syncthreads();
    }

    // Load the fragments for Q.
    gemm_q_k.load_k();
    // Load the fragments for K^T.
    typename Smem_tile_qt::Fragment frag_qt[2][Mma_tile_dk::MMAS_N];
    smem_qt.load(frag_qt[0], 0);
    // typename Smem_tile_qt::Fragment frag_qt[Mma_tile_dk::MMAS_K][Mma_tile_dk::MMAS_N];
    // #pragma unroll
    // for( int ki = 0; ki < Mma_tile_dk::MMAS_K; ++ki ) {
    //     smem_qt.load(frag_qt[ki], ki);
    // }

    // Create the object to do the softmax.
    // We won't be using the shared memory for either of the softmax at all
    Softmax softmax(params, smem_, tidx);
    Softmax softmax_dp(params, smem_, tidx);
    Gmem_softmax_sum gmem_softmax_sum(params.softmax_lse_ptr, params, tidx);
    Gmem_softmax_sum gmem_softmax_d(params.dsoftmax_sum, params, tidx);

    int warp = tidx / Cta_tile_p::THREADS_PER_WARP;
    int lane = tidx % Cta_tile_p::THREADS_PER_WARP;
    int rows[Mma_tile_p::MMAS_N * 4];
    for (int ni = 0; ni < Mma_tile_p::MMAS_N; ni++) {
        rows[ni * 4 + 0] = ni * Cta_tile_p::WARPS_N * 16 + warp * 16 + (lane % 4) * 2;
        rows[ni * 4 + 1] = ni * Cta_tile_p::WARPS_N * 16 + warp * 16 + (lane % 4) * 2 + 1;
        rows[ni * 4 + 2] = ni * Cta_tile_p::WARPS_N * 16 + warp * 16 + (lane % 4) * 2 + 8;
        rows[ni * 4 + 3] = ni * Cta_tile_p::WARPS_N * 16 + warp * 16 + (lane % 4) * 2 + 9;
    }
    float p_lse[Mma_tile_p::MMAS_N * 4];
    gmem_softmax_sum.load_row(reinterpret_cast<uint32_t(&)[Mma_tile_p::MMAS_N * 4]>(p_lse), rows);
    float dp_sum[Mma_tile_p::MMAS_N * 4];
    gmem_softmax_d.load_row(reinterpret_cast<uint32_t(&)[Mma_tile_p::MMAS_N * 4]>(dp_sum), rows);
    // int qid = lane / 8;
    // int rows_shfl[Mma_tile_p::MMAS_N];
    // for (int ni = 0; ni < Mma_tile_p::MMAS_N; ni++) {
    //     rows_shfl[ni] = ni * Cta_tile_p::WARPS_N * 16 + warp * 16 + (lane % 4) * 2 + (qid / 2) * 8 + (qid % 2);
    // }
    // float p_lse[Mma_tile_p::MMAS_N];
    // gmem_softmax_sum.load_row(reinterpret_cast<uint32_t(&)[Mma_tile_p::MMAS_N]>(p_lse), rows_shfl);
    // float dp_sum[Mma_tile_p::MMAS_N];
    // gmem_softmax_d.load_row(reinterpret_cast<uint32_t(&)[Mma_tile_p::MMAS_N]>(dp_sum), rows_shfl);

    constexpr int STEPS = Cta_tile_p::N / Cta_tile_p::M;
    // Load over the entire sequence length.
    for( int l = 0; l < STEPS; l++ ) {
        const int loop = l * Cta_tile_p::M;
        if( loop >= binfo.actual_seqlen )
            break;

        typename Smem_tile_dot::Fragment frag_dot[2][Mma_tile_p::MMAS_N];
        // smem_mmat.store(frag_do, 0);
        // smem_mmat.load(frag_dot[0]);
        // smem_mmat.transpose(frag_do, frag_dot[0], 0);

        // Declare the accumulators for the 1st gemm.
        fmha::Fragment_accumulator acc_pt[Mma_tile_p::MMAS_M][Mma_tile_p::MMAS_N];
        fmha::Clear_accumulator<typename fmha::Accumulator_type, Cta_tile_p::WARPS_K>::apply(acc_pt);

        // Do this part of P^T = (Q * K^T)^T.
        gemm_q_k(acc_pt);

        // Trigger the load for the next K values.
        if( l < STEPS - 1) {
            gemm_q_k.smem_q.move_to_next_write_buffer();
            gmem_k.move();
            gmem_k.load();
        }

        // Load the mask for that iteration.
        mask.load(l);

        // Convert from the accumulator type to FP32 for Softmax.
        softmax.unpack_noscale(acc_pt);

        // Apply the mask.
        softmax.apply_mask(mask);

        // Scale by log-sum-exp of the softmax
        softmax.template apply_exp_col</*max_in_base2=*/true>(p_lse);

        using Frag_p = fmha::Fragment_a<fmha::Row>;
        Frag_p frag_p[Mma_tile_dk::MMAS_K][Mma_tile_dk::MMAS_M];
        softmax.pack(frag_p);

        // Declare the accumulators for the 1st gemm.
        fmha::Fragment_accumulator acc_dv[Mma_tile_dk::MMAS_M][Mma_tile_dk::MMAS_N];
        fmha::Clear_accumulator<typename fmha::Accumulator_type, Cta_tile_dkv::WARPS_K>::apply(acc_dv);

        smem_mmat.transpose(frag_do, frag_dot[0], 0);
        // Do this part of O = P^T * V^T.
        #pragma unroll
        for( int ki = 0; ki < Mma_tile_dk::MMAS_K; ++ki ) {
            // fmha::gemm(acc_dv, frag_p[ki], frag_dot[ki]);
            if (ki + 1 < Mma_tile_dk::MMAS_K) {
                // smem_mmat.store(frag_do, ki + 1);
                // smem_mmat.load(frag_dot[(ki + 1) % 2]);
                smem_mmat.transpose(frag_do, frag_dot[(ki + 1) % 2], ki + 1);
            }
            fmha::gemm(acc_dv, frag_p[ki], frag_dot[ki % 2]);
        }

        __syncthreads();
        // Loop over MMAS_M.
        #pragma unroll
        for( int ii = 0; ii < Gmem_tile_dkv::LOOPS; ++ii ) {
            // Swizzle the elements and do the final reduction.
            smem_dkv.store(acc_dv, ii);
            // Make sure the data is in shared memory.
            __syncthreads();
            // Load from shared memory.
            uint4 out[Gmem_tile_dkv::STGS_PER_LOOP];
            smem_dkv.load(out);
            // Make sure the data was read from shared memory.
            if( ii < Gmem_tile_dkv::LOOPS - 1 ) {
                __syncthreads();
            }
            // Output the values.
            gmem_dv.store(out, ii);
        }
        // Move to the next part of the output.
        gmem_dv.move();

        if( Kernel_traits::SHARE_SMEM_FOR_K_AND_V && l == 0 ) {
            // if we share K and V, it could be that V was not fully read yet but we write into smem for reduction
            __syncthreads();
        }

        fmha::Fragment_accumulator acc_dpt[Mma_tile_p::MMAS_M][Mma_tile_p::MMAS_N];
        fmha::Clear_accumulator<fmha::Accumulator_type, Cta_tile_p::WARPS_K>::apply(acc_dpt);
        // Do this part of dP^T = (dO * V^T)^T.
        #pragma unroll
        for( int ki = 1; ki < Mma_tile_p::MMAS_K; ++ki ) {
            // Trigger the load from shared memory for the next series of dO values.
            smem_v.load(frag_v[ki & 1], ki);
            // Do the math for the values already in registers.
            fmha::gemm(acc_dpt, frag_v[(ki - 1) & 1], frag_do[ki - 1]);
        }

        // Do the final stage of math.
        {
            int ki = Mma_tile_p::MMAS_K;
            // fmha::gemm(acc_dpt, frag_v[(ki - 1) & 1], frag_do[(ki - 1) & 1]);
            fmha::gemm(acc_dpt, frag_v[(ki - 1) & 1], frag_do[(ki - 1)]);
        }

        // Trigger the load for the next V values.
        if( l < STEPS - 1) {
            smem_v.move_to_next_write_buffer();
            gmem_v.move();
            gmem_v.load();
        }

        softmax_dp.unpack_noscale(acc_dpt);
        // TD [2022-04-01]: Don't need to apply mask since the corresponding value in softmax
        // will be zero.

        #pragma unroll
        for( int mi = 0; mi < Mma_tile_p::MMAS_M * 2; mi++ ) {
            #pragma unroll
            for( int ni = 0; ni < Mma_tile_p::MMAS_N * 4; ni++ ) {
                // softmax.elt_[mi][ni] *= (softmax_dp.elt_[mi][ni] - dp_sum[ni]);
                softmax_dp.elt_[mi][ni] -= dp_sum[ni];
                // const float tmp = __shfl_sync(0xffffffff, dp_sum[ni / 4], (ni % 4) * 8 + threadIdx.x % 8);
                // softmax_dp.elt_[mi][ni] -= tmp;
            }
        }

        Frag_p frag_dp[Mma_tile_dk::MMAS_K][Mma_tile_dk::MMAS_M];
        softmax_dp.pack(frag_dp);

        #pragma unroll
        for( int mi = 0; mi < Mma_tile_p::MMAS_M; mi++ ) {
            #pragma unroll
            for( int ni = 0; ni < Mma_tile_p::MMAS_N; ni++ ) {
                frag_p[mi][ni].hmul(frag_dp[mi][ni]);
            }
        }

        // using Frag_p = fmha::Fragment_a<fmha::Row>;
        // Frag_p frag_p[Mma_tile_dk::MMAS_K][Mma_tile_dk::MMAS_M];
        // softmax.pack(frag_p);
        // softmax_dp.pack(frag_p);
        // gmem_s.store(frag_p, mask);
        // gmem_s.move();

        __syncthreads();
        // Commit the values for K and V into shared memory.
        if(l < STEPS - 1) {
            gmem_k.commit(gemm_q_k.smem_q);
            gmem_v.commit(smem_v);
        }

        // Declare the accumulators for the 2nd gemm.
        fmha::Fragment_accumulator acc_dk[Mma_tile_dk::MMAS_M][Mma_tile_dk::MMAS_N];
        fmha::Clear_accumulator<typename fmha::Accumulator_type, Cta_tile_dkv::WARPS_K>::apply(acc_dk);

        // Do this part of O = P^T * V^T.
        #pragma unroll
        for( int ki = 1; ki < Mma_tile_dk::MMAS_K; ++ki ) {
            // Trigger the load from shared memory for the next series of Q values.
            smem_qt.load(frag_qt[ki & 1], ki);
            // Do the math for the values already in registers.
            fmha::gemm(acc_dk, frag_p[ki - 1], frag_qt[(ki - 1) & 1]);
            // fmha::gemm(acc_dk, frag_p[ki - 1], frag_qt[(ki - 1)]);
        }
        // Do the final stage of math.
        {
            int ki = Mma_tile_p::MMAS_K;
            fmha::gemm(acc_dk, frag_p[ki - 1], frag_qt[(ki - 1) & 1]);
            // fmha::gemm(acc_dk, frag_p[ki - 1], frag_qt[(ki - 1)]);
        }

        // Loop over MMAS_M.
        #pragma unroll
        for( int ii = 0; ii < Gmem_tile_dkv::LOOPS; ++ii ) {
            // Swizzle the elements and do the final reduction.
            smem_dkv.store(acc_dk, ii);
            // Make sure the data is in shared memory.
            __syncthreads();
            // Load from shared memory.
            uint4 out[Gmem_tile_dkv::STGS_PER_LOOP];
            smem_dkv.load(out);
            // Make sure the data was read from shared memory.
            if( ii < Gmem_tile_dkv::LOOPS - 1 ) {
                __syncthreads();
            }
            // Output the values.
            gmem_dk.store(out, ii);
        }
        // Move to the next part of the output.
        gmem_dk.move();

        gemm_q_k.reload_k();
        smem_qt.load(frag_qt[0], 0);

        // Make sure the data is in shared memory.
        // __syncthreads();

        // Commit the values for Q and dO into shared memory.
        if(l < STEPS - 1) {
            gemm_q_k.smem_q.move_to_next_read_buffer();
            gemm_q_k.reload_q();
            smem_v.move_to_next_read_buffer();
            smem_v.load(frag_v[0], 0);
        }

    }  // Outer loop over the sequence length.
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace fmha
