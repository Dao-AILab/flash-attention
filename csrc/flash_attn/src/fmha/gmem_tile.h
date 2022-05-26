/******************************************************************************
 * Copyright (c) 2011-2021, NVIDIA CORPORATION.  All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

namespace fmha {

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile_,
    // The number of bits per element.
    int BITS_PER_ELEMENT,
    // The number of rows of Q, K or V loaded by this tile.
    int ROWS_,
    // The number of columns.
    int COLS,
    // The number of matrics.
    int NUM_MATS = 3
>
struct Gmem_tile_qkv {

    using Cta_tile = Cta_tile_;

    // The size of each LDG.
    static constexpr int BYTES_PER_LDG = 16;
    // The size of a row in bytes.
    static constexpr int BYTES_PER_ROW = COLS * BITS_PER_ELEMENT / 8;

    // The number of threads to load a "row" of the matrix.
    static constexpr int THREADS_PER_ROW = BYTES_PER_ROW / BYTES_PER_LDG;

    static constexpr int ROWS = ROWS_;
    // The number of "rows" loaded per LDG.
    static constexpr int ROWS_PER_LDG = Cta_tile::THREADS_PER_CTA / THREADS_PER_ROW;
    // The number of LDGs needed to load a chunk of the Q matrix.
    static constexpr int LDGS = DivUpConstexpr(ROWS, ROWS_PER_LDG);

    // Ctor.
    template< typename Params, typename BInfo >
    inline __device__ Gmem_tile_qkv(const Params &params, const int qkv_offset, const BInfo &binfo, const int tidx)
        : params_qkv_stride_in_bytes_(params.qkv_stride_in_bytes)
        , actual_seqlen(binfo.actual_seqlen)
        , qkv_ptr_(reinterpret_cast<char *>(params.qkv_ptr))
        , tidx_(tidx) {

        // Compute the position in the sequence (within the CTA for the moment).
        int row = tidx / THREADS_PER_ROW;
        // Compute the position of the thread in the row.
        int col = tidx % THREADS_PER_ROW;

        // Store the row as we need it to disable the loads.
        // TD [2022-04-16]: To minimize registers, we'll recompute row_ instead of storing it
        // row_ = row;

        // The row offset in the batched GEMM. For each seq element, we store QKV in that order.
        // int64_t row_offset = (int64_t)row * params.qkv_stride_in_bytes;
        uint32_t row_offset = (uint32_t)row * params.qkv_stride_in_bytes;
        // Add the block index.
        // row_offset += (int64_t)((binfo.sum_s * NUM_MATS + qkv_offset) * binfo.h + binfo.bidh) * BYTES_PER_ROW;
        row_offset += (uint32_t)((binfo.sum_s * NUM_MATS + qkv_offset) * binfo.h + binfo.bidh) * BYTES_PER_ROW;

        // Assemble the final pointer.
        qkv_ptr_ += row_offset + col * BYTES_PER_LDG;
    }

    // Store data to shared memory.
    template< typename Smem_tile >
    inline __device__ void commit(Smem_tile &smem_tile) {
        smem_tile.store(fetch_);
    }

    inline __device__ void load() {
        int row_ = tidx_ / THREADS_PER_ROW;
        const void *ptrs[LDGS];
        uint32_t preds[LDGS];
        #pragma unroll
        for( int ii = 0; ii < LDGS; ++ii ) {
            // ptrs[ii] = qkv_ptr_ + (int64_t)ii * ROWS_PER_LDG * params_qkv_stride_in_bytes_;
            ptrs[ii] = qkv_ptr_ + (uint32_t)ii * ROWS_PER_LDG * params_qkv_stride_in_bytes_;
            preds[ii] = ((row_ + ii * ROWS_PER_LDG) < min(ROWS, actual_seqlen));
            fetch_[ii] = make_uint4(0, 0, 0, 0);
        }

        // not packing predicates removes restrictions (e.g. FP16 384, 4 warps)
        Ldg_functor<uint4, LDGS> fct(fetch_, ptrs);
        #pragma unroll
        for( int ii = 0; ii < LDGS; ++ii ) {
            fct.load(ii, preds[ii]);
        }
    }

    // Store data to memory.
    inline __device__ void store(const uint4 (&data)[LDGS]) {
        int row_ = tidx_ / THREADS_PER_ROW;
        #pragma unroll
        for( int ii = 0; ii < LDGS; ++ii ) {
            // char *ptr = qkv_ptr_ + (int64_t)ii * ROWS_PER_LDG * params_qkv_stride_in_bytes_;
            char *ptr = qkv_ptr_ + (uint32_t)ii * ROWS_PER_LDG * params_qkv_stride_in_bytes_;
            if( (row_ + ii * ROWS_PER_LDG) < min(ROWS, actual_seqlen) ) {
                fmha::stg(ptr, data[ii]);
            }
        }
    }

    // Move the pointer to the next location.
    inline __device__ void move() {
        // qkv_ptr_ += (int64_t)ROWS * params_qkv_stride_in_bytes_;
        qkv_ptr_ += (uint32_t)ROWS * params_qkv_stride_in_bytes_;
        actual_seqlen -= ROWS;
    }

    inline __device__ void move(int steps) {
        // qkv_ptr_ += (int64_t)ROWS * params_qkv_stride_in_bytes_ * steps;
        qkv_ptr_ += (uint32_t)ROWS * params_qkv_stride_in_bytes_ * steps;
        actual_seqlen -= ROWS * steps;
    }

    // The stride between rows for the QKV matrice.
    // int64_t params_qkv_stride_in_bytes_;
    uint32_t params_qkv_stride_in_bytes_;
    // The pointer.
    char *qkv_ptr_;
    // The fetch registers.
    uint4 fetch_[LDGS];
    // Keep track of the row the thread is processing as we move the tile.
    // int row_;
    const int tidx_;
    // The length of the sequence loaded by that memory tile.
    int actual_seqlen;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    typename Cta_tile,
    int BYTES_PER_ELEMENT = 2
>
struct Gmem_tile_o {

    static_assert(BYTES_PER_ELEMENT == 2 || BYTES_PER_ELEMENT == 4);

    // The mma tile.
    using Mma_tile = fmha::Hmma_tile<Cta_tile>;

    // The size of each element.
    // static constexpr int BYTES_PER_ELEMENT = 2;
    // The size of each STG.
    static constexpr int BYTES_PER_STG = BYTES_PER_ELEMENT * 4;
    static constexpr int COLS = Cta_tile::N;
    // The size of a row in bytes.
    static constexpr int BYTES_PER_ROW = COLS * BYTES_PER_ELEMENT;

    // The number of threads to store a "row" of the matrix.
    static constexpr int THREADS_PER_ROW = BYTES_PER_ROW / BYTES_PER_STG;
    // The number of "rows" stored per iteration of the loop. The output of 1 MMA.
    static constexpr int ROWS = Cta_tile::M;
    // The number of "rows" stored per iteration of the loop. The output of 1 MMA.
    static constexpr int ROWS_PER_LOOP = ROWS <= 64 ? ROWS : (int)Mma_tile::M_PER_MMA_PER_CTA;
    // The number of outter loop for the stores.
    static constexpr int LOOPS = ROWS / ROWS_PER_LOOP;

    // The number of "rows" stored per STG.
    static constexpr int ROWS_PER_STG = Cta_tile::THREADS_PER_CTA / THREADS_PER_ROW;
    // Do we have to guard against partial writes/reads.
    static constexpr bool HAS_INCOMPLETE_STG = Cta_tile::M % ROWS_PER_STG != 0;
    // The number of STGs needed to store a chunk of the Q matrix.
    static constexpr int STGS_PER_LOOP = DivUpConstexpr(ROWS_PER_LOOP, ROWS_PER_STG);
    // The number of STGs needed to store a chunk of the Q matrix in total.
    static constexpr int STGS = STGS_PER_LOOP * LOOPS;

    // Ctor.
    template<typename BInfo>
    // inline __device__ Gmem_tile_o(void *ptr, const size_t stride_in_elts, const BInfo &binfo, const int tidx)
    inline __device__ Gmem_tile_o(void *ptr, const uint32_t stride_in_elts, const BInfo &binfo, const int tidx)
        : stride_in_bytes_(stride_in_elts * BYTES_PER_ELEMENT)
        , actual_seqlen_(binfo.actual_seqlen)
        , actual_seqlen(binfo.actual_seqlen)
        , ptr_(reinterpret_cast<char *>(ptr))
        , tidx_(tidx) {

        // Compute the position in the sequence (within the CTA for the moment).
        int row = tidx / THREADS_PER_ROW;
        // Compute the position of the thread in the row.
        int col = tidx % THREADS_PER_ROW;

        // Store the row as we need it to disable loads.
        // row_ = row;

        // The row offset in the batched GEMM.
        // int64_t row_offset = (int64_t)row * stride_in_bytes_ + binfo.bidx * BYTES_PER_ROW;
        uint32_t row_offset = (uint32_t)row * stride_in_bytes_ + binfo.bidx * BYTES_PER_ROW;
        // Assemble the final pointer.
        ptr_ += row_offset + col * BYTES_PER_STG;

        // Is that thread active on the last STG?
        if( HAS_INCOMPLETE_STG ) {
            is_active_for_last_stg_ = row + (STGS - 1) * ROWS_PER_STG < Cta_tile::M;
        }
    }

    template<typename Params, typename BInfo>
    inline __device__ Gmem_tile_o(const Params &params, const BInfo &binfo, const int tidx)
        : Gmem_tile_o(params.o_ptr, params.o_stride_in_elts, binfo, tidx) {}

    // Store data to global memory.
    inline __device__ void store(const uint4 (&src)[STGS_PER_LOOP], int mi) {
        int row_ = tidx_ / THREADS_PER_ROW;
        #pragma unroll
        for( int ii = 0; ii < STGS_PER_LOOP; ++ii ) {
            int jj = mi * STGS_PER_LOOP + ii;
            // if( this->row_ + jj * ROWS_PER_STG >= this->actual_seqlen_ ) {
            //     break;
            if( row_ + jj * ROWS_PER_STG >= this->actual_seqlen ) {
                break;
            }

            if (BYTES_PER_ELEMENT == 4) {
                if( !HAS_INCOMPLETE_STG || (jj < STGS - 1 || this->is_active_for_last_stg_) ) {
                    fmha::stg(this->ptr_ + jj * ROWS_PER_STG * this->stride_in_bytes_, src[ii]);
                }
            } else if (BYTES_PER_ELEMENT == 2) {
                float x = reinterpret_cast<const float &>(src[ii].x);
                float y = reinterpret_cast<const float &>(src[ii].y);
                float z = reinterpret_cast<const float &>(src[ii].z);
                float w = reinterpret_cast<const float &>(src[ii].w);
                uint2 out = float4_to_half4(x, y, z, w);
                if( !HAS_INCOMPLETE_STG || (jj < STGS - 1 || this->is_active_for_last_stg_) ) {
                    fmha::stg(this->ptr_ + jj * ROWS_PER_STG * this->stride_in_bytes_, out);
                }
            }
        }
    }

    // Store data to global memory.
    inline __device__ void load(uint4 (&dst)[STGS_PER_LOOP], int mi) {
        static_assert(BYTES_PER_ELEMENT == 4);
        int row_ = tidx_ / THREADS_PER_ROW;
        #pragma unroll
        for( int ii = 0; ii < STGS_PER_LOOP; ++ii ) {
            int jj = mi * STGS_PER_LOOP + ii;
            if( row_ + jj * ROWS_PER_STG >= this->actual_seqlen ) {
                break;
            }

            if( !HAS_INCOMPLETE_STG || (jj < STGS - 1 || this->is_active_for_last_stg_) ) {
                fmha::ldg(dst[ii], this->ptr_ + jj * ROWS_PER_STG * this->stride_in_bytes_);
            }
        }
    }

    // Move the pointer to the next location.
    inline __device__ void move() {
        // row_ += ROWS;
        // ptr_ += (int64_t)ROWS * stride_in_bytes_;
        ptr_ += (uint32_t)ROWS * stride_in_bytes_;
        actual_seqlen -= ROWS;
    }

    inline __device__ void move(const int steps) {
        // row_ += ROWS * steps;
        // ptr_ += (int64_t)ROWS * stride_in_bytes_ * steps;
        ptr_ += (uint32_t)ROWS * stride_in_bytes_ * steps;
        actual_seqlen -= ROWS * steps;
    }

    // The stride between rows for the QKV matrice.
    // int64_t stride_in_bytes_;
    uint32_t stride_in_bytes_;
    // The pointer.
    char *ptr_;
    // Is the thread active for the last STG?
    int is_active_for_last_stg_;
    // Keep track of the row to disable loads.
    // int row_;
    // The length of the sequence loaded by that memory tile.
    const int actual_seqlen_;
    int actual_seqlen;
    const int tidx_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, int BYTES_PER_ELEMENT >
struct Gmem_tile_mma_sd {

    // The mma tile.
    using Mma_tile = fmha::Hmma_tile<Cta_tile>;

    // Each STG stores 8 elements.
    static constexpr int BYTES_PER_STG = BYTES_PER_ELEMENT * 8;
    // The number of MMAs in the M dimension.
    static constexpr int MMAS_M = Mma_tile::MMAS_M;
    // The number of MMAs in the N dimension.
    static constexpr int MMAS_N = Mma_tile::MMAS_N;
    // The number of rows computed per MMA per thread block.
    static constexpr int M_PER_MMA_PER_CTA = Mma_tile::M_PER_MMA_PER_CTA;
    // The number of cols computed per MMA per thread block.
    static constexpr int N_PER_MMA_PER_CTA = Mma_tile::N_PER_MMA_PER_CTA;
    // The number of threads per block.
    static constexpr int THREADS_PER_CTA = Cta_tile::THREADS_PER_CTA;
    // The size of each row in bytes. I.e. how many bytes are stored per STG.
    static constexpr int BYTES_PER_ROW = THREADS_PER_CTA * BYTES_PER_STG;
    // The distance between elements stored per loop (in bytes).
    static constexpr int LOOP_STRIDE_BYTES = MMAS_M * MMAS_N * BYTES_PER_ROW;

    // The type of elements stored per STG.
    using Type = typename fmha::Uint_from_size_in_bytes<BYTES_PER_STG>::Type;

    // Ctor.
    template<typename Params>
    inline __device__ Gmem_tile_mma_sd(void *ptr, const Params &params, const int bidb, const int bidh, const int tidx) 
        : ptr_(static_cast<char *>(ptr)) {

        // The block index.
        // size_t bidx = bidb * params.h + bidh;
        uint32_t bidx = bidb * params.h + bidh;

        // The distance between two blocks (in bytes).
        // const size_t block_stride_bytes = params.s * params.s * BYTES_PER_ELEMENT;
        const uint32_t block_stride_bytes = params.s * params.s * BYTES_PER_ELEMENT;
        // Set store location for each thread at the beginning of the loop
        ptr_ += bidx * block_stride_bytes + tidx * BYTES_PER_STG;
    }

    // Store to global memory.
    inline __device__ void store(const Type &data, const int mi, const int ni) {
        // size_t offset = (mi * MMAS_N + ni) * BYTES_PER_ROW;
        uint32_t offset = (mi * MMAS_N + ni) * BYTES_PER_ROW;
        fmha::stg(ptr_ + offset, data);
    }

    // Load from global memory.
    inline __device__ void load(Type &data, const int mi, const int ni) {
        // size_t offset = (mi * MMAS_N + ni) * BYTES_PER_ROW;
        uint32_t offset = (mi * MMAS_N + ni) * BYTES_PER_ROW;
        fmha::ldg(data, ptr_ + offset);
    }

    // Move to the next tile.
    inline __device__ void move() {
        ptr_ += LOOP_STRIDE_BYTES;
    }
    inline __device__ void move(const int steps) {
        ptr_ += LOOP_STRIDE_BYTES * steps;
    }

    // The pointer in global memory.
    char *ptr_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, typename Base = Gmem_tile_mma_sd<Cta_tile, sizeof(uint16_t)> >
struct Gmem_tile_mma_s : public Base {

    // The number of mmas in the vertical dimension.
    static constexpr int M = Base::MMAS_M;
    // The number of mmas in the horizontal dimension.
    static constexpr int N = Base::MMAS_N;
    // The type of the vectors stored by each STG.
    using Type = typename Base::Type;

    // Ctor.
    template< typename Params, typename Block_info >
    inline __device__ Gmem_tile_mma_s(const Params &params, const Block_info& binfo, const int tidx) 
        : Base(params.s_ptr, params, binfo.bidb, binfo.bidh, tidx) {
    }

    // Store to global memory.
    template<typename Mask>
    inline __device__ void store(const float (&softmax)[2 * M][4 * N], const Mask &mask) {
        #pragma unroll
        for( int mi = 0; mi < M; mi++ ) {
            #pragma unroll
            for( int ni = 0; ni < N; ni++ ) {

                float tmp00 = softmax[2 * mi + 0][4 * ni + 0];
                float tmp01 = softmax[2 * mi + 0][4 * ni + 1];
                float tmp02 = softmax[2 * mi + 0][4 * ni + 2];
                float tmp03 = softmax[2 * mi + 0][4 * ni + 3];

                float tmp10 = softmax[2 * mi + 1][4 * ni + 0];
                float tmp11 = softmax[2 * mi + 1][4 * ni + 1];
                float tmp12 = softmax[2 * mi + 1][4 * ni + 2];
                float tmp13 = softmax[2 * mi + 1][4 * ni + 3];

                uint4 dst;
                dst.x = fmha::float2_to_half2(tmp00, tmp01);
                dst.y = fmha::float2_to_half2(tmp02, tmp03);
                dst.z = fmha::float2_to_half2(tmp10, tmp11);
                dst.w = fmha::float2_to_half2(tmp12, tmp13);
                if( mask.is_valid(mi, ni, 0, 0) ) {
                    Base::store(dst, mi, ni);
                }
            }
        }
    }

    // Store to global memory.
    template<typename Mask, typename Fragment>
    inline __device__ void store(const Fragment (&frag)[N][M], const Mask& mask){
        #pragma unroll
        for( int mi = 0; mi < M; mi++ ) {
            #pragma unroll
            for( int ni = 0; ni < N; ni++ ) {
                uint4 dst;
                dst.x = frag[ni][mi].reg(0);
                dst.y = frag[ni][mi].reg(2);
                dst.z = frag[ni][mi].reg(1);
                dst.w = frag[ni][mi].reg(3);
                if( mask.any_valid(mi, ni) ) {
                    Base::store(dst, mi, ni);
                }
            }
        }
    }

    // Load from global memory.
    template<typename Mask>
    inline __device__ void load(uint4 (&regs)[M][N], const Mask &mask) {
        #pragma unroll
        for( int mi = 0; mi < M; mi++ ) {
            #pragma unroll
            for( int ni = 0; ni < N; ni++ ) {
                regs[mi][ni] = make_uint4(0, 0, 0, 0);
                if( mask.any_valid(mi, ni) ) {
                    Base::load(regs[mi][ni], mi, ni);
                }
            }
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The base class.
    typename Base = fmha::Gmem_tile_qkv<Cta_tile, fmha::BITS_PER_ELEMENT_A, Cta_tile::M, Cta_tile::K>
>
struct Gmem_tile_dout : public Base {

    // Ctor.
    template<typename Params, typename BInfo>
    inline __device__ Gmem_tile_dout(void *ptr, const Params &params, const BInfo &binfo, int tidx)
        : Base(params, 0, binfo, tidx) {

        // this->qkv_ptr_ = reinterpret_cast<char *>(params.do_ptr);
        this->qkv_ptr_ = static_cast<char *>(ptr);
        this->params_qkv_stride_in_bytes_ = params.o_stride_in_bytes;  // needed for move

        // Compute the position in the sequence (within the CTA for the moment).
        int row = tidx / Base::THREADS_PER_ROW;
        // Compute the position of the thread in the row.
        int col = tidx % Base::THREADS_PER_ROW;

        // The row offset in the batched GEMM. For each seq element, we store O in that order.
        // int64_t row_offset = (int64_t)this->row_ * params.o_stride_in_bytes + binfo.bidx * Base::BYTES_PER_ROW;
        // int64_t row_offset = (int64_t)row * params.o_stride_in_bytes + binfo.bidx * Base::BYTES_PER_ROW;
        uint32_t row_offset = (uint32_t)row * params.o_stride_in_bytes + binfo.bidx * Base::BYTES_PER_ROW;

        // Assemble the final pointer.
        this->qkv_ptr_ += row_offset + col * Base::BYTES_PER_LDG;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, typename Base = fmha::Gmem_tile_o<Cta_tile> >
struct Gmem_tile_dq : public Base {

    // Ctor.
    template<typename Params, typename BInfo>
    inline __device__ Gmem_tile_dq(const Params &params, const int qkv_offset, const BInfo &binfo, int tidx)
        : Base(params.dqkv_ptr, params.qkv_stride_in_elts, binfo, tidx) {
        this->ptr_ = reinterpret_cast<char *>(params.dqkv_ptr);

        // Compute the position in the sequence (within the CTA for the moment).
        int row = tidx / Base::THREADS_PER_ROW;
        // Compute the position of the thread in the row.
        int col = tidx % Base::THREADS_PER_ROW;

        // The row offset in the batched GEMM. For each seq element, we store O in that order.
        // int64_t row_offset = (int64_t)this->row_ * params.qkv_stride_in_bytes +
        //     ((binfo.sum_s * 3 + qkv_offset) * binfo.h + binfo.bidh) * Base::BYTES_PER_ROW;
        // int64_t row_offset = (int64_t)row * this->stride_in_bytes_ +
        //     ((binfo.sum_s * 3 + qkv_offset) * binfo.h + binfo.bidh) * Base::BYTES_PER_ROW;
        uint32_t row_offset = (uint32_t)row * this->stride_in_bytes_ +
            ((binfo.sum_s * 3 + qkv_offset) * binfo.h + binfo.bidh) * Base::BYTES_PER_ROW;

        // Assemble the final pointer.
        this->ptr_ += row_offset + col * Base::BYTES_PER_STG;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile
>
struct Gmem_summary_stats {

    // The Mma tile.
    using Mma_tile = fmha::Hmma_tile<Cta_tile>;

    // The number of MMAs in M/N dimensions.
    static constexpr int MMAS_M = Mma_tile::MMAS_M;

    // The size of each element.
    static constexpr int BYTES_PER_ELEMENT = 4;
    static constexpr int BYTES_PER_MMA = (Cta_tile::THREADS_PER_WARP / 4) * 2 * BYTES_PER_ELEMENT;
    static constexpr int ROWS = Cta_tile::M;

    // Ctor.
    template<typename Params>
    inline __device__ Gmem_summary_stats(void *ptr, const Params &params, const int tidx)
        : ptr_(reinterpret_cast<char *>(ptr)), tidx_(tidx) {

        // The block index for the batch.
        const int bidb = blockIdx.y;
        // The block index for the head.
        const int bidh = blockIdx.x;
        // The block index.
        // size_t bidx = bidb * params.h + bidh;
        uint32_t bidx = bidb * params.h + bidh;

        // Extract the position in the warp.
        int warp = tidx / Cta_tile::THREADS_PER_WARP;
        int lane = tidx % Cta_tile::THREADS_PER_WARP;

        // The distance between two blocks (in bytes).
        // size_t block_stride_bytes = params.s * BYTES_PER_ELEMENT;
        uint32_t block_stride_bytes = params.s * BYTES_PER_ELEMENT;

        // Set store location for each thread at the beginning of the loop
        ptr_row_ = ptr_ + bidx * block_stride_bytes;
        ptr_ += bidx * block_stride_bytes + (lane / 4) * BYTES_PER_ELEMENT;
    }

    // Store data to global memory.
    inline __device__ void store(const uint32_t (&data)[MMAS_M * 2]) {
        int warp = tidx_ / Cta_tile::THREADS_PER_WARP;
        int lane = tidx_ % Cta_tile::THREADS_PER_WARP;
        if ((warp == 0) && (lane % 4 == 0)) {
            #pragma unroll
            for (int mi = 0; mi < MMAS_M; ++mi) {
                // TODO: Not sure if it's right for MMAS_M > 1
                fmha::stg(ptr_ + mi * BYTES_PER_MMA + 0 * BYTES_PER_ELEMENT, data[mi * 2 + 0]);
                fmha::stg(ptr_ + mi * BYTES_PER_MMA + 8 * BYTES_PER_ELEMENT, data[mi * 2 + 1]);
            }
        }
    }

    // Store data to global memory.
    inline __device__ void store_row(const uint32_t (&data)[MMAS_M], const int row) {
        #pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi) {
            // TODO: Not sure if it's right for MMAS_M > 1
            fmha::stg(ptr_row_ + mi * BYTES_PER_MMA + row * BYTES_PER_ELEMENT, data[mi]);
        }
    }

    // Load from global memory.
    inline __device__ void load(uint32_t (&data)[MMAS_M * 2]) {
        #pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi) {
            // TODO: Not sure if it's right for MMAS_M > 1
            fmha::ldg(data[mi * 2 + 0], ptr_ + mi * BYTES_PER_MMA + 0 * BYTES_PER_ELEMENT);
            fmha::ldg(data[mi * 2 + 1], ptr_ + mi * BYTES_PER_MMA + 8 * BYTES_PER_ELEMENT);
        }
    }

    // Load from global memory.
    inline __device__ void load_next(uint32_t (&data)[MMAS_M * 2], int move_steps=1) {
        char *ptr_next = ptr_ + move_steps * ROWS * BYTES_PER_ELEMENT;
        #pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi) {
            // TODO: Not sure if it's right for MMAS_M > 1
            fmha::ldg(data[mi * 2 + 0], ptr_next + mi * BYTES_PER_MMA + 0 * BYTES_PER_ELEMENT);
            fmha::ldg(data[mi * 2 + 1], ptr_next + mi * BYTES_PER_MMA + 8 * BYTES_PER_ELEMENT);
        }
    }

    // Store data to global memory.
    template <int N>
    inline __device__ void load_row(uint32_t (&data)[N], const int row[N]) {
        #pragma unroll
        for (int ni = 0; ni < N; ++ni) {
            fmha::ldg(data[ni], ptr_row_ + row[ni] * BYTES_PER_ELEMENT);
        }
    }

    // Move the pointer to the next location.
    inline __device__ void move() {
        ptr_ += ROWS * BYTES_PER_ELEMENT;
        ptr_row_ += ROWS * BYTES_PER_ELEMENT;
    }

    // Move the pointer to the next location.
    inline __device__ void move(const int steps) {
        ptr_ += ROWS * BYTES_PER_ELEMENT * steps;
        ptr_row_ += ROWS * BYTES_PER_ELEMENT * steps;
    }

    // The pointer.
    char *ptr_;
    char *ptr_row_;
    const int tidx_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace fmha

