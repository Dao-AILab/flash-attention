#include <cute/util/debug.hpp>
#include "block_info.h"

#pragma once

#define KIN_PRINT(statement) \
    if (thread0()) { \
        printf("\n[kin:start:%s]\n", #statement); \
        statement; \
        printf("\n[kin:end:%s]\n", #statement); \
    }

#define KIN_PRINT_BOOL(BOOL) \
    if (thread0()) { \
        printf("\n[kin:start:%s]\n", #BOOL); \
        printf("%s", BOOL ? "true" : "false"); \
        printf("\n[kin:end:%s]\n", #BOOL); \
    }

__forceinline__ __device__ 
void print_qkv_params(const Qkv_params& params) {
    // LLM generated
    printf("Qkv_params:\n");
    printf("q_ptr: %p\n", params.q_ptr);
    printf("k_ptr: %p\n", params.k_ptr);
    printf("v_ptr: %p\n", params.v_ptr);
    printf("q_batch_stride: %" PRId64 "\n", params.q_batch_stride);
    printf("k_batch_stride: %" PRId64 "\n", params.k_batch_stride);
    printf("v_batch_stride: %" PRId64 "\n", params.v_batch_stride);
    printf("q_row_stride: %" PRId64 "\n", params.q_row_stride);
    printf("k_row_stride: %" PRId64 "\n", params.k_row_stride);
    printf("v_row_stride: %" PRId64 "\n", params.v_row_stride);
    printf("q_head_stride: %" PRId64 "\n", params.q_head_stride);
    printf("k_head_stride: %" PRId64 "\n", params.k_head_stride);
    printf("v_head_stride: %" PRId64 "\n", params.v_head_stride);
    printf("h: %d\n", params.h);
    printf("h_k: %d\n", params.h_k);
    printf("h_h_k_ratio: %d\n", params.h_h_k_ratio);
}

__forceinline__ __device__ 
void print_flash_fwd_params(const Flash_fwd_params& params) {
    print_qkv_params(params);
    // LLM generated
    printf("struct Flash_fwd_params:\n");
    printf("o_ptr: %p\n", params.o_ptr);
    printf("oaccum_ptr: %p\n", params.oaccum_ptr);
    printf("o_batch_stride: %ld\n", params.o_batch_stride);
    printf("o_row_stride: %ld\n", params.o_row_stride);
    printf("o_head_stride: %ld\n", params.o_head_stride);
    printf("p_ptr: %p\n", params.p_ptr);
    printf("softmax_lse_ptr: %p\n", params.softmax_lse_ptr);
    printf("softmax_lseaccum_ptr: %p\n", params.softmax_lseaccum_ptr);
    printf("b: %d\n", params.b);
    printf("seqlen_q: %d\n", params.seqlen_q);
    printf("seqlen_k: %d\n", params.seqlen_k);
    printf("seqlen_knew: %d\n", params.seqlen_knew);
    printf("d: %d\n", params.d);
    printf("seqlen_q_rounded: %d\n", params.seqlen_q_rounded);
    printf("seqlen_k_rounded: %d\n", params.seqlen_k_rounded);
    printf("d_rounded: %d\n", params.d_rounded);
    printf("rotary_dim: %d\n", params.rotary_dim);
    printf("scale_softmax: %f\n", params.scale_softmax);
    printf("scale_softmax_log2: %f\n", params.scale_softmax_log2);
    printf("cu_seqlens_q: %p\n", params.cu_seqlens_q);
    printf("cu_seqlens_k: %p\n", params.cu_seqlens_k);
    printf("seqused_k: %p\n", params.seqused_k);
    printf("blockmask: %p\n", params.blockmask);
    printf("knew_ptr: %p\n", params.knew_ptr);
    printf("vnew_ptr: %p\n", params.vnew_ptr);
    printf("knew_batch_stride: %ld\n", params.knew_batch_stride);
    printf("vnew_batch_stride: %ld\n", params.vnew_batch_stride);
    printf("knew_row_stride: %ld\n", params.knew_row_stride);
    printf("vnew_row_stride: %ld\n", params.vnew_row_stride);
    printf("knew_head_stride: %ld\n", params.knew_head_stride);
    printf("vnew_head_stride: %ld\n", params.vnew_head_stride);
    printf("rotary_cos_ptr: %p\n", params.rotary_cos_ptr);
    printf("rotary_sin_ptr: %p\n", params.rotary_sin_ptr);
    printf("cache_batch_idx: %p\n", params.cache_batch_idx);
    printf("block_table: %p\n", params.block_table);
    printf("block_table_batch_stride: %ld\n", params.block_table_batch_stride);
    printf("page_block_size: %d\n", params.page_block_size);
    printf("p_dropout: %f\n", params.p_dropout);
    printf("p_dropout_in_uint8_t: %u\n", params.p_dropout_in_uint8_t);
    printf("rp_dropout: %f\n", params.rp_dropout);
    printf("scale_softmax_rp_dropout: %f\n", params.scale_softmax_rp_dropout);
    printf("window_size_left: %d\n", params.window_size_left);
    printf("window_size_right: %d\n", params.window_size_right);
    printf("philox_args: %p\n", &(params.philox_args));
    printf("rng_state: %p\n", params.rng_state);
    printf("is_bf16: %d\n", params.is_bf16);
    printf("is_causal: %d\n", params.is_causal);
    printf("is_seqlens_k_cumulative: %d\n", params.is_seqlens_k_cumulative);
    printf("is_rotary_interleaved: %d\n", params.is_rotary_interleaved);
    printf("num_splits: %d\n", params.num_splits);
    printf("alibi_slopes_ptr: %p\n", params.alibi_slopes_ptr);
    printf("alibi_slopes_batch_stride: %ld\n", params.alibi_slopes_batch_stride);
}

template<typename Kernel_traits>
__forceinline__ __device__ 
void print_traits() {
    // bool
    printf("Kernel_traits::Share_Q_K_smem    : %s\n", Kernel_traits::Share_Q_K_smem ? "true" : "false");
    printf("Kernel_traits::Is_Q_in_regs      : %s\n", Kernel_traits::Is_Q_in_regs ? "true" : "false");

    // int
    printf("Kernel_traits::kNWarps           : %d\n", Kernel_traits::kNWarps );
    printf("Kernel_traits::kNThreads         : %d\n", Kernel_traits::kNThreads );
    printf("Kernel_traits::kBlockM           : %d\n", Kernel_traits::kBlockM );
    printf("Kernel_traits::kBlockN           : %d\n", Kernel_traits::kBlockN );
    printf("Kernel_traits::kHeadDim          : %d\n", Kernel_traits::kHeadDim );
    printf("Kernel_traits::kBlockKSmem       : %d\n", Kernel_traits::kBlockKSmem );
    printf("Kernel_traits::kBlockKGmem       : %d\n", Kernel_traits::kBlockKGmem );
    printf("Kernel_traits::kSwizzle          : %d\n", Kernel_traits::kSwizzle );
    printf("Kernel_traits::kSmemQSize        : %d\n", Kernel_traits::kSmemQSize );
    printf("Kernel_traits::kSmemKVSize       : %d\n", Kernel_traits::kSmemKVSize );
    printf("Kernel_traits::kSmemSize         : %d\n", Kernel_traits::kSmemSize );
    printf("Kernel_traits::kGmemRowsPerThread: %d\n", Kernel_traits::kGmemRowsPerThread);
    printf("Kernel_traits::kGmemThreadsPerRow: %d\n", Kernel_traits::kGmemThreadsPerRow);
    printf("Kernel_traits::kGmemElemsPerLoad : %d\n", Kernel_traits::kGmemElemsPerLoad );

    // cute object
    printf("Kernel_traits::GmemLayoutAtom    : ");
    cute::print(Kernel_traits::GmemLayoutAtom());
    printf("\n");
    printf("Kernel_traits::GmemTiledCopyQKV  :\n");
    cute::print(Kernel_traits::GmemTiledCopyQKV());
    printf("\n");
    
}

template<typename BlockInfo>
__forceinline__ __device__ void
print_binfo(const BlockInfo& binfo) {
    printf("binfo.sum_s_q           : %d\n", binfo.sum_s_q);
    printf("binfo.sum_s_k           : %d\n", binfo.sum_s_k);
    printf("binfo.actual_seqlen_q   : %d\n", binfo.actual_seqlen_q);
    printf("binfo.seqlen_k_cache    : %d\n", binfo.seqlen_k_cache);
    printf("binfo.actual_seqlen_k   : %d\n", binfo.actual_seqlen_k);
}
