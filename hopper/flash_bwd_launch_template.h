/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include "cute/tensor.hpp"

#include "cutlass/cluster_launch.hpp"

#include "static_switch.h"
#include "flash.h"
#include "flash_bwd_preprocess_kernel.h"
#include "flash_bwd_kernel.h"
#include "kernel_traits.h"
#include "utils.h"

template<bool Clear_dQaccum=true, typename Kernel_traits>
__global__ void flash_bwd_dot_do_o_kernel(const Flash_bwd_params params) {
    flash::compute_dot_do_o<Clear_dQaccum, Kernel_traits>(params);
}

// template<typename Kernel_traits>
// __global__ void flash_bwd_convert_dq_kernel(const Flash_bwd_params params, const int nsplits) {
//     flash::convert_dQ<Kernel_traits>(params, nsplits);
// }

template<typename Kernel_traits>
__global__ void flash_bwd_convert_dkv_kernel(const Flash_bwd_params params) {
    flash::convert_dKV<Kernel_traits>(params);
}

template<typename Kernel_traits, bool Is_causal>
void run_flash_bwd(Flash_bwd_params &params, cudaStream_t stream) {
    int num_m_block = cute::ceil_div(params.seqlen_q, Kernel_traits::kBlockM);
    dim3 grid_m(num_m_block, params.b, params.h);
    flash_bwd_dot_do_o_kernel<true, Kernel_traits><<<grid_m, Kernel_traits::kNThreadsNonWS, 0, stream>>>(params);
    // If we use both TMA_STORE (for n_block=0) and TMA_REDUCE_ADD (for n_block>0), we don't need to clear dQaccum
    // flash_bwd_dot_do_o_kernel<false, Kernel_traits><<<grid_m, Kernel_traits::kNThreadsNonWS, 0, stream>>>(params);
    CHECK_CUDA_KERNEL_LAUNCH();

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using TileShape_MNK = typename Kernel_traits::TileShape_MNK;
    using ClusterShape = typename Kernel_traits::ClusterShape_MNK;

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)),
                            make_shape(params.seqlen_q, params.d, params.h, params.b),
                            make_stride(params.q_row_stride, _1{}, params.q_head_stride, params.q_batch_stride));
    auto tma_load_Q = make_tma_copy(
        typename Kernel_traits::GmemTiledCopyQdO{},
        mQ,
        typename Kernel_traits::SmemLayoutQ{}(_, _, _0{}),
        // typename Kernel_traits::SmemLayoutQ{},
        select<0, 2>(TileShape_MNK{}),
        size<1>(ClusterShape{})); // mcast along N mode for this M load, if any
    Tensor mdO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.do_ptr)),
                             make_shape(params.seqlen_q, params.d, params.h, params.b),
                             make_stride(params.do_row_stride, _1{}, params.do_head_stride, params.do_batch_stride));
    auto tma_load_dO = make_tma_copy(
        typename Kernel_traits::GmemTiledCopyQdO{},
        mdO,
        typename Kernel_traits::SmemLayoutdO{}(_, _, _0{}),
        // typename Kernel_traits::SmemLayoutdO{},
        select<0, 2>(TileShape_MNK{}),
        size<1>(ClusterShape{})); // mcast along N mode for this M load, if any
    Tensor mK = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr)),
                            make_shape(params.seqlen_k, params.d, params.h, params.b),
                            make_stride(params.k_row_stride, _1{}, params.k_head_stride, params.k_batch_stride));
    auto tma_load_K = make_tma_copy(
        typename Kernel_traits::GmemTiledCopyKV{},
        mK,
        typename Kernel_traits::SmemLayoutK{},
        // typename Kernel_traits::SmemLayoutK{}(_, _, _0{}),
        select<1, 2>(TileShape_MNK{}),
        _1{}); // no mcast for K
    Tensor mV = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.v_ptr)),
                            make_shape(params.seqlen_k, params.d, params.h, params.b),
                            make_stride(params.v_row_stride, _1{}, params.v_head_stride, params.v_batch_stride));
    auto tma_load_V = make_tma_copy(
        typename Kernel_traits::GmemTiledCopyKV{},
        mV,
        typename Kernel_traits::SmemLayoutV{},
        // typename Kernel_traits::SmemLayoutV{}(_, _, _0{}),
        select<1, 2>(TileShape_MNK{}),
        _1{}); // no mcast for V
    Tensor mdK = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.dk_ptr)),
                             make_shape(params.seqlen_k, params.d, params.h, params.b),
                             make_stride(params.dk_row_stride, _1{}, params.dk_head_stride, params.dk_batch_stride));
    auto tma_store_dK = make_tma_copy(
        typename Kernel_traits::GmemTiledCopydKV{},
        mdK,
        typename Kernel_traits::SmemLayoutdK{},
        select<1, 2>(TileShape_MNK{}),
        _1{}); // no mcast for output
    Tensor mdV = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.dv_ptr)),
                             make_shape(params.seqlen_k, params.d, params.h, params.b),
                             make_stride(params.dv_row_stride, _1{}, params.dv_head_stride, params.dv_batch_stride));
    auto tma_store_dV = make_tma_copy(
        typename Kernel_traits::GmemTiledCopydKV{},
        mdV,
        typename Kernel_traits::SmemLayoutdV{},
        select<1, 2>(TileShape_MNK{}),
        _1{}); // no mcast for output
    Tensor mdQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.dq_ptr)),
                             make_shape(params.seqlen_q, params.d, params.h, params.b),
                             make_stride(params.dq_row_stride, _1{}, params.dq_head_stride, params.dq_batch_stride));
    Tensor mdQaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum*>(params.dq_accum_ptr)),
                                  make_shape(params.seqlen_q, params.d, params.h, params.b),
                                  make_stride(params.d * params.h, _1{}, params.d, params.d * params.h * params.seqlen_q_rounded));
    auto tma_store_dQaccum = make_tma_copy(
        // typename Kernel_traits::GmemTiledCopydKV{},
        typename cute::SM90_TMA_STORE{},
        // mdQ,
        mdQaccum,
        // typename Kernel_traits::SmemLayoutdQTMA{},
        typename Kernel_traits::SmemLayoutdQaccTMA{},
        select<0, 2>(TileShape_MNK{}),
        _1{}); // no mcast for output
    auto tma_reduce_add_dQaccum = make_tma_copy(
        // typename Kernel_traits::GmemTiledCopydKV{},
        typename cute::SM90_TMA_REDUCE_ADD{},
        // mdQ,
        mdQaccum,
        // typename Kernel_traits::SmemLayoutdQTMA{},
        typename Kernel_traits::SmemLayoutdQaccTMA{},
        select<0, 2>(TileShape_MNK{}),
        _1{}); // no mcast for output
    // print(typename Kernel_traits::SmemLayoutVt{}); printf("\n"); print(typename Kernel_traits::SmemLayoutVt_tmp{});

    // print(typename Kernel_traits::TiledMmaSdP{}); printf("\n");
    // print(typename Kernel_traits::TiledMmadKV{}); printf("\n");
    // print(typename Kernel_traits::TiledMmadQ{}); printf("\n");
    // print(typename Kernel_traits::SmemLayoutAtomK{}); printf("\n");
    // print(typename Kernel_traits::SmemLayoutK{}); printf("\n");
    // print(typename Kernel_traits::SmemLayoutKt{}); printf("\n");
    // Get the ptr to kernel function.
    void *kernel;
    if constexpr (!Kernel_traits::Is_WS) {
       kernel = (void *)flash::compute_dqkv<Kernel_traits, Is_causal, decltype(tma_load_Q), decltype(tma_load_dO),
        decltype(tma_load_K), decltype(tma_load_V), decltype(tma_store_dK), decltype(tma_store_dV)>;
    } else {
       kernel = (void *)flash::compute_dqkv_ws<Kernel_traits, Is_causal, decltype(tma_load_Q), decltype(tma_load_dO),
        decltype(tma_load_K), decltype(tma_load_V), decltype(tma_store_dK), decltype(tma_store_dV), decltype(tma_store_dQaccum), decltype(tma_reduce_add_dQaccum)>;
    }
    // void *kernel = (void *)flash::compute_dqkv_seqqpar<Kernel_traits, Is_causal, decltype(tma_load_Q), decltype(tma_load_dO),
        // decltype(tma_load_K), decltype(tma_load_V), decltype(tma_store_dQaccum), decltype(tma_store_dK), decltype(tma_store_dV)>;
    auto shared_storage = typename Kernel_traits::SharedStorage{};
    int smem_size = sizeof(typename Kernel_traits::SharedStorage);
    int smem_size_q = sizeof(decltype(shared_storage.smem_q));
    int smem_size_do = sizeof(decltype(shared_storage.smem_do));
    int smem_size_k = sizeof(decltype(shared_storage.smem_k));
    int smem_size_v = sizeof(decltype(shared_storage.smem_v));
    // int smem_size_p = sizeof(decltype(shared_storage.smem_p));
    int smem_size_ds = sizeof(decltype(shared_storage.smem_ds));
    // printf("smem_size = %d, q = %d, do = %d, k = %d, v = %d, p = %d, ds = %d\n", smem_size, smem_size_q, smem_size_do, smem_size_k, smem_size_v, smem_size_p, smem_size_ds);
    // printf("smem_size = %d, q = %d, do = %d, k = %d, v = %d, ds = %d\n", smem_size, smem_size_q, smem_size_do, smem_size_k, smem_size_v, smem_size_ds);
    if (smem_size >= 48 * 1024) {
       CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }

    static constexpr int ctaSize = Kernel_traits::kNWarps * 32;
    int num_blocks_n = cutlass::ceil_div(params.seqlen_k, Kernel_traits::kBlockN);
    num_blocks_n = cutlass::ceil_div(num_blocks_n, size<1>(ClusterShape{})) * size<1>(ClusterShape{});
    dim3 grid_dims(num_blocks_n, params.h, params.b);
    // int num_blocks_m = cutlass::ceil_div(params.seqlen_q, Kernel_traits::kBlockM);
    // num_blocks_m = cutlass::ceil_div(num_blocks_m, size<0>(ClusterShape{})) * size<0>(ClusterShape{});
    // dim3 grid_dims(num_blocks_m, params.h, params.b);
    dim3 block_dims(ctaSize);
    dim3 cluster_dims(size<0>(ClusterShape{}), size<1>(ClusterShape{}), size<2>(ClusterShape{}));
    cutlass::ClusterLaunchParams launch_params{grid_dims, block_dims, cluster_dims, smem_size, stream};
    if constexpr (!Kernel_traits::Is_WS) {
        cutlass::launch_kernel_on_cluster(launch_params, kernel, params, tma_load_Q, tma_load_dO,
                                          tma_load_K, tma_load_V, tma_store_dK, tma_store_dV);
    } else {
        cutlass::launch_kernel_on_cluster(launch_params, kernel, params, tma_load_Q, tma_load_dO,
                                          tma_load_K, tma_load_V, tma_store_dK, tma_store_dV, tma_store_dQaccum, tma_reduce_add_dQaccum);
    }
    // cutlass::launch_kernel_on_cluster(launch_params, kernel, params, tma_load_Q, tma_load_dO,
                                      // tma_load_K, tma_load_V, tma_store_dQaccum, tma_store_dK, tma_store_dV);
    CHECK_CUDA_KERNEL_LAUNCH();

    auto tma_load_dQaccum = make_tma_copy(
        typename cute::SM90_TMA_LOAD{},
        mdQaccum,
        typename Kernel_traits::SmemLayoutdQaccTMA{},
        select<0, 2>(TileShape_MNK{}),
        _1{}); // no mcast for output
    // auto kernel_dq = &flash_bwd_convert_dq_kernel<Kernel_traits>;
    auto kernel_dq = &flash::convert_dQ<Kernel_traits, decltype(tma_load_dQaccum)>;
    if (Kernel_traits::kSmemdQSize * 2 + 8 >= 48 * 1024)  {
        CHECK_CUDA(cudaFuncSetAttribute(
            kernel_dq, cudaFuncAttributeMaxDynamicSharedMemorySize, Kernel_traits::kSmemdQSize * 2 + 8));
    }
    kernel_dq<<<grid_m, Kernel_traits::kNThreadsdQ, Kernel_traits::kSmemdQSize * 2 + 8, stream>>>(params, tma_load_dQaccum);
    CHECK_CUDA_KERNEL_LAUNCH();
    // auto kernel_dkv = &flash_bwd_convert_dkv_kernel<Kernel_traits>;
    // if (Kernel_traits::kSmemdKVSize >= 48 * 1024)  {
        // CHECK_CUDA(cudaFuncSetAttribute(
            // kernel_dkv, cudaFuncAttributeMaxDynamicSharedMemorySize, Kernel_traits::kSmemdKVSize));
    // }
    // int num_n_block = cute::ceil_div(params.seqlen_k, Kernel_traits::kBlockN);
    // dim3 grid_n(num_n_block, params.b, params.h);
    // kernel_dkv<<<grid_n, Kernel_traits::kNThreads, Kernel_traits::kSmemdKVSize, stream>>>(params);
    // CHECK_CUDA_KERNEL_LAUNCH();
}


template<typename T>
void run_mha_bwd_hdim64(Flash_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 64;
    // BOOL_SWITCH(params.is_causal, Is_causal, [&] {
    //     run_flash_bwd<T, Headdim, Is_causal>(params, stream);
    // });
    // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 128, 128, 8, false, false, false, 2, 2, 2, 1, T>, false>(params, stream);
    run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 128, 128, 12, true, false, false, 1, 2, 2, 1, T>, false>(params, stream);
    // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 96, 128, 12, true, false, true, 1, 2, 2, 1, T>, false>(params, stream);
}

template<typename T>
void run_mha_bwd_hdim128(Flash_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    // BOOL_SWITCH(params.is_causal, Is_causal, [&] {
    //     run_flash_bwd<T, Headdim, Is_causal>(params, stream);
    // });
    // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 128, 64, 8, false, 2, 1, 2, 1, T>, false>(params, stream);
    // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 128, 8, false, false, false, 1, 2, 1, 1, T>, false>(params, stream);
    // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 128, 96, 8, false, true, false, 2, 1, 2, 1, T>, false>(params, stream);
    // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 128, 96, 8, false, true, true, 2, 1, 1, 1, T>, false>(params, stream);
    // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 128, 8, true, false, true, 1, 2, 1, 1, T>, false>(params, stream);
    run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 128, 12, true, false, true, 1, 2, 1, 1, T>, false>(params, stream);
    // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 128, 12, true, false, false, 1, 2, 1, 1, T>, false>(params, stream);
    // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 128, 12, false, false, false, 1, 2, 1, 1, T>, false>(params, stream);
    // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 80, 128, 12, true, false, true, 1, 2, 1, 1, T>, false>(params, stream);
    // run_flash_bwd<Flash_bwd_seqqpar_kernel_traits<Headdim, 128, 64, 8, false, true, false, 2, 1, 2, 1, T>, false>(params, stream);
    // run_flash_bwd<Flash_bwd_seqqpar_kernel_traits<Headdim, 96, 128, 8, true, false, true, 1, 2, 1, 1, T>, false>(params, stream);
}

template<typename T>
void run_mha_bwd_hdim256(Flash_bwd_params &params, cudaStream_t stream) {
    // constexpr static int Headdim = 256;
    // BOOL_SWITCH(params.is_causal, Is_causal, [&] {
    //     run_flash_bwd<T, Headdim, Is_causal>(params, stream);
    // });
}
