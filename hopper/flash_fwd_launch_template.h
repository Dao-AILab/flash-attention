/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/cluster_launch.hpp"

#include "static_switch.h"
#include "flash.h"
#include "tile_scheduler.hpp"
#include "flash_fwd_kernel.h"
#include "kernel_traits.h"
#include "seq_len.h"
#include "utils.h"

#include "combine.h"

template<typename Kernel_traits, bool Is_causal, bool Is_local, typename Seqlen_traits, typename Seqlen_traits_Q = Seqlen_traits>
void run_flash_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    static_assert(!(Is_causal && Is_local), "Is_causal and Is_local cannot be true at the same time.");
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using OutputType = typename Kernel_traits::OutputType;
    using TileShape_MNK = typename Kernel_traits::TileShape_MNK;
    using ClusterShape = typename Kernel_traits::ClusterShape_MNK;

    constexpr static bool Is_split = Kernel_traits::Is_split;

    static_assert(Seqlen_traits_Q::DecodingGQA == (Kernel_traits::kBlockH > 1));

    static_assert(!(Is_split && Seqlen_traits::UseVarSeqLen), "Split KV not supported for varseqlen.");

    // print(typename Kernel_traits::SmemLayoutVt{}); printf("\n"); print(typename Kernel_traits::SmemLayoutVt_tmp{});
    using CollectiveMainloop = flash::CollectiveMainloopFwd<Kernel_traits, Is_causal, Is_local, Seqlen_traits, Seqlen_traits_Q>;
    using CollectiveEpilogue = flash::CollectiveEpilogueFwd<Kernel_traits, Seqlen_traits_Q>;
    using Scheduler = std::conditional_t<
        Seqlen_traits::UseVarSeqLen || Is_local, 
        flash::SingleTileScheduler,
        std::conditional_t<!Is_causal && !Is_split,
            flash::StaticPersistentTileScheduler<Is_split>,
            flash::DynamicPersistentTileScheduler<
                Kernel_traits::kNThreads - cutlass::NumThreadsPerWarpGroup,
                Kernel_traits::NumProducerThreads,
                Is_split
            >
    >>;
    // using Scheduler = flash::SingleTileScheduler;
    Seqlen_traits_Q seqlen_traits_q(
        params.total_q, params.seqlen_q, params.cu_seqlens_q, params.seqused_q);
    Seqlen_traits seqlen_traits_k(
        params.total_k, params.seqlen_k, params.cu_seqlens_k, params.seqused_k);

    // print("Q layout: ");
    // print(seqlen_traits_q.get_gmem_layout(
    //             params.seqlen_q, params.d, params.h_k, params.b, params.h_h_k_ratio, 
    //             params.q_row_stride, params.q_head_stride, params.q_batch_stride));
    // print("\n");
    // print("Q smem layout: ");
    // using SmemLayoutQCopy = typename Kernel_traits::SmemLayoutQCopy;
    // print(SmemLayoutQCopy{});
    // print("\n");
    typename CollectiveMainloop::Params mainloop_params =
        CollectiveMainloop::to_underlying_arguments({
            static_cast<Element const*>(params.q_ptr),            
            seqlen_traits_q.get_gmem_layout(
                params.seqlen_q, params.d, params.h_k, params.b, params.h_h_k_ratio, 
                params.q_row_stride, params.q_head_stride, params.q_batch_stride
            ),  // layout_Q
            static_cast<Element const*>(params.k_ptr),
            seqlen_traits_k.get_gmem_layout(
                params.seqlen_k, params.d, params.h_k, params.b_k, 
                params.k_row_stride, params.k_head_stride, params.k_batch_stride
            ),  // layout_K
            static_cast<Element const*>(params.v_ptr),
            seqlen_traits_k.get_gmem_layout(
                params.seqlen_k, params.d, params.h_k, params.b_k, 
                params.v_row_stride, params.v_head_stride, params.v_batch_stride
            ),  // layout_V
            params.scale_softmax_log2,
            params.descale_q_ptr,
            params.descale_k_ptr,
            params.descale_v_ptr,
            params.window_size_left,
            params.window_size_right,
            ceil_div(params.h_h_k_ratio, Kernel_traits::kBlockH),
            params.cache_batch_idx,
            Is_split ? params.num_splits : 1
        });
    typename CollectiveEpilogue::Params epilogue_params = [&] {
        if constexpr(!Is_split) {
            return CollectiveEpilogue::to_underlying_arguments({            
                static_cast<OutputType*>(params.o_ptr),
                seqlen_traits_q.get_gmem_layout(
                    params.seqlen_q, params.d, params.h_k, params.b, params.h_h_k_ratio, 
                    params.o_row_stride, params.o_head_stride, params.o_batch_stride
                ),  // layout_O
                static_cast<float*>(params.softmax_lse_ptr),            
                seqlen_traits_q.get_lse_gmem_layout(
                    params.seqlen_q, params.h, params.b
                )  // layout_LSE
            });
        } else {
            return CollectiveEpilogue::to_underlying_arguments({
                static_cast<OutputType*>(params.oaccum_ptr), 
                seqlen_traits_q.get_oaccum_gmem_layout(
                    params.seqlen_q, params.d, params.h_k, params.b, params.h_h_k_ratio, params.num_splits,
                    params.oaccum_row_stride, params.oaccum_head_stride, params.oaccum_batch_stride,  
                    params.oaccum_split_stride
                ), // layout_O
                static_cast<float*>(params.softmax_lseaccum_ptr),            
                seqlen_traits_q.get_lseaccum_gmem_layout(
                    params.seqlen_q, params.h, params.b, params.num_splits
                ), // layout_LSE
            });
        }
    }();

    // int num_blocks_m = cutlass::ceil_div(params.seqlen_q, Kernel_traits::kBlockM);
    int num_blocks_m = cutlass::ceil_div(params.seqlen_q, Kernel_traits::kBlockM/Kernel_traits::kBlockH);
    num_blocks_m = cutlass::ceil_div(num_blocks_m, size<0>(ClusterShape{})) * size<0>(ClusterShape{});    
    int num_grid_heads = params.h_k * ceil_div(params.h_h_k_ratio, Kernel_traits::kBlockH);

    // std::cout << "num blocks m = " << num_blocks_m << " num grid heads" << num_grid_heads << std::endl;
    typename Scheduler::Arguments scheduler_args =
        {num_blocks_m, Is_split ? params.num_splits : 1, num_grid_heads, params.b, params.tile_count_semaphore};
    typename Scheduler::Params scheduler_params = Scheduler::to_underlying_arguments(scheduler_args);    

    // Get the ptr to kernel function.
    void *kernel;
    if constexpr(cutlass::sizeof_bits_v<Element> == 8)
        kernel = (void *)flash::compute_attn_ws_fp8<Kernel_traits, Is_causal, Scheduler, Seqlen_traits, Seqlen_traits_Q>;
    else
        kernel = (void *)flash::compute_attn_ws<Kernel_traits, Is_causal, Is_local, Scheduler, Seqlen_traits, Seqlen_traits_Q>;
    int smem_size = sizeof(typename Kernel_traits::SharedStorage);
    // int smem_size_q = sizeof(decltype((typename Kernel_traits::SharedStorage{}).smem_q));
    // int smem_size_k = sizeof(decltype((typename Kernel_traits::SharedStorage{}).smem_k));
    // int smem_size_v = sizeof(decltype((typename Kernel_traits::SharedStorage{}).smem_v));
    // int smem_size_o = sizeof(decltype((typename Kernel_traits::SharedStorage{}).smem_o));
    // printf("smem_size = %d, q = %d, k = %d, v = %d, o = %d.\n", smem_size, smem_size_q, smem_size_k, smem_size_v, smem_size_o);
    if (smem_size >= 48 * 1024) {
       CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }

    int device;
    cudaGetDevice(&device);
    int multiprocessor_count;
    CHECK_CUDA(cudaDeviceGetAttribute(&multiprocessor_count, cudaDevAttrMultiProcessorCount, device));
    dim3 grid_dims = Scheduler::get_grid_dim(scheduler_args, multiprocessor_count);
    static constexpr int ctaSize = Kernel_traits::kNWarps * 32;
    dim3 block_dims(ctaSize);
    if constexpr(size(ClusterShape{}) > 1) {
        dim3 cluster_dims(size<0>(ClusterShape{}), size<1>(ClusterShape{}), size<2>(ClusterShape{}));
        cutlass::ClusterLaunchParams launch_params{grid_dims, block_dims, cluster_dims, smem_size, stream};
        cutlass::launch_kernel_on_cluster(
            launch_params, kernel, mainloop_params, epilogue_params, 
            scheduler_params, seqlen_traits_q, seqlen_traits_k);
    } else {
        if constexpr(cutlass::sizeof_bits_v<Element> == 8) {
            flash::compute_attn_ws_fp8<Kernel_traits, Is_causal, Scheduler, Seqlen_traits, Seqlen_traits_Q>
                <<<grid_dims, block_dims, smem_size, stream>>>
                (mainloop_params, epilogue_params, scheduler_params, seqlen_traits_q, seqlen_traits_k);
        } else {
            flash::compute_attn_ws<Kernel_traits, Is_causal, Scheduler, Seqlen_traits, Seqlen_traits_Q>
                <<<grid_dims, block_dims, smem_size, stream>>>
                (mainloop_params, epilogue_params, scheduler_params, seqlen_traits_q, seqlen_traits_k);
        }

    }
    CHECK_CUDA_KERNEL_LAUNCH();

    if constexpr (Is_split) {
      using FinalOutputType = Element;
      static_assert(is_same_v<OutputType, float>, "Assume OutputType of main kernel is float.");
      static_assert(is_same_v<ElementAccum, float>, "ElementAccum must be float.");
      // We want kBlockM to be as small as possible for more parallelism.
      // With 128 threads we can load 512 elements at a time, so if headdim is divisible by 128, kBlockM = 4.
      // If headdim is divisible by 64, then we set kBlockM = 8, etc.
      constexpr static int kHeadDim = Kernel_traits::kHeadDim;
      constexpr static int kBlockM = kHeadDim % 128 == 0 ? 4 : (kHeadDim % 64 == 0 ? 8 : 16);
      constexpr static bool Is_even_K = true; // always true for our current setting
      void *kernel_combine;
      int smem_size_combine;
      if (params.num_splits <= 2) {
        kernel_combine = (void *) flash::combine_attn_seqk_parallel<FinalOutputType, ElementAccum, kHeadDim, kBlockM, 1, Is_even_K, Flash_fwd_params>;
        smem_size_combine = sizeof(flash::SharedStorageLSE<float, Shape<Int<2>, Int<kBlockM+1>>, Shape<Int<2>>>);
      } else if (params.num_splits <= 4) {
        kernel_combine = (void *) flash::combine_attn_seqk_parallel<FinalOutputType, ElementAccum, kHeadDim, kBlockM, 2, Is_even_K, Flash_fwd_params>;
        smem_size_combine = sizeof(flash::SharedStorageLSE<float, Shape<Int<4>, Int<kBlockM+1>>, Shape<Int<4>>>);
      } else if (params.num_splits <= 8) {
        kernel_combine = (void *) flash::combine_attn_seqk_parallel<FinalOutputType, ElementAccum, kHeadDim, kBlockM, 3, Is_even_K, Flash_fwd_params>;
        smem_size_combine = sizeof(flash::SharedStorageLSE<float, Shape<Int<8>, Int<kBlockM+1>>, Shape<Int<8>>>);
      } else if (params.num_splits <= 16) {
        kernel_combine = (void *) flash::combine_attn_seqk_parallel<FinalOutputType, ElementAccum, kHeadDim, kBlockM, 4, Is_even_K, Flash_fwd_params>;
        smem_size_combine = sizeof(flash::SharedStorageLSE<float, Shape<Int<16>, Int<kBlockM+1>>, Shape<Int<16>>>);
      } else if (params.num_splits <= 32) {
        kernel_combine = (void *) flash::combine_attn_seqk_parallel<FinalOutputType, ElementAccum, kHeadDim, kBlockM, 5, Is_even_K, Flash_fwd_params>;
        smem_size_combine = sizeof(flash::SharedStorageLSE<float, Shape<Int<32>, Int<kBlockM+1>>, Shape<Int<32>>>);
      } else if (params.num_splits <= 64) {
        kernel_combine = (void *) flash::combine_attn_seqk_parallel<FinalOutputType, ElementAccum, kHeadDim, kBlockM, 6, Is_even_K, Flash_fwd_params>;
        smem_size_combine = sizeof(flash::SharedStorageLSE<float, Shape<Int<64>, Int<kBlockM+1>>, Shape<Int<64>>>);
      } else if (params.num_splits <= 128) {
        kernel_combine = (void *) flash::combine_attn_seqk_parallel<FinalOutputType, ElementAccum, kHeadDim, kBlockM, 7, Is_even_K, Flash_fwd_params>;
        smem_size_combine = sizeof(flash::SharedStorageLSE<float, Shape<Int<128>, Int<kBlockM+1>>, Shape<Int<128>>>);
      } else {
        // don't support > 128 splits
        return;
      }
      if (smem_size_combine >= 48 * 1024) {
        CHECK_CUDA(cudaFuncSetAttribute(kernel_combine, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_combine));
      }
      dim3 grid_combine((params.b * params.h * params.seqlen_q + kBlockM - 1) / kBlockM);
      dim3 block_dims_combine(128);
      dim3 cluster_dims_combine(1, 1, 1);
      cutlass::ClusterLaunchParams launch_params_combine{
          grid_combine, block_dims_combine, cluster_dims_combine, smem_size_combine, stream};
      cutlass::launch_kernel_on_cluster(launch_params_combine, kernel_combine, params);
      CHECK_CUDA_KERNEL_LAUNCH();
    }
}

template<typename T>
void run_mha_fwd_hdim64(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 64;
    // constexpr static bool UseCluster = false;

    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
      MMA_3WG_SWITCH(params.seqlen_q, kNumMmaWGs, [&] {
        SEQLEN_SWITCH(params.cu_seqlens_q, Seqlen_traits, [&] {
          BOOL_SWITCH(params.num_splits > 1, Is_split, [&] {
            run_flash_fwd<
                Flash_fwd_kernel_traits<Headdim, kNumMmaWGs * 64, 128, 4 + kNumMmaWGs * 4,
                    2, false, 1, T, !Seqlen_traits::UseVarSeqLen && Is_split>,
                Is_causal, Seqlen_traits
            >(params, stream);
          });
        });
      });
    });
}

template<typename T>
void run_mha_fwd_hdim64_gqa_decoding(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 64;
    // constexpr static bool UseCluster = false;
    using Seqlen_traits = flash::FixedSeqLenTraits;
    using Seqlen_traits_Q = flash::DecodingGQASeqLenTraits;

    QUERYHEAD_SWITCH(params.h_h_k_ratio, kBlockH, [&] {
      MMA_3WG_SWITCH(kBlockH * params.seqlen_q, kNumMmaWGs, [&] {
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
          BOOL_SWITCH(params.num_splits > 1, Is_split, [&] {
            run_flash_fwd<
                Flash_fwd_kernel_traits<Headdim, kNumMmaWGs * 64, 128, 4 + kNumMmaWGs * 4,
                    2, false, 1, T, !Seqlen_traits::UseVarSeqLen && Is_split, kBlockH>,
                Is_causal, Seqlen_traits, Seqlen_traits_Q
            >(params, stream);
          });
        });
      });
    });
}

template<typename T>
void run_mha_fwd_hdim128(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    
    MMA_2WG_SWITCH(params.seqlen_q, kNumMmaWGs, [&] {
      BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        SEQLEN_SWITCH(params.cu_seqlens_q, Seqlen_traits, [&] {
          BOOL_SWITCH(params.num_splits > 1, Is_split, [&] {
            // Only use Cluster if number of tiles along seqlen_q is even
            // and not Is_causal, Is_split, or varseqlen
            BOOL_SWITCH(cutlass::ceil_div(params.seqlen_q, 128) % 2 == 0 && !Is_causal
                        && !Seqlen_traits::UseVarSeqLen && !Is_split, UseCluster, [&] {
                run_flash_fwd<
                    Flash_fwd_kernel_traits<Headdim, kNumMmaWGs * 64, Is_causal ? 128 : 176,
                        4 + kNumMmaWGs * 4, 2, false, UseCluster ? 2 : 1, 
                        T, !Seqlen_traits::UseVarSeqLen && Is_split>, 
                    Is_causal, Seqlen_traits
                >(params, stream);
            });
          });
        });
      });
    });
}

template<typename T>
void run_mha_fwd_hdim128_gqa_decoding(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    // constexpr static bool UseCluster = false;
    using Seqlen_traits = flash::FixedSeqLenTraits;
    using Seqlen_traits_Q = flash::DecodingGQASeqLenTraits;

    QUERYHEAD_SWITCH(params.h_h_k_ratio, kBlockH, [&] {
      MMA_2WG_SWITCH(kBlockH * params.seqlen_q, kNumMmaWGs, [&] {
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
          BOOL_SWITCH(params.num_splits > 1, Is_split, [&] {
            run_flash_fwd<
                Flash_fwd_kernel_traits<Headdim, kNumMmaWGs * 64, Is_causal ? 128 : 176,
                    4 + kNumMmaWGs * 4, 2, false, 1, T, Is_split, kBlockH>, 
                Is_causal, Seqlen_traits, Seqlen_traits_Q
            >(params, stream);
          });
        });
      });
    });
}

template<typename T>
void run_mha_fwd_hdim256(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 256;

    MMA_2WG_SWITCH(params.seqlen_q, kNumMmaWGs, [&] {
      BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        SEQLEN_SWITCH(params.cu_seqlens_q, Seqlen_traits, [&] {
          BOOL_SWITCH(params.num_splits > 1, Is_split, [&] {
            // Only use Cluster if number of tiles along seqlen_q is even
            // and not Is_causal, Is_split, or varseqlen
            BOOL_SWITCH(cutlass::ceil_div(params.seqlen_q, 128) % 2 == 0 && !Is_causal
                        && !Seqlen_traits::UseVarSeqLen && !Is_split, UseCluster, [&] {
                run_flash_fwd<
                    Flash_fwd_kernel_traits<Headdim, kNumMmaWGs * 64, kNumMmaWGs == 1 ? 96 : 80,
                        4 + kNumMmaWGs * 4, 2, false, UseCluster ? 2 : 1,
                        T, !Seqlen_traits::UseVarSeqLen && Is_split>, 
                    Is_causal, Seqlen_traits
                >(params, stream);
            });
          });
        });
      });
    });
}

template<typename T>
void run_mha_fwd_hdim256_gqa_decoding(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 256;
    // constexpr static bool UseCluster = false;
    using Seqlen_traits = flash::FixedSeqLenTraits;
    using Seqlen_traits_Q = flash::DecodingGQASeqLenTraits;

    QUERYHEAD_SWITCH(params.h_h_k_ratio, kBlockH, [&] {
      MMA_2WG_SWITCH(kBlockH * params.seqlen_q, kNumMmaWGs, [&] {
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
          BOOL_SWITCH(params.num_splits > 1, Is_split, [&] {
            run_flash_fwd<
                Flash_fwd_kernel_traits<Headdim, kNumMmaWGs * 64, kNumMmaWGs == 1 ? 96 : 80,
                    4 + kNumMmaWGs * 4, 2, false, 1, T, Is_split, kBlockH>, 
                Is_causal, Seqlen_traits, Seqlen_traits_Q
            >(params, stream);
          });
        });
      });
    });
}

template<typename T>
void run_mha_fwd_hdim64_fp8(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 64;
    constexpr static int kBlockM = 192;
    constexpr static int kBlockN = 128;
    constexpr static int kNWarps = 4 + kBlockM/16;
    constexpr static int kStages = 4;    
    using Seqlen_traits = flash::FixedSeqLenTraits;
    if(params.is_causal) {
        run_flash_fwd<Flash_fwd_kernel_traits_fp8<Headdim, kBlockM, kBlockN, kNWarps, kStages,
                        false, 1, T>, /*Is_causal=*/true, /*Is_local=*/false, Seqlen_traits>(params, stream);
    } else {
        BOOL_SWITCH(cutlass::ceil_div(params.seqlen_q, kBlockM) % 2 == 0, UseCluster, [&] {
            run_flash_fwd<Flash_fwd_kernel_traits_fp8<Headdim, kBlockM, kBlockN, kNWarps, kStages,
                            false, UseCluster ? 2 : 1, T>, /*Is_causal=*/false, /*Is_local=*/false, Seqlen_traits>(params, stream);
        });
    }
    // BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        // SEQLEN_SWITCH(params.cu_seqlens_q, Seqlen_traits, [&] {
            // Only use Cluster if number of tiles along seqlen_q is even
            // BOOL_SWITCH(cutlass::ceil_div(params.seqlen_q, kBlockM) % 2 == 0 && !Is_causal &&
            //             !Seqlen_traits::kUseVarSeqLen, UseCluster, [&] {
            //     run_flash_fwd<Flash_fwd_kernel_traits_fp8<Headdim, kBlockM, kBlockN, kNWarps, kStages,
            //                   false, UseCluster ? 2 : 1, T>, Is_causal, Seqlen_traits>(params, stream);            
            // });
        // });
    // });
}

template<typename T>
void run_mha_fwd_hdim128_fp8(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    constexpr static int kBlockM = 128;
    constexpr static int kBlockN = 256;
    constexpr static int kNWarps = 4 + kBlockM/16;
    constexpr static int kStages = 2;
    using Seqlen_traits = flash::FixedSeqLenTraits;
    if(params.is_causal) {
        run_flash_fwd<Flash_fwd_kernel_traits_fp8<Headdim, kBlockM, kBlockN, kNWarps, kStages,
                        false, 1, T>, /*Is_causal=*/true, /*Is_local=*/false, Seqlen_traits>(params, stream);
    } else {
        BOOL_SWITCH(cutlass::ceil_div(params.seqlen_q, kBlockM) % 2 == 0, UseCluster, [&] {
            run_flash_fwd<Flash_fwd_kernel_traits_fp8<Headdim, kBlockM, kBlockN, kNWarps, kStages,
                            false, UseCluster ? 2 : 1, T>, /*Is_causal=*/false, /*Is_local=*/false, Seqlen_traits>(params, stream);
        });
    }
    // BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        // SEQLEN_SWITCH(params.cu_seqlens_q, Seqlen_traits, [&] {
            // Only use Cluster if number of tiles along seqlen_q is even
            // BOOL_SWITCH(cutlass::ceil_div(params.seqlen_q, kBlockM) % 2 == 0 && !Is_causal &&
            //             !Seqlen_traits::kUseVarSeqLen, UseCluster, [&] {
            //     run_flash_fwd<Flash_fwd_kernel_traits_fp8<Headdim, kBlockM, kBlockN, kNWarps, kStages,
            //                   false, UseCluster ? 2 : 1, T>, Is_causal, Seqlen_traits>(params, stream);
            // });
        // });
    // });
}

template<typename T>
void run_mha_fwd_hdim256_fp8(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 256; 
    constexpr static int kBlockM = 128;
    constexpr static int kBlockN = 128;
    constexpr static int kNWarps = 4 + kBlockM/16;
    constexpr static int kStages = 2;
    using Seqlen_traits = flash::FixedSeqLenTraits;
    if(params.is_causal) {
        run_flash_fwd<Flash_fwd_kernel_traits_fp8<Headdim, kBlockM, kBlockN, kNWarps, kStages,
                        false, 1, T>, /*Is_causal=*/true, /*Is_local=*/false, Seqlen_traits>(params, stream);
    } else {
        BOOL_SWITCH(cutlass::ceil_div(params.seqlen_q, kBlockM) % 2 == 0, UseCluster, [&] {
            run_flash_fwd<Flash_fwd_kernel_traits_fp8<Headdim, kBlockM, kBlockN, kNWarps, kStages,
                            false, UseCluster ? 2 : 1, T>, /*Is_causal=*/false, /*Is_local=*/false, Seqlen_traits>(params, stream);
        });
    }
    // BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        // SEQLEN_SWITCH(params.cu_seqlens_q, Seqlen_traits, [&] {
            // Only use Cluster if number of tiles along seqlen_q is even
            // BOOL_SWITCH(cutlass::ceil_div(params.seqlen_q, kBlockM) % 2 == 0 && !Is_causal &&
            //             !Seqlen_traits::kUseVarSeqLen, UseCluster, [&] {
            //     run_flash_fwd<Flash_fwd_kernel_traits_fp8<Headdim, kBlockM, kBlockN, kNWarps, kStages,
            //                   false, UseCluster ? 2 : 1, T>, Is_causal, Seqlen_traits>(params, stream);
            // });
        // });
    // });
}
