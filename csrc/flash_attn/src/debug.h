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

template<typename Kernel_traits>
__forceinline__ __device__ void
print_traits() {
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
    printf("Kernel_traits::kGmemRowsPerThread: %d\n", Kernel_traits::kGmemRowsPerThread );
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
