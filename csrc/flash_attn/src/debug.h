#include <cute/util/debug.hpp>

#define KIN_PRINT(tag, statement) \
    if (cute::thread0()) { \
        printf("[kin:start:%s]\n", tag); \
        statement; \
        printf("\n[kin:end:%s]\n", tag); \
    }

template<typename Kernel_traits>
void
print_traits() {
    // bool
    printf("Kernel_traits::Share_Q_K_smem : %s\n", Kernel_traits::Share_Q_K_smem );
    printf("Kernel_traits::Is_Q_in_regs : %s\n", Kernel_traits::Is_Q_in_regs );

    // int
    printf("Kernel_traits::kNWarps : %s\n", Kernel_traits::kNWarps );
    printf("Kernel_traits::kNThreads : %s\n", Kernel_traits::kNThreads );
    printf("Kernel_traits::kBlockM : %s\n", Kernel_traits::kBlockM );
    printf("Kernel_traits::kBlockN : %s\n", Kernel_traits::kBlockN );
    printf("Kernel_traits::kHeadDim : %s\n", Kernel_traits::kHeadDim );
    printf("Kernel_traits::kBlockKSmem : %s\n", Kernel_traits::kBlockKSmem );
    printf("Kernel_traits::kBlockKGmem : %s\n", Kernel_traits::kBlockKGmem );
    printf("Kernel_traits::kSwizzle : %s\n", Kernel_traits::kSwizzle );
    printf("Kernel_traits::kSmemQSize : %s\n", Kernel_traits::kSmemQSize );
    printf("Kernel_traits::kSmemKVSize : %s\n", Kernel_traits::kSmemKVSize );
    printf("Kernel_traits::kSmemSize : %s\n", Kernel_traits::kSmemSize );
    printf("Kernel_traits::kGmemElemsPerLoad : %s\n", Kernel_traits::kGmemElemsPerLoad );

    // cute object
    printf("Kernel_traits::GmemLayoutAtom : "); print(Kernel_traits::GmemLayoutAtom); printf("\n");
    printf("Kernel_traits::GmemTiledCopyQKV : "); print(Kernel_traits::GmemTiledCopyQKV); printf("\n");
    printf("Kernel_traits::GmemTiledCopyO : "); print(Kernel_traits::GmemTiledCopyO); printf("\n");
    printf("Kernel_traits::SmemCopyAtom : "); print(Kernel_traits::SmemCopyAtom); printf("\n");
    printf("Kernel_traits::SmemCopyAtomTransposed : "); print(Kernel_traits::SmemCopyAtomTransposed); printf("\n");
    printf("Kernel_traits::MMA_Atom_Arch : "); print(Kernel_traits::MMA_Atom_Arch); printf("\n");
}

