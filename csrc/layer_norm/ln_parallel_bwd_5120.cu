#include "ln_parallel_residual_bwd_kernels.cuh"

// Create backward launch function and register. Macro signature:
//  HIDDEN_SIZE, WTYPE, ITYPE, RTYPE, OTYPE, CTYPE, CTAS_PER_ROW, WARPS_M, WARPS_N, BYTES_PER_LDG, BYTES_PER_LDG_FINAL

// Use 8 warps otherwise there's a lot of register spilling

REGISTER_PARALLEL_BWD_LAUNCHER( 5120, fp32, fp32, fp32, fp32, fp32, 1, 1, 8, 16, 4);
REGISTER_PARALLEL_BWD_LAUNCHER( 5120, fp16, fp32, fp32, fp32, fp32, 1, 1, 8, 16, 4);
REGISTER_PARALLEL_BWD_LAUNCHER( 5120, fp32, fp16, fp32, fp16, fp32, 1, 1, 8,  8, 4);
REGISTER_PARALLEL_BWD_LAUNCHER( 5120, fp16, fp16, fp32, fp16, fp32, 1, 1, 8,  8, 4);
REGISTER_PARALLEL_BWD_LAUNCHER( 5120, fp32, fp16, fp16, fp16, fp32, 1, 1, 8,  8, 4);
REGISTER_PARALLEL_BWD_LAUNCHER( 5120, fp32, bf16, fp32, bf16, fp32, 1, 1, 8,  8, 4);
REGISTER_PARALLEL_BWD_LAUNCHER( 5120, bf16, bf16, fp32, bf16, fp32, 1, 1, 8,  8, 4);
REGISTER_PARALLEL_BWD_LAUNCHER( 5120, fp32, bf16, bf16, bf16, fp32, 1, 1, 8,  8, 4);
REGISTER_PARALLEL_BWD_LAUNCHER( 5120, fp16, fp16, fp16, fp16, fp32, 1, 1, 8,  8, 4);
REGISTER_PARALLEL_BWD_LAUNCHER( 5120, bf16, bf16, bf16, bf16, fp32, 1, 1, 8,  8, 4);