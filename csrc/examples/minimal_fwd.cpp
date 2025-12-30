#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include <iostream>

// Include the local API header from the repo
#include "../flash_attn/flash_api.h" 

/**
 * Minimal Example: Calling FlashAttention-3 Forward via C++
 * This bypasses the Python layer for high-performance inference.
 */

void run_minimal_fa3_fwd() {
    // 1. Define dimensions [Batch, SeqLen, Heads, HeadDim]
    // Hopper (H100) works best with HeadDim 64, 128, or 256.
    int64_t b = 1; 
    int64_t s = 2048; 
    int64_t h = 16; 
    int64_t d = 128;

    // 2. Create tensors on CUDA using BFloat16 (Recommended for FA3)
    auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kBFloat16);
    at::Tensor q = torch::randn({b, s, h, d}, options);
    at::Tensor k = torch::randn({b, s, h, d}, options);
    at::Tensor v = torch::randn({b, s, h, d}, options);
    
    // 3. Prepare output and Softmax Log-Sum-Exp (LSE) metadata
    at::Tensor out = torch::empty_like(q);
    at::Tensor softmax_lse = torch::empty({b, h, s}, options.dtype(torch::kFloat));

    // 4. Set hyperparameters
    float softmax_scale = 1.0f / sqrt(static_cast<float>(d));
    bool is_causal = true;
    int window_size_left = -1; // -1 means no windowing (full attention)
    int window_size_right = -1;

    // 5. Call the official mha_fwd API
    // This function automatically selects the FA3 Hopper kernel if on SM90 hardware
    mha_fwd(
        q, k, v, out, 
        softmax_lse, 
        /*dropout_p=*/0.0, 
        softmax_scale, 
        is_causal, 
        window_size_left, 
        window_size_right, 
        /*alibi_slopes=*/at::Tensor(), 
        /*return_softmax=*/false
    );

    std::cout << "Successfully executed FlashAttention-3 forward pass in C++!" << std::endl;
}

int main() {
    try {
        run_minimal_fa3_fwd();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
