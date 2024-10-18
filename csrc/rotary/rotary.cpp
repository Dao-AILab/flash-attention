/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

#define CHECK_DEVICE(x) TORCH_CHECK(x.device().type() == torch::kCUDA, #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")

void apply_rotary_cuda(const torch::Tensor x1, const torch::Tensor x2,
                       const torch::Tensor cos, const torch::Tensor sin,
                       torch::Tensor out1, torch::Tensor out2,
                       const bool conj);

void apply_rotary(const torch::Tensor x1, const torch::Tensor x2,
                  const torch::Tensor cos, const torch::Tensor sin,
                  torch::Tensor out1, torch::Tensor out2,
                  const bool conj) {
    CHECK_DEVICE(x1); CHECK_DEVICE(x2);
    CHECK_DEVICE(cos); CHECK_DEVICE(sin);
    CHECK_DEVICE(out1); CHECK_DEVICE(out1);
    TORCH_CHECK(x1.dtype() == x2.dtype());
    TORCH_CHECK(cos.dtype() == sin.dtype());
    TORCH_CHECK(out1.dtype() == out2.dtype());
    TORCH_CHECK(x1.dtype() == cos.dtype());
    TORCH_CHECK(x1.dtype() == out1.dtype());
    TORCH_CHECK(x1.sizes() == x2.sizes());
    TORCH_CHECK(cos.sizes() == sin.sizes());
    TORCH_CHECK(out1.sizes() == out2.sizes());

    // Otherwise the kernel will be launched from cuda:0 device
    at::cuda::CUDAGuard device_guard{x1.device()};

    apply_rotary_cuda(x1, x2, cos, sin, out1, out2, conj);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("apply_rotary", &apply_rotary, "Apply rotary embedding");
}
