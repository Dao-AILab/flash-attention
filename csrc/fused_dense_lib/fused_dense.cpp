// Adapted from https://github.com/NVIDIA/apex/blob/master/csrc/fused_dense.cpp
// We make it work for bfloat16
#include <torch/extension.h>
#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>

#include <stdio.h>

#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")

// https://github.com/NVIDIA/apex/blob/master/csrc/type_shim.h
// #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define DISPATCH_HALF_AND_BF16(TYPE, NAME, ...)                                \
  switch (TYPE) {                                                              \
  case at::ScalarType::Half: {                                                 \
    using scalar_t = at::Half;                                                 \
    __VA_ARGS__();                                                             \
    break;                                                                     \
  }                                                                            \
  case at::ScalarType::BFloat16: {                                             \
    using scalar_t = at::BFloat16;                                             \
    __VA_ARGS__();                                                             \
    break;                                                                     \
  }                                                                            \
  default:                                                                     \
    AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'");            \
  }

template <typename T>
int linear_bias_wgrad_cuda(const T *input, const T *d_output, int64_t in_features, int64_t batch_size, int64_t out_features, T *d_weight, T *d_bias, void *lt_workspace, size_t workspaceSize);

template <typename T>
int linear_act_forward_cuda(const T *input, const T *weight, const T *bias, int64_t in_features, int64_t batch_size, int64_t out_features, bool is_gelu, int heuristic, T *output, void *pre_act, void *lt_workspace, size_t workspaceSize);

template <typename T>
int bias_act_linear_dgrad_bgrad_cuda(const T *weight, const T *d_output, const void *pre_act, int64_t in_features, int64_t batch_size, int64_t out_features, bool is_gelu, int heuristic, T *d_input, T *d_bias, void *lt_workspace, size_t workspaceSize);

std::vector<at::Tensor> linear_bias_wgrad(at::Tensor input, at::Tensor d_output, bool has_d_bias) {

  int64_t batch_size = input.size(0);
  int64_t in_features = input.size(1);
  int64_t out_features = d_output.size(1);

  TORCH_CHECK(input.dtype() == torch::kFloat16 || input.dtype() == torch::kBFloat16);
  TORCH_CHECK(input.dtype() == d_output.dtype());
  TORCH_CHECK(input.is_cuda());
  TORCH_CHECK(d_output.is_cuda());
  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(d_output.is_contiguous());
  CHECK_SHAPE(input, batch_size, in_features);
  CHECK_SHAPE(d_output, batch_size, out_features);

  // Otherwise the kernel will be launched from cuda:0 device
  // Cast to char to avoid compiler warning about narrowing
  at::cuda::CUDAGuard device_guard{(char)input.get_device()};

  // create output/workspace tensor
  auto opts = input.options();
  auto d_weight = at::empty({out_features, in_features}, opts);
  at::Tensor d_bias;
  if (has_d_bias) {
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION < 11600
    d_bias = d_output.view({-1, out_features}).sum(0, false);
#else
    d_bias = at::empty({out_features}, opts);
#endif
  }
  // See https://github.com/pytorch/pytorch/issues/73328 for reasoning behind setting this to 1M.
  // However, Apex sets it to 4M and TransformerEngine sets to 32M for Hopper and 4M for other GPUs
  // https://github.com/NVIDIA/TransformerEngine/blob/a0f0065498bbcfc1da78cf9e8b166f5381613fbc/transformer_engine/pytorch/module.py#L91
  size_t workspaceSize = 1024 * 1024 * (at::cuda::getCurrentDeviceProperties()->major >= 9 ? 32 : 4);
  auto lt_workspace = at::empty({static_cast<int64_t>(workspaceSize)}, opts.dtype(torch::kUInt8));

  DISPATCH_HALF_AND_BF16(input.scalar_type(), "linear_bias_wgrad", [&] {
    auto result = linear_bias_wgrad_cuda<scalar_t>(
        input.data_ptr<scalar_t>(),
        d_output.data_ptr<scalar_t>(),
        in_features,
        batch_size,
        out_features,
        d_weight.data_ptr<scalar_t>(),
        has_d_bias ? d_bias.data_ptr<scalar_t>() : nullptr,
        (void*) (lt_workspace.data_ptr()),
        workspaceSize);
    TORCH_CHECK(result == 0, "linear_bias_wgrad failed.");
  });

  return {d_weight, d_bias};
}

std::vector<at::Tensor> linear_act_forward(at::Tensor input, at::Tensor weight,
                                           c10::optional<at::Tensor> bias_,
                                           bool is_gelu, bool save_pre_act, int heuristic) {

  int64_t batch_size = input.size(0);
  int64_t in_features = input.size(1);
  int64_t out_features = weight.size(0);

  TORCH_CHECK(input.dtype() == torch::kFloat16 || input.dtype() == torch::kBFloat16);
  TORCH_CHECK(input.dtype() == weight.dtype());
  TORCH_CHECK(input.is_cuda());
  TORCH_CHECK(weight.is_cuda());
  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(weight.is_contiguous());
  CHECK_SHAPE(input, batch_size, in_features);
  CHECK_SHAPE(weight, out_features, in_features);
  if (bias_.has_value()) {
    auto bias = bias_.value();
    TORCH_CHECK(bias.dtype() == input.dtype());
    TORCH_CHECK(bias.is_cuda());
    TORCH_CHECK(bias.is_contiguous());
    CHECK_SHAPE(bias, out_features);
  }

  // Otherwise the kernel will be launched from cuda:0 device
  // Cast to char to avoid compiler warning about narrowing
  at::cuda::CUDAGuard device_guard{(char)input.get_device()};

  // create output/workspace tensor
  auto opts = input.options();
  auto output = at::empty({batch_size, out_features}, opts);
  at::Tensor pre_act;
  // If ReLU, cuBlasLT stores a bit-mask (1 bit per element)
  if (save_pre_act) { pre_act = at::empty({batch_size, is_gelu ? out_features : out_features / 8},
                                          is_gelu ? opts : opts.dtype(torch::kUInt8)); }
  // See https://github.com/pytorch/pytorch/issues/73328 for reasoning behind setting this to 1M.
  // However, Apex sets it to 4M and TransformerEngine sets to 32M for Hopper and 4M for other GPUs
  // https://github.com/NVIDIA/TransformerEngine/blob/a0f0065498bbcfc1da78cf9e8b166f5381613fbc/transformer_engine/pytorch/module.py#L91
  size_t workspaceSize = 1024 * 1024 * (at::cuda::getCurrentDeviceProperties()->major >= 9 ? 32 : 4);
  auto lt_workspace = at::empty({static_cast<int64_t>(workspaceSize)}, opts.dtype(torch::kUInt8));

  DISPATCH_HALF_AND_BF16(input.scalar_type(), "linear_act_forward", [&] {
    auto result = linear_act_forward_cuda<scalar_t>(
        input.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        bias_.has_value()? bias_.value().data_ptr<scalar_t>() : nullptr,
        in_features,
        batch_size,
        out_features,
        is_gelu,
        heuristic,
        output.data_ptr<scalar_t>(),
        save_pre_act ? pre_act.data_ptr() : nullptr,
        (void*) (lt_workspace.data_ptr()),
        workspaceSize);
    TORCH_CHECK(result == 0, "linear_act_forward failed.");
  });

  std::vector<at::Tensor> result = {output};
  if (save_pre_act) { result.push_back(pre_act); };
  return result;
}

std::vector<at::Tensor> bias_act_linear_dgrad_bgrad(
  at::Tensor weight, at::Tensor d_output, at::Tensor pre_act, bool is_gelu, int heuristic
) {

  int64_t batch_size = d_output.size(0);
  int64_t out_features = d_output.size(1);
  int64_t in_features = weight.size(1);

  TORCH_CHECK(weight.dtype() == torch::kFloat16 || weight.dtype() == torch::kBFloat16);
  TORCH_CHECK(weight.dtype() == d_output.dtype());
  TORCH_CHECK(is_gelu ? (pre_act.dtype() == weight.dtype()) : (pre_act.dtype() == torch::kUInt8));
  TORCH_CHECK(weight.is_cuda());
  TORCH_CHECK(d_output.is_cuda());
  TORCH_CHECK(pre_act.is_cuda());
  TORCH_CHECK(weight.is_contiguous());
  TORCH_CHECK(d_output.is_contiguous());
  TORCH_CHECK(pre_act.is_contiguous());
  CHECK_SHAPE(weight, out_features, in_features);
  CHECK_SHAPE(d_output, batch_size, out_features);
  // If ReLU, cuBlasLT stores a bit-mask (1 bit per element)
  CHECK_SHAPE(pre_act, batch_size, is_gelu ? in_features : in_features / 8);

  // Otherwise the kernel will be launched from cuda:0 device
  // Cast to char to avoid compiler warning about narrowing
  at::cuda::CUDAGuard device_guard{(char)weight.get_device()};

  // create output/workspace tensor
  auto opts = weight.options();
  auto d_bias = at::empty({in_features}, opts);
  auto d_input = at::empty({batch_size, in_features}, opts);
  // See https://github.com/pytorch/pytorch/issues/73328 for reasoning behind setting this to 1M.
  // However, Apex sets it to 4M and TransformerEngine sets to 32M for Hopper and 4M for other GPUs
  // https://github.com/NVIDIA/TransformerEngine/blob/a0f0065498bbcfc1da78cf9e8b166f5381613fbc/transformer_engine/pytorch/module.py#L91
  size_t workspaceSize = 1024 * 1024 * (at::cuda::getCurrentDeviceProperties()->major >= 9 ? 32 : 4);
  auto lt_workspace = at::empty({static_cast<int64_t>(workspaceSize)}, opts.dtype(torch::kUInt8));

  DISPATCH_HALF_AND_BF16(weight.scalar_type(), "bias_act_linear_dgrad_bgrad", [&] {
    auto result = bias_act_linear_dgrad_bgrad_cuda<scalar_t>(
        weight.data_ptr<scalar_t>(),
        d_output.data_ptr<scalar_t>(),
        pre_act.data_ptr(),
        in_features,
        batch_size,
        out_features,
        is_gelu,
        heuristic,
        d_input.data_ptr<scalar_t>(),
        d_bias.data_ptr<scalar_t>(),
        (void*) (lt_workspace.data_ptr()),
        workspaceSize);
    TORCH_CHECK(result == 0, "bias_act_linear_dgrad_bgrad failed.");
  });

  return {d_input, d_bias};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("linear_bias_wgrad", &linear_bias_wgrad, "linear bias wgrad");
  m.def("linear_act_forward", &linear_act_forward, "linear gelu/relu forward");
  m.def("bias_act_linear_dgrad_bgrad", &bias_act_linear_dgrad_bgrad, "bias gelu/relu linear dgrad bgrad");
}
