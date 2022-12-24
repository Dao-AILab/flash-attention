// Adapted from https://github.com/NVIDIA/apex/blob/master/csrc/fused_dense.cpp
// We make it work for bfloat16
#include <torch/extension.h>
#include <torch/torch.h>
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
int linear_bias_wgrad_cuda(T *input, T *d_output, int in_features, int batch_size, int out_features, T *d_weight, T *d_bias, void *lt_workspace);

template <typename T>
int linear_gelu_forward_cuda(T *input, T *weight, T *bias, int in_features, int batch_size, int out_features, int heuristic, T *output, T *gelu_in, void *lt_workspace) ;

template <typename T>
int bias_gelu_linear_dgrad_bgrad_cuda(T *weight, T *d_output, T *gelu_in, int in_features, int batch_size, int out_features, int heuristic, T *d_input, T *d_bias, void *lt_workspace);

std::vector<at::Tensor> linear_bias_wgrad(at::Tensor input, at::Tensor d_output, bool has_d_bias) {

  int batch_size = input.size(0);
  int in_features = input.size(1);
  int out_features = d_output.size(1);

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
  // allocate fixed 4MB workspace for cublaslt for now, and this gets at least 4 MB
  auto lt_workspace = at::empty({1 << 22}, opts);

  DISPATCH_HALF_AND_BF16(input.scalar_type(), "linear_bias_wgrad", [&] {
    auto result = linear_bias_wgrad_cuda<scalar_t>(
        input.data_ptr<scalar_t>(),
        d_output.data_ptr<scalar_t>(),
        in_features,
        batch_size,
        out_features,
        d_weight.data_ptr<scalar_t>(),
        has_d_bias ? d_bias.data_ptr<scalar_t>() : nullptr,
        (void*) (lt_workspace.data_ptr<scalar_t>()));
    TORCH_CHECK(result == 0, "linear_bias_wgrad failed.");
  });

  return {d_weight, d_bias};
}

std::vector<at::Tensor> linear_gelu_forward(at::Tensor input, at::Tensor weight,
                                            c10::optional<at::Tensor> bias_,
                                            bool save_gelu_in, int heuristic) {

  int batch_size = input.size(0);
  int in_features = input.size(1);
  int out_features = weight.size(0);

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
  at::Tensor gelu_in;
  if (save_gelu_in) { gelu_in = at::empty({batch_size, out_features}, opts); }
  // allocate fixed 4MB workspace for cublaslt for now, and this gets at least 4 MB
  auto lt_workspace = at::empty({1 << 22}, opts);

  DISPATCH_HALF_AND_BF16(input.scalar_type(), "linear_gelu_forward", [&] {
    auto result = linear_gelu_forward_cuda<scalar_t>(
        input.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        bias_.has_value()? bias_.value().data_ptr<scalar_t>() : nullptr,
        in_features,
        batch_size,
        out_features,
        heuristic,
        output.data_ptr<scalar_t>(),
        save_gelu_in ? gelu_in.data_ptr<scalar_t>() : nullptr,
        (void*) (lt_workspace.data_ptr<scalar_t>()));
    TORCH_CHECK(result == 0, "linear_gelu_forward failed.");
  });

  std::vector<at::Tensor> result = {output};
  if (save_gelu_in) { result.push_back(gelu_in); };
  return result;
}

std::vector<at::Tensor> bias_gelu_linear_dgrad_bgrad(
  at::Tensor weight, at::Tensor d_output, at::Tensor gelu_in, int heuristic
) {

  int batch_size = d_output.size(0);
  int out_features = d_output.size(1);
  int in_features = weight.size(1);

  TORCH_CHECK(weight.dtype() == torch::kFloat16 || weight.dtype() == torch::kBFloat16);
  TORCH_CHECK(weight.dtype() == d_output.dtype());
  TORCH_CHECK(weight.dtype() == gelu_in.dtype());
  TORCH_CHECK(weight.is_cuda());
  TORCH_CHECK(d_output.is_cuda());
  TORCH_CHECK(gelu_in.is_cuda());
  TORCH_CHECK(weight.is_contiguous());
  TORCH_CHECK(d_output.is_contiguous());
  TORCH_CHECK(gelu_in.is_contiguous());
  CHECK_SHAPE(weight, out_features, in_features);
  CHECK_SHAPE(d_output, batch_size, out_features);
  CHECK_SHAPE(gelu_in, batch_size, in_features);

  // Otherwise the kernel will be launched from cuda:0 device
  // Cast to char to avoid compiler warning about narrowing
  at::cuda::CUDAGuard device_guard{(char)weight.get_device()};

  // create output/workspace tensor
  auto opts = weight.options();
  auto d_bias = at::empty({in_features}, opts);
  auto d_input = at::empty({batch_size, in_features}, opts);
  // allocate fixed 4MB workspace for cublaslt for now, and this gets at least 4 MB
  auto lt_workspace = at::empty({1 << 22}, opts);

  DISPATCH_HALF_AND_BF16(weight.scalar_type(), "bias_gelu_linear_dgrad_bgrad", [&] {
    auto result = bias_gelu_linear_dgrad_bgrad_cuda<scalar_t>(
        weight.data_ptr<scalar_t>(),
        d_output.data_ptr<scalar_t>(),
        gelu_in.data_ptr<scalar_t>(),
        in_features,
        batch_size,
        out_features,
        heuristic,
        d_input.data_ptr<scalar_t>(),
        d_bias.data_ptr<scalar_t>(),
        (void*) (lt_workspace.data_ptr<scalar_t>()));
    TORCH_CHECK(result == 0, "bias_gelu_linear_dgrad_bgrad failed.");
  });

  return {d_input, d_bias};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("linear_bias_wgrad", &linear_bias_wgrad, "linear bias wgrad");
  m.def("linear_gelu_forward", &linear_gelu_forward, "linear gelu forward");
  m.def("bias_gelu_linear_dgrad_bgrad", &bias_gelu_linear_dgrad_bgrad, "bias gelu linear dgrad bgrad");
}
