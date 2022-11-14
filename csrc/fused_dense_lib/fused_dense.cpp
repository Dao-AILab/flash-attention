// Adapted from https://github.com/NVIDIA/apex/blob/master/csrc/fused_dense.cpp
// We make it work for bfloat16
#include <torch/extension.h>
#include <torch/torch.h>
#include <vector>

#include <stdio.h>

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

#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")

template <typename T>
int linear_bias_forward_cuda(at::Tensor input, T *weight, at::Tensor bias, int in_features, int batch_size, int out_features, at::Tensor output, void *lt_workspace);

template <typename T>
int linear_bias_backward_cuda(T *input, T *weight, T *d_output, int in_features, int batch_size, int out_features, T *d_weight, T *d_bias, T *d_input,  bool residual, void *lt_workspace);

template <typename T>
int linear_bias_wgrad_cuda(T *input, T *d_output, int in_features, int batch_size, int out_features, T *d_weight, T *d_bias, void *lt_workspace);

template <typename T>
int linear_gelu_forward_cuda(T *input, T *weight, T *bias, int in_features, int batch_size, int out_features, int heuristic, T *output, T *gelu_in, void *lt_workspace) ;

template <typename T>
int linear_gelu_linear_backward_cuda(T *input, T *gelu_in, T *output1, T *weight1, T *weight2, T *d_output1, T *d_output2, int in_features, int batch_size, int hidden_features, int out_features, int heuristic, T *d_weight1, T *d_weight2, T *d_bias1, T *d_bias2, T *d_input, bool residual, void *lt_workspace);

at::Tensor linear_bias_forward(at::Tensor input, at::Tensor weight, at::Tensor bias) {

  auto batch_size = input.size(0);
  auto in_features = input.size(1);

  int out_features = weight.size(0);

  //auto reserved_size = get_mlp_reserved_space(batch_size, num_layers, output_features.data());

  // create output/workspace tensor
  auto out = at::empty({batch_size, out_features}, at::dtype(input.dtype()).device(input.device()));
  //auto reserved_space = at::empty({reserved_size}, inputs[0].type());
  // allocate fixed 4MB workspace for cublaslt for now, and this gets at least 4 MB
  auto lt_workspace = at::empty({1 << 22}, at::dtype(input.dtype()).device(input.device()));

  DISPATCH_HALF_AND_BF16(input.scalar_type(), "linear_bias_forward", [&] {
    scalar_t* w_ptr = weight.data_ptr<scalar_t>();
    auto result = linear_bias_forward_cuda<scalar_t>(
        input,
        w_ptr,
        bias,
        in_features,
        batch_size,
        out_features,
        out,
        //out.data_ptr<scalar_t>(),
       // reserved_space.data_ptr<scalar_t>(),
        (void*) (lt_workspace.data_ptr<scalar_t>()));
    TORCH_CHECK(result == 0, "linear_bias_forward failed.")
  });

  return {out};
}

std::vector<at::Tensor> linear_bias_backward(at::Tensor input, at::Tensor weight, at::Tensor d_output) {

  auto batch_size = input.size(0);
  auto in_features = input.size(1);

  int out_features = weight.size(0);

  //auto reserved_size = get_mlp_reserved_space(batch_size, num_layers, output_features.data());

  // create output/workspace tensor
  auto opts = input.options();
  auto d_weight = at::empty({out_features, in_features}, opts);
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION < 11600
  auto d_bias = d_output.view({-1, out_features}).sum(0, false);
#else
  auto d_bias = at::empty({out_features}, opts);
#endif
  auto d_input = at::empty({batch_size, in_features}, opts);
  //auto reserved_space = at::empty({reserved_size}, inputs[0].type());
  // allocate fixed 4MB workspace for cublaslt for now, and this gets at least 4 MB
  auto lt_workspace = at::empty({1 << 22}, opts);

  DISPATCH_HALF_AND_BF16(input.scalar_type(), "linear_bias_backward", [&] {
    scalar_t* w_ptr = weight.data_ptr<scalar_t>();
    auto result = linear_bias_backward_cuda<scalar_t>(
        input.data_ptr<scalar_t>(),
        w_ptr,
        d_output.data_ptr<scalar_t>(),
        in_features,
        batch_size,
        out_features,
        d_weight.data_ptr<scalar_t>(),
        d_bias.data_ptr<scalar_t>(),
        d_input.data_ptr<scalar_t>(),
       // reserved_space.data_ptr<scalar_t>(),
        /*residual=*/false,
        (void*) (lt_workspace.data_ptr<scalar_t>()));
    TORCH_CHECK(result == 0, "linear_bias_backward failed.")
  });

  return {d_input, d_weight, d_bias};
}

std::vector<at::Tensor> linear_bias_wgrad(at::Tensor input, at::Tensor d_output) {

  auto batch_size = input.size(0);
  auto in_features = input.size(1);

  int out_features = d_output.size(1);

  //auto reserved_size = get_mlp_reserved_space(batch_size, num_layers, output_features.data());

  // create output/workspace tensor
  auto opts = input.options();
  auto d_weight = at::empty({out_features, in_features}, opts);
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION < 11600
  auto d_bias = d_output.view({-1, out_features}).sum(0, false);
#else
  auto d_bias = at::empty({out_features}, opts);
#endif
  //auto reserved_space = at::empty({reserved_size}, inputs[0].type());
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
        d_bias.data_ptr<scalar_t>(),
       // reserved_space.data_ptr<scalar_t>(),
        (void*) (lt_workspace.data_ptr<scalar_t>()));
    TORCH_CHECK(result == 0, "linear_bias_wgrad failed.")
  });

  return {d_weight, d_bias};
}

std::vector<at::Tensor> linear_bias_residual_backward(at::Tensor input, at::Tensor weight, at::Tensor d_output, at::Tensor d_input) {

  auto batch_size = input.size(0);
  auto in_features = input.size(1);

  int out_features = weight.size(0);

  //auto reserved_size = get_mlp_reserved_space(batch_size, num_layers, output_features.data());

  // create output/workspace tensor
  auto opts = input.options();
  auto d_weight = at::empty({out_features, in_features}, opts);
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION < 11600
  auto d_bias = d_output.view({-1, out_features}).sum(0, false);
#else
  auto d_bias = at::empty({out_features}, opts);
#endif
  CHECK_SHAPE(d_input, batch_size, in_features);
  //auto reserved_space = at::empty({reserved_size}, inputs[0].type());
  // allocate fixed 4MB workspace for cublaslt for now, and this gets at least 4 MB
  auto lt_workspace = at::empty({1 << 22}, opts);

  DISPATCH_HALF_AND_BF16(input.scalar_type(), "linear_bias_backward", [&] {
    scalar_t* w_ptr = weight.data_ptr<scalar_t>();
    auto result = linear_bias_backward_cuda<scalar_t>(
        input.data_ptr<scalar_t>(),
        w_ptr,
        d_output.data_ptr<scalar_t>(),
        in_features,
        batch_size,
        out_features,
        d_weight.data_ptr<scalar_t>(),
        d_bias.data_ptr<scalar_t>(),
        d_input.data_ptr<scalar_t>(),
       // reserved_space.data_ptr<scalar_t>(),
        /*residual=*/true,
        (void*) (lt_workspace.data_ptr<scalar_t>()));
    TORCH_CHECK(result == 0, "linear_bias_residual_backward failed.")
  });

  return {d_input, d_weight, d_bias};
}

std::vector<at::Tensor> linear_gelu_forward(at::Tensor input, at::Tensor weight, at::Tensor bias,
                                            bool save_gelu_in, int heuristic) {

  auto batch_size = input.size(0);
  auto in_features = input.size(1);

  int out_features = weight.size(0);

  //auto reserved_size = get_mlp_reserved_space(batch_size, num_layers, output_features.data());

  // create output/workspace tensor
  auto opts = input.options();
  auto output = at::empty({batch_size, out_features}, opts);
  at::Tensor gelu_in;
  if (save_gelu_in) { gelu_in = at::empty({batch_size, out_features}, opts); }
  //auto reserved_space = at::empty({reserved_size}, inputs[0].type());
  // allocate fixed 4MB workspace for cublaslt for now, and this gets at least 4 MB
  auto lt_workspace = at::empty({1 << 22}, opts);

  DISPATCH_HALF_AND_BF16(input.scalar_type(), "linear_gelu_forward", [&] {
    scalar_t* w_ptr = weight.data_ptr<scalar_t>();
    scalar_t* b_ptr = bias.data_ptr<scalar_t>();
    auto result = linear_gelu_forward_cuda<scalar_t>(
        input.data_ptr<scalar_t>(),
        w_ptr,
        b_ptr,
        in_features,
        batch_size,
        out_features,
        heuristic,
        output.data_ptr<scalar_t>(),
        save_gelu_in ? gelu_in.data_ptr<scalar_t>() : nullptr,
       // reserved_space.data_ptr<scalar_t>(),
        (void*) (lt_workspace.data_ptr<scalar_t>()));
    TORCH_CHECK(result == 0, "linear_gelu_forward failed.")
  });

  std::vector<at::Tensor> result = {output};
  if (save_gelu_in) { result.push_back(gelu_in); };
  return result;
}

std::vector<at::Tensor> linear_gelu_linear_backward(at::Tensor input, at::Tensor gelu_in, at::Tensor output1, at::Tensor weight1, at::Tensor weight2, at::Tensor d_output2, int heuristic) {

  auto batch_size = input.size(0);
  auto in_features = input.size(1);

  int hidden_features = weight1.size(0);
  int out_features = weight2.size(0);

  //auto reserved_size = get_mlp_reserved_space(batch_size, num_layers, output_features.data());

  // create output/workspace tensor
  auto opts = input.options();
  auto d_weight1 = at::empty({hidden_features, in_features}, opts);
  auto d_weight2 = at::empty({out_features, hidden_features}, opts);
  auto d_bias1 = at::empty({hidden_features}, opts);
  auto d_bias2 = at::empty({out_features}, opts);
  auto d_input = at::empty({batch_size, in_features}, opts);
  auto d_output1 = at::empty({batch_size, hidden_features}, opts);
  //auto reserved_space = at::empty({reserved_size}, inputs[0].type());
  // allocate fixed 4MB workspace for cublaslt for now, and this gets at least 4 MB
  auto lt_workspace = at::empty({1 << 22}, opts);

  DISPATCH_HALF_AND_BF16(input.scalar_type(), "linear_bias_backward", [&] {
    //scalar_t* w_ptr = weight.data_ptr<scalar_t>();
    //scalar_t* d_b_ptr = d_bias.data_ptr<scalar_t>();
    auto result = linear_gelu_linear_backward_cuda<scalar_t>(
        input.data_ptr<scalar_t>(),
        gelu_in.data_ptr<scalar_t>(),
        output1.data_ptr<scalar_t>(),
        weight1.data_ptr<scalar_t>(),
        weight2.data_ptr<scalar_t>(),
        d_output1.data_ptr<scalar_t>(),
        d_output2.data_ptr<scalar_t>(),
        in_features,
        batch_size,
        hidden_features,
        out_features,
        heuristic,
        d_weight1.data_ptr<scalar_t>(),
        d_weight2.data_ptr<scalar_t>(),
        d_bias1.data_ptr<scalar_t>(),
        d_bias2.data_ptr<scalar_t>(),
        d_input.data_ptr<scalar_t>(),
       // reserved_space.data_ptr<scalar_t>(),
        /*residual=*/false,
        (void*) (lt_workspace.data_ptr<scalar_t>()));
    TORCH_CHECK(result == 0, "linear_gelu_linear_backward failed.")
  });

  return {d_input, d_weight1, d_bias1, d_weight2, d_bias2};
}

std::vector<at::Tensor> linear_residual_gelu_linear_backward(at::Tensor input, at::Tensor gelu_in, at::Tensor output1, at::Tensor weight1, at::Tensor weight2, at::Tensor d_output2, at::Tensor d_input, int heuristic) {

  auto batch_size = input.size(0);
  auto in_features = input.size(1);

  int hidden_features = weight1.size(0);
  int out_features = weight2.size(0);

  //auto reserved_size = get_mlp_reserved_space(batch_size, num_layers, output_features.data());

  // create output/workspace tensor
  auto opts = input.options();
  auto d_weight1 = at::empty({hidden_features, in_features}, opts);
  auto d_weight2 = at::empty({out_features, hidden_features}, opts);
  auto d_bias1 = at::empty({hidden_features}, opts);
  auto d_bias2 = at::empty({out_features}, opts);
  CHECK_SHAPE(d_input, batch_size, in_features);
  auto d_output1 = at::empty({batch_size, hidden_features}, opts);
  //auto reserved_space = at::empty({reserved_size}, inputs[0].type());
  // allocate fixed 4MB workspace for cublaslt for now, and this gets at least 4 MB
  auto lt_workspace = at::empty({1 << 22}, opts);

  DISPATCH_HALF_AND_BF16(input.scalar_type(), "linear_bias_backward", [&] {
    //scalar_t* w_ptr = weight.data_ptr<scalar_t>();
    //scalar_t* d_b_ptr = d_bias.data_ptr<scalar_t>();
    auto result = linear_gelu_linear_backward_cuda<scalar_t>(
        input.data_ptr<scalar_t>(),
        gelu_in.data_ptr<scalar_t>(),
        output1.data_ptr<scalar_t>(),
        weight1.data_ptr<scalar_t>(),
        weight2.data_ptr<scalar_t>(),
        d_output1.data_ptr<scalar_t>(),
        d_output2.data_ptr<scalar_t>(),
        in_features,
        batch_size,
        hidden_features,
        out_features,
        heuristic,
        d_weight1.data_ptr<scalar_t>(),
        d_weight2.data_ptr<scalar_t>(),
        d_bias1.data_ptr<scalar_t>(),
        d_bias2.data_ptr<scalar_t>(),
        d_input.data_ptr<scalar_t>(),
       // reserved_space.data_ptr<scalar_t>(),
        /*residual=*/true,
        (void*) (lt_workspace.data_ptr<scalar_t>()));
    TORCH_CHECK(result == 0, "linear_residual_gelu_linear_backward failed.")
  });

  return {d_input, d_weight1, d_bias1, d_weight2, d_bias2};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("linear_bias_forward", &linear_bias_forward, "linear bias forward");
  m.def("linear_bias_backward", &linear_bias_backward, "linear bias backward");
  m.def("linear_bias_wgrad", &linear_bias_wgrad, "linear bias wgrad");
  m.def("linear_bias_residual_backward", &linear_bias_residual_backward, "linear bias residual backward");
  m.def("linear_gelu_forward", &linear_gelu_forward, "linear gelu forward");
  m.def("linear_gelu_linear_backward", &linear_gelu_linear_backward, "linear gelu linear backward");
  m.def("linear_residual_gelu_linear_backward", &linear_residual_gelu_linear_backward, "linear residual gelu linear backward");
}
