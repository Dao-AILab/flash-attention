#include <torch/extension.h>

// CUDA forward declarations
std::vector<at::Tensor> softmax_xentropy_cuda(
    const at::Tensor &input,
    const at::Tensor &labels,
    const float smoothing);

at::Tensor softmax_xentropy_backward_cuda(
    const at::Tensor &grad_loss,
    at::Tensor &logits,
    const at::Tensor &max_log_sum_exp,
    const at::Tensor &labels,
    const float smoothing,
    const bool inplace);

// C++ interface

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<at::Tensor> softmax_xentropy_forward(
    const at::Tensor &input,
    const at::Tensor &labels,
    const float smoothing) {
    CHECK_CUDA(input);
    CHECK_INPUT(labels);

    return softmax_xentropy_cuda(input, labels, smoothing);
}

at::Tensor softmax_xentropy_backward(
    const at::Tensor &grad_loss,
    at::Tensor &logits,
    const at::Tensor &max_log_sum_exp,
    const at::Tensor &labels,
    const float smoothing,
    const bool inplace)  {
    CHECK_CUDA(grad_loss);
    CHECK_CUDA(logits);
    CHECK_INPUT(max_log_sum_exp);
    CHECK_INPUT(labels);

    return softmax_xentropy_backward_cuda(grad_loss, logits, max_log_sum_exp, labels, smoothing, inplace);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &softmax_xentropy_forward, "Softmax cross entropy loss with label smoothing forward (CUDA)");
    m.def("backward", &softmax_xentropy_backward, "Softmax cross entropy loss with label smoothing backward (CUDA)");
}
