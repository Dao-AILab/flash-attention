#include <torch/extension.h>

// CUDA forward declarations
std::vector<at::Tensor> softmax_xentropy_cuda(
    const at::Tensor &input,
    const at::Tensor &labels,
    const float smoothing,
    const int total_classes);

at::Tensor softmax_xentropy_backward_cuda(
    const at::Tensor &grad_loss,
    at::Tensor &logits,
    const at::Tensor &max_log_sum_exp,
    const at::Tensor &labels,
    const float smoothing,
    const bool inplace,
    const int total_classes);

// C++ interface

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<at::Tensor> softmax_xentropy_forward(
    const at::Tensor &input,
    const at::Tensor &labels,
    const float smoothing,
    const int total_classes=-1) {
    // For tensor parallel cross entropy with smoothing, we want to pass in the total number
    // of classes so that smoothing can be applied correctly. If total_classes=-1, use the
    // last dimension of the input tensor.
    CHECK_INPUT(input);
    CHECK_INPUT(labels);

    return softmax_xentropy_cuda(input, labels, smoothing, total_classes);
}

at::Tensor softmax_xentropy_backward(
    const at::Tensor &grad_loss,
    at::Tensor &logits,
    const at::Tensor &max_log_sum_exp,
    const at::Tensor &labels,
    const float smoothing,
    const bool inplace,
    const int total_classes=-1)  {
    CHECK_INPUT(grad_loss);
    CHECK_INPUT(logits);
    CHECK_INPUT(max_log_sum_exp);
    CHECK_INPUT(labels);

    return softmax_xentropy_backward_cuda(grad_loss, logits, max_log_sum_exp, labels,
                                          smoothing, inplace, total_classes);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &softmax_xentropy_forward, "Softmax cross entropy loss with label smoothing forward (CUDA)", py::arg("input"), py::arg("labels"), py::arg("smoothing"), py::arg("total_classes")=-1);
    m.def("backward", &softmax_xentropy_backward, "Softmax cross entropy loss with label smoothing backward (CUDA)", py::arg("grad_loss"), py::arg("logits"), py::arg("max_log_sum_exp"), py::arg("labels"), py::arg("smoothing"), py::arg("inplace"), py::arg("total_classes")=-1);
}
