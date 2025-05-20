#include <pybind11/numpy.h>
#include <torch/extension.h>

namespace py = pybind11;

torch::Tensor logic_layer_cuda_forward(torch::Tensor x, torch::Tensor a, torch::Tensor b, torch::Tensor w);
torch::Tensor logic_layer_cuda_backward_w(torch::Tensor x, torch::Tensor a, torch::Tensor b, torch::Tensor grad_y);
torch::Tensor logic_layer_cuda_backward_x(torch::Tensor x, torch::Tensor a, torch::Tensor b, torch::Tensor w,
                                          torch::Tensor grad_y, torch::Tensor given_x_indices_of_y_start,
                                          torch::Tensor given_x_indices_of_y);
torch::Tensor logic_layer_cuda_eval(torch::Tensor x, torch::Tensor a, torch::Tensor b, torch::Tensor w);
std::tuple<torch::Tensor, int> tensor_packbits_cuda(torch::Tensor t, const int bit_count);
torch::Tensor groupbitsum(torch::Tensor b, const int pad_len, const int k);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &logic_layer_cuda_forward, "Logic layer forward (CUDA)");
    m.def("backward_w", &logic_layer_cuda_backward_w, "Logic layer backward w (CUDA)");
    m.def("backward_x", &logic_layer_cuda_backward_x, "Logic layer backward x (CUDA)");
    m.def("eval", &logic_layer_cuda_eval, "Logic layer eval (CUDA)");
    m.def("tensor_packbits_cuda", &tensor_packbits_cuda, "Pack bits (CUDA)");
    m.def(
        "groupbitsum",
        [](torch::Tensor b, const int pad_len, const unsigned int k)
        {
            if (b.size(0) % k != 0)
            {
                throw py::value_error("in_dim (" + std::to_string(b.size(0)) + ") has to be divisible by k (" +
                                      std::to_string(k) + ") but it is not");
            }
            return groupbitsum(b, pad_len, k);
        },
        "groupbitsum (CUDA)");
}
