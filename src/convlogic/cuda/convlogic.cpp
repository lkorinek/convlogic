#include <pybind11/numpy.h>
#include <torch/extension.h>

namespace py = pybind11;

torch::Tensor convlogic_cuda_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor selection, int stride_h,
                                     int stride_w, int pad_h, int pad_w, int kernel_h, int kernel_w);
torch::Tensor convlogic_cuda_backward_x(torch::Tensor grad_y, torch::Tensor x, torch::Tensor weight,
                                        torch::Tensor selection, int stride_h, int stride_w, int pad_h, int pad_w);
torch::Tensor convlogic_cuda_backward_weight(torch::Tensor grad_y, torch::Tensor x, torch::Tensor selection,
                                             int stride_h, int stride_w, int pad_h, int pad_w);

torch::Tensor treelogic_cuda_forward(torch::Tensor x, torch::Tensor weights);
torch::Tensor treelogic_cuda_backward_x(torch::Tensor grad_y, torch::Tensor x, torch::Tensor weights);
torch::Tensor treelogic_cuda_backward_weight(torch::Tensor grad_y, torch::Tensor x);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
full_convlogic_cuda_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor selection, int stride_h, int stride_w,
                            int pad_h, int pad_w, int kernel_h, int kernel_w);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
full_convlogic_cuda_backward_x(torch::Tensor grad_y, torch::Tensor x1, torch::Tensor x2, torch::Tensor x3,
                               torch::Tensor weight, torch::Tensor indices, torch::Tensor selection, int stride_h,
                               int stride_w, int pad_h, int pad_w);
torch::Tensor full_convlogic_cuda_backward_weight(torch::Tensor grad_x2, torch::Tensor grad_x3, torch::Tensor grad_y,
                                                  torch::Tensor x1, torch::Tensor x2, torch::Tensor x3,
                                                  torch::Tensor indices, torch::Tensor selection, int stride_h,
                                                  int stride_w, int pad_h, int pad_w);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("convlogic_forward", &convlogic_cuda_forward, "ConvLogic forward (CUDA)");
    m.def("convlogic_backward_x", &convlogic_cuda_backward_x, "ConvLogic backward input (CUDA)");
    m.def("convlogic_backward_weight", &convlogic_cuda_backward_weight, "ConvLogic backward weight (CUDA)");

    m.def("treelogic_forward", &treelogic_cuda_forward, "TreeLogic forward (CUDA)");
    m.def("treelogic_backward_x", &treelogic_cuda_backward_x, "TreeLogic backward input (CUDA)");
    m.def("treelogic_backward_weight", &treelogic_cuda_backward_weight, "TreeLogic backward weight (CUDA)");

    m.def("full_convlogic_forward", &full_convlogic_cuda_forward, "Full ConvLogic forward with tree depth 2 (CUDA)");
    m.def("full_convlogic_backward_x", &full_convlogic_cuda_backward_x,
          "Full ConvLogic backward input with tree depth 2 (CUDA)");
    m.def("full_convlogic_backward_weight", &full_convlogic_cuda_backward_weight,
          "Full ConvLogic backward weight with tree depth 2 (CUDA)");
}
