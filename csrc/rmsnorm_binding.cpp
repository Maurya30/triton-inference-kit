// Python binding for the CUDA RMSNorm kernel.
//
// This file is compiled by the host C++ compiler, rmsnorm.cu by nvcc.
// Keeping the pybind11 glue out of the .cu file keeps nvcc away from
// pybind11's heavy templates, which shortens the JIT build.

#include <torch/extension.h>

// Defined in rmsnorm.cu.
torch::Tensor rmsnorm_cuda(torch::Tensor x, torch::Tensor weight, double eps);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "triton_inference_kit CUDA C++ kernels";
  m.def("rmsnorm", &rmsnorm_cuda,
        "Fused RMSNorm over the last dim of a 2-D CUDA tensor (float16 / bfloat16 / float32).",
        py::arg("x"), py::arg("weight"), py::arg("eps"));
}
