#include <common.h>
#include <nico.h>
#include <torch/extension.h>

// functions to export
void _init_nccl(c10d::ProcessGroupNCCL &p, at::Device dev);

void _broadcast(torch::Tensor t);

void testing(torch::Tensor t) {
  fprintf(stderr, "%ld\n", t.size(0));
  fprintf(stderr, "%ld\n", t.numel());
  fprintf(stderr, "%ld\n", t.dim());
  fprintf(stderr, "%d\n", t.is_contiguous());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "An extensively tailored CUDA library for machine learning.";

  auto m_nico = m.def_submodule("nico", "A backend to bypass pytorch NCCL.");
  m_nico.def("init_nccl", &_init_nccl,
             "Initialize NCCL communication object and store it in "
             "CudaContextManager.");
  m_nico.def("broadcast", &_broadcast, "Nico's broadcast operation.");

  m_nico.def("testing", &testing, "A testing function.");
}