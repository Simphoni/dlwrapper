#include <common.h>
#include <nico.h>
#include <torch/extension.h>

void testing(torch::Tensor t) {
  fprintf(stderr, "%ld\n", t.size(0));
  fprintf(stderr, "%ld\n", t.numel());
  fprintf(stderr, "%ld\n", t.dim());
  fprintf(stderr, "%d\n", t.is_contiguous());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "An extensively tailored CUDA library for machine learning.";

  auto m_nico = m.def_submodule("nico", "A backend to bypass pytorch NCCL.");
  m_nico.def("init_nccl", &_init_nccl, "Initialize NCCL communicator.");
  m_nico.def("sync_stream", &_sync_stream,
             "Nico's CUDA stream synchronize operation.");
  m_nico.def("sendrecv", &_sendrecv, "Nico's sendrecv operation.");
  m_nico.def("broadcast", &_broadcast, "Nico's broadcast operation.");
  m_nico.def("allgather_into_tensor_doubling", &_allgather_into_tensor_doubling,
             "Nico's allgather operation w/ recursive doubling.");
  m_nico.def("export_summary", &_manager_export_summary,
             "Export Nico's internal performance summary.");

  m_nico.def("testing", &testing, "A testing function.");
}