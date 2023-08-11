#include "common.h"
#include "device_manager.h"
#include "nico.h"
#include <torch/extension.h>

void testing(torch::Tensor t) {
  static int callnum = 0;
  callnum++;
  auto manager = DeviceContextManager::get();
  int data = manager->get_local_rank() + callnum;
  char **result = manager->ipc_allgather((char *)&data, sizeof(int));
  int res[8];
  for (int i = 0; i < 8; i++) {
    res[i] = *reinterpret_cast<int *>(result[i]);
    assert(res[i] == callnum + i);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "An extensively tailored CUDA library for machine learning.";
  // == nico submodule ==
  auto m_nico = m.def_submodule("nico", "A backend to bypass pytorch NCCL.");

  m_nico.def("init_nccl", &_init_nccl, "Initialize nico's device context.",
             py::arg("pg"), py::arg("device"), py::arg("enable_uva") = false);

  m_nico.def("sync_stream", &_sync_stream,
             "Nico's CUDA stream synchronize operation.",
             py::arg("stream_index") = 0);

  m_nico.def("sendrecv", &_sendrecv, "Nico's sendrecv operation.");
  m_nico.def("broadcast", &_broadcast, "Nico's broadcast operation.");
  m_nico.def("allgather_into_tensor_doubling", &_allgather_into_tensor_doubling,
             "Nico's allgather operation w/ recursive doubling.");
  m_nico.def("export_summary", &_manager_export_summary,
             "Export Nico's internal performance summary.");
  m_nico.def("memcpy_peer", &_memcpy_peer, "feature during testing");

  m_nico.def("testing", &testing, "A testing function.");
}