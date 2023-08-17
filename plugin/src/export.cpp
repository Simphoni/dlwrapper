#include "common.h"
#include "device_manager.h"
#include "nico.h"
#include <torch/extension.h>

void test_ipc_allgather();

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "An extensively tailored CUDA library for machine learning.";
  // == nico submodule ==
  auto m_nico =
      m.def_submodule("nico_native", "A backend to bypass pytorch NCCL.");

  m_nico.def("init_nico", &_init_nico, "Initialize nico's device context.",
             py::arg("pg"), py::arg("enable_uva") = false);
  m_nico.def("destroy_nico", &_destroy_nico, "Destroy nico.");

  // m_nico.def("sync_stream", &_sync_stream,
  //            "Nico's CUDA stream synchronize operation.",
  //            py::arg("stream_index") = 0);

  m_nico.def("sendrecv", &_sendrecv, "Nico's sendrecv operation.");
  m_nico.def("broadcast", &_broadcast, "Nico's broadcast operation.");
  m_nico.def("allgather_into_tensor_doubling", &_allgather_into_tensor_doubling,
             "Nico's allgather operation w/ recursive doubling.");

  m_nico.def("allgather_with_peer_access", &_allgather_with_peer_access,
             "Nico's ring all_gather operation.", py::arg("dst"),
             py::arg("src"), py::arg("idx") = 0, py::arg("prof") = false);

  m_nico.def("export_summary", &_manager_export_summary,
             "Export Nico's internal performance summary.");

  m_nico.def("testing", &test_ipc_allgather, "A testing function.");
}