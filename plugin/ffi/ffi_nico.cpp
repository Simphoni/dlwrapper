#include "ffi.h"
#include "nico.h"

void test_ipc_allgather();
namespace ffi {
void init_ffi_nico(py::module_ &m) {
  m.def("init_nico", &_init_nico, "Initialize nico's device context.", py::arg("pg"),
        py::arg("enable_uva") = false);
  m.def("destroy_nico", &_destroy_nico, "Destroy nico.");

  m.def("sync_stream", &_sync_stream, "Nico's CUDA stream synchronize operation.",
        py::arg("stream_index") = 0);

  m.def("sendrecv", &_sendrecv, "Nico's sendrecv operation.");
  m.def("broadcast", &_broadcast, "Nico's broadcast operation.");

  m.def("allgather", &_allgather, "Nico's ring all_gather operation.", py::arg("dst"),
        py::arg("src"), py::arg("idx") = 0, py::arg("prof") = false);

  m.def("scatter", &_scatter, "Nico's scatter operation.", py::arg("tensor"), py::arg("src_rank"),
        py::arg("prof") = false);

  m.def("export_summary", &_manager_export_summary, "Export Nico's internal performance summary.");

  m.def("testing", &test_ipc_allgather, "A testing function.");
}
} // namespace ffi