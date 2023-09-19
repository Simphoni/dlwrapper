#include "cuda/cuda_torch_api.h"
#include "ffi.h"

namespace ffi {

void init_ffi_cuda_native(py::module_ &m) {
  auto m_F = m.def_submodule("F");

  m_F.def("grid_sample", &grid_sample, "grid_sample", "input"_a, "grid"_a, "output"_a,
          "output_ncdhw"_a = true);
}

} // namespace ffi