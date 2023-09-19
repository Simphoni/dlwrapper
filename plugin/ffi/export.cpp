#include "ffi.h"

PYBIND11_MODULE(dlwrapperffi, m) {
  m.doc() = "An extensively tailored CUDA library for machine learning.";

  // == nico submodule ==
  auto m_nico = m.def_submodule("nico_native", "A backend to bypass pytorch NCCL.");
  ffi::init_ffi_nico(m_nico);

  auto m_torch_backend =
      m.def_submodule("torch_backend", "Pytorch model parsing and tensor management.");
  ffi::init_ffi_tensor(m_torch_backend);

  auto m_cuda = m.def_submodule("cuda", "Optimized high performance CUDA kernels.");
  ffi::init_ffi_cuda_native(m_cuda);
}