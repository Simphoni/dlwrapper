#include "common.h"
#include "nico.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nico_steal_nccl", &_steal_nccl, "Steal NCCL comm from torch");
}