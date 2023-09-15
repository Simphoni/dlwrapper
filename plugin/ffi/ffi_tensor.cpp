#include "fast_unpickler.h"
#include "ffi.h"
#include "model_manager.h"
#include "tensor.h"

namespace ffi {

void init_ffi_tensor(py::module_ &m) {
  py::enum_<MemoryType>(m, "MemoryType")
      .value("DEVICE", MemoryType::DEVICE)
      .value("PINNED", MemoryType::PINNED)
      .value("PAGEABLE", MemoryType::PAGEABLE);

  py::class_<BaseTensor, std::shared_ptr<BaseTensor>>(m, "BaseTensor")
      .def_property_readonly("shape", &BaseTensor::get_shape)
      .def("__str__", &BaseTensor::to_string)
      .def("torch_get_contiguous", &BaseTensor::torch_get_contiguous, py::arg("memtype"));

  py::class_<OriginTensor, std::shared_ptr<OriginTensor>, BaseTensor>(m, "OriginTensor")
      .def_property("segment", &OriginTensor::get_segment, &OriginTensor::set_segment)
      .def("create_tensor_grid", &OriginTensor::create_tensor_grid)
      .def("set_as_managed", &OriginTensor::set_as_managed)
      .def("move_to", &OriginTensor::move_to, py::arg("memtype"))
      .def("wait", &OriginTensor::wait);

  py::class_<DerivedTensor, std::shared_ptr<DerivedTensor>, BaseTensor>(m, "DerivedTensor")
      .def("move_to", &DerivedTensor::move_to, py::arg("memtype"))
      .def("wait", &DerivedTensor::wait);

  py::class_<TensorGrid, std::shared_ptr<TensorGrid>>(m, "TensorGrid")
      .def("get_slice", &TensorGrid::get_slice);

  py::class_<PyTorchModelManager, std::shared_ptr<PyTorchModelManager>>(m, "ModelManagerTorch")
      .def(py::init<const std::string &>())
      .def("load",
           [](PyTorchModelManager &self) {
             if (self.ffi_result == std::nullopt) {
               self.load();
               self.ffi_result = self.parse_result->to_pyobject();
             }
             return self.ffi_result;
           },
           py::return_value_policy::copy);
}

} // namespace ffi