#include "fast_unpickler.h"
#include "ffi.h"
#include "model_manager.h"
#include "tensor.h"

namespace ffi {

void init_ffi_tensor(py::module_ &m) {
  py::class_<BaseTensor, std::shared_ptr<BaseTensor>>(m, "BaseTensor")
      .def_property_readonly("shape", &BaseTensor::get_shape)
      .def("__str__", &BaseTensor::to_string);

  py::class_<OriginTensor, std::shared_ptr<OriginTensor>, BaseTensor>(m, "OriginTensor")
      .def_property("segment", &OriginTensor::get_segment, &OriginTensor::set_segment)
      .def("create_tensor_grid", &OriginTensor::create_tensor_grid)
      .def("set_as_managed", &OriginTensor::set_as_managed);

  py::class_<DerivedTensor, std::shared_ptr<DerivedTensor>, BaseTensor>(m, "DerivedTensor");

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