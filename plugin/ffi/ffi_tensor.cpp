#include "fast_unpickler.h"
#include "ffi.h"
#include "model_manager.h"
#include "tensor.h"

namespace ffi {

void init_ffi_tensor(py::module_ &m) {
  py::class_<OriginTensor, std::shared_ptr<OriginTensor>>(m, "OriginTensor")
      .def("set_segment", &OriginTensor::set_segment)
      .def("get_shape", &OriginTensor::get_shape)
      .def("get_segment", &OriginTensor::get_segment);
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