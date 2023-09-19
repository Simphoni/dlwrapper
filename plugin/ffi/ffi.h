#pragma once

#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

namespace py = pybind11;
using py::literals::operator""_a;

namespace ffi {
void init_ffi_nico(py::module_ &m);
void init_ffi_tensor(py::module_ &m);
void init_ffi_cuda_native(py::module_ &m);
} // namespace ffi