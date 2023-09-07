#pragma once

#include "fast_unpickler.h"
#include "tensor.h"
#include <map>
#include <memory>
#include <optional>
#include <string>

class PyTorchModelManager {
  // Support for PyTorch models
private:
  std::string filename;
  std::string modelname;
  std::shared_ptr<ZipFileParser> fileReader;
  std::shared_ptr<FastUnpickler> unpickler;
  std::map<std::string, std::shared_ptr<OriginTensor>> tensorMap;
  bool loaded;

public:
  std::shared_ptr<FastUnpickler::object> parse_result;
  std::optional<py::object> ffi_result;
  PyTorchModelManager() = default;
  PyTorchModelManager(std::string filename) : filename(filename), loaded(false) {}

  void load();
};