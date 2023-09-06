#pragma once

#include "fast_unpickler.h"
#include "tensor.h"
#include <map>
#include <memory>
#include <string>

class PyTorchModelManager {
  // Support for PyTorch models
private:
  std::string filename;
  std::string modelname;
  std::shared_ptr<ZipFileParser> fileReader;
  std::shared_ptr<FastUnpickler> unpickler;
  std::map<std::string, std::shared_ptr<UntypedTensor>> tensorMap;
  bool loaded;

public:
  PyTorchModelManager() = default;
  PyTorchModelManager(std::string filename) : filename(filename), loaded(false) {}

  void load();
};