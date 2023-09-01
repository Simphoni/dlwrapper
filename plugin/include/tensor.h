#pragma once
#include "misc.h"
#include <memory>
#include <vector>

class Storage {
private:
  char *data;
  std::string dtype, location;
  uint64_t numel, dsize;
#define bytes (numel * dsize)

public:
  Storage() = default;
  Storage(char *data, std::string dtype, std::string location, uint64_t numel, uint64_t dsize)
      : data(data), dtype(dtype), location(location), numel(numel), dsize(dsize) {}

#undef bytes
};

class UntypedTensor {
private:
  std::shared_ptr<Storage> storage;
  int64_t offset;
  std::vector<int64_t> dims, strides;
  bool requires_grad;

public:
  UntypedTensor(std::shared_ptr<Storage> storage, int64_t offset, std::vector<int64_t> dims,
                std::vector<int64_t> strides, bool requires_grad)
      : storage(storage), offset(offset), dims(dims), strides(strides),
        requires_grad(requires_grad) {}

  std::string to_string() {
    std::string ret = "Tensor(dims=[";
    for (auto &dim : dims) {
      ret += std::to_string(dim) + ", ";
    }
    ret += "], strides=[";
    for (auto &stride : strides) {
      ret += std::to_string(stride) + ", ";
    }
    ret += "])";
    return ret;
  }
};