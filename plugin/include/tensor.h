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
  int64_t ndim;
  bool requires_grad;
  bool is_cstyle;

public:
  UntypedTensor(std::shared_ptr<Storage> storage, int64_t offset, std::vector<int64_t> dims,
                std::vector<int64_t> strides, bool requires_grad)
      : storage(storage), offset(offset), dims(dims), strides(strides),
        requires_grad(requires_grad) {
    ndim      = dims.size();
    is_cstyle = (strides[ndim - 1] == 1);
    for (int i = 1; i < ndim - 1; i++) {
      is_cstyle = is_cstyle && (strides[i] == strides[i + 1] * dims[i + 1]);
    }
  }

  std::string to_string() {
    std::string ret = "Tensor(dims=[";
    for (auto &dim : dims) {
      ret += std::to_string(dim) + ", ";
    }
    ret += "], strides=[";
    for (auto &stride : strides) {
      ret += std::to_string(stride) + ", ";
    }
    ret += "]";
    if (is_cstyle) {
      ret += ", is cstyle contiguous";
    }
    ret += ")";
    return ret;
  }
};