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
  std::vector<uint64_t> dims, strides;
  std::string name;
};