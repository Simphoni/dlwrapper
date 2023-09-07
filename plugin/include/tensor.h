// Tensor abstraction api. Provide big-endian tensor abstraction.
// ?Is it a good choice to base internal logic on big-endian

#pragma once

#include "misc.h"
#include <cassert>
#include <memory>
#include <vector>

#define DIV_CEIL(x, y) ((x - 1) / (y) + 1)

class CachedStorage {
private:
  char *data;
  std::string dtype, location;
  uint64_t numel, dsize;
#define bytes (numel * dsize)

public:
  CachedStorage() = default;
  CachedStorage(char *data, std::string dtype, std::string location, uint64_t numel, uint64_t dsize)
      : data(data), dtype(dtype), location(location), numel(numel), dsize(dsize) {}

#undef bytes
};

class BaseTensor {
protected:
  std::shared_ptr<CachedStorage> storage;
  int64_t offset;
  std::vector<int64_t> shape, strides;
  int64_t ndim;
  bool requires_grad;
  bool is_cstyle;

public:
  BaseTensor() = default;
  BaseTensor(std::shared_ptr<CachedStorage> storage, int64_t offset, std::vector<int64_t> shape,
             std::vector<int64_t> strides, bool requires_grad)
      : storage(storage), offset(offset), shape(shape), strides(strides),
        requires_grad(requires_grad) {
    ndim = shape.size();
    assert((size_t)ndim == strides.size());
    for (int i = 0; i < ndim; i++) {
      assert(shape[i] > 0);
      assert(strides[i] > 0);
    }
    is_cstyle = (strides[ndim - 1] == 1);
    for (int i = 1; i < ndim - 1; i++) {
      is_cstyle = is_cstyle && (strides[i] == strides[i + 1] * shape[i + 1]);
    }
    if (ndim > 1 && shape[0] != 1) {
      is_cstyle = is_cstyle && (strides[0] == strides[1] * shape[1]);
    }
  }

  std::string to_string() {
    std::string ret = "Tensor(shape=[";
    for (auto &dim : shape) {
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

  std::vector<int64_t> get_shape() const noexcept { return shape; }
};

class OriginTensor : public BaseTensor {
protected:
  std::vector<int64_t> segment;
  friend class DerivedTensor;

public:
  OriginTensor(std::shared_ptr<CachedStorage> storage, int64_t offset, std::vector<int64_t> shape,
               std::vector<int64_t> strides, bool requires_grad)
      : BaseTensor(storage, offset, shape, strides, requires_grad) {
    segment = std::vector<int64_t>(ndim, 1);
  }
  void set_segment(std::vector<int64_t> segment) {
    assert((int64_t)segment.size() == ndim);
    this->segment = segment;
  }
  std::vector<int64_t> get_segment() const noexcept { return segment; }
  // std::vector<DerivedTensor> derive() const;
};

class DerivedTensor : public BaseTensor {
private:
  std::shared_ptr<OriginTensor> origin;
  int64_t cstyle_order;
  std::vector<int64_t> grid_index, grid_offset, grid_segment;

public:
  DerivedTensor(std::shared_ptr<OriginTensor> origin, int64_t cstyle_order)
      : origin(origin), cstyle_order(cstyle_order) {
    assert(origin->is_cstyle);
    is_cstyle    = false; // do not derive!
    ndim         = origin->ndim;
    storage      = origin->storage;
    offset       = origin->offset;
    strides      = origin->strides;
    grid_segment = origin->segment;

    int64_t tmp = cstyle_order;
    grid_index.resize(ndim);
    grid_offset.resize(ndim);
    shape.resize(ndim);
    for (int i = ndim - 1; i >= 0; i--) {
      auto seg_size  = DIV_CEIL(origin->shape[i], grid_segment[i]);
      grid_index[i]  = tmp % grid_segment[i];
      grid_offset[i] = grid_index[i] * seg_size;
      shape[i]       = std::min(seg_size, origin->shape[i] - grid_offset[i]);
      offset += grid_offset[i] * strides[i];
      tmp /= grid_segment[i];
    }
  }
};