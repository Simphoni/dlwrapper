// Tensor abstraction api. Provide big-endian tensor abstraction.
// ?Is it a good choice to base internal logic on big-endian

#pragma once

#include "misc.h"
#include <cassert>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#define DIV_CEIL(x, y) ((x - 1) / (y) + 1)

using shape_t = std::vector<int64_t>;

template <> struct std::hash<shape_t> {
  std::size_t operator()(const shape_t &shape) const noexcept {
    std::size_t ret = 0;
    for (const auto &dim : shape) {
      ret = ((ret % 22024454644109u) << 18) ^ std::hash<int64_t>()(dim);
    }
    return ret;
  }
};

// return the number of elements in a shape
inline int64_t numel(const shape_t &shape) {
  int64_t ret = 1;
  for (const auto &dim : shape) {
    ret *= dim;
  }
  return ret;
}

enum MemoryType : uint8_t {
  DISK     = 1,
  PAGEABLE = 2,
  PINNED   = 3,
  DEVICE   = 4,
};

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

  size_t hash() { return (size_t)(data) ^ std::hash<std::string>()(dtype) ^ numel ^ dsize; }
  char *get_data() const noexcept { return data; }
  uint64_t get_dsize() const noexcept { return dsize; }
#undef bytes
};

class BaseTensor {
protected:
  std::shared_ptr<CachedStorage> storage;
  int64_t offset;
  shape_t shape, strides;
  int64_t ndim;
  bool requires_grad;
  bool is_cstyle;

public:
  BaseTensor() = default;
  BaseTensor(std::shared_ptr<CachedStorage> storage, int64_t offset, shape_t shape, shape_t strides,
             bool requires_grad)
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

  std::string to_string();

  shape_t get_shape() const noexcept { return shape; }

  size_t hash() {
    return storage->hash() ^ offset ^ std::hash<shape_t>()(shape) ^ std::hash<shape_t>()(strides) ^
           ndim ^ requires_grad ^ is_cstyle;
  }
};

class TensorGrid;

class OriginTensor : public BaseTensor {
private:
  shape_t segment;
  bool managed;
  std::string name;
  // memtype is used for unmanaged tensor
  // when tensor is managed, use tensor grid to control data movement
  MemoryType memtype, memtype_next;
  std::mutex mu; // guard for a running thread
  char *p_dev;

  friend class DerivedTensor;

public:
  OriginTensor(std::shared_ptr<CachedStorage> storage, int64_t offset, shape_t shape,
               shape_t strides, bool requires_grad)
      : BaseTensor(storage, offset, shape, strides, requires_grad), managed(false),
        memtype(MemoryType::DISK) {
    segment = shape_t(ndim, 1);
  }

  std::shared_ptr<OriginTensor> clone() const {
    auto ret     = std::make_shared<OriginTensor>(storage, offset, shape, strides, requires_grad);
    ret->segment = segment;
    return ret;
  }

  void set_name(std::string s) {
    if (name.length() == 0)
      name = s;
  }

  void set_segment(shape_t segment) {
    assert((int64_t)segment.size() == ndim);
    this->segment = segment;
  }
  shape_t get_segment() const noexcept { return segment; }
  std::shared_ptr<TensorGrid> create_tensor_grid() { return std::make_shared<TensorGrid>(clone()); }
  void set_as_managed() { managed = true; }

  void move_to(MemoryType _memtype);
  void issue_move_thread();
  size_t hash() { return BaseTensor::hash() ^ std::hash<std::string>()(name); }
};

class DerivedTensor : public BaseTensor {
private:
  std::shared_ptr<OriginTensor> origin;
  int64_t grid_order;
  shape_t grid_index, grid_offset, grid_segment;

public:
  DerivedTensor(std::shared_ptr<OriginTensor> origin, int64_t grid_order)
      : origin(origin), grid_order(grid_order) {
    assert(origin->is_cstyle);
    is_cstyle    = false; // do not derive!
    ndim         = origin->ndim;
    storage      = origin->storage;
    offset       = origin->offset;
    strides      = origin->strides;
    grid_segment = origin->segment;

    int64_t tmp = grid_order;
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
  size_t hash() {
    return origin->hash() ^ std::hash<shape_t>()(grid_segment) ^ std::hash<shape_t>()(grid_index) ^
           grid_order;
  }
};

class TensorGrid {
private:
  static const int64_t MAX_MANAGEABLE_GRID = 1 << 16;
  std::shared_ptr<OriginTensor> origin;
  shape_t segment;
  // data movement can be controlled by other threads
  // mutex do not support copy, so use unordered_map
  std::vector<MemoryType> states;
  std::unordered_map<int64_t, std::mutex> locks;

public:
  TensorGrid(std::shared_ptr<OriginTensor> _origin) {
    origin  = _origin->clone(); // Always make sure the origin is owned
    segment = _origin->get_segment();
    if (numel(segment) > MAX_MANAGEABLE_GRID) {
      throw std::runtime_error("TensorGrid: segment count is not manageable.");
      return;
    }
    states = std::vector<MemoryType>(numel(segment), MemoryType::DISK);
  }
  std::shared_ptr<DerivedTensor> get_slice(int64_t grid_order) {
    return std::make_shared<DerivedTensor>(origin, grid_order);
  }
};