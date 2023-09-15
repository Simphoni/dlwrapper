#include "tensor.h"
#include "device_manager.h"
#include "host_manager.h"
#include <chrono>
using namespace std::chrono_literals;

std::string BaseTensor::to_string() {
  std::string ret = "Tensor(shape=[";
  for (auto &dim : shape) {
    ret += std::to_string(dim) + ", ";
  }
  ret += "], strides=[";
  for (auto &stride : strides) {
    ret += std::to_string(stride) + ", ";
  }
  ret += "])";
  return ret;
}

torch::Tensor BaseTensor::torch_get_contiguous(MemoryType memtype) const {
  if (memtype == MemoryType::DEVICE && device_ptr == nullptr) {
    throw std::runtime_error("BaseTensor::torch_get_contiguous: device_ptr is nullptr");
  }
  if (memtype == MemoryType::PINNED && pinned_ptr == nullptr) {
    throw std::runtime_error("BaseTensor::torch_get_contiguous: pinned_ptr is nullptr");
  }
  torch::TensorOptions options;
  auto dtype = storage->get_dtype();
  // add dtype option
  if (dtype == "Float") {
    options = options.dtype(torch::kFloat32);
  } else if (dtype == "Half") {
    options = options.dtype(torch::kHalf);
  } else {
    throw std::runtime_error("BaseTensor::torch_get_contiguous: unsupported dtype " + dtype);
  }
  options = options.layout(torch::kStrided).requires_grad(requires_grad);
  // add device option
  if (memtype == MemoryType::DEVICE) {
    options = options.device(torch::kCUDA, 0);
    return torch::from_blob(device_ptr, shape, stride_from_shape(shape), options);
  } else {
    options = options.device(torch::kCPU);
    return torch::from_blob(pinned_ptr, shape, stride_from_shape(shape), options);
  }
}

void OriginTensor::issue_movement() {
  // run by a new thread
  assert(memtype_next == MemoryType::DEVICE);
  size_t dsize  = storage->get_dsize();
  size_t bytes  = numel(shape) * dsize;
  auto HMM      = HostMemoryManager::get();
  auto DMM      = DeviceMemoryManager::get();
  char *src     = storage->get_data() + dsize * offset;
  char *pin_ptr = HMM->get_temporary_buffer();
  char *dev_ptr = DMM->get_fixed(bytes);
  auto stream   = DMM->get_move_stream();

  int64_t unit_size  = HostMemoryManager::TEMP_BUFFER_SIZE;
  int64_t rounds     = (bytes - 1) / unit_size + 1;
  char *to_pin       = pin_ptr;
  char *to_dev       = pin_ptr + unit_size;
  int64_t to_pin_rem = bytes;
  int64_t to_dev_rem = bytes;
  cudaEvent_t event;
  cudaEventCreate(&event);
  for (int i = 0; i <= rounds; i++) {
    if (i > 0) {
      int64_t copy_size = std::min(unit_size, to_dev_rem);
      CUDA_SAFE_CALL(cudaMemcpyAsync(dev_ptr + (i - 1) * unit_size, to_dev, copy_size,
                                     cudaMemcpyHostToDevice, stream));
      CUDA_SAFE_CALL(cudaEventRecord(event, stream));
      to_dev_rem -= copy_size;
    }
    if (i < rounds) {
      int64_t copy_size = std::min(unit_size, to_pin_rem);
      memcpy(to_pin, src + i * unit_size, copy_size);
      to_pin_rem -= copy_size;
    }
    if (i > 0) {
      CUDA_SAFE_CALL(cudaEventSynchronize(event));
    }
    std::swap(to_pin, to_dev);
  }
  cudaEventDestroy(event);
  device_ptr = dev_ptr;
  memtype    = memtype_next;
}

bool OriginTensor::thread_running() {
  bool ret = move_future.valid() && move_future.wait_for(0ms) != std::future_status::ready;
  return ret;
}

void OriginTensor::wait() {
  if (move_future.valid()) {
    move_future.get(); // this will invalidate move_future
  }
}

bool OriginTensor::move_to(MemoryType _memtype) {
  if (managed) {
    INFO("Cannot move managed tensor. Use TensorGrid to control data movement%s", ".");
    return false;
  }
  if (_memtype == MemoryType::DISK || _memtype == MemoryType::PINNED || _memtype == memtype) {
    // a unmanaged tensor do not need to stay in pinned memory
    // pinned memory is used simply for temporary data transfer
    // also we cannot explicitly move a tensor to disk
    return true;
  }
  if (_memtype < memtype) {
    // TODO: just release memory
    return true;
  }
  if (!thread_running()) {
    memtype_next = _memtype;
    move_future  = std::async(std::launch::async, &OriginTensor::issue_movement, this);
    return true;
  } else {
    return false;
  }
}

void DerivedTensor::issue_movement() {
  size_t dsize = storage->get_dsize();
  size_t bytes = numel(shape) * dsize;
  auto HMM     = HostMemoryManager::get();
  auto DMM     = DeviceMemoryManager::get();
  if (pinned_ptr == nullptr && memtype_next >= MemoryType::PINNED) {
    pinned_ptr = HMM->get_permanent_buffer(bytes);
  }
  if (device_ptr == nullptr && memtype_next >= MemoryType::DEVICE) {
    device_ptr = DMM->get_fixed(bytes);
  }
  cudaStream_t stream = DMM->get_move_stream();
  cudaEvent_t event;
  cudaEventCreate(&event);
  if (memtype_next == MemoryType::DEVICE) {
    if (memtype == MemoryType::PINNED) {
      CUDA_SAFE_CALL(
          cudaMemcpyAsync(device_ptr, pinned_ptr, bytes, cudaMemcpyHostToDevice, stream));
      CUDA_SAFE_CALL(cudaEventRecord(event, stream));
    }
  } else if (memtype_next == MemoryType::PINNED) {
    // be careful, it should be hard to implement
    shape_t compress_shape  = shape;
    shape_t compress_stride = strides;
    int dim                 = shape.size();
    int64_t segsize         = 1;
    for (int i = dim - 1; i >= 0; i--) {
      if (compress_stride[i] == segsize) {
        segsize *= compress_shape[i];
        compress_shape.pop_back();
        compress_stride.pop_back();
      } else {
        break;
      }
    }
    if (compress_shape.size() == 0) {
      compress_shape.push_back(1);
      compress_stride.push_back(segsize);
    }
    std::reverse(compress_shape.begin(), compress_shape.end());
    std::reverse(compress_stride.begin(), compress_stride.end());

    dim                = compress_shape.size();
    int64_t segnum     = numel(compress_shape);
    int64_t cur_offset = 0;
    int64_t segbytes   = segsize * dsize;
    char *disk_ptr     = storage->get_data() + dsize * offset;
    shape_t cur_index(dim, 0);
    for (int64_t i = 0; i < segnum; i++) {
      memcpy(pinned_ptr + i * segbytes, disk_ptr + cur_offset * dsize, segbytes);
      cur_index[0]++;
      cur_offset += compress_stride[0];
      for (int j = 0; j < dim - 1; j++) {
        if (cur_index[j] == compress_shape[j]) {
          cur_index[j] = 0;
          cur_index[j + 1]++;
          cur_offset += compress_stride[j + 1] - compress_shape[j] * compress_stride[j];
        } else {
          break;
        }
      }
    }
  }
  memtype = memtype_next;
}

bool DerivedTensor::thread_running() {
  bool ret = move_future.valid() && move_future.wait_for(0ms) != std::future_status::ready;
  return ret;
}

void DerivedTensor::wait() {
  if (move_future.valid()) {
    move_future.get(); // this will invalidate move_future
  }
}

bool DerivedTensor::move_to(MemoryType _memtype) {
  if (memtype == _memtype) {
    return true;
  } else if (memtype < _memtype) {
    if (!thread_running()) {
      memtype_next = _memtype;
      move_future  = std::async(std::launch::async, &DerivedTensor::issue_movement, this);
      return true;
    } else {
      return false;
    }
  } else {
    // TODO: release memory
    return true;
  }
}