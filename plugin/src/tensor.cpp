#include "tensor.h"
#include "device_manager.h"
#include "host_manager.h"
#include <future>
#include <thread>

std::string BaseTensor::to_string() {
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

void OriginTensor::issue_move_thread() {
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
  p_dev = dev_ptr;
  mu.unlock();
}

void OriginTensor::move_to(MemoryType _memtype) {
  if (managed) {
    throw std::runtime_error("OriginTensor: cannot move managed tensor.");
    return;
  }
  if (_memtype == MemoryType::DISK || _memtype == MemoryType::PINNED || _memtype == memtype) {
    // a unmanaged tensor do not need to stay in pinned memory
    // pinned memory is used simply for temporary data transfer
    // also we cannot explicitly move a tensor to disk
    return;
  }
  if (_memtype < memtype) {
    // TODO: just release memory
    return;
  }
  if (mu.try_lock()) {
    memtype_next = _memtype;
    // std::async(std::launch::async, &OriginTensor::issue_move_thread, this);
  } else {
    throw std::runtime_error("OriginTensor: cannot issue multiple move threads.");
  }
}