#pragma once

#include "misc.h"
#include "nv_common.h"
#include <memory>
#include <mutex>
#include <vector>

#include <chrono>
#include <thread>

class HostMemoryManager {
  // manage pinned memory
public:
  static const size_t TEMP_BUFFER_SIZE = 1lu * 1024 * 1024 * 1024;
  static const int TEMP_BUFFER_NUM     = 8;
  // TODO: enable scaling temp_buffer_num

private:
  // pinned
  char *temp_buffer[TEMP_BUFFER_NUM];
  std::vector<int> free_temp_buffer;
  std::mutex mu_get;

  // TODO: support reusing permanent buffers
  std::vector<std::pair<char *, size_t>> permanent_buffers;

  static std::mutex _mu;
  static HostMemoryManager *_manager;

  HostMemoryManager() {
    char *tmp;
    CUDA_SAFE_CALL(cudaMallocHost(&tmp, TEMP_BUFFER_SIZE * TEMP_BUFFER_NUM * 2));
    free_temp_buffer.reserve(TEMP_BUFFER_NUM);
    for (int i = 0; i < TEMP_BUFFER_NUM; i++) {
      temp_buffer[i] = tmp + i * TEMP_BUFFER_SIZE * 2;
      free_temp_buffer.push_back(i);
    }
  }

public:
  static HostMemoryManager *get();

  ~HostMemoryManager() { cudaFreeHost(temp_buffer[0]); }

  char *get_temporary_buffer() {
    char *ret = nullptr;
    while (true) {
      mu_get.lock();
      if (!free_temp_buffer.empty()) {
        ret = temp_buffer[free_temp_buffer.back()];
        free_temp_buffer.pop_back();
      }
      mu_get.unlock();
      if (ret != nullptr)
        return ret;
      std::this_thread::sleep_for(std::chrono::microseconds(500));
    }
  }

  char *get_permanent_buffer(size_t bytes) {
    char *ret = nullptr;
    CUDA_SAFE_CALL(cudaMallocHost(&ret, bytes));
    return ret;
  }

  void release_temporary_buffer();
};