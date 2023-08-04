#pragma once
#include <cassert>
#include <common.h>
#include <mutex>

static std::mutex _mu;

class CudaContextManager {
  // cuda environment manager
  // get the handles & streams from its singleton instance
private:
  static constexpr int STREAMS_CAP = 4;
  static CudaContextManager *_manager;
  cudaStream_t *_streams;
  ncclComm_t comm;
  int local_rank;

  CudaContextManager() {
    // create singleton instance
    // always assume we work on one device, aka index==0
    CUDA_SAFE_CALL(cudaSetDevice(0));
    _streams = new cudaStream_t[STREAMS_CAP];
    for (int i = 0; i < STREAMS_CAP; i++) {
      CUDA_SAFE_CALL(cudaStreamCreate(&_streams[i]));
    }
  }

public:
  static CudaContextManager *get() {
    if (_manager != nullptr) // most cases
      return _manager;
    std::lock_guard<std::mutex> guard(_mu);
    if (_manager == nullptr)
      _manager = new CudaContextManager();
    return _manager;
  }

  cudaStream_t stream(int idx = 0) {
    assert(idx < STREAMS_CAP);
    return _streams[idx];
  }

  void sync(int idx = 0) {
    assert(idx < STREAMS_CAP);
    CUDA_SAFE_CALL(cudaStreamSynchronize(_streams[idx]));
  }

  void setCommWorld(std::pair<ncclComm_t, int> _comm) {
    comm = _comm.first;
    local_rank = _comm.second;
  }

  std::pair<ncclComm_t, int> getCommWorld() {
    return std::make_pair(comm, local_rank);
  }
};

CudaContextManager *CudaContextManager::_manager = nullptr;