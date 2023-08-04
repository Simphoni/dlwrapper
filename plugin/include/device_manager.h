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
  static CudaContextManager *p_manager;
  cudaStream_t *p_streams;
  ncclComm_t comm;
  int local_rank;

  CudaContextManager() {
    // create singleton instance
    // always assume we work on one device, aka index==0
    CUDA_SAFE_CALL(cudaSetDevice(0));
    p_streams = new cudaStream_t[STREAMS_CAP];
    for (int i = 0; i < STREAMS_CAP; i++) {
      CUDA_SAFE_CALL(cudaStreamCreate(&p_streams[i]));
    }
  }

public:
  static CudaContextManager *get() {
    if (p_manager != nullptr) // most cases
      return p_manager;
    std::lock_guard<std::mutex> guard(_mu);
    if (p_manager == nullptr)
      p_manager = new CudaContextManager();
    return p_manager;
  }

  cudaStream_t getStream(int idx) {
    assert(idx < STREAMS_CAP);
    return p_streams[idx];
  }

  void setCommWorld(std::pair<ncclComm_t, int> _comm) {
    comm = _comm.first;
    local_rank = _comm.second;
  }

  std::pair<ncclComm_t, int> getCommWorld() {
    return std::make_pair(comm, local_rank);
  }
};

CudaContextManager *CudaContextManager::p_manager = nullptr;