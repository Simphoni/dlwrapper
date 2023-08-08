#pragma once
#include <cassert>
#include <common.h>
#include <map>
#include <mutex>
#include <string>

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
  int world_size;
  // internal profiler with cuda initiatives
  std::map<std::string, cudaEvent_t> _events[STREAMS_CAP];
  std::map<std::string, std::pair<float, int>> _cumtime;

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
  // return singleton instance
  static CudaContextManager *get() {
    if (_manager != nullptr) // most cases
      return _manager;
    _mu.lock();
    if (_manager == nullptr)
      _manager = new CudaContextManager();
    _mu.unlock();
    return _manager;
  }

  // get the specified cuda stream
  cudaStream_t stream(int idx = 0) const {
    assert(idx < STREAMS_CAP);
    return _streams[idx];
  }

  // sync the specified cuda stream
  void sync(int idx = 0) {
    assert(idx < STREAMS_CAP);
    CUDA_SAFE_CALL(cudaStreamSynchronize(_streams[idx]));
  }

  // initialize communicator and other info
  void setCommWorld(std::pair<ncclComm_t, int> _comm) {
    _mu.lock();
    comm = _comm.first;
    local_rank = _comm.second >> 12;
    world_size = _comm.second & ((1 << 12) - 1);
    _mu.unlock();
  }

  ncclComm_t getCommWorld() const { return comm; }

  int getRank() const { return local_rank; }

  int getWorldSize() const { return world_size; }

  // place a notifier in the specified stream,
  // bind with a std::string to identifiy
  void event_start(std::string s, int idx = 0) {
    cudaEvent_t e;
    cudaEventCreate(&e);
    cudaEventRecord(e, _streams[idx]);
    _mu.lock();
    assert(_events[idx].find(s) == _events[idx].end());
    _events[idx][s] = e;
    _mu.unlock();
  }

  float event_stop(std::string s, int idx = 0) {
    cudaEvent_t e;
    cudaEventCreate(&e);
    cudaEventRecord(e, _streams[idx]);
    sync(idx);
    _mu.lock();
    auto it = _events[idx].find(s);
    assert(it != _events[idx].end());
    auto start_e = it->second;
    _events[idx].erase(it);
    float ellapsed_ms = 0;
    cudaEventElapsedTime(&ellapsed_ms, start_e, e);
    auto map_it = _cumtime.find(s);
    if (map_it == _cumtime.end()) {
      _cumtime[s] = std::make_pair(ellapsed_ms, 1);
    } else {
      map_it->second.first += ellapsed_ms;
      map_it->second.second++;
    }
    _mu.unlock();
    cudaEventDestroy(start_e);
    cudaEventDestroy(e);
    return ellapsed_ms;
  }

  void manual_record(std::string s, float ellapsed_ms) {
    _mu.lock();
    auto map_it = _cumtime.find(s);
    if (map_it == _cumtime.end()) {
      _cumtime[s] = std::make_pair(ellapsed_ms, 1);
    } else {
      map_it->second.first += ellapsed_ms;
      map_it->second.second++;
    }
    _mu.unlock();
  }

  void export_summary() const {
    _mu.lock();
    DEBUG(
        "rank[%d]: time summary for %ld events: (item / total_ms / evoke_num)",
        local_rank, _cumtime.size());
    for (auto const kv : _cumtime) {
      DEBUG("rank[%d]: %s\t%f\t%d", local_rank, kv.first.data(),
            kv.second.first, kv.second.second);
    }
    _mu.unlock();
  }
};

CudaContextManager *CudaContextManager::_manager = nullptr;