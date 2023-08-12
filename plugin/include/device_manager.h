#pragma once
#include "common.h"
#include <map>
#include <mutex>
#include <semaphore.h>
#include <string>
#include <sys/ipc.h>
#include <sys/sem.h>
#include <sys/shm.h>
#include <sys/types.h>
#include <vector>

static std::mutex _mu;

class DeviceContextManager {
  // device environment manager - cuda & unix ipc
  // get the handles & streams from its singleton instance
private:
  static constexpr int MAX_PROCS = 8;
  static constexpr int STREAMS_CAP = 4;
  static constexpr int IPC_KEY = 0xbada991e; // bad apple
  static constexpr int IPC_SHMEM_SEG_SIZE = 1 << 10;
  static constexpr int IPC_PROC_SEG_SIZE = 1 << 6;
  static DeviceContextManager *_manager;
  cudaStream_t *_streams;
  ncclComm_t comm;
  bool initialized;
  bool peer_access_enabled;

  // topo info
  int worldSize;
  int worldRank;   // rank in world
  int nodeCount;   // node count in world
  int nodeRank;    // node rank in world
  int deviceCount; // devices per node
  int localRank;   // rank in node

  // IPC data structures
  int shmid, semid;
  int segment_offset;
  char *shmdata;
  char recvbuf[IPC_SHMEM_SEG_SIZE];

  // internal profiler with cuda initiatives
  std::map<std::string, cudaEvent_t> _events[STREAMS_CAP];
  std::map<std::string, std::pair<float, int>> _cumtime;

  DeviceContextManager() {
    // create singleton instance
    initialized = false;
    peer_access_enabled = false;
    cudaGetDeviceCount(&deviceCount);
    assert(deviceCount > 0);
  }

public:
  // return singleton instance
  static DeviceContextManager *get() {
    if (_manager != nullptr) // most cases
      return _manager;
    _mu.lock();
    if (_manager == nullptr)
      _manager = new DeviceContextManager();
    _mu.unlock();
    return _manager;
  }

  // initialize communicator and other info
  void set_comm_world(std::pair<ncclComm_t, int> _comm) {
    _mu.lock();
    assert(initialized == false);
    initialized = true;
    comm = _comm.first;
    worldRank = _comm.second >> 12;
    worldSize = _comm.second & ((1 << 12) - 1);
    nodeCount = worldSize / deviceCount;
    assert(nodeCount * deviceCount == worldSize);
    localRank = worldRank % deviceCount;
    nodeRank = worldRank / deviceCount;

    _streams = new cudaStream_t[STREAMS_CAP];
    for (int i = 0; i < STREAMS_CAP; i++) {
      CUDA_SAFE_CALL(cudaStreamCreate(&_streams[i]));
    }
    _mu.unlock();
  }

  void ipc_init_process_group();

  void enable_peer_access() {
    if (peer_access_enabled) {
      return;
    }
    _mu.lock();
    for (int i = 0; i < deviceCount; i++) {
      if (i == localRank) {
        continue;
      }
      CUDA_SAFE_CALL(cudaDeviceEnablePeerAccess(i, 0));
    }
    ipc_init_process_group();
    peer_access_enabled = true;
    _mu.unlock();
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

  ncclComm_t get_comm_world() const { return comm; }

  int get_local_rank() const { return localRank; }

  int get_world_size() const { return worldSize; }

  // place a notifier in the specified stream,
  // bind with a std::string to identifiy
  void event_start(std::string s, int idx = 0);

  float event_stop(std::string s, int idx = 0);

  void manual_record(std::string s, float ellapsed_ms);

  void export_summary() const;

  // ~ 18us
  char *ipc_allgather(const void *input, size_t bytes);

  void ipc_allgather_device_pointer(std::vector<void *> &ptrs);
};