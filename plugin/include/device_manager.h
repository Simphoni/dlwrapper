#pragma once
#include "common.h"
#include <algorithm>
#include <map>
#include <memory>
#include <mutex>
#include <semaphore.h>
#include <string>
#include <vector>

#include <sys/ipc.h>
#include <sys/sem.h>
#include <sys/shm.h>
#include <sys/types.h>
static std::mutex _mu;

class DeviceContextManager;
class NicoProcessGroup;

class DeviceContextManager {
  // device environment manager - cuda & nico process groups
  // get the handles & streams from its singleton instance
  // see src/device_manager.cpp for implementation
private:
  static constexpr int MAX_PROCS = 8;
  static constexpr int STREAMS_CAP = 4;
  static constexpr int IPC_KEY_BASE = 0xbada991e; // bad apple
  static DeviceContextManager *_manager;
  cudaStream_t *_streams;
  ncclComm_t comm;
  bool initialized;
  bool peerAccessEnabled;

  // topo info
  int worldSize;
  int worldRank;   // rank in world
  int nodeCount;   // node count in world
  int nodeRank;    // node rank in world
  int deviceCount; // devices per node
  int localRank;   // rank in node

  // derived communication groups
  std::vector<std::shared_ptr<NicoProcessGroup>> procGroups;

  // internal profiler with cuda initiatives
  std::map<std::string, cudaEvent_t> _events[STREAMS_CAP];
  std::map<std::string, std::pair<float, int>> _cumtime;

  DeviceContextManager() {
    // create singleton instance
    initialized = false;
    peerAccessEnabled = false;
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

  // initialize communicator && a default process group
  void set_comm_world(std::pair<ncclComm_t, int> _comm);
  void enable_peer_access();
  int create_process_group(const std::vector<int> &members);

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

  std::shared_ptr<NicoProcessGroup> get_process_group(int idx) const {
    assert(idx < procGroups.size());
    return procGroups[idx];
  }
  int get_local_rank() const { return localRank; }
  int get_world_size() const { return worldSize; }
  // returns the number of devices per node
  int get_device_count() const { return deviceCount; }
  int get_world_rank() const { return worldRank; }
  bool peer_access_enabled() const { return peerAccessEnabled; }

  // place a notifier in the specified stream,
  // bind with a std::string to identifiy
  void event_start(std::string s, int idx = 0);
  float event_stop(std::string s, int idx = 0);
  void manual_record(std::string s, float ellapsed_ms);
  void export_summary() const;
};

class NicoProcessGroup {
  //! not thread-safe
  // see src/device_manager.cpp for implementation
private:
  static constexpr int MAX_PROCS = 8;
  static constexpr int IPC_PROC_SEG_SIZE = CUDA_IPC_HANDLE_SIZE << 2;
  static constexpr int IPC_SHMEM_SEG_SIZE = IPC_PROC_SEG_SIZE * MAX_PROCS;
  int groupRank;
  bool isLocal;
  bool isInGroup;
  bool isLeader;
  // for groups that sit in the same node, IPC can be utilized
  bool ipcInitialized;
  std::vector<int> members;
  int memberNum;

  // IPC data structures
  int shmid, semid;
  char *shmdata;
  char recvbuf[IPC_SHMEM_SEG_SIZE];
  int IPC_KEY;

  // nico's process groups are managed by DeviceContextManager
  friend class DeviceContextManager;
  NicoProcessGroup(const std::vector<int> &_members, int ipc_key);
  void ipc_init_process_group();

public:
  bool is_local() { return isLocal; }
  void *ipc_allgather(const void *input, size_t bytes);

  // input: device pointers
  // output: gathered device pointers from other processes
  std::vector<std::vector<void *>>
  ipc_allgather_device_pointer(const std::vector<void *> &ptrs);
};