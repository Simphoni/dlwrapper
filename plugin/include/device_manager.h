#pragma once
#include "common.h"
#include <algorithm>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
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
public:
  static constexpr int MAX_PROCS = 8;
  static constexpr int MAX_STREAMS = 4;
  static constexpr int IPC_KEY_MASTER = 0x00001000;
  static constexpr int IPC_KEY_PER_GROUP = 16;

private:
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
  std::vector<NicoProcessGroup *> procGroups;

  // internal profiler with cuda initiatives
  std::map<std::string, cudaEvent_t> _events[MAX_STREAMS];
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

  void destroy_existing_pg();

  // initialize communicator && a default process group
  void set_comm_world(std::pair<ncclComm_t, int> _comm);
  void enable_peer_access();
  int create_process_group(const std::vector<int> &members);
  NicoProcessGroup *get_process_group(int idx) const {
    assert(idx < (int)procGroups.size());
    return procGroups[idx];
  }

  // get the specified cuda stream
  cudaStream_t stream(int idx = 0) const {
    assert(idx < MAX_STREAMS);
    return _streams[idx];
  }
  // sync the specified cuda stream
  void sync(int idx = 0) {
    assert(idx < MAX_STREAMS);
    CUDA_SAFE_CALL(cudaStreamSynchronize(_streams[idx]));
  }

  ncclComm_t get_comm_world() const { return comm; }

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
  //! shared memroy is system-wide persistent, do not change segment size,
  //! otherwise shmget() with fixed key will fail on the next run
  // see src/device_manager.cpp for implementation
public:
  static constexpr int MAX_PROCS = 8;
  static constexpr size_t IPC_SHMEM_SEG_SIZE = 1ul << 12; //! do not change
  static constexpr size_t IPC_PROC_SEG_SIZE = IPC_SHMEM_SEG_SIZE / MAX_PROCS;
  static constexpr int MAX_CUDA_MEM_HANDLES =
      IPC_PROC_SEG_SIZE / CUDA_IPC_HANDLE_SIZE;

private:
  int IPC_KEY_BASE;
  int group_id;

  enum semUsage : int {
    SEM_IPC_ALLGATHER = 0, // IPC allgather before collective calls
    SEM_GPU_ALLGATHER = 1, // GPU allgather barriers
    SEM_TOTAL,
  };

  int groupRank;
  bool isIntraNode;
  bool isInGroup;
  bool isLeader;
  // for groups that sit in the same node, IPC can be utilized
  bool ipcInitialized;
  std::vector<int> members;
  std::queue<void *> remoteDevPtrToClose;
  int memberNum;

  // IPC data structures
  int _shmid;
  int _semid[DeviceContextManager::IPC_KEY_PER_GROUP];
  char *shmdata;

  // nico's process groups are managed by DeviceContextManager
  friend class DeviceContextManager;
  NicoProcessGroup(const std::vector<int> &_members, int ipc_key,
                   int _group_id);
  void ipc_init_process_group();
  ~NicoProcessGroup();

public:
  bool is_intranode() { return isIntraNode; }
  std::vector<char> ipc_allgather(const void *input, size_t bytes);

  // gathered device pointers from other processes
  //! NOTE: this will cause memory mapping that needs to be manually undone
  //! call close_all_mem_handles() to release memory and leave the rest to
  //! cuda_runtime
  std::vector<std::vector<void *>>
  ipc_allgather_device_pointer(const std::vector<void *> &ptrs);
  void close_all_mem_handles();

  void allgather_with_peer_access(char *dst, char *src, int64_t numel_dst,
                                  int64_t numel_src, bool prof = false);
};