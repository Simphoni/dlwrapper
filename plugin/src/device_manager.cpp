#include "device_manager.h"

DeviceContextManager *DeviceContextManager::_manager = nullptr;

void DeviceContextManager::set_comm_world(std::pair<ncclComm_t, int> _comm) {
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
  std::vector<int> members;
  for (int i = 0; i < deviceCount; ++i) {
    members.push_back(i + nodeRank * deviceCount);
  }
  // initialize a default process group
  procGroups.emplace_back(std::shared_ptr<NicoProcessGroup>(
      new NicoProcessGroup(members, IPC_KEY_BASE + 0)));
  _mu.unlock();
}

void DeviceContextManager::enable_peer_access() {
  if (peerAccessEnabled) {
    return;
  }
  _mu.lock();
  for (int i = 0; i < deviceCount; i++) {
    if (i == localRank) {
      continue;
    }
    CUDA_SAFE_CALL(cudaDeviceEnablePeerAccess(i, 0));
  }
  peerAccessEnabled = true; // pg will check this flag
  for (const auto &pg : procGroups) {
    if (pg->is_local())
      pg->ipc_init_process_group();
  }
  _mu.unlock();
}

int DeviceContextManager::create_process_group(
    const std::vector<int> &members) {
  _mu.lock();
  assert(initialized);
  int ret = procGroups.size();
  procGroups.emplace_back(std::shared_ptr<NicoProcessGroup>(
      new NicoProcessGroup(members, IPC_KEY_BASE + ret)));
  if (peerAccessEnabled) {
    procGroups[ret]->ipc_init_process_group();
  }
  _mu.unlock();
  return ret;
}

void DeviceContextManager::event_start(std::string s, int idx) {
  cudaEvent_t e;
  cudaEventCreate(&e);
  cudaEventRecord(e, _streams[idx]);
  _mu.lock();
  assert(_events[idx].find(s) == _events[idx].end());
  _events[idx][s] = e;
  _mu.unlock();
}

float DeviceContextManager::event_stop(std::string s, int idx) {
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

void DeviceContextManager::manual_record(std::string s, float ellapsed_ms) {
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

void DeviceContextManager::export_summary() const {
  _mu.lock();
  DEBUG("rank[%d]: time summary for %ld events: (item / total_ms / "
        "evoke_num)",
        localRank, _cumtime.size());
  for (auto const kv : _cumtime) {
    DEBUG("rank[%d]: %s\t%f\t%d", localRank, kv.first.data(), kv.second.first,
          kv.second.second);
  }
  _mu.unlock();
}

// -------------------------- NicoProcessGroup --------------------------

NicoProcessGroup::NicoProcessGroup(const std::vector<int> &_members,
                                   int ipc_key) {
  IPC_KEY = ipc_key;
  members.resize(_members.size());
  std::copy(_members.begin(), _members.end(), members.begin());
  std::sort(members.begin(), members.end());
  auto it = std::unique(members.begin(), members.end());
  members.erase(it, members.end());

  memberNum = members.size();
  auto manager = DeviceContextManager::get();
  int deviceCount = manager->get_device_count();
  int worldRank = manager->get_world_rank();
  groupRank = -1;
  for (int i = 0; i < memberNum; ++i) {
    assert(members[i] >= 0 && members[i] < MAX_PROCS);
    if (members[i] == worldRank) {
      isInGroup = true;
      groupRank = i;
    }
  }
  if (!isInGroup)
    return;
  isLeader = (groupRank == 0);
  // check if all members are in the same node
  isLocal = true;
  for (auto const &rank : members) {
    if (rank / deviceCount != worldRank / deviceCount) {
      isLocal = false;
      break;
    }
  }
  if (!isLocal) {
    return;
  }
  if (manager->peer_access_enabled()) {
    ipc_init_process_group();
  }
}

// ~ 18us
void NicoProcessGroup::ipc_init_process_group() {
  // ipc init is called when:
  // 1. all processes are in the same node
  // 2. peer access is enabled
  // 3. the process is in the group
  if (ipcInitialized || !isLocal || !isInGroup ||
      !DeviceContextManager::get()->peer_access_enabled()) {
    return;
  }
  shmid = shmget(IPC_KEY, IPC_SHMEM_SEG_SIZE, 0666 | IPC_CREAT);
  assert(shmid != -1);
  /*
  ** sem[0] write lock
  ** sem[1] read lock
  ** barrier is performed by all process waiting for a sem to reach 0
  */
  semid = semget(IPC_KEY, 2, 0666 | IPC_CREAT);
  assert(semid != -1);
  shmdata = (char *)shmat(shmid, NULL, 0);
  ipcInitialized = true;
}

struct sembuf write_wait = {0, -1, 0};
struct sembuf write_exhaust = {0, 0, 0};
struct sembuf read_wait = {1, -1, 0};
struct sembuf read_exhaust = {1, 0, 0};
void *NicoProcessGroup::ipc_allgather(const void *input, size_t bytes) {
  // gather all `input` data into `recvbuf`
  // shared memory needn't remap to ensure coherence
  assert(bytes <= IPC_PROC_SEG_SIZE);
  struct sembuf write_free = {0, (short)memberNum, 0};
  struct sembuf read_free = {1, (short)memberNum, 0};

  // data is placed tightly in shared memory
  int segmentOffset = groupRank * bytes;
  if (isLeader) {
    // enable write lock
    semop(semid, &write_free, 1);
  }
  // write begin after master has enabled write lock
  if (input != NULL) {
    memcpy(shmdata + segmentOffset, input, bytes);
  } else {
    memset(shmdata + segmentOffset, 0, bytes);
  }
  // signal write completion
  semop(semid, &write_wait, 1);
  // wait until all processes have written
  // read can begin immediately
  semop(semid, &write_exhaust, 1);
  if (isLeader) {
    semop(semid, &read_free, 1);
  }
  memcpy(recvbuf, shmdata, bytes * memberNum);
  // signal read completion
  semop(semid, &read_wait, 1);
  // wait until all processes have read
  semop(semid, &read_exhaust, 1);
  return (void *)recvbuf;
}

std::vector<std::vector<void *>> NicoProcessGroup::ipc_allgather_device_pointer(
    const std::vector<void *> &ptrs) {
  int items = ptrs.size();
  std::vector<cudaIpcMemHandle_t> sendbuf;
  sendbuf.reserve(items);
  for (auto devPtr : ptrs) {
    cudaIpcMemHandle_t handle;
    cudaIpcGetMemHandle(&handle, devPtr);
    sendbuf.push_back(handle);
  }
  cudaIpcMemHandle_t *recvbuf = (cudaIpcMemHandle_t *)ipc_allgather(
      sendbuf.data(), sizeof(cudaIpcMemHandle_t) * sendbuf.size());
  std::vector<std::vector<void *>> result;
  result.resize(memberNum);
  for (int i = 0; i < memberNum; i++) {
    result[i].resize(items);
    for (int j = 0; j < items; j++) {
      cudaIpcOpenMemHandle(&result[i][j], recvbuf[i * items + j],
                           cudaIpcMemLazyEnablePeerAccess);
    }
  }
  return result;
}