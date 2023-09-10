#include "device_manager.h"
#include "mem_handle_manager.h"
#include <chrono>
#include <mutex>
#include <unistd.h>
namespace ch = std::chrono;

static std::mutex _mu; // for singleton instance
DeviceContextManager *DeviceContextManager::_manager = nullptr;

void DeviceContextManager::destroy_existing_pg() {
  _mu.lock();
  INFO("rank[%d]: destroying process groups.", worldRank);
  for (auto pg : procGroups) {
    delete pg;
  }
  INFO("rank[%d]: done.", worldRank);
  procGroups.clear();
  _mu.unlock();
}

DeviceContextManager *DeviceContextManager::get() {
  if (_manager != nullptr) // most cases
    return _manager;
  _mu.lock();
  if (_manager == nullptr)
    _manager = new DeviceContextManager();
  _mu.unlock();
  return _manager;
}

void DeviceContextManager::set_comm_world(std::pair<ncclComm_t, int> _comm) {
  _mu.lock();
  assert(initialized == false);
  initialized = true;
  comm        = _comm.first;
  worldRank   = _comm.second >> 12;
  worldSize   = _comm.second & ((1 << 12) - 1);
  nodeCount   = worldSize / deviceCount;
  assert(nodeCount * deviceCount == worldSize);
  localRank = worldRank % deviceCount;
  nodeRank  = worldRank / deviceCount;

  _streams = new cudaStream_t[MAX_STREAMS];
  for (int i = 0; i < MAX_STREAMS; i++) {
    CUDA_SAFE_CALL(cudaStreamCreate(&_streams[i]));
  }
  std::vector<int> members;
  for (int i = 0; i < deviceCount; ++i) {
    members.push_back(i + nodeRank * deviceCount);
  }
  // initialize a default process group
  procGroups.emplace_back(new NicoProcessGroup(members, IPC_KEY_MASTER + 0 * IPC_KEY_PER_GROUP, 0));
  _mu.unlock();
}

void DeviceContextManager::enable_peer_access() {
  _mu.lock();
  if (peerAccessEnabled) {
    return;
  }
  for (int i = 0; i < deviceCount; i++) {
    if (i == localRank) {
      continue;
    }
    CUDA_SAFE_CALL(cudaDeviceEnablePeerAccess(i, 0));
  }
  peerAccessEnabled = true; // pg will check this flag
  DEBUG("rank[%d] cudaDeviceEnablePeerAccess succeeded.", worldRank);
  for (const auto &pg : procGroups) {
    if (pg->is_intranode())
      pg->ipc_init_process_group();
  }
  _mu.unlock();
}

int DeviceContextManager::create_process_group(const std::vector<int> &members) {
  _mu.lock();
  assert(initialized);
  int ret = procGroups.size();
  procGroups.emplace_back(
      new NicoProcessGroup(members, IPC_KEY_MASTER + ret * IPC_KEY_PER_GROUP, ret));
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

void DeviceContextManager::export_summary() {
  _mu.lock();
  INFO("rank[%d]: time summary for %ld events: (item / total_ms / "
       "evoke_num)",
       localRank, _cumtime.size());
  for (auto const kv : _cumtime) {
    INFO("rank[%d]: %s\t%f\t%d", localRank, kv.first.data(), kv.second.first, kv.second.second);
  }
  _cumtime.clear();
  _mu.unlock();
}

// -------------------------- NicoProcessGroup --------------------------

NicoProcessGroup::NicoProcessGroup(const std::vector<int> &_members, int ipc_key, int _group_id) {
  IPC_KEY_BASE = ipc_key;
  groupId      = _group_id;
  members.resize(_members.size());
  std::copy(_members.begin(), _members.end(), members.begin());
  std::sort(members.begin(), members.end());
  auto it = std::unique(members.begin(), members.end());
  members.erase(it, members.end());

  memberNum       = members.size();
  auto manager    = DeviceContextManager::get();
  int deviceCount = manager->get_device_count();
  int worldRank   = manager->get_world_rank();
  groupRank       = -1;
  for (int i = 0; i < memberNum; ++i) {
    assert(members[i] >= 0 && members[i] < DeviceContextManager::get()->get_world_size());
    if (members[i] == worldRank) {
      isInGroup = true;
      groupRank = i;
    }
  }
  if (!isInGroup)
    return;
  isLeader = (groupRank == 0);
  // check if all members are in the same node
  isIntraNode = true;
  for (auto const &rank : members) {
    if (rank / deviceCount != worldRank / deviceCount) {
      isIntraNode = false;
      break;
    }
  }
  if (!isIntraNode)
    return;
  if (manager->peer_access_enabled()) {
    ipc_init_process_group();
  }
}

NicoProcessGroup::~NicoProcessGroup() {
  if (!ipcInitialized)
    return;
  try {
    shmdt(shmdata);
  } catch (...) { // do nothing
  }
  if (!isLeader)
    return;
  for (int i = 0; i < SEM_TOTAL; i++) {
    try {
      semctl(_semid[i], 0, IPC_RMID);
    } catch (...) { // do nothing
    }
  }
  try {
    shmctl(_shmid, IPC_RMID, NULL);
  } catch (...) { // do nothing
  }
  DEBUG("rank[%d]-group[%d]: Leader successfully deleted shmid && semid.",
        DeviceContextManager::get()->get_world_rank(), groupId);
}

// ~ 18us
void NicoProcessGroup::ipc_init_process_group() {
  // ipc init is called when:
  // 1. all processes are in the same node
  // 2. peer access is enabled
  // 3. the process is in the group
  if (ipcInitialized || !isIntraNode || !isInGroup ||
      !DeviceContextManager::get()->peer_access_enabled()) {
    return;
  }
  if (!isLeader) {
    sleep(1);
  }
  _shmid = shmget(IPC_KEY_BASE, IPC_SHMEM_SEG_SIZE, 0666 | IPC_CREAT);
  assert(_shmid != -1);
  shmdata = (char *)shmat(_shmid, NULL, 0);
  /*
  ** sem[0] write lock
  ** sem[1] read lock
  ** barrier is performed by all process waiting for a sem to reach 0
  */
  int semnum[2] = {2, memberNum};
  std::vector<short> sarr(32, 0);
  for (int i = 0; i < SEM_TOTAL; i++) {
    _semid[i] = semget(IPC_KEY_BASE + i, semnum[i], 0666 | IPC_CREAT);
    assert(_semid[i] != -1);
    if (isLeader) {
      semctl(_semid[i], 0, SETALL, sarr.data());
    }
  }
  ipcInitialized = true;
  DEBUG("rank[%d] IPC is enabled for group[%d]", DeviceContextManager::get()->get_world_rank(),
        groupId);
}

struct sembuf write_wait    = {0, -1, 0};
struct sembuf write_exhaust = {0, 0, 0};
struct sembuf read_wait     = {1, -1, 0};
struct sembuf read_exhaust  = {1, 0, 0};
std::vector<char> NicoProcessGroup::ipc_allgather(const void *input, size_t bytes) {
  // gather all `input` data into `recvbuf`
  // shared memory needn't remap to ensure coherence
  assert(ipcInitialized);
  assert(bytes <= IPC_PROC_SEG_SIZE);
  struct sembuf write_free = {0, (short)memberNum, 0};
  struct sembuf read_free  = {1, (short)memberNum, 0};
  std::vector<char> recvbuf(bytes * memberNum, 0);

  // data is placed tightly in shared memory
  int segmentOffset = groupRank * bytes;
  int semid         = _semid[SEM_IPC_ALLGATHER];
  if (isLeader) {
    semop(semid, &write_free, 1);
  }
  if (input != NULL) {
    memcpy(shmdata + segmentOffset, input, bytes);
  } else {
    memset(shmdata + segmentOffset, 0, bytes);
  }
  // signal write completion
  semop(semid, &write_wait, 1);
  // wait until all processes have written
  semop(semid, &write_exhaust, 1);
  if (isLeader) {
    //! this free must happen after write_exhaust
    //! otherwise a straggler can stuck at read_exhaust
    semop(semid, &read_free, 1);
  }
  memcpy(recvbuf.data(), shmdata, bytes * memberNum);
  // signal read completion
  semop(semid, &read_wait, 1);
  // wait until all processes have read
  semop(semid, &read_exhaust, 1);
  return recvbuf;
}

std::vector<cudaIpcMemHandle_t>
NicoProcessGroup::ipc_allgather_device_pointer(const std::vector<void *> &ptrs) {
  int items = ptrs.size();
  std::vector<cudaIpcMemHandle_t> sendbuf;
  sendbuf.resize(items);
  for (int i = 0; i < items; ++i) {
    if (ptrs[i] != nullptr)
      CUDA_SAFE_CALL(cudaIpcGetMemHandle(&sendbuf[i], ptrs[i]));
  }
  auto raw_chars = ipc_allgather(sendbuf.data(), sizeof(cudaIpcMemHandle_t) * items);
  std::vector<cudaIpcMemHandle_t> ret;
  ret.resize(items * memberNum);
  memcpy(ret.data(), raw_chars.data(), raw_chars.size());
  return ret;
}

DeviceMemoryManager *DeviceMemoryManager::_manager = nullptr;

DeviceMemoryManager *DeviceMemoryManager::get() {
  if (_manager != nullptr) // most cases
    return _manager;
  _mu.lock();
  if (_manager == nullptr)
    _manager = new DeviceMemoryManager();
  _mu.unlock();
  return _manager;
}
