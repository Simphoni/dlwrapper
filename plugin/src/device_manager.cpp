#include "device_manager.h"

DeviceContextManager *DeviceContextManager::_manager = nullptr;

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

void DeviceContextManager::ipc_init_process_group() {
  assert(initialized == true);
  shmid = shmget(IPC_KEY, IPC_SHMEM_SEG_SIZE, 0666 | IPC_CREAT);
  assert(shmid != -1);
  /*
  ** allocate 3 semaphores,
  ** sem[0] write lock
  ** sem[1] read lock
  ** sem[2] a counter for each process to increment
  ** when rank0 realizes that all processes have incremented the counter, it
  ** enables the read lock
  */
  semid = semget(IPC_KEY, 3, 0666 | IPC_CREAT);
  assert(semid != -1);
  shmdata = (char *)shmat(shmid, NULL, 0);
}

char *DeviceContextManager::ipc_allgather(const void *input, size_t bytes) {
  // gather all `input` data into `recvbuf`
  // shared memory needn't remap to ensure coherence
  assert(bytes <= IPC_PROC_SEG_SIZE);
  static struct sembuf write_wait = {0, -1, 0};
  static struct sembuf write_free = {0, (short)deviceCount, 0};
  static struct sembuf write_exhaust = {0, 0, 0};
  static struct sembuf read_wait = {1, -1, 0};
  static struct sembuf read_free = {1, (short)deviceCount, 0};
  static struct sembuf read_exhaust = {1, 0, 0};

  // data is placed tightly in shared memory
  segment_offset = localRank * bytes;
  if (localRank == 0) {
    // enable write lock
    semop(semid, &write_free, 1);
  }
  // write begin after master has enabled write lock
  if (input != NULL) {
    memcpy(shmdata + segment_offset, input, bytes);
  } else {
    memset(shmdata + segment_offset, 0, bytes);
  }
  // signal write completion
  semop(semid, &write_wait, 1);
  // wait until all processes have written
  // read can begin immediately
  semop(semid, &write_exhaust, 1);
  if (localRank == 0) {
    semop(semid, &read_free, 1);
  }
  memcpy(recvbuf, shmdata, bytes * deviceCount);
  // signal read completion
  semop(semid, &read_wait, 1);
  // wait until all processes have read
  semop(semid, &read_exhaust, 1);
  return recvbuf;
}