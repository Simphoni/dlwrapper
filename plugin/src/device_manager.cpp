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
  shmid = shmget(IPC_KEY, IPC_SHMEM_SEG_SIZE, 0644 | IPC_CREAT);
  assert(shmid != -1);
  segment_offset = localRank * IPC_PROC_SEG_SIZE;
  for (int i = 0; i < MAX_PROCS; i++) {
    recvbuf[i] = new char[IPC_PROC_SEG_SIZE];
  }
  /*
  ** allocate 3 semaphores,
  ** sem[0] write lock
  ** sem[1] read lock
  ** sem[2] a counter for each process to increment
  ** when rank0 realizes that all processes have incremented the counter, it
  ** enables the read lock
  */
  semid = semget(IPC_KEY, 3, 0644 | IPC_CREAT);
  assert(semid != -1);
}

char **DeviceContextManager::ipc_allgather(char *input, size_t bytes) {
  // gather all `input` data into `recvbuf[]`
  // shared memory attach/detach must be done here
  // to ensure the input data gets flushed into memory
  assert(bytes <= IPC_PROC_SEG_SIZE);
  struct sembuf write_wait = {0, -1, 0};
  struct sembuf write_release = {0, 1, 0};
  struct sembuf read_wait = {1, -1, 0};
  struct sembuf read_release = {1, 1, 0};
  struct sembuf counter_wait = {2, -(short)deviceCount, 0};
  struct sembuf counter_release = {2, 1, 0};

  if (localRank == 0) {
    // enable write lock
    assert(semop(semid, &write_release, 1) != -1);
  }
  assert(semop(semid, &write_wait, 1) != -1);
  shmdata = (char *)shmat(shmid, NULL, 0);
  if (input != NULL) {
    memcpy(shmdata + segment_offset, input, bytes);
  } else {
    memset(shmdata + segment_offset, 0, bytes);
  }
  shmdt(shmdata);
  assert(semop(semid, &write_release, 1) != -1);
  // update counter
  assert(semop(semid, &counter_release, 1) != -1);

  if (localRank == 0) {
    //! wait lock must be taken before read
    //! other processes may proceed to next allgather, stop them
    assert(semop(semid, &counter_wait, 1) != -1);
    assert(semop(semid, &write_wait, 1) != -1);
    assert(semop(semid, &read_release, 1) != -1);
  }
  assert(semop(semid, &read_wait, 1) != -1);
  shmdata = (char *)shmat(shmid, NULL, 0);
  for (int i = 0; i < deviceCount; i++) {
    memcpy(recvbuf[i], shmdata + i * IPC_PROC_SEG_SIZE, bytes);
  }
  shmdt(shmdata);
  assert(semop(semid, &read_release, 1) != -1);
  // update counter
  assert(semop(semid, &counter_release, 1) != -1);
  if (localRank == 0) {
    //! all semaphore must be reset to 0 before the next allgather
    //! must wait for all processes to release the read lock
    assert(semop(semid, &counter_wait, 1) != -1);
    assert(semop(semid, &read_wait, 1) != -1);
  }
  return recvbuf;
}