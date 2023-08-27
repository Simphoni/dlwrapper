#include "device_manager.h"
#include "mem_handle_manager.h"
#include <chrono>

namespace ch = std::chrono;

#define IS_POW2(x) ((x) == 2 || (x) == 4 || (x) == 8 || (x) == 16)
#define ALIGN_EXP2(x, y) (((x) >> (y)) << (y))
void NicoProcessGroup::allgather_cuda_uva(char *dst, char *src, int64_t numel_dst,
                                          int64_t numel_src, bool prof) {
  if (!isIntraNode || !isInGroup || !IS_POW2(memberNum) || !ipcInitialized) {
    return;
  }
  assert(numel_dst == numel_src * memberNum);
  auto start_clk = ch::steady_clock::now();
  auto manager = DeviceContextManager::get();
  int semid = _semid[SEM_GPU_COLL];
  struct sembuf sem_release = {(unsigned short int)groupRank, 1, 0};
  struct sembuf sem_wait = {0, -1, 0};

  std::vector<void *> ptrs{dst, src};
  auto peer_handles = ipc_allgather_device_pointer(ptrs);

  int stages = 0;
  void *peer_ptr;
  for (int i = 0; i < 4; ++i) {
    if ((1 << i) == memberNum) {
      stages = i;
      break;
    }
  }

  int peer = groupRank ^ 1;
  auto peer_manager = PeerMemHandleManager::get();
  peer_ptr = peer_manager->openHandle(peer_handles[peer * 2 + 1]);
  CUDA_SAFE_CALL(cudaMemcpyAsync(dst + numel_src * peer, peer_ptr, numel_src,
                                 cudaMemcpyDeviceToDevice, manager->stream(0)));
  CUDA_SAFE_CALL(cudaMemcpyAsync(dst + numel_src * groupRank, src, numel_src,
                                 cudaMemcpyDeviceToDevice, manager->stream(1)));
  for (int i = 1; i < stages; ++i) {
    peer = groupRank ^ (1 << i);
    peer_ptr = peer_manager->openHandle(peer_handles[peer * 2]);
    manager->sync(0);
    assert(semop(semid, &sem_release, 1) != -1);
    sem_wait.sem_num = peer;
    assert(semop(semid, &sem_wait, 1) != -1);
    size_t offset = ALIGN_EXP2(peer, i) * numel_src;
    CUDA_SAFE_CALL(cudaMemcpyAsync(dst + offset, (char *)peer_ptr + offset, numel_src << i,
                                   cudaMemcpyDeviceToDevice, manager->stream(0)));
  }
  manager->sync(0);
  manager->sync(1);

  auto stop_clk = ch::steady_clock::now();
  if (prof) {
    ch::duration<double> duration = stop_clk - start_clk;
    manager->manual_record("allgather_intranode_uva", duration.count() * 1000);
  }
}
#undef ALIGN_EXP2

void NicoProcessGroup::scatter_cuda_uva(char *data, int src_rank, int64_t numel_dst, bool prof) {
  if (!isIntraNode || !isInGroup || !ipcInitialized) {
    return;
  }
  auto start_clk = ch::steady_clock::now();
  auto manager = DeviceContextManager::get();
  [[maybe_unused]] int semid = _semid[SEM_GPU_COLL];

  std::vector<void *> ptrs{data};
  if (groupRank != src_rank) {
    ptrs[0] = nullptr;
  }
  auto peer_handles = ipc_allgather_device_pointer(ptrs);

  if (groupRank != src_rank) {
    void *src_ptr = PeerMemHandleManager::get()->openHandle(peer_handles[src_rank]);
    size_t offset = groupRank * numel_dst;
    CUDA_SAFE_CALL(cudaMemcpyAsync(data, (char *)src_ptr + offset, numel_dst,
                                   cudaMemcpyDeviceToDevice, manager->stream(0)));
    manager->sync(0);
  }

  auto stop_clk = ch::steady_clock::now();
  if (prof) {
    ch::duration<double> duration = stop_clk - start_clk;
    manager->manual_record("scatter_intranode_uva", duration.count() * 1000);
  }
}

#undef IS_POW2