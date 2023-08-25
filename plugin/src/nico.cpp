#include "nico.h"
#include "device_manager.h"
#include "mem_handle_manager.h"
#include <cstdlib>

std::pair<ncclComm_t, int> ProcessGroupNico::getComm() {
  ncclUniqueId ncclID;
  int rank = getRank();
  int world_size = getSize();
  CUDA_SAFE_CALL(cudaSetDevice(rank));
  if (rank == 0) {
    ncclGetUniqueId(&ncclID);
  }
  broadcastUniqueNCCLID(&ncclID, false, "nico_comm", rank);
  ncclComm_t comm;
  NCCL_SAFE_CALL(ncclCommInitRank(&comm, getSize(), ncclID, rank));
  return std::make_pair(comm, rank << 12 | world_size);
}

void _init_nico(c10d::ProcessGroupNCCL &p, bool enable_uva) {
  ProcessGroupNico *h = static_cast<ProcessGroupNico *>(&p);
  DeviceContextManager::get()->set_comm_world(h->getComm());
  if (enable_uva) {
    DeviceContextManager::get()->enable_peer_access();
  }
  atexit(_destroy_nico);
}

void _destroy_nico() {
  DeviceContextManager::get()->destroy_existing_pg();
  PeerMemHandleManager::get()->closeAllHandles();
}

void _sync_stream(int idx) { DeviceContextManager::get()->sync(idx); }

void _memcpy_peer(torch::Tensor dst, torch::Tensor src, int peer, int bytes, bool prof) {
  auto manager = DeviceContextManager::get();
  if (prof)
    manager->event_start("p2p comm");
  cudaMemcpyPeerAsync(dst.data_ptr<float>(), manager->get_local_rank(), src.data_ptr<float>(), peer,
                      bytes, manager->stream(0));
  if (prof)
    manager->event_stop("p2p comm");
  else
    manager->sync(0);
}

void _sendrecv(torch::Tensor t, int src_rnk, int dst_rnk, bool prof) {
  if (src_rnk == dst_rnk)
    return;
  assert(t.is_contiguous());
  auto manager = DeviceContextManager::get();
  ncclComm_t comm = manager->get_comm_world();
  int rank = manager->get_local_rank();
  assert(t.scalar_type() == dtype_torch::Float);
  float *buf = t.data_ptr<float>();
  int stream = 0;
  if (prof) {
    manager->event_start(std::string("sendrecv"));
  }
  if (rank == src_rnk) {
    ncclSend(buf, t.numel(), ncclFloat32, dst_rnk, comm, manager->stream(stream));
  }
  if (rank == dst_rnk) {
    ncclRecv(buf, t.numel(), ncclFloat32, src_rnk, comm, manager->stream(stream));
  }
  if (prof) {
    manager->event_stop(std::string("sendrecv"));
  } else {
    manager->sync(stream);
  }
}

void _broadcast(torch::Tensor t, bool prof) {
  assert(t.is_contiguous());
  // if (t.scalar_type() != dtype_torch::Float) {
  //   fprintf(stderr, "broadcast: tensor dtype not Float32.\n");
  //   throw;
  // }
  auto manager = DeviceContextManager::get();
  ncclComm_t comm = manager->get_comm_world();
  if (prof) {
    manager->event_start(std::string("broadcast"));
  }
  NCCL_SAFE_CALL(ncclBroadcast(t.data_ptr<float>(), t.data_ptr<float>(), t.numel(), ncclFloat32, 0,
                               comm, manager->stream(0)));
  if (prof) {
    manager->event_stop(std::string("broadcast"));
  } else {
    manager->sync(0);
  }
}

void _allgather(torch::Tensor dst, torch::Tensor src, int idx, bool prof) {
  assert(dst.is_contiguous());
  assert(src.is_contiguous());
  auto pg = DeviceContextManager::get()->get_process_group(idx);
  float *dstbuf = dst.data_ptr<float>();
  float *srcbuf = src.data_ptr<float>();
  pg->allgather_cuda_uva((char *)dstbuf, (char *)srcbuf, dst.numel() * 4, src.numel() * 4, prof);
}

void _scatter(torch::Tensor tensor, int src_rank, bool prof) {
  assert(tensor.is_contiguous());
  auto pg = DeviceContextManager::get()->get_process_group(0);
  float *buf = tensor.data_ptr<float>();
  int64_t dst_numel = 0;
  if (DeviceContextManager::get()->get_local_rank() == src_rank)
    dst_numel = tensor.numel() / pg->get_member_num();
  else
    dst_numel = tensor.numel();
  dst_numel *= 4;
  pg->scatter_cuda_uva((char *)buf, src_rank, dst_numel, prof);
}

void _manager_export_summary() { DeviceContextManager::get()->export_summary(); }