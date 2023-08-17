#include <device_manager.h>
#include <nico.h>

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
}

void _destroy_nico() { DeviceContextManager::get()->destroy_existing_pg(); }

void _sync_stream(int idx) { DeviceContextManager::get()->sync(idx); }

void _memcpy_peer(torch::Tensor dst, torch::Tensor src, int peer, int bytes,
                  bool prof) {
  auto manager = DeviceContextManager::get();
  if (prof)
    manager->event_start("p2p comm");
  cudaMemcpyPeerAsync(dst.data_ptr<float>(), manager->get_local_rank(),
                      src.data_ptr<float>(), peer, bytes, manager->stream(0));
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
    ncclSend(buf, t.numel(), ncclFloat32, dst_rnk, comm,
             manager->stream(stream));
  }
  if (rank == dst_rnk) {
    ncclRecv(buf, t.numel(), ncclFloat32, src_rnk, comm,
             manager->stream(stream));
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
  NCCL_SAFE_CALL(ncclBroadcast(t.data_ptr<float>(), t.data_ptr<float>(),
                               t.numel(), ncclFloat32, 0, comm,
                               manager->stream(0)));
  if (prof) {
    manager->event_stop(std::string("broadcast"));
  } else {
    manager->sync(0);
  }
}

#define ALIGN_EXP2(x, y) (((x) >> (y)) << (y))

void _allgather_into_tensor_doubling(torch::Tensor dst, torch::Tensor src,
                                     bool prof) {
  assert(DeviceContextManager::get()->get_world_size() == 8);
  assert(dst.size(0) == src.size(0) * 8);
  assert(src.is_contiguous());
  assert(dst.is_contiguous());

  auto manager = DeviceContextManager::get();
  auto start_clk = ch::steady_clock::now();

  float *recvbuf = dst.data_ptr<float>();
  float *sendbuf = src.data_ptr<float>();
  int64_t const len = src.numel();
  int const rank = manager->get_local_rank();
  ncclComm_t comm = manager->get_comm_world();
  // phase 1 is carefully dealt with
  int peer = rank ^ 1;
  CUDA_SAFE_CALL(cudaMemcpyAsync(recvbuf + len * rank, sendbuf,
                                 len * sizeof(float), cudaMemcpyDeviceToDevice,
                                 manager->stream(1)));
  // nccl p2p operation is blocking
  NCCL_SAFE_CALL(ncclGroupStart());
  NCCL_SAFE_CALL(
      ncclSend(sendbuf, len, ncclFloat32, peer, comm, manager->stream(0)));
  NCCL_SAFE_CALL(ncclRecv(recvbuf + len * peer, len, ncclFloat32, peer, comm,
                          manager->stream(0)));
  NCCL_SAFE_CALL(ncclGroupEnd());
  manager->sync(0);
  manager->sync(1);
  // phase 2
  peer = rank ^ 2;
  NCCL_SAFE_CALL(ncclGroupStart());
  NCCL_SAFE_CALL(ncclSend(recvbuf + len * ALIGN_EXP2(rank, 1), len * 2,
                          ncclFloat32, peer, comm, manager->stream(0)));
  NCCL_SAFE_CALL(ncclRecv(recvbuf + len * ALIGN_EXP2(peer, 1), len * 2,
                          ncclFloat32, peer, comm, manager->stream(0)));
  NCCL_SAFE_CALL(ncclGroupEnd());
  manager->sync(0);
  // phase 3
  peer = rank ^ 4;
  NCCL_SAFE_CALL(ncclGroupStart());
  NCCL_SAFE_CALL(ncclSend(recvbuf + len * ALIGN_EXP2(rank, 2), len * 4,
                          ncclFloat32, peer, comm, manager->stream(0)));
  NCCL_SAFE_CALL(ncclRecv(recvbuf + len * ALIGN_EXP2(peer, 2), len * 4,
                          ncclFloat32, peer, comm, manager->stream(0)));
  NCCL_SAFE_CALL(ncclGroupEnd());
  manager->sync(0);

  auto stop_clk = ch::steady_clock::now();
  if (prof) {
    ch::duration<double> duration = stop_clk - start_clk;
    manager->manual_record("allgather_into_tensor_doubling",
                           duration.count() * 1000);
  }
}
#undef ALIGN_EXP2

void _allgather_with_peer_access(torch::Tensor dst, torch::Tensor src, int idx,
                                 bool prof) {
  assert(dst.is_contiguous());
  assert(src.is_contiguous());
  auto pg = DeviceContextManager::get()->get_process_group(idx);
  float *dstbuf = dst.data_ptr<float>();
  float *srcbuf = src.data_ptr<float>();
  pg->allgather_with_peer_access((char *)dstbuf, (char *)srcbuf,
                                 dst.numel() * 4, src.numel() * 4, prof);
}

void _manager_export_summary() {
  DeviceContextManager::get()->export_summary();
}