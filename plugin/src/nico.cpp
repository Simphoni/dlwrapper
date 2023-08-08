#include <device_manager.h>
#include <nico.h>

std::pair<ncclComm_t, int> ProcessGroupNico::getComm(at::Device dev) {
  ncclUniqueId ncclID;
  int rank = getRank();
  int world_size = getSize();
  if (rank == 0) {
    ncclGetUniqueId(&ncclID);
  }
  broadcastUniqueNCCLID(&ncclID, false, "nico_comm", rank);
  ncclComm_t comm;
  NCCL_SAFE_CALL(ncclCommInitRank(&comm, getSize(), ncclID, rank));
  return std::make_pair(comm, rank << 12 | world_size);
}

void _init_nccl(c10d::ProcessGroupNCCL &p, at::Device dev) {
  ProcessGroupNico *h = static_cast<ProcessGroupNico *>(&p);
  CudaContextManager::get()->setCommWorld(h->getComm(dev));
}

void _sync_stream(int idx) { CudaContextManager::get()->sync(idx); }

void _sendrecv(torch::Tensor t, int src_rnk, int dst_rnk, bool prof) {
  if (src_rnk == dst_rnk)
    return;
  assert(t.is_contiguous());
  ncclComm_t comm = CudaContextManager::get()->getCommWorld();
  int rank = CudaContextManager::get()->getRank();
  assert(t.scalar_type() == dtype_torch::Float);
  float *buf = t.data_ptr<float>();
  int stream = 0;
  if (prof) {
    CudaContextManager::get()->event_start(std::string("sendrecv"));
  }
  if (rank == src_rnk) {
    ncclSend(buf, t.numel(), ncclFloat32, dst_rnk, comm,
             CudaContextManager::get()->stream(stream));
  }
  if (rank == dst_rnk) {
    ncclRecv(buf, t.numel(), ncclFloat32, src_rnk, comm,
             CudaContextManager::get()->stream(stream));
  }
  if (prof) {
    CudaContextManager::get()->event_stop(std::string("sendrecv"));
  } else {
    CudaContextManager::get()->sync(stream);
  }
}

void _broadcast(torch::Tensor t, bool prof) {
  assert(t.is_contiguous());
  // if (t.scalar_type() != dtype_torch::Float) {
  //   fprintf(stderr, "broadcast: tensor dtype not Float32.\n");
  //   throw;
  // }
  ncclComm_t comm = CudaContextManager::get()->getCommWorld();
  if (prof) {
    CudaContextManager::get()->event_start(std::string("broadcast"));
  }
  NCCL_SAFE_CALL(ncclBroadcast(t.data_ptr<float>(), t.data_ptr<float>(),
                               t.numel(), ncclFloat32, 0, comm,
                               CudaContextManager::get()->stream(0)));
  if (prof) {
    CudaContextManager::get()->event_stop(std::string("broadcast"));
  } else {
    CudaContextManager::get()->sync(0);
  }
}

#define ALIGN_EXP2(x, y) (((x) >> (y)) << (y))

void _allgather_into_tensor_doubling(torch::Tensor dst, torch::Tensor src,
                                     bool prof) {
  assert(CudaContextManager::get()->getWorldSize() == 8);
  assert(dst.size(0) == src.size(0) * 8);
  assert(src.is_contiguous());
  assert(dst.is_contiguous());

  auto start_clk = ch::steady_clock::now();

  float *recvbuf = dst.data_ptr<float>();
  float *sendbuf = src.data_ptr<float>();
  int64_t const len = src.numel();
  int const rank = CudaContextManager::get()->getRank();
  ncclComm_t comm = CudaContextManager::get()->getCommWorld();
  // phase 1 is carefully dealt with
  int peer = rank ^ 1;
  CUDA_SAFE_CALL(cudaMemcpyAsync(recvbuf + len * rank, sendbuf,
                                 len * sizeof(float), cudaMemcpyDeviceToDevice,
                                 CudaContextManager::get()->stream(1)));
  // nccl p2p operation is blocking
  NCCL_SAFE_CALL(ncclGroupStart());
  NCCL_SAFE_CALL(ncclSend(sendbuf, len, ncclFloat32, peer, comm,
                          CudaContextManager::get()->stream(0)));
  NCCL_SAFE_CALL(ncclRecv(recvbuf + len * peer, len, ncclFloat32, peer, comm,
                          CudaContextManager::get()->stream(0)));
  NCCL_SAFE_CALL(ncclGroupEnd());
  CudaContextManager::get()->sync(0);
  CudaContextManager::get()->sync(1);
  // phase 2
  peer = rank ^ 2;
  NCCL_SAFE_CALL(ncclGroupStart());
  NCCL_SAFE_CALL(ncclSend(recvbuf + len * ALIGN_EXP2(rank, 1), len * 2,
                          ncclFloat32, peer, comm,
                          CudaContextManager::get()->stream(0)));
  NCCL_SAFE_CALL(ncclRecv(recvbuf + len * ALIGN_EXP2(peer, 1), len * 2,
                          ncclFloat32, peer, comm,
                          CudaContextManager::get()->stream(0)));
  NCCL_SAFE_CALL(ncclGroupEnd());
  CudaContextManager::get()->sync(0);
  // phase 3
  peer = rank ^ 4;
  NCCL_SAFE_CALL(ncclGroupStart());
  NCCL_SAFE_CALL(ncclSend(recvbuf + len * ALIGN_EXP2(rank, 2), len * 4,
                          ncclFloat32, peer, comm,
                          CudaContextManager::get()->stream(0)));
  NCCL_SAFE_CALL(ncclRecv(recvbuf + len * ALIGN_EXP2(peer, 2), len * 4,
                          ncclFloat32, peer, comm,
                          CudaContextManager::get()->stream(0)));
  NCCL_SAFE_CALL(ncclGroupEnd());
  CudaContextManager::get()->sync(0);

  auto stop_clk = ch::steady_clock::now();
  if (prof) {
    ch::duration<double> duration = stop_clk - start_clk;
    CudaContextManager::get()->manual_record("allgather_into_tensor_doubling",
                                             duration.count() * 1000);
  }
}
#undef ALIGN_EXP2

void _manager_export_summary() { CudaContextManager::get()->export_summary(); }