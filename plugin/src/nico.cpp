#include <device_manager.h>
#include <nico.h>

std::pair<ncclComm_t, int> ProcessGroupNico::getComm(at::Device dev) {
  ncclUniqueId ncclID;
  int rank = getRank();
  if (rank == 0) {
    ncclGetUniqueId(&ncclID);
  }
  broadcastUniqueNCCLID(&ncclID, false, "nico_comm", rank);
  ncclComm_t comm;
  NCCL_SAFE_CALL(ncclCommInitRank(&comm, getSize(), ncclID, rank));
  //DEBUG("stolen %d, id=%s, dev=%d", rank, ncclID.internal, dev.index());
  return std::make_pair(comm, rank);
}

void _init_nccl(c10d::ProcessGroupNCCL &p, at::Device dev) {
  ProcessGroupNico *h = static_cast<ProcessGroupNico *>(&p);
  CudaContextManager::get()->setCommWorld(h->getComm(dev));
}

void _broadcast(torch::Tensor t) {
  if (!t.is_contiguous()) {
    fprintf(stderr, "broadcast: tensor not contiguous.\n");
    throw;
  }
  if (t.scalar_type() != dtype_torch::Float) {
    fprintf(stderr, "broadcast: tensor dtype not Float32.\n");
    throw;
  }
  ncclComm_t comm;
  int rank;
  auto ret = CudaContextManager::get()->getCommWorld();
  comm = ret.first;
  rank = ret.second;
  NCCL_SAFE_CALL(ncclBroadcast(t.data_ptr<float>(), t.data_ptr<float>(),
                               t.numel(), ncclFloat32, 0, comm,
                               CudaContextManager::get()->stream(0)));
  CudaContextManager::get()->sync(0);
}