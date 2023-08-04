#include <nico.h>
#include <device_manager.h>

std::pair<ncclComm_t, int> ProcessGroupNico::getComm(at::Device dev) {
  ncclUniqueId ncclID;
  int rank = getRank();
  if (rank == 0) {
    ncclGetUniqueId(&ncclID);
  }
  broadcastUniqueNCCLID(&ncclID, false, "nico_comm", rank);
  ncclComm_t comm;
  NCCL_SAFE_CALL(ncclCommInitRank(&comm, getSize(), ncclID, rank));
  DEBUG("stolen %d, id=%s, dev=%d", rank, ncclID.internal, dev.index());
  return std::make_pair(comm, rank);
}

void _init_nccl(c10d::ProcessGroupNCCL &p, at::Device dev) {
  ProcessGroupNico *h = static_cast<ProcessGroupNico *>(&p);
  CudaContextManager::get()->setCommWorld(h->getComm(dev));
}

void _broadcast(torch::Tensor t) {}