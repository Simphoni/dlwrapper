#ifndef NICO_H
#define NICO_H

#include "common.h"
#include <torch/extension.h>
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>

class ProcessGroupNico : public c10d::ProcessGroupNCCL {
public:
  ncclComm_t getcomm(at::Device dev) {
    ncclUniqueId ncclID;
    int rank = getRank();
    if (rank == 0) {
      ncclGetUniqueId(&ncclID);
    }
    broadcastUniqueNCCLID(&ncclID, false, "nico_comm", rank);
    ncclComm_t comm;
    NCCL_SAFE_CALL(ncclCommInitRank(&comm, getSize(), ncclID, rank));
    return comm;
  }
};

void _steal_nccl(c10d::ProcessGroupNCCL &p, torch::Tensor t);

#endif