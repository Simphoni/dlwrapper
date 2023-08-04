#pragma once
#include "common.h"
#include <utility>
#include <torch/extension.h>
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>

using dtype_torch = c10::ScalarType;

class ProcessGroupNico : public c10d::ProcessGroupNCCL {
public:
  std::pair<ncclComm_t, int> getComm(at::Device dev);
};

void _init_nccl(c10d::ProcessGroupNCCL &p, at::Device dev);

void _broadcast(torch::Tensor t);
