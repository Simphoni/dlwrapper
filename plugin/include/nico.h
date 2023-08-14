#pragma once
#include "common.h"
#include <chrono>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/extension.h>
#include <utility>

namespace ch = std::chrono;
using dtype_torch = c10::ScalarType;

class ProcessGroupNico : public c10d::ProcessGroupNCCL {
public:
  std::pair<ncclComm_t, int> getComm(at::Device dev);
};

void _sync_stream(int idx = 0);

void _init_nccl(c10d::ProcessGroupNCCL &p, at::Device dev,
                bool enable_uva = false);

void _sendrecv(torch::Tensor t, int src_rnk, int dst_rnk, bool prof);

void _broadcast(torch::Tensor t, bool prof);

void _allgather_into_tensor_doubling(torch::Tensor dst, torch::Tensor src,
                                     bool prof);

void _manager_export_summary();