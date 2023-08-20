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
  std::pair<ncclComm_t, int> getComm();
};

void _sync_stream(int idx = 0);

void _init_nico(c10d::ProcessGroupNCCL &p, bool enable_uva = false);

void _destroy_nico();

void _sendrecv(torch::Tensor t, int src_rnk, int dst_rnk, bool prof);

void _broadcast(torch::Tensor t, bool prof);

void _allgather(torch::Tensor dst, torch::Tensor src, int idx = 0,
                bool prof = false);
void _scatter(torch::Tensor tensor, int src_rank, bool prof = false);

void _manager_export_summary();