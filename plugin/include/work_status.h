#pragma once

#include "misc.h"
#include "tensor.h"
#include <set>

struct WorkInfo {
  uint64_t workId;
  uint64_t dataSize;
};

class MoverStatus {
  // keep track of memory movement operations in progress
private:
  std::set<WorkInfo> movers;
  std::mutex mu;
  uint8_t count[6];

public:
  bool register_mover(size_t bytes) {
    mu.lock();

    mu.unlock();
  }
};