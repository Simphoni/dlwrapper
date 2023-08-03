#include "nico.h"

void _steal_nccl(c10d::ProcessGroupNCCL &p, torch::Tensor t) {
  ProcessGroupNico *h = static_cast<ProcessGroupNico *>(&p);
  h->getcomm(t.device());
}