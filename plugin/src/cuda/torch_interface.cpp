#include "misc.h"
#include "nv_common.h"

#include "cuda/cuda_ops.h"
#include "cuda/torch_helper.h"

void grid_sample(torch::Tensor input, torch::Tensor grid, torch::Tensor output,
                 cudaStream_t stream) {
  CHECK_INPUT(input);
  CHECK_INPUT(grid);
  CHECK_INPUT(output);
}
