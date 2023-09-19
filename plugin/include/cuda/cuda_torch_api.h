// file name: cuda_ops.h
// expose cuda invoke functions with pytorch api to external callers

#pragma once

#include "misc.h"
#include "nv_common.h"
#include <torch/extension.h>

void grid_sample(torch::Tensor input, torch::Tensor grid, torch::Tensor output, bool output_ncdhw);
