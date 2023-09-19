#include "device_manager.h"
#include "misc.h"
#include "nv_common.h"

#include "cuda/cuda_kernels.h"
#include "cuda/cuda_torch_api.h"
#include "cuda/torch_helper.h"

// DO NOT use AT_DISPATCH_XXX_TYPES macro here, they are clumsy

void grid_sample(torch::Tensor input, torch::Tensor grid, torch::Tensor output, bool output_ncdhw) {
  cudaStream_t stream = DeviceMemoryManager::get()->get_move_stream();
  CHECK_INPUT(input);
  CHECK_INPUT(grid);
  CHECK_INPUT(output);
  auto i_shape = input.sizes();
  auto g_shape = grid.sizes();
  auto o_shape = output.sizes();
  if (i_shape.size() == 5) {
    int n = i_shape[0];
    int d = i_shape[1];
    int h = i_shape[2];
    int w = i_shape[3];
    int c = i_shape[4];
    TORCH_CHECK_EQ(o_shape.size(), 5);
    TORCH_CHECK_EQ(o_shape[0], n);
    if (output_ncdhw) {
      TORCH_CHECK_EQ(o_shape[1], c);
    } else {
      TORCH_CHECK_EQ(o_shape[4], c);
    }
    TORCH_CHECK_EQ(g_shape.size(), 5);
    TORCH_CHECK_EQ(g_shape[0], n);
    TORCH_CHECK_EQ(g_shape[4], 3);
    int batch_size = g_shape[1] * g_shape[2] * g_shape[3];
    if (output_ncdhw) {
      TORCH_CHECK_EQ(o_shape[2] * o_shape[3] * o_shape[4], batch_size);
    } else {
      TORCH_CHECK_EQ(o_shape[1] * o_shape[2] * o_shape[3], batch_size);
    }

    switch (input.scalar_type()) {
    case c10::ScalarType::Float:
      launch_grid_sample_3d_float(input.data_ptr<float>(), grid.data_ptr<float>(),
                                  output.data_ptr<float>(), n, d, h, w, c, batch_size, output_ncdhw,
                                  stream);
      break;
    default:
      throw;
    }
  } else {
  }
  cudaStreamSynchronize(stream);
}
