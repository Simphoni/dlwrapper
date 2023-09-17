#include "cuda/cuda_kernels.cuh"

namespace GridSampler {
/*
 * Concepts
 * in each iteration:
 * - nanobatch: batch size at per-warp level
 * - microbatch: batch size at per-block level
 * - minibatch: batch size at per-grid level
 */

/**
 * @brief calculate optimal tile size for a given channel size
 *
 * @param c channel size
 * @return optimal tile size
 */
__device__ __forceinline__ int calc_channel_tile_size(int c) {
  if (c % 32 == 0)
    return 32;
  else if (c % 16 == 0)
    return 16;
  else if (c % 8 == 0)
    return 8;
  else if (c % 4 == 0)
    return 4;
  else {
  }
}

// clang-format off
__global__ void grid_sample_3d_ndhwc(float *__restrict__ input,
                                     float *__restrict__ sample_grid,
                                     float *__restrict__ output,
                                     int n, int d, int h, int w, int c,
                                     int batch_size) {
  // clang-format on
  __shared__ float output_cache[1024];
}

} // namespace GridSampler

void launch_grid_sample_3d(float *input, float *sample_grid, float *output, int n, int d, int h,
                           int w, int c, int batch_size, cudaStream_t stream) {
  dim3 grid_size(1, 1, 1);
  dim3 block_size(1, 1, 1);
  GridSampler::grid_sample_3d_ndhwc<<<grid_size, block_size, 0, stream>>>(
      input, sample_grid, output, n, d, h, w, c, batch_size);
}