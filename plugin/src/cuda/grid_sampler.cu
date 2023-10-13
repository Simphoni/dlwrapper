#include "cuda/cuda_kernels.h"
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
namespace cg = cooperative_groups;

namespace GridSampler {
/*
 * Concepts
 * in each iteration:
 * - nanobatch: batch size at per-warp/workgroup level
 * - microbatch: batch size at per-block level
 * - minibatch: batch size at per-grid level
 */

int const WARP_NUM = 8;

/**
 * @brief calculate optimal tile size for a given channel size
 *
 * @param c channel size
 * @return optimal tile size
 */
__device__ __forceinline__ int calc_channel_tile_size(int c) {
  if (c % 32 == 0) {
    return 32;
  } else if (c % 16 == 0) {
    return 16;
  } else if (c % 8 == 0) {
    return 8;
  } else {
    return 4;
  }
}

/**
 * @brief perform unnormailization on coordinate within [-1, 1]
 */
template <typename scalar_t>
__forceinline__ __device__ scalar_t grid_sampler_unnormalize(scalar_t coord, int size,
                                                             bool align_corners) {
  if (align_corners) {
    // unnormalize coord from [-1, 1] to [0, size - 1]
    return ((coord + 1.f) / 2) * (size - 1);
  } else {
    // unnormalize coord from [-1, 1] to [-0.5, size - 0.5]
    return ((coord + 1.f) * size - 1) / 2;
  }
}

template <typename scalar_t>
__forceinline__ __device__ scalar_t safe_downgrade_to_int_range(scalar_t x) {
  if (x > INT_MAX - 1 || x < INT_MIN || !::isfinite(static_cast<double>(x)))
    return static_cast<scalar_t>(-100.0);
  return x;
}

__forceinline__ __device__ bool inrange(int x, int d) { return x >= 0 && x < d; }

// fixed variables for compiler optimization
bool const align_corners            = true;
bool const allow_arithmatic_reorder = false;

template <typename scalar_t>
__global__ void grid_sample_3d_ndhwc2ndhwc(scalar_t *__restrict__ input,
                                           scalar_t *__restrict__ sample_grid,
                                           scalar_t *__restrict__ output, int n, int d, int h,
                                           int w, int c, int batch_size) {
  // Each workgroup find its work by indexing into [batch_size, c/c_tile]
  // workgroup's size is equal to c_tile
  int const threadid   = threadIdx.x + threadIdx.y * blockDim.x;
  int const blockid    = blockIdx.x + blockIdx.y * gridDim.x;
  int const c_tile     = calc_channel_tile_size(c);
  int const N_c_tile   = (c + c_tile - 1) / c_tile;
  int const warp_idx   = threadid / 32;
  int const lane_idx   = threadid % 32;
  int const group_idx  = threadid / c_tile;
  int const worker_idx = threadid % c_tile;
  int const nanobatch  = 8;
  int const microbatch = blockDim.x * blockDim.y / c_tile * nanobatch;
  int const minibatch  = gridDim.x * gridDim.y * microbatch;
  int const njobs      = n * batch_size * N_c_tile;
  int job_offset_base  = blockid * microbatch + group_idx * nanobatch;
  uint32_t mask        = 0;

  int wg_lane_start = lane_idx - worker_idx;
  int wg_lane_end   = wg_lane_start + c_tile;

  mask = wg_lane_end == 32 ? 0xffffffff : (1 << wg_lane_end) - 1;
  mask = mask ^ (1 << wg_lane_start) - 1;

  __shared__ float sample_grid_shared[64 * 29];
  float *sample_grid_wg = sample_grid_shared + group_idx * 29;

  for (int it = job_offset_base; it < njobs; it += minibatch) {
    for (int readit = worker_idx; readit < nanobatch * 3; readit += c_tile) {
      int nanoit   = readit / 3;
      int read_idx = readit % 3;
      int job_idx  = it + nanoit;
      if (job_idx < njobs) {
        float raw = sample_grid[(job_idx / N_c_tile) * 3 + read_idx];
        int dim   = read_idx == 0 ? w : (read_idx == 1 ? h : d);
        sample_grid_wg[readit] =
            safe_downgrade_to_int_range(grid_sampler_unnormalize(raw, dim, align_corners));
      }
    }
    __syncwarp(mask);
    for (int nanoit = 0; nanoit < nanobatch; nanoit++) {
      int const job_idx = it + nanoit;
      if (job_idx >= njobs) {
        break;
      }
      int const c_offset  = job_idx % N_c_tile * c_tile + worker_idx;
      int const batch_idx = (job_idx / N_c_tile) % batch_size;
      int const n_idx     = (job_idx / N_c_tile) / batch_size;

      using index_t = int;

      float ix = sample_grid_wg[nanoit * 3 + 0];
      float iy = sample_grid_wg[nanoit * 3 + 1];
      float iz = sample_grid_wg[nanoit * 3 + 2];

      index_t ix_tnw = static_cast<index_t>(::floor(ix));
      index_t iy_tnw = static_cast<index_t>(::floor(iy));
      index_t iz_tnw = static_cast<index_t>(::floor(iz));

      index_t ix_bse = ix_tnw + 1;
      index_t iy_bse = iy_tnw + 1;
      index_t iz_bse = iz_tnw + 1;

      if (c_offset < c) {
        float sum = 0;
        scalar_t *input_base =
            input + (((n_idx * d + iz_tnw) * h + iy_tnw) * w + ix_tnw) * c + c_offset;
        scalar_t *output_base = output + (n_idx * batch_size + batch_idx) * c + c_offset;

        int inp_sD = h * w * c;
        int inp_sH = w * c;
        int inp_sW = c;

        if (allow_arithmatic_reorder && 0 <= iz_tnw && iz_bse < d && 0 <= iy_tnw && iy_bse < h &&
            0 <= ix_tnw && ix_bse < w) {
          // strictly inside
          float tnw = (ix_bse - ix) * (iy_bse - iy);
          float tne = (ix - ix_tnw) * (iy_bse - iy);
          float tsw = (ix_bse - ix) * (iy - iy_tnw);
          float tse = (ix - ix_tnw) * (iy - iy_tnw);
          // clang-format off
          sum += (tnw * input_base[0] +
                  tne * input_base[inp_sW] +
                  tsw * input_base[inp_sH] +
                  tse * input_base[inp_sH + inp_sW]) * (iz_bse - iz);
          input_base += inp_sD;
          sum += (tnw * input_base[0] +
                  tne * input_base[inp_sW] +
                  tsw * input_base[inp_sH] +
                  tse * input_base[inp_sH + inp_sW]) * (iz - iz_tnw);
          // clang-format on
        } else {
          float coef_00 = (ix - ix_tnw) * (iy - iy_tnw);
          float coef_01 = (ix - ix_tnw) * (iy_bse - iy);
          float coef_10 = (ix_bse - ix) * (iy - iy_tnw);
          float coef_11 = (ix_bse - ix) * (iy_bse - iy);
          float z_1     = iz_bse - iz;
          float z_0     = iz - iz_tnw;

          if (inrange(iz_tnw, d)) {
            if (inrange(iy_tnw, h)) {
              if (inrange(ix_tnw, w))
                sum += coef_11 * z_1 * input_base[0];
              if (inrange(ix_tnw + 1, w))
                sum += coef_01 * z_1 * input_base[inp_sW];
            }
            if (inrange(iy_tnw + 1, h)) {
              if (inrange(ix_tnw, w))
                sum += coef_10 * z_1 * input_base[inp_sH];
              if (inrange(ix_tnw + 1, w))
                sum += coef_00 * z_1 * input_base[inp_sH + inp_sW];
            }
          }
          input_base += inp_sD;

          if (inrange(iz_tnw + 1, d)) {
            if (inrange(iy_tnw, h)) {
              if (inrange(ix_tnw, w))
                sum += coef_11 * z_0 * input_base[0];
              if (inrange(ix_tnw + 1, w))
                sum += coef_01 * z_0 * input_base[inp_sW];
            }
            if (inrange(iy_tnw + 1, h)) {
              if (inrange(ix_tnw, w))
                sum += coef_10 * z_0 * input_base[inp_sH];
              if (inrange(ix_tnw + 1, w))
                sum += coef_00 * z_0 * input_base[inp_sH + inp_sW];
            }
          }
        }
        output_base[0] = sum;
      }
    }
  }
}

__device__ __forceinline__ int calc_channel_roundup(int c) {
  if (c > 16)
    return 32;
  if (c > 8)
    return 16;
  return 8;
}

// adopting a rough strategy that doesn't tile channel can bring more opportunities
template <typename scalar_t>
__global__ void grid_sample_3d_ndhwc2ndhwc_rough(scalar_t *__restrict__ input,
                                                 scalar_t *__restrict__ sample_grid,
                                                 scalar_t *__restrict__ output, int n, int d, int h,
                                                 int w, int c, int batch_size) {
  int const threadid   = threadIdx.x + threadIdx.y * blockDim.x;
  int const blockid    = blockIdx.x + blockIdx.y * gridDim.x;
  int const c_tile     = calc_channel_roundup(c);
  int const warp_idx   = threadid / 32;
  int const lane_idx   = threadid % 32;
  int const group_idx  = threadid / c_tile;
  int const worker_idx = threadid % c_tile;
  int const nanobatch  = 8;
  int const microbatch = blockDim.x * blockDim.y / c_tile * nanobatch;
  int const minibatch  = gridDim.x * gridDim.y * microbatch;
  int const njobs      = n * batch_size;
  int job_offset_base  = blockid * microbatch + group_idx * nanobatch;
  uint32_t mask        = 0;

  int wg_lane_start = lane_idx - worker_idx;
  int wg_lane_end   = wg_lane_start + c_tile;

  mask = wg_lane_end == 32 ? 0xffffffff : (1 << wg_lane_end) - 1;
  mask = mask ^ (1 << wg_lane_start) - 1;

  __shared__ float sample_grid_shared[32 * 26];
  float *sample_grid_wg = sample_grid_shared + group_idx * 26;

  for (int it = job_offset_base; it < njobs; it += minibatch) {
    for (int readit = worker_idx; readit < nanobatch * 3; readit += c_tile) {
      int nanoit   = readit / 3;
      int read_idx = readit % 3;
      int job_idx  = it + nanoit;
      if (job_idx < njobs) {
        float raw = sample_grid[job_idx * 3 + read_idx];
        int dim   = read_idx == 0 ? w : (read_idx == 1 ? h : d);
        sample_grid_wg[readit] =
            safe_downgrade_to_int_range(grid_sampler_unnormalize(raw, dim, align_corners));
      }
    }
    __syncwarp(mask);
    for (int nanoit = 0; nanoit < nanobatch; nanoit++) {
      int const job_idx = it + nanoit;
      if (job_idx >= njobs) {
        break;
      }
      int const c_offset  = worker_idx;
      int const batch_idx = job_idx % batch_size;
      int const n_idx     = job_idx / batch_size;

      using index_t = int;

      float ix = sample_grid_wg[nanoit * 3 + 0];
      float iy = sample_grid_wg[nanoit * 3 + 1];
      float iz = sample_grid_wg[nanoit * 3 + 2];

      index_t ix_tnw = static_cast<index_t>(::floor(ix));
      index_t iy_tnw = static_cast<index_t>(::floor(iy));
      index_t iz_tnw = static_cast<index_t>(::floor(iz));

      index_t ix_bse = ix_tnw + 1;
      index_t iy_bse = iy_tnw + 1;
      index_t iz_bse = iz_tnw + 1;

      if (c_offset < c) {
        float sum = 0;
        scalar_t *input_base =
            input + (((n_idx * d + iz_tnw) * h + iy_tnw) * w + ix_tnw) * c + c_offset;
        scalar_t *output_base = output + job_idx * c + c_offset;

        int inp_sD = h * w * c;
        int inp_sH = w * c;
        int inp_sW = c;

        if (allow_arithmatic_reorder && 0 <= iz_tnw && iz_bse < d && 0 <= iy_tnw && iy_bse < h &&
            0 <= ix_tnw && ix_bse < w) {
          // strictly inside
          float tnw = (ix_bse - ix) * (iy_bse - iy);
          float tne = (ix - ix_tnw) * (iy_bse - iy);
          float tsw = (ix_bse - ix) * (iy - iy_tnw);
          float tse = (ix - ix_tnw) * (iy - iy_tnw);
          // clang-format off
          sum += (tnw * input_base[0] +
                  tne * input_base[inp_sW] +
                  tsw * input_base[inp_sH] +
                  tse * input_base[inp_sH + inp_sW]) * (iz_bse - iz);
          input_base += inp_sD;
          sum += (tnw * input_base[0] +
                  tne * input_base[inp_sW] +
                  tsw * input_base[inp_sH] +
                  tse * input_base[inp_sH + inp_sW]) * (iz - iz_tnw);
          // clang-format on
        } else {
          float coef_00 = (ix - ix_tnw) * (iy - iy_tnw);
          float coef_01 = (ix - ix_tnw) * (iy_bse - iy);
          float coef_10 = (ix_bse - ix) * (iy - iy_tnw);
          float coef_11 = (ix_bse - ix) * (iy_bse - iy);
          float z_1     = iz_bse - iz;
          float z_0     = iz - iz_tnw;

          if (inrange(iz_tnw, d)) {
            if (inrange(iy_tnw, h)) {
              if (inrange(ix_tnw, w))
                sum += coef_11 * z_1 * input_base[0];
              if (inrange(ix_tnw + 1, w))
                sum += coef_01 * z_1 * input_base[inp_sW];
            }
            if (inrange(iy_tnw + 1, h)) {
              if (inrange(ix_tnw, w))
                sum += coef_10 * z_1 * input_base[inp_sH];
              if (inrange(ix_tnw + 1, w))
                sum += coef_00 * z_1 * input_base[inp_sH + inp_sW];
            }
          }
          input_base += inp_sD;

          if (inrange(iz_tnw + 1, d)) {
            if (inrange(iy_tnw, h)) {
              if (inrange(ix_tnw, w))
                sum += coef_11 * z_0 * input_base[0];
              if (inrange(ix_tnw + 1, w))
                sum += coef_01 * z_0 * input_base[inp_sW];
            }
            if (inrange(iy_tnw + 1, h)) {
              if (inrange(ix_tnw, w))
                sum += coef_10 * z_0 * input_base[inp_sH];
              if (inrange(ix_tnw + 1, w))
                sum += coef_00 * z_0 * input_base[inp_sH + inp_sW];
            }
          }
        }
        output_base[0] = sum;
      }
    }
  }
}

template <typename scalar_t>
__global__ void grid_sample_3d_ndhwc2ncdhw(const scalar_t *__restrict__ input,
                                           const scalar_t *__restrict__ sample_grid,
                                           scalar_t *__restrict__ output, int n, int d, int h,
                                           int w, int c, int batch_size) {
  // Each workgroup find its work by indexing into [batch_size, c/c_tile]
  // workgroup's size is equal to c_tile
  int const threadid   = threadIdx.x + threadIdx.y * blockDim.x;
  int const blockid    = blockIdx.x + blockIdx.y * gridDim.x;
  int const c_tile     = calc_channel_tile_size(c);
  int const N_c_tile   = (c + c_tile - 1) / c_tile;
  int const warp_idx   = threadid / 32;
  int const lane_idx   = threadid % 32;
  int const group_idx  = threadid / c_tile;
  int const worker_idx = threadid % c_tile;
  int const nanobatch  = 8;
  int const microbatch = blockDim.x * blockDim.y / c_tile * nanobatch;
  int const minibatch  = gridDim.x * gridDim.y * microbatch;
  int const njobs      = n * batch_size * N_c_tile;
  int job_offset_base  = blockid * microbatch + group_idx * nanobatch;
  uint32_t mask        = 0;

  int const wg_lane_start = lane_idx - worker_idx;
  {
    int wg_lane_end = wg_lane_start + c_tile;

    mask = wg_lane_end == 32 ? 0xffffffff : (1 << wg_lane_end) - 1;
    mask ^= (1 << wg_lane_start) - 1;
  }

  // workgroup = 256 / c_tile <= 64
  // wg_grid = nanobatch * 3
  __shared__ float sample_grid_shared[64 * 29];
  __shared__ scalar_t output_shared[WARP_NUM * nanobatch * 33];

  float *sample_grid_wg = sample_grid_shared + group_idx * 29;
  scalar_t *output_wg   = output_shared + warp_idx * nanobatch * 33;

  for (int it = job_offset_base; it < njobs; it += minibatch) {
    for (int readit = worker_idx; readit < nanobatch * 3; readit += c_tile) {
      int nanoit   = readit / 3;
      int read_idx = readit % 3;
      int job_idx  = it + nanoit;
      if (job_idx < njobs) {
        float raw = sample_grid[(job_idx / N_c_tile) * 3 + read_idx];
        int dim   = read_idx == 0 ? w : (read_idx == 1 ? h : d);
        raw       = grid_sampler_unnormalize(raw, dim, align_corners);
        raw       = safe_downgrade_to_int_range(raw);

        sample_grid_wg[readit] = raw;
      }
    }
    __syncwarp(mask);

    for (int nanoit = 0; nanoit < nanobatch; nanoit++) {
      int const job_idx = it + nanoit;
      if (job_idx >= njobs) {
        break;
      }
      int64_t const c_offset = job_idx % N_c_tile * c_tile + worker_idx;
      int64_t const n_idx    = (job_idx / N_c_tile) / batch_size;

      using index_t = int;

      float ix = sample_grid_wg[nanoit * 3 + 0];
      float iy = sample_grid_wg[nanoit * 3 + 1];
      float iz = sample_grid_wg[nanoit * 3 + 2];
      // 8 grid points to sample from
      index_t ix_tnw = static_cast<index_t>(::floor(ix));
      index_t iy_tnw = static_cast<index_t>(::floor(iy));
      index_t iz_tnw = static_cast<index_t>(::floor(iz));
      index_t ix_bse = ix_tnw + 1;
      index_t iy_bse = iy_tnw + 1;
      index_t iz_bse = iz_tnw + 1;

      if (c_offset < c) {
        float sum = 0;
        const scalar_t *input_base =
            input + (((n_idx * d + iz_tnw) * h + iy_tnw) * w + ix_tnw) * c + c_offset;

        int inp_sD = h * w * c;
        int inp_sH = w * c;
        int inp_sW = c;

        if (0 <= iz_tnw && iz_bse < d && 0 <= iy_tnw && iy_bse < h && 0 <= ix_tnw && ix_bse < w) {
          // strictly inside
          float tnw = (ix_bse - ix) * (iy_bse - iy);
          float tne = (ix - ix_tnw) * (iy_bse - iy);
          float tsw = (ix_bse - ix) * (iy - iy_tnw);
          float tse = (ix - ix_tnw) * (iy - iy_tnw);
          // clang-format off
          sum += (tnw * input_base[0] +
                  tne * input_base[inp_sW] +
                  tsw * input_base[inp_sH] +
                  tse * input_base[inp_sH + inp_sW]) * (iz_bse - iz);
          input_base += inp_sD;
          sum += (tnw * input_base[0] +
                  tne * input_base[inp_sW] +
                  tsw * input_base[inp_sH] +
                  tse * input_base[inp_sH + inp_sW]) * (iz - iz_tnw);
          // clang-format on
        } else {
          float tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz);
          float tne = (ix - ix_tnw) * (iy_bse - iy) * (iz_bse - iz);
          float tsw = (ix_bse - ix) * (iy - iy_tnw) * (iz_bse - iz);
          float tse = (ix - ix_tnw) * (iy - iy_tnw) * (iz_bse - iz);
          float bnw = (ix_bse - ix) * (iy_bse - iy) * (iz - iz_tnw);
          float bne = (ix - ix_tnw) * (iy_bse - iy) * (iz - iz_tnw);
          float bsw = (ix_bse - ix) * (iy - iy_tnw) * (iz - iz_tnw);
          float bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw);
          if (inrange(iz_tnw, d)) {
            if (inrange(iy_tnw, h)) {
              if (inrange(ix_tnw, w))
                sum += tnw * input_base[0];
              if (inrange(ix_tnw + 1, w))
                sum += tne * input_base[inp_sW];
            }
            if (inrange(iy_tnw + 1, h)) {
              if (inrange(ix_tnw, w))
                sum += tsw * input_base[inp_sH];
              if (inrange(ix_tnw + 1, w))
                sum += tse * input_base[inp_sH + inp_sW];
            }
          }
          input_base += inp_sD;

          if (inrange(iz_tnw + 1, d)) {
            if (inrange(iy_tnw, h)) {
              if (inrange(ix_tnw, w))
                sum += bnw * input_base[0];
              if (inrange(ix_tnw + 1, w))
                sum += bne * input_base[inp_sW];
            }
            if (inrange(iy_tnw + 1, h)) {
              if (inrange(ix_tnw, w))
                sum += bsw * input_base[inp_sH];
              if (inrange(ix_tnw + 1, w))
                sum += bse * input_base[inp_sH + inp_sW];
            }
          }
        }
        // nanobatch * 32/c_tile * c_tile
        output_wg[nanoit * 33 + lane_idx] = sum;
      }
    }
    __syncwarp(mask);
    for (int writeit = worker_idx; writeit < nanobatch * c_tile; writeit += c_tile) {
      int const row      = writeit % nanobatch;
      int const col      = writeit / nanobatch;
      int const job_idx  = it + row;
      int const c_offset = (job_idx % N_c_tile) * c_tile + col;
      if (job_idx < njobs && c_offset < c) {
        int const batch_idx = (job_idx / N_c_tile) % batch_size;
        int const n_idx     = (job_idx / N_c_tile) / batch_size;
        output[(n_idx * c + c_offset) * batch_size + batch_idx] =
            output_wg[row * 33 + wg_lane_start + col];
      }
    }
  }
}

} // namespace GridSampler

bool should_use_rough(int c) {
  static int exp[] = {8, 16, 32};
  for (int i = 0; i < 3; i++) {
    if (c <= exp[i] && c > exp[i] / 4 * 3) {
      return true;
    }
  }
  return false;
}

void launch_grid_sample_3d_float(float *input, float *sample_grid, float *output, int n, int d,
                                 int h, int w, int c, int batch_size, bool output_ncdhw,
                                 cudaStream_t stream) {
  dim3 grid_size(216, 1, 1);
  dim3 block_size(32, GridSampler::WARP_NUM, 1);
  if (output_ncdhw) {
    GridSampler::grid_sample_3d_ndhwc2ncdhw<float><<<grid_size, block_size, 0, stream>>>(
        input, sample_grid, output, n, d, h, w, c, batch_size);
  } else {
    if (should_use_rough(c)) {
      GridSampler::grid_sample_3d_ndhwc2ndhwc_rough<float><<<grid_size, block_size, 0, stream>>>(
          input, sample_grid, output, n, d, h, w, c, batch_size);
    } else {
      GridSampler::grid_sample_3d_ndhwc2ndhwc<float><<<grid_size, block_size, 0, stream>>>(
          input, sample_grid, output, n, d, h, w, c, batch_size);
    }
  }
}