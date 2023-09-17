// file name: cuda_kernels.cuh
// expose cuda kernels to invoke functions in cuda_kernels.cpp

#pragma once
#include "nv_common.h"

/**
 * @brief
 *
 * @param input tensor of size `[n,d,h,w,c]`
 * @param sample_grid tensor of size `[n,d_out,h_out,w_out,3]`
 * @param output tensor of size `[n,d_out,h_out,w_out,c]`
 * @param batch equal to `d_out*h_out*w_out`
 */
void launch_grid_sample_3d(float *input, float *sample_grid, float *output, int n, int d, int h,
                           int w, int c, int batch, cudaStream_t stream);