// file name: cuda_kernels.cuh
// expose cuda kernels to invoke functions in cuda_kernels.cpp

#pragma once
#include "nv_common.h"
#include <type_traits>

/**
 * @brief Grid Sample 3D on float32 with NDHWC input layout
 *
 * @param input tensor of size `[n,d,h,w,c]`
 * @param sample_grid tensor of size `[n,d_out,h_out,w_out,3]`
 * @param output tensor of size `[n,d_out,h_out,w_out,c]`
 * @param batch equal to `d_out*h_out*w_out`
 * @param output_ndhwc if true, output layout is NDHWC, else NCDHW
 */
void launch_grid_sample_3d_float(float *input, float *sample_grid, float *output, int n, int d,
                                 int h, int w, int c, int batch, bool output_ncdhw,
                                 cudaStream_t stream);