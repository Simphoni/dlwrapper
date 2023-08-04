// common.h
// includes macros of cpp extension and nvidia headers
// DOES NOT include torch headers
#pragma once

#define PLUGIN_ENABLE_NCCL
#define PLUGIN_ENABLE_DEBUG

#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#define CUDA_SAFE_CALL(__fn__) _cuda_safe_call(__fn__, __FILE__, __LINE__)
inline void _cuda_safe_call(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA Error at %s:%d message %s\n", file, line,
            cudaGetErrorName(err));
    exit(-1);
  }
}

#ifdef PLUGIN_ENABLE_NCCL

#define USE_C10D_NCCL // enable ProcessGroupNCCL
#include "nccl.h"

#define NCCL_SAFE_CALL(__fn__) _nccl_safe_call(__fn__, __FILE__, __LINE__)
inline void _nccl_safe_call(ncclResult_t err, const char *file, int line) {
  if (err != ncclSuccess) {
    fprintf(stderr, "NCCL Error at %s:%d value %d\n", file, line, err);
    exit(-1);
  }
}

#endif

#ifdef PLUGIN_ENABLE_DEBUG
#define DEBUG(fmt, ...)                                                        \
  fprintf(stderr, "%s:%d\t" fmt "\n", __FILE__, __LINE__, __VA_ARGS__)
#else
#define DEBUG(...)
#endif
