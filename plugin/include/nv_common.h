// nv_common.h
// includes macros of cpp extension and nvidia headers
// DOES NOT include torch headers
#pragma once

#define PLUGIN_ENABLE_NCCL

#include "misc.h"
#include <cassert>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <string>

#define CUDA_SAFE_CALL(__fn__) _cuda_safe_call(__fn__, __FILE_BRIEF__, __LINE__)
inline void _cuda_safe_call(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA Error at %s:%d message %s\n", file, line, cudaGetErrorName(err));
    exit(-1);
  }
}

#ifdef PLUGIN_ENABLE_NCCL

#ifndef USE_C10D_NCCL
#define USE_C10D_NCCL
#endif
#include "nccl.h"

#define NCCL_SAFE_CALL(__fn__) _nccl_safe_call(__fn__, __FILE_BRIEF__, __LINE__)
inline void _nccl_safe_call(ncclResult_t err, const char *file, int line) {
  if (err != ncclSuccess) {
    fprintf(stderr, "NCCL Error at %s:%d value %d\n", file, line, err);
    exit(-1);
  }
}

#endif
