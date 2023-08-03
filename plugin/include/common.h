// common.h 
// includes macros of cpp extension and nvidia headers
// DOES NOT include torch headers

#ifndef COMMON_H
#define COMMON_H

#define PLUGIN_ENABLE_NCCL
#define PLUGIN_ENABLE_DEBUG

#include "cuda.h"
#include <cstdio>

#ifdef PLUGIN_ENABLE_NCCL

#define USE_C10D_NCCL // enable ProcessGroupNCCL
#include "nccl.h"

#define NCCL_SAFE_CALL(__fn__) { \
    auto __res__ = __fn__; \
    if (__res__ != ncclSuccess) { \
        fprintf(stderr, "NCCL Error at %s:%d value %d\n", __FILE__, __LINE__, __res__); \
        exit(-1); \
    } \
}

#endif

#ifdef PLUGIN_ENABLE_DEBUG
#define DEBUG(x) fprintf(stderr, x)
#else
#define DEBUG(x)
#endif

#endif