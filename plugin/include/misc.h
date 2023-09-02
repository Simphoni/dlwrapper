#pragma once
#define PLUGIN_ENABLE_INFO
#define PLUGIN_ENABLE_DEBUG

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cstdlib>
#include <stdexcept>

/* /cpfs01/user/xingjingze/dlwrapper/plugin/include/device_manager.h */
#define __FILE_BRIEF__ std::string(__FILE__).substr(41).data()

#ifdef PLUGIN_ENABLE_INFO
#define INFO(fmt, ...) fprintf(stderr, "[INFO]\t" fmt "\n", __VA_ARGS__)
#else
#define INFO(...)
#endif

#ifdef PLUGIN_ENABLE_DEBUG
#define DEBUG(fmt, ...) fprintf(stderr, "%s:%d\t" fmt "\n", __FILE_BRIEF__, __LINE__, __VA_ARGS__)
#else
#define DEBUG(...)
#endif