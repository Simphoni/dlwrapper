#include "mem_handle_manager.h"
#include "common.h"
#include "device_manager.h"
#include <mutex>
#include <unordered_map>

PeerMemHandleManager *PeerMemHandleManager::_manager = nullptr;
static std::mutex _mu;

PeerMemHandleManager *PeerMemHandleManager::get() {
  if (_manager != nullptr) // most cases
    return _manager;
  _mu.lock();
  if (_manager == nullptr)
    _manager = new PeerMemHandleManager();
  _mu.unlock();
  return _manager;
}

void *PeerMemHandleManager::openHandle(cudaIpcMemHandle_t key) {
  auto it = openedHandles.find(key);
  if (it != openedHandles.end())
    return it->second;
  void *devPtr;
  CUDA_SAFE_CALL(
      cudaIpcOpenMemHandle(&devPtr, key, cudaIpcMemLazyEnablePeerAccess));
  openedHandles[key] = devPtr;
  return devPtr;
}

void PeerMemHandleManager::closeAllHandles() {
  INFO("rank[%d]: closing all handles.",
       DeviceContextManager::get()->get_world_rank());
  for (auto it = openedHandles.begin(); it != openedHandles.end(); it++) {
    // Calling cudaFree on an exported memory region before calling
    // `cudaIpcCloseMemHandle` in the importing context will result in undefined
    // behavior.
    try {
      cudaIpcCloseMemHandle(it->second);
    } catch (...) {
      // do nothing
    }
  }
  openedHandles.clear();
}