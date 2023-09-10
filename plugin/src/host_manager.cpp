#include "host_manager.h"

std::mutex HostMemoryManager::singleton_mutex{};
HostMemoryManager *HostMemoryManager::_manager = nullptr;

HostMemoryManager *HostMemoryManager::get() {
  if (_manager != nullptr) // most cases
    return _manager;
  singleton_mutex.lock();
  if (_manager == nullptr)
    _manager = new HostMemoryManager();
  singleton_mutex.unlock();
  return _manager;
}
