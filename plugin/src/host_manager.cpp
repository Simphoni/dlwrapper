#include "host_manager.h"

std::mutex HostMemoryManager::_mu{};
HostMemoryManager *HostMemoryManager::_manager = nullptr;

HostMemoryManager *HostMemoryManager::get() {
  if (_manager != nullptr) // most cases
    return _manager;
  _mu.lock();
  if (_manager == nullptr)
    _manager = new HostMemoryManager();
  _mu.unlock();
  return _manager;
}
