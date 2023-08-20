#include "common.h"
#include <unordered_map>

class PeerMemHandleManager {
private:
  struct cudaMemHandleWrapper {
    cudaIpcMemHandle_t handle;
    cudaMemHandleWrapper(cudaIpcMemHandle_t handle) : handle(handle) {}
    bool operator==(const cudaMemHandleWrapper &other) const {
      return memcmp(handle.reserved, other.handle.reserved,
                    CUDA_IPC_HANDLE_SIZE) == 0;
    }
  };
  struct cudaMemHandleHash {
    std::size_t operator()(const cudaMemHandleWrapper &wrapper) const {
      const size_t *p =
          reinterpret_cast<const size_t *>(wrapper.handle.reserved);
      size_t result = 0;
      for (int i = 0; i < CUDA_IPC_HANDLE_SIZE / 8; i++) {
        result ^= std::hash<size_t>()(p[i]);
      }
      return result;
    }
  };
  std::unordered_map<cudaMemHandleWrapper, void *, cudaMemHandleHash>
      openedHandles;
  static PeerMemHandleManager *_manager;

public:
  static PeerMemHandleManager *get();
  void *openHandle(cudaIpcMemHandle_t key);
  void closeAllHandles();
};