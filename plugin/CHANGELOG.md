# Changelog

All notable changes to DLWrapper will be documented in this file.

## 0.0.1 - 2023-8-24

### Added

- Class `NicoProcessGroup` administrated by `DeviceContextManager` to better custom domains
- Class `PeerMemHandleManager` to help manage cuda memory handles from other processes
- Fully tested IPC functionality
- Intranode all_gather and scatter operator, subsequent calls reached >300GB/s

### TODO

- Support internode communication domains with NCCL
- Implement lazy unpickler to manage PyTorch model files

## 0.1.0 - 2023-9-7

### Added

- Partial implementation of PyTorch model manager and its unpickler
- Changed build system to py-build-cmake
- Removed old `setup.py`
- Refactored ffi code organization, exposed OriginTensor and PyTorchModelManager

### TODO

- Fully implement PyTorch model manager