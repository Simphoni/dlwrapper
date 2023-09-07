__all__ = ["nico_native", "torch_backend", "OriginTensor", "ModelManagerTorch"]

import torch
from . import dlwrapperffi

nico_native = dlwrapperffi.nico_native
torch_backend = dlwrapperffi.torch_backend

OriginTensor = torch_backend.OriginTensor
ModelManagerTorch = torch_backend.ModelManagerTorch
