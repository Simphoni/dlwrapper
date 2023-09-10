__all__ = [
    "nico_native",
    "torch_backend",
    # torch_backend members
    "BaseTensor",
    "OriginTensor",
    "ModelManagerTorch",
    "TensorGrid",
]

import torch
from . import dlwrapperffi

nico_native = dlwrapperffi.nico_native
torch_backend = dlwrapperffi.torch_backend

BaseTensor = torch_backend.BaseTensor
OriginTensor = torch_backend.OriginTensor
ModelManagerTorch = torch_backend.ModelManagerTorch
TensorGrid = torch_backend.TensorGrid
