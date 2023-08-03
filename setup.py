# file name: setup.py
import os
import torch
from setuptools import setup
from torch.utils import cpp_extension

plugin_path = "plugin"
sources = [f"{plugin_path}/src/nico.cpp", f"{plugin_path}/src/cuda_plugins.cpp"]
include_dirs = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), plugin_path, "include")
]

module = cpp_extension.CUDAExtension(
    name="cuda_plugins",
    sources=sources,
    include_dirs=include_dirs,
)

setup(
    name="CUDA-Plugins",
    ext_modules=[module],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
