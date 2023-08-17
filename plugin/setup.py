# file name: setup.py
# install cuda_plugins as
import os
import torch
from setuptools import setup
from torch.utils import cpp_extension

sources = [
    "src/nico.cpp",
    "src/export.cpp",
    "src/device_manager.cpp",
    "src/unit_test.cpp",
]
include_dirs = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "include"),
]
cxx_flags = [
    "-lrt",  # librt for POSIX shared memory objects
    "-UNDEBUG",
]

module = cpp_extension.CUDAExtension(
    name="dlwrapper",
    sources=sources,
    include_dirs=include_dirs,
    extra_compile_args={"cxx": cxx_flags, "nvcc": cxx_flags},
)

setup(
    name="DLWrapper",
    ext_modules=[module],
    packages=["nico"],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)