# file name: setup.py
# install cuda_plugins as
import os
import torch
from setuptools import setup
from torch.utils import cpp_extension

plugin_path = "plugin"
sources = [f"{plugin_path}/src/nico.cpp", f"{plugin_path}/src/export.cpp"]
include_dirs = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), plugin_path, "include"),
]
cxx_flags = ["-O3"]

module = cpp_extension.CUDAExtension(
    name="dlwrapper",
    sources=sources,
    include_dirs=include_dirs,
    extra_compile_args={"cxx": cxx_flags, "nvcc": cxx_flags},
)

setup(
    name="DLWrapper",
    ext_modules=[module],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
