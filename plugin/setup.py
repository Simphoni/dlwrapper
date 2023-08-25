# file name: setup.py
import os
from setuptools import setup
from torch.utils import cpp_extension

extdir = os.path.dirname(os.path.abspath(__file__))

sources = [
    "src/nico.cpp",
    "src/export.cpp",
    "src/device_manager.cpp",
    "src/comm_ops.cpp",
    "src/unit_test.cpp",
    "src/mem_handle_manager.cpp",
    "src/lazy_unpickler.cpp",
]
include_dirs = [
    os.path.join(extdir, "include"),
]
cxx_flags = [
    "-lrt",  # librt for POSIX shared memory objects
    "-UNDEBUG",
    "-std=c++17",
]

dlwrapper = cpp_extension.CUDAExtension(
    name="dlwrapper",
    sources=sources,
    include_dirs=include_dirs,
    extra_compile_args={"cxx": cxx_flags, "nvcc": cxx_flags},
)

setup(
    name="DLWrapper",
    version="0.0.1",
    ext_modules=[dlwrapper],
    # packages=["nico"],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
lib_file = os.path.join(
    extdir, "build/lib.linux-x86_64-cpython-39/dlwrapper.cpython-39-x86_64-linux-gnu.so"
)
