cmake_minimum_required(VERSION 3.22)
project(dlwrapper C CXX CUDA)
find_package(Python3 COMPONENTS Interpreter Development)

# vscode cmake extension helper
set(Python3_SHORT_VERSION "python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}")

set(Python3_SITE_PACKAGES "${Python3_LIBRARY_DIRS}/${Python3_SHORT_VERSION}/site-packages")
if (NOT EXISTS ${Python3_SITE_PACKAGES})
    set(Python3_SITE_PACKAGES "$ENV{HOME}/.local/lib/${Python3_SHORT_VERSION}}/site-packages")
endif()
if (EXISTS ${Python3_SITE_PACKAGES})
    if (EXISTS ${Python3_SITE_PACKAGES}/torch)
        set(Torch_ROOT "${Python3_SITE_PACKAGES}/torch")
        set(CUDNN_LIBRARY_PATH "${Torch_ROOT}/lib/libcudnn.so.8")
    endif()
    if (EXISTS ${Python3_SITE_PACKAGES}/nvidia)
        set(CUDNN_INCLUDE_DIR "${Python3_SITE_PACKAGES}/nvidia/cudnn/include")
    endif()
    if (EXISTS ${Python3_SITE_PACKAGES}/pybind11)
        set(pybind11_ROOT "${Python3_SITE_PACKAGES}/pybind11")
    endif()
endif()

set(CCACHE_EXECUTABLE "/usr/local/bin/ccache")
if (EXISTS ${CCACHE_EXECUTABLE})
    message(STATUS "Using ccache: ${CCACHE_EXECUTABLE}")
    set(CMAKE_C_LAUNCHER ${CCACHE_EXECUTABLE})
    set(CMAKE_CXX_LAUNCHER ${CCACHE_EXECUTABLE})
    set(CMAKE_CUDA_LAUNCHER ${CCACHE_EXECUTABLE})
endif()

set(TORCH_CUDA_ARCH_LIST "8.0")
find_package(Torch REQUIRED)
find_package(pybind11 REQUIRED)
# provide pybind11 casts for pytorch classes
append_torchlib_if_found(torch_python)

set(DEFAULT_BUILD_TYPE "RelWithDebInfo")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -lrt -UNDEBUG -O2")

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# INSTALL_RPATH_USE_LINK_PATH prevents INSTALL_RPATH from overwriting linked libraries
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

include_directories(${Python3_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(include)

file(GLOB_RECURSE SRC src/*.cpp src/cuda/*.cu)
file(GLOB_RECURSE SRC_FFI ffi/*.cpp)
file(GLOB_RECURSE SRC_STANDALONE standalone/*.cpp)

add_library(dlwrapper SHARED ${SRC})
target_include_directories(dlwrapper PUBLIC ${TORCH_INCLUDE_DIRS})
target_link_libraries(dlwrapper ${TORCH_LIBRARIES} ${Python3_LIBRARIES})

pybind11_add_module(dlwrapperffi SHARED ${SRC_FFI})
target_include_directories(dlwrapperffi PUBLIC ${TORCH_INCLUDE_DIRS})
target_link_libraries(dlwrapperffi PRIVATE dlwrapper)

set_target_properties(dlwrapper dlwrapperffi PROPERTIES INSTALL_RPATH "$ORIGIN")
install(TARGETS dlwrapper LIBRARY DESTINATION ${PY_BUILD_CMAKE_MODULE_NAME} COMPONENT torch_extension)
install(TARGETS dlwrapperffi LIBRARY DESTINATION ${PY_BUILD_CMAKE_MODULE_NAME} COMPONENT torch_extension)

foreach(sourcefile ${SRC_STANDALONE})
    get_filename_component(executable ${sourcefile} NAME_WE)
    add_executable(${executable} ${sourcefile})
    set_target_properties(${executable} PROPERTIES INSTALL_RPATH "$ORIGIN")
    target_link_libraries(${executable} dlwrapper ${Python_LIBRARIES})
    install(TARGETS ${executable} RUNTIME DESTINATION ${PY_BUILD_CMAKE_MODULE_NAME} COMPONENT)
endforeach(sourcefile ${SRC_STANDALONE})