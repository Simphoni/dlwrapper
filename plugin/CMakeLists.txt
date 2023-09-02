cmake_minimum_required(VERSION 3.27)
project(dlwrapper C CXX CUDA)
find_package(Python3 COMPONENTS Interpreter Development)
find_package(Torch REQUIRED)
find_package(pybind11 REQUIRED)

set(DEFAULT_BUILD_TYPE "RelWithDebInfo")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -lrt -UNDEBUG")
# INSTALL_RPATH_USE_LINK_PATH prevents INSTALL_RPATH from overwriting linked libraries
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

include_directories(${Python3_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(include)

file(GLOB_RECURSE SRC src/*.cpp)
file(GLOB_RECURSE SRC_FFI ffi/*.cpp)
file(GLOB_RECURSE SRC_STANDALONE standalone/*.cpp)

add_library(dlwrapper SHARED ${SRC})
pybind11_add_module(dlwrapperffi SHARED ${SRC_FFI})
target_include_directories(dlwrapper PUBLIC ${TORCH_INCLUDE_DIRS})
target_link_libraries(dlwrapper ${TORCH_LIBRARIES} ${Python3_LIBRARIES})
target_link_libraries(dlwrapperffi PRIVATE dlwrapper PRIVATE ${pybind11_LIBRARIES})

set_target_properties(dlwrapper dlwrapperffi PROPERTIES INSTALL_RPATH "$ORIGIN")
set_target_properties(dlwrapper dlwrapperffi PROPERTIES CUDA_ARCHITECTURES "70;80;86")
install(TARGETS dlwrapper LIBRARY DESTINATION ${PY_BUILD_CMAKE_MODULE_NAME} COMPONENT torch_extension)
install(TARGETS dlwrapperffi LIBRARY DESTINATION ${PY_BUILD_CMAKE_MODULE_NAME} COMPONENT torch_extension)

foreach(sourcefile ${SRC_STANDALONE})
    get_filename_component(executable ${sourcefile} NAME_WE)
    add_executable(${executable} ${sourcefile})
    set_target_properties(${executable} PROPERTIES INSTALL_RPATH "$ORIGIN")
    target_link_libraries(${executable} dlwrapper ${Python_LIBRARIES})
    install(TARGETS ${executable} RUNTIME DESTINATION ${PY_BUILD_CMAKE_MODULE_NAME} COMPONENT torch_extension)
endforeach(sourcefile ${TEST_SOURCES})