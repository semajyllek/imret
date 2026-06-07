cmake_minimum_required(VERSION 3.15)
project(imret_python)

find_package(pybind11 CONFIG REQUIRED)

# Go up one level to read the root CMakeLists.txt and build the core library
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../ ${CMAKE_CURRENT_BINARY_DIR}/core_build)

# Build the Python wrapper module
pybind11_add_module(_core src/bind.cpp)

# Link the Python wrapper to the core library built by the root CMake
target_link_libraries(_core PRIVATE imret_core)
