cmake_minimum_required(VERSION 3.15)
project(imret)


# Enforce C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
find_package(faiss REQUIRED)

# 1. Build the core engine as a static library
add_library(imret_core STATIC 
    src/extractor.cpp 
    src/vault.cpp
)
target_include_directories(imret_core PUBLIC include)
target_link_libraries(imret_core PUBLIC ${OpenCV_LIBS} faiss)

# 2. Build the CLI executable
add_executable(imret_cli main.cpp)
target_link_libraries(imret_cli PRIVATE imret_core)
