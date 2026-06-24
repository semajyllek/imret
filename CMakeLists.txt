cmake_minimum_required(VERSION 3.15)
project(imret)


set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Enforce C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED COMPONENTS core imgproc features2d)

# Apple Clang doesn't ship OpenMP — locate Homebrew libomp automatically
if(APPLE)
    execute_process(
        COMMAND brew --prefix libomp
        OUTPUT_VARIABLE LIBOMP_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(LIBOMP_PREFIX)
        set(OpenMP_C_FLAGS   "-Xpreprocessor -fopenmp")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib")
    endif()
endif()

find_package(OpenMP REQUIRED)
find_package(faiss REQUIRED)

# 1. Build the core engine as a static library
add_library(imret_core STATIC 
    src/extractor.cpp 
    src/vault.cpp
)
target_include_directories(imret_core PUBLIC include)
target_link_libraries(imret_core PUBLIC ${OpenCV_LIBS} faiss OpenMP::OpenMP_CXX)
if(APPLE AND LIBOMP_PREFIX)
    target_include_directories(imret_core PRIVATE "${LIBOMP_PREFIX}/include")
endif()

if(NOT SKBUILD)
    # 2. Build the CLI executable
    add_executable(imret_cli main.cpp)
    target_link_libraries(imret_cli PRIVATE imret_core)

    # 3. Tests
    enable_testing()
    add_subdirectory(tests)
endif()
