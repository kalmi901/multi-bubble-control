cmake_minimum_required(VERSION 3.24)

project(
    MultiBubbleControl
    LANGUAGES CXX CUDA
)

include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

find_package(CUDAToolkit REQUIRED)

add_subdirectory(native)

if(BUILD_TESTING)
    add_subdirectory(tests/cuda)
endif()