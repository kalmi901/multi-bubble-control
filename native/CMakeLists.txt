add_library(
    mbc_cuda_core
    STATIC
    src/cuda/smoke.cu
)

target_include_directories(
    mbc_cuda_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(
    mbc_cuda_core
    PUBLIC
        CUDA::cudart
)

target_compile_features(
    mbc_cuda_core
    PUBLIC
        cxx_std_17
)