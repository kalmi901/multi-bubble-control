add_executable(
    mbc_cuda_smoke_test
    smoke_test.cpp
)

target_link_libraries(
    mbc_cuda_smoke_test
    PRIVATE
        mbc_cuda_core
)

add_test(
    NAME cuda_smoke
    COMMAND mbc_cuda_smoke_test
)

set_tests_properties(
    cuda_smoke
    PROPERTIES
        LABELS "cuda;gpu;smoke"
        SKIP_RETURN_CODE 77
)