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


add_executable(
    mbc_cuda_device_buffer_test
    device_buffer_test.cpp
)

target_link_libraries(
    mbc_cuda_device_buffer_test
    PRIVATE
        mbc_cuda_core
)

add_test(
    NAME cuda_device_buffer
    COMMAND mbc_cuda_device_buffer_test
)

set_tests_properties(
    cuda_device_buffer
    PROPERTIES
        LABELS "cuda;gpu;unit"
        SKIP_RETURN_CODE 77
)


add_executable(
    mbc_cuda_stream_test
    stream_test.cpp
)

target_link_libraries(
    mbc_cuda_stream_test
    PRIVATE
        mbc_cuda_core
)

add_test(
    NAME cuda_stream
    COMMAND mbc_cuda_stream_test
)

set_tests_properties(
    cuda_stream
    PROPERTIES
        LABELS "cuda;gpu;unit"
        SKIP_RETURN_CODE 77
)


add_executable(
    mbc_cuda_device_view_test
    device_view_test.cpp
)

target_link_libraries(
    mbc_cuda_device_view_test
    PRIVATE
        mbc_cuda_core
)

add_test(
    NAME cuda_device_view
    COMMAND mbc_cuda_device_view_test
)

set_tests_properties(
    cuda_device_view
    PROPERTIES
        LABELS "cuda;gpu;unit"
        SKIP_RETURN_CODE 77  
)


add_executable(
    mbc_cuda_infrastructure_test
    infrastructure_test.cpp
)

target_link_libraries(
    mbc_cuda_infrastructure_test
    PRIVATE
        mbc_cuda_core
)

add_test(
    NAME cuda_infrastructure
    COMMAND mbc_cuda_infrastructure_test
)

set_tests_properties(
    cuda_infrastructure
    PROPERTIES
        LABELS "cuda;gpu;integration"
        SKIP_RETURN_CODE 77
)

add_executable(
    mbc_oscillator_rhs_test
    oscillator_rhs_test.cpp
)

target_link_libraries(
    mbc_oscillator_rhs_test
    PRIVATE
        mbc_cuda_core
)

add_test(
    NAME oscillator_rhs
    COMMAND mbc_oscillator_rhs_test
)

set_tests_properties(
    oscillator_rhs
    PROPERTIES
        LABELS "unit;host;ode"
)

add_executable(
    mbc_harmonic_oscillator_rhs_device_test
    oscillator_rhs_device_test.cu
)

target_link_libraries(
    mbc_harmonic_oscillator_rhs_device_test
    PRIVATE
        mbc_cuda_core
)

add_test(
    NAME harmonic_oscillator_rhs_device
    COMMAND mbc_harmonic_oscillator_rhs_device_test
)

set_tests_properties(
    harmonic_oscillator_rhs_device
    PROPERTIES
        LABELS "cuda;gpu;unit;ode"
        SKIP_RETURN_CODE 77
)