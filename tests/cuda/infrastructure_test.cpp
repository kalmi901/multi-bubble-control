#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <cuda_runtime_api.h>

#include "mbc/cuda/check.hpp"
#include "mbc/cuda/device_buffer.hpp"
#include "mbc/cuda/device_view.hpp"
#include "mbc/cuda/smoke.hpp"
#include "mbc/cuda/stream.hpp"


void test_kernel_execution()
{
    constexpr std::size_t size = 4096;

    mbc::cuda::DeviceBuffer<float> buffer(0, size);
    mbc::cuda::Stream stream(0);

    auto view = mbc::cuda::make_device_view(buffer);

    std::vector<float> host_data(size);

    mbc::cuda::fill_index_async(
        view.data(),
        view.size(),
        stream.get()
    );

    MBC_CUDA_CHECK(
        cudaMemcpyAsync(
            host_data.data(),
            view.data(),
            view.size() * sizeof(float),
            cudaMemcpyDeviceToHost,
            stream.get()
        )
    );

    stream.synchronize();

    for (std::size_t index = 0; index < size; ++index) {
        const float expected = static_cast<float>(index);

        if (host_data[index] != expected) {
            throw std::runtime_error(
                "Kernel output mismatch"
            );
        }
    }
}


int main()
{
    try {
        int device_count = 0;

        const cudaError_t status =
            cudaGetDeviceCount(&device_count);

        if (
            status == cudaErrorNoDevice
            || status == cudaErrorInsufficientDriver
            || device_count == 0
        ) {
            std::cout
                << "CUDA device unavailable; skipping test.\n";

            return 77;
        }

        MBC_CUDA_CHECK(status);

        test_kernel_execution();

        std::cout
            << "CUDA infrastructure test passed.\n";

        return 0;
    }
    catch (const std::exception& error) {
        std::cerr
            << "CUDA infrastructure test failed:\n"
            << error.what()
            << '\n';

        return 1;
    }
}