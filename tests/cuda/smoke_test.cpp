#include <cmath>
#include <cstddef>
#include <iostream>

#include <cuda_runtime_api.h>

#include "mbc/cuda/check.hpp"
#include "mbc/cuda/smoke.hpp"


int main() {
    try {
        int device_count = 0;

        const cudaError_t device_status =
            cudaGetDeviceCount(&device_count);

        if (
            device_status == cudaErrorNoDevice
            || device_status == cudaErrorInsufficientDriver
            || device_count == 0
        ) {
            std::cout
                << "CUDA device unavailable; skipping test.\n";

            return 77;
        }

        MBC_CUDA_CHECK(device_status);

        constexpr std::size_t size = 4096;

        float* device_data = nullptr;
        float* host_data = nullptr;

        cudaStream_t stream = nullptr;

        MBC_CUDA_CHECK(cudaSetDevice(0));

        MBC_CUDA_CHECK(
            cudaStreamCreateWithFlags(
                &stream,
                cudaStreamNonBlocking
            )
        );

        MBC_CUDA_CHECK(
            cudaMalloc(
                reinterpret_cast<void**>(&device_data),
                size * sizeof(float)
            )
        );

        MBC_CUDA_CHECK(
            cudaMallocHost(
                reinterpret_cast<void**>(&host_data),
                size * sizeof(float)
            )
        );

        mbc::cuda::fill_index_async(
            device_data,
            size,
            stream
        );

        MBC_CUDA_CHECK(
            cudaMemcpyAsync(
                host_data,
                device_data,
                size * sizeof(float),
                cudaMemcpyDeviceToHost,
                stream
            )
        );

        MBC_CUDA_CHECK(
            cudaStreamSynchronize(stream)
        );

        for (std::size_t index = 0; index < size; ++index) {
            const float expected =
                static_cast<float>(index);

            if (host_data[index] != expected) {
                std::cerr
                    << "Mismatch at index "
                    << index
                    << ": expected "
                    << expected
                    << ", got "
                    << host_data[index]
                    << '\n';

                return 1;
            }
        }

        MBC_CUDA_CHECK(cudaFree(device_data));
        MBC_CUDA_CHECK(cudaFreeHost(host_data));
        MBC_CUDA_CHECK(cudaStreamDestroy(stream));

        std::cout << "CUDA smoke test passed.\n";

        return 0;
    }
    catch (const std::exception& error) {
        std::cerr
            << "CUDA smoke test failed:\n"
            << error.what()
            << '\n';

        return 1;
    }
}