#include <cstddef>
#include <iostream>
#include <utility>
#include <cuda_runtime_api.h>
#include "mbc/cuda/check.hpp"
#include "mbc/cuda/device_buffer.hpp"


void test_allocation()
{
    mbc::cuda::DeviceBuffer<float> buffer(0, 1024);

    if (buffer.data() == nullptr) {
        throw std::runtime_error("DeviceBuffer allocation failed");
    }

    if (buffer.size() != 1024) {
        throw std::runtime_error("DeviceBuffer size mismatch");
    }

    if (buffer.device_id() != 0) {
        throw std::runtime_error("DeviceBuffer device ID mismatch");
    }
}


void test_move_constructor()
{
    mbc::cuda::DeviceBuffer<float> source(0, 1024);

    float* original_ptr = source.data();

    mbc::cuda::DeviceBuffer<float> destination(
        std::move(source)
    );

    if (destination.data() != original_ptr) {
        throw std::runtime_error(
            "Move constructor did not transfer pointer"
        );
    }

    if (source.data() != nullptr) {
        throw std::runtime_error(
            "Moved-from buffer still owns memory"
        );
    }

    if (source.size() != 0) {
        throw std::runtime_error(
            "Moved-from buffer size was not reset"
        );
    }
}


void test_move_assignment()
{
    mbc::cuda::DeviceBuffer<float> source(0, 1024);
    mbc::cuda::DeviceBuffer<float> destination(0, 2048);

    float* original_ptr = source.data();

    destination = std::move(source);

    if (destination.data() != original_ptr) {
        throw std::runtime_error(
            "Move assignment did not transfer pointer"
        );
    }

    if (destination.size() != 1024) {
        throw std::runtime_error(
            "Move assignment did not transfer size"
        );
    }

    if (source.data() != nullptr) {
        throw std::runtime_error(
            "Moved-from buffer still owns memory"
        );
    }

    if (source.size() != 0) {
        throw std::runtime_error(
            "Moved-from buffer size was not reset"
        );
    }
}


int main() {
    try {
        int device_count = 0;
        const cudaError_t device_status = cudaGetDeviceCount(&device_count);
        if (
            device_status == cudaErrorNoDevice
            || device_status == cudaErrorInsufficientDriver
            || device_count == 0
        ) {
            std::cout << "CUDA device unavailable; skipping test.\n";

            return 77;
        }



    }
    catch (const std::exception& error) {
        std::cerr
            << "CUDA device buffer failed:\n"
            << error.what()
            << "\n";
        return 1;
    }
}