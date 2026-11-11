#include <iostream>
#include <stdexcept>
#include <utility>
#include <cuda_runtime_api.h>
#include "mbc/cuda/check.hpp"
#include "mbc/cuda/stream.hpp"


void test_creation()
{
    mbc::cuda::Stream stream(0);

    if (stream.get() == nullptr) {
        throw std::runtime_error(
            "Stream creation returned nullptr"
        );
    }

    if (stream.device_id() != 0) {
        throw std::runtime_error(
            "Stream device ID mismatch"
        );
    }

    stream.synchronize();
}


void test_move_constructor()
{
    mbc::cuda::Stream source(0);

    const cudaStream_t original_stream =
        source.get();

    mbc::cuda::Stream destination(
        std::move(source)
    );

    if (destination.get() != original_stream) {
        throw std::runtime_error(
            "Move constructor did not transfer stream"
        );
    }

    if (source.get() != nullptr) {
        throw std::runtime_error(
            "Moved-from Stream still owns CUDA stream"
        );
    }

    destination.synchronize();
}


void test_move_assignment()
{
    mbc::cuda::Stream source(0);
    mbc::cuda::Stream destination(0);

    const cudaStream_t original_stream =
        source.get();

    destination = std::move(source);

    if (destination.get() != original_stream) {
        throw std::runtime_error(
            "Move assignment did not transfer stream"
        );
    }

    if (source.get() != nullptr) {
        throw std::runtime_error(
            "Moved-from Stream still owns CUDA stream"
        );
    }

    destination.synchronize();
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

        test_creation();
        test_move_constructor();
        test_move_assignment();

        std::cout
            << "Stream tests passed.\n";

        return 0;
    }
    catch (const std::exception& error) {
        std::cerr
            << "Stream test failed:\n"
            << error.what()
            << '\n';

        return 1;
    }
}