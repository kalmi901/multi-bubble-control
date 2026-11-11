#include <iostream>
#include <stdexcept>

#include "mbc/cuda/check.hpp"
#include "mbc/cuda/device_buffer.hpp"
#include "mbc/cuda/device_view.hpp"

void test_default_view()
{
    mbc::cuda::DeviceView<float> view;

    if (view.data() != nullptr) {
        throw std::runtime_error(
            "Default DeviceView data is not nullptr"
        );
    }

    if (!view.empty()) {
        throw std::runtime_error(
            "Default DeviceView is not empty"
        );
    }
}


void test_pointer_and_size()
{
    mbc::cuda::DeviceBuffer<float> buffer(0, 1024);

    mbc::cuda::DeviceView<float> view(
        buffer.data(),
        buffer.size()
    );

    if (view.data() != buffer.data()) {
        throw std::runtime_error(
            "DeviceView pointer mismatch"
        );
    }

    if (view.size() != buffer.size()) {
        throw std::runtime_error(
            "DeviceView size mismatch"
        );
    }

    if (view.empty()) {
        throw std::runtime_error(
            "DeviceView unexpectedly empty"
        );
    }
}

void test_copy()
{
    mbc::cuda::DeviceBuffer<float> buffer(0, 1024);

    mbc::cuda::DeviceView<float> source(
        buffer.data(),
        buffer.size()
    );

    mbc::cuda::DeviceView<float> copy = source;

    if (copy.data() != source.data()) {
        throw std::runtime_error(
            "Copied DeviceView pointer mismatch"
        );
    }

    if (copy.size() != source.size()) {
        throw std::runtime_error(
            "Copied DeviceView size mismatch"
        );
    }
}


void test_make_device_view()
{
    mbc::cuda::DeviceBuffer<float> buffer(0, 1024);

    auto view = mbc::cuda::make_device_view(buffer);

    if (view.data() != buffer.data()) {
        throw std::runtime_error(
            "DeviceView pointer mismatch"
        );
    }

    if (view.size() != buffer.size()) {
        throw std::runtime_error(
            "DeviceView size mismatch"
        );
    }

    if (view.empty()) {
        throw std::runtime_error(
            "DeviceView unexpectedly empty"
        );
    }

}

void test_make_const_device_view()
{
    mbc::cuda::DeviceBuffer<float> buffer(0, 1024);

    const auto& const_buffer = buffer;
    auto view = mbc::cuda::make_device_view(const_buffer);

    if (view.data() != const_buffer.data()) {
        throw std::runtime_error(
            "Const DeviceView pointer mismatch"
        );
    }

    if (view.size() != const_buffer.size()) {
        throw std::runtime_error(
            "Const DeviceView size mismatch"
        );
    }
}


int main(){
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

            test_default_view();
            test_pointer_and_size();
            test_copy();
            test_make_device_view();
            test_make_const_device_view();

            std::cout << "DeviceView tests passed.\n";

            return 0;
    }
    catch (const std::exception& error) {
        std::cerr
            << "DeviceView test failed:\n"
            << error.what()
            << "\n";

            return 1;
    }
}