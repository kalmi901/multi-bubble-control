#pragma once
#include <cuda_runtime_api.h>
#include "mbc/cuda/check.hpp"

class DeviceGuard {
public:
    explicit DeviceGuard(int requested_device)
    {
        MBC_CUDA_CHECK(cudaGetDevice(&previous_device_));
        MBC_CUDA_CHECK(cudaSetDevice(requested_device));
    }

    ~DeviceGuard() noexcept
    {
        cudaSetDevice(previous_device_);
    }

private:
    int previous_device_{0};
};