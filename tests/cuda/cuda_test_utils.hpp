#pragma once

#include <cuda_runtime_api.h>

#include "mbc/cuda/check.hpp"

namespace mbc::test {
inline bool cuda_device_available(int device_id = 0) {
    int device_count = 0;

    const cudaError_t status = cudaGetDeviceCount(&device_count);

    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
        device_count <= device_id) {
        return false;
    }

    MBC_CUDA_CHECK(status);

    return true;
}

} // namespace mbc::test