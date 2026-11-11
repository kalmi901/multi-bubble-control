#pragma once

#include <sstream>
#include <stdexcept>

#include <cuda_runtime_api.h>

namespace mbc::cuda {

inline void
check_cuda(cudaError_t status, const char* expression, const char* file, int line) {
    if (status == cudaSuccess) {
        return;
    }

    std::ostringstream message;

    message << "CUDA error: " << cudaGetErrorString(status)
            << "\nExpression: " << expression << "\nLocation: " << file << ":" << line;

    throw std::runtime_error(message.str());
}

} // namespace mbc::cuda

#define MBC_CUDA_CHECK(expression)                                                     \
    ::mbc::cuda::check_cuda((expression), #expression, __FILE__, __LINE__)