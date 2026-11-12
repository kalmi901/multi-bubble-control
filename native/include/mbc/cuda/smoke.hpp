#pragma once

#include <cstddef>

#include <cuda_runtime_api.h>

namespace mbc::cuda {

void fill_index_async(float* output, std::size_t size, cudaStream_t stream);

} // namespace mbc::cuda