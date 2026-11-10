#include "mbc/cuda/smoke.hpp"

#include "mbc/cuda/check.hpp"

namespace mbc::cuda {

namespace {

__global__ void fill_index_kernel(
    float* output,
    int size
) {
    const int thread_index =
    blockIdx.x * blockDim.x + threadIdx.x;

    const int stride =
        blockDim.x * gridDim.x;

    for (
        int index = thread_index;
        index < size;
        index += stride
    ) {
        output[index] = static_cast<float>(index);
    }
}

}  // namespace

void fill_index_async(
    float* output,
    std::size_t size,
    cudaStream_t stream
) {
    if (size == 0) {
        return;
    }

    constexpr int block_size = 256;

    const int grid_size =
        static_cast<int>((size + block_size - 1) / block_size);

    fill_index_kernel<<<grid_size, block_size, 0, stream>>>(output, static_cast<int>(size));

    MBC_CUDA_CHECK(cudaGetLastError());
}

}  // namespace mbc::cuda