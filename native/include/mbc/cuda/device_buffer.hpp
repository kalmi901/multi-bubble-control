#pragma once
#include <cstddef>
#include <cstdio>
#include <utility>
#include <cuda_runtime_api.h>
#include "mbc/cuda/check.hpp"
#include "mbc/cuda/device_guard.hpp"

namespace mbc::cuda {

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer(
        int device_id,
        std::size_t size):  
        device_id_(device_id),
        size_(size) {
        if (size_ == 0) {
            return;
        }

        mbc::cuda::DeviceGuard guard(device_id_);

        MBC_CUDA_CHECK(
            cudaMalloc(
                reinterpret_cast<void**>(&data_),
                size_ * sizeof(T)
            )
        );
    }

    ~DeviceBuffer() noexcept {
        release();
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : data_(other.data_),
        size_(other.size_),
        device_id_(other.device_id_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            release();

            data_ = other.data_;
            size_ = other.size_;
            device_id_ = other.device_id_;

            other.data_ = nullptr;
            other.size_ = 0;
        }

        return *this;
    }

    T* data() noexcept {
        return data_;
    }

    const T* data() const noexcept {
        return data_;
    }

    std::size_t size() const noexcept {
        return size_;
    }

    int device_id() const noexcept {
        return device_id_;
    }

private:
    void release() noexcept
    {
        if (data_ == nullptr) {
            return;
        }

        int previous_device = 0;
        bool have_previous_device = false;

        cudaError_t err = cudaGetDevice(&previous_device);

        if (err == cudaSuccess) {
            have_previous_device = true;
        }

        err = cudaSetDevice(device_id_);

        if (err == cudaSuccess) {
            err = cudaFree(data_);
        }

        if (err != cudaSuccess) {
            std::fprintf(
                stderr,
                "DeviceBuffer release failed: %s\n",
                cudaGetErrorString(err)
            );
        }

        if (have_previous_device) {
            const cudaError_t restore_err =
                cudaSetDevice(previous_device);

            if (restore_err != cudaSuccess) {
                std::fprintf(
                    stderr,
                    "CUDA device restore failed: %s\n",
                    cudaGetErrorString(restore_err)
                );
            }
        }

        data_ = nullptr;
        size_ = 0;
    }

    T* data_{nullptr};
    std::size_t size_{0};
    int device_id_{0};
};

}  // namespace mbc::cuda