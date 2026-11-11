#pragma once

#include <cstdio>

#include <cuda_runtime_api.h>

#include "mbc/cuda/check.hpp"
#include "mbc/cuda/device_guard.hpp"

namespace mbc::cuda {

class Stream {
public:
    explicit Stream(int device_id)
        : device_id_(device_id) {
        mbc::cuda::DeviceGuard guard(device_id_);

        MBC_CUDA_CHECK(
            cudaStreamCreateWithFlags(
                &stream_,
                cudaStreamNonBlocking
            )
        );
    }

    ~Stream() noexcept {
        release();
    }

    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;

    Stream(Stream&& other) noexcept
        : stream_(other.stream_),
          device_id_(other.device_id_) {
        other.stream_ = nullptr;
    }

    Stream& operator=(Stream&& other) noexcept {
        if (this != &other) {
            release();

            stream_ = other.stream_;
            device_id_ = other.device_id_;

            other.stream_ = nullptr;
        }

        return *this;
    }

    cudaStream_t get() const noexcept {
        return stream_;
    }

    int device_id() const noexcept {
        return device_id_;
    }

    void synchronize() {
        if (stream_ == nullptr) {
            return;
        }

        DeviceGuard guard(device_id_);

        MBC_CUDA_CHECK(
            cudaStreamSynchronize(stream_)
        );
    }
 

private:
    void release() noexcept {
        if (stream_ == nullptr) {
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
            err = cudaStreamDestroy(stream_);
        }

        if (err != cudaSuccess) {
            std::fprintf(
                stderr,
                "CUDA stream release failed: %s\n",
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

        stream_ = nullptr;
    }

    cudaStream_t stream_{nullptr};
    int device_id_{0};
};

}  // namespace mbc::cuda