#pragma once
#include <cstddef>
#include "mbc/cuda/device_buffer.hpp"


namespace mbc::cuda{

template <typename T>
class DeviceView {
public:
    DeviceView() noexcept = default;

    DeviceView(
        T* data,
        std::size_t size
    ) noexcept
        : data_(data),
          size_(size)
    {}

    T* data() const noexcept {
        return data_;
    }

    std::size_t size() const noexcept {
        return size_;
    }

    bool empty() const noexcept {
        return size_ == 0;
    }

private:
    T* data_{nullptr};
    std::size_t size_{0};
};

template<typename T>
DeviceView<T> make_device_view(DeviceBuffer<T> &buffer) noexcept {
    return DeviceView<T>{buffer.data(), buffer.size()};
}

template<typename T>
DeviceView<const T> make_device_view(const DeviceBuffer<T> &buffer) noexcept {
    return DeviceView<const T>{buffer.data(), buffer.size()};
}

} // namespace mbc::cuda